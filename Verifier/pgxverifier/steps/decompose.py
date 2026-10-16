"""Decomposition: a text into atomic claims, in two passes plus one deduplication.

  pass 1  recommendation facts (prompts/decompose_recommendation.txt)
  pass 2  context facts        (prompts/decompose_context.txt)
          both over the same text, in parallel; the lists are concatenated, pass 1 first
  dedup   one call over the concatenation with the boundary between the two lists
          (prompts/dedup.txt); every distinct fact is kept once, copied verbatim

All three calls go to the decomposer (Gemma-4-31B-it) at temperature 0.

Two failure modes are handled rather than passed on:
  * output ceiling: a reply cut at max_tokens whose JSON does not parse is retried once at the
    largest ceiling the context allows; if it is still cut, DecompositionError is raised. A
    partial claim list is never returned.
  * context overflow: before every call the prompt is counted with the server's /tokenize route
    and max_tokens is clamped so prompt and output fit. If /tokenize is unavailable the call
    runs unclamped.
"""
import difflib
import hashlib
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from openai import BadRequestError, OpenAI

from .. import config
from ..jsonparse import clean, first_json_value, repair

PROMPT_FILES = {"pass1": "decompose_recommendation.txt", "pass2": "decompose_context.txt",
                "dedup": "dedup.txt"}
# Every placeholder the three templates use; the warm-up fills all of them.
PLACEHOLDERS = ("{question_text}", "{text}", "{claims}", "{stream1_last}")
_THINK_CLOSE = "</think>"


class DecompositionError(RuntimeError):
    """A decomposition or deduplication call could not produce a complete result."""


def strip_thinking(text):
    """Keep only what follows the final </think>; a text without one is returned unchanged."""
    idx = text.rfind(_THINK_CLOSE)
    if idx == -1:
        return text
    return text[idx + len(_THINK_CLOSE):].lstrip()


def load_prompts():
    """{key: {"file", "text", "sha256"}} for the three templates."""
    out = {}
    for key, name in PROMPT_FILES.items():
        text = (config.PROMPT_DIR / name).read_text(encoding="utf-8")
        out[key] = {"file": name, "text": text,
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return out


def numbered_list(claims):
    return "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))


def split_index(line):
    """'[<i>] <text>' -> (i, text); a line without a well-formed bracket -> (None, line)."""
    head, sep, rest = line.partition("] ")
    if sep and head.startswith("[") and head[1:].isdigit():
        return int(head[1:]), rest.strip()
    return None, line


def absorbed_groups(absorbed_into):
    """{absorbed index: survivor index} -> [(survivor, [absorbed, ...]), ...], first-seen order."""
    by_survivor = {}
    for k, v in absorbed_into.items():
        by_survivor.setdefault(int(v), []).append(int(k))
    return list(by_survivor.items())


class Endpoint:
    """The decomposer's OpenAI-compatible server. Thread-safe."""

    def __init__(self, base_url=None, model=None, min_output_tokens=64, parse_retries=2):
        self.base_url = (base_url or config.DECOMPOSER_URL).rstrip("/")
        self.model = model or config.DECOMPOSER_MODEL
        self.temperature = config.DECOMPOSER_TEMPERATURE
        self.min_output_tokens = min_output_tokens
        self.parse_retries = parse_retries
        # max_retries=0: a retried generation would queue a second copy of a long request.
        self._c = OpenAI(base_url=self.base_url, api_key="EMPTY",
                         timeout=config.DECOMPOSER_TIMEOUT, max_retries=0)
        root = self.base_url[:-len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        self._tokenize_url = root + "/tokenize"
        self._tokenize_warned = False
        self._lock = threading.Lock()

    def count_prompt_tokens(self, prompt):
        """(prompt tokens, max_model_len) from the vLLM /tokenize route, or (None, None)."""
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "add_generation_prompt": True}).encode("utf-8")
        req = urllib.request.Request(self._tokenize_url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                obj = json.loads(r.read().decode("utf-8"))
            return int(obj["count"]), int(obj["max_model_len"])
        except Exception as e:
            with self._lock:
                if not self._tokenize_warned:
                    self._tokenize_warned = True
                    print(f"  WARN {self.base_url}: /tokenize unavailable "
                          f"({type(e).__name__}: {str(e)[:100]}); max_tokens is not "
                          f"clamped to the context")
            return None, None

    def warm(self, template):
        """Prefill the template prefix with an 8-token completion. A cold prefill of a full
        prompt was seen never to be scheduled on a small KV cache. Failures are ignored."""
        p = template
        for ph in PLACEHOLDERS:
            p = p.replace(ph, "warmup" if ph != "{stream1_last}" else "0")
        try:
            self._c.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": p}],
                temperature=self.temperature, max_tokens=8)
        except Exception:
            pass

    def _complete(self, prompt, max_tokens):
        r = self._c.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature, max_tokens=max_tokens)
        ch = r.choices[0]
        u = getattr(r, "usage", None)
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                 "completion_tokens": getattr(u, "completion_tokens", None)}
        return (ch.message.content or ""), (ch.finish_reason or ""), usage

    def call_json(self, prompt, expect, max_tokens):
        """-> (parsed first JSON value, meta). Raises DecompositionError when the reply is cut
        and cannot be made to fit, when the server rejects the request, or when a complete
        reply is not the expected JSON type."""
        n_prompt, ctx = self.count_prompt_tokens(prompt)
        ceiling = None
        budget = max_tokens
        if n_prompt is not None:
            ceiling = ctx - n_prompt
            if ceiling < self.min_output_tokens:
                raise DecompositionError(
                    f"{self.base_url}: prompt is {n_prompt} tokens, leaving {ceiling} of the "
                    f"{ctx}-token context for output (< {self.min_output_tokens}); not sent")
            budget = min(max_tokens, ceiling)
        meta = {"model": self.model, "prompt_tokens": n_prompt, "max_model_len": ctx,
                "requested_max_tokens": max_tokens, "attempts": [], "escalated": False,
                "warnings": []}
        parse_failures = 0
        while True:
            try:
                text, finish, usage = self._complete(prompt, budget)
            except BadRequestError as e:
                raise DecompositionError(
                    f"{self.base_url} rejected the request (max_tokens={budget}, "
                    f"prompt_tokens={n_prompt}, max_model_len={ctx}): {e}") from e
            meta["attempts"].append({"max_tokens": budget, "finish_reason": finish, **usage})
            try:
                obj = first_json_value(clean(text), expect)
            except json.JSONDecodeError as e:
                obj, err = None, e
                try:
                    o, rep = repair(clean(text))
                    if (expect == "dict" and isinstance(o, dict)) or \
                       (expect == "list" and isinstance(o, list)):
                        obj, err = o, None
                        meta["warnings"].append(
                            f"FORMAT-ONLY JSON REPAIR applied: passes={rep.get('passes')} "
                            f"chars_removed={rep.get('chars_removed')}")
                        meta["format_repair"] = rep
                except Exception as re_err:
                    meta["warnings"].append(f"format repair did not apply: {re_err}")
            else:
                err = None
            if err is None:
                if expect == "list" and not isinstance(obj, list):
                    raise DecompositionError(f"{self.base_url}: expected a JSON array, got "
                                             f"{type(obj).__name__}: {clean(text)[:200]!r}")
                if expect == "dict" and not isinstance(obj, dict):
                    raise DecompositionError(f"{self.base_url}: expected a JSON object, got "
                                             f"{type(obj).__name__}: {clean(text)[:200]!r}")
                if finish == "length":
                    meta["warnings"].append(f"finish_reason=length after a complete JSON value "
                                            f"(max_tokens={budget}); trailing output was cut")
                meta["completion_text"] = text
                return obj, meta
            if finish == "length":
                if ceiling is not None and budget < ceiling and not meta["escalated"]:
                    meta["escalated"] = True
                    budget = ceiling
                    continue
                raise DecompositionError(
                    f"{self.base_url}: output cut at max_tokens={budget} (prompt {n_prompt} + "
                    f"output {usage.get('completion_tokens')} of a {ctx}-token context) and the "
                    f"JSON does not parse: {err}. No partial claim list is returned.")
            parse_failures += 1
            if parse_failures > self.parse_retries:
                raise DecompositionError(
                    f"{self.base_url}: malformed JSON after {parse_failures} attempts "
                    f"(finish_reason={finish}): {err}; head={clean(text)[:200]!r}")


def _claims(obj):
    return [str(c).strip() for c in (obj or []) if str(c).strip()]


def two_passes(ep, prompts, question_text, text):
    """Pass 1 and pass 2 over the same text, in parallel. -> (pass-1 claims, pass-2 claims, meta)."""

    def one(key):
        tmpl = prompts[key]["text"]
        p = tmpl.replace("{question_text}", question_text).replace("{text}", text)
        ep.warm(tmpl)
        obj, meta = ep.call_json(p, "list", config.DECOMPOSE_MAX_TOKENS)
        return _claims(obj), meta

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1, f2 = ex.submit(one, "pass1"), ex.submit(one, "pass2")
        (s1, m1), (s2, m2) = f1.result(), f2.result()
    return s1, s2, {"pass1_call": m1, "pass2_call": m2,
                    "warnings": [f"pass1: {w}" for w in m1["warnings"]]
                                + [f"pass2: {w}" for w in m2["warnings"]]}


def _survivors(raw, lines, warnings):
    """(claim texts, input index of each) from the survivor lines '[<i>] <text>'. A line
    without an index bracket is matched to the first unused verbatim input; its index is None
    if there is none. The text is kept either way."""
    stripped = [c.strip() for c in raw]
    used = set()
    texts, idxs = [], []
    for k, line in enumerate(lines):
        idx, text = split_index(line)
        if idx is None:
            for i, c in enumerate(stripped):
                if i not in used and c == text:
                    idx = i
                    break
            if idx is None:
                warnings.append(f"survivor [{k}] has no index bracket and is not verbatim "
                                f"from the input: {text[:80]!r}")
        elif not 0 <= idx < len(raw):
            warnings.append(f"survivor [{k}] bracket index {idx} is out of range "
                            f"(have {len(raw)} inputs): {text[:80]!r}")
            idx = None
        elif stripped[idx] != text:
            warnings.append(f"survivor [{k}] text differs from input [{idx}]: {text[:80]!r}")
        if idx is not None:
            used.add(idx)
        texts.append(text)
        idxs.append(idx)
    return texts, idxs


def dedup(ep, prompts, pass1, pass2):
    """One deduplication call over pass1 + pass2. -> {"claims", "merge_groups",
    "survivor_indices", "dedup_call", "warnings"} (indices refer to pass1 + pass2)."""
    raw = list(pass1) + list(pass2)
    stream1_last = len(pass1) - 1
    if len(raw) < 2:
        return {"claims": raw, "merge_groups": [], "survivor_indices": list(range(len(raw))),
                "dedup_call": None, "warnings": []}
    tmpl = prompts["dedup"]["text"]
    ep.warm(tmpl)
    p = tmpl.replace("{stream1_last}", str(stream1_last)).replace("{claims}", numbered_list(raw))
    obj, meta = ep.call_json(p, "dict", config.DEDUP_MAX_TOKENS)
    warnings = list(meta["warnings"])
    survivors, idxs = _survivors(raw, _claims(obj.get("deduplicated_claims")), warnings)

    groups = []
    absorbed = obj.get("absorbed_into")
    if absorbed is not None:
        if not isinstance(absorbed, dict):
            warnings.append(f"absorbed_into ignored (not an object): {absorbed!r}"[:160])
            absorbed = {}
        ok_entries = {}
        for k, v in absorbed.items():
            if (k.isdigit() and 0 <= int(k) < len(raw) and isinstance(v, int)
                    and 0 <= v < len(raw) and int(k) != v):
                ok_entries[k] = v
            else:
                warnings.append(f"absorbed_into entry ignored (malformed indices): "
                                f"{k!r}: {v!r}"[:160])
        groups = [{"kept_index": s, "merged_from_indices": ks}
                  for s, ks in absorbed_groups(ok_entries)]

    # Every input index must appear once: in a survivor's bracket or as an absorbed key.
    seen = {}
    for idx in idxs:
        if idx is not None:
            seen[idx] = seen.get(idx, 0) + 1
    for g in groups:
        for m in g["merged_from_indices"]:
            seen[m] = seen.get(m, 0) + 1
    missing = [i for i in range(len(raw)) if i not in seen]
    doubled = sorted(i for i, n in seen.items() if n > 1)
    if missing:
        warnings.append(f"index invariant: {len(missing)} input index(es) neither survived nor "
                        f"were absorbed: {missing[:20]}")
    if doubled:
        warnings.append(f"index invariant: input index(es) counted twice: {doubled[:20]}")
    return {"claims": survivors, "merge_groups": groups, "survivor_indices": idxs,
            "dedup_call": meta, "warnings": warnings}


def _norm(s):
    return " ".join(s.lower().split())


def check_dedup(original, deduplicated, merge_groups):
    """Warnings: fewer than half the claims kept; a kept claim that is not a near-match of any
    input; a merge index out of range."""
    warnings = []
    n_orig, n_out = len(original), len(deduplicated)
    if n_orig > 0 and n_out < 0.5 * n_orig:
        warnings.append(f"dedup reduced {n_orig} -> {n_out} claims (< 50% retained)")
    norm_orig = [_norm(c) for c in original]
    for i, claim in enumerate(deduplicated):
        nc = _norm(claim)
        if any(nc in o or o in nc for o in norm_orig):
            continue
        best = max((difflib.SequenceMatcher(None, nc, o).ratio() for o in norm_orig), default=0.0)
        if best < 0.6:
            warnings.append(f"output claim [{i}] is not a near-match of any original "
                            f"(best ratio {best:.2f}): {claim[:80]!r}")
    for gi, g in enumerate(merge_groups):
        for idx in g["merged_from_indices"]:
            if idx < 0 or idx >= n_orig:
                warnings.append(f"merge_group[{gi}] references invalid original index {idx} "
                                f"(have {n_orig} originals)")
    return warnings


def decompose(ep, prompts, question_text, text):
    """A text -> its deduplicated claims and the record of how they were made."""
    text = strip_thinking(text)
    if not text.strip():
        raise ValueError("empty text: nothing to decompose")
    s1, s2, m = two_passes(ep, prompts, question_text, text)
    d = dedup(ep, prompts, s1, s2)
    raw = s1 + s2
    warnings = m["warnings"] + (check_dedup(raw, d["claims"], d["merge_groups"])
                                if len(raw) >= 2 else []) + d["warnings"]
    return {"claims": d["claims"], "pass1_claims": s1, "pass2_claims": s2,
            "stream1_last": len(s1) - 1, "merge_groups": d["merge_groups"],
            "survivor_indices": d["survivor_indices"], "warnings": warnings,
            "prompts": {k: {"file": v["file"], "sha256": v["sha256"]} for k, v in prompts.items()},
            "calls": {"pass1": m["pass1_call"], "pass2": m["pass2_call"], "dedup": d["dedup_call"]}}
