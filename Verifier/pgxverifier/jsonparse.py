"""Reading JSON out of model replies.

clean / first_json_value parse every reply (decomposer, deduplicator and labelling judge).
repair is a format-only fallback for the decomposer, used only after first_json_value fails.
"""
import json
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def clean(text):
    """Strip <think> blocks and a single surrounding markdown code fence."""
    t = _THINK_RE.sub("", text or "").strip()
    m = _FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    return t


def first_json_value(text, expect=None):
    """Parse the first complete JSON value in text (expect: "list", "dict" or None).

    A reply may wrap the JSON in prose; a reply with no valid JSON at the expected position
    still raises json.JSONDecodeError. A backslash that does not start a valid JSON escape is
    escaped and the parse retried once."""
    s = text.strip()
    starts = []
    if expect != "dict":
        i = s.find("[")
        if i >= 0:
            starts.append(i)
    if expect != "list":
        i = s.find("{")
        if i >= 0:
            starts.append(i)
    if not starts:
        return json.loads(s)
    try:
        obj, _ = json.JSONDecoder().raw_decode(s, min(starts))
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
        obj, _ = json.JSONDecoder().raw_decode(repaired, min(starts))
    return obj


# ------------------------------------------------------------------ format-only repair
# Two shapes the decomposer was seen to produce: a backslash before a quote that is JSON syntax
# (\"absorbed_into\": ...) after the claim array has closed, and a trailing comma before ] or }.
# Only text after the first parse error is touched, and the text before it is asserted
# unchanged, so a claim before that point cannot be altered.
MAX_PASSES = 8


def _strip_fence(t):
    s = t.strip()
    if s.startswith("```"):
        i = s.find("\n")
        s = s[i + 1:] if i > 0 else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _drop_trailing_comma(text, pos):
    """Remove a comma followed only by whitespace and then ] or }.
    -> (text, removed position) or (text unchanged, -1).

    Python before 3.13 reports this error at the bracket, 3.13 and later at the comma."""
    if text[pos] == ",":
        j = pos + 1
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        if j < len(text) and text[j] in "]}":
            return text[:pos] + text[pos + 1:], pos
        return text, -1
    j = pos
    while j < len(text) and text[j] in " \t\r\n":
        j += 1
    if j >= len(text) or text[j] not in "]}":
        return text, -1
    i = pos - 1
    while i >= 0 and text[i] in " \t\r\n":
        i -= 1
    if i < 0 or text[i] != ",":
        return text, -1
    return text[:i] + text[i + 1:], i


def _drop_backslash_before_quote(tail):
    out = []
    i = 0
    while i < len(tail):
        if tail[i] == "\\" and i + 1 < len(tail) and tail[i + 1] == '"':
            i += 1
            continue
        out.append(tail[i])
        i += 1
    return "".join(out)


def repair(text):
    """-> (object, report). Raises if the text cannot be repaired."""
    orig = _strip_fence(text)
    rep = {"fenced": orig != text.strip(), "passes": [], "chars_removed": 0}
    cur = orig
    for _ in range(MAX_PASSES):
        try:
            obj = json.loads(cur)
            rep["ok"] = True
            return obj, rep
        except json.JSONDecodeError as e:
            p = e.pos
            if p is None or p <= 0 or p >= len(cur):
                raise
            fixed, cut = _drop_trailing_comma(cur, p)
            if cut >= 0:
                rep["passes"].append({"pos": p, "comma_at": cut, "removed": 1,
                                      "msg": e.msg, "kind": "trailing_comma"})
                rep["chars_removed"] += 1
                assert fixed[:cut] == cur[:cut], "text before the removed comma changed"
                cur = fixed
                continue
            head, tail = cur[:p], cur[p:]
            fixed_tail = _drop_backslash_before_quote(tail)
            if fixed_tail == tail:
                raise
            rep["passes"].append({"pos": p, "removed": len(tail) - len(fixed_tail),
                                  "msg": e.msg})
            rep["chars_removed"] += len(tail) - len(fixed_tail)
            assert head == orig[:p], "text before the parse error changed"
            cur = head + fixed_tail
    raise ValueError(f"still unparsable after {MAX_PASSES} repair passes")
