"""Labelling judge (Grok 4.5, xAI Responses API): one call per answer labels every claim.

  forward   each answer claim: addressed / partially_addressed / wrong / extra, with the GT
            claims it matches
  backward  each GT claim: covered / partial / missing
  check     the judge lists the claims it labelled inconsistently across the two directions and
            corrects them; the corrections are applied to fwd_labels / bwd_labels

The raw forward / backward lists are kept as returned (the score counts extra claims from them).
"""
import json
import os
import threading
import time

from openai import (APIConnectionError, APITimeoutError, InternalServerError, OpenAI,
                    PermissionDeniedError, RateLimitError)

from .. import config
from ..jsonparse import clean, first_json_value


class LabelError(RuntimeError):
    """The labelling judge returned something that cannot be used."""


class _Transient(LabelError):
    """An empty reply: intermittent on reasoning models, worth retrying."""


class GrokClient:
    """Thread-safe client of the xAI Responses API with retries and usage accounting."""

    def __init__(self, api_key=None, model=None):
        api_key = (api_key or os.getenv("XAI_API_KEY") or "").strip()
        if not api_key:
            raise LabelError("no xAI API key: set XAI_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=config.XAI_BASE_URL,
                              timeout=config.LABEL_TIMEOUT)
        self.model = model or config.LABEL_MODEL
        self._lock = threading.Lock()
        self.n_calls = 0
        self.total_tokens = 0
        self.cost_usd = 0.0

    def _raw(self, prompt):
        time.sleep(config.LABEL_CALL_SLEEP)
        # No output cap: the reasoning tokens of a capped call can leave no room for the JSON.
        resp = self._client.responses.create(
            model=self.model, input=[{"role": "user", "content": prompt}],
            temperature=config.LABEL_TEMPERATURE)
        usage = getattr(resp, "usage", None)
        with self._lock:
            self.n_calls += 1
            if usage is not None:
                self.total_tokens += getattr(usage, "total_tokens", 0) or 0
                ticks = getattr(usage, "cost_in_usd_ticks", None)
                if ticks:
                    self.cost_usd += ticks / 1e9
        text = (getattr(resp, "output_text", None) or "").strip()
        if not text:
            raise _Transient(f"empty output_text (model={self.model})")
        return text

    def call_json(self, prompt, expect=None):
        """Call and parse the first JSON value. Connection, timeout, rate-limit and server
        errors and empty replies are retried with exponential backoff; a content-safety 403
        (a flaky false positive on clinical text) and malformed JSON are retried after a short
        sleep."""
        retries = config.LABEL_RETRIES
        for attempt in range(retries + 1):
            try:
                raw = self._raw(prompt)
            except PermissionDeniedError as e:
                if attempt >= retries:
                    raise LabelError(f"content-safety 403 after {retries + 1} attempts: {e}") from e
                time.sleep(config.LABEL_SAFETY_SLEEP)
                continue
            except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError,
                    _Transient) as e:
                if attempt >= retries:
                    raise LabelError(f"API error after {retries + 1} attempts: "
                                     f"{type(e).__name__}: {e}") from e
                time.sleep(min(config.LABEL_BACKOFF * (2 ** attempt), 30.0))
                continue
            cleaned = clean(raw)
            try:
                parsed = first_json_value(cleaned, expect)
            except json.JSONDecodeError as e:
                if attempt >= retries:
                    raise LabelError(f"malformed JSON after {retries + 1} attempts: {e}; "
                                     f"head={cleaned[:200]!r}") from e
                time.sleep(config.LABEL_SAFETY_SLEEP)
                continue
            if expect == "list" and not isinstance(parsed, list):
                raise LabelError(f"expected a JSON array, got {type(parsed).__name__}: "
                                 f"{cleaned[:200]!r}")
            if expect == "dict" and not isinstance(parsed, dict):
                raise LabelError(f"expected a JSON object, got {type(parsed).__name__}: "
                                 f"{cleaned[:200]!r}")
            return parsed

    def stats(self):
        with self._lock:
            return {"n_calls": self.n_calls, "total_tokens": self.total_tokens,
                    "cost_usd": round(self.cost_usd, 4)}


def _numbered(claims):
    return "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))


def judge(client, template, question, answer_claims, gt_claims):
    """One labelling call. -> raw forward / backward / inconsistencies plus the label maps with
    the judge's own corrections applied."""
    prompt = (template.replace("{question_text}", question)
              .replace("{numbered_gt_claims}", _numbered(gt_claims))
              .replace("{numbered_response_claims}", _numbered(answer_claims)))
    out = client.call_json(prompt, "dict") or {}
    fwd = {r["i"]: r for r in out.get("forward", []) if isinstance(r.get("i"), int)}
    bwd = {r["j"]: r for r in out.get("backward", []) if isinstance(r.get("j"), int)}
    fwd_labels = {i: str(r.get("label", "")).strip() for i, r in fwd.items()}
    bwd_labels = {j: str(r.get("label", "")).strip() for j, r in bwd.items()}
    for inc in out.get("inconsistencies", []) or []:
        corr = inc.get("corrected") or {}
        ri, gj = inc.get("response_idx"), inc.get("gt_idx")
        if isinstance(ri, int) and ri in fwd_labels and corr.get("forward"):
            fwd_labels[ri] = str(corr["forward"]).strip()
        if isinstance(gj, int) and gj in bwd_labels and corr.get("backward"):
            bwd_labels[gj] = str(corr["backward"]).strip()
    return {"forward": out.get("forward", []),
            "backward": out.get("backward", []),
            "inconsistencies": out.get("inconsistencies", []) or [],
            "fwd_labels": {str(k): v for k, v in fwd_labels.items()},
            "bwd_labels": {str(k): v for k, v in bwd_labels.items()},
            "matching_gt_indices": {str(i): (r.get("matching_gt_indices") or [])
                                    for i, r in fwd.items()}}
