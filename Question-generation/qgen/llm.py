"""Calls to Claude (Messages API, streamed). Credentials come from the environment
(ANTHROPIC_API_KEY, or an `ant auth login` profile)."""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic

from . import config

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client = None
_lock = threading.Lock()


class Refused(RuntimeError):
    """The model declined the request (stop_reason "refusal")."""


@dataclass
class Reply:
    text: str
    requested: str
    model: str                                   # the model that served the call
    usage: dict = field(default_factory=dict)

    def record(self):
        return {"requested_model": self.requested, "served_by": self.model, "usage": self.usage}


def client():
    global _client
    with _lock:
        if _client is None:
            _client = anthropic.Anthropic(max_retries=4, timeout=3600)
    return _client


def ask(model, prompt, system=None, effort=None, max_tokens=None):
    """One user turn -> Reply. Raises Refused on a refusal and RuntimeError on truncation."""
    kw = {"model": model, "max_tokens": max_tokens or config.MAX_TOKENS,
          "messages": [{"role": "user", "content": prompt}],
          "thinking": {"type": "adaptive"},
          "output_config": {"effort": effort or config.EFFORT}}
    if system:
        kw["system"] = system
    if config.FALLBACKS and model.startswith("claude-fable"):
        stream = client().beta.messages.stream(betas=[FALLBACK_BETA], fallbacks="default", **kw)
    else:
        stream = client().messages.stream(**kw)
    with stream as s:
        msg = s.get_final_message()
    if msg.stop_reason == "refusal":
        details = getattr(msg, "stop_details", None)
        raise Refused(f"{model} declined the request: {getattr(details, 'category', None)}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError(f"{model} reply hit max_tokens ({kw['max_tokens']})")
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if msg.model != model:
        log.warning("%s was served by %s", model, msg.model)
    u = msg.usage
    return Reply(text, model, msg.model, {"input_tokens": u.input_tokens,
                                          "output_tokens": u.output_tokens})


def extract_json(text):
    """The JSON object or list in a reply, with or without a code fence around it."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in the reply")
    start = min(starts)
    end = t.rfind("}" if t[start] == "{" else "]")
    return json.loads(t[start:end + 1])


def ask_json(model, prompt, system=None, effort=None, tries=2):
    """Like ask(), for a prompt that asks for JSON only -> (parsed, Reply). Re-asks once on a
    reply that does not parse."""
    last = None
    for _ in range(tries):
        reply = ask(model, prompt, system=system, effort=effort)
        try:
            return extract_json(reply.text), reply
        except ValueError as e:                  # json.JSONDecodeError is a ValueError
            last = e
            log.warning("%s: reply did not parse as JSON (%s); asking again", model, e)
    raise RuntimeError(f"{model}: no parsable JSON after {tries} tries: {last}")


def ask_json_each(models, prompt):
    """The same prompt to each model independently and in parallel -> {model: (parsed, Reply)}.
    No model sees another's reply."""
    with ThreadPoolExecutor(len(models)) as ex:
        futures = {m: ex.submit(ask_json, m, prompt) for m in models}
        return {m: f.result() for m, f in futures.items()}
