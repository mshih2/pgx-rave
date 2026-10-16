"""Reasoner call (all three pipelines): one chat completion against the vLLM server."""
import logging
import re

import requests

from .. import config

log = logging.getLogger(__name__)

# Rendered in place of an empty CPIC or retrieval section, so the model reads "nothing was
# retrieved" as a fact on the page instead of filling the gap from its own knowledge.
EMPTY_SECTION = ("(EMPTY — nothing was retrieved into this section for this case. It states "
                 "nothing. Do not attribute any value, mechanism, direction, alternative agent or "
                 "monitoring item to it, and do not supply its content from your own knowledge.)")


def _clamp(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n]


def build_messages(template, question, cpic_text="", context="", user_instruction=""):
    system = template.format(context=_clamp(context, config.CONTEXT_MAX_CHARS) or EMPTY_SECTION,
                             CPIC=_clamp(cpic_text, config.CPIC_MAX_CHARS) or EMPTY_SECTION,
                             user=question or "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user_instruction}]


def clean_answer(text):
    """Answer text from a raw completion.

    Keeps what follows the LAST </think> (a model can reopen its deliberation), unless the
    completion is only a think block; strips remaining tags, simple LaTeX wrappers and '$',
    and collapses whitespace."""
    if not (text and isinstance(text, str)):
        return ""
    after = re.split(r"<\s*/\s*think\s*>", text, flags=re.IGNORECASE)
    if len(after) > 1 and after[-1].strip():
        text = after[-1]
    s = re.sub(r"<\s*think\b[^>]*>", "", text, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*think\s*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    for tag in (r'\\boxed', r'\\text', r'\\mathrm', r'\\mathbf', r'\\unit'):
        s = re.sub(tag + r'\{(.*?)\}', r'\1', s)
    s = s.replace('$', '')
    return re.sub(r'\s+', ' ', s).strip()


def generate(messages):
    """-> (raw completion, usage). Raises on a truncated or persistently empty completion."""
    payload = {
        "model": config.REASONER_MODEL,
        "messages": messages,
        "temperature": config.REASONER_TEMPERATURE,
        "top_p": config.REASONER_TOP_P,
        "max_tokens": config.REASONER_MAX_TOKENS,
    }
    if config.REASONER_SEED is not None:
        payload["seed"] = config.REASONER_SEED
    if "qwen3" in config.REASONER_MODEL.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": config.REASONER_THINKING}
        if config.REASONER_THINKING and config.REASONER_EFFORT:
            if config.REASONER_EFFORT not in ("xhigh", "medium", "low"):
                raise ValueError(f"REASONER_EFFORT must be xhigh, medium or low, "
                                 f"got {config.REASONER_EFFORT!r}")
            payload["chat_template_kwargs"]["reasoning_effort"] = config.REASONER_EFFORT

    url = config.REASONER_BASE_URL + "/chat/completions"
    text, usage = "", {"input_tokens": 0, "output_tokens": 0}
    for attempt in range(1, config.REASONER_EMPTY_RETRIES + 1):
        body = dict(payload)
        if attempt > 1:            # an empty completion is a serving glitch; retry off-greedy
            body["temperature"] = max(0.3, config.REASONER_TEMPERATURE)
        r = requests.post(url, json=body, timeout=config.REASONER_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        text = (choice["message"]["content"] or "").strip()
        u = data.get("usage") or {}
        usage = {"input_tokens": int(u.get("input_tokens") or u.get("prompt_tokens") or 0),
                 "output_tokens": int(u.get("output_tokens") or u.get("completion_tokens") or 0),
                 "finish_reason": choice.get("finish_reason")}
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"reasoner hit max_tokens={config.REASONER_MAX_TOKENS} "
                               f"(completion_tokens={usage['output_tokens']}); a truncated "
                               f"answer is not recorded")
        if text and usage["output_tokens"] > 0:
            return text, usage
        log.warning("empty completion on attempt %d/%d", attempt, config.REASONER_EMPTY_RETRIES)
    raise RuntimeError(f"reasoner returned an empty completion "
                       f"{config.REASONER_EMPTY_RETRIES} times")
