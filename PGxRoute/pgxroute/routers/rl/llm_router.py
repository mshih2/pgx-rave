"""LLM routers: Qwen3-8B reads a prompt that describes each pipeline as a tool and names one.

  fg  facet-guided: the prompt lists six fixed clinical facets (prompts/routers/fg_system.txt)
  sf  self-faceted: the model lists in its own words what the question asks for
      (prompts/routers/sf_system.txt); the three tools are listed in an order drawn per question
      from random.Random("v17:" + question), as in training

The same code serves the untrained router (ROUTER_MODEL = Qwen/Qwen3-8B) and a GRPO-trained one
(ROUTER_MODEL = the served name of its checkpoint); only the weights differ. Checkpoints are not
part of this repository.

A reply counts only if the text after the last </think> holds a JSON object whose "name" is
tool_a, tool_b or tool_c. An unparsable reply is resampled up to ROUTER_RESAMPLES times, then the
question goes to Reasoning-only.

Sampling: thinking on, temperature 0.6, top-p 0.95, 2,048 new tokens (ROUTER_* in config).
"""
import json
import logging
import random

import requests

from ... import config
from .. import TOOL_TO_PIPELINE

log = logging.getLogger(__name__)

PROMPTS = ("fg", "sf")
FALLBACK = "reasoning_only"
_cache = {}


def _read(name):
    if name not in _cache:
        _cache[name] = (config.PROMPT_DIR / "routers" / name).read_text(encoding="utf-8")
    return _cache[name]


def system_prompt(prompt, question):
    if prompt == "fg":
        return _read("fg_system.txt").rstrip("\n")
    if prompt == "sf":
        lines = json.loads(_read("sf_tool_lines.json"))
        order = ["tool_a", "tool_b", "tool_c"]
        random.Random(f"v17:{question}").shuffle(order)
        return _read("sf_system.txt").rstrip("\n").replace(
            "{tool_list}", "\n".join(lines[t] for t in order))
    raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")


def build_messages(prompt, question):
    return [{"role": "system", "content": system_prompt(prompt, question)},
            {"role": "user", "content": "## Question\n" + question}]


def parse_tool(content, reasoning=None):
    """The tool named in a reply, or None if the reply does not count.

    With a server that splits the reasoning out (a vLLM reasoning parser), `content` is already
    the text after </think>."""
    content = content or ""
    if "</think>" in content:
        tail = content.split("</think>")[-1]
    elif reasoning:
        tail = content
    else:
        return None
    i, j = tail.find("{"), tail.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(tail[i:j + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    return name if name in TOOL_TO_PIPELINE else None


class LLMRouter:
    def __init__(self, prompt="fg", base_url=None, model=None, seed=None):
        if prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")
        self.prompt = prompt
        self.name = prompt
        self.base_url = (base_url or config.ROUTER_BASE_URL).rstrip("/")
        self.model = model or config.ROUTER_MODEL
        self.seed = seed
        self.last = None          # replies and attempts of the latest call, for inspection

    def _sample(self, messages, attempt):
        body = {"model": self.model, "messages": messages,
                "temperature": config.ROUTER_TEMPERATURE, "top_p": config.ROUTER_TOP_P,
                "max_tokens": config.ROUTER_MAX_TOKENS,
                "chat_template_kwargs": {"enable_thinking": True}}
        if self.seed is not None:
            body["seed"] = self.seed + attempt
        r = requests.post(self.base_url + "/chat/completions", json=body,
                          timeout=config.ROUTER_TIMEOUT)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return msg.get("content"), msg.get("reasoning_content") or msg.get("reasoning")

    def route(self, question):
        messages = build_messages(self.prompt, question)
        replies = []
        for attempt in range(1 + config.ROUTER_RESAMPLES):
            content, reasoning = self._sample(messages, attempt)
            replies.append(content)
            tool = parse_tool(content, reasoning)
            if tool:
                self.last = {"tool": tool, "attempts": attempt + 1, "replies": replies}
                return TOOL_TO_PIPELINE[tool]
        log.warning("router reply unparsable after %d attempts; using %s", len(replies), FALLBACK)
        self.last = {"tool": None, "attempts": len(replies), "replies": replies}
        return FALLBACK
