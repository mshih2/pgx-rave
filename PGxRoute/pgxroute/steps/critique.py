"""Critique step of the Full pipeline (Gemma-4-31B-it).

Reviews the reasoner's draft against the CPIC API output and the retrieved units: keeps what
they support, corrects or removes what they do not, and adds source content the draft missed.
If every attempt fails, the caller keeps the draft.
"""
import json
import time
import urllib.request

from .. import config

_PROMPT = None
NO_CPIC_ROW = "(the CPIC recommendation lookup returned no row for this patient's genotype)"


def critique(question, cpic_text, context, draft):
    """-> (revised answer or None, usage)."""
    global _PROMPT
    if _PROMPT is None:
        _PROMPT = (config.PROMPT_DIR / "full_critique.txt").read_text(encoding="utf-8")
    cpic_text = cpic_text.strip() or NO_CPIC_ROW
    prompt = _PROMPT.format(q=question.strip(), cpic=cpic_text[:config.CPIC_MAX_CHARS],
                            ext=context[:config.CONTEXT_MAX_CHARS], draft=draft.strip())
    body = json.dumps({"model": config.CRITIC_MODEL, "temperature": config.CRITIC_TEMPERATURE,
                       "max_tokens": config.CRITIC_MAX_TOKENS,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for _ in range(config.CRITIC_ATTEMPTS):
        try:
            req = urllib.request.Request(config.CRITIC_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=config.CRITIC_TIMEOUT))
            u = d.get("usage") or {}
            return (d["choices"][0]["message"]["content"].strip(),
                    {"input_tokens": int(u.get("prompt_tokens") or 0),
                     "output_tokens": int(u.get("completion_tokens") or 0)})
        except Exception:
            time.sleep(4)
    return None, {"input_tokens": 0, "output_tokens": 0}
