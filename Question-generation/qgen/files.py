"""Work-directory layout, JSON helpers and prompt rendering.

  WORK_DIR/
    cpic_snapshot/            CPIC API tables used for candidates
    candidates.jsonl          one CPIC recommendation row per line
    unit_categories.jsonl     information categories of every corpus unit
    registry.json             tags of the questions already in the bank
    coverage.json             questions per information category
    questions/<qid>/          everything one question goes through (question.json, gt.md, ...)
    bank.jsonl                the assembled question bank
"""
import json
import os
import re
import tempfile
import time
from pathlib import Path

from . import config

TIERS = ("hard", "intermediate", "common")


def work(*parts):
    return config.WORK_DIR.joinpath(*parts)


def qdir(qid):
    return work("questions", str(qid))


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        if default is not None:
            return default
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def write_text(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def log_event(qid, step, **info):
    """Append one line to the question's log.jsonl (which model ran which step, and when)."""
    p = qdir(qid) / "log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, **info},
                           ensure_ascii=False) + "\n")


def keep_version(path):
    """Copy path to path.v<n> before it is overwritten, so earlier versions stay readable."""
    p = Path(path)
    if not p.exists():
        return
    n = 1
    while p.with_name(f"{p.stem}.v{n}{p.suffix}").exists():
        n += 1
    p.with_name(f"{p.stem}.v{n}{p.suffix}").write_text(p.read_text(encoding="utf-8"),
                                                       encoding="utf-8")


def tag(model):
    """A model ID as a file-name part."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model)


def questions(tier=None):
    """question.json of every question in the work directory (optionally one tier), by qid."""
    out = []
    root = work("questions")
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        q = d / "question.json"
        if q.exists():
            obj = read_json(q)
            if tier is None or obj["tier"] == tier:
                out.append(obj)
    return out


def next_qid(tier):
    prefix = {"hard": "h", "intermediate": "i", "common": "c"}[tier]
    taken = {q["qid"] for q in questions()}
    n = 1
    while f"{prefix}{n:03d}" in taken:
        n += 1
    return f"{prefix}{n:03d}"


# ---------------------------------------------------------------- prompts

def prompt(name, **fields):
    """prompts/<name> with every {{FIELD}} replaced. A placeholder left unfilled raises."""
    text = read_text(config.PROMPT_DIR / name)
    for k, v in fields.items():
        text = text.replace("{{" + k.upper() + "}}", str(v))
    left = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", text)))
    if left:
        raise KeyError(f"{name}: unfilled placeholders {left}")
    return text


def principles():
    return read_text(config.PROMPT_DIR / "principles.md").strip()


def facets():
    return read_json(config.PROMPT_DIR / "facets.json")


def facets_text():
    return "\n".join(f"- {f['name']}: {f['covers']}" for f in facets())


def taxonomy():
    return read_json(config.PROMPT_DIR / "categories.json")


def examples_text(limit=None):
    """The pharmacist-written pairs, as the model questions and answers of every generation and
    GT-writing prompt."""
    ex = [q for q in questions("hard") if q.get("source") == "pharmacist"]
    if not ex:
        raise RuntimeError("no pharmacist example pairs in the work directory; "
                           "run `python run.py import-examples <file>` first")
    blocks = []
    for i, q in enumerate(ex[:limit] if limit else ex, 1):
        gt = read_text(qdir(q["qid"]) / "gt.md").strip()
        blocks.append(f"### Example {i}\n\nQuestion:\n{q['question'].strip()}\n\nAnswer:\n{gt}")
    return "\n\n".join(blocks)
