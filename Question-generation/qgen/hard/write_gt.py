"""Write a hard question's GT from its CPIC section and its final passages (config.GT_WRITER).

The writer is given the construction principles and the pharmacist example pairs as the model of
format and style, the question, the CPIC section, the final passages in full and the converge
step's notes. Content may come only from the CPIC section and the passages.

The exact prompt is saved as gt_prompt.md beside gt.md; the review reads the same material.
"""
import json
import logging

from .. import config, files, llm
from . import select

log = logging.getLogger(__name__)

NOTE_TITLES = [
    ("api_only", "Answered by the CPIC section alone; state it, do not attribute it to a passage"),
    ("sole_sources", "Single-support facts; dropping one leaves that ask unanswered"),
    ("conflicts_to_name", "Printed entries that differ; name both, do not average them"),
    ("silence_traps", "Spans that tempt an assertion the sources do not print; do not make it"),
    ("column_traps", "Values that belong to a different column or row than they appear to"),
    ("must_not_say", "Claims the sources do not support"),
]


def material(qid):
    """The question, CPIC section and final passages, as the writer and the reviewers read them.
    A pharmacist pair has no passage selection, so all its in-scope passages are given."""
    b = select.bundle(qid)
    final = files.qdir(qid) / "final.json"
    if not final.exists():
        return select.bundle_text(b)
    return select.bundle_text(b, ids={m["id"] for m in files.read_json(final)["final"]})


def _notes(final):
    h = final.get("for_the_gt_writer") or {}
    out = []
    for key, title in NOTE_TITLES:
        items = h.get(key) or []
        if items:
            out.append(f"**{title}**\n" + "\n".join(
                f"- {json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x}"
                for x in items))
    return "\n\n".join(out) or "(none)"


def unfence(text):
    """Remove a code fence wrapping the whole reply (and only then)."""
    lines = text.strip().split("\n")
    if lines and lines[0].lstrip().startswith("```") and len(lines) > 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def build_prompt(qid):
    final = files.read_json(files.qdir(qid) / "final.json")
    return files.prompt("gt_writer.txt", principles=files.principles(),
                        examples=files.examples_text(), material=material(qid),
                        notes=_notes(final))


def write(qid, force=False):
    out = files.qdir(qid) / "gt.md"
    if out.exists() and not force:
        log.info("%s: gt.md exists (use force to rewrite)", qid)
        return files.read_text(out)
    prompt = build_prompt(qid)
    reply = llm.ask(config.GT_WRITER, prompt)
    files.keep_version(out)
    files.write_text(out, unfence(reply.text))
    files.write_text(files.qdir(qid) / "gt_prompt.md", prompt)
    files.log_event(qid, "write_gt", **reply.record())
    q = files.read_json(files.qdir(qid) / "question.json")
    q["stage"] = "gt_written"
    files.write_json(files.qdir(qid) / "question.json", q)
    return files.read_text(out)
