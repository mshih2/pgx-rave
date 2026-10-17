"""Curate an intermediate question's GT from its CPIC lookup and trim it to the question.

The GT writer receives the question and the CPIC section PGxRoute's lookup returns for it (all
rows the lookup gives for the question's drugs) and keeps, in CPIC's own words, only what the
question asks for. Sentences that are not verbatim in the CPIC section are listed in gt_meta.json
for the review and the human check.
"""
import logging
import re

from .. import config, files, llm
from ..hard.write_gt import unfence
from ..sources import cpic

log = logging.getLogger(__name__)


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def section(qid):
    path = files.qdir(qid) / "section.json"
    if path.exists():
        return files.read_json(path)
    q = files.read_json(files.qdir(qid) / "question.json")
    s = cpic.cpic_section(q["question"])
    files.write_json(path, s)
    return s


def material(qid):
    q = files.read_json(files.qdir(qid) / "question.json")
    return (f"## THE QUESTION\n\n{q['question'].strip()}\n\n## THE CPIC SECTION\n\n"
            f"```text\n{section(qid)['cpic_text'].strip() or '(empty)'}\n```")


def not_verbatim(gt, source):
    """Sentences of the GT, minus a leading 'Label:', that the source does not contain."""
    src = norm(source)
    out = []
    for line in gt.splitlines():
        body = re.sub(r"^\s*[-*]?\s*[A-Za-z][A-Za-z /&()-]{0,40}:\s*", "", line).strip()
        for sent in re.split(r"(?<=[.;])\s+", body):
            if len(sent) > 25 and norm(sent).rstrip(".;") not in src:
                out.append(sent)
    return out


def write(qid, force=False):
    out = files.qdir(qid) / "gt.md"
    if out.exists() and not force:
        return files.read_text(out)
    s = section(qid)
    if not s["cpic_text"].strip():
        raise RuntimeError(f"{qid}: the lookup returns nothing for this question")
    prompt = files.prompt("intermediate_gt.txt", principles=files.principles(),
                          material=material(qid))
    reply = llm.ask(config.GT_WRITER, prompt)
    gt = unfence(reply.text)
    files.keep_version(out)
    files.write_text(out, gt)
    files.write_json(files.qdir(qid) / "gt_meta.json",
                     {"not_verbatim": not_verbatim(gt, s["cpic_text"]), **reply.record()})
    files.log_event(qid, "write_gt", **reply.record())
    q = files.read_json(files.qdir(qid) / "question.json")
    q["stage"] = "gt_written"
    files.write_json(files.qdir(qid) / "question.json", q)
    return gt
