"""Cross-review of each new question and its GT, before human curation.

Two LLM readers (config.READERS) each review the question and its GT against the material the GT
was built from, independently and with byte-identical prompts. A settler (config.SETTLER) then
reads both reports with the material:

  - a finding is acted on only if both readers made it independently (the same span, even at a
    different severity or label); a finding from one reader is recorded, not acted on;
  - the settler may still overrule a corroborated finding when the material shows it is wrong,
    and must not order a repair the material cannot support.

The settled changes go back to the GT writer, which applies them and nothing else. This repeats
up to config.REVIEW_ROUNDS times or until no change is settled. Every round is kept under
review_r<n>/.
"""
import json
import logging

from . import config, files, llm
from .common import gt as common_gt
from .hard import write_gt as hard_gt
from .intermediate import gt as intermediate_gt

log = logging.getLogger(__name__)

PROMPTS = {"hard": "review_gt_cpic.txt", "intermediate": "review_gt_cpic.txt",
           "common": "review_gt_common.txt"}


def material(q):
    return {"hard": hard_gt.material, "intermediate": intermediate_gt.material,
            "common": common_gt.material}[q["tier"]](q["qid"])


def _strip_meta(obj):
    return {k: v for k, v in obj.items() if k not in ("requested_model", "served_by", "usage")}


def review_round(qid, n):
    q = files.read_json(files.qdir(qid) / "question.json")
    d = files.qdir(qid) / f"review_r{n}"
    mat, answer = material(q), files.read_text(files.qdir(qid) / "gt.md")
    prompt = files.prompt(PROMPTS[q["tier"]], principles=files.principles(), material=mat,
                          answer=answer)
    reports = {}
    for model, (obj, reply) in llm.ask_json_each(config.READERS, prompt).items():
        files.write_json(d / f"reader_{files.tag(model)}.json", {**obj, **reply.record()})
        files.log_event(qid, f"review_r{n}", **reply.record())
        reports[model] = obj
    readers = "\n\n".join(f"### Reader {i}\n```json\n{json.dumps(_strip_meta(r), ensure_ascii=False, indent=1)}\n```"
                          for i, r in enumerate(reports.values(), 1))
    settled, reply = llm.ask_json(config.SETTLER, files.prompt(
        "settle.txt", material=mat, answer=answer, readers=readers))
    files.write_json(d / "settled.json", {**settled, **reply.record()})
    files.log_event(qid, f"settle_r{n}", **reply.record())
    return q, mat, answer, settled


def revise(q, mat, answer, settled):
    prompt = files.prompt("gt_revise.txt", principles=files.principles(), material=mat,
                          answer=answer,
                          changes=json.dumps(settled.get("actions") or [], ensure_ascii=False, indent=1))
    reply = llm.ask(config.GT_WRITER, prompt)
    out = files.qdir(q["qid"]) / "gt.md"
    files.keep_version(out)
    files.write_text(out, hard_gt.unfence(reply.text))
    files.log_event(q["qid"], "revise_gt", **reply.record())


def _refresh_section(q):
    """After a question edit, look its CPIC section up again (the genotype may have changed)."""
    if q["tier"] == "common":
        return
    from .sources import cpic
    s = cpic.cpic_section(q["question"])
    files.write_json(files.qdir(q["qid"]) / "section.json", s)
    if q["tier"] == "hard":
        b = files.read_json(files.qdir(q["qid"]) / "bundle.json")
        b.update(question=q["question"], cpic_text=s["cpic_text"])
        files.write_json(files.qdir(q["qid"]) / "bundle.json", b)


def review(qid, rounds=None):
    for n in range(1, (rounds or config.REVIEW_ROUNDS) + 1):
        q, mat, answer, settled = review_round(qid, n)
        change = (settled.get("question_change") or {}).get("new_question")
        if change:
            files.keep_version(files.qdir(qid) / "question.json")
            q["question"] = change.strip()
            q.setdefault("question_changes", []).append(
                {"round": n, "why": settled["question_change"].get("why")})
            files.write_json(files.qdir(qid) / "question.json", q)
            _refresh_section(q)
            mat = material(q)
        if not settled.get("actions") and not change:
            break
        if change:
            settled.setdefault("actions", []).append(
                {"change": f"The question was edited and now reads: {q['question']} "
                           f"Make the answer answer it as it now stands, with the least change."})
        revise(q, mat, answer, settled)
    q = files.read_json(files.qdir(qid) / "question.json")
    q["stage"] = "reviewed"
    files.write_json(files.qdir(qid) / "question.json", q)
