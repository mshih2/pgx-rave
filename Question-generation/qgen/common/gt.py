"""Reach a common question's GT through a discussion between the two LLMs.

config.COMMON_DRAFTER writes the answer; config.COMMON_CRITIC checks it and either agrees or
lists the changes it wants, each with its reason; the drafter accepts or rebuts each change and
returns a revised answer. The exchange ends when the critic agrees, or after
config.COMMON_GT_ROUNDS rounds, in which case the question is marked for the human check with
the open points. The whole exchange is saved as discussion.json.
"""
import json
import logging

from .. import config, files, llm
from ..hard.write_gt import unfence

log = logging.getLogger(__name__)


def material(qid):
    q = files.read_json(files.qdir(qid) / "question.json")
    return f"## THE QUESTION\n\n{q['question'].strip()}"


def write(qid, force=False):
    out = files.qdir(qid) / "gt.md"
    if out.exists() and not force:
        return files.read_text(out)
    q = files.read_json(files.qdir(qid) / "question.json")
    reply = llm.ask(config.COMMON_DRAFTER, files.prompt(
        "common_gt_draft.txt", principles=files.principles(), question=q["question"]))
    answer = unfence(reply.text)
    turns = [{"role": "draft", "answer": answer, **reply.record()}]
    agreed = False
    for _ in range(config.COMMON_GT_ROUNDS):
        crit, r1 = llm.ask_json(config.COMMON_CRITIC, files.prompt(
            "common_gt_critique.txt", principles=files.principles(), question=q["question"],
            answer=answer))
        turns.append({"role": "critique", **crit, **r1.record()})
        if crit.get("agree") and not crit.get("changes"):
            agreed = True
            break
        rev, r2 = llm.ask_json(config.COMMON_DRAFTER, files.prompt(
            "common_gt_revise.txt", principles=files.principles(), question=q["question"],
            answer=answer, critique=json.dumps(crit.get("changes") or [], ensure_ascii=False, indent=1)))
        answer = unfence(rev.get("answer") or answer)
        turns.append({"role": "revision", **rev, **r2.record()})
    files.keep_version(out)
    files.write_text(out, answer)
    files.write_json(files.qdir(qid) / "discussion.json", {"agreed": agreed, "turns": turns})
    files.log_event(qid, "write_gt", agreed=agreed, rounds=len(turns))
    q["stage"] = "gt_written"
    if not agreed:
        q["open_points"] = turns[-1].get("changes") or turns[-2].get("changes") or []
    files.write_json(files.qdir(qid) / "question.json", q)
    return answer
