"""Generate common questions: everyday medication questions with no PGx content.

Questions are written per everyday-medication category (prompts/common_categories.json). The
writer sees the questions already in that category so it does not repeat a drug or a wording.
A question is kept only if the structured input finds no gene in it.
"""
import logging

from .. import config, files, llm
from ..sources import cpic

log = logging.getLogger(__name__)


def category_list():
    return files.read_json(config.PROMPT_DIR / "common_categories.json")


def generate(per_category, only=None):
    made = []
    for cat in category_list():
        if only and cat["name"] != only:
            continue
        existing = [q for q in files.questions("common") if q.get("category") == cat["name"]]
        listing = "\n".join(f"- {q['question']}" for q in existing) or "(none)"
        prompt = files.prompt("common_question.txt", principles=files.principles(),
                              category=cat["name"], category_asks=cat["asks"],
                              existing=listing, n=per_category)
        obj, reply = llm.ask_json(config.QUESTION_WRITER, prompt)
        taken = {d.lower() for q in existing for d in q.get("drugs") or []}
        for item in obj.get("questions") or []:
            drugs = [d.lower() for d in item.get("drugs") or []]
            if not item.get("question") or set(drugs) & taken:
                continue
            section = cpic.cpic_section(item["question"])
            if section["genes"]:
                log.info("dropped (names a gene): %s", item["question"])
                continue
            qid = files.next_qid("common")
            q = {"qid": qid, "tier": "common", "source": "generated", "stage": "generated",
                 "category": cat["name"], "question": item["question"].strip(), "drugs": drugs,
                 "genes": [], "generated_by": reply.record()}
            files.write_json(files.qdir(qid) / "question.json", q)
            files.log_event(qid, "generate", **reply.record())
            taken |= set(drugs)
            made.append(qid)
    return made
