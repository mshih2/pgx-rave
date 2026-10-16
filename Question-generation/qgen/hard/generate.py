"""Generate hard questions for sparsely covered information categories.

For each target category (least covered first), a CPIC recommendation row is drawn whose gene
and drug have current guideline or supplement text in that category and whose (gene, drug) pair
no hard question has taken. The question writer then sees the construction principles, the six
clinical question facets, the pharmacist example pairs, the row and the category's passages, and
writes a question that needs at least one point only those passages carry.

A question is kept only if the passage quote it names is verbatim in that passage, the structured
input recognises its gene and drug, and it does not collide with a question already in the bank.
"""
import logging
import random
import re

from .. import config, files, llm
from ..sources import categories, corpus, cpic

log = logging.getLogger(__name__)

MAX_CATEGORY_PASSAGES = 25
MAX_TRIES = 5                    # candidate rows tried per target category


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _passages(units):
    return "\n\n".join(f"### {u['uid']}\n```text\n{u['text'].strip()}\n```" for u in units)


def _try(category, cand, units):
    tax = {c["name"]: c for c in files.taxonomy()}
    prompt = files.prompt(
        "hard_question.txt", principles=files.principles(), facets=files.facets_text(),
        examples=files.examples_text(), category=category,
        category_definition=tax[category]["definition"],
        categories="\n".join(f"- {n}" for n in tax), candidate=cpic.render_candidate(cand),
        passages=_passages(units))
    obj, reply = llm.ask_json(config.QUESTION_WRITER, prompt)
    if obj.get("skip"):
        return None, f"writer declined: {obj['skip']}"
    point = obj.get("corpus_only_point") or {}
    unit = next((u for u in units if u["uid"] == point.get("uid")), None)
    if unit is None or not norm(point.get("quote")) or norm(point.get("quote")) not in norm(unit["text"]):
        return None, "corpus-only point is not a verbatim quote of the named passage"
    section = cpic.cpic_section(obj["question"])
    if not set(cand["genes"]) & set(section["genes"]) or \
            cand["drug"].lower() not in [d.lower() for d in section["drugs"]]:
        return None, (f"structured input found genes={section['genes']} drugs={section['drugs']}, "
                      f"not the row's gene and drug")
    cats = [c for c in obj.get("categories") or [] if c in tax]
    tags = {"categories": sorted(set(cats) | {category}), "facets": obj.get("facets") or [],
            "answer_direction": obj.get("answer_direction"), "population": obj.get("population")}
    why = categories.collides("hard", section["genes"], cand["drug"], tags=tags)
    if why:
        return None, why
    return {"question": obj["question"].strip(), "asks": obj.get("asks") or [],
            "corpus_only_point": point, "tags": tags, "section": section,
            "generated_by": reply.record()}, None


def generate(n, only_category=None):
    """Write up to n new hard questions, one target category at a time."""
    cov = categories.coverage()
    order = [only_category] if only_category else categories.sparse(cov)
    labels = categories.unit_categories()
    rng = random.Random(config.SEED)
    pool = [c for c in cpic.candidates() if c.get("corpus_docs")]
    made = []
    for i in range(n):
        category = order[i % len(order)]
        rng.shuffle(pool)
        tries = 0
        for cand in pool:
            if tries == MAX_TRIES:
                break
            if categories.collides("hard", cand["genes"], cand["drug"]):
                continue
            units = [u for u in corpus.scope(cand["genes"], [cand["drug"]])
                     if category in (labels.get(u["uid"]) or {}).get("categories", [])]
            if not units:
                continue
            tries += 1
            got, why = _try(category, cand, units[:MAX_CATEGORY_PASSAGES])
            if got is None:
                log.info("%s / %s+%s: %s", category, cand["drug"], "+".join(cand["genes"]), why)
                continue
            qid = files.next_qid("hard")
            section = got.pop("section")
            q = {"qid": qid, "tier": "hard", "source": "generated", "stage": "generated",
                 "cand_id": cand["cand_id"], "target_category": category,
                 "genes": section["genes"], "drugs": section["drugs"], **got}
            files.write_json(files.qdir(qid) / "question.json", q)
            files.write_json(files.qdir(qid) / "section.json", section)
            files.log_event(qid, "generate", **q["generated_by"])
            log.info("%s: %s", qid, q["question"][:100])
            made.append(qid)
            break
        else:
            log.warning("no candidate produced a question for category %s", category)
    return made
