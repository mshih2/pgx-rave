"""Generate intermediate questions: answerable from one CPIC recommendation lookup.

Candidate rows are drawn one gene family at a time, so the questions spread across gene
families. The question writer is shown the construction principles, the six clinical question
facets, the pharmacist example pairs (the model of what a question states and asks) and one
CPIC row with its example diplotypes.

A question is kept only if the structured input recognises its gene and drug and the lookup it
produces returns the row it was written from.
"""
import logging
import random
import re
from collections import defaultdict

from .. import config, files, llm
from ..sources import categories, cpic

log = logging.getLogger(__name__)

MAX_TRIES = 5                    # rows tried per question


def gene_family(gene):
    """CYP2C9 and CYP2C19 -> CYP2C; HLA-A and HLA-B -> HLA; any other gene is its own family."""
    g = gene.upper()
    if g.startswith("HLA-"):
        return "HLA"
    m = re.match(r"(CYP\d+[A-Z])\d+", g)
    if m:
        return m.group(1)
    m = re.match(r"(UGT\d+[A-Z])\d+", g)
    return m.group(1) if m else g


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _try(cand):
    prompt = files.prompt("intermediate_question.txt", principles=files.principles(),
                          facets=files.facets_text(), examples=files.examples_text(),
                          candidate=cpic.render_candidate(cand))
    obj, reply = llm.ask_json(config.QUESTION_WRITER, prompt)
    if obj.get("skip"):
        return None, f"writer declined: {obj['skip']}"
    section = cpic.cpic_section(obj["question"])
    rec = norm(cand["row"].get("drugrecommendation"))
    if not section["cpic_text"] or rec[:120] not in norm(section["cpic_text"]):
        return None, "the question's lookup does not return the row it was written from"
    return {"question": obj["question"].strip(), "asks": obj.get("asks") or [],
            "tags": {"facets": obj.get("facets") or [], "population": obj.get("population"),
                     "answer_direction": obj.get("answer_direction")},
            "section": section, "generated_by": reply.record()}, None


def generate(n):
    rng = random.Random(config.SEED)
    by_family = defaultdict(list)
    for c in cpic.candidates():
        if any(c["example_diplotypes"].values()) and c["row"].get("drugrecommendation"):
            by_family[gene_family(c["genes"][0])].append(c)
    families = sorted(by_family)
    rng.shuffle(families)
    made = []
    for i in range(n):
        fam = families[i % len(families)]
        pool = by_family[fam]
        rng.shuffle(pool)
        tries = 0
        for cand in pool:
            if tries == MAX_TRIES:
                break
            if categories.collides("intermediate", cand["genes"], cand["drug"], cand_id=cand["cand_id"]):
                continue
            tries += 1
            got, why = _try(cand)
            if got is None:
                log.info("%s / %s: %s", cand["cand_id"], cand["drug"], why)
                continue
            qid = files.next_qid("intermediate")
            section = got.pop("section")
            q = {"qid": qid, "tier": "intermediate", "source": "generated", "stage": "generated",
                 "cand_id": cand["cand_id"], "gene_family": fam,
                 "genes": section["genes"], "drugs": section["drugs"], **got}
            files.write_json(files.qdir(qid) / "question.json", q)
            files.write_json(files.qdir(qid) / "section.json", section)
            files.log_event(qid, "generate", **q["generated_by"])
            log.info("%s: %s", qid, q["question"][:100])
            made.append(qid)
            break
        else:
            log.warning("no row of gene family %s produced a question", fam)
    return made
