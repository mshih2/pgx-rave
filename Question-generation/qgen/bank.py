"""The question bank: pharmacist example pairs in, human check sheets out, bank and split.

  import_examples  pharmacist-written pairs (JSONL of {"question", "answer"}) -> hard questions
                   with source "pharmacist"; their answers are kept as GT and go through claims
  sheet            one Markdown sheet per tier for the authors' word-by-word check
  mark_checked     record that a question, its GT and its claims were checked by hand
  split            tuning / evaluation split and two folds of the evaluation questions
  assemble         -> WORK_DIR/bank.jsonl, the format Router-training/scripts/build_data.py reads
"""
import logging
import random
import re
from collections import defaultdict

from . import config, files
from .intermediate.generate import gene_family
from .sources import cpic

log = logging.getLogger(__name__)

TUNING_PER_TIER = 10
COMMON_TUNING_CATEGORIES = 5


def import_examples(path):
    made = []
    for i, row in enumerate(files.read_jsonl(path), 1):
        qid = row.get("qid") or f"p{i:02d}"
        section = cpic.cpic_section(row["question"])
        q = {"qid": qid, "tier": "hard", "source": "pharmacist", "stage": "reviewed",
             "question": row["question"].strip(), "genes": section["genes"],
             "drugs": section["drugs"]}
        files.write_json(files.qdir(qid) / "question.json", q)
        files.write_json(files.qdir(qid) / "section.json", section)
        files.write_text(files.qdir(qid) / "gt.md", row["answer"].strip())
        made.append(qid)
    log.info("%d example pairs imported", len(made))
    return made


def sheet(tier):
    out = [f"# {tier} questions: word-by-word check\n",
           "Edit question.json, gt.md or claims.json in the question's folder, then run "
           "`python run.py mark-checked <qid>`.\n"]
    for q in files.questions(tier):
        d = files.qdir(q["qid"])
        claims = files.read_json(d / "claims.json", default={}).get("claims") or []
        out.append(f"\n## {q['qid']}  ({q.get('source')}, stage {q.get('stage')})\n")
        out.append(f"- [{'x' if q.get('checked') else ' '}] question, GT and claims checked\n")
        out.append(f"**Question**\n\n{q['question']}\n")
        if (d / "gt.md").exists():
            out.append(f"**GT**\n\n{files.read_text(d / 'gt.md').strip()}\n")
        if claims:
            out.append("**Claims**\n\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(claims)) + "\n")
        for r in sorted(d.glob("review_r*/settled.json")):
            s = files.read_json(r)
            notes = [a.get("change") for a in s.get("actions") or []] + \
                    [f"(one reader) {x.get('finding') or x}" for x in s.get("single_reader") or []]
            if notes:
                out.append(f"**Review {r.parent.name}**\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
        curation = files.read_json(d / "claims.json", default={}).get("rounds") or []
        notes = [n for r in curation for n in r.get("notes_for_human_check") or []]
        if notes:
            out.append("**Notes from claim curation**\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
        if q.get("open_points"):
            out.append("**Open points from the GT discussion**\n\n" +
                       "\n".join(f"- {p}" for p in q["open_points"]) + "\n")
    path = files.work("sheets", f"{tier}.md")
    files.write_text(path, "\n".join(out))
    return path


def mark_checked(qid, by=""):
    q = files.read_json(files.qdir(qid) / "question.json")
    q["checked"] = True
    q["checked_by"] = by
    q["stage"] = "checked"
    files.write_json(files.qdir(qid) / "question.json", q)


def _phenotype(qid):
    s = files.read_json(files.qdir(qid) / "section.json", default={})
    m = re.search(r"Phenotype:\s*(?:[A-Za-z0-9-]+:\s*)?([^;\n]+)", s.get("cpic_text") or "")
    return m.group(1).strip().lower() if m else "unknown"


def _spread(groups, k, rng):
    """k items taken one per group in turn, groups in random order."""
    order = sorted(groups)
    rng.shuffle(order)
    for g in order:
        rng.shuffle(groups[g])
    picked = []
    while len(picked) < k and any(groups[g] for g in order):
        for g in order:
            if groups[g] and len(picked) < k:
                picked.append(groups[g].pop())
    return picked


def split():
    """Ten tuning questions per tier: hard ones spread over gene-variant categories (gene family
    and phenotype), intermediate ones one per gene family, common ones two from each of five
    everyday-medication categories. The rest are evaluation questions, in two folds per tier."""
    rng = random.Random(config.SEED)
    tuning = []
    hard, inter, common = (defaultdict(list) for _ in range(3))
    for q in files.questions("hard"):
        hard[(gene_family((q.get("genes") or ["?"])[0]), _phenotype(q["qid"]))].append(q["qid"])
    for q in files.questions("intermediate"):
        inter[gene_family((q.get("genes") or ["?"])[0])].append(q["qid"])
    for q in files.questions("common"):
        common[q.get("category")].append(q["qid"])
    tuning += _spread(hard, TUNING_PER_TIER, rng)
    tuning += _spread(inter, TUNING_PER_TIER, rng)
    cats = sorted(common, key=lambda c: (-len(common[c]), c))[:COMMON_TUNING_CATEGORIES]
    tuning += _spread({c: common[c] for c in cats}, TUNING_PER_TIER, rng)
    out = {}
    for tier in files.TIERS:
        rest = [q["qid"] for q in files.questions(tier) if q["qid"] not in tuning]
        rng.shuffle(rest)
        for i, qid in enumerate(rest):
            out[qid] = {"split": "eval", "fold": i % 2}
    for qid in tuning:
        out[qid] = {"split": "tuning", "fold": None}
    files.write_json(files.work("split.json"), out)
    return out


def assemble(include_unchecked=False):
    split_of = files.read_json(files.work("split.json"), default={})
    rows, skipped = [], []
    for q in files.questions():
        d = files.qdir(q["qid"])
        if not (d / "claims.json").exists() or not (include_unchecked or q.get("checked")):
            skipped.append(q["qid"])
            continue
        s = split_of.get(q["qid"], {})
        row = {"question_id": q["qid"], "tier": q["tier"], "question": q["question"],
               "ground_truth": files.read_text(d / "gt.md").strip(),
               "gt_claims": files.read_json(d / "claims.json")["claims"],
               "category": q.get("category") or q.get("target_category") or q["tier"],
               "source": q.get("source"), "split": s.get("split")}
        if s.get("fold") is not None:
            row["fold"] = s["fold"]
        rows.append(row)
    files.write_jsonl(files.work("bank.jsonl"), rows)
    if skipped:
        log.info("left out (no claims yet, or not checked by hand): %s", skipped)
    return rows
