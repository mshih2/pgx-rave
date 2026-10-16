"""Question and GT generation, step by step. Run from this folder: python run.py <step> ...

  Sources (once)
    snapshot                      fetch the CPIC API tables used for candidates
    candidates                    CPIC recommendation rows -> candidates
    categorize                    sort guideline and supplement text into information categories
    import-examples FILE          pharmacist-written example pairs (JSONL: question, answer)
    tag                           tag bank questions with categories, facets, answer direction
    coverage                      questions per information category; sparse categories first

  Questions and GT
    generate --tier T --n N       new questions (hard: sparse categories; common: N per category)
    select QID...                 hard: two blind passage selections, then converge
    write-gt QID...               write (hard), curate (intermediate) or discuss (common) the GT
    review QID...                 two LLM readers and a settler review the question and GT
    claims QID...                 decompose, coverage review, curation of the GT claims
    pipeline --tier T [QID...]    every remaining step for each question, in order

  Human check and bank
    sheet --tier T                Markdown sheet for the word-by-word check
    mark-checked QID --by NAME    record the human check
    split                         tuning / evaluation split and folds
    bank [--include-unchecked]    -> work/bank.jsonl

Instead of QIDs, --tier T selects every question of that tier that has reached the step.
"""
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

from qgen import bank, claims, config, files, review
from qgen.common import generate as common_generate, gt as common_gt
from qgen.hard import generate as hard_generate, select, write_gt as hard_gt
from qgen.intermediate import generate as intermediate_generate, gt as intermediate_gt
from qgen.sources import categories, cpic

log = logging.getLogger("qgen")

# step -> the stage a question must be at for the step to run on it
NEEDS = {"select": ("generated",), "write-gt": ("generated", "passages_selected"),
         "review": ("gt_written",), "claims": ("reviewed",)}


def write_gt(qid, force=False):
    tier = files.read_json(files.qdir(qid) / "question.json")["tier"]
    return {"hard": hard_gt.write, "intermediate": intermediate_gt.write,
            "common": common_gt.write}[tier](qid, force=force)


def pipeline(qid):
    q = files.read_json(files.qdir(qid) / "question.json")
    if q["tier"] == "hard" and q["stage"] == "generated":
        select.run(qid)
    q = files.read_json(files.qdir(qid) / "question.json")
    if q["stage"] in ("generated", "passages_selected"):
        write_gt(qid)
    if files.read_json(files.qdir(qid) / "question.json")["stage"] == "gt_written":
        review.review(qid)
    if files.read_json(files.qdir(qid) / "question.json")["stage"] == "reviewed":
        claims.run(qid)


STEPS = {"select": select.run, "write-gt": lambda q: write_gt(q, force=True),
         "review": review.review, "claims": claims.run, "pipeline": pipeline}


def targets(args, step):
    if args.qids:
        return args.qids
    if not args.tier:
        sys.exit("give QIDs or --tier")
    qs = files.questions(args.tier)
    if step == "select":
        qs = [q for q in qs if q["tier"] == "hard"]
    if step in NEEDS:
        qs = [q for q in qs if q.get("stage") in NEEDS[step]]
    if step == "write-gt":                       # a hard GT needs its passages selected first
        qs = [q for q in qs if not (q["tier"] == "hard" and q["stage"] == "generated")]
    return [q["qid"] for q in qs if not q.get("checked")]


def for_each(fn, qids, workers):
    def one(qid):
        try:
            fn(qid)
            return qid, None
        except Exception as e:                   # one question's failure does not stop the rest
            log.exception("%s failed", qid)
            return qid, e
    with ThreadPoolExecutor(workers) as ex:
        failed = [q for q, e in ex.map(one, qids) if e]
    log.info("%d done, %d failed%s", len(qids) - len(failed), len(failed),
             f": {failed}" if failed else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in ("snapshot", "candidates", "categorize", "tag", "coverage", "split"):
        sub.add_parser(name)
    p = sub.add_parser("import-examples")
    p.add_argument("file")
    p = sub.add_parser("generate")
    p.add_argument("--tier", required=True, choices=files.TIERS)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--category", help="hard: one information category; common: one category")
    for name in STEPS:
        p = sub.add_parser(name)
        p.add_argument("qids", nargs="*")
        p.add_argument("--tier", choices=files.TIERS)
    p = sub.add_parser("sheet")
    p.add_argument("--tier", required=True, choices=files.TIERS)
    p = sub.add_parser("mark-checked")
    p.add_argument("qid")
    p.add_argument("--by", default="")
    p = sub.add_parser("bank")
    p.add_argument("--include-unchecked", action="store_true")
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.step == "snapshot":
        cpic.snapshot()
    elif args.step == "candidates":
        cpic.build_candidates()
    elif args.step == "categorize":
        categories.categorize(args.workers)
    elif args.step == "import-examples":
        bank.import_examples(args.file)
    elif args.step == "tag":
        categories.tag(args.workers)
    elif args.step == "coverage":
        for name, v in categories.coverage().items():
            print(f"{name:45s} units {v['units']:5d}  hard questions {v['hard_questions']:3d}")
        print("\nsparse first:", ", ".join(categories.sparse()))
    elif args.step == "generate":
        made = {"hard": lambda: hard_generate.generate(args.n, args.category),
                "intermediate": lambda: intermediate_generate.generate(args.n),
                "common": lambda: common_generate.generate(args.n, args.category)}[args.tier]()
        print("new questions:", " ".join(made) or "(none)")
    elif args.step in STEPS:
        for_each(STEPS[args.step], targets(args, args.step), args.workers)
    elif args.step == "sheet":
        print(bank.sheet(args.tier))
    elif args.step == "mark-checked":
        bank.mark_checked(args.qid, args.by)
    elif args.step == "split":
        s = bank.split()
        print(sum(v["split"] == "tuning" for v in s.values()), "tuning,",
              sum(v["split"] == "eval" for v in s.values()), "evaluation")
    elif args.step == "bank":
        rows = bank.assemble(args.include_unchecked)
        print(len(rows), "questions ->", files.work("bank.jsonl"))


if __name__ == "__main__":
    main()
