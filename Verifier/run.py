"""Command-line runner.

  python run.py --input items.jsonl --out results/

items.jsonl: one {"question_id": ..., "question": ..., "answer": ..., "gt_claims": [...]} per
line ("ground_truth": "<text>" instead of gt_claims has the GT decomposed first). Each result is
written to <out>/<question_id>.json; existing files are skipped, so a run can be resumed. A
failed item is logged and left without an output file. <out>/scores.csv collects the scores of
every result in <out>.
"""
import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from pgxverifier import config
from pgxverifier.verifier import Verifier

COLUMNS = ["question_id", "precision", "recall", "F1.5", "n_answer_claims", "n_gt_claims",
           "n_extra", "n_wrong", "n_missing"]


def write_scores(out):
    rows = []
    for name in sorted(os.listdir(out)):
        if name.endswith(".json"):
            d = json.load(open(os.path.join(out, name), encoding="utf-8"))
            rows.append({"question_id": d["question_id"], **d["scores"]})
    with open(os.path.join(out, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with question_id, question, answer "
                                                   "and gt_claims (or ground_truth)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    os.makedirs(args.out, exist_ok=True)
    items = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    todo = [it for it in items
            if not os.path.exists(os.path.join(args.out, f"{it['question_id']}.json"))]
    logging.info("%d items, %d already scored, %d to score", len(items), len(items) - len(todo),
                 len(todo))
    v = Verifier()

    def run_one(it):
        res = v.verify(it["question"], it["answer"], gt_claims=it.get("gt_claims"),
                       ground_truth=it.get("ground_truth"))
        res = {"question_id": it["question_id"], "question": it["question"], **res}
        path = os.path.join(args.out, f"{it['question_id']}.json")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        os.replace(path + ".tmp", path)
        return res["scores"]

    failed = 0
    with ThreadPoolExecutor(max_workers=config.WORKERS) as ex:
        futures = {ex.submit(run_one, it): it["question_id"] for it in todo}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                s = fut.result()
                logging.info("%s: P %.3f R %.3f F1.5 %.3f (%d answer / %d GT claims)", qid,
                             s["precision"], s["recall"], s["F1.5"], s["n_answer_claims"],
                             s["n_gt_claims"])
            except Exception as e:
                failed += 1
                logging.error("%s failed: %s: %s", qid, type(e).__name__, e)

    rows = write_scores(args.out)
    if rows:
        logging.info("mean over %d scored: F1.5 %.4f  P %.4f  R %.4f", len(rows),
                     sum(r["F1.5"] for r in rows) / len(rows),
                     sum(r["precision"] for r in rows) / len(rows),
                     sum(r["recall"] for r in rows) / len(rows))
    if v._labeller is not None:
        logging.info("labelling judge usage: %s", v.labeller.stats())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
