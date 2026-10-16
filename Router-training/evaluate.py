"""Evaluate a router end to end: route each question, answer it with the chosen pipeline, and
score the answer with the Verifier.

  python evaluate.py --bank bank.jsonl --fold 1 --router fg --model router-fg-fold0 --out eval/fg_fold0

--router is any PGxRoute router (rule, logreg, xgboost, fg, sf); --model and --base-url select
the served checkpoint of an LLM router, --model-path a trained classifier. Each result is written
to <out>/<question_id>.json (existing ones are skipped); <out>/summary.json holds the mean
F1.5, mean tokens and the pipeline counts.
"""
import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("PGxRoute", "Verifier"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

from pgxroute.pipelines import answer                     # noqa: E402
from pgxroute.router import ROUTERS, get_router           # noqa: E402
from pgxverifier.verifier import Verifier                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="JSONL with question_id, question, gt_claims (and fold)")
    ap.add_argument("--fold", type=int, help="evaluate only this fold")
    ap.add_argument("--router", choices=ROUTERS, required=True)
    ap.add_argument("--model", help="LLM router: served model name")
    ap.add_argument("--base-url", help="LLM router: server URL")
    ap.add_argument("--model-path", help="text classifier: trained model file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    bank = [json.loads(l) for l in open(args.bank, encoding="utf-8") if l.strip()]
    if args.fold is not None:
        bank = [b for b in bank if b.get("fold") == args.fold]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    router = get_router(args.router, model_path=args.model_path, base_url=args.base_url,
                        model=args.model)
    verifier = Verifier()

    def one(item):
        dst = out / f"{item['question_id']}.json"
        if dst.exists():
            return
        try:
            pipeline = router.route(item["question"])
            res = answer(item["question"], pipeline)
            v = verifier.verify(item["question"], res["answer"], gt_claims=item["gt_claims"])
        except Exception as e:
            logging.error("%s failed: %s: %s", item["question_id"], type(e).__name__, e)
            return
        rec = {"question_id": item["question_id"], "router": args.router, "pipeline": pipeline,
               "answer": res["answer"], "usage": res["usage"], "scores": v["scores"]}
        dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        logging.info("%s -> %s: F1.5 %.3f, %d tokens", item["question_id"], pipeline,
                     v["scores"]["F1.5"], res["usage"]["input_tokens"] + res["usage"]["output_tokens"])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, bank))

    ids = {str(b["question_id"]) for b in bank}
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*.json")
            if p.name != "summary.json"]
    recs = [r for r in recs if str(r["question_id"]) in ids]
    if not recs:
        logging.error("no question was scored")
        return 1
    summary = {"router": args.router, "n_scored": len(recs), "n_questions": len(bank),
               "mean_F1.5": sum(r["scores"]["F1.5"] for r in recs) / len(recs),
               "mean_tokens": sum(r["usage"]["input_tokens"] + r["usage"]["output_tokens"]
                                  for r in recs) / len(recs),
               "pipelines": dict(Counter(r["pipeline"] for r in recs))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0 if len(recs) == len(bank) else 1


if __name__ == "__main__":
    sys.exit(main())
