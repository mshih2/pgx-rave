"""Command-line runner.

  python run.py --pipeline full --question "A 45-year-old woman ..."
  python run.py --pipeline api_only --questions questions.jsonl --out answers/
  python run.py --router xgboost --questions questions.jsonl --out answers/

--router lets a router (pgxroute.router) pick the pipeline of each question; its settings come
from the environment (see pgxroute/config.py).

questions.jsonl: one {"question_id": ..., "question": ...} per line. Each answer is written to
<out>/<question_id>__<pipeline>.json (<question_id>__router-<name>.json with --router); existing
files are skipped, so a run can be resumed. A failed question is logged and left without an
output file.
"""
import argparse
import json
import logging
import os
import sys

from pgxroute.pipelines import PIPELINES, answer
from pgxroute.router import ROUTERS, get_router


def main():
    ap = argparse.ArgumentParser()
    how = ap.add_mutually_exclusive_group(required=True)
    how.add_argument("--pipeline", choices=PIPELINES)
    how.add_argument("--router", choices=ROUTERS, help="let this router pick the pipeline")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--question")
    src.add_argument("--questions", help="JSONL with question_id and question")
    ap.add_argument("--out", default="answers")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    router = get_router(args.router) if args.router else None
    tag = f"router-{args.router}" if router else args.pipeline

    def run_one(question):
        if router is None:
            return answer(question, args.pipeline)
        res = answer(question, router.route(question))
        res["router"] = args.router
        return res

    if args.question:
        print(json.dumps(run_one(args.question), ensure_ascii=False, indent=1))
        return 0

    os.makedirs(args.out, exist_ok=True)
    failed = 0
    for line in open(args.questions, encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        qid = item["question_id"]
        path = os.path.join(args.out, f"{qid}__{tag}.json")
        if os.path.exists(path):
            continue
        try:
            res = run_one(item["question"])
        except Exception as e:
            failed += 1
            logging.error("q%s %s failed: %s", qid, tag, e)
            continue
        res["question_id"] = qid
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        os.replace(path + ".tmp", path)
        logging.info("q%s %s done (%s): %d tokens", qid, tag, res["pipeline"],
                     res["usage"]["input_tokens"] + res["usage"]["output_tokens"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
