"""Pick the pipeline for a question with one of five routers.

  rule      rule baseline over the SI's genes, variants and drugs   (routers/rule.py)
  logreg    TF-IDF + logistic regression                            (routers/ml)
  xgboost   TF-IDF + XGBoost                                        (routers/ml)
  fg        LLM router, facet-guided prompt                         (routers/rl)
  sf        LLM router, self-faceted prompt                         (routers/rl)

  from pgxroute.router import get_router, route
  route(question, "rule")                        # -> "api_only", "reasoning_only" or "full"
  r = get_router("fg", model="my-trained-ckpt")  # reuse one router for many questions
  r.route(question)

  python -m pgxroute.router --router fg --question "A 45-year-old woman ..."

The classifiers load a trained model from CLASSIFIER_DIR/<name>.pkl (or model_path=...); train one
with `python -m pgxroute.routers.ml.text_classifier`. The LLM routers call ROUTER_BASE_URL with
ROUTER_MODEL (or base_url=..., model=...).
"""
import argparse
import json
import sys
from pathlib import Path

from . import config

ROUTERS = ("rule", "logreg", "xgboost", "fg", "sf")


def get_router(name, model_path=None, base_url=None, model=None, seed=None):
    """A router object whose .route(question) returns a pipeline name."""
    if name == "rule":
        from .routers.rule import RuleRouter
        return RuleRouter()
    if name in ("logreg", "xgboost"):
        from .routers.ml.text_classifier import TextClassifierRouter
        path = Path(model_path or config.CLASSIFIER_DIR / f"{name}.pkl")
        if not path.exists():
            raise FileNotFoundError(f"no trained {name} router at {path}; train one with "
                                    f"python -m pgxroute.routers.ml.text_classifier --kind {name} ...")
        r = TextClassifierRouter.load(path)
        if r.kind != name:
            raise ValueError(f"{path} holds a {r.kind} router, not {name}")
        return r
    if name in ("fg", "sf"):
        from .routers.rl.llm_router import LLMRouter
        return LLMRouter(name, base_url=base_url, model=model, seed=seed)
    raise ValueError(f"router must be one of {ROUTERS}, got {name!r}")


def route(question, router="rule", **kwargs):
    """The pipeline the named router picks for one question."""
    return get_router(router, **kwargs).route(question)


def main():
    ap = argparse.ArgumentParser(description="Print the pipeline a router picks.")
    ap.add_argument("--router", choices=ROUTERS, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--question")
    src.add_argument("--questions", help="JSONL with question_id and question")
    ap.add_argument("--model-path", help="trained classifier (logreg / xgboost)")
    ap.add_argument("--base-url", help="LLM router server (fg / sf)")
    ap.add_argument("--model", help="LLM router model name (fg / sf)")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    r = get_router(args.router, model_path=args.model_path, base_url=args.base_url,
                   model=args.model, seed=args.seed)
    if args.question:
        print(r.route(args.question))
        return 0
    for line in open(args.questions, encoding="utf-8"):
        if line.strip():
            item = json.loads(line)
            print(json.dumps({"question_id": item["question_id"], "router": args.router,
                              "pipeline": r.route(item["question"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
