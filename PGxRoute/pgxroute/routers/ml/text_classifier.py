"""Text-classifier routers: TF-IDF features with logistic regression or XGBoost.

They read the question text only. Settings: TF-IDF over word
1-2-grams, min_df 2, sublinear tf, fitted on the training questions only; logistic regression
with max_iter 4000; XGBoost with 300 trees, depth 4, learning rate 0.1, row/column subsampling
0.9/0.7, random_state 0.

Labels are pipelines; a question's tier can be mapped to its pipeline with TIER_TO_PIPELINE.
`two_fold_predict` trains on one fold, predicts the other, then swaps.

Train and save a router:

  python -m pgxroute.routers.ml.text_classifier --kind xgboost --data bank.jsonl \
      --out data/routers/xgboost.pkl

bank.jsonl: one {"question": ..., "pipeline": ...} or {"question": ..., "tier": ...} per line.
A saved router is a pickle; load only files you trust.
"""
import argparse
import json
import pickle

from .. import PIPELINE_ORDER, TIER_TO_PIPELINE

KINDS = ("logreg", "xgboost")


def make_model(kind):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=4000)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.7, eval_metric="mlogloss", verbosity=0,
                             random_state=0)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def _vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)


class TextClassifierRouter:
    def __init__(self, kind="xgboost"):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.name = kind
        self.vectorizer = None
        self.model = None

    def fit(self, questions, pipelines):
        """Fit on question texts and the pipeline each should go to."""
        y = [PIPELINE_ORDER.index(p) for p in pipelines]
        self.vectorizer = _vectorizer()
        self.model = make_model(self.kind).fit(self.vectorizer.fit_transform(list(questions)), y)
        return self

    def route_many(self, questions):
        if self.model is None:
            raise RuntimeError("router is not trained; call fit() or load()")
        pred = self.model.predict(self.vectorizer.transform(list(questions)))
        return [PIPELINE_ORDER[int(k)] for k in pred]

    def route(self, question):
        return self.route_many([question])[0]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"kind": self.kind, "vectorizer": self.vectorizer, "model": self.model}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        r = cls(d["kind"])
        r.vectorizer, r.model = d["vectorizer"], d["model"]
        return r


def two_fold_predict(questions, pipelines, in_first_fold, kind="xgboost"):
    """Out-of-fold picks: train on one fold, predict the other, then swap."""
    picks = [None] * len(questions)
    folds = [[i for i, f in enumerate(in_first_fold) if f], [i for i, f in enumerate(in_first_fold) if not f]]
    for train, test in ((folds[0], folds[1]), (folds[1], folds[0])):
        r = TextClassifierRouter(kind).fit([questions[i] for i in train], [pipelines[i] for i in train])
        for i, p in zip(test, r.route_many([questions[i] for i in test])):
            picks[i] = p
    return picks


def _label(row):
    if row.get("pipeline"):
        return row["pipeline"]
    return TIER_TO_PIPELINE[row["tier"]]


def main():
    ap = argparse.ArgumentParser(description="Train and save a text-classifier router.")
    ap.add_argument("--kind", choices=KINDS, default="xgboost")
    ap.add_argument("--data", required=True, help="JSONL with question and pipeline (or tier)")
    ap.add_argument("--out", required=True, help="where to write the trained router (pickle)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    r = TextClassifierRouter(args.kind).fit([x["question"] for x in rows], [_label(x) for x in rows])
    r.save(args.out)
    print(f"trained {args.kind} on {len(rows)} questions -> {args.out}")


if __name__ == "__main__":
    main()
