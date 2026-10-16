"""Score: precision, recall and F1.5 from the graded credits.

  P     = forward credit / (n - (1 - EXTRA_WEIGHT) * n_extra)
          (n answer claims; each extra claim counts EXTRA_WEIGHT = 0.1 of a false positive)
  R     = backward credit / number of GT claims
  F1.5  = 3.25 P R / (2.25 P + R)

n_extra is counted from the labelling judge's forward labels as it returned them, before its
consistency corrections. An answer with no countable claims gets
P = 1.0 (vacuous); with no GT claims, R = 0.
"""
from .. import config


def f_beta(p, r, beta):
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    d = b2 * p + r
    return 0.0 if d == 0 else (1.0 + b2) * p * r / d


def score(labels, credits, n_answer, n_gt):
    fwd_labels = labels["fwd_labels"].values()
    bwd_labels = labels["bwd_labels"].values()
    n_extra = sum(1 for r in labels["forward"] if isinstance(r, dict) and r.get("label") == "extra")
    fwd = sum(credits["fwd_credit"].values())
    bwd = sum(credits["bwd_credit"].values())
    den = n_answer - (1.0 - config.EXTRA_WEIGHT) * n_extra
    p = fwd / den if den > 0 else 1.0
    r = bwd / n_gt if n_gt > 0 else 0.0
    return {"precision": p, "recall": r, f"F{config.F_BETA:g}": f_beta(p, r, config.F_BETA),
            "n_answer_claims": n_answer, "n_gt_claims": n_gt, "n_extra": n_extra,
            "n_wrong": sum(1 for v in fwd_labels if v == "wrong"),
            "n_missing": sum(1 for v in bwd_labels if v == "missing"),
            "n_inconsistencies": len(labels.get("inconsistencies", [])),
            "forward_credit": round(fwd, 4), "backward_credit": round(bwd, 4)}
