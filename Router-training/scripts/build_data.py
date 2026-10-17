"""Build the training files for the LLM router.

  python scripts/build_data.py --bank bank.jsonl --prompt fg --out data/fg

bank.jsonl: one {"question_id", "question", "tier", "gt_claims": [...]} per line, optionally with
"fold" (0 or 1) and "ground_truth" (GT text). "tier" is hard, intermediate or common; it becomes the
row's category, which the rollout loop requires to be a tier. --folds names a file of
"question_id fold" lines (header optional) that sets the fold of the questions it lists.
Writes <out>/fold<k>.jsonl per fold (or <out>/all.jsonl) in the row format the rollout loop
reads, with the router prompt built by pgxroute.routers.rl.llm_router, and <out>/gt_claims.json
({question_id: [claim, ...]}), which the live reward reads through GT_CLAIMS_FILE.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PGxRoute"))

from pgxroute.routers.rl.llm_router import PROMPTS, build_messages  # noqa: E402

TIER_TO_TOOL = {"common": "tool_b", "intermediate": "tool_a", "hard": "tool_c"}
TIER_ALIASES = {"filtered": "intermediate"}


def _qid(v):
    s = str(v)
    return int(s) if s.isdigit() else s


def to_row(item, prompt):
    qid = _qid(item["question_id"])
    q = item["question"]
    gt_text = item.get("ground_truth") or "\n".join(str(c) for c in item["gt_claims"])
    tier = str(item.get("tier", "")).strip().lower()
    tier = TIER_ALIASES.get(tier, tier)
    if tier not in TIER_TO_TOOL:
        raise ValueError(f"question {item['question_id']}: tier must be one of {sorted(TIER_TO_TOOL)}, "
                         f"got {item.get('tier')!r}")
    tool = TIER_TO_TOOL[tier]
    return {"data_source": "pgx",
            "prompt": build_messages(prompt, q),
            "ability": "pharmacogenomics",
            "reward_model": {"style": "rule", "ground_truth": gt_text},
            "extra_info": {"index": qid, "split": "train", "need_tools_kwargs": True,
                           "question": q, "answer": gt_text, "expected_mode": tool,
                           "category": tier,
                           "level": tier, "pgx_analysis": "{}",
                           "tools_kwargs": {"pgx_query": {"create_kwargs": {
                               "question": q, "answer": gt_text, "expected_mode": tool,
                               "id": str(qid)}}}},
            "pref_vec": {"accuracy": 1.0, "latency": 0.0, "cost": 0.0},
            "index": qid}


def read_folds(path):
    """A "question_id fold" file (whitespace-separated, header optional) -> {question_id: fold}."""
    folds = {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2 and parts[1] in ("0", "1"):
            folds[_qid(parts[0])] = int(parts[1])
    if not folds:
        raise SystemExit(f"{path}: no 'question_id fold' lines")
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--prompt", choices=PROMPTS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", help="file of 'question_id fold' lines that sets each listed question's fold")
    args = ap.parse_args()
    bank = [json.loads(l) for l in open(args.bank, encoding="utf-8") if l.strip()]
    if args.folds:
        folds = read_folds(args.folds)
        for b in bank:
            if _qid(b["question_id"]) in folds:
                b["fold"] = folds[_qid(b["question_id"])]
    missing = [b.get("question_id") for b in bank if not b.get("gt_claims")]
    if missing:
        raise SystemExit(f"questions without gt_claims: {missing[:10]}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for b in bank:
        groups.setdefault(f"fold{b['fold']}" if "fold" in b else "all", []).append(to_row(b, args.prompt))
    for key, rows in sorted(groups.items()):
        with open(out / f"{key}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{out / f'{key}.jsonl'}: {len(rows)} questions")
    gt = {str(b["question_id"]): [str(c) for c in b["gt_claims"]] for b in bank}
    (out / "gt_claims.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out / 'gt_claims.json'}: {len(gt)} questions")


if __name__ == "__main__":
    main()
