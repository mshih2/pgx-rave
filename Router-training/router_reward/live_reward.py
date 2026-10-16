"""Live reward table for REWARD_ADV_MODE=cache_baseline.

Every training step, each question in the batch is answered by every active tool's pipeline
(covaph_env.run_mode, which reuses what the rollouts already ran) and each answer is scored
against the question's GT claims with the Verifier. The result has the shape
compute_cache_baseline_rewards expects:

  {"<index>__<tool>": {"F15": <claim-level F-score>, "cost": <input + output tokens>}}

The F-score uses REWARD_BETA (default 1.0) and the extra-claim weight REWARD_EXTRA_WEIGHT
(default 0.2). GT claims come from GT_CLAIMS_FILE, a JSON object {index: [claim, ...]} written by
scripts/build_data.py. If any pipeline or scoring of a question fails, all its tools get F = 0 and equal
cost, so its rollouts get a zero reward and the failure is logged.
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(os.environ.get("PGX_REPO_ROOT") or Path(__file__).resolve().parents[2])
for _sub in ("PGxRoute", "Verifier"):
    if str(_REPO / _sub) not in sys.path:
        sys.path.insert(0, str(_REPO / _sub))

from router_reward import covaph_env                # noqa: E402
from pgxverifier.steps.score import f_beta          # noqa: E402
from pgxverifier.verifier import Verifier           # noqa: E402

TOOL_TO_MODE = {"tool_a": "skip_retrieve", "tool_b": "module4_only", "tool_c": "full_rules_top8"}
REWARD_BETA = float(os.environ.get("REWARD_BETA", "1.0"))
REWARD_EXTRA_WEIGHT = float(os.environ.get("REWARD_EXTRA_WEIGHT", "0.2"))
REWARD_WORKERS = int(os.environ.get("REWARD_WORKERS", "8"))

_verifier = None
_gt = None
_lock = threading.Lock()


def _get_verifier():
    global _verifier
    with _lock:
        if _verifier is None:
            _verifier = Verifier()
        return _verifier


def _gt_claims():
    global _gt
    with _lock:
        if _gt is None:
            path = os.environ.get("GT_CLAIMS_FILE", "")
            if not path or not os.path.exists(path):
                raise RuntimeError(f"GT_CLAIMS_FILE not found ({path!r}); build it with build_data.py")
            _gt = {str(k): [str(c) for c in v] for k, v in json.load(open(path, encoding="utf-8")).items()}
        return _gt


def answer_f(verified):
    """Claim-level F-score of a verified answer with the reward's beta and extra-claim weight."""
    labels, credits = verified["labels"], verified["credits"]
    n, n_gt = len(verified["answer_claims"]), len(verified["gt_claims"])
    n_extra = sum(1 for r in labels["forward"] if isinstance(r, dict) and r.get("label") == "extra")
    den = n - (1.0 - REWARD_EXTRA_WEIGHT) * n_extra
    p = sum(credits["fwd_credit"].values()) / den if den > 0 else 1.0
    r = sum(credits["bwd_credit"].values()) / n_gt if n_gt else 0.0
    return f_beta(p, r, REWARD_BETA)


def _score(question, gt, tool):
    res = covaph_env.run_mode(question, TOOL_TO_MODE[tool])
    if res.get("error"):
        raise RuntimeError(res["error"])
    text = res.get("response") or ""
    f = answer_f(_get_verifier().verify(question, text, gt_claims=gt)) if text.strip() else 0.0
    return {"F15": f, "cost": res["cost"]}


def build_reward_table(all_turn_index, all_turn_problems, active_tools, global_steps, output_dir=None):
    gt = _gt_claims()
    questions = {}
    for idx, q in zip(all_turn_index, all_turn_problems):
        questions.setdefault(idx, q)
    missing = [i for i in questions if str(i) not in gt]
    if missing:
        raise RuntimeError(f"no GT claims for question index {missing[:10]} in GT_CLAIMS_FILE")
    jobs = [(idx, t) for idx in questions for t in active_tools]

    def job(j):
        idx, t = j
        try:
            return j, _score(questions[idx], gt[str(idx)], t)
        except Exception as e:
            return j, {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=max(1, REWARD_WORKERS)) as ex:
        results = dict(ex.map(job, jobs))

    table, failed = {}, {}
    for idx in questions:
        errs = {t: results[(idx, t)]["error"] for t in active_tools if "error" in results[(idx, t)]}
        for t in active_tools:
            table[f"{idx}__{t}"] = ({"F15": 0.0, "cost": 0} if errs else results[(idx, t)])
        if errs:
            failed[str(idx)] = errs
            print(f"[Live Reward] question {idx}: {errs}; its rollouts get reward 0", flush=True)
    if output_dir:
        d = os.path.join(output_dir, f"global_step_{global_steps}", "rl_training_inputs")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "live_reward_table.json"), "w", encoding="utf-8") as f:
            json.dump({"table": table, "failed": failed}, f, indent=1)
    covaph_env.clear_step_results()
    print(f"[Live Reward] step {global_steps}: {len(questions)} questions x {len(active_tools)} tools "
          f"scored live, {len(failed)} failed", flush=True)
    return table
