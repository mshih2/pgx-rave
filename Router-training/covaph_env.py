"""Runs the answer pipeline a router tool stands for, through PGxRoute.

  tool_a  skip_retrieve    -> api_only
  tool_b  module4_only     -> reasoning_only
  tool_c  full_rules_top8  -> full

Results are kept per (question, mode) until clear_step_results() is called, so within one
training step a pipeline runs once per question however many rollouts pick it, and the reward
(live_reward.py) reuses what the rollouts already ran.
"""
import os
import sys
import threading
import time
from pathlib import Path

_REPO = Path(os.environ.get("PGX_REPO_ROOT") or Path(__file__).resolve().parents[1])
for _sub in ("PGxRoute", "Verifier"):
    if str(_REPO / _sub) not in sys.path:
        sys.path.insert(0, str(_REPO / _sub))

from pgxroute.pipelines import answer as _answer   # noqa: E402

MODE_TO_PIPELINE = {"skip_retrieve": "api_only", "module4_only": "reasoning_only",
                    "full_rules_top8": "full"}
ATTEMPTS = 2

_results = {}
_running = {}
_lock = threading.Lock()


def _run(question, mode):
    pipeline = MODE_TO_PIPELINE[mode]
    err = None
    for _ in range(ATTEMPTS):
        t = time.time()
        try:
            res = _answer(question, pipeline)
            usage = res["usage"]
            return {"response": res["answer"], "mode": mode, "pipeline": pipeline,
                    "eval_result": None,
                    "cost": int(usage["input_tokens"]) + int(usage["output_tokens"]),
                    "usage": usage, "latency_seconds": time.time() - t}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[covaph_env] {pipeline} failed: {err}", flush=True)
    return {"response": f"Error: {err}", "mode": mode, "pipeline": pipeline, "eval_result": None,
            "cost": 0, "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_seconds": 0.0, "error": err}


def run_mode(question, mode):
    """The pipeline result for (question, mode), run once per step."""
    if mode not in MODE_TO_PIPELINE:
        raise ValueError(f"unknown pipeline mode {mode!r}; valid: {sorted(MODE_TO_PIPELINE)}")
    key = (question, mode)
    with _lock:
        if key in _results:
            return _results[key]
        event = _running.get(key)
        owner = event is None
        if owner:
            event = _running[key] = threading.Event()
    if not owner:
        event.wait()
        with _lock:
            return _results[key]
    res = _run(question, mode)
    with _lock:
        _results[key] = res
        _running.pop(key, None)
    event.set()
    return res


def clear_step_results():
    with _lock:
        _results.clear()


def call_covaph_tool(arguments):
    """Entry point used by the rollout loop (generation_quick3.call_tool)."""
    mode = arguments.get("pipeline_mode") or arguments.get("mode")
    question = arguments.get("question") or arguments.get("problem")
    if not question:
        raise ValueError("missing 'question'")
    return dict(run_mode(question, mode))
