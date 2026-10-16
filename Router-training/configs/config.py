"""Settings read as the module `config` by the rollout loop (lead_agent/llm_agent/generation_quick3.py,
loaded from $CONFIG_ROOT/config.py) and by the verl rollout workers (configs/ is on PYTHONPATH).
"""
import os

# cache_baseline: reward = F[chosen] - mean_t F[t] - COST_WEIGHT * cost_feature, used as the
# advantage as is (pair with algorithm.adv_estimator=cache_baseline). The F and token tables are
# built from live pipeline runs every step (router_reward/live_reward.py). "grpo" selects the older
# preference-vector reward, which needs evaluators that are not part of this repository.
REWARD_ADV_MODE = os.environ.get("REWARD_ADV_MODE", "cache_baseline").strip().lower()
if REWARD_ADV_MODE not in ("grpo", "cache_baseline"):
    raise ValueError(f"REWARD_ADV_MODE must be 'grpo' or 'cache_baseline', got {REWARD_ADV_MODE!r}")

REWARD_COST_WEIGHT = float((os.environ.get("COST_WEIGHT") or "0").strip() or 0)
if REWARD_COST_WEIGHT < 0:
    raise ValueError(f"COST_WEIGHT must be >= 0, got {REWARD_COST_WEIGHT}")

REWARD_EVAL_MODE = "existing"
REWARD_NORMALIZATION_SCOPE = "per_question"
# Read only by the older answer-evaluator path (REWARD_ADV_MODE=grpo).
REWARD_EVALUATION_MODE = "batch_per_question"
REWARD_MULTI_QUESTION_COUNT = 3

# Rollouts run the chosen pipeline live; no per-question result files are read or written.
USE_EXISTING_QUESTION_TOOL_RESULTS = False
QUESTION_TOOL_RESULT_DIR = ""

# Pipeline modes the router's tools stand for (router_reward/covaph_env.py maps them to PGxRoute pipelines).
VALID_PIPELINE_MODES = ["skip_retrieve", "module4_only", "full_rules_top8"]

# Rollout sampling temperature used by the verl rollout workers when the trainer config sets none
# (train.sh sets actor_rollout_ref.rollout.temperature, which takes precedence).
RL_ACTOR_TEMPERATURE = float(os.getenv("RL_ACTOR_TEMPERATURE", "1.2"))
