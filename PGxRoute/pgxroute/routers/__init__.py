"""Routers that pick one answer pipeline per question.

  rule.py   rule baseline over the structured input's genes, variants and drugs
  ml/       text classifiers: TF-IDF + logistic regression or XGBoost
  rl/       LLM routers (Qwen3-8B, facet-guided or self-faceted prompt), untrained or
            GRPO-trained

Use them through pgxroute.router. The LLM routers name the pipelines tool_a / tool_b / tool_c,
as in the prompts they were trained on; TOOL_TO_PIPELINE maps those names.
"""

PIPELINE_ORDER = ("reasoning_only", "api_only", "full")
TOOL_TO_PIPELINE = {"tool_a": "api_only", "tool_b": "reasoning_only", "tool_c": "full"}
# The question tiers and the pipeline built for each ("filtered" is the old name of intermediate).
TIER_TO_PIPELINE = {"common": "reasoning_only", "intermediate": "api_only", "filtered": "api_only",
                    "hard": "full"}
