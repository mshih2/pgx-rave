"""Placeholder for the external LLM helper the rollout module imports.

Router training calls no external LLM through this module: the chosen pipeline runs through
covaph_env.py and the reward through live_reward.py.
"""


def get_llm_response(*args, **kwargs):
    raise NotImplementedError("get_llm_response is not used in router training")
