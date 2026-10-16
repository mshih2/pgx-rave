# Router training

GRPO training of the LLM router (Qwen3-8B, `fg` or `sf` prompt of PGxRoute) with rewards computed
live: every training step, the questions in the batch are answered by the PGxRoute pipelines and
the answers are scored by the Verifier. The verl used for training is included in this folder.

## Files

```
train.sh               launches training (python -m recipe.algo.main_grpo_covaph with its settings)
build_data.py          question bank -> training rows (router prompt) + gt_claims.json
live_reward.py         per-step reward table: pipelines run live, answers scored by the Verifier
covaph_env.py          runs the pipeline a tool stands for, through PGxRoute
config.py              reward settings read by the rollout loop
merge_checkpoint.py    trained checkpoint -> Hugging Face folder for serving
evaluate.py            route -> answer -> verify on held-out questions, with any PGxRoute router
LLM_CALL.py            placeholder for a helper the rollout module imports but does not call
recipe/algo/           training entry point, trainer and its config
lead_agent/llm_agent/  rollout loop (generation_quick3.py): tool parsing, tool execution, reward
covaph_reward.py       reward manager that hands the rollout loop's rewards to the trainer
verl/                  the verl this training uses (a modified 0.4.0.dev; see LICENSE and NOTICE)
covaph_tools_3tool.json   the router's tool set (tool_a, tool_b, tool_c)
```

The other packages (Ray, vLLM, PyTorch, Transformers, ...) are used unmodified; versions are in
`requirements.txt`.

## Reward

The router names one of three tools: `tool_a` = API-only, `tool_b` = Reasoning-only, `tool_c` =
Full pipeline. Each rollout's pick is rewarded against the question's other pipelines:

```
reward = F[chosen] - mean_t F[t] - COST_WEIGHT * cost_feature[chosen]
cost_feature[p] = (tokens[p] - mean_t tokens[t]) / (max_t tokens[t] - min_t tokens[t] + 1e-6)
```

`F` is the answer's claim-level F-score from the Verifier's labels and credits (beta
`REWARD_BETA`, default 1.0; extra-claim weight `REWARD_EXTRA_WEIGHT`, default 0.2) and `tokens`
the pipeline's input + output tokens. The reward is used as the advantage as is
(`algorithm.adv_estimator=cache_baseline`, an identity estimator), so a question whose rollouts
all pick the same tool still gets a signed update. A rollout whose answer has no tool after
`</think>` scores `F = 0` and the largest cost feature.

Every step, each question in the batch runs all three pipelines once (the rollouts' own tool
calls are reused) and each answer is verified once. If a pipeline or its scoring fails twice,
that question's rollouts get reward 0 and the failure is logged. Each step's table is saved to
`outputs/<run>/global_step_<n>/rl_training_inputs/live_reward_table.json`.

## Setup

Training needs four GPUs (FSDP actor, vLLM rollouts with tensor parallel 2). The reward calls
the model servers of the other folders, which must run on other GPUs:

| Used by | Server | Settings |
|---|---|---|
| PGxRoute pipelines | Qwen3.8-27B (reasoner, reranker), Gemma-4-31B-it (critique), BGE-M3 | `REASONER_BASE_URL`, `CRITIC_URL`, `EMBED_PYTHON`, `EMBED_DEVICE`, `CORPUS_DIR` (see `PGxRoute/README.md`) |
| Verifier | Gemma-4-31B-it (decomposition), Mistral-Small-24B (grading), Grok 4.5 | `DECOMPOSER_URL`, `GRADER_URL`, `XAI_API_KEY` (see `Verifier/README.md`) |

```bash
pip install -r requirements.txt
```

Run BGE-M3 in its own environment (`EMBED_PYTHON`) if its Transformers version differs from the
training one. The Full pipeline needs a retrieval corpus (`PGxRoute/README.md`).

## Running

```bash
# 1. training rows from a question bank (one {"question_id", "question", "gt_claims", "fold"} per line)
python build_data.py --bank bank.jsonl --prompt fg --out data/fg

# 2. train on one fold (COST_WEIGHT penalises tokens; 0 = off)
COST_WEIGHT=0.4 bash train.sh data/fg/fold0.jsonl data/fg/gt_claims.json router_fg_fold0

# 3. serve a checkpoint and evaluate it on the other fold
python merge_checkpoint.py checkpoints/router_fg_fold0/global_step_27/actor
vllm serve checkpoints/router_fg_fold0/global_step_27/actor/_merged --served-model-name router-fg-fold0 --port 8020
python evaluate.py --bank bank.jsonl --fold 1 --router fg --model router-fg-fold0 \
    --base-url http://127.0.0.1:8020/v1 --out eval/fg_fold0
```

`train.sh` passes any further arguments to the trainer as Hydra overrides (for example
`trainer.total_training_steps=12`). Checkpoints are saved every 3 steps under `checkpoints/<run>/`
(`CKPT_DIR`); `MODEL_PATH` (default `Qwen/Qwen3-8B`) and `N_GPUS` (default 4) can be set too.

## Settings in train.sh

| Setting | Value |
|---|---|
| Base model | Qwen3-8B |
| Rollouts per question | 3, temperature 1.0, top-p 0.95, top-k 10, up to 2,048 new tokens |
| Batch | 16 questions per step, mini-batch 16, 1 PPO epoch |
| Optimiser | learning rate 2e-6, cosine decay to 0.1x |
| Loss | clip 0.1 / 0.3 (dual clip 10), KL loss (mse) 0.001, entropy 0.001 |
| Steps | 36, checkpoint every 3 |
| Tool parsing | only the JSON after the last `</think>` counts (`STRICT_TOOL_PARSE=1`) |

## Licence

The verl code in `verl/` and the training code derived from it are under the Apache License 2.0
(`LICENSE`, `NOTICE`). The rest of this folder: Apache License 2.0.
