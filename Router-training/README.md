# Router training

GRPO training of the LLM router of PGxRoute (Qwen3-8B with the `fg` or `sf` prompt). This is the
Evolve loop in which the verifier's score trains the router: at every training step, the
questions in the batch are answered by the PGxRoute pipelines and the answers are scored by the
Verifier, live. The verl used for training is included in this folder.

In the paper, routers are trained with two-fold cross-validation on the 135 evaluation questions:
each fold's model routes the other fold, so every question is held out once. Each prompt is
trained with cost weights 0, 0.2, 0.4, 0.6 and 0.8, and the checkpoint at step 27 (the step with
the highest average held-out F1.5) is reported for every weight.

## Reward

The router names one of three tools: `tool_a` = API-only, `tool_b` = Reasoning-only, `tool_c` =
Full pipeline. Each rollout's pick is rewarded against the question's other pipelines:

```
reward = F[chosen] - mean_t F[t] - COST_WEIGHT * cost_feature[chosen]
cost_feature[p] = (tokens[p] - mean_t tokens[t]) / (max_t tokens[t] - min_t tokens[t] + 1e-6)
```

`F` is the claim-level F-score of the answer on a 0–1 scale, computed from the Verifier's labels
and credits (`REWARD_BETA`, default 1.0, i.e. F1, and the extra-claim weight
`REWARD_EXTRA_WEIGHT`, default 0.2), and `tokens` the pipeline's input + output tokens. `COST_WEIGHT` penalises token cost (0 = off). The reward is
used as the advantage as is (`algorithm.adv_estimator=cache_baseline`, an identity estimator), so
a question whose rollouts all pick the same tool still gets a signed update. A rollout with no
parsable tool after `</think>` scores `F = 0` and the largest cost feature.

At every step, each question in the batch runs all three pipelines once (the rollouts' own tool
calls are reused), and each answer is verified once. If a pipeline or its scoring fails, that
question's rollouts get reward 0 and the failure is logged. Each step's table is saved to
`outputs/<run>/global_step_<n>/rl_training_inputs/live_reward_table.json`.

## Settings in train.sh

| Setting | Value |
|---|---|
| Base model | Qwen3-8B, full-parameter fine-tuning |
| Rollouts per question | 3, temperature 1.0, top-p 0.95, top-k 10, thinking on, up to 2,048 new tokens |
| Batch | 16 questions per step, mini-batch 16, 1 PPO epoch |
| Optimiser | learning rate 2e-6, cosine decay to 0.1x |
| Loss | clip 0.1 / 0.3 (dual clip 10), KL loss (mse) 0.001 (not in the reward), entropy 0.001 |
| Steps | 36, checkpoint every 3 |
| Tool parsing | only the JSON after the last `</think>` counts (`STRICT_TOOL_PARSE=1`) |

## Setup

Training needs four GPUs (FSDP actor, vLLM rollouts with tensor parallel 2). The reward calls the
model servers of the other folders, which must run on other GPUs:

| Used by | Server | Settings |
|---|---|---|
| PGxRoute pipelines | Qwen3.8-27B (reasoner, reranker), Gemma-4-31B-it (critique), BGE-M3 | `REASONER_BASE_URL`, `CRITIC_URL`, `EMBED_PYTHON`, `EMBED_DEVICE`, `CORPUS_DIR` (see `../PGxRoute/README.md`) |
| Verifier | Gemma-4-31B-it (decomposition), Mistral-Small-24B-Instruct-2501 (grading), Grok 4.5 (labelling) | `DECOMPOSER_URL`, `GRADER_URL`, `XAI_API_KEY` (see `../Verifier/README.md`) |

```bash
pip install -r requirements.txt
```

The other packages (Ray, vLLM, PyTorch, Transformers, ...) are used unmodified; their versions are
in `requirements.txt`. These requirements do not include FlagEmbedding and pin a different
Transformers version from PGxRoute's, so BGE-M3 must run in its own environment: install
FlagEmbedding there and set `EMBED_PYTHON` to its interpreter. The Full pipeline needs a
retrieval corpus at `CORPUS_DIR` (with `VERIFY_CORPUS=0` for one you built; see
`../PGxRoute/README.md`).

## Running

```bash
# 1. training rows from a question bank (one {"question_id", "question", "tier", "gt_claims", "fold"} per line)
python scripts/build_data.py --bank bank.jsonl --prompt fg --out data/fg

# 2. train on one fold (COST_WEIGHT penalises tokens; 0 = off)
COST_WEIGHT=0.4 bash train.sh data/fg/fold0.jsonl data/fg/gt_claims.json router_fg_fold0

# 3. serve a checkpoint and evaluate it on the other fold
python scripts/merge_checkpoint.py checkpoints/router_fg_fold0/global_step_27/actor
vllm serve checkpoints/router_fg_fold0/global_step_27/actor/_merged --served-model-name router-fg-fold0 --port 8020
python scripts/evaluate.py --bank bank.jsonl --fold 1 --router fg --model router-fg-fold0 \
    --base-url http://127.0.0.1:8020/v1 --out eval/fg_fold0
```

`build_data.py` writes one file per `fold` (0 or 1); rows without a `fold` (tuning questions) go
to `all.jsonl`. The bank written by `../Question-generation` gives each evaluation question its
fold. For `../Question-bank/ai-assisted/questions.jsonl`, add `--folds ../Question-bank/folds.txt`,
which assigns each evaluation question its fold by question ID.
`evaluate.py` runs route → answer → verify on held-out questions with any PGxRoute router.

`train.sh` passes any further arguments to the trainer as Hydra overrides (for example
`trainer.total_training_steps=12`). Checkpoints are saved every 3 steps under `checkpoints/<run>/`
(`CKPT_DIR`); `MODEL_PATH` (default `Qwen/Qwen3-8B`) and `N_GPUS` (default 4) can be set too.

## Files

```
train.sh                     launches training (python -m recipe.algo.main_grpo_pgx with its settings)
configs/
  config.py                  settings read as the module `config` by the rollout loop and workers
  pgx_tools_3tool.json       the router's tool set (tool_a, tool_b, tool_c)
  vllm_model_configs.json
router_reward/               live reward (calls PGxRoute and the Verifier)
  live_reward.py             per-step reward table: pipelines run live, answers scored by the Verifier
  pgx_env.py                 runs the pipeline a tool stands for, through PGxRoute
  pgx_reward.py              reward manager that hands the rollout loop's rewards to the trainer
scripts/
  build_data.py              question bank -> training rows (router prompt) + gt_claims.json
  merge_checkpoint.py        trained checkpoint -> Hugging Face folder for serving
  evaluate.py                route -> answer -> verify on held-out questions, with any PGxRoute router
recipe/algo/                 training entry point, trainer and its Hydra config
lead_agent/llm_agent/        rollout loop (generation_quick3.py): tool parsing, tool execution, reward
verl/                        the verl this training uses (a modified 0.4.0.dev; see LICENSE and NOTICE)
```

`train.sh` puts this folder, `configs/`, `../PGxRoute` and `../Verifier` on `PYTHONPATH` and sets
`CONFIG_ROOT=configs`; the trainer passes the same paths to its Ray workers.

## Licence

The verl code in `verl/` and the training code derived from it are under the Apache License 2.0
(`LICENSE`, `NOTICE`). The rest of this folder: Apache License 2.0.
