#!/usr/bin/env bash
# Train the LLM router (Qwen3-8B) with GRPO on live rewards, using the verl in this folder.
#
#   bash train.sh data/fg/fold0.jsonl data/fg/gt_claims.json router_fg_fold0 [hydra overrides ...]
#
# Every step, the questions in the batch are answered by all three PGxRoute pipelines and scored
# by the Verifier (router_reward/live_reward.py), so their model servers must be up and configured through the
# environment before launching (see README.md). COST_WEIGHT sets the token-cost penalty.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USAGE="usage: bash train.sh <train.jsonl> <gt_claims.json> <run name> [overrides ...]"
TRAIN_FILE="$(cd "$(dirname "${1:?$USAGE}")" && pwd)/$(basename "$1")"
GT_FILE="$(cd "$(dirname "${2:?$USAGE}")" && pwd)/$(basename "$2")"
RUN_NAME="${3:?$USAGE}"
shift 3

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
N_GPUS="${N_GPUS:-4}"
CKPT_DIR="${CKPT_DIR:-$HERE/checkpoints}"
OUT_DIR="${OUT_DIR:-$HERE/outputs/$RUN_NAME}"

export REWARD_ADV_MODE=cache_baseline
export COST_WEIGHT="${COST_WEIGHT:-0}"
export STRICT_TOOL_PARSE=1
export GT_CLAIMS_FILE="$GT_FILE"
export BASE_DIR="$HERE" CONFIG_ROOT="$HERE/configs" PGX_REPO_ROOT="$(dirname "$HERE")"
export PYTHONPATH="$HERE:$HERE/configs:$PGX_REPO_ROOT/PGxRoute:$PGX_REPO_ROOT/Verifier${PYTHONPATH:+:$PYTHONPATH}"
export RAY_USAGE_STATS_ENABLED=0 PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$HERE"
python3 -u -m recipe.algo.main_grpo_covaph \
    data.shuffle=True \
    algorithm.adv_estimator=cache_baseline \
    data.train_files="$TRAIN_FILE" \
    data.train_tool_config_path="$HERE/configs/covaph_tools_3tool.json" \
    data.test_tool_config_path="$HERE/configs/covaph_tools_3tool.json" \
    +data.my_output_dir="$OUT_DIR" \
    +data.cur_transfer_dir="$OUT_DIR/transfer" \
    +data.model_type="$MODEL_PATH" \
    +data.exp_tag=covaph \
    +data.use_llm_reward=true \
    +data.efficiency_reward=true \
    data.prompt_template=qwen-base \
    data.train_batch_size=16 \
    data.gen_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.truncation=left \
    +data.prompt_mode=pgx \
    data.rollout_diversity_hint_until=0 \
    +data.vllm_model_configs="$HERE/configs/vllm_model_configs.json" \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.metric=seq_reward \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.num_cycles=0.5 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.clip_ratio_low=0.1 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    critic.model.enable_activation_offload=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=10 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.correct_sample_advantage_boost_value=0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    stop_properly_penalty.penalty_coef=0.1 \
    trainer.critic_warmup=0 \
    'trainer.logger=["console"]' \
    trainer.project_name=pgx_router \
    trainer.experiment_name="$RUN_NAME" \
    trainer.log_val_generations=6 \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=3 \
    max_turns=1 \
    trainer.default_local_dir="$CKPT_DIR/$RUN_NAME" \
    trainer.total_epochs=36 \
    trainer.total_training_steps=36 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.resume_mode=auto \
    'actor_rollout_ref.actor.checkpoint.contents=[model,extra]' \
    +trainer.save_critic_checkpoint=False \
    reward_manager.type=covaph \
    "$@"
