# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CoVaPh GRPO Training Main Module
Trains an orchestrator model to select the best CoVaPh pipeline mode.
"""

import os
import sys
import json
import hydra
import ray
import time
import torch
from pathlib import Path
from collections import defaultdict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Add paths
SCRIPT_DIR = Path(__file__).parent.resolve()
TRAINING_DIR = SCRIPT_DIR.parent.parent.resolve()   # Router-training/
REPO_ROOT = TRAINING_DIR.parent.resolve()            # holds PGxRoute/ and Verifier/

# `verl`, `recipe`, `lead_agent` and the training modules live under TRAINING_DIR.
sys.path.insert(0, str(TRAINING_DIR))

# Import the canonical CoVaPh reward manager (single source of truth).
# Must come after sys.path setup since covaph_reward.py lives in TRAINING_DIR.
from covaph_reward import CoVaPhRewardManager

# Debug: Print paths for troubleshooting
if os.getenv("DEBUG_PATHS"):
    print(f"SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"TRAINING_DIR: {TRAINING_DIR}")
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"sys.path: {sys.path[:3]}")

# Hydra searchpath in grpo_trainer_covaph.yaml uses `file://verl/trainer/config` (relative to cwd).
# Ray unpacks working_dir to a temp root that already contains `verl/`; chdir to recipe/algo breaks that.
os.chdir(TRAINING_DIR)

# Import verl - ensure TRAINING_DIR is in path
try:
    from verl.trainer.ppo.reward import get_custom_reward_fn
except ImportError as e:
    print(f"ERROR: Cannot import verl module")
    print(f"TRAINING_DIR: {TRAINING_DIR}")
    print(f"verl path should be: {TRAINING_DIR / 'verl'}")
    print(f"verl exists: {(TRAINING_DIR / 'verl').exists()}")
    print(f"sys.path contains TRAINING_DIR: {str(TRAINING_DIR) in sys.path}")
    raise

# Import RayGRPOTrainer - use absolute import since we're executing directly
# Need to ensure recipe is importable from TRAINING_DIR
try:
    from recipe.algo.grpo_ray_trainer_quick3 import RayGRPOTrainer
except ImportError:
    # Fallback: try direct import if recipe is not a package
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "grpo_ray_trainer_quick3",
        SCRIPT_DIR / "grpo_ray_trainer_quick3.py"
    )
    grpo_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grpo_module)
    RayGRPOTrainer = grpo_module.RayGRPOTrainer



@hydra.main(config_path=str(SCRIPT_DIR / "config"), config_name="grpo_trainer_covaph", version_base=None)
def main(config):
    # Log tee is set up inside TaskRunner.run() (the Ray actor where all
    # real work happens).  Setting it up here as well would create a second
    # log file with nearly-identical content because Ray forwards the
    # actor's stdout back to the driver.
    run_grpo(config)


def run_grpo(config) -> None:
    """Main GRPO training function."""
    # Fail fast on a reward/advantage mismatch: cache_baseline rewards are already the
    # final advantage and must pair with the identity 'cache_baseline' estimator; otherwise GRPO would
    # silently re-normalize them per group (and vice versa: the identity estimator on min-max rewards
    # would train on un-centered rewards).
    _adv_mode = os.environ.get("REWARD_ADV_MODE", "grpo").strip().lower()
    _est = str(config.algorithm.adv_estimator)
    if (_adv_mode == "cache_baseline") != (_est == "cache_baseline"):
        raise ValueError(f"REWARD_ADV_MODE={_adv_mode!r} but algorithm.adv_estimator={_est!r}: "
                         f"set both to 'cache_baseline' or neither.")
    print(f"[Advantage] REWARD_ADV_MODE={_adv_mode} algorithm.adv_estimator={_est}")
    if not ray.is_initialized():
        # When using ray job submit (--address), we're connecting to existing cluster
        # and must not provide num_cpus/num_gpus.
        # For local initialization, Ray auto-detects resources, so not passing num_cpus is fine too.
        # Therefore, always initialize without num_cpus/num_gpus.
        # IMPORTANT:
        # - If we are connecting to an existing Ray cluster (e.g., ray job submit / RAY_ADDRESS),
        #   the Job already has a runtime_env. Passing another runtime_env here will conflict.
        # - Only set runtime_env for local, self-initialized runs.
        connecting_address = (os.getenv("RAY_ADDRESS") or "").strip()
        if connecting_address:
            ray.init(address=connecting_address, ignore_reinit_error=True)
        else:
            ray.init(
                runtime_env={
                    # Ship the training folder; PGxRoute and Verifier are imported from REPO_ROOT.
                    "working_dir": str(TRAINING_DIR),
                    "excludes": ["/checkpoints/", "/outputs/", "/data/"],
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                        "VLLM_LOGGING_LEVEL": "WARN",
                        "PYTHONPATH": ":".join([str(TRAINING_DIR), str(REPO_ROOT / "PGxRoute"),
                                                str(REPO_ROOT / "Verifier"),
                                                os.getenv("PYTHONPATH", "")]).rstrip(":"),
                        "BASE_DIR": str(TRAINING_DIR),
                        "CONFIG_ROOT": str(TRAINING_DIR),
                        "PGX_REPO_ROOT": str(REPO_ROOT),
                    },
                },
                ignore_reinit_error=True,
            )
    
    runner = TaskRunner.remote()
    try:
        ray.get(runner.run.remote(config))
    finally:
        # Always clean up Ray actors on exit (success or failure)
        # to prevent GPU memory leaks from orphaned workers
        ray.kill(runner)
        ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        import time
        from run_log import setup_run_log

        log_dir = os.environ.get("RESULTS_DIR") or os.path.dirname(config.trainer.default_local_dir)
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        setup_run_log(os.path.join(log_dir, f"run_log_{ts}.log"))

        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        
        OmegaConf.resolve(config)
        model_path = config.actor_rollout_ref.model.path
        if not model_path or not isinstance(model_path, str) or len(model_path) == 0:
            raise ValueError(
                f"Model path is empty or invalid: {model_path}. "
                f"Please set actor_rollout_ref.model.path in your configuration."
            )
        local_path = copy_to_local(model_path)
        
        from verl.utils import hf_processor, hf_tokenizer
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)
        
        # Configure workers based on strategy
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError
        
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        
        # Initialize CoVaPh reward manager
        reward_manager_name = config.reward_manager.get("type", "naive")
        if reward_manager_name == 'covaph':
            reward_fn = CoVaPhRewardManager(
                tokenizer=tokenizer,
                num_examine=1,
            )
        elif reward_manager_name == 'match':
            # Fallback to original reward manager
            from .main_grpo_quick3 import RewardManager
            reward_fn = RewardManager(
                tokenizer=tokenizer,
                num_examine=1,
                use_format_score=True
            )
        else:
            raise NotImplementedError(f"Unknown reward manager: {reward_manager_name}")
        
        strategy = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False
        )
        val_reward_fn = None
        
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping
        )
        
        trainer = RayGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )
        
        trainer.init_workers()
        
        # Check if we should stop after model loading (for debugging/preloading)
        stop_after_load = os.getenv('STOP_AFTER_MODEL_LOAD', 'false').lower() == 'true'
        if stop_after_load:
            print("=" * 80)
            print("MODELS LOADED SUCCESSFULLY!")
            print("=" * 80)
            print("\nStopped after model loading as requested (STOP_AFTER_MODEL_LOAD=true).")
            print("Models are loaded in Ray workers and ready for training.")
            print("\nTo continue training, run the same script with STOP_AFTER_MODEL_LOAD=false")
            print("or remove the environment variable.")
            
            # Create a ready file to signal that models are loaded
            ready_file = os.getenv('MODEL_READY_FILE', '/tmp/covaph_models_ready.flag')
            with open(ready_file, 'w') as f:
                f.write(f"Models loaded at: {os.getenv('BASE_DIR', 'unknown')}\n")
                f.write(f"Timestamp: {time.time()}\n")
            
            # Keep the workers alive by waiting (or you can exit and reconnect later)
            # For now, we'll wait indefinitely to keep workers alive
            print("\nKeeping workers alive. Press Ctrl+C to stop (this will unload models).")
            while True:
                time.sleep(60)
        
        trainer.fit()


if __name__ == "__main__":
    main()

