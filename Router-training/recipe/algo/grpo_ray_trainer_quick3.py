# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------------

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import time
import numpy as np
import ray
import os
import json
import torch
import tensordict
from tensordict import TensorDict
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
import copy
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from torchdata.stateful_dataloader import StatefulDataLoader
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask
from lead_agent.llm_agent.generation_quick3 import LLMGenerationManager, GenerationConfig

class DataToolProto(DataProto):

    @classmethod
    def from_dict(cls, tensors, non_tensors=None, meta_info=None, num_batch_dims=1, auto_padding=False):
        # assert len(tensors) > 0, 'tensors must not be empty'
        assert num_batch_dims > 0, 'num_batch_dims must be greater than zero'
        if non_tensors is not None:
            assert num_batch_dims == 1, 'only support num_batch_dims=1 when non_tensors is not None.'

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, \
                    f'Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}'

        for key, val in non_tensors.items():
            if batch_size is None:
                batch_size = len(val)
            else:
                if isinstance(batch_size,torch.Size):
                    assert batch_size[0] == len(val), f"batch_size: {batch_size}, len(val): {len(val)}"
                else:
                    assert batch_size == len(val), f"{key}, batch_size: {batch_size}, len(val): {len(val)}"
            non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def repeat(self, repeat_times=2, interleave=True):
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        if not 'repeat_id' in repeated_non_tensor_batch:
            repeated_non_tensor_batch['repeat_id'] = []
            index_count = defaultdict(int)
            for example_index in repeated_non_tensor_batch['index']:
                repeated_non_tensor_batch['repeat_id'].append(index_count[example_index])
                index_count[example_index] += 1
            repeated_non_tensor_batch['repeat_id'] = np.array(repeated_non_tensor_batch['repeat_id'], dtype=object)

        return DataToolProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

def _strip_pgx_from_prompt(prompt_messages):
    """Strip PGx Analysis from prompt messages for 'raw' prompt mode (question → actor).

    Transforms:
      - System message: removes Input Format / Instructions sections that reference PGx Analysis,
        replaces with a simpler instruction block.
      - User message: keeps only the ``## Question`` section, drops ``## PGx Analysis``.

    Returns a new list (does not mutate the original).
    """
    import copy, re
    out = copy.deepcopy(prompt_messages)
    for msg in out:
        if msg.get("role") == "system":
            content = msg["content"]
            # Remove "## Input Format" block (up to the next ## heading or end)
            content = re.sub(
                r"## Input Format\n.*?(?=\n## |\Z)", "", content, flags=re.DOTALL
            )
            # Rewrite "## Instructions" to remove PGx references
            content = re.sub(
                r"## Instructions\n.*?(?=\n## |\Z)",
                (
                    "## Instructions\n"
                    "- Read the question to understand the clinical intent.\n"
                    "- Based on the question semantics and the tool descriptions, choose exactly ONE tool.\n"
                ),
                content,
                flags=re.DOTALL,
            )
            # In Output Format, remove "PGx Analysis" from reasoning description
            content = content.replace(
                "based on the question and PGx Analysis",
                "based on the question",
            )
            # Clean up consecutive blank lines
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            msg["content"] = content

        elif msg.get("role") == "user":
            content = msg["content"]
            # Keep everything before "## PGx Analysis"
            pgx_idx = content.find("## PGx Analysis")
            if pgx_idx >= 0:
                content = content[:pgx_idx].rstrip()
            msg["content"] = content
    return out


class JsonlDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 file_path,
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 prompt_template='v1',
                 tool_config_path=None,
                 vllm_model_config_path=None,
                 my_output_dir=None,
                 cur_transfer_dir=None,
                 model_type=None,
                 prompt_mode='pgx',
        ):

        self.file_path = file_path
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.model_type = model_type
        self.prompt_mode = prompt_mode

        self.dataset = []
        
        # Helper function to resolve file path (defined as instance method)
        def resolve_file_path(path_str):
            """Resolve file path, trying current dir, BASE_DIR env var, and original path"""
            path = os.path.expanduser(path_str)
            
            # If absolute path exists, use it
            if os.path.isabs(path) and os.path.exists(path):
                return path
            
            # Try relative to current working directory
            if os.path.exists(path):
                return os.path.abspath(path)
            
            # Try relative to BASE_DIR environment variable (set by Ray job)
            base_dir = os.getenv('BASE_DIR')
            if base_dir:
                candidate = os.path.join(base_dir, path)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
            
            # Try relative to PGX_ROOT environment variable
            pgx_root = os.getenv('PGX_ROOT')
            if pgx_root:
                # Try training directory under PGX_ROOT
                candidate = os.path.join(pgx_root, 'ToolOrchestra', 'training', path)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
                # Try direct under PGX_ROOT
                candidate = os.path.join(pgx_root, path)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
            
            # If still not found, return original path (will raise error on open)
            return path
        
        # Store resolve_file_path as instance method so it can be used in __getitem__
        self.resolve_file_path = resolve_file_path
        
        # Load dataset files
        files_to_load = [self.file_path] if isinstance(self.file_path, str) else self.file_path
        loaded_count = 0
        
        for file_path_item in files_to_load:
            resolved_path = resolve_file_path(file_path_item)
            
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"File not found: {file_path_item} (resolved to: {resolved_path})")
            
            try:
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    file_lines = 0
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                self.dataset.append(json.loads(line))
                                file_lines += 1
                            except json.JSONDecodeError as e:
                                print(f"[JsonlDataset] WARNING: Failed to parse JSON line in {resolved_path}: {e}")
                                continue
                    print(f"[JsonlDataset] Loaded {file_lines} lines from {resolved_path}")
                    loaded_count += file_lines
            except Exception as e:
                error_msg = f"Error reading file {resolved_path}: {e}"
                print(f"[JsonlDataset] ERROR: {error_msg}")
                raise
        
        print(f"[JsonlDataset] Total loaded: {len(self.dataset)} items from {len(files_to_load)} file(s)")
        print(f"[JsonlDataset] prompt_mode={self.prompt_mode} ({'question only' if self.prompt_mode == 'raw' else 'question + PGx Analysis'})")
        if len(self.dataset) == 0:
            raise ValueError(f"No data loaded from files: {files_to_load}. Check that files exist and contain valid JSONL data.")

        if self.prompt_mode == 'raw':
            n_stripped = 0
            for row in self.dataset:
                if 'prompt' in row and isinstance(row['prompt'], list):
                    row['prompt'] = _strip_pgx_from_prompt(row['prompt'])
                    n_stripped += 1
            print(f"[JsonlDataset] Stripped PGx Analysis from {n_stripped}/{len(self.dataset)} prompts (raw mode)")

        # Validate pref_vec presence — empty pref_vec causes silent fallback
        # to default weights in reward computation (generation_quick3.py).
        n_missing_pref = sum(
            1 for row in self.dataset
            if not row.get('pref_vec') or (isinstance(row.get('pref_vec'), dict) and len(row['pref_vec']) == 0)
        )
        if n_missing_pref > 0:
            print(f"[JsonlDataset] WARNING: {n_missing_pref}/{len(self.dataset)} samples have "
                  f"empty or missing 'pref_vec'. These will use default weights "
                  f"(accuracy=1.0, latency=0.1, cost=0.1) during reward computation.")

        with open(vllm_model_config_path) as vllm_model_configs_file:
            vllm_model_configs = json.load(vllm_model_configs_file)
        self.vllm_model_configs = vllm_model_configs
        self.my_output_dir = my_output_dir
        self.cur_transfer_dir = cur_transfer_dir
        self.tool_config_path = tool_config_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[item]
        row_dict['turn_id'] = 0
        row_dict['id'] = row_dict.get('index', item)
        row_dict['vllm_model_configs'] = self.vllm_model_configs
        row_dict['my_output_dir'] = self.my_output_dir
        row_dict['cur_transfer_dir'] = self.cur_transfer_dir
        row_dict['model_type'] = self.model_type
        
        # Extract fields from extra_info if available (for PGx format)
        if 'extra_info' in row_dict and isinstance(row_dict['extra_info'], dict):
            extra_info = row_dict['extra_info']
            
            # Extract question and problem
            if 'question' in extra_info:
                row_dict['question'] = extra_info['question']
                row_dict['problem'] = extra_info['question']  # Alias for compatibility
            if 'answer' in extra_info:
                row_dict['answer'] = extra_info['answer']
            
            # Extract category (required for generation_quick3.py)
            if 'category' in extra_info:
                row_dict['category'] = extra_info['category']
            else:
                # Default to 'func_call' for PGx (since PGx uses tool calling)
                row_dict['category'] = 'func_call'
            
            # Extract tools_kwargs if available (needed for tool execution)
            if 'tools_kwargs' in extra_info:
                row_dict['tools_kwargs'] = extra_info['tools_kwargs']
            
            # Extract expected_mode if available
            if 'expected_mode' in extra_info:
                row_dict['expected_mode'] = extra_info['expected_mode']
            # Extract level (key-derived: hard / filtered / simple) for summary
            if 'level' in extra_info:
                row_dict['level'] = extra_info['level']
        else:
            # If no extra_info, set default category for PGx
            row_dict['category'] = 'func_call'
        
        # Load tools from tool_config_path if available and tools not already set
        if 'tools' not in row_dict and self.tool_config_path:
            try:
                import json
                tool_config_path_resolved = self.resolve_file_path(self.tool_config_path)
                if os.path.exists(tool_config_path_resolved):
                    with open(tool_config_path_resolved, 'r') as f:
                        tools_schemas = json.load(f)
                        # Convert to OpenAI format if needed
                        row_dict['tools'] = tools_schemas if isinstance(tools_schemas, list) else [tools_schemas]
                else:
                    # If tool config not found, use empty list (will be loaded later if needed)
                    row_dict['tools'] = []
            except Exception as e:
                print(f"Warning: Could not load tools from {self.tool_config_path}: {e}")
                row_dict['tools'] = []
        
        # Set default values for PGx-specific fields if missing
        # These fields are expected by generation_quick3.py
        if 'tools' not in row_dict:
            # Default: empty tools list (will be populated from tools_kwargs if available)
            row_dict['tools'] = []
        
        if 'model_mapping' not in row_dict:
            # Default model mapping (can be customized per PGx needs)
            row_dict['model_mapping'] = {}
        
        if 'tool_pricing' not in row_dict:
            # Default tool pricing (placeholder, may need to be configured)
            row_dict['tool_pricing'] = {}
        
        if 'pref_vec' not in row_dict:
            # Default preference vector (empty dict)
            # Note: pref_vec should be a dict with keys like 'accuracy', 'latency', 'cost', etc.
            row_dict['pref_vec'] = {}
        elif isinstance(row_dict.get('pref_vec'), list):
            # Convert list to dict if it's a list (for backward compatibility)
            # If it's an empty list, use empty dict
            if len(row_dict['pref_vec']) == 0:
                row_dict['pref_vec'] = {}
            else:
                # If list has items, try to convert (this is a fallback, ideally pref_vec should be a dict)
                print(f"Warning: pref_vec is a list, expected dict. Converting list to dict may not work correctly.")
                row_dict['pref_vec'] = {}
        
        if 'example' not in row_dict:
            # Default example (can use the row_dict itself or None)
            row_dict['example'] = row_dict.copy() if row_dict else None
        
        return row_dict

def collate_fn(data_list):
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output

class RayGRPOTrainer(RayPPOTrainer):
                
    def _create_dataloader(self,place_holder1,place_holder2,place_holder3,train_sampler):
        from torch.utils.data import DataLoader
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        self.train_dataset = JsonlDataset(
                                    file_path=self.config.data.train_files,
                                    tokenizer=self.tokenizer,
                                    prompt_key=self.config.data.prompt_key,
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    filter_prompts=True,
                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    truncation='error',
                                    prompt_template=self.config.data.prompt_template,
                                    tool_config_path=self.config.data.train_tool_config_path,
                                    vllm_model_config_path=self.config.data.vllm_model_configs,
                                    my_output_dir=self.config.data.my_output_dir,
                                    cur_transfer_dir=self.config.data.cur_transfer_dir,
                                    model_type=self.config.data.model_type,
                                    prompt_mode=self.config.data.get('prompt_mode', 'pgx'),
                                )

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        if self.val_reward_fn is not None:
            self.val_dataset = JsonlDataset(
                                    file_path=self.config.data.val_files,
                                    tokenizer=self.tokenizer,
                                    prompt_key=self.config.data.prompt_key,
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    filter_prompts=True,
                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    truncation='error',
                                    prompt_template=self.config.data.prompt_template,
                                    tool_config_path=self.config.data.test_tool_config_path,
                                    vllm_model_config_path=self.config.data.vllm_model_configs,
                                    my_output_dir=self.config.data.my_output_dir,
                                    cur_transfer_dir=self.config.data.cur_transfer_dir,
                                    model_type=self.config.data.model_type,
                                    prompt_mode=self.config.data.get('prompt_mode', 'pgx'),
                                )

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.data.val_batch_size,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
            val_dataset_len = len(self.val_dataset)
            val_batch_size = self.config.data.val_batch_size
            if val_dataset_len == 0:
                raise ValueError(
                    f"Validation dataset is empty (0 items). "
                    f"Config val_files: {self.config.data.val_files}. "
                    f"Please check that the file(s) exist and contain valid JSONL data."
                )
            assert len(self.val_dataloader) >= 1, (
                f"Val dataloader has {len(self.val_dataloader)} batches "
                f"(dataset: {val_dataset_len} items, batch_size: {val_batch_size})"
            )
        else:
            self.val_dataset = None
            self.val_dataloader = None
            print("[Dataloader] Validation disabled (val_reward_fn=None), skipping val dataset loading")

        train_dataset_len = len(self.train_dataset)
        train_batch_size = self.config.data.train_batch_size

        if self.val_dataloader is not None:
            print(f"[Dataloader] train={train_dataset_len} samples / bs={train_batch_size} -> {len(self.train_dataloader)} batches  |  "
                  f"val={len(self.val_dataset)} samples / bs={self.config.data.val_batch_size} -> {len(self.val_dataloader)} batches")
        else:
            print(f"[Dataloader] train={train_dataset_len} samples / bs={train_batch_size} -> {len(self.train_dataloader)} batches  |  val=disabled")

        if train_dataset_len == 0:
            raise ValueError(f"Train dataset is empty. Config train_files: {self.config.data.train_files}")
        if len(self.train_dataloader) == 0:
            raise ValueError(f"Train dataloader has 0 batches (dataset={train_dataset_len}, bs={train_batch_size}, drop_last=True)")
        assert len(self.train_dataloader) >= 1, (
            f"Train dataloader has {len(self.train_dataloader)} batches "
            f"(dataset: {train_dataset_len} items, batch_size: {train_batch_size})"
        )

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _create_loss_mask(self, batch, metrics):
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        batch.batch['loss_mask'] = response_mask
        return batch, metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Get retriever config with defaults if not present
        # GenerationConfig has defaults: search_url=None, topk=3
        search_url = OmegaConf.select(self.config, "retriever.url", default=None)
        topk = OmegaConf.select(self.config, "retriever.topk", default=3)

        # Get min_reward_std_threshold from actor config, default to 0.1
        min_reward_std_threshold = OmegaConf.select(
            self.config, 
            "actor_rollout_ref.actor.min_reward_std_threshold", 
            default=0.1
        )
        rollout_diversity_hint_until = int(OmegaConf.select(self.config, "data.rollout_diversity_hint_until", default=-1))
        rollout_n_agent = OmegaConf.select(self.config, "actor_rollout_ref.rollout.n_agent", default=1)
        print(f"[Training] GenerationConfig: rollout_diversity_hint_until={rollout_diversity_hint_until}, rollout_n_agent={rollout_n_agent} (hint adds a per-rollout tool preference to the prompt)")
        
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            search_url=search_url,
            topk=topk,
            min_reward_std_threshold=min_reward_std_threshold,
            rollout_diversity_hint_until=rollout_diversity_hint_until,
            rollout_n_agent=rollout_n_agent,
        )

        # Check if reward model is available
        rm_wg_to_pass = None
        if hasattr(self, 'rm_wg') and self.use_rm:
            rm_wg_to_pass = self.rm_wg
        
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            train_tool_config_path=self.config.data.train_tool_config_path,
            test_tool_config_path=self.config.data.test_tool_config_path,
            rm_wg=rm_wg_to_pass,  # Pass reward model if available
        )

        # load checkpoint before doing anything
        self._load_checkpoint()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        print(f"[Training] Starting training loop. Total epochs: {self.config.trainer.total_epochs}, "
              f"DataLoader batches: {len(self.train_dataloader)}")
        
        skipped_batches = 0  # Track batches skipped due to insufficient samples
        for epoch in range(self.config.trainer.total_epochs):
            print(f"[Training] Starting epoch {epoch + 1}/{self.config.trainer.total_epochs}")
            batch_idx = 0
            for batch_dict in self.train_dataloader:
                batch_idx += 1
                metrics = {}
                timing_raw = {}
                batch: DataToolProto = DataToolProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        # Get optional config values with defaults
                        topk_doc = OmegaConf.select(self.config.data, "topk_doc", default=3)
                        use_llm_reward = OmegaConf.select(self.config.data, "use_llm_reward", default=True)
                        efficiency_reward = OmegaConf.select(self.config.data, "efficiency_reward", default=False)
                        exp_tag = OmegaConf.select(self.config.data, "exp_tag", default="pgx")
                        use_qa_reward = OmegaConf.select(self.config.data, "use_qa_reward", default=False)
                        
                        final_gen_batch_output,tool_avg = generation_manager.run_llm_loop(
                            gen_batch=batch,
                            tokenizer_config={
                                'tokenizer': self.tokenizer,
                                'max_prompt_length': self.config.data.max_prompt_length,
                                'max_response_length': self.config.data.max_response_length,
                                'truncation': 'error'
                            },
                            global_steps=self.global_steps,
                            topk_doc=topk_doc,
                            use_llm_reward=use_llm_reward,
                            efficiency_reward=efficiency_reward,
                            exp_tag=exp_tag,
                            use_qa_reward=use_qa_reward
                        )
                    # Check if run_llm_loop returned None (insufficient samples for training)
                    if final_gen_batch_output is None:
                        skipped_batches += 1
                        total_batches_so_far = batch_idx + (epoch * len(self.train_dataloader))
                        print(f"[Training] WARNING: run_llm_loop returned None for batch {batch_idx} "
                              f"(insufficient samples after filtering). Skipping this batch. "
                              f"[{skipped_batches} skipped / {total_batches_so_far} total batches so far]")
                        if skipped_batches > 0 and skipped_batches == total_batches_so_far:
                            print(f"[Training] CRITICAL: ALL {skipped_batches} batches have been skipped so far! "
                                  f"Check your training data quality and filtering thresholds.")
                        continue
                    
                    # Cast only integer-semantic tensors to long; leave any float
                    # tensors (e.g. pre-computed scores) untouched.
                    _INT_KEYS = {'input_ids', 'attention_mask', 'position_ids', 'responses'}
                    for key in final_gen_batch_output.batch.keys():
                        if key in _INT_KEYS:
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()
                    batch = final_gen_batch_output
                    batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                    assert self.config.actor_rollout_ref.rollout.n==1
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    for key in batch.batch.keys():
                        if key in _INT_KEYS:
                            batch.batch[key] = batch.batch[key].long()
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        
                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        reward_tensor = self.reward_fn(batch, global_step=self.global_steps)
                        batch.batch['token_level_scores'] = reward_tensor

                        # Concise reward summary (per-example, not per-token)
                        example_rewards = reward_tensor.sum(dim=-1)
                        n_examples = example_rewards.numel()
                        n_nonzero = (example_rewards != 0).sum().item()
                        if n_nonzero > 0:
                            nz = example_rewards[example_rewards != 0]
                            print(f"[Reward] step={self.global_steps}  examples={n_examples}  "
                                  f"nonzero={n_nonzero}  mean={nz.mean():.4f}  "
                                  f"std={nz.std():.4f}  min={nz.min():.4f}  max={nz.max():.4f}")
                        else:
                            print(f"[Reward] step={self.global_steps}  WARNING: all {n_examples} example rewards are ZERO")
                        
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            # apply_kl_penalty expects batch to have token_level_rewards (reads then overwrites)
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"].clone()
                            if hasattr(self, "kl_ctrl_in_reward") and self.kl_ctrl_in_reward is not None:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            # else: token_level_rewards already set above (no KL penalty)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        batch = compute_advantage(batch,
                                                    adv_estimator=self.config.algorithm.adv_estimator,
                                                    gamma=self.config.algorithm.gamma,
                                                    lam=self.config.algorithm.lam,
                                                    num_repeat=self.config.actor_rollout_ref.rollout.n)
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            batch, metrics = self._create_loss_mask(batch, metrics)
                            print(f"[Update] step={self.global_steps} starting actor update...")
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        print(f"[Update] step={self.global_steps} done  "
                              f"pg_loss={actor_output_metrics.get('actor/pg_loss', 'N/A')}  "
                              f"grad_norm={actor_output_metrics.get('actor/grad_norm', 'N/A')}  "
                              f"ppo_kl={actor_output_metrics.get('actor/ppo_kl', 'N/A')}")

                    is_last_step = self.global_steps >= self.total_training_steps
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                logger.log(data=metrics, step=self.global_steps)
                
                self.global_steps += 1
                print(f"[Training] epoch={epoch+1}/{self.config.trainer.total_epochs}  "
                      f"batch={batch_idx}/{len(self.train_dataloader)}  step={self.global_steps}  "
                      f"pg_loss={metrics.get('actor/pg_loss', 'N/A')}")

                if self.global_steps > self.total_training_steps:
                    break
            if self.global_steps > self.total_training_steps:
                break







