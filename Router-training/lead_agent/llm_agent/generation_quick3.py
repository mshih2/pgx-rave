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

from math import remainder
import random
import torch
import re
import asyncio
from collections import defaultdict
import os
import copy
import json
import time
import subprocess
import csv
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.trainer.ppo.ray_trainer import compute_response_mask
import shutil
import requests
import sys
from pathlib import Path

# Require CONFIG_ROOT (passed via the Ray runtime_env): the folder holding config.py.
_env_mr = (os.environ.get("CONFIG_ROOT") or "").strip()
if not _env_mr:
    raise ImportError("Missing env CONFIG_ROOT (required to locate config.py on Ray workers).")
_CONFIG_ROOT = Path(_env_mr)
try:
    _root_ok = _CONFIG_ROOT.exists()
except OSError:
    _root_ok = False
if _root_ok:
    _p = str(_CONFIG_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -----------------------------------------------------------------------------
# Pin config to ${CONFIG_ROOT}/config.py (Ray runtime_env).
# -----------------------------------------------------------------------------
import importlib.util as _importlib_util

_pgx_config_path = _CONFIG_ROOT / "config.py"
try:
    _cfg_ok = _pgx_config_path.exists()
except OSError:
    _cfg_ok = False
if not _cfg_ok:
    raise ImportError(f"Missing config: {_pgx_config_path}")

_cfg_spec = _importlib_util.spec_from_file_location("config", _pgx_config_path)
if _cfg_spec is None or _cfg_spec.loader is None:
    raise ImportError(f"Failed to create import spec: {_pgx_config_path}")

_cfg_mod = _importlib_util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)
sys.modules["config"] = _cfg_mod

print(f"[Tool Orchestration][Debug] pinned config to: {getattr(_cfg_mod, '__file__', None)}")

for _req in ("USE_EXISTING_QUESTION_TOOL_RESULTS", "QUESTION_TOOL_RESULT_DIR"):
    if not hasattr(_cfg_mod, _req):
        raise ImportError(f"Config missing required {_req} in {_pgx_config_path}")

import multiprocessing as mp
from transformers import AutoTokenizer

from datetime import datetime
import string
def generate_random_string(length):
    """Generates a random string of specified length using alphanumeric characters."""
    characters = string.ascii_letters + string.digits + '~!@#$%^&*()-=_+[]'
    return ''.join(random.choice(characters) for _ in range(length))

# Lazy initialization of oss_client - only create when needed
# Don't initialize at module level to avoid requiring OSS_KEY on import
_oss_client = None

def get_oss_client():
    """Get or create the OpenAI client for OSS (NVIDIA API). Only creates it when needed."""
    global _oss_client
    if _oss_client is None:
        oss_key = os.getenv("OSS_KEY")
        if oss_key:
            _oss_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=oss_key
            )
        else:
            _oss_client = None
    return _oss_client

# Create a wrapper class to maintain backward compatibility with oss_client.chat.completions.create()
class OSSClientWrapper:
    """Wrapper to maintain backward compatibility with oss_client usage."""
    @property
    def chat(self):
        client = get_oss_client()
        if client is None:
            raise ValueError("OSS_KEY environment variable is not set. Cannot use OSS client.")
        return client.chat

oss_client = OSSClientWrapper()

def merge_documents(main_list,sub_list):
    if len(sub_list)==0:
        return main_list
    if len(main_list)<len(sub_list):
        return main_list+sub_list
    merged_list = []
    multiple = len(main_list)//len(sub_list)
    assert multiple>0
    idx_main = 0
    idx_sub = 0
    while idx_sub<len(sub_list):
        if not sub_list[idx_sub] in merged_list:
            merged_list.append(sub_list[idx_sub])
        assert idx_main+multiple<=len(main_list)
        for iter_idx in range(idx_main,idx_main+multiple):
            if not main_list[iter_idx] in merged_list:
                merged_list.append(main_list[iter_idx])
        idx_main = idx_main+multiple
        idx_sub += 1
    merged_list += main_list[multiple*len(sub_list):]
    return merged_list

# Router tool set: tool_a (API-only), tool_b (Reasoning-only), tool_c (Full pipeline).
ALL_TOOLS = {
    "tool_a": {},
    "tool_b": {},
    "tool_c": {},
}

# Actor is restricted to only these tools
VALID_PGX_TOOL_NAMES = ['tool_a', 'tool_b', 'tool_c']

# Tool name -> pipeline mode (router_reward/pgx_env.py maps modes to PGxRoute pipelines)
TOOL_TO_MODE = {
    'tool_a': 'skip_retrieve',
    'tool_b': 'module4_only',
    'tool_c': 'full_rules_top8',
}

# A row is a routing question when its category is one of the question tiers.
PGX_CATEGORIES = ('hard', 'intermediate', 'common')


def compute_cache_baseline_rewards(all_turn_index, all_turn_repeat_ids, all_turn_tools, active_tools, reward_cache,
                                   cost_weight: float = 0.0):
    """REWARD_ADV_MODE=cache_baseline.

    reward(rollout) = [F15[qid, chosen] - mean_{t in active_tools} F15[qid, t]]
                      - cost_weight * cost_feature[qid, chosen]
    cost_feature    = (cost[qid, chosen] - mean_t cost[qid, t]) / (max_t cost - min_t cost + 1e-6)   in [-1, 1]
    (cost = input+output tokens of the pipeline's answer; centered over the active tools like the accuracy term so the
    per-question mean of the total advantage stays 0; bigger cost -> larger feature -> more negative reward.)
    The (qid, tool) table is built live every step (live_reward.build_reward_table). This value is the FINAL advantage (pair with
    algorithm.adv_estimator=cache_baseline, an identity pass-through): no sibling-group mean/std
    re-normalization, so a question whose rollouts all picked the same tool still gets a correctly
    signed nonzero advantage, and magnitudes are preserved (a 0.07 gap != a 0.38 gap).

    Chosen tool is parsed from all_turn_tools[i] with the same convention as _process_weighted_scores.
    Guards:
      - an ACTIVE tool missing from the table for a qid -> RuntimeError (halt; the table must be complete)
      - cost_weight > 0 and any active tool has no numeric cost for a qid -> RuntimeError (halt)
      - chosen tool unparsable / not active -> scored as F15 = 0 (worst) and, if cost_weight > 0,
        the most expensive tool's cost_feature (worst), i.e. reward = -baseline - cost_weight * max_feature
    Returns (rewards_by_rollout_id, details_by_rollout_id).
    """
    rewards, details = {}, {}
    cost_weight = float(cost_weight or 0.0)
    n_idx = len(all_turn_index)
    tools_aligned = all_turn_tools is not None and len(all_turn_tools) == n_idx
    if not tools_aligned:
        raise RuntimeError(f"[Reward Cache][cache_baseline] all_turn_tools ({None if all_turn_tools is None else len(all_turn_tools)}) "
                           f"not aligned with all_turn_index ({n_idx}); cannot map rollouts to chosen tools")
    for i, (example_idx, repeat_id) in enumerate(zip(all_turn_index, all_turn_repeat_ids)):
        rollout_id = f"{example_idx}_____{repeat_id}"
        if rollout_id in rewards:
            continue
        tlist = all_turn_tools[i]
        tname = None
        if tlist:
            first = tlist[0]
            if isinstance(first, (list, tuple)) and len(first) > 0:
                tname = first[0]
            elif isinstance(first, str):
                tname = first
        per_tool = {}
        for t in active_tools:
            key = f"{example_idx}__{t}"
            entry = reward_cache.get(key)
            if entry is None or entry.get("F15") is None:
                raise RuntimeError(f"[Reward Cache][cache_baseline] missing F15 for key={key!r} (rollout {rollout_id}); "
                                   f"every active tool must be cached for every training qid")
            per_tool[t] = float(entry["F15"])
        baseline = sum(per_tool.values()) / len(per_tool)
        if tname in per_tool:
            chosen = per_tool[tname]
        else:
            chosen = 0.0
            print(f"[Reward Cache][cache_baseline][GUARD] rollout {rollout_id}: chosen tool={tname!r} not in active "
                  f"{list(active_tools)} -> F15=0 (reward = -baseline)")
        acc_adv = chosen - baseline
        # --- cost term (v8): centered + range-scaled token cost over the active tools ---
        cost_by_tool, cost_feature_by_tool, cost_feature, cost_term = None, None, 0.0, 0.0
        if cost_weight > 0:
            cost_by_tool = {}
            for t in active_tools:
                c = reward_cache[f"{example_idx}__{t}"].get("cost")
                if not isinstance(c, (int, float)) or isinstance(c, bool):
                    raise RuntimeError(f"[Reward Cache][cache_baseline] COST_WEIGHT={cost_weight} but key={example_idx}__{t} "
                                       f"has no numeric cost ({c!r}); every active tool needs a cost for every training qid")
                cost_by_tool[t] = abs(float(c))
            c_mean = sum(cost_by_tool.values()) / len(cost_by_tool)
            c_range = max(cost_by_tool.values()) - min(cost_by_tool.values()) + 1e-6
            cost_feature_by_tool = {t: (c - c_mean) / c_range for t, c in cost_by_tool.items()}
            cost_feature = cost_feature_by_tool[tname] if tname in cost_feature_by_tool else max(cost_feature_by_tool.values())
            cost_term = -cost_weight * cost_feature
        rewards[rollout_id] = acc_adv + cost_term
        details[rollout_id] = {"tool": tname, "F15_chosen": chosen, "F15_by_tool": per_tool,
                               "baseline_mean": baseline, "acc_adv": acc_adv,
                               "cost_weight": cost_weight, "cost_by_tool": cost_by_tool,
                               "cost_feature_by_tool": cost_feature_by_tool, "cost_feature": cost_feature,
                               "cost_term": cost_term, "reward": acc_adv + cost_term}
    return rewards, details

def is_pgx_category(category: str) -> bool:
    """Check if a row's category is a question tier (hard, intermediate or common)."""
    return str(category or '').strip().lower() in PGX_CATEGORIES

def get_pgx_mode_from_tool(tool_name: str) -> str:
    """Get pipeline mode from tool name."""
    return TOOL_TO_MODE.get(tool_name, 'full')


def _load_active_tools(tool_config_path):
    """Active PGx tool set = the tool names advertised in the selected tool-config JSON
    (TOOL_CONFIG / data.train_tool_config_path), restricted to VALID_PGX_TOOL_NAMES.
    Raises if the config is unreadable."""
    import json as _json, os as _os
    for p in [tool_config_path, _os.path.join(_os.environ.get("BASE_DIR", ""), tool_config_path or "")]:
        try:
            if p and _os.path.exists(p):
                names = []
                for t in _json.load(open(p)):
                    n = (t.get("function", {}) or {}).get("name") or t.get("name")
                    if n in VALID_PGX_TOOL_NAMES:
                        names.append(n)
                if names:
                    return names
        except Exception as e:
            print(f"[Tool Orchestration] WARN: could not read tool config {p!r}: {e}")
    raise RuntimeError(
        f"[Tool Orchestration] could not load active tools from {tool_config_path!r} "
        f"(tried the relative name and the BASE_DIR-joined path). Refusing to default to the "
        f"full set {VALID_PGX_TOOL_NAMES}, which would silently admit tools the run excludes. "
        f"Fix TOOL_CONFIG / the Ray working_dir so the config is readable."
    )


def _can_parse_pgx_prediction(prediction: str) -> bool:
    """Quick check: would `postprocess_predictions` succeed on this PGx prediction?

    Mirrors the success conditions in `postprocess_predictions` without raising.
    Returns True if exactly one PGx tool name can be uniquely identified.
    """
    if not isinstance(prediction, str) or not prediction:
        return False
    import re as _re
    if '<tool_call>' in prediction:
        # Look for a JSON-like {"name": "tool_X"} inside <tool_call> tags
        m = _re.findall(r'<tool_call>(.*?)</tool_call>', prediction, _re.DOTALL)
        for tag_content in m:
            for t in VALID_PGX_TOOL_NAMES:
                if f'"{t}"' in tag_content:
                    return True
        return False
    matches = _re.findall(
        rf'"name"\s*:\s*"({"|".join(VALID_PGX_TOOL_NAMES)})"',
        prediction,
    )
    matches = list(dict.fromkeys(matches))  # de-dup preserving order
    return len(matches) == 1


def _normalize_pgx_tool_arguments(args: Any) -> Dict[str, Any]:
    """Normalize PGx tool arguments. Models often output {"parameters": {}} per prompt; treat as empty."""
    if args is None or args == {}:
        return {}
    if isinstance(args, dict) and set(args.keys()) == {"parameters"}:
        val = args.get("parameters")
        if val is None or val == {} or val == []:
            return {}
    return args if isinstance(args, dict) else {}


def is_pgx_tool(tool_name: str) -> bool:
    """Check if a tool name is a valid PGx tool."""
    return tool_name in VALID_PGX_TOOL_NAMES


def _write_scores_result_csv_from_summary(summary_file: str) -> str:
    """
    Export rollout_details in summary.json to a flat CSV (scores_result.csv).
    This is a lightweight equivalent of running eval_score_reader.py after each global step.
    """
    if not os.path.isfile(summary_file):
        raise FileNotFoundError(summary_file)

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    rollout_details = summary.get("rollout_details", {}) or {}
    out_path = os.path.join(os.path.dirname(summary_file), "scores_result.csv")

    fieldnames = [
        "question_id",
        "repeat_id",
        "pipeline_name",
        "accuracy",
        "relevance",
        "completeness",
        "clarity",
        "weighted_score",
        "full_reward",
        "level",
        "rollout_id",
    ]

    def _get_metric(d: dict, key: str):
        if key in d:
            return d.get(key)
        titled = key[:1].upper() + key[1:]
        if titled in d:
            return d.get(titled)
        upper = key.upper()
        if upper in d:
            return d.get(upper)
        return None

    rows = []
    for rollout_id, details in rollout_details.items():
        if not isinstance(details, dict):
            continue
        qid = None
        rid = None
        if isinstance(rollout_id, str) and "_____" in rollout_id:
            parts = rollout_id.split("_____")
            if len(parts) >= 2:
                qid, rid = parts[0], parts[1]
        else:
            qid = str(rollout_id)
        rows.append(
            {
                "question_id": qid,
                "repeat_id": rid,
                "pipeline_name": details.get("pipeline_name"),
                "accuracy": _get_metric(details, "accuracy"),
                "relevance": _get_metric(details, "relevance"),
                "completeness": _get_metric(details, "completeness"),
                "clarity": _get_metric(details, "clarity"),
                "weighted_score": details.get("weighted_score"),
                "full_reward": details.get("full_reward"),
                "level": details.get("level"),
                "rollout_id": rollout_id,
            }
        )

    def _to_int_or_none(v):
        try:
            return int(v)
        except Exception:
            return None

    rows.sort(
        key=lambda r: (
            _to_int_or_none(r.get("question_id")) is None,
            _to_int_or_none(r.get("question_id")) if _to_int_or_none(r.get("question_id")) is not None else (r.get("question_id") or ""),
            _to_int_or_none(r.get("repeat_id")) is None,
            _to_int_or_none(r.get("repeat_id")) if _to_int_or_none(r.get("repeat_id")) is not None else (r.get("repeat_id") or ""),
            r.get("pipeline_name") or "",
        )
    )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return out_path

def _infer_question_type_from_id(question_id: Optional[str]) -> str:
    """Infer question type label from dataset key/id."""
    s = (str(question_id or "")).lower()
    if "hard" in s:
        return "hard"
    if "filtered" in s:
        return "filtered"
    return "common"

def _safe_cache_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s or "")).strip("_") or "unknown"

def _extract_question_id_from_arguments(arguments: Dict[str, Any]) -> str:
    """
    Return the stable question identifier (jsonl key) for caching.
    Do NOT use per-rollout item index (`id`) because it changes with batching/rollouts.
    """
    qid = arguments.get("cur_index")
    if qid is None or str(qid).strip() == "":
        for k in ("question_id", "key", "jsonl_key", "example_id", "query_id"):
            v = arguments.get(k)
            if v is not None and str(v).strip() != "":
                qid = v
                break
    if qid is None or str(qid).strip() == "":
        raise ValueError("Missing required jsonl key for caching (expected 'cur_index' or equivalent).")
    return str(qid)

def _extract_question_type_from_arguments(arguments: Dict[str, Any], question_id: Optional[str]) -> str:
    """
    Prefer extra_info.level (propagated as `level`) for question_type.
    Fallback to legacy heuristic if not available.
    """
    level = arguments.get("level")
    if level is not None and str(level).strip() != "":
        return str(level)
    return _infer_question_type_from_id(question_id)

def _write_json_atomic(path: Path, payload: Any) -> None:
    """
    Write JSON atomically (best-effort) to avoid partial files under Ray.
    Uses `default=str` to survive numpy/scalar/Path-like objects in payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

def cut_seq(tokenizer,seq,l):
    if len(seq)==0:
        return {
            'effective_length': 0,
            'string_after_cut': ''
        }
    token_ids = tokenizer(seq)['input_ids']
    rs = tokenizer.batch_decode(token_ids[-l:], skip_special_tokens=True)
    return {
        'effective_length': len(token_ids),
        'string_after_cut': ''.join(rs)
    }

def cut_middle_turns(tokenizer,messages,max_length):
    messages_str = ''
    start_identifier = generate_random_string(15)
    end_identifier = generate_random_string(15)
    assert not start_identifier in str(messages) and not end_identifier in str(messages) and start_identifier!=end_identifier
    for mid,m in enumerate(messages):
        messages_str += f"{m}{start_identifier}{mid}{end_identifier}"
    token_ids = tokenizer(str(messages_str))['input_ids']
    if len(token_ids)<=max_length:
        return messages
    p1_tokens = tokenizer.batch_decode(token_ids[:max_length//2])
    p1 = ''.join(p1_tokens)
    p1_idx = int(p1.split(start_identifier)[-1].split(end_identifier)[0])
    p2_tokens = tokenizer.batch_decode(token_ids[-max_length//2:])
    p2 = ''.join(p2_tokens)
    p2_idx = int(p2.split(end_identifier)[0].split(start_identifier)[-1])
    return messages[:p1_idx+1]+messages[p2_idx:]

def seq_length(tokenizer,seq):
    input_data = tokenizer(seq, return_tensors='pt', add_special_tokens=True)
    return len(tokenizer.convert_ids_to_tokens(input_data['input_ids'][0]))

def call_tool(arguments):
    start_time = time.time()
    category = arguments['category']
    category_is_pgx = is_pgx_category(category)
    
    if arguments['category']=='qa' or category_is_pgx:
        tool_name = arguments['tool']
        
        if is_pgx_tool(tool_name):
            # Optional cache: reuse existing per-question tool results to avoid re-running PGx pipeline
            from config import USE_EXISTING_QUESTION_TOOL_RESULTS, QUESTION_TOOL_RESULT_DIR
            import config as _cfg
            _cfg_path = getattr(_cfg, "__file__", None)
            print(f"[Tool Orchestration][Debug] imported config from: {_cfg_path}")

            mode = get_pgx_mode_from_tool(tool_name)
            question = arguments.get('problem') or arguments.get('question')
            if not question:
                raise ValueError("Missing required 'problem' or 'question' argument")
            query_id = arguments.get('id')
            if not query_id:
                raise ValueError("Missing required 'id' argument")
            # Use jsonl key for caching (stable across rollouts)
            question_id = _extract_question_id_from_arguments(arguments)
            ground_truth = arguments.get('answer')
            cur_output_dir = arguments.get('cur_output_dir')
            if not cur_output_dir:
                raise ValueError("Missing required 'cur_output_dir' argument")

            cache_path = None
            if QUESTION_TOOL_RESULT_DIR:
                cache_dir = Path(str(QUESTION_TOOL_RESULT_DIR)).expanduser().resolve()
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"{_safe_cache_name(str(question_id))}__{_safe_cache_name(str(tool_name))}.json"

            if USE_EXISTING_QUESTION_TOOL_RESULTS and cache_path and cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                pgx_result = cached.get("result") or {}
                # Fill required fields to match normal execution path
                latency_sec = pgx_result.get('latency_seconds') or 0.0
                arguments['latency'] = latency_sec
                arguments['tool'] = tool_name
                arguments['response'] = pgx_result.get('response')
                arguments['pgx_result'] = pgx_result
                arguments['cost'] = pgx_result.get('cost', 0)
                arguments['mode'] = pgx_result.get('mode') or mode
                print(f"[Tool Orchestration] ✓ Loaded cached PGx result: question_id={question_id}, tool={tool_name} (cache={cache_path.name})")
                return arguments
            
            # Import PGx tool runner.
            # Prefer the packaged ToolOrchestra within the current working dir (Ray working_dir_files),
            # and only fall back to external PGX_ROOT if needed.
            # Under Ray job submit, `working_dir` is the training folder. Import directly from there.
            from router_reward.pgx_env import call_pgx_tool
            
            category_label = category if category_is_pgx else 'qa'
            print(f"[Tool Orchestration] ========== PGx Tool Execution ({category_label} category) ==========")
            print(f"[Tool Orchestration] 🔧 Pipeline Mode: {mode.upper()} (Tool: {tool_name})")
            print(f"[Tool Orchestration] Query ID: {query_id}, Category: {category}")
            
            pgx_arguments = {
                'tool': 'pgx_query',  # Keep 'pgx_query' for compatibility with pgx_env
                'mode': mode,
                'question': question,
                'id': str(query_id),
                'answer': ground_truth,
                'output_dir': cur_output_dir,
            }
            
            exec_start_time = time.time()
            pgx_result = call_pgx_tool(pgx_arguments)
            exec_end_time = time.time()
            # Use pipeline latency (module 1 start → module 4 end) when available; else wall clock
            latency_sec = pgx_result.get('latency_seconds')
            if latency_sec is None:
                latency_sec = exec_end_time - exec_start_time
            arguments['latency'] = latency_sec
            
            arguments['tool'] = tool_name
            arguments['response'] = pgx_result['response']
            arguments['pgx_result'] = pgx_result
            arguments['cost'] = pgx_result['cost']
            arguments['mode'] = mode
            # Preserve item_index if present (needed for marking as used after execution)
            if 'item_index' in arguments:
                arguments['item_index'] = arguments['item_index']
            
            print(f"[Tool Orchestration] ✓ PGx tool execution completed ({category_label} category) in {latency_sec:.2f}s (pipeline module1→module4)")
            print(f"[Tool Orchestration] Cost: {arguments['cost']}")

            # Save per-question tool result for reuse
            if cache_path:
                payload = {
                    "question_id": str(question_id),
                    "question": question,
                    "question_type": _extract_question_type_from_arguments(arguments, question_id=question_id),
                    "tool_name": str(tool_name),
                    "mode": str(mode),
                    "result": pgx_result,
                }
                _write_json_atomic(cache_path, payload)
                print(f"[Tool Orchestration] ✓ Saved cached PGx result to {cache_path}")
            
            return arguments
        
        # Only PGx tools are supported
        raise ValueError(f"Unsupported tool '{arguments['tool']}' called. Only {VALID_PGX_TOOL_NAMES} are allowed in PGx training.")
    elif arguments['category']=='func_call':
        # For PGx training, only support PGx tools in func_call category
        assert isinstance(arguments['tool'],list)
        cost = 0
        cur_tool_pricing = arguments['cur_tool_pricing']
        valid_tools = []
        
        # Filter only PGx tools - remove all other tools
        for iter_one_tool_call in arguments['tool']:
            if isinstance(iter_one_tool_call, dict):
                tool_name = iter_one_tool_call['name']
                if is_pgx_tool(tool_name):
                    valid_tools.append(iter_one_tool_call)
                else:
                    raise ValueError(f"Unsupported tool '{tool_name}' in func_call category. Only {VALID_PGX_TOOL_NAMES} are allowed.")
        
        # For PGx training, only execute PGx tools directly
        # No call_expert or intermediate LLM calls needed
        if len(valid_tools) == 0:
            raise ValueError("No valid PGx tool call found in func_call category")
        
        # Execute PGx tools directly
        for tool_call in valid_tools:
            if isinstance(tool_call, dict):
                tool_name = tool_call['name']
                if not is_pgx_tool(tool_name):
                    raise ValueError(f"Tool '{tool_name}' is not a PGx tool")

                # Optional cache: reuse existing per-question tool results to avoid re-running PGx pipeline
                from config import USE_EXISTING_QUESTION_TOOL_RESULTS, QUESTION_TOOL_RESULT_DIR
                import config as _cfg
                _cfg_path = getattr(_cfg, "__file__", None)
                print(f"[Tool Orchestration][Debug] imported config from: {_cfg_path}")

                mode = get_pgx_mode_from_tool(tool_name)
                question = arguments.get('problem') or arguments.get('question')
                if not question:
                    raise ValueError("Missing required 'problem' or 'question' argument")
                query_id = arguments.get('id')
                if not query_id:
                    raise ValueError("Missing required 'id' argument")
                question_id = _extract_question_id_from_arguments(arguments)
                ground_truth = arguments.get('answer')
                cur_output_dir = arguments.get('cur_output_dir')
                if not cur_output_dir:
                    raise ValueError("Missing required 'cur_output_dir' argument")

                cache_path = None
                if QUESTION_TOOL_RESULT_DIR:
                    cache_dir = Path(str(QUESTION_TOOL_RESULT_DIR)).expanduser().resolve()
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = cache_dir / f"{_safe_cache_name(str(question_id))}__{_safe_cache_name(str(tool_name))}.json"

                if USE_EXISTING_QUESTION_TOOL_RESULTS and cache_path and cache_path.exists():
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    pgx_result = cached.get("result") or {}
                    latency_sec = pgx_result.get('latency_seconds') or 0.0
                    arguments['latency'] = latency_sec
                    print(f"[Tool Orchestration] ✓ Loaded cached PGx result (func_call): question_id={question_id}, tool={tool_name} (cache={cache_path.name})")
                    arguments['tool'] = 'pgx_query'
                    arguments['response'] = pgx_result.get('response')
                    arguments['pgx_result'] = pgx_result
                    arguments['cost'] = pgx_result.get('cost', 0)
                    arguments['used_llm'] = 'pgx_pipeline'
                    if 'tokenizer' in arguments:
                        arguments.pop('tokenizer')
                    return arguments
                
                # Setup PGX_ROOT path - must be set via environment variable
                # PGX_ROOT should point to ToolOrchestra directory
                # Import PGx tool runner.
                # Prefer the packaged ToolOrchestra within the current working dir (Ray working_dir_files),
                # and only fall back to external PGX_ROOT if needed.
                # Under Ray job submit, `working_dir` is the training folder. Import directly from there.
                from router_reward.pgx_env import call_pgx_tool
                
                print(f"[Tool Orchestration] ========== PGx Tool Execution (func_call category) ==========")
                print(f"[Tool Orchestration] Tool: {tool_name}, Mode: {mode}, Query ID: {query_id}")
                print(f"[Tool Orchestration] Question: {question[:200]}...")
                print(f"[Tool Orchestration] Output Dir: {cur_output_dir}")
                
                pgx_arguments = {
                    'tool': 'pgx_query',  # Keep 'pgx_query' for compatibility with pgx_env
                    'mode': mode,
                    'question': question,
                    'id': str(query_id),
                    'answer': ground_truth,
                    'output_dir': cur_output_dir,
                }
                
                exec_start_time = time.time()
                pgx_result = call_pgx_tool(pgx_arguments)
                exec_end_time = time.time()
                latency_sec = pgx_result.get('latency_seconds')
                if latency_sec is None:
                    latency_sec = exec_end_time - exec_start_time
                arguments['latency'] = latency_sec
                
                print(f"[Tool Orchestration] ✓ PGx tool execution completed (func_call category) in {latency_sec:.2f}s (pipeline module1→module4)")
                print(f"[Tool Orchestration] Tool: {tool_name}, Cost: {pgx_result['cost']}")
                
                arguments['tool'] = 'pgx_query'
                arguments['response'] = pgx_result['response']
                arguments['pgx_result'] = pgx_result
                arguments['cost'] = pgx_result['cost']
                arguments['used_llm'] = 'pgx_pipeline'

                # Save per-question tool result for reuse
                if cache_path:
                    payload = {
                        "question_id": str(question_id),
                        "question": question,
                        "question_type": _extract_question_type_from_arguments(arguments, question_id=question_id),
                        "tool_name": str(tool_name),
                        "mode": str(mode),
                        "result": pgx_result,
                    }
                    _write_json_atomic(cache_path, payload)
                    print(f"[Tool Orchestration] ✓ Saved cached PGx result to {cache_path}")
                
                if 'tokenizer' in arguments:
                    arguments.pop('tokenizer')
                return arguments
        
        raise ValueError("No valid PGx tool call executed in func_call category")
        
    else:
        # Unknown category - not supported for PGx training
        raise ValueError(f"Unknown category '{arguments['category']}' - only 'qa' and 'func_call' are supported for PGx training.")


def call_tool_all(all_arguments):
    all_return_arguments = []
    call_id = all_arguments['id']
    all_call_args = all_arguments['all_call_arguments']
    print(f"[Tool Orchestration] call_tool_all called for id={call_id}, {len(all_call_args)} argument(s)")
    for idx, one_arguments in enumerate(all_call_args):
        category = one_arguments['category']
        tool_field = one_arguments['tool']
        # For func_call category, tool is a list; for qa category, tool is a string
        if category == 'func_call' and isinstance(tool_field, list):
            tool_display = f"list({len(tool_field)} tools)" if tool_field else "empty list"
        else:
            tool_display = tool_field
        print(f"[Tool Orchestration] call_tool_all: Executing tool call {idx+1}/{len(all_call_args)} (id={call_id}, tool={tool_display}, category={category})")
        result = call_tool(one_arguments)
        all_return_arguments.append(result)
        print(f"[Tool Orchestration] call_tool_all: Tool call {idx+1} completed (id={call_id})")
    print(f"[Tool Orchestration] call_tool_all completed for id={call_id}, returning {len(all_return_arguments)} result(s)")
    return {
        'id': call_id,
        'all_tool_call_results': all_return_arguments
    }

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple, Any, Callable

# task_list is an iterable of (func, arg) pairs
async def run_all(
    task_list: Iterable[Tuple[Callable[[Any], Any], Any]],
    progress: bool = False,
    return_exceptions: bool = False,
):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        # wrap each task so it obeys the semaphore
        async def run_one(idx: int, func: Callable, arg: Any):
            async with sem:
                if asyncio.iscoroutinefunction(func):
                    res = await func(arg)
                else:
                    res = await loop.run_in_executor(executor, func, arg)
                return idx, res, None

        task_list = list(task_list)
        tasks = [asyncio.create_task(run_one(i, f, a))
                 for i, (f, a) in enumerate(task_list)]

        results = [None] * len(tasks)

        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks))
        else:
            pbar = None

        # update progress as tasks complete
        for fut in asyncio.as_completed(tasks):
            idx, res, err = await fut
            if err is None:
                results[idx] = res
            else:
                if return_exceptions:
                    results[idx] = err
                else:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise err
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()

        return results

@dataclass
class GenerationConfig:
    max_turns: int
    max_prompt_length: int
    max_response_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str = None
    topk: int = 3
    min_reward_std_threshold: float = 0.1
    # -1 = off; N > 0 = diversity hint active for first N global steps, then off
    rollout_diversity_hint_until: int = -1
    rollout_n_agent: int = 1

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        train_tool_config_path=None,
        test_tool_config_path=None,
        rm_wg=None,  # Reward model worker group (optional)
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.rm_wg = rm_wg  # Store reward model worker group (not used, kept for compatibility)
        # Always use AnswerRewardEvaluator (Grok/XAI backend) for RL reward
        print(f"[Reward Model] Using AnswerRewardEvaluator (from utils.answer_reward_grok, Grok/XAI) for RL reward computation")
        print(f"[Reward Model] Will compute normalized scores (0-1 range) to replace accuracy values")

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length
        ))

        # Active PGx tool set (single source of truth = the selected tool-config JSON).
        # hint / fallback / choke-point / reward-guard all read this, so dropping or re-adding
        # a tool is a TOOL_CONFIG change (+ data), not a code change.
        self.train_tool_config_path = train_tool_config_path
        self.active_tools = _load_active_tools(train_tool_config_path)
        self.fallback_tool = "tool_b" if "tool_b" in self.active_tools else self.active_tools[0]
        print(f"[Tool Orchestration] Active PGx tools = {self.active_tools} (from {train_tool_config_path!r}); fallback = {self.fallback_tool}")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _compute_feature_min_max(
        self,
        example_idx: Optional[str],
        all_turn_index: list,
        all_turn_repeat_ids: list,
        example_costs: dict,
        example_latency: dict,
        rm_scores_by_rollout_id: dict,
        example_correctness: dict,
        use_rm_for_accuracy: bool
    ) -> dict:
        """
        Compute min and max values for accuracy, latency, and cost.
        
        If example_idx is None, pools over all (index, repeat_id) rows in this batch (per_batch scope).
        Otherwise pools only rows whose index equals example_idx (per_question scope).
        
        Args:
            example_idx: Example index to restrict pooling, or None for whole batch
            all_turn_index: List of all example indices
            all_turn_repeat_ids: List of all repeat IDs
            example_costs: Dictionary mapping rollout_id to cost values (negative)
            example_latency: Dictionary mapping rollout_id to latency values (negative)
            rm_scores_by_rollout_id: Dictionary mapping rollout_id to RM scores
            example_correctness: Dictionary mapping rollout_id to correctness (0/1)
            use_rm_for_accuracy: Whether to use RM scores for accuracy
            
        Returns:
            Dictionary with 'min' and 'max' keys for each feature
        """
        # Collect values for the requested pooling scope
        accuracy_values = []
        cost_values = []
        latency_values = []
        
        for idx, repeat_id in zip(all_turn_index, all_turn_repeat_ids):
            if example_idx is not None and idx != example_idx:
                continue
            rollout_id = f"{idx}_____{repeat_id}"
            
            # Collect accuracy value
            if use_rm_for_accuracy and rollout_id in rm_scores_by_rollout_id:
                accuracy_values.append(rm_scores_by_rollout_id[rollout_id])
            elif rollout_id in example_correctness:
                accuracy_values.append(float(example_correctness[rollout_id]))
            
            # Collect cost value
            if rollout_id in example_costs:
                cost_values.append(example_costs[rollout_id])
            
            # Collect latency value
            if rollout_id in example_latency:
                latency_values.append(example_latency[rollout_id])
        
        # Compute min/max for each feature
        result = {}
        
        # Accuracy
        if accuracy_values:
            result['accuracy'] = {'min': min(accuracy_values), 'max': max(accuracy_values)}
        else:
            result['accuracy'] = {'min': 0.0, 'max': 1.0}  # Default range
        
        # Cost
        if cost_values:
            result['cost'] = {'min': min(cost_values), 'max': max(cost_values)}
        else:
            result['cost'] = {'min': 0.0, 'max': 0.0}  # Default if no values
        
        # Latency
        if latency_values:
            result['latency'] = {'min': min(latency_values), 'max': max(latency_values)}
        else:
            result['latency'] = {'min': 0.0, 'max': 0.0}  # Default if no values
        
        return result

    def _process_weighted_scores(
        self,
        all_turn_index: list,
        all_turn_repeat_ids: list,
        all_turn_responses: list,
        all_turn_problems: list,
        all_turn_answers: list,
        precomputed_eval_results_by_rollout_id: dict,
        response_json_by_rollout_id: dict,
        global_steps: int,
        my_output_dir: str,
            example_correctness: dict,
        all_turn_tools: Optional[list] = None,
        all_turn_levels: Optional[list] = None,
    ) -> tuple[dict, bool, Optional[str], dict]:
        """
        Process weighted scores from AnswerRewardEvaluator.
        
        Args:
            all_turn_index: List of example indices
            all_turn_repeat_ids: List of repeat IDs
            all_turn_responses: List of response token sequences
            all_turn_problems: List of problems/questions
            all_turn_answers: List of ground truth answers
            precomputed_eval_results_by_rollout_id: Dictionary of pre-computed evaluation results
            response_json_by_rollout_id: Dictionary mapping rollout_id to response JSON
            global_steps: Current global step number
            my_output_dir: Output directory for saving results
            example_correctness: Dictionary mapping rollout_id to correctness (0/1)
            all_turn_tools: Optional list of tool summaries per rollout (for pipeline_name in summary)
            all_turn_levels: Optional list of level per rollout (key-derived: hard / filtered / simple)
            
        Returns:
            Tuple of (rm_scores_by_rollout_id, use_rm_for_accuracy, reward_output_dir, detailed_eval_results)
        """
        rm_scores_by_rollout_id = {}  # Store reward model scores to replace accuracy values
        use_rm_for_accuracy = False
        
        # Use AnswerRewardEvaluator (Grok/XAI backend) instead of neural network reward model
        use_answer_reward_evaluator = True  # Always use AnswerRewardEvaluator for RL reward
        
        if use_answer_reward_evaluator:
            print(f"\n{'='*80}")
            print(f"[Reward Model] ========== Starting AnswerRewardEvaluator Computation (Grok/XAI) ==========")
            print(f"[Reward Model] Step: {global_steps}")
            print(f"[Reward Model] Using AnswerRewardEvaluator from utils.answer_reward_grok (Grok/XAI)")
            print(f"[Reward Model] Processing {len(all_turn_index)} turns to compute normalized reward scores (0-1 range)")
            from utils.answer_reward_grok import AnswerRewardEvaluator
            from config import (
                REWARD_EVALUATION_MODE,
                REWARD_MULTI_QUESTION_COUNT,
                REWARD_EVAL_MODE,
            )
            if REWARD_EVALUATION_MODE not in ("individual", "batch_per_question"):
                REWARD_EVALUATION_MODE = "individual"
            if REWARD_MULTI_QUESTION_COUNT < 1:
                REWARD_MULTI_QUESTION_COUNT = 1
            print(
                f"[Reward Model] Evaluation mode: {REWARD_EVALUATION_MODE}, "
                f"multi_question_count={REWARD_MULTI_QUESTION_COUNT}"
            )
            
            # Initialize AnswerRewardEvaluator
            print(f"[Reward Model] Initializing AnswerRewardEvaluator...")
            reward_evaluator = AnswerRewardEvaluator()
            print(f"[Reward Model] ✓ AnswerRewardEvaluator initialized")
            
            # Decode responses from token sequences to text
            print(f"[Reward Model] Decoding {len(all_turn_responses)} response sequences to text...")
            import time
            rm_start_time = time.time()
            
            # Decode response tokens to text
            decoded_responses = self.tokenizer.batch_decode(
                all_turn_responses, 
                skip_special_tokens=True
            )
            print(f"[Reward Model] ✓ Decoded {len(decoded_responses)} responses")
            
            # Ensure we have questions and answers
            assert len(all_turn_problems) == len(all_turn_index), f"Mismatch: {len(all_turn_problems)} problems vs {len(all_turn_index)} indices"
            assert len(all_turn_answers) == len(all_turn_index), f"Mismatch: {len(all_turn_answers)} answers vs {len(all_turn_index)} indices"
            assert len(decoded_responses) == len(all_turn_index), f"Mismatch: {len(decoded_responses)} responses vs {len(all_turn_index)} indices"

            # Map rollout_id -> decoded actor text, used by the facet-reasoning
            # bonus applied inside every weighted_score store path below.
            response_text_by_rollout_id: dict = {
                f"{ex_idx}_____{rep_id}": resp_text
                for ex_idx, rep_id, resp_text in zip(
                    all_turn_index, all_turn_repeat_ids, decoded_responses
                )
            }
            # --- Reward eval cache: (qid, tool) keyed; miss = fatal ---
            # Cache produced by training/build_qid_tool_eval_cache.py.
            # Every rollout's chosen tool must correspond to a precomputed
            # (qid, tool) entry — otherwise training halts.
            import hashlib as _hashlib  # kept for legacy writeback (unreachable when cache is fully populated)
            # REWARD_CACHE_FILE selects a different metric's cache (same schema, F-beta of choice
            # stored under the "F15" key). Unset -> the historical F1.5 cache, behaviour unchanged.
            _eval_cache_name = os.environ.get("REWARD_CACHE_FILE", "eval_cache_qid_tool_claims.json")
            _eval_cache_path = Path(os.environ.get("BASE_DIR", "")) / "output" / "eval_records" / _eval_cache_name
            if not _eval_cache_path.exists():
                raise FileNotFoundError(
                    f"[Reward Cache] required (qid, tool) claims cache not found at {_eval_cache_path}. "
                    f"Run training/build_qid_tool_claims_cache.py first."
                )
            with open(_eval_cache_path) as _f:
                _reward_cache = json.load(_f)
            # REWARD_ROUND_SAMPLING=1 turns on per-rollout sampling over the repeat rounds
            # stored in a multi-round cache. Default off -> unchanged behaviour.
            # REWARD_ROUND_SEED only shifts which round each rollout draws (reproducible).
            _round_sampling = os.environ.get("REWARD_ROUND_SAMPLING", "0") not in ("0", "", "false", "False")
            _round_seed = os.environ.get("REWARD_ROUND_SEED", "0")
            # REWARD_METRIC picks which F-beta becomes the accuracy signal, from a cache that
            # stores several. "F15" (default) keeps historical behaviour; "F1" uses the F1 field.
            # This replaces needing a separate *_F1.json file when the cache already has both.
            _reward_metric = os.environ.get("REWARD_METRIC", "F15").strip()
            _ALLOWED_METRICS = ("F15", "F1", "F0.5", "P", "R")
            if _reward_metric not in _ALLOWED_METRICS:
                raise RuntimeError(
                    f"[Reward Cache] REWARD_METRIC={_reward_metric!r} not one of {_ALLOWED_METRICS}."
                )
            _probe = next((_e for _e in _reward_cache.values() if isinstance(_e, dict)), None)
            if _probe is not None and _reward_metric not in _probe:
                raise RuntimeError(
                    f"[Reward Cache] REWARD_METRIC={_reward_metric!r} but entries in "
                    f"{_eval_cache_path} have no such field (available: {sorted(_probe)})."
                )
            print(f"[Reward Cache] metric={_reward_metric} (accuracy signal)")
            _n_multi = sum(1 for _e in _reward_cache.values()
                           if isinstance(_e, dict) and len(_e.get("rounds") or []) > 1)
            print(f"[Reward Cache] Loaded {len(_reward_cache)} (qid, tool) entries from {_eval_cache_path}"
                  f"; round_sampling={_round_sampling} seed={_round_seed}"
                  f"; {_n_multi} entries have >1 round")
            if _round_sampling and _n_multi == 0:
                raise RuntimeError(
                    "[Reward Cache] REWARD_ROUND_SAMPLING=1 but no entry in "
                    f"{_eval_cache_path} has more than one round. Point REWARD_CACHE_FILE at "
                    "eval_cache_qid_tool_claims_multiround.json (built by build_multiround_cache.py)."
                )

            # Build rollout_id -> chosen tool_name (tool_a / tool_b / tool_c) for cache key construction.
            # Mirrors the late construction at the summary writer below; we need it earlier.
            tool_name_by_rollout_id = {}
            if all_turn_tools is not None and len(all_turn_tools) == len(all_turn_index):
                for _i in range(len(all_turn_index)):
                    _rid = f"{all_turn_index[_i]}_____{all_turn_repeat_ids[_i]}"
                    _tlist = all_turn_tools[_i] if _i < len(all_turn_tools) else []
                    _tname = None
                    if _tlist:
                        _first = _tlist[0]
                        if isinstance(_first, (list, tuple)) and len(_first) > 0:
                            _tname = _first[0]
                        elif isinstance(_first, str):
                            _tname = _first
                    tool_name_by_rollout_id[_rid] = _tname if _tname in VALID_PGX_TOOL_NAMES else None

            print(f"[Reward Model] Computing normalized reward scores using AnswerRewardEvaluator...")
            if REWARD_EVALUATION_MODE == "batch_per_question":
                print(f"[Reward Model] Batch-per-question mode: one LLM call per question (all rollouts for that question in one prompt)")
            else:
                print(f"[Reward Model] This may take some time as it calls LLM API for each response...")

            # Store detailed evaluation results for saving to file
            detailed_eval_results = {}  # rollout_id -> eval_result dict
            batch_eval_reasoning_records = []  # (question_id, rollout_ids, batch_meta, eval_results)

            def _make_cache_key(question, response_for_eval):
                """Legacy (question, response) hash key — retained only for the dead-code
                writeback path below. Live lookup uses _lookup_qid_tool_cached."""
                if isinstance(response_for_eval, dict):
                    resp_str = json.dumps(response_for_eval, sort_keys=True, ensure_ascii=False)
                else:
                    resp_str = str(response_for_eval)
                return _hashlib.md5((question + resp_str).encode()).hexdigest()

            def _lookup_qid_tool_cached(example_idx, rollout_id):
                """Look up the precomputed reward for (qid, tool). Raise on any miss
                so training halts immediately rather than silently calling the LLM."""
                tool_name = tool_name_by_rollout_id.get(rollout_id)
                if tool_name not in self.active_tools:
                    # Not in the active tool set (None / any invalid name): score 0, do NOT crash,
                    # so one stray rollout can't halt training. A genuine miss of an ACTIVE tool still
                    # raises below.
                    print(f"[Reward Cache][GUARD] rollout {rollout_id}: tool={tool_name!r} not in active {self.active_tools} -> score 0")
                    return {"weighted_score": 0.0, "F15": 0.0, "Accuracy": 0.0, "Relevance": 0.0,
                            "Completeness": 0.0, "Clarity": 0.0, "cost": 0.0, "latency_seconds": 0.0,
                            "f15_source": "invalid_tool_zero"}
                key = f"{example_idx}__{tool_name}"
                if key not in _reward_cache:
                    raise RuntimeError(
                        f"[Reward Cache] miss for key={key!r} (rollout {rollout_id}). "
                        f"Every rollout must hit a precomputed (qid, tool) entry built by "
                        f"build_qid_tool_claims_cache.py. Re-run that script if entries are missing."
                    )
                entry = _reward_cache[key]
                # Multi-round cache (eval_cache_qid_tool_claims_multiround.json): the entry
                # carries a "rounds" list of same-setting repeat runs. With sampling ON we
                # draw ONE round per (step, qid, tool, rollout) so the reward is a sample of
                # the tool's own distribution instead of a single fixed draw. The top-level
                # fields are the round averages, so sampling OFF (the default) behaves exactly
                # like a single-round cache and leaves existing runs byte-identical.
                if _round_sampling and entry.get("rounds"):
                    _rs = [r for r in entry["rounds"] if r.get("F15") is not None]
                    if _rs:
                        _h = _hashlib.md5(
                            f"{_round_seed}|{global_steps}|{example_idx}|{tool_name}|{rollout_id}".encode()
                        ).hexdigest()
                        _pick = _rs[int(_h, 16) % len(_rs)]
                        entry = {**entry, **{k: v for k, v in _pick.items() if k != "round"},
                                 "sampled_round": _pick["round"], "n_rounds_available": len(_rs)}
                # Expose the chosen F-beta as "weighted_score" so every downstream accuracy
                # read uses it unchanged. All of these live in [0,1], so the downstream
                # [0,1] clamp is a no-op. REWARD_METRIC selects which one (default F15 =
                # historical behaviour); "metric" records the choice in the saved records.
                if _reward_metric in entry and entry[_reward_metric] is not None:
                    entry = {**entry, "weighted_score": entry[_reward_metric],
                             "metric": _reward_metric}
                elif "weighted_score" not in entry and "F15" in entry:
                    entry = {**entry, "weighted_score": entry["F15"], "metric": "F15"}
                return entry

            _cache_hits = 0
            _cache_misses = 0

            if REWARD_EVALUATION_MODE == "batch_per_question":
                # Group by example_idx (one question per example_idx; multiple rollouts = multiple repeat_ids)
                from collections import defaultdict
                groups = defaultdict(list)
                for idx, (example_idx, example_repeat_id, response_text, question, ground_truth) in enumerate(zip(
                    all_turn_index, all_turn_repeat_ids, decoded_responses, all_turn_problems, all_turn_answers
                )):
                    rollout_id = f"{example_idx}_____{example_repeat_id}"
                    cur_level = all_turn_levels[idx] if all_turn_levels is not None and idx < len(all_turn_levels) else None
                    groups[example_idx].append((rollout_id, response_text, question, ground_truth, cur_level))
                num_processed = 0
                def _build_answers_for_eval(_rollout_ids, _response_texts):
                    _answers_for_eval = []
                    for _rollout_id, _response_text in zip(_rollout_ids, _response_texts):
                        if _rollout_id in response_json_by_rollout_id:
                            _rj = response_json_by_rollout_id[_rollout_id]
                            if isinstance(_rj, dict) and (_rj.get("title") or _rj.get("details")):
                                _answers_for_eval.append(_rj)
                                continue
                        _answers_for_eval.append(_response_text)
                    return _answers_for_eval

                if REWARD_EVALUATION_MODE == "batch_per_question" and REWARD_MULTI_QUESTION_COUNT > 1:
                    group_keys = list(groups.keys())
                    group_payload = {}
                    for example_idx, group_list in groups.items():
                        rollout_ids = [x[0] for x in group_list]
                        response_texts = [x[1] for x in group_list]
                        question = group_list[0][2]
                        ground_truth = group_list[0][3]
                        ability = group_list[0][4]
                        group_payload[example_idx] = {
                            "rollout_ids": rollout_ids,
                            "response_texts": response_texts,
                            "question": question,
                            "ground_truth": ground_truth,
                            "ability": ability,
                            "answers_for_eval": _build_answers_for_eval(rollout_ids, response_texts),
                        }

                    def _sample_context_examples(target_example_idx, target_ability):
                        n_total = max(1, int(REWARD_MULTI_QUESTION_COUNT))
                        n_other = max(0, n_total - 1)
                        if n_other == 0:
                            return []
                        candidates = [k for k in group_keys if k != target_example_idx]
                        if not candidates:
                            return []
                        different_ability = []
                        same_or_unknown = []
                        t_ability = str(target_ability) if target_ability is not None else ""
                        for k in candidates:
                            k_ability = group_payload[k].get("ability")
                            k_ability_s = str(k_ability) if k_ability is not None else ""
                            if t_ability and k_ability_s and (k_ability_s != t_ability):
                                different_ability.append(k)
                            else:
                                same_or_unknown.append(k)
                        random.shuffle(different_ability)
                        random.shuffle(same_or_unknown)
                        selected = different_ability[:n_other]
                        if len(selected) < n_other:
                            selected.extend(same_or_unknown[: (n_other - len(selected))])
                        return selected

                    for example_idx in group_keys:
                        target_payload = group_payload[example_idx]
                        rollout_ids = target_payload["rollout_ids"]
                        answers_for_eval = target_payload["answers_for_eval"]
                        question = target_payload["question"]

                        # (qid, tool) cache lookup — every rollout must hit; misses raise.
                        for rollout_id in rollout_ids:
                            eval_result = _lookup_qid_tool_cached(example_idx, rollout_id)
                            weighted_score = eval_result["weighted_score"]
                            if weighted_score is None:
                                weighted_score = 0.2
                            if not (0.0 <= weighted_score <= 1.0):
                                weighted_score = max(0.0, min(1.0, weighted_score))
                            detailed_eval_results[rollout_id] = eval_result
                            rm_scores_by_rollout_id[rollout_id] = weighted_score
                            num_processed += 1
                            _cache_hits += 1
                        print(f"[Reward Cache] ✓ Question {example_idx}: all {len(rollout_ids)} rollouts from (qid, tool) cache")
                else:
                    for example_idx, group_list in groups.items():
                        rollout_ids = [x[0] for x in group_list]

                        # (qid, tool) cache lookup — every rollout must hit; misses raise.
                        for rollout_id in rollout_ids:
                            eval_result = _lookup_qid_tool_cached(example_idx, rollout_id)
                            weighted_score = eval_result["weighted_score"]
                            if weighted_score is None:
                                weighted_score = 0.2
                            if not (0.0 <= weighted_score <= 1.0):
                                weighted_score = max(0.0, min(1.0, weighted_score))
                            detailed_eval_results[rollout_id] = eval_result
                            rm_scores_by_rollout_id[rollout_id] = weighted_score
                            num_processed += 1
                            _cache_hits += 1
                        print(f"[Reward Cache] ✓ Question {example_idx}: all {len(rollout_ids)} rollouts from (qid, tool) cache")
                print(f"[Reward Model] Batch mode: evaluated {len(groups)} questions -> {num_processed} rollouts")
            else:
                # Individual mode: (qid, tool) cache lookup per rollout — misses raise.
                num_processed = 0
                for idx, (example_idx, example_repeat_id, response_text, question, ground_truth) in enumerate(zip(
                    all_turn_index, all_turn_repeat_ids, decoded_responses, all_turn_problems, all_turn_answers
                )):
                    rollout_id = f"{example_idx}_____{example_repeat_id}"
                    eval_result = _lookup_qid_tool_cached(example_idx, rollout_id)
                    _cache_hits += 1

                    detailed_eval_results[rollout_id] = {
                        "weighted_score": eval_result.get('weighted_score'),
                        "Accuracy": eval_result.get('Accuracy'),
                        "Relevance": eval_result.get('Relevance'),
                        "Completeness": eval_result.get('Completeness'),
                        "Clarity": eval_result.get('Clarity'),
                        "raw_scores": eval_result.get('raw_scores', {}),
                        "evaluation_time_seconds": eval_result.get('evaluation_time_seconds'),
                    }
                    weighted_score = eval_result.get('weighted_score', None)
                    if weighted_score is None:
                        raise ValueError(f"AnswerRewardEvaluator returned None for weighted_score for rollout {rollout_id}")
                    if not (0.0 <= weighted_score <= 1.0):
                        print(f"[Reward Model] ⚠️  Warning: Score {weighted_score:.4f} for rollout {rollout_id} is not in [0,1] range, clamping...")
                        weighted_score = max(0.0, min(1.0, weighted_score))
                    rm_scores_by_rollout_id[rollout_id] = weighted_score
                    num_processed += 1
                    
                    if num_processed <= 3:
                        accuracy_score = eval_result.get('Accuracy', 'N/A')
                        relevance_score = eval_result.get('Relevance', 'N/A')
                        completeness_score = eval_result.get('Completeness', 'N/A')
                        clarity_score = eval_result.get('Clarity', 'N/A')
                        if rollout_id in response_json_by_rollout_id:
                            rj = response_json_by_rollout_id[rollout_id]
                            if isinstance(rj, dict) and (rj.get("title") or rj.get("details")):
                                answer_text_for_length = (rj.get('title', '') + ' ' + rj.get('details', '')).strip()
                            else:
                                answer_text_for_length = response_text
                        else:
                            answer_text_for_length = response_text
                        print(f"[Reward Model]   Example {num_processed}: rollout={rollout_id}")
                        print(f"[Reward Model]     Normalized weighted_score={weighted_score:.4f} (0-1 range)")
                        print(f"[Reward Model]     Raw scores: Accuracy={accuracy_score}, Relevance={relevance_score}, "
                              f"Completeness={completeness_score}, Clarity={clarity_score}")
                        print(f"[Reward Model]     Question length: {len(question)}, Answer length: {len(answer_text_for_length)}, Ground truth length: {len(ground_truth)}")
            
            rm_compute_time = time.time() - rm_start_time
            print(f"[Reward Model] ✓ Completed evaluation in {rm_compute_time:.2f}s")
            if _cache_hits or _cache_misses:
                print(f"[Reward Cache] Summary: {_cache_hits} cache hits, {_cache_misses} API calls (saved {_cache_hits}/{_cache_hits + _cache_misses} Grok calls)")
            print(f"[Reward Model] ✓ Successfully processed {num_processed}/{len(all_turn_index)} responses")
            if num_processed != len(all_turn_index):
                raise ValueError(f"Expected to process {len(all_turn_index)} responses but only processed {num_processed}")
            
            # Log statistics
            if rm_scores_by_rollout_id:
                scores_list = list(rm_scores_by_rollout_id.values())
                print(f"[Reward Model] Score statistics: min={min(scores_list):.4f}, max={max(scores_list):.4f}, "
                      f"mean={sum(scores_list)/len(scores_list):.4f}, std={(sum((s - sum(scores_list)/len(scores_list))**2 for s in scores_list) / len(scores_list))**0.5:.4f}")
                print(f"[Reward Model] ✓ All scores are normalized in [0, 1] range")
            
            # Map to unique rollout_ids (handle multiple turns)
            unique_rollout_ids = set(rm_scores_by_rollout_id.keys())
            print(f"[Reward Model] ✓ Mapped {len(rm_scores_by_rollout_id)} RM scores to {len(unique_rollout_ids)} unique rollout_ids")
            
            use_rm_for_accuracy = True
            print(f"[Reward Model] ✓ Successfully computed {len(rm_scores_by_rollout_id)} normalized reward scores")
            print(f"[Reward Model] ✓ Normalized scores (0-1 range) will replace accuracy values in preference vector calculation")
            
            # Save reward model results to file
            reward_output_dir = os.path.join(my_output_dir, f"global_step_{global_steps}", "reward_model_results")
            if not os.path.isdir(reward_output_dir):
                os.makedirs(reward_output_dir, exist_ok=True)

            # Batch eval: per-rollout files with shared prompts + full batch reasoning (same text for all rollouts in that question)
            if batch_eval_reasoning_records:
                eval_reasoning_dir = os.path.join(reward_output_dir, "eval_reasoning")
                os.makedirs(eval_reasoning_dir, exist_ok=True)
                for qid, rids, meta, evs in batch_eval_reasoning_records:
                    for j, (rollout_id, ev) in enumerate(zip(rids, evs), start=1):
                        safe_name = f"q_{qid.replace('/', '_')}__{rollout_id.replace('/', '_')}.json"
                        record = {
                            "question_id": qid,
                            "rollout_id": rollout_id,
                            "rollout_index_in_batch": j,
                            "global_step": global_steps,
                            "system_prompt": meta.get("system_prompt", ""),
                            "user_prompt": meta.get("user_prompt", ""),
                            "batch_reasoning_and_scores_raw": meta.get("batch_raw_llm_response", ""),
                            "note": (
                                "batch_reasoning_and_scores_raw applies to the whole batch call; "
                                "scores below are for this rollout only."
                            ),
                            "rollout_scores": {
                                "Accuracy": ev.get("Accuracy"),
                                "Relevance": ev.get("Relevance"),
                                "Completeness": ev.get("Completeness"),
                                "Clarity": ev.get("Clarity"),
                                "weighted_score": ev.get("weighted_score"),
                            },
                        }
                        if meta.get("parse_error"):
                            record["parse_error"] = meta["parse_error"]
                        if meta.get("exception"):
                            record["batch_exception"] = meta["exception"]
                        er_path = os.path.join(eval_reasoning_dir, safe_name)
                        with open(er_path, "w", encoding="utf-8") as erf:
                            json.dump(record, erf, indent=2, ensure_ascii=False)
                print(
                    f"[Reward Model] ✓ Saved batch eval reasoning under {eval_reasoning_dir} "
                    f"({sum(len(x[1]) for x in batch_eval_reasoning_records)} files)"
                )

            # Save detailed results for each rollout
            reward_results = []
            for idx, (example_idx, example_repeat_id, response_text, question, ground_truth) in enumerate(zip(
                all_turn_index, all_turn_repeat_ids, decoded_responses, all_turn_problems, all_turn_answers
            )):
                rollout_id = f"{example_idx}_____{example_repeat_id}"
                if rollout_id in rm_scores_by_rollout_id and rollout_id in detailed_eval_results:
                    eval_detail = detailed_eval_results[rollout_id]
                    # Prefer response_json (short version) over response_text (long version)
                    response_to_save = response_text
                    if rollout_id in response_json_by_rollout_id:
                        response_json = response_json_by_rollout_id[rollout_id]
                        if isinstance(response_json, dict) and (response_json.get("title") or response_json.get("details")):
                            response_to_save = response_json
                    
                    result_entry = {
                        "rollout_id": rollout_id,
                        "example_idx": example_idx,
                        "repeat_id": example_repeat_id,
                        "weighted_score": rm_scores_by_rollout_id[rollout_id],
                        "Accuracy": eval_detail.get("Accuracy"),
                        "Relevance": eval_detail.get("Relevance"),
                        "Completeness": eval_detail.get("Completeness"),
                        "Clarity": eval_detail.get("Clarity"),
                        "raw_scores": eval_detail.get("raw_scores", {}),
                        "evaluation_time_seconds": eval_detail.get("evaluation_time_seconds"),
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response_to_save,  # Use response_json if available, otherwise response_text
                    }
                    reward_results.append(result_entry)
            
            # Save summary statistics
            if rm_scores_by_rollout_id:
                # Build pipeline_name per rollout from all_turn_tools (tool_a -> skip_retrieve, etc.)
                # Also keep raw tool name (tool_a/b/d) — used by the eval-cache writeback below.
                pipeline_by_rollout_id = {}
                tool_name_by_rollout_id = {}
                if all_turn_tools is not None and len(all_turn_tools) == len(all_turn_index):
                    for i in range(len(all_turn_index)):
                        rollout_id = f"{all_turn_index[i]}_____{all_turn_repeat_ids[i]}"
                        tool_list = all_turn_tools[i] if i < len(all_turn_tools) else []
                        tool_name = None
                        if tool_list:
                            first = tool_list[0]
                            if isinstance(first, (list, tuple)) and len(first) > 0:
                                tool_name = first[0]
                            elif isinstance(first, str):
                                tool_name = first
                        valid_tool = tool_name if tool_name and tool_name in VALID_PGX_TOOL_NAMES else None
                        pipeline_by_rollout_id[rollout_id] = get_pgx_mode_from_tool(valid_tool) if valid_tool else None
                        tool_name_by_rollout_id[rollout_id] = valid_tool
                # Level per rollout (key-derived: hard / filtered / simple)
                level_by_rollout_id = {}
                if all_turn_levels is not None and len(all_turn_levels) == len(all_turn_index):
                    for i in range(len(all_turn_index)):
                        rid = f"{all_turn_index[i]}_____{all_turn_repeat_ids[i]}"
                        level_by_rollout_id[rid] = all_turn_levels[i] if i < len(all_turn_levels) else None
                # Per-rollout: 4 normalized eval scores (before weighted), cost/efficiency normalized (filled later), pipeline_name, level
                rollout_details = {}
                for rid in rm_scores_by_rollout_id:
                    eval_detail = detailed_eval_results.get(rid, {})
                    rollout_details[rid] = {
                        "weighted_score": float(rm_scores_by_rollout_id[rid]),
                        "Accuracy": eval_detail.get("Accuracy"),
                        "Relevance": eval_detail.get("Relevance"),
                        "Completeness": eval_detail.get("Completeness"),
                        "Clarity": eval_detail.get("Clarity"),
                        "cost_normalized": None,
                        "efficiency_normalized": None,
                        "pipeline_name": pipeline_by_rollout_id.get(rid),
                        "level": level_by_rollout_id.get(rid),
                    }
                summary = {
                    "global_step": global_steps,
                    "total_rollouts": len(rm_scores_by_rollout_id),
                    "scores_note": "scores = accuracy only (RM weighted_score 0-1). full_reward_by_rollout_id (added after reward loop) = accuracy + latency + cost, clipped [-1,1].",
                    "scores": {rollout_id: float(score) for rollout_id, score in rm_scores_by_rollout_id.items()},
                    "statistics": {
                        "min": float(min(rm_scores_by_rollout_id.values())),
                        "max": float(max(rm_scores_by_rollout_id.values())),
                        "mean": float(sum(rm_scores_by_rollout_id.values()) / len(rm_scores_by_rollout_id)),
                        "std": float((sum((s - sum(rm_scores_by_rollout_id.values())/len(rm_scores_by_rollout_id))**2 for s in rm_scores_by_rollout_id.values()) / len(rm_scores_by_rollout_id))**0.5),
                    },
                    "rollout_details": rollout_details,
                }
                
                summary_file = os.path.join(reward_output_dir, "summary.json")
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"[Reward Model] ✓ Saved summary to {summary_file}")
                
                # Save detailed results
                detailed_file = os.path.join(reward_output_dir, "detailed_results.json")
                with open(detailed_file, 'w') as f:
                    json.dump(reward_results, f, indent=2)
                print(f"[Reward Model] ✓ Saved detailed results to {detailed_file}")
                
                # Save rm_scores_by_rollout_id separately for easy access
                rm_scores_file = os.path.join(reward_output_dir, "rm_scores_by_rollout_id.json")
                with open(rm_scores_file, 'w') as f:
                    json.dump({rollout_id: float(score) for rollout_id, score in rm_scores_by_rollout_id.items()}, f, indent=2)
                print(f"[Reward Model] ✓ Saved rm_scores_by_rollout_id to {rm_scores_file}")

                # Update centralized eval cache with new Grok results (cache misses only).
                # Always run regardless of REWARD_EVAL_MODE — any time we paid for a Grok call
                # we should capture it so future runs hit the cache.
                # NOTE: cost / latency_seconds / input_tokens / output_tokens are NOT captured here
                # because they live at the tool-execution layer (pgx_env.py) and aren't passed
                # into _process_weighted_scores. Existing cache entries built from detailed_results
                # also lack these for ~half the entries, so this is consistent behavior.
                if _cache_misses > 0:
                    try:
                        _eval_cache_dir = Path(os.environ.get("BASE_DIR", "")) / "output" / "eval_records"
                        _eval_cache_path = _eval_cache_dir / "eval_cache.json"
                        _eval_cache_dir.mkdir(parents=True, exist_ok=True)
                        _existing_cache = {}
                        if _eval_cache_path.exists():
                            with open(_eval_cache_path) as _f:
                                _existing_cache = json.load(_f)
                        _new_entries = 0
                        for _rr in reward_results:
                            _ck = _make_cache_key(_rr["question"], _rr["response"])
                            if _ck not in _existing_cache:
                                _rid = _rr.get("rollout_id")
                                _ex_idx = _rr.get("example_idx")
                                _existing_cache[_ck] = {
                                    "weighted_score": _rr["weighted_score"],
                                    "Accuracy": _rr.get("Accuracy"),
                                    "Relevance": _rr.get("Relevance"),
                                    "Completeness": _rr.get("Completeness"),
                                    "Clarity": _rr.get("Clarity"),
                                    "n_evals": 1,
                                    "question_preview": _rr["question"][:120],
                                    "question_index": int(_ex_idx) if _ex_idx is not None else None,
                                    "level": level_by_rollout_id.get(_rid),
                                    "tool": tool_name_by_rollout_id.get(_rid),
                                }
                                _new_entries += 1
                        with open(_eval_cache_path, "w", encoding="utf-8") as _f:
                            json.dump(_existing_cache, _f, indent=2, ensure_ascii=False)
                        if _new_entries:
                            print(f"[Reward Cache] ✓ Added {_new_entries} new entries to {_eval_cache_path} (total: {len(_existing_cache)})")
                    except Exception as _e:
                        print(f"[Reward Cache] ⚠️  Failed to update eval cache: {_e}")

                return rm_scores_by_rollout_id, use_rm_for_accuracy, reward_output_dir, detailed_eval_results
        else:
            print(f"[Reward Model] AnswerRewardEvaluator not enabled, using original accuracy (0/1) values")
            use_rm_for_accuracy = False
        
        return rm_scores_by_rollout_id, use_rm_for_accuracy, None, {}

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        # [t.shape for t in tensors_with_mask] [torch.Size([2560, 0]), torch.Size([2560, 657]), torch.Size([2560, 500])]
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        # concatenated_with_info.shape torch.Size([2560, 1157])
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        # concatenated.shape torch.Size([2560, 1004]) sorted_indices.shape torch.Size([2560, 1004])
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _initialize_run_llm_loop(
        self,
        gen_batch,
        tokenizer_config: dict,
        global_steps: int
    ) -> dict:
        """
        Initialize variables and directories for run_llm_loop.
        
        Args:
            gen_batch: Generation batch containing input data
            global_steps: Current global training step
            
        Returns:
            Dictionary containing initialized variables
        """
        # Handle both 'problem' (for QA datasets) and 'question' (for PGx datasets)
        if 'problem' in gen_batch.non_tensor_batch:
            problem_key = 'problem'
        elif 'question' in gen_batch.non_tensor_batch:
            problem_key = 'question'
        else:
            # Try to find question in extra_info
            if 'extra_info' in gen_batch.non_tensor_batch:
                # Extract question from extra_info if available
                extra_infos = gen_batch.non_tensor_batch['extra_info']
                questions = []
                for extra_info in extra_infos:
                    if isinstance(extra_info, dict):
                        questions.append(extra_info.get('question', ''))
                    else:
                        questions.append('')
                gen_batch.non_tensor_batch['question'] = np.array(questions, dtype=object)
                problem_key = 'question'
            else:
                raise KeyError(f"Neither 'problem' nor 'question' found in non_tensor_batch. Available keys: {list(gen_batch.non_tensor_batch.keys())}")
        
        loop_batch_size = len(gen_batch.non_tensor_batch[problem_key])
        active_mask = torch.ones(loop_batch_size, dtype=torch.bool)
        tokenizer = tokenizer_config['tokenizer']
        retrieved_documents = [[] for _ in range(loop_batch_size)]
        code_snippets = [[] for _ in range(loop_batch_size)]
        attempts = [[] for _ in range(loop_batch_size)]
        total_costs = [0 for _ in range(loop_batch_size)]
        
        # Initialize lists to store all turn data
        all_turn_input_ids = []
        all_turn_attention_mask = []
        all_turn_position_ids = []
        all_turn_responses = []
        all_turn_index = []
        all_turn_ids = []
        all_turn_turn_ids = []
        all_turn_repeat_ids = []
        all_turn_answers = []
        all_turn_answer_preds = []
        all_turn_problems = []
        all_turn_tools = []
        all_turn_success = []
        all_turn_costs = []
        all_turn_latency = []
        all_turn_formats = []
        all_turn_valid_answers_generated = []
        all_turn_used_llms = []
        all_turn_steps = []
        all_turn_categories = []
        all_pref_vecs = {}
        meta_info = {}
        vllm_model_configs = {}
        example_correct_by_rollout_id = {}
        
        tool_total = 0
        TRAINED_MODEL_TYPE = None
        cur_batch_indices = copy.deepcopy(gen_batch.non_tensor_batch['index'])
        cur_batch_repeat_ids = copy.deepcopy(gen_batch.non_tensor_batch['repeat_id'])
        cur_batch_output_files = []
        tmp_my_output_dir = gen_batch.non_tensor_batch['my_output_dir'][0]
        my_output_dir = gen_batch.non_tensor_batch['my_output_dir'][0]
        
        # Clean up existing directories for this global_step
        if os.path.isdir(os.path.join(tmp_my_output_dir,f"global_step_{global_steps}")):
            shutil.rmtree(os.path.join(tmp_my_output_dir,f"global_step_{global_steps}"))
        tmp_transfer_dir = os.path.join(gen_batch.non_tensor_batch['cur_transfer_dir'][0],str(global_steps))
        if os.path.isdir(tmp_transfer_dir):
            shutil.rmtree(tmp_transfer_dir)
        
        # Track which items have already called a PGx tool across all steps
        # This ensures that within one question-to-answer flow, only one PGx tool can be used
        pgx_tool_used_per_item = [False] * loop_batch_size
        
        return {
            'problem_key': problem_key,
            'loop_batch_size': loop_batch_size,
            'active_mask': active_mask,
            'tokenizer': tokenizer,
            'retrieved_documents': retrieved_documents,
            'code_snippets': code_snippets,
            'attempts': attempts,
            'total_costs': total_costs,
            'all_turn_input_ids': all_turn_input_ids,
            'all_turn_attention_mask': all_turn_attention_mask,
            'all_turn_position_ids': all_turn_position_ids,
            'all_turn_responses': all_turn_responses,
            'all_turn_index': all_turn_index,
            'all_turn_ids': all_turn_ids,
            'all_turn_turn_ids': all_turn_turn_ids,
            'all_turn_repeat_ids': all_turn_repeat_ids,
            'all_turn_answers': all_turn_answers,
            'all_turn_answer_preds': all_turn_answer_preds,
            'all_turn_problems': all_turn_problems,
            'all_turn_tools': all_turn_tools,
            'all_turn_success': all_turn_success,
            'all_turn_costs': all_turn_costs,
            'all_turn_latency': all_turn_latency,
            'all_turn_formats': all_turn_formats,
            'all_turn_valid_answers_generated': all_turn_valid_answers_generated,
            'all_turn_used_llms': all_turn_used_llms,
            'all_turn_steps': all_turn_steps,
            'all_turn_categories': all_turn_categories,
            'all_pref_vecs': all_pref_vecs,
            'meta_info': meta_info,
            'vllm_model_configs': vllm_model_configs,
            'example_correct_by_rollout_id': example_correct_by_rollout_id,
            'tool_total': tool_total,
            'TRAINED_MODEL_TYPE': TRAINED_MODEL_TYPE,
            'cur_batch_indices': cur_batch_indices,
            'cur_batch_repeat_ids': cur_batch_repeat_ids,
            'cur_batch_output_files': cur_batch_output_files,
            'my_output_dir': my_output_dir,
            'pgx_tool_used_per_item': pgx_tool_used_per_item,
        }

    def run_llm_loop(self, gen_batch,tokenizer_config,global_steps,topk_doc,use_llm_reward,efficiency_reward,exp_tag,use_qa_reward):
        """
        Main loop for running LLM generation across multiple turns.
        
        This function orchestrates the entire generation process:
        1. Initializes variables and directories
        2. Runs multi-turn generation loop
        3. Processes results and extracts metadata
        4. Computes rewards using preference vectors
        5. Selects and normalizes training samples
        6. Returns formatted data for RL training
        
        Args:
            gen_batch: Generation batch containing input data
            tokenizer_config: Tokenizer configuration dictionary
            global_steps: Current global training step
            topk_doc: Top-k documents parameter
            use_llm_reward: Whether to use LLM-based reward
            efficiency_reward: Whether to use efficiency-based reward
            exp_tag: Experiment tag
            use_qa_reward: Whether to use QA-based reward
            
        Returns:
            Tuple of (DataProto containing training data, average tool usage count)
        """
        # Initialize variables and directories
        init_vars = self._initialize_run_llm_loop(gen_batch, tokenizer_config, global_steps)
        problem_key = init_vars['problem_key']
        loop_batch_size = init_vars['loop_batch_size']
        active_mask = init_vars['active_mask']
        tokenizer = init_vars['tokenizer']
        retrieved_documents = init_vars['retrieved_documents']
        code_snippets = init_vars['code_snippets']
        attempts = init_vars['attempts']
        total_costs = init_vars['total_costs']
        all_turn_input_ids = init_vars['all_turn_input_ids']
        all_turn_attention_mask = init_vars['all_turn_attention_mask']
        all_turn_position_ids = init_vars['all_turn_position_ids']
        all_turn_responses = init_vars['all_turn_responses']
        all_turn_index = init_vars['all_turn_index']
        all_turn_ids = init_vars['all_turn_ids']
        all_turn_turn_ids = init_vars['all_turn_turn_ids']
        all_turn_repeat_ids = init_vars['all_turn_repeat_ids']
        all_turn_answers = init_vars['all_turn_answers']
        all_turn_answer_preds = init_vars['all_turn_answer_preds']
        all_turn_problems = init_vars['all_turn_problems']
        all_turn_tools = init_vars['all_turn_tools']
        all_turn_success = init_vars['all_turn_success']
        all_turn_costs = init_vars['all_turn_costs']
        all_turn_latency = init_vars['all_turn_latency']
        all_turn_formats = init_vars['all_turn_formats']
        all_turn_valid_answers_generated = init_vars['all_turn_valid_answers_generated']
        all_turn_used_llms = init_vars['all_turn_used_llms']
        all_turn_steps = init_vars['all_turn_steps']
        all_turn_categories = init_vars['all_turn_categories']
        all_pref_vecs = init_vars['all_pref_vecs']
        meta_info = init_vars['meta_info']
        vllm_model_configs = init_vars['vllm_model_configs']
        example_correct_by_rollout_id = init_vars['example_correct_by_rollout_id']
        tool_total = init_vars['tool_total']
        TRAINED_MODEL_TYPE = init_vars['TRAINED_MODEL_TYPE']
        cur_batch_indices = init_vars['cur_batch_indices']
        cur_batch_repeat_ids = init_vars['cur_batch_repeat_ids']
        cur_batch_output_files = init_vars['cur_batch_output_files']
        my_output_dir = init_vars['my_output_dir']
        pgx_tool_used_per_item = init_vars['pgx_tool_used_per_item']
        
        rollings = gen_batch
        base_batch_size = loop_batch_size  # number of prompts; when rollout.n>1 we have loop_batch_size * n_rollouts_per_prompt items
        n_rollouts_per_prompt = 1

        # Create all rollout_step_1 .. rollout_step_max_turns dirs upfront so folder structure is complete
        # even when we break early (e.g. all samples done after turn 2)
        for s in range(1, self.config.max_turns + 1):
            step_dir = os.path.join(my_output_dir, f"global_step_{global_steps}", f"rollout_step_{s}")
            if not os.path.isdir(step_dir):
                os.makedirs(step_dir)
            code_exec_dir = os.path.join(step_dir, "code_execution")
            if not os.path.isdir(code_exec_dir):
                os.makedirs(code_exec_dir)

        # ==================== Main Multi-Turn Generation Loop ====================
        # Iterate through multiple turns (conversation rounds) until max_turns or all items are done
        # Note: range(1, max_turns + 1) ensures we execute exactly max_turns rounds (step 1 to max_turns)

        for step in range(1, self.config.max_turns + 1):
            if not active_mask.sum():
                break
            
            # Log turn start with active samples count
            active_count = active_mask.sum().item() if hasattr(active_mask, 'sum') else sum(active_mask)
            print(f"\n[Multi-Turn Dialogue] ========== Turn {step}/{self.config.max_turns} (Global Step {global_steps}) ==========")
            print(f"[Multi-Turn Dialogue] Active samples: {active_count}/{loop_batch_size}")
            if step==1:
                for item_idx in range(loop_batch_size):
                    category = gen_batch.non_tensor_batch['category'][item_idx]
                    item_index = gen_batch.non_tensor_batch['index'][item_idx]
                    item_repeat_id = gen_batch.non_tensor_batch['repeat_id'][item_idx]
                    iter_rollout_id = f"{item_index}_____{item_repeat_id}"
                    all_pref_vecs[iter_rollout_id] = gen_batch.non_tensor_batch['pref_vec'][item_idx]
                    if category!='func_call':
                        continue
                    cur_transfer_dir = os.path.join(gen_batch.non_tensor_batch['cur_transfer_dir'][item_idx],str(global_steps),iter_rollout_id)
                    task_path = os.path.join(cur_transfer_dir,'task.json')
                    cur_func_call_output_path = os.path.join(cur_transfer_dir,'output.json')
                    if not os.path.isdir(cur_transfer_dir):
                        os.makedirs(cur_transfer_dir,exist_ok=True)
                    with open(task_path,'w') as f:
                        json.dump([gen_batch.non_tensor_batch['example'][item_idx]],f,indent=2)
                    cur_tool = gen_batch.non_tensor_batch['tools'][item_idx]
                    cur_domain = item_index.split('____')[0]
                    func_call_cmd = ['python','rollout/tau2/cli.py','--domain',cur_domain,'--agent-llm','train',
                        '--user-llm','gpt-5','--num-trials','1','--task_path',str(task_path),'--max-steps','40','--cur_transfer_dir',
                        str(cur_transfer_dir),'--output_file',str(cur_func_call_output_path),'--use_model_tool']
                    subprocess.Popen(func_call_cmd)
            all_input_ids = []
            all_attention_mask = []
            all_categories = []
            all_indices = []
            all_levels = []
            all_transfer_dirs = []
            all_input_messages = []
            all_input_tools = []
            all_transfer_paths = []
            all_output_file_path = []
            all_model_mappings = []
            all_tool_pricing = []
            all_model_types = []
            assert len(retrieved_documents)==len(code_snippets)
            assert len(retrieved_documents)==len(attempts)
            item_idx = -1
            cached_inputs = []
            for doc_list,code_list,attempt_list in zip(retrieved_documents,code_snippets,attempts):
                item_idx += 1
                base_idx = item_idx // n_rollouts_per_prompt if n_rollouts_per_prompt > 1 else item_idx
                category = gen_batch.non_tensor_batch['category'][base_idx]
                all_categories.append(category)
                item_index = gen_batch.non_tensor_batch['index'][base_idx]
                item_repeat_id = gen_batch.non_tensor_batch['repeat_id'][base_idx]
                cur_model_mapping = gen_batch.non_tensor_batch['model_mapping'][base_idx]
                cur_tool_pricing = gen_batch.non_tensor_batch['tool_pricing'][base_idx]
                cur_model_type = gen_batch.non_tensor_batch['model_type'][base_idx]
                TRAINED_MODEL_TYPE = cur_model_type  # Keep for backward compatibility
                all_indices.append(item_index)
                # extra_info.level (precomputed and stored as non_tensor_batch['level']) if available
                if 'level' in gen_batch.non_tensor_batch:
                    all_levels.append(gen_batch.non_tensor_batch['level'][base_idx])
                else:
                    all_levels.append(None)
                all_model_mappings.append(cur_model_mapping)
                all_tool_pricing.append(cur_tool_pricing)
                all_model_types.append(cur_model_type)
                vllm_model_configs = gen_batch.non_tensor_batch['vllm_model_configs'][base_idx]
                
                # Initialize chat and tools variables to avoid UnboundLocalError
                chat = None
                tools = None
                
                if category=='qa':
                    problem = gen_batch.non_tensor_batch[problem_key][base_idx]
                    raw_tools = gen_batch.non_tensor_batch['tools'][base_idx]
                    my_output_dir = gen_batch.non_tensor_batch['my_output_dir'][base_idx]
                    # Convert to list to handle numpy arrays or other array types
                    if raw_tools is None:
                        tools = []
                    elif isinstance(raw_tools, (list, tuple)):
                        tools = list(raw_tools)
                    elif hasattr(raw_tools, '__iter__'):
                        tools = list(raw_tools)
                    else:
                        tools = []
                    doc_str = ''
                    for doc_idx, doc in enumerate(doc_list):
                        doc_str += f"Doc {doc_idx+1}: {doc[:4000]}\n\n"
                    code_str = ''
                    for code_idx, code_piece in enumerate(code_list):
                        code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                    attempt_str = ''
                    for attempt_idx, attempt in enumerate(attempt_list):
                        attempt_str += f"Attempt{attempt_idx+1} answer by {attempt['model']}: {attempt['answer']}\n"
                    str_cut = cut_seq(tokenizer=tokenizer,seq=attempt_str,l=8000)
                    attempt_str = str_cut['string_after_cut']
                    if not attempt_str.startswith('Attempt') and len(attempt_str)>0:
                        attempt_str = 'Attempt answer: '+attempt_str
                    str_cut = cut_seq(tokenizer=tokenizer,seq=code_str+attempt_str,l=16000)
                    code_attempt_str = str_cut['string_after_cut']
                    code_attempt_str_len = str_cut['effective_length']
                    if not code_attempt_str.startswith('```') and len(code_attempt_str)>0:
                        code_attempt_str = '```\n'+code_attempt_str
                    doc_flag = False
                    if code_attempt_str_len<24000:
                        context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_attempt_str,l=24000)
                        context_str = context_str['string_after_cut']
                        if len(doc_str)>0:
                            doc_flag = True
                            context_str = 'Documents:\n'+context_str
                    else:
                        context_str = code_attempt_str
                    chat = [
                                {"role": "system", "content": "You are good at using tools. "},
                                {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an approriate tool."}
                            ]
                    all_transfer_dirs.append('')
                    all_input_messages.append(chat)
                    # Ensure tools is a list before appending
                    if isinstance(tools, list):
                        all_input_tools.append(tools)
                    elif tools is not None and hasattr(tools, '__iter__'):
                        all_input_tools.append(list(tools))
                    else:
                        all_input_tools.append([])
                    all_transfer_paths.append(None)
                    all_output_file_path.append('')
                    if step==1:
                        cur_batch_output_files.append('')
                elif category=='func_call':
                    iter_rollout_id = f"{item_index}_____{item_repeat_id}"
                    cur_transfer_dir = os.path.join(gen_batch.non_tensor_batch['cur_transfer_dir'][base_idx],str(global_steps),iter_rollout_id)
                    all_transfer_dirs.append(cur_transfer_dir)
                    task_path = os.path.join(cur_transfer_dir,'task.json')
                    cur_func_call_output_path = os.path.join(cur_transfer_dir,'output.json')
                    transfer_idx = 0
                    while os.path.isfile(os.path.join(cur_transfer_dir,f"output_{transfer_idx}.json")) and os.path.isfile(os.path.join(cur_transfer_dir,f"input_{transfer_idx}.json")):
                        transfer_idx += 1
                    receive_end_signal = False
                    debug_file_path = os.path.join(cur_transfer_dir,f"wait_input_{transfer_idx}")
                    while not os.path.isfile(os.path.join(cur_transfer_dir,f"input_{transfer_idx}.json")):
                        if os.path.isfile(os.path.join(cur_transfer_dir,'done')):
                            with open(os.path.join(cur_transfer_dir,'done')) as f:
                                tmp_result = f.read()
                            if tmp_result=="Done!":
                                correct = 0
                                for subfile in os.listdir(os.path.join(cur_transfer_dir,'output')):
                                    if subfile.endswith('.json'):
                                        with open(os.path.join(cur_transfer_dir,'output',subfile)) as f:
                                            r = json.load(f)
                                        correct += r["reward_info"]["reward"]
                                if correct>0:
                                    simulation_reward = 1
                                else:
                                    simulation_reward = 0
                                example_correct_by_rollout_id[iter_rollout_id] = simulation_reward
                            active_mask[item_idx] = 0
                            receive_end_signal = True
                            break
                        time.sleep(5)
                    with open(os.path.join(cur_transfer_dir,f"input_{transfer_idx}.json")) as f:
                        input_dict = json.load(f)
                    if receive_end_signal:
                        with open('tools_debug.json') as f:
                            debug_tool = json.load(f)
                        original_tools = debug_tool
                        input_dict = {
                            'tools': debug_tool,
                            'original_tools': debug_tool,
                            'messages': [{'role': 'user','content': 'fake input'}],
                            'original_messages': [{'role': 'user','content': 'fake input'}]
                        }
                    else:
                        with open(os.path.join(cur_transfer_dir,f"input_{transfer_idx}.json")) as f:
                            input_dict = json.load(f)
                        original_tools = input_dict['original_tools']
                    tools = input_dict['tools']
                    tools_length = len(tokenizer(str(tools))['input_ids'])
                    chat = cut_middle_turns(tokenizer=tokenizer,messages=input_dict['messages'],max_length=23000-tools_length)
                    all_input_messages.append(input_dict['original_messages'])
                    all_input_tools.append(original_tools)
                    all_transfer_paths.append(os.path.join(cur_transfer_dir,f"output_{transfer_idx}.json"))
                    all_output_file_path.append(cur_func_call_output_path)
                    if step==1:
                        cur_batch_output_files.append(cur_transfer_dir)
                else:
                    # Handle other category types (e.g., the PGx tiers 'hard', 'intermediate' and 'common')
                    # For PGx, we should treat them as func_call since they use tool calling
                    # Extract problem/question for fallback
                    problem_key = 'problem' if 'problem' in gen_batch.non_tensor_batch else 'question'
                    if problem_key in gen_batch.non_tensor_batch:
                        problem = gen_batch.non_tensor_batch[problem_key][base_idx]
                    else:
                        problem = "Unknown question"
                    
                    # Load tools from tools field or use empty list
                    # Convert to list to handle numpy arrays or other array types
                    raw_tools = gen_batch.non_tensor_batch.get('tools', [None])[base_idx] if 'tools' in gen_batch.non_tensor_batch else None
                    if raw_tools is None:
                        tools = []
                    elif isinstance(raw_tools, (list, tuple)):
                        tools = list(raw_tools)
                    elif hasattr(raw_tools, '__iter__'):
                        tools = list(raw_tools)
                    else:
                        tools = []
                    
                    # For PGx categories, create output directory similar to func_call
                    # This is needed for reward reading later
                    iter_rollout_id = f"{item_index}_____{item_repeat_id}"
                    cur_transfer_dir = os.path.join(gen_batch.non_tensor_batch['cur_transfer_dir'][base_idx],str(global_steps),iter_rollout_id)
                    os.makedirs(cur_transfer_dir, exist_ok=True)
                    os.makedirs(os.path.join(cur_transfer_dir, 'output'), exist_ok=True)
                    
                    # For PGx categories, use the prompt from the data (which includes system message with tool instructions)
                    # Check if prompt exists in the data
                    if 'prompt' in gen_batch.non_tensor_batch:
                        chat = gen_batch.non_tensor_batch['prompt'][base_idx]
                        # Per-rollout diversity hint: nudge different rollouts to try different tools for meaningful comparison
                        if getattr(self.config, 'rollout_diversity_hint_until', -1) > 0 and global_steps <= self.config.rollout_diversity_hint_until:
                            import copy
                            chat = copy.deepcopy(chat)
                            # Diversity rotation over the active tool set: one distinct tool per rollout (item_repeat_id + base_idx).
                            preferred_tool = self.active_tools[(int(item_repeat_id) + int(item_index)) % len(self.active_tools)]
                            hint = f"\n[Variation {item_repeat_id}: Prefer choosing {preferred_tool} for this response.]"
                            for i in range(len(chat) - 1, -1, -1):
                                if isinstance(chat[i], dict) and chat[i].get('role') == 'user':
                                    chat[i] = {**chat[i], 'content': (chat[i].get('content') or '') + hint}
                                    break
                            if os.getenv("VERBOSE_TOOL_LOG", "0") == "1":
                                print(f"[Tool Orchestration] Diversity hint applied (item_idx={item_idx}, repeat_id={item_repeat_id}): prefer {preferred_tool}")
                        # Ensure tools are passed correctly
                        if len(tools) == 0 and len(chat) > 0:
                            # Tools should be loaded from tools field, but if empty, try to infer from prompt
                            print(f"[Tool Orchestration] PGx category '{category}' (idx={item_idx}): using prompt from data, tools={len(tools)} tools")
                    else:
                        # Fallback: create a simple chat template (still add diversity hint when enabled)
                        print(f"[Tool Orchestration] WARNING: PGx category '{category}' (idx={item_idx}) has no 'prompt' field, using fallback chat template")
                        user_content = problem
                        if getattr(self.config, 'rollout_diversity_hint_until', -1) > 0 and global_steps <= self.config.rollout_diversity_hint_until:
                            preferred_tool = self.active_tools[(int(item_repeat_id) + int(item_index)) % len(self.active_tools)]
                            user_content = user_content + f"\n[Variation {item_repeat_id}: Prefer choosing {preferred_tool} for this response.]"
                        chat = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": user_content}
                        ]
                    
                    # Log tools availability for PGx (tools is already converted to list above)
                    if len(tools) > 0:
                        tool_names = []
                        for t in tools:
                            if isinstance(t, dict):
                                # Tool format can be either {'function': {'name': ...}} or {'name': ...}
                                if 'function' in t and isinstance(t['function'], dict):
                                    tool_names.append(t['function']['name'])
                                elif 'name' in t:
                                    tool_names.append(t['name'])
                                else:
                                    raise ValueError(f"Tool definition missing 'name' field: {t}")
                            else:
                                tool_names.append(str(t))
                        print(f"[Tool Orchestration] PGx category '{category}' (idx={item_idx}): loaded {len(tools)} tool(s): {tool_names}")
                        
                        # Validate that only PGx tools are present - filter out any non-PGx tools
                        filtered_tools = []
                        for t in tools:
                            if isinstance(t, dict):
                                tool_name = None
                                if 'function' in t and isinstance(t['function'], dict):
                                    tool_name = t['function']['name']
                                elif 'name' in t:
                                    tool_name = t['name']
                                else:
                                    raise ValueError(f"Tool definition missing 'name' field: {t}")
                                if is_pgx_tool(tool_name):
                                    filtered_tools.append(t)
                                else:
                                    raise ValueError(f"Non-PGx tool '{tool_name}' found in tools list. Only {VALID_PGX_TOOL_NAMES} are allowed.")
                        if len(filtered_tools) != len(tools):
                            non_pgx_tools = [name for name in tool_names if not is_pgx_tool(name)]
                            print(f"[Tool Orchestration] ERROR: Found non-PGx tools in tools list: {non_pgx_tools}. Filtering them out - only {VALID_PGX_TOOL_NAMES} will be used.")
                            tools = filtered_tools
                    else:
                        print(f"[Tool Orchestration] WARNING: PGx category '{category}' (idx={item_idx}): no tools loaded! This may prevent tool calling.")
                    
                    all_transfer_dirs.append(cur_transfer_dir)
                    all_input_messages.append(chat)
                    # Ensure tools is a list before appending (already converted above)
                    all_input_tools.append(tools if isinstance(tools, list) else [])
                    all_transfer_paths.append(None)
                    all_output_file_path.append('')
                    if step==1:
                        cur_batch_output_files.append(cur_transfer_dir)
                    
                # Ensure chat and tools are defined before using them
                if chat is None:
                    raise ValueError(f"chat is None for category '{category}' at item_idx {item_idx}. This should not happen.")
                # Ensure tools is a list (handle numpy arrays or None)
                if tools is None:
                    tools = []
                elif not isinstance(tools, list):
                    tools = list(tools) if hasattr(tools, '__iter__') else []

                # PGx orchestrator: render the tool list INLINE in the system prompt (baked into the
                # data), NOT via the chat template's tools= block (which injects "You may call one or more
                # functions" + the <tools> schema). Pass tools=None so that block is gone, and
                # enable_thinking=True so the orchestrator reasons before emitting the tool JSON (thinking-off
                # collapsed filtered->tool_c; predict budget must be >= ~1300 tokens to avoid truncation).
                prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tools=None, tokenize=False, enable_thinking=True)
                cached_inputs.append(prompt_with_chat_template)
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                                tokenizer=tokenizer,
                                                                                max_length=tokenizer_config['max_prompt_length'],
                                                                                pad_token_id=tokenizer.pad_token_id,
                                                                                left_pad=True,
                                                                                truncation='middle')
                # assert attention_mask[0][0]==0,f"input_ids: {input_ids}\nattention_mask: {attention_mask}"
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)

            if not active_mask.sum():
                break
            # exit(0)
            if not os.path.isdir(os.path.join(my_output_dir,f"global_step_{global_steps}",f"rollout_step_{step}",'code_execution')):
                os.makedirs(os.path.join(my_output_dir,f"global_step_{global_steps}",f"rollout_step_{step}",'code_execution'))
            assert my_output_dir is not None
            input_ids = torch.cat(all_input_ids, dim=0)
            attention_mask = torch.cat(all_attention_mask, dim=0)
            input_tokens_count = torch.sum(attention_mask,dim=1)
            position_ids = compute_position_id_with_mask(attention_mask)
            rollings.batch['input_ids'] = input_ids
            rollings.batch['attention_mask'] = attention_mask
            rollings.batch['position_ids'] = position_ids
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

            # Parse-failure retry: if any PGx-category sample produces an
            # unparseable response, regenerate just those samples up to
            # PGX_MAX_PARSE_RETRY times. Done BEFORE n_expand and downstream
            # state mutations so we can simply replace rows in responses_ids/str.
            # After exhausting retries, downstream postprocess_predictions will
            # raise (with full prediction dump) as before.
            PGX_MAX_PARSE_RETRY = 2
            _active_local_idxs = [i for i, m in enumerate(active_mask.tolist()) if m]
            _active_categories = [all_categories[i] for i in _active_local_idxs]
            for _retry_attempt in range(PGX_MAX_PARSE_RETRY):
                _bad_idxs = [
                    j for j, (_resp, _cat) in enumerate(zip(responses_str, _active_categories))
                    if is_pgx_category(_cat) and not _can_parse_pgx_prediction(_resp)
                ]
                if not _bad_idxs:
                    break
                print(
                    f"[Tool Orchestration] Parse-retry {_retry_attempt+1}/{PGX_MAX_PARSE_RETRY}: "
                    f"{len(_bad_idxs)} unparseable PGx sample(s) at active idxs={_bad_idxs} — regenerating."
                )
                _bad_batch_dict = {k: v[_bad_idxs] for k, v in rollings_active.batch.items()}
                _bad_batch = DataProto.from_dict(_bad_batch_dict)
                _new_gen = self._generate_with_gpu_padding(_bad_batch)
                _new_responses_ids, _new_responses_str = self._postprocess_responses(_new_gen.batch['responses'])
                # Replace rows. Tensor row-replacement requires matching width;
                # both come from the same tokenizer/max_response_length so widths align.
                responses_ids[_bad_idxs] = _new_responses_ids
                for _j_new, _j_orig in enumerate(_bad_idxs):
                    responses_str[_j_orig] = _new_responses_str[_j_new]

            # When rollout worker uses n>1 (multiple samples per prompt), responses have shape (num_prompts*n,) but rollings_active has (num_prompts,). Expand rollings_active by repeat_interleave so sizes match.
            num_prompts = rollings_active.batch['input_ids'].shape[0]
            num_responses = responses_ids.shape[0]
            if num_responses != num_prompts and num_responses % num_prompts == 0:
                n_expand = num_responses // num_prompts
                expanded_batch = {k: v.repeat_interleave(n_expand, dim=0) for k, v in rollings_active.batch.items()}
                rollings_active = DataProto.from_dict(expanded_batch)
            else:
                n_expand = 1
            
            # Log model responses for PGx debugging
            if global_steps % 10 == 0 or os.getenv("VERBOSE_TOOL_LOG", "0") == "1":
                print(f"[Tool Orchestration] Model responses preview (step {step}, global_step {global_steps}):")
                for idx, resp_str in enumerate(responses_str[:min(3, len(responses_str))]):
                    category = all_categories[idx] if idx < len(all_categories) else 'unknown'
                    print(f"  Response {idx} (category={category}): {resp_str[:300]}...")
                    # Check if contains tool call indicators
                    if 'pgx_query' in resp_str.lower() or '<tool_call>' in resp_str or 'tool_call' in resp_str.lower():
                        print(f"    ⚠️  Contains potential tool call keywords!")
            
            input_padding = torch.ones((len(rollings_active.batch['input_ids']), tokenizer_config['max_response_length']+tokenizer_config['max_prompt_length']-len(rollings_active.batch['input_ids'][0])-len(responses_ids[0])), dtype=torch.long)*tokenizer.pad_token_id
            mask_padding = torch.zeros((len(rollings_active.batch['input_ids']), tokenizer_config['max_response_length']+tokenizer_config['max_prompt_length']-len(rollings_active.batch['input_ids'][0])-len(responses_ids[0])), dtype=torch.long)
            round_input_ids = torch.cat([input_padding,rollings_active.batch['input_ids'],responses_ids], dim=1)
            round_attention_mask = torch.cat([mask_padding,self.tensor_fn.create_attention_mask(rollings_active.batch['input_ids']),self.tensor_fn.create_attention_mask(responses_ids)], dim=1)
            round_position_ids = self.tensor_fn.create_position_ids(round_attention_mask)
            all_turn_input_ids.append(round_input_ids)
            all_turn_attention_mask.append(round_attention_mask)
            all_turn_position_ids.append(round_position_ids)
            all_turn_responses.append(responses_ids)
            def _repeat_for_rollout_n(lst, n):
                if n <= 1:
                    return lst
                return [lst[i] for i in range(len(lst)) for _ in range(n)]
            _base_index = [gen_batch.non_tensor_batch['index'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['index'])) if active_mask[local_idx]]
            cur_turn_index = _repeat_for_rollout_n(_base_index, n_expand)
            assert len(cur_turn_index)==round_input_ids.shape[0],f"len(cur_turn_index): {len(cur_turn_index)}, round_input_ids.shape[0]: {round_input_ids.shape[0]}"
            all_turn_index += cur_turn_index
            cur_turn_turn_ids = _repeat_for_rollout_n([gen_batch.non_tensor_batch['turn_id'][local_idx]+step for local_idx in range(len(gen_batch.non_tensor_batch['turn_id'])) if active_mask[local_idx]], n_expand)
            assert len(cur_turn_turn_ids)==round_input_ids.shape[0]
            all_turn_turn_ids += cur_turn_turn_ids
            cur_turn_repeat_ids = _repeat_for_rollout_n([gen_batch.non_tensor_batch['repeat_id'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['repeat_id'])) if active_mask[local_idx]], n_expand)
            assert len(cur_turn_repeat_ids) == round_input_ids.shape[0]
            all_turn_repeat_ids += cur_turn_repeat_ids
            cur_turn_categories = _repeat_for_rollout_n([gen_batch.non_tensor_batch['category'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['category'])) if active_mask[local_idx]], n_expand)
            assert len(cur_turn_categories) == round_input_ids.shape[0]
            all_turn_categories += cur_turn_categories
            cur_turn_ids = _repeat_for_rollout_n([gen_batch.non_tensor_batch['id'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['id'])) if active_mask[local_idx]], n_expand)
            assert len(cur_turn_ids) == round_input_ids.shape[0]
            all_turn_ids += cur_turn_ids
            cur_turn_problems = _repeat_for_rollout_n([gen_batch.non_tensor_batch[problem_key][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]], n_expand)
            assert len(cur_turn_problems) == round_input_ids.shape[0]
            all_turn_problems += cur_turn_problems
            cur_turn_steps = _repeat_for_rollout_n([step for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]], n_expand)
            assert len(cur_turn_steps) == round_input_ids.shape[0]
            all_turn_steps += cur_turn_steps
            cur_answers = _repeat_for_rollout_n([gen_batch.non_tensor_batch['answer'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['answer'])) if active_mask[local_idx]], n_expand)
            assert len(cur_answers) == round_input_ids.shape[0]
            all_turn_answers += cur_answers
            if n_expand > 1:
                active_mask = torch.tensor([active_mask[local_idx].item() for local_idx in range(active_mask.shape[0]) for _ in range(n_expand)], dtype=torch.bool)
                retrieved_documents = [copy.deepcopy(retrieved_documents[i]) for i in range(len(retrieved_documents)) for _ in range(n_expand)]
                code_snippets = [copy.deepcopy(code_snippets[i]) for i in range(len(code_snippets)) for _ in range(n_expand)]
                attempts = [copy.deepcopy(attempts[i]) for i in range(len(attempts)) for _ in range(n_expand)]
                total_costs = _repeat_for_rollout_n(total_costs, n_expand)
                all_tool_pricing = _repeat_for_rollout_n(all_tool_pricing, n_expand)
                all_model_types = _repeat_for_rollout_n(all_model_types, n_expand)
                all_input_messages = _repeat_for_rollout_n(all_input_messages, n_expand)
                all_input_tools = _repeat_for_rollout_n(all_input_tools, n_expand)
                all_transfer_dirs = _repeat_for_rollout_n(all_transfer_dirs, n_expand)
                all_transfer_paths = _repeat_for_rollout_n(all_transfer_paths, n_expand)
                all_model_mappings = _repeat_for_rollout_n(all_model_mappings, n_expand)
                all_categories = _repeat_for_rollout_n(all_categories, n_expand)
                all_indices = _repeat_for_rollout_n(all_indices, n_expand)
                all_levels = _repeat_for_rollout_n(all_levels, n_expand)
                pgx_tool_used_per_item = _repeat_for_rollout_n(pgx_tool_used_per_item, n_expand)
            if n_expand <= 1:
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            else:
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            response_id_mask = (responses_ids!=tokenizer.pad_token_id)
            response_tokens_count = torch.sum(response_id_mask,dim=1)
            if n_expand > 1:
                input_tokens_count = torch.sum(rollings_active.batch['attention_mask'], dim=1)
            main_agent_cost = []
            assert len(all_tool_pricing)==len(input_tokens_count)
            assert len(all_tool_pricing)==len(response_tokens_count)
            
            # Default pricing fallback for common model types
            DEFAULT_MODEL_PRICING = {
                'Qwen/Qwen3-8B': {
                    'input_tokens_per_million': 0.2/1000000,
                    'output_tokens_per_million': 0.2/1000000
                },
                'Qwen/Qwen3-32B': {
                    'input_tokens_per_million': 0.8/1000000,
                    'output_tokens_per_million': 0.8/1000000
                },
                'Qwen/Qwen2.5-Coder-32B-Instruct': {
                    'input_tokens_per_million': 0.8/1000000,
                    'output_tokens_per_million': 0.8/1000000
                },
                'Qwen/Qwen2.5-Math-72B-Instruct': {
                    'input_tokens_per_million': 0.9/1000000,
                    'output_tokens_per_million': 0.9/1000000
                },
                'Qwen/Qwen2.5-Math-7B-Instruct': {
                    'input_tokens_per_million': 0.2/1000000,
                    'output_tokens_per_million': 0.2/1000000
                },
                'meta-llama/Llama-3.3-70B-Instruct': {
                    'input_tokens_per_million': 0.9/1000000,
                    'output_tokens_per_million': 0.9/1000000
                },
                'gpt-5': {
                    'input_tokens_per_million': 1.25/1000000,
                    'output_tokens_per_million': 10/1000000
                },
                'gpt-5-mini': {
                    'input_tokens_per_million': 0.25/1000000,
                    'output_tokens_per_million': 2/1000000
                },
            }
            
            for one_tool_pricing, model_type, ic, rc in zip(all_tool_pricing, all_model_types, input_tokens_count.tolist(), response_tokens_count.tolist()):
                # Try to get pricing from one_tool_pricing, fallback to default if not found
                if one_tool_pricing and model_type in one_tool_pricing:
                    model_pricing = one_tool_pricing[model_type]
                elif model_type in DEFAULT_MODEL_PRICING:
                    model_pricing = DEFAULT_MODEL_PRICING[model_type]
                else:
                    # Last resort: use Qwen/Qwen3-8B pricing as default
                    model_pricing = DEFAULT_MODEL_PRICING.get('Qwen/Qwen3-8B', {
                        'input_tokens_per_million': 0.2/1000000,
                        'output_tokens_per_million': 0.2/1000000
                    })
                
                main_agent_cost.append(model_pricing['input_tokens_per_million']*ic+model_pricing['output_tokens_per_million']*rc)
            assert len(total_costs)==len(active_mask)
            assert len(total_costs)==len(main_agent_cost)
            for iter_idx in range(len(total_costs)):
                if active_mask[iter_idx]:
                    total_costs[iter_idx] += main_agent_cost[iter_idx]
            assert len(all_categories)==len(responses_str)
            assert len(all_categories)==len(responses_str)
            assert len(all_indices)==len(all_categories)
            assert len(all_levels)==len(all_categories)
            _user_problems = cur_turn_problems if n_expand > 1 else gen_batch.non_tensor_batch[problem_key]
            _answers = cur_answers if n_expand > 1 else gen_batch.non_tensor_batch['answer']
            # Pass base path so execute_predictions can put each item (n_agent rollout) in rollout_{i}/code_execution
            cur_output_dir_base = os.path.join(my_output_dir, f"global_step_{global_steps}", f"rollout_step_{step}")
            new_documents, new_code, new_attempts, dones, new_attempt_correct, new_costs, tool_summary,format_correctness, latencies, all_attempt_results,valid_answers_generated,used_llms, updated_pgx_tool_used = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, retrieved_documents=retrieved_documents, user_problems=_user_problems,
                code_snippets=code_snippets,attempts=attempts,tokenizer=tokenizer,vllm_model_configs=vllm_model_configs,cur_output_dir=cur_output_dir_base,
                answers=_answers,total_costs=total_costs,all_categories=all_categories,all_transfer_dirs=all_transfer_dirs,
                all_input_messages=all_input_messages,all_input_tools=all_input_tools,all_transfer_paths=all_transfer_paths,all_model_mappings=all_model_mappings,
                all_tool_pricing=all_tool_pricing,all_indices=all_indices,all_levels=all_levels,topk_doc=topk_doc,pgx_tool_used_per_item=pgx_tool_used_per_item
            )
            # Update tracking: if a PGx tool was used in this step, mark it
            pgx_tool_used_per_item = updated_pgx_tool_used
            for tmp_tool_l1 in tool_summary:
                assert isinstance(tmp_tool_l1,list)
                for tmp_tool in tmp_tool_l1:
                    if tmp_tool[0]!='':
                        tool_total += 1
            for i,iter_docs in enumerate(new_documents):
                assert isinstance(iter_docs,list)
                retrieved_documents[i] = merge_documents(main_list=retrieved_documents[i],sub_list=iter_docs)
            for i,c in enumerate(new_code):
                assert isinstance(c,list)
                if len(c)>0:
                    code_snippets[i] += c
            for i,a in enumerate(new_attempts):
                assert isinstance(a,list)
                if len(a)>0:
                    attempts[i] += a
            assert len(tool_summary)==len(new_attempt_correct)
            assert len(new_costs)==len(total_costs)
            assert len(active_mask)==len(total_costs)
            for iter_idx in range(len(total_costs)):
                if active_mask[iter_idx]:
                    total_costs[iter_idx] += new_costs[iter_idx]
            if n_expand > 1:
                cur_turn_tools = list(tool_summary)
                cur_turn_success = list(new_attempt_correct)
                cur_turn_costs = [new_costs[i] + main_agent_cost[i] for i in range(len(new_costs))]
                cur_turn_used_llms = list(used_llms)
                cur_turn_valid_answers_generated = list(valid_answers_generated)
                cur_turn_latency = list(latencies)
                cur_turn_formats = list(format_correctness)
            else:
                cur_turn_tools = [tool_summary[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]]
                cur_turn_success = [new_attempt_correct[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]]
                cur_turn_costs = [new_costs[local_idx]+main_agent_cost[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]]
                cur_turn_used_llms = [used_llms[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]]
                cur_turn_valid_answers_generated = [valid_answers_generated[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]]
                cur_turn_latency = [latencies[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]]
                cur_turn_formats = [format_correctness[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch[problem_key])) if active_mask[local_idx]]
            assert len(cur_turn_tools) == round_input_ids.shape[0]
            all_turn_tools += cur_turn_tools
            assert len(cur_turn_success) == round_input_ids.shape[0]
            all_turn_success += cur_turn_success
            assert len(cur_turn_costs) == round_input_ids.shape[0]
            all_turn_costs += cur_turn_costs
            assert len(cur_turn_used_llms) == round_input_ids.shape[0]
            all_turn_used_llms += cur_turn_used_llms
            assert len(cur_turn_valid_answers_generated) == round_input_ids.shape[0]
            all_turn_valid_answers_generated += cur_turn_valid_answers_generated
            assert len(cur_turn_latency) == round_input_ids.shape[0]
            all_turn_latency += cur_turn_latency
            if step == 1 and len(cur_turn_latency) > 0:
                print(f"[Latency Debug] Turn {step}: Collected {len(cur_turn_latency)} latency values: {[f'{l:.2f}s' for l in cur_turn_latency[:3]]}{'...' if len(cur_turn_latency) > 3 else ''}")
            assert len(cur_turn_formats) == round_input_ids.shape[0]
            all_turn_formats += cur_turn_formats

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            if n_expand > 1:
                n_rollouts_per_prompt = n_expand
                loop_batch_size = num_responses
                cur_batch_indices = _repeat_for_rollout_n(cur_batch_indices, n_expand)
                cur_batch_repeat_ids = _repeat_for_rollout_n(cur_batch_repeat_ids, n_expand)
                cur_batch_output_files = _repeat_for_rollout_n(cur_batch_output_files, n_expand)

        # ==================== Post-Processing: Extract Results and Metadata ====================
        # Process attempts to extract valid answers, response_json, and pre-computed eval results
        valid_answers = {}
        # Build mapping from rollout_id to response_json (for PGx results)
        response_json_by_rollout_id = {}  # rollout_id -> response_json dict
        # Build mapping from rollout_id to pre-computed eval_result (from pgx_env.py)
        # This avoids duplicate LLM evaluation calls
        precomputed_eval_results_by_rollout_id = {}  # rollout_id -> eval_result dict
        assert len(attempts)==len(cur_batch_indices)
        assert len(attempts)==len(cur_batch_repeat_ids)
        assert len(attempts)==len(cur_batch_output_files)
        for a,tmp_idx,tmp_repeat_id,tmp_output_file in zip(attempts,cur_batch_indices,cur_batch_repeat_ids,cur_batch_output_files):
            rollout_id = f"{tmp_idx}_____{tmp_repeat_id}"
            if not tmp_output_file:
                print(f"[ERROR] Missing output file for rollout {rollout_id} (example_idx={tmp_idx}, repeat_id={tmp_repeat_id})")
                raise ValueError(f"Missing output file for rollout {rollout_id}. tmp_output_file is None or empty.")
            for tmp_a in a:
                if len(tmp_a['answer'].strip())>0:
                    valid_answers[rollout_id] = True
                # Extract response_json and eval_result from PGx results
                if 'pgx_result' in tmp_a:
                    pgx_result = tmp_a['pgx_result']
                    response_json = pgx_result.get('response_json') or pgx_result.get('final_answer_json')
                    if response_json and isinstance(response_json, dict):
                        response_json_by_rollout_id[rollout_id] = response_json
                    # Extract pre-computed eval_result if available (from pgx_env.py call_tool)
                    eval_result = pgx_result.get('eval_result')
                    if eval_result and isinstance(eval_result, dict) and 'weighted_score' in eval_result:
                        precomputed_eval_results_by_rollout_id[rollout_id] = eval_result
                        print(f"[Reward Model] Found pre-computed eval_result for rollout {rollout_id} (from pgx_env.py)")
            correct = 0
            for subfile in os.listdir(os.path.join(tmp_output_file,'output')):
                if subfile.endswith('.json'):
                    with open(os.path.join(tmp_output_file,'output',subfile)) as f:
                        r = json.load(f)
                    correct += r["reward_info"]["reward"]
            if correct>-1:
                valid_answers[rollout_id] = True
            if correct>0:
                example_correct_by_rollout_id[rollout_id] = 1
            else:
                example_correct_by_rollout_id[rollout_id] = 0
        assert len(all_turn_index)==len(all_turn_valid_answers_generated)
        for example_idx,example_repeat_id,turn_valid_answer in zip(all_turn_index,all_turn_repeat_ids,all_turn_valid_answers_generated):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            if not rollout_id in valid_answers:
                valid_answers[rollout_id] = turn_valid_answer
            elif not valid_answers[rollout_id]:
                valid_answers[rollout_id] = turn_valid_answer

        # Concatenate all turn data into tensors
        all_turn_input_ids = torch.cat(all_turn_input_ids, dim=0)
        all_turn_attention_mask = torch.cat(all_turn_attention_mask, dim=0)
        all_turn_position_ids = torch.cat(all_turn_position_ids, dim=0)
        all_turn_responses = torch.cat(all_turn_responses, dim=0)

        # ==================== Compute Example Correctness, Costs, and Latency ====================
        # Validate data consistency
        assert len(all_turn_index)==len(all_turn_success)
        assert len(all_turn_index)==len(all_turn_repeat_ids)
        assert len(all_turn_index)==len(all_turn_used_llms)
        assert len(all_turn_index)==len(all_turn_categories)
        assert len(all_turn_index)==len(all_turn_tools)
        assert len(all_turn_index)==len(all_turn_steps)
        
        # Compute example_correctness: True if turn was successful or example was correct
        example_correctness = {}
        for example_idx,example_repeat_id,turn_success in zip(all_turn_index,all_turn_repeat_ids,all_turn_success):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            if turn_success or (rollout_id in example_correct_by_rollout_id and example_correct_by_rollout_id[rollout_id]):
                example_correctness[rollout_id] = True
            elif not rollout_id in example_correctness:
                example_correctness[rollout_id] = False
        
        # Compute example_costs and example_latency (accumulated across all turns)
        # IMPORTANT: cost and latency should be NEGATIVE values (lower is better)
        # So we convert them to negative when accumulating
        example_costs = defaultdict(int)
        example_latency = defaultdict(int)
        for example_idx,example_repeat_id,turn_cost,turn_latency in zip(all_turn_index,all_turn_repeat_ids,all_turn_costs,all_turn_latency):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            # Convert to negative: cost and latency are penalties (lower is better)
            example_costs[rollout_id] += -turn_cost
            example_latency[rollout_id] += -turn_latency
            # Debug: Print latency accumulation for first few rollouts
            if global_steps == 1 and len([r for r in example_latency.keys() if r.startswith(f"{example_idx}_____")]) <= 3:
                print(f"[Latency Debug] Rollout {rollout_id}: turn_latency={turn_latency:.2f}s (positive), accumulated example_latency={example_latency[rollout_id]:.2f}s (negative)")
        
        # ==================== Legacy Efficiency Reward Calculation ====================
        # Note: This section uses a different reward calculation method (efficiency_reward)
        # ==================== Legacy Efficiency Reward Calculation ====================
        # Since example_costs and example_latency are now NEGATIVE, we need to adjust the calculation
        # However, this section might not be used if preference vector method is used instead
        example_rewards = defaultdict(list)
        rewards_by_rollout_id = {}
        for turn_format,example_idx,example_repeat_id in zip(all_turn_formats,all_turn_index,all_turn_repeat_ids):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            if str(efficiency_reward).lower()=='true' and example_correctness[rollout_id]:
                # example_costs and example_latency are now NEGATIVE, so we use absolute values for this calculation
                cost_reward = abs(example_costs[rollout_id])*5
                latency_reward = abs(example_latency[rollout_id])/500
                if cost_reward+latency_reward>0.8:
                    rewards_by_rollout_id[rollout_id] = int(example_correctness[rollout_id])-0.8
                    example_rewards[example_idx].append(int(example_correctness[rollout_id])-0.8)
                else:
                    rewards_by_rollout_id[rollout_id] = int(example_correctness[rollout_id])-latency_reward-cost_reward
                    example_rewards[example_idx].append(int(example_correctness[rollout_id])-latency_reward-cost_reward)
            else:
                rewards_by_rollout_id[rollout_id] = int(example_correctness[rollout_id])
                example_rewards[example_idx].append(int(example_correctness[rollout_id]))
        
        # Build level (hard / filtered / simple) per rollout from key-derived extra_info.level
        all_turn_levels = None
        if 'level' in gen_batch.non_tensor_batch:
            index_arr = gen_batch.non_tensor_batch['index']
            level_arr = gen_batch.non_tensor_batch['level']
            index_to_level = {}
            for idx in range(len(index_arr)):
                index_to_level[index_arr[idx]] = level_arr[idx] if idx < len(level_arr) else None
            all_turn_levels = [index_to_level.get(ex_idx) for ex_idx in all_turn_index]
        # ==================== Process Weighted Scores (AnswerRewardEvaluator) ====================
        # Under REWARD_ADV_MODE=cache_baseline the reward is built from live pipeline runs and the
        # Verifier (live_reward.py, below), so the answer evaluator is not run.
        from config import REWARD_ADV_MODE as _reward_adv_mode_pre
        if _reward_adv_mode_pre == "cache_baseline":
            rm_scores_by_rollout_id, use_rm_for_accuracy, reward_output_dir, detailed_eval_results = {}, False, None, {}
        else:
          rm_scores_by_rollout_id, use_rm_for_accuracy, reward_output_dir, detailed_eval_results = self._process_weighted_scores(
            all_turn_index=all_turn_index,
            all_turn_repeat_ids=all_turn_repeat_ids,
            all_turn_responses=all_turn_responses,
            all_turn_problems=all_turn_problems,
            all_turn_answers=all_turn_answers,
            precomputed_eval_results_by_rollout_id=precomputed_eval_results_by_rollout_id,
            response_json_by_rollout_id=response_json_by_rollout_id,
            global_steps=global_steps,
            my_output_dir=my_output_dir,
            example_correctness=example_correctness,
            all_turn_tools=all_turn_tools,
            all_turn_levels=all_turn_levels,
        )

        # ==================== Override Cost/Latency from eval_cache (eval_mode=existing) ====================
        # Only active when REWARD_EVAL_MODE == "existing". When active, cost & latency are
        # read STRICTLY from eval_cache_qid_tool_claims.json — any rollout without those fields
        # raises (rebuild via build_qid_tool_claims_cache.py to populate them).
        from config import REWARD_EVAL_MODE as _reward_eval_mode
        if _reward_eval_mode == "existing" and detailed_eval_results:
            _cache_override_count = 0
            _missing_fields = []
            for rollout_id, eval_result in detailed_eval_results.items():
                cost = eval_result.get("cost")
                latency = eval_result.get("latency_seconds")
                if cost is None or latency is None:
                    _missing_fields.append(rollout_id)
                    continue
                example_costs[rollout_id] = -abs(cost)
                example_latency[rollout_id] = -abs(latency)
                _cache_override_count += 1
            if _missing_fields:
                raise RuntimeError(
                    f"[Reward Cache] {len(_missing_fields)} rollouts have no cost/latency in cache "
                    f"(e.g. {_missing_fields[:3]}). Re-run apply_cache_overrides.py to populate them."
                )
            if _cache_override_count > 0:
                print(f"[Reward Cache] ✓ Overrode cost/latency for {_cache_override_count} rollouts from eval_cache")

        # ==================== Save Metrics to JSON Files ====================
        # Save example_costs and example_latency to JSON files for analysis
        metrics_output_dir = os.path.join(my_output_dir, f"global_step_{global_steps}", "metrics")
        if not os.path.isdir(metrics_output_dir):
            os.makedirs(metrics_output_dir, exist_ok=True)
        
        # Save example_costs
        costs_file = os.path.join(metrics_output_dir, "example_costs.json")
        with open(costs_file, 'w') as f:
            json.dump({rollout_id: float(cost) for rollout_id, cost in example_costs.items()}, f, indent=2)
        print(f"[Metrics] ✓ Saved example_costs to {costs_file}")
        
        # Save example_latency
        latency_file = os.path.join(metrics_output_dir, "example_latency.json")
        with open(latency_file, 'w') as f:
            json.dump({rollout_id: float(latency) for rollout_id, latency in example_latency.items()}, f, indent=2)
        print(f"[Metrics] ✓ Saved example_latency to {latency_file}")
        
        # ==================== Compute Rewards Using Preference Vector ====================
        # Compute rewards using preference vector (always use this approach, but replace accuracy with RM score if available)
        # Reset rewards_by_rollout_id for preference vector calculation
        from config import REWARD_NORMALIZATION_SCOPE as _reward_norm_scope
        _norm_scope = (_reward_norm_scope or "per_question").strip().lower()
        if _norm_scope not in ("per_question", "per_batch"):
            _norm_scope = "per_question"
        if global_steps == 1:
            print(f"[Reward Normalization] REWARD_NORMALIZATION_SCOPE={_norm_scope} (min-max + z-score pooling)")

        rewards_by_rollout_id = {}
        example_rewards = defaultdict(list)
        if use_rm_for_accuracy:
            reward_source = "AnswerRewardEvaluator (normalized 0-1 scores replace accuracy)"
        else:
            reward_source = "Preference Vector (original accuracy 0/1)"
        
        # Debug: Check pref_vec for first example
        if global_steps == 1:
            print(f"\n[Reward Debug] Checking preference vectors for first few rollouts:")
            for idx, (example_idx, example_repeat_id) in enumerate(zip(all_turn_index[:3], all_turn_repeat_ids[:3])):
                rollout_id = f"{example_idx}_____{example_repeat_id}"
                if rollout_id in all_pref_vecs:
                    pref_vec = all_pref_vecs[rollout_id]
                    print(f"  Rollout {rollout_id}: pref_vec={pref_vec} (type={type(pref_vec)}, len={len(pref_vec) if hasattr(pref_vec, '__len__') else 'N/A'})")
                else:
                    print(f"  Rollout {rollout_id}: pref_vec NOT FOUND in all_pref_vecs")
            
            # Check if all pref_vecs are empty
            empty_pref_vecs = sum(1 for pv in all_pref_vecs.values() if (isinstance(pv, dict) and len(pv) == 0) or (isinstance(pv, list) and len(pv) == 0))
            total_pref_vecs = len(all_pref_vecs)
            print(f"[Reward Debug] Total pref_vecs: {total_pref_vecs}, Empty pref_vecs: {empty_pref_vecs}")
            if empty_pref_vecs == total_pref_vecs and total_pref_vecs > 0:
                print(f"[Reward Debug] ⚠️  WARNING: All preference vectors are empty! This will cause all rewards to be 0.")
                print(f"[Reward Debug] Please check that your training data includes 'pref_vec' field with non-empty dictionaries.")
        
        # ==================== Compute Rewards Using Preference Vector ====================
        # For each rollout, compute reward by combining contributions from:
        # 1. Accuracy (either RM score or 0/1, normalized by min/max)
        # 2. Latency (normalized by min/max, already negative)
        # 3. Cost (normalized by min/max, already negative)
        # Each contribution = pref_weight * normalized_value
        # Store normalized scores per rollout for summary.json (before weighted sum)
        normalized_scores_by_rollout_id = {}
        feature_min_max_cache = {}
        batch_feature_min_max = None
        if _norm_scope == "per_batch":
            batch_feature_min_max = self._compute_feature_min_max(
                example_idx=None,
                all_turn_index=all_turn_index,
                all_turn_repeat_ids=all_turn_repeat_ids,
                example_costs=example_costs,
                example_latency=example_latency,
                rm_scores_by_rollout_id=rm_scores_by_rollout_id,
                example_correctness=example_correctness,
                use_rm_for_accuracy=use_rm_for_accuracy,
            )
        for turn_format,example_idx,example_repeat_id in zip(all_turn_formats,all_turn_index,all_turn_repeat_ids):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            if rollout_id in rewards_by_rollout_id:
                continue
            rewards_by_rollout_id[rollout_id] = 0
            
            # ==================== Centralized Feature Values Dictionary ====================
            # Store all feature values in a unified dictionary structure
            # Use reward model score as accuracy if available, otherwise use original correctness
            if use_rm_for_accuracy and rollout_id in rm_scores_by_rollout_id:
                accuracy_value = rm_scores_by_rollout_id[rollout_id]  # Use RM score as accuracy
                if global_steps % 50 == 0 and iter_index < 3:  # Log first 3 examples every 50 steps
                    print(f"[Reward Model] Using RM score for rollout {rollout_id}: accuracy={accuracy_value:.4f} (replaced original {int(example_correctness[rollout_id])})")
            else:
                accuracy_value = int(example_correctness[rollout_id])  # Use original 0/1 accuracy
            
            # Skip if using original accuracy and correctness is False
            if not use_rm_for_accuracy and not example_correctness[rollout_id]:
                example_rewards[example_idx].append(rewards_by_rollout_id[rollout_id])
                continue
            
            # Centralized feature values dictionary
            feature_values = {
                'accuracy': accuracy_value,
                'latency': example_latency[rollout_id],
                'cost': example_costs[rollout_id]
            }
            cur_pref_vec_raw = all_pref_vecs.get(rollout_id, {})
            
            # Ensure cur_pref_vec is a dict (fail fast if wrong type)
            if isinstance(cur_pref_vec_raw, list):
                print(f"[ERROR] pref_vec for {rollout_id} is a list (length={len(cur_pref_vec_raw)}), expected dict. pref_vec={cur_pref_vec_raw}")
                raise TypeError(f"pref_vec for rollout {rollout_id} must be a dict, got list instead.")
            elif not isinstance(cur_pref_vec_raw, dict):
                print(f"[ERROR] pref_vec for {rollout_id} has unexpected type {type(cur_pref_vec_raw)}, expected dict. pref_vec={cur_pref_vec_raw}")
                raise TypeError(f"pref_vec for rollout {rollout_id} must be a dict, got {type(cur_pref_vec_raw)} instead.")
            
            # Use default preference weights if pref_vec is empty
            # Default weights: accuracy=1.0, latency=0.1, cost=0.1
            # This ensures rewards are computed even when training data doesn't include pref_vec
            DEFAULT_PREF_VEC = {
                'accuracy': 1.0,
                'latency': 0.1,
                'cost': 0.1
            }
            if len(cur_pref_vec_raw) == 0:
                cur_pref_vec = DEFAULT_PREF_VEC
                if global_steps == 1 and example_idx == all_turn_index[0] and example_repeat_id == all_turn_repeat_ids[0]:
                    print(f"[Reward Debug] ⚠️  Using default preference vector: {DEFAULT_PREF_VEC}")
            else:
                cur_pref_vec = cur_pref_vec_raw
            
            # Min-max (uniform) normalization: per question or over full batch (see config.REWARD_NORMALIZATION_SCOPE)
            if _norm_scope == "per_batch":
                feature_min_max = batch_feature_min_max
            else:
                if example_idx not in feature_min_max_cache:
                    feature_min_max_cache[example_idx] = self._compute_feature_min_max(
                        example_idx=example_idx,
                        all_turn_index=all_turn_index,
                        all_turn_repeat_ids=all_turn_repeat_ids,
                        example_costs=example_costs,
                        example_latency=example_latency,
                        rm_scores_by_rollout_id=rm_scores_by_rollout_id,
                        example_correctness=example_correctness,
                        use_rm_for_accuracy=use_rm_for_accuracy,
                    )
                feature_min_max = feature_min_max_cache[example_idx]
            normalized_scores_by_rollout_id[rollout_id] = {}
            
            # Debug: Log pref_vec and values for first rollout
            if global_steps == 1 and example_idx == all_turn_index[0] and example_repeat_id == all_turn_repeat_ids[0]:
                print(f"\n[Reward Debug] Computing reward for first rollout {rollout_id}:")
                print(f"  cur_pref_vec={cur_pref_vec} (type={type(cur_pref_vec)})")
                print(f"  feature_values={feature_values}")
                print(f"    accuracy={feature_values['accuracy']}, cost={abs(feature_values['cost']):.4f} (raw: {feature_values['cost']:.4f} negative), latency={abs(feature_values['latency']):.2f}s (raw: {feature_values['latency']:.2f}s negative)")
                print(f"  feature_min_max={feature_min_max}")
                # Verify latency is being used
                latency_pref_weight = cur_pref_vec.get('latency', 0.0)
                if latency_pref_weight > 0:
                    print(f"  ✓ Latency will contribute to reward (pref_weight={latency_pref_weight:.4f})")
                else:
                    print(f"  ⚠️  Latency pref_weight is {latency_pref_weight:.4f}, will NOT contribute to reward")
            
            # ==================== Compute Rewards for All Features ====================
            # Loop over accuracy, latency, and cost using centralized feature_values dictionary
            for feature_name in ['accuracy', 'latency', 'cost']:
                # Get feature value and preference weight
                feature_value = feature_values[feature_name]
                pref_weight = cur_pref_vec.get(feature_name, 0.0)
                
                # Get min/max values for this feature
                min_val = feature_min_max[feature_name]['min']
                max_val = feature_min_max[feature_name]['max']
                
                # Compute normalized value using min-max normalization
                # Formula: normalized_value = (value - min) / (max - min + 1e-6)
                # Add 1e-6 to denominator to avoid division by zero
                normalized_value = (feature_value - min_val) / (max_val - min_val + 1e-6)
                # Store for summary.json (before weighted sum): accuracy, cost, efficiency (latency)
                key = 'efficiency' if feature_name == 'latency' else feature_name
                normalized_scores_by_rollout_id[rollout_id][key] = float(normalized_value)
                
                # Compute reward contribution
                reward_contribution = pref_weight * normalized_value
                rewards_by_rollout_id[rollout_id] += reward_contribution
                
                # Debug logging for first rollout
                if global_steps == 1 and example_idx == all_turn_index[0] and example_repeat_id == all_turn_repeat_ids[0]:
                    if feature_name in ['latency', 'cost']:
                        # Display negative values in positive form for readability
                        value_positive = abs(feature_value)
                        min_positive = abs(min_val)
                        max_positive = abs(max_val)
                        print(f"    {feature_name.capitalize()}: pref_weight={pref_weight}, value={value_positive:.4f} (raw: {feature_value:.4f} negative), "
                              f"min={min_positive:.4f} (raw: {min_val:.4f}), max={max_positive:.4f} (raw: {max_val:.4f}), "
                              f"normalized_value={normalized_value:.4f}, contribution={reward_contribution:.4f}")
                    else:
                        # Accuracy (already positive)
                        print(f"    {feature_name.capitalize()}: pref_weight={pref_weight}, value={feature_value}, "
                              f"min={min_val}, max={max_val}, normalized_value={normalized_value:.4f}, contribution={reward_contribution:.4f}")
                
                # Periodic logging for latency if pref_weight > 0
                if feature_name == 'latency' and pref_weight > 0 and global_steps % 10 == 0:
                    latency_positive = abs(feature_value)
                    print(f"[Latency Reward] Rollout {rollout_id}: latency={latency_positive:.2f}s, pref_weight={pref_weight:.4f}, contribution={reward_contribution:.4f}")
            
            if global_steps == 1 and example_idx == all_turn_index[0] and example_repeat_id == all_turn_repeat_ids[0]:
                print(f"  Final reward for {rollout_id}: {rewards_by_rollout_id[rollout_id]:.4f}\n")
        
        # ==================== REWARD_ADV_MODE=cache_baseline override ====================
        # Replace the per-question min-max reward computed above with
        #   F15[qid, chosen_tool] - mean_t F15[qid, t]   (t over the active tool set; F15 and cost come
        #   from live pipeline runs scored by the Verifier this step, live_reward.build_reward_table).
        # This value IS the final advantage: algorithm.adv_estimator=cache_baseline is an identity
        # pass-through, so the trainer does NOT re-normalize by sibling mean/std. Unanimous-tool groups
        # therefore keep a correctly-signed nonzero advantage, and reward magnitudes are preserved.
        from config import REWARD_ADV_MODE as _reward_adv_mode, REWARD_COST_WEIGHT as _reward_cost_weight
        if _reward_adv_mode == "cache_baseline":
            from router_reward import live_reward
            _live_table = live_reward.build_reward_table(
                all_turn_index, all_turn_problems, self.active_tools, global_steps, my_output_dir)
            _cb_rewards, _cb_details = compute_cache_baseline_rewards(
                all_turn_index, all_turn_repeat_ids, all_turn_tools, self.active_tools, _live_table,
                cost_weight=_reward_cost_weight)
            rewards_by_rollout_id = dict(_cb_rewards)
            example_rewards = defaultdict(list)
            reward_source = (f"cache_baseline (F15[qid,chosen] - mean_t F15[qid,t] - {_reward_cost_weight} * cost_feature; "
                             f"final advantage, no group re-normalization)")
            _cb_dir = os.path.join(my_output_dir, f"global_step_{global_steps}", "rl_training_inputs")
            os.makedirs(_cb_dir, exist_ok=True)
            with open(os.path.join(_cb_dir, "cache_baseline_details.json"), "w") as _f:
                json.dump(_cb_details, _f, indent=2)
            _vals = list(rewards_by_rollout_id.values())
            print(f"[Reward][cache_baseline] step {global_steps}: n={len(_vals)} mean={sum(_vals)/len(_vals):+.4f} "
                  f"min={min(_vals):+.4f} max={max(_vals):+.4f} nonzero={sum(1 for v in _vals if abs(v) > 1e-9)}")

        # Fill example_rewards so stats and display show what is actually used for training
        for example_idx, example_repeat_id in zip(all_turn_index, all_turn_repeat_ids):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            example_rewards[example_idx].append(rewards_by_rollout_id.get(rollout_id, 0.0))
        
        # Z-score (standardize) over the whole batch when REWARD_NORMALIZATION_SCOPE=per_batch
        from statistics import stdev as _batch_stdev
        batch_reward_average: Optional[float] = None
        batch_reward_std: Optional[float] = None
        if _norm_scope == "per_batch":
            _batch_vals = [
                rewards_by_rollout_id.get(f"{i}_____{r}", 0.0)
                for i, r in zip(all_turn_index, all_turn_repeat_ids)
            ]
            n_b = len(_batch_vals)
            if n_b == 0:
                batch_reward_average, batch_reward_std = 0.0, 0.01
            else:
                batch_reward_average = sum(_batch_vals) / n_b
                if n_b < 2:
                    batch_reward_std = 0.01
                elif n_b == 2:
                    batch_reward_std = max(_batch_stdev(_batch_vals), 0.01)
                else:
                    batch_reward_std = _batch_stdev(_batch_vals)
            if global_steps == 1 or global_steps % 10 == 0:
                print(f"[RL Reward Stats] Batch-level z-score: mean={batch_reward_average:.4f}, std={batch_reward_std:.4f} (n={n_b})")
        
        if global_steps % 10 == 0:
            rmin, rmax = min(rewards_by_rollout_id.values()), max(rewards_by_rollout_id.values())
            print(f"[Reward Stats] Reward range: min={rmin:.4f}, max={rmax:.4f}")
        
        # Log summary statistics
        if use_rm_for_accuracy:
            rm_replaced_count = sum(1 for rid in rewards_by_rollout_id.keys() if rid in rm_scores_by_rollout_id)
            total_count = len(rewards_by_rollout_id)
            print(f"[Reward Model] Summary: Replaced accuracy for {rm_replaced_count}/{total_count} rollouts with AnswerRewardEvaluator normalized scores (0-1 range)")
        
        # Update reward_model_results/summary.json with cost_normalized, efficiency_normalized, and full_reward per rollout
        # Note: summary "scores" / rm_scores_by_rollout_id are accuracy-only (RM weighted_score). full_reward = accuracy + latency + cost contributions, clipped to [-1,1].
        if reward_output_dir and normalized_scores_by_rollout_id and os.path.isdir(reward_output_dir):
            summary_file = os.path.join(reward_output_dir, "summary.json")
            if os.path.isfile(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                if "rollout_details" in summary:
                    for rid, details in summary["rollout_details"].items():
                        ns = normalized_scores_by_rollout_id.get(rid, {})
                        details["cost_normalized"] = ns.get("cost")
                        details["efficiency_normalized"] = ns.get("efficiency")
                        details["full_reward"] = float(rewards_by_rollout_id[rid]) if rid in rewards_by_rollout_id else None
                    summary["full_reward_by_rollout_id"] = {rid: float(r) for rid, r in rewards_by_rollout_id.items()}
                    summary["reward_after_latency_and_cost_by_rollout_id"] = {rid: float(r) for rid, r in rewards_by_rollout_id.items()}
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                    print(f"[Reward Model] ✓ Updated summary.json with cost_normalized, efficiency_normalized, full_reward, and reward_after_latency_and_cost_by_rollout_id")
                    try:
                        csv_path = _write_scores_result_csv_from_summary(summary_file)
                        print(f"[Reward Model] ✓ Exported scores_result.csv to {csv_path}")
                    except Exception as e:
                        print(f"[Reward Model] ⚠️  Failed to export scores_result.csv: {e}")
                    # Save reward_after_latency_and_cost_by_rollout_id as standalone JSON (accuracy + latency + cost summed, clipped [-1,1])
                    reward_after_file = os.path.join(reward_output_dir, "reward_after_latency_and_cost_by_rollout_id.json")
                    with open(reward_after_file, 'w') as f:
                        json.dump({rid: float(r) for rid, r in rewards_by_rollout_id.items()}, f, indent=2)
                    print(f"[Reward Model] ✓ Saved reward_after_latency_and_cost_by_rollout_id to {reward_after_file}")
        
        # Log raw rewards by rollout_id (detailed view)
        if global_steps % 10 == 0 or os.getenv("VERBOSE_REWARD_LOG", "0") == "1":  # Log every 10 steps or when verbose
            print(f"\n[RL Reward Details] Step {global_steps} - Raw rewards by rollout (source: {reward_source}):")
            if use_rm_for_accuracy:
                print(f"[RL Reward Details] Note: accuracy values are from AnswerRewardEvaluator (normalized 0-1 range), not original 0/1 correctness")
            
            rm_stats = None
            if use_rm_for_accuracy and rm_scores_by_rollout_id:
                rm_scores_list = list(rm_scores_by_rollout_id.values())
                rm_stats = {
                    'min': min(rm_scores_list),
                    'max': max(rm_scores_list),
                    'mean': sum(rm_scores_list) / len(rm_scores_list),
                }
                print(f"[RL Reward Details] RM accuracy stats: min={rm_stats['min']:.4f}, max={rm_stats['max']:.4f}, mean={rm_stats['mean']:.4f}")
            
            for rollout_id in sorted(rewards_by_rollout_id.keys()):
                example_idx, repeat_id = rollout_id.split('_____')
                raw_reward = rewards_by_rollout_id[rollout_id]
                correctness = example_correctness.get(rollout_id, False)
                cost = example_costs.get(rollout_id, 0)
                latency = example_latency.get(rollout_id, 0)
                
                # Display latency in positive seconds for readability (latency is stored as negative)
                latency_positive = abs(latency) if latency < 0 else latency
                if use_rm_for_accuracy and rollout_id in rm_scores_by_rollout_id:
                    rm_accuracy = rm_scores_by_rollout_id[rollout_id]
                    original_acc = int(correctness)
                    print(f"  Rollout {rollout_id}: raw_reward={raw_reward:.4f}, "
                          f"rm_accuracy={rm_accuracy:.4f} (replaced {original_acc}), "
                          f"correctness={correctness}, cost={abs(cost):.4f}, latency={latency_positive:.2f}s")
                else:
                    print(f"  Rollout {rollout_id}: raw_reward={raw_reward:.4f}, "
                          f"correctness={correctness}, cost={abs(cost):.4f}, latency={latency_positive:.2f}s")
            
        # ==================== Compute Reward Statistics ====================
        # Calculate average and standard deviation of rewards for each example
        example_indices = list(example_rewards.keys())
        example_reward_average = {}
        example_reward_std = {}
        from statistics import stdev
        for example_idx in example_indices:
            reward_count = len(example_rewards[example_idx])
            if reward_count < 2:
                # If we have fewer than 2 rewards, we can't compute std meaningfully
                # This can happen if some rollouts were filtered out (e.g., invalid format, no valid answer)
                print(f"[Warning] Example {example_idx} has only {reward_count} reward(s), expected at least 2. "
                      f"This may happen if some rollouts were filtered. Using fallback: std=0.0")
                example_reward_average[example_idx] = sum(example_rewards[example_idx])/reward_count if reward_count > 0 else 0.0
                example_reward_std[example_idx] = 0.0  # Set std to 0 when we can't compute it
            elif reward_count == 2:
                # With exactly 2 values, stdev can be computed but may be less reliable
                # Use a small epsilon to ensure std > 0 for normalization
                example_reward_average[example_idx] = sum(example_rewards[example_idx])/reward_count
                computed_std = stdev(example_rewards[example_idx])
                example_reward_std[example_idx] = max(computed_std, 0.01)  # Minimum std of 0.01 for stability
            else:
                # Normal case: 3+ rewards, compute std normally
                example_reward_average[example_idx] = sum(example_rewards[example_idx])/reward_count
                example_reward_std[example_idx] = stdev(example_rewards[example_idx])
        
        # Log reward statistics for each training example
        print(f"\n[RL Reward Stats] Step {global_steps} - Reward breakdown by example:")
        for example_idx in sorted(example_indices):
            raw_rewards = example_rewards[example_idx]
            avg_reward = example_reward_average[example_idx]
            std_reward = example_reward_std[example_idx]
            print(f"  Example {example_idx}: avg={avg_reward:.4f}, std={std_reward:.4f}, "
                  f"raw_rewards={[f'{r:.4f}' for r in raw_rewards]}")

        # ==================== Normalize Rewards and Select Training Samples ====================
        # Normalize rewards by example (z-score normalization) and select samples for training
        rewards = []
        selected_indices = []
        iter_index = -1
        for turn_format,example_idx,example_repeat_id,turn_used_llms,turn_category,turn_step,turn_tools in zip(all_turn_formats,all_turn_index,all_turn_repeat_ids,all_turn_used_llms,all_turn_categories,all_turn_steps,all_turn_tools):
            iter_index += 1
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            # Safety check: ensure rollout_id has a reward
            if rollout_id not in rewards_by_rollout_id:
                print(f"[Warning] Rollout {rollout_id} not found in rewards_by_rollout_id, using 0.0")
                rewards_by_rollout_id[rollout_id] = 0.0
            # ==================== Pass Raw Reward (Pure GRPO) ====================
            # Keep raw reward here; compute_grpo_outcome_advantage does the only
            # group z-score normalization. Previously we z-scored here AND again
            # inside the trainer, which double-normalized. We still compute
            # per-example/per-batch std below purely for the filtering condition.
            if _norm_scope == "per_batch":
                _r_avg = batch_reward_average if batch_reward_average is not None else 0.0
                _r_std = batch_reward_std if batch_reward_std is not None else 0.01
            else:
                _r_avg = example_reward_average[example_idx]
                _r_std = example_reward_std[example_idx]
            if rollout_id in valid_answers and valid_answers[rollout_id] and turn_format:
                selected_indices.append(iter_index)
            rewards.append(rewards_by_rollout_id[rollout_id])

        # Save rewards_by_rollout_id and normalized rewards BEFORE potential early return
        # This ensures data is saved even if training samples are insufficient
        rl_inputs_output_dir = os.path.join(my_output_dir, f"global_step_{global_steps}", "rl_training_inputs")
        if not os.path.isdir(rl_inputs_output_dir):
            os.makedirs(rl_inputs_output_dir, exist_ok=True)
        
        # Save rewards_by_rollout_id (raw rewards before normalization)
        raw_rewards_file = os.path.join(rl_inputs_output_dir, "rewards_by_rollout_id.json")
        with open(raw_rewards_file, 'w') as f:
            json.dump({rollout_id: float(reward) for rollout_id, reward in rewards_by_rollout_id.items()}, f, indent=2)
        print(f"[RL Training Inputs] ✓ Saved rewards_by_rollout_id (raw rewards) to {raw_rewards_file}")
        
        # Save normalized rewards (mapped to rollout_ids)
        normalized_rewards_dict = {}
        for i, (idx, repeat_id, reward_val) in enumerate(zip(all_turn_index, all_turn_repeat_ids, rewards)):
            rollout_id = f"{idx}_____{repeat_id}"
            normalized_rewards_dict[rollout_id] = float(reward_val)
        
        normalized_rewards_file = os.path.join(rl_inputs_output_dir, "normalized_rewards.json")
        with open(normalized_rewards_file, 'w') as f:
            json.dump(normalized_rewards_dict, f, indent=2)
        print(f"[RL Training Inputs] ✓ Saved normalized_rewards (actual RL input) to {normalized_rewards_file}")

        def selection_list(original_list,selection_indices):
            if isinstance(original_list,list):
                new_list = []
                for sidx in selection_indices:
                    new_list.append(original_list[sidx])
                return new_list
            else:
                return torch.index_select(original_list, dim=0, index=selection_indices)

        total_nodes = self.config.num_gpus 
        # For small datasets, allow at least 1 sample instead of requiring num_gpus samples
        # This prevents skipping batches when filtered samples < num_gpus
        # Original requirement: len(selected_indices) >= total_nodes
        # Modified for small datasets: len(selected_indices) >= 1 (if any samples exist)
        min_required_samples = 1 if len(selected_indices) > 0 else total_nodes
        if len(selected_indices) < min_required_samples:
            print(f"[Sample Selection] WARNING: Only {len(selected_indices)} samples passed filtering (required: {min_required_samples}, num_gpus: {total_nodes}). Returning None.")
            return None,None
        
        indices = torch.tensor(selected_indices)
        all_turn_input_ids = torch.index_select(all_turn_input_ids, dim=0, index=indices)
        all_turn_attention_mask = torch.index_select(all_turn_attention_mask, dim=0, index=indices)
        all_turn_position_ids = torch.index_select(all_turn_position_ids, dim=0, index=indices)
        all_turn_responses = torch.index_select(all_turn_responses, dim=0, index=indices)
        all_turn_ids = selection_list(all_turn_ids,selected_indices)
        all_turn_index = selection_list(all_turn_index,selected_indices)
        all_turn_turn_ids = selection_list(all_turn_turn_ids,selected_indices)
        all_turn_repeat_ids = selection_list(all_turn_repeat_ids,selected_indices)
        all_turn_answers = selection_list(all_turn_answers,selected_indices)
        all_turn_problems = selection_list(all_turn_problems,selected_indices)
        all_turn_tools = selection_list(all_turn_tools,selected_indices)
        all_turn_success = selection_list(all_turn_success,selected_indices)
        all_turn_costs = selection_list(all_turn_costs,selected_indices)
        all_turn_formats = selection_list(all_turn_formats,selected_indices)
        rewards = selection_list(rewards,selected_indices)

        cur_remainder = len(all_turn_ids)%total_nodes
        if cur_remainder!=0:
            all_turn_input_ids = all_turn_input_ids[:-cur_remainder]
            all_turn_attention_mask = all_turn_attention_mask[:-cur_remainder]
            all_turn_position_ids = all_turn_position_ids[:-cur_remainder]
            all_turn_responses = all_turn_responses[:-cur_remainder]
            all_turn_ids = all_turn_ids[:-cur_remainder]
            all_turn_index = all_turn_index[:-cur_remainder]
            all_turn_turn_ids = all_turn_turn_ids[:-cur_remainder]
            all_turn_repeat_ids = all_turn_repeat_ids[:-cur_remainder]
            all_turn_answers = all_turn_answers[:-cur_remainder]
            all_turn_problems = all_turn_problems[:-cur_remainder]
            all_turn_tools = all_turn_tools[:-cur_remainder]
            all_turn_success = all_turn_success[:-cur_remainder]
            all_turn_costs = all_turn_costs[:-cur_remainder]
            all_turn_formats = all_turn_formats[:-cur_remainder]
            rewards = rewards[:-cur_remainder]
        # For small datasets, allow at least 1 sample instead of requiring num_gpus samples
        # Original requirement: len(all_turn_input_ids) >= total_nodes
        # Modified for small datasets: len(all_turn_input_ids) >= 1 (if any samples exist)
        min_required_samples_after_trim = 1 if len(all_turn_input_ids) > 0 else total_nodes
        if len(all_turn_input_ids) < min_required_samples_after_trim:
            print(f"[Sample Selection] WARNING: After trimming remainder, only {len(all_turn_input_ids)} samples remain (required: {min_required_samples_after_trim}, num_gpus: {total_nodes}). Returning None.")
            return None, None

        assert len(all_turn_input_ids)==len(rewards)
        
        # ==================== Log Final Normalized Rewards ====================
        # Log final normalized rewards used for training
        if len(rewards) > 0:
            print(f"[RL Reward Stats] Step {global_steps} - Final raw rewards (for training, GRPO normalizes inside trainer):")
            print(f"  Total training samples: {len(rewards)}")
            print(f"  Reward range: [{min(rewards):.4f}, {max(rewards):.4f}], mean={sum(rewards)/len(rewards):.4f}")
            # Show reward for each selected training example
            for i, (idx, repeat_id, reward_val) in enumerate(zip(all_turn_index[:min(10, len(rewards))],
                                                                  all_turn_repeat_ids[:min(10, len(rewards))],
                                                                  rewards[:min(10, len(rewards))])):
                print(f"    Sample {i+1}: example_idx={idx}, repeat_id={repeat_id}, raw_reward={reward_val:.4f}")
            if len(rewards) > 10:
                print(f"    ... (showing first 10 of {len(rewards)} samples)")
        
        # Note: normalized_rewards are already saved before potential early return (above)
        # This ensures data is saved even if training samples are insufficient
        
        # ==================== Build Final Output Dictionary ====================
        # Construct final dictionary with all training data
        final_dict = {
            'input_ids': all_turn_input_ids,
            'attention_mask': all_turn_attention_mask,
            'position_ids': all_turn_position_ids,
            'responses': all_turn_responses,
            'id': np.array(all_turn_ids),
            'index': np.array(all_turn_index),
            'turn_id': np.array(all_turn_turn_ids),
            'repeat_id': np.array(all_turn_repeat_ids),
            'answer': np.array(all_turn_answers),
            'reward': np.array(rewards),
        }
        return_dict = DataProto.from_single_dict(final_dict,meta_info=meta_info)
        return return_dict,tool_total/loop_batch_size

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, retrieved_documents=None, user_problems=None,
                                code_snippets=None, attempts=None, tokenizer=None,vllm_model_configs=None,cur_output_dir=None,answers=None,
                                total_costs=None,all_categories=None,all_transfer_dirs=None,all_input_messages=None,all_input_tools=None,all_transfer_paths=None,
                                all_model_mappings=None,all_tool_pricing=None,all_indices=None,all_levels=None,topk_doc=30,pgx_tool_used_per_item=None):
        # Initialize tracking if not provided
        if pgx_tool_used_per_item is None:
            pgx_tool_used_per_item = [False] * len(predictions)
        
        # Create local reference to module-level functions to avoid UnboundLocalError
        _is_pgx_tool = is_pgx_tool
        _get_mode = get_pgx_mode_from_tool
        
        # Log raw predictions before parsing
        verbose_log = os.getenv("VERBOSE_TOOL_LOG", "0") == "1"
        if verbose_log:
            print(f"[Tool Orchestration] Raw predictions before parsing:")
            for idx, (pred, cat) in enumerate(zip(predictions[:min(3, len(predictions))], all_categories[:min(3, len(all_categories))])):
                print(f"  Prediction {idx} (category={cat}): {str(pred)[:200]}...")
        
        cur_tool_calls, contents, format_correctness = self.postprocess_predictions(predictions,all_categories)
        
        # Log parsed tool calls and pipeline modes
        pgx_tool_calls_found = 0
        selected_modes = []  # Track selected pipeline modes
        for idx, (tool_call_list, cat) in enumerate(zip(cur_tool_calls, all_categories)):
            if tool_call_list:
                for tc in tool_call_list:
                    if isinstance(tc, dict):
                        tool_name = tc['name']
                        if _is_pgx_tool(tool_name):
                            pgx_tool_calls_found += 1
                            mode = _get_mode(tool_name)
                            selected_modes.append((idx, mode, tool_name, cat))
                            print(f"[Tool Orchestration] ✓ Parsed PGx tool call (category={cat}, idx={idx}): tool={tool_name}, mode={mode}")
        
        # Print summary of selected pipeline modes
        if selected_modes:
            print(f"[Actor Pipeline Selection] ========== Selected Pipeline Modes (Round Summary) ==========")
            for idx, mode, tool_name, cat in selected_modes:
                print(f"[Actor Pipeline Selection]   Sample {idx}: Pipeline Mode = {mode.upper()} (Tool: {tool_name}, Category: {cat})")
            print(f"[Actor Pipeline Selection] ========== Total: {len(selected_modes)} pipeline mode(s) selected ==========")
        
        if pgx_tool_calls_found == 0:
            print(f"[Tool Orchestration] ⚠️  No PGx tool calls parsed from {len(predictions)} predictions")
            print(f"  Categories found: {set(all_categories)}")
            print(f"  Format correctness: {sum(format_correctness)}/{len(format_correctness)}")

        assert len(cur_tool_calls) == len(active_mask)
        assert len(cur_tool_calls) == len(retrieved_documents)
        assert len(cur_tool_calls) == len(user_problems)
        assert len(cur_tool_calls) == len(code_snippets)
        assert len(cur_tool_calls) == len(attempts)
        assert len(cur_tool_calls) == len(answers)
        assert len(cur_tool_calls) == len(all_categories)
        assert len(cur_tool_calls) == len(all_transfer_dirs)
        assert len(cur_tool_calls) == len(all_input_messages)
        assert len(cur_tool_calls) == len(all_input_tools)
        assert len(cur_tool_calls) == len(all_transfer_paths)
        assert len(cur_tool_calls) == len(all_model_mappings)

        from tqdm import tqdm
        all_call_tool_arguments = []
        # cur_output_dir is the base path (rollout_step_{step}); each item gets rollout_{i}/code_execution under it (n_agent separation)
        cur_output_dir_base = cur_output_dir
        # semaphore = asyncio.Semaphore(512)
        for i, (iter_tool_calls, active, doc_list, user_problem, code_list, attempt_list, answer, cur_category, cur_transfer_dir,cur_input_messages,cur_input_tools,cur_transfer_path,cur_model_mapping,cur_tool_pricing,cur_index, cur_level) in enumerate(zip(cur_tool_calls, active_mask, retrieved_documents, user_problems, code_snippets, attempts, answers, all_categories, all_transfer_dirs, all_input_messages, all_input_tools,all_transfer_paths,all_model_mappings,all_tool_pricing,all_indices, all_levels)):
            # Per-item (per n_agent rollout) output dir so 3 rollouts -> rollout_0, rollout_1, rollout_2
            rollout_dir = os.path.join(cur_output_dir_base, f"rollout_{i}") if cur_output_dir_base else None
            item_output_dir = os.path.join(rollout_dir, "code_execution") if rollout_dir else None
            if item_output_dir:
                os.makedirs(item_output_dir, exist_ok=True)
            # Persist actor reasoning + raw output (orchestrator sometimes emits reasoning before JSON).
            # `contents` is the extracted thought/reasoning for this prediction.
            if rollout_dir:
                try:
                    os.makedirs(rollout_dir, exist_ok=True)
                    actor_record = {
                        "index_in_batch": i,
                        "category": cur_category,
                        "question_index": cur_index,
                        "level": cur_level,
                        "actor_reasoning": contents[i] if i < len(contents) else "",
                        "raw_prediction": predictions[i] if i < len(predictions) else "",
                        "parsed_tool_calls": iter_tool_calls,
                    }
                    with open(os.path.join(rollout_dir, "actor_output.json"), "w", encoding="utf-8") as f:
                        json.dump(actor_record, f, ensure_ascii=False, indent=2)
                    with open(os.path.join(rollout_dir, "actor_reasoning.txt"), "w", encoding="utf-8") as f:
                        f.write((actor_record.get("actor_reasoning") or "") + ("\n" if actor_record.get("actor_reasoning") else ""))
                except Exception as e:
                    print(f"[Tool Orchestration] Warning: failed to save actor reasoning for rollout_{i}: {e}")
            call_tool_arguments = []
            is_pgx_tool_category = is_pgx_category(cur_category)
            if cur_category=='qa' or is_pgx_tool_category:
                # Debug: Log tool calls received for this item
                print(f"[Tool Orchestration] Processing item {i} (category={cur_category}, active={active}): {len(iter_tool_calls) if isinstance(iter_tool_calls, list) else 'N/A'} tool call(s), already_used={pgx_tool_used_per_item[i] if i < len(pgx_tool_used_per_item) else 'N/A'}")
                
                # Check if this item has already used a PGx tool in a previous step
                if i < len(pgx_tool_used_per_item) and pgx_tool_used_per_item[i]:
                    raise ValueError(f"Item {i} has already used a PGx tool in a previous step. Cannot use PGx tool again for the same item.")
                
                added_pgx_tools = set()
                pgx_tool_executed = False
                print(f"[Tool Orchestration] Item {i}: Processing {len(iter_tool_calls)} tool call(s) for category {cur_category}")
                for tid,tool_call in enumerate(iter_tool_calls):
                    if not isinstance(tool_call, dict):
                        raise TypeError(f"Item {i}, tid={tid}: Expected dict for tool_call, got {type(tool_call)}")
                    
                    # Normalize tool call format: convert 'parameters' to 'arguments' if needed
                    # Some models generate 'parameters' instead of 'arguments'
                    if 'parameters' in tool_call and 'arguments' not in tool_call:
                        tool_call['arguments'] = tool_call.pop('parameters')
                        print(f"[Tool Orchestration] Normalized tool call format: converted 'parameters' to 'arguments' (qa, i={i}, tid={tid})")
                    elif 'parameters' in tool_call and 'arguments' in tool_call:
                        # Both exist - prefer 'arguments', but log the conflict
                        print(f"[Tool Orchestration] Warning: Tool call has both 'parameters' and 'arguments', using 'arguments' (qa, i={i}, tid={tid})")
                        tool_call.pop('parameters', None)
                    
                    tool_name_in_call = tool_call.get('name')
                    if not tool_name_in_call:
                        raise ValueError(f"Item {i}, tid={tid}: Tool call missing required 'name' field")
                    is_pgx_call = _is_pgx_tool(tool_name_in_call)
                    if is_pgx_call:
                        print(f"[Tool Orchestration] Validating PGx tool call (qa, i={i}, tid={tid}): {tool_call}")
                    
                    if not active:
                        raise ValueError(f"Item {i} is not active, cannot process tool call")
                    if not isinstance(tool_call, dict):
                        raise TypeError(f"Item {i}, tid={tid}: Expected dict for tool_call, got {type(tool_call)}")
                    tool_call_keys = set(tool_call.keys())
                    # After normalization, we should have 'name' and 'arguments'
                    if 'name' not in tool_call_keys:
                        raise ValueError(f"Item {i}, tid={tid}: Tool call missing required 'name' field. Got keys: {tool_call_keys}")
                    # Allow 'arguments' to be missing (will default to {})
                    if tool_call_keys - {'name', 'arguments'}:
                        unexpected_keys = tool_call_keys - {'name', 'arguments'}
                        raise ValueError(f"Item {i}, tid={tid}: Tool call has unexpected keys: {unexpected_keys}. Expected only 'name' and 'arguments'.")
                    tool_call_name = tool_call['name']
                    
                    if tool_call_name not in ALL_TOOLS:
                        raise ValueError(f"Item {i}, tid={tid}: Tool '{tool_call_name}' not in ALL_TOOLS. Allowed tools: {list(ALL_TOOLS.keys())}")
                    
                    if not _is_pgx_tool(tool_call_name):
                        raise ValueError(f"Tool '{tool_call_name}' is not allowed in PGx training. Only {VALID_PGX_TOOL_NAMES} are allowed. (category={cur_category}, i={i})")
                    
                    valid_tool_call = True
                    
                    # Ensure only one tool is selected per turn
                    if is_pgx_call:
                        tool_name = tool_call['name']
                        mode = _get_mode(tool_name)
                        if len(added_pgx_tools) > 0:
                            first_tool = list(added_pgx_tools)[0]
                            first_mode = _get_mode(first_tool)
                            raise ValueError(f"Multiple PGx tool calls detected (i={i}, tid={tid}). Only one PGx tool call is allowed per turn. Already added: {first_tool} (mode={first_mode}), trying to add: {tool_name} (mode={mode}).")
                        added_pgx_tools.add(tool_name)
                        print(f"[Actor Pipeline Selection] Sample {i}: Selected Pipeline Mode = {mode.upper()} (Tool: {tool_name}, Category: {cur_category})")
                        print(f"[Tool Orchestration] ✓ PGx tool call validated (qa, i={i}): tool={tool_name}, mode={mode}")
                    
                    # Validate tool signature (PGx tools have no parameters)
                    tool_call_name = tool_call['name']
                    func_signature = ALL_TOOLS.get(tool_call_name, {})  # Use get with default {} as this is a lookup
                    tool_arguments = tool_call.get('arguments', {})  # arguments may not exist for tools with no parameters
                    tool_arguments = _normalize_pgx_tool_arguments(tool_arguments)
                    if func_signature:
                        if not isinstance(tool_arguments, dict):
                            raise TypeError(f"Tool call 'arguments' must be a dict, got {type(tool_arguments)} (qa, i={i})")
                        for parameter_name, parameter_values in func_signature.items():
                            if parameter_name not in tool_arguments:
                                raise ValueError(f"PGx tool call missing required parameter '{parameter_name}' (qa, i={i})")
                            if (tool_arguments[parameter_name] not in parameter_values) and parameter_values != 'any':
                                raise ValueError(f"PGx tool call invalid parameter value '{parameter_name}'='{tool_arguments[parameter_name]}' (qa, i={i}). Valid values: {parameter_values}")
                    elif tool_arguments not in [{}, None]:
                        if not isinstance(tool_arguments, dict):
                            print(f"[Tool Orchestration][FAIL-SOFT] PGx tool '{tool_call_name}' got non-dict arguments {type(tool_arguments)} (qa, i={i}); coercing to empty.")
                            tool_call['arguments'] = {}
                        else:
                            unexpected_params = list(tool_arguments.keys())
                            if unexpected_params:
                                print(f"[Tool Orchestration][FAIL-SOFT] PGx tool '{tool_call_name}' does not require parameters, but found: {unexpected_params} (qa, i={i}); stripping.")
                                tool_call['arguments'] = {}
                    
                    if not valid_tool_call:
                        raise ValueError(f"Tool call validation failed for item {i}, tid={tid}")
                    
                    assert isinstance(doc_list,list) and isinstance(code_list,list) and isinstance(attempt_list,list)
                    
                    # PGx tool handling
                    if is_pgx_call:
                        if pgx_tool_executed:
                            raise ValueError(f"Duplicate PGx tool call detected (i={i}, tid={tid}). Only one PGx tool call is allowed per turn.")
                        
                        tool_name = tool_call['name']
                        mode = _get_mode(tool_name)
                        question = user_problem if user_problem else "Unknown question"
                        query_id = cur_index
                        ground_truth = answer
                        
                        print(f"[Actor Pipeline Selection] ========== Executing Selected Pipeline Mode ==========")
                        print(f"[Actor Pipeline Selection] Sample {i}: Pipeline Mode = {mode.upper()} (Tool: {tool_name})")
                        print(f"[Tool Orchestration] ========== PGx Tool Call in execute_predictions ==========")
                        print(f"[Tool Orchestration] Category: {cur_category}, Index: {i}, Query ID: {query_id}")
                        print(f"[Tool Orchestration] 🔧 Selected Pipeline Mode: {mode.upper()} (Tool: {tool_name})")
                        print(f"[Tool Orchestration] Question: {question[:200]}...")
                        print(f"[Tool Orchestration] Output dir: {item_output_dir}")
                        
                        call_tool_arguments.append({
                            'tool': tool_name,  # Use actual tool name
                            'mode': mode,
                            'question': question,
                            'id': str(i),  # Use item index as id so tool_responses[i] works correctly
                            'answer': ground_truth,
                            'cur_output_dir': item_output_dir,  # Per n_agent rollout: rollout_{i}/code_execution
                            'category': cur_category,
                            'tid': tid,
                            'cur_index': cur_index,
                            'level': cur_level,
                            'cur_tool_pricing': cur_tool_pricing,
                            'problem': user_problem,
                            'item_index': i,  # Store item index to mark as used after successful execution
                            'query_id': query_id,  # Keep original query_id for reference
                        })
                        pgx_tool_executed = True  # Mark that a PGx tool has been added to queue in this step
                        print(f"[Tool Orchestration] ✓ PGx tool call added to execution queue (category={cur_category}, tool={tool_name}, mode={mode})")
                        print(f"[Tool Orchestration] Note: Item {i} will be marked as having used a PGx tool only after successful execution")
            elif cur_category=='func_call':
                assert cur_transfer_dir
                
                # Check if this is a PGx tool call
                has_pgx_tool_call = False
                if isinstance(iter_tool_calls, list) and len(iter_tool_calls) > 0:
                    for tc in iter_tool_calls:
                        if isinstance(tc, dict):
                            tool_name = tc['name']
                            if _is_pgx_tool(tool_name):
                                has_pgx_tool_call = True
                                mode = _get_mode(tool_name)
                                print(f"[Tool Orchestration] Preparing PGx tool execution (func_call, i={i}): tool={tool_name}, mode={mode}")
                                break
                
                call_tool_arguments.append({
                    'tool': iter_tool_calls,
                    'cur_output_dir': item_output_dir,
                    'id': str(i),
                    'category': cur_category,
                    'input_messages': cur_input_messages,
                    'input_tools': cur_input_tools,
                    'tid': 0,
                    'transfer_path': cur_transfer_path,
                    'cur_model_mapping': cur_model_mapping,
                    'cur_tool_pricing': cur_tool_pricing,
                    'tokenizer': self.tokenizer,
                    'cur_index': cur_index,
                    'level': cur_level,
                    'vllm_model_configs': vllm_model_configs,
                    'topk_doc': topk_doc,
                    'problem': user_problems[i] if i < len(user_problems) else None,
                    'answer': answers[i] if i < len(answers) else None,
                })
            print(f"[Tool Orchestration] Item {i}: Prepared {len(call_tool_arguments)} tool call argument(s) for execution")
            all_call_tool_arguments.append(call_tool_arguments)

        tool_responses = {}

        all_cache_arguments = []
        for call_tool_arguments in all_call_tool_arguments:
            cache_arguments = []
            for iter_a in call_tool_arguments:
                tmp_arg = {}
                for k,v in iter_a.items():
                    if not k in ['semaphore','tokenizer']:
                        tmp_arg[k] = v
                cache_arguments.append(tmp_arg)
            all_cache_arguments.append(cache_arguments)
        if not os.path.isdir(os.path.join(cur_output_dir,'tool_return')):
            os.makedirs(os.path.join(cur_output_dir,'tool_return'),exist_ok=True)
        # Log summary before building tool_call_list
        total_pgx_calls = 0
        total_tool_calls = 0
        for call_tool_arguments in all_call_tool_arguments:
            total_tool_calls += len(call_tool_arguments)
            for arg in call_tool_arguments:
                tool_name = arg['tool']
                if isinstance(tool_name, list):
                    # func_call category: tool is a list of tool calls
                    for tc in tool_name:
                        if isinstance(tc, dict):
                            tc_name = tc['name']
                            if _is_pgx_tool(tc_name):
                                total_pgx_calls += 1
                elif isinstance(tool_name, dict):
                    # Should not happen, but handle it
                    if 'name' in tool_name:
                        if _is_pgx_tool(tool_name['name']):
                            total_pgx_calls += 1
                elif isinstance(tool_name, str):
                    # qa category: tool is a string (tool name)
                    if _is_pgx_tool(tool_name):
                        total_pgx_calls += 1
                else:
                    raise TypeError(f"Unexpected type for 'tool' field: {type(tool_name)}")
        
        print(f"[Tool Orchestration] Summary: {total_tool_calls} total tool call(s) prepared, {total_pgx_calls} PGx tool call(s)")
        
        tool_call_list = []
        for idx, iter_call_argument in enumerate(all_call_tool_arguments):
            if len(iter_call_argument)>0:
                call_id = iter_call_argument[0]['id']
                print(f"[Tool Orchestration] Adding tool call batch {idx+1}/{len(all_call_tool_arguments)} to execution queue (id={call_id}, {len(iter_call_argument)} tool call(s))")
                tool_call_list.append([call_tool_all,{
                    'id': call_id,
                    'all_call_arguments': iter_call_argument
                }])
        
        print(f"[Tool Orchestration] Tool execution summary: {total_pgx_calls} PGx call(s), {len(tool_call_list)} batch(es) to execute, {total_tool_calls} total tool call(s)")
        
        if total_pgx_calls > 0:
            print(f"[Tool Orchestration] ========== Executing {total_pgx_calls} PGx tool call(s) from {len(tool_call_list)} batch(es) ==========")
        
        if len(tool_call_list) == 0:
            print(f"[Tool Orchestration] WARNING: No tool calls to execute (all items produced malformed output). Skipping async execution.")
            tool_call_results = []
        else:
            print(f"[Tool Orchestration] Starting async execution of {len(tool_call_list)} tool call batch(es)...")
            tool_call_results = asyncio.run(run_all(tool_call_list))
        print(f"[Tool Orchestration] Async execution completed, received {len(tool_call_results)} result(s)")
        for return_contents in tool_call_results:
            result_id = return_contents['id']
            results = return_contents['all_tool_call_results']
            print(f"[Tool Orchestration] Processing tool call results for id={result_id}: {len(results)} result(s)")
            if len(results) > 0:
                for idx, result in enumerate(results):
                    tool_name_in_result = result['tool']
                    # Non-PGx tools should not exist - throw error immediately
                    if not _is_pgx_tool(tool_name_in_result):
                        raise ValueError(f"Non-PGx tool '{tool_name_in_result}' found in results. Only PGx tools are allowed.")
                    
                    # For PGx tools, these fields should always exist
                    cost = result['cost']
                    print(f"[Tool Orchestration]   Result {idx+1}/{len(results)}: tool={tool_name_in_result}, cost={cost}")
                    
                    # Process PGx tool result
                    if _is_pgx_tool(tool_name_in_result):
                        # result_id should be the item index (we set id=str(i) when adding to queue)
                        if not result_id.isdigit():
                            raise ValueError(f"Expected numeric result_id for PGx tool result, got: {result_id}")
                        item_idx = int(result_id)
                        # Verify item_index if present (should match)
                        if 'item_index' in result:
                            if result['item_index'] != item_idx:
                                raise ValueError(f"Item index mismatch: result_id={result_id}, result['item_index']={result['item_index']}")
                        if item_idx >= len(pgx_tool_used_per_item):
                            raise IndexError(f"Item index {item_idx} out of range for pgx_tool_used_per_item (len={len(pgx_tool_used_per_item)})")
                        response_text = result['response']
                        pgx_result = result['pgx_result']
                        # Only mark as used if we actually got a response (execution succeeded)
                        if not (response_text or pgx_result):
                            raise ValueError(f"PGx tool execution failed for item {item_idx}: no response or result returned")
                        pgx_tool_used_per_item[item_idx] = True
                        print(f"[Tool Orchestration] ✓ Item {item_idx} marked as having used a PGx tool (execution confirmed from result processing)")
            tool_responses[result_id] = results
        
        if total_pgx_calls > 0:
            print(f"[Tool Orchestration] ========== PGx tool execution completed ==========")
            print(f"[Tool Orchestration] Tool responses collected: {len(tool_responses)} item(s)")
        elif len(tool_call_list) > 0:
            raise ValueError("Tool calls were executed but no PGx calls were detected in results")

        cur_documents, cur_code, cur_attempts, dones, attempt_correct,costs, tool_summary, latencies,all_attempt_results,valid_answers_generated,used_llms = [], [], [], [], [], [], [], [], [], [], []
        # for i,(active,previous_cost) in enumerate(zip(active_mask,total_costs)):
        for i,active in enumerate(active_mask):
            if not active:
                cur_documents.append([])
                cur_code.append([])
                cur_attempts.append([])
                dones.append(1)
                attempt_correct.append(False)
                costs.append(0)
                tool_summary.append([])
                latencies.append(0)
                all_attempt_results.append([])
                valid_answers_generated.append(False)
                used_llms.append([])
            elif not str(i) in tool_responses:
                print(f"[Tool Orchestration] WARNING: Active item {i} has no tool_responses entry (model likely produced malformed output with no valid tool calls). Treating as failed attempt.")
                cur_documents.append([])
                cur_code.append([])
                cur_attempts.append([])
                dones.append(1)
                attempt_correct.append(False)
                costs.append(0)
                tool_summary.append([])
                latencies.append(0)
                all_attempt_results.append([])
                valid_answers_generated.append(False)
                used_llms.append([])
            else:
                assert isinstance(tool_responses[str(i)],list)
                iter_tool_summary = []
                iter_documents = []
                iter_code = []
                iter_attempt = []
                iter_attempt_correct = []
                iter_dones = []
                iter_costs = []
                iter_latency = []
                iter_valid_answer = False
                iter_used_llms = []
                for cur_response in tool_responses[str(i)]:
                    # Log PGx tool execution results
                    tool_name = cur_response['tool']
                    is_pgx_tool_result = _is_pgx_tool(tool_name) or (isinstance(cur_response, dict) and 'pgx_result' in cur_response)
                    if is_pgx_tool_result:
                        print(f"[Tool Orchestration] Processing PGx tool result (i={i}):")
                        if 'pgx_result' in cur_response:
                            pgx_result = cur_response['pgx_result']
                            run_id = pgx_result.get('run_id') or pgx_result.get('run_dir', '')  # run_dir is path string
                            if isinstance(run_id, str) and '/' in run_id:
                                run_id = run_id.split('/')[-1]  # use dir name
                            pipeline_mode = pgx_result.get('mode', 'unknown')
                            print(f"  🔧 Run: {run_id} | Pipeline mode: {pipeline_mode.upper()} (Tool: {tool_name})")
                            eval_result = pgx_result.get('eval_result', None)
                            if eval_result:
                                print(f"  Weighted score: {eval_result.get('weighted_score', 'N/A')}")
                            print(f"  Cost: {pgx_result['cost']}")
                            timing = pgx_result.get('timing') or {}
                            pipeline_t = {k: v for k, v in timing.items() if k != 'answer_evaluation'}
                            if pipeline_t:
                                print(f"  Pipeline timing (modules): {pipeline_t}")
                            if 'answer_evaluation' in timing:
                                print(f"  Answer evaluation: {timing['answer_evaluation']:.2f}s (not part of module4)")
                    
                    if not 'used_llm' in cur_response:
                        pass
                    else:
                        iter_used_llms.append(cur_response['used_llm'])
                    if 'model' in cur_response:
                        iter_tool_summary.append([cur_response['tool'],cur_response['model']])
                    elif 'query' in cur_response:
                        iter_tool_summary.append([cur_response['tool'],cur_response['query']])
                    else:
                        # tool_summary.append([cur_response['tool'],''])
                        iter_tool_summary.append([cur_response['tool'],''])
                    # Only PGx tools are supported
                    if _is_pgx_tool(tool_name):
                        # Handle PGx tool results (similar to answer tool)
                        print(f"[Tool Orchestration] Processing PGx tool result (i={i}):")
                        pgx_result = cur_response['pgx_result']
                        mode = cur_response['mode']
                        response_text = cur_response['response']
                        eval_result = pgx_result.get('eval_result', None)
                        
                        # Determine if answer is correct based on eval_result weighted_score
                        is_correct = False
                        if eval_result is not None:
                            weighted_score = eval_result.get('weighted_score', 0.0)
                            is_correct = weighted_score >= 0.6
                        
                        print(f"  Mode: {mode}, Response length: {len(response_text)}, Weighted score: {eval_result.get('weighted_score', 0.0) if eval_result else 'N/A'}")
                        
                        # Mark this item as having successfully executed a PGx tool
                        # Only mark after we confirm the tool was actually executed (has response)
                        if not (response_text or pgx_result):
                            raise ValueError(f"PGx tool execution failed for item {i}: no response or result returned")
                        if i >= len(pgx_tool_used_per_item):
                            raise IndexError(f"Item index {i} out of range for pgx_tool_used_per_item (len={len(pgx_tool_used_per_item)})")
                        pgx_tool_used_per_item[i] = True
                        print(f"[Tool Orchestration] ✓ Item {i} marked as having used a PGx tool (execution confirmed)")
                        
                        iter_attempt.append({
                            'model': f'pgx_{mode}',
                            'answer': response_text,
                            'response': response_text,
                            'problem': user_problems[i] if i < len(user_problems) else '',
                            'ground_truth': answers[i] if i < len(answers) else '',
                            'correct': is_correct,
                            'pgx_result': pgx_result,
                            'mode': mode,
                        })
                        iter_dones.append(1)
                        if is_correct:
                            iter_attempt_correct.append(True)
                        iter_valid_answer = True
                        print(f"[Tool Orchestration] ✓ PGx result processed: done=1, correct={is_correct}, valid_answer=True")
                        iter_latency.append(cur_response['latency'])
                        iter_costs.append(cur_response['cost'])
                        break
                    # All non-PGx tools removed - only PGx tools are supported
                    # If we reach here, it means a non-PGx tool was executed, which should not happen
                    raise ValueError(f"Unexpected non-PGx tool '{tool_name}' executed. Only PGx tools are allowed.")
                cur_documents.append(iter_documents)
                cur_code.append(iter_code)
                cur_attempts.append(iter_attempt)
                if len(iter_dones)>0:
                    dones.append(iter_dones[-1])
                else:
                    dones.append(0)
                if len(iter_attempt_correct)>0:
                    attempt_correct.append(iter_attempt_correct[-1])
                else:
                    attempt_correct.append(False)
                all_attempt_results.append(iter_attempt_correct)
                valid_answers_generated.append(iter_valid_answer)
                costs.append(sum(iter_costs))
                latencies.append(sum(iter_latency))
                tool_summary.append(iter_tool_summary)
                used_llms.append(iter_used_llms)

        if cur_output_dir:
            try:
                global_step_dir = os.path.dirname(cur_output_dir)
                actor_decision_dir = os.path.join(global_step_dir, "actor_decision")
                os.makedirs(actor_decision_dir, exist_ok=True)
                for i in range(len(predictions)):
                    q_index = all_indices[i] if i < len(all_indices) else i
                    decision_record = {
                        "question_index": q_index,
                        "category": all_categories[i] if i < len(all_categories) else "",
                        "level": all_levels[i] if i < len(all_levels) else "",
                        "actor_reasoning": contents[i] if i < len(contents) else "",
                        "raw_prediction": predictions[i] if i < len(predictions) else "",
                        "parsed_tool_calls": cur_tool_calls[i] if i < len(cur_tool_calls) else [],
                        "format_correct": format_correctness[i] if i < len(format_correctness) else False,
                    }
                    safe_qid = str(q_index).replace("/", "_").replace("\\", "_")
                    fname = f"question_{safe_qid}_rollout_{i}.json"
                    with open(os.path.join(actor_decision_dir, fname), "w", encoding="utf-8") as f:
                        json.dump(decision_record, f, ensure_ascii=False, indent=2)
                print(f"[Actor Decision] Saved {len(predictions)} actor decision records to {actor_decision_dir}")
            except Exception as e:
                print(f"[Actor Decision] Warning: failed to save actor decisions: {e}")

        return cur_documents, cur_code, cur_attempts, dones, attempt_correct,costs,tool_summary,format_correctness,latencies,all_attempt_results,valid_answers_generated,used_llms,pgx_tool_used_per_item

    def postprocess_predictions(self, predictions,all_categories):
        thoughts = []
        tool_calls = []
        formats = []
        assert len(all_categories)==len(predictions)
        
        # Create local reference to module-level functions to avoid UnboundLocalError
        _is_pgx_tool = is_pgx_tool
        _get_mode = get_pgx_mode_from_tool
        _is_pgx_category = is_pgx_category
        
        # Track PGx tool calls for logging
        pgx_tool_call_count = 0
                
        def _extract_prefix_before_json(text: str) -> str:
            """
            Store everything before the first JSON object ("{...}") verbatim.
            If JSON starts at position 0 (or JSON is missing), return empty string.
            """
            if not isinstance(text, str):
                return ""
            j = text.find("{")
            return text[:j] if j > 0 else ""

        for idx, (prediction, category) in enumerate(zip(predictions, all_categories)):
            cur_thought = ''
            cur_all_tool_calls = []
            format_correct = False
            
            # Check if this is a PGx category (not 'qa' or 'func_call')
            category_is_pgx = _is_pgx_category(category)
            
            if isinstance(prediction, str): # for llm output
                # Capture reasoning even if parser later succeeds (for logging/persistence).
                cur_thought = _extract_prefix_before_json(prediction)
                if category=='qa':
                    components = prediction.split('<tool_call>')
                    added_tools = set()
                    added_pgx_tools = set()
                    
                    for c in components:
                        components1 = c.split('</tool_call>')
                        for c1 in components1:
                            tmp_tool_call = json.loads(c1)
                            
                            # Normalize tool call format: convert 'parameters' to 'arguments' if needed
                            if 'parameters' in tmp_tool_call and 'arguments' not in tmp_tool_call:
                                tmp_tool_call['arguments'] = tmp_tool_call.pop('parameters')
                                print(f"[Tool Orchestration] Normalized tool call format: converted 'parameters' to 'arguments' (qa category, idx={idx})")
                            elif 'parameters' in tmp_tool_call and 'arguments' in tmp_tool_call:
                                # Both exist - prefer 'arguments', but log the conflict
                                print(f"[Tool Orchestration] Warning: Tool call has both 'parameters' and 'arguments', using 'arguments' (qa category, idx={idx})")
                                tmp_tool_call.pop('parameters', None)
                            
                            # Validate structure after normalization
                            tool_call_keys = set(tmp_tool_call.keys())
                            if 'name' not in tool_call_keys:
                                raise ValueError(f"Tool call missing required 'name' field (qa category, idx={idx}). Got keys: {tool_call_keys}")
                            if tool_call_keys - {'name', 'arguments'}:
                                unexpected_keys = tool_call_keys - {'name', 'arguments'}
                                raise ValueError(f"Tool call has unexpected keys: {unexpected_keys} (qa category, idx={idx}). Expected only 'name' and 'arguments'.")
                            
                            # Ensure 'arguments' exists (default to empty dict if missing)
                            if 'arguments' not in tmp_tool_call:
                                tmp_tool_call['arguments'] = {}
                            
                            # Validate tool name is in ALL_TOOLS (tool_a/tool_b/tool_c)
                            _name = tmp_tool_call['name']
                            if _name not in ALL_TOOLS:
                                raise ValueError(f"Tool '{_name}' not in ALL_TOOLS (postprocess, idx={idx}). Allowed: {list(ALL_TOOLS.keys())}")
                            
                            if not _is_pgx_tool(tmp_tool_call['name']):
                                print(f"[Tool Orchestration] ERROR: Tool '{tmp_tool_call['name']}' is not allowed in PGx training. Only {VALID_PGX_TOOL_NAMES} are allowed. (category={category}, idx={idx})")
                                continue
                            
                            # Log PGx tool call detection
                            if _is_pgx_tool(tmp_tool_call['name']):
                                tool_name = tmp_tool_call['name']
                                mode = _get_mode(tool_name)
                                print(f"[Actor Pipeline Selection] Sample {idx}: Detected Pipeline Mode = {mode.upper()} (Tool: {tool_name}, Category: qa)")
                                print(f"[Tool Orchestration] Detected PGx tool call (qa category, idx={idx}): tool={tool_name}, mode={mode}")
                                
                                # Ensure only one tool is selected per turn
                                if len(added_pgx_tools) > 0:
                                    first_tool = list(added_pgx_tools)[0]
                                    first_mode = _get_mode(first_tool)
                                    print(f"[Tool Orchestration] WARNING: Multiple PGx tool calls detected (idx={idx}). Only the first one (tool={first_tool}, mode={first_mode}) will be used. Ignoring tool={tool_name} (mode={mode}).")
                                    continue
                                
                                added_pgx_tools.add(tool_name)
                                pgx_tool_call_count += 1
                            
                            # General duplicate tool check (for non-PGx tools, though they shouldn't exist)
                            if tmp_tool_call['name'] in added_tools:
                                continue
                            added_tools.add(tmp_tool_call['name'])
                            # Validate tool signature (PGx tools have no parameters)
                            func_signature = ALL_TOOLS.get(tmp_tool_call['name'], {})
                            if func_signature != {}:
                                # Tool has parameters - validate them
                                for parameter_name, parameter_values in func_signature.items():
                                    assert tmp_tool_call["arguments"][parameter_name] in parameter_values or parameter_values=='any'
                            else:
                                # PGx tools have no parameters - allow empty arguments or {"parameters": {}}
                                tool_args = _normalize_pgx_tool_arguments(tmp_tool_call.get('arguments', {}))
                                if tool_args not in [{}, None]:
                                    unexpected_params = list(tool_args.keys()) if isinstance(tool_args, dict) else []
                                    if unexpected_params:
                                        print(f"[Tool Orchestration][FAIL-SOFT] PGx tool '{tmp_tool_call['name']}' does not require parameters, but found: {unexpected_params} (postprocess qa, idx={idx}); stripping.")
                                        tmp_tool_call['arguments'] = {}
                            
                            cur_all_tool_calls.append(tmp_tool_call)
                    if len(cur_all_tool_calls)>0:
                        format_correct = True
                elif category=='func_call' or category_is_pgx:
                    # For func_call or PGx categories, handle tool calling similarly
                    if category_is_pgx:
                        print(f"[Tool Orchestration] Processing PGx category '{category}' (idx={idx}): attempting to parse tool calls")
                        print(f"  Prediction preview: {prediction[:300]}...")
                    
                    # Preserve thought captured above (everything before the first '{').
                    
                    # Track PGx tools to ensure only one is selected
                    added_pgx_tools = set()
                    added_tools = set()

                    fallback_fired = False
                    def _pgx_fallback(reason: str) -> None:
                        """
                        Fail-soft for PGx formatting issues. The earlier rollout-level
                        retry (run_llm_loop's PGX_MAX_PARSE_RETRY) already gave the model
                        up to 3 attempts to produce a valid output. If we still can't parse,
                        default to the fallback tool (self.fallback_tool) and mark
                        format_correct=False so the reward layer drops the bad format.
                        """
                        nonlocal fallback_fired
                        fallback_fired = True
                        should_print = True
                        if torch.distributed.is_initialized():
                            should_print = (torch.distributed.get_rank() == 0)
                        if should_print:
                            print(
                                f"[Tool Orchestration][FAIL-SOFT] PGx output format not parseable (idx={idx}, category='{category}'). "
                                f"Reason: {reason}. Defaulting to {self.fallback_tool}."
                            )
                            print(f"[Tool Orchestration][FAIL-SOFT] FULL PREDICTION (idx={idx}, len={len(prediction)} chars) BEGIN >>>")
                            print(prediction)
                            print(f"<<< END FULL PREDICTION (idx={idx})")
                        # Append the fallback tool so downstream rollout has something to execute.
                        # Under STRICT_TOOL_PARSE the rollout made no decision, so substituting one would
                        # credit the model with a choice it never made: leave the list empty instead. The
                        # reward path already handles that (tool not in active_tools -> score 0).
                        if os.environ.get("STRICT_TOOL_PARSE", "0") == "1" and category_is_pgx:
                            return
                        if not any(_is_pgx_tool(tc.get('name')) for tc in cur_all_tool_calls if isinstance(tc, dict)):
                            cur_all_tool_calls.append({"name": self.fallback_tool, "arguments": {}})
                        # The caller still expects to see something; format_correct is set to
                        # False at the call site (we don't return True here, and the existing
                        # code only flips format_correct=True on the success branches).
                    
                    # Try parsing <tool_call> tags (text format)
                    if '<tool_call>' in prediction:
                        print(f"[Tool Orchestration] Found <tool_call> tag in prediction (category={category}, idx={idx})")
                        import re
                        # Use regex to properly extract content between <tool_call> and </tool_call> tags
                        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                        matches = re.finditer(tool_call_pattern, prediction, re.DOTALL)
                        for match in matches:
                            tag_content = match.group(1).strip()
                            # Try to extract JSON from the content (may contain other text/tags like <think>)
                            # Find the first { and match it with its closing }
                            json_start = tag_content.find('{')
                            if json_start == -1:
                                if category_is_pgx:
                                    _pgx_fallback("No JSON object found inside <tool_call> tag")
                                    break
                                raise ValueError(f"No JSON object (starting with '{{') found in <tool_call> tag content (category={category}, idx={idx}). Content preview: {tag_content[:200]}...")
                            
                            # Extract JSON by matching braces
                            brace_count = 0
                            json_end = json_start
                            for i, char in enumerate(tag_content[json_start:], start=json_start):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_end = i + 1
                                        break
                            
                            if brace_count != 0:
                                if category_is_pgx:
                                    _pgx_fallback("Unmatched braces in <tool_call> JSON")
                                    break
                                raise ValueError(f"Unmatched braces in JSON object (category={category}, idx={idx}). Content preview: {tag_content[json_start:json_start+200]}...")
                            
                            json_str = tag_content[json_start:json_end]
                            tmp_tool_call = json.loads(json_str)
                            
                            if not isinstance(tmp_tool_call, dict):
                                if category_is_pgx:
                                    _pgx_fallback(f"Parsed <tool_call> JSON is not a dict: {type(tmp_tool_call)}")
                                    break
                                raise TypeError(f"Parsed tool_call is not a dict: {type(tmp_tool_call)} (category={category}, idx={idx})")
                            
                            tool_name = tmp_tool_call['name']
                            
                            if _is_pgx_tool(tool_name):
                                mode = _get_mode(tool_name)
                                print(f"[Actor Pipeline Selection] Sample {idx}: Detected Pipeline Mode = {mode.upper()} (Tool: {tool_name}, Category: {category})")
                                print(f"[Tool Orchestration] ✓ Detected PGx tool call from <tool_call> tag (category={category}, idx={idx}): tool={tool_name}, mode={mode}")
                                
                                # Ensure only one tool is selected per turn
                                if len(added_pgx_tools) > 0:
                                    first_tool = list(added_pgx_tools)[0]
                                    first_mode = _get_mode(first_tool)
                                    if category_is_pgx:
                                        _pgx_fallback("Multiple PGx tool calls detected in <tool_call> tags")
                                        break
                                    raise ValueError(f"Multiple PGx tool calls detected (idx={idx}). Only one PGx tool call is allowed per turn. Already added: {first_tool} (mode={first_mode}), trying to add: {tool_name} (mode={mode}).")
                                
                                added_pgx_tools.add(tool_name)
                                pgx_tool_call_count += 1
                            
                            # General duplicate tool check
                            if tool_name in added_tools:
                                if category_is_pgx:
                                    _pgx_fallback(f"Duplicate tool call '{tool_name}' detected")
                                    break
                                raise ValueError(f"Duplicate tool call '{tool_name}' detected (category={category}, idx={idx})")
                            added_tools.add(tool_name)
                            cur_all_tool_calls.append(tmp_tool_call)
                        if len(cur_all_tool_calls)>0 and not fallback_fired:
                            format_correct = True
                    # Try parsing JSON tool calls in text (alternative format)
                    elif any(tool_name in prediction.lower() for tool_name in VALID_PGX_TOOL_NAMES) or '"name":' in prediction:
                        print(f"[Tool Orchestration] Attempting to extract tool call from JSON-like text (category={category}, idx={idx})")
                        import re
                        # Look for JSON-like structures matching any PGx tool
                        tool_patterns = '|'.join(VALID_PGX_TOOL_NAMES)
                        # STRICT_TOOL_PARSE: the decision is the JSON AFTER the last </think>. A name the
                        # model wrote inside its own reasoning is not a decision, and a rollout that never
                        # closed </think> was truncated and made no decision at all.
                        _strict = os.environ.get("STRICT_TOOL_PARSE", "0") == "1" and category_is_pgx
                        _search_text = prediction
                        if _strict:
                            if "</think>" not in prediction:
                                _pgx_fallback("STRICT_TOOL_PARSE: no </think> — thinking was truncated, no final answer")
                                _search_text = ""
                            else:
                                _search_text = prediction.split("</think>")[-1]
                        # Extract tool names rather than trying to parse full JSON (more robust).
                        # TWO_STAGE_ROUTER: the L1 two-stage prompt answers {"retrieve": <bool>,
                        # "tool": "<name>"}; accept "tool" as the key alongside "name". A
                        # retrieve flag that contradicts the named tool is a parse failure, not
                        # something to guess at.
                        _key = '(?:name|tool)' if os.environ.get("TWO_STAGE_ROUTER", "0") == "1" else 'name'
                        matches = re.findall(rf'"{_key}"\s*:\s*"({tool_patterns})"', _search_text)
                        matches = list(dict.fromkeys(matches))  # de-dup preserving order
                        if os.environ.get("TWO_STAGE_ROUTER", "0") == "1" and len(matches) == 1:
                            # The flag is either a bool (needs_reference style) or the words
                            # direct / reference. Both spellings of "no lookup" are accepted.
                            _fk = os.environ.get("TWO_STAGE_FLAG_KEY", "mode")
                            _flag = re.findall(rf'"{_fk}"\s*:\s*(?:"(direct|reference)"|(true|false))',
                                               _search_text, flags=re.I)
                            _nort = os.environ.get("NO_RETRIEVAL_TOOL", "tool_b")
                            _direct = None
                            if len(_flag) == 1:
                                _w, _b = _flag[0]
                                _direct = (_w.lower() == "direct") if _w else (_b.lower() == "false")
                            if _direct is None or _direct != (matches[0] == _nort):
                                _pgx_fallback(f"TWO_STAGE_ROUTER: {_fk} flag {_flag} "
                                                 f"disagrees with tool {matches[0]}")
                                matches = []
                        if len(matches) != 1 and category_is_pgx:
                            _pgx_fallback("Could not uniquely identify a single PGx tool name in JSON-like text")
                            matches = []
                        for tool_name in matches:
                            tmp_tool_call = {"name": tool_name, "arguments": {}}
                            if not isinstance(tmp_tool_call, dict):
                                raise TypeError(f"Parsed tool_call is not a dict: {type(tmp_tool_call)} (category={category}, idx={idx})")
                            
                            tool_name = tmp_tool_call['name']
                            if _is_pgx_tool(tool_name):
                                mode = _get_mode(tool_name)
                                print(f"[Actor Pipeline Selection] Sample {idx}: Extracted Pipeline Mode = {mode.upper()} (Tool: {tool_name}, Category: {category})")
                                print(f"[Tool Orchestration] ✓ Extracted PGx tool call from JSON-like text (category={category}, idx={idx}): tool={tool_name}, mode={mode}")
                                
                                # Ensure only one tool is selected per turn
                                if len(added_pgx_tools) > 0:
                                    first_tool = list(added_pgx_tools)[0]
                                    first_mode = _get_mode(first_tool)
                                    if category_is_pgx:
                                        _pgx_fallback("Multiple PGx tool calls detected in JSON-like text")
                                        break
                                    raise ValueError(f"Multiple PGx tool calls detected (idx={idx}). Only one PGx tool call is allowed per turn. Already added: {first_tool} (mode={first_mode}), trying to add: {tool_name} (mode={mode}).")
                                
                                added_pgx_tools.add(tool_name)
                                pgx_tool_call_count += 1
                                cur_all_tool_calls.append(tmp_tool_call)
                                format_correct = True
                    elif '<message>' in prediction:
                        cur_all_tool_calls.append(prediction.split('<message>')[-1].split('</message>')[0])
                    else:
                        # No tool call found in text format
                        if category_is_pgx:
                            _pgx_fallback("No <tool_call> tags or JSON tool call detected")
            
            # Choke point: a parsed PGx tool not in the active set must NOT be executed and
            # must not be trained. Route it to the fallback tool and mark the format incorrect.
            _nonactive = [tc.get("name") for tc in cur_all_tool_calls
                          if isinstance(tc, dict) and _is_pgx_tool(tc.get("name"))
                          and tc.get("name") not in self.active_tools]
            if _nonactive:
                print(f"[Tool Orchestration][CHOKE] idx={idx}: non-active tool(s) {_nonactive} "
                      f"not in {self.active_tools} -> {self.fallback_tool}, format_correct=False")
                cur_all_tool_calls = [{"name": self.fallback_tool, "arguments": {}}]
                format_correct = False

            thoughts.append(cur_thought)
            tool_calls.append(cur_all_tool_calls)
            formats.append(format_correct)
        
        # Summary log
        if pgx_tool_call_count > 0:
            print(f"[Tool Orchestration] Summary: Found {pgx_tool_call_count} PGx tool call(s) in {len(predictions)} predictions")
            
        return tool_calls, thoughts, formats

    def batch_search(self, queries: List[str] = None):
        if len(queries)>0:
            results = self._batch_search(queries)['result']
            return results
        else:
            return []

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            format_reference += f"Doc {idx+1}: {content}\n"

        return format_reference

    def serialize_documents(self, documents):
        documents_str = ''
        for idx, doc in enumerate(documents):
            documents_str += f"Doc {idx+1}: {doc}\n\n"
        return documents_str
