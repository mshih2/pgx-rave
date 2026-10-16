# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CoVaPh Reward Functions for ToolOrchestra GRPO Training
Computes rewards based on:
1. Outcome reward: Answer quality (using CoVaPh answer_reward with multi-dimensional evaluation)
2. Efficiency reward: Cost/time savings from using simpler pipelines
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add CoVaPh modularized_src to path


# Pipeline mode efficiency scores (higher = more efficient)
MODE_EFFICIENCY = {
    "full": 0.0,           # Baseline, no efficiency bonus
    "skip_retrieve": 0.3,  # Moderate efficiency
    "skip_cpic_api": 0.2,  # Some efficiency
    "module4_only": 0.5,   # Most efficient
}



def compute_covaph_reward(
    response: str,
    tool_result: Dict[str, Any],
    ground_truth: str,
    question: str,
    reward_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the total reward for a CoVaPh tool selection (fallback only; training uses pref_vec reward from generation_quick3).
    
    Args:
        response: The orchestrator's response (containing tool selection)
        tool_result: Result from executing the CoVaPh tool
        ground_truth: Ground truth answer
        question: Original question
        reward_weights: Optional weights for reward dimensions (Accuracy, Relevance, Completeness, Clarity)
        
    Returns:
        Tuple of (total_reward, reward_breakdown)
    """
    reward_breakdown = {
        "outcome": 0.0,
        "efficiency": 0.0,
        "total": 0.0,
        "eval_details": None,
    }
    
    # 1. Extract mode from tool result
    mode = tool_result.get("mode", "full")
    
    # 3. Outcome reward: Based on answer quality using reward evaluation
    eval_result = tool_result.get("eval_result", None)
    if eval_result is not None:
        # Use pre-computed evaluation result from tool_result
        outcome_score = eval_result.get("weighted_score", 0.0)
        reward_breakdown["eval_details"] = eval_result
    else:
        # Evaluate using reward evaluator
        # Prefer response_json (short version) over response (long version)
        # Check if response_json or final_answer_json has valid content
        response_json = tool_result.get("response_json") or tool_result.get("final_answer_json")
        if (isinstance(response_json, dict) and 
            (response_json.get("title") or response_json.get("details"))):
            answer_for_eval = response_json
        else:
            answer_for_eval = tool_result.get("response", "")
        outcome_score, eval_details = compute_outcome_score(
            answer_for_eval,
            ground_truth,
            question,
            reward_weights=reward_weights
        )
        reward_breakdown["eval_details"] = eval_details
    reward_breakdown["outcome"] = outcome_score
    
    # 2. Efficiency reward: Based on mode selection
    # Only give efficiency bonus if the answer is correct
    if outcome_score > 0.5:
        efficiency_score = MODE_EFFICIENCY.get(mode, 0.0)
    else:
        # Penalize efficiency if wrong answer with efficient mode
        # (should have used more thorough mode)
        efficiency_score = -MODE_EFFICIENCY.get(mode, 0.0) * 0.5
    reward_breakdown["efficiency"] = efficiency_score
    
    # 3. Calculate total reward (fixed weights for fallback only)
    total_reward = 0.6 * outcome_score + 0.3 * efficiency_score
    reward_breakdown["total"] = total_reward
    
    return total_reward, reward_breakdown


def compute_outcome_score(
    predicted: str,
    ground_truth: str,
    question: str,
    reward_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute outcome score based on answer quality using reward evaluation.
    Uses CoVaPh's AnswerRewardEvaluator for multi-dimensional evaluation.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        question: Original question
        reward_weights: Optional weights for reward dimensions (Accuracy, Relevance, Completeness, Clarity)
        
    Returns:
        Tuple of (outcome_score (0.0 to 1.0), evaluation_details)
    """
    if not predicted or not ground_truth:
        return 0.0, {
            "weighted_score": 0.0,
            "raw_scores": {
                "Accuracy": 1.0,
                "Relevance": 1.0,
                "Completeness": 1.0,
                "Clarity": 1.0,
            }
        }
    

    # The answer evaluator this fallback used is not part of this repository; rewards come from
    # the rollout loop (REWARD_ADV_MODE=cache_baseline with live_reward.py).
    raise RuntimeError("no reward was computed by the rollout loop for this sample; "
                       "train with REWARD_ADV_MODE=cache_baseline")
        


class CoVaPhRewardManager:
    """
    Reward manager for CoVaPh GRPO training.
    Integrates with ToolOrchestra's training loop.
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int = 1,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the reward manager.
        
        Args:
            tokenizer: Tokenizer for processing responses
            num_examine: Number of examples to examine for debugging
            reward_weights: Optional weights for reward dimensions (Accuracy, Relevance, Completeness, Clarity)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_weights = reward_weights
    
    def __call__(self, data, global_step: int):
        """
        Compute rewards for a batch of data.
        
        Args:
            data: Batch data from training loop
            global_step: Current training step
            
        Returns:
            Reward tensor
        """
        # Priority 1: Use pre-computed reward from generation_quick3.py (preference vector reward with AnswerRewardEvaluator's weighted_score)
        # This is the preferred method as it uses AnswerRewardEvaluator's weighted_score via preference vector calculation
        # Priority 2: Use rm_scores if available (neural network reward model, but we don't use this anymore)
        # Priority 3: Compute reward from tool result (fallback)
        
        # Initialize reward tensor
        reward_tensor = torch.zeros_like(
            data.batch['responses'], 
            dtype=torch.float32
        )
        
        # Get attention mask to find last valid token
        attention_mask = data.batch.get('attention_mask', None)
        
        # Check if we should use rm_scores (neural network reward model)
        # Note: We prefer non_tensor_batch['reward'] over rm_scores because it uses AnswerRewardEvaluator
        use_rm_scores = False
        if 'rm_scores' in data.batch.keys():
            # Only use rm_scores if no reward is available in non_tensor_batch
            has_reward_in_non_tensor = any('reward' in data[i].non_tensor_batch for i in range(len(data)))
            if not has_reward_in_non_tensor:
                use_rm_scores = True
                print(f"[Reward Manager] Using rm_scores from neural network reward model (no reward found in non_tensor_batch)")
        
        if use_rm_scores:
            return data.batch['rm_scores']
        
        # Compute rewards for each item
        for i in range(len(data)):
            data_item = data[i]
            
            # Priority 1: Get pre-computed reward from generation_quick3.py (preference vector with AnswerRewardEvaluator)
            if 'reward' in data_item.non_tensor_batch:
                reward = data_item.non_tensor_batch['reward']
            else:
                # Priority 2: Compute reward from tool result (fallback)
                response = data_item.non_tensor_batch.get('response', '')
                tool_result = data_item.non_tensor_batch.get('tool_result', {})
                ground_truth = data_item.non_tensor_batch.get('answer', '')
                question = data_item.non_tensor_batch.get('question', '')
                
                print(f"[Reward Manager] ⚠️  Warning: No pre-computed reward found for item {i}, computing from tool_result (fallback)")
                reward, _ = compute_covaph_reward(
                    response=response,
                    tool_result=tool_result,
                    ground_truth=ground_truth,
                    question=question,
                    reward_weights=self.reward_weights,
                )
            
            # Assign reward to the last valid token position.
            # In GRPO the sequence-level reward must land on the final
            # non-padding token so that the advantage is propagated
            # correctly through the response mask.
            # reward_tensor has shape (bsz, response_length); attention_mask
            # covers prompt+response, so slice the response portion first.
            if attention_mask is not None:
                response_length = reward_tensor.shape[1]
                response_mask_i = attention_mask[i, -response_length:]
                last_valid_idx = int(response_mask_i.sum().item()) - 1
                last_valid_idx = max(last_valid_idx, 0)
                reward_tensor[i, last_valid_idx] = reward
            else:
                reward_tensor[i, 0] = reward
        
        return reward_tensor


def get_reward_fn(
    tokenizer,
    reward_weights: Optional[Dict[str, float]] = None,
) -> CoVaPhRewardManager:
    """
    Factory function to create a CoVaPh reward manager.
    
    Args:
        tokenizer: Tokenizer for processing
        reward_weights: Optional weights for reward dimensions (Accuracy, Relevance, Completeness, Clarity)
        
    Returns:
        CoVaPhRewardManager instance
    """
    return CoVaPhRewardManager(
        tokenizer=tokenizer,
        reward_weights=reward_weights,
    )


if __name__ == "__main__":
    # Test the reward functions
    test_response = """<think>
    The question asks about CYP2D6 poor metabolizer dosing for codeine.
    This requires specific allele information and CPIC recommendations.
    I should use the full pipeline to get accurate dosing information.
    </think>
    
    Tool call: covaph_query with mode="full"
    """
    
    test_tool_result = {
        "mode": "full",
        "response": "For CYP2D6 poor metabolizers, codeine should be avoided.",
    }
    
    test_ground_truth = "Codeine should be avoided in CYP2D6 poor metabolizers."
    test_question = "What is the recommended dosing for codeine in a CYP2D6 poor metabolizer?"
    
    reward, breakdown = compute_covaph_reward(
        response=test_response,
        tool_result=test_tool_result,
        ground_truth=test_ground_truth,
        question=test_question,
    )
    
    print(f"Total reward: {reward:.4f}")
    print(f"Reward breakdown: {json.dumps(breakdown, indent=2)}")

