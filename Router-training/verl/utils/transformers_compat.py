# SPDX-License-Identifier: Apache-2.0
"""Hugging Face ``transformers`` API differences across versions.

``AutoModelForVision2Seq`` exists in some 4.4x lines but is not exported in all
5.x installs; verl still branches on it for VLMs. Text-only training (e.g.
Qwen3) only needs :class:`transformers.AutoModelForCausalLM`.
"""
from __future__ import annotations

from typing import Any, Type

try:
    from transformers import AutoModelForVision2Seq as AutoModelForVision2Seq  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore[misc, assignment]


def is_vision2seq_config(config: Any) -> bool:
    if AutoModelForVision2Seq is None:
        return False
    return type(config) in AutoModelForVision2Seq._model_mapping.keys()


def auto_model_class_for_actor_config(config: Any):
    """Return Auto class for FSDP actor/ref loading (causal LM vs vision-seq)."""
    from transformers import AutoModelForCausalLM

    if is_vision2seq_config(config):
        return AutoModelForVision2Seq
    return AutoModelForCausalLM


def auto_class_for_conditional_generation_checkpoint() -> Type:
    """For HF checkpoints whose architecture is *ForConditionalGeneration*."""
    if AutoModelForVision2Seq is not None:
        return AutoModelForVision2Seq
    from transformers import AutoModel

    return AutoModel
