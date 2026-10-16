"""Merge a trained router checkpoint's FSDP shards into a Hugging Face model folder.

  python merge_checkpoint.py checkpoints/router_fg_fold0/global_step_27/actor

Writes <actor>/_merged (model.safetensors plus the config and tokenizer files), which a vLLM
server can load, e.g. `vllm serve <actor>/_merged --served-model-name router-fg`. A checkpoint
without the .save_complete marker (a save that did not finish) is refused.
"""
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def merge_fsdp_shards(shards):
    """Merge FSDP sharded state dicts by concatenating along dim 0."""
    if len(shards) == 1:
        return shards[0]
    merged = OrderedDict()
    for key in shards[0].keys():
        tensors = [s[key] for s in shards if key in s]
        if len(tensors) == 1:
            merged[key] = tensors[0]
        elif tensors[0].shape == tensors[1].shape and tensors[0].dim() >= 1:
            merged[key] = torch.cat(tensors, dim=0)
        else:
            merged[key] = tensors[0]
    return merged


def merge(ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not (ckpt_path / ".save_complete").exists():
        raise FileNotFoundError(f"{ckpt_path}/.save_complete missing: the save may not have finished")
    shard_files = sorted(ckpt_path.glob("model_world_size_*_rank_*.pt")) or \
        sorted(ckpt_path.glob("model_world_size_*_rank_*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"no model shard files in {ckpt_path}")
    if not (ckpt_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in {ckpt_path}")
    out = ckpt_path / "_merged"
    if not (out / "model.safetensors").exists():
        shards = []
        for sf in shard_files:
            if sf.suffix == ".safetensors":
                from safetensors.torch import load_file
                sd = load_file(str(sf), device="cpu")
            else:
                sd = torch.load(str(sf), map_location="cpu", weights_only=False)
            shards.append({k: v._local_tensor if hasattr(v, "_local_tensor") else v for k, v in sd.items()})
        merged = merge_fsdp_shards(shards)
        out.mkdir(exist_ok=True)
        from safetensors.torch import save_file
        save_file(merged, str(out / "model.safetensors"))
        print(f"merged {len(shard_files)} shard(s), {len(merged)} tensors -> {out / 'model.safetensors'}")
    for f in ckpt_path.iterdir():
        if f.name in ("_merged", ".save_complete") or f.name.startswith(("model_world_size", "extra_state")):
            continue
        if f.is_file():
            shutil.copy2(f, out / f.name)
    print(f"Hugging Face folder: {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    merge(sys.argv[1])
