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
Contains a resharding manager that binds weights from FSDP zero3 to XPerfGPT
"""

import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.device import get_torch_device
from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group

from .base import BaseShardingManager


class FSDPUlyssesShardingManager(BaseShardingManager):
    """
    Sharding manager to support data resharding when using FSDP + Ulysses
    """

    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        self.device_mesh = device_mesh
        self.seed_offset = 12345

    def __enter__(self):
        if self.device_mesh is not None:
            # We have a global SP group
            # so we have to change to use model-specific sp group
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())
            # TODO: check how to set seed for each model

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        if self.device_mesh is not None:
            # revert to previous sp group
            set_ulysses_sequence_parallel_group(self.prev_sp_group)
            # TODO: check how to set seed for each model

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        Replicate one canonical batch across the Ulysses SP group.

        ``all_gather_data_proto`` (concat on batch dim-0) is only valid when each SP rank
        holds a *different* shard of the global batch. In the common FSDP + Ulysses setup
        here, ranks already receive the same ``DataProto`` from DP routing; gathering then
        multiplies batch size by ``sp_size`` and can make ``embed_tokens`` try to allocate
        hundreds of GiB (CUDA OOM).

        Broadcasting rank-0 tensors keeps a single logical batch per rank, which matches
        sequence-parallel forward. Ray job submit packs ``working_dir``; ensure this file
        is the one under that directory or workers still run stale code.
        """
        if self.device_mesh is None:
            return data
        group = self.device_mesh["sp"].get_group()
        if data.batch is None:
            if data.non_tensor_batch is not None:
                obj_list = [data.non_tensor_batch]
                dist.broadcast_object_list(obj_list, src=0, group=group)
                data.non_tensor_batch = obj_list[0]
            return data

        prev_device = data.batch.device
        batch_size = data.batch.batch_size
        dev = get_torch_device().current_device()
        out = {}
        for key in sorted(data.batch.keys()):
            t = data.batch[key].to(dev, non_blocking=True).contiguous()
            dist.broadcast(t, src=0, group=group)
            out[key] = t
        data.batch = TensorDict(source=out, batch_size=batch_size).to(prev_device)

        obj_list = [data.non_tensor_batch]
        dist.broadcast_object_list(obj_list, src=0, group=group)
        data.non_tensor_batch = obj_list[0]
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """
        Split the data to follow FSDP partition
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh["sp"].size()
            sp_rank = self.device_mesh["sp"].get_local_rank()
            data = data.chunk(chunks=sp_size)[sp_rank]
        return data
