# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused projection layouts, builders, splitters and reform rules.

Public surface for the layout package. The pieces fit together as follows:

* :mod:`.dense` owns :class:`FusedColumnLayout` and :class:`FusedSegment`,
  the declarative descriptors used by dense fused projections (Q/K/V,
  gate/up). :func:`dense_qkv_layout` and :func:`dense_gate_up_layout`
  build the two canonical instances.
* :mod:`.moe` owns :class:`FusedExpertLayout`, the MoE expert counterpart
  whose expert / TP axis layout differs from dense projections.
* :mod:`.builders` exposes the high-level ``build_fused_*`` constructors
  that wire a layout onto a fresh :class:`ColumnParallelLinear`, plus
  the matching ``split_fused_*`` activation splitters.
* :mod:`.reform` provides the checkpoint reform rule generators
  (``*_fusion_reform_param``) consumed by the EasyDeL checkpoint loader.
* :mod:`.runtime` owns the pure-JAX runtime splitters and the TP
  sharding helpers; :mod:`.torch_packing` provides the torch-side
  counterparts used at checkpoint load time.
* :mod:`.types` centralises :data:`ReformRule` / :data:`ReformParam`
  type aliases.
"""

from __future__ import annotations

from ._builders import (
    build_fused_gate_up_projection,
    build_fused_qkv_projection,
    split_fused_gate_up_projection,
    split_fused_qkv_projection,
)
from ._dense import FusedColumnLayout, FusedSegment, dense_gate_up_layout, dense_qkv_layout
from ._moe import FusedExpertLayout
from ._reform import (
    gate_up_fusion_reform_param,
    hf_per_expert_swiglu_reform_param,
    interleaved_fusion_reform_param,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
    moe_gate_up_fusion_reform_param,
    qkv_fusion_reform_param,
)
from ._runtime import (
    keep_interleaved_segments_last_axis,
    normalize_segment_sizes,
    split_contiguous_segments_last_axis,
    split_interleaved_pair_last_axis,
    split_interleaved_segments_last_axis,
    tensor_parallel_axis,
    tensor_parallel_size,
    with_tp_last_axis_sharding,
)
from ._torch_packing import (
    torch_deinterleave_axis_segments_for_tp,
    torch_deinterleave_segments_for_tp,
    torch_interleave_axis_segments_for_tp,
    torch_interleave_segments_for_tp,
)
from ._types import ReformParam, ReformRule

__all__ = [
    "FusedColumnLayout",
    "FusedExpertLayout",
    "FusedSegment",
    "ReformParam",
    "ReformRule",
    "build_fused_gate_up_projection",
    "build_fused_qkv_projection",
    "dense_gate_up_layout",
    "dense_qkv_layout",
    "gate_up_fusion_reform_param",
    "hf_per_expert_swiglu_reform_param",
    "interleaved_fusion_reform_param",
    "keep_interleaved_segments_last_axis",
    "moe_down_projection_reform_param",
    "moe_fused_gate_up_reform_param",
    "moe_gate_up_fusion_reform_param",
    "normalize_segment_sizes",
    "qkv_fusion_reform_param",
    "split_contiguous_segments_last_axis",
    "split_fused_gate_up_projection",
    "split_fused_qkv_projection",
    "split_interleaved_pair_last_axis",
    "split_interleaved_segments_last_axis",
    "tensor_parallel_axis",
    "tensor_parallel_size",
    "torch_deinterleave_axis_segments_for_tp",
    "torch_deinterleave_segments_for_tp",
    "torch_interleave_axis_segments_for_tp",
    "torch_interleave_segments_for_tp",
    "with_tp_last_axis_sharding",
]
