# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Efficient Ring Attention Implementation for Single-Device Execution

This module provides an optimized implementation of ring attention,
originally inspired by the work of Liu et al. (2023)
([https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)).
It incorporates the following enhancements:

- Single-Device Focus: Adapted for efficient execution on a single device,
  removing the need for parallel communication primitives.
- Enhanced JIT Compatibility: Streamlined for smoother integration with
  JAX's Just-In-Time (JIT) compilation.
- Performance Optimizations:  Includes code optimizations for improved speed
  and memory usage.

Note: While based on existing implementations, this version offers significant
modifications to enhance its usability and performance in single-device and multi-host
settings.
- also adding softmax scale option to support custom scales
"""

from functools import partial

import chex
import jax
from jax import Array

from ._backward_pallas import _ring_flash_attention_bwd_tpu
from ._forward_pallas import _ring_flash_attention_fwd_tpu
from ._utils import SegmentIds


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def ring_attention(
    q: Array,
    k: Array,
    v: Array,
    attn_bias: Array | None,
    segment_ids: SegmentIds | None,
    cache_idx,
    axis_name: str,
    float32_logits,
    query_chunk_size,
    key_chunk_size,
    causal_block_size,
) -> chex.Array:
    """
    Computes ring attention using FlashAttention on TPU.
    """
    y, _ = _ring_flash_attention_fwd_tpu(
        q,
        k,
        v,
        attn_bias,
        segment_ids,
        cache_idx,
        axis_name,
        float32_logits,
        query_chunk_size,
        key_chunk_size,
        causal_block_size,
    )
    return y


ring_attention.defvjp(
    _ring_flash_attention_fwd_tpu,
    _ring_flash_attention_bwd_tpu,
)
