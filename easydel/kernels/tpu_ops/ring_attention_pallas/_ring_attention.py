# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import typing as tp
from functools import partial

import chex
import jax

from ._utils import SegmentIds
from ._forward_pallas import _ring_flash_attention_fwd_tpu
from ._backward_pallas import _ring_flash_attention_bwd_tpu


@partial(
	jax.custom_vjp,
	nondiff_argnums=(6, 7, 8, 9, 10, 11),
)
def ring_attention(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	bias: tp.Optional[chex.Array] = None,
	segment_ids: tp.Optional[SegmentIds] = None,
	cache_idx: tp.Optional[int] = None,
	axis_name: tp.Optional[str] = None,
	float32_logits: bool = True,
	softmax_scale: tp.Optional[float] = None,
	blocksize_q: int = 256,
	blocksize_k: int = 256,
	blocksize_c: tp.Optional[int] = None,
) -> chex.Array:
	"""Computes ring attention using FlashAttention on TPU.

	Args:
	    query: Query array of shape (batch, query_len, num_heads, dim_per_head).
	    key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
	    value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
	    bias: tp.Optional bias array.  Its shape depends on the attention mechanism.
	    segment_ids: tp.Optional segment ids for Q and KV sequences.
	    cache_idx: tp.Optional cache index for use with caching.
	    axis_name: tp.Optional name of the axis to ppermute over (for multi-host support).
	    float32_logits: Whether to compute logits in float32.
	    softmax_scale: tp.Optional scaling factor for the softmax function.
	    blocksize_q: Block size for the query sequence.
	    blocksize_k: Block size for the key/value sequence.
	    blocksize_c: tp.Optional block size for causal masking.


	Returns:
	    Output array of shape (batch, query_len, num_heads, dim_per_head).
	"""
	y, _ = _ring_flash_attention_fwd_tpu(
		query,
		key,
		value,
		bias,
		segment_ids,
		cache_idx,
		axis_name,
		float32_logits,
		softmax_scale,
		blocksize_q,
		blocksize_k,
		blocksize_c,
	)
	return y


ring_attention.defvjp(
	_ring_flash_attention_fwd_tpu,
	_ring_flash_attention_bwd_tpu,
)
