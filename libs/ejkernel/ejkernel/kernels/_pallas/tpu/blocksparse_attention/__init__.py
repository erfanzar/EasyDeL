# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Block-Sparse (Splash) Attention implementation for TPU using Pallas.

This module provides TPU-optimized block-sparse attention using JAX's Pallas
library with Mosaic backend. Based on Google's Splash Attention implementation,
it enables efficient attention computation by exploiting sparsity patterns
in the attention matrix.

Block-sparse attention is particularly useful for:
1. Long sequence modeling where full attention is prohibitively expensive
2. Structured sparsity patterns like local windows and causal masks
3. Memory-efficient attention with predictable memory usage
4. High throughput on TPU hardware

Key Features:
    - Multiple mask types: CausalMask, LocalMask, ChunkedCausalMask, FullMask
    - Multi-head attention (MHA) with separate masks per head
    - Multi-query attention (MQA) for efficient inference
    - Single-device and multi-device (ring) execution
    - Custom mask builders for arbitrary sparsity patterns
    - Composable masks via logical AND/OR operations

Supported Mask Types:
    - CausalMask: Lower triangular mask for autoregressive attention
    - LocalMask: Sliding window mask for local attention patterns
    - ChunkedCausalMask: Block-diagonal causal mask (Llama4-style)
    - FullMask: No masking (dense attention)
    - NumpyMask: Custom mask from numpy array
    - MultiHeadMask: Different masks per attention head

Algorithm overview:
1. Mask is preprocessed to identify non-zero blocks
2. Only non-zero blocks are computed, skipping masked regions
3. Block indices are used for efficient memory access
4. Results are accumulated in the correct output positions

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._pallas.tpu.blocksparse_attention import (
    ...     blocksparse_attention,
    ...     CausalMask,
    ...     LocalMask,
    ... )
    >>>
    >>> # Create causal attention
    >>> q = jnp.ones((batch, heads, seq_len, head_dim))
    >>> k = jnp.ones((batch, heads, seq_len, head_dim))
    >>> v = jnp.ones((batch, heads, seq_len, head_dim))
    >>>
    >>> mask = CausalMask(shape=(seq_len, seq_len))
    >>> output = blocksparse_attention(q, k, v, mask)
    >>>
    >>> # Sliding window with causal
    >>> window_mask = LocalMask(shape=(seq_len, seq_len), window_size=(512, 512))
    >>> combined = mask & window_mask  # Combine masks
    >>> output = blocksparse_attention(q, k, v, combined)

Public API:
    - blocksparse_attention: Main attention function
    - BlockSizes: Configuration for block tiling
    - Mask types: CausalMask, LocalMask, ChunkedCausalMask, FullMask, etc.
    - Factory functions: make_splash_mha, make_splash_mqa, etc.

Reference:
    Splash Attention: High-Performance Attention on TPU
    https://research.google/blog/splash-attention-high-performance-attention-on-tpu/
"""

from ._info import MaskInfo
from ._interface import blocksparse_attention
from ._kernel import (
    BlockSizes,
    make_attention_reference,
    make_masked_mha_reference,
    make_masked_mqa_reference,
    make_splash_mha,
    make_splash_mha_single_device,
    make_splash_mqa,
    make_splash_mqa_single_device,
)
from ._masks import (
    CausalMask,
    ChunkedCausalMask,
    FullMask,
    LocalMask,
    Mask,
    MultiHeadMask,
    NumpyMask,
    make_causal_mask,
    make_chunk_attention_mask,
    make_local_attention_mask,
    make_random_mask,
)

__all__ = (
    "BlockSizes",
    "CausalMask",
    "ChunkedCausalMask",
    "FullMask",
    "LocalMask",
    "Mask",
    "MaskInfo",
    "MultiHeadMask",
    "NumpyMask",
    "blocksparse_attention",
    "make_attention_reference",
    "make_causal_mask",
    "make_chunk_attention_mask",
    "make_local_attention_mask",
    "make_masked_mha_reference",
    "make_masked_mqa_reference",
    "make_random_mask",
    "make_splash_mha",
    "make_splash_mha_single_device",
    "make_splash_mqa",
    "make_splash_mqa_single_device",
)
