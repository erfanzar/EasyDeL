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

"""Utility functions and tuned block sizes for Ragged Paged Attention V2.

This module provides utility functions for TPU device detection, block size
selection, and performance tuning for the ragged paged attention V2 kernel.
It includes pre-tuned block size configurations for different TPU generations
and workload characteristics.

Key Components:
    - TUNED_BLOCK_SIZES: Dictionary of empirically tuned (bkv, bq) block sizes
      indexed by TPU version and workload parameters
    - next_power_of_2: Helper for power-of-2 alignment calculations
    - simplify_key: Normalizes workload parameters for block size lookup
    - get_tpu_version: Detects TPU generation from device info
    - get_tuned_block_sizes: Retrieves optimal block sizes for given workload

Block Size Parameters:
    - bkv (num_kv_pages_per_block): Number of KV cache pages processed per
      Flash Attention block. Larger values improve HBM bandwidth utilization
      but increase VMEM usage
    - bq (num_queries_per_block): Number of query tokens processed per block.
      Affects parallelism and memory footprint

Tuning Dimensions:
    The tuned configurations are indexed by:
    - q_dtype, kv_dtype: Data types for queries and KV cache
    - num_q_heads_per_blk: Query heads per processing block
    - num_kv_heads_per_blk: KV heads per processing block
    - head_dim: Attention head dimension (128, 256, etc.)
    - page_size: KV cache page size (16, 32, 64, etc.)
    - max_num_batched_tokens: Maximum tokens in a batch
    - pages_per_seq: Maximum pages per sequence

TPU Version Support:
    - TPU v4: Conservative block sizes for older architecture
    - TPU v5: Optimized for enhanced memory bandwidth
    - TPU v6: Maximum performance with larger block sizes

Example:
    >>> # Get optimal block sizes for a workload
    >>> bkv_pages, bq_size = get_tuned_block_sizes(
    ...     q_dtype=jnp.bfloat16,
    ...     kv_dtype=jnp.bfloat16,
    ...     num_q_heads_per_blk=32,
    ...     num_kv_heads_per_blk=8,
    ...     head_dim=128,
    ...     page_size=64,
    ...     max_num_batched_tokens=2048,
    ...     pages_per_seq=256,
    ... )
    >>> print(f"Block sizes: KV pages={bkv_pages}, queries={bq_size}")

Note:
    The tuned block sizes are empirically determined through benchmarking
    and may not be optimal for all workloads. The simplify_key function
    normalizes parameters to reduce the tuning space while maintaining
    good performance across similar configurations.
"""

import jax
import jax.numpy as jnp

MAX_PAGES_PER_SEQ = 16


TUNED_BLOCK_SIZES = {
    "TPU v6": {
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 1024, 1024): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 1024, 2048): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 1024, 4096): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 1024, 512): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 2048, 1024): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 2048, 2048): (16, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 2048, 4096): (32, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 2048, 512): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 4096, 1024): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 4096, 2048): (16, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 4096, 4096): (32, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 4096, 512): (4, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 512, 1024): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 512, 2048): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 512, 4096): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 512, 512): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 1024): (64, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 2048): (128, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 512): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 64): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 1024): (64, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 2048): (128, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 512): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 64): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 1024): (64, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 2048): (128, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 512): (32, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 64): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 1024): (64, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 2048): (128, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 512): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 64): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 1024, 1024): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 1024, 2048): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 1024, 4096): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 2048, 1024): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 2048, 2048): (8, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 2048, 4096): (16, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 4096, 1024): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 4096, 2048): (8, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 4096, 4096): (16, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 512, 1024): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 512, 2048): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 512, 4096): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 1024): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 128): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 2048): (64, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 256): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 4096): (128, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 512): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 1024): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 128): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 2048): (64, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 256): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 4096): (128, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 512): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 1024): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 128): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 2048): (64, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 256): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 4096): (128, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 512): (16, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 1024): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 128): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 2048): (64, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 256): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 4096): (128, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 512): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 1024, 1024): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 1024, 2048): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 1024, 256): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 1024, 4096): (64, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 1024, 512): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 2048, 1024): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 2048, 2048): (32, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 2048, 256): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 2048, 4096): (64, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 2048, 512): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 4096, 1024): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 4096, 2048): (32, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 4096, 256): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 4096, 4096): (64, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 4096, 512): (8, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 512, 1024): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 512, 2048): (32, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 512, 256): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 512, 4096): (64, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 512, 512): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 1024, 1024): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 1024, 2048): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 1024, 4096): (32, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 1024, 512): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 2048, 1024): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 2048, 2048): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 2048, 4096): (32, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 2048, 512): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 4096, 1024): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 4096, 2048): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 4096, 4096): (32, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 4096, 512): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 512, 1024): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 512, 2048): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 512, 4096): (32, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 512, 512): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 1024): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 128): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 2048): (128, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 256): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 512): (32, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 64): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 1024): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 128): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 2048): (128, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 256): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 512): (32, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 64): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 1024): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 128): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 2048): (128, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 256): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 512): (32, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 64): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 1024): (64, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 128): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 2048): (128, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 256): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 512): (32, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 64): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 1024, 1024): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 1024, 2048): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 1024, 4096): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 2048, 1024): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 2048, 2048): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 2048, 4096): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 4096, 1024): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 4096, 2048): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 4096, 4096): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 512, 1024): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 512, 2048): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 512, 4096): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 1024): (32, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 128): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 2048): (64, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 256): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 4096): (128, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 512): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 1024): (32, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 128): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 2048): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 256): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 4096): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 512): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 1024): (32, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 128): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 2048): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 256): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 4096): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 512): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 1024): (32, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 128): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 2048): (64, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 256): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 4096): (128, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 512): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 1024, 1024): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 1024, 2048): (32, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 1024, 256): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 1024, 4096): (64, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 1024, 512): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 2048, 1024): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 2048, 2048): (32, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 2048, 256): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 2048, 4096): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 2048, 512): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 4096, 1024): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 4096, 2048): (32, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 4096, 256): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 4096, 4096): (64, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 4096, 512): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 512, 1024): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 512, 2048): (32, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 512, 256): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 512, 4096): (64, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 512, 512): (8, 32),
    },
    "TPU v5": {
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 1024, 1024): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 1024, 2048): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 1024, 512): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 2048, 1024): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 2048, 2048): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 2048, 512): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 4096, 1024): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 4096, 2048): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 4096, 512): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 512, 1024): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 512, 2048): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 128, 512, 512): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 1024, 64): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 2048, 64): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 4096, 64): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 16, 512, 64): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 1024, 1024): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 1024, 2048): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 1024, 4096): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 2048, 1024): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 2048, 2048): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 2048, 4096): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 4096, 1024): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 4096, 2048): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 4096, 4096): (16, 64),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 512, 1024): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 512, 2048): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 256, 512, 4096): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 128): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 256): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 1024, 512): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 128): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 256): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 2048, 512): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 128): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 256): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 4096, 512): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 128): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 256): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 32, 512, 512): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 1024, 1024): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 1024, 256): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 1024, 512): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 2048, 1024): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 2048, 256): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 2048, 512): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 4096, 1024): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 4096, 256): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 4096, 512): (8, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 512, 1024): (16, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 512, 256): (4, 32),
        ("bfloat16", "bfloat16", 32, 8, 128, 64, 512, 512): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 1024, 1024): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 1024, 2048): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 1024, 512): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 2048, 1024): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 2048, 2048): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 2048, 512): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 4096, 1024): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 4096, 2048): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 4096, 512): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 512, 1024): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 512, 2048): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 512, 512): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 128): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 256): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 1024, 64): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 128): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 256): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 2048, 64): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 128): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 256): (16, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 4096, 64): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 128): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 256): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 16, 512, 64): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 1024, 1024): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 1024, 2048): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 1024, 4096): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 2048, 1024): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 2048, 2048): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 2048, 4096): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 4096, 1024): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 4096, 2048): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 4096, 4096): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 512, 1024): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 512, 2048): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 256, 512, 4096): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 128): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 256): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 1024, 512): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 128): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 256): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 2048, 512): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 128): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 256): (8, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 4096, 512): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 128): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 256): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 32, 512, 512): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 1024, 1024): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 1024, 256): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 1024, 512): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 2048, 1024): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 2048, 256): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 2048, 512): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 4096, 1024): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 4096, 256): (4, 64),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 4096, 512): (8, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 512, 1024): (16, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 512, 256): (4, 32),
        ("bfloat16", "bfloat16", 8, 1, 128, 64, 512, 512): (8, 32),
    },
}


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
    """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def simplify_key(key):
    """Normalize workload parameters to reduce the tuning space.

    Rounds parameters to the nearest power of 2 and aligns head dimensions
    to 128-byte boundaries. This allows a smaller set of tuned block sizes
    to cover a wider range of workload configurations with similar performance
    characteristics.

    Args:
        key: Tuple of (q_dtype, kv_dtype, num_q_heads_per_blk,
            num_kv_heads_per_blk, head_dim, page_size,
            max_num_batched_tokens, pages_per_seq).

    Returns:
        Simplified tuple with rounded/normalized values suitable for
        lookup in the TUNED_BLOCK_SIZES dictionary.
    """
    (
        q_dtype,
        kv_dtype,
        num_q_heads_per_blk,
        num_kv_heads_per_blk,
        head_dim,
        page_size,
        max_num_batched_tokens,
        pages_per_seq,
    ) = key
    return (
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_blk),
        next_power_of_2(num_kv_heads_per_blk),
        (head_dim + 127) // 128 * 128,
        next_power_of_2(page_size),
        next_power_of_2(max_num_batched_tokens),
        next_power_of_2(page_size * pages_per_seq),
    )


def get_tpu_version() -> int:
    """Detect and return the numeric TPU generation version.

    Inspects the first JAX device's kind string to determine the TPU version.
    Handles both standard TPU and TPU lite variants.

    Returns:
        Integer TPU version (e.g., 4, 5, 6), or -1 if not running on TPU.
    """
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])


def get_device_name(num_devices: int | None = None):
    """Get a standardized device name string for block size lookup.

    Extracts the first two words of the device kind (e.g., "TPU v6") and
    optionally appends the device count for multi-device configurations.

    Args:
        num_devices: Optional number of devices to append to the name.

    Returns:
        Device name string like "TPU v6" or "TPU v6-8".
    """
    name = " ".join(jax.devices()[0].device_kind.split()[:2])
    if num_devices is not None:
        name += f"-{num_devices}"
    return name


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    num_q_heads_per_blk,
    num_kv_heads_per_blk,
    head_dim,
    page_size,
    max_num_batched_tokens,
    pages_per_seq,
) -> tuple[int, int]:
    """Look up optimal block sizes from the auto-tuned configuration table.

    Searches the TUNED_BLOCK_SIZES dictionary for the best (bkv, bq) pair
    matching the given workload parameters and TPU version. Falls back to
    conservative defaults if no tuned configuration is found.

    Args:
        q_dtype: Query tensor dtype (e.g., jnp.bfloat16).
        kv_dtype: KV cache dtype (e.g., jnp.bfloat16).
        num_q_heads_per_blk: Number of query heads per processing block.
        num_kv_heads_per_blk: Number of KV heads per processing block.
        head_dim: Attention head dimension.
        page_size: Number of tokens per KV cache page.
        max_num_batched_tokens: Maximum tokens in a batch.
        pages_per_seq: Maximum number of pages per sequence.

    Returns:
        Tuple of (num_kv_pages_per_blk, num_queries_per_blk), clamped to
        not exceed pages_per_seq and max_num_batched_tokens respectively.

    Raises:
        NotImplementedError: If TPU version is less than 4.
    """
    tpu_version = get_tpu_version()
    if tpu_version < 4:
        raise NotImplementedError("TPU version must be 4 or higher.")
    key = (
        q_dtype,
        kv_dtype,
        num_q_heads_per_blk,
        num_kv_heads_per_blk,
        head_dim,
        page_size,
        max_num_batched_tokens,
        pages_per_seq,
    )
    key = simplify_key(key)
    device_name = get_device_name()

    bkv, bq = (128, 32)
    if tpu_version == 4:
        bkv, bq = (32, 32)
    elif device_name in TUNED_BLOCK_SIZES:
        if key in TUNED_BLOCK_SIZES[device_name]:
            bkv, bq = TUNED_BLOCK_SIZES[device_name][key]
    return (min(pages_per_seq, bkv), min(max_num_batched_tokens, bq))


def get_min_page_size(max_model_len, min_page_size=16):
    """Compute the recommended minimum page size for the attention kernel.

    Determines the smallest page size that keeps the total number of pages
    per sequence at or below MAX_PAGES_PER_SEQ, ensuring efficient page
    table lookups.

    Args:
        max_model_len: Maximum model sequence length to support.
        min_page_size: Minimum allowed page size. Defaults to 16.

    Returns:
        Recommended page size (power of 2, at least min_page_size).
    """
    return max(next_power_of_2(max_model_len) // MAX_PAGES_PER_SEQ, min_page_size)
