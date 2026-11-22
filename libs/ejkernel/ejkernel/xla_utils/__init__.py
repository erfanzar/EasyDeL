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


"""
XLA-specific utilities for sequence processing and sharding.

This module provides utilities for working with packed sequences, cumulative
operations, and sharding specifications in JAX/XLA computations. These utilities
are essential for efficient transformer implementations with variable-length
sequences and distributed training.

Key Features:
    - Packed sequence utilities for efficient batch processing
    - Chunked cumulative sum operations for attention mechanisms
    - Sharding utilities for distributed computation
    - Sequence reordering for ring attention patterns

Packed Sequences:
    Packed sequences concatenate multiple variable-length sequences into a single
    tensor, using cumulative sequence lengths (cu_seqlens) to track boundaries.
    This is more memory-efficient than padding all sequences to maximum length.

    >>> import jax.numpy as jnp
    >>> from ejkernel.xla_utils import prepare_lens, prepare_position_ids
    >>>
    >>> # Three sequences of lengths 3, 2, 4 packed together
    >>> cu_seqlens = jnp.array([0, 3, 5, 9])
    >>> lens = prepare_lens(cu_seqlens)  # [3, 2, 4]
    >>> pos_ids = prepare_position_ids(cu_seqlens)  # [0,1,2, 0,1, 0,1,2,3]

Sequence Utilities:
    - cdiv: Ceiling division for computing chunk/block counts
    - prepare_lens: Calculate sequence lengths from cumulative lengths
    - prepare_lens_from_mask: Calculate lengths from attention mask
    - prepare_position_ids: Generate position IDs for packed sequences
    - prepare_sequence_ids: Generate sequence IDs for packed sequences
    - prepare_token_indices: Generate (seq_id, pos_id) pairs
    - prepare_chunk_indices: Generate chunk indices for tiled processing
    - prepare_chunk_offsets: Generate cumulative chunk offsets
    - prepare_cu_seqlens_from_mask: Create cumulative lengths from masks
    - identity_dtype_convert: Identity function with gradient dtype conversion

Cumulative Sum Operations:
    Used in linear attention mechanisms (GLA, RetNet, etc.) for computing
    running sums of gating values or state accumulations.

    - chunk_local_cumsum: Cumsum within fixed-size chunks (resets at boundaries)
    - chunk_global_cumsum: Cumsum across entire sequence (respects boundaries)

    >>> from ejkernel.xla_utils import chunk_local_cumsum
    >>> # Chunked cumsum resets every chunk_size positions
    >>> result = chunk_local_cumsum(g, chunk_size=128)

Sharding Utilities:
    For distributed training with JAX device meshes.

    - get_corrected_named_sharding: Create valid shardings by correcting specs
    - reorder_sequence: Reorder sequences for ring attention communication

    >>> from ejkernel.xla_utils import get_corrected_named_sharding
    >>> from jax.sharding import PartitionSpec, Mesh
    >>> sharding = get_corrected_named_sharding(shape, spec, mesh)

Example:
    >>> from ejkernel.xla_utils import (
    ...     prepare_position_ids,
    ...     prepare_sequence_ids,
    ...     prepare_cu_seqlens_from_mask,
    ... )
    >>> import jax.numpy as jnp
    >>>
    >>> # From attention mask to packed sequence utilities
    >>> mask = jnp.array([[True, True, True, False],
    ...                   [True, True, False, False]])
    >>> cu_seqlens = prepare_cu_seqlens_from_mask(mask)  # [0, 3, 5]
    >>> pos_ids = prepare_position_ids(cu_seqlens)  # [0, 1, 2, 0, 1]
    >>> seq_ids = prepare_sequence_ids(cu_seqlens)  # [0, 0, 0, 1, 1]
"""

from .cumsum import chunk_global_cumsum, chunk_local_cumsum
from .shardings import get_corrected_named_sharding, reorder_sequence
from .utils import (
    cdiv,
    identity_dtype_convert,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_cu_seqlens_from_mask,
    prepare_lens,
    prepare_lens_from_mask,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_token_indices,
)

__all__ = [
    "cdiv",
    "chunk_global_cumsum",
    "chunk_local_cumsum",
    "get_corrected_named_sharding",
    "identity_dtype_convert",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "prepare_cu_seqlens_from_mask",
    "prepare_lens",
    "prepare_lens_from_mask",
    "prepare_position_ids",
    "prepare_sequence_ids",
    "prepare_token_indices",
    "reorder_sequence",
]
