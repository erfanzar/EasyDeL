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


"""Data carrier dataclasses for kernel tiling and execution configuration.

This module provides :class:`FwdParams` and :class:`BwdParams` — lightweight
dataclasses that carry the block-size and GPU-execution parameters commonly
needed by forward and backward kernel variants (primarily attention and
matrix-multiplication operations).

All fields default to ``None``, which signals to the kernel that it should
select an appropriate value automatically (via heuristics or autotuning).

Custom hashing:
    Both dataclasses override ``__hash__`` with :func:`ejkernel.callib.hash_fn`,
    which builds a hash from the concatenated string representations of the
    object's numeric/collection attributes. This makes the objects usable as
    dictionary keys and in :class:`~ejkernel.ops.config.ConfigCache` lookups
    without requiring them to be ``frozen=True``.

Classes:
    FwdParams: Block-size and GPU-execution parameters for forward kernels.
    BwdParams: Block-size and GPU-execution parameters for backward kernels.
"""

from dataclasses import dataclass

from ejkernel.callib import hash_fn


@dataclass
class FwdParams:
    """Forward pass parameters for kernel configuration.

    Encapsulates block sizes and execution parameters for forward pass kernels,
    particularly for attention and matrix multiplication operations.

    Attributes:
        blocksize_m: Block size for M dimension (rows of output matrix)
        blocksize_k: Block size for K dimension (reduction dimension)
        blocksize_n: Block size for N dimension (columns of output matrix)
        q_blocksize: Block size for query dimension in attention
        kv_blocksize: Block size for key/value dimension in attention
        blocksize_heads: Block size for head dimension in multi-head attention
        blocksize_keys: Block size for key sequence length
        num_key_splits: Number of splits for key computation
        num_warps: Number of GPU warps for thread block execution
        num_stages: Number of pipeline stages for memory optimization

    Note:
        All parameters are optional (None) to allow automatic selection
        during kernel execution or autotuning.
    """

    blocksize_m: int | None = None
    blocksize_k: int | None = None
    blocksize_n: int | None = None
    q_blocksize: int | None = None
    kv_blocksize: int | None = None

    blocksize_heads: int | None = None
    blocksize_keys: int | None = None
    num_key_splits: int | None = None

    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn


@dataclass
class BwdParams:
    """Backward pass parameters for kernel configuration.

    Encapsulates block sizes and execution parameters for backward pass kernels,
    used in gradient computation for attention and matrix multiplication operations.

    Attributes:
        blocksize_m: Block size for M dimension (rows of output matrix)
        blocksize_k: Block size for K dimension (reduction dimension)
        blocksize_n: Block size for N dimension (columns of output matrix)
        q_blocksize: Block size for query dimension in attention gradients
        kv_blocksize: Block size for key/value dimension in attention gradients
        num_warps: Number of GPU warps for thread block execution
        num_stages: Number of pipeline stages for memory optimization

    Note:
        Parameters are typically smaller than forward pass due to different
        memory access patterns in gradient computation.
    """

    blocksize_m: int | None = None
    blocksize_k: int | None = None
    blocksize_n: int | None = None
    q_blocksize: int | None = None
    kv_blocksize: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn
