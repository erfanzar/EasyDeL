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

"""Utility functions and constants for Flash MLA TPU Pallas kernel."""

from __future__ import annotations

import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
"""Large negative value for masked attention positions."""

NUM_LANES = 128
"""Number of lanes in TPU vector processing unit."""

NUM_SUBLANES = 8
"""Number of sublanes for efficient broadcasting."""

MIN_BLOCK_SIZE = 128
"""Minimum efficient block size for TPU MXU operations."""

TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))
"""Einsum dimension numbers for A @ B^T: contracts last dim of both."""


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
    """Verify that a block size is valid for a given dimension.

    Args:
        block_name: Name of the block parameter for error messages.
        dim_name: Name of the dimension for error messages.
        block: The block size to verify.
        dim: The dimension size to check against.
        should_divide: If True, require dim to be divisible by block.

    Raises:
        ValueError: If block > dim or if should_divide and dim % block != 0.
    """
    if block > dim:
        raise ValueError(f"{block_name}={block} should be <= {dim_name}={dim}")
    if should_divide and dim % block != 0:
        raise ValueError(f"{dim_name}={dim} should be divisible by {block_name}={block}")


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    """Check if query block can attend to KV block under causal masking.

    Returns True if the last row of query block *r* is at or after the
    first column of KV block *c*.

    Args:
        r: Query block index.
        r_blk_size: Query block size.
        c: KV block index.
        c_blk_size: KV block size.

    Returns:
        Boolean scalar.
    """
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)
