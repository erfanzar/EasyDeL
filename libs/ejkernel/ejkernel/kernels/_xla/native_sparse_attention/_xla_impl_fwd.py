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

"""Native Sparse Attention forward pass implementation using XLA/JAX.

This module provides the forward pass for Native Sparse Attention (NSA),
implementing block-sparse attention where each query only attends to a
selected subset of key-value blocks.

Key Components:
    - _sparse_attention_fwd: Main forward function with block selection

Algorithm:
    Block-sparse attention with dynamic block selection:
    1. For each query token, look up which K/V blocks to attend to
    2. Gather the selected K/V blocks using block_indices
    3. Compute attention only over selected blocks
    4. Apply softmax and weighted sum over selected blocks

Features:
    - Dynamic block selection via block_indices
    - GQA/MQA support with head broadcasting
    - Variable block counts per query via block_counts
    - Efficient JAX vmap/scan implementation

Memory Complexity:
    - O(N * S * B) where N=seq_len, S=num_selected_blocks, B=block_size
    - Much less than full O(N²) for sparse patterns

Note:
    Uses ejit decorator for efficient JIT compilation with
    static argument handling.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.callib import ejit


@ejit(static_argnums=(5,))
def _sparse_attention_fwd(
    q: Float[Array, "batch seq_len num_q_heads head_dim"],
    k: Float[Array, "batch seq_len num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"],
    block_size: int,
    softmax_scale: float,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Compute forward pass for block-sparse attention with per-token selection and causal masking.

    For each query token, this function gathers the selected K/V blocks
    specified by block_indices, computes scaled dot-product attention over
    those blocks with causal masking, and returns the weighted sum of values.

    The block selection is dynamic and per-token: each query position can
    attend to a different set of K/V blocks, controlled by block_indices
    and block_counts.

    Args:
        q: Query tensor [batch, seq_len, num_q_heads, head_dim].
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim].
            Padded to block boundaries internally.
        v: Value tensor [batch, seq_len, num_kv_heads, head_dim].
            Padded to block boundaries internally.
        block_indices: Per-token block selection indices
            [batch, seq_len, num_kv_heads, num_selected_blocks].
            Values of -1 are treated as invalid (sentinel) and masked out.
        block_counts: Number of valid blocks per token
            [batch, seq_len, num_kv_heads].
        block_size: Size of each K/V block (static argument).
        softmax_scale: Scaling factor applied to QK^T dot products.

    Returns:
        Attention output [batch, seq_len, num_q_heads, head_dim].
    """
    B, T, HQ, D = q.shape
    HKV = k.shape[2]
    G = HQ // HKV

    NB = (T + block_size - 1) // block_size
    pad_len = NB * block_size - T
    if pad_len > 0:
        k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))

    k_blocks = k.reshape(B, NB, block_size, HKV, D)
    v_blocks = v.reshape(B, NB, block_size, HKV, D)

    def attend_for_head(hq, q_b, k_blocks_b, v_blocks_b, bi_b, bc_b):
        """Compute sparse attention for a single query head across all tokens.

        Maps each query head to its corresponding KV head group, then
        vectorizes attention computation across all token positions.

        Args:
            hq: Query head index.
            q_b: Query tensor for this batch element [seq_len, num_q_heads, head_dim].
            k_blocks_b: Key blocks [num_blocks, block_size, num_kv_heads, head_dim].
            v_blocks_b: Value blocks [num_blocks, block_size, num_kv_heads, head_dim].
            bi_b: Block indices for this batch [seq_len, num_kv_heads, num_selected].
            bc_b: Block counts for this batch [seq_len, num_kv_heads].

        Returns:
            Attention output for this head [seq_len, head_dim].
        """
        kvh = hq // G

        def attend_for_token(t):
            """Compute sparse attention output for a single query token.

            Gathers selected K/V blocks, applies causal and validity
            masking, computes softmax-weighted sum over selected blocks.

            Args:
                t: Token position index within the sequence.

            Returns:
                Attention output vector [head_dim] for this token.
            """
            q_vec = q_b[t, hq]
            inds = bi_b[t, kvh]
            cnt = bc_b[t, kvh]

            nb = k_blocks_b.shape[0]
            inds_valid = (inds >= 0) & (inds < nb)
            inds_safe = jnp.where(inds_valid, inds, 0)

            k_sel = k_blocks_b[inds_safe, :, kvh, :]
            v_sel = v_blocks_b[inds_safe, :, kvh, :]

            bs = jnp.arange(block_size)
            local_limit = t - inds * block_size
            pos_mask = bs[None, :] <= local_limit[:, None]
            s = jnp.arange(inds.shape[0])
            blk_mask = s < cnt
            valid_mask = pos_mask & blk_mask[:, None] & inds_valid[:, None]

            scores = jnp.einsum("d,sbd->sb", q_vec, k_sel) * softmax_scale
            scores = jnp.where(valid_mask, scores, jnp.full_like(scores, -1e9))
            w = jax.nn.softmax(scores.reshape(-1), axis=-1)

            v_flat = v_sel.reshape(-1, D)
            return (w[:, None] * v_flat).sum(axis=0)

        return jax.vmap(attend_for_token, in_axes=(0,), out_axes=0)(jnp.arange(T))

    outs = jax.vmap(
        lambda q_b, kb_b, vb_b, bi_b, bc_b: jax.vmap(
            attend_for_head,
            in_axes=(0, None, None, None, None, None),
            out_axes=0,
        )(jnp.arange(HQ), q_b, kb_b, vb_b, bi_b, bc_b).transpose(1, 0, 2),
        in_axes=(0, 0, 0, 0, 0),
        out_axes=0,
    )(q, k_blocks, v_blocks, block_indices, block_counts)

    return outs
