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

"""Native Sparse Attention backward pass implementation using XLA/JAX.

This module provides the backward pass (gradient computation) for Native
Sparse Attention, computing gradients dQ, dK, dV for the block-sparse
attention pattern.

Key Components:
    - _sparse_attention_bwd: Main backward function

Algorithm:
    Gradient computation for sparse attention:
    1. For each query, compute dQ from do and selected K/V blocks
    2. Scatter dK, dV gradients back to original positions using block_indices
    3. Handle gradient accumulation for overlapping block selections

Gradient Equations:
    - dQ: dL/dQ = do @ K^T (only selected K blocks)
    - dK: dL/dK = Q^T @ do * attention_weights (scattered to block positions)
    - dV: dL/dV = attention_weights^T @ do (scattered to block positions)

Features:
    - Efficient scatter-based gradient accumulation
    - Handles variable block_counts per query
    - GQA/MQA gradient broadcasting

Note:
    Uses ejit decorator for efficient JIT compilation. The scatter
    operations for dK/dV can be memory-intensive for highly overlapping
    block selection patterns.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.callib import ejit


@ejit(static_argnums=(5,))
def _sparse_attention_bwd(
    q: Float[Array, "batch seq_len num_q_heads head_dim"],
    k: Float[Array, "batch seq_len num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"],
    block_size: int,
    softmax_scale: float,
    do: Float[Array, "batch seq_len num_q_heads head_dim"],
) -> tuple[
    Float[Array, "batch seq_len num_q_heads head_dim"],
    Float[Array, "batch seq_len num_kv_heads head_dim"],
    Float[Array, "batch seq_len num_kv_heads head_dim"],
]:
    """Compute backward pass for block-sparse attention.

    Computes gradients dQ, dK, dV for the sparse attention forward pass.
    For each query token, recomputes the forward attention over selected
    blocks, then computes and scatters gradients back to the original
    K/V positions.

    The gradient flow follows standard attention backward:
        - dQ: Accumulated from softmax-weighted dScores times K
        - dK: Scattered from Q times dScores to selected block positions
        - dV: Scattered from attention weights times dO to selected positions

    Args:
        q: Query tensor [batch, seq_len, num_q_heads, head_dim].
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim].
        v: Value tensor [batch, seq_len, num_kv_heads, head_dim].
        block_indices: Per-token block selection indices
            [batch, seq_len, num_kv_heads, num_selected_blocks].
        block_counts: Number of valid blocks per token
            [batch, seq_len, num_kv_heads].
        block_size: Size of each K/V block (static argument).
        softmax_scale: Scaling factor applied to QK^T dot products.
        do: Gradient of the loss w.r.t. the attention output
            [batch, seq_len, num_q_heads, head_dim].

    Returns:
        Tuple of (dq, dk, dv) where:
            - dq: Gradient w.r.t. queries [batch, seq_len, num_q_heads, head_dim]
            - dk: Gradient w.r.t. keys [batch, seq_len, num_kv_heads, head_dim]
            - dv: Gradient w.r.t. values [batch, seq_len, num_kv_heads, head_dim]
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

    dq = jnp.zeros_like(q)
    bs = jnp.arange(block_size)

    def bkvh_backward(b, kvh):
        """Compute backward pass for one (batch, kv_head) pair.

        Processes all query head groups within this KV head, computing
        gradients for Q, K, V by iterating over all token positions.
        Uses JAX vmap over tokens for parallel gradient computation.

        Args:
            b: Batch index within the batch dimension.
            kvh: KV head index within the key/value head dimension.

        Returns:
            Tuple of (dq_b, dk_b, dv_b) where:
                - dq_b: Query gradients [seq_len, num_q_heads, head_dim]
                - dk_b: Key gradients [num_blocks, block_size, head_dim]
                - dv_b: Value gradients [num_blocks, block_size, head_dim]
        """
        hq_start = kvh * G

        q_b = q[b]
        do_b = do[b]
        q_grp = jax.lax.dynamic_slice(q_b, start_indices=(0, hq_start, 0), slice_sizes=(T, G, D))
        do_grp = jax.lax.dynamic_slice(do_b, start_indices=(0, hq_start, 0), slice_sizes=(T, G, D))

        inds_bt = block_indices[b, :, kvh, :]
        cnt_bt = block_counts[b, :, kvh]

        def token_bwd(t):
            """Compute backward gradients for a single token position.

            Recomputes forward attention weights, then derives dQ for
            each query head group and scatters dK, dV to the selected
            block positions.

            Args:
                t: Token position index within the sequence.

            Returns:
                Tuple of (dQ_g, dk_upd, dv_upd) where dQ_g has shape
                [G, head_dim] and dk_upd/dv_upd have shape
                [num_blocks, block_size, head_dim] for scatter accumulation.
            """
            inds = inds_bt[t]
            cnt = cnt_bt[t]

            k_sel = k_blocks[b, inds, :, kvh, :]
            v_sel = v_blocks[b, inds, :, kvh, :]

            local_limit = t - inds * block_size
            pos_mask = bs[None, :] <= local_limit[:, None]
            s_ar = jnp.arange(inds.shape[0])
            blk_mask = s_ar < cnt
            valid_mask = pos_mask & blk_mask[:, None]
            mask_flat = valid_mask.reshape(-1)

            k_flat = k_sel.reshape(-1, D)
            v_flat = v_sel.reshape(-1, D)

            def head_bwd(g):
                """Compute backward gradients for a single query head within the group.

                Recomputes softmax weights from Q and K, then applies the
                standard attention backward formula to get dQ, dK, dV
                contributions for this head.

                Args:
                    g: Query head index within the KV head group (0 to G-1).

                Returns:
                    Tuple of (dQ, dK_flat, dV_flat) gradients for this head.
                """
                q_vec = q_grp[t, g]
                do_vec = do_grp[t, g]

                scores = (k_flat @ q_vec) * softmax_scale
                scores = jnp.where(mask_flat, scores, -1e9)
                w = jax.nn.softmax(scores, axis=-1)

                z = v_flat @ do_vec
                mu = (w * z).sum()
                ds = w * (z - mu)
                ds = jnp.where(mask_flat, ds, 0.0)

                dQ = softmax_scale * (ds[:, None] * k_flat).sum(axis=0)
                dK_flat = softmax_scale * (ds[:, None] * q_vec[None, :])
                dV_flat = w[:, None] * do_vec[None, :]

                dV_flat = jnp.where(mask_flat[:, None], dV_flat, 0.0)

                return dQ, dK_flat.reshape(-1, block_size, D), dV_flat.reshape(-1, block_size, D)

            dQ_g, dK_sel_g, dV_sel_g = jax.vmap(head_bwd, in_axes=(0,), out_axes=(0, 0, 0))(jnp.arange(G))

            dK_sel_sum = dK_sel_g.sum(axis=0)
            dV_sel_sum = dV_sel_g.sum(axis=0)

            dk_upd = jnp.zeros((NB, block_size, D))
            dv_upd = jnp.zeros((NB, block_size, D))
            dk_upd = dk_upd.at[inds].add(dK_sel_sum)
            dv_upd = dv_upd.at[inds].add(dV_sel_sum)

            return dQ_g, dk_upd, dv_upd

        dq_heads_t, dk_bt, dv_bt = jax.vmap(token_bwd, in_axes=(0,))(jnp.arange(T))

        dq_b = jnp.zeros((T, HQ, D))
        dq_b = jax.lax.dynamic_update_slice(dq_b, dq_heads_t, (0, hq_start, 0))

        dk_b = dk_bt.sum(axis=0)
        dv_b = dv_bt.sum(axis=0)
        return dq_b, dk_b, dv_b

    dq_b, dk_b, dv_b = jax.vmap(
        jax.vmap(bkvh_backward, in_axes=(None, 0), out_axes=(0, 0, 0)),
        in_axes=(0, None),
        out_axes=(0, 0, 0),
    )(jnp.arange(B), jnp.arange(HKV))

    dq = dq_b.sum(axis=1)

    dk = dk_b.reshape(B, HKV, NB * block_size, D).transpose(0, 2, 1, 3)
    dv = dv_b.reshape(B, HKV, NB * block_size, D).transpose(0, 2, 1, 3)

    dk = dk[:, :T, :, :]
    dv = dv[:, :T, :, :]

    return dq, dk, dv
