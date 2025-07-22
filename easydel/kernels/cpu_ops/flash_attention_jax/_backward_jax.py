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


# Implementation based on FlashAttention 2 (https://arxiv.org/pdf/2307.08691) by @erfanzar,
# with a few bug fixes and adjustments.

import functools

import jax
import jax.numpy as jnp
import jax.sharding
from jax import lax

from easydel.utils.compiling_utils import ejit


@functools.partial(jax.named_call, name="_bwd_flash_attn")
def _bwd_flash_attn(
    dropout: float,
    inference: bool,
    key: jax.random.PRNGKey,
    blocksize_q: int,
    blocksize_k: int,
    dtype: jnp.dtype | None,
    precision: lax.PrecisionLike,
    residuals,
    grad_in: jax.Array,
):
    """Backward pass of FlashAttention."""

    del dtype
    (
        O,  # noqa: E741
        L,
        query_state,
        key_state,
        value_state,
        mask,
        bias,
    ) = residuals
    dO = grad_in

    b, h, _, d = query_state.shape
    q_seq = query_state.shape[2]
    k_seq = key_state.shape[2]
    assert q_seq % blocksize_q == 0
    assert k_seq % blocksize_k == 0
    Tr = q_seq // blocksize_q
    Tc = k_seq // blocksize_k

    D = jnp.sum(dO * O, axis=-1)

    dQ = (query_state * 0.0).astype(query_state.dtype)
    dK = (key_state * 0.0).astype(key_state.dtype)
    dV = (value_state * 0.0).astype(value_state.dtype)
    global_mask = mask
    is_causal = mask is not None

    @ejit
    @functools.partial(jax.named_call, name="_bwd_flash_attn_call_o")
    def call_o(state):
        j, dQ, dK, dV = state
        k_j = jax.lax.dynamic_slice_in_dim(key_state, j * blocksize_k, blocksize_k, 2)
        v_j = jax.lax.dynamic_slice_in_dim(value_state, j * blocksize_k, blocksize_k, 2)

        dK_j = jax.lax.dynamic_slice_in_dim(dK, j * blocksize_k, blocksize_k, 2)
        dV_j = jax.lax.dynamic_slice_in_dim(dV, j * blocksize_k, blocksize_k, 2)

        @ejit
        @functools.partial(jax.named_call, name="_bwd_flash_attn_call_o_call_qk")
        def do_inner_block(state):
            i, j, dQ, dK_j, dV_j = state
            q_i = jax.lax.dynamic_slice_in_dim(query_state, i * blocksize_q, blocksize_q, 2)
            dQ_i = jax.lax.dynamic_slice_in_dim(dQ, i * blocksize_q, blocksize_q, 2)
            dO_i = jax.lax.dynamic_slice_in_dim(dO, i * blocksize_q, blocksize_q, 2)

            L_i = jax.lax.dynamic_slice_in_dim(L, i * blocksize_q, blocksize_q, 2)
            D_i = jax.lax.dynamic_slice_in_dim(D, i * blocksize_q, blocksize_q, 2)
            s_ij = q_i @ k_j.transpose(0, 1, 3, 2)
            if dropout > 0 and not inference:
                rng = jax.random.fold_in(key, i * Tc + j)
                keep_prob = 1.0 - dropout
                broadcast_shape = list(s_ij.shape)
                mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
                mask = jnp.broadcast_to(mask, s_ij.shape)
                s_ij = lax.select(mask, s_ij / keep_prob, jnp.zeros_like(s_ij))

            if bias is not None:
                b_i = jax.lax.dynamic_slice_in_dim(bias, i * blocksize_q, blocksize_q, 2)
                b_ij = jax.lax.dynamic_slice_in_dim(b_i, j * blocksize_k, blocksize_k, 3)
                s_ij = s_ij + b_ij

            if global_mask is not None:
                ma_i = jax.lax.dynamic_slice_in_dim(global_mask, i * blocksize_q, blocksize_q, 2)
                ma_ij = jax.lax.dynamic_slice_in_dim(ma_i, j * blocksize_k, blocksize_k, 3)
                s_ij = jnp.where(ma_ij, s_ij, -1e10)

            p_ij = jnp.exp(s_ij - jnp.expand_dims(L_i, -1))

            if dropout > 0 and not inference:
                rng = jax.random.fold_in(key, i * Tc + j)
                keep_prob = 1.0 - dropout
                broadcast_shape = list(p_ij.shape)
                mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
                mask = jnp.broadcast_to(mask, p_ij.shape)
                p_ij = lax.select(mask, p_ij / keep_prob, jnp.zeros_like(p_ij))

            dV_j = dV_j + jnp.matmul(p_ij.transpose(0, 1, 3, 2), dO_i, precision=precision)

            dP_ij = jnp.matmul(dO_i, v_j.transpose(0, 1, 3, 2), precision=precision)

            dS_ij = p_ij * (dP_ij - D_i[..., None])
            dQ_i = dQ_i + jnp.matmul(dS_ij, k_j, precision=precision)
            dK_j = dK_j + jnp.matmul(dS_ij.transpose(0, 1, 3, 2), q_i, precision=precision)
            dQ = jax.lax.dynamic_update_slice_in_dim(
                dQ,
                dQ_i.astype(dQ.dtype),
                i * blocksize_q,
                2,
            )
            return (
                i + 1,
                j,
                dQ.astype(query_state.dtype),
                dK_j.astype(key_state.dtype),
                dV_j.astype(value_state.dtype),
            )

        i_start = j if is_causal else 0
        _, j, dQ, dK_j, dV_j = jax.lax.while_loop(
            lambda state: state[0] < Tr,
            do_inner_block,
            (i_start, j, dQ, dK_j, dV_j),
        )

        dK = jax.lax.dynamic_update_slice_in_dim(dK, dK_j.astype(dK.dtype), j * blocksize_q, 2)
        dV = jax.lax.dynamic_update_slice_in_dim(dV, dV_j.astype(dV.dtype), j * blocksize_q, 2)

        return j + 1, dQ, dK, dV

    _, dQ, dK, dV = jax.lax.while_loop(
        lambda state: state[0] < Tc,
        call_o,
        (0, dQ, dK, dV),
    )

    return dQ, dK, dV, None, None
