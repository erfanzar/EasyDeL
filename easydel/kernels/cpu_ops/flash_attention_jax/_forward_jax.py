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

import functools

import jax
import jax.numpy as jnp
import jax.sharding
from eformer.escale import with_sharding_constraint
from jax import lax

from easydel.utils.compiling_utils import ejit


@functools.partial(jax.named_call, name="_fwd_flash_attn")
def _fwd_flash_attn(
    query_state: jax.Array,
    key_state: jax.Array,
    value_state: jax.Array,
    mask: jax.Array | None,
    bias: jax.Array | None,
    dropout: float,
    inference: bool,
    key: jax.random.PRNGKey,
    blocksize_q: int,
    blocksize_k: int,
    dtype: jnp.dtype | None,
    precision: lax.PrecisionLike,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Forward pass of FlashAttention."""
    b, h, _, d = query_state.shape
    q_seq = query_state.shape[2]
    k_seq = key_state.shape[2]
    assert q_seq % blocksize_q == 0, "Query sequence length is not visible by queryblock size"
    assert k_seq % blocksize_k == 0, "Key sequence length is not visible by keyblock size"
    Tr = q_seq // blocksize_q
    Tc = k_seq // blocksize_k
    o_shape = jax.eval_shape(lambda: (query_state @ key_state.transpose(0, 1, 3, 2)) @ value_state).shape
    o = jnp.zeros(o_shape, dtype=dtype)

    lse = jnp.full((b, h, q_seq), fill_value=-jnp.inf, dtype=jnp.float32)
    if hasattr(query_state, "sharding"):
        if isinstance(query_state.sharding, jax.sharding.NamedSharding):
            with query_state.sharding.mesh:
                o = with_sharding_constraint(
                    arr=o,
                    sharding=query_state.sharding,
                )
                lse = with_sharding_constraint(
                    arr=lse,
                    sharding=jax.sharding.PartitionSpec(*query_state.sharding.spec[:3]),
                )
        elif isinstance(query_state.sharding, jax.sharding.SingleDeviceSharding) and hasattr(
            query_state.sharding, "_device"
        ):
            o = jax.device_put(o, query_state.sharding._device)
            lse = jax.device_put(lse, query_state.sharding._device)

    global_mask = mask

    @ejit
    @functools.partial(jax.named_call, name="_fwd_flash_attn_call_o")
    def call_o(state):
        i, o, lse = state
        q_i = jax.lax.dynamic_slice_in_dim(query_state, i * blocksize_q, blocksize_q, 2)
        o_i = jax.lax.dynamic_slice_in_dim(o, i * blocksize_q, blocksize_q, 2)
        lse_i = jax.lax.dynamic_slice_in_dim(lse, i * blocksize_q, blocksize_q, 2)
        m_i = jnp.full((b, h, blocksize_q), fill_value=-jnp.inf, dtype=dtype)

        @ejit
        @functools.partial(jax.named_call, name="_fwd_flash_attn_call_o_call_qk")
        def call_qk(state):
            i, j, o_i, q_i, lse_i, m_i = state
            k_j = jax.lax.dynamic_slice_in_dim(key_state, j * blocksize_k, blocksize_k, 2)
            v_j = jax.lax.dynamic_slice_in_dim(value_state, j * blocksize_k, blocksize_k, 2)

            s_ij = jnp.einsum(
                "bhqd,bhdk->bhqk",
                q_i,
                k_j.transpose(0, 1, 3, 2),
                precision=precision,
            )

            if bias is not None:
                b_i = jax.lax.dynamic_slice_in_dim(bias, i * blocksize_q, blocksize_q, 2)
                b_ij = jax.lax.dynamic_slice_in_dim(b_i, j * blocksize_k, blocksize_k, 3)
                s_ij = s_ij + b_ij
            if global_mask is not None:
                ma_i = jax.lax.dynamic_slice_in_dim(global_mask, i * blocksize_q, blocksize_q, 2)
                ma_ij = jax.lax.dynamic_slice_in_dim(ma_i, j * blocksize_k, blocksize_k, 3)
                s_ij = jnp.where(ma_ij, s_ij, -1e10)

            if dropout > 0 and not inference:
                rng = jax.random.fold_in(key, i * Tc + j)
                keep_prob = 1.0 - dropout
                broadcast_shape = list(s_ij.shape)
                mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
                mask = jnp.broadcast_to(mask, s_ij.shape)
                s_ij = lax.select(mask, s_ij / keep_prob, jnp.zeros_like(s_ij))

            m_ij = jnp.maximum(m_i, jnp.max(s_ij, axis=-1))
            p = jnp.exp(s_ij - jnp.expand_dims(m_ij, -1))

            l_ij = jnp.sum(p, -1)

            o_scale = jnp.exp(m_i - m_ij)
            o_i = o_i * jnp.expand_dims(o_scale, -1)

            o_i = o_i + jnp.einsum(
                "bhqk,bhkd->bhqd",
                p,
                v_j,
                precision=precision,
            )

            return (
                i,
                j + 1,
                o_i.astype(dtype),
                q_i.astype(dtype),
                jnp.log(jnp.exp(lse_i - m_ij) + l_ij) + m_ij,
                m_ij.astype(dtype),
            )

        j_end = jnp.minimum(i + 1, Tc) if mask is not None else Tc

        _, _, o_i, _, lse_i, m_i = jax.lax.while_loop(
            lambda state: state[1] < j_end,
            call_qk,
            (i, 0, o_i, q_i, lse_i, m_i),
        )
        o_scale = jnp.exp(m_i - lse_i)
        o_i = o_i * jnp.expand_dims(o_scale, -1)

        o = jax.lax.dynamic_update_slice_in_dim(o, o_i.astype(o.dtype), i * blocksize_q, 2)
        lse = jax.lax.dynamic_update_slice_in_dim(
            lse,
            lse_i.astype(lse.dtype),
            i * blocksize_q,
            2,
        )
        return i + 1, o, lse

    _, o, lse = jax.lax.while_loop(lambda state: state[0] < Tr, call_o, (0, o, lse))

    return o, (
        o,
        lse,
        query_state,  #: jax.Array
        key_state,  #: jax.Array
        value_state,  #: jax.Array
        mask,
        bias,
    )
