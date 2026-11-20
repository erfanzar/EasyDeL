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


"""XLA reference for the fused KDA decode step: causal-conv window update, conv, SiLU, L2 norm and recurrence.

Token-major contract (one token per advancing row, as in an eSurge pure-decode step): token ``t`` of the packed stream
advances pool slot ``token_slots[t]`` (``< 0`` or ``>= S``: the token outputs zeros and changes nothing), starting from
the slot's windows and state when ``token_init[t]`` is set and from zeros otherwise. Per advancing token and for each of
q, k and v, the slot's conv window ``[dim, d_conv]`` is shifted by the raw token, dotted with the depthwise kernel in
fp32 (``t0 * w0``, then ``+= t_j * w_j``), passed through SiLU and rounded to the compute and conv dtypes; the shifted
window is written back. The conv outputs then take one ``ragged_kda_decode`` step (``S = S * exp(g)``,
``delta = beta * (v - k^T S)``, ``S += k delta^T``, ``o = q^T S``, q / k L2-normalised and q scaled by ``K**-0.5``).
This is the arithmetic of EasyDeL's ``kda_decode_rows_forward`` (its conv glue plus the decode operation).
"""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp
from jaxtyping import Array

from ..ragged_kda._xla_impl import conv_compute_dtype
from ..ragged_kda_decode._xla_impl import ragged_kda_decode_xla


def validate_shapes(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    conv_weights: tp.Sequence[Array],
    conv_pools: tp.Sequence[Array],
    recurrent_state: Array,
    token_slots: Array,
    token_init: Array,
) -> None:
    """Check the ``ragged_kda_conv_decode`` array contract.

    Args:
        query: ``[T, H*K]``.
        key: ``[T, H*K]``.
        value: ``[T, H*V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]``.
        conv_weights: ``(q [H*K, dc], k [H*K, dc], v [H*V, dc])``.
        conv_pools: ``(q [S, H*K, dc], k [S, H*K, dc], v [S, H*V, dc])``.
        recurrent_state: ``[S, H, K, V]``.
        token_slots: ``[T]``.
        token_init: ``[T]``.

    Raises:
        ValueError: If any shape disagrees with the contract above.
    """
    if query.ndim != 2 or key.shape != query.shape or value.ndim != 2 or value.shape[0] != query.shape[0]:
        raise ValueError(
            f"query / key must be [T, H*K] and value [T, H*V], got {query.shape}, {key.shape}, {value.shape}"
        )
    num_tokens = query.shape[0]
    if beta.ndim != 2 or beta.shape[0] != num_tokens:
        raise ValueError(f"beta must be [{num_tokens}, H], got {beta.shape}")
    num_heads = beta.shape[1]
    if query.shape[1] % num_heads or value.shape[1] % num_heads:
        raise ValueError(f"{num_heads} heads do not divide the widths {query.shape[1]} / {value.shape[1]}")
    head_dim, v_dim = query.shape[1] // num_heads, value.shape[1] // num_heads
    if decay.shape != (num_tokens, num_heads, head_dim):
        raise ValueError(f"decay must be per-channel {(num_tokens, num_heads, head_dim)}, got {decay.shape}")
    if recurrent_state.ndim != 4 or recurrent_state.shape[1:] != (num_heads, head_dim, v_dim):
        raise ValueError(f"recurrent_state must be [S, {num_heads}, {head_dim}, {v_dim}], got {recurrent_state.shape}")
    num_slots = recurrent_state.shape[0]
    if len(conv_weights) != 3 or len(conv_pools) != 3:
        raise ValueError("conv_weights and conv_pools must hold (q, k, v)")
    d_conv = conv_weights[0].shape[-1]
    for dim, w, p in zip((query.shape[1], key.shape[1], value.shape[1]), conv_weights, conv_pools, strict=True):
        if w.shape != (dim, d_conv):
            raise ValueError(f"conv weight must be [{dim}, {d_conv}], got {w.shape}")
        if p.shape != (num_slots, dim, d_conv):
            raise ValueError(f"conv pool must be [{num_slots}, {dim}, {d_conv}], got {p.shape}")
    if token_slots.shape != (num_tokens,) or token_init.shape != (num_tokens,):
        raise ValueError(f"token_slots / token_init must be [{num_tokens}], got {token_slots.shape}, {token_init.shape}")


def ragged_kda_conv_decode_xla(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    conv_weights: tp.Sequence[Array],
    conv_pools: tp.Sequence[Array],
    recurrent_state: Array,
    token_slots: Array,
    token_init: Array,
    *,
    conv_dtype: tp.Any,
    use_qk_l2norm: bool = True,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Fused KDA decode step over the packed token stream (XLA reference; see the module docstring).

    Args:
        query: Raw (pre-conv) queries ``[T, H*K]``.
        key: Raw keys ``[T, H*K]``.
        value: Raw values ``[T, H*V]``.
        beta: Write strengths ``[T, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[T, H, K]``.
        conv_weights: Depthwise conv kernels ``(q, k, v)``, each ``[dim, d_conv]`` (last tap = current token).
        conv_pools: Conv windows ``(q, k, v)``, each ``[S, dim, d_conv]``.
        recurrent_state: State pool ``[S, H, K, V]``.
        token_slots: ``[T]`` pool slot advanced by token ``t`` (out of range: none).
        token_init: ``[T]`` whether that slot's windows and state carry over (``False``: zeros).
        conv_dtype: Dtype of the conv outputs.
        use_qk_l2norm: L2-normalise queries and keys.

    Returns:
        ``(out [T, H, V] fp32, (q_pool, k_pool, v_pool, recurrent_state))``.
    """
    f32 = jnp.float32
    num_tokens = query.shape[0]
    num_heads = beta.shape[1]
    head_dim, v_dim = query.shape[1] // num_heads, value.shape[1] // num_heads
    num_slots = recurrent_state.shape[0]
    slots = jnp.asarray(token_slots).astype(jnp.int32)
    live = (slots >= 0) & (slots < num_slots)
    init = jnp.asarray(token_init).astype(jnp.bool_)
    read = jnp.clip(slots, 0, num_slots - 1)
    write = jnp.where(live, slots, num_slots)

    def conv(stream: Array, kernel: Array, pool: Array) -> tuple[Array, Array]:
        compute = conv_compute_dtype(stream.dtype, pool.dtype)
        window = jnp.where(init[:, None, None], pool[read].astype(compute), jnp.zeros((), compute))
        new_window = jnp.concatenate([window[:, :, 1:], stream.astype(compute)[:, :, None]], axis=-1)
        taps = new_window.astype(f32)
        weights = kernel.astype(f32)
        mixed = taps[:, :, 0] * weights[None, :, 0]
        for j in range(1, int(kernel.shape[-1])):
            mixed = mixed + taps[:, :, j] * weights[None, :, j]
        out = jax.nn.silu(mixed).astype(compute).astype(conv_dtype)
        return out, pool.at[write].set(new_window.astype(pool.dtype), mode="drop")

    q, q_pool = conv(query, conv_weights[0], conv_pools[0])
    k, k_pool = conv(key, conv_weights[1], conv_pools[1])
    v, v_pool = conv(value, conv_weights[2], conv_pools[2])
    out, rec = ragged_kda_decode_xla(
        q.reshape(num_tokens, num_heads, head_dim),
        k.reshape(num_tokens, num_heads, head_dim),
        v.reshape(num_tokens, num_heads, v_dim),
        beta.astype(f32) if jnp.dtype(beta.dtype).itemsize <= 1 else beta,
        decay,
        recurrent_state,
        slots,
        live,
        init,
        use_qk_l2norm=use_qk_l2norm,
    )
    return out, (q_pool, k_pool, v_pool, rec)


__all__ = ("ragged_kda_conv_decode_xla", "validate_shapes")
