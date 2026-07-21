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

"""Pure-JAX reference implementation of the GDN speculative-window state scan.

Speculative decoding on GDN (gated-delta-net) hybrids verifies ``num_steps``
draft tokens in one fused forward. The scheduler later commits the recurrent
state matching the ACCEPTED prefix length, so the model must stash the
per-step ("per accepted prefix") recurrent states for every step of the
window. This op advances the gated-delta-rule recurrence over the window and
returns the state after every step, stacked step-major — exactly the layout
the prefix-major candidate rows use.

The recurrence per step ``t`` (all in float32, matching the sequential decode
path of Qwen3-Next / Qwen3.5 linear attention):

    k_mem   = k_t · S                     (contraction over ``key_dim``)
    v_new   = beta_t * (v_t - g_t * k_mem)
    S       = S * g_t + k_t ⊗ v_new       (only where ``step_valid[t]``)

where ``g_t = exp(-exp(A_log) * softplus(a_t + dt_bias))`` and
``beta_t = sigmoid(b_t)`` are derived from the raw gate projections, and
``k``/``v`` are sliced out of the packed post-conv ``mixed_qkv`` rows
(keys L2-normalised in the input dtype, matching the decode path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def _prep_window_inputs(
    mixed_qkv: Float[Array, "batch num_steps qkv_dim"],
    b: Float[Array, "batch num_steps num_v_heads"],
    a: Float[Array, "batch num_steps num_v_heads"],
    A_log: Float[Array, "num_v_heads"],
    dt_bias: Float[Array, "num_v_heads"],
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    apply_silu: bool,
) -> tuple[Array, Array, Array, Array]:
    """Slice/normalise the packed window rows into per-step GDR inputs.

    Shared between the XLA reference and the Pallas TPU wrapper so both
    backends consume bit-identical ``(key, value, beta, gate)`` tensors.

    Args:
        mixed_qkv: Post-conv packed rows ``[batch, num_steps, qkv_dim]`` where
            ``qkv_dim = 2 * n_kq * d_k + n_v * d_v`` (query block unused).
        b: Raw beta gate rows ``[batch, num_steps, n_v]`` (pre-sigmoid).
        a: Raw alpha gate rows ``[batch, num_steps, n_v]`` (pre-softplus).
        A_log: Per-value-head log-decay parameter ``[n_v]``.
        dt_bias: Per-value-head delta-time bias ``[n_v]``.
        n_kq: Number of key/query heads in ``mixed_qkv``.
        n_v: Number of value heads.
        d_k: Key/query head dim.
        d_v: Value head dim.
        apply_silu: Apply SiLU to ``mixed_qkv`` first (used when the upstream
            ragged conv skipped its fused activation).

    Returns:
        ``(key, value, beta, gate)`` with shapes
        ``[batch, num_steps, n_v, d_k]`` (keys repeated to value heads and
        L2-normalised in the input dtype), ``[batch, num_steps, n_v, d_v]``,
        and two ``[batch, num_steps, n_v]`` float32 gate tensors, where
        ``gate`` is the multiplicative decay ``exp(g)``.
    """
    batch, num_steps = mixed_qkv.shape[:2]
    key_dim = n_kq * d_k
    x = jax.nn.silu(mixed_qkv.astype(jnp.float32)).astype(mixed_qkv.dtype) if apply_silu else mixed_qkv
    key = x[:, :, key_dim : 2 * key_dim].reshape(batch, num_steps, n_kq, d_k)
    value = x[:, :, 2 * key_dim :].reshape(batch, num_steps, n_v, d_v)
    if n_v != n_kq:
        key = jnp.repeat(key, n_v // n_kq, axis=2)
    inv_norm = jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    key = key * inv_norm
    beta = jax.nn.sigmoid(b.astype(jnp.float32))
    g = -jnp.exp(A_log.astype(jnp.float32))[None, None, :] * jax.nn.softplus(
        a.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, None, :]
    )
    gate = jnp.exp(g)
    return key, value, beta, gate


def gdn_spec_window_states(
    mixed_qkv: Float[Array, "batch num_steps qkv_dim"],
    b: Float[Array, "batch num_steps num_v_heads"],
    a: Float[Array, "batch num_steps num_v_heads"],
    A_log: Float[Array, "num_v_heads"],
    dt_bias: Float[Array, "num_v_heads"],
    step_valid: Bool[Array, "batch num_steps"],
    recurrent_state: Float[Array, "batch num_v_heads key_dim value_dim"],
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    apply_silu: bool = False,
) -> Float[Array, "num_steps batch num_v_heads key_dim value_dim"]:
    """Advance the GDN recurrence over a speculative window, returning every prefix state.

    Args:
        mixed_qkv: Post-conv packed rows for the window tokens,
            ``[batch, num_steps, 2 * n_kq * d_k + n_v * d_v]``.
        b: Raw beta gate rows ``[batch, num_steps, n_v]``.
        a: Raw alpha gate rows ``[batch, num_steps, n_v]``.
        A_log: Per-value-head log-decay parameter ``[n_v]``.
        dt_bias: Per-value-head delta-time bias ``[n_v]``.
        step_valid: ``[batch, num_steps]`` mask; the state is carried
            unchanged through invalid steps (their row inputs are ignored).
        recurrent_state: Window-entry state
            ``[batch, n_v, d_k, d_v]``; its dtype is the storage dtype of the
            returned stack (compute is float32).
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key/query head dim.
        d_v: Value head dim.
        apply_silu: Apply SiLU to ``mixed_qkv`` before slicing.

    Returns:
        ``[num_steps, batch, n_v, d_k, d_v]`` — the recurrent state after each
        window step (step-major, matching prefix-major candidate rows), cast
        back to ``recurrent_state.dtype``.
    """
    key, value, beta, gate = _prep_window_inputs(
        mixed_qkv, b, a, A_log, dt_bias, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v, apply_silu=apply_silu
    )
    num_steps = mixed_qkv.shape[1]
    state = recurrent_state.astype(jnp.float32)
    stacked = []
    for t in range(num_steps):
        key_t = key[:, t].astype(jnp.float32)
        value_t = value[:, t].astype(jnp.float32)
        beta_t = beta[:, t]
        gate_t = gate[:, t]
        valid_t = step_valid[:, t]
        k_mem = jnp.einsum("bhd,bhdm->bhm", key_t, state)
        v_new = beta_t[..., None] * (value_t - gate_t[..., None] * k_mem)
        state_new = state * gate_t[..., None, None] + key_t[..., :, None] * v_new[..., None, :]
        state = jnp.where(valid_t[:, None, None, None], state_new, state)
        stacked.append(state)
    return jnp.stack(stacked, axis=0).astype(recurrent_state.dtype)
