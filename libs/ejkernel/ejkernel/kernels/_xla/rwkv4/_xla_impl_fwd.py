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

"""RWKV-4 recurrent time-mix kernel (XLA).

This module provides a pure JAX/XLA implementation of the RWKV-4 time-mix
recurrence, compatible with Flash-Linear-Attention's fused recurrent operation.
RWKV-4 achieves O(N) complexity through a numerically-stable recurrent formulation
that avoids the quadratic attention matrix.

RWKV-4 is a recurrent language model architecture that combines the efficient
parallelizable training of Transformers with the efficient inference of RNNs.
The time-mix mechanism uses exponential moving averages with learned decay rates.

State layout follows the common RWKV-4 formulation:
    state[:, 0, :] = alpha (numerator accumulator)
    state[:, 1, :] = beta (denominator accumulator)
    state[:, 2, :] = eps (log-scale normalization term)

The core recurrence computes WKV (Weighted Key-Value) attention:
    tau_t = max(u + k_t, eps_{t-1})
    wkv_t = (exp(eps_{t-1} - tau_t) * alpha_{t-1} + exp(u + k_t - tau_t) * v_t) /
            (exp(eps_{t-1} - tau_t) * beta_{t-1} + exp(u + k_t - tau_t))

    eps_t = max(w + eps_{t-1}, k_t)
    alpha_t = exp(w + eps_{t-1} - eps_t) * alpha_{t-1} + exp(k_t - eps_t) * v_t
    beta_t = exp(w + eps_{t-1} - eps_t) * beta_{t-1} + exp(k_t - eps_t)

Where:
    - w is the time-decay parameter (log space, uses -exp(w) internally)
    - u is the bonus/bias for current token attention
    - The log-sum-exp trick via eps ensures numerical stability

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.rwkv4 import rwkv4
    >>>
    >>> batch, seq_len, channels = 2, 100, 512
    >>> w = jnp.zeros(channels)  # time decay
    >>> u = jnp.zeros(channels)  # bonus
    >>> k = jnp.ones((batch, seq_len, channels))
    >>> v = jnp.ones((batch, seq_len, channels))
    >>>
    >>> wkv, final_state = rwkv4(w, u, k, v)
    >>> wkv.shape
    (2, 100, 512)
    >>> final_state.shape
    (2, 3, 512)

Reference:
    RWKV: Reinventing RNNs for the Transformer Era
    https://arxiv.org/abs/2305.13048
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _rwkv4_step(
    carry: tuple[Array, Array, Array],
    x: tuple[Array, Array],
    *,
    w: Array,
    u: Array,
) -> tuple[tuple[Array, Array, Array], Array]:
    """Single step of the RWKV-4 recurrence with log-sum-exp stability.

    Computes one timestep of the RWKV-4 time-mix mechanism using the
    numerically stable formulation that tracks a log-scale normalization
    term (eps) to prevent overflow/underflow.

    Args:
        carry: Tuple of (alpha, beta, eps) state tensors, each [B, C]:
            - alpha: Numerator accumulator (weighted sum of values)
            - beta: Denominator accumulator (sum of weights)
            - eps: Log-scale normalization term for stability
        x: Tuple of (k_t, v_t) for current timestep, each [B, C]:
            - k_t: Key vector at time t
            - v_t: Value vector at time t
        w: Time-decay parameter in log space [C], uses -exp(w) internally
        u: Bonus/bias for current token attention [C]

    Returns:
        Tuple of:
            - Updated (alpha_next, beta_next, eps_next) state
            - wkv_t: Output for current timestep [B, C]
    """
    alpha, beta, eps = carry
    kt, vt = x

    ukt = u + kt
    tau = jnp.maximum(ukt, eps)
    e1a = jnp.exp(eps - tau)
    e2a = jnp.exp(ukt - tau)
    wkv = (e1a * alpha + e2a * vt) / (e1a * beta + e2a)

    w_eps = w + eps
    eps_next = jnp.maximum(w_eps, kt)
    e1b = jnp.exp(w_eps - eps_next)
    e2b = jnp.exp(kt - eps_next)
    alpha_next = e1b * alpha + e2b * vt
    beta_next = e1b * beta + e2b

    return (alpha_next, beta_next, eps_next), wkv


def rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
) -> tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """RWKV-4 recurrent time-mix attention using XLA backend.

    Implements the RWKV-4 time-mixing mechanism which computes a weighted
    key-value (WKV) attention using an O(N) recurrent formulation. This
    replaces the O(N^2) attention matrix with a sequential scan that
    maintains running statistics.

    The mechanism uses exponential decay controlled by `w` and adds a
    bonus term `u` for the current token's contribution. The computation
    is numerically stabilized using log-sum-exp tracking.

    Args:
        w: Time-decay parameter in log space. The actual decay used is
            -exp(w), so larger w values mean faster decay (less memory
            of past tokens). Shape: [channels]
        u: Time-mix bonus/bias that controls how much the current token
            contributes relative to historical context. Shape: [channels]
        k: Key tensor representing input features to match against.
            Shape: [batch, seq_len, channels]
        v: Value tensor representing content to aggregate.
            Shape: [batch, seq_len, channels]
        state: Optional initial state tuple packed as [B, 3, C] where:
            - state[:, 0, :] = alpha (numerator accumulator)
            - state[:, 1, :] = beta (denominator accumulator)
            - state[:, 2, :] = eps (log-scale normalization)
            If None, initializes with alpha=0, beta=0, eps=-1e30.

    Returns:
        Tuple of:
            - wkv: Weighted key-value output matching input dtype.
                Shape: [batch, seq_len, channels]
            - final_state: Final state for continuation, always float32.
                Shape: [batch, 3, channels]

    Raises:
        ValueError: If k and v shapes don't match, or dimension mismatches.

    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> # Basic usage
        >>> batch, seq_len, channels = 2, 100, 512
        >>> w = -jnp.ones(channels) * 0.5  # moderate decay
        >>> u = jnp.zeros(channels)
        >>> k = jnp.ones((batch, seq_len, channels))
        >>> v = jnp.ones((batch, seq_len, channels))
        >>>
        >>> wkv, state = rwkv4(w, u, k, v)
        >>> wkv.shape
        (2, 100, 512)
        >>>
        >>> # Incremental inference with state
        >>> k_next = jnp.ones((batch, 1, channels))
        >>> v_next = jnp.ones((batch, 1, channels))
        >>> wkv_next, state_next = rwkv4(w, u, k_next, v_next, state=state)
    """
    if k.shape != v.shape:
        raise ValueError(f"`k` and `v` must have the same shape, got k={k.shape}, v={v.shape}.")
    if k.ndim != 3:
        raise ValueError(f"`k` must be rank-3 [B, T, C], got shape {k.shape}.")
    if w.ndim != 1 or u.ndim != 1:
        raise ValueError(f"`w` and `u` must be rank-1 [C], got w={w.shape}, u={u.shape}.")
    if w.shape[0] != k.shape[-1] or u.shape[0] != k.shape[-1]:
        raise ValueError(f"Channel dim mismatch: C={k.shape[-1]}, w={w.shape}, u={u.shape}.")

    orig_dtype = v.dtype
    w_f32 = -jnp.exp(w.astype(jnp.float32))
    u_f32 = u.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    bsz, _, chans = k.shape
    if state is None:
        alpha0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        beta0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        eps0 = jnp.full((bsz, chans), -1e30, dtype=jnp.float32)
    else:
        if state.shape != (bsz, 3, chans):
            raise ValueError(f"`state` must have shape [B, 3, C]={(bsz, 3, chans)}, got {state.shape}.")
        alpha0 = state[:, 0, :].astype(jnp.float32)
        beta0 = state[:, 1, :].astype(jnp.float32)
        eps0 = state[:, 2, :].astype(jnp.float32)

    xs = (jnp.swapaxes(k_f32, 0, 1), jnp.swapaxes(v_f32, 0, 1))
    (alphaT, betaT, epsT), wkvT = jax.lax.scan(
        lambda carry, x: _rwkv4_step(carry, x, w=w_f32, u=u_f32),
        (alpha0, beta0, eps0),
        xs,
    )
    wkv = jnp.swapaxes(wkvT, 0, 1).astype(orig_dtype)
    final_state = jnp.stack([alphaT, betaT, epsT], axis=1)
    return wkv, final_state
