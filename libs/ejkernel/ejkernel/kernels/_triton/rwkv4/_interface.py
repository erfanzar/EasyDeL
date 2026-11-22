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

"""RWKV-4 recurrent time-mix kernel implementation using Triton.

This module provides a GPU-accelerated implementation of the RWKV-4 time-mix
recurrence mechanism. RWKV-4 is a linear attention variant that replaces the
quadratic attention mechanism with a recurrent formulation, achieving O(N)
complexity for sequence processing.

RWKV-4 computes attention through a weighted combination of key-value pairs
using exponential decay, where the decay rate is learned per channel. The
model maintains a running state that accumulates information from previous
tokens, allowing efficient sequential processing.

Key components:
- Time decay (w): Controls how quickly past information decays, stored in
  log-space for numerical stability
- Time-mix bias (u): Adds a direct contribution from the current timestep
- State: A 3-tuple (alpha, beta, eps) tracking the accumulated numerator,
  denominator, and max logit for numerical stability

Algorithm:
    For each timestep t:
        wkv_t = (exp(u + k_t) * v_t + alpha_t) / (exp(u + k_t) + beta_t)
        alpha_{t+1} = exp(w) * alpha_t + exp(k_t) * v_t
        beta_{t+1} = exp(w) * beta_t + exp(k_t)

Key features:
- O(N) time complexity for sequence processing
- Numerically stable through log-space computation
- Custom Triton kernel for GPU acceleration
- Stateful processing for autoregressive generation
- Compatible with standard RWKV model architectures

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.rwkv4 import rwkv4
    >>>
    >>> batch, seq_len, channels = 2, 1024, 512
    >>> w = jnp.zeros((channels,))  # Time decay parameters
    >>> u = jnp.zeros((channels,))  # Time-mix bias
    >>> k = jnp.ones((batch, seq_len, channels))
    >>> v = jnp.ones((batch, seq_len, channels))
    >>>
    >>> output, final_state = rwkv4(w, u, k, v)

Reference:
    RWKV: Reinventing RNNs for the Transformer Era
    https://arxiv.org/abs/2305.13048
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.interpreters import ad
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_bwd import bwd_triton_impl
from ._triton_impl_fwd import fwd_triton_impl, fwd_triton_impl_with_history


def _fwd_call(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None,
):
    """Forward pass for RWKV-4 time-mix in a custom VJP.

    Computes the RWKV-4 recurrence and saves residuals for backward pass.
    Initializes state if not provided.

    Args:
        w: Time decay parameter in log-space of shape `[C]`.
        u: Time-mix bias parameter of shape `[C]`.
        k: Key tensor of shape `[B, T, C]`.
        v: Value tensor of shape `[B, T, C]`.
        state: Optional initial state `[B, 3, C]` containing (alpha, beta, eps).

    Returns:
        A tuple containing (output, final_state) and residuals for backward.
    """
    state_input = state
    state_was_none = state is None
    if state is None:
        bsz, _, chans = k.shape
        alpha0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        beta0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        eps0 = jnp.full((bsz, chans), -1e30, dtype=jnp.float32)
        state = jnp.stack([alpha0, beta0, eps0], axis=1)

    w_neg = -jnp.exp(w.astype(jnp.float32))
    o, final_state, state_hist = fwd_triton_impl_with_history(
        w_neg, u.astype(jnp.float32), k, v, state.astype(jnp.float32)
    )
    residual = (w, u, k, v, w_neg, state_hist, state_was_none, state_input)
    return (o, final_state), residual


def _bwd_call(
    residual,
    grads,
):
    """Backward pass for RWKV-4 time-mix in a custom VJP.

    Args:
        residual: Tensors saved from the forward pass (w, u, k, v, state).
        grads: A tuple containing gradients (do, dstate) of output and final state.
    """
    w, u, k, v, w_neg, state_hist, state_was_none, state_input = residual
    do, dstate = grads

    do = ad.instantiate_zeros(do)
    dstate = ad.instantiate_zeros(dstate)
    if dstate is None:
        dstate = jnp.zeros((k.shape[0], 3, k.shape[2]), dtype=jnp.float32)
    else:
        dstate = dstate.astype(jnp.float32)

    dw_neg, du, dk, dv, dstate0 = bwd_triton_impl(
        w_neg=w_neg,
        u=u.astype(jnp.float32),
        k=k,
        v=v,
        state_hist=state_hist,
        do=do,
        dstate=dstate,
    )
    dw = (dw_neg * w_neg).astype(w.dtype)
    du = du.astype(u.dtype)
    dk = dk.astype(k.dtype)
    dv = dv.astype(v.dtype)

    if state_was_none:
        state_cotangent = None
    else:
        state_cotangent = dstate0.astype(state_input.dtype)
    return dw, du, dk, dv, state_cotangent


@partial(jax.custom_vjp)
def _rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
) -> tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """Core RWKV-4 function with custom VJP.

    This internal function directly calls the Triton implementation and is
    registered with JAX's custom differentiation system for memory-efficient
    gradient computation.

    Args:
        w: Time decay parameter in log-space of shape `[C]`.
        u: Time-mix bias parameter of shape `[C]`.
        k: Key tensor of shape `[B, T, C]`.
        v: Value tensor of shape `[B, T, C]`.
        state: Optional initial state `[B, 3, C]` containing (alpha, beta, eps).

    Returns:
        A tuple containing:
            - output: The WKV output tensor of shape `[B, T, C]`.
            - final_state: The final recurrent state of shape `[B, 3, C]`.
    """
    if state is None:
        bsz, _, chans = k.shape
        alpha0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        beta0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        eps0 = jnp.full((bsz, chans), -1e30, dtype=jnp.float32)
        state = jnp.stack([alpha0, beta0, eps0], axis=1)

    w_neg = -jnp.exp(w.astype(jnp.float32))
    return fwd_triton_impl(w_neg, u.astype(jnp.float32), k, v, state.astype(jnp.float32))


_rwkv4.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("rwkv4", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
    *,
    block_c: int = 64,
) -> tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """RWKV-4 time-mix recurrence (Triton GPU implementation).

    Args:
        w: Time-decay parameter in log space `[C]`.
        u: Time-mix bias `[C]`.
        k: Key tensor `[B, T, C]`.
        v: Value tensor `[B, T, C]`.
        state: Optional initial state `[B, 3, C]` (alpha, beta, eps).
        block_c: Accepted for signature parity with the tilelang backend;
            the Triton implementation chooses its own meta-parameters.

    Returns:
        Tuple of (output `[B, T, C]`, final_state `[B, 3, C]`).
    """
    del block_c
    return _rwkv4(w, u, k, v, state)
