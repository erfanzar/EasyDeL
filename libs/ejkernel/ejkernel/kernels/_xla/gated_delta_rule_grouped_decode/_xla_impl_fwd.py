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

"""Pure-JAX reference implementation of the grouped gated-delta-rule (GDR) decode step.

This module contains the backend-agnostic numerics for a single autoregressive decode step of
grouped gated delta-rule linear attention. It is the body that the registry interface in
:mod:`._interface` wraps with kernel registration and runtime type checking, and it serves as
the XLA fallback when no fused (Pallas/Triton/CUDA) decode kernel is available.

The single public function, :func:`gated_delta_rule_grouped_decode`, advances the linear
recurrent state by one token using the gated delta rule (optionally applying a per-step
log-space decay), reads the attention output for the new token, and returns both the output and
the updated state.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def gated_delta_rule_grouped_decode(
    query: Float[Array, "batch num_k_heads head_dim"],
    key: Float[Array, "batch num_k_heads head_dim"],
    value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
    beta: Float[Array, "batch num_k_heads expand_ratio"],
    decay: Float[Array, "batch num_k_heads expand_ratio"] | None,
    recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
) -> tuple[
    Float[Array, "batch num_v_heads value_dim"],
    Float[Array, "batch num_v_heads head_dim value_dim"],
]:
    """Advance grouped gated-delta-rule linear attention by one decode token in pure JAX.

    Implements a single autoregressive step of grouped GDR linear attention. Every key head is
    shared by ``expand_ratio`` value heads, so ``num_v_heads == num_k_heads * expand_ratio`` and
    the recurrent state is internally viewed as
    ``[batch, num_k_heads, expand_ratio, head_dim, value_dim]``.

    The step does, per ``(batch, key-head, group)``:

        1. Cast ``query``/``key`` to the state dtype and scale ``query`` by ``head_dim ** -0.5``.
        2. If ``decay`` is given, multiply the prior state by ``exp(decay)`` (gated forgetting).
        3. Read the current memory for this key: ``kv_mem = state @ key`` over ``head_dim``.
        4. Form the delta-rule write ``delta = (value - kv_mem) * beta`` and add the outer
           product ``key ⊗ delta`` into the state.
        5. Read the output for the new token: ``output = updated_state @ query`` over ``head_dim``.

    All intermediate math runs in ``recurrent_state.dtype`` (the compute dtype); the outputs are
    cast back to ``recurrent_state.dtype`` (i.e. unchanged) before returning.

    Args:
        query: Query for the new token, ``[batch, num_k_heads, head_dim]``. Cast to the compute
            dtype and scaled by ``head_dim ** -0.5``.
        key: Key for the new token, ``[batch, num_k_heads, head_dim]``. Cast to the compute dtype.
        value: Value for the new token, ``[batch, num_k_heads, expand_ratio, value_dim]``; the
            ``expand_ratio`` axis indexes the value-head groups sharing each key head.
        beta: Per-group delta-rule gating coefficient, ``[batch, num_k_heads, expand_ratio]``,
            multiplying ``(value - kv_mem)`` before it is written into the state.
        decay: Optional per-group log-space decay, ``[batch, num_k_heads, expand_ratio]``. When
            provided, the state is multiplied by ``exp(decay)`` (broadcast over ``head_dim`` and
            ``value_dim``) prior to the update; when ``None`` the state is carried unchanged.
        recurrent_state: Prior linear recurrent state,
            ``[batch, num_v_heads, head_dim, value_dim]``. Its dtype is the compute dtype and is
            preserved on the returned state. Treated as read-only input; the update is returned
            rather than written in place.

    Returns:
        tuple: ``(output, new_state)`` where:
            - ``output`` is the decoded attention output, ``[batch, num_v_heads, value_dim]``,
              cast to ``recurrent_state.dtype``.
            - ``new_state`` is the post-update recurrent state,
              ``[batch, num_v_heads, head_dim, value_dim]``, cast to ``recurrent_state.dtype``,
              to be fed back as ``recurrent_state`` on the next decode step.
    """
    batch, num_k_heads, head_dim = query.shape
    expand_ratio = value.shape[2]
    compute_dtype = recurrent_state.dtype

    query = query.astype(compute_dtype)
    key = key.astype(compute_dtype)
    scale = jnp.asarray(head_dim**-0.5, dtype=compute_dtype)
    query = query * scale

    value_dim = value.shape[-1]
    num_v_heads = num_k_heads * expand_ratio

    gs = recurrent_state.astype(compute_dtype).reshape(batch, num_k_heads, expand_ratio, head_dim, value_dim)
    value_c = value.astype(compute_dtype)
    beta_c = beta.astype(compute_dtype)

    if decay is not None:
        gs = gs * jnp.exp(decay.astype(compute_dtype))[:, :, :, None, None]

    kv_mem = jnp.einsum("bkehv,bkh->bkev", gs, key)
    delta = (value_c - kv_mem) * beta_c[:, :, :, None]
    gs = gs + jnp.einsum("bkh,bkev->bkehv", key, delta)
    output = jnp.einsum("bkehv,bkh->bkev", gs, query)

    return (
        output.reshape(batch, num_v_heads, value_dim).astype(recurrent_state.dtype),
        gs.reshape(batch, num_v_heads, head_dim, value_dim).astype(recurrent_state.dtype),
    )
