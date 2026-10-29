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

"""Shared building blocks for the XLA per-row fused distillation losses.

Every helper takes an optional ``vocab_axis``. With ``None`` the vocabulary is
whole on the device and the helper is plain dense JAX. With a mesh-axis name the
inputs are per-shard ``(..., V_local)`` slices and the helper must run inside
``jax.shard_map``: normalizers are merged with an online-softmax ``pmax`` +
``psum`` so the ``[..., V]`` tensor is never gathered.

``pmax`` has no JAX differentiation rule, and the global maximum only
re-centers the exponentials (the value is invariant to it), so every ``pmax``
input here is ``stop_gradient``-ed; autodiff then flows through the
differentiable ``psum`` collectives only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

REDUCTIONS = ("none", "sum", "mean")


def check_reduction(reduction: str) -> None:
    """Raise ``ValueError`` unless ``reduction`` is ``"none"``, ``"sum"`` or ``"mean"``."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"Invalid reduction '{reduction}'; expected one of none/sum/mean.")


def check_temperature(temperature: float) -> None:
    """Raise ``ValueError`` unless ``temperature`` is strictly positive."""
    if not temperature > 0.0:
        raise ValueError(f"temperature must be positive; got {temperature}")


def row_weights(weights, leading: tuple[int, ...]):
    """Return float32 per-row weights, defaulting to ones.

    Args:
        weights: ``None`` or an array shaped like the logits' leading dims.
        leading: ``logits.shape[:-1]``.

    Raises:
        ValueError: If ``weights`` is given and its shape differs from ``leading``.
    """
    if weights is None:
        return jnp.ones(leading, dtype=jnp.float32)
    if tuple(weights.shape) != tuple(leading):
        raise ValueError(f"weights.shape={weights.shape} must equal logits.shape[:-1]={leading}")
    return weights.astype(jnp.float32)


def reduce_rows(per_row, wts, reduction: str):
    """Reduce already-weighted per-row values; ``"mean"`` divides by ``max(sum(wts), 1e-8)``."""
    if reduction == "none":
        return per_row
    total = jnp.sum(per_row)
    if reduction == "sum":
        return total
    return total / jnp.maximum(jnp.sum(wts), 1e-8)


def psum_if(x, vocab_axis: str | None):
    """``jax.lax.psum`` over ``vocab_axis`` (any pytree), identity when ``vocab_axis is None``."""
    return x if vocab_axis is None else jax.lax.psum(x, vocab_axis)


def vocab_max(x, vocab_axis: str | None):
    """Detached max over the (possibly sharded) last axis."""
    m = jnp.max(jax.lax.stop_gradient(x), axis=-1)
    return m if vocab_axis is None else jax.lax.pmax(m, vocab_axis)


def vocab_sum(x, vocab_axis: str | None):
    """Sum over the (possibly sharded) last axis; differentiable."""
    return psum_if(jnp.sum(x, axis=-1), vocab_axis)


def logsumexp(x, vocab_axis: str | None = None):
    """``logsumexp`` over the (possibly sharded) last axis."""
    if vocab_axis is None:
        return jax.scipy.special.logsumexp(x, axis=-1)
    global_max = vocab_max(x, vocab_axis)
    global_se = vocab_sum(jnp.exp(x - global_max[..., None]), vocab_axis)
    return jnp.log(global_se) + global_max


def log_softmax(z, vocab_axis: str | None = None):
    """``log_softmax`` over the (possibly sharded) last axis."""
    if vocab_axis is None:
        return jax.nn.log_softmax(z, axis=-1)
    return z - logsumexp(z, vocab_axis)[..., None]
