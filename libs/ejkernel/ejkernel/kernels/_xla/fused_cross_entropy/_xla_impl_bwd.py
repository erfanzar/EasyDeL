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

"""Backward rules for the XLA fused cross-entropy custom VJPs.

This module owns the VJP primal (forward-with-residuals) and backward
rules for both the single-shard and vocab-parallel custom VJPs declared
in :mod:`_xla_impl_fwd`.

The single-shard rules cache ``(exp_shifted, sum_exp, target, weight,
logits)`` so the backward is one analytic ``(softmax - onehot) * weight``
broadcast multiply.

The vocab-parallel rules use ``pmax`` + an online-softmax ``psum`` to
build the global ``(max, sum_exp, target_logit)``, cache the per-shard
``logits`` and shard metadata, and then write the local ``dlogits``
slab without any further collectives in the backward (each shard already
owns its softmax slice and the local piece of the ``onehot`` subtraction).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _ce_fwd(logits, targets, weights, ignore_index):
    """VJP primal for the single-shard fused cross-entropy.

    Saves only the per-row ``lse`` (and the input ``logits``, which is already
    live) rather than the full ``[..., V]`` ``exp_shifted``. The backward
    recomputes ``softmax = exp(logits - lse)``. On bandwidth-bound hardware this
    trades a ``[..., V]`` HBM round-trip (write in fwd + read in bwd) for a cheap
    in-place ``exp`` recompute — ~25% less HBM traffic than caching the softmax.
    """
    del ignore_index
    vocab = logits.shape[-1]
    safe_targets = jnp.clip(targets, 0, vocab - 1)
    lse = jax.nn.logsumexp(logits, axis=-1)
    target_logit = jnp.take_along_axis(logits, safe_targets[..., None], axis=-1)[..., 0]
    per_row = (lse - target_logit) * weights
    residual = (logits, lse, safe_targets, weights)
    return per_row.astype(jnp.float32), residual


def _ce_bwd(ignore_index, residual, dy):
    """Analytic backward: ``dlogits = weight * (softmax - onehot) * dy``.

    Recomputes ``softmax = exp(logits - lse)`` from the saved per-row ``lse``.
    """
    del ignore_index
    logits, lse, safe_targets, weights = residual
    probs = jnp.exp(logits - lse[..., None])
    onehot = jax.nn.one_hot(safe_targets, probs.shape[-1], dtype=probs.dtype)
    factor = (weights.astype(probs.dtype) * dy.astype(probs.dtype))[..., None]
    dlogits = (probs - onehot) * factor
    return (dlogits.astype(logits.dtype), None, None)


def _ce_tp_fwd(logits_local, targets, weights, ignore_index, vocab_axis):
    """VJP primal for vocab-parallel cross-entropy.

    Online-merges per-shard ``(max, sum_exp)`` across ``vocab_axis`` so the
    forward returns the *globally correct* per-row loss, while caching the
    minimal state needed for a fully local backward.
    """
    del ignore_index
    v_local = logits_local.shape[-1]
    tp_idx = jax.lax.axis_index(vocab_axis)
    vocab_start = tp_idx * v_local

    local_max = jnp.max(logits_local, axis=-1)
    local_se = jnp.sum(jnp.exp(logits_local - local_max[..., None]), axis=-1)

    is_local = (targets >= vocab_start) & (targets < vocab_start + v_local)
    local_idx = jnp.where(is_local, targets - vocab_start, 0)
    local_target_logit = jnp.where(
        is_local,
        jnp.take_along_axis(logits_local, local_idx[..., None], axis=-1)[..., 0],
        0.0,
    )

    global_max = jax.lax.pmax(local_max, vocab_axis)
    scaled_local_se = local_se * jnp.exp(local_max - global_max)
    global_se = jax.lax.psum(scaled_local_se, vocab_axis)
    global_target_logit = jax.lax.psum(local_target_logit, vocab_axis)

    lse = jnp.log(global_se) + global_max
    per_row = (lse - global_target_logit) * weights
    residual = (logits_local, global_max, global_se, is_local, local_idx, weights)
    return per_row.astype(jnp.float32), residual


def _ce_tp_bwd(ignore_index, vocab_axis, residual, dy):
    """Backward for vocab-parallel CE.

    No collectives: each shard already owns its slice of the (global)
    softmax and the local piece of the onehot subtraction, so the local
    ``(softmax - onehot) * weight * dy`` is exactly the gradient slab for
    this shard.
    """
    del ignore_index, vocab_axis
    logits_local, global_max, global_se, is_local, local_idx, weights = residual
    probs_local = jnp.exp(logits_local - global_max[..., None]) / global_se[..., None]
    onehot_local = is_local[..., None].astype(probs_local.dtype) * jax.nn.one_hot(
        local_idx, probs_local.shape[-1], dtype=probs_local.dtype
    )
    factor = (weights.astype(probs_local.dtype) * dy.astype(probs_local.dtype))[..., None]
    dlogits_local = (probs_local - onehot_local) * factor
    return (dlogits_local.astype(logits_local.dtype), None, None)


def _soft_ce_tp_fwd(logits_local, soft_local, vocab_axis):
    """VJP primal for vocab-parallel *dense* (soft-target) cross-entropy.

    Builds the global log-sum-exp via an online-softmax ``pmax`` + ``psum`` over the vocab axis (the
    ``pmax`` lives only here, never under autodiff), then ``psum``-reduces the per-row
    ``soft·(logits - lse)`` dot product across shards. Returns the *unweighted* per-row loss; the caller
    multiplies by the token weights. Caches the per-shard softmax state + global soft mass for a fully
    local backward.
    """
    local_max = jnp.max(logits_local, axis=-1)
    local_se = jnp.sum(jnp.exp(logits_local - local_max[..., None]), axis=-1)
    global_max = jax.lax.pmax(local_max, vocab_axis)
    scaled_local_se = local_se * jnp.exp(local_max - global_max)
    global_se = jax.lax.psum(scaled_local_se, vocab_axis)
    lse = jnp.log(global_se) + global_max
    local_dot = jnp.sum(soft_local * (logits_local - lse[..., None]), axis=-1)
    per_row = -jax.lax.psum(local_dot, vocab_axis)
    soft_mass = jax.lax.psum(jnp.sum(soft_local, axis=-1), vocab_axis)
    residual = (logits_local, global_max, global_se, soft_local, soft_mass)
    return per_row.astype(jnp.float32), residual


def _soft_ce_tp_bwd(vocab_axis, residual, dy):
    """Backward for vocab-parallel dense CE: ``dlogits = (softmax·S - soft) · dy`` (local, no collectives).

    ``S`` is the per-row global soft mass (1 for a normalized distribution); each shard owns its slice
    of the (global) softmax and its slice of ``soft_targets``, so the slab is exact without any psum.
    """
    del vocab_axis
    logits_local, global_max, global_se, soft_local, soft_mass = residual
    probs_local = jnp.exp(logits_local - global_max[..., None]) / global_se[..., None]
    factor = dy.astype(probs_local.dtype)[..., None]
    dlogits_local = (probs_local * soft_mass[..., None] - soft_local) * factor
    return (dlogits_local.astype(logits_local.dtype), None)


__all__ = (
    "_ce_bwd",
    "_ce_fwd",
    "_ce_tp_bwd",
    "_ce_tp_fwd",
    "_soft_ce_tp_bwd",
    "_soft_ce_tp_fwd",
)
