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

"""Pure-JAX forward implementation of fused sparse cross-entropy.

Two execution modes:

* ``vocab_parallel_axis is None`` — single-device / DP / SP / FSDP.
  The forward computes the row-wise ``log-softmax``, target gather, and
  returns the per-row loss; the analytic backward (in
  :mod:`_xla_impl_bwd`) returns ``(softmax - onehot) * weight``. Leading
  dims keep their sharding because no reshape is performed.
* ``vocab_parallel_axis="<name>"`` — vocab-parallel. Must be called
  inside ``shard_map`` with ``V`` sharded on that mesh axis. The forward
  online-merges per-shard ``(max, sum_exp)`` across the TP axis via
  ``pmax`` / ``psum`` and gathers the (possibly cross-shard) target
  logit via ``psum``; the backward needs no collectives.

The ``jax.custom_vjp`` cores defined here pair with the ``_*_fwd`` /
``_*_bwd`` rules in :mod:`_xla_impl_bwd`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ._xla_impl_bwd import _ce_bwd, _ce_fwd, _ce_tp_bwd, _ce_tp_fwd, _soft_ce_tp_bwd, _soft_ce_tp_fwd


def _per_token_weights(targets, weights, ignore_index):
    """Build per-token float32 weights consistent with the TileLang interface."""
    if weights is None:
        return (targets != ignore_index).astype(jnp.float32)
    return weights.astype(jnp.float32)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fused_ce_core(logits, targets, weights, ignore_index):
    """Per-row sparse cross-entropy on a rank-N ``(..., V)`` tensor.

    No reshape is performed: every op is either elementwise on ``V`` or a
    reduction along ``V``. Sharding on the leading dims propagates through
    XLA SPMD without any explicit constraints.
    """
    del ignore_index
    vocab = logits.shape[-1]
    safe_targets = jnp.clip(targets, 0, vocab - 1)
    max_logit = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logit
    lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1)) + max_logit[..., 0]
    target_logit = jnp.take_along_axis(logits, safe_targets[..., None], axis=-1)[..., 0]
    per_row = lse - target_logit
    return (per_row * weights).astype(jnp.float32)


_fused_ce_core.defvjp(_ce_fwd, _ce_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _fused_ce_core_tp(logits_local, targets, weights, ignore_index, vocab_axis):
    """Vocab-parallel CE. ``logits_local`` is the per-shard ``V_local`` slice.

    Must be called inside ``shard_map`` with ``vocab_axis`` as the mesh axis
    along which the vocab dimension is sharded.
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
    return per_row.astype(jnp.float32)


_fused_ce_core_tp.defvjp(_ce_tp_fwd, _ce_tp_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _fused_soft_ce_core_tp(logits_local, soft_local, vocab_axis):
    """Vocab-parallel *dense* (soft-target) CE on the per-shard ``V_local`` slice.

    Must be called inside ``shard_map`` with ``vocab_axis`` as the sharded vocab mesh axis. Returns the
    *unweighted* per-row loss ``-Σ_v soft_v·log_softmax_v`` (global softmax via ``pmax``/``psum``). The
    custom VJP keeps ``pmax`` out of autodiff and emits a fully local ``(softmax·S - soft)`` backward.
    """
    local_max = jnp.max(logits_local, axis=-1)
    local_se = jnp.sum(jnp.exp(logits_local - local_max[..., None]), axis=-1)
    global_max = jax.lax.pmax(local_max, vocab_axis)
    scaled_local_se = local_se * jnp.exp(local_max - global_max)
    global_se = jax.lax.psum(scaled_local_se, vocab_axis)
    lse = jnp.log(global_se) + global_max
    local_dot = jnp.sum(soft_local * (logits_local - lse[..., None]), axis=-1)
    per_row = -jax.lax.psum(local_dot, vocab_axis)
    return per_row.astype(jnp.float32)


_fused_soft_ce_core_tp.defvjp(_soft_ce_tp_fwd, _soft_ce_tp_bwd)


def _label_smoothing_correction(logits, lse, targets, weights, label_smoothing):
    """Compute the label-smoothing loss correction in pure JAX.

    Returns ``per_row_correction`` that should be added to the standard
    CE loss ``lse - target_logit`` to get the smoothed loss
    ``lse - eff_target_w · target_logit - low_conf · sum_logits - norm_const``.
    """
    import math

    if label_smoothing <= 0.0:
        return jnp.zeros_like(weights)
    vocab = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_conf = label_smoothing / (vocab - 1)
    eff_target_w = confidence - low_conf
    norm_const = -(
        confidence * math.log(max(confidence, 1e-20)) + (vocab - 1) * low_conf * math.log(max(low_conf, 1e-20))
    )
    safe_targets = jnp.clip(targets, 0, vocab - 1)
    target_logit = jnp.take_along_axis(logits, safe_targets[..., None], axis=-1)[..., 0]
    sum_logits = jnp.sum(logits, axis=-1)
    return (1.0 - eff_target_w) * target_logit - low_conf * sum_logits - norm_const


def fused_cross_entropy(
    logits,
    targets=None,
    weights=None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    soft_targets=None,
    reduction: str = "mean",
    vocab_parallel_axis: str | None = None,
):
    """JAX/XLA reference for fused cross-entropy.

    Mirrors the TileLang interface bit-for-bit: same sparse/dense
    target modes, same ``label_smoothing`` / ``z_loss`` semantics, same
    masking and reduction handling. Use this on non-NVIDIA backends
    (TPU/CPU) or when TileLang is unavailable.
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Invalid reduction '{reduction}'; expected one of none/sum/mean.")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1); got {label_smoothing}")
    if z_loss < 0.0:
        raise ValueError(f"z_loss must be non-negative; got {z_loss}")

    if soft_targets is not None:
        if label_smoothing > 0.0:
            raise ValueError("`label_smoothing` cannot combine with `soft_targets` — apply smoothing externally.")
        if soft_targets.shape != logits.shape:
            raise ValueError(f"soft_targets.shape={soft_targets.shape} must equal logits.shape={logits.shape}")
        if vocab_parallel_axis is None:
            max_logit = jnp.max(logits, axis=-1, keepdims=True)
            shifted = logits - max_logit
            lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1)) + max_logit[..., 0]
            log_softmax = logits - lse[..., None]
            per_row = -jnp.sum(soft_targets * log_softmax, axis=-1)
            if z_loss > 0.0:
                per_row = per_row + z_loss * lse * lse
        else:
            # Vocab-parallel dense (soft-target) CE via the custom-VJP core: builds the global softmax
            # with pmax+psum over the vocab axis (kept out of autodiff) and emits a local
            # ``(softmax·S - soft)`` backward. Returns the unweighted per-row loss (weights applied below).
            if z_loss > 0.0:
                raise NotImplementedError(
                    "z_loss is not yet wired through the vocab-parallel dense (soft-target) XLA path."
                )
            per_row = _fused_soft_ce_core_tp(logits, soft_targets, vocab_parallel_axis)
        if weights is not None:
            if weights.shape != logits.shape[:-1]:
                raise ValueError(f"weights.shape={weights.shape} must equal logits.shape[:-1]={logits.shape[:-1]}")
            per_row = per_row * weights
            wts = weights.astype(jnp.float32)
        else:
            wts = jnp.ones(per_row.shape, dtype=jnp.float32)
    else:
        if targets is None:
            raise ValueError("either `targets` (sparse) or `soft_targets` (dense) must be provided.")
        if targets.shape != logits.shape[:-1]:
            raise ValueError(
                f"fused_cross_entropy: targets.shape={targets.shape} must equal logits.shape[:-1]={logits.shape[:-1]}"
            )
        if weights is not None and weights.shape != targets.shape:
            raise ValueError(
                f"fused_cross_entropy: weights.shape={weights.shape} must equal targets.shape={targets.shape}"
            )

        targets_i32 = targets.astype(jnp.int32)
        wts = _per_token_weights(targets_i32, weights, ignore_index)

        if vocab_parallel_axis is None:
            per_row = _fused_ce_core(logits, targets_i32, wts, ignore_index)
        else:
            if label_smoothing > 0.0 or z_loss > 0.0:
                raise NotImplementedError(
                    "label_smoothing / z_loss are not yet wired through the vocab-parallel XLA path."
                )
            per_row = _fused_ce_core_tp(logits, targets_i32, wts, ignore_index, vocab_parallel_axis)

        if label_smoothing > 0.0 or z_loss > 0.0:
            max_logit = jnp.max(logits, axis=-1, keepdims=True)
            shifted = logits - max_logit
            lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1)) + max_logit[..., 0]
            if label_smoothing > 0.0:
                correction = _label_smoothing_correction(logits, lse, targets_i32, wts, label_smoothing)
                per_row = per_row + correction * wts
            if z_loss > 0.0:
                per_row = per_row + z_loss * lse * lse * wts

    if soft_targets is None and targets is not None and vocab_parallel_axis is None:
        preds = jnp.argmax(logits, axis=-1)
        active = (targets_i32 != ignore_index).astype(jnp.float32)
        per_row_correct = (preds == targets_i32).astype(jnp.float32) * active
    else:
        per_row_correct = jnp.full(per_row.shape, -1.0, dtype=jnp.float32)

    if reduction == "none":
        return per_row.astype(jnp.float32), per_row_correct
    total = jnp.sum(per_row)
    if reduction == "sum":
        return total, per_row_correct
    denom = jnp.maximum(jnp.sum(wts), 1e-8)
    return total / denom, per_row_correct
