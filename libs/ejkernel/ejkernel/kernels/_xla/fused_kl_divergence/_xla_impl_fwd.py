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

"""Pure-JAX forward implementation of fused forward-KL between two logit tensors.

Two execution modes:

* ``vocab_parallel_axis is None`` — single-device / DP / SP / FSDP.
  Computes ``KL(softmax(teacher) || softmax(student))`` row-wise; the
  analytic backward (in :mod:`_xla_impl_bwd`) returns
  ``softmax(student) - softmax(teacher)`` for the student and zero for
  the teacher (which is treated as detached for distillation).
* ``vocab_parallel_axis="<name>"`` — vocab-parallel. Must be called
  inside ``shard_map`` with ``V`` sharded on that mesh axis. The forward
  online-merges per-shard ``(max, sum_exp)`` for both teacher and student
  across the TP axis via ``pmax`` / ``psum`` and then ``psum``s the local
  KL contribution; the backward needs no collectives.

The ``jax.custom_vjp`` cores defined here pair with the ``_*_fwd`` /
``_*_bwd`` rules in :mod:`_xla_impl_bwd`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ._xla_impl_bwd import _kl_bwd, _kl_fwd, _kl_tp_bwd, _kl_tp_fwd


@jax.custom_vjp
def _fused_kl_core(student, teacher, weights):
    """Per-row forward KL on rank-N ``(..., V)`` tensors."""
    # fp32 softmax to match the reverse/JSD paths -- a bf16 log_softmax diverges from dense by ~2.5e-2.
    student = student.astype(jnp.float32)
    teacher = teacher.astype(jnp.float32)
    log_p_t = jax.nn.log_softmax(teacher, axis=-1)
    log_p_s = jax.nn.log_softmax(student, axis=-1)
    p_t = jnp.exp(log_p_t)
    per_row = jnp.sum(p_t * (log_p_t - log_p_s), axis=-1)
    return (per_row * weights).astype(jnp.float32)


_fused_kl_core.defvjp(_kl_fwd, _kl_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fused_kl_core_tp(student_local, teacher_local, weights, vocab_axis):
    """Vocab-parallel forward KL.

    Both inputs are ``(..., V_local)`` per-shard slices. Must be called
    inside ``shard_map`` with ``vocab_axis`` as the TP mesh axis.
    """
    # fp32 softmax merge -- the cross-shard online ``exp(local_max - global_max)`` accumulates bf16 error
    # across shards, diverging from dense by ~2.5e-2; the reverse/JSD paths already upcast.
    student_local = student_local.astype(jnp.float32)
    teacher_local = teacher_local.astype(jnp.float32)
    local_max_t = jnp.max(teacher_local, axis=-1)
    local_se_t = jnp.sum(jnp.exp(teacher_local - local_max_t[..., None]), axis=-1)
    local_max_s = jnp.max(student_local, axis=-1)
    local_se_s = jnp.sum(jnp.exp(student_local - local_max_s[..., None]), axis=-1)

    global_max_t = jax.lax.pmax(local_max_t, vocab_axis)
    global_se_t = jax.lax.psum(local_se_t * jnp.exp(local_max_t - global_max_t), vocab_axis)
    lse_t = jnp.log(global_se_t) + global_max_t

    global_max_s = jax.lax.pmax(local_max_s, vocab_axis)
    global_se_s = jax.lax.psum(local_se_s * jnp.exp(local_max_s - global_max_s), vocab_axis)
    lse_s = jnp.log(global_se_s) + global_max_s

    log_p_t_local = teacher_local - lse_t[..., None]
    log_p_s_local = student_local - lse_s[..., None]
    p_t_local = jnp.exp(log_p_t_local)
    local_loss_part = jnp.sum(p_t_local * (log_p_t_local - log_p_s_local), axis=-1)
    per_row = jax.lax.psum(local_loss_part, vocab_axis) * weights
    return per_row.astype(jnp.float32)


_fused_kl_core_tp.defvjp(_kl_tp_fwd, _kl_tp_bwd)


def _kl_per_row(student_logits, teacher_logits, weights, direction, temperature, beta):
    """Pure-JAX per-row KL — supports forward / reverse / JSD + temperature.

    Used as the XLA fallback for the operation. JAX autodiff handles
    the gradient through this directly (we don't define a custom_vjp
    for the temperature/JSD paths — the closed-form gradient is the
    same as autodiff would produce).
    """
    inv_T = 1.0 / float(temperature)
    # Accumulate in float32 for numerical stability (bf16/fp16 logits lose precision in the
    # softmax / log-mixture, and the 1/T scaling amplifies it). The gradient is still taken
    # w.r.t. the original student dtype. Matches the fp32 accumulation of reference KL losses.
    s = student_logits.astype(jnp.float32) * inv_T
    t = teacher_logits.astype(jnp.float32) * inv_T
    log_p_s = jax.nn.log_softmax(s, axis=-1)
    log_p_t = jax.nn.log_softmax(t, axis=-1)
    p_s = jnp.exp(log_p_s)
    p_t = jnp.exp(log_p_t)

    if direction == "forward":
        per_row = jnp.sum(p_t * (log_p_t - log_p_s), axis=-1)
    elif direction == "reverse":
        per_row = jnp.sum(p_s * (log_p_s - log_p_t), axis=-1)
    elif direction == "jsd":
        # Mixture weights in float32 (a bf16 beta loses precision in log/log1p).
        log_beta = jnp.log(jnp.asarray(beta, dtype=jnp.float32))
        log_one_minus_beta = jnp.log1p(-jnp.asarray(beta, dtype=jnp.float32))
        # Mixture m = beta * p_t + (1 - beta) * p_s (matches the module docstring and the
        # standard GKD / Agarwal et al. convention): teacher pairs with log(beta).
        log_m = jax.scipy.special.logsumexp(
            jnp.stack([log_p_t + log_beta, log_p_s + log_one_minus_beta]),
            axis=0,
        )
        per_row = beta * jnp.sum(p_t * (log_p_t - log_m), axis=-1) + (1.0 - beta) * jnp.sum(
            p_s * (log_p_s - log_m), axis=-1
        )
    else:
        raise ValueError(f"direction must be forward / reverse / jsd; got {direction!r}")

    if temperature != 1.0:
        per_row = per_row * (float(temperature) ** 2)
    return (per_row * weights).astype(jnp.float32)


def _kl_per_row_tp(student_logits, teacher_logits, weights, direction, temperature, beta, vocab_axis):
    """Vocab-parallel per-row KL for *any* direction (forward / reverse / JSD) + temperature.

    Mirrors :func:`_kl_per_row` but the inputs are the per-shard ``(..., V_local)`` slices and the
    softmax normalizers (and the JSD log-mixture) are built globally with an online-softmax
    ``pmax`` + ``psum`` over ``vocab_axis``; the per-row divergence is then ``psum``-reduced across
    shards. ``pmax`` is fed a ``stop_gradient`` input -- the global max only re-centers the exp and the
    loss is invariant to it -- so plain autodiff differentiates the (differentiable) ``psum`` collectives
    directly, with no hand-written VJP. Must be called inside ``shard_map``.
    """
    inv_T = 1.0 / float(temperature)
    s = student_logits.astype(jnp.float32) * inv_T
    t = teacher_logits.astype(jnp.float32) * inv_T

    def _log_softmax_tp(z):
        global_max = jax.lax.pmax(jax.lax.stop_gradient(jnp.max(z, axis=-1)), vocab_axis)
        global_se = jax.lax.psum(jnp.sum(jnp.exp(z - global_max[..., None]), axis=-1), vocab_axis)
        return z - (jnp.log(global_se) + global_max)[..., None]

    log_p_s = _log_softmax_tp(s)
    log_p_t = _log_softmax_tp(t)
    p_s = jnp.exp(log_p_s)
    p_t = jnp.exp(log_p_t)

    if direction == "forward":
        local = jnp.sum(p_t * (log_p_t - log_p_s), axis=-1)
    elif direction == "reverse":
        local = jnp.sum(p_s * (log_p_s - log_p_t), axis=-1)
    elif direction == "jsd":
        log_beta = jnp.log(jnp.asarray(beta, dtype=jnp.float32))
        log_one_minus_beta = jnp.log1p(-jnp.asarray(beta, dtype=jnp.float32))
        log_m = jax.scipy.special.logsumexp(
            jnp.stack([log_p_t + log_beta, log_p_s + log_one_minus_beta]),
            axis=0,
        )
        local = beta * jnp.sum(p_t * (log_p_t - log_m), axis=-1) + (1.0 - beta) * jnp.sum(
            p_s * (log_p_s - log_m), axis=-1
        )
    else:
        raise ValueError(f"direction must be forward / reverse / jsd; got {direction!r}")

    per_row = jax.lax.psum(local, vocab_axis)
    if temperature != 1.0:
        per_row = per_row * (float(temperature) ** 2)
    return (per_row * weights).astype(jnp.float32)


def fused_kl_divergence(
    student_logits,
    teacher_logits,
    weights=None,
    *,
    reduction: str = "mean",
    direction: str = "forward",
    temperature: float = 1.0,
    beta: float = 0.5,
    vocab_parallel_axis: str | None = None,
):
    """JAX/XLA reference for fused KL (forward / reverse / JSD, optional T).

    Mirrors the TileLang interface bit-for-bit: same direction +
    temperature + beta semantics, same masking + reduction. Use on
    non-NVIDIA backends or when TileLang is unavailable.

    Args:
        student_logits, teacher_logits: ``(..., V)``.
        weights: ``logits.shape[:-1]``; ``completion_mask`` for
            assistant-only loss.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        direction: ``"forward"`` / ``"reverse"`` / ``"jsd"``.
        temperature: Softmax temperature ``T``; loss scaled by ``T²``.
        beta: JSD interpolation factor (only used when
            ``direction="jsd"``).
        vocab_parallel_axis: Mesh axis name for TP. Forward, reverse, and JSD are
            all supported under vocab parallelism; any temperature is supported via
            a ``1/T`` pre-scale + ``T**2`` post-scale around the vocab-parallel core.
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Invalid reduction '{reduction}'; expected one of none/sum/mean.")
    if direction not in ("forward", "reverse", "jsd"):
        raise ValueError(f"Invalid direction '{direction}'; expected forward/reverse/jsd.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive; got {temperature}")
    if direction == "jsd" and not 0.0 < beta < 1.0:
        raise ValueError(f"JSD requires beta in (0, 1); got {beta}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"fused_kl_divergence: shape mismatch student={student_logits.shape} vs teacher={teacher_logits.shape}"
        )

    teacher_logits = teacher_logits.astype(student_logits.dtype)
    leading = student_logits.shape[:-1]
    if weights is None:
        wts = jnp.ones(leading, dtype=jnp.float32)
    else:
        if weights.shape != leading:
            raise ValueError(f"weights.shape={weights.shape} must equal logits.shape[:-1]={leading}")
        wts = weights.astype(jnp.float32)

    if vocab_parallel_axis is not None:
        if direction == "forward":
            # Forward KL keeps the memory-optimal custom_vjp core. Temperature is a linear 1/T pre-scale
            # on the (still per-shard ``V_local``) logits plus a T^2 post-scale on the per-row loss; both
            # compose through ``_fused_kl_core_tp``'s custom_vjp via outer autodiff, yielding the exact
            # T^2*(1/T)=T-scaled distillation gradient WITHOUT ever materializing the full vocabulary.
            if temperature != 1.0:
                inv_T = 1.0 / float(temperature)
                per_row = _fused_kl_core_tp(
                    student_logits * inv_T,
                    teacher_logits * inv_T,
                    wts,
                    vocab_parallel_axis,
                ) * (float(temperature) ** 2)
            else:
                per_row = _fused_kl_core_tp(student_logits, teacher_logits, wts, vocab_parallel_axis)
        else:
            # Reverse / JSD vocab-parallel via the plain-autodiff per-row path (global softmax + log-mixture
            # built with pmax+psum over the vocab axis; gradient flows through the differentiable psum).
            per_row = _kl_per_row_tp(
                student_logits, teacher_logits, wts, direction, float(temperature), float(beta), vocab_parallel_axis
            )
    elif direction == "forward" and temperature == 1.0:
        per_row = _fused_kl_core(student_logits, teacher_logits, wts)
    else:
        per_row = _kl_per_row(student_logits, teacher_logits, wts, direction, float(temperature), float(beta))

    if reduction == "none":
        return per_row
    total = jnp.sum(per_row)
    if reduction == "sum":
        return total
    denom = jnp.maximum(jnp.sum(wts), 1e-8)
    return total / denom
