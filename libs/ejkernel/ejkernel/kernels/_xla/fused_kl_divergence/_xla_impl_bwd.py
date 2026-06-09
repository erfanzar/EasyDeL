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

"""Backward rules for the XLA fused forward-KL custom VJPs.

This module owns the VJP primal (forward-with-residuals) and backward
rules for both the single-shard and vocab-parallel custom VJPs declared
in :mod:`_xla_impl_fwd`.

The single-shard rules cache the per-row teacher and student softmaxes
so the backward is a single analytic
``(softmax(student) - softmax(teacher)) * weight`` broadcast multiply.
The teacher cotangent is forced to zero (the teacher is treated as
detached for distillation).

The vocab-parallel rules online-merge per-shard ``(max, sum_exp)`` for
both teacher and student across the TP axis to compute the global
``lse_*``, cache the per-shard rescaled probabilities, and write the
local ``dstudent`` slab without further collectives.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _kl_fwd(student, teacher, weights):
    """VJP primal for single-shard forward KL.

    Caches both softmaxes for the analytic backward.
    """
    # fp32 softmax (bf16 log_softmax diverges ~2.5e-2); residual keeps the original dtype so the
    # backward casts ``dstudent`` back to the student's dtype.
    log_p_t = jax.nn.log_softmax(teacher.astype(jnp.float32), axis=-1)
    log_p_s = jax.nn.log_softmax(student.astype(jnp.float32), axis=-1)
    p_t = jnp.exp(log_p_t)
    p_s = jnp.exp(log_p_s)
    per_row = jnp.sum(p_t * (log_p_t - log_p_s), axis=-1) * weights
    residual = (p_s, p_t, weights, student, teacher)
    return per_row.astype(jnp.float32), residual


def _kl_bwd(residual, dy):
    """Analytic backward: ``dstudent = weight * (p_s - p_t) * dy``; teacher → 0."""
    p_s, p_t, weights, student, teacher = residual
    factor = (weights.astype(p_s.dtype) * dy.astype(p_s.dtype))[..., None]
    dstudent = (p_s - p_t) * factor
    dteacher = jnp.zeros_like(teacher)
    return (dstudent.astype(student.dtype), dteacher, None)


def _kl_tp_fwd(student_local, teacher_local, weights, vocab_axis):
    """VJP primal for vocab-parallel forward KL.

    Online-merges per-shard ``(max, sum_exp)`` separately for teacher and
    student via ``pmax``/``psum`` over ``vocab_axis``, then ``psum``s
    the local KL contribution so the returned per-row value is the
    *globally correct* KL. Caches the rescaled per-shard probabilities
    so the backward is fully local.
    """
    # fp32 softmax merge (bf16 cross-shard online-softmax diverges ~2.5e-2); ``s_in``/``t_in`` keep the
    # original dtype so the backward casts ``dstudent`` back to the student's dtype.
    s_in, t_in = student_local, teacher_local
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
    p_s_local = jnp.exp(log_p_s_local)

    local_loss_part = jnp.sum(p_t_local * (log_p_t_local - log_p_s_local), axis=-1)
    per_row = jax.lax.psum(local_loss_part, vocab_axis) * weights
    residual = (p_s_local, p_t_local, weights, s_in, t_in)
    return per_row.astype(jnp.float32), residual


def _kl_tp_bwd(vocab_axis, residual, dy):
    """Backward for vocab-parallel KL.

    No collectives: the cached ``p_s_local`` / ``p_t_local`` are already
    the global softmax slices restricted to this shard, so the local
    ``(p_s - p_t) * weight * dy`` is the gradient slab for this shard.
    """
    del vocab_axis
    p_s_local, p_t_local, weights, student_local, teacher_local = residual
    factor = (weights.astype(p_s_local.dtype) * dy.astype(p_s_local.dtype))[..., None]
    dstudent_local = (p_s_local - p_t_local) * factor
    dteacher_local = jnp.zeros_like(teacher_local)
    return (dstudent_local.astype(student_local.dtype), dteacher_local, None)


__all__ = ("_kl_bwd", "_kl_fwd", "_kl_tp_bwd", "_kl_tp_fwd")
