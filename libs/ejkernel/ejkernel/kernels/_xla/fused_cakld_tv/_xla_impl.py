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

"""Pure-JAX single-pass fused dynamic-CAKLD + TV blend between two logit tensors.

Per row, with ``p = softmax(t / T)``, ``q = softmax(s / T)`` and the per-row
teacher confidence gate ``gamma = max_v p_v`` (detached)::

    blend = gamma * KL(q || p) + (1 - gamma) * KL(p || q)
    loss  = ((1 - rho) * blend + rho * 2 * TV(p, q)) * T**2

where ``TV(p, q) = 1 - sum_v min(p_v, q_v)`` and ``rho = tv_weight``. This is a
per-token confidence-gated forward/reverse KL blend (peaked teacher tokens get
mode-seeking reverse KL, flat tokens get mass-covering forward KL) with an
optional total-variation mix. It is fused so BOTH logit tensors are read once
per call rather than once per component (gamma, reverse KL, forward KL, TV).

An optional gated Renyi student-entropy penalty
``entropy_weight * 1[gamma > entropy_gate] * H_alpha(q)`` is added per row
before the shared ``T**2`` scaling; it sharpens the student only where the
teacher is committed.

One code path serves both execution modes: with ``vocab_parallel_axis=None`` the
vocabulary is whole on the device; with a mesh-axis name the inputs are
per-shard ``(..., V_local)`` slices (must run inside ``shard_map``) and the
three per-row partial reductions (forward, reverse, TV overlap) share ONE
``psum``, with one extra detached ``pmax`` for the gate. The teacher and the
gate are detached on every path, so the teacher cotangent is exactly zero and
gamma is a weight rather than a gradient path. Gradients are plain autodiff.
Logits must be finite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._row_loss import (
    check_reduction,
    check_temperature,
    log_softmax,
    psum_if,
    reduce_rows,
    row_weights,
    vocab_max,
)
from ..fused_renyi_entropy._xla_impl import renyi_entropy_rows


def _blend_per_row(
    student_logits,
    teacher_logits,
    wts,
    temperature: float,
    tv_weight: float,
    entropy_weight: float,
    entropy_alpha: float,
    entropy_gate: float | None,
    vocab_axis: str | None,
):
    """Weighted per-row fused blend for any temperature and ``rho`` in ``[0, 1]``."""
    inv_t = 1.0 / temperature
    # fp32 throughout: a bf16 log_softmax diverges from dense by ~2.5e-2 at large V, and the
    # blend subtracts O(1) divergences. The gradient is still taken w.r.t. the original student dtype.
    s = student_logits.astype(jnp.float32) * inv_t
    t = jax.lax.stop_gradient(teacher_logits.astype(jnp.float32)) * inv_t
    log_p_s = log_softmax(s, vocab_axis)
    log_p_t = log_softmax(t, vocab_axis)
    p_s = jnp.exp(log_p_s)
    p_t = jnp.exp(log_p_t)

    diff = log_p_s - log_p_t
    partials = (
        jnp.sum(p_s * diff, axis=-1),  # reverse KL(q || p)
        -jnp.sum(p_t * diff, axis=-1),  # forward KL(p || q)
        jnp.sum(jnp.exp(jnp.minimum(log_p_t, log_p_s)), axis=-1) if tv_weight != 0.0 else None,  # TV overlap
    )
    reverse_kl, forward_kl, overlap = psum_if(partials, vocab_axis)
    # log_p_t is globally normalized, so the global gate is the exp of one detached (p)max of the local max.
    gamma = jnp.exp(vocab_max(log_p_t, vocab_axis))

    per_row = gamma * reverse_kl + (1.0 - gamma) * forward_kl
    if tv_weight != 0.0:
        per_row = (1.0 - tv_weight) * per_row + tv_weight * 2.0 * (1.0 - overlap)

    if entropy_weight != 0.0:
        # log_p_s is already normalized, so the Renyi entropy skips its own normalizer reduction.
        h = renyi_entropy_rows(log_p_s, entropy_alpha, vocab_axis, normalized=True)
        gate = (gamma > jnp.float32(entropy_gate)).astype(jnp.float32) if entropy_gate is not None else 1.0
        per_row = per_row + jnp.float32(entropy_weight) * gate * h

    if temperature != 1.0:
        per_row = per_row * (temperature**2)
    return (per_row * wts).astype(jnp.float32)


def fused_cakld_tv(
    student_logits,
    teacher_logits,
    weights=None,
    *,
    reduction: str = "mean",
    temperature: float = 1.0,
    tv_weight: float = 0.0,
    entropy_weight: float = 0.0,
    entropy_alpha: float = 1.0,
    entropy_gate: float | None = None,
    vocab_parallel_axis: str | None = None,
):
    """JAX/XLA single-pass fused dynamic-CAKLD (+ optional TV) distillation loss.

    Args:
        student_logits: ``(..., V)`` student logits (differentiable).
        teacher_logits: ``(..., V)`` teacher logits; detached internally and
            allowed to have a different float dtype than the student.
        weights: ``logits.shape[:-1]`` per-row weights; ``completion_mask`` for
            assistant-only loss. Zero-weight rows contribute zero loss and zero
            gradient.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"`` (mean over
            ``max(sum(weights), 1e-8)``).
        temperature: Softmax temperature ``T``; the loss is scaled by ``T**2``
            (Hinton convention, matching ``fused_kl_divergence``).
        tv_weight: TV mixing weight ``rho`` in ``[0, 1]``. ``0.0`` is plain
            per-token dynamic CAKLD and skips the TV accumulators entirely.
        entropy_weight: Gated Renyi student-entropy penalty coefficient ``>= 0``.
            ``0.0`` skips the entropy reductions entirely.
        entropy_alpha: Renyi order ``> 0``; ``1.0`` is the exact Shannon branch,
            ``2.0`` is collision entropy (head-focused sharpening).
        entropy_gate: Teacher-confidence threshold in ``(0, 1)`` for a hard gate
            on gamma; ``None`` applies the penalty on every row.
        vocab_parallel_axis: Mesh axis name for vocab parallelism; requires
            per-shard inputs inside ``shard_map``.

    Returns:
        float32 per-row array (``reduction="none"``) or scalar.

    Raises:
        ValueError: On an invalid ``reduction``, ``temperature``, ``tv_weight``,
            ``entropy_weight``, ``entropy_alpha`` or ``entropy_gate``, or when the
            logits (or ``weights``) shapes disagree.
    """
    check_reduction(reduction)
    check_temperature(temperature)
    if not 0.0 <= float(tv_weight) <= 1.0:
        raise ValueError(f"tv_weight must be within [0, 1]; got {tv_weight}")
    if float(entropy_weight) < 0.0:
        raise ValueError(f"entropy_weight must be non-negative; got {entropy_weight}")
    if not float(entropy_alpha) > 0.0:
        raise ValueError(f"entropy_alpha must be positive; got {entropy_alpha}")
    if entropy_gate is not None and not 0.0 < float(entropy_gate) < 1.0:
        raise ValueError(f"entropy_gate must be within (0, 1) when set; got {entropy_gate}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"fused_cakld_tv: shape mismatch student={student_logits.shape} vs teacher={teacher_logits.shape}"
        )

    wts = row_weights(weights, student_logits.shape[:-1])
    per_row = _blend_per_row(
        student_logits,
        teacher_logits,
        wts,
        float(temperature),
        float(tv_weight),
        float(entropy_weight),
        float(entropy_alpha),
        None if entropy_gate is None else float(entropy_gate),
        vocab_parallel_axis,
    )
    return reduce_rows(per_row, wts, reduction)
