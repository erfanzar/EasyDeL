# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
from easydel.trainers.distillation_trainer._fn import chunked_distillation_loss, distillation_loss


def _identity_lm_head(hidden: jnp.ndarray) -> jnp.ndarray:
    return hidden


def _build_expected_ce(
    student_logits: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    attention_mask: jnp.ndarray | None,
    loss_mask: jnp.ndarray | None,
    shift_tokens: bool = True,
) -> jnp.ndarray:
    """Independent reference for the distillation hard-label CE.

    When ``shift_tokens`` (the ``LossConfig.shift_tokens`` default) the reference is
    causally shifted -- ``student_logits[:, :-1]`` scored against ``labels[:, 1:]`` and
    the completion / attention ``mask[:, 1:]`` -- so it matches ``ForCausalLMLoss`` and
    the fixed ``distillation_loss``. With ``shift_tokens=False`` it reproduces the old
    unshifted ``CE(logits[t], labels[t])`` behavior.
    """
    dtype = student_logits.dtype
    if shift_tokens:
        student_logits = student_logits[:, :-1]
        labels = labels[:, 1:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, 1:]
        if loss_mask is not None:
            loss_mask = loss_mask[:, 1:]

    if loss_mask is not None:
        mask = loss_mask.astype(dtype)
    elif attention_mask is not None:
        mask = attention_mask.astype(dtype)
    else:
        mask = None

    valid_label_mask = (labels != -100).astype(dtype)
    mask = valid_label_mask if mask is None else mask * valid_label_mask

    safe_labels = jnp.where(labels == -100, 0, labels)
    per_token_ce = optax.softmax_cross_entropy_with_integer_labels(
        student_logits.astype(jnp.float32),
        safe_labels,
    ).astype(dtype)
    normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
    return jnp.sum(per_token_ce * mask) / normalizer


@pytest.mark.parametrize(
    ("attention_mask", "loss_mask"),
    [
        (None, None),
        (jnp.array([[1, 1, 0], [1, 0, 0]], dtype=jnp.int32), None),
        (None, jnp.array([[1, 0, 1], [0, 1, 0]], dtype=jnp.int32)),
    ],
)
def test_kl_loss_zero_when_student_equals_teacher(attention_mask, loss_mask):
    logits = jnp.array(
        [
            [[1.0, 0.5, -0.1, 0.2], [0.3, -0.4, 1.1, 0.9], [0.2, -0.1, 0.0, 0.7]],
            [[-0.2, 0.4, 0.9, 0.6], [1.4, -0.8, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    _, metrics = distillation_loss(
        student_logits=logits,
        teacher_logits=logits,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=None,
        use_hard_labels=False,
        temperature=2.0,
        alpha=1.0,
    )
    assert jnp.allclose(metrics["kl_loss"], 0.0, atol=1e-6)
    assert jnp.allclose(
        metrics["kl_loss"],
        metrics["distill_xent_loss"] - metrics["teacher_entropy_loss"],
        atol=1e-6,
    )


def test_kl_identity_holds():
    student_logits = jnp.array(
        [
            [[2.0, 0.0, -1.0], [0.5, 1.2, -0.3], [1.1, -0.1, 0.2]],
            [[0.1, -0.4, 0.3], [0.9, 1.5, -0.2], [-1.0, 0.2, 0.7]],
        ],
        dtype=jnp.float32,
    )
    teacher_logits = jnp.array(
        [
            [[1.2, 0.3, -0.7], [0.2, 0.8, 0.0], [0.9, -0.3, 0.4]],
            [[-0.2, -0.1, 0.6], [1.0, 0.4, 0.2], [-0.8, 0.6, 0.3]],
        ],
        dtype=jnp.float32,
    )
    loss_mask = jnp.array([[1, 1, 0], [1, 0, 1]], dtype=jnp.int32)

    _, metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        loss_mask=loss_mask,
        use_hard_labels=False,
        temperature=3.0,
        alpha=0.9,
    )
    assert jnp.allclose(
        metrics["kl_loss"],
        metrics["distill_xent_loss"] - metrics["teacher_entropy_loss"],
        atol=1e-6,
    )


def test_chunked_matches_non_chunked():
    student_logits = jnp.array(
        [
            [
                [0.2, 0.3, 0.1, -0.5, 1.1],
                [0.7, -0.4, 0.0, 0.5, 0.9],
                [-0.1, 1.0, 0.3, 0.4, 0.2],
                [0.0, -0.2, 0.8, 0.6, 0.1],
                [1.3, 0.4, -0.6, 0.2, 0.0],
            ],
            [
                [0.4, 0.2, 0.5, 0.3, -0.2],
                [0.1, 0.0, -0.1, 1.0, 0.9],
                [0.6, 0.7, -0.2, 0.1, 0.3],
                [-0.3, 0.8, 0.9, 0.0, 0.4],
                [0.2, -0.5, 0.4, 0.6, 0.7],
            ],
        ],
        dtype=jnp.float32,
    )
    teacher_logits = jnp.array(
        [
            [
                [0.3, 0.1, 0.2, -0.2, 1.0],
                [0.6, -0.1, 0.2, 0.4, 0.8],
                [0.0, 0.9, 0.2, 0.5, 0.1],
                [0.2, -0.3, 0.6, 0.7, 0.0],
                [1.0, 0.5, -0.4, 0.1, 0.2],
            ],
            [
                [0.5, 0.0, 0.4, 0.2, -0.1],
                [0.2, -0.1, 0.0, 0.9, 1.0],
                [0.4, 0.8, -0.3, 0.2, 0.1],
                [-0.2, 0.6, 0.7, 0.1, 0.5],
                [0.3, -0.4, 0.5, 0.4, 0.8],
            ],
        ],
        dtype=jnp.float32,
    )
    attention_mask = jnp.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=jnp.int32)
    labels = jnp.array([[1, 2, -100, 0, -100], [4, -100, 3, -100, -100]], dtype=jnp.int32)

    non_chunked_total, non_chunked_metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        attention_mask=attention_mask,
        labels=labels,
        use_hard_labels=True,
        temperature=2.5,
        alpha=0.35,
    )
    chunked_total, chunked_metrics = chunked_distillation_loss(
        student_hidden=student_logits,
        teacher_hidden=teacher_logits,
        student_lm_head_fn=_identity_lm_head,
        teacher_lm_head_fn=_identity_lm_head,
        attention_mask=attention_mask,
        labels=labels,
        use_hard_labels=True,
        temperature=2.5,
        alpha=0.35,
        chunk_size=3,
    )

    assert jnp.allclose(non_chunked_total, chunked_total, atol=1e-6)
    for key in ("kl_loss", "distill_xent_loss", "teacher_entropy_loss", "ce_loss"):
        # fp32 cancellation: the non-chunked path uses the ejkernel fused-KL (per-row
        # KL before reduction) vs the chunked sum-of-parts; equal to ~1e-5, not 1e-6.
        assert jnp.allclose(non_chunked_metrics[key], chunked_metrics[key], atol=1e-5)


def test_supervised_ce_unchanged():
    student_logits = jnp.array(
        [
            [[1.0, 0.2, -0.1, 0.3], [0.1, 0.4, 0.7, -0.5], [0.9, -0.3, 0.2, 0.0]],
            [[-0.1, 0.5, 0.6, 0.2], [0.3, 0.2, -0.4, 0.8], [0.0, 0.1, 0.2, 0.3]],
        ],
        dtype=jnp.float32,
    )
    teacher_logits_a = jnp.array(
        [
            [[0.2, 0.0, 0.1, 0.5], [0.6, -0.2, 0.4, 0.1], [0.8, 0.3, -0.1, 0.0]],
            [[0.1, 0.2, 0.0, 0.4], [-0.2, 0.5, 0.6, 0.3], [0.3, -0.1, 0.2, 0.0]],
        ],
        dtype=jnp.float32,
    )
    teacher_logits_b = teacher_logits_a * 2.0 - 0.7
    attention_mask = jnp.array([[1, 1, 1], [1, 1, 0]], dtype=jnp.int32)
    loss_mask = jnp.array([[1, 0, 1], [1, 1, 1]], dtype=jnp.int32)
    labels = jnp.array([[3, -100, 0], [1, 2, -100]], dtype=jnp.int32)

    expected_ce = _build_expected_ce(
        student_logits=student_logits,
        labels=labels,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )

    _, metrics_a = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits_a,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        use_hard_labels=True,
        temperature=3.5,
        alpha=0.4,
    )
    _, metrics_b = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits_b,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        use_hard_labels=True,
        temperature=3.5,
        alpha=0.4,
    )

    assert jnp.allclose(metrics_a["ce_loss"], expected_ce, atol=1e-6)
    assert jnp.allclose(metrics_b["ce_loss"], expected_ce, atol=1e-6)


def test_pure_ce_is_causally_shifted():
    """alpha=0 pure-CE trains logits[:, :-1] against labels[:, 1:] (BUG 1 regression).

    The old (buggy) path scored ``CE(logits[t], labels[t])``; the fix scores
    ``CE(logits[t], labels[t+1])``. Data is chosen so the shifted and unshifted
    references genuinely differ, so this fails on the old code and passes on the fix.
    """
    student_logits = jnp.array(
        [[[2.0, 0.1, -1.0, 0.3], [0.2, 1.5, -0.4, 0.0], [-0.5, 0.2, 1.1, 0.8]]],
        dtype=jnp.float32,
    )
    # Unshifted labels (trainer convention): label[t] == input_ids[t], -100 outside completion.
    labels = jnp.array([[-100, 1, 2]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)

    shifted_ref = _build_expected_ce(
        student_logits=student_logits,
        labels=labels,
        attention_mask=attention_mask,
        loss_mask=None,
        shift_tokens=True,
    )
    unshifted_ref = _build_expected_ce(
        student_logits=student_logits,
        labels=labels,
        attention_mask=attention_mask,
        loss_mask=None,
        shift_tokens=False,
    )
    # Guard against a vacuous assertion: the two references must actually differ.
    assert not jnp.allclose(shifted_ref, unshifted_ref, atol=1e-4)

    # teacher == student so KL == 0 and, at alpha=0, total loss == the hard-label CE.
    _, metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=student_logits,
        attention_mask=attention_mask,
        labels=labels,
        use_hard_labels=True,
        temperature=2.0,
        alpha=0.0,
    )
    assert jnp.allclose(metrics["ce_loss"], shifted_ref, atol=1e-6)
    assert not jnp.allclose(metrics["ce_loss"], unshifted_ref, atol=1e-4)

    # shift_tokens=False must reproduce the old unshifted behavior exactly.
    _, metrics_noshift = distillation_loss(
        student_logits=student_logits,
        teacher_logits=student_logits,
        attention_mask=attention_mask,
        labels=labels,
        use_hard_labels=True,
        temperature=2.0,
        alpha=0.0,
        shift_tokens=False,
    )
    assert jnp.allclose(metrics_noshift["ce_loss"], unshifted_ref, atol=1e-6)


def test_forward_kl_mask_is_causally_shifted():
    """Completion-only KD masks the KL by completion_mask[t+1] (BUG 2 regression).

    Position t predicts token t+1, so the forward-KL masked mean uses ``mask[:, 1:]``
    over student/teacher ``logits[:, :-1]``. The mask + data are chosen so the shifted
    and unshifted references differ, so this fails on the old code and passes on the fix.
    """
    student_logits = jnp.array(
        [[[1.0, 0.0, -0.5], [0.2, 0.9, -0.3], [-0.4, 0.5, 1.2]]],
        dtype=jnp.float32,
    )
    teacher_logits = jnp.array(
        [[[0.1, 0.6, -0.2], [0.4, -0.1, 0.3], [0.7, 0.2, -0.5]]],
        dtype=jnp.float32,
    )
    completion_mask = jnp.array([[0, 1, 1]], dtype=jnp.int32)
    temperature = 2.0
    t_sq = temperature**2

    def _forward_kl_masked(student, teacher, mask):
        s_lp = jax.nn.log_softmax(student / temperature, axis=-1)
        t_lp = jax.nn.log_softmax(teacher / temperature, axis=-1)
        per_token = jnp.sum(jnp.exp(t_lp) * (t_lp - s_lp), axis=-1)
        return jnp.sum(per_token * mask) / jnp.sum(mask) * t_sq

    shifted_ref = _forward_kl_masked(
        student_logits[:, :-1],
        teacher_logits[:, :-1],
        completion_mask[:, 1:].astype(jnp.float32),
    )
    unshifted_ref = _forward_kl_masked(
        student_logits,
        teacher_logits,
        completion_mask.astype(jnp.float32),
    )
    assert not jnp.allclose(shifted_ref, unshifted_ref, atol=1e-4)

    # Default classic-KD (beta=None) forward KL, pure distillation (alpha=1.0).
    _, metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        loss_mask=completion_mask,
        use_hard_labels=False,
        temperature=temperature,
        alpha=1.0,
    )
    assert jnp.allclose(metrics["kl_loss"], shifted_ref, atol=1e-5)
    assert not jnp.allclose(metrics["kl_loss"], unshifted_ref, atol=1e-4)

    _, metrics_noshift = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        loss_mask=completion_mask,
        use_hard_labels=False,
        temperature=temperature,
        alpha=1.0,
        shift_tokens=False,
    )
    assert jnp.allclose(metrics_noshift["kl_loss"], unshifted_ref, atol=1e-5)


def test_distillation_beta_uses_generalized_jsd():
    student_logits = jnp.array([[[1.0, 0.0, -0.5], [0.2, 0.5, -0.3]]], dtype=jnp.float32)
    teacher_logits = jnp.array([[[0.1, 0.6, -0.2], [0.4, -0.1, 0.3]]], dtype=jnp.float32)
    mask = jnp.array([[1, 0]], dtype=jnp.int32)
    beta = 0.25

    # shift_tokens=False: verify the GJSD math on a fixed (unshifted) window; the causal
    # shift is exercised separately by test_forward_kl_mask_is_causally_shifted.
    _, metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        loss_mask=mask,
        temperature=1.5,
        alpha=1.0,
        beta=beta,
        shift_tokens=False,
    )

    student_log_probs = jax.nn.log_softmax(student_logits / 1.5, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits / 1.5, axis=-1)
    mixture_log_probs = jax.scipy.special.logsumexp(
        jnp.stack(
            [
                student_log_probs + jnp.log1p(-beta),
                teacher_log_probs + jnp.log(beta),
            ]
        ),
        axis=0,
    )
    student_probs = jnp.exp(student_log_probs)
    teacher_probs = jnp.exp(teacher_log_probs)
    kl_teacher = jnp.sum(teacher_probs * (teacher_log_probs - mixture_log_probs), axis=-1)
    kl_student = jnp.sum(student_probs * (student_log_probs - mixture_log_probs), axis=-1)
    per_token = beta * kl_teacher + (1 - beta) * kl_student
    expected = jnp.sum(per_token * mask) / jnp.sum(mask) * (1.5**2)

    assert jnp.allclose(metrics["kl_loss"], expected, atol=1e-6)


def test_distillation_beta_endpoints_match_gkd_convention():
    """beta=0 -> forward KL(p_t || p_s); beta=1 -> reverse KL(p_s || p_t) (GKD-paper / TRL)."""
    student_logits = jnp.array([[[1.0, 0.0, -0.5], [0.2, 0.5, -0.3]]], dtype=jnp.float32)
    teacher_logits = jnp.array([[[0.1, 0.6, -0.2], [0.4, -0.1, 0.3]]], dtype=jnp.float32)
    mask = jnp.array([[1, 1]], dtype=jnp.int32)
    temperature = 2.0

    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)
    forward_kl = jnp.sum(jnp.exp(teacher_log_probs) * (teacher_log_probs - student_log_probs), axis=-1)
    reverse_kl = jnp.sum(jnp.exp(student_log_probs) * (student_log_probs - teacher_log_probs), axis=-1)
    t_sq = temperature**2

    # shift_tokens=False: this checks the forward/reverse KL endpoint math over both
    # positions; the causal shift is covered by test_forward_kl_mask_is_causally_shifted.
    _, metrics_fwd = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        loss_mask=mask,
        temperature=temperature,
        alpha=1.0,
        beta=0.0,
        shift_tokens=False,
    )
    assert jnp.allclose(metrics_fwd["kl_loss"], jnp.mean(forward_kl) * t_sq, atol=1e-5)

    _, metrics_rev = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        loss_mask=mask,
        temperature=temperature,
        alpha=1.0,
        beta=1.0,
        shift_tokens=False,
    )
    assert jnp.allclose(metrics_rev["kl_loss"], jnp.mean(reverse_kl) * t_sq, atol=1e-5)


def test_distillation_cakld_blends_forward_and_reverse_kl():
    student_logits = jnp.array([[[1.0, 0.0, -0.5], [0.2, 0.5, -0.3]]], dtype=jnp.float32)
    teacher_logits = jnp.array([[[0.1, 0.6, -0.2], [0.4, -0.1, 0.3]]], dtype=jnp.float32)
    mask = jnp.array([[1, 0]], dtype=jnp.int32)
    temperature = 1.5
    gamma = 0.7

    # shift_tokens=False: CAKLD forward/reverse blend on a fixed window.
    _, metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        loss_mask=mask,
        temperature=temperature,
        alpha=1.0,
        cakld_gamma=gamma,
        shift_tokens=False,
    )

    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)
    forward_kl = jnp.sum(jnp.exp(teacher_log_probs) * (teacher_log_probs - student_log_probs), axis=-1)
    reverse_kl = jnp.sum(jnp.exp(student_log_probs) * (student_log_probs - teacher_log_probs), axis=-1)
    per_token = gamma * reverse_kl + (1.0 - gamma) * forward_kl
    expected = jnp.sum(per_token * mask) / jnp.sum(mask) * (temperature**2)

    assert jnp.allclose(metrics["kl_loss"], expected, atol=1e-6)
    assert jnp.allclose(metrics["distill_xent_loss"], expected, atol=1e-6)
    assert jnp.allclose(metrics["teacher_entropy_loss"], 0.0, atol=1e-6)


def test_distillation_cakld_rejects_beta():
    logits = jnp.array([[[1.0, 0.0, -0.5]]], dtype=jnp.float32)
    with pytest.raises(ValueError, match="cakld_gamma"):
        distillation_loss(
            student_logits=logits,
            teacher_logits=logits,
            beta=0.5,
            cakld_gamma=0.5,
        )


def test_gkd_jsd_beta_endpoints_match_trl_convention():
    """GKD generalized_jsd_loss: beta=0 -> forward KL, beta=1 -> reverse KL (TRL parity, no T^2)."""
    from easydel.trainers.generalized_knowledge_distillation_trainer._fn import generalized_jsd_loss

    student_logits = jnp.array([[[1.0, 0.0, -0.5], [0.2, 0.5, -0.3]]], dtype=jnp.float32)
    teacher_logits = jnp.array([[[0.1, 0.6, -0.2], [0.4, -0.1, 0.3]]], dtype=jnp.float32)
    temperature = 1.5

    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)
    forward_kl = jnp.mean(jnp.sum(jnp.exp(teacher_log_probs) * (teacher_log_probs - student_log_probs), axis=-1))
    reverse_kl = jnp.mean(jnp.sum(jnp.exp(student_log_probs) * (student_log_probs - teacher_log_probs), axis=-1))

    got_fwd = generalized_jsd_loss(student_logits, teacher_logits, beta=0.0, temperature=temperature)
    got_rev = generalized_jsd_loss(student_logits, teacher_logits, beta=1.0, temperature=temperature)
    assert jnp.allclose(got_fwd, forward_kl, atol=1e-5)
    assert jnp.allclose(got_rev, reverse_kl, atol=1e-5)


def test_distillation_topk_tail_matches_bucketed_loss():
    student_logits = jnp.array([[[0.1, 0.7, -0.2, 0.5]]], dtype=jnp.float32)
    teacher_logits = jnp.array([[[1.2, 0.4, -0.1, 0.2]]], dtype=jnp.float32)

    # Single-position example: shift_tokens=False keeps the position (a shift would empty it)
    # and lets the top-k tail bucketing be checked directly against the manual reference.
    _, metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=1.0,
        alpha=1.0,
        loss_top_k=2,
        loss_add_tail=True,
        shift_tokens=False,
    )

    teacher_log_probs = jax.nn.log_softmax(teacher_logits, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    top_teacher_log_probs, top_indices = jax.lax.top_k(teacher_log_probs, 2)
    top_student_log_probs = jnp.take_along_axis(student_log_probs, top_indices, axis=-1)
    top_teacher_probs = jnp.exp(top_teacher_log_probs)
    top_student_probs = jnp.exp(top_student_log_probs)
    teacher_tail = 1.0 - jnp.sum(top_teacher_probs, axis=-1)
    student_tail = 1.0 - jnp.sum(top_student_probs, axis=-1)
    xent = -(jnp.sum(top_teacher_probs * top_student_log_probs, axis=-1) + teacher_tail * jnp.log(student_tail))
    entropy = -(jnp.sum(top_teacher_probs * top_teacher_log_probs, axis=-1) + teacher_tail * jnp.log(teacher_tail))

    assert jnp.allclose(metrics["kl_loss"], jnp.mean(xent - entropy), atol=1e-6)


def test_teacher_branch_is_stop_gradient():
    student_logits = jnp.array(
        [
            [[0.2, 0.3, -0.1], [0.5, -0.2, 0.7]],
            [[0.1, -0.4, 0.6], [0.9, 0.2, -0.3]],
        ],
        dtype=jnp.float32,
    )
    teacher_logits = jnp.array(
        [
            [[0.0, 0.1, 0.2], [0.3, 0.4, -0.2]],
            [[-0.1, 0.5, 0.0], [0.2, -0.3, 0.7]],
        ],
        dtype=jnp.float32,
    )

    def _loss_for_teacher(t_logits):
        loss, _ = distillation_loss(
            student_logits=student_logits,
            teacher_logits=t_logits,
            use_hard_labels=False,
            temperature=2.0,
            alpha=1.0,
        )
        return loss

    teacher_grad = jax.grad(_loss_for_teacher)(teacher_logits)
    assert jnp.allclose(teacher_grad, jnp.zeros_like(teacher_grad), atol=1e-7)


def test_chunked_z_loss_matches_manual_reference():
    """z_loss = coef * mean over valid tokens of logsumexp(raw student logits)^2,
    folded into the chunked total and reported as a component."""
    key = jax.random.PRNGKey(7)
    ks, kt = jax.random.split(key)
    B, L, V = 2, 8, 16
    student_hidden = jax.random.normal(ks, (B, L, V), jnp.float32)
    teacher_hidden = jax.random.normal(kt, (B, L, V), jnp.float32)
    attention_mask = jnp.asarray([[1] * 8, [1] * 5 + [0] * 3], jnp.int32)
    coef = 1e-2

    # shift_tokens=False: the manual z-loss reference below is over the full (unshifted)
    # logsumexp window; the causal shift itself is covered by the CE/KL shift tests.
    base_total, base_components = chunked_distillation_loss(
        student_hidden=student_hidden,
        teacher_hidden=teacher_hidden,
        student_lm_head_fn=_identity_lm_head,
        teacher_lm_head_fn=_identity_lm_head,
        attention_mask=attention_mask,
        temperature=1.0,
        alpha=1.0,
        chunk_size=4,
        shift_tokens=False,
    )
    total, components = chunked_distillation_loss(
        student_hidden=student_hidden,
        teacher_hidden=teacher_hidden,
        student_lm_head_fn=_identity_lm_head,
        teacher_lm_head_fn=_identity_lm_head,
        attention_mask=attention_mask,
        temperature=1.0,
        alpha=1.0,
        chunk_size=4,
        z_loss=coef,
        shift_tokens=False,
    )

    mask = attention_mask.astype(jnp.float32)
    lse = jax.nn.logsumexp(student_hidden.astype(jnp.float32), axis=-1)
    expected_z = coef * jnp.sum(jnp.square(lse) * mask) / jnp.sum(mask)

    assert "z_loss" not in base_components
    assert jnp.allclose(components["z_loss"], expected_z, rtol=1e-5)
    assert jnp.allclose(total, base_total + expected_z, rtol=1e-5)
    # KL terms untouched by the regularizer
    assert jnp.allclose(components["kl_loss"], base_components["kl_loss"], rtol=1e-6)


def test_chunked_z_loss_zero_coefficient_is_identity():
    key = jax.random.PRNGKey(3)
    hidden = jax.random.normal(key, (1, 8, 8), jnp.float32)
    kwargs = dict(
        student_hidden=hidden,
        teacher_hidden=hidden,
        student_lm_head_fn=_identity_lm_head,
        teacher_lm_head_fn=_identity_lm_head,
        temperature=1.0,
        alpha=1.0,
        chunk_size=4,
        shift_tokens=False,
    )
    total_off, comps_off = chunked_distillation_loss(**kwargs)
    total_default, comps_default = chunked_distillation_loss(**kwargs, z_loss=0.0)
    assert jnp.allclose(total_off, total_default)
    assert "z_loss" not in comps_off and "z_loss" not in comps_default
