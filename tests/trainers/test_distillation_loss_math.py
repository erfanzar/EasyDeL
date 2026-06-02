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
) -> jnp.ndarray:
    dtype = student_logits.dtype
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


def test_distillation_beta_uses_generalized_jsd():
    student_logits = jnp.array([[[1.0, 0.0, -0.5], [0.2, 0.5, -0.3]]], dtype=jnp.float32)
    teacher_logits = jnp.array([[[0.1, 0.6, -0.2], [0.4, -0.1, 0.3]]], dtype=jnp.float32)
    mask = jnp.array([[1, 0]], dtype=jnp.int32)
    beta = 0.25

    _, metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        loss_mask=mask,
        temperature=1.5,
        alpha=1.0,
        beta=beta,
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


def test_distillation_topk_tail_matches_bucketed_loss():
    student_logits = jnp.array([[[0.1, 0.7, -0.2, 0.5]]], dtype=jnp.float32)
    teacher_logits = jnp.array([[[1.2, 0.4, -0.1, 0.2]]], dtype=jnp.float32)

    _, metrics = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=1.0,
        alpha=1.0,
        loss_top_k=2,
        loss_add_tail=True,
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
