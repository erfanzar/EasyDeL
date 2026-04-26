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

"""Internal functions for sparse (gray-box) knowledge distillation.

This module implements partial KL divergence loss where only the teacher's
top-k logprobs are available.  The student minimises KL between its full
distribution and the teacher's sparse top-k distribution (renormalised).

Key function:
    ``partial_kl_distillation_loss``: Computes KL(teacher_topk || student)
    using only the teacher's top-k token indices and log-probabilities.
"""

import collections.abc
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    sanitize_model_call_kwargs,
    update_metrics,
    update_state_respectfully,
)


def partial_kl_distillation_loss(
    student_logits: Array,
    teacher_top_k_indices: Array,
    teacher_top_k_logprobs: Array,
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    labels: Array | None = None,
    use_hard_labels: bool = False,
    temperature: float = 4.0,
    alpha: float = 0.9,
) -> tuple[Array, dict[str, Array]]:
    """Compute partial KL distillation loss from sparse teacher logprobs.

    Given the teacher's top-k token indices and their log-probabilities, this
    function computes a partial KL divergence between the renormalised teacher
    distribution (over top-k tokens) and the student's full distribution
    evaluated at those same indices.

    The loss is:
        partial_KL = -sum_k( teacher_prob_k * student_logprob_k )

    where ``teacher_prob_k`` is obtained by applying temperature scaling and
    softmax over the teacher's top-k logprobs (renormalising them to sum to
    1), and ``student_logprob_k`` is the student's log-softmax evaluated at
    the teacher's top-k token indices.

    Args:
        student_logits: Student model logits. Shape: ``[B, L, V]``.
        teacher_top_k_indices: Token indices for teacher's top-k.
            Shape: ``[B, L, K]``.
        teacher_top_k_logprobs: Log-probabilities for teacher's top-k tokens.
            Shape: ``[B, L, K]``.
        attention_mask: Mask for valid tokens (1=valid, 0=pad).
            Shape: ``[B, L]``.
        loss_mask: Task-specific token mask. Takes priority over
            ``attention_mask`` when provided. Shape: ``[B, L]``.
        labels: Ground truth labels for optional supervised loss.
            Shape: ``[B, L]``.
        use_hard_labels: Whether to include supervised CE loss.
        temperature: Temperature for softening distributions.
        alpha: Weight for distillation loss (1.0=pure distillation).

    Returns:
        Tuple of ``(total_loss, {"kl_loss": ..., "ce_loss": ...})``.
    """
    dtype = student_logits.dtype
    alpha_s = jnp.array(alpha, dtype=dtype)
    temp_sq = jnp.array(temperature * temperature, dtype=dtype)

    student_log_probs = jax.nn.log_softmax(
        student_logits.astype(jnp.float32) / temperature,
        axis=-1,
    )  # [B, L, V]

    # Gather student log-probs at teacher's top-k indices
    student_logprobs_at_k = jnp.take_along_axis(
        student_log_probs,
        teacher_top_k_indices,
        axis=-1,
    )  # [B, L, K]

    teacher_probs_at_k = jax.nn.softmax(
        teacher_top_k_logprobs.astype(jnp.float32) / temperature,
        axis=-1,
    )  # [B, L, K]

    per_token_partial_kl = -jnp.sum(
        teacher_probs_at_k * student_logprobs_at_k,
        axis=-1,
    ).astype(dtype)  # [B, L]

    if loss_mask is not None:
        mask = loss_mask.astype(dtype)
    elif attention_mask is not None:
        mask = attention_mask.astype(dtype)
    else:
        mask = None

    if labels is not None:
        valid_label_mask = (labels != -100).astype(dtype)
        mask = valid_label_mask if mask is None else mask * valid_label_mask

    if mask is not None:
        normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
        kl_loss = jnp.sum(per_token_partial_kl * mask) / normalizer
    else:
        kl_loss = jnp.mean(per_token_partial_kl)
    kl_loss = kl_loss * temp_sq

    total_loss = alpha_s * kl_loss

    ce_loss = jnp.array(0.0, dtype=dtype)
    if use_hard_labels and labels is not None:
        safe_labels = jnp.where(labels == -100, 0, labels)
        per_token_ce = optax.softmax_cross_entropy_with_integer_labels(
            student_logits.astype(jnp.float32),
            safe_labels,
        ).astype(dtype)
        if mask is not None:
            ce_loss = jnp.sum(per_token_ce * mask) / normalizer
        else:
            ce_loss = jnp.mean(per_token_ce)
        total_loss = total_loss + (jnp.array(1.0, dtype=dtype) - alpha_s) * ce_loss

    metrics = {
        "kl_loss": jnp.asarray(kl_loss, dtype=dtype),
        "ce_loss": jnp.asarray(ce_loss, dtype=dtype),
    }
    return total_loss, metrics


def sparse_distillation_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    temperature: float = 4.0,
    alpha: float = 0.9,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Training/evaluation step for sparse (gray-box) distillation.

    The batch is expected to contain pre-computed teacher top-k data:
        - ``teacher_top_k_indices``: ``[B, L, K]``
        - ``teacher_top_k_logprobs``: ``[B, L, K]``

    No ``teacher_state`` is needed because teacher scoring has already been
    performed in ``_preprocess_batch_input``.

    Args:
        state: Current student model state.
        batch: Batch with input_ids, attention_mask, completion_mask,
            teacher_top_k_indices, teacher_top_k_logprobs.
        loss_config: Optional loss configuration.
        learning_rate_fn: Learning rate schedule function.
        partition_spec: Sharding specification for the batch.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        is_training: Whether this is a training step.
        temperature: Temperature for softening distributions in KL loss.
        alpha: Weight for distillation loss.
        straight_through_emulator: Optional quantization emulator.

    Returns:
        If training: ``(updated_state, metrics)``
        If evaluation: ``metrics``
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree)

        input_ids = minibatch["input_ids"]
        attention_mask = minibatch["attention_mask"]
        completion_mask = minibatch.get("completion_mask")
        labels = minibatch.get("labels")
        teacher_top_k_indices = minibatch["teacher_top_k_indices"]
        teacher_top_k_logprobs = minibatch["teacher_top_k_logprobs"]

        call_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        call_kwargs = filter_kwargs_for_callable(getattr(module, "forward", module), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        student_outputs = module(**call_kwargs)

        total_loss, loss_components = partial_kl_distillation_loss(
            student_logits=student_outputs.logits,
            teacher_top_k_indices=teacher_top_k_indices,
            teacher_top_k_logprobs=teacher_top_k_logprobs,
            attention_mask=attention_mask,
            loss_mask=completion_mask,
            labels=labels,
            use_hard_labels=(labels is not None),
            temperature=temperature,
            alpha=alpha,
        )

        metrics = LossMetrics(
            loss=total_loss,
            other_metrics={key: jnp.asarray(value) for key, value in loss_components.items()},
        )
        return total_loss, metrics

    if is_training:
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
        state = update_state_respectfully(
            state=state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=update_metrics(
                metrics=metrics,
                learning_rate_fn=learning_rate_fn,
                step=state.step,
                gradients=gradients,
            ),
        )
        return state, metrics
    else:
        _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics
