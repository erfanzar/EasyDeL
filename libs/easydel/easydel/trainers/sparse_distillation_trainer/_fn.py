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
    shift_tokens: bool = True,
) -> tuple[Array, dict[str, Array]]:
    """Compute partial KL distillation loss from sparse teacher logprobs.

    Given the teacher's top-k token indices and their log-probabilities, this
    function computes a partial KL divergence between the renormalised teacher
    distribution (over top-k tokens) and the student's full distribution
    evaluated at those same indices.

    The loss is the true partial KL divergence:
        partial_KL = sum_k( teacher_prob_k * (log teacher_prob_k - student_logprob_k) )

    where ``teacher_prob_k`` is obtained by applying temperature scaling and
    softmax over the teacher's top-k logprobs (renormalising them to sum to
    1), and ``student_logprob_k`` is the student's log-softmax evaluated at
    the teacher's top-k token indices. All masks, normalizers, and loss
    reductions accumulate in ``float32`` to preserve precision under
    low-precision (e.g. ``bfloat16``) logits.

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
        shift_tokens: When ``True`` (default), ``loss_mask`` / ``attention_mask`` / ``labels`` are token-indexed
            and are shifted left by one so that the distribution at position ``t`` (which predicts token
            ``t+1``) is weighted by the mask of token ``t+1``. Set ``False`` if the masks are already aligned
            with the logits.

    Returns:
        Tuple of ``(total_loss, {"kl_loss": ..., "ce_loss": ...})``.
    """
    alpha_s = jnp.asarray(alpha, dtype=jnp.float32)
    temp_sq = jnp.asarray(temperature * temperature, dtype=jnp.float32)

    student_log_probs = jax.nn.log_softmax(
        student_logits.astype(jnp.float32) / temperature,
        axis=-1,
    )  # [B, L, V]

    student_logprobs_at_k = jnp.take_along_axis(
        student_log_probs,
        teacher_top_k_indices,
        axis=-1,
    )  # [B, L, K]

    teacher_scaled_logprobs = jax.lax.stop_gradient(
        teacher_top_k_logprobs.astype(jnp.float32) / temperature
    )
    teacher_log_probs_at_k = jax.nn.log_softmax(teacher_scaled_logprobs, axis=-1)  # [B, L, K]
    teacher_probs_at_k = jnp.exp(teacher_log_probs_at_k)  # [B, L, K]

    # True partial KL = sum_k p_t(k) * (log p_t(k) - log p_s(k)) = CE(p_t, p_s) - H(p_t).
    # Guard p_t == 0 so padded -inf slots never produce 0 * (-inf) = NaN.
    per_k_kl = jnp.where(
        teacher_probs_at_k > 0.0,
        teacher_probs_at_k * (teacher_log_probs_at_k - student_logprobs_at_k),
        0.0,
    )
    per_token_partial_kl = jnp.sum(per_k_kl, axis=-1)  # [B, L] float32

    if loss_mask is not None:
        mask = loss_mask.astype(jnp.float32)
    elif attention_mask is not None:
        mask = attention_mask.astype(jnp.float32)
    else:
        mask = None

    if shift_tokens:
        if mask is None:
            mask = jnp.ones(student_logits.shape[:-1], dtype=jnp.float32)
        # Logits/top-k at position t describe token t+1, so the token-level masks and labels (which are indexed
        # by the token they describe) must be shifted left by one to line up with the per-position distributions.
        if mask is not None:
            mask = jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)
        if labels is not None:
            labels = jnp.concatenate([labels[:, 1:], jnp.full_like(labels[:, :1], -100)], axis=1)

    if labels is not None:
        valid_label_mask = (labels != -100).astype(jnp.float32)
        mask = valid_label_mask if mask is None else mask * valid_label_mask

    if mask is not None:
        normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=jnp.float32))
        kl_loss = jnp.sum(per_token_partial_kl * mask) / normalizer
    else:
        normalizer = jnp.array(1.0, dtype=jnp.float32)
        kl_loss = jnp.mean(per_token_partial_kl)
    kl_loss = kl_loss * temp_sq

    ce_loss = jnp.array(0.0, dtype=jnp.float32)
    if use_hard_labels and labels is not None:
        safe_labels = jnp.where(labels == -100, 0, labels)
        per_token_ce = optax.softmax_cross_entropy_with_integer_labels(
            student_logits.astype(jnp.float32),
            safe_labels,
        )
        if mask is not None:
            ce_loss = jnp.sum(per_token_ce * mask) / normalizer
        else:
            ce_loss = jnp.mean(per_token_ce)
        total_loss = alpha_s * kl_loss + (jnp.array(1.0, dtype=jnp.float32) - alpha_s) * ce_loss
    else:
        total_loss = kl_loss

    metrics = {
        "kl_loss": jnp.asarray(kl_loss, dtype=jnp.float32),
        "ce_loss": jnp.asarray(ce_loss, dtype=jnp.float32),
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
    scope_root = "easydel/trainer/sparse_distillation/" + ("train_step" if is_training else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=partition_spec,
        )
        batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    shift_tokens = bool(getattr(loss_config, "shift_tokens", True)) if loss_config is not None else True
    z_loss_coef = float(getattr(loss_config, "z_loss", 0.0) or 0.0) if loss_config is not None else 0.0

    def loss_fn(tree, minibatch):
        """Compute the sparse-distillation loss for one minibatch.

        Args:
            tree: Differentiable parameter tree.
            minibatch (collections.abc.Mapping[str, jax.Array]): Minibatch
                with ``input_ids``, ``attention_mask``,
                ``teacher_top_k_indices``, ``teacher_top_k_logprobs`` and
                optional ``completion_mask`` / ``labels``.

        Returns:
            tuple[jax.Array, LossMetrics]: ``(scalar_loss, metrics)``.
        """
        with jax.named_scope(scope_root + "/loss_fn"):
            if is_training and straight_through_emulator is not None:
                with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                    tree = straight_through_emulator(tree)
            with jax.named_scope(scope_root + "/loss_fn/merge_state"):
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
            with jax.named_scope(scope_root + "/loss_fn/student_forward"):
                student_outputs = module(**call_kwargs)

            with jax.named_scope(scope_root + "/loss_fn/partial_kl_loss"):
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
                    shift_tokens=shift_tokens,
                )

            if z_loss_coef > 0.0:
                z_logits = student_outputs.logits
                z_mask = completion_mask if completion_mask is not None else attention_mask
                z_mask = (
                    jnp.ones(z_logits.shape[:2], dtype=jnp.float32) if z_mask is None else z_mask.astype(jnp.float32)
                )
                z_labels = labels
                if shift_tokens:
                    z_mask = jnp.concatenate([z_mask[:, 1:], jnp.zeros_like(z_mask[:, :1])], axis=1)
                    if z_labels is not None:
                        z_labels = jnp.concatenate([z_labels[:, 1:], jnp.full_like(z_labels[:, :1], -100)], axis=1)
                if z_labels is not None:
                    z_mask = z_mask * (z_labels != -100).astype(jnp.float32)
                lse = jax.nn.logsumexp(z_logits.astype(jnp.float32), axis=-1)
                z_value = (
                    jnp.asarray(z_loss_coef, jnp.float32)
                    * jnp.sum(jnp.square(lse) * z_mask)
                    / jnp.maximum(jnp.sum(z_mask), 1.0)
                )
                total_loss = total_loss + z_value
                loss_components["z_loss"] = z_value

            aux_loss = getattr(student_outputs, "aux_loss", None)
            if aux_loss is not None:
                aux_loss = aux_loss.astype(jnp.float32)
                total_loss = total_loss + aux_loss
                loss_components["aux_loss"] = aux_loss

            metrics = LossMetrics(
                loss=total_loss,
                z_loss=loss_components.get("z_loss"),
                other_metrics={key: jnp.asarray(value) for key, value in loss_components.items()},
            )
            return total_loss, metrics

    def loss_token_mass(_tree, minibatch) -> jax.Array:
        """Compute supervised token mass per microbatch for exact GA weighting."""
        mask = minibatch.get("completion_mask", minibatch.get("attention_mask"))
        ids = minibatch["input_ids"]
        mask_f32 = jnp.ones(tuple(ids.shape[:2]), dtype=jnp.float32) if mask is None else mask.astype(jnp.float32)
        lbls = minibatch.get("labels")
        if shift_tokens:
            mask_f32 = mask_f32[:, 1:]
            if lbls is not None:
                lbls = lbls[:, 1:]
        if lbls is not None:
            mask_f32 = mask_f32 * (lbls != -100).astype(jnp.float32)
        return jnp.sum(mask_f32)

    if is_training:
        with jax.named_scope(scope_root + "/grad_and_minibatch"):
            gradients, metrics = minibatch_call(
                state=state,
                batch=batch,
                minibatch_size=minibatch_size,
                grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
                microbatch_weight_fn=loss_token_mass if gradient_accumulation_steps > 1 else None,
            )
        with jax.named_scope(scope_root + "/update_state"):
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
        with jax.named_scope(scope_root + "/eval_call"):
            _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics
