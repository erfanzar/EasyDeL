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

import collections.abc
import typing as tp

import jax
from jax import numpy as jnp
from jax.nn import log_sigmoid
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers._logprob_utils import (
    compute_sequence_scores_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)
from easydel.trainers._shared import apply_paired_truncation, gather_multimodal_kwargs

from ..training_utils import (
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    sanitize_model_call_kwargs,
    update_metrics,
    update_state_respectfully,
)


class RunningMoments:
    """Simple running mean/variance tracker for BCO delta parameter.

    Tracks running statistics compatible with host-side updates for maintaining
    the BCO delta parameter across training steps.
    """

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-24

    def update(self, values: collections.abc.Sequence[float] | jnp.ndarray | None):
        """Update running statistics with new values.

        Args:
            values: New values to incorporate into running statistics.
        """
        if values is None:
            return
        arr = jnp.asarray(values).reshape(-1)
        if arr.size == 0:
            return
        batch_mean = float(arr.mean())
        batch_var = float(arr.var())
        batch_count = float(arr.size)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_sum = batch_var * batch_count
        old_sum = self.var * self.count + delta * delta * self.count * batch_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * batch_count / tot_count
        self.var = tot_sum / tot_count
        self.count = tot_count

    def as_dict(self) -> dict[str, float]:
        """Export statistics as dictionary.

        Returns:
            Dictionary with mean, var, and count.
        """
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_dict(self, data: dict[str, float]):
        """Load statistics from dictionary.

        Args:
            data: Dictionary with mean, var, and count.
        """
        self.mean = float(data.get("mean", 0.0))
        self.var = float(data.get("var", 1.0))
        self.count = float(data.get("count", 1e-24))


def concatenated_forward(
    model: EasyDeLBaseModule,
    batch: dict[str, jax.Array],
    *,
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: int,
    max_length: int | None = None,
    truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    aux_loss_enabled: bool = False,
    logprob_vocab_chunk_size: int | None = None,
) -> dict[str, jax.Array]:
    """Run model forward pass to compute completion log probabilities.

    Args:
        model: Model to run forward pass on.
        batch: Input batch with prompts and completions.
        is_encoder_decoder: Whether model uses encoder-decoder architecture.
        label_pad_token_id: Token ID for padding labels.
        padding_value: Padding token ID.
        max_length: Maximum sequence length.
        truncation_mode: How to truncate sequences.
        aux_loss_enabled: Whether to compute auxiliary loss.

    Returns:
        Dictionary with completion log probabilities and logits.
    """

    prompt_input_ids = batch["prompt_input_ids"]
    prompt_attention_mask = batch["prompt_attention_mask"]
    completion_input_ids = batch["completion_input_ids"]
    completion_attention_mask = batch["completion_attention_mask"]
    completion_labels = batch["completion_labels"]

    model_kwargs: dict[str, jax.Array] = gather_multimodal_kwargs(batch, aux_loss_enabled=aux_loss_enabled)

    lmhead_chunksize = None

    if is_encoder_decoder:
        call_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "decoder_input_ids": batch.get("completion_decoder_input_ids"),
            "labels": completion_labels,
            **model_kwargs,
        }
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        outputs = model(**call_kwargs)
        logits = outputs.logits
        loss_mask = completion_attention_mask.astype(bool)
        labels_safe = jnp.where(loss_mask, completion_labels, 0)
        gathered_logps, _ = compute_token_logps_and_entropies_chunked(
            logits,
            labels_safe,
            return_entropy=False,
            chunk_size=logprob_vocab_chunk_size,
        )
        completion_logps = jnp.where(loss_mask, gathered_logps, 0.0).sum(axis=1)
        token_logit_sums = jnp.sum(
            jnp.where(loss_mask, logits.astype(jnp.float32).sum(axis=-1), 0.0),
            axis=1,
        )
        token_counts = jnp.sum(loss_mask.astype(jnp.float32), axis=1)
        mean_logits = token_logit_sums.sum() / jnp.maximum(token_counts.sum(), 1.0)
    else:
        input_ids, attention_mask, completion_labels = apply_paired_truncation(
            completion_input_ids,
            completion_attention_mask,
            completion_labels,
            max_length=max_length,
            truncation_mode=truncation_mode,
        )
        call_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **model_kwargs,
        }
        lmhead_chunksize = resolve_lmhead_chunksize(model)
        if lmhead_chunksize is not None:
            call_kwargs["apply_lm_head"] = False
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        outputs = model(**call_kwargs)
        logits = outputs.logits

        labels_shifted = completion_labels[:, 1:]
        loss_mask = labels_shifted != label_pad_token_id
        labels_safe = jnp.where(loss_mask, labels_shifted, 0)
        if logits is None and lmhead_chunksize is not None:
            hidden_states = outputs.last_hidden_state
            if hidden_states is None:
                raise TypeError(
                    f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
                )
            hidden_states = hidden_states[:, :-1, :]
            completion_logps, token_logit_sums, token_counts = compute_sequence_scores_from_hidden_states(
                model=model,
                hidden_states=hidden_states,
                labels=labels_safe,
                loss_mask=loss_mask,
                token_chunk_size=lmhead_chunksize,
                vocab_chunk_size=logprob_vocab_chunk_size,
            )
        else:
            logits_shifted = logits[:, :-1, :]
            gathered_logps, _ = compute_token_logps_and_entropies_chunked(
                logits_shifted,
                labels_safe,
                return_entropy=False,
                chunk_size=logprob_vocab_chunk_size,
            )
            completion_logps = jnp.where(loss_mask, gathered_logps, 0.0).sum(axis=1)
            token_logit_sums = jnp.sum(
                jnp.where(loss_mask, logits_shifted.astype(jnp.float32).sum(axis=-1), 0.0),
                axis=1,
            )
            token_counts = jnp.sum(loss_mask.astype(jnp.float32), axis=1)

        mean_logits = token_logit_sums.sum() / jnp.maximum(token_counts.sum(), 1.0)

    output = {
        "completion_logps": completion_logps,
        "completion_logits": logits,
        "mean_completion_logits": mean_logits,
    }

    if aux_loss_enabled and hasattr(outputs, "aux_loss"):
        output["aux_loss"] = outputs.aux_loss
    return output


def compute_bco_loss(
    policy_logps: jax.Array,
    reference_logps: jax.Array,
    chosen_mask: jax.Array,
    rejected_mask: jax.Array,
    *,
    beta: float,
    delta: float,
    udm_weights: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute BCO loss and rewards for desirable/undesirable examples.

    Args:
        policy_logps: Policy model log probabilities.
        reference_logps: Reference model log probabilities.
        chosen_mask: Mask for desirable examples.
        rejected_mask: Mask for undesirable examples.
        beta: Temperature parameter.
        delta: Running delta parameter.
        udm_weights: Optional UDM weights for handling distribution mismatch.

    Returns:
        Tuple of (loss, chosen_rewards, rejected_rewards, chosen_losses, rejected_losses, chosen_mask_f, rejected_mask_f).
    """

    chosen_mask_f = chosen_mask.astype(policy_logps.dtype)
    rejected_mask_f = rejected_mask.astype(policy_logps.dtype)

    reward_delta = beta * (policy_logps - reference_logps)
    chosen_rewards = reward_delta * chosen_mask_f
    rejected_rewards = reward_delta * rejected_mask_f

    chosen_losses = jnp.where(chosen_mask_f > 0, -log_sigmoid(chosen_rewards - delta), 0.0)
    rejected_losses = jnp.where(rejected_mask_f > 0, -log_sigmoid(-(rejected_rewards - delta)), 0.0)

    chosen_count = jnp.maximum(chosen_mask_f.sum(), 0.0)
    rejected_count = jnp.maximum(rejected_mask_f.sum(), 0.0)

    if udm_weights is not None:
        rejected_weights = udm_weights.astype(policy_logps.dtype) * rejected_mask_f
        total_weight = jnp.maximum(chosen_count + rejected_weights.sum(), 1.0)
        loss = (chosen_losses.sum() + (rejected_losses * rejected_weights).sum()) / total_weight
    else:
        total_weight = jnp.maximum(chosen_count + rejected_count, 1.0)
        loss = (chosen_losses.sum() + rejected_losses.sum()) / total_weight

    return (
        loss,
        chosen_rewards,
        rejected_rewards,
        chosen_losses,
        rejected_losses,
        chosen_mask_f,
        rejected_mask_f,
    )


def training_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState | None,
    learning_rate_fn: tp.Callable | None,
    concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]],
    beta: float,
    loss_config: LossConfig | None,
    partition_spec: PartitionSpec | None,
    gradient_accumulation_steps: int,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Execute BCO training step with gradient computation.

    Args:
        state: Current model state.
        batch: Training batch.
        reference_state: Reference model state.
        learning_rate_fn: Function mapping step to learning rate.
        concatenated_forward_fn: Forward function.
        beta: Temperature parameter.
        loss_config: Optional loss configuration.
        partition_spec: Sharding specification.
        gradient_accumulation_steps: Number of gradient accumulation steps.

    Returns:
        Updated state and loss metrics.
    """

    running_delta = batch.get("running_mean")
    if running_delta is None:
        running_delta = jnp.array(0.0, dtype=jnp.float32)
    else:
        running_delta = jnp.asarray(running_delta).reshape(())

    step_batch = dict(batch)
    step_batch.pop("running_mean", None)

    _, minibatch_size, batch_partition_spec = make_assertions_and_get_sizes(
        batch=step_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    step_batch = with_sharding_constraint(step_batch, batch_partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def calculate_loss(tree: jax.ArrayTree, call_batch: dict[str, jax.Array]):
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        model = state.merge(tree=tree)
        policy_outputs = concatenated_forward_fn(model, call_batch)
        completion_logps = policy_outputs["completion_logps"]

        labels = call_batch["label"].astype(bool)
        chosen_mask = labels
        rejected_mask = jnp.logical_not(labels)

        if "reference_logps" in call_batch:
            reference_completion_logps = call_batch["reference_logps"]
        else:
            if reference_state is None:
                reference_model = model
            else:
                reference_model = reference_state.model
            ref_outputs = concatenated_forward_fn(reference_model, call_batch)
            reference_completion_logps = jax.lax.stop_gradient(ref_outputs["completion_logps"])

        udm_weights = call_batch.get("udm_weights", None)
        if udm_weights is not None:
            rejected_weights = jnp.where(rejected_mask, udm_weights.astype(completion_logps.dtype), 0.0)
        else:
            rejected_weights = None

        (
            losses,
            chosen_rewards_masked,
            rejected_rewards_masked,
            chosen_losses,
            rejected_losses,
            chosen_valid,
            rejected_valid,
        ) = compute_bco_loss(
            completion_logps,
            reference_completion_logps,
            chosen_mask,
            rejected_mask,
            beta=beta,
            delta=running_delta,
            udm_weights=rejected_weights,
        )

        if policy_outputs.get("aux_loss") is not None:
            losses = losses + policy_outputs["aux_loss"]

        metrics = LossMetrics(
            loss=losses,
            chosen_rewards=chosen_rewards_masked,
            rejected_rewards=rejected_rewards_masked,
            other_metrics={
                "delta": running_delta,
                "logps/chosen": (completion_logps * chosen_mask.astype(completion_logps.dtype)).sum()
                / jnp.maximum(chosen_valid.sum(), 1.0),
                "logps/rejected": (completion_logps * rejected_mask.astype(completion_logps.dtype)).sum()
                / jnp.maximum(rejected_valid.sum(), 1.0),
                "logits/mean": policy_outputs["mean_completion_logits"],
                "count/chosen": chosen_valid.sum(),
                "count/rejected": rejected_valid.sum(),
                "loss/chosen": chosen_losses.sum() / jnp.maximum(chosen_valid.sum(), 1.0),
                "loss/rejected": rejected_losses.sum() / jnp.maximum(rejected_valid.sum(), 1.0),
            },
        )
        return metrics.loss, metrics

    gradients, metrics = minibatch_call(
        state=state,
        batch=step_batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
    )

    metrics = update_metrics(
        metrics=metrics,
        learning_rate_fn=learning_rate_fn,
        step=state.step,
        gradients=gradients,
    )
    new_state = update_state_respectfully(
        state=state,
        gradients=gradients,
        loss_config=loss_config,
        metrics=metrics,
    )
    return new_state, metrics


def evaluation_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState | None,
    concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]],
    beta: float,
) -> LossMetrics:
    """Execute BCO evaluation step without gradients.

    Args:
        state: Current model state.
        batch: Evaluation batch.
        reference_state: Reference model state.
        concatenated_forward_fn: Forward function.
        beta: Temperature parameter.

    Returns:
        Loss metrics.
    """

    running_delta = batch.get("running_mean")
    if running_delta is None:
        running_delta = jnp.array(0.0, dtype=jnp.float32)
    else:
        running_delta = jnp.asarray(running_delta).reshape(())

    policy_outputs = concatenated_forward_fn(state.model, batch)
    completion_logps = policy_outputs["completion_logps"]

    labels = batch["label"].astype(bool)
    chosen_mask = labels
    rejected_mask = jnp.logical_not(labels)

    if "reference_logps" in batch:
        reference_completion_logps = batch["reference_logps"]
    else:
        if reference_state is None:
            reference_model = state.model
        else:
            reference_model = reference_state.model
        ref_outputs = concatenated_forward_fn(reference_model, batch)
        reference_completion_logps = ref_outputs["completion_logps"]

    udm_weights = batch.get("udm_weights", None)
    if udm_weights is not None:
        rejected_weights = jnp.where(rejected_mask, udm_weights.astype(completion_logps.dtype), 0.0)
    else:
        rejected_weights = None

    (
        losses,
        chosen_rewards,
        rejected_rewards,
        chosen_losses,
        rejected_losses,
        chosen_valid,
        rejected_valid,
    ) = compute_bco_loss(
        completion_logps,
        reference_completion_logps,
        chosen_mask,
        rejected_mask,
        beta=beta,
        delta=running_delta,
        udm_weights=rejected_weights,
    )

    metrics = LossMetrics(
        loss=losses,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        other_metrics={
            "delta": running_delta,
            "logps/chosen": (completion_logps * chosen_mask.astype(completion_logps.dtype)).sum()
            / jnp.maximum(chosen_valid.sum(), 1.0),
            "logps/rejected": (completion_logps * rejected_mask.astype(completion_logps.dtype)).sum()
            / jnp.maximum(rejected_valid.sum(), 1.0),
            "logits/mean": policy_outputs["mean_completion_logits"],
            "count/chosen": chosen_valid.sum(),
            "count/rejected": rejected_valid.sum(),
            "loss/chosen": chosen_losses.sum() / jnp.maximum(chosen_valid.sum(), 1.0),
            "loss/rejected": rejected_losses.sum() / jnp.maximum(rejected_valid.sum(), 1.0),
            "mask/chosen": chosen_valid,
            "mask/rejected": rejected_valid,
        },
    )
    return metrics


def detach_metrics(metrics: LossMetrics) -> LossMetrics:
    """Convert metrics to host scalars for serialization.

    Args:
        metrics: Loss metrics to detach.

    Returns:
        Detached metrics with scalar values.
    """

    if metrics.other_metrics is not None:
        filtered_other = {}
        for key, value in metrics.other_metrics.items():
            if key.startswith("mask/"):
                continue
            filtered_other[key] = float(value)
    else:
        filtered_other = None

    return metrics.replace(
        loss=float(metrics.loss),
        chosen_rewards=None,
        rejected_rewards=None,
        other_metrics=filtered_other,
    )


__all__ = [
    "RunningMoments",
    "compute_bco_loss",
    "concatenated_forward",
    "detach_metrics",
    "evaluation_step",
    "training_step",
]
