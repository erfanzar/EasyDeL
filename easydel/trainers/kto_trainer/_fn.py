# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import typing as tp

import flax
import jax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.nn import sigmoid
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)

KTO_LOSS_TYPES = ("kto", "apo_zero_unpaired")


def _build_kl_batch(batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Create mismatched prompt/completion batch for KL estimation.

    Rolls completion sequences by one position to create mismatched pairs,
    enabling estimation of KL divergence between policy and reference.

    Args:
        batch: Original batch with prompts and completions.

    Returns:
        Batch with rolled completions for KL computation.
    """

    kl_batch: dict[str, jax.Array] = {
        "prompt_input_ids": batch["prompt_input_ids"],
        "prompt_attention_mask": batch["prompt_attention_mask"],
    }

    for key in ("pixel_values", "pixel_attention_mask", "image_sizes"):
        if key in batch:
            kl_batch[key] = batch[key]

    def _rolled(name: str):
        if name in batch:
            kl_batch[name] = jnp.roll(batch[name], shift=1, axis=0)

    for field in (
        "completion_input_ids",
        "completion_attention_mask",
        "completion_labels",
        "completion_decoder_input_ids",
    ):
        _rolled(field)

    return kl_batch


def kto_objective(
    policy_logps: jax.Array,
    reference_logps: jax.Array,
    labels: jax.Array,
    *,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    loss_type: str,
    policy_kl_logps: jax.Array | None = None,
    reference_kl_logps: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute KTO or APO unpaired losses and rewards.

    Args:
        policy_logps: Log probabilities from the policy model.
        reference_logps: Log probabilities from the reference model.
        labels: Binary labels (True=desirable, False=undesirable).
        beta: Temperature parameter controlling deviation from reference.
        desirable_weight: Weight for desirable examples.
        undesirable_weight: Weight for undesirable examples.
        loss_type: Loss variant ('kto' or 'apo_zero_unpaired').
        policy_kl_logps: Optional policy log probs for KL estimation.
        reference_kl_logps: Optional reference log probs for KL estimation.

    Returns:
        Tuple of (loss, chosen_rewards, rejected_rewards, kl).
    """

    if loss_type not in KTO_LOSS_TYPES:
        raise ValueError(f"Unsupported KTO loss type: {loss_type}")

    dtype = policy_logps.dtype
    labels_bool = labels.astype(bool)
    chosen_mask = labels_bool.astype(dtype)
    rejected_mask = (~labels_bool).astype(dtype)

    logratios = policy_logps - reference_logps

    if policy_kl_logps is not None and reference_kl_logps is not None:
        kl = jnp.maximum(jnp.mean(policy_kl_logps - reference_kl_logps), 0.0)
    else:
        kl = jnp.zeros((), dtype=dtype)
    kl = jax.lax.stop_gradient(kl)

    def _safe_sigmoid(x):
        return sigmoid(jnp.clip(x, -30.0, 30.0))

    if loss_type == "kto":
        chosen_term = beta * (logratios - kl)
        rejected_term = beta * (kl - logratios)
        chosen_losses = chosen_mask * (1.0 - _safe_sigmoid(chosen_term))
        rejected_losses = rejected_mask * (1.0 - _safe_sigmoid(rejected_term))
    else:  # apo_zero_unpaired
        chosen_term = beta * logratios
        rejected_term = beta * logratios
        chosen_losses = chosen_mask * (1.0 - _safe_sigmoid(chosen_term))
        rejected_losses = rejected_mask * _safe_sigmoid(rejected_term)

    chosen_rewards = beta * logratios * chosen_mask
    rejected_rewards = beta * logratios * rejected_mask

    total_examples = jnp.maximum(chosen_mask.sum() + rejected_mask.sum(), 1.0)
    weighted_chosen = desirable_weight * chosen_losses.sum()
    weighted_rejected = undesirable_weight * rejected_losses.sum()
    loss = (weighted_chosen + weighted_rejected) / total_examples

    return loss, chosen_rewards, rejected_rewards, kl


def training_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState,
    learning_rate_fn: tp.Callable[[jax.Array], jax.Array],
    forward_fn: tp.Callable[[EasyDeLState | EasyDeLState.model, dict[str, jax.Array]], dict[str, jax.Array]],
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    loss_type: str,
    calculate_kl: bool,
    aux_loss_coef: float,
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Execute KTO training step with gradient computation.

    Args:
        state: Current model state.
        batch: Training batch.
        reference_state: Reference model state.
        learning_rate_fn: Function mapping step to learning rate.
        forward_fn: Forward pass function.
        beta: Temperature parameter.
        desirable_weight: Weight for desirable examples.
        undesirable_weight: Weight for undesirable examples.
        loss_type: Loss variant to use.
        calculate_kl: Whether to compute KL divergence.
        aux_loss_coef: Coefficient for auxiliary loss.
        loss_config: Optional loss configuration.
        partition_spec: Sharding specification.
        gradient_accumulation_steps: Number of gradient accumulation steps.

    Returns:
        Updated state and loss metrics.
    """

    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def _loss_fn(tree: flax.nnx.GraphState, minibatch: dict[str, jax.Array]):
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree=tree)
        policy_out = forward_fn(module, minibatch)
        policy_logps = policy_out["completion_logps"]

        if "reference_logps" in minibatch:
            reference_logps = jax.lax.stop_gradient(minibatch["reference_logps"])
        else:
            ref_out = forward_fn(reference_state.model, minibatch)
            reference_logps = jax.lax.stop_gradient(ref_out["completion_logps"])

        if calculate_kl:
            kl_batch = _build_kl_batch(minibatch)
            policy_kl_logps = jax.lax.stop_gradient(forward_fn(module, kl_batch)["completion_logps"])
            reference_kl_logps = jax.lax.stop_gradient(forward_fn(reference_state.model, kl_batch)["completion_logps"])
        else:
            policy_kl_logps = reference_kl_logps = None

        loss, chosen_rewards, rejected_rewards, kl = kto_objective(
            policy_logps,
            reference_logps,
            minibatch["label"],
            beta=beta,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
            loss_type=loss_type,
            policy_kl_logps=policy_kl_logps,
            reference_kl_logps=reference_kl_logps,
        )

        if aux_loss_coef > 0.0 and "aux_loss" in policy_out:
            loss = loss + aux_loss_coef * policy_out["aux_loss"]

        metrics = LossMetrics(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            other_metrics={"kl": kl},
        )
        return metrics.loss, metrics

    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(_loss_fn, has_aux=True),
    )

    metrics = update_metrics(
        metrics=metrics,
        learning_rate_fn=learning_rate_fn,
        step=state.step,
        gradients=gradients,
    )
    state = update_state_respectfully(
        state=state,
        gradients=gradients,
        loss_config=loss_config,
        metrics=metrics,
    )
    return state, metrics


def evaluation_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState,
    forward_fn: tp.Callable[[EasyDeLState | EasyDeLState.model, dict[str, jax.Array]], dict[str, jax.Array]],
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    loss_type: str,
    calculate_kl: bool,
    aux_loss_coef: float,
    partition_spec: PartitionSpec | None = None,
) -> LossMetrics:
    """Execute KTO evaluation step without gradients.

    Args:
        state: Current model state.
        batch: Evaluation batch.
        reference_state: Reference model state.
        forward_fn: Forward pass function.
        beta: Temperature parameter.
        desirable_weight: Weight for desirable examples.
        undesirable_weight: Weight for undesirable examples.
        loss_type: Loss variant to use.
        calculate_kl: Whether to compute KL divergence.
        aux_loss_coef: Coefficient for auxiliary loss.
        partition_spec: Sharding specification.

    Returns:
        Loss metrics.
    """

    *_, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=1,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    policy_out = forward_fn(state.model, batch)
    policy_logps = policy_out["completion_logps"]

    if "reference_logps" in batch:
        reference_logps = batch["reference_logps"]
    else:
        reference_logps = forward_fn(reference_state.model, batch)["completion_logps"]

    if calculate_kl:
        kl_batch = _build_kl_batch(batch)
        policy_kl_logps = forward_fn(state.model, kl_batch)["completion_logps"]
        reference_kl_logps = forward_fn(reference_state.model, kl_batch)["completion_logps"]
    else:
        policy_kl_logps = reference_kl_logps = None

    loss, chosen_rewards, rejected_rewards, kl = kto_objective(
        policy_logps,
        reference_logps,
        batch["label"],
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        loss_type=loss_type,
        policy_kl_logps=policy_kl_logps,
        reference_kl_logps=reference_kl_logps,
    )

    if aux_loss_coef > 0.0 and "aux_loss" in policy_out:
        loss = loss + aux_loss_coef * policy_out["aux_loss"]

    return LossMetrics(
        loss=loss,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        other_metrics={"kl": kl},
    )
