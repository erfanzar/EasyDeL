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
import flax.nnx
import jax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.nn import log_sigmoid, relu
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers.direct_preference_optimization_trainer._fn import concatenated_inputs

from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)

LOSS_TYPES = tp.Literal["sigmoid", "hinge", "ipo", "simpo"]


def concatenated_forward(
    model: EasyDeLBaseModule,
    batch: dict[str, tp.Any],
    *,
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: int,
    max_length: int | None = None,
    truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    aux_loss_enabled: bool = False,
    loss_type: LOSS_TYPES = "sigmoid",
) -> dict[str, jax.Array]:
    """Runs the policy model on concatenated chosen/rejected sequences.

    This mirrors the behaviour of TRL's CPO forward helper while leveraging the
    JAX-specific utilities already used by the DPO trainer. We concatenate the
    chosen and rejected completions to share a single forward pass, compute
    per-token log-probabilities and expose additional statistics required by the
    CPO objective (raw log-prob sums and token lengths).
    """

    num_examples = batch["prompt_input_ids"].shape[0]
    concatenated_batch = concatenated_inputs(batch=batch, padding_value=padding_value)

    model_kwargs: dict[str, jax.Array] = {}
    if aux_loss_enabled:
        model_kwargs["output_router_logits"] = True

    if "pixel_values" in concatenated_batch:
        model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
    if "pixel_attention_mask" in concatenated_batch:
        model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
    if "image_sizes" in concatenated_batch:
        model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]

    if is_encoder_decoder:
        labels = jnp.where(
            completion_attention_mask == 0,
            label_pad_token_id,
            completion_input_ids,
        )
        outputs = model(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            labels=labels,
            **model_kwargs,
        )
        logits = outputs.logits
        loss_mask = completion_attention_mask.astype(bool)
    else:
        input_ids = jnp.concatenate([prompt_input_ids, completion_input_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_attention_mask, completion_attention_mask], axis=1)
        loss_mask = jnp.concatenate(
            [jnp.zeros_like(prompt_attention_mask), completion_attention_mask],
            axis=1,
        )
        if max_length is not None:
            if truncation_mode == "keep_end":
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
                loss_mask = loss_mask[:, -max_length:]
            elif truncation_mode == "keep_start":
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
                loss_mask = loss_mask[:, :max_length]
            else:
                raise ValueError(
                    f"Unknown truncation mode: '{truncation_mode}'. Should be one of ['keep_end', 'keep_start']."
                )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
        logits = outputs.logits
        labels = jnp.roll(input_ids, shift=-1, axis=1)
        loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype(bool)

    if logits.shape[:2] != loss_mask.shape:
        seq_len = loss_mask.shape[1]
        logits = logits[:, -seq_len:]

    if is_encoder_decoder:
        labels = labels
    else:
        labels = jnp.where(loss_mask, labels, 0)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    batch_size, seq_len = labels.shape
    per_token_logps = jnp.where(
        loss_mask,
        log_probs[jnp.arange(batch_size)[:, None], jnp.arange(seq_len)[None, :], labels],
        0,
    )

    if not is_encoder_decoder:
        per_token_logps = jnp.roll(per_token_logps, shift=1, axis=1)

    sum_logps = per_token_logps.sum(axis=1)
    token_counts = jnp.maximum(loss_mask.sum(axis=1), 1)
    if loss_type in ("ipo", "simpo"):
        scaled_logps = jnp.where(token_counts > 0, sum_logps / token_counts, 0.0)
    else:
        scaled_logps = sum_logps

    chosen_logps = scaled_logps[:num_examples]
    rejected_logps = scaled_logps[num_examples:]
    chosen_logps_raw = sum_logps[:num_examples]
    rejected_logps_raw = sum_logps[num_examples:]
    chosen_lengths = token_counts[:num_examples]
    rejected_lengths = token_counts[num_examples:]

    chosen_logits_sum = jnp.where(
        loss_mask[:num_examples, :, None],
        logits[:num_examples],
        0,
    ).sum()
    rejected_logits_sum = jnp.where(
        loss_mask[num_examples:, :, None],
        logits[num_examples:],
        0,
    ).sum()
    chosen_denom = jnp.maximum(jnp.sum(loss_mask[:num_examples]), 1)
    rejected_denom = jnp.maximum(jnp.sum(loss_mask[num_examples:]), 1)
    mean_chosen_logits = chosen_logits_sum / chosen_denom
    mean_rejected_logits = rejected_logits_sum / rejected_denom

    outputs_dict: dict[str, jax.Array] = {
        "chosen_logps": chosen_logps,
        "rejected_logps": rejected_logps,
        "chosen_logps_raw": chosen_logps_raw,
        "rejected_logps_raw": rejected_logps_raw,
        "chosen_lengths": chosen_lengths,
        "rejected_lengths": rejected_lengths,
        "mean_chosen_logits": mean_chosen_logits,
        "mean_rejected_logits": mean_rejected_logits,
    }
    if aux_loss_enabled and hasattr(outputs, "aux_loss"):
        outputs_dict["aux_loss"] = outputs.aux_loss
    return outputs_dict


def cpo_loss(
    policy_chosen_logps: jax.Array,
    policy_rejected_logps: jax.Array,
    *,
    beta: float,
    label_smoothing: float,
    loss_type: LOSS_TYPES,
    simpo_gamma: float,
    alpha: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute CPO losses and rewards for chosen/rejected pairs.

    Args:
        policy_chosen_logps: Policy log probs for chosen completions.
        policy_rejected_logps: Policy log probs for rejected completions.
        beta: Temperature parameter.
        label_smoothing: Label smoothing factor.
        loss_type: Loss variant (sigmoid, hinge, ipo, simpo).
        simpo_gamma: Margin for SimPO loss.
        alpha: AlphaPO reward shaping parameter.

    Returns:
        Tuple of (losses, chosen_rewards, rejected_rewards).
    """

    if alpha != 0.0:
        chosen_probs = jnp.exp(policy_chosen_logps)
        rejected_probs = jnp.exp(policy_rejected_logps)
        chosen_rewards = (1.0 - jnp.power(chosen_probs, -alpha)) / alpha
        rejected_rewards = (1.0 - jnp.power(rejected_probs, -alpha)) / alpha
        logits = chosen_rewards - rejected_rewards
    else:
        chosen_rewards = policy_chosen_logps
        rejected_rewards = policy_rejected_logps
        logits = policy_chosen_logps - policy_rejected_logps

    if loss_type == "simpo":
        gamma_logratios = simpo_gamma / beta
        logits = logits - gamma_logratios
        losses = -log_sigmoid(beta * logits) * (1.0 - label_smoothing) - log_sigmoid(-beta * logits) * label_smoothing
    elif loss_type == "sigmoid":
        losses = -log_sigmoid(beta * logits) * (1.0 - label_smoothing) - log_sigmoid(-beta * logits) * label_smoothing
    elif loss_type == "hinge":
        losses = relu(1.0 - beta * logits)
    elif loss_type == "ipo":
        losses = jnp.square(logits - (1.0 / (2.0 * beta)))
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Expected one of ['sigmoid', 'hinge', 'ipo', 'simpo'].")

    if alpha != 0.0:
        chosen_rewards = beta * jnp.asarray(chosen_rewards)
        rejected_rewards = beta * jnp.asarray(rejected_rewards)
    else:
        chosen_rewards = beta * policy_chosen_logps
        rejected_rewards = beta * policy_rejected_logps

    return losses, chosen_rewards, rejected_rewards


def _policy_nll_loss(
    chosen_logps_raw: jax.Array,
    chosen_lengths: jax.Array,
) -> jax.Array:
    """Compute negative log-likelihood loss for policy regularization.

    Args:
        chosen_logps_raw: Raw log probabilities for chosen completions.
        chosen_lengths: Token counts for normalization.

    Returns:
        Scalar NLL loss.
    """
    total_tokens = jnp.maximum(jnp.sum(chosen_lengths), 1)
    total_logprob = jnp.sum(chosen_logps_raw)
    return -total_logprob / total_tokens


def training_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    learning_rate_fn: tp.Callable[[jax.Array], jax.Array] | None,
    concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]],
    beta: float,
    label_smoothing: float,
    loss_type: LOSS_TYPES,
    cpo_alpha: float,
    simpo_gamma: float,
    alpha: float,
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Execute CPO training step with gradient computation.

    Args:
        state: Current model state.
        batch: Training batch with chosen/rejected pairs.
        learning_rate_fn: Function mapping step to learning rate.
        concatenated_forward_fn: Forward function.
        beta: Temperature parameter.
        label_smoothing: Label smoothing factor.
        loss_type: Loss variant to use.
        cpo_alpha: Weight for behavior cloning regularization.
        simpo_gamma: Margin for SimPO.
        alpha: AlphaPO reward shaping parameter.
        loss_config: Optional loss configuration.
        partition_spec: Sharding specification.
        gradient_accumulation_steps: Number of gradient accumulation steps.

    Returns:
        Updated state and loss metrics.
    """

    _, minibatch_size, batch_partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=batch_partition_spec)

    def calculate_loss(tree: flax.nnx.GraphState, call_batch: dict[str, jax.Array]):
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        policy_model = state.merge(tree=tree)
        model_outputs = concatenated_forward_fn(policy_model, call_batch)

        losses, chosen_rewards, rejected_rewards = cpo_loss(
            model_outputs["chosen_logps"],
            model_outputs["rejected_logps"],
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            simpo_gamma=simpo_gamma,
            alpha=alpha,
        )

        chosen_rewards = jax.lax.stop_gradient(chosen_rewards)
        rejected_rewards = jax.lax.stop_gradient(rejected_rewards)
        policy_nll_loss = _policy_nll_loss(
            model_outputs["chosen_logps_raw"],
            model_outputs["chosen_lengths"],
        )

        loss = losses.mean() + cpo_alpha * policy_nll_loss
        aux_loss = model_outputs.get("aux_loss")
        if aux_loss is not None:
            loss = loss + aux_loss

        reward_margin = jnp.mean(chosen_rewards - rejected_rewards)
        reward_accuracy = jnp.mean((chosen_rewards > rejected_rewards).astype(jnp.float32))

        other_metrics = {
            "policy_nll_loss": policy_nll_loss,
            "reward_margin": reward_margin,
            "reward_accuracy": reward_accuracy,
            "mean_chosen_logits": model_outputs["mean_chosen_logits"],
            "mean_rejected_logits": model_outputs["mean_rejected_logits"],
        }
        metrics = LossMetrics(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            other_metrics=other_metrics,
        )
        return loss, metrics

    grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=grad_fn,
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
    concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]],
    beta: float,
    label_smoothing: float,
    loss_type: LOSS_TYPES,
    cpo_alpha: float,
    simpo_gamma: float,
    alpha: float,
    partition_spec: PartitionSpec | None = None,
) -> LossMetrics:
    """Execute CPO evaluation step without gradients.

    Args:
        state: Current model state.
        batch: Evaluation batch.
        concatenated_forward_fn: Forward function.
        beta: Temperature parameter.
        label_smoothing: Label smoothing factor.
        loss_type: Loss variant to use.
        cpo_alpha: Weight for behavior cloning regularization.
        simpo_gamma: Margin for SimPO.
        alpha: AlphaPO reward shaping parameter.
        partition_spec: Sharding specification.

    Returns:
        Loss metrics.
    """
    del partition_spec

    model_outputs = concatenated_forward_fn(state.model, batch)
    losses, chosen_rewards, rejected_rewards = cpo_loss(
        model_outputs["chosen_logps"],
        model_outputs["rejected_logps"],
        beta=beta,
        label_smoothing=label_smoothing,
        loss_type=loss_type,
        simpo_gamma=simpo_gamma,
        alpha=alpha,
    )

    chosen_rewards = jax.lax.stop_gradient(chosen_rewards)
    rejected_rewards = jax.lax.stop_gradient(rejected_rewards)
    policy_nll_loss = _policy_nll_loss(
        model_outputs["chosen_logps_raw"],
        model_outputs["chosen_lengths"],
    )

    loss = losses.mean() + cpo_alpha * policy_nll_loss
    aux_loss = model_outputs.get("aux_loss")
    if aux_loss is not None:
        loss = loss + aux_loss

    reward_margin = jnp.mean(chosen_rewards - rejected_rewards)
    reward_accuracy = jnp.mean((chosen_rewards > rejected_rewards).astype(jnp.float32))

    other_metrics = {
        "policy_nll_loss": policy_nll_loss,
        "reward_margin": reward_margin,
        "reward_accuracy": reward_accuracy,
        "mean_chosen_logits": model_outputs["mean_chosen_logits"],
        "mean_rejected_logits": model_outputs["mean_rejected_logits"],
    }

    return LossMetrics(
        loss=loss,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        other_metrics=other_metrics,
    )
