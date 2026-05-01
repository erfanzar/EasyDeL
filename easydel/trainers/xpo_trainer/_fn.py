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

import typing as tp

import jax
import jax.nn as jnn
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..group_relative_policy_optimization._fn import get_per_token_logps
from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    constrain_scheduled_batch,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    scheduled_loss_cache_key,
    sync_module_schedule_config,
    update_metrics,
    update_state_respectfully,
)

_XPO_REFERENCE_FORWARD_CACHE: dict[int, tp.Callable[..., tp.Any]] = {}


def _compute_logps(
    module: spx.Module,
    prompt_ids: jax.Array,
    prompt_mask: jax.Array,
    completion_ids: jax.Array,
    completion_mask: jax.Array,
    logprob_vocab_chunk_size: int | None,
) -> jax.Array:
    """Compute per-token log probabilities for completion tokens.

    Concatenates prompts and completions, then extracts log probabilities
    for tokens in the completion portion only.

    Args:
        module: The language model module to evaluate.
        prompt_ids: Token IDs for the prompt portion.
        prompt_mask: Attention mask for the prompt.
        completion_ids: Token IDs for the completion portion.
        completion_mask: Attention mask for the completion.

    Returns:
        Per-token log probabilities for completion tokens.
    """
    input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
    attention_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)
    prompt_length = prompt_ids.shape[-1]
    return get_per_token_logps(
        module,
        input_ids,
        attention_mask,
        prompt_length,
        logprob_vocab_chunk_size=logprob_vocab_chunk_size,
    )


def _compute_pair_logps(
    module: spx.Module,
    prompt_ids: jax.Array,
    prompt_mask: jax.Array,
    policy_completion_ids: jax.Array,
    policy_completion_mask: jax.Array,
    ref_completion_ids: jax.Array,
    ref_completion_mask: jax.Array,
    logprob_vocab_chunk_size: int | None,
) -> tuple[jax.Array, jax.Array]:
    prompt_ids = jnp.concatenate([prompt_ids, prompt_ids], axis=0)
    prompt_mask = jnp.concatenate([prompt_mask, prompt_mask], axis=0)
    completion_ids = jnp.concatenate([policy_completion_ids, ref_completion_ids], axis=0)
    completion_mask = jnp.concatenate([policy_completion_mask, ref_completion_mask], axis=0)
    token_logps = _compute_logps(
        module,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        logprob_vocab_chunk_size,
    )
    split = policy_completion_ids.shape[0]
    return token_logps[:split], token_logps[split:]


def _sum_logps(token_logps: jax.Array, completion_mask: jax.Array) -> jax.Array:
    """Sum log probabilities over completion tokens, respecting the attention mask.

    Args:
        token_logps: Per-token log probabilities.
        completion_mask: Attention mask indicating valid completion tokens.

    Returns:
        Sum of log probabilities for each sequence in the batch.
    """
    mask = completion_mask.astype(token_logps.dtype)
    return (token_logps * mask).sum(axis=1)


def _xpo_pair_forward(module: spx.Module, batch: dict[str, tp.Any], logprob_vocab_chunk_size: int | None):
    return _compute_pair_logps(
        module,
        batch["prompt_ids"],
        batch["prompt_mask"],
        batch["policy_completion_ids"],
        batch["policy_completion_mask"],
        batch["ref_completion_ids"],
        batch["ref_completion_mask"],
        logprob_vocab_chunk_size,
    )


def _cached_xpo_reference_forward(mesh: spx.SpxMesh) -> tp.Callable[..., tp.Any]:
    key = id(mesh)
    cached = _XPO_REFERENCE_FORWARD_CACHE.get(key)
    if cached is None:
        cached = spx.jit(_xpo_pair_forward, mesh=mesh, static_argnums=(2,))
        _XPO_REFERENCE_FORWARD_CACHE[key] = cached
    return cached


def xpo_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState,
    logprob_vocab_chunk_size: int | None,
    loss_config: LossConfig | None,
    learning_rate_fn,
    partition_spec: PartitionSpec | None,
    gradient_accumulation_steps: int,
    is_train: bool,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Execute a single XPO training or evaluation step.

    Implements the Exploratory Preference Optimization objective, which combines
    DPO-style preference learning with an exploratory term that encourages the
    policy to assign probability mass to reference completions.

    Args:
        state: Current model state containing parameters and optimizer state.
        batch: Input batch containing prompt IDs, completion IDs, masks, and hyperparameters.
        reference_state: Frozen reference model state for computing log probability ratios.
        loss_config: Optional configuration for loss computation and clipping.
        learning_rate_fn: Function that returns the learning rate for the current step.
        partition_spec: Sharding specification for distributed training.
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating.
        is_train: Whether this is a training step (True) or evaluation step (False).

    Returns:
        If is_train is True, returns tuple of (updated_state, metrics).
        If is_train is False, returns only metrics.
    """

    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    ref_module = reference_state.merge(reference_state.graphstate)
    batch = dict(batch)
    batch["_ref_on_policy"] = jax.lax.stop_gradient(
        _compute_logps(
            ref_module,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["policy_completion_ids"],
            batch["policy_completion_mask"],
            logprob_vocab_chunk_size,
        )
    )
    batch["_ref_on_ref"] = jax.lax.stop_gradient(
        _compute_logps(
            ref_module,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["ref_completion_ids"],
            batch["ref_completion_mask"],
            logprob_vocab_chunk_size,
        )
    )
    del ref_module

    def loss_fn(tree: spx.State, minibatch: dict[str, jax.Array]):
        if is_train and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree=tree)

        prompt_ids = minibatch["prompt_ids"]
        prompt_mask = minibatch["prompt_mask"]
        policy_completion_ids = minibatch["policy_completion_ids"]
        policy_completion_mask = minibatch["policy_completion_mask"]
        ref_completion_ids = minibatch["ref_completion_ids"]
        ref_completion_mask = minibatch["ref_completion_mask"]
        chosen_mask = minibatch["chosen_mask"].astype(bool)
        beta = minibatch["beta"][0]
        alpha = minibatch["alpha"][0]

        policy_on_policy = _compute_logps(
            module,
            prompt_ids,
            prompt_mask,
            policy_completion_ids,
            policy_completion_mask,
            logprob_vocab_chunk_size,
        )
        policy_on_ref = _compute_logps(
            module,
            prompt_ids,
            prompt_mask,
            ref_completion_ids,
            ref_completion_mask,
            logprob_vocab_chunk_size,
        )
        ref_on_policy = jax.lax.stop_gradient(minibatch["_ref_on_policy"])
        ref_on_ref = jax.lax.stop_gradient(minibatch["_ref_on_ref"])

        policy_logps_policy = _sum_logps(policy_on_policy, policy_completion_mask)
        policy_logps_ref = _sum_logps(policy_on_ref, ref_completion_mask)
        ref_logps_policy = _sum_logps(ref_on_policy, policy_completion_mask)
        ref_logps_ref = _sum_logps(ref_on_ref, ref_completion_mask)

        chosen_policy_logps = jnp.where(chosen_mask, policy_logps_policy, policy_logps_ref)
        chosen_ref_logps = jnp.where(chosen_mask, ref_logps_policy, ref_logps_ref)
        rejected_policy_logps = jnp.where(chosen_mask, policy_logps_ref, policy_logps_policy)
        rejected_ref_logps = jnp.where(chosen_mask, ref_logps_ref, ref_logps_policy)

        chosen_log_ratio = chosen_policy_logps - chosen_ref_logps
        rejected_log_ratio = rejected_policy_logps - rejected_ref_logps
        logits = chosen_log_ratio - rejected_log_ratio

        loss_type = minibatch["loss_type"][0]
        sigmoid_losses = -jnn.log_sigmoid(beta * logits)
        ipo_losses = (logits - 1.0 / (2.0 * beta)) ** 2
        dpo_losses = jnp.where(loss_type == 0, sigmoid_losses, ipo_losses)

        xpo_losses = alpha * policy_logps_ref
        total_loss = (dpo_losses + xpo_losses).mean()

        chosen_rewards = beta * chosen_log_ratio
        rejected_rewards = beta * rejected_log_ratio
        margin = chosen_rewards - rejected_rewards
        accuracy = jnp.mean(margin > 0)

        kl_policy = ((policy_on_policy - ref_on_policy) * policy_completion_mask).sum(axis=1)
        kl_ref = ((policy_on_ref - ref_on_ref) * ref_completion_mask).sum(axis=1)
        mean_kl = jnp.mean((kl_policy + kl_ref) / 2)

        entropy_policy = -_sum_logps(policy_on_policy, policy_completion_mask)
        entropy_ref = -_sum_logps(policy_on_ref, ref_completion_mask)
        mean_entropy = jnp.mean((entropy_policy + entropy_ref) / 2)

        metrics = LossMetrics(
            loss=total_loss,
            other_metrics={
                "loss_dpo": dpo_losses.mean(),
                "loss_xpo": xpo_losses.mean(),
                "kl": mean_kl,
                "entropy": mean_entropy,
                "chosen_rewards": jnp.mean(chosen_rewards),
                "rejected_rewards": jnp.mean(rejected_rewards),
                "margin": jnp.mean(margin),
                "accuracy": accuracy,
                "beta": beta,
                "alpha": alpha,
            },
        )
        return total_loss, metrics

    if is_train:
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
        metrics = update_metrics(
            metrics=metrics, learning_rate_fn=learning_rate_fn, step=state.step, gradients=gradients
        )
        state = update_state_respectfully(
            state=state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=metrics,
        )
        return state, metrics

    _, metrics = loss_fn(state.graphstate, batch)
    return metrics


def _xpo_loss_from_logps(
    *,
    policy_on_policy: jax.Array,
    policy_on_ref: jax.Array,
    ref_on_policy: jax.Array,
    ref_on_ref: jax.Array,
    policy_completion_mask: jax.Array,
    ref_completion_mask: jax.Array,
    chosen_mask: jax.Array,
    beta: jax.Array,
    alpha: jax.Array,
    loss_type: jax.Array,
) -> jax.Array:
    policy_logps_policy = _sum_logps(policy_on_policy, policy_completion_mask)
    policy_logps_ref = _sum_logps(policy_on_ref, ref_completion_mask)
    ref_logps_policy = _sum_logps(ref_on_policy, policy_completion_mask)
    ref_logps_ref = _sum_logps(ref_on_ref, ref_completion_mask)

    chosen_policy_logps = jnp.where(chosen_mask, policy_logps_policy, policy_logps_ref)
    chosen_ref_logps = jnp.where(chosen_mask, ref_logps_policy, ref_logps_ref)
    rejected_policy_logps = jnp.where(chosen_mask, policy_logps_ref, policy_logps_policy)
    rejected_ref_logps = jnp.where(chosen_mask, ref_logps_ref, ref_logps_policy)

    chosen_log_ratio = chosen_policy_logps - chosen_ref_logps
    rejected_log_ratio = rejected_policy_logps - rejected_ref_logps
    logits = chosen_log_ratio - rejected_log_ratio
    sigmoid_losses = -jnn.log_sigmoid(beta * logits)
    ipo_losses = (logits - 1.0 / (2.0 * beta)) ** 2
    dpo_losses = jnp.where(loss_type == 0, sigmoid_losses, ipo_losses)
    xpo_losses = alpha * policy_logps_ref
    return (dpo_losses + xpo_losses).mean()


def _prepare_xpo_scheduled_batch(call) -> dict[str, tp.Any]:
    batch = dict(call.batch)
    if "_ref_on_policy" in batch and "_ref_on_ref" in batch:
        return batch
    reference_state = call.get("reference_state")
    if reference_state is None:
        raise RuntimeError("XPO scheduled MPMD training requires reference_state.")
    ref_module = reference_state.merge(reference_state.graphstate)
    ref_module.eval()
    sync_module_schedule_config(ref_module, call.schedule)
    partition_spec = call.get("partition_spec")
    batch_for_ref = constrain_scheduled_batch(ref_module, batch, partition_spec)
    ref_forward = _cached_xpo_reference_forward(ref_module.mesh)
    ref_on_policy, ref_on_ref = ref_forward(ref_module, batch_for_ref, call.get("logprob_vocab_chunk_size"))
    batch["_ref_on_policy"] = jax.lax.stop_gradient(ref_on_policy)
    batch["_ref_on_ref"] = jax.lax.stop_gradient(ref_on_ref)
    return batch


def _xpo_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    return scheduled_loss_cache_key(
        call,
        value_fields=("logprob_vocab_chunk_size", "partition_spec"),
        object_fields=("straight_through_emulator",),
    )


def _make_xpo_scheduled_loss(call):
    logprob_vocab_chunk_size = call.get("logprob_vocab_chunk_size")
    partition_spec = call.get("partition_spec")

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        module = bind_scheduled_module(call, tree)
        batch = constrain_scheduled_batch(module, batch, partition_spec)
        policy_on_policy, policy_on_ref = _compute_pair_logps(
            module,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["policy_completion_ids"],
            batch["policy_completion_mask"],
            batch["ref_completion_ids"],
            batch["ref_completion_mask"],
            logprob_vocab_chunk_size,
        )
        return _xpo_loss_from_logps(
            policy_on_policy=policy_on_policy,
            policy_on_ref=policy_on_ref,
            ref_on_policy=jax.lax.stop_gradient(batch["_ref_on_policy"]),
            ref_on_ref=jax.lax.stop_gradient(batch["_ref_on_ref"]),
            policy_completion_mask=batch["policy_completion_mask"],
            ref_completion_mask=batch["ref_completion_mask"],
            chosen_mask=batch["chosen_mask"].astype(bool),
            beta=batch["beta"][0],
            alpha=batch["alpha"][0],
            loss_type=batch["loss_type"][0],
        )

    return scheduled_loss


register_scheduled_loss_adapter(
    xpo_step,
    ScheduledLossAdapter(
        name="xpo",
        make_loss=_make_xpo_scheduled_loss,
        make_cache_key=_xpo_scheduled_loss_cache_key,
        prepare_batch=_prepare_xpo_scheduled_batch,
    ),
)
