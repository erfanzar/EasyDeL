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

"""Compiled actor/critic steps and shared full-sequence scoring helpers."""

from __future__ import annotations

import collections.abc
import copy
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
import spectrax as spx
from eformer.optimizers import OptimizerFactory, SchedulerConfig
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.etils import EasyDeLSchedulers
from easydel.infra.loss_utils import LossConfig, LossMetrics

from .._logprob_utils import (
    compute_per_token_logps_and_entropies_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)
from ..training_utils import (
    ScheduledLossAdapter,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)

if tp.TYPE_CHECKING:
    from .config import OnlineActorCriticConfig


def build_critic_trainable_selector(mode: str) -> spx.Selector:
    """Build the default structural selector for a separately trained critic.

    Args:
        mode: ``"all"`` or ``"full_except_attention"``.

    Returns:
        A SpectraX selector over parameter variables.

    Raises:
        ValueError: If ``mode`` is not supported.
    """
    parameters = spx.select().variables("parameters")
    if mode == "all":
        return parameters
    if mode == "full_except_attention":

        def is_not_attention(_variable, path: str) -> bool:
            """Reject parameters under attention modules by canonical path."""
            normalized = path.lower()
            return "attention" not in normalized and "attn" not in normalized

        return parameters.where_variable(is_not_attention)
    raise ValueError(f"Unsupported critic trainable mode: {mode!r}.")


def build_critic_optimizer(
    arguments: OnlineActorCriticConfig,
    steps: int | None,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Create the critic optimizer without mutating actor arguments.

    Args:
        arguments: Online actor/critic configuration.
        steps: Number of critic scheduler steps.

    Returns:
        ``(optimizer, scheduler)`` configured from actor optimizer defaults
        plus critic-specific overrides.
    """
    optimizer_kwargs = copy.deepcopy(arguments.optimizer_kwargs)
    configured_steps = steps or optimizer_kwargs["steps"] or 2**63 - 1
    optimizer_kwargs["steps"] = max(
        int(configured_steps),
        int(arguments.critic_warmup_steps) + 1,
    )
    optimizer_kwargs["learning_rate"] = float(arguments.critic_learning_rate)
    optimizer_kwargs["learning_rate_end"] = arguments.critic_learning_rate_end
    optimizer_kwargs["warmup_steps"] = int(arguments.critic_warmup_steps)
    if arguments.critic_optimizer is not None:
        optimizer_kwargs["optimizer"] = arguments.critic_optimizer
    if arguments.critic_scheduler is not None:
        optimizer_kwargs["scheduler"] = arguments.critic_scheduler

    scheduler_type = optimizer_kwargs.pop("scheduler", None)
    if scheduler_type == "none" or scheduler_type == EasyDeLSchedulers.NONE:
        scheduler_type = None
    scheduler_config = SchedulerConfig(
        scheduler_type=scheduler_type,
        steps=optimizer_kwargs.pop("steps"),
        learning_rate=optimizer_kwargs.pop("learning_rate"),
        learning_rate_end=optimizer_kwargs.pop("learning_rate_end"),
        warmup_steps=optimizer_kwargs.pop("warmup_steps"),
        exponent=optimizer_kwargs.pop("exponent", 1),
    )
    optimizer_kwargs.pop("gradient_accumulation_steps", None)
    optimizer, scheduler = OptimizerFactory.create(
        optimizer_type=optimizer_kwargs.pop("optimizer"),
        scheduler_config=scheduler_config,
        clip_grad=optimizer_kwargs.pop("clip_grad"),
        weight_decay=optimizer_kwargs.pop("weight_decay"),
        custom_scheduler=None,
        **optimizer_kwargs,
    )
    return optimizer, scheduler


def score_actor_tokens(
    model,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    *,
    logprob_vocab_chunk_size: int | None,
    return_entropy: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """Score every causal target position in a full trajectory row.

    Outputs align with ``input_ids[:, 1:]``.  Prompt, observation, and
    padding targets remain present in the returned arrays; callers select
    optimized positions with a separate ``action_mask``.

    Args:
        model: Bound causal language model (a value-head wrapper is accepted).
        input_ids: Token IDs with shape ``[batch, sequence]``.
        attention_mask: Full attention mask with the same leading shape.
        logprob_vocab_chunk_size: Optional vocabulary-axis chunk size.
        return_entropy: Whether to return token entropies.

    Returns:
        ``(logps, entropies)`` with shape ``[batch, sequence - 1]``.
    """
    target_ids = input_ids[:, 1:]
    call_kwargs: dict[str, tp.Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    lmhead_chunksize = resolve_lmhead_chunksize(model)
    if lmhead_chunksize is not None:
        call_kwargs["apply_lm_head"] = False
        call_kwargs["output_hidden_states"] = True
    outputs = model(**call_kwargs)

    if outputs.logits is None and lmhead_chunksize is not None:
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None:
                hidden_states = hidden_states[-1]
        if hidden_states is None:
            raise ValueError("Model outputs do not provide hidden states for headless online-RL scoring.")
        return compute_per_token_logps_and_entropies_from_hidden_states(
            model,
            hidden_states[:, :-1, :],
            target_ids,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=return_entropy,
        )

    logits = outputs.logits
    if logits is None:
        raise ValueError("Model outputs do not provide logits for online-RL scoring.")
    return compute_token_logps_and_entropies_chunked(
        logits[:, :-1, :],
        target_ids,
        return_entropy=return_entropy,
        chunk_size=logprob_vocab_chunk_size,
    )


def score_critic_values(
    model,
    input_ids: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """Predict values aligned with every causal target position.

    Args:
        model: Bound value-head model.
        input_ids: Token IDs with shape ``[batch, sequence]``.
        attention_mask: Full attention mask.

    Returns:
        Values with shape ``[batch, sequence - 1]``.

    Raises:
        ValueError: If hidden states or a ``value_head`` are unavailable.
    """
    call_kwargs: dict[str, tp.Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
    }
    if resolve_lmhead_chunksize(model) is not None:
        call_kwargs["apply_lm_head"] = False
    outputs = model(**call_kwargs)
    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            hidden_states = hidden_states[-1]
    if hidden_states is None:
        raise ValueError("Critic model outputs do not provide hidden states.")
    value_head = getattr(model, "value_head", None)
    if value_head is None:
        raise ValueError("Critic model must expose a `value_head`.")
    return value_head(hidden_states[:, :-1, :]).squeeze(-1).astype(jnp.float32)


def _token_normalizer(
    minibatch: collections.abc.Mapping[str, jax.Array],
    local_count: jax.Array,
    gradient_accumulation_steps: int,
    use_global_normalizer: bool,
) -> jax.Array:
    """Return an accumulation-correct token denominator."""
    if not use_global_normalizer:
        return jnp.maximum(local_count, 1.0)
    global_count = minibatch.get("num_items_in_batch")
    if global_count is None:
        return jnp.maximum(local_count, 1.0)
    return jnp.maximum(
        global_count.astype(jnp.float32) / max(int(gradient_accumulation_steps), 1),
        1.0,
    )


def _masked_mean(value: jax.Array, mask: jax.Array) -> jax.Array:
    """Compute a numerically safe masked mean."""
    mask = mask.astype(jnp.float32)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def actor_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    algorithm: tp.Literal["sao", "ppo"],
    dis_epsilon_low: float,
    dis_epsilon_high: float,
    cliprange: float,
    entropy_coef: float,
    logprob_vocab_chunk_size: int | None,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    use_global_normalizer: bool = True,
    is_training: bool = True,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run one policy-only SAO or clipped-PPO actor update.

    The SAO branch implements strict double-sided rejection in log space and
    detaches the retained importance ratio.  The PPO branch implements the
    standard clipped surrogate.  Neither branch differentiates a value head;
    critic regression is handled by :func:`critic_step`.
    """
    scope_root = f"easydel/trainer/{algorithm}/actor/" + ("train_step" if is_training else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=partition_spec,
        )
        batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """Compute a policy-only loss for one accumulation microbatch."""
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree)
        action_mask = minibatch["action_mask"].astype(jnp.float32)
        behavior_logps = jax.lax.stop_gradient(minibatch["behavior_logps"].astype(jnp.float32))
        advantages = jax.lax.stop_gradient(minibatch["advantages"].astype(jnp.float32))
        current_logps, entropies = score_actor_tokens(
            module,
            minibatch["input_ids"],
            minibatch["attention_mask"],
            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=entropy_coef > 0,
        )
        log_ratio = current_logps - behavior_logps
        ratio = jnp.exp(log_ratio)

        if algorithm == "sao":
            lower = jnp.log1p(-jnp.asarray(dis_epsilon_low, dtype=jnp.float32))
            upper = jnp.log1p(jnp.asarray(dis_epsilon_high, dtype=jnp.float32))
            keep = ((log_ratio > lower) & (log_ratio < upper)).astype(jnp.float32) * action_mask
            detached_weight = jax.lax.stop_gradient(ratio * keep)
            per_token_policy_loss = -detached_weight * advantages * current_logps
            clip_fraction = 1.0 - _masked_mean(keep, action_mask)
        elif algorithm == "ppo":
            unclipped = -advantages * ratio
            clipped = -advantages * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
            per_token_policy_loss = jnp.maximum(unclipped, clipped)
            keep = action_mask
            clip_fraction = _masked_mean((clipped > unclipped).astype(jnp.float32), action_mask)
        else:
            raise ValueError(f"Unknown online actor algorithm: {algorithm!r}.")

        local_count = jnp.sum(action_mask)
        normalizer = _token_normalizer(
            minibatch,
            local_count,
            gradient_accumulation_steps,
            use_global_normalizer,
        )
        policy_loss = jnp.sum(per_token_policy_loss * action_mask) / normalizer
        if entropies is None:
            entropy = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            entropy = jnp.sum(entropies * action_mask) / normalizer
        loss = policy_loss - entropy_coef * entropy
        approx_kl = 0.5 * _masked_mean(jnp.square(log_ratio), action_mask)
        metrics = LossMetrics(
            loss=loss,
            accuracy=jnp.asarray(1.0, dtype=jnp.float32),
            other_metrics={
                "actor/policy_loss": policy_loss,
                "actor/entropy": entropy,
                "actor/approx_kl": approx_kl,
                "actor/clip_fraction": clip_fraction,
                "actor/ratio_mean": _masked_mean(ratio, action_mask),
                "actor/advantage_mean": _masked_mean(advantages, action_mask),
                "actor/action_tokens": local_count,
                "actor/dis_keep_fraction": _masked_mean(keep, action_mask),
            },
        )
        return loss, metrics

    if not is_training:
        _, metrics = loss_fn(state.graphstate, batch)
        return metrics

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


def critic_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    cliprange_value: float,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    use_global_normalizer: bool = True,
    is_training: bool = True,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run one independently optimized critic regression update."""
    scope_root = "easydel/trainer/online_rl/critic/" + ("train_step" if is_training else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=partition_spec,
        )
        batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """Compute clipped value regression for one accumulation microbatch."""
        module = state.merge(tree)
        action_mask = minibatch["action_mask"].astype(jnp.float32)
        old_values = jax.lax.stop_gradient(minibatch["old_values"].astype(jnp.float32))
        returns = jax.lax.stop_gradient(minibatch["critic_returns"].astype(jnp.float32))
        values = score_critic_values(
            module,
            minibatch["input_ids"],
            minibatch["attention_mask"],
        )
        clipped_values = old_values + jnp.clip(values - old_values, -cliprange_value, cliprange_value)
        losses = jnp.square(values - returns)
        clipped_losses = jnp.square(clipped_values - returns)
        per_token_loss = 0.5 * jnp.maximum(losses, clipped_losses)
        local_count = jnp.sum(action_mask)
        normalizer = _token_normalizer(
            minibatch,
            local_count,
            gradient_accumulation_steps,
            use_global_normalizer,
        )
        value_loss = jnp.sum(per_token_loss * action_mask) / normalizer
        value_mean = _masked_mean(values, action_mask)
        return_mean = _masked_mean(returns, action_mask)
        return_var = _masked_mean(jnp.square(returns - return_mean), action_mask)
        residual_var = _masked_mean(jnp.square(returns - values), action_mask)
        explained_variance = 1.0 - residual_var / jnp.maximum(return_var, 1e-8)
        clip_fraction = _masked_mean((clipped_losses > losses).astype(jnp.float32), action_mask)
        metrics = LossMetrics(
            loss=value_loss,
            accuracy=jnp.asarray(1.0, dtype=jnp.float32),
            other_metrics={
                "critic/value_loss": value_loss,
                "critic/value_mean": value_mean,
                "critic/return_mean": return_mean,
                "critic/explained_variance": explained_variance,
                "critic/clip_fraction": clip_fraction,
                "critic/action_tokens": local_count,
            },
        )
        return value_loss, metrics

    if not is_training:
        _, metrics = loss_fn(state.graphstate, batch)
        return metrics

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


def _actor_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build the MPMD scheduled-loss cache key for the actor step."""
    return scheduled_loss_cache_key(
        call,
        value_fields=(
            "algorithm",
            "dis_epsilon_low",
            "dis_epsilon_high",
            "cliprange",
            "entropy_coef",
            "logprob_vocab_chunk_size",
            "partition_spec",
            "gradient_accumulation_steps",
            "use_global_normalizer",
        ),
        object_fields=("loss_config", "learning_rate_fn", "straight_through_emulator"),
    )


def _make_actor_scheduled_loss(call):
    """Build a scalar actor loss for SpectraX scheduled VJP."""
    algorithm = call.get("algorithm")
    dis_epsilon_low = call.get("dis_epsilon_low")
    dis_epsilon_high = call.get("dis_epsilon_high")
    cliprange = call.get("cliprange")
    entropy_coef = call.get("entropy_coef")
    logprob_vocab_chunk_size = call.get("logprob_vocab_chunk_size")
    loss_config = call.get("loss_config")
    learning_rate_fn = call.get("learning_rate_fn")
    partition_spec = call.get("partition_spec")
    gradient_accumulation_steps = call.get("gradient_accumulation_steps")
    use_global_normalizer = call.get("use_global_normalizer")
    straight_through_emulator = call.get("straight_through_emulator")

    def scheduled_loss(tree, batch):
        """Evaluate the policy-only scalar loss on one scheduled microbatch."""
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        metrics = actor_step(
            call.state.replace(graphstate=tree),
            batch,
            algorithm,
            dis_epsilon_low,
            dis_epsilon_high,
            cliprange,
            entropy_coef,
            logprob_vocab_chunk_size,
            loss_config=loss_config,
            learning_rate_fn=learning_rate_fn,
            partition_spec=partition_spec,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_global_normalizer=use_global_normalizer,
            is_training=False,
            straight_through_emulator=None,
        )
        return tp.cast(LossMetrics, metrics).loss

    return scheduled_loss


def _critic_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build the MPMD scheduled-loss cache key for the critic step."""
    return scheduled_loss_cache_key(
        call,
        value_fields=(
            "cliprange_value",
            "partition_spec",
            "gradient_accumulation_steps",
            "use_global_normalizer",
        ),
        object_fields=("loss_config", "learning_rate_fn"),
    )


def _make_critic_scheduled_loss(call):
    """Build a scalar critic loss for SpectraX scheduled VJP."""
    cliprange_value = call.get("cliprange_value")
    loss_config = call.get("loss_config")
    learning_rate_fn = call.get("learning_rate_fn")
    partition_spec = call.get("partition_spec")
    gradient_accumulation_steps = call.get("gradient_accumulation_steps")
    use_global_normalizer = call.get("use_global_normalizer")

    def scheduled_loss(tree, batch):
        """Evaluate the critic-only scalar loss on one scheduled microbatch."""
        metrics = critic_step(
            call.state.replace(graphstate=tree),
            batch,
            cliprange_value,
            loss_config=loss_config,
            learning_rate_fn=learning_rate_fn,
            partition_spec=partition_spec,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_global_normalizer=use_global_normalizer,
            is_training=False,
        )
        return tp.cast(LossMetrics, metrics).loss

    return scheduled_loss


register_scheduled_loss_adapter(
    step_fn=actor_step,
    adapter=ScheduledLossAdapter(
        name="online_rl_actor",
        make_loss=_make_actor_scheduled_loss,
        make_cache_key=_actor_scheduled_loss_cache_key,
    ),
)
register_scheduled_loss_adapter(
    step_fn=critic_step,
    adapter=ScheduledLossAdapter(
        name="online_rl_critic",
        make_loss=_make_critic_scheduled_loss,
        make_cache_key=_critic_scheduled_loss_cache_key,
    ),
)


def recover_rewards_from_gae(
    advantages: jax.Array,
    values: jax.Array,
    mask: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Invert standard contiguous GAE to recover the original token rewards.

    This adapter lets the online actor/critic substrate reuse PPO's mature
    generation and reward-routing path while replacing PPO's joint value head
    with a separate critic.
    """
    advantages = advantages.astype(jnp.float32)
    values = values.astype(jnp.float32)
    mask = mask.astype(jnp.float32)
    next_advantages = jnp.concatenate([advantages[:, 1:], jnp.zeros_like(advantages[:, :1])], axis=1)
    next_values = jnp.concatenate([values[:, 1:], jnp.zeros_like(values[:, :1])], axis=1)
    next_mask = jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)
    delta = advantages - float(gamma) * float(gae_lambda) * next_advantages * next_mask
    rewards = delta + values - float(gamma) * next_values * next_mask
    return rewards * mask


__all__ = (
    "actor_step",
    "build_critic_optimizer",
    "build_critic_trainable_selector",
    "critic_step",
    "recover_rewards_from_gae",
    "score_actor_tokens",
    "score_critic_values",
)
