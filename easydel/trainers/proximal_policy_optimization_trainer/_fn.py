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

"""Internal functions for PPO training.

Implements the PPO clipped objective for language-model RLHF training, including:
- Per-token log-probabilities and entropies
- Value head predictions
- Clipped policy loss and clipped value loss
"""

from __future__ import annotations

import collections.abc
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers._logprob_utils import (
    compute_per_token_logps_and_entropies_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)

from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)


def _masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Compute the mean of masked elements.

    Args:
        x: Input array of values.
        mask: Binary mask indicating which elements to include.

    Returns:
        Mean of elements where mask is non-zero, or 0 if mask is empty.
    """
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(x * mask) / denom


def compute_per_token_logps(logits: jax.Array, input_ids: jax.Array, prompt_length: int) -> jax.Array:
    """Slice and gather per-token log-probabilities for completion positions.

    Used by the simple (non-chunked) PPO log-prob path: the function
    converts the full vocabulary logits into log-probabilities via a
    standard log-softmax, then gathers the log-probability of the
    *realised* completion token at every completion position. The
    prompt prefix is dropped wholesale -- PPO only differentiates the
    policy at completion positions.

    Note that this helper does **not** perform the causal shift; the
    caller is expected to align ``logits`` and ``input_ids`` such that
    ``logits[:, t]`` already predicts ``input_ids[:, t]`` (i.e. the
    standard causal-LM shift has already been applied). PPO uses
    :func:`get_per_token_logps_values_entropies` for the shifted path.

    Args:
        logits: Model output logits of shape
            ``(batch, seq_len, vocab_size)``.
        input_ids: Full input sequence with shape
            ``(batch, seq_len)`` covering prompt + completion.
        prompt_length: Number of leading prompt tokens to drop.

    Returns:
        Float array of shape ``(batch, seq_len - prompt_length)``
        containing ``log p(token_t | logits_t)`` for each completion
        position.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_ids = input_ids[:, prompt_length:]
    token_log_probs = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(target_ids, axis=-1),
        axis=-1,
    )
    return jnp.squeeze(token_log_probs, axis=-1)


def get_per_token_logps_values_entropies(
    model,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    prompt_length: int,
    logprob_vocab_chunk_size: int | None = None,
):
    """Compute per-token log probabilities, values, and entropies for PPO.

    Performs a forward pass through the model (with value head) to obtain
    all quantities needed for PPO loss computation. When the model's
    ``lmhead_chunksize`` is configured, the forward pass is run in
    *headless* mode (``apply_lm_head=False``) and log probabilities and
    entropies are derived directly from the hidden states using chunked
    projection through the LM head, avoiding materialization of the full
    ``[batch, seq, vocab]`` logit tensor.

    Args:
        model: CausalLMWithValueHead model instance. Must expose a
            ``value_head`` module and, when headless chunking is active,
            must return ``last_hidden_state`` in its outputs.
        input_ids: Full input sequence (prompt + completion) of shape
            ``(batch_size, seq_len)``.
        attention_mask: Attention mask of shape ``(batch_size, seq_len)``.
        prompt_length: Length of the prompt prefix. Log probabilities,
            entropies, and values are extracted only for the completion
            portion (tokens after the prompt).
        logprob_vocab_chunk_size: When set to a positive value, the log-softmax
            and entropy computations over the vocabulary dimension are
            performed in chunks of this size to reduce peak memory.
            ``None`` disables vocabulary chunking.

    Returns:
        Tuple of ``(token_log_probs, values, entropies)``:
            - ``token_log_probs``: Per-token log probabilities for the
              completion tokens, shape ``(batch, completion_len)``.
            - ``values``: Value head predictions for completion positions,
              shape ``(batch, completion_len)``.
            - ``entropies``: Per-token entropy of the output distribution,
              shape ``(batch, completion_len)``.

    Raises:
        ValueError: If the model outputs do not provide hidden states
            (required for the value head and, when headless mode is
            active, for chunked log-probability computation).
    """
    call_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
    }
    lmhead_chunksize = resolve_lmhead_chunksize(model)
    if lmhead_chunksize is not None:
        call_kwargs["apply_lm_head"] = False
    outputs = model(**call_kwargs)
    targets = input_ids[:, prompt_length:]

    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model outputs do not provide hidden states; cannot compute value head outputs.")
        hidden_states = hidden_states[-1]

    if outputs.logits is None and lmhead_chunksize is not None:
        score_hidden_states = hidden_states[:, prompt_length - 1 : -1, :]
        token_log_probs, entropies = compute_per_token_logps_and_entropies_from_hidden_states(
            model,
            score_hidden_states,
            targets,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=True,
        )
    else:
        logits = outputs.logits[:, prompt_length - 1 :]
        logits = logits[:, :-1, :]
        token_log_probs, entropies = compute_token_logps_and_entropies_chunked(
            logits,
            targets,
            return_entropy=True,
            chunk_size=logprob_vocab_chunk_size,
        )

    values_full = model.value_head(hidden_states).squeeze(-1)
    values = values_full[:, prompt_length - 1 : -1]
    return token_log_probs, values, entropies


def ppo_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    prompt_length: int,
    cliprange: float,
    vf_coef: float,
    cliprange_value: float,
    entropy_coef: float,
    logprob_vocab_chunk_size: int | None,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run one PPO update (forward + clipped objective + optimiser step).

    The step implements the joint PPO loss

    ``L = pg_loss + vf_coef * vf_loss - entropy_coef * entropy``

    where each term is computed under the completion mask:

    * **Policy loss** (clipped surrogate): for the importance ratio
      ``ratio = exp(new_logps - old_logps)`` PPO uses
      ``pg_loss = mean(max(-A * ratio, -A * clip(ratio, 1-cliprange, 1+cliprange)))``.
      ``A`` is the GAE advantage, detached from the gradient via
      ``stop_gradient``.
    * **Value loss** (clipped regression): the new values are clipped
      to a trust region around the rollout values
      ``V_clipped = V_old + clip(V - V_old, -cliprange_value, +cliprange_value)``
      and the loss is ``0.5 * mean(max((V - R)^2, (V_clipped - R)^2))``.
      Returns ``R`` are detached.
    * **Entropy bonus**: ``mean(entropy)`` of the per-token policy
      distribution, scaled by ``entropy_coef`` and *subtracted* so that
      higher-entropy policies are preferred.

    Diagnostics emitted via :class:`LossMetrics` include the approx KL
    ``0.5 * mean(log_ratio**2)``, the policy and value clip-fractions,
    and masked means of ratio/advantages/returns -- the standard PPO
    debugging panel.

    Sharding constraints are applied to the batch up front (``ignore_mpmd``
    is set so MPMD pipelines do not re-shard mid-step), and the body
    runs through :func:`minibatch_call` for gradient accumulation. When
    ``is_training=False`` no gradients are computed and only the
    metrics dict is returned.

    Args:
        state: Current model state for the value-head-augmented policy.
        batch: Mapping carrying the rollout-computed quantities --
            ``input_ids``, ``attention_mask``, ``completion_mask`` (1s
            where the loss applies), ``old_logps`` and ``old_values``
            from the rollout policy, plus precomputed ``advantages``
            and ``returns``.
        prompt_length: Number of leading prompt tokens, used to index
            into the per-token outputs of
            :func:`get_per_token_logps_values_entropies`.
        cliprange: ``epsilon`` in the clipped policy surrogate.
        vf_coef: Scalar weight on the value-function loss.
        cliprange_value: Symmetric clip range for the value update.
        entropy_coef: Weight on the entropy bonus (subtracted from the
            loss).
        logprob_vocab_chunk_size: Optional vocabulary-axis chunk size
            forwarded to the chunked log-prob/entropy reductions; pass
            ``None`` to disable chunking.
        loss_config: Optional :class:`LossConfig` consumed by
            :func:`update_state_respectfully` (e.g. for grad
            clipping/scaling).
        learning_rate_fn: Optional schedule used by
            :func:`update_metrics` for per-step LR logging.
        partition_spec: Sharding spec for the input batch; resolved by
            :func:`make_assertions_and_get_sizes`.
        gradient_accumulation_steps: Number of microbatches whose
            gradients are accumulated per optimiser update.
        is_training: When ``True`` differentiate, accumulate, and apply
            the optimiser update; when ``False`` evaluate the loss and
            return only metrics.
        straight_through_emulator: Optional STE wrapper applied to the
            parameter tree inside the loss closure to simulate
            quantised forward passes during training.

    Returns:
        ``(updated_state, metrics)`` in training mode; in eval mode
        only the :class:`LossMetrics`.
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """Compute the clipped PPO objective for a single minibatch.

        Args:
            tree: Differentiable parameter tree (the policy graph state).
            minibatch (collections.abc.Mapping[str, jax.Array]): Minibatch
                with ``input_ids``, ``attention_mask``, ``completion_mask``,
                ``old_logps``, ``old_values``, ``advantages``, and ``returns``.

        Returns:
            tuple[jax.Array, LossMetrics]: Scalar loss and a
            :class:`LossMetrics` instance with diagnostic statistics
            (policy/value losses, approx KL, clip fractions, etc.).
        """
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree)

        input_ids = minibatch["input_ids"]
        attention_mask = minibatch["attention_mask"]
        completion_mask = minibatch["completion_mask"].astype(jnp.float32)

        old_logps = minibatch["old_logps"]
        old_values = minibatch["old_values"]
        advantages = jax.lax.stop_gradient(minibatch["advantages"])
        returns = jax.lax.stop_gradient(minibatch["returns"])

        new_logps, new_values, entropies = get_per_token_logps_values_entropies(
            module,
            input_ids,
            attention_mask,
            prompt_length,
            logprob_vocab_chunk_size,
        )

        log_ratio = new_logps - old_logps
        ratio = jnp.exp(log_ratio)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
        pg_loss = _masked_mean(jnp.maximum(pg_losses, pg_losses2), completion_mask)

        values_clipped = old_values + jnp.clip(new_values - old_values, -cliprange_value, cliprange_value)
        vf_losses1 = jnp.square(new_values - returns)
        vf_losses2 = jnp.square(values_clipped - returns)
        vf_loss = 0.5 * _masked_mean(jnp.maximum(vf_losses1, vf_losses2), completion_mask)

        entropy = _masked_mean(entropies, completion_mask)
        loss = pg_loss + vf_coef * vf_loss - entropy_coef * entropy

        approx_kl = 0.5 * _masked_mean(jnp.square(log_ratio), completion_mask)
        pg_clipfrac = _masked_mean(((pg_losses2 > pg_losses).astype(jnp.float32)), completion_mask)
        vf_clipfrac = _masked_mean(((vf_losses2 > vf_losses1).astype(jnp.float32)), completion_mask)

        other_metrics: dict[str, jax.Array] = {
            "policy_loss": pg_loss,
            "value_loss": vf_loss,
            "mean_entropy": entropy,
            "approx_kl": approx_kl,
            "pg_clipfrac": pg_clipfrac,
            "vf_clipfrac": vf_clipfrac,
            "ratio_mean": _masked_mean(ratio, completion_mask),
            "advantages_mean": _masked_mean(advantages, completion_mask),
            "returns_mean": _masked_mean(returns, completion_mask),
        }

        return loss, LossMetrics(loss=loss, accuracy=1, other_metrics=other_metrics)

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

    _, metrics = loss_fn(state.graphstate, batch)
    return metrics
