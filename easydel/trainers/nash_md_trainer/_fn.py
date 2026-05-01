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
"""Loss and step implementations for the Nash-MD trainer.

Implements the Nash mirror-descent objective: a regularised
expected-reward gradient against a mixture between the current policy
and a reference, with optional KL clipping and missing-EOS reward
penalty.  The scheduled-VJP variant supports MPMD pipeline parallelism.
"""

from __future__ import annotations

import typing as tp

import jax
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
    update_metrics,
    update_state_respectfully,
)


def _compute_policy_logps(
    module: spx.Module,
    prompt_ids: jax.Array,
    prompt_mask: jax.Array,
    completion_ids: jax.Array,
    completion_mask: jax.Array,
    logprob_vocab_chunk_size: int | None,
) -> jax.Array:
    """Compute policy log probabilities for completion tokens.

    Args:
        module: Policy model module.
        prompt_ids: Prompt token IDs.
        prompt_mask: Prompt attention mask.
        completion_ids: Completion token IDs.
        completion_mask: Completion attention mask.

    Returns:
        Per-token log probabilities for completions.
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


def nash_md_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    beta: float,
    logprob_vocab_chunk_size: int | None,
    loss_config: LossConfig | None,
    learning_rate_fn,
    partition_spec: PartitionSpec | None,
    gradient_accumulation_steps: int,
    is_train: bool,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run one Nash-MD training or evaluation step.

    Consumes a batch that the trainer has already augmented with:

    * ``ref_token_logps`` -- per-token reference logps for the
      completion tokens (frozen).
    * ``probabilities`` -- the preference oracle's win probability of
      the policy completion against the mixture-sampled completion.

    The mirror-descent objective is

    ``L = beta * KL(policy || reference)
          - (probabilities - 0.5) * policy_token_logps``

    summed over completion tokens and masked by ``completion_mask``.
    The KL term is computed in token-space using the policy and
    reference per-token logps; the second term scales the policy
    log-likelihood by the centred oracle preference, pushing the
    policy toward completions the oracle preferred.

    Args:
        state: Policy ``EasyDeLState`` being differentiated.
        batch: Nash-MD minibatch with prompt/completion ids and
            masks, ``ref_token_logps``, and ``probabilities``.
        beta: KL coefficient against the reference.
        logprob_vocab_chunk_size: Vocab-axis chunk size used by the
            policy logp computation.
        loss_config: ``LossConfig`` controlling NaN handling.
        learning_rate_fn: Schedule mapping step to learning rate.
        partition_spec: Sharding spec applied to the input batch.
        gradient_accumulation_steps: Gradient-accumulation factor.
        is_train: When ``False`` skips gradient computation and
            returns only ``LossMetrics``.
        straight_through_emulator: Optional STE callable applied to
            the graphstate before the forward (QAT path).

    Returns:
        ``(new_state, metrics)`` when ``is_train`` is ``True``;
        otherwise ``LossMetrics`` alone. ``metrics.other_metrics``
        records the score, KL, mean oracle probability, and mean
        policy log-prob for diagnostics.
    """

    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)
    beta = jnp.asarray(beta, dtype=jnp.float32)

    def loss_fn(tree: spx.State, minibatch: dict[str, jax.Array]):
        """Compute the Nash-MD loss for one minibatch.

        Computes per-token policy log-probabilities, evaluates the
        ``beta * KL - (oracle_prob - 0.5) * policy_logp`` mirror-descent
        objective against the precomputed reference per-token logps,
        and records score / KL / probability / mean policy-logp as
        diagnostics.

        Args:
            tree: Policy graphstate to differentiate against.
            minibatch: Dict carrying prompt and completion ids/masks,
                ``ref_token_logps``, and the oracle ``probabilities``.

        Returns:
            ``(loss, metrics)`` ready for ``minibatch_call``.
        """
        if is_train and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree=tree)

        prompt_ids = minibatch["prompt_ids"]
        prompt_mask = minibatch["prompt_mask"]
        completion_ids = minibatch["completion_ids"]
        completion_mask = minibatch["completion_mask"]
        ref_token_logps = minibatch["ref_token_logps"]
        probabilities = minibatch["probabilities"]

        policy_token_logps = _compute_policy_logps(
            module,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            logprob_vocab_chunk_size,
        )
        mask = completion_mask.astype(policy_token_logps.dtype)
        policy_token_logps = policy_token_logps * mask
        ref_token_logps = ref_token_logps * mask

        policy_logps = policy_token_logps.sum(axis=1)
        log_ratio = policy_token_logps - ref_token_logps
        kl_vector = log_ratio.sum(axis=1)
        kl_loss = (log_ratio * policy_token_logps).sum(axis=1)

        score = (probabilities - 0.5) * policy_logps
        loss_vector = beta * kl_loss - score
        loss = loss_vector.mean()

        metrics = LossMetrics(
            loss=loss,
            other_metrics={
                "score": score.mean(),
                "kl": kl_vector.mean(),
                "probability": probabilities.mean(),
                "policy_logps": policy_logps.mean(),
            },
        )
        return loss, metrics

    if is_train:
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
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

    _, metrics = loss_fn(state.graphstate, batch)
    return metrics


def _nash_md_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the Nash-MD scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple covering ``beta``, the logprob vocab chunk size, the
        partition spec, and the quantization emulator identity.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=("beta", "logprob_vocab_chunk_size", "partition_spec"),
        object_fields=("straight_through_emulator",),
    )


def _make_nash_md_scheduled_loss(call):
    """Build a SpectraX-scheduled Nash-MD scalar-loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` carrying loss config.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` ready for
        :func:`spx.sxvalue_and_grad`.
    """
    beta = jnp.asarray(call.get("beta"), dtype=jnp.float32)
    logprob_vocab_chunk_size = call.get("logprob_vocab_chunk_size")
    partition_spec = call.get("partition_spec")

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the scalar Nash-MD loss inside the SpectraX scheduled VJP.

        Args:
            tree: Policy graphstate to differentiate against.
            batch: Minibatch dict with prompt / completion ids, masks,
                ``ref_token_logps`` and ``probabilities``.

        Returns:
            The scalar Nash-MD loss.
        """
        module = bind_scheduled_module(call, tree)
        batch = constrain_scheduled_batch(module, batch, partition_spec)
        completion_mask = batch["completion_mask"]
        policy_token_logps = _compute_policy_logps(
            module,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["completion_ids"],
            completion_mask,
            logprob_vocab_chunk_size,
        )
        mask = completion_mask.astype(policy_token_logps.dtype)
        policy_token_logps = policy_token_logps * mask
        ref_token_logps = jax.lax.stop_gradient(batch["ref_token_logps"]) * mask
        policy_logps = policy_token_logps.sum(axis=1)
        log_ratio = policy_token_logps - ref_token_logps
        kl_loss = (log_ratio * policy_token_logps).sum(axis=1)
        score = (batch["probabilities"] - 0.5) * policy_logps
        return (beta * kl_loss - score).mean()

    return scheduled_loss


register_scheduled_loss_adapter(
    nash_md_step,
    ScheduledLossAdapter(
        name="nash_md",
        make_loss=_make_nash_md_scheduled_loss,
        make_cache_key=_nash_md_scheduled_loss_cache_key,
    ),
)
