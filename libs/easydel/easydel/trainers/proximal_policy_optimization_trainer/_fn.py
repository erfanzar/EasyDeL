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
- Generalised Advantage Estimation with scalar or per-trajectory lambda
- Clipped policy loss (optionally asymmetric, "clip-higher") and clipped value loss
- VAPO extensions: global token-level normalisation, positive-example NLL and
  value-pretraining gating
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

from .._shared.policy_losses import (
    _MAX_SAFE_LOG_RATIO,
    _MIN_SAFE_LOG_RATIO,
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


def _token_normalizer(
    local_mask: jax.Array,
    global_count: jax.Array | None,
    *,
    token_level_loss: bool,
    gradient_accumulation_steps: int,
) -> jax.Array:
    """Return the denominator of a masked token-sum loss.

    With ``token_level_loss=False`` this is the microbatch-local token count
    (a masked mean). With ``token_level_loss=True`` the full rollout-batch
    count ``global_count`` is used instead. :func:`minibatch_call` does not
    slice 0-dim leaves and averages microbatch gradients by
    ``1/gradient_accumulation_steps``, so the global count is divided by the
    accumulation steps to recombine to ``sum(loss) / global_count`` exactly.

    Args:
        local_mask: Mask of the tokens contributing to the loss in this
            microbatch.
        global_count: 0-dim token count over the whole rollout batch, or
            ``None`` to fall back to the local count.
        token_level_loss: Whether to normalise by the global count.
        gradient_accumulation_steps: Number of accumulated microbatches.

    Returns:
        A float32 scalar denominator. For local counts it is clamped to at
        least ``1.0``; for global counts ``global_count`` is clamped to at
        least ``1.0`` *before* dividing by ``gradient_accumulation_steps`` so
        ``0 < global_count < gradient_accumulation_steps`` still recombines to
        ``sum(loss) / global_count`` under ``1 / K`` averaging.
    """
    if token_level_loss and global_count is not None:
        return jnp.maximum(jnp.asarray(global_count, dtype=jnp.float32), 1.0) / max(int(gradient_accumulation_steps), 1)
    return jnp.maximum(jnp.sum(local_mask.astype(jnp.float32)), 1.0)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    mask: jax.Array,
    *,
    gamma: float,
    lam: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Generalised Advantage Estimation over right-padded completions.

    Runs the backward recursion

    ``delta_t = r_t + gamma * V_{t+1} * m_{t+1} - V_t``

    ``A_t = (delta_t + gamma * lambda * A_{t+1} * m_{t+1}) * m_t``

    with :func:`jax.lax.scan`. ``lam`` may be a scalar or one value per
    trajectory row, which is what length-adaptive GAE needs.

    Args:
        rewards: ``(batch, completion_len)`` per-token rewards.
        values: ``(batch, completion_len)`` value predictions aligned with
            ``rewards``.
        mask: Same-shape mask of valid completion positions.
        gamma: Discount factor.
        lam: GAE trace decay; a scalar or a ``(batch,)`` array.

    Returns:
        ``(advantages, returns)`` with ``returns = advantages + values``. Both
        are float32 ``(batch, completion_len)``. Advantages are zero outside
        ``mask``.
    """
    rewards = rewards.astype(jnp.float32)
    values = values.astype(jnp.float32)
    mask = mask.astype(jnp.float32)
    batch_size = rewards.shape[0]
    gamma = float(gamma)
    if isinstance(lam, int | float):
        trace_decay: float | jax.Array = gamma * float(lam)
    else:
        trace_decay = gamma * jnp.broadcast_to(jnp.asarray(lam, dtype=jnp.float32), (batch_size,))

    values_next = jnp.concatenate([values[:, 1:], jnp.zeros((batch_size, 1), dtype=values.dtype)], axis=1)
    mask_next = jnp.concatenate([mask[:, 1:], jnp.zeros((batch_size, 1), dtype=mask.dtype)], axis=1)

    def scan_fn(adv_next, inputs):
        r_t, v_t, v_next_t, m_t, m_next_t = inputs
        delta = r_t + gamma * v_next_t * m_next_t - v_t
        adv_t = delta + trace_decay * adv_next * m_next_t
        adv_t = adv_t * m_t
        return adv_t, adv_t

    inputs = (
        rewards[:, ::-1].T,
        values[:, ::-1].T,
        values_next[:, ::-1].T,
        mask[:, ::-1].T,
        mask_next[:, ::-1].T,
    )
    _, adv_rev = jax.lax.scan(scan_fn, jnp.zeros((batch_size,), dtype=jnp.float32), inputs)
    advantages = adv_rev.T[:, ::-1]
    return advantages, advantages + values


def length_adaptive_gae_lambda(lengths: jax.Array, *, alpha: float) -> jax.Array:
    """VAPO length-adaptive policy lambda ``clip(1 - 1 / (alpha * l), 0, 1)``.

    Short completions get a small lambda (low variance, more bootstrapping).
    Long completions get a lambda near one, so the terminal reward is not
    exponentially decayed before it reaches early tokens. Rows of length zero
    get ``0``; they are fully masked anyway.

    Args:
        lengths: ``(batch,)`` number of completion tokens per trajectory.
        alpha: Positive length scale (VAPO uses ``0.05``).

    Returns:
        Float32 ``(batch,)`` lambdas in ``[0, 1]``.

    Raises:
        ValueError: If ``alpha`` is not positive.
    """
    if alpha <= 0:
        raise ValueError(f"`alpha` must be positive, got {alpha}.")
    lengths = jnp.asarray(lengths, dtype=jnp.float32)
    scheduled = 1.0 - 1.0 / (float(alpha) * jnp.maximum(lengths, 1.0))
    return jnp.where(lengths > 0, jnp.clip(scheduled, 0.0, 1.0), 0.0)


def clipped_policy_loss(
    new_logps: jax.Array,
    old_logps: jax.Array,
    advantages: jax.Array,
    *,
    cliprange: float,
    cliprange_high: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-token PPO clipped surrogate with an optional decoupled upper clip.

    ``loss_t = max(-A_t * r_t, -A_t * clip(r_t, 1 - cliprange, 1 + eps_high))``
    where ``eps_high = cliprange_high`` when set ("clip-higher", DAPO/VAPO) and
    ``cliprange`` otherwise.

    Args:
        new_logps: Current-policy per-token log-probabilities.
        old_logps: Rollout-policy per-token log-probabilities.
        advantages: Per-token advantages (treated as constants).
        cliprange: Lower clip ``eps_low``; also the upper clip when
            ``cliprange_high`` is ``None``.
        cliprange_high: Optional upper clip ``eps_high``.

    Returns:
        ``(per_token_loss, ratio, clipped)`` where ``clipped`` is a float mask
        of tokens whose clipped branch was selected.
    """
    upper = cliprange if cliprange_high is None else cliprange_high
    log_ratio = (new_logps - old_logps).astype(jnp.float32)
    safe_log_ratio = jnp.clip(log_ratio, _MIN_SAFE_LOG_RATIO, _MAX_SAFE_LOG_RATIO)
    ratio = jnp.exp(safe_log_ratio)
    low = 1.0 - cliprange
    high = 1.0 + upper
    use_clipped = ((advantages > 0.0) & (ratio > high)) | ((advantages < 0.0) & (ratio < low))
    safe_unclipped_ratio = jnp.where(~use_clipped, ratio, 1.0)
    unclipped = -advantages * safe_unclipped_ratio
    clipped_branch = -advantages * jnp.where(advantages > 0.0, high, low)
    per_token_loss = jnp.where(use_clipped, clipped_branch, unclipped)
    return per_token_loss, ratio, use_clipped.astype(jnp.float32)


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
    log_probs = jax.nn.log_softmax(logits[:, prompt_length:, :], axis=-1)
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
    detach_value_hidden: bool | jax.Array = False,
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
        detach_value_hidden: When true (a Python bool or a traced scalar
            bool), the value head reads ``stop_gradient(hidden_states)``.
            Value-loss gradients then reach only the value head and leave
            the shared policy backbone alone (used by value pretraining).

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
        logits = outputs.logits
        if logits is None:
            raise ValueError("Model outputs do not provide logits for PPO scoring.")
        logits = logits[:, prompt_length - 1 :]
        logits = logits[:, :-1, :]
        token_log_probs, entropies = compute_token_logps_and_entropies_chunked(
            logits,
            targets,
            return_entropy=True,
            chunk_size=logprob_vocab_chunk_size,
        )

    value_hidden_states = hidden_states
    if detach_value_hidden is not False:
        value_hidden_states = jnp.where(detach_value_hidden, jax.lax.stop_gradient(hidden_states), hidden_states)
    values_full = model.value_head(value_hidden_states).squeeze(-1)
    values = values_full[:, prompt_length - 1 : -1]
    return token_log_probs, values, entropies


VALUE_HEAD_MODULE_NAME = "value_head"


def _is_value_head_path(path: tuple[tp.Any, ...]) -> bool:
    """Return whether a pytree key path addresses a leaf of the ``value_head`` module."""
    return VALUE_HEAD_MODULE_NAME in jax.tree_util.keystr(path)


def freeze_policy_during_value_pretrain(
    old_state: EasyDeLState,
    new_state: EasyDeLState,
    in_value_pretrain: jax.Array,
) -> EasyDeLState:
    """Keep every non-value-head parameter (and its optimiser moments) at its pre-step value.

    Zeroing the policy gradient is not enough to freeze the policy: decoupled weight decay
    (e.g. AdamW) still shrinks the parameters and momentum keeps moving them. While
    ``in_value_pretrain`` is true this selects, per leaf, the pre-step value for every
    parameter outside the ``value_head`` module and for every optimiser-state leaf that mirrors
    such a parameter (moment trees share the parameter's key path as a suffix). Value-head
    leaves, the step counter and shared optimiser scalars (e.g. Adam's ``count``) advance.

    Args:
        old_state: State before the optimiser update.
        new_state: State after the optimiser update.
        in_value_pretrain: Traced scalar bool; when false ``new_state`` is returned unchanged
            (as values).

    Returns:
        ``new_state`` with policy parameters and their optimiser moments restored during
        value pretraining.
    """

    def select(path, old, new):
        if _is_value_head_path(path):
            return new
        return jnp.where(in_value_pretrain, old, new)

    graphstate = jax.tree_util.tree_map_with_path(select, old_state.graphstate, new_state.graphstate)

    policy_paths = {
        tuple(path)
        for path, _ in jax.tree_util.tree_leaves_with_path(old_state.graphstate)
        if not _is_value_head_path(path)
    }

    def mirrors_policy_param(path) -> bool:
        path = tuple(path)
        return any(path[len(path) - n :] in policy_paths for n in range(1, len(path) + 1))

    def select_opt(path, old, new):
        if not mirrors_policy_param(path):
            return new
        return jnp.where(in_value_pretrain, old, new)

    opt_state = new_state.opt_state
    if opt_state is not None and old_state.opt_state is not None:
        opt_state = jax.tree_util.tree_map_with_path(select_opt, old_state.opt_state, opt_state)
    return new_state.replace(graphstate=graphstate, opt_state=opt_state)


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
    cliprange_high: float | None = None,
    token_level_loss: bool = False,
    positive_nll_coef: float = 0.0,
    value_pretrain_steps: int = 0,
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

    VAPO extensions (all disabled by default, which leaves the loss above
    unchanged):

    * ``cliprange_high`` decouples the upper ratio clip (clip-higher).
    * ``token_level_loss`` divides the policy-gradient token sum by the
      rollout-batch token count ``num_items_in_batch`` instead of the
      microbatch-local count.
    * ``positive_nll_coef`` adds ``coef * NLL`` over the completion tokens of
      trajectories flagged by ``positive_mask``.
    * ``value_pretrain_steps``: while ``state.step < value_pretrain_steps``,
      the loss is only ``vf_coef * vf_loss`` and the value head reads
      detached hidden states, so the policy backbone receives zero gradient.

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
        cliprange_high: Optional upper clip ``eps_high``; ``None`` keeps the
            symmetric ``cliprange``.
        token_level_loss: Normalise the policy loss by the global rollout
            token count carried in ``batch["num_items_in_batch"]``.
        positive_nll_coef: Weight of the positive-example NLL term; requires
            ``batch["positive_mask"]`` (and ``num_positive_tokens`` when
            ``token_level_loss`` is set) when positive.
        value_pretrain_steps: Number of initial optimiser steps that train
            only the value head.

    Returns:
        ``(updated_state, metrics)`` in training mode; in eval mode
        only the :class:`LossMetrics`.
    """
    normalizer_accumulation_steps = int(gradient_accumulation_steps) if is_training else 1
    scope_root = "easydel/trainer/ppo/" + ("train_step" if is_training else "eval_step")
    effective_accum_steps = max(int(gradient_accumulation_steps), 1) if is_training else 1
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=effective_accum_steps,
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
        with jax.named_scope(scope_root + "/loss_fn"):
            if is_training and straight_through_emulator is not None:
                with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                    tree = straight_through_emulator(tree)
            with jax.named_scope(scope_root + "/loss_fn/merge_state"):
                module = state.merge(tree)

            input_ids = minibatch["input_ids"]
            attention_mask = minibatch["attention_mask"]
            completion_mask = minibatch["completion_mask"].astype(jnp.float32)
            mask_bool = completion_mask > 0.0

            old_logps = jax.lax.stop_gradient(minibatch["old_logps"].astype(jnp.float32))
            old_values = jax.lax.stop_gradient(jnp.where(mask_bool, minibatch["old_values"].astype(jnp.float32), 0.0))
            advantages = jax.lax.stop_gradient(jnp.where(mask_bool, minibatch["advantages"].astype(jnp.float32), 0.0))
            returns = jax.lax.stop_gradient(jnp.where(mask_bool, minibatch["returns"].astype(jnp.float32), 0.0))
            in_value_pretrain = jnp.asarray(state.step) < value_pretrain_steps if value_pretrain_steps > 0 else None

            with jax.named_scope(scope_root + "/loss_fn/policy_forward"):
                new_logps, new_values, entropies = get_per_token_logps_values_entropies(
                    module,
                    input_ids,
                    attention_mask,
                    prompt_length,
                    logprob_vocab_chunk_size,
                    detach_value_hidden=False if in_value_pretrain is None else in_value_pretrain,
                )

            with jax.named_scope(scope_root + "/loss_fn/policy_loss"):
                safe_new_logps = jnp.where(mask_bool, new_logps.astype(jnp.float32), 0.0)
                safe_old_logps = jnp.where(mask_bool, old_logps, 0.0)
                safe_log_ratio = safe_new_logps - safe_old_logps
                per_token_pg_loss, ratio, pg_clipped = clipped_policy_loss(
                    safe_new_logps,
                    safe_old_logps,
                    advantages,
                    cliprange=cliprange,
                    cliprange_high=cliprange_high,
                )
                safe_ratio = jnp.where(mask_bool, ratio, 0.0)
                pg_denominator = _token_normalizer(
                    completion_mask,
                    minibatch.get("num_items_in_batch"),
                    token_level_loss=token_level_loss,
                    gradient_accumulation_steps=normalizer_accumulation_steps,
                )
                pg_loss = jnp.sum(jnp.where(mask_bool, per_token_pg_loss, 0.0)) / pg_denominator

            with jax.named_scope(scope_root + "/loss_fn/value_loss"):
                safe_new_values = jnp.where(mask_bool, new_values.astype(jnp.float32), 0.0)
                values_clipped = old_values + jnp.clip(safe_new_values - old_values, -cliprange_value, cliprange_value)
                vf_losses1 = jnp.square(safe_new_values - returns)
                vf_losses2 = jnp.square(values_clipped - returns)
                vf_loss = 0.5 * _masked_mean(jnp.maximum(vf_losses1, vf_losses2), completion_mask)

            with jax.named_scope(scope_root + "/loss_fn/entropy_and_combine"):
                entropy = _masked_mean(jnp.where(mask_bool, entropies.astype(jnp.float32), 0.0), completion_mask)
                loss = pg_loss + vf_coef * vf_loss - entropy_coef * entropy
                positive_nll = jnp.zeros((), dtype=jnp.float32)
                if positive_nll_coef > 0.0:
                    positive_tokens = completion_mask * minibatch["positive_mask"].astype(jnp.float32)[:, None]
                    nll_denominator = _token_normalizer(
                        positive_tokens,
                        minibatch.get("num_positive_tokens"),
                        token_level_loss=token_level_loss,
                        gradient_accumulation_steps=normalizer_accumulation_steps,
                    )
                    positive_nll = -jnp.sum(safe_new_logps * positive_tokens) / nll_denominator
                    loss = loss + positive_nll_coef * positive_nll
                if in_value_pretrain is not None:
                    loss = jnp.where(in_value_pretrain, vf_coef * vf_loss, loss)

            with jax.named_scope(scope_root + "/loss_fn/diagnostics"):
                approx_kl = 0.5 * _masked_mean(jnp.square(safe_log_ratio), completion_mask)
                pg_clipfrac = _masked_mean(pg_clipped, completion_mask)
                vf_clipfrac = _masked_mean(((vf_losses2 > vf_losses1).astype(jnp.float32)), completion_mask)

                other_metrics: dict[str, jax.Array] = {
                    "policy_loss": pg_loss,
                    "value_loss": vf_loss,
                    "mean_entropy": entropy,
                    "approx_kl": approx_kl,
                    "pg_clipfrac": pg_clipfrac,
                    "vf_clipfrac": vf_clipfrac,
                    "ratio_mean": _masked_mean(safe_ratio, completion_mask),
                    "advantages_mean": _masked_mean(advantages, completion_mask),
                    "returns_mean": _masked_mean(returns, completion_mask),
                }
                if positive_nll_coef > 0.0:
                    other_metrics["positive_nll"] = positive_nll
                if in_value_pretrain is not None:
                    other_metrics["value_pretrain_active"] = in_value_pretrain.astype(jnp.float32)

            return loss, LossMetrics(loss=loss, accuracy=1, other_metrics=other_metrics)

    if is_training:
        with jax.named_scope(scope_root + "/grad_and_minibatch"):
            gradients, metrics = minibatch_call(
                state=state,
                batch=batch,
                minibatch_size=minibatch_size,
                grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
            )
        pre_update_state = state
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
        if value_pretrain_steps > 0:
            with jax.named_scope(scope_root + "/freeze_policy_during_value_pretrain"):
                state = freeze_policy_during_value_pretrain(
                    pre_update_state,
                    state,
                    jnp.asarray(pre_update_state.step) < value_pretrain_steps,
                )
        return state, metrics

    with jax.named_scope(scope_root + "/eval_call"):
        _, metrics = loss_fn(state.graphstate, batch)
    return metrics
