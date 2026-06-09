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

"""Internal functions for Odds Ratio Preference Optimization training.

This module contains the core computational functions used by the ORPO trainer,
implementing odds ratio-based preference optimization without requiring a reference
model. ORPO formulates preference learning through odds ratios, providing a
mathematically principled and computationally efficient alternative to DPO.

The module provides functions for:
- Computing log probabilities and odds ratios for chosen/rejected samples
- Implementing the ORPO loss function with log-odds differences
- Handling both encoder-decoder and decoder-only architectures
- Efficient batch processing with concatenated forward passes

ORPO's key innovation is using odds ratios (p/(1-p)) instead of raw probabilities,
which provides better gradient properties and eliminates the need for a reference model.

All functions are JAX-compatible and support distributed training.
"""

import collections.abc
import typing as tp

import jax
import spectrax as spx
from ejkernel.modules.operations import fused_cross_entropy as _fused_cross_entropy
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array
from spectrax import with_sharding_constraint
from spectrax.common_types import BATCH, LENGTH, MODE_TRAIN, VOCAB

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers._logprob_utils import (
    compute_sequence_scores_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)

from ..training_utils import (
    ScheduledLossAdapter,
    _scheduled_terminal_stage_rank,
    bind_scheduled_module,
    constrain_scheduled_batch,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)


def _log1mexp(x: Array) -> Array:
    """Compute ``log(1 - exp(x))`` stably for non-positive log-probabilities."""
    cutoff = jnp.array(-0.6931471805599453, dtype=x.dtype)
    return jnp.where(x < cutoff, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


def concatenated_forward(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, list | Array],
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: tp.Any,
    max_length: int | None = None,
    logprob_vocab_chunk_size: int | None = None,
    vocab_shard_stage: int | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Compute concatenated chosen/rejected log-probs, logits, NLL, and accuracy.

    Concatenates the chosen and rejected batches along the batch axis,
    performs a single forward pass through the policy, then splits the
    per-sequence statistics back into chosen and rejected halves. When the
    model exposes an ``lmhead_chunksize`` the forward is run in headless
    mode and per-token log probabilities are recomputed from the hidden
    states with vocabulary chunking to avoid materialising
    ``[2 * batch, seq, vocab]`` logit tensors.

    Args:
        state (EasyDeLState): Current model state holding parameters and the
            bound policy module.
        batch (collections.abc.Mapping[str, list | Array]): Mapping with input
            arrays for chosen and rejected examples (keys prefixed by
            ``chosen``/``rejected``) plus any extra fields required for an
            encoder-decoder model.
        is_encoder_decoder (bool): Whether the model is an encoder-decoder
            architecture.
        label_pad_token_id (int): Token ID used to mark padding positions in
            the labels; ignored by the loss and accuracy reductions.
        padding_value (Any): Pad token value used by the collator. Must not
            be ``None``.
        max_length (int | None): Maximum sequence length used by the
            encoder-decoder branches when reconciling label shapes; defaults
            to ``None``.
        logprob_vocab_chunk_size (int | None): Vocabulary chunk size used by
            the chunked log-prob path. ``None`` disables vocabulary
            chunking.
        vocab_shard_stage (int | None): Optional MPMD pipeline stage rank
            forwarded to the chunked LM-head projection so the vocab shard
            lands on the terminal stage.

    Returns:
        tuple[Array, Array, Array, Array, Array, Array]: Tuple of
        ``(chosen_log_probs, rejected_log_probs, chosen_logits,
        rejected_logits, chosen_nll_loss, chosen_accuracy)`` where the
        logit entries are per-example mean logit summaries and the NLL /
        accuracy are reduced scalars over chosen examples.

    Raises:
        ValueError: If ``padding_value`` is ``None``.
        TypeError: If the model returns neither ``logits`` nor
            ``last_hidden_state`` when the chunked path is active.
    """
    if padding_value is None:
        raise ValueError("`padding_value` can not be set as `None` it must be an integer.")
    model = state.model if isinstance(state, EasyDeLState) else getattr(state, "model", state)

    # Concatenate inputs from chosen and rejected examples.
    concatenated_batch = concatenated_inputs(batch, is_encoder_decoder)

    len_chosen = batch["chosen_labels"].shape[0]

    # Prepare model keyword arguments for encoder-decoder architectures.
    model_kwargs = (
        {
            "labels": concatenated_batch["concatenated_labels"],
            "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
        }
        if is_encoder_decoder
        else {}
    )
    lmhead_chunksize = None
    if not is_encoder_decoder:
        lmhead_chunksize = resolve_lmhead_chunksize(model)
        if lmhead_chunksize is not None:
            model_kwargs["apply_lm_head"] = False

    # Forward pass through the model.
    call_kwargs = {
        "input_ids": concatenated_batch["concatenated_input_ids"],
        "attention_mask": concatenated_batch["concatenated_attention_mask"],
        **model_kwargs,
    }
    call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
    call_kwargs = sanitize_model_call_kwargs(call_kwargs)
    outputs = model(**call_kwargs)
    all_logits = getattr(outputs, "logits", None)

    effective_labels = concatenated_batch["concatenated_labels"]
    if is_encoder_decoder and all_logits is not None and effective_labels.shape != all_logits.shape[:-1]:
        candidate_labels = call_kwargs.get("labels")
        if candidate_labels is None:
            candidate_labels = call_kwargs.get("decoder_input_ids")
        if candidate_labels is None:
            candidate_labels = call_kwargs.get("input_ids")
        if candidate_labels is not None and candidate_labels.shape == all_logits.shape[:-1]:
            effective_labels = candidate_labels
        else:
            target_seq_len = all_logits.shape[1]
            current_seq_len = effective_labels.shape[1]
            if current_seq_len >= target_seq_len:
                effective_labels = effective_labels[:, :target_seq_len]
            else:
                pad_shape = (effective_labels.shape[0], target_seq_len - current_seq_len)
                pad_values = jnp.full(pad_shape, label_pad_token_id, dtype=effective_labels.dtype)
                effective_labels = jnp.concatenate((effective_labels, pad_values), axis=1)

    def cross_entropy_loss(logits, labels):
        """Compute cross-entropy loss and token-level accuracy.

        For decoder-only models the logits/labels are causally shifted
        (drop the last logit, drop the first label) before scoring.

        Args:
            logits (Array): Logits produced by the model with shape
                ``(batch, seq_len, vocab_size)``.
            labels (Array): Ground-truth label ids with shape
                ``(batch, seq_len)``.

        Returns:
            tuple[Array, Array]: ``(loss, accuracy)`` -- scalar
            cross-entropy averaged over non-padding tokens and the
            corresponding masked top-1 accuracy.
        """
        if not is_encoder_decoder:
            logits = logits[..., :-1, :]
            labels = labels[..., 1:]
        # masked-mean NLL over non-padding tokens via the ejkernel fused kernel (XLA backend). On a real
        # (>1) TP mesh, resolve [BATCH, LENGTH, VOCAB] / [BATCH, LENGTH] specs and run inside shard_map so
        # the vocab stays TP-sharded -- the full [B, S, V] logits are never all-reduced for the softmax.
        ce_kwargs: dict[str, tp.Any] = dict(ignore_index=label_pad_token_id, reduction="mean", platform="xla")
        ce_mesh = spx.get_current_stage_mesh(spx.get_incontext_mesh(raise_error=False), raise_error=False)
        ce_logits, ce_labels = logits, labels
        # Vocab-parallel NLL only for a real (>1) ``tp`` axis (no vocab to shard at tp=1). shard_map needs
        # every meshed leading dim evenly divisible -- the chosen-only half is frequently indivisible
        # (e.g. 1 row over fsdp=2), so pad [batch, seq] up to the sharded-axis product first. Padded rows
        # carry ``label_pad_token_id`` and are dropped by the masked mean (and excluded from accuracy).
        if ce_mesh is not None and "tp" in ce_mesh.axis_names and int(ce_mesh.shape["tp"]) > 1:
            _pm = spx.PartitionManager(paxis=model.config.partition_axis)
            _logit_spec = _pm.resolve([BATCH, LENGTH, VOCAB], MODE_TRAIN)
            _token_spec = _pm.resolve([BATCH, LENGTH], MODE_TRAIN)

            def _axis_prod(entry):
                names = entry if isinstance(entry, tuple) else ((entry,) if entry is not None else ())
                prod = 1
                for nm in names:
                    prod *= int(ce_mesh.shape[nm])
                return prod

            pads = [
                (-int(d)) % max(_axis_prod(_token_spec[i]) if i < len(_token_spec) else 1, 1)
                for i, d in enumerate(logits.shape[:-1])
            ]
            if any(p > 0 for p in pads):
                ce_logits = jnp.pad(logits, [(0, p) for p in pads] + [(0, 0)])
                ce_labels = jnp.pad(labels, [(0, p) for p in pads], constant_values=label_pad_token_id)
            ce_kwargs.update(mesh=ce_mesh, in_specs=(_logit_spec, _token_spec), out_specs=PartitionSpec())
        loss = _fused_cross_entropy(ce_logits, ce_labels, **ce_kwargs).loss
        valid = labels != label_pad_token_id
        safe_labels = jnp.where(valid, labels, 0)
        accuracy = jnp.sum(
            valid.astype(jnp.float32) * (jnp.argmax(logits, axis=-1) == safe_labels).astype(jnp.float32)
        ) / jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        return loss, accuracy

    # Set labels for computing loss.
    if is_encoder_decoder:
        labels = effective_labels
    else:
        labels = concatenated_batch["concatenated_input_ids"]
        attention_mask = concatenated_batch["concatenated_attention_mask"]
        labels = jnp.where(attention_mask == 1, labels, label_pad_token_id)

    if not is_encoder_decoder and all_logits is None and lmhead_chunksize is not None:
        shifted_labels = labels[:, 1:]
        loss_mask = shifted_labels != label_pad_token_id
        labels_safe = jnp.where(loss_mask, shifted_labels, 0)
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        hidden_states = hidden_states[:, :-1, :]
        sum_logps, token_logit_sums, token_counts, correct_counts = compute_sequence_scores_from_hidden_states(
            model=model,
            hidden_states=hidden_states,
            labels=labels_safe,
            loss_mask=loss_mask,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_correct_counts=True,
            vocab_shard_stage=vocab_shard_stage,
        )
        token_counts = jnp.maximum(token_counts, 1.0)
        all_log_probs = sum_logps / token_counts
        chosen_log_probs = all_log_probs[:len_chosen]
        rejected_log_probs = all_log_probs[len_chosen:]
        chosen_logits = jnp.where(
            token_counts[:len_chosen] > 0,
            token_logit_sums[:len_chosen] / token_counts[:len_chosen],
            0.0,
        )
        rejected_logits = jnp.where(
            token_counts[len_chosen:] > 0,
            token_logit_sums[len_chosen:] / token_counts[len_chosen:],
            0.0,
        )
        chosen_nll_loss = -sum_logps[:len_chosen].sum() / jnp.maximum(token_counts[:len_chosen].sum(), 1.0)
        chosen_accuracy = correct_counts[:len_chosen].sum() / jnp.maximum(token_counts[:len_chosen].sum(), 1.0)
    else:
        if all_logits is None:
            raise TypeError(f"{type(model).__name__} did not return logits.")
        # Compute negative log likelihood loss and accuracy for the chosen examples.
        chosen_nll_loss, chosen_accuracy = cross_entropy_loss(
            all_logits[:len_chosen],
            labels[:len_chosen],
        )

        # Compute log probabilities for the entire batch.
        all_log_probs = get_batch_logps(
            all_logits,
            effective_labels,
            average_log_prob=True,
            is_encoder_decoder=is_encoder_decoder,
            label_pad_token_id=label_pad_token_id,
            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
        )

        # Split log probabilities and logit summaries into chosen and rejected.
        chosen_log_probs = all_log_probs[:len_chosen]
        rejected_log_probs = all_log_probs[len_chosen:]
        all_logit_summaries = get_batch_mean_logit_summaries(
            all_logits,
            effective_labels,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=is_encoder_decoder,
        )
        chosen_logits = all_logit_summaries[:len_chosen]
        rejected_logits = all_logit_summaries[len_chosen:]
    return (
        chosen_log_probs,
        rejected_log_probs,
        chosen_logits,
        rejected_logits,
        chosen_nll_loss,
        chosen_accuracy,
    )


def get_batch_logps(
    logits: Array,
    labels: Array,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
    logprob_vocab_chunk_size: int | None = None,
) -> Array:
    """Compute per-sequence log probabilities from logits and labels.

    Applies a log-softmax over the vocabulary, gathers the log probability of
    each realised label token, and masks out positions marked with
    ``label_pad_token_id``. For decoder-only models the logits/labels are
    causally shifted before reduction.

    Args:
        logits (Array): Logits output by the model with shape
            ``(..., sequence_length, vocab_size)``.
        labels (Array): Ground-truth labels with shape matching ``logits``
            except for the vocabulary dimension.
        average_log_prob (bool): If ``True`` return the average log
            probability per sequence; otherwise return the sum. Defaults to
            ``False``.
        label_pad_token_id (int): Token id marking padding positions in the
            labels. Defaults to ``-100``.
        is_encoder_decoder (bool): Whether the model is an encoder-decoder
            architecture. Defaults to ``False``.
        logprob_vocab_chunk_size (int | None): Vocabulary chunk size used by
            the chunked log-prob reduction. ``None`` disables chunking.

    Returns:
        Array: Per-sequence log-probability scores with shape ``(batch,)``.

    Raises:
        ValueError: If the batch / sequence-length dimensions of ``logits``
            (ignoring the vocab axis) do not match ``labels``.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    # For non encoder-decoder models, adjust logits and labels for proper alignment.
    if not is_encoder_decoder:
        labels = labels[:, 1:]
        logits = logits[:, :-1, :]

    # Create a mask to ignore the padded tokens.
    loss_mask = labels != label_pad_token_id
    # Replace pad token indices in labels with 0 (since they are masked out later).
    labels = jnp.expand_dims(jnp.where(labels == label_pad_token_id, 0, labels), -1)
    per_token_logps, _ = compute_token_logps_and_entropies_chunked(
        logits,
        jnp.squeeze(labels, axis=-1),
        return_entropy=False,
        chunk_size=logprob_vocab_chunk_size,
    )

    # Return averaged or summed log probabilities based on the flag.
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def get_batch_mean_logit_summaries(
    logits: Array,
    labels: Array,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> Array:
    """Compute a per-example mean logit summary over loss-bearing token positions.

    This utility replaces the earlier approach of returning full logit tensors
    (which are very large for big vocabularies) with a single scalar summary
    per example.  For each example in the batch it:

    1. Identifies the *loss-bearing* positions -- those whose label is not the
       padding sentinel ``label_pad_token_id``.
    2. Sums the raw logit values across the entire vocabulary at each
       loss-bearing position.
    3. Averages those sums over the number of loss-bearing tokens, producing
       one scalar per example.

    For decoder-only models (``is_encoder_decoder=False``), the labels and
    logits are shifted so that position *t* of the logits predicts position
    *t + 1* of the labels, matching the standard causal-LM alignment
    convention.

    Args:
        logits: Float array of shape ``(batch, seq_len, vocab_size)`` with the
            unnormalized model predictions.
        labels: Integer array of shape ``(batch, seq_len)`` with target token
            ids.  Positions set to ``label_pad_token_id`` are excluded from
            the summary.
        label_pad_token_id: The sentinel value used to mark padding / ignored
            positions in *labels*.  Defaults to ``-100``.
        is_encoder_decoder: If ``False`` (the default), the function applies
            the standard causal shift (drop the last logit, drop the first
            label) before computing the summary.

    Returns:
        Float array of shape ``(batch,)`` where each entry is the mean logit
        sum across the loss-bearing positions of that example.

    Raises:
        ValueError: If the batch and sequence-length dimensions of *logits*
            (ignoring the vocab axis) do not match the shape of *labels*.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:]
        logits = logits[:, :-1, :]

    loss_mask = labels != label_pad_token_id
    token_logit_sums = jnp.sum(logits.astype(jnp.float32), axis=-1)
    token_counts = jnp.maximum(loss_mask.astype(jnp.float32).sum(-1), 1.0)
    return jnp.where(loss_mask, token_logit_sums, 0.0).sum(-1) / token_counts


def concatenated_inputs(
    batch: dict[str, list | Array],
    is_encoder_decoder: bool = False,
) -> dict[str, Array]:
    """Concatenate chosen and rejected entries of a preference batch.

    Pairs every ``chosen<suffix>`` array with the matching
    ``rejected<suffix>`` array and emits a single ``concatenated<suffix>``
    entry that stacks them along the batch axis. For encoder-decoder
    models the prompt inputs and attention masks are additionally
    repeated twice along the batch axis so each branch sees its prompt.

    Args:
        batch (dict[str, list | Array]): Mapping carrying the preference
            batch. Expected keys include those prefixed by ``"chosen"`` and
            ``"rejected"`` (and, for encoder-decoder models,
            ``"prompt_input_ids"`` and ``"prompt_attention_mask"``).
        is_encoder_decoder (bool): Whether the model is encoder-decoder.
            Defaults to ``False``.

    Returns:
        dict[str, Array]: Mapping whose keys are prefixed with
        ``"concatenated"``, containing the stacked chosen+rejected
        tensors.
    """
    concatenated_batch = {}

    # Process chosen examples.
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], jax.Array):
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = batch[k]
    # Process rejected examples and concatenate with chosen examples.
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], jax.Array):
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = jnp.concatenate(
                (concatenated_batch[concatenated_key], batch[k]), axis=0
            )

    # For encoder-decoder models, duplicate prompt inputs along the BATCH axis to line up with the
    # chosen/rejected completions (which are concatenated on axis 0 above). ``.repeat(2, 1)`` wrongly doubled
    # the sequence axis -> (batch, 2*seq); use a batch-axis concatenate to get (2*batch, seq), matching DPO.
    if is_encoder_decoder:
        concatenated_batch["concatenated_input_ids"] = jnp.concatenate(
            [batch["prompt_input_ids"], batch["prompt_input_ids"]], axis=0
        )
        concatenated_batch["concatenated_attention_mask"] = jnp.concatenate(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], axis=0
        )

    return concatenated_batch


def odds_ratio_loss(
    beta: float,
    policy_chosen_logps: Array,
    policy_rejected_logps: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Compute the ORPO odds-ratio loss and accompanying reward statistics.

    The log-odds quantity is

    ``log_odds = (logp_chosen - logp_rejected)
                  - (log1mexp(logp_chosen) - log1mexp(logp_rejected))``

    where ``log1mexp(x) = log(1 - exp(x))``. The base loss is
    ``beta * log_sigmoid(log_odds)`` (negated by the caller before adding
    the NLL term). Detached ``beta * logp_*`` quantities are returned as
    per-sample implicit rewards for diagnostic reporting.

    Args:
        beta (float): Scaling hyperparameter applied to both the loss term
            and the implicit rewards.
        policy_chosen_logps (Array): Per-sequence log probabilities of the
            chosen branch.
        policy_rejected_logps (Array): Per-sequence log probabilities of
            the rejected branch.

    Returns:
        tuple[Array, Array, Array, Array, Array]: ``(losses, chosen_rewards,
        rejected_rewards, mean_ratio, mean_log_odds)`` where ``losses`` is
        per-example, the reward tensors are detached from the gradient,
        ``mean_ratio`` is the mean ``log_sigmoid`` summary, and
        ``mean_log_odds`` is the mean ``log_odds`` value.
    """
    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        _log1mexp(policy_chosen_logps) - _log1mexp(policy_rejected_logps)
    )
    ratio = jax.nn.log_sigmoid(log_odds)
    losses = beta * ratio

    chosen_rewards = beta * jax.lax.stop_gradient(policy_chosen_logps)
    rejected_rewards = beta * jax.lax.stop_gradient(policy_rejected_logps)

    return losses, chosen_rewards, rejected_rewards, jnp.mean(ratio), jnp.mean(log_odds)


def orpo_step(
    state: EasyDeLState,
    batch: dict,
    concatenated_forward: tp.Callable,
    beta: float = 0.1,
    learning_rate_fn: tp.Callable | None = None,
    mode: tp.Literal["train", "eval"] = "train",
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Execute a single ORPO training or evaluation step.

    Builds the chosen/rejected log probabilities via ``concatenated_forward``,
    composes the ORPO loss ``policy_nll_loss - mean(odds_ratio_loss)``, and
    either runs gradient accumulation + optimizer update (``mode == "train"``)
    or returns the diagnostic metrics only (``mode == "eval"``).

    Args:
        state (EasyDeLState): Current model state containing parameters and
            optimizer state.
        batch (dict): Input batch data carrying chosen/rejected tensors and
            (optionally) prompt arrays.
        concatenated_forward (tp.Callable): Callable returning
            ``(chosen_logps, rejected_logps, chosen_logits, rejected_logits,
            nll_loss, accuracy)`` for the merged chosen+rejected batch.
        beta (float): Scaling factor used in the odds-ratio loss. Defaults
            to ``0.1``.
        learning_rate_fn (tp.Callable | None): Optional callable mapping
            step -> learning rate used for metric reporting. Defaults to
            ``None``.
        mode (tp.Literal["train", "eval"]): Selects the train or eval
            branch. Defaults to ``"train"``.
        loss_config (LossConfig | None): Optional loss configuration
            forwarded to :func:`update_state_respectfully`.
        partition_spec (PartitionSpec | None): Sharding specification
            applied to the input batch.
        gradient_accumulation_steps (int): Number of microbatches whose
            gradients are accumulated per optimizer update (training only).
            Defaults to ``1``.
        straight_through_emulator (tp.Callable | None): Optional STE
            wrapper applied to the parameter tree inside the loss closure
            to simulate quantised forward passes during training.

    Returns:
        tuple[EasyDeLState, LossMetrics] | LossMetrics: In ``"train"`` mode
        returns ``(updated_state, metrics)``; in ``"eval"`` mode returns
        only the :class:`LossMetrics`.
    """
    scope_root = "easydel/trainer/orpo/" + ("train_step" if mode == "train" else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            batch_partition_spec=partition_spec,
            gradient_accumulation_steps=gradient_accumulation_steps if mode == "train" else 1,
        )

        # Apply sharding constraints to the batch.
        batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def calculate_loss(tree: spx.State, batch: dict):
        """Compute the ORPO scalar loss and metrics for a minibatch.

        Runs the policy forward via ``concatenated_forward``, computes the
        odds-ratio loss, and assembles the diagnostic metrics dictionary
        (rewards, log-probs, logit summaries, NLL/accuracy, log-odds).

        Args:
            tree (spx.State): Current graph state for the differentiable
                policy parameters.
            batch (dict): Minibatch with chosen/rejected tensors.

        Returns:
            tuple[Array, LossMetrics]: Scalar loss value plus a
            :class:`LossMetrics` instance containing the per-step metrics
            (prefixed with ``"eval_"`` in eval mode).
        """
        with jax.named_scope(scope_root + "/loss_fn"):
            if mode == "train" and straight_through_emulator is not None:
                with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                    tree = straight_through_emulator(tree)
            with jax.named_scope(scope_root + "/loss_fn/policy_forward"):
                (
                    policy_chosen_logps,
                    policy_rejected_logps,
                    policy_chosen_logits,
                    policy_rejected_logits,
                    policy_nll_loss,
                    policy_accuracy,
                ) = concatenated_forward(state.merge_to_state(tree), batch)

            with jax.named_scope(scope_root + "/loss_fn/compute_orpo_loss"):
                (
                    losses,
                    chosen_rewards,
                    rejected_rewards,
                    log_odds_ratio,
                    log_odds_chosen,
                ) = odds_ratio_loss(beta, policy_chosen_logps, policy_rejected_logps)

                loss = policy_nll_loss - losses.mean()

            with jax.named_scope(scope_root + "/loss_fn/build_metrics"):
                reward_accuracies = (chosen_rewards > rejected_rewards).astype("float32")
                metrics = {
                    "rewards/chosen": chosen_rewards.mean(),
                    "rewards/rejected": rejected_rewards.mean(),
                    "rewards/accuracies": reward_accuracies.mean(),
                    "rewards/margins": (chosen_rewards - rejected_rewards).mean(),
                    "logps/rejected": policy_rejected_logps.mean(),
                    "logps/chosen": policy_chosen_logps.mean(),
                    "logits/rejected": policy_rejected_logits.mean(),
                    "logits/chosen": policy_chosen_logits.mean(),
                    "nll_loss": policy_nll_loss.mean(),
                    "nll_accuracy": policy_accuracy.mean(),
                    "log_odds_ratio": log_odds_ratio,
                    "log_odds_chosen": log_odds_chosen,
                }

                if mode == "eval":
                    # Prefix metric names with 'eval_' in evaluation mode.
                    metrics = {f"eval_{k}": v for k, v in metrics.items()}

            return loss, LossMetrics(
                loss=loss,
                other_metrics=metrics,
            )

    if mode == "train":
        with jax.named_scope(scope_root + "/grad_and_minibatch"):
            # Compute gradients and metrics via minibatch processing.
            gradients, metrics = minibatch_call(
                state=state,
                batch=batch,
                minibatch_size=minibatch_size,
                grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
            )
        with jax.named_scope(scope_root + "/update_state"):
            # Update model state with computed gradients.
            state = update_state_respectfully(
                state=state,
                gradients=gradients,
                loss_config=loss_config,
                metrics=metrics,
            )
            # Update metrics with learning rate and step information.
            metrics = update_metrics(
                metrics=metrics,
                learning_rate_fn=learning_rate_fn,
                step=state.step,
                gradients=gradients,
            )
        return state, metrics
    else:
        with jax.named_scope(scope_root + "/eval_call"):
            # In evaluation mode, compute loss metrics without updating the state.
            _, metrics = calculate_loss(state.graphstate, batch)
        return metrics


def orpo_training_step(
    state: EasyDeLState,
    batch: dict,
    concatenated_forward: tp.Callable,
    beta: float = 0.1,
    learning_rate_fn: tp.Callable | None = None,
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Execute one ORPO training step (forward, backward, optimizer update).

    Thin wrapper around :func:`orpo_step` with ``mode="train"``. Suitable as
    the JIT entry point compiled by the trainer.

    Args:
        state (EasyDeLState): Current model state (parameters, optimizer state).
        batch (dict): Mapping containing chosen/rejected token tensors.
        concatenated_forward (tp.Callable): Forward function returning the
            tuple of (chosen_logps, rejected_logps, chosen_logits,
            rejected_logits, nll_loss, accuracy).
        beta (float): Scaling factor for the odds ratio loss term.
        learning_rate_fn (tp.Callable | None): Optional callable mapping
            step -> learning rate, used for metric reporting.
        loss_config (LossConfig | None): Optional loss configuration override.
        partition_spec (PartitionSpec | None): Sharding spec to apply to the
            input batch.
        gradient_accumulation_steps (int): Number of microbatches whose
            gradients are accumulated before an optimizer update.
        straight_through_emulator (tp.Callable | None): Optional STE callable
            wrapping the parameter tree to emulate quantized forward passes.

    Returns:
        tuple[EasyDeLState, LossMetrics]: Updated state and computed metrics.
    """
    return tp.cast(
        tuple[EasyDeLState, LossMetrics],
        orpo_step(
            state=state,
            batch=batch,
            concatenated_forward=concatenated_forward,
            beta=beta,
            learning_rate_fn=learning_rate_fn,
            mode="train",
            loss_config=loss_config,
            partition_spec=partition_spec,
            gradient_accumulation_steps=gradient_accumulation_steps,
            straight_through_emulator=straight_through_emulator,
        ),
    )


def _orpo_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build the cache key identifying a scheduled ORPO loss specialization.

    Args:
        call: A scheduled call descriptor produced by the training utilities,
            holding the bound static arguments for the loss function.

    Returns:
        tuple[tp.Any, ...]: A hashable tuple uniquely identifying the
        ``(beta, partition_spec, concatenated_forward, straight_through_emulator)``
        specialization.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=("beta", "partition_spec"),
        object_fields=("concatenated_forward", "straight_through_emulator"),
    )


def _make_orpo_scheduled_loss(call):
    """Build a scalar loss closure for the scheduled-loss adapter.

    Args:
        call: Scheduled call descriptor carrying ``concatenated_forward``,
            ``beta``, and ``partition_spec`` entries.

    Returns:
        tp.Callable: A function ``(tree, batch) -> Array`` returning the
        scalar ORPO objective ``policy_nll_loss - mean(odds_ratio_loss)``.
    """
    concatenated_forward = call.get("concatenated_forward")
    beta = call.get("beta")
    partition_spec = call.get("partition_spec")

    def scheduled_loss(tree: spx.State, batch: dict):
        """Compute the ORPO scalar loss for the scheduled-loss adapter.

        Args:
            tree (spx.State): Current model parameter tree.
            batch (dict): Input minibatch with chosen/rejected entries.

        Returns:
            Array: Scalar loss value.
        """
        with jax.named_scope("easydel/trainer/orpo/scheduled_loss"):
            with jax.named_scope("easydel/trainer/orpo/scheduled_loss/bind_module"):
                module = bind_scheduled_module(call, tree)
                call_batch = constrain_scheduled_batch(module, batch, partition_spec)
                _terminal_rank = _scheduled_terminal_stage_rank(module, call.schedule)
            with jax.named_scope("easydel/trainer/orpo/scheduled_loss/policy_forward"):
                (
                    policy_chosen_logps,
                    policy_rejected_logps,
                    _policy_chosen_logits,
                    _policy_rejected_logits,
                    policy_nll_loss,
                    _policy_accuracy,
                ) = concatenated_forward(module, call_batch, vocab_shard_stage=_terminal_rank)
            with jax.named_scope("easydel/trainer/orpo/scheduled_loss/compute_orpo_loss"):
                losses, *_ = odds_ratio_loss(beta, policy_chosen_logps, policy_rejected_logps)
                return policy_nll_loss - losses.mean()

    return scheduled_loss


register_scheduled_loss_adapter(
    step_fn=orpo_training_step,
    adapter=ScheduledLossAdapter(
        name="orpo",
        make_loss=_make_orpo_scheduled_loss,
        make_cache_key=_orpo_scheduled_loss_cache_key,
    ),
)
