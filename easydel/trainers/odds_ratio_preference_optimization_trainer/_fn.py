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
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics, dynamic_cross_entropy_loss
from easydel.trainers._logprob_utils import (
    compute_sequence_scores_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)

from ..training_utils import (
    ScheduledLossAdapter,
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


def concatenated_forward(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, list | Array],
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: tp.Any,
    max_length: int | None = None,
    logprob_vocab_chunk_size: int | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    Computes log-probabilities and logits for both chosen and rejected examples by concatenating
    the inputs and performing a forward pass through the model.

    The function processes the batch by concatenating the chosen and rejected examples. It then
    calls the model (stored in `state`) to obtain the logits, computes the negative log-likelihood
    loss for the chosen examples using a dynamic cross entropy loss function, and splits the logits
    and log-probabilities into those corresponding to the chosen and rejected examples.

    Args:
        state (EasyDeLState): The current state of the model containing parameters and the model itself.
        batch (collections.abc.Mapping[str, tp.Union[tp.List, Array]]): A dictionary containing input arrays for
            chosen and rejected examples as well as other necessary inputs.
        is_encoder_decoder (bool): Flag indicating whether the model is an encoder-decoder.
        label_pad_token_id (int): The token ID used to mark padding positions in the labels.
        padding_value (Any): The value used for padding. Must not be None.
        max_length (int | None, optional): Maximum length for the inputs (if applicable). Defaults to None.

    Returns:
        tp.Tuple[Array, Array, Array, Array, Array, Array]:
            A tuple containing:
                - chosen_log_probs: Log probabilities for the chosen examples.
                - rejected_log_probs: Log probabilities for the rejected examples.
                - chosen_logits: Per-example mean logit summaries for the chosen examples.
                - rejected_logits: Per-example mean logit summaries for the rejected examples.
                - chosen_nll_loss: Negative log-likelihood loss for the chosen examples.
                - chosen_accuracy: Accuracy metric computed on the chosen examples.
    """
    if padding_value is None:
        raise ValueError("`padding_value` can not be set as `None` it must be an integer.")
    model = state.model if isinstance(state, EasyDeLState) else state

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
        """
        Computes the cross entropy loss and accuracy between the logits and labels.

        For non encoder-decoder models, the logits and labels are shifted appropriately.

        Args:
            logits (Array): Logits produced by the model.
            labels (Array): Ground truth labels.

        Returns:
            tp.Tuple[Array, Array]: The computed loss and accuracy.
        """
        if not is_encoder_decoder:
            logits = logits[..., :-1, :]
            labels = labels[..., 1:]
        loss, _ = dynamic_cross_entropy_loss(
            logits,
            labels,
            ignore_index=label_pad_token_id,
        )
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
    """
    Computes the log probabilities for a batch of sequences given the model logits and labels.

    The function applies a log-softmax over the logits and extracts the log probability of each
    token corresponding to the label. It also masks out the padding tokens using `label_pad_token_id`.

    Args:
        logits (Array): The logits output by the model with shape (..., sequence_length, vocab_size).
        labels (Array): The ground truth labels with shape matching logits except for the vocabulary dimension.
        average_log_prob (bool, optional): If True, returns the average log probability per sequence.
            Otherwise, returns the sum of log probabilities per sequence. Defaults to False.
        label_pad_token_id (int, optional): The token ID used for padding in the labels. Defaults to -100.
        is_encoder_decoder (bool, optional): Flag indicating whether the model is an encoder-decoder.
            Defaults to False.

    Returns:
        Array: An array of log probabilities for each sequence in the batch.
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
    """
    Concatenates chosen and rejected examples from the batch into unified arrays.

    For each key in the batch that starts with "chosen" or "rejected", the function creates a new key
    starting with "concatenated" and combines the corresponding arrays. In the case of an encoder-decoder
    model, the prompt inputs and attention masks are also repeated accordingly.

    Args:
        batch (tp.Dict[str, tp.Union[tp.List, Array]]): A dictionary containing the batch of data.
            Expected keys include those starting with "chosen", "rejected", "prompt_input_ids", and
            "prompt_attention_mask".
        is_encoder_decoder (bool, optional): Indicates whether the model is encoder-decoder.
            Defaults to False.

    Returns:
        tp.Dict[str, Array]: A dictionary containing concatenated arrays with keys prefixed with
            "concatenated".
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

    # For encoder-decoder models, repeat prompt inputs and attention masks.
    if is_encoder_decoder:
        concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
        concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

    return concatenated_batch


def odds_ratio_loss(
    beta: float,
    policy_chosen_logps: Array,
    policy_rejected_logps: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """
    Computes the odds ratio loss used for training based on the log probabilities of chosen and rejected examples.

    The odds ratio is calculated as the difference between the chosen and rejected log probabilities
    (with a correction term for numerical stability). The sigmoid of this log odds is then taken, and the
    log of this sigmoid forms the basis of the loss. The function also computes reward values for both
    chosen and rejected examples, as well as summary statistics.

    Args:
        beta (float): A scaling hyperparameter applied to the loss and rewards.
        policy_chosen_logps (Array): Log probabilities for the chosen examples.
        policy_rejected_logps (Array): Log probabilities for the rejected examples.

    Returns:
        tp.Tuple[Array, Array, Array, Array, Array]:
            A tuple containing:
                - losses: The computed odds ratio loss.
                - chosen_rewards: Rewards computed from the chosen log probabilities (detached).
                - rejected_rewards: Rewards computed from the rejected log probabilities (detached).
                - mean_ratio: The mean of the log sigmoid ratio.
                - mean_log_odds: The mean log odds difference.
    """
    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        jnp.log1p(-jnp.exp(policy_chosen_logps)) - jnp.log1p(-jnp.exp(policy_rejected_logps))
    )
    sig_ratio = jax.nn.sigmoid(log_odds)
    ratio = jnp.log(sig_ratio)
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
    """
    Performs a single training or evaluation step for the ORPO method.

    The function handles both forward and backward passes (when in training mode) and computes
    the loss metrics. It supports minibatch processing and gradient accumulation. In training mode,
    the model state is updated based on the computed gradients, while in evaluation mode, only loss
    metrics are returned.

    Args:
        state (EasyDeLState): The current model state containing parameters, optimizer state, etc.
        batch (dict): The input batch data.
        concatenated_forward (tp.Callable): A callable that performs the forward pass and returns
            logits and loss values for chosen and rejected examples.
        beta (float, optional): Scaling factor used in the odds ratio loss. Defaults to 0.1.
        learning_rate_fn (tp.Optional[tp.Callable], optional): A callable to compute the learning rate
            at the current step. Defaults to None.
        mode (tp.Literal["train", "eval"], optional): Specifies whether the step is for training or evaluation.
            Defaults to "train".
        loss_config (tp.Optional[LossConfig], optional): Configuration for the loss computation. Defaults to None.
        partition_spec (tp.Optional[PartitionSpec], optional): Specification for sharding the batch data.
            Defaults to None.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients
            (only relevant in training mode). Defaults to 1.

    Returns:
        tp.Union[tp.Tuple[EasyDeLState, LossMetrics], LossMetrics]:
            - In "train" mode: A tuple containing the updated model state and the computed loss metrics.
            - In "eval" mode: The computed loss metrics.
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        batch_partition_spec=partition_spec,
        gradient_accumulation_steps=gradient_accumulation_steps if mode == "train" else 1,
    )

    # Apply sharding constraints to the batch.
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def calculate_loss(tree: spx.State, batch: dict):
        """
        Computes the loss and metrics for a given minibatch.

        This inner function performs a forward pass using the concatenated_forward function,
        computes the odds ratio loss, and aggregates various metrics.

        Args:
            tree (spx.State): The current state of the model graph.
            batch (tp.Dict): The input batch data.

        Returns:
            tp.Tuple[Array, LossMetrics]: The computed loss and a LossMetrics object containing
            additional metrics.
        """
        if mode == "train" and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
            policy_accuracy,
        ) = concatenated_forward(state.merge_to_state(tree), batch)

        (
            losses,
            chosen_rewards,
            rejected_rewards,
            log_odds_ratio,
            log_odds_chosen,
        ) = odds_ratio_loss(beta, policy_chosen_logps, policy_rejected_logps)

        loss = policy_nll_loss - losses.mean()

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
        # Compute gradients and metrics via minibatch processing.
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
        )
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
    return orpo_step(
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
        module = bind_scheduled_module(call, tree)
        call_batch = constrain_scheduled_batch(module, batch, partition_spec)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            _policy_chosen_logits,
            _policy_rejected_logits,
            policy_nll_loss,
            _policy_accuracy,
        ) = concatenated_forward(module, call_batch)
        losses, *_ = odds_ratio_loss(beta, policy_chosen_logps, policy_rejected_logps)
        return policy_nll_loss - losses.mean()

    return scheduled_loss


register_scheduled_loss_adapter(
    orpo_training_step,
    ScheduledLossAdapter(
        name="orpo",
        make_loss=_make_orpo_scheduled_loss,
        make_cache_key=_orpo_scheduled_loss_cache_key,
    ),
)
