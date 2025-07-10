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
import typing as tp

import chex
import jax
from eformer.escale import with_sharding_constraint
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics, dynamic_cross_entropy_loss

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully


def concatenated_forward(
    state: EasyDeLState,
    batch: tp.Mapping[str, list | chex.Array],
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: tp.Any,
    max_length: int | None = None,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Computes log-probabilities and logits for both chosen and rejected examples by concatenating
    the inputs and performing a forward pass through the model.

    The function processes the batch by concatenating the chosen and rejected examples. It then
    calls the model (stored in `state`) to obtain the logits, computes the negative log-likelihood
    loss for the chosen examples using a dynamic cross entropy loss function, and splits the logits
    and log-probabilities into those corresponding to the chosen and rejected examples.

    Args:
        state (EasyDeLState): The current state of the model containing parameters and the model itself.
        batch (tp.Mapping[str, tp.Union[tp.List, chex.Array]]): A dictionary containing input arrays for
            chosen and rejected examples as well as other necessary inputs.
        is_encoder_decoder (bool): Flag indicating whether the model is an encoder-decoder.
        label_pad_token_id (int): The token ID used to mark padding positions in the labels.
        padding_value (Any): The value used for padding. Must not be None.
        max_length (int | None, optional): Maximum length for the inputs (if applicable). Defaults to None.

    Returns:
        tp.Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
            A tuple containing:
                - chosen_log_probs: Log probabilities for the chosen examples.
                - rejected_log_probs: Log probabilities for the rejected examples.
                - chosen_logits: Logits for the chosen examples.
                - rejected_logits: Logits for the rejected examples.
                - chosen_nll_loss: Negative log-likelihood loss for the chosen examples.
                - chosen_accuracy: Accuracy metric computed on the chosen examples.
    """
    assert padding_value is not None, "`padding_value` can not be set as `None` it must be an integer."

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

    # Forward pass through the model.
    all_logits = state.model(
        input_ids=concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
        **model_kwargs,
    ).logits

    def cross_entropy_loss(logits, labels):
        """
        Computes the cross entropy loss and accuracy between the logits and labels.

        For non encoder-decoder models, the logits and labels are shifted appropriately.

        Args:
            logits (chex.Array): Logits produced by the model.
            labels (chex.Array): Ground truth labels.

        Returns:
            tp.Tuple[chex.Array, chex.Array]: The computed loss and accuracy.
        """
        if not is_encoder_decoder:
            logits = logits[..., :-1, :]
            labels = labels[..., 1:]
        loss, accuracy = dynamic_cross_entropy_loss(
            logits,
            labels,
            ignore_index=label_pad_token_id,
        )
        return loss, accuracy

    # Set labels for computing loss.
    if is_encoder_decoder:
        labels = concatenated_batch["concatenated_labels"]
    else:
        labels = concatenated_batch["concatenated_input_ids"]
        attention_mask = concatenated_batch["concatenated_attention_mask"]
        labels = jnp.where(attention_mask == 1, labels, label_pad_token_id)

    # Compute negative log likelihood loss and accuracy for the chosen examples.
    chosen_nll_loss, chosen_accuracy = cross_entropy_loss(
        all_logits[:len_chosen],
        labels[:len_chosen],
    )

    # Compute log probabilities for the entire batch.
    all_log_probs = get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        average_log_prob=True,
        is_encoder_decoder=is_encoder_decoder,
        label_pad_token_id=label_pad_token_id,
    )

    # Split log probabilities and logits into chosen and rejected.
    chosen_log_probs = all_log_probs[:len_chosen]
    rejected_log_probs = all_log_probs[len_chosen:]
    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]
    return (
        chosen_log_probs,
        rejected_log_probs,
        chosen_logits,
        rejected_logits,
        chosen_nll_loss,
        chosen_accuracy,
    )


def get_batch_logps(
    logits: chex.Array,
    labels: chex.Array,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> chex.Array:
    """
    Computes the log probabilities for a batch of sequences given the model logits and labels.

    The function applies a log-softmax over the logits and extracts the log probability of each
    token corresponding to the label. It also masks out the padding tokens using `label_pad_token_id`.

    Args:
        logits (chex.Array): The logits output by the model with shape (..., sequence_length, vocab_size).
        labels (chex.Array): The ground truth labels with shape matching logits except for the vocabulary dimension.
        average_log_prob (bool, optional): If True, returns the average log probability per sequence.
            Otherwise, returns the sum of log probabilities per sequence. Defaults to False.
        label_pad_token_id (int, optional): The token ID used for padding in the labels. Defaults to -100.
        is_encoder_decoder (bool, optional): Flag indicating whether the model is an encoder-decoder.
            Defaults to False.

    Returns:
        chex.Array: An array of log probabilities for each sequence in the batch.
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
    # Compute the log softmax along the vocabulary dimension.
    lsmax = jax.nn.log_softmax(logits, axis=-1)
    # Extract log probabilities for the corresponding label tokens.
    per_token_logps = jnp.take_along_axis(lsmax, axis=2, indices=labels).squeeze(2)

    # Return averaged or summed log probabilities based on the flag.
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(
    batch: dict[str, list | chex.Array],
    is_encoder_decoder: bool = False,
) -> dict[str, chex.Array]:
    """
    Concatenates chosen and rejected examples from the batch into unified arrays.

    For each key in the batch that starts with "chosen" or "rejected", the function creates a new key
    starting with "concatenated" and combines the corresponding arrays. In the case of an encoder-decoder
    model, the prompt inputs and attention masks are also repeated accordingly.

    Args:
        batch (tp.Dict[str, tp.Union[tp.List, chex.Array]]): A dictionary containing the batch of data.
            Expected keys include those starting with "chosen", "rejected", "prompt_input_ids", and
            "prompt_attention_mask".
        is_encoder_decoder (bool, optional): Indicates whether the model is encoder-decoder.
            Defaults to False.

    Returns:
        tp.Dict[str, chex.Array]: A dictionary containing concatenated arrays with keys prefixed with
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
    policy_chosen_logps: chex.Array,
    policy_rejected_logps: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Computes the odds ratio loss used for training based on the log probabilities of chosen and rejected examples.

    The odds ratio is calculated as the difference between the chosen and rejected log probabilities
    (with a correction term for numerical stability). The sigmoid of this log odds is then taken, and the
    log of this sigmoid forms the basis of the loss. The function also computes reward values for both
    chosen and rejected examples, as well as summary statistics.

    Args:
        beta (float): A scaling hyperparameter applied to the loss and rewards.
        policy_chosen_logps (chex.Array): Log probabilities for the chosen examples.
        policy_rejected_logps (chex.Array): Log probabilities for the rejected examples.

    Returns:
        tp.Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
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
    batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        batch_partition_spec=partition_spec,
        gradient_accumulation_steps=gradient_accumulation_steps if mode == "train" else 1,
    )

    # Apply sharding constraints to the batch.
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def calculate_loss(tree: nn.GraphState, batch: dict):
        """
        Computes the loss and metrics for a given minibatch.

        This inner function performs a forward pass using the concatenated_forward function,
        computes the odds ratio loss, and aggregates various metrics.

        Args:
            tree (nn.GraphState): The current state of the model graph.
            batch (tp.Dict): The input batch data.

        Returns:
            tp.Tuple[chex.Array, LossMetrics]: The computed loss and a LossMetrics object containing
            additional metrics.
        """
        (
            mean_chosen_logits,
            mean_rejected_logits,
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
        ) = odds_ratio_loss(beta, mean_chosen_logits, mean_rejected_logits)

        loss = policy_nll_loss - losses.mean()

        reward_accuracies = (chosen_rewards > rejected_rewards).astype("float32")
        metrics = {
            "rewards/chosen": chosen_rewards.mean(),
            "rewards/rejected": rejected_rewards.mean(),
            "rewards/accuracies": reward_accuracies.mean(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean(),
            "logps/rejected": mean_rejected_logits.mean(),
            "logps/chosen": mean_chosen_logits.mean(),
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
