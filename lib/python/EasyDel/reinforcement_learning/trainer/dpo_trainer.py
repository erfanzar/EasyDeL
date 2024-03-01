import copy
import os
import sys
import time
import typing
import warnings
from abc import ABC
from collections import defaultdict

import IPython
import chex
import flax.core
import jax
import termcolor
import wandb
from fjformer import match_partition_rules, make_shard_and_gather_fns
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.collectors import DPODataCollatorWithPadding
from typing import Optional, Literal, Dict, Union, Any, Tuple, List, Callable, Mapping
from .utils import pad_to_length
from jax.experimental.pjit import pjit
from datasets import Dataset
from jax import numpy as jnp

from ...etils.etils import get_logger
from ...smi import get_capacity_matrix, initialise_tracking, get_mem
from ...trainer.causal_language_model_trainer import TrainerOutput
from ...trainer.training_configurations import TrainArguments
from ...trainer.base_trainer import (
    BaseTrainer,
    TrainerConfigureFunctionFuncOutput,
    TrainerConfigureDataloaderFuncOutput,
    TrainerConfigureModelFuncOutput
)
from ...etils import EasyDelState, EasyDelTimerError
from transformers import PreTrainedTokenizerBase
from .partitioner_config import PartitionerConfig
from jax.sharding import PartitionSpec

from ...utils import Timers
from flax.struct import dataclass
from contextlib import contextmanager


@contextmanager
def leave_alone_context_manager():
    # Perform setup actions (none in this case)
    yield


logger = get_logger(__name__)


@dataclass
class DPOStepOut:
    loss: chex.Array
    chosen_rewards: chex.Array
    rejected_rewards: chex.Array


def create_concatenated_forward(
        is_encoder_decoder,
        label_pad_token_id,
        padding_value,
        truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
        fixed_max_length: int | None = None
):
    """
    The create_concatenated_forward function is a helper function that creates a forward pass function for the
    model. The forward pass function takes in an apply_fn, which is the model's apply_fn, and runs it on concatenated
    inputs. It returns chosen log probs, rejected log probs, chosen logits and rejected logits.

    :param is_encoder_decoder: Determine whether the model is an encoder-decoder model or not
    :param label_pad_token_id: Pad the labels to the same length
    :param padding_value: Pad the inputs to the same length
    :param truncation_mode: typing.Literal["keep_end","keep_start"]: where to pad and where to keep.
    :param fixed_max_length : int|None: by providing fixed_max_length the func will always return a fixed sequence length
    and won't use dynamic methods.
    :return: A function that takes in a apply_fn, params and a batch of inputs,
    """

    def concatenated_forward(
            apply_fn: Callable,
            params: dict | flax.core.FrozenDict,
            batch: Dict[str, Union[List, chex.Array]]

    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        The concatenated_forward function is used to compute the log-probabilities of both chosen and rejected labels.

        :param apply_fn: Callable: Pass in the model function
        :param params: dict | flax.core.FrozenDict: Pass the model parameters to the function
        :param batch: Dict[str, Union[List, chex.Array]] : Pass the batch of data to the concatenated_forward function
        :return: The log_probs of the chosen and rejected labels, as well as their corresponding logits
        """
        assert padding_value is not None, "`padding_value` can not be set as `None` it must be an integer."
        concatenated_batch = concatenated_inputs(
            batch,
            is_encoder_decoder=is_encoder_decoder,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            fixed_max_length=fixed_max_length
        )
        len_chosen = batch["chosen_labels"].shape[0]
        concatenated_batch["concatenated_input_ids"] = concatenated_batch["concatenated_input_ids"].reshape(
            concatenated_batch["concatenated_input_ids"].shape[0], -1
        )
        concatenated_batch["concatenated_labels"] = concatenated_batch["concatenated_labels"].reshape(
            concatenated_batch["concatenated_labels"].shape[0], -1
        )
        concatenated_batch["concatenated_attention_mask"] = concatenated_batch["concatenated_attention_mask"].reshape(
            concatenated_batch["concatenated_attention_mask"].shape[0], -1
        )
        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if is_encoder_decoder
            else {}
        )
        all_logits = apply_fn(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            params=params,
            **model_kwargs,
        ).logits

        all_log_probs = get_batch_log_probs(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            is_encoder_decoder=is_encoder_decoder,
            label_pad_token_id=label_pad_token_id,
        )

        chosen_log_probs = all_log_probs[:len_chosen]
        rejected_log_probs = all_log_probs[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits

    return concatenated_forward


def get_batch_log_probs(
        logits: chex.Array,
        labels: chex.Array,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
) -> chex.Array:
    """
    The get_batch_log_probs function computes the log probability of a batch of sequences.

    :param logits: chex.Array: Compute the log_softmax of the input
    :param labels: chex.Array: Mask the logits
    :param average_log_prob: bool: Determine whether to average the log prob over the sequence length
    :param label_pad_token_id: int: Mask out the padding tokens in the labels
    :param is_encoder_decoder: bool: Indicate whether the model is an encoder-decoder model
    :param : Determine whether to average the log probability over all tokens or not
    :return: The log probability of the labels given the logits
    """

    # sudo code
    # (per_token_log_probs * loss_mask).sum(-1)
    # or
    # (per_token_log_probs * loss_mask).sum(-1) / loss_mask.sum(-1)

    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:]
        logits = logits[:, :-1, :]

    batch, seq_len, dim = logits.shape
    loss_mask = labels != label_pad_token_id
    labels = jax.lax.select(
        labels == label_pad_token_id,
        jnp.zeros(labels.shape, dtype=labels.dtype),
        labels
    )
    logits_log_s = jax.nn.log_softmax(
        logits, -1
    )
    per_token_log_probs = jnp.take_along_axis(
        logits_log_s,
        axis=2,
        indices=labels[:, :, None]
    ).reshape(batch, seq_len)

    if average_log_prob:
        log_prob = jnp.sum((per_token_log_probs * loss_mask), axis=-1) / jnp.sum(loss_mask, axis=-1)
    else:
        log_prob = jnp.sum((per_token_log_probs * loss_mask), axis=-1)

    return log_prob


def concatenated_inputs(
        batch: Dict[str, Union[List, chex.Array]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
        fixed_max_length: int | None = None
) -> Dict[str, chex.Array]:
    """
    The concatenated_inputs function takes a batch of chosen and rejected examples,
    and concatenates them together. This is useful for training the model to predict whether     an example was chosen
    by the human annotator. The function also pads all inputs to
    the same length as the longest input in that batch.

    :param batch: Dict[str,Union[List,chex.Array]]: Pass the batch of data into the function,
    Allow for the batch to be a list of arrays or just an array,
     Specify the type of data that is being passed in
    :param is_encoder_decoder: bool: Determine whether the model is an encoder-decoder model
    :param label_pad_token_id: int: Pad the labels with a value of -100
    :param padding_value: int: Pad the input_ids and attention_mask arrays to the same length
    :param truncation_mode: typing.Literal["keep_end", "keep_start"]: is left padded or not should it keep start of the
    array or the end of the array?.
    :param fixed_max_length : int|None: by providing fixed_max_length the func will always return a fixed sequence length
    and won't use dynamic methods.
    :return: A dictionary of the concatenated inputs
    """
    concatenated_batch = {}
    if fixed_max_length is None:
        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[-1], batch["rejected_labels"].shape[-1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[-1], batch["rejected_input_ids"].shape[-1])
    else:
        max_length = fixed_max_length
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], jax.Array):
            if "labels" in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith("_input_ids"):
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            else:
                raise KeyError("couldn't find pad_value [Dataset Issue]")
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], jax.Array):
            if "labels" in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith("_input_ids"):
                assert padding_value is not None, "`padding_value` can not be set as `None`"
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            else:
                raise KeyError("couldn't find pad_value [Dataset Issue]")
            concatenated_key = k.replace("rejected", "concatenated")
            v2d = lambda ar: ar.reshape(ar.shape[0], -1)
            concatenated_batch[concatenated_key] = jnp.concatenate(
                (
                    v2d(concatenated_batch[concatenated_key]),
                    pad_to_length(v2d(batch[k]), max_length, pad_value=pad_value),
                ),
                axis=0,
            )
    for k in list(concatenated_batch.keys()):
        val = concatenated_batch[k]
        if val.ndim == 3:
            # making 3d array 2d
            concatenated_batch[k] = val.reshape(val.shape[0], -1)
    if is_encoder_decoder:
        concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
        concatenated_batch["concatenated_attention_mask"] = (
            batch["prompt_attention_mask"].repeat(2, 1)
        )

    return concatenated_batch


def create_dpo_train_function(
        concatenated_forward: Callable,
        ref_state: EasyDelState = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
        reference_free: bool = False,
):
    """
    The create_dpo_train_function function is a helper function that creates the DPO training step.

    :param concatenated_forward: Callable: Define the forward pass of the model
    :param ref_state: EasyDelState: Specify the reference policy
    :param beta: float: Scale the logits
    :param label_smoothing: float: Smooth the labels
    :param loss_type:  Literal["sigmoid", "hinge", "ipo", "kto"]: Determine the loss function
    :param reference_free: bool: Indicate whether the reference policy is used or not
    :return: A function that takes in a state and a batch
    """

    def _sigmoid_dpo_loss(
            logits: chex.Array,
            policy_chosen_log_probs: chex.Array = None,  # IGNORED
            reference_chosen_log_probs: chex.Array = None,  # IGNORED
            policy_rejected_log_probs: chex.Array = None,  # IGNORED
            reference_rejected_log_probs: chex.Array = None  # IGNORED
    ):

        """
        The _sigmoid_dpo_loss function is a helper function for the sigmoid_dpo_loss
            function. It computes the loss of each example in a batch, given its logits
            and (optionally) its chosen/rejected log probabilities under both policies.

        :param logits: chex.Array: Compute the loss
        :param policy_chosen_log_probs: chex.Array: Calculate the policy loss
        :param # IGNORED
                    reference_chosen_log_probs: chex.Array: Compute the loss for the reference policy
        :param # IGNORED
                    policy_rejected_log_probs: chex.Array: Calculate the loss for the rejected samples
        :param # IGNORED
                    reference_rejected_log_probs: chex.Array: Calculate the loss of rejected samples
        :return: an array represent loss
        """
        losses = (
                -jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing)
                - jax.nn.log_sigmoid(-beta * logits) * label_smoothing
        )
        return losses

    def _hinge_dpo_loss(
            logits: chex.Array,
            policy_chosen_log_probs: chex.Array,  # IGNORED
            reference_chosen_log_probs: chex.Array,  # IGNORED
            policy_rejected_log_probs: chex.Array,  # IGNORED
            reference_rejected_log_probs: chex.Array  # IGNORED
    ):

        """
        The _hinge_dpo_loss function is a helper function that computes the loss for DPO.

        :param logits: chex.Array: Calculate the hinge loss
        :param policy_chosen_log_probs: chex.Array: Compute the policy loss
        :param # IGNORED
                    reference_chosen_log_probs: chex.Array: Compute the loss
        :param # IGNORED
                    policy_rejected_log_probs: chex.Array: Calculate the loss
        :param # IGNORED
                    reference_rejected_log_probs: chex.Array  # IGNORED: Calculate the loss
        :return: an array represent The hinge loss
        """
        return jax.relu(1 - beta * logits)

    def _ipo_dpo_loss(
            logits: chex.Array,
            policy_chosen_log_probs: chex.Array,  # IGNORED
            reference_chosen_log_probs: chex.Array,  # IGNORED
            policy_rejected_log_probs: chex.Array,  # IGNORED
            reference_rejected_log_probs: chex.Array  # IGNORED
    ):
        """
         The _ipo_dpo_loss function is a helper function that calculates the loss for
         the IPO-DPO algorithm. It takes in the logits, policy_chosen_log_probs,
         reference_chosen_log_probs, policy rejected log probs and reference rejected
         log probs as inputs. The output of this function is used to calculate the loss
         for each batch of data.

         :param logits: chex.Array: Calculate the loss
         :param policy_chosen_log_probs: chex.Array: Compute the
         :param # IGNORED
                     reference_chosen_log_probs: chex.Array: Compute the loss
         :param # IGNORED
                     policy_rejected_log_probs: chex.Array: Calculate the probability of rejecting a policy
         :param # IGNORED
                     reference_rejected_log_probs: chex.Array  # IGNORED: Make sure that the function
         :return: an array represent loss
         """
        return (logits - 1 / (2 * beta)) ** 2

    def _kto_pair_dpo_loss(
            logits: chex.Array,  # IGNORED
            policy_chosen_log_probs: chex.Array,
            reference_chosen_log_probs: chex.Array,
            policy_rejected_log_probs: chex.Array,
            reference_rejected_log_probs: chex.Array
    ):

        """
        The _kto_pair_dpo_loss function is a helper function that computes the loss for
        a single pair of trajectories. It takes in two sets of log probabilities, one from
        the policy and one from the reference distribution. The first set are the log
        probabilities for actions taken by each agent in a trajectory, while the second set
        are those for actions not taken by each agent (i.e., rejected). The function then
        computes KL divergences between these two sets of distributions and uses them to compute losses.

        :param logits: chex.Array: Calculate the log_probs
        :param # IGNORED
                    policy_chosen_log_probs: chex.Array: Calculate the chosen_kl
        :param reference_chosen_log_probs: chex.Array: Calculate the chosen_kl
        :param policy_rejected_log_probs: chex.Array: Calculate the rejected_kl variable
        :param reference_rejected_log_probs: chex.Array: Calculate the rejected_kl variable
        :return: an array represent loss
        """
        chosen_kl = jax.lax.clamp(
            min=0,
            x=jnp.mean(policy_chosen_log_probs - reference_chosen_log_probs),
            max=1e9
        )
        rejected_kl = jax.lax.clamp(
            min=0,
            x=jnp.mean(policy_rejected_log_probs - reference_rejected_log_probs),
            max=1e9
        )

        chosen_log_ratios = policy_chosen_log_probs - reference_chosen_log_probs
        rejected_log_ratios = policy_rejected_log_probs - reference_rejected_log_probs
        losses = jnp.concatenate(
            (
                1 - jax.nn.sigmoid(beta * (chosen_log_ratios - rejected_kl)),
                1 - jax.nn.sigmoid(beta * (chosen_kl - rejected_log_ratios)),
            ),
            0,
        )

        return losses

    if loss_type == "sigmoid":
        _loss_func = _sigmoid_dpo_loss
    elif loss_type == "hinge":
        _loss_func = _hinge_dpo_loss
    elif loss_type == "ipo":
        _loss_func = _ipo_dpo_loss
    elif loss_type == "kto_pair":
        _loss_func = _kto_pair_dpo_loss
    else:
        raise ValueError(f"UnKnown loss_type {loss_type}")

    def dpo_step(
            state: EasyDelState,
            batch: dict
    ) -> tuple[EasyDelState, DPOStepOut]:

        """
        The dpo_step function is the core of DPO. It takes a state and a batch,
        and returns an updated state. The update is done by calculating the loss
        for each example in the batch, then taking its gradient with respect to
        the parameters of the policy network (which are stored in `state`). This
        gradient is then used to update `state`.

        :param state: EasyDelState: Store the parameters of the model
        :param batch: dict: Pass the data to the model
        :return: A new state, which is a collection of the parameters and apply_fn
        """

        def calculate_loss(params: dict | flax.core.FrozenDict):
            (
                policy_chosen_log_probs,
                policy_rejected_log_probs,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = concatenated_forward(
                state.apply_fn,
                params,
                batch
            )

            if "reference_chosen_log_probs" in batch and "reference_rejected_log_probs" in batch:
                reference_chosen_log_probs = batch["reference_chosen_log_probs"]
                reference_rejected_log_probs = batch["reference_rejected_log_probs"]
            else:
                if ref_state is None:
                    (
                        reference_chosen_log_probs,
                        reference_rejected_log_probs,
                        _,
                        _,
                    ) = concatenated_forward(
                        state.apply_fn,
                        state.params,
                        batch
                    )
                else:
                    (
                        reference_chosen_log_probs,
                        reference_rejected_log_probs,
                        _,
                        _,
                    ) = concatenated_forward(
                        ref_state.apply_fn,
                        ref_state.params,
                        batch
                    )

            pi_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs

            if reference_free:
                ref_log_ratios = 0
            else:
                ref_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs

            logits = pi_log_ratios - ref_log_ratios
            losses = _loss_func(
                logits,
                policy_chosen_log_probs,
                reference_chosen_log_probs,
                policy_rejected_log_probs,
                reference_rejected_log_probs
            )
            chosen_rewards = (
                    beta
                    * (
                            policy_chosen_log_probs - reference_chosen_log_probs
                    )
            )
            rejected_rewards = (
                    beta
                    * (
                            policy_rejected_log_probs
                            - reference_rejected_log_probs
                    )
            )
            return losses[0], (chosen_rewards, rejected_rewards)

        grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
        (__loss, (__chosen_rewards, __rejected_rewards)), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, DPOStepOut(
            loss=__loss,
            rejected_rewards=__rejected_rewards,
            chosen_rewards=__chosen_rewards
        )

    return dpo_step


def create_dpo_eval_function(
        concatenated_forward: Callable,
        ref_state: EasyDelState = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
        reference_free: bool = False,
):
    """
    The create_dpo_eval_function function is a helper function that creates the DPO evaluating step.

    :param concatenated_forward: Callable: Define the forward pass of the model
    :param ref_state: EasyDelState: Specify the reference policy
    :param beta: float: Scale the logits
    :param label_smoothing: float: Smooth the labels
    :param loss_type:  Literal["sigmoid", "hinge", "ipo", "kto"]: Determine the loss function
    :param reference_free: bool: Indicate whether the reference policy is used or not
    :return: A function that takes in a state and a batch
    """

    def _sigmoid_dpo_loss(
            logits: chex.Array,
            policy_chosen_log_probs: chex.Array = None,  # IGNORED
            reference_chosen_log_probs: chex.Array = None,  # IGNORED
            policy_rejected_log_probs: chex.Array = None,  # IGNORED
            reference_rejected_log_probs: chex.Array = None  # IGNORED
    ):

        """
        The _sigmoid_dpo_loss function is a helper function for the sigmoid_dpo_loss
            function. It computes the loss of each example in a batch, given its logits
            and (optionally) its chosen/rejected log probabilities under both policies.

        :param logits: chex.Array: Compute the loss
        :param policy_chosen_log_probs: chex.Array: Calculate the policy loss
        :param # IGNORED
                    reference_chosen_log_probs: chex.Array: Compute the loss for the reference policy
        :param # IGNORED
                    policy_rejected_log_probs: chex.Array: Calculate the loss for the rejected samples
        :param # IGNORED
                    reference_rejected_log_probs: chex.Array: Calculate the loss of rejected samples
        :return: an array represent loss
        """
        losses = (
                -jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing)
                - jax.nn.log_sigmoid(-beta * logits) * label_smoothing
        )
        return losses

    def _hinge_dpo_loss(
            logits: chex.Array,
            policy_chosen_log_probs: chex.Array,  # IGNORED
            reference_chosen_log_probs: chex.Array,  # IGNORED
            policy_rejected_log_probs: chex.Array,  # IGNORED
            reference_rejected_log_probs: chex.Array  # IGNORED
    ):

        """
        The _hinge_dpo_loss function is a helper function that computes the loss for DPO.

        :param logits: chex.Array: Calculate the hinge loss
        :param policy_chosen_log_probs: chex.Array: Compute the policy loss
        :param # IGNORED
                    reference_chosen_log_probs: chex.Array: Compute the loss
        :param # IGNORED
                    policy_rejected_log_probs: chex.Array: Calculate the loss
        :param # IGNORED
                    reference_rejected_log_probs: chex.Array  # IGNORED: Calculate the loss
        :return: an array represent The hinge loss
        """
        return jax.relu(1 - beta * logits)

    def _ipo_dpo_loss(
            logits: chex.Array,
            policy_chosen_log_probs: chex.Array,  # IGNORED
            reference_chosen_log_probs: chex.Array,  # IGNORED
            policy_rejected_log_probs: chex.Array,  # IGNORED
            reference_rejected_log_probs: chex.Array  # IGNORED
    ):
        """
         The _ipo_dpo_loss function is a helper function that calculates the loss for
         the IPO-DPO algorithm. It takes in the logits, policy_chosen_log_probs,
         reference_chosen_log_probs, policy rejected log probs and reference rejected
         log probs as inputs. The output of this function is used to calculate the loss
         for each batch of data.

         :param logits: chex.Array: Calculate the loss
         :param policy_chosen_log_probs: chex.Array: Compute the
         :param # IGNORED
                     reference_chosen_log_probs: chex.Array: Compute the loss
         :param # IGNORED
                     policy_rejected_log_probs: chex.Array: Calculate the probability of rejecting a policy
         :param # IGNORED
                     reference_rejected_log_probs: chex.Array  # IGNORED: Make sure that the function
         :return: an array represent loss
         """
        return (logits - 1 / (2 * beta)) ** 2

    def _kto_pair_dpo_loss(
            logits: chex.Array,  # IGNORED
            policy_chosen_log_probs: chex.Array,
            reference_chosen_log_probs: chex.Array,
            policy_rejected_log_probs: chex.Array,
            reference_rejected_log_probs: chex.Array
    ):

        """
        The _kto_pair_dpo_loss function is a helper function that computes the loss for
        a single pair of trajectories. It takes in two sets of log probabilities, one from
        the policy and one from the reference distribution. The first set are the log
        probabilities for actions taken by each agent in a trajectory, while the second set
        are those for actions not taken by each agent (i.e., rejected). The function then
        computes KL divergences between these two sets of distributions and uses them to compute losses.

        :param logits: chex.Array: Calculate the log_probs
        :param # IGNORED
                    policy_chosen_log_probs: chex.Array: Calculate the chosen_kl
        :param reference_chosen_log_probs: chex.Array: Calculate the chosen_kl
        :param policy_rejected_log_probs: chex.Array: Calculate the rejected_kl variable
        :param reference_rejected_log_probs: chex.Array: Calculate the rejected_kl variable
        :return: an array represent loss
        """
        chosen_kl = jax.lax.clamp(
            min=0,
            x=jnp.mean(policy_chosen_log_probs - reference_chosen_log_probs),
            max=1e9
        )
        rejected_kl = jax.lax.clamp(
            min=0,
            x=jnp.mean(policy_rejected_log_probs - reference_rejected_log_probs),
            max=1e9
        )

        chosen_log_ratios = policy_chosen_log_probs - reference_chosen_log_probs
        rejected_log_ratios = policy_rejected_log_probs - reference_rejected_log_probs
        losses = jnp.concatenate(
            (
                1 - jax.nn.sigmoid(beta * (chosen_log_ratios - rejected_kl)),
                1 - jax.nn.sigmoid(beta * (chosen_kl - rejected_log_ratios)),
            ),
            0,
        )

        return losses

    if loss_type == "sigmoid":
        _loss_func = _sigmoid_dpo_loss
    elif loss_type == "hinge":
        _loss_func = _hinge_dpo_loss
    elif loss_type == "ipo":
        _loss_func = _ipo_dpo_loss
    elif loss_type == "kto_pair":
        _loss_func = _kto_pair_dpo_loss
    else:
        raise ValueError(f"UnKnown loss_type {loss_type}")

    def dpo_step(
            state: EasyDelState,
            batch: dict
    ) -> DPOStepOut:

        """
        The dpo_step function is the core of DPO. It takes a state and a batch,
        and returns an updated state. The update is done by calculating the loss
        for each example in the batch, then taking its gradient with respect to
        the parameters of the policy network (which are stored in `state`). This
        gradient is then used to update `state`.

        :param state: EasyDelState: Store the parameters of the model
        :param batch: dict: Pass the data to the model
        :return: A `DPOStepOut` class
        """

        def calculate_loss(params: dict | flax.core.FrozenDict):
            (
                policy_chosen_log_probs,
                policy_rejected_log_probs,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = concatenated_forward(
                state.apply_fn,
                params,
                batch
            )

            if "reference_chosen_log_probs" in batch and "reference_rejected_log_probs" in batch:
                reference_chosen_log_probs = batch["reference_chosen_log_probs"]
                reference_rejected_log_probs = batch["reference_rejected_log_probs"]
            else:
                if ref_state is None:
                    (
                        reference_chosen_log_probs,
                        reference_rejected_log_probs,
                        _,
                        _,
                    ) = concatenated_forward(
                        state.apply_fn,
                        state.params,
                        batch
                    )
                else:
                    (
                        reference_chosen_log_probs,
                        reference_rejected_log_probs,
                        _,
                        _,
                    ) = concatenated_forward(
                        ref_state.apply_fn,
                        ref_state.params,
                        batch
                    )

            pi_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs

            if reference_free:
                ref_log_ratios = 0
            else:
                ref_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs

            logits = pi_log_ratios - ref_log_ratios
            losses = _loss_func(
                logits,
                policy_chosen_log_probs,
                reference_chosen_log_probs,
                policy_rejected_log_probs,
                reference_rejected_log_probs
            )
            chosen_rewards = (
                    beta
                    * (
                            policy_chosen_log_probs - reference_chosen_log_probs
                    )
            )
            rejected_rewards = (
                    beta
                    * (
                            policy_rejected_log_probs
                            - reference_rejected_log_probs
                    )
            )
            return losses[0], (chosen_rewards, rejected_rewards)

        __loss, (__chosen_rewards, __rejected_rewards) = calculate_loss(state.params)

        return DPOStepOut(
            loss=__loss,
            rejected_rewards=__rejected_rewards,
            chosen_rewards=__chosen_rewards
        )

    return dpo_step


class DPOTrainer(BaseTrainer, ABC):
    """
    EasyDel DPO Trainer Class
    """

    def __init__(
            self,
            arguments: TrainArguments,
            model_state: EasyDelState | str,
            ref_model_state: Optional[EasyDelState | str] = None,
            partitioner_config: Optional[PartitionerConfig] = PartitionerConfig(),
            beta: float = 0.1,
            label_smoothing: float = .0,
            loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
            label_pad_token_id: int = -100,
            padding_value: int = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            data_collator: Optional[Callable] = None,
            max_length: Optional[int] = None,
            max_prompt_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
            precompute_ref_log_probs: bool = False,
            model_init_kwargs: Optional[Dict] = None,
            ref_model_init_kwargs: Optional[Dict] = None,
            reference_free: bool = False,
            auto_shard_model_state: bool = True,
            auto_shard_ref_model_state: bool = True,
            is_encoder_decoder: Optional[bool] = False,
            dataset_map_arguments: Optional[dict] = None,
            low_mem_usage: bool = True,
            auto_fix_data: bool = True,
            _do_init_fns: bool = True,
    ):

        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an object.


        :param self: Refer to the object itself
        :param model_state: EasyDelState | str: Pass the model state to the trainer
        :param ref_model_state: Optional[EasyDelState | str]: Pass the reference model state
        :param partitioner_config: Optional[PartitionerConfig]: Specify the partitioner configuration
        :param beta: float: Control the strength of the regularization term
        :param label_smoothing: float: Smooth the labels
        :param loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] : Determine the loss function used
        :param arguments: TrainArguments: Pass the arguments to the trainer
        :param label_pad_token_id: int: Pad the labels
        :param padding_value: int: Specify the value that is used for padding
        :param train_dataset: Optional[Dataset]: Load the training dataset
        :param eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] : Pass the evaluation dataset to the trainer
        :param tokenizer: Optional[PreTrainedTokenizerBase]: Pass the tokenizer to the trainer
        :param max_length: Optional[int]: Set the maximum length of the input sequence
        :param max_prompt_length: Optional[int]: Set the maximum length of the prompt
        :param max_target_length: Optional[int]: Truncate the target sequence
        :param data_collator: Optional[Callable]: Function to be used for creating datasets.
        :param precompute_ref_log_probs: bool: Precompute the log probabilities of the reference model
        :param model_init_kwargs: Optional[Dict]: Pass in the model_kwargs to model for init process
        :param ref_model_init_kwargs: Optional[Dict]: Pass the ref_model_init_kwargs to ref_model for init process
        :param auto_shard_model_state: bool: whenever to automatically shard `model_state`
        :param auto_shard_ref_model_state: bool: whenever to automatically shard `ref_model_state`
        :param dataset_map_arguments: Optional[dict]: arguments to be passed to train and eval datasets for
        tokenizing process with `dataset.map`.
        :param _do_init_fns: bool : preferred to set ture to trainer will automatically configure
        model with provided training Arguments
        :param : Set the padding value for the model
        """
        if partitioner_config is None:
            partitioner_config = PartitionerConfig()
        self.partitioner_config = partitioner_config
        assert arguments is not None, (
            "You Have to pass arguments that will be used for training but you have passed"
            "`arguments=None`"
        )
        assert isinstance(arguments, TrainArguments), (
            f"arguments type must be `TrainArguments` but got {type(arguments)}"
        )
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model_state, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model_state, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model_state, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDelModelForCausalLM` for you."
            )
            model_state = EasyDelState.from_pretrained(
                model_state,
                **model_init_kwargs
            )
        if isinstance(ref_model_state, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDelModelForCausalLM`"
            )
            ref_model_state = EasyDelState.from_pretrained(
                ref_model_state,
                **ref_model_init_kwargs
            )

        data_collator = DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=False,
        ) if data_collator is None else data_collator

        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )
        self.auto_fix_data = auto_fix_data

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the "
                "DPOTrainer's init it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = arguments.truncation_mode

        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self.reference_free = reference_free
        self.is_encoder_decoder = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.low_mem_usage = low_mem_usage

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if dataset_map_arguments is None:
            dataset_map_arguments = {}
        train_dataset = train_dataset.map(
            self.tokenize_row,
            **dataset_map_arguments
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.tokenize_row,
                **dataset_map_arguments
            )

        self.arguments = arguments
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.ref_model_state = ref_model_state
        self.model_state = model_state
        self._loggers_initialized = False
        self.mesh = self.arguments.get_mesh()
        assert padding_value is not None, "`padding_value` can not be set as `None` it must be an integer."

        self.concatenated_forward = create_concatenated_forward(
            is_encoder_decoder=self.is_encoder_decoder,
            padding_value=padding_value,
            label_pad_token_id=label_pad_token_id,
        )
        self.auto_shard_ref_model_state = auto_shard_ref_model_state
        self.auto_shard_model_state = auto_shard_model_state
        self._cached_p_l_s = None
        self._cached_c_l_s = None
        self._cached_r_l_s = None
        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            finetune=True,
            checkpoint_path=None,
            _do_init_fns=_do_init_fns
        )

    def initialize_trainer_utils(self):
        """
        The initialize_trainer_utils function is responsible for initializing the following:
            - wandb_runtime (if you use_wandb is True)
            - timer object (for logging time taken by various functions)
            - dataloader objects for training and evaluation data, along with max steps per epoch.
              The configure_dataloader function accomplishes this task.

        :param self: Represent the instance of the class
        :return: A tuple of functions

        """
        self.wandb_runtime = self.arguments.get_wandb_init() if self.arguments.use_wandb else None
        self.timer = Timers(
            use_wandb=False,
            tensorboard_writer=self.arguments.get_board()
        )

        self.timer("configure dataloaders").start()
        dataset_configurations = self.configure_dataloader()
        self.dataloader_train = dataset_configurations.dataloader_train
        self.max_training_steps = dataset_configurations.max_training_steps
        self.dataloader_eval = dataset_configurations.dataloader_eval
        self.max_evaluation_steps = dataset_configurations.max_evaluation_steps

        self.timer("configure dataloaders").stop()

        self.timer.log(["configure dataloaders"])

        self.timer("configure Model, Optimizer, Scheduler and Config").start()
        model_configurations = self.configure_model()
        model = model_configurations.model
        tx = model_configurations.tx
        scheduler = model_configurations.scheduler
        config = model_configurations.config
        self.model = model
        self.tx = tx
        self.scheduler = scheduler
        self.config = config
        if self.rapture is not None:
            lora_modules = self.rapture.apply_lora(
                module=model,
                parameters=self.arguments.rapture_config.parameters,
                tx=tx,
            )
            self.lora_parameters = lora_modules.lora_parameters
            self.lora_apply_fn = lora_modules.lora_module.__call__
            self.lora_opt_state = lora_modules.lora_opt_state
            self.lora_model = lora_modules.lora_module
            self.lora_tx = lora_modules.lora_tx

        self.timer("configure Model, Optimizer, Scheduler and Config").stop()
        self.timer.log(["configure Model, Optimizer, Scheduler and Config"])

        self.timer("configure functions and sharding them").start()

        if self.auto_shard_model_state:
            self.timer("Sharding Model State").start()
            self.model_state: EasyDelState = self.shard_states(
                self.model_state,
                self.model_state.module.config.get_partition_rules(self.arguments.fully_sharded_data_parallel)
            )

            termcolor.cprint("initializing TX and Schedulers for `model_state`", force_color=True, color="cyan")

            params_with_opt = (
                self.model_state.params[
                    'params'
                ] if '_overwrite_with_gradient' in self.model_state.params else self.model_state.params
            )
            opt_state = self.tx.init(params_with_opt)

            self.model_state = self.model_state.replace(
                opt_state=opt_state,
                tx=self.tx
            )

            self.timer("Sharding Model State").stop()
            self.timer.log(["Sharding Model State"])
        if self.auto_shard_ref_model_state and self.ref_model_state is not None:
            self.timer("Sharding Ref Model State").start()
            self.ref_model_state = self.shard_states(
                self.ref_model_state,
                self.ref_model_state.module.config.get_partition_rules(self.arguments.fully_sharded_data_parallel)
            )
            self.timer("Sharding Ref Model State").stop()
            self.timer.log(["Sharding Ref Model State"])

        function_configurations = self.configure_functions()
        self.create_sharded_state_from_params_function = (
            function_configurations.create_sharded_state_from_params_function
        )
        self.sharded_train_step_function = function_configurations.sharded_train_step_function
        self.sharded_eval_step_function = function_configurations.sharded_eval_step_function
        self.mesh = function_configurations.mesh
        self.checkpoint_manager = function_configurations.checkpoint_manager
        self.initialize_state_function = function_configurations.initialize_state_function
        self.timer("configure functions and sharding them").stop()
        self.timer.log(["configure functions and sharding them"])

    def create_collate_function(
            self,
            max_sequence_length: int,
            truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> Callable:
        return self.data_collator

    def shard_states(self, state, rules):
        with self.arguments.get_mesh():
            partition_spec = match_partition_rules(rules=rules, params=jax.eval_shape(lambda: state))

            def _shard(x):
                return x

            shard = pjit(
                _shard,
                in_shardings=(PartitionSpec(),),
                out_shardings=partition_spec
            )
            return shard(state)

    def configure_dataloader(self) -> TrainerConfigureDataloaderFuncOutput:
        dataloader_train = self.get_train_dataloader()
        max_evaluation_steps = None
        dataloader_eval = None

        max_training_steps = self.arguments.num_train_epochs * len(
            dataloader_train
        ) if self.arguments.max_training_steps is None else self.arguments.max_training_steps
        if self.eval_dataset is not None:
            dataloader_eval = self.get_eval_dataloader(self.eval_dataset)
            max_evaluation_steps = len(dataloader_eval)
        return TrainerConfigureDataloaderFuncOutput(
            dataloader_train=dataloader_train,
            max_training_steps=max_training_steps,
            dataloader_eval=dataloader_eval,
            max_evaluation_steps=max_evaluation_steps
        )

    def configure_functions(self) -> TrainerConfigureFunctionFuncOutput:
        def initialize_state_function():
            initialized_parameters = self.model.init_weights(
                jax.random.PRNGKey(0),
                self.arguments.init_input_shape
            )

            if self.arguments.dtype == jnp.bfloat16:
                initialized_parameters = self.model.to_bf16(initialized_parameters)
            elif self.arguments.dtype == jnp.float16:
                initialized_parameters = self.model.to_fp16(initialized_parameters)

            tx = self.tx
            parameters = flax.core.freeze({"params": initialized_parameters})
            tx_init = copy.deepcopy(self.arguments.optimizer_kwargs)

            if self.rapture is not None:
                lora_parameters = self.lora_parameters
                if self.arguments.dtype == jnp.bfloat16:
                    lora_parameters = self.model.to_bf16(lora_parameters)
                elif self.arguments.dtype == jnp.float16:
                    lora_parameters = self.model.to_fp16(lora_parameters)

                return EasyDelState(
                    step=0,
                    apply_fn=self.lora_apply_fn,
                    params=lora_parameters,
                    tx=self.lora_tx,
                    opt_state=self.lora_opt_state,
                    tx_init=EasyDelState.safe_dict(tx_init),
                    hyperparameters=EasyDelState.create_hyperparameters(self.model.config.model_type),
                    module=self.lora_model,
                    module_config=self.model_state.module.config,
                    module_config_args=None,
                )
            else:
                return EasyDelState.create(
                    tx=tx,
                    params=parameters,
                    apply_fn=self.model.__call__,
                    module_config=copy.deepcopy(self.model_state.module.config),
                    tx_init=tx_init,
                    hyperparameters=EasyDelState.create_hyperparameters(self.model.config.model_type),
                    module=self.model,
                    module_config_args=None
                )

        def create_state_from_params_function(parameters):
            if self.rapture is None:
                return EasyDelState.create(
                    tx=self.tx,
                    params=parameters,
                    apply_fn=self.model.__call__,
                    module_config=copy.deepcopy(self.model_state.module.config),
                    tx_init=copy.deepcopy(self.arguments.optimizer_kwargs),
                    hyperparameters=EasyDelState.create_hyperparameters(self.model.config.model_type),
                    module=self.model,
                    module_config_args=None
                )
            else:
                return EasyDelState(
                    step=0,
                    apply_fn=self.lora_apply_fn,
                    params=parameters,
                    tx=self.lora_tx,
                    opt_state=self.lora_opt_state,
                    tx_init=EasyDelState.safe_dict(copy.deepcopy(self.arguments.optimizer_kwargs)),
                    hyperparameters=EasyDelState.create_hyperparameters(self.model.config.model_type),
                    module=self.lora_model,
                    module_config=self.model_state.module.config,
                    module_config_args=None,
                )

        state_shape = jax.eval_shape(lambda: self.model_state)

        state_partition_spec = match_partition_rules(
            self.config.get_partition_rules(
                fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
            ) if self.arguments.custom_rule is None else self.arguments.custom_rule,
            state_shape
        )
        create_sharded_state_from_params_function = pjit(
            create_state_from_params_function,
            in_shardings=(state_partition_spec.params,),
            out_shardings=state_partition_spec,
            donate_argnums=(0,)
        )
        train_function = create_dpo_train_function(
            concatenated_forward=self.concatenated_forward,
            ref_state=self.ref_model_state,
            loss_type=self.loss_type,
            reference_free=self.reference_free,
            label_smoothing=self.label_smoothing,
            beta=self.beta
        )
        sharded_train_step_function = pjit(
            train_function,
            in_shardings=(state_partition_spec, self.arguments.step_partition_spec),
            out_shardings=(state_partition_spec, PartitionSpec()),
        )

        eval_function = create_dpo_eval_function(
            concatenated_forward=self.concatenated_forward,
            ref_state=self.ref_model_state,
            loss_type=self.loss_type,
            reference_free=self.reference_free,
            label_smoothing=self.label_smoothing,
            beta=self.beta
        )

        sharded_eval_step_function = pjit(
            eval_function,
            in_shardings=(state_partition_spec, self.arguments.step_partition_spec),
            out_shardings=(state_partition_spec, PartitionSpec()),
        )

        self.arguments.ckpt_path_exists()
        self.state_partition_spec = state_partition_spec
        self.state_shape = state_shape
        checkpoint_manager = self.arguments.get_streaming_checkpointer()
        mesh = self.arguments.get_mesh()
        return TrainerConfigureFunctionFuncOutput(
            initialize_state_function=initialize_state_function,
            sharded_train_step_function=sharded_train_step_function,
            create_sharded_state_from_params_function=create_sharded_state_from_params_function,
            checkpoint_manager=checkpoint_manager,
            mesh=mesh,
            sharded_eval_step_function=sharded_eval_step_function
        )

    def configure_model(self) -> TrainerConfigureModelFuncOutput:
        config = self.model_state.module.config
        tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_training_steps)
        return TrainerConfigureModelFuncOutput(
            model=self.model_state.module,
            config=config,
            scheduler=scheduler,
            tx=tx
        )

    def _get_train_dataloader(self) -> DataLoader:

        """
        The _get_train_dataloader function is used to create a DataLoader object for the training dataset.

        :param self: Represent the instance of the class
        :return: A dataloader object
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self.arguments.total_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.arguments.dataloader_num_workers,
            "pin_memory": self.arguments.dataloader_pin_memory,
        }

        return DataLoader(
            train_dataset,
            drop_last=True,
            **dataloader_params
        )

    def _get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        dataloader_params = {
            "batch_size": self.arguments.total_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.arguments.dataloader_num_workers,
            "pin_memory": self.arguments.dataloader_pin_memory,
            "shuffle": False,
            "drop_last": True,
        }

        return DataLoader(
            eval_dataset,
            **dataloader_params
        )

    def get_train_dataloader(
            self,
    ) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.arguments.total_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.arguments.dataloader_num_workers,
                "pin_memory": self.arguments.dataloader_pin_memory,
                "shuffle": False,
            }

            data_loader = DataLoader(self.train_dataset, **dataloader_params)
            reference_chosen_log_probs = []
            reference_rejected_log_probs = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(
                    self.model_state,
                    padded_batch,
                )
                reference_chosen_log_probs.append(reference_chosen_logp.cpu())
                reference_rejected_log_probs.append(reference_rejected_logp.cpu())

            all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
            all_reference_rejected_log_probs = jnp.concatenate(reference_rejected_log_probs)
            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_log_probs", column=all_reference_chosen_log_probs
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_log_probs", column=all_reference_rejected_log_probs
            )

            self._precomputed_train_ref_log_probs = True
        return self._get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.arguments.total_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.arguments.dataloader_num_workers,
                "pin_memory": self.arguments.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = DataLoader(eval_dataset, **dataloader_params)

            reference_chosen_log_probs = []
            reference_rejected_log_probs = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(
                    self.model_state,
                    padded_batch
                )
                reference_chosen_log_probs.append(reference_chosen_logp.cpu())
                reference_rejected_log_probs.append(reference_rejected_logp.cpu())

            all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
            all_reference_rejected_log_probs = jnp.concatenate(reference_rejected_log_probs)

            eval_dataset = eval_dataset.add_column(name="reference_chosen_log_probs",
                                                   column=all_reference_chosen_log_probs)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_log_probs", column=all_reference_rejected_log_probs
            )

            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return self._get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]
        prompt_input_ids = jnp.asarray(prompt_input_ids, dtype="i4")
        answer_input_ids = jnp.asarray(answer_input_ids, dtype="i4")
        full_concat_input_ids = jnp.concatenate(
            (
                prompt_input_ids,
                answer_input_ids
            )
        )

        # Prepare input tokens for token by token comparison
        full_input_ids = jnp.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids.tolist() != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=jnp.array(prompt_input_ids, dtype="i4"),
            prompt_attention_mask=jnp.array(prompt_attention_mask, dtype="i4"),
            input_ids=jnp.array(answer_input_ids, dtype="i4"),
            attention_mask=jnp.array(answer_attention_mask, dtype="i4"),
        )

    def tokenize_row(self, feature, state: EasyDelState = None) -> Dict:

        """
        The tokenize_row function is responsible for taking a single row of data and converting it into the format that
        the model expects. This includes:
        - Tokenizing the text (using HuggingFace's tokenizer)
        - Padding/truncating sequences to a fixed length (if necessary)
        - Creating attention masks, which tell the model which tokens are padding and which aren't.

        :param self: Represent the instance of the class
        :param feature: Pass in the data from the dataset
        :param state: EasyDelState: Keep track of the state of the tokenizer
        :return: A dictionary of the following keys
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)} , {prompt}")
        prompt_tokens = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="np",
        )
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)} , {chosen}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)
        v2d = lambda ar: ar.reshape(1, -1) if ar.ndim == 1 else ar

        def add_tkn(n, ar):
            return jnp.concatenate(
                (
                    jnp.array(n).reshape(1, 1),
                    v2d(ar)
                ), axis=-1
            )

        def add_post_tkn(n, ar):
            return jnp.concatenate(
                (
                    v2d(ar),
                    jnp.array(n).reshape(1, 1)
                ), axis=-1
            )

        prompt_tokens["prompt_input_ids"] = add_tkn(
            self.tokenizer.bos_token_id,
            prompt_tokens["prompt_input_ids"]
        )
        chosen_tokens["prompt_input_ids"] = add_tkn(
            self.tokenizer.bos_token_id,
            chosen_tokens["prompt_input_ids"]
        )
        rejected_tokens["prompt_input_ids"] = add_tkn(
            self.tokenizer.bos_token_id,
            rejected_tokens["prompt_input_ids"]
        )

        prompt_tokens["prompt_attention_mask"] = add_tkn(
            1, prompt_tokens["prompt_attention_mask"]
        )
        chosen_tokens["prompt_attention_mask"] = add_tkn(
            1, chosen_tokens["prompt_attention_mask"]
        )
        rejected_tokens["prompt_attention_mask"] = add_tkn(
            1, rejected_tokens["prompt_attention_mask"]
        )

        # add EOS token to end of answer
        chosen_tokens["input_ids"] = add_post_tkn(self.tokenizer.eos_token_id, chosen_tokens["input_ids"])
        chosen_tokens["attention_mask"] = add_post_tkn(1, chosen_tokens["attention_mask"])

        rejected_tokens["input_ids"] = add_post_tkn(self.tokenizer.eos_token_id, rejected_tokens["input_ids"])
        rejected_tokens["attention_mask"] = add_post_tkn(1, rejected_tokens["attention_mask"])

        longer_response_length = max(chosen_tokens["input_ids"].shape[-1], rejected_tokens["input_ids"].shape[-1])

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            length_rn = answer_tokens["prompt_input_ids"].shape[-1] + longer_response_length
            if length_rn > self.max_length:

                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][:, : self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][:, -self.max_prompt_length:]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if answer_tokens["prompt_input_ids"].shape[-1] + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][:, : self.max_length - self.max_prompt_length]

        chosen_sequence_tokens = {
            k: jnp.concatenate(
                (v2d(chosen_tokens[f"prompt_{k}"]), v2d(chosen_tokens[k])),
                axis=-1
            ) for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: jnp.concatenate(
                (v2d(rejected_tokens[f"prompt_{k}"]), v2d(rejected_tokens[k])),
                axis=-1
            ) for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["labels"].at[
                                           : len(chosen_tokens["prompt_input_ids"])
                                           ].set([self.label_pad_token_id] * len(chosen_tokens["prompt_input_ids"]))
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["labels"].at[
                                             : len(rejected_tokens["prompt_input_ids"])
                                             ].set(
            ([self.label_pad_token_id] * len(rejected_tokens["prompt_input_ids"]))
        )

        for k, tokens_ in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in tokens_.items():
                if type_key == "token_type_ids":
                    continue

                b, s = tokens.shape

                if self.max_prompt_length > s:
                    if k == "chosen_":
                        if type_key == "input_ids":
                            tokens = pad_to_length(
                                tokens,
                                self.max_target_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )
                        elif type_key == "attention_mask":
                            tokens = pad_to_length(
                                tokens,
                                self.max_target_length,
                                pad_value=0,
                                axis=-1
                            )
                        elif type_key == "labels":
                            tokens = pad_to_length(
                                tokens,
                                self.max_target_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )

                        tokens = tokens[..., :self.max_target_length]

                        if tokens.shape[-1] != self.max_target_length:
                            raise ValueError(
                                f"there was an error in padding token with `type_key` of {type_key}"
                                f". it must have sequence_length of {self.max_target_length} but we got {tokens.shape[-1]}"
                                f" From {k}{type_key}"
                            )
                        tokens = tokens[..., :self.max_target_length]
                    elif k == "rejected_":
                        if type_key == "input_ids":
                            tokens = pad_to_length(
                                tokens,
                                self.max_target_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )
                        elif type_key == "attention_mask":
                            tokens = pad_to_length(
                                tokens,
                                self.max_target_length,
                                pad_value=0,
                                axis=-1
                            )
                        elif type_key == "labels":
                            tokens = pad_to_length(
                                tokens,
                                self.max_target_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )
                        tokens = tokens[..., :self.max_target_length]
                        if tokens.shape[-1] != self.max_target_length:
                            raise ValueError(
                                f"there was an error in padding token with `type_key` of {type_key}"
                                f". it must have sequence_length of {self.max_target_length} but we got {tokens.shape[-1]}"
                                f" From {k}{type_key}"
                            )
                    elif k == "":
                        if type_key == "prompt_input_ids":
                            tokens = pad_to_length(
                                tokens,
                                self.max_prompt_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )
                        elif type_key == "prompt_attention_mask":
                            tokens = pad_to_length(
                                tokens,
                                self.max_prompt_length,
                                pad_value=0,
                                axis=-1
                            )
                        elif type_key == "prompt_labels":
                            tokens = pad_to_length(
                                tokens,
                                self.max_prompt_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )
                        tokens = tokens[..., :self.max_prompt_length]
                        if tokens.shape[-1] != self.max_prompt_length:
                            raise ValueError(
                                f"there was an error in padding token with `type_key` of {type_key}"
                                f". it must have sequence_length of {self.max_prompt_length} but we got {tokens.shape[-1]}"
                                f" From {k}{type_key}"
                            )
                batch[f"{k}{type_key}"] = tokens
        return batch

    def compute_reference_log_probs(
            self,
            state: EasyDelState,
            padded_batch: Dict,
    ) -> tuple[Any, Any]:
        """
        Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset.
        """

        if self.ref_model_state is None:
            (
                reference_chosen_log_probs,
                reference_rejected_log_probs,
                _,
                _,
            ) = self.concatenated_forward(
                apply_fn=state.apply_fn,
                params=state.params,
                batch=padded_batch,
            )
        else:
            (
                reference_chosen_log_probs,
                reference_rejected_log_probs,
                _,
                _,
            ) = self.concatenated_forward(
                apply_fn=self.ref_model_state.apply_fn,
                params=self.ref_model_state.params,
                batch=padded_batch,
            )

        return reference_chosen_log_probs, reference_rejected_log_probs

    def _save_state(
            self,
            state: EasyDelState,
            gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
            milestone: bool = False
    ) -> str:
        step = int(
            jax.device_get(
                state.step
            )
        ) + self.arguments.step_start_point if self.arguments.step_start_point is not None else int(
            jax.device_get(
                state.step
            )
        )
        checkpoint_name = f"{self.arguments.model_name}-S{step}"
        filename = f"{checkpoint_name}_{step}" if milestone else f"{checkpoint_name}"
        filename += ".easy"
        termcolor.cprint(f"Saving Model {filename}.", color="cyan", force_color=True)
        state.save_state(
            filename=filename,
            checkpoint_dir=os.path.join(self.arguments.save_dir, self.arguments.model_name),
            gather_fns=gather_fns,
            float_dtype=self.dtype,
            verbose=self.arguments.verbose,
            save_optimizer=self.arguments.save_optimizer_state,
        )
        return filename

    def train(self):
        assert self.model_state is not None, "model_state can not be None for training purpose"
        with self.mesh:
            with jax.default_device(jax.devices("cpu")[0]) if self.low_mem_usage else leave_alone_context_manager():
                dir_prefix: str = "/dev/shm" if sys.platform != "win32" else "."
                checkpoint_path = "SAVING_SKIPPED"
                if self.arguments.track_memory:
                    initialise_tracking(dir_prefix=dir_prefix)

                pbar = tqdm(total=self.max_training_steps)
                pbar.set_description("Training")
                current_step = self.model_state.step.tolist() if isinstance(
                    self.model_state.step,
                    jax.Array
                ) else self.model_state.step

                loss_sum = None
                chosen_rewards_sum = None
                rejected_rewards_sum = None

                mem_res = "Tracking Option is OFF"

                try:
                    for epoch_index in range(self.arguments.num_train_epochs):
                        for batch in self.dataloader_train:
                            current_step += 1
                            if self.arguments.step_start_point > current_step:
                                ...
                            elif current_step < self.max_training_steps:
                                time_start = time.time()

                                for key in self.arguments.ids_to_pop_from_dataset:
                                    _ = batch.pop(key, None)
                                for key in list(batch.keys()):
                                    if not (
                                            key.endswith("_input_ids")
                                            or key.endswith("_attention_mask")
                                            or key.endswith("_labels")
                                            or key.endswith("_log_probs")
                                    ):
                                        _ = batch.pop(key, None)
                                for k in list(batch.keys()):
                                    v = batch[k]
                                    batch[k] = v.reshape(v.shape[0], -1)
                                if self.auto_fix_data:
                                    batch["rejected_input_ids"] = batch["rejected_input_ids"][
                                                                  ..., :self.max_target_length
                                                                  ]
                                    batch["rejected_attention_mask"] = batch["rejected_attention_mask"][
                                                                       ..., :self.max_target_length
                                                                       ]
                                    batch["rejected_labels"] = batch["rejected_labels"][
                                                               ..., :self.max_target_length
                                                               ]
                                    batch["chosen_input_ids"] = batch["chosen_input_ids"][
                                                                ..., :self.max_target_length
                                                                ]
                                    batch["chosen_attention_mask"] = batch["chosen_attention_mask"][
                                                                     ..., :self.max_target_length
                                                                     ]
                                    batch["chosen_labels"] = batch["chosen_labels"][
                                                             ..., :self.max_target_length
                                                             ]
                                    batch["prompt_input_ids"] = batch["prompt_input_ids"][
                                                                ..., :self.max_prompt_length
                                                                ]
                                    batch["prompt_attention_mask"] = batch["prompt_attention_mask"][
                                                                     ..., :self.max_prompt_length
                                                                     ]
                                try:
                                    if self._cached_p_l_s is None:
                                        self._cached_p_l_s = batch["prompt_input_ids"].shape
                                    else:
                                        assert self._cached_p_l_s == batch["prompt_input_ids"].shape
                                    if self._cached_c_l_s is None:
                                        self._cached_c_l_s = batch["chosen_input_ids"].shape
                                    else:
                                        assert self._cached_c_l_s == batch["chosen_input_ids"].shape
                                    if self._cached_r_l_s is None:
                                        self._cached_r_l_s = batch["rejected_input_ids"].shape
                                    else:
                                        assert self._cached_r_l_s == batch["rejected_input_ids"].shape
                                except AssertionError as e:
                                    warnings.warn(
                                        "there's an issue with dataset and seems like compilation process will be much"
                                        " slower and function (jax.jit) will be re-compiled and that should not happen "
                                        "in case of seeing this error open an issue "
                                        "https://github.com/erfanzar/EasyDeL/issues/new?assignees=&labels=&projects="
                                        "&template=bug_report.md&title=\n or turn `auto_fix_data=True` in DPOTrainer"
                                        f"Extra information error \n{e}"
                                    )

                                self.model_state, metrics = self.sharded_train_step_function(
                                    self.model_state,
                                    batch
                                )
                                total_time = time.time() - time_start
                                (
                                    loss, chosen_rewards, rejected_rewards
                                ) = metrics.loss, metrics.chosen_rewards[0], metrics.rejected_rewards[0]

                                loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss

                                rejected_rewards_sum = (
                                    rejected_rewards.tolist() if (
                                            rejected_rewards_sum is None
                                    ) else rejected_rewards_sum + rejected_rewards
                                )
                                chosen_rewards_sum = (
                                    chosen_rewards.tolist() if (
                                            chosen_rewards_sum is None
                                    ) else chosen_rewards_sum + chosen_rewards
                                )
                                information_queries = {}
                                if self.arguments.track_memory:
                                    mem_res = get_mem(dir_prefix=dir_prefix)

                                    for key in ["Used", "Usage Percent"]:
                                        for device, info in get_capacity_matrix(dir_prefix=dir_prefix).items():
                                            information_queries[f"{device.replace('_', ' ')} ({key})"] = float(
                                                info[key].replace("%", "").replace("GB", ""))
                                train_metrics = {
                                    "loss": loss.tolist(),
                                    "mean_loss": loss_sum / (current_step - self.arguments.step_start_point),
                                    "mean_rejected_rewards": rejected_rewards_sum / (
                                            current_step - self.arguments.step_start_point
                                    ),
                                    "mean_chosen_rewards": chosen_rewards_sum / (
                                            current_step - self.arguments.step_start_point
                                    ),
                                    "learning_rate": self.scheduler(
                                        jax.device_get(self.model_state.step)
                                    ).tolist(),
                                    "step": current_step,
                                    "step_time": total_time,
                                    "perplexity": jnp.exp(loss).tolist(),
                                    "accelerators": information_queries,
                                    "epoch": epoch_index
                                }
                                if self.arguments.use_wandb:
                                    with jax.spmd_mode("allow_all"):
                                        self.wandb_runtime.log(
                                            train_metrics
                                        ),
                                        wandb.summary["captured_memory_log"] = mem_res

                                if self.arguments.track_memory:
                                    IPython.display.clear_output(True)
                                    pbar.display(mem_res)
                                pbar.update(1)
                                log_metrics = copy.deepcopy(train_metrics)
                                _ = log_metrics.pop("accelerators")
                                pbar.set_postfix(
                                    **log_metrics
                                )
                            else:
                                break
                except KeyboardInterrupt:
                    termcolor.cprint(
                        "KeyboardInterrupt At training model Will return Current State of the Model with Parameters.",
                        color="cyan",
                        force_color=True
                    )

                except EasyDelTimerError:
                    termcolor.cprint(
                        "Training reached out maximum training Time Killing training Process "
                        "and Will return Current State of the Model with Parameters.",
                        color="cyan",
                        force_color=True
                    )

                if self.arguments.merge_lora_rapture_parameters and self.rapture is not None:
                    print(
                        termcolor.colored(
                            "Info : ", color="red", force_color=True
                        ),
                        termcolor.colored(
                            "Merging LoRA Parameters.", color="white", force_color=True
                        )
                    )
                    self.model_state = self.model_state.replace(
                        params=self.rapture.merge_parameters(self.model_state.params)
                    )

                shard_fns, gather_fns = make_shard_and_gather_fns(
                    partition_specs=match_partition_rules(
                        rules=self.model_state.module.config.get_partition_rules(
                            self.arguments.fully_sharded_data_parallel
                        ),
                        params=jax.eval_shape(lambda: self.model_state)
                    ),
                    dtype_specs=self.arguments.dtype
                )
                output = TrainerOutput(
                    state=self.model_state,
                    mesh=self.mesh,
                    shard_fns=shard_fns,
                    gather_fns=gather_fns,
                    checkpoint_manager=self.checkpoint_manager,
                )
                if self.arguments.save_steps is None and self.arguments.do_last_save:
                    shard_fns, gather_fns = make_shard_and_gather_fns(
                        match_partition_rules(
                            self.config.get_partition_rules(
                                fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
                            ) if self.arguments.custom_rule is None else self.arguments.custom_rule,
                            jax.eval_shape(lambda: self.model_state)
                        ),
                        dtype_specs=self.dtype
                    )  # You have to re-init the new shard and gather functions in order to be able to skip LoRA weight
                    # crashing errors and saving errors
                    filename = self._save_state(
                        state=self.model_state,
                        gather_fns=gather_fns
                    )
                    checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"

                if self.arguments.do_eval:
                    for _ in self.eval(
                            self.model_state
                    ):
                        ...

                output.checkpoint_path = checkpoint_path
                output.last_save_file_name = filename
                wandb.finish()

        return output

    def eval(self, model_state: EasyDelState) -> typing.Iterator[dict]:
        """Evaluate the Given Model State and yield the eval metrics"""
        assert self.eval_dataset is not None, "`dataloader_eval` is required by evaluator function."
        with self.mesh:

            dir_prefix: str = "/dev/shm" if sys.platform != "win32" else "."

            if self.arguments.track_memory:
                initialise_tracking(dir_prefix=dir_prefix)

            pbar = tqdm(total=self.max_evaluation_steps)
            pbar.set_description("Evaluating")
            current_step = 0
            loss_sum = None
            chosen_rewards_sum = None
            rejected_rewards_sum = None
            mem_res = "Tracking Option is OFF"

            try:
                for batch in self.dataloader_eval:
                    current_step += 1
                    time_start = time.time()
                    for key in self.arguments.ids_to_pop_from_dataset:
                        _ = batch.pop(key, None)
                    for key in list(batch.keys()):
                        if not (
                                key.endswith("_input_ids")
                                or key.endswith("_attention_mask")
                                or key.endswith("_labels")
                        ):
                            _ = batch.pop(key, None)

                    metrics = self.sharded_eval_step_function(
                        model_state,
                        batch
                    )
                    total_time = time.time() - time_start
                    (
                        loss, chosen_rewards, rejected_rewards
                    ) = metrics.loss, metrics.chosen_rewards[0], metrics.rejected_rewards[0]

                    loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
                    rejected_rewards_sum = (
                        rejected_rewards.tolist() if (
                                rejected_rewards_sum is None
                        ) else rejected_rewards_sum + rejected_rewards
                    )
                    chosen_rewards_sum = (
                        chosen_rewards.tolist() if (
                                chosen_rewards_sum is None
                        ) else chosen_rewards_sum + chosen_rewards
                    )

                    information_queries = {}
                    if self.arguments.track_memory:
                        mem_res = get_mem(dir_prefix=dir_prefix)
                        for key in ["Used", "Usage Percent"]:
                            for device, info in get_capacity_matrix(dir_prefix=dir_prefix).items():
                                information_queries[f"{device.replace('_', ' ')} ({key})"] = float(
                                    info[key].replace("%", "").replace("GB", ""))
                    eval_metrics = {
                        "eval_loss": loss.tolist(),
                        "eval_mean_loss": loss_sum / (current_step - self.arguments.step_start_point),
                        "eval_mean_rejected_rewards": rejected_rewards_sum / (
                                current_step - self.arguments.step_start_point
                        ),
                        "eval_mean_chosen_rewards": chosen_rewards_sum / (
                                current_step - self.arguments.step_start_point
                        ),
                        "eval_step": current_step,
                        "eval_step_time": total_time,
                        "eval_perplexity": jnp.exp(loss).tolist(),
                        "eval_accelerators": information_queries,
                    }
                    if self.arguments.use_wandb:
                        with jax.spmd_mode("allow_all"):
                            self.wandb_runtime.log(
                                eval_metrics
                            ),
                            wandb.summary["eval_captured_memory_log"] = mem_res

                    if self.arguments.track_memory:
                        IPython.display.clear_output(True)
                        pbar.display(mem_res)
                    pbar.update(1)
                    log_metrics = copy.deepcopy(eval_metrics)
                    _ = log_metrics.pop("eval_accelerators")
                    pbar.set_postfix(
                        **log_metrics
                    )
                    yield eval_metrics
            except KeyboardInterrupt:
                termcolor.cprint(
                    "KeyboardInterrupt At Evaluation model Will return Nothing and just pass.",
                    color="cyan",
                    force_color=True
                )

    def __repr__(self):

        """
        The __repr__ function is used to generate a string representation of an object.
        This function should return a string that can be parsed by the Python interpreter
        to recreate the object. The __repr__ function is called when you use print() on an
        object, or when you type its name in the REPL.

        :param self: Refer to the instance of the class
        :return: A string representation of the object
        """
        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                try:
                    repr_src = f"\t{k} : " + v.__str__().replace("\n", "\n\t") + "\n"
                    string += repr_src if len(repr_src) < 350 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
                except:
                    repr_src = f"\t{k} : " + "EasyDeLReadingError" + "\n"
                    string += repr_src if len(repr_src) < 350 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"

        return string + ")"

    def __str__(self):

        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()
