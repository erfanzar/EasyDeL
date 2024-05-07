import typing
import warnings

import chex
import fjformer
import flax.core
import jax

from typing import Literal, Dict, Union, Tuple, List, Callable

from jax import numpy as jnp
from ...etils import EasyDeLState
from flax.struct import dataclass
from jax.sharding import PartitionSpec


def pad_to_length(tensor: chex.Array, length: int, pad_value: Union[int, float], axis: int = -1) -> chex.Array:
    if tensor.shape[axis] >= length:
        if tensor.ndim == 2:
            tensor = tensor[:, :length]
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[axis] = length - tensor.shape[axis]
        return jax.numpy.concatenate(
            [
                tensor,
                pad_value * jax.numpy.ones(pad_size, dtype=tensor.dtype),
            ],
            axis=axis,
        )


@dataclass
class ORPOStepOut:
    loss: chex.Array
    metrics: Dict[str, Union[chex.Array, str]]


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

    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
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

        def cross_entropy_loss(logits, labels, mask):
            if not is_encoder_decoder:
                logits = logits[..., :-1, :]
                labels = labels[..., 1:]
                mask = mask[..., 1:]
            loss = fjformer.cross_entropy_loss_and_accuracy(logits, labels, mask)[0]
            return loss

        if is_encoder_decoder:
            labels = concatenated_batch["concatenated_labels"]
        else:
            labels = concatenated_batch["concatenated_input_ids"]

        chosen_nll_loss = cross_entropy_loss(
            all_logits[:len_chosen],
            labels[:len_chosen],
            concatenated_batch["concatenated_attention_mask"][:len_chosen]
        )
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
        return chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits, chosen_nll_loss

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

    labels = jnp.where(labels == label_pad_token_id, 0, labels)

    per_token_logps = jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1), axis=2, indices=labels[:, :, None]
    ).reshape(batch, seq_len)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


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
    and concatenates them together. This is useful for training the model to predict whether an example was chosen
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

    :param fixed_max_length : int|None: by providing fixed_max_length the func will always return a fixed sequence
     length and won't use dynamic methods.

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
        warnings.warn("`concatenated_input_ids` will be repeated (encoder decoder model detected)")
        concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
        concatenated_batch["concatenated_attention_mask"] = (
            batch["prompt_attention_mask"].repeat(2, 1)
        )

    return concatenated_batch


def odds_ratio_loss(
        beta: float,
        policy_chosen_logps: chex.Array,
        policy_rejected_logps: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            jnp.log1p(-jnp.exp(policy_chosen_logps)) - jnp.log1p(-jnp.exp(policy_rejected_logps))
    )
    sig_ratio = jax.nn.sigmoid(log_odds)
    ratio = jnp.log(sig_ratio)
    losses = beta * ratio
    # jax.debug.print("policy_chosen_logps   : {x}", x=policy_chosen_logps)
    # jax.debug.print("policy_rejected_logps : {x}", x=policy_rejected_logps)
    # jax.debug.print("sig_ratio : {x}", x=sig_ratio)
    # jax.debug.print("ratio     : {x}", x=ratio)
    # jax.debug.print("log_odds  : {x}", x=log_odds)
    # jax.debug.print("losses    : {x}", x=losses)

    chosen_rewards = beta * jax.lax.stop_gradient(policy_chosen_logps)
    rejected_rewards = beta * jax.lax.stop_gradient(policy_rejected_logps)

    return (
        losses,
        chosen_rewards,
        rejected_rewards,
        jnp.mean(ratio),
        jnp.mean(log_odds)
    )


def create_orpo_step_function(
        concatenated_forward: Callable,
        beta: float = 0.1,
        mode: Literal["train", "eval"] = "train",
        batch_partition_spec: PartitionSpec = PartitionSpec(("fsdp", "dp"), "sp")
):
    """
    The create_orpo_step_function function is a helper function that creates the ORPO training step.

    :param concatenated_forward: Callable: Define the forward pass of the model
    :param beta: float: Scale the logits
    :param mode: Literal["train", "eval"] : "train", "eval" function modes
    :param batch_partition_spec: PartitionSpec: Batch PartitionSpec
    :return: A function that takes in a state and a batch
    """

    def orpo_step(
            state: EasyDeLState,
            batch: dict
    ) -> tuple[EasyDeLState, ORPOStepOut]:
        """
        The orpo_step function is the core of ORPO. It takes a state and a batch,
        and returns an updated state. The update is done by calculating the loss
        for each example in the batch, then taking its gradient with respect to
        the parameters of the policy network (which are stored in `state`). This
        gradient is then used to update `state`.

        :param state: EasyDeLState: Store the parameters of the model
        :param batch: dict: Pass the data to the model
        :return: A new state, which is a collection of the parameters and apply_fn
        """
        batch = fjformer.with_sharding_constraint(batch, partition_specs=batch_partition_spec)

        def calculate_loss(params: dict | flax.core.FrozenDict):
            (
                policy_chosen_log_probs,
                policy_rejected_log_probs,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_nll_loss
            ) = concatenated_forward(
                state.apply_fn,
                params,
                batch
            )

            losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = odds_ratio_loss(
                beta, policy_chosen_log_probs, policy_rejected_log_probs
            )

            loss = policy_nll_loss - losses.mean()

            reward_accuracies = (chosen_rewards > rejected_rewards).astype("float32")
            metrics = {}
            prefix = "eval_" if mode == "eval" else ""
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
            metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
            metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
            metrics[f"{prefix}logps/rejected"] = policy_rejected_log_probs.mean()
            metrics[f"{prefix}logps/chosen"] = policy_chosen_log_probs.mean()
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean()
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean()
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.mean()
            metrics[f"{prefix}log_odds_ratio"] = log_odds_ratio
            metrics[f"{prefix}log_odds_chosen"] = log_odds_chosen
            return loss, metrics

        if mode == "train":
            grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
            (__loss, (__metrics)), grads = grad_fn(state.params)
            new_state = state.apply_gradients(grads=grads)
        else:
            __loss, __metrics = calculate_loss(state.params)
            new_state = state
        return new_state, ORPOStepOut(
            loss=__loss,
            metrics=__metrics
        )

    return orpo_step
