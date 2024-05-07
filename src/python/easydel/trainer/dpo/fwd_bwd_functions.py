import typing
import chex
import flax.core
import jax

from typing import Literal, Dict, Union, Tuple, List, Callable

from jax import numpy as jnp
from ...etils import EasyDeLState
from flax.struct import dataclass
from .utils import pad_to_length


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
        concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
        concatenated_batch["concatenated_attention_mask"] = (
            batch["prompt_attention_mask"].repeat(2, 1)
        )

    return concatenated_batch


def create_dpo_train_function(
        concatenated_forward: Callable,
        ref_state: EasyDeLState = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
        reference_free: bool = False,
):
    """
    The create_dpo_train_function function is a helper function that creates the DPO training step.

    :param concatenated_forward: Callable: Define the forward pass of the model
    :param ref_state: EasyDeLState: Specify the reference policy
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
        :param reference_chosen_log_probs: chex.Array: Compute the loss for the reference policy # IGNORED
        :param policy_rejected_log_probs: chex.Array: Calculate the loss for the rejected samples # IGNORED
        :param reference_rejected_log_probs: chex.Array: Calculate the loss of rejected samples # IGNORED
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
        :param reference_chosen_log_probs: chex.Array: Compute the loss for the reference policy # IGNORED
        :param policy_rejected_log_probs: chex.Array: Calculate the loss for the rejected samples # IGNORED
        :param reference_rejected_log_probs: chex.Array: Calculate the loss of rejected samples # IGNORED
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
        :param reference_chosen_log_probs: chex.Array: Compute the loss for the reference policy # IGNORED
        :param policy_rejected_log_probs: chex.Array: Calculate the loss for the rejected samples # IGNORED
        :param reference_rejected_log_probs: chex.Array: Calculate the loss of rejected samples # IGNORED
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
        :param  policy_chosen_log_probs: chex.Array: Calculate the chosen_kl # IGNORED
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
            state: EasyDeLState,
            batch: dict
    ) -> tuple[EasyDeLState, DPOStepOut]:

        """
        The dpo_step function is the core of DPO. It takes a state and a batch,
        and returns an updated state. The update is done by calculating the loss
        for each example in the batch, then taking its gradient with respect to
        the parameters of the policy network (which are stored in `state`). This
        gradient is then used to update `state`.

        :param state: EasyDeLState: Store the parameters of the model
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
        ref_state: EasyDeLState = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
        reference_free: bool = False,
):
    """
    The create_dpo_eval_function function is a helper function that creates the DPO evaluating step.

    :param concatenated_forward: Callable: Define the forward pass of the model
    :param ref_state: EasyDeLState: Specify the reference policy
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
        :param reference_chosen_log_probs: chex.Array: Compute the loss for the reference policy # IGNORED
        :param policy_rejected_log_probs: chex.Array: Calculate the loss for the rejected samples # IGNORED
        :param reference_rejected_log_probs: chex.Array: Calculate the loss of rejected samples # IGNORED
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
        :param reference_chosen_log_probs: chex.Array: Compute the loss for the reference policy # IGNORED
        :param policy_rejected_log_probs: chex.Array: Calculate the loss for the rejected samples # IGNORED
        :param reference_rejected_log_probs: chex.Array: Calculate the loss of rejected samples # IGNORED
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
        :param reference_chosen_log_probs: chex.Array: Compute the loss for the reference policy # IGNORED
        :param policy_rejected_log_probs: chex.Array: Calculate the loss for the rejected samples # IGNORED
        :param reference_rejected_log_probs: chex.Array: Calculate the loss of rejected samples # IGNORED
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
        :param  policy_chosen_log_probs: chex.Array: Calculate the chosen_kl # IGNORED
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
            state: EasyDeLState,
            batch: dict
    ) -> DPOStepOut:

        """
        The dpo_step function is the core of DPO. It takes a state and a batch,
        and returns an updated state. The update is done by calculating the loss
        for each example in the batch, then taking its gradient with respect to
        the parameters of the policy network (which are stored in `state`). This
        gradient is then used to update `state`.

        :param state: EasyDeLState: Store the parameters of the model
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
