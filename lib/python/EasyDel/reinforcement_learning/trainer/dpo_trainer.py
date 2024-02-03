import warnings
from collections import defaultdict

import chex
import flax.core
import jax
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.collectors import DPODataCollatorWithPadding
from typing import Optional, Literal, Dict, Union, Any, Tuple, List, Callable
from .utils import pad_to_length
from datasets import Dataset
from jax import numpy as jnp
from ...trainer.training_configurations import TrainArguments
from ...etils.easystate import EasyDelState
from transformers import PreTrainedTokenizerBase
from .partitioner_config import PartitionerConfig


def create_concatenated_forward(
        is_encoder_decoder,
        label_pad_token_id,
        padding_value
):
    """
    The create_concatenated_forward function is a helper function that creates a forward pass function for the
    model. The forward pass function takes in an apply_fn, which is the model's apply_fn, and runs it on concatenated
    inputs. It returns chosen log probs, rejected log probs, chosen logits and rejected logits.

    :param is_encoder_decoder: Determine whether the model is an encoder-decoder model or not
    :param label_pad_token_id: Pad the labels to the same length
    :param padding_value: Pad the inputs to the same length
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
        concatenated_batch = concatenated_inputs(
            batch,
            is_encoder_decoder=is_encoder_decoder,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
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
) -> Dict[str, chex.Array]:
    """
    The concatenated_inputs function takes a batch of chosen and rejected examples,
    and concatenates them together. This is useful for training the model to predict whether     an example was chosen by the human annotator. The function also pads all inputs to
    the same length as the longest input in that batch.

    :param batch: Dict[str,Union[List,chex.Array]]: Pass the batch of data into the function,
    Allow for the batch to be a list of arrays or just an array,
     Specify the type of data that is being passed in
    :param is_encoder_decoder: bool: Determine whether the model is an encoder-decoder model
    :param label_pad_token_id: int: Pad the labels with a value of -100
    :param padding_value: int: Pad the input_ids and attention_mask arrays to the same length
    :return: A dictionary of the concatenated inputs
    """
    concatenated_batch = {}

    if is_encoder_decoder:
        max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
    else:
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

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
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            else:
                raise KeyError("couldn't find pad_value [Dataset Issue]")
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = jnp.concatenate(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                axis=0,
            )

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
    ) -> EasyDelState:

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

            return losses, chosen_rewards, rejected_rewards

        grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
        (__loss, __chosen_rewards, __rejected_rewards), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    return dpo_step


class DPOTrainer:
    """
    EasyDel DPO Trainer Class
    """

    def __init__(
            self,
            model_state: EasyDelState | str = None,
            ref_model_state: Optional[EasyDelState | str] = None,
            partitioner_config: Optional[PartitionerConfig] = PartitionerConfig(),
            beta: float = 0.1,
            label_smoothing: float = 0,
            loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
            arguments: TrainArguments = None,
            label_pad_token_id: int = -100,
            padding_value: int = None,
            truncation_mode: str = "keep_end",
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            max_length: Optional[int] = None,
            max_prompt_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
            precompute_ref_log_probs: bool = False,
            model_init_kwarguments: Optional[Dict] = None,
            ref_model_init_kwarguments: Optional[Dict] = None,
            reference_free: bool = False,
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
        :param truncation_mode: str: Truncate the input text
        :param train_dataset: Optional[Dataset]: Load the training dataset
        :param eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] : Pass the evaluation dataset to the trainer
        :param tokenizer: Optional[PreTrainedTokenizerBase]: Pass the tokenizer to the trainer
        :param max_length: Optional[int]: Set the maximum length of the input sequence
        :param max_prompt_length: Optional[int]: Set the maximum length of the prompt
        :param max_target_length: Optional[int]: Truncate the target sequence
        :param precompute_ref_log_probs: bool: Precompute the log probabilities of the reference model
        :param model_init_kwarguments: Optional[Dict]: Pass in the model_kwarguments to model for init process
        :param ref_model_init_kwarguments: Optional[Dict]: Pass the ref_model_init_kwarguments to ref_model for init process
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
        if model_init_kwarguments is None:
            model_init_kwarguments = {}
        elif not isinstance(model_state, str):
            raise ValueError("You passed model_kwarguments to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwarguments is None:
            ref_model_init_kwarguments = {}
        elif not isinstance(ref_model_state, str):
            raise ValueError(
                "You passed ref_model_kwarguments to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model_state, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDelModelForCausalLM` for you."
            )
            model_state = EasyDelState.from_pretrained(
                model_state,
                **model_init_kwarguments
            )
        if isinstance(ref_model_state, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDelModelForCausalLM`"
            )
            ref_model_state = EasyDelState.from_pretrained(
                ref_model_state,
                **ref_model_init_kwarguments
            )

        data_collator = DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=False,
        )

        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self.reference_free = reference_free
        self.is_encoder_decoder = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        train_dataset = train_dataset.map(self.tokenize_row, )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(self.tokenize_row)

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

        self.concatenated_forward = create_concatenated_forward(
            is_encoder_decoder=self.is_encoder_decoder,
            padding_value=padding_value,
            label_pad_token_id=label_pad_token_id
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
        if hasattr(self, "_remove_unused_columns"):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            warnings.warn(
                "Couldn't find function `_remove_unused_columns` if you are the "
                "developer fix this otherwise ignore this warning"
            )
        dataloader_params = {
            "batch_size": self.arguments.total_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.arguments.dataloader_num_workers,
            "pin_memory": self.arguments.dataloader_pin_memory,
        }

        return DataLoader(
            train_dataset,
            **dataloader_params
        )

    def get_train_dataloader(
            self,
    ) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
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
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
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
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)} , {chosen}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)

        # add BOS token to head of prompt
        prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
        rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                                                                                         self.label_pad_token_id
                                                                                     ] * len(
            chosen_tokens["prompt_input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                                                                                             self.label_pad_token_id
                                                                                         ] * len(
            rejected_tokens["prompt_input_ids"])

        for k, tokens_ in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in tokens_.items():
                if type_key == "token_type_ids":
                    continue
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

    def get_mesh(self) -> jax.sharding.Mesh:

        """
        The get_mesh function returns the mesh of a given instance of the class.

        :param self: Bind the method to an object
        :return: The mesh of the device
        """
        return self.mesh

    def train(self):
        step = 0
        train_function = create_dpo_train_function(
            concatenated_forward=self.concatenated_forward,
            ref_state=self.ref_model_state,
            loss_type=self.loss_type,
            reference_free=self.reference_free,
            label_smoothing=self.label_smoothing,
            beta=self.beta
        )
        for epoch_index in range(self.arguments.num_train_epochs):
            for batch in self.get_train_dataloader():
                step += 1
                if self.arguments.step_start_point > step:
                    ...
                else:
                    self.model_state, mt = train_function(self.model_state, batch=batch)
                    print(mt)
                    break

    def eval(self):
        """
        Process is Under Progress ...
        """
        # TODO : Finish Eval Step
        ...

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
                repr_src = f"\t{k} : " + v.__str__().replace("\n", "\n\t") + "\n"
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
