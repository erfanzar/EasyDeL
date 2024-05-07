import copy
import os
import sys
import time
import typing
import warnings
from abc import ABC
from collections import defaultdict
from glob import glob

import flax.core
import jax
import tensorflow.data
import tensorflow_datasets
import termcolor
import wandb
from fjformer import match_partition_rules, make_shard_and_gather_fns
from flax.core import FrozenDict
from tqdm import tqdm

from typing import (
    Optional,
    Literal,
    Dict,
    Union,
    Any,
    Callable,
    Mapping,
    Tuple
)

from jax.experimental.pjit import pjit
from datasets import Dataset
from jax import numpy as jnp

from ...etils.etils import get_logger
from ..training_configurations import TrainArguments
from ..base_trainer import (
    BaseTrainer,
    TrainerConfigureFunctionFuncOutput,
    TrainerConfigureDataloaderFuncOutput,
    TrainerConfigureModelFuncOutput
)
from ...etils import EasyDeLState, EasyDeLTimerError
from transformers import PreTrainedTokenizerBase
from jax.sharding import PartitionSpec

from ...utils import Timers, prefix_print
from ..dpo.utils import (
    pad_to_length,
    DPODataCollatorWithPadding,
    leave_alone_context_manager
)
from .fwd_bwd_functions import (
    create_orpo_step_function,
    create_concatenated_forward,
)
from .modelling_output import ORPOTrainerOutput

logger = get_logger(__name__)


class ORPOTrainer(BaseTrainer, ABC):
    """
    easydel ORPO Trainer Class
    """

    def __init__(
            self,
            arguments: TrainArguments,
            max_length: Optional[int] = None,
            max_prompt_length: Optional[int] = None,
            max_completion_length: Optional[int] = None,
            beta: float = 0.1,
            disable_dropout: bool = True,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
            padding_value: int = None,
            data_collator: Optional[DPODataCollatorWithPadding] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            _do_init_fns: bool = True,
            dataset_map_arguments: Optional[Dict[str, Any]] = None,
            low_mem_usage: bool = False,
    ):

        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an object.


        :param self: Refer to the object itself
        :param beta: float: Control the strength of the regularization term
        :param arguments: TrainArguments: Pass the arguments to the trainer
        :param label_pad_token_id: int: Pad the labels
        :param padding_value: int: Specify the value that is used for padding
        :param train_dataset: Optional[Dataset]: Load the training dataset
        :param eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] : Pass the evaluation dataset to the trainer
        :param tokenizer: Optional[PreTrainedTokenizerBase]: Pass the tokenizer to the trainer
        :param max_length: Optional[int]: Set the maximum length of the input sequence
        :param max_prompt_length: Optional[int]: Set the maximum length of the prompt
        :param max_completion_length: Optional[int]: Truncate the target sequence
        :param data_collator: Optional[Callable]: Function to be used for creating datasets.
        tokenizing process with `dataset.map`.
        :param _do_init_fns: bool : preferred to set ture to trainer will automatically configure
        model with provided training Arguments
        :param : Set the padding value for the model
        """

        assert arguments is not None, (
            "You Have to pass arguments that will be used for training but you have passed"
            "`arguments=None`"
        )
        assert isinstance(arguments, TrainArguments), (
            f"arguments type must be `TrainArguments` but got {type(arguments)}"
        )

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a ORPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the ORPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the ORPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_completion_length is None:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_completion_length` in the "
                "ORPOTrainer's init it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_completion_length = 128

        padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = arguments.truncation_mode
        self.disable_dropout = disable_dropout
        self.max_completion_length = max_completion_length
        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder
        self.low_mem_usage = low_mem_usage
        self.beta = beta
        data_collator = DPODataCollatorWithPadding(
            max_prompt_length=self.max_prompt_length,
            max_target_length=self.max_completion_length,
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=False,
        ) if data_collator is None else data_collator
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
        self._loggers_initialized = False
        self.mesh = self.arguments.get_mesh()
        assert padding_value is not None, "`padding_value` can not be set as `None` it must be an integer."

        self.concatenated_forward = create_concatenated_forward(
            is_encoder_decoder=self.is_encoder_decoder,
            padding_value=padding_value,
            label_pad_token_id=label_pad_token_id,
        )

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

    def tokenize_row(self, feature, state: EasyDeLState = None) -> Dict:

        """
        The tokenize_row function is responsible for taking a single row of data and converting it into the format that
        the model expects. This includes:
        - Tokenizing the text (using HuggingFace's tokenizer)
        - Padding/truncating sequences to a fixed length (if necessary)
        - Creating attention masks, which tell the model which tokens are padding and which aren't.

        :param self: Represent the instance of the class
        :param feature: Pass in the data from the dataset
        :param state: EasyDeLState: Keep track of the state of the tokenizer
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
                                self.max_completion_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )
                        elif type_key == "attention_mask":
                            tokens = pad_to_length(
                                tokens,
                                self.max_completion_length,
                                pad_value=0,
                                axis=-1
                            )
                        elif type_key == "labels":
                            tokens = pad_to_length(
                                tokens,
                                self.max_completion_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )

                        tokens = tokens[..., :self.max_completion_length]

                        if tokens.shape[-1] != self.max_completion_length:
                            raise ValueError(
                                f"there was an error in padding token with `type_key` of {type_key}"
                                f". it must have sequence_length of {self.max_completion_length} but we got {tokens.shape[-1]}"
                                f" From {k}{type_key}"
                            )
                        tokens = tokens[..., :self.max_completion_length]
                    elif k == "rejected_":
                        if type_key == "input_ids":
                            tokens = pad_to_length(
                                tokens,
                                self.max_completion_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )
                        elif type_key == "attention_mask":
                            tokens = pad_to_length(
                                tokens,
                                self.max_completion_length,
                                pad_value=0,
                                axis=-1
                            )
                        elif type_key == "labels":
                            tokens = pad_to_length(
                                tokens,
                                self.max_completion_length,
                                pad_value=self.padding_value,
                                axis=-1
                            )
                        tokens = tokens[..., :self.max_completion_length]
                        if tokens.shape[-1] != self.max_completion_length:
                            raise ValueError(
                                f"there was an error in padding token with `type_key` of {type_key}"
                                f". it must have sequence_length of {self.max_completion_length} but we got {tokens.shape[-1]}"
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

    def configure_functions(self) -> TrainerConfigureFunctionFuncOutput:
        """
        The configure_functions function is responsible for configuring the functions that will be used in training.
        It does this by first defining a function called function_configurations, which initializes the model parameters
         and returns
        them as a EasyDeLState object. The EasyDeLState object contains all the information needed to train or evaluate
        on a batch of data, including:
        :param self: Access the class attributes
        :return: A TrainerConfigureFunctionFuncOutput object

        """

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

                return EasyDeLState(
                    step=0,
                    apply_fn=self.lora_apply_fn,
                    params=lora_parameters,
                    tx=self.lora_tx,
                    opt_state=self.lora_opt_state,
                    tx_init=EasyDeLState.safe_dict(tx_init),
                    hyperparameters=EasyDeLState.create_hyperparameters(self.model.config.model_type),
                    module=self.lora_model,
                    module_config=self.model.config,
                    module_config_args=None,
                )
            else:
                return EasyDeLState.create(
                    tx=tx,
                    params=parameters,
                    apply_fn=self.model.__call__,
                    module_config=copy.deepcopy(self.model.config),
                    tx_init=tx_init,
                    hyperparameters=EasyDeLState.create_hyperparameters(self.model.config.model_type),
                    module=self.model,
                    module_config_args=None
                )

        def create_state_from_params_function(parameters):
            if self.rapture is None:
                return EasyDeLState.create(
                    tx=self.tx,
                    params=parameters,
                    apply_fn=self.model.__call__,
                    module_config=copy.deepcopy(self.model.config),
                    tx_init=copy.deepcopy(self.arguments.optimizer_kwargs),
                    hyperparameters=EasyDeLState.create_hyperparameters(self.model.config.model_type),
                    module=self.model,
                    module_config_args=None
                )
            else:
                return EasyDeLState(
                    step=0,
                    apply_fn=self.lora_apply_fn,
                    params=parameters,
                    tx=self.lora_tx,
                    opt_state=self.lora_opt_state,
                    tx_init=EasyDeLState.safe_dict(copy.deepcopy(self.arguments.optimizer_kwargs)),
                    hyperparameters=EasyDeLState.create_hyperparameters(self.model.config.model_type),
                    module=self.lora_model,
                    module_config=self.model.config,
                    module_config_args=None,
                )

        state_shape = jax.eval_shape(initialize_state_function)
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
        sharded_train_step_function = pjit(
            create_orpo_step_function(
                mode="train",
                beta=self.beta,
                concatenated_forward=self.concatenated_forward,
                batch_partition_spec=self.arguments.step_partition_spec
            ),
            in_shardings=(state_partition_spec, PartitionSpec()),
            out_shardings=(state_partition_spec, PartitionSpec(),),

        )

        sharded_eval_step_function = pjit(
            create_orpo_step_function(
                mode="eval",
                beta=self.beta,
                concatenated_forward=self.concatenated_forward,
                batch_partition_spec=self.arguments.step_partition_spec
            ),
            in_shardings=(state_partition_spec, PartitionSpec()),
            out_shardings=(state_partition_spec, PartitionSpec(),),

        )

        mesh = self.arguments.get_mesh()
        self.arguments.ckpt_path_exists()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()
        self.state_partition_spec = state_partition_spec
        self.state_shape = state_shape

        return TrainerConfigureFunctionFuncOutput(
            create_sharded_state_from_params_function=create_sharded_state_from_params_function,
            sharded_train_step_function=sharded_train_step_function,
            sharded_eval_step_function=sharded_eval_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
            initialize_state_function=initialize_state_function
        )

    def initialize_state(
            self,
            model_parameters: Optional[flax.core.FrozenDict] = None,
            state: Optional[EasyDeLState] = None,
    ) -> Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
        if model_parameters is None and state is None and self.rapture is None and self.checkpoint_path is None:
            raise RuntimeError(
                "You are passing `model_parameters=None`, `state=None`, and `checkpoint_path=None` and also you are not"
                " using LoRA, if you are "
                "Using LoRA make sure to pass parameters and Rapture Config correctly otherwise pass the "
                "model_parameters or state."
            )
        if model_parameters is None and state is None:
            model_parameters = self.lora_parameters
        with self.mesh:
            shard_fns, gather_fns = make_shard_and_gather_fns(
                self.state_partition_spec,
                dtype_specs=self.dtype
            )
            if state is not None:
                sharded_state = state
                params = sharded_state.params if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                    lambda f, x: f(x),
                    shard_fns.params,
                    sharded_state.params
                )
                sharded_state.params = params
                if sharded_state.opt_state is None:
                    prefix_print(
                        "Action", "Optimizer State is not Found!, initializing one."
                    )
                    with jax.default_device(self.arguments.offload_device):
                        sharded_state = sharded_state.init_opt_state()
                        opt_state = sharded_state.opt_state if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                            lambda f, x: f(x),
                            shard_fns.opt_state,
                            sharded_state.opt_state
                        )
                        sharded_state = sharded_state.replace(
                            opt_state=opt_state
                        )
            elif self.finetune:

                if model_parameters is None and self.checkpoint_path is not None:
                    prefix_print(
                        "Action", f"Loading Model From {self.checkpoint_path}"
                    )
                    with jax.default_device(self.arguments.offload_device):
                        sharded_state = EasyDeLState.load_state(
                            verbose=self.arguments.verbose,
                            state_shard_fns=shard_fns,
                            init_optimizer_state=True,
                            checkpoint_path=self.checkpoint_path,
                            input_shape=self.arguments.init_input_shape,
                            config_kwargs=self.arguments.loaded_model_config_kwargs
                        )
                        state_shape = jax.eval_shape(lambda: sharded_state)
                        state_partition_spec = match_partition_rules(
                            self.config.get_partition_rules(
                                fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
                            ) if self.arguments.custom_rule is None else self.arguments.custom_rule,
                            state_shape
                        )
                        sharded_train_step_function = pjit(
                            create_orpo_step_function(
                                mode="train",
                                beta=self.beta,
                                concatenated_forward=self.concatenated_forward,
                                batch_partition_spec=self.arguments.step_partition_spec
                            ),
                            in_shardings=(state_partition_spec, PartitionSpec()),
                            out_shardings=(state_partition_spec, PartitionSpec(),),

                        )

                        sharded_eval_step_function = pjit(
                            create_orpo_step_function(
                                mode="eval",
                                beta=self.beta,
                                concatenated_forward=self.concatenated_forward,
                                batch_partition_spec=self.arguments.step_partition_spec
                            ),
                            in_shardings=(state_partition_spec, PartitionSpec()),
                            out_shardings=(state_partition_spec, PartitionSpec(),),
                        )

                        self.state_partition_spec = state_partition_spec
                        self.state_shape = state_shape
                        self.sharded_train_step_function = sharded_train_step_function
                        self.sharded_eval_step_function = sharded_eval_step_function

                    if self.arguments.remove_ckpt_after_load:
                        os.remove(self.checkpoint_path)
                elif model_parameters is not None and self.checkpoint_path is None:
                    prefix_print(
                        "Action", f"Sharding Passed Parameters"
                    )
                    from flax.core import unfreeze
                    if not isinstance(model_parameters, flax.core.FrozenDict):
                        prefix_print(
                            "Warning",
                            "Model Parameters should be like FrozenDict({'params': params}) make sure to "
                            "pass as type FrozenDict in case of not getting UnExcepted Errors "
                        )

                    model_parameters = model_parameters if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                        lambda f, x: f(x),
                        shard_fns.params,
                        model_parameters,
                    )
                    sharded_state = self.create_sharded_state_from_params_function(model_parameters)
                elif model_parameters is not None and self.checkpoint_path is not None:
                    raise EasyDeLTimerError(
                        "You can't pass `model_parameters` and `checkpoint_path` at same time"
                    )
                else:
                    raise EasyDeLTimerError(
                        "You should pass `model_parameters` or `checkpoint_path` to trainer in order to load model"
                    )
            else:
                sharded_state = self.initialize_state_function()
                params = sharded_state.params if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                    lambda f, x: f(x),
                    shard_fns.params,
                    sharded_state.params
                )
                sharded_state.params = params

            self.sharded_state = sharded_state
            return sharded_state, shard_fns, gather_fns

    def _save_state(
            self,
            state: EasyDeLState,
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

        checkpoint_dir = os.path.join(self.arguments.save_dir, self.arguments.model_name)
        filename_extension = ".easy"
        if self.arguments.save_total_limit:
            checkpoint_files = glob(os.path.join(checkpoint_dir, f"*{filename_extension}"))
            checkpoint_files.sort(key=os.path.getmtime)
            for old_checkpoint in checkpoint_files[:-self.arguments.save_total_limit]:
                os.remove(old_checkpoint)
                termcolor.cprint(f"Removed old checkpoint: {old_checkpoint}", color="red", force_color=True)

        checkpoint_name = f"{self.arguments.model_name}-S{step}"
        filename = f"{checkpoint_name}_{step}" if milestone else f"{checkpoint_name}"
        filename += ".easy"
        termcolor.cprint(f"Saving Model {filename}.", color="cyan", force_color=True)
        state.save_state(
            filename=filename,
            checkpoint_dir=checkpoint_dir,
            gather_fns=gather_fns,
            float_dtype=self.dtype,
            verbose=self.arguments.verbose,
            save_optimizer=self.arguments.save_optimizer_state,
        )
        return filename

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
            dataloader_train=dataloader_train,  # type:ignore
            max_training_steps=max_training_steps,
            dataloader_eval=dataloader_eval,
            max_evaluation_steps=max_evaluation_steps
        )

    def _get_train_dataloader(self) -> tensorflow.data.Dataset:

        """
        The _get_train_dataloader function is used to create a tensorflow.data.Dataset object for the training dataset.

        :param self: Represent the instance of the class
        :return: A dataloader object
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        return tensorflow_datasets.as_numpy(
            train_dataset.to_tf_dataset(
                batch_size=self.arguments.total_batch_size,
                collate_fn=data_collator,
                num_workers=self.arguments.dataloader_num_workers,
                shuffle=True,
                drop_remainder=True
            )
        )

    def _get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> tensorflow.data.Dataset:
        """
        Returns the evaluation [`~tensorflow.data.Dataset`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return tensorflow_datasets.as_numpy(
            eval_dataset.to_tf_dataset(
                batch_size=self.arguments.total_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.arguments.dataloader_num_workers,
                shuffle=False,
                drop_remainder=True
            )
        )

    def get_train_dataloader(
            self,
    ) -> tensorflow.data.Dataset:
        """
        Returns the training [`~tensorflow.data.Dataset`].
        """
        return self._get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> tensorflow.data.Dataset:
        """
        Returns the evaluation [`~tensorflow.data.Dataset`].
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return self._get_eval_dataloader(eval_dataset=eval_dataset)

    def train(
            self,
            model_parameters: Optional[flax.core.FrozenDict] = None,
            state: Optional[EasyDeLState] = None
    ) -> ORPOTrainerOutput:
        def get_layer_names(frozen_dict, prefix=""):
            layer_names = {}
            for key, value in frozen_dict.items():
                if isinstance(value, FrozenDict):
                    layer_names.update(get_layer_names(value, prefix=f"{prefix}_{key}"))
                else:
                    layer_name = f"{prefix}_{key}".lstrip("/")
                    layer_names[layer_name] = value
            return layer_names

        def count_model_parameters(_p):
            termcolor.cprint(
                f"Model Contain {sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0]) / 1e9} "
                f"Billion Parameters",
                color="red", force_color=True
            )

        checkpoint_path = "SAVING_SKIPPED"
        if self.arguments.performance_mode:
            termcolor.cprint(
                "Performance Mode is ON, we will ignore the Memory Tracking, WANDB Logging, and extra information "
                "Process.",
                color="red",
                force_color=True
            )
        sharded_state, shard_fns, gather_fns = self.initialize_state(
            model_parameters=model_parameters,
            state=state
        )
        self.model_state = sharded_state
        count_model_parameters(sharded_state.params)
        with self.mesh:
            with jax.default_device(jax.devices("cpu")[0]) if self.low_mem_usage else leave_alone_context_manager():
                dir_prefix: str = "/dev/shm" if sys.platform != "win32" else "."
                checkpoint_path = "SAVING_SKIPPED"

                pbar = tqdm(total=self.max_training_steps)
                pbar.set_description("Training")
                current_step = self.model_state.step.tolist() if isinstance(
                    self.model_state.step,
                    jax.Array
                ) else self.model_state.step

                loss_sum = None

                try:
                    for epoch_index in range(self.arguments.num_train_epochs):
                        for batch in self.dataloader_train:
                            current_step += 1
                            if self.arguments.step_start_point > current_step:
                                ...
                            elif current_step < self.max_training_steps:
                                time_start = time.time()

                                self.model_state, outputs = self.sharded_train_step_function(
                                    self.model_state,
                                    batch
                                )
                                total_time = time.time() - time_start
                                (loss, metrics) = outputs.loss, outputs.metrics

                                loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss

                                train_metrics = {
                                    "train/loss": loss.tolist(),
                                    "train/mean_loss": loss_sum / (current_step - self.arguments.step_start_point),
                                    "train/learning_rate": self.scheduler(
                                        jax.device_get(self.model_state.step)).tolist(),
                                    "train/step": current_step,
                                    "train/step_time": total_time,
                                    "train/perplexity": jnp.exp(loss).tolist(),
                                    "train/epoch": epoch_index
                                }
                                train_metrics.update(metrics)
                                log_metrics = copy.deepcopy(train_metrics)
                                train_metrics.update(self.arguments.captured_memory)
                                if self.arguments.use_wandb:
                                    with jax.spmd_mode("allow_all"):
                                        self.wandb_runtime.log(
                                            train_metrics
                                        )
                                pbar.update(1)
                                pbar.set_postfix(**{k.replace("train/", ""): v for k, v in log_metrics.items()})
                            else:
                                break
                except KeyboardInterrupt:
                    termcolor.cprint(
                        "KeyboardInterrupt At training model Will return Current State of the Model with Parameters.",
                        color="cyan",
                        force_color=True
                    )

                except EasyDeLTimerError:
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
                output = ORPOTrainerOutput(
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

    def eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
        """Evaluate the Given Model State and yield the eval metrics"""
        assert self.eval_dataset is not None, "`dataloader_eval` is required by evaluator function."
        with self.mesh:
            pbar = tqdm(total=self.max_evaluation_steps)
            pbar.set_description("Evaluating")
            current_step = 0
            loss_sum = None
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

                    _, outputs = self.sharded_eval_step_function(
                        model_state,
                        batch
                    )
                    total_time = time.time() - time_start
                    (
                        loss, metrics
                    ) = outputs.loss, outputs.metrics

                    loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss

                    eval_metrics = {
                        "eval/loss": loss.tolist(),
                        "eval/mean_loss": loss_sum / (current_step - self.arguments.step_start_point),
                        "eval/step": current_step,
                        "eval/step_time": total_time,
                        "eval/perplexity": jnp.exp(loss).tolist(),
                    }
                    eval_metrics.update(metrics)
                    log_metrics = copy.deepcopy(eval_metrics)
                    eval_metrics.update(self.arguments.captured_memory)
                    if self.arguments.use_wandb:
                        with jax.spmd_mode("allow_all"):
                            self.wandb_runtime.log(
                                eval_metrics
                            )

                    pbar.update(1)
                    pbar.set_postfix(**{k.replace("eval/", ""): v for k, v in log_metrics.items()})
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
                except TypeError:
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
