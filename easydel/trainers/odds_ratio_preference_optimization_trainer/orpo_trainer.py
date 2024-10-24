# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import copy
import os
import time
import typing
import warnings
from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import flax.core
import jax
import numpy as np
import termcolor
from fjformer.sharding import make_shard_and_gather_fns, match_partition_rules
from flax.core import FrozenDict
from jax import jit
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import get_logger
from easydel.trainers.base_trainer import (
	BaseTrainer,
	TrainerConfigureDataloaderOutput,
	TrainerConfigureFunctionOutput,
)
from easydel.trainers.direct_preference_optimization_trainer.utils import (
	DPODataCollatorWithPadding,
	leave_alone_context_manager,
)
from easydel.trainers.odds_ratio_preference_optimization_trainer.fwd_bwd_functions import (
	create_orpo_concatenated_forward,
	create_orpo_step_function,
)
from easydel.trainers.odds_ratio_preference_optimization_trainer.modelling_output import (
	ORPOTrainerOutput,
)
from easydel.trainers.training_configurations import TrainingArguments

logger = get_logger(__name__)


class ORPOTrainer(BaseTrainer, ABC):
	"""
	Trainer for Odds Ratio Preference Optimization (ORPO).

	This trainer handles the training, evaluation, and checkpointing of language models
	using the ORPO algorithm. It supports sharding, gradient accumulation, mixed precision
	training, LoRA.

	Attributes:
	    arguments (TrainingArguments): The training arguments.
	    max_length (Optional[int]): The maximum sequence length.
	    max_prompt_length (Optional[int]): The maximum prompt length.
	    max_completion_length (Optional[int]): The maximum completion length.
	    beta (float): The strength of the regularization term in the ORPO loss.
	    disable_dropout (bool): Whether to disable dropout during training.
	    label_pad_token_id (int): The ID of the padding token for labels.
	    is_encoder_decoder (bool): Whether the model is an encoder-decoder architecture.
	    padding_value (int): The padding value for input sequences.
	    data_collator (Optional[DPODataCollatorWithPadding]): The data collator used for batching.
	    train_dataset (Optional[Dataset]): The training dataset.
	    eval_dataset (Optional[Union[Dataset, Dict[str, Dataset]]]): The evaluation dataset.
	    tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer used for preprocessing.
	    dataset_num_proc (Optional[int]): The number of processes to use for dataset mapping.
	    low_mem_usage (bool): Whether to prioritize low memory usage during training.

	Methods:
	    build_tokenized_answer(self, prompt: str, answer: str) -> Dict: Tokenizes a prompt and answer pair, handling special tokens and padding/truncation.
	    tokenize_row(self, feature: Dict, state: EasyDeLState = None) -> Dict: Tokenizes a single row of data from the ORPO dataset.
	    configure_functions(self) -> TrainerConfigureFunctionOutput: Configures and JIT-compiles the training and evaluation step functions.
	    initialize_state(self, model_parameters: Optional[flax.core.FrozenDict] = None, state: Optional[EasyDeLState] = None) -> Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
	        Initializes the training state, either from scratch, pretrained parameters, or a checkpoint.
	    initialize_trainer_utils(self): Initializes trainer utilities (logging, timer, dataloaders, model, etc.).
	    _configure_dataloaders(self): Configures the dataloaders for training and evaluation.
	    _configure_model(self): Configures the model, optimizer, scheduler, and configuration.
	    _configure_functions(self):  Configures and JIT-compiles the training and evaluation step functions.
	    create_collect_function(self, max_sequence_length: int, truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end") -> Callable:
	        Creates a data collection function for batching.
	    configure_dataloaders(self) -> TrainerConfigureDataloaderOutput: Configures the dataloaders for training and evaluation.
	    _get_train_dataloader(self) -> tensorflow.data.Dataset: Creates the training dataloader.
	    _get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> tensorflow.data.Dataset: Creates the evaluation dataloader.
	    get_train_dataloader(self) -> tensorflow.data.Dataset: Returns the training dataloader.
	    get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> tensorflow.data.Dataset: Returns the evaluation dataloader.
	    train(self, model_parameters: Optional[flax.core.FrozenDict] = None, state: Optional[EasyDeLState] = None) -> ORPOTrainerOutput:
	        Trains the ORPO model and returns the training output.
	    eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]: Evaluates the ORPO model and yields evaluation metrics.
	"""

	def __init__(
		self,
		arguments: TrainingArguments,
		max_length: Optional[int] = None,
		max_prompt_length: Optional[int] = None,
		max_completion_length: Optional[int] = None,
		beta: float = 0.1,
		disable_dropout: bool = True,
		label_pad_token_id: int = -100,
		is_encoder_decoder: bool = False,
		padding_value: int = None,
		data_collator: Optional[DPODataCollatorWithPadding] = None,
		train_dataset: Optional["Dataset"] = None,  # noqa #type:ignore
		eval_dataset: Optional[
			Union["Dataset", Dict[str, "Dataset"]]  # noqa #type:ignore
		] = None,
		tokenizer: Optional[
			"transformers.PreTrainedTokenizerBase"  # noqa #type:ignore
		] = None,
		dataset_num_proc: Optional[int] = None,
		_do_init_fns: bool = True,
		dataset_map_arguments: Optional[Dict[str, Any]] = None,
		low_mem_usage: bool = False,
		apply_chat_template: bool = False,
	):
		"""
		Initializes the ORPOTrainer.

		Args:
		    arguments (TrainingArguments): The training arguments.
		    max_length (Optional[int], optional): The maximum sequence length. Defaults to None.
		    max_prompt_length (Optional[int], optional): The maximum prompt length. Defaults to None.
		    max_completion_length (Optional[int], optional): The maximum completion length. Defaults to None.
		    beta (float, optional): The strength of the regularization term in the ORPO loss. Defaults to 0.1.
		    disable_dropout (bool, optional): Whether to disable dropout during training. Defaults to True.
		    label_pad_token_id (int, optional): The ID of the padding token for labels. Defaults to -100.
		    is_encoder_decoder (bool, optional): Whether the model is an encoder-decoder architecture. Defaults to False.
		    padding_value (int, optional): The padding value for input sequences. Defaults to None.
		    data_collator (Optional[DPODataCollatorWithPadding], optional): The data collator used for batching. Defaults to None.
		    train_dataset (Optional[Dataset], optional): The training dataset. Defaults to None.
		    eval_dataset (Optional[Union[Dataset, Dict[str, Dataset]]], optional): The evaluation dataset. Defaults to None.
		    tokenizer (Optional[PreTrainedTokenizerBase], optional): The tokenizer used for preprocessing. Defaults to None.
		    dataset_num_proc (Optional[int], optional): The number of processes to use for dataset mapping. Defaults to None.
		    _do_init_fns (bool, optional): Whether to automatically initialize trainer functions. Defaults to True.
		    dataset_map_arguments (Optional[Dict[str, Any]], optional): Arguments to pass to the dataset `map` function for tokenization. Defaults to None.
		    low_mem_usage (bool, optional): Whether to prioritize low memory usage during training. Defaults to False.
		    apply_chat_template (bool): Whether to apply chat template from tokenizer on `rejected` and `chosen` fields in dataset.

		Raises:
		    ValueError: If `arguments` is not provided or is not a `TrainingArguments` instance, or if `tokenizer` is not provided.
		"""

		assert arguments is not None, (
			"You Have to pass arguments that will be used for training but you have passed"
			"`arguments=None`"
		)
		assert isinstance(
			arguments, TrainingArguments
		), f"arguments type must be `TrainingArguments` but got {type(arguments)}"

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

		padding_value = (
			padding_value if padding_value is not None else tokenizer.pad_token_id
		)
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
		self.dataset_num_proc = dataset_num_proc
		self.apply_chat_template = apply_chat_template
		data_collator = (
			DPODataCollatorWithPadding(
				max_prompt_length=self.max_prompt_length,
				max_completion_length=self.max_completion_length,
				pad_token_id=tokenizer.pad_token_id,
				label_pad_token_id=label_pad_token_id,
				is_encoder_decoder=False,
			)
			if data_collator is None
			else data_collator
		)
		self._stored_metrics = defaultdict(lambda: defaultdict(list))
		if dataset_map_arguments is None:
			dataset_map_arguments = {}
		with jax.default_device(jax.devices("cpu")[0]):
			train_dataset = train_dataset.map(
				self.tokenize_row,
				num_proc=dataset_num_proc,
				**dataset_map_arguments,
			)
			if eval_dataset is not None:
				eval_dataset = eval_dataset.map(
					self.tokenize_row,
					num_proc=dataset_num_proc,
					**dataset_map_arguments,
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
		assert (
			padding_value is not None
		), "`padding_value` can not be set as `None` it must be an integer."

		self.concatenated_forward = create_orpo_concatenated_forward(
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
			_do_init_fns=_do_init_fns,
		)

	def build_tokenized_answer(self, prompt: str, answer: str) -> Dict[str, np.ndarray]:
		"""
		Tokenizes a prompt and answer pair, handling special tokens and padding/truncation.

		Args:
		    prompt (str): The prompt text.
		    answer (str): The answer text.

		Returns:
		    Dict[str, np.ndarray]: A dictionary containing the tokenized prompt and answer, along with attention masks.

		Raises:
		    ValueError: If there's a mismatch in token lengths.
		"""
		full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
		prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

		# Convert to numpy arrays for consistency
		full_input_ids = np.array(full_tokenized["input_ids"])
		prompt_input_ids = np.array(prompt_input_ids)

		response_token_ids_start_idx = len(prompt_input_ids)

		# Check for potential off-by-one error
		if not np.array_equal(
			prompt_input_ids, full_input_ids[:response_token_ids_start_idx]
		):
			response_token_ids_start_idx -= 1
			if not np.array_equal(
				prompt_input_ids, full_input_ids[:response_token_ids_start_idx]
			):
				raise ValueError("Mismatch in prompt tokenization")

		prompt_attention_mask = full_tokenized["attention_mask"][
			:response_token_ids_start_idx
		]
		answer_input_ids = full_input_ids[response_token_ids_start_idx:]
		answer_attention_mask = full_tokenized["attention_mask"][
			response_token_ids_start_idx:
		]

		if len(prompt_input_ids) != len(prompt_attention_mask):
			raise ValueError(
				"Prompt input ids and attention mask should have the same length."
			)

		return {
			"prompt_input_ids": prompt_input_ids.astype(np.int32),
			"prompt_attention_mask": np.array(prompt_attention_mask, dtype=np.int32),
			"input_ids": answer_input_ids.astype(np.int32),
			"attention_mask": np.array(answer_attention_mask, dtype=np.int32),
		}

	def tokenize_row(
		self,
		feature: Dict[str, str],
		state: Optional[object] = None,
	) -> Dict[str, np.ndarray]:
		"""
		Tokenizes a single row of data from the ORPO dataset.

		This method tokenizes the prompt, chosen response, and rejected response,
		handles padding and truncation, and prepares the data for input to the DPO model.

		Args:
		    feature (Dict): A dictionary containing the "prompt", "chosen", and "rejected" texts.
		    state (EasyDeLState, optional): Not used in this implementation. Defaults to None.

		Returns:
		    Dict: A dictionary containing the tokenized prompt, chosen response, and rejected response,
		          along with attention masks and labels.

		Raises:
		    ValueError: If the input data types are incorrect.
		"""
		prompt = self._validate_input(feature, "prompt")
		chosen = self._validate_input(feature, "chosen")
		rejected = self._validate_input(feature, "rejected")

		prompt_tokens = self._tokenize_prompt(prompt)
		chosen_tokens = self._tokenize_answer(prompt, chosen)
		rejected_tokens = self._tokenize_answer(prompt, rejected)

		chosen_sequence = self._create_sequence(prompt_tokens, chosen_tokens)
		rejected_sequence = self._create_sequence(prompt_tokens, rejected_tokens)

		return self._prepare_final_batch(
			prompt_tokens,
			chosen_sequence,
			rejected_sequence,
		)

	def _validate_input(self, feature: Dict[str, str], key: str) -> str:
		"""Validates input and returns the corresponding value."""
		value = feature[key]
		if self.apply_chat_template and key in ["chosen", "rejected"]:
			value = self.tokenizer.apply_chat_template(value, tokenize=False)
		if not isinstance(value, str):
			raise ValueError(f"{key} should be a string but got {type(value)}")
		return value

	def _tokenize_prompt(self, prompt: str) -> Dict[str, np.ndarray]:
		"""Tokenizes the prompt."""
		tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
		return {f"prompt_{k}": v for k, v in tokens.items()}

	def _tokenize_answer(self, prompt: str, answer: str) -> Dict[str, np.ndarray]:
		"""Tokenizes the answer in context of the prompt."""
		full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
		prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

		return {
			"prompt_input_ids": np.array(
				full_tokenized["input_ids"][:prompt_length], dtype=np.int32
			),
			"prompt_attention_mask": np.array(
				full_tokenized["attention_mask"][:prompt_length], dtype=np.int32
			),
			"input_ids": np.array(
				full_tokenized["input_ids"][prompt_length:], dtype=np.int32
			),
			"attention_mask": np.array(
				full_tokenized["attention_mask"][prompt_length:], dtype=np.int32
			),
		}

	def _create_sequence(
		self,
		prompt_tokens: Dict[str, np.ndarray],
		answer_tokens: Dict[str, np.ndarray],
	) -> Dict[str, np.ndarray]:
		"""Creates a full sequence by combining prompt and answer tokens."""
		sequence = {}
		for key in ["input_ids", "attention_mask"]:
			sequence[key] = np.concatenate(
				[
					self._add_special_token(
						prompt_tokens[f"prompt_{key}"],
						self.tokenizer.bos_token_id,
						start=True,
					),
					self._add_special_token(
						answer_tokens[key],
						self.tokenizer.eos_token_id,
						start=False,
					),
				],
				axis=1,
			)
		sequence["labels"] = self._create_labels(
			sequence["input_ids"], len(prompt_tokens["prompt_input_ids"])
		)
		return sequence

	def _add_special_token(
		self,
		array: np.ndarray,
		token_id: int,
		start: bool,
	) -> np.ndarray:
		"""Adds a special token to the start or end of an array."""
		token = np.array([[token_id]], dtype=np.int32)
		array = np.atleast_2d(array)
		return np.concatenate([token, array] if start else [array, token], axis=1)

	def _create_labels(self, input_ids: np.ndarray, prompt_length: int) -> np.ndarray:
		"""Creates labels for the sequence, masking the prompt part."""
		labels = input_ids.copy()
		labels[:, :prompt_length] = self.label_pad_token_id
		return labels

	def _prepare_final_batch(
		self,
		prompt_tokens: Dict[str, np.ndarray],
		chosen_sequence: Dict[str, np.ndarray],
		rejected_sequence: Dict[str, np.ndarray],
	) -> Dict[str, np.ndarray]:
		"""Prepares the final batch by padding and truncating sequences."""
		batch = {}
		for prefix, tokens in [
			("", prompt_tokens),
			("chosen_", chosen_sequence),
			("rejected_", rejected_sequence),
		]:
			for key, value in tokens.items():
				if key == "token_type_ids":
					continue
				max_length = (
					self.max_prompt_length if prefix == "" else self.max_completion_length
				)
				pad_value = self.padding_value if key in ["input_ids", "labels"] else 0
				batch[f"{prefix}{key}"] = self._pad_and_truncate(value, max_length, pad_value)
		return batch

	def _pad_and_truncate(
		self,
		array: np.ndarray,
		max_length: int,
		pad_value: int,
	) -> np.ndarray:
		"""Pads or truncates an array to the specified length."""
		if array.shape[1] < max_length:
			padding = np.full(
				(array.shape[0], max_length - array.shape[1]),
				pad_value,
				dtype=array.dtype,
			)
			return np.concatenate([array, padding], axis=1)
		return array[:, :max_length]

	def configure_functions(self) -> TrainerConfigureFunctionOutput:
		"""
		Configures and JIT-compiles the training and evaluation step functions.

		This method sets up the necessary functions for training and evaluation, including:
		    - Initialization of the model state.
		    - Sharding of the model parameters and optimizer state.
		    - JIT-compilation of the training and evaluation step functions.

		Returns:
		    TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
		"""
		mesh = self.arguments.get_mesh()

		def initialize_state_function():
			initialized_parameters = self.model.init_weights(
				jax.random.PRNGKey(0), self.arguments.init_input_shape
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
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
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
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
					module=self.model,
					module_config_args=None,
				)

		def create_state_from_params_function(parameters):
			if self.rapture is None:
				return EasyDeLState.create(
					tx=self.tx,
					params=parameters,
					apply_fn=self.model.__call__,
					module_config=copy.deepcopy(self.model.config),
					tx_init=copy.deepcopy(self.arguments.optimizer_kwargs),
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
					module=self.model,
					module_config_args=None,
				)
			else:
				return EasyDeLState(
					step=0,
					apply_fn=self.lora_apply_fn,
					params=parameters,
					tx=self.lora_tx,
					opt_state=self.lora_opt_state,
					tx_init=EasyDeLState.safe_dict(
						copy.deepcopy(self.arguments.optimizer_kwargs)
					),
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
					module=self.lora_model,
					module_config=self.model.config,
					module_config_args=None,
				)

		state_shape = jax.eval_shape(initialize_state_function)
		state_partition_spec = match_partition_rules(
			(
				self.config.get_partition_rules(
					fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
				)
				if self.arguments.custom_rule is None
				else self.arguments.custom_rule
			),
			state_shape,
		)

		spec_named_sharding = self.specs_to_name_sharding(state_partition_spec)
		empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=mesh)
		create_sharded_state_from_params_function = jit(
			create_state_from_params_function,
			in_shardings=(spec_named_sharding.params,),
			out_shardings=spec_named_sharding,
			donate_argnums=(0,),
		)
		sharded_train_step_function = jit(
			create_orpo_step_function(
				mode="train",
				beta=self.beta,
				concatenated_forward=self.concatenated_forward,
				batch_partition_spec=self.arguments.step_partition_spec,
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(
				spec_named_sharding,
				empty_sharding,
			),
		)

		sharded_eval_step_function = jit(
			create_orpo_step_function(
				mode="eval",
				beta=self.beta,
				concatenated_forward=self.concatenated_forward,
				batch_partition_spec=self.arguments.step_partition_spec,
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(
				spec_named_sharding,
				empty_sharding,
			),
		)

		self.arguments.ensure_checkpoint_path()
		checkpoint_manager = self.arguments.get_streaming_checkpointer()
		self.state_partition_spec = state_partition_spec
		self.state_named_sharding = spec_named_sharding
		self.state_shape = state_shape

		return TrainerConfigureFunctionOutput(
			create_sharded_state_from_params_function=create_sharded_state_from_params_function,
			sharded_train_step_function=sharded_train_step_function,
			sharded_eval_step_function=sharded_eval_step_function,
			mesh=mesh,
			checkpoint_manager=checkpoint_manager,
			initialize_state_function=initialize_state_function,
		)

	def initialize_state(
		self,
		model_parameters: Optional[flax.core.FrozenDict] = None,
		state: Optional[EasyDeLState] = None,
	) -> Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
		if (
			model_parameters is None
			and state is None
			and self.rapture is None
			and self.checkpoint_path is None
		):
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
				self.state_partition_spec, mesh=self.mesh
			)
			if state is not None:
				sharded_state = state
				if sharded_state.opt_state is None:
					logger.info("Optimizer State is not Found!, initializing one.")
					with jax.default_device(self.arguments.offload_device):
						sharded_state = sharded_state.init_opt_state()
			elif self.finetune:
				if model_parameters is None and self.checkpoint_path is not None:
					logger.info(f"Loading Model From {self.checkpoint_path}")
					with jax.default_device(self.arguments.offload_device):
						sharded_state = EasyDeLState.load_state(
							verbose=self.arguments.verbose,
							state_shard_fns=shard_fns,
							init_optimizer_state=True,
							checkpoint_path=self.checkpoint_path,
							input_shape=self.arguments.init_input_shape,
							config_kwargs=self.arguments.loaded_model_config_kwargs,
						)
						state_shape = jax.eval_shape(lambda: sharded_state)
						state_partition_spec = match_partition_rules(
							(
								self.config.get_partition_rules(
									fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
								)
								if self.arguments.custom_rule is None
								else self.arguments.custom_rule
							),
							state_shape,
						)

						spec_named_sharding = self.specs_to_name_sharding(state_partition_spec)
						empty_sharding = jax.sharding.NamedSharding(
							spec=PartitionSpec(), mesh=self.arguments.get_mesh()
						)
						sharded_train_step_function = jit(
							create_orpo_step_function(
								mode="train",
								beta=self.beta,
								concatenated_forward=self.concatenated_forward,
								batch_partition_spec=self.arguments.step_partition_spec,
							),
							in_shardings=(spec_named_sharding, empty_sharding),
							out_shardings=(
								spec_named_sharding,
								empty_sharding,
							),
						)

						sharded_eval_step_function = jit(
							create_orpo_step_function(
								mode="eval",
								beta=self.beta,
								concatenated_forward=self.concatenated_forward,
								batch_partition_spec=self.arguments.step_partition_spec,
							),
							in_shardings=(spec_named_sharding, empty_sharding),
							out_shardings=(
								spec_named_sharding,
								empty_sharding,
							),
						)

						self.state_partition_spec = state_partition_spec
						self.state_named_sharding = spec_named_sharding
						self.state_shape = state_shape
						self.sharded_train_step_function = sharded_train_step_function
						self.sharded_eval_step_function = sharded_eval_step_function

					if self.arguments.remove_ckpt_after_load:
						os.remove(self.checkpoint_path)
				elif model_parameters is not None and self.checkpoint_path is None:
					if not isinstance(model_parameters, flax.core.FrozenDict):
						logger.warn(
							"Model Parameters should be like FrozenDict({'params': params}) make sure to "
							"pass as type FrozenDict in case of not getting UnExcepted Errors ",
						)
					sharded_state = self.create_sharded_state_from_params_function(
						model_parameters
					)
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

			self.sharded_state = sharded_state
			return sharded_state, shard_fns, gather_fns

	def initialize_trainer_utils(self):
		"""
		Initializes various utilities used by the trainer.

		This includes setting up Weights & Biases, initializing the training timer,
		configuring dataloaders, configuring the model and optimizer, sharding the
		model and reference model states, and configuring the training and evaluation functions.
		"""
		self._initialize_wandb()
		self._initialize_timer()
		self._configure_dataloaders()
		self._configure_model()
		self._configure_functions()

	def _configure_dataloaders(self):
		"""
		Configures the dataloaders for training and evaluation.

		This method retrieves the dataloaders from the `configure_dataloaders` method,
		sets the maximum training and evaluation steps, and logs the time taken for
		this configuration.
		"""

		operation_name = "configure dataloaders"

		with self.timer(operation_name):
			dataset_configurations = self.configure_dataloaders()
			self.dataloader_train = dataset_configurations.dataloader_train
			self.max_training_steps = dataset_configurations.max_training_steps
			self.dataloader_eval = dataset_configurations.dataloader_eval
			self.max_evaluation_steps = dataset_configurations.max_evaluation_steps
		self.timer.log(operation_name)

	def _configure_model(self):
		"""
		Configures the model, optimizer, scheduler, and configuration.

		This method retrieves the model, optimizer, scheduler, and configuration from
		the `configure_model` method and configures LoRA (if enabled). It also logs
		the time taken for this configuration.
		"""
		operation_name = "configure Model, Optimizer, Scheduler and Config"
		with self.timer(operation_name):
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

		self.timer.log(operation_name)

	def _configure_functions(self):
		"""
		Configures and JIT-compiles the training and evaluation step functions.

		This method retrieves the configured functions from the `configure_functions`
		method, sets up the mesh, checkpoint manager, and state initialization
		function, and logs the time taken for this configuration.
		"""
		operation_name = "configure functions and sharding them"
		with self.timer(operation_name):
			function_configurations = self.configure_functions()
			self.create_sharded_state_from_params_function = (
				function_configurations.create_sharded_state_from_params_function
			)
			self.sharded_train_step_function = (
				function_configurations.sharded_train_step_function
			)
			self.sharded_eval_step_function = (
				function_configurations.sharded_eval_step_function
			)
			self.mesh = function_configurations.mesh
			self.checkpoint_manager = function_configurations.checkpoint_manager
			self.initialize_state_function = function_configurations.initialize_state_function
		self.timer.log(operation_name)

	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
	) -> Callable:
		return self.data_collator

	def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
		"""
		Configures the dataloaders for training and evaluation.

		This method creates the training and evaluation dataloaders using the provided
		datasets and data collator. It also determines the maximum number of training
		and evaluation steps based on the dataset sizes and training arguments.

		Returns:
		    TrainerConfigureDataloaderOutput: An object containing the configured dataloaders and the
		                                    maximum number of training and evaluation steps.
		"""
		dataloader_train = self.get_train_dataloader()
		max_evaluation_steps = None
		dataloader_eval = None

		max_training_steps = (
			self.arguments.num_train_epochs * len(dataloader_train)
			if self.arguments.max_training_steps is None
			else self.arguments.max_training_steps
		)
		if self.eval_dataset is not None:
			dataloader_eval = self.get_eval_dataloader(self.eval_dataset)
			max_evaluation_steps = len(dataloader_eval)
		return TrainerConfigureDataloaderOutput(
			dataloader_train=dataloader_train,  # type:ignore
			max_training_steps=max_training_steps,
			dataloader_eval=dataloader_eval,
			max_evaluation_steps=max_evaluation_steps,
		)

	def _get_train_dataloader(self) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Creates the training dataloader as a TensorFlow Dataset.

		This method retrieves the training dataset, applies the data collator, and converts
		it into a TensorFlow Dataset for efficient batching and data loading during training.

		Returns:
		    tensorflow.data.Dataset: The training dataloader.

		Raises:
		    ValueError: If the training dataset is not set.
		"""

		import tensorflow_datasets

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
				drop_remainder=True,
			)
		)

	def _get_eval_dataloader(
		self,
		eval_dataset: Optional["Dataset"] = None,  # noqa #type:ignore
	) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Creates the evaluation dataloader as a TensorFlow Dataset.

		This method retrieves the evaluation dataset (either provided as an argument or
		from the `self.eval_dataset` attribute), applies the data collator, and converts
		it into a TensorFlow Dataset for efficient batching and data loading during evaluation.

		Args:
		    eval_dataset (Optional[Dataset], optional):
		        An optional evaluation dataset to use. If None, `self.eval_dataset` is used. Defaults to None.

		Returns:
		    tensorflow.data.Dataset: The evaluation dataloader.

		Raises:
		    ValueError: If no evaluation dataset is provided or set.
		"""

		import tensorflow_datasets

		if eval_dataset is None and self.eval_dataset is None:
			raise ValueError("Trainer: evaluation requires an eval_dataset.")
		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

		return tensorflow_datasets.as_numpy(
			eval_dataset.to_tf_dataset(
				batch_size=self.arguments.eval_batch_size,
				collate_fn=self.data_collator,
				num_workers=self.arguments.dataloader_num_workers,
				shuffle=False,
				drop_remainder=True,
			)
		)

	def get_train_dataloader(self) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Returns the training dataloader

		Returns:
		    tensorflow.data.Dataset: The training dataloader.
		"""
		return self._get_train_dataloader()

	def get_eval_dataloader(
		self,
		eval_dataset: Optional["Dataset"] = None,  # noqa #type:ignore
	) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Returns the evaluation dataloader
		Args:
		    eval_dataset (Optional[Dataset], optional):
		        An optional evaluation dataset to use. If None, `self.eval_dataset` is used. Defaults to None.

		Returns:
		    tensorflow.data.Dataset: The evaluation dataloader.
		"""
		if eval_dataset is None and self.eval_dataset is None:
			raise ValueError("Trainer: evaluation requires an eval_dataset.")
		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
		return self._get_eval_dataloader(eval_dataset=eval_dataset)

	def train(
		self,
		model_parameters: Optional[flax.core.FrozenDict] = None,
		state: Optional[EasyDeLState] = None,
	) -> ORPOTrainerOutput:
		"""
		Trains the ORPO model.

		This method orchestrates the training process, iterating over epochs and batches,
		performing training steps, logging metrics, saving checkpoints, handling keyboard
		interrupts and timeouts, and optionally evaluating the model.

		Args:
		    model_parameters (Optional[flax.core.FrozenDict], optional):
		        Pretrained model parameters for initialization. Defaults to None.
		    state (Optional[EasyDeLState], optional):
		        An existing EasyDeLState to resume training from. Defaults to None.

		Returns:
		    ORPOTrainerOutput: An object containing the trained model state and other training information.
		"""

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
				color="red",
				force_color=True,
			)

		checkpoint_path = "SAVING_SKIPPED"
		if self.arguments.performance_mode:
			termcolor.cprint(
				"Performance Mode is ON, we will ignore the Memory Tracking, WANDB Logging, and extra information "
				"Process.",
				color="red",
				force_color=True,
			)
		sharded_state, shard_fns, gather_fns = self.initialize_state(
			model_parameters=model_parameters, state=state
		)
		self.model_state = sharded_state
		count_model_parameters(sharded_state.params)
		flops_per_device = (
			self.calculate_number_total_flops_per_device(params=sharded_state.params) / 1e12
		)
		with self.mesh:
			with (
				jax.default_device(jax.devices("cpu")[0])
				if self.low_mem_usage
				else leave_alone_context_manager()
			):
				checkpoint_path = "SAVING_SKIPPED"

				pbar = tqdm(total=self.max_training_steps)
				pbar.set_description("Training")
				current_step = (
					self.model_state.step.tolist()
					if isinstance(self.model_state.step, jax.Array)
					else self.model_state.step
				)

				loss_sum = None
				filename = None

				try:
					for epoch_index in range(self.arguments.num_train_epochs):
						for batch in self.dataloader_train:
							if self.arguments.step_start_point > current_step:
								...
							elif current_step < self.max_training_steps:
								time_start = time.time()
								# for k, v in batch.items():
								#     print(k, v.shape)
								#     try:
								#         print(self.tokenizer.decode(v[0]))
								#     except:
								#         ...
								# # print()
								self.model_state, outputs = self.sharded_train_step_function(
									self.model_state, batch
								)
								total_time = time.time() - time_start
								flops = flops_per_device / total_time
								(loss, metrics) = outputs.loss, outputs.metrics
								loss.block_until_ready()
								loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss

								train_metrics = {
									"train/loss": loss.tolist(),
									"train/mean_loss": loss_sum
									/ ((current_step + 1) - self.arguments.step_start_point),
									"train/learning_rate": self.scheduler(
										jax.device_get(self.model_state.step)
									).tolist(),
									"train/step": current_step,
									"train/step_time": total_time,
									"train/perplexity": jnp.exp(loss).tolist(),
									"train/epoch": epoch_index,
									"train/TFLOPs": flops,
								}
								train_metrics.update(metrics)
								log_metrics = copy.deepcopy(train_metrics)
								train_metrics.update(self.arguments._captured_memory)
								self.arguments.log_metrics(
									metrics=train_metrics,
									step=current_step,
								)
								pbar.update(1)
								pbar.set_postfix(
									**{k.replace("train/", ""): v for k, v in log_metrics.items()}
								)
							else:
								break

							current_step += 1
				except KeyboardInterrupt:
					termcolor.cprint(
						"KeyboardInterrupt At training model Will return Current State of the Model with Parameters.",
						color="cyan",
						force_color=True,
					)

				except EasyDeLTimerError:
					termcolor.cprint(
						"Training reached out maximum training Time Killing training Process "
						"and Will return Current State of the Model with Parameters.",
						color="cyan",
						force_color=True,
					)

				if self.arguments.merge_lora_rapture_parameters and self.rapture is not None:
					print(
						termcolor.colored("Info : ", color="red", force_color=True),
						termcolor.colored(
							"Merging LoRA Parameters.", color="white", force_color=True
						),
					)
					self.model_state = self.model_state.replace(
						params=self.rapture.merge_parameters(self.model_state.params)
					)

				shard_fns, gather_fns = make_shard_and_gather_fns(
					partition_specs=match_partition_rules(
						rules=self.model_state.module_config.get_partition_rules(
							self.arguments.fully_sharded_data_parallel
						),
						params=jax.eval_shape(lambda: self.model_state),
					),
					mesh=self.mesh,
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
							(
								self.config.get_partition_rules(
									fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
								)
								if self.arguments.custom_rule is None
								else self.arguments.custom_rule
							),
							jax.eval_shape(lambda: self.model_state),
						),
						mesh=self.mesh,
					)  # You have to re-init the new shard and gather functions in order to be able to skip LoRA weight
					# crashing errors and saving errors
					filename = self._save_state(state=self.model_state, gather_fns=gather_fns)
					checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"

				if self.arguments.do_eval:
					for _ in self.eval(self.model_state):
						...

				output.checkpoint_path = checkpoint_path
				output.last_save_file_name = filename
				self.finish()

		return output

	def eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
		"""
		Evaluates the ORPO model using the provided model state.

		This method iterates over the evaluation dataset, performs evaluation steps,
		calculates metrics, logs metrics, and yields a dictionary of metrics for each step.

		Args:
		    model_state (EasyDeLState): The EasyDeLState object containing the model parameters
		                                and other relevant information.

		Yields:
		    Iterator[dict]: An iterator that yields a dictionary of evaluation metrics for each step.

		Raises:
		    AssertionError: If the evaluation dataset is not set.
		"""
		assert (
			self.eval_dataset is not None
		), "`dataloader_eval` is required by evaluator function."
		with self.mesh:
			pbar = tqdm(total=self.max_evaluation_steps)
			pbar.set_description("Evaluating")
			current_step = 0
			flops_per_device = (
				self.calculate_number_total_flops_per_device(params=model_state.params) / 1e12
			)
			loss_sum = None
			try:
				for batch in self.dataloader_eval:
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

					_, outputs = self.sharded_eval_step_function(model_state, batch)
					total_time = time.time() - time_start
					flops = flops_per_device / total_time
					loss, metrics = outputs.loss, outputs.metrics
					loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
					eval_metrics = {
						"eval/loss": loss.tolist(),
						"eval/mean_loss": loss_sum
						/ ((current_step + 1) - self.arguments.step_start_point),
						"eval/step": current_step,
						"eval/step_time": total_time,
						"eval/perplexity": jnp.exp(loss).tolist(),
						"eval/TFLOPs": flops,
					}
					eval_metrics.update(metrics)
					log_metrics = copy.deepcopy(eval_metrics)
					eval_metrics.update(self.arguments._captured_memory)
					self.arguments.log_metrics(metrics=eval_metrics, step=current_step)

					pbar.update(1)
					pbar.set_postfix(
						**{k.replace("eval/", ""): v for k, v in log_metrics.items()}
					)
					yield eval_metrics
					current_step += 1
			except KeyboardInterrupt:
				termcolor.cprint(
					"KeyboardInterrupt At Evaluation model Will return Nothing and just pass.",
					color="cyan",
					force_color=True,
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
					repr_src = f"  {k} : " + v.__str__().replace("\n", "\n  ") + "\n"
					string += (
						repr_src
						if len(repr_src) < 350
						else f"  {k} : " + f"{v.__class__.__name__}(...)" + "\n"
					)
				except TypeError:
					repr_src = f"\t{k} : " + "EasyDeLReadingError" + "\n"
					string += (
						repr_src
						if len(repr_src) < 350
						else f"  {k} : " + f"{v.__class__.__name__}(...)" + "\n"
					)

		return string + ")"

	def __str__(self):
		"""
		The __str__ function is called when you use the print function or when str() is used.
		It should return a string representation of the object.

		:param self: Refer to the instance of the class
		:return: The object's string representation
		"""
		return self.__repr__()
