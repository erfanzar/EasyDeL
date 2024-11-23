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

from __future__ import annotations

import copy
import os
import time
import typing
import warnings
from abc import ABC
from collections import defaultdict
from functools import partial  # noqa
from typing import Any, Callable, Dict, Mapping, Optional

import flax.core
import jax
import termcolor
from fjformer.sharding import make_shard_and_gather_fns, match_partition_rules
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm
from transformers import PreTrainedTokenizerBase

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import get_logger
from easydel.trainers.base_trainer import (
	BaseTrainer,
	MetricsTracker,
	StepMetrics,
	TrainerConfigureDataloaderOutput,
	TrainerConfigureFunctionOutput,
	TrainerConfigureModelOutput,
)
from easydel.trainers.direct_preference_optimization_trainer.dpo_config import DPOConfig
from easydel.trainers.direct_preference_optimization_trainer.func_utils import (
	create_dpo_concatenated_forward,
	create_dpo_eval_function,
	create_dpo_train_function,
)
from easydel.trainers.direct_preference_optimization_trainer.modelling_output import (
	DPOTrainerOutput,
)
from easydel.trainers.direct_preference_optimization_trainer.utils import (
	DPODataCollatorWithPadding,
	build_tokenize,
)
from easydel.trainers.prompt_utils import (
	maybe_apply_chat_template,
	maybe_extract_prompt,
)

logger = get_logger(__name__)


class DPOTrainer(BaseTrainer, ABC):
	"""
	Trainer for Direct Preference Optimization (DPO).

	This trainer handles the training, evaluation, and checkpointing of language models
	using the DPO algorithm. It supports sharding, gradient accumulation, mixed precision
	training, LoRA, and precomputed reference model log probabilities.

	Attributes:
			arguments (DPOConfig): The dpo training config.
			model_state (EasyDeLState): The EasyDeLState object for the model being trained.
			ref_model_state (Optional[EasyDeLState]): The EasyDeLState object for the reference model (if used).
			beta (float): The strength of the regularization term in the DPO loss.
			label_smoothing (float): The amount of label smoothing to apply.
			loss_type (Literal["sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust", "bco_pair", "sppo_hard", "aot", "aot_pair", "apo_zero", "apo_down"]): The type of loss function to use.
			label_pad_token_id (int): The ID of the padding token for labels.
			padding_value (int): The padding value for input sequences.
			train_dataset (Optional[Dataset]): The training dataset.
			eval_dataset (Optional[Union[Dataset, Dict[str, Dataset]]]): The evaluation dataset.
			tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer used for preprocessing.
			data_collator (Optional[Callable]): The data collator used for batching.
			max_length (Optional[int]): The maximum sequence length.
			max_prompt_length (Optional[int]): The maximum prompt length.
			max_completion_length (Optional[int]): The maximum target length.
			precompute_ref_log_probs (bool): Whether to precompute reference model log probabilities.
			reference_free (bool): Whether to use a reference-free DPO variant.
			is_encoder_decoder (bool): Whether the model is an encoder-decoder architecture.
			dataset_map_arguments (Optional[dict]): Arguments to pass to the dataset `map` function for tokenization.
			low_mem_usage (bool): Whether to prioritize low memory usage during training.
			auto_fix_data (bool): Whether to automatically fix data issues.
			_do_init_fns (bool): Whether to automatically initialize trainer functions.

	Methods:
			initialize_trainer_utils(self): Initializes trainer utilities (logging, timer, dataloaders, model, etc.).
			configure_dataloaders(self) -> TrainerConfigureDataloaderOutput: Configures the dataloaders for training and evaluation.
			configure_model(self) -> TrainerConfigureModelOutput: Configures the model, optimizer, scheduler, and configuration.
			configure_functions(self) -> TrainerConfigureFunctionOutput: Configures and JIT-compiles the training and evaluation step functions.
			_configure_lora(self): Configures LoRA if enabled.
			shard_states(self, state: EasyDeLState, rules: Any) -> EasyDeLState: Shards the provided state according to the given rules.
			create_collect_function(self, max_sequence_length: int, truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end") -> Callable:
					Creates a data collection function for batching.
			_get_train_dataloader(self) -> tensorflow.data.Dataset: Creates the training dataloader.
			_get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> tensorflow.data.Dataset: Creates the evaluation dataloader.
			get_train_dataloader(self) -> tensorflow.data.Dataset: Returns the training dataloader, potentially with precomputed reference log probabilities.
			get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> tensorflow.data.Dataset: Returns the evaluation dataloader, potentially with precomputed reference log probabilities.
			compute_reference_log_probs(self, state: EasyDeLState, padded_batch: Dict) -> tuple[Any, Any]: Computes log probabilities for the chosen and rejected responses using the reference model.
			_save_state(self, state: EasyDeLState, gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]], milestone: bool = False) -> str:
					Saves the model state to a checkpoint file.
			train(self) -> DPOTrainerOutput: Trains the DPO model and returns the training output.
			eval(self, model_state: EasyDeLState) -> Iterator[dict]: Evaluates the DPO model and yields evaluation metrics.
	"""

	def __init__(
		self,
		arguments: DPOConfig,
		model_state: EasyDeLState,
		ref_model_state: Optional[EasyDeLState] = None,
		tokenizer: Optional[PreTrainedTokenizerBase] = None,
		train_dataset: Optional["datasets.Dataset"] = None,  # type:ignore #noqa
		eval_dataset: Optional["datasets.Dataset"] = None,  # type:ignore #noqa
		data_collator: Optional[Callable] = None,
		dataset_map_arguments: Optional[dict] = None,
		low_mem_usage: bool = True,
		auto_fix_data: bool = True,
		_do_init_fns: bool = True,
	):
		assert arguments is not None, (
			"You Have to pass arguments that will be used for training but you have passed"
			"`arguments=None`"
		)
		assert isinstance(
			arguments, DPOConfig
		), f"arguments type must be `DPOConfig` but got {type(arguments)}"

		assert (
			tokenizer is not None
		), "tokenizer must be specified to tokenize a DPO dataset."
		self.arguments = arguments
		self.auto_fix_data = auto_fix_data
		self.truncation_mode = arguments.truncation_mode
		self.tokenizer = tokenizer
		self.is_encoder_decoder = False
		self._precomputed_train_ref_log_probs = False
		self._precomputed_eval_ref_log_probs = False
		self.low_mem_usage = low_mem_usage

		arguments.padding_value = (
			arguments.padding_value
			if arguments.padding_value is not None
			else tokenizer.pad_token_id
		)
		data_collator = (
			DPODataCollatorWithPadding(
				max_prompt_length=arguments.max_prompt_length,
				max_completion_length=arguments.max_completion_length,  # type: ignore
				pad_token_id=tokenizer.pad_token_id,  # type: ignore
				label_pad_token_id=arguments.label_pad_token_id,
				is_encoder_decoder=arguments.is_encoder_decoder,
			)
			if data_collator is None
			else data_collator
		)
		self._stored_metrics = defaultdict(lambda: defaultdict(list))

		if dataset_map_arguments is None:
			dataset_map_arguments = {}

		processing_class = tokenizer

		if train_dataset[-1].get("chosen", None) is None:
			warnings.warn(
				"couldn't find `chosen` column in dataset"
				", this might make DPOTrainer break down if you haven't customize trainer.",
				stacklevel=2,
			)
		if train_dataset[-1].get("rejected", None) is None:
			warnings.warn(
				"couldn't find `rejected` column in dataset"
				", this might make DPOTrainer break down if you haven't customize trainer.",
				stacklevel=2,
			)
		if train_dataset[-1].get("prompt", None) is None:
			warnings.warn(
				"couldn't find `prompt` column in dataset"
				", this might make DPOTrainer break down if you haven't customize trainer.",
				stacklevel=2,
			)
		train_dataset = train_dataset.map(
			maybe_extract_prompt,
			num_proc=arguments.dataset_num_proc,
			desc="Extracting Prompts",
		)
		train_dataset = train_dataset.map(
			maybe_apply_chat_template,
			fn_kwargs={"tokenizer": processing_class},
			num_proc=arguments.dataset_num_proc,
			desc="Apply Chat Template",
		)
		if eval_dataset is not None:
			eval_dataset = eval_dataset.map(
				maybe_extract_prompt,
				num_proc=arguments.dataset_num_proc,
				desc="Eval - Extracting Prompts",
			)
			eval_dataset = eval_dataset.map(
				maybe_apply_chat_template,
				fn_kwargs={"tokenizer": processing_class},
				num_proc=arguments.dataset_num_proc,
				desc="Eval - Apply Chat Template",
			)
		fn_kwargs = {
			"tokenizer": self.tokenizer,
			"processor": None,
		}
		_tokenize = build_tokenize(
			model=model_state if arguments.is_encoder_decoder else None,
			args=arguments,
		)

		def to_jax_arrays(x):
			return {k: jnp.array(v) for k, v in x.items()}

		train_dataset = train_dataset.map(
			_tokenize,
			fn_kwargs=fn_kwargs,
			batched=True,
			num_proc=arguments.dataset_num_proc,
			writer_batch_size=10,
			desc="Tokenizing train dataset",
		)
		if eval_dataset is not None:
			eval_dataset = eval_dataset.map(
				_tokenize,
				fn_kwargs=fn_kwargs,
				batched=True,
				num_proc=arguments.dataset_num_proc,
				writer_batch_size=10,
				desc="Tokenizing eval dataset",
			)

		self.arguments = arguments

		self.is_in_train = False
		self.data_collator = data_collator
		self.train_dataset = train_dataset
		self.eval_dataset = eval_dataset
		self.tokenizer = tokenizer
		self.ref_model_state = ref_model_state
		self.model_state = model_state
		self._loggers_initialized = False
		self.mesh = self.arguments.get_mesh()

		self.concatenated_forward = jax.jit(
			create_dpo_concatenated_forward(
				is_encoder_decoder=arguments.is_encoder_decoder,
				padding_value=arguments.padding_value,
				label_pad_token_id=arguments.label_pad_token_id,
				truncation_mode=arguments.truncation_mode,
			),
			static_argnums=[0],
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
		self._shard_states()
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
		with self.timer("configure Model, Optimizer, Scheduler and Config"):
			model_configurations = self.configure_model()
			self.model = model_configurations.model
			self.tx = model_configurations.tx
			self.scheduler = model_configurations.scheduler
			self.config = model_configurations.config
			self._configure_lora()
		self.timer.log("configure Model, Optimizer, Scheduler and Config")

	def _configure_functions(self):
		"""
		Configures and JIT-compiles the training and evaluation step functions.

		This method retrieves the configured functions from the `configure_functions`
		method, sets up the mesh, checkpoint manager, and state initialization
		function, and logs the time taken for this configuration.
		"""
		operation_name = "configure functions and sharding them"
		with self.timer(operation_name):
			functions = self.configure_functions()

			self.create_sharded_state_from_params_function = (
				functions.create_sharded_state_from_params_function
			)
			self.sharded_train_step_function = functions.sharded_train_step_function
			self.sharded_eval_step_function = functions.sharded_eval_step_function
			self.mesh = functions.mesh
			self.checkpoint_manager = functions.checkpoint_manager
			self.initialize_state_function = functions.initialize_state_function
		self.timer.log(operation_name)

	def _configure_lora(self):
		"""
		Configures LoRA (Low-Rank Adaptation) if enabled in the training arguments.

		This method applies LoRA to the model, sets up the LoRA parameters, apply function,
		optimizer state, model, and optimizer, and logs the time taken for this configuration.
		"""
		if self.rapture is not None:
			lora_modules = self.rapture.apply_lora(
				module=self.model,
				parameters=self.arguments.rapture_config.parameters,
				tx=self.tx,
			)
			self.lora_parameters = lora_modules.lora_parameters
			self.lora_apply_fn = lora_modules.lora_module.__call__
			self.lora_opt_state = lora_modules.lora_opt_state
			self.lora_model = lora_modules.lora_module
			self.lora_tx = lora_modules.lora_tx

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
			dataloader_eval=dataloader_eval,
			dataloader_train=dataloader_train,
			max_evaluation_steps=max_evaluation_steps,
			max_training_steps=max_training_steps,
		)

	def configure_model(self) -> TrainerConfigureModelOutput:
		"""
		Configures the model, optimizer, scheduler, and configuration.

		This method retrieves the model configuration from the model state, creates
		the optimizer and scheduler using the training arguments, and returns an
		object containing the configured model, optimizer, scheduler, and configuration.

		Returns:
				TrainerConfigureModelOutput: An object containing the configured model, optimizer, scheduler, and configuration.
		"""
		config = self.model_state.module.config
		tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_training_steps)
		model = (self.model_state.module,)
		return TrainerConfigureModelOutput(
			model=model, tx=tx, scheduler=scheduler, config=config
		)

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

		def initialize_state_function():
			"""
			Initializes the EasyDeLState object, which holds model parameters, optimizer state, and other training information.

			Returns:
					EasyDeLState: The initialized EasyDeLState object.
			"""
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
					module_config=self.model_state.module.config,
					module_config_args=None,
				)
			else:
				return EasyDeLState.create(
					tx=tx,
					params=parameters,
					apply_fn=self.model.__call__,
					module_config=copy.deepcopy(self.model_state.module.config),
					tx_init=tx_init,
					hyperparameters=EasyDeLState.create_hyperparameters(
						self.model.config.model_type
					),
					module=self.model,
					module_config_args=None,
				)

		def create_state_from_params_function(parameters):
			"""
			Creates an EasyDeLState object from given parameters.

			This function is used when loading a model from pretrained parameters
			or a checkpoint.

			Args:
					parameters (FrozenDict): The model parameters.

			Returns:
					EasyDeLState: The EasyDeLState object initialized with the provided parameters.
			"""
			if self.rapture is None:
				return EasyDeLState.create(
					tx=self.tx,
					params=parameters,
					apply_fn=self.model.__call__,
					module_config=copy.deepcopy(self.model_state.module.config),
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
					module_config=self.model_state.module.config,
					module_config_args=None,
				)

		state_shape = jax.eval_shape(lambda: self.model_state)

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
			spec=PartitionSpec(),
			mesh=self.arguments.get_mesh(),
		)
		create_sharded_state_from_params_function = jax.jit(
			create_state_from_params_function,
			in_shardings=(spec_named_sharding.params,),
			out_shardings=spec_named_sharding,
			donate_argnums=(0,),
		)
		train_function = create_dpo_train_function(
			concatenated_forward=self.concatenated_forward,
			ref_state=self.ref_model_state,
			loss_type=self.arguments.loss_type,
			reference_free=self.arguments.reference_free,
			label_smoothing=self.arguments.label_smoothing,
			beta=self.arguments.beta,
		)
		sharded_train_step_function = jax.jit(
			train_function,
			in_shardings=(
				spec_named_sharding,
				jax.sharding.NamedSharding(
					spec=self.arguments.step_partition_spec,
					mesh=self.mesh,
				),
			),
			out_shardings=(spec_named_sharding, empty_sharding),
		)

		eval_function = create_dpo_eval_function(
			concatenated_forward=self.concatenated_forward,
			ref_state=self.ref_model_state,
			loss_type=self.arguments.loss_type,
			reference_free=self.arguments.reference_free,
			label_smoothing=self.arguments.label_smoothing,
			beta=self.arguments.beta,
		)

		sharded_eval_step_function = jax.jit(
			eval_function,
			in_shardings=(
				spec_named_sharding,
				jax.sharding.NamedSharding(
					spec=self.arguments.step_partition_spec,
					mesh=self.mesh,
				),
			),
			out_shardings=(spec_named_sharding, empty_sharding),
		)

		self.arguments.ensure_checkpoint_path()
		self.state_partition_spec = state_partition_spec
		self.state_named_sharding = spec_named_sharding
		self.state_shape = state_shape
		checkpoint_manager = self.arguments.get_streaming_checkpointer()
		mesh = self.arguments.get_mesh()
		return TrainerConfigureFunctionOutput(
			create_sharded_state_from_params_function=create_sharded_state_from_params_function,
			sharded_train_step_function=sharded_train_step_function,
			sharded_eval_step_function=sharded_eval_step_function,
			mesh=mesh,
			checkpoint_manager=checkpoint_manager,
			initialize_state_function=initialize_state_function,
		)

	def _shard_states(self):
		"""
		Shards the model and reference model states if automatic sharding is enabled.

		This method shards the `model_state` and `ref_model_state` using the sharding rules
		defined in the model configuration. It also initializes the optimizer and scheduler
		for the sharded model state.
		"""
		if self.model_state.tx is None or self.model_state.opt_state is None:
			inner_module_operation_name = "initializing TX and Schedulers for `model_state`"
			with self.timer(inner_module_operation_name):
				params_with_opt = (
					self.model_state.params["params"]
					if "_overwrite_with_gradient" in self.model_state.params
					else self.model_state.params
				)
				opt_state = self.tx.init(params_with_opt)

				self.model_state = self.model_state.replace(
					opt_state=opt_state,
					tx=self.tx,
				)
			self.timer.log(inner_module_operation_name)
		else:
			logger.info(
				"Found an existing TX and OptimizerState for "
				"model_state (ignore sharding and tx_init)."
			)

	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
	) -> Callable:
		"""
		Creates a data collection function for batching.

		For DPO training, this method simply returns the pre-configured `data_collator`.

		Args:
				max_sequence_length (int): The maximum sequence length (not used in this implementation).
				truncation_mode (typing.Literal["keep_end", "keep_start"], optional):
						The truncation mode (not used in this implementation). Defaults to "keep_end".

		Returns:
				Callable: The data collator function.
		"""
		return self.data_collator

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
		Returns the training dataloader, potentially with precomputed reference log probabilities.

		If `precompute_ref_log_probs` is enabled, this method computes the reference model's log
		probabilities for the chosen and rejected responses in the training dataset and adds
		them as columns to the dataset.

		Returns:
				tensorflow.data.Dataset: The training dataloader.
		"""

		import tensorflow_datasets

		if (
			self.arguments.precompute_ref_log_probs
			and not self._precomputed_train_ref_log_probs
		):
			data_loader = tensorflow_datasets.as_numpy(
				self.train_dataset.to_tf_dataset(
					batch_size=self.arguments.total_batch_size,
					collate_fn=self.data_collator,
					num_workers=self.arguments.dataloader_num_workers,
					shuffle=False,
					drop_remainder=True,
				)
			)
			reference_chosen_log_probs = []
			reference_rejected_log_probs = []
			for padded_batch in tqdm(
				iterable=data_loader, desc="Train dataset reference log probs"
			):
				reference_chosen_logp, reference_rejected_logp = (
					self.compute_reference_log_probs(
						self.model_state,
						padded_batch,
					)
				)
				reference_chosen_log_probs.append(reference_chosen_logp)
				reference_rejected_log_probs.append(reference_rejected_logp)

			all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
			all_reference_rejected_log_probs = jnp.concatenate(reference_rejected_log_probs)
			self.train_dataset = self.train_dataset.add_column(
				name="reference_chosen_log_probs", column=all_reference_chosen_log_probs
			)
			self.train_dataset = self.train_dataset.add_column(
				name="reference_rejected_log_probs",
				column=all_reference_rejected_log_probs,
			)

			self._precomputed_train_ref_log_probs = True
		return self._get_train_dataloader()

	def get_eval_dataloader(
		self,
		eval_dataset: Optional["Dataset"] = None,  # noqa #type:ignore
	) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Returns the evaluation dataloader, potentially with precomputed reference log probabilities.

		If `precompute_ref_log_probs` is enabled, this method computes the reference model's log
		probabilities for the chosen and rejected responses in the evaluation dataset and adds
		them as columns to the dataset.

		Args:
				eval_dataset (Optional[Dataset], optional):
						An optional evaluation dataset to use. If None, `self.eval_dataset` is used. Defaults to None.

		Returns:
				tensorflow.data.Dataset: The evaluation dataloader.
		"""

		import tensorflow_datasets

		if eval_dataset is None and self.eval_dataset is None:
			raise ValueError("Trainer: evaluation requires an eval_dataset.")
		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

		if (
			self.arguments.precompute_ref_log_probs
			and not self._precomputed_eval_ref_log_probs
		):
			# prepare dataloader
			data_loader = tensorflow_datasets.as_numpy(
				eval_dataset.to_tf_dataset(
					batch_size=self.arguments.eval_batch_size,
					collate_fn=self.data_collator,
					num_workers=self.arguments.dataloader_num_workers,
					shuffle=False,
					drop_remainder=True,
				)
			)

			reference_chosen_log_probs = []
			reference_rejected_log_probs = []
			for padded_batch in tqdm(
				iterable=data_loader, desc="Eval dataset reference log probs"
			):
				reference_chosen_logp, reference_rejected_logp = (
					self.compute_reference_log_probs(self.model_state, padded_batch)
				)
				reference_chosen_log_probs.append(reference_chosen_logp.cpu())
				reference_rejected_log_probs.append(reference_rejected_logp.cpu())

			all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
			all_reference_rejected_log_probs = jnp.concatenate(reference_rejected_log_probs)

			eval_dataset = eval_dataset.add_column(
				name="reference_chosen_log_probs", column=all_reference_chosen_log_probs
			)
			eval_dataset = eval_dataset.add_column(
				name="reference_rejected_log_probs",
				column=all_reference_rejected_log_probs,
			)

			if self.eval_dataset is not None:
				self.eval_dataset = eval_dataset
			self._precomputed_eval_ref_log_probs = True

		return self._get_eval_dataloader(eval_dataset=eval_dataset)

	def compute_reference_log_probs(
		self,
		state: EasyDeLState,
		padded_batch: Dict,
	) -> tuple[Any, Any]:
		"""
		Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset.

		Args:
				state (EasyDeLState): The EasyDeLState object of the model (used if no reference model is provided).
				padded_batch (Dict): The padded batch of data.

		Returns:
				tuple[Any, Any]: A tuple containing the log probabilities for the chosen and rejected responses.
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
		state: EasyDeLState,
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		milestone: bool = False,
	) -> str:
		"""
		Saves the model state to a checkpoint file.

		This method constructs the checkpoint file name, prints a message indicating the save operation,
		and uses the `save_state` method of the `EasyDeLState` object to save the state to disk.

		Args:
				state (EasyDeLState): The EasyDeLState object to be saved.
				gather_fns (Optional[Any | Mapping[str, Callable] | dict[Callable]]):
						Gather functions used to collect sharded data before saving.
				milestone (bool, optional): Whether this save is a milestone (e.g., end of epoch). Defaults to False.

		Returns:
				str: The filename of the saved checkpoint.
		"""
		step = (
			int(jax.device_get(state.step)) + self.arguments.step_start_point
			if self.arguments.step_start_point is not None
			else int(jax.device_get(state.step))
		)
		checkpoint_name = f"{self.arguments.model_name}-S{step}"
		filename = f"{checkpoint_name}_{step}" if milestone else f"{checkpoint_name}"
		filename += ".easy"
		termcolor.cprint(f"Saving Model {filename}.", color="red", force_color=True)
		state.save_state(
			filename=filename,
			checkpoint_dir=os.path.join(self.arguments.save_dir, self.arguments.model_name),
			gather_fns=gather_fns,
			float_dtype=self.dtype,
			verbose=self.arguments.verbose,
			save_optimizer=self.arguments.save_optimizer_state,
		)
		return filename

	def _run_training_loop(
		self,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		start_time: float,
		shard_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
	):
		"""Core training loop implementation."""
		pbar = tqdm(total=self.max_training_steps)
		current_step = int(jax.device_get(self.model_state.step))
		run_exception = None
		with self.mesh:
			for epoch in range(self.arguments.num_train_epochs):
				current_step, run_exception = self._train_epoch(
					self.dataloader_train,
					current_step,
					metrics_tracker,
					step_metrics,
					pbar,
					start_time,
					epoch,
					shard_fns,
					gather_fns,
				)

				if current_step >= self.max_training_steps:
					break
				if run_exception is not None:
					break
		return self._prepare_training_output(
			sharded_state=self.model_state,
			shard_fns=shard_fns,
			gather_fns=gather_fns,
			run_exception=run_exception,
		), run_exception

	def _run_evaluation(
		self,
		sharded_state: EasyDeLState,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		start_time: float,
	):
		"""Core evaluation loop implementation."""
		pbar = tqdm(total=self.max_evaluation_steps)
		pbar.set_description("evaluation process")
		current_step = int(jax.device_get(sharded_state.step))
		with self.mesh:
			for eval_metrics in self._eval_epoch(
				sharded_state,
				self.dataloader_eval,
				current_step,
				metrics_tracker,
				step_metrics,
				pbar,
				start_time,
			):
				yield eval_metrics

	def _train_epoch(
		self,
		train_dataset,
		current_step: int,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		pbar: tqdm,
		start_time: float,
		epoch: int,
		shard_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
	):
		"""Handles training for a single epoch."""
		train_iter = iter(train_dataset)
		for _ in range(self.max_training_steps // self.arguments.num_train_epochs):
			try:  # to make training loop safer if user wants to break that.
				batch = self._get_next_batch(train_iter)
				if self._should_skip_step(current_step):
					pbar.update(1)
					continue
				step_metrics.start_step()
			except (KeyboardInterrupt, EasyDeLTimerError, StopIteration) as exect:
				return self.model_state, current_step, exect

			# Execute training step
			loss, metrics, run_exception = self._execute_train_step(batch)
			# Update and log metrics
			try:
				mean_loss = metrics_tracker.update(loss, float("inf"), current_step)

				train_metrics = step_metrics.calculate(
					loss=loss,
					metrics=metrics,
					current_step=current_step,
					learning_rate=self.scheduler(current_step)
					if self.scheduler is not None
					else self.arguments.learning_rate,
					epoch=epoch,
					flops_per_device=getattr(self, "_flops_per_device", 0),
					batch_size=self.arguments.total_batch_size,
					seq_length=self.arguments.max_prompt_length
					+ self.arguments.max_completion_length * 2,
					mean_loss=mean_loss,
					mode="train",
				)

				self._log_metrics(
					metrics=train_metrics,
					pbar=pbar,
					step=current_step,
					mode="train",
				)

				# Save checkpoint if needed
				if self._should_save_checkpoint(current_step):
					_ = self._save_state(
						state=self.model_state,
						gather_fns=gather_fns,
						milestone=True,
						save_dir=self.arguments.save_dir,
					)

				current_step += 1
			except (KeyboardInterrupt, EasyDeLTimerError):
				return current_step, run_exception
			if run_exception is not None:
				break
		return current_step, run_exception

	def _eval_epoch(
		self,
		sharded_state: EasyDeLState,
		eval_dataset,
		current_step: int,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		pbar: tqdm,
		start_time: float,
	):
		"""Handles training for a single epoch."""
		eval_iter = iter(eval_dataset)
		for _ in range(self.max_evaluation_steps):
			try:
				batch = self._get_next_batch(eval_iter)
				step_metrics.start_step()
				loss, metrics = self._execute_eval_step(sharded_state, batch)
				mean_loss = metrics_tracker.update(
					loss,
					float("inf"),
					current_step,  # Disable accuracy
				)
				eval_metrics = step_metrics.calculate(
					loss=loss,
					metrics=metrics,
					current_step=current_step,
					learning_rate=0.000,
					epoch=0,
					flops_per_device=getattr(self, "_flops_per_device", 0),
					batch_size=self.arguments.total_batch_size,
					seq_length=self.arguments.max_prompt_length
					+ self.arguments.max_completion_length * 2,
					mean_loss=mean_loss,
					mode="eval",
				)
				self._log_metrics(
					metrics=eval_metrics,
					pbar=pbar,
					step=current_step,
					mode="eval",
				)
				current_step += 1

				yield eval_metrics
			except (KeyboardInterrupt, EasyDeLTimerError) as _:
				break

	def _execute_eval_step(self, state, batch):
		"""Execute a single eval step."""
		batch = {key: jnp.asarray(value) for key, value in batch.items()}
		dpo_out = self.sharded_eval_step_function(state, batch)
		loss = dpo_out.loss
		metrics = dict(
			loss=loss,
			chosen_rewards=dpo_out.chosen_rewards,
			rejected_rewards=dpo_out.rejected_rewards,
		)
		return loss, metrics

	def _execute_train_step(self, batch):
		"""Execute a single training step."""
		if self.pruning_module is not None:
			self.model_state = self.model_state.replace(
				params=self.pruning_module.pre_forward_update(
					self.model_state.params,
					self.model_state.opt_state,
				)
			)

		# Forward and backward pass
		try:
			batch = {key: jnp.asarray(value) for key, value in batch.items()}

			self.model_state, dpo_out = self.sharded_train_step_function(
				self.model_state, batch
			)
			# Apply post-gradient updates
			loss = dpo_out.loss
			metrics = dict(
				loss=loss,
				chosen_rewards=dpo_out.chosen_rewards,
				rejected_rewards=dpo_out.rejected_rewards,
			)
			if self.pruning_module is not None:
				self.model_state = self.model_state.replace(
					params=self.pruning_module.post_gradient_update(
						self.model_state.params,
						self.model_state.opt_state,
					)
				)

			return loss, metrics, None
		except (KeyboardInterrupt, EasyDeLTimerError) as run_exception:
			return loss, metrics, run_exception

	def _finalize_training(self, output, run_exception):
		"""Finalize training and prepare output."""
		if run_exception is None:
			if self.arguments.merge_lora_rapture_parameters and self.rapture:
				termcolor.cprint("Merging LoRA Parameters.", color="cyan", force_color=True)
				output.state = output.state.replace(
					params=self.rapture.merge_parameters(output.state.params)
				)

		if self.arguments.do_eval:
			for _ in self.eval(output.state):
				...

		self.finish()

		return output

	def train(self) -> DPOTrainerOutput:
		start_time = time.time()
		rules = self.model_state.module.config.get_partition_rules(
			self.arguments.fully_sharded_data_parallel
		)
		shard_fns, gather_fns = make_shard_and_gather_fns(
			partition_specs=match_partition_rules(
				rules=rules,
				params=jax.eval_shape(lambda: self.model_state),
			),
			mesh=self.mesh,
		)

		metrics_tracker = MetricsTracker()
		step_metrics = StepMetrics(self.arguments)

		# Setup initial metrics and logging
		self._setup_initial_metrics(self.model_state)

		output, run_exception = self._run_training_loop(
			metrics_tracker=metrics_tracker,
			step_metrics=step_metrics,
			start_time=start_time,
			shard_fns=shard_fns,
			gather_fns=gather_fns,
		)
		return self._finalize_training(output, run_exception)

	def eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
		"""
		Evaluates the DPO using the provided model state.

		This method iterates over the evaluation dataset, performs forward passes,
		calculates evaluation metrics, logs the metrics, and yields the metrics for
		each evaluation step.

		Args:
				model_state (EasyDeLState): The EasyDeLState object containing the model parameters
																		and other relevant information.

		Yields:
				Iterator[dict]: An iterator yielding a dictionary of evaluation metrics for each step.

		Raises:
				AssertionError: If `self.dataloader_eval` is not set (meaning the evaluation dataloader is missing).
		"""
		start_time = time.time()

		metrics_tracker = MetricsTracker()
		step_metrics = StepMetrics(self.arguments)

		for metrics in self._run_evaluation(
			sharded_state=model_state,
			metrics_tracker=metrics_tracker,
			step_metrics=step_metrics,
			start_time=start_time,
		):
			yield metrics


def check_unimplemented_abstract_methods():
	from inspect import getmembers, isfunction

	base_class = BaseTrainer
	derived_class = DPOTrainer
	abstract_methods = [
		method
		for method in getmembers(base_class, predicate=isfunction)
		if getattr(getattr(base_class, method[0]), "__isabstractmethod__", False)
	]

	not_implemented = [
		method[0]
		for method in abstract_methods
		if getattr(derived_class, method[0]) is getattr(base_class, method[0])
	]
	if len(not_implemented) != 0:
		warnings.warn(
			f"{DPOConfig.__name__} does not implement the following abstract methods: {', '.join(not_implemented)}. "
			f"Please ensure these methods are implemented in {DPOConfig.__name__}.",
			stacklevel=1,
			category=UserWarning,
		)


check_unimplemented_abstract_methods()
