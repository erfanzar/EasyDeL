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
	TrainerConfigureDataloaderOutput,
	TrainerConfigureFunctionOutput,
	TrainerConfigureModelOutput,
)
from easydel.trainers.direct_preference_optimization_trainer.dpo_config import DPOConfig
from easydel.trainers.direct_preference_optimization_trainer.jax_funcs import (
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
	leave_alone_context_manager,
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
		train_dataset: Optional["Dataset"] = None,  # noqa # type:ignore
		eval_dataset: Optional["Dataset"] = None,  # noqa # type:ignore
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

		train_dataset = train_dataset.map(
			maybe_extract_prompt,
			num_proc=arguments.dataset_num_proc,
		)
		train_dataset = train_dataset.map(
			maybe_apply_chat_template,
			fn_kwargs={"tokenizer": processing_class},
			num_proc=arguments.dataset_num_proc,
		)
		if eval_dataset is not None:
			eval_dataset = eval_dataset.map(
				maybe_extract_prompt, num_proc=arguments.dataset_num_proc
			)
			eval_dataset = eval_dataset.map(
				maybe_apply_chat_template,
				fn_kwargs={"tokenizer": processing_class},
				num_proc=arguments.dataset_num_proc,
			)

		fn_kwargs = {
			"tokenizer": self.tokenizer,
			"processor": None,
		}
		_tokenize = build_tokenize(
			model=model_state if arguments.is_encoder_decoder else None,
			args=arguments,
		)
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

	def train(self) -> DPOTrainerOutput:
		"""
		Trains the DPO model.

		This method orchestrates the training process, iterating over epochs and batches,
		performing training steps, logging metrics, saving checkpoints, handling keyboard
		interrupts and timeouts, and optionally evaluating the model.

		Returns:
		    DPOTrainerOutput: An object containing the trained model state and other training information.

		Raises:
		    AssertionError: If the model state is None.
		"""
		assert (
			self.model_state is not None
		), "model_state can not be None for training purpose"
		with self.mesh:
			with (
				jax.default_device(jax.devices("cpu")[0])
				if self.low_mem_usage
				else leave_alone_context_manager
			):
				checkpoint_path = "SAVING_SKIPPED"
				flops_per_device = (
					self.calculate_number_total_flops_per_device(params=self.model_state.params)
					/ 1e12
				)
				pbar = tqdm(total=self.max_training_steps)
				pbar.set_description("Training")
				current_step = (
					self.model_state.step.tolist()
					if isinstance(self.model_state.step, jax.Array)
					else self.model_state.step
				)

				loss_sum = None
				chosen_rewards_sum = None
				rejected_rewards_sum = None
				filename = None

				try:
					for epoch_index in range(self.arguments.num_train_epochs):
						for batch in self.dataloader_train:
							if self.arguments.step_start_point > current_step:
								...
							elif current_step < self.max_training_steps:
								for k, v in batch.items():
									break
								time_start = time.time()
								self.model_state, metrics = self.sharded_train_step_function(
									self.model_state, batch
								)
								total_time = time.time() - time_start
								flops = flops_per_device / total_time
								(loss, chosen_rewards, rejected_rewards) = (
									metrics.loss,
									metrics.chosen_rewards[0],
									metrics.rejected_rewards[0],
								)
								loss.block_until_ready()
								loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss

								rejected_rewards_sum = (
									rejected_rewards.tolist()
									if (rejected_rewards_sum is None)
									else rejected_rewards_sum + rejected_rewards
								)
								chosen_rewards_sum = (
									chosen_rewards.tolist()
									if (chosen_rewards_sum is None)
									else chosen_rewards_sum + chosen_rewards
								)
								train_metrics = {
									"train/loss": loss.tolist(),
									"train/mean_loss": loss_sum
									/ ((current_step + 1) - self.arguments.step_start_point),
									"train/mean_rejected_rewards": rejected_rewards_sum
									/ ((current_step + 1) - self.arguments.step_start_point),
									"train/mean_chosen_rewards": chosen_rewards_sum
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
						rules=self.model_state.module.config.get_partition_rules(
							self.arguments.fully_sharded_data_parallel
						),
						params=jax.eval_shape(lambda: self.model_state),
					),
					mesh=self.mesh,
				)
				output = DPOTrainerOutput(
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
		Evaluates the DPO model using the provided model state.

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
			loss_sum = None
			chosen_rewards_sum = None
			rejected_rewards_sum = None
			flops_per_device = (
				self.calculate_number_total_flops_per_device(params=self.model_state.params)
				/ 1e12
			)

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

					metrics = self.sharded_eval_step_function(model_state, batch)
					total_time = time.time() - time_start
					flops = flops_per_device / total_time
					(loss, chosen_rewards, rejected_rewards) = (
						metrics.loss,
						metrics.chosen_rewards[0],
						metrics.rejected_rewards[0],
					)

					loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
					rejected_rewards_sum = (
						rejected_rewards.tolist()
						if (rejected_rewards_sum is None)
						else rejected_rewards_sum + rejected_rewards
					)
					chosen_rewards_sum = (
						chosen_rewards.tolist()
						if (chosen_rewards_sum is None)
						else chosen_rewards_sum + chosen_rewards
					)

					eval_metrics = {
						"eval/loss": loss.tolist(),
						"eval/mean_loss": loss_sum
						/ ((current_step + 1) - self.arguments.step_start_point),
						"eval/mean_rejected_rewards": rejected_rewards_sum
						/ ((current_step + 1) - self.arguments.step_start_point),
						"eval/mean_chosen_rewards": chosen_rewards_sum
						/ ((current_step + 1) - self.arguments.step_start_point),
						"eval/step": current_step,
						"eval/step_time": total_time,
						"eval/perplexity": jnp.exp(loss).tolist(),
						"eval/TFLOPs": flops,
					}
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
