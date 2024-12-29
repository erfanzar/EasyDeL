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

import time
import typing as tp
import warnings
from collections import defaultdict

import jax
from fjformer.sharding import make_shard_and_gather_fns, match_partition_rules
from jax import numpy as jnp
from jax.experimental import sparse
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import get_logger
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.utils import ProcessingClassType

from ..base_trainer import (
	BaseTrainer,
	TrainerConfigureDataloaderOutput,
	TrainerConfigureFunctionOutput,
	TrainerConfigureModelOutput,
)
from ..prompt_utils import maybe_apply_chat_template, maybe_extract_prompt
from ..trainer_protocol import MetricsTracker, StepMetrics
from .dpo_config import DPOConfig
from .func_utils import (
	create_concatenated_forward,
	create_eval_function,
	create_train_function,
)
from .modelling_output import DPOTrainerOutput
from .utils import DPODataCollatorWithPadding, build_tokenize

if tp.TYPE_CHECKING:
	from datasets import Dataset
else:
	Dataset = tp.Any

logger = get_logger(__name__)


class DPOTrainer(BaseTrainer):
	"""
	Trainer for Direct Preference Optimization (DPO).

	This trainer handles the training, evaluation, and checkpointing of language models
	using the DPO algorithm. It supports sharding, gradient accumulation, mixed precision
	training, LoRA, and precomputed reference model log probabilities.
	"""

	def __init__(
		self,
		arguments: DPOConfig,
		model: EasyDeLBaseModule,
		ref_model: tp.Optional[tp.Union[EasyDeLBaseModule, EasyDeLState]] = None,
		processing_class: tp.Optional[ProcessingClassType] = None,
		train_dataset: tp.Optional[Dataset] = None,
		eval_dataset: tp.Optional[Dataset] = None,
		data_collator: tp.Optional[tp.Callable] = None,
		dataset_map_arguments: tp.Optional[dict] = None,
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
			processing_class is not None
		), "processing_class must be specified to tokenize a DPO dataset."
		self.arguments = arguments
		self.auto_fix_data = auto_fix_data
		self.truncation_mode = arguments.truncation_mode
		self.processing_class = processing_class
		self.is_encoder_decoder = False
		self._precomputed_train_ref_log_probs = False
		self._precomputed_eval_ref_log_probs = False
		self.low_mem_usage = low_mem_usage

		arguments.padding_value = (
			arguments.padding_value
			if arguments.padding_value is not None
			else processing_class.pad_token_id
		)
		data_collator = (
			DPODataCollatorWithPadding(
				max_prompt_length=arguments.max_prompt_length,
				max_completion_length=arguments.max_completion_length,  # type: ignore
				pad_token_id=processing_class.pad_token_id,  # type: ignore
				label_pad_token_id=arguments.label_pad_token_id,
				is_encoder_decoder=arguments.is_encoder_decoder,
			)
			if data_collator is None
			else data_collator
		)
		self._stored_metrics = defaultdict(lambda: defaultdict(list))

		if dataset_map_arguments is None:
			dataset_map_arguments = {}

		processing_class = processing_class

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
			fn_kwargs={"processing_class": processing_class},
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
				fn_kwargs={"processing_class": processing_class},
				num_proc=arguments.dataset_num_proc,
				desc="Eval - Apply Chat Template",
			)
		fn_kwargs = {
			"processing_class": self.processing_class,
			"processor": None,
		}
		_tokenize = build_tokenize(
			model=model if arguments.is_encoder_decoder else None,
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
		self.processing_class = processing_class
		if not isinstance(ref_model, EasyDeLState):
			ref_model = ref_model.to_state()
		self.ref_model = ref_model
		self.model = model
		self._loggers_initialized = False
		self.mesh = self.model.mesh

		self.concatenated_forward = jax.jit(
			create_concatenated_forward(
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
			model=model,
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
		self._configure_functions()
		self._configure_state()

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

			self.create_state_sharded = functions.create_state_sharded
			self.sharded_training_step_function = functions.sharded_training_step_function
			self.sharded_evaluation_step_function = functions.sharded_evaluation_step_function
			self.mesh = functions.mesh
			self.checkpoint_manager = functions.checkpoint_manager
		self.timer.log(operation_name)

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
		tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_training_steps)
		if self.pruning_module is not None:
			tx = self.pruning_module.wrap_optax(tx)
		return TrainerConfigureModelOutput(
			model=self.model,
			tx=tx,
			scheduler=scheduler,
			config=self.model.config,
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

		if self.arguments.sparsify_module:
			self.model.__call__ = sparse.sparsify(self.model.__call__)

		def create_state():
			"""
			Creates an EasyDeLState object.
			Returns:
			    EasyDeLState: The EasyDeLState object initialized.
			"""
			return EasyDeLState.create(
				model=self.model,
				tx=self.tx,
				init_opt_state=True,
			)

		state_shape = jax.eval_shape(lambda: create_state())
		state_partition_spec = match_partition_rules(
			self.model.config.get_partition_rules(),
			state_shape,
		)

		spec_named_sharding = self.specs_to_name_sharding(state_partition_spec)
		empty_sharding = jax.sharding.NamedSharding(
			spec=PartitionSpec(),
			mesh=self.model.mesh,
		)
		create_state_sharded = jax.jit(create_state, out_shardings=spec_named_sharding)
		train_function = create_train_function(
			concatenated_forward=self.concatenated_forward,
			ref_state=self.ref_model,
			loss_type=self.arguments.loss_type,
			reference_free=self.arguments.reference_free,
			label_smoothing=self.arguments.label_smoothing,
			beta=self.arguments.beta,
		)
		sharded_training_step_function = jax.jit(
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

		eval_function = create_eval_function(
			concatenated_forward=self.concatenated_forward,
			ref_state=self.ref_model,
			loss_type=self.arguments.loss_type,
			reference_free=self.arguments.reference_free,
			label_smoothing=self.arguments.label_smoothing,
			beta=self.arguments.beta,
		)

		sharded_evaluation_step_function = jax.jit(
			eval_function,
			in_shardings=(
				spec_named_sharding,
				jax.sharding.NamedSharding(
					spec=self.arguments.step_partition_spec,
					mesh=self.mesh,
				),
			),
			out_shardings=empty_sharding,
		)

		self.arguments.ensure_checkpoint_path()
		self.state_partition_spec = state_partition_spec
		self.state_named_sharding = spec_named_sharding
		self.state_shape = state_shape
		checkpoint_manager = self.arguments.get_streaming_checkpointer()
		mesh = self.model.mesh
		return TrainerConfigureFunctionOutput(
			create_state_sharded=create_state_sharded,
			sharded_training_step_function=sharded_training_step_function,
			sharded_evaluation_step_function=sharded_evaluation_step_function,
			mesh=mesh,
			checkpoint_manager=checkpoint_manager,
		)

	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
	) -> tp.Callable:
		"""
		Creates a data collection function for batching.

		For DPO training, this method simply returns the pre-configured `data_collator`.

		Args:
				max_sequence_length (int): The maximum sequence length (not used in this implementation).
				truncation_mode (tp.Literal["keep_end", "keep_start"], optional):
						The truncation mode (not used in this implementation). Defaults to "keep_end".

		Returns:
				tp.Callable: The data collator function.
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
				batch_size=self.arguments.total_batch_size
				* self.arguments.gradient_accumulation_steps,
				collate_fn=data_collator,
				num_workers=self.arguments.dataloader_num_workers,
				shuffle=True,
				drop_remainder=True,
			)
		)

	def _get_eval_dataloader(
		self,
		eval_dataset: tp.Optional["Dataset"] = None,  # noqa #type:ignore
	) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Creates the evaluation dataloader as a TensorFlow Dataset.

		This method retrieves the evaluation dataset (either provided as an argument or
		from the `self.eval_dataset` attribute), applies the data collator, and converts
		it into a TensorFlow Dataset for efficient batching and data loading during evaluation.

		Args:
				eval_dataset (tp.Optional[Dataset], optional):
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
					batch_size=self.arguments.total_batch_size
					* self.arguments.gradient_accumulation_steps,
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
						self.state,
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
		eval_dataset: tp.Optional["Dataset"] = None,  # noqa #type:ignore
	) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Returns the evaluation dataloader, potentially with precomputed reference log probabilities.

		If `precompute_ref_log_probs` is enabled, this method computes the reference model's log
		probabilities for the chosen and rejected responses in the evaluation dataset and adds
		them as columns to the dataset.

		Args:
				eval_dataset (tp.Optional[Dataset], optional):
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
					self.compute_reference_log_probs(self.state, padded_batch)
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
		padded_batch: tp.Dict,
	) -> tuple[tp.Any, tp.Any]:
		"""
		Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset.

		Args:
				state (EasyDeLState): The EasyDeLState object of the model (used if no reference model is provided).
				padded_batch (tp.Dict): The padded batch of data.

		Returns:
				tuple[tp.Any, tp.Any]: A tuple containing the log probabilities for the chosen and rejected responses.
		"""

		if self.ref_model is None:
			(
				reference_chosen_log_probs,
				reference_rejected_log_probs,
				_,
				_,
			) = self.concatenated_forward(
				state,
				batch=padded_batch,
			)
		else:
			(
				reference_chosen_log_probs,
				reference_rejected_log_probs,
				_,
				_,
			) = self.concatenated_forward(
				self.ref_model,
				batch=padded_batch,
			)

		return reference_chosen_log_probs, reference_rejected_log_probs

	def _run_training_loop(
		self,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		start_time: float,
		shard_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
		gather_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
	):
		"""Core training loop implementation."""
		pbar = tqdm(total=self.max_training_steps)
		current_step = int(jax.device_get(self.state.step))
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
			state=self.state,
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
		shard_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
		gather_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
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
				return self.state, current_step, exect

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
					batch_size=self.arguments.total_batch_size
					* self.arguments.gradient_accumulation_steps,
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
						state=self.state,
						gather_fns=gather_fns,
						milestone=True,
					)
				if self._should_run_evaluation(current_step):
					for _ in self.eval(model_state=self.state):
						...
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
				metrics = self._execute_eval_step(sharded_state, batch)

				loss = metrics.loss
				mean_loss = metrics_tracker.update(loss, float("inf"), current_step)
				eval_metrics = step_metrics.calculate(
					loss=loss,
					metrics=metrics,
					current_step=current_step,
					learning_rate=0.000,
					epoch=0,
					flops_per_device=getattr(self, "_flops_per_device", 0),
					batch_size=self.arguments.total_batch_size
					* self.arguments.gradient_accumulation_steps,
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

		metrics = self.sharded_evaluation_step_function(state, batch)
		return metrics

	def _execute_train_step(self, batch):
		"""Execute a single training step."""
		if self.pruning_module is not None:
			self.state = self.state.replace(
				graphstate=self.pruning_module.pre_forward_update(
					self.state.graphstate,
					self.state.opt_state,
				)
			)

		# Forward and backward pass
		try:
			batch = {key: jnp.asarray(value) for key, value in batch.items()}

			self.state, metrics = self.sharded_training_step_function(self.state, batch)
			# Apply post-gradient updates
			loss = metrics.loss
			if self.pruning_module is not None:
				self.state = self.state.replace(
					graphstate=self.pruning_module.post_gradient_update(
						self.state.graphstate,
						self.state.opt_state,
					)
				)

			return loss, metrics, None
		except (KeyboardInterrupt, EasyDeLTimerError) as run_exception:
			return loss, metrics, run_exception

	def _finalize_training(self, output, run_exception):
		"""Finalize training and prepare output."""

		if self.arguments.do_eval:
			for _ in self.eval(output.state):
				...

		self.finish()

		return output

	def train(self) -> DPOTrainerOutput:
		start_time = time.time()
		rules = self.model.config.get_partition_rules()
		shard_fns, gather_fns = make_shard_and_gather_fns(
			partition_specs=match_partition_rules(rules, jax.eval_shape(lambda: self.state)),
			mesh=self.mesh,
		)

		metrics_tracker = MetricsTracker()
		step_metrics = StepMetrics(self.arguments)

		# Setup initial metrics and logging
		self._setup_initial_metrics(self.state)

		output, run_exception = self._run_training_loop(
			metrics_tracker=metrics_tracker,
			step_metrics=step_metrics,
			start_time=start_time,
			shard_fns=shard_fns,
			gather_fns=gather_fns,
		)
		return self._finalize_training(output, run_exception)

	def eval(self, model_state: EasyDeLState) -> tp.Iterator[dict]:
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
