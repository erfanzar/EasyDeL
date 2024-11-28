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
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import flax
import jax
import termcolor
from fjformer.sharding import make_shard_and_gather_fns, match_partition_rules
from jax import numpy as jnp
from jax.experimental import sparse
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import get_logger
from easydel.modules.modeling_utils import EasyDeLBaseModule
from easydel.trainers.base_trainer import (
	BaseTrainer,
	MetricsTracker,
	StepMetrics,
	TrainerConfigureFunctionOutput,
)
from easydel.trainers.sequence_to_sequence_trainer.functions import (
	create_seq2seq_model_evaluation_step,
	create_seq2seq_model_train_step,
)
from easydel.trainers.sequence_to_sequence_trainer.modeling_output import (
	Seq2SeqTrainerOutput,
)
from easydel.trainers.training_configurations import TrainingArguments

logger = get_logger(__name__)


class Seq2SeqTrainer(BaseTrainer):
	"""
	Trainer for SequenceToSequence Models (Seq2Seq).

	This trainer handles training, evaluation, and checkpointing of Seq2Seq Models
	using EasyDeL. It supports features like sharding, gradient accumulation, mixed precision training, and LoRA.

	Attributes:
			(Inherited from BaseTrainer)

	Methods:
			create_collect_function(self, max_sequence_length: int, truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end") -> Callable:
					Creates a function to collect and pad/truncate batches of data.
			configure_functions(self) -> TrainerConfigureFunctionOutput:
					Configures and JIT-compiles the training and evaluation step functions.
			initialize_state(self, model_parameters: Optional[flax.core.FrozenDict] = None, state: Optional[EasyDeLState] = None) -> Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
					Initializes the training state, either from scratch, pretrained parameters, or a checkpoint.
			train(self, model_parameters: Optional[flax.core.FrozenDict] = None, state: Optional[EasyDeLState] = None) -> CausalLMTrainerOutput:
					Trains the Seq2Seq and returns the training output.
			eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
					Evaluates the Seq2Seq and yields evaluation metrics.
	"""

	def __init__(
		self,
		arguments: Optional[TrainingArguments] = None,
		model: Optional[EasyDeLBaseModule] = None,
		data_collator=None,
		dataset_train: Optional["Dataset"] = None,  # noqa: F821 # type:ignore
		dataset_eval: Optional["Dataset"] = None,  # noqa: F821 # type:ignore
		finetune: bool = True,
		checkpoint_path: Optional[Union[str, os.PathLike]] = None,
		_do_init_fns: bool = True,
	):
		assert dataset_train is not None, "`dataset_train` must be provided."
		assert data_collator is not None, "`data_collator` must be provided."
		self.data_collator = data_collator
		if arguments.init_input_shape is None:
			if model is not None:
				arguments.init_input_shape = (
					jax.device_count(),
					model.config.num_mel_bins,
					model.config.max_source_positions * 2,
				)
			else:
				config = arguments.configs_to_initialize_model_class.get("config", None)
				if config is None:
					raise NotImplementedError(
						"please provide `init_input_shape` in `TrainingArguments`"
					)
				arguments.init_input_shape = (
					jax.device_count(),
					config.num_mel_bins,
					config.max_source_positions * 2,
				)
		super().__init__(
			arguments=arguments,
			checkpoint_path=checkpoint_path,
			dataset_eval=dataset_eval,
			dataset_train=dataset_train,
			finetune=finetune,
			model=model,
			_do_init_fns=_do_init_fns,
		)

	def create_collect_function(self, *args, **kwargs) -> Callable:
		"""
		Creates a function to collect and process batches of data for training or evaluation.
		Returns:
		    Callable: A function that takes a batch of data and returns a processed batch.
		"""

		def collate_fn(batch):
			return self.data_collator(batch)

		return collate_fn

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

		def initialize_state_function():
			initialized_parameters = self.model.init_weights(
				jax.random.PRNGKey(0),
				self.arguments.init_input_shape,
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
		sharded_train_step_function = jax.jit(
			create_seq2seq_model_train_step(
				partition_spec=self.arguments.step_partition_spec,
				gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(spec_named_sharding, empty_sharding, empty_sharding),
			donate_argnums=(0, 0),
		)

		sharded_eval_step_function = jax.jit(
			create_seq2seq_model_evaluation_step(
				partition_spec=self.arguments.step_partition_spec
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(empty_sharding, empty_sharding, empty_sharding),
			donate_argnums=(0, 0),
		)

		mesh = self.arguments.get_mesh()
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
		"""
		Initializes the training state, either from scratch, pretrained parameters, or a checkpoint.

		This method handles different initialization scenarios:
		1.  **No parameters, no state, no checkpoint, no LoRA:** Raises an error, as there's nothing to initialize from.
		2.  **Using LoRA, no parameters, no state:** Initializes from `self.lora_parameters`.
		3.  **State provided:** Uses the provided `state` directly.
		4.  **Finetuning and checkpoint path provided:** Loads the state from the checkpoint.
		5.  **Finetuning and model parameters provided:** Initializes the state from the provided `model_parameters`.
		6.  **No finetuning:** Initializes the state from scratch.

		Args:
		    model_parameters (Optional[flax.core.FrozenDict], optional):
		        Pretrained model parameters to initialize from. Defaults to None.
		    state (Optional[EasyDeLState], optional):
		        An existing EasyDeLState object to use. Defaults to None.

		Returns:
		    Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
		        A tuple containing the initialized or loaded EasyDeLState, shard functions, and gather functions.

		Raises:
		    RuntimeError: If no initialization source is provided and LoRA is not being used.
		    EasyDeLTimerError: If both `model_parameters` and `checkpoint_path` are provided,
		                        or if neither is provided when finetuning.
		"""
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
				params = sharded_state.params
				sharded_state.params = params
				if sharded_state.opt_state is None:
					logger.info("Optimizer State is not Found!, initializing one.")
					with jax.default_device(self.arguments.offload_device):
						sharded_state = sharded_state.init_opt_state()
						opt_state = sharded_state.opt_state
						sharded_state = sharded_state.replace(opt_state=opt_state)
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
						sharded_train_step_function = jax.jit(
							create_seq2seq_model_train_step(
								partition_spec=self.arguments.step_partition_spec,
								gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
							),
							in_shardings=(spec_named_sharding, empty_sharding),
							out_shardings=(
								spec_named_sharding,
								empty_sharding,
								empty_sharding,
							),
							donate_argnums=(0, 0),
						)

						sharded_eval_step_function = jax.jit(
							create_seq2seq_model_evaluation_step(
								partition_spec=self.arguments.step_partition_spec,
							),
							in_shardings=(spec_named_sharding, empty_sharding),
							out_shardings=(
								empty_sharding,
								empty_sharding,
								empty_sharding,
							),
							donate_argnums=(0, 0),
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
						warnings.warn(
							"Model Parameters should be like FrozenDict({'params': params}). (auto correct)",
							stacklevel=1,
						)
						model_parameters = flax.core.FrozenDict(model_parameters)

					model_parameters = model_parameters
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
			if self.arguments.sparsify_module:
				...  # disabled at the moment to fix shardings..
				# sharded_state = sharded_state.replace(
				# 	params=apply_sparsity_to_params(
				# 		params=sharded_state.params,
				# 		sparsify_module=self.arguments.sparse_module_type,
				# 		verbose=True,
				# 	)
				# )
			self.sharded_state = sharded_state
			return sharded_state, shard_fns, gather_fns

	def _run_training_loop(
		self,
		sharded_state: EasyDeLState,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		start_time: float,
		shard_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
	):
		"""Core training loop implementation."""
		pbar = tqdm(total=self.max_training_steps)
		current_step = int(jax.device_get(sharded_state.step))
		run_exception = None
		with self.mesh:
			for epoch in range(self.arguments.num_train_epochs):
				sharded_state, current_step, run_exception = self._train_epoch(
					sharded_state,
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
			sharded_state=sharded_state,
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
		sharded_state: EasyDeLState,
		train_dataset: int,
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
				return sharded_state, current_step, exect

			# Execute training step
			sharded_state, loss, metrics, run_exception = self._execute_train_step(
				sharded_state, batch
			)
			# Update and log metrics
			try:
				mean_loss, mean_accuracy = metrics_tracker.update(
					loss, metrics["accuracy"], current_step
				)

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
					seq_length=self.arguments.max_sequence_length,
					mean_loss=mean_loss,
					mean_accuracy=mean_accuracy,
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
						state=sharded_state,
						gather_fns=gather_fns,
						milestone=True,
						save_dir=self.arguments.save_dir,
					)

				current_step += 1
			except (KeyboardInterrupt, EasyDeLTimerError):
				return sharded_state, current_step, run_exception
			if run_exception is not None:
				break
		return sharded_state, current_step, run_exception

	def _eval_epoch(
		self,
		sharded_state: EasyDeLState,
		eval_dataset: int,
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
				mean_loss, mean_accuracy = metrics_tracker.update(
					loss, metrics["accuracy"], current_step
				)
				eval_metrics = step_metrics.calculate(
					loss=loss,
					metrics=metrics,
					current_step=current_step,
					learning_rate=0.000,
					epoch=0,
					flops_per_device=getattr(self, "_flops_per_device", 0),
					batch_size=self.arguments.total_batch_size,
					seq_length=self.arguments.max_sequence_length,
					mean_loss=mean_loss,
					mean_accuracy=mean_accuracy,
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
		loss, accuracy, aux_loss = self.sharded_eval_step_function(state, batch)
		return loss, dict(loss=loss, accuracy=accuracy, aux_loss=aux_loss)

	def _execute_train_step(self, state, batch):
		"""Execute a single training step."""
		if self.pruning_module is not None:
			state = state.replace(
				params=self.pruning_module.pre_forward_update(
					state.params,
					state.opt_state,
				)
			)

		# Forward and backward pass
		try:
			state, loss, metrics = self.sharded_train_step_function(state, batch)
			# Apply post-gradient updates
			if self.pruning_module is not None:
				state = state.replace(
					params=self.pruning_module.post_gradient_update(
						state.params,
						state.opt_state,
					)
				)

			if self.arguments.log_grad_norms:
				metrics.update(
					{
						"train/max_grad_norm": metrics["max_grad_norm"].tolist(),
						"train/mean_grad_norm": metrics["mean_grad_norm"].tolist(),
					}
				)
			return state, loss, metrics, None
		except (KeyboardInterrupt, EasyDeLTimerError) as run_exception:
			return state, loss, metrics, run_exception

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

	def train(
		self,
		model_parameters: Optional[flax.core.FrozenDict] = None,
		state: Optional[EasyDeLState] = None,
	) -> Seq2SeqTrainerOutput:
		start_time = time.time()

		sharded_state, shard_fns, gather_fns = self.initialize_state(
			model_parameters=model_parameters, state=state
		)

		metrics_tracker = MetricsTracker()
		step_metrics = StepMetrics(self.arguments)

		# Setup initial metrics and logging
		self._setup_initial_metrics(sharded_state)

		output, run_exception = self._run_training_loop(
			sharded_state=sharded_state,
			metrics_tracker=metrics_tracker,
			step_metrics=step_metrics,
			start_time=start_time,
			shard_fns=shard_fns,
			gather_fns=gather_fns,
		)
		return self._finalize_training(output, run_exception)

	def eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
		"""
		Evaluates the Seq2Seq (Seq2Seq) using the provided model state.

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
