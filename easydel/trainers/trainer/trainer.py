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
import time
import typing as tp

import jax
from fjformer.sharding import match_partition_rules
from jax import numpy as jnp
from jax.experimental import sparse
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import get_logger
from easydel.infra.loss_utils import LossMetrics

from ..base_trainer import (
	BaseTrainer,
	TrainerConfigureFunctionOutput,
)
from ..trainer_protocol import MetricsTracker, StepMetrics
from ._fn import create_evaluation_step, create_training_step
from .modeling_output import TrainerOutput

logger = get_logger(__name__)


class Trainer(BaseTrainer):
	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
	) -> tp.Callable:
		"""
		Creates a function to collect and process batches of data for training or evaluation.

		This function handles padding or truncating sequences to the specified `max_sequence_length`
		based on the chosen `truncation_mode`.

		Args:
		    max_sequence_length (int): The maximum allowed sequence length.
		    truncation_mode (typing.Literal["keep_end", "keep_start"], optional):
		        The truncation mode. Defaults to "keep_end".

		Returns:
		    Callable: A function that takes a batch of data and returns a processed batch.
		"""

		def collate_fn(batch):
			results = {}
			for key in batch[0].keys():
				if self.model.loss_function.__name__ == "ForCausalLMLoss":
					if truncation_mode == "keep_end":
						corrected_sequence = [
							jnp.array(f[key])[..., -max_sequence_length:] for f in batch
						]
					else:
						corrected_sequence = [
							jnp.array(f[key])[..., :max_sequence_length] for f in batch
						]
				else:
					corrected_sequence = [jnp.array(f[key]) for f in batch]

				results[key] = jnp.stack(corrected_sequence).reshape(
					-1,
					corrected_sequence[0].shape[-1],
				)
			return results

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
		sharded_training_step_function = jax.jit(
			create_training_step(
				partition_spec=self.arguments.step_partition_spec,
				loss_config=self.arguments.loss_config,
				learning_rate_fn=self.scheduler,
				gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(spec_named_sharding, empty_sharding, empty_sharding),
			donate_argnums=(0, 0),
		)

		sharded_evaluation_step_function = jax.jit(
			create_evaluation_step(
				partition_spec=self.arguments.step_partition_spec,
				loss_config=self.arguments.loss_config,
			),
			in_shardings=(spec_named_sharding, empty_sharding),
			out_shardings=(empty_sharding),
			donate_argnums=(0, 0),
		)

		mesh = self.model.mesh
		self.arguments.ensure_checkpoint_path()
		checkpoint_manager = self.arguments.get_streaming_checkpointer()
		self.state_partition_spec = state_partition_spec
		self.state_named_sharding = spec_named_sharding
		self.state_shape = state_shape

		return TrainerConfigureFunctionOutput(
			create_state_sharded=create_state_sharded,
			sharded_training_step_function=sharded_training_step_function,
			sharded_evaluation_step_function=sharded_evaluation_step_function,
			mesh=mesh,
			checkpoint_manager=checkpoint_manager,
		)

	def _run_training_loop(
		self,
		state: EasyDeLState,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
	):
		"""Core training loop implementation."""
		pbar = tqdm(total=self.max_training_steps)
		current_step = int(jax.device_get(state.step))
		run_exception = None
		with self.mesh:
			for epoch in range(self.arguments.num_train_epochs):
				state, current_step, run_exception = self._train_epoch(
					state=state,
					train_dataset=self.dataloader_train,
					current_step=current_step,
					metrics_tracker=metrics_tracker,
					step_metrics=step_metrics,
					pbar=pbar,
					epoch=epoch,
				)

				if current_step >= self.max_training_steps:
					break
				if run_exception is not None:
					break
		return self._prepare_training_output(
			state=state,
			run_exception=run_exception,
		), run_exception

	def _run_evaluation(
		self,
		state: EasyDeLState,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		start_time: float,
	):
		"""Core evaluation loop implementation."""
		pbar = tqdm(total=self.max_evaluation_steps)
		pbar.set_description("evaluation process")
		current_step = int(jax.device_get(state.step))
		with self.mesh:
			for eval_metrics in self._eval_epoch(
				state=state,
				eval_dataset=self.dataloader_eval,
				current_step=current_step,
				metrics_tracker=metrics_tracker,
				step_metrics=step_metrics,
				pbar=pbar,
				start_time=start_time,
			):
				yield eval_metrics

	def _train_epoch(
		self,
		state: EasyDeLState,
		train_dataset: int,
		current_step: int,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		pbar: tqdm,
		epoch: int,
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
				return state, current_step, exect

			# Execute training step
			state, loss, metrics, run_exception = self._execute_train_step(
				state=state, batch=batch
			)
			# Update and log metrics
			try:
				mean_loss, mean_accuracy = metrics_tracker.update(
					loss=loss,
					accuracy=metrics.accuracy,
					step=current_step,
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
					batch_size=self.arguments.total_batch_size
					* self.arguments.gradient_accumulation_steps,
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
						state=state,
						milestone=True,
						save_directory=self.arguments.save_directory,
					)
				if self._should_run_evaluation(current_step):
					for _ in self.eval(model_state=state):
						...

				current_step += 1
			except (KeyboardInterrupt, EasyDeLTimerError):
				return state, current_step, run_exception
			if run_exception is not None:
				break
		return state, current_step, run_exception

	def _eval_epoch(
		self,
		state: EasyDeLState,
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
				loss, metrics = self._execute_eval_step(state, batch)
				mean_loss, mean_accuracy = metrics_tracker.update(
					loss,
					metrics.accuracy,
					current_step,
				)
				eval_metrics = step_metrics.calculate(
					loss=loss,
					metrics=metrics,
					current_step=current_step,
					learning_rate=0.000,
					epoch=0,
					flops_per_device=getattr(self, "_flops_per_device", 0),
					batch_size=self.arguments.total_batch_size
					* self.arguments.gradient_accumulation_steps,
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

	def _execute_eval_step(self, state, batch) -> tp.Tuple[float, LossMetrics]:
		"""Execute a single eval step."""
		metrics = self.sharded_evaluation_step_function(state, batch)
		return metrics.loss, metrics

	def _execute_train_step(self, state, batch):
		"""Execute a single training step."""
		if self.pruning_module is not None:
			state = state.replace(
				graphstate=self.pruning_module.pre_forward_update(
					state.graphstate,
					state.opt_state,
				)
			)

		# Forward and backward pass
		try:
			state, loss, metrics = self.sharded_training_step_function(state, batch)
			# Apply post-gradient updates
			if self.pruning_module is not None:
				state = state.replace(
					graphstate=self.pruning_module.post_gradient_update(
						state.graphstate,
						state.opt_state,
					)
				)

			return state, loss, metrics, None
		except (KeyboardInterrupt, EasyDeLTimerError) as run_exception:
			return state, loss, metrics, run_exception

	def _finalize_training(self, output, run_exception):
		"""Finalize training and prepare output."""

		if self.arguments.do_eval:
			for _ in self.eval(output.state):
				...

		self.finish()

		return output

	def train(self) -> TrainerOutput:
		state = self.state
		metrics_tracker = MetricsTracker()
		step_metrics = StepMetrics(self.arguments)

		# Setup initial metrics and logging
		self._setup_initial_metrics(state)

		output, run_exception = self._run_training_loop(
			state=self.state,
			metrics_tracker=metrics_tracker,
			step_metrics=step_metrics,
		)
		return self._finalize_training(output, run_exception)

	def eval(self, model_state: EasyDeLState) -> tp.Iterator[dict]:
		"""
		Evaluates the Causal Language Model (CLM) using the provided model state.

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
			state=model_state,
			metrics_tracker=metrics_tracker,
			step_metrics=step_metrics,
			start_time=start_time,
		):
			yield metrics
