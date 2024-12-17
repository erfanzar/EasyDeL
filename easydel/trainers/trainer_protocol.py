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

import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Literal, Mapping, Optional, Union

import flax
import flax.core
import jax
import numpy as np
import optax
import tqdm
from fjformer.checkpoint import CheckpointManager
from jax.sharding import Mesh
from optax import GradientTransformation, Schedule

from easydel.etils.easystate import EasyDeLState
from easydel.infra.loss_utils import LossMetrics
from easydel.utils.traversals import flatten_dict

try:
	import wandb  # noqa: F821 # type:ignore
except ImportError:
	wandb = None

from jax import numpy as jnp

from easydel.etils.etils import get_logger
from easydel.infra.base_module import (
	EasyDeLBaseConfig,
	EasyDeLBaseModule,
)
from easydel.trainers.training_configurations import TrainingArguments
from easydel.utils import Timers

logger = get_logger(__name__)


@dataclass
class TrainerConfigureDataloaderOutput:
	dataloader_train: Iterator[np.ndarray]
	max_training_steps: int
	dataloader_eval: Optional[Iterator[np.ndarray]] = None
	max_evaluation_steps: Optional[int] = None


@dataclass
class TrainerConfigureModelOutput:
	model: EasyDeLBaseModule
	tx: GradientTransformation
	scheduler: Schedule
	config: Optional[EasyDeLBaseConfig] = None


@dataclass
class TrainerConfigureFunctionOutput:
	create_state_sharded: Callable
	sharded_training_step_function: Callable
	mesh: Mesh
	checkpoint_manager: CheckpointManager
	sharded_evaluation_step_function: Optional[Callable] = None


@dataclass
class TrainerOutput:
	state: EasyDeLState
	mesh: Optional[jax.sharding.Mesh]
	last_save_file_name: Optional[str] = None
	checkpoint_path: Optional[str] = None


class BaseTrainerProtocol(ABC):
	# Required attributes for all trainers
	arguments: TrainingArguments
	dataset_train: Optional["Dataset"]  # type: ignore #noqa
	dataset_eval: Optional["Dataset"]  # type: ignore #noqa
	finetune: bool
	checkpoint_path: Optional[Union[str, os.PathLike]]
	dtype: Any  # jax.numpy.dtype
	param_dtype: Any  # jax.numpy.dtype

	timer: Timers
	wandb_runtime: Any  # wandb runtime
	dataloader_train: Iterator[np.ndarray]
	dataloader_eval: Optional[Iterator[np.ndarray]]
	max_training_steps: int
	max_evaluation_steps: int
	model: EasyDeLBaseModule
	config: EasyDeLBaseConfig
	scheduler: optax.Schedule
	tx: optax.GradientTransformation  # optax
	model_state: Any  # flax.core.FrozenDict
	create_state_sharded: Callable

	sharded_training_step_function: Callable
	sharded_evaluation_step_function: Callable

	mesh: Any
	checkpoint_manager: Any
	state_shape: Any
	state_partition_spec: Any
	state_named_sharding: Any
	state: Any
	pruning_module: Any

	_base_model: Any

	@abstractmethod
	def __init__(
		self,
		arguments: TrainingArguments,
		model: Any,
		dataset_train: Optional["Dataset"] = None,  # type:ignore #noqa
		dataset_eval: Optional["Dataset"] = None,  # type:ignore #noqa
		finetune: bool = True,
		checkpoint_path: Optional[Union[str, os.PathLike]] = None,
		_do_init_fns: bool = True,
	):
		"""
		Initializes the trainer.
		"""
		...

	@abstractmethod
	def _initialize_attributes(self):
		"""
		Initializes attributes with default values.
		"""
		...

	@abstractmethod
	def _initialize_memory_tracking(self):
		"""
		Initializes memory tracking if enabled.
		"""
		...

	@abstractmethod
	def initialize_trainer_utils(self):
		"""
		Initializes all trainer utilities.
		"""
		...

	@abstractmethod
	def _initialize_wandb(self):
		"""
		Initializes Weights & Biases if enabled.
		"""
		...

	@abstractmethod
	def _initialize_timer(self):
		"""
		Initializes the training timer.
		"""
		...

	@abstractmethod
	def _configure_dataloaders(self):
		"""
		Configures the dataloaders for training and evaluation.
		"""
		...

	@abstractmethod
	def _configure_model(self):
		"""
		Configures the model, optimizer, scheduler, and configuration.
		"""
		...

	@abstractmethod
	def _configure_functions(self):
		"""
		Configures and JIT-compiles the training and evaluation step functions.
		"""
		...

	@abstractmethod
	def _configure_state(self):
		"""Configures and JIT-compiles the sharded state"""
		...

	@abstractmethod
	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: Literal["keep_end", "keep_start"],
	) -> Callable:
		"""
		Creates a function to collect and process batches of data for training or evaluation.
		"""
		...

	@abstractmethod
	def configure_functions(self) -> TrainerConfigureFunctionOutput:
		"""
		Configures and JIT-compiles the training and evaluation step functions.
		"""
		...

	@abstractmethod
	def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
		"""
		Configures the dataloaders for training and evaluation.
		"""
		...

	@abstractmethod
	def configure_model(self) -> TrainerConfigureModelOutput:
		"""
		Configures the model, optimizer, scheduler, and configuration.
		"""
		...

	@abstractmethod
	def _save_state(
		self,
		state: "EasyDeLState",  # noqa: F821 # type:ignore
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		milestone: bool,
		save_dir: Optional[str] = None,
	) -> str:
		"""
		Saves the model state to a checkpoint file.
		"""
		...

	@abstractmethod
	def _get_current_step(self, state):
		"""
		Get the current step number.
		"""
		...

	@abstractmethod
	def _get_save_directory(self, save_dir):
		"""
		Get the checkpoint directory.
		"""
		...

	@abstractmethod
	def _manage_checkpoint_limit(self, checkpoint_dir):
		"""
		Manages the checkpoint limit by deleting old checkpoints.
		"""
		...

	@abstractmethod
	def _generate_checkpoint_directory_name(self, step, milestone):
		"""
		Generates a checkpoint filename.
		"""
		...

	@abstractmethod
	def _save_readme(self, checkpoint_dir):
		"""
		Saves a README file with model and training information.
		"""
		...

	@abstractmethod
	def _format_partition_rules(self) -> str:
		"""Format partition rules with proper indentation and formatting."""
		...

	@abstractmethod
	def _get_device_info(self) -> dict:
		"""Get information about available devices."""
		...

	@abstractmethod
	def _get_information(self) -> str:
		"""
		Generate formatted information about the model and training setup.
		"""
		...

	@abstractmethod
	def save_information(self, output_path: Union[str, Path]) -> None:
		"""
		Save the generated information to a markdown file.
		"""
		...

	@abstractmethod
	def save_pretrained(
		self,
		state: "EasyDeLState",  # noqa: F821 # type:ignore
		save_dir: Optional[str] = None,
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]] = None,
		to_torch: bool = False,
		base_hf_auto_class=None,
		easystate_to_huggingface_model_kwargs: Optional[dict] = None,
		add_params_field_to_torch_convertation: bool = False,
		torch_save_pretrained_kwargs: Optional[dict] = None,
	):
		"""
		Saves the model state as a checkpoint file or to a Torch compatible directory.
		"""
		...

	@abstractmethod
	def _save_to_torch(
		self,
		state,
		save_dir,
		base_hf_auto_class,
		easystate_to_huggingface_model_kwargs,
		torch_save_pretrained_kwargs,
	):
		"""
		Saves the model state to a Torch compatible directory.
		"""
		...

	@abstractmethod
	def _create_hf_model_config(
		self,
		state: "EasyDeLState",  # noqa: F821 # type:ignore
		model_config,
		model_type,
	):
		"""
		Creates a Hugging Face model config from the current state
		"""
		...

	@abstractmethod
	def specs_to_name_sharding(self, tree, mesh=None):
		"""Convert specs to named sharding."""
		...

	@abstractmethod
	def calculate_number_total_flops_per_device(self, params):
		"""Calculate total FLOPs per device."""
		...

	@staticmethod
	@abstractmethod
	def count_model_parameters(prm):
		"""Prints the number of model parameters in billions."""
		...

	@abstractmethod
	def _should_skip_step(self, current_step):
		"""Determine if current step should be skipped."""
		...

	@abstractmethod
	def _should_save_checkpoint(self, current_step):
		"""Determine if checkpoint should be saved at current step."""
		...

	@abstractmethod
	def _prepare_training_output(
		self,
		state: EasyDeLState,
		shard_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		run_exception: Optional[Exception] = None,
	):
		"""Prepare training output after training loop completion."""
		...

	@abstractmethod
	def _handle_training_interruption(
		self,
		state: EasyDeLState,
		exception: Exception,
		shard_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
	):
		"""Handle training interruption gracefully."""
		...

	@abstractmethod
	def _setup_initial_metrics(self, state):
		"""Setup initial metrics logging."""
		...

	@abstractmethod
	def _get_next_batch(self, train_iter):
		"""Get next batch from iterator, reinitializing if needed."""
		...

	@abstractmethod
	def _log_metrics(
		self,
		metrics: Dict[str, float],
		pbar: tqdm.tqdm,
		step: int,
		mode: str = "train",
	):
		"""Log metrics and update progress bar."""
		...

	@abstractmethod
	def _run_training_loop(
		self,
		state: EasyDeLState,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		start_time: float,
		shard_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
		gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
	):
		"""Core training loop implementation."""
		...

	@abstractmethod
	def _run_evaluation(
		self,
		state: EasyDeLState,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		start_time: float,
	):
		"""Core evaluation implementation."""
		...

	@abstractmethod
	def _train_epoch(
		self,
		state: EasyDeLState,
		train_iter: int,
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
		...

	@abstractmethod
	def _eval_epoch(
		self,
		state: EasyDeLState,
		eval_iter: int,
		current_step: int,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		pbar: tqdm,
		start_time: float,
	):
		"""Handles training for a single epoch."""
		...

	@abstractmethod
	def _execute_eval_step(self, state, batch):
		"""Execute a single eval step."""
		...

	@abstractmethod
	def _execute_train_step(self, state, batch):
		"""Execute a single train step."""
		...

	@abstractmethod
	def _finalize_training(self, output, run_exception):
		"""Finalize training and prepare output."""
		...

	@abstractmethod
	def train(
		self,
		model_parameters: Optional[flax.core.FrozenDict] = None,
		state: Optional[EasyDeLState] = None,
	) -> Any:
		"""Train using the provided model state."""
		...

	@abstractmethod
	def eval(self, model_state: EasyDeLState) -> Iterator[dict]:
		"""
		Evaluates using the provided model state.
		"""
		...


class StepMetrics:
	"""Handles calculation and tracking of training metrics."""

	def __init__(self, arguments):
		self.arguments = arguments
		self.start_time = time.time()
		self.step_start_time = time.time()

	def start_step(self):
		"""Mark the start of a training step."""
		self.step_start_time = time.time()

	def calculate(
		self,
		loss,
		metrics: LossMetrics,
		current_step,
		epoch,
		flops_per_device,
		batch_size,
		seq_length,
		learning_rate,
		mode: Optional[Literal["eval", "train"]] = None,
		**extras,
	) -> Dict[str, float]:
		"""Calculate comprehensive metrics for the training step."""
		step_time = time.time() - self.step_start_time
		total_time = time.time() - self.start_time

		visited_tokens = jnp.multiply(seq_length, jnp.multiply(current_step, batch_size))

		flops = flops_per_device / step_time

		basic_metrics = {
			"loss": loss.tolist(),
			"learning_rate": learning_rate,
			"step": current_step,
			"step_time": step_time,
			"perplexity": jnp.exp(loss).tolist(),
			"visited_tokens": visited_tokens,
			"epoch": epoch,
			"TFLOPs": flops,
			"total_time": total_time,
			**extras,
		}

		basic_metrics.update({"accuracy": metrics.accuracy})

		# if metrics.get("mae", None) is not None:
		# 	basic_metrics.update({"mae": metrics.get("mae", 0.0)})
		# if metrics.get("mse", None) is not None:
		# 	basic_metrics.update({"mse": metrics.get("mse", 0.0)})

		if not self.arguments.performance_mode and (mode == "train" or mode is None):
			detailed_metrics = self._calculate_detailed_metrics(metrics)
			basic_metrics.update(detailed_metrics)
		if mode is not None:
			basic_metrics = {f"{mode}/{k}": v for k, v in basic_metrics.items()}
		return basic_metrics

	def _calculate_detailed_metrics(self, metrics):
		"""Calculate additional detailed metrics."""
		detailed_metrics = {}
		getattr_in = lambda x: x if not hasattr(x, "value") else x.value  # noqa
		if self.arguments.log_grad_norms:
			detailed_metrics.update(
				{
					"train/max_grad_norm": getattr_in(metrics.max_grad_norm).tolist(),
					"train/mean_grad_norm": getattr_in(metrics.mean_grad_norm).tolist(),
				}
			)

			# Add per-layer gradient norms
			detailed_metrics.update(
				{
					f"grad_norm/{layer_name}": getattr_in(grad_norm).tolist()
					for layer_name, grad_norm in flatten_dict(metrics.grad_norms).items()
					if getattr_in(grad_norm) is not None
				}
			)

		return detailed_metrics


class MetricsTracker:
	"""Tracks and aggregates training metrics over time."""

	def __init__(self):
		self.loss_sum = None
		self.accuracy_sum = None
		self.metrics_history = defaultdict(list)
		self.step_offset = 0

	def update(self, loss, accuracy, step):
		"""Update tracked metrics with new values."""
		with jax.spmd_mode("allow_all"):
			self.loss_sum = loss if self.loss_sum is None else self.loss_sum + loss
			mean_loss = self.loss_sum / (step + 1 - self.step_offset)
			if accuracy != float("inf"):
				if accuracy is None:
					accuracy = 0.0
				self.accuracy_sum = (
					accuracy if self.accuracy_sum is None else self.accuracy_sum + accuracy
				)
				mean_accuracy = self.accuracy_sum / (step + 1 - self.step_offset)

				return mean_loss, mean_accuracy
			return mean_loss

	def reset(self, step):
		"""Reset tracked metrics."""
		self.loss_sum = None
		self.accuracy_sum = None
		self.step_offset = step
