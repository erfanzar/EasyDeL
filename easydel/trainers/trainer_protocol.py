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

import abc
import os
import time
import typing as tp
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path

import flax
import flax.core
import jax
import numpy as np
import optax
import tqdm
from jax.sharding import Mesh
from optax import GradientTransformation, Schedule
from rich.progress import (
	Progress,
	ProgressColumn,
	Task,
	TaskID,
)
from rich.text import Text

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.utils import CompilationTracker
from easydel.utils import traversals as etr
from easydel.utils.checkpoint_managers.streamer import CheckpointManager
from easydel.utils.traversals import flatten_dict

try:
	import wandb  # noqa: F821 # type:ignore
except ImportError:
	wandb = None

from jax import numpy as jnp

from easydel.infra.base_module import (
	EasyDeLBaseConfig,
	EasyDeLBaseModule,
)
from easydel.utils import Timers
from easydel.utils.helpers import get_logger

from .training_configurations import MetricsType, TrainingArguments

if tp.TYPE_CHECKING:
	from datasets import Dataset, IterableDataset
	from jax._src.pjit import JitWrapped
else:
	JitWrapped = tp.Callable
	Dataset = tp.Any
	IterableDataset = tp.Any

logger = get_logger(__name__)


@etr.auto_pytree
class TrainerConfigureDataloaderOutput:
	dataloader_train: tp.Iterator[np.ndarray]
	max_training_steps: int
	dataloader_eval: tp.Optional[tp.Iterator[np.ndarray]] = None
	max_evaluation_steps: tp.Optional[int] = None


@etr.auto_pytree
class TrainerConfigureModelOutput:
	model: EasyDeLBaseModule
	tx: GradientTransformation
	scheduler: Schedule
	config: tp.Optional[EasyDeLBaseConfig] = None


@etr.auto_pytree
class TrainerConfigureFunctionOutput:
	sharded_training_step_function: JitWrapped
	mesh: Mesh
	checkpoint_manager: CheckpointManager
	sharded_evaluation_step_function: tp.Optional[JitWrapped] = None


@etr.auto_pytree
class TrainerOutput:
	state: EasyDeLState
	mesh: tp.Optional[jax.sharding.Mesh]
	last_save_file_name: tp.Optional[str] = None
	checkpoint_path: tp.Optional[str] = None


class BaseTrainerProtocol(metaclass=ABCMeta):
	# Required attributes for all trainers
	arguments: TrainingArguments
	dataset_train: tp.Optional[Dataset]
	dataset_eval: tp.Optional[Dataset]
	data_collator: tp.Optional[tp.Callable]
	finetune: bool
	checkpoint_path: tp.Optional[tp.Union[str, os.PathLike]]
	dtype: tp.Any  # jax.numpy.dtype
	param_dtype: tp.Any  # jax.numpy.dtype

	timer: Timers
	wandb_runtime: tp.Any  # wandb runtime
	dataloader_train: tp.Iterator[np.ndarray]
	dataloader_eval: tp.Optional[tp.Iterator[np.ndarray]]
	max_training_steps: int
	max_evaluation_steps: int
	_model: EasyDeLBaseModule
	config: EasyDeLBaseConfig
	scheduler: optax.Schedule
	tx: optax.GradientTransformation
	model_state: EasyDeLState

	sharded_training_step_function: JitWrapped
	train_tracker: CompilationTracker
	sharded_evaluation_step_function: JitWrapped
	evalu_tracker: CompilationTracker

	mesh: tp.Any
	checkpoint_manager: tp.Any
	state_shape: tp.Any
	state_partition_spec: tp.Any
	state_named_sharding: tp.Any
	state: tp.Any
	pruning_module: tp.Any
	memory_monitor: tp.Any

	@abstractmethod
	def __init__(
		self,
		arguments: TrainingArguments,
		model: EasyDeLBaseModule,
		dataset_train: tp.Optional[Dataset] = None,
		dataset_eval: tp.Optional[Dataset] = None,
		finetune: bool = True,
		checkpoint_path: tp.Optional[tp.Union[str, os.PathLike]] = None,
	):
		"""
		Initializes the trainer.
		"""
		...

	@property
	@abstractmethod
	def model(self): ...

	@property
	@abstractmethod
	def mesh(self): ...

	@property
	@abstractmethod
	def training_batch_size(self): ...

	@property
	@abstractmethod
	def evaluation_batch_size(self): ...

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
		truncation_mode: tp.Literal["keep_end", "keep_start"],
	) -> tp.Callable:
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
	def _save_state(self, state: EasyDeLState, *args, **kwargs) -> str:
		"""
		Saves the model state to a checkpoint file.

		This method handles generating the filename, managing the checkpoint limit, saving the
		state using the `EasyDeLState.save_state` method, and creating a README file with
		information about the trained model.
		Args:
				state (EasyDeLState): The EasyDeLState object containing the model state to save.
		Returns:
				str: The filename of the saved checkpoint.
		"""
		...

	@abstractmethod
	def _get_current_step(self, state):
		"""
		Get the current step number.
		"""
		...

	@abstractmethod
	def _manage_checkpoint_limit(self, checkpoint_dir):
		"""
		Manages the checkpoint limit by deleting old checkpoints.
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
	def save_information(self, output_path: tp.Union[str, Path]) -> None:
		"""
		Save the generated information to a markdown file.
		"""
		...

	@abstractmethod
	def save_pretrained(
		self,
		state: EasyDeLState,
		save_directory: tp.Optional[str] = None,
		gather_fns: tp.Optional[
			tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]
		] = None,
		to_torch: bool = False,
		base_hf_auto_class=None,
		easystate_to_huggingface_model_kwargs: tp.Optional[dict] = None,
		torch_save_pretrained_kwargs: tp.Optional[dict] = None,
	):
		"""
		Saves the model state as a checkpoint file or to a Torch compatible directory.
		"""
		...

	@abstractmethod
	def _save_to_torch(
		self,
		state,
		save_directory,
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
		state: EasyDeLState,
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

	@staticmethod
	@abstractmethod
	def count_model_parameters(prm):
		"""Prints the number of model parameters in billions."""
		...

	@abstractmethod
	def apply_training_hooks(self, metrics: LossMetrics) -> LossMetrics:
		"""Apply training hooks to the model."""
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
	def _should_run_evaluation(self, current_step):
		"""Determine if evaluation process should be runned current step."""
		...

	@abstractmethod
	def _prepare_training_output(
		self,
		state: EasyDeLState,
		shard_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
		gather_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
		run_exception: tp.Optional[Exception] = None,
	):
		"""Prepare training output after training loop completion."""
		...

	@abstractmethod
	def _handle_training_interruption(
		self,
		state: EasyDeLState,
		exception: Exception,
		shard_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
		gather_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
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
	def create_progress_bar(
		total: int,
		desc: str = "",
		disabled: bool = False,
	) -> BaseProgressBar:
		"""Create a progress bar of the specified type."""

	@abstractmethod
	def log_weight_distribution(self, state: EasyDeLState, step: int):
		"""Log distribution of weights."""

	@abstractmethod
	def log_metrics(
		self,
		metrics: tp.Dict[str, float],
		pbar: BaseProgressBar,
		step: int,
		mode: str = "train",
	) -> None:
		"""Log metrics and update progress bar."""

	@abstractmethod
	def _run_training_loop(
		self,
		state: EasyDeLState,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		start_time: float,
		shard_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
		gather_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
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
		train_dataset: int,
		metrics_tracker: MetricsTracker,
		step_metrics: StepMetrics,
		pbar: BaseProgressBar,
		epoch: int,
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
		pbar: BaseProgressBar,
		start_time: float,
	):
		"""Handles training for a single epoch."""
		...

	@property
	@abstractmethod
	def _train_shared_fn_extra_args(self) -> tp.Tuple[tp.Any]: ...

	@property
	@abstractmethod
	def _eval_shared_fn_extra_args(self) -> tp.Tuple[tp.Any]: ...

	@abstractmethod
	def _execute_eval_step(self, state, batch) -> LossMetrics:
		"""Execute a single eval step."""
		...

	@abstractmethod
	def _execute_train_step(
		self, state, batch
	) -> tp.Tuple[EasyDeLState, LossMetrics, Exception]:
		"""Execute a single train step."""
		...

	@abstractmethod
	def _finalize_training(self, output, run_exception):
		"""Finalize training and prepare output."""
		...

	@abstractmethod
	def train(
		self,
		model_parameters: tp.Optional[flax.core.FrozenDict] = None,
		state: tp.Optional[EasyDeLState] = None,
	) -> tp.Any:
		"""Train using the provided model state."""
		...

	@abstractmethod
	def eval(self, model_state: EasyDeLState) -> tp.Iterator[dict]:
		"""
		Evaluates using the provided model state.
		"""
		...

	@abstractmethod
	def start_training_hook(self):
		"""Hook to run before training starts."""

	@abstractmethod
	def start_evaluation_hook(self):
		"""Hook to run before evaluation starts."""

	@abstractmethod
	def compile_aot(self) -> bool:
		"""Compile the state ahead of time for faster execution."""
		...

	@abstractmethod
	def finish(self):
		"""Finalize the training process."""
		...

	@abstractmethod
	def on_step_start(
		self,
		state: EasyDeLState,
		step: int,
	) -> EasyDeLState:
		"""hook process to call in start of the step."""
		...

	@abstractmethod
	def on_step_end(
		self,
		state: EasyDeLState,
		metrics: MetricsType,
		step: int,
	) -> tp.Tuple[EasyDeLState, MetricsType]:
		"""hook process to call in start of the step."""
		...

	@abstractmethod
	def _preprocess_batch_input(
		self,
		state: EasyDeLState,
		batch: tp.Dict[str, jax.Array],
		is_train: bool,
	) -> tp.Tuple[tp.Dict[str, jax.Array], tp.Dict[str, tp.Union[float, int, str]]]:
		"""hook call before passing data to function (called in `_execute` functions)"""

	@abstractmethod
	def get_runstage_flops(self, is_training: bool) -> float:
		"""Return the total number of FLOPs for the model."""
		...

	@abstractmethod
	def _ensure_functions_compiled(self):
		"""Ensure functions are compiled."""
		...

	@abstractmethod
	def __repr__(self):
		"""Return a string representation of the trainer."""
		...

	@abstractmethod
	def __str__(self):
		"""Return a string representation of the trainer."""
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
		metrics: LossMetrics,
		current_step: int,
		epoch: int,
		flops: float,
		batch_size: int,
		seq_length: int,
		learning_rate: float,
		mode: tp.Optional[tp.Literal["eval", "train"]] = None,
		**extras,
	) -> tp.Dict[str, float]:
		"""Calculate comprehensive metrics for the training step."""
		step_time = time.time() - self.step_start_time
		total_time = time.time() - self.start_time

		visited_tokens = seq_length * (current_step) * batch_size
		throughput = (seq_length * batch_size) / step_time
		flops_per_token = flops / visited_tokens
		flops_per_sequence = flops / ((current_step) * batch_size)

		flops_pre_second = flops / step_time
		flops_token_pre_second = flops / visited_tokens
		flops_sequence_pre_second = flops / ((current_step) * batch_size)
		mlperf_metrics = {
			"mlperf/flops": float(flops),
			"mlperf/flops_per_token": float(flops_per_token),
			"mlperf/flops_per_sequence": float(flops_per_sequence),
			"mlperf/flops_pre_second": float(flops_pre_second),
			"mlperf/flops_token_pre_second": float(flops_token_pre_second),
			"mlperf/flops_sequence_pre_second": float(flops_sequence_pre_second),
			"mlperf/throughput": throughput,
			"mlperf/step_time": float(step_time),
			"mlperf/execution_time": float(metrics.execution_time),
			"mlperf/total_time": float(total_time),
		}
		loss = metrics.loss
		z_loss = metrics.z_loss
		basic_metrics = {
			"loss": float(loss),
			"z_loss": float(z_loss) if z_loss is not None else None,
			"learning_rate": float(np.array(learning_rate).item()),
			"step": int(current_step),
			"perplexity": float(jnp.exp(loss)),
			"visited_tokens": visited_tokens,
			"epoch": int(epoch),
			**extras,
		}
		if metrics.accuracy is not None:
			basic_metrics["accuracy"] = float(metrics.accuracy)

		if metrics.chosen_rewards is not None:
			basic_metrics["chosen_rewards"] = float(jnp.mean(metrics.chosen_rewards).item())
		if metrics.rejected_rewards is not None:
			basic_metrics["rejected_rewards"] = float(
				jnp.mean(metrics.rejected_rewards).item()
			)
		if metrics.other_metrics is not None:
			basic_metrics.update(metrics.other_metrics)
		if not self.arguments.performance_mode and (mode == "train" or mode is None):
			detailed_metrics = self._calculate_detailed_metrics(metrics)
			basic_metrics.update(detailed_metrics)
		if mode is not None:
			basic_metrics = {f"{mode}/{k}": v for k, v in basic_metrics.items()}
		basic_metrics.update(mlperf_metrics)

		return basic_metrics

	def _calculate_detailed_metrics(self, metrics: LossMetrics):
		"""Calculate additional detailed metrics."""
		detailed_metrics = {}
		getattr_in = lambda x: x if not hasattr(x, "value") else x.value  # noqa
		if self.arguments.log_grad_norms:
			if metrics.max_grad_norm is not None:
				detailed_metrics.update(
					{"train/max_grad_norm": getattr_in(metrics.max_grad_norm).tolist()}
				)

			if metrics.mean_grad_norm is not None:
				detailed_metrics.update(
					{"train/mean_grad_norm": getattr_in(metrics.mean_grad_norm).tolist()}
				)

			# Add per-layer gradient norms
			if metrics.grad_norms is not None:
				detailed_metrics.update(
					{
						f"grad_norm/{'.'.join([str(s) for s in layer_name])}": getattr_in(
							grad_norm
						).tolist()
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
			mean_loss = self.loss_sum / (step - self.step_offset)
			if accuracy != float("inf"):
				if accuracy is None:
					accuracy = 0.0
				self.accuracy_sum = (
					accuracy if self.accuracy_sum is None else self.accuracy_sum + accuracy
				)
				mean_accuracy = self.accuracy_sum / (step - self.step_offset)

				return float(mean_loss), float(mean_accuracy)
			return float(mean_loss)

	def reset(self, step):
		"""Reset tracked metrics."""
		self.loss_sum = None
		self.accuracy_sum = None
		self.step_offset = step


class MetricsColumn(ProgressColumn):
	"""A custom progress column for displaying metrics."""

	def __init__(self, metrics_to_show=None):
		super().__init__()
		self.metrics_to_show = metrics_to_show

	def render(self, task: Task) -> Text:
		"""Render the metrics in a organized way."""
		if not task.fields.get("metrics"):
			return Text("")

		metrics = task.fields["metrics"]
		display_items = []

		for key, value in metrics.items():
			if self.metrics_to_show is None:
				if isinstance(value, float):
					if abs(value) < 0.01 or abs(value) > 1000:
						formatted_value = f"{value:.4e}"
					else:
						formatted_value = f"{value:.4f}"
				else:
					formatted_value = str(value)

				display_items.append(f"{key}={formatted_value}")
			else:
				if any(metric in key for metric in self.metrics_to_show):
					if isinstance(value, float):
						if abs(value) < 0.01 or abs(value) > 1000:
							formatted_value = f"{value:.4e}"
						else:
							formatted_value = f"{value:.4f}"
					else:
						formatted_value = str(value)

					display_items.append(f"{key}={formatted_value}")

		return Text(" • ".join(display_items), style="cyan")


class BaseProgressBar(abc.ABC):
	"""Abstract base class for progress bar implementations."""

	@abc.abstractmethod
	def update(self, n: int = 1) -> None:
		pass

	@abc.abstractmethod
	def set_postfix(self, **kwargs) -> None:
		pass

	@abc.abstractmethod
	def reset(self) -> None:
		pass

	@abc.abstractmethod
	def close(self) -> None:
		pass


class NullProgressBar(BaseProgressBar):
	"""Dummy progress bar that does nothing - useful for multiprocessing."""

	def update(self, n: int = 1) -> None:
		pass

	def set_postfix(self, **kwargs) -> None:
		pass

	def reset(self) -> None:
		pass

	def close(self) -> None:
		pass


class TqdmProgressBar(BaseProgressBar):
	"""Wrapper for tqdm progress bar."""

	def __init__(self, pbar: tqdm.tqdm):
		self.pbar = pbar

	def update(self, n: int = 1) -> None:
		self.pbar.update(n)

	def set_postfix(self, **kwargs) -> None:
		self.pbar.set_postfix(**kwargs)

	def reset(self) -> None:
		self.pbar.n = 0
		self.pbar.start_t = self.pbar._time()

	def close(self) -> None:
		self.pbar.close()


class JSONProgressBar(BaseProgressBar):
	"""Wrapper for JSON"""

	def __init__(self, desc=""):
		self.desc = desc

	def update(self, n: int = 1) -> None: ...

	def set_postfix(self, **kwargs) -> None:
		for k in list(kwargs.keys()):
			val = kwargs.get(k)
			if hasattr(val, "size") and val.size == 1:
				kwargs[k] = val.item()

		print(kwargs)

	def reset(self) -> None: ...

	def close(self) -> None: ...


class RichProgressBar(BaseProgressBar):
	"""Wrapper for rich progress bar."""

	def __init__(self, progress: Progress, task_id: TaskID):
		"""Initialize RichProgressBar with an existing Progress instance and task_id."""
		self.progress = progress
		self.task_id = task_id
		self._postfix = {}

	def update(self, n: int = 1) -> None:
		self.progress.update(self.task_id, advance=n)

	def set_postfix(self, **kwargs) -> None:
		self._postfix.update(kwargs)
		self.progress.update(self.task_id, metrics=self._postfix)

	def reset(self) -> None:
		self.progress.reset(self.task_id)
		self._postfix = {}

	def close(self) -> None:
		try:
			self.progress.remove_task(self.task_id)
		except KeyError:
			pass
