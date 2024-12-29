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

import functools
import re
import typing as tp
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import (
	AVAILABLE_OPTIMIZERS,
	AVAILABLE_PRUNING_TYPE,
	AVAILABLE_SCHEDULERS,
	AVAILABLE_SPARSE_MODULE_TYPES,
	EasyDeLOptimizers,
	EasyDeLSchedulers,
	get_logger,
)
from easydel.infra.loss_utils import LossConfig

from .utils import JaxDistributedConfig

try:
	import wandb  # type: ignore # noqa: F821
except ImportError:
	wandb = None


import flax.metrics.tensorboard

logger = get_logger(__name__)


# Constants
AVAILABLE_BACKENDS: tp.List[str] = ["cpu", "gpu", "tpu", None]


@dataclass
class TrainingArguments:
	model_name: str = "EasyDeL-Model"
	num_train_epochs: int = 10
	total_batch_size: int = 32
	eval_batch_size: tp.Optional[int] = None
	max_training_steps: tp.Optional[int] = None
	max_evaluation_steps: tp.Optional[int] = None
	optimizer: AVAILABLE_OPTIMIZERS = EasyDeLOptimizers.ADAMW
	scheduler: AVAILABLE_SCHEDULERS = EasyDeLSchedulers.NONE
	learning_rate: float = 5e-5
	learning_rate_end: tp.Optional[float] = None
	gradient_accumulation_steps: int = 1
	clip_grad: tp.Optional[float] = None
	weight_decay: float = 0.01
	loss_config: tp.Optional[LossConfig] = None
	frozen_parameters: tp.Optional[str] = None
	max_sequence_length: tp.Optional[int] = 4096
	is_fine_tuning: bool = True
	do_train: bool = True
	do_eval: bool = False
	train_on_inputs: bool = True
	backend: tp.Optional[str] = None
	extra_optimizer_kwargs: dict = field(default_factory=dict)
	evaluation_steps: tp.Optional[int] = None
	save_steps: tp.Optional[int] = None
	save_directory: str = "EasyDeL-Checkpoints"
	save_total_limit: tp.Optional[int] = None
	use_wandb: bool = True
	ids_to_pop_from_dataset: tp.Optional[list] = field(default_factory=list)
	remove_ckpt_after_load: bool = False
	do_last_save: bool = True
	model_parameters: tp.Optional[dict] = None
	track_memory: tp.Optional[bool] = None

	truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end"
	warmup_steps: int = 500
	step_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp")
	training_time: tp.Optional[str] = None
	dataloader_num_workers: tp.Optional[int] = 0
	dataloader_pin_memory: tp.Optional[bool] = False
	jax_distributed_config: tp.Optional[dict] = None
	log_all_workers: bool = False
	wandb_entity: tp.Optional[str] = None
	save_optimizer_state: bool = False
	step_start_point: tp.Optional[int] = None
	verbose: bool = True
	offload_device: jax.Device = jax.devices("cpu")[0]
	pruning_module: AVAILABLE_PRUNING_TYPE = None
	sparsify_module: bool = False
	sparse_module_type: AVAILABLE_SPARSE_MODULE_TYPES = "bcoo"
	state_apply_fn_kwarguments_to_model: tp.Optional[dict] = None
	remove_unused_columns: bool = True
	performance_mode: bool = False
	log_grad_norms: bool = True

	def __post_init__(self):
		"""
		Validates the configuration, sets up distributed training,
		initializes the optimizer, configures logging.
		This method is automatically called after the object is initialized.
		"""
		self._validate_config()
		self._setup_distributed()
		self._setup_optimizer()
		self._setup_logging()
		self._ensure_variables()

	def _validate_config(self):
		"""
		Performs validation checks on the provided configuration settings.
		Raises ValueError if any configuration is invalid.
		"""
		assert (
			self.gradient_accumulation_steps > 0
		), "`gradient_accumulation_steps` can't be lower than 1."
		if self.total_batch_size % self.gradient_accumulation_steps != 0:
			raise ValueError(
				"Number of `total_batch_size` should be even with `gradient_accumulation_steps`"
			)
		if self.backend not in AVAILABLE_BACKENDS:
			raise ValueError(
				f"Backend {self.backend} is not recognized. Available backends: {AVAILABLE_BACKENDS}"
			)

	def _setup_distributed(self):
		"""
		Sets up JAX distributed training based on the chosen backend and sharding configuration.
		Determines the number of available devices and sets up the device mesh.
		"""
		self.available_backends = len(jax.devices(self.backend))

		JaxDistributedConfig.initialize(self.jax_distributed_config)

	def _setup_optimizer(self):
		"""
		Configures the optimizer and learning rate scheduler based on the provided arguments.
		Sets up the optimizer_kwargs dictionary.
		"""
		extra_optimizer_kwargs = (
			self.extra_optimizer_kwargs if self.extra_optimizer_kwargs is not None else {}
		)
		self.optimizer_kwargs = {
			"learning_rate": self.learning_rate,
			"learning_rate_end": self.learning_rate_end,
			"optimizer": self.optimizer,
			"scheduler": self.scheduler,
			"warmup_steps": self.warmup_steps,
			"gradient_accumulation_steps": self.gradient_accumulation_steps,
			"weight_decay": self.weight_decay,
			"steps": self.max_training_steps,
			**extra_optimizer_kwargs,
		}

	def _setup_logging(self):
		"""
		Sets up logging for training using TensorBoard and Weights & Biases.
		Handles warnings if performance mode is enabled and disables WandB logging accordingly.
		"""
		if self.use_wandb and self.performance_mode:
			warnings.warn("WandB logging disabled due to performance mode", stacklevel=1)
			self.use_wandb = False

		self._stop_capturing_memory = False
		self._captured_memory = {}

	def _ensure_variables(self):
		"""
		Checks and sets up variables for start.
		"""
		self.step_start_point = self.step_start_point or 0
		self.eval_batch_size = (
			self.eval_batch_size
			if self.eval_batch_size is not None
			else self.total_batch_size
		)

	@staticmethod
	def _time_to_seconds(time_str: str) -> int:
		"""
		Converts a time string in the format "50min" or "23h" to seconds.

		Args:
		    time_str (str): The time string to convert.

		Returns:
		    int: The equivalent time in seconds.
		"""
		match = re.match(r"(\d+)\s*(h|min)", time_str.lower())
		if not match:
			raise ValueError(
				"Invalid time format. Use `50min` for minutes or `23h` for hours."
			)
		value, unit = match.groups()
		return int(value) * (3600 if unit == "h" else 60)

	def get_path(self) -> Path:
		"""
		Returns the path to the checkpoint directory.

		Returns:
		    Path: The path to the checkpoint directory.
		"""
		return Path(self.save_directory, self.model_name)

	def ensure_checkpoint_path(self):
		"""
		Creates the checkpoint directory if it doesn't exist.
		"""
		path = self.get_path()
		path.mkdir(parents=True, exist_ok=True)

	def get_optimizer_and_scheduler(self, steps: tp.Optional[int] = None):
		"""
		Returns the configured optimizer and learning rate scheduler.

		Args:
		    steps (tp.Optional[int]): The number of training steps.
		        If not provided, uses the value from `self.optimizer_kwargs`.

		Returns:
		    tuple: A tuple containing the optimizer and scheduler.
		"""
		from easydel.etils.auto_tx import get_optimizer_and_scheduler

		self.optimizer_kwargs["steps"] = steps or self.optimizer_kwargs["steps"]
		tx, sc = get_optimizer_and_scheduler(**self.optimizer_kwargs)
		return tx, sc

	def get_streaming_checkpointer(self):
		"""
		Returns the checkpoint manager, responsible for saving model checkpoints.

		Returns:
		    fjformer.CheckpointManager: The checkpoint manager.
		"""
		import os.path

		from easydel.utils.checkpoint_managers import CheckpointManager

		return CheckpointManager(
			checkpoint_dir=os.path.join(self.save_directory, self.model_name),
			save_optimizer_state=self.save_optimizer_state,
			verbose=self.verbose,
		)

	@functools.cached_property
	def _tensorboard(self):
		return flax.metrics.tensorboard.SummaryWriter(
			log_dir=str(self._get_save_directory(create=True))
		)

	def get_tensorboard(self):
		"""
		Returns the TensorBoard SummaryWriter, used for logging metrics.

		Returns:
		    flax.metrics.tensorboard.SummaryWriter: The TensorBoard SummaryWriter.
		"""

		return self._tensorboard

	def get_wandb_init(self):
		"""
		Initializes Weights & Biases for experiment tracking if enabled.

		Returns:
		    tp.Optional[wandb.sdk.wandb_run.Run]: The WandB run object if initialized, else None.
		"""
		if not self.use_wandb or wandb is None:
			warnings.warn(
				"you have used `use_wandb=True` but you haven't install wandb.",
				stacklevel=1,
			)
			return None

		if not self.log_all_workers and jax.process_index() != 0:
			return None

		return wandb.init(
			project=f"EasyDeL-{self.model_name}",
			config=self.to_dict(),
			tags=["EasyDeL", "Jax/Flax"],
			entity=self.wandb_entity,
		)

	def ensure_training_time(self, time_passed):
		if self.training_time is not None and time_passed > self._time_to_seconds(
			self.training_time
		):
			raise EasyDeLTimerError("Time Out")

	def log_metrics(
		self,
		metrics: tp.Dict[
			str,
			tp.Union[
				float,
				tp.List,
				tp.Tuple,
				np.ndarray,
				"jnp.ndarray",
				"torch.Tensor",  # type: ignore # noqa: F821
			],
		],
		step: int,
	):
		"""
		Logs training metrics to Weights & Biases and/or TensorBoard.

		Args:
		    metrics (tp.Dict[str, tp.Union[float, tp.List, tp.Tuple, np.ndarray, 'jnp.ndarray', 'torch.Tensor']]):
		        A dictionary where keys are metric names and values are metric values.
		    step (int): The current training step or iteration.
		"""

		def restructure_metric_name(metric_name):
			if metric_name.startswith("train/grad_norm/"):
				return metric_name.replace("train/grad_norm/", "grad_norm/")
			return metric_name

		with jax.spmd_mode("allow_all"):
			metrics = {restructure_metric_name(k): v for k, v in metrics.items()}
			self._log_to_wandb(metrics, step)
			self._log_to_tensorboard(metrics, step)

	def _log_to_wandb(self, metrics, step):
		"""
		Log metrics to Weights & Biases (wandb).

		This method processes the given metrics and logs them to wandb if it's enabled and properly initialized.

		Args:
		    metrics (dict): A dictionary of metrics to log. Keys are metric names, values are the metric values.
		    step (int): The current step or iteration number.

		Notes:
		    - If a metric value is a list, tuple, or numpy array, it's converted to a wandb.Histogram.
		    - For JAX arrays or PyTorch tensors, they're converted to numpy arrays before creating a histogram.
		    - Other types of values are logged as-is.
		    - Any exceptions during logging are caught and warned about, allowing the process to continue.
		"""
		if self.use_wandb and wandb is not None:

			class Tensor: ...

			try:
				from torch import Tensor
			except ModuleNotFoundError:
				...
			wandb_metrics = {}
			for key, value in metrics.items():
				try:
					if isinstance(value, (list, tuple, np.ndarray)):
						wandb_metrics[key] = self._create_wandb_histogram(value)
					elif isinstance(
						value,
						(
							jnp.ndarray,
							Tensor,
						),
					):  # Use string for Tensor to avoid import issues
						wandb_metrics[key] = self._create_wandb_histogram(
							value.cpu().numpy() if hasattr(value, "cpu") else np.array(value)
						)
					else:
						wandb_metrics[key] = value
				except Exception as e:
					warnings.warn(f"Failed to log metric {key} to wandb: {e}", stacklevel=3)

			wandb.log(wandb_metrics, step=step)

	def _log_to_tensorboard(self, metrics, step):
		"""
		Log metrics to TensorBoard.

		This method processes the given metrics and logs them to TensorBoard.

		Args:
		    metrics (dict): A dictionary of metrics to log. Keys are metric names, values are the metric values.
		    step (int): The current step or iteration number.

		Notes:
		    - Scalar values (float, int) are logged using summary_writer.scalar().
		    - Lists, tuples, numpy arrays, JAX arrays, and PyTorch tensors are logged as histograms.
		    - JAX arrays and PyTorch tensors are converted to numpy arrays before logging.
		    - tp.Any exceptions during logging are caught and warned about, allowing the process to continue.
		    - The summary writer is flushed after logging all metrics.
		"""

		class Tensor: ...

		try:
			from torch import Tensor
		except ModuleNotFoundError:
			...
		summary_writer = self.get_tensorboard()
		for key, value in metrics.items():
			try:
				if isinstance(value, (float, int)):
					summary_writer.scalar(key, value, step)
				elif isinstance(
					value,
					(
						list,
						tuple,
						np.ndarray,
						jnp.ndarray,
						Tensor,
					),
				):
					if hasattr(value, "cpu"):
						value = value.cpu().numpy()
					elif isinstance(value, jnp.ndarray):
						value = np.array(value)
					summary_writer.histogram(key, value, step)
			except Exception as e:
				warnings.warn(f"Failed to log metric {key} to TensorBoard: {e}", stacklevel=1)

		summary_writer.flush()

	def _create_wandb_histogram(self, value):
		"""
		Create a wandb.Histogram object from the given value.

		This method handles the conversion of various data types to a format suitable for wandb histograms.

		Args:
		    value: The value to convert into a wandb.Histogram. Can be a list, tuple, numpy array, etc.

		Returns:
		    wandb.Histogram or None: A wandb.Histogram object if successful, None if an error occurs.

		Notes:
		    - Non-numpy array inputs are converted to numpy arrays.
		    - float16 and bfloat16 dtypes are converted to float32 to avoid potential issues.
		    - tp.Any exceptions during histogram creation are caught and logged, returning None in such cases.
		"""
		try:
			# Convert to numpy array if it's not already
			if not isinstance(value, np.ndarray):
				value = np.array(value)

			# Handle different dtypes
			if value.dtype in [np.float16, np.bfloat16]:
				value = value.astype(np.float32)

			return wandb.Histogram(value)
		except Exception as e:
			(f"Failed to create wandb histogram: {e}")
			return None

	def to_dict(self) -> tp.Dict[str, tp.Any]:
		"""
		Converts the TrainingArguments object into a dictionary.

		Returns:
		    tp.Dict[str, tp.Any]: A dictionary representation of the TrainingArguments.
		"""
		return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

	@classmethod
	def from_dict(cls, config: tp.Dict[str, tp.Any]) -> "TrainingArguments":
		"""
		Creates a TrainingArguments instance from a dictionary.

		Args:
		    config (tp.Dict[str, tp.Any]): The configuration dictionary.

		Returns:
		    TrainingArguments: A TrainingArguments object initialized with values from the dictionary.
		"""
		return cls(**config)

	def __str__(self):
		"""
		Returns a formatted string representation of the configuration.

		Returns:
		    str: A formatted string showing the configuration settings.
		"""
		return self._pretty_print(self.to_dict())

	@staticmethod
	def _pretty_print(d: tp.Dict[str, tp.Any], indent: int = 0) -> str:
		"""
		Helper function for pretty-printing a dictionary.

		Args:
		    d (tp.Dict[str, tp.Any]): The dictionary to pretty-print.
		    indent (int): The indentation level.

		Returns:
		    str: The pretty-printed string representation of the dictionary.
		"""
		result = []
		for key, value in d.items():
			result.append(" " * indent + str(key) + ":")
			if isinstance(value, dict):
				result.append(TrainingArguments._pretty_print(value, indent + 2))
			else:
				result.append(" " * (indent + 2) + str(value))
		return "\n".join(result)

	def _get_save_directory(self, create: bool = True) -> Path:
		bd = Path(self.save_directory)
		dir = bd / Path(self.model_name)
		if create:
			dir.mkdir(exist_ok=True, parents=True)
		return dir

	def _get_save_directory_milestone(self, step, create: bool = True) -> Path:
		directory_name = f"run-{step}"
		save_directory = self._get_save_directory(create=create) / directory_name
		if create:
			save_directory.mkdir(exist_ok=True, parents=True)
		return save_directory
