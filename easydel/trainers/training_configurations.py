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

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from easydel.infra.errors import EasyDeLTimerError
from easydel.infra.etils import (
	AVAILABLE_OPTIMIZERS,
	AVAILABLE_PRUNING_TYPE,
	AVAILABLE_SCHEDULERS,
	AVAILABLE_SPARSE_MODULE_TYPES,
	EasyDeLOptimizers,
	EasyDeLSchedulers,
)
from easydel.infra.loss_utils import LossConfig
from easydel.utils.helpers import get_logger

from .utils import JaxDistributedConfig

try:
	import wandb  # type: ignore # noqa: F821
except ImportError:
	wandb = None

if tp.TYPE_CHECKING:
	from flax.metrics.tensorboard import SummaryWriter
	from jax import Array
	from torch import Tensor

	MetricsType = tp.Dict[
		str,
		tp.Union[float, tp.List, tp.Tuple, np.ndarray, Array, Tensor],
	]
else:
	SummaryWriter = tp.Any
	MetricsType = tp.Any
logger = get_logger(__name__)


# Constants
AVAILABLE_BACKENDS: tp.List[str] = ["cpu", "gpu", "tpu", None]


@dataclass
class TrainingArguments:
	auto_shard_states: bool = True
	backend: tp.Optional[str] = None
	clip_grad: tp.Optional[float] = None
	dataloader_num_workers: tp.Optional[int] = 0
	dataloader_pin_memory: tp.Optional[bool] = False
	do_eval: bool = False
	do_last_save: bool = True
	do_train: bool = True
	eval_batch_size: tp.Optional[int] = None
	evaluation_steps: tp.Optional[int] = None
	extra_optimizer_kwargs: dict = field(default_factory=dict)
	frozen_parameters: tp.Optional[str] = None
	gradient_accumulation_steps: int = 1
	ids_to_pop_from_dataset: tp.Optional[list] = field(default_factory=list)
	is_fine_tuning: bool = True
	jax_distributed_config: tp.Optional[dict] = None
	learning_rate: float = 5e-5
	learning_rate_end: tp.Optional[float] = None
	log_all_workers: bool = False
	log_grad_norms: bool = True
	report_metrics: bool = True
	log_steps: int = 10
	loss_config: tp.Optional[LossConfig] = None
	max_evaluation_steps: tp.Optional[int] = None
	max_sequence_length: tp.Optional[int] = 4096
	max_training_steps: tp.Optional[int] = None
	model_name: str = "EasyDeL-Model"
	model_parameters: tp.Optional[dict] = None
	metrics_to_show_in_rich_pbar: tp.Optional[list] = None
	num_train_epochs: int = 10
	offload_device: jax.Device = jax.devices("cpu")[0]
	optimizer: AVAILABLE_OPTIMIZERS = EasyDeLOptimizers.ADAMW
	performance_mode: bool = False
	pruning_module: AVAILABLE_PRUNING_TYPE = None
	process_zero_is_admin: bool = True
	progress_bar_type: tp.Literal["tqdm", "rich", "json"] = "tqdm"
	remove_ckpt_after_load: bool = False
	remove_unused_columns: bool = True
	save_directory: str = "EasyDeL-Checkpoints"
	save_optimizer_state: bool = True
	save_steps: tp.Optional[int] = None
	save_total_limit: tp.Optional[int] = None
	scheduler: AVAILABLE_SCHEDULERS = EasyDeLSchedulers.NONE
	sparsify_module: bool = False
	sparse_module_type: AVAILABLE_SPARSE_MODULE_TYPES = "bcoo"
	state_apply_fn_kwarguments_to_model: tp.Optional[dict] = None
	step_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp")
	step_start_point: tp.Optional[int] = None
	total_batch_size: int = 32
	training_time_limit: tp.Optional[str] = None
	train_on_inputs: bool = True
	truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end"
	tx_mu_dtype: tp.Optional[jnp.dtype] = None
	track_memory: bool = False
	use_data_collactor: bool = True
	use_wandb: bool = True
	verbose: bool = True
	wandb_entity: tp.Optional[str] = None
	warmup_steps: int = 500
	weight_decay: float = 0.01

	@property
	def training_time_seconds(self) -> int:
		if self.training_time_limit is None:
			return None
		return self._time_to_seconds(self.training_time_limit)

	@functools.cached_property
	def is_process_zero(self):
		return jax.process_index() == 0

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
		assert self.gradient_accumulation_steps > 0, (
			"`gradient_accumulation_steps` can't be lower than 1."
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
			"mu_dtype": self.tx_mu_dtype,
			**extra_optimizer_kwargs,
		}

	def _setup_logging(self):
		"""
		Sets up logging for training using TensorBoard and Weights & Biases.
		Handles warnings if performance mode is enabled and disables WandB logging accordingly.
		"""
		if self.use_wandb and self.performance_mode:
			logger.info("WandB logging disabled due to performance mode")
			self.use_wandb = False
		if self.report_metrics and self.performance_mode:
			logger.info("Metrics reporting disabled due to performance mode")
			self.report_metrics = False
		if self.report_metrics:
			if not self.is_process_zero and not self.log_all_workers:
				logger.info(
					"Metrics reporting disabled and it's only working on process index 0 or "
					"admin process (`log_all_workers` is `False`)."
				)
				self.report_metrics = False

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
		if self.loss_config is None:
			self.loss_config = LossConfig()

	@staticmethod
	def _time_to_seconds(time_str: str) -> int:
		"""
		Converts a time string in the format "50min" or "23h" to seconds.

		Args:
		    time_str (str): The time string to convert.

		Returns:
		    int: The equivalent time in seconds.
		"""
		match = re.match(
			r"(\d+)\s*(h|hour|hours|min|m|minutes|s|sec|seconds)", time_str.lower()
		)
		if not match:
			raise ValueError(
				"Invalid time format. Use `50min` for minutes, `23h` for hours, or `30s` for seconds."
			)
		value, unit = match.groups()
		unit_to_seconds = {
			"h": 3600,
			"hour": 3600,
			"hours": 3600,
			"min": 60,
			"m": 60,
			"minutes": 60,
			"s": 1,
			"sec": 1,
			"seconds": 1,
		}.get(unit.lower())
		return int(value) * unit_to_seconds

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
		from easydel.trainers.auto_tx import get_optimizer_and_scheduler

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
		from flax.metrics.tensorboard import SummaryWriter

		return SummaryWriter(log_dir=str(self._get_save_directory(create=True)))

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
		if self.report_metrics:
			if not self.use_wandb or wandb is None:
				warnings.warn(
					"you have used `use_wandb=True` but you haven't install wandb.",
					stacklevel=1,
				)
				return None

			return wandb.init(
				project=f"EasyDeL-{self.model_name}",
				config=self.to_dict(),
				tags=["EasyDeL", "JAX/Flax"],
				entity=self.wandb_entity,
			)
		return None

	def ensure_training_time_limit(self, time_passed):
		if self.training_time_limit is not None and time_passed > self._time_to_seconds(
			self.training_time_limit
		):
			raise EasyDeLTimerError("Time Out")

	def log_metrics(self, metrics: MetricsType, step: int):
		"""
		Logs training metrics to Weights & Biases and/or TensorBoard.

		Args:
		  metrics (tp.Dict[str, tp.Union[float, tp.List, tp.Tuple, np.ndarray, 'jnp.ndarray', 'torch.Tensor']]):
		    A dictionary where keys are metric names and values are metric values.
		  step (int): The current training step or iteration.
		"""
		if self.report_metrics:
			metrics = {
				self._restructure_metric_name(k): v for k, v in metrics.items() if v is not None
			}
			self._log_to_wandb(metrics, step)
			self._log_to_tensorboard(metrics, step)

	def _restructure_metric_name(self, metric_name: str) -> str:
		"""
		Restructures the metric name for logging.

		Args:
		  metric_name (str): The original metric name.

		Returns:
		  str: The restructured metric name.
		"""
		if metric_name.startswith("train/grad_norm/"):
			return metric_name.replace("train/grad_norm/", "grad_norm/")
		return metric_name

	def _log_to_wandb(self, metrics, step):
		"""
		Log metrics to Weights & Biases (wandb).

		This method processes the given metrics and logs them to wandb if it's enabled and properly initialized.

		Args:
		  metrics (dict): A dictionary of metrics to log. Keys are metric names, values are the metric values.
		  step (int): The current step or iteration number.
		"""
		if self.use_wandb and wandb is not None:
			wandb_metrics = {}
			for key, value in metrics.items():
				try:
					wandb_metrics[key] = (
						self._create_wandb_histogram(value)
						if isinstance(value, (list, tuple, np.ndarray, jnp.ndarray))
						else value
					)
				except Exception as e:
					warnings.warn(f"Failed to log metric {key} to wandb: {e}", stacklevel=3)
			try:
				wandb.log(wandb_metrics, step=step)
			except Exception:
				...

	def _log_to_tensorboard(self, metrics, step):
		"""
		Log metrics to TensorBoard.

		This method processes the given metrics and logs them to TensorBoard.

		Args:
		    metrics (dict): A dictionary of metrics to log. Keys are metric names, values are the metric values.
		    step (int): The current step or iteration number.
		"""
		summary_writer = self.get_tensorboard()
		for key, value in metrics.items():
			try:
				if isinstance(value, (float, int)):
					summary_writer.scalar(key, value, step)
				elif isinstance(value, (list, tuple, np.ndarray, jnp.ndarray)):
					summary_writer.histogram(key, np.array(value), step)
			except Exception as e:
				warnings.warn(f"Failed to log metric {key} to TensorBoard: {e}", stacklevel=1)
			finally:
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
