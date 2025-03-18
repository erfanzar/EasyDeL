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
from copy import deepcopy
from dataclasses import field, fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from eformer.optimizers import OptimizerFactory, SchedulerConfig
from jax.sharding import PartitionSpec

from easydel.infra.errors import EasyDeLTimerError
from easydel.infra.etils import (
	AVAILABLE_OPTIMIZERS,
	AVAILABLE_SCHEDULERS,
	AVAILABLE_SPARSE_MODULE_TYPES,
	EasyDeLOptimizers,
	EasyDeLSchedulers,
)
from easydel.infra.loss_utils import LossConfig
from easydel.utils import traversals as etr
from easydel.utils.compiling_utils import hash_fn
from easydel.utils.helpers import get_logger

from .utils import JaxDistributedConfig, compute_weight_stats

try:
	import wandb  # type: ignore # noqa: F821
except ImportError:
	wandb = None

if tp.TYPE_CHECKING:
	from flax.metrics.tensorboard import SummaryWriter
	from jax import Array
	from torch import Tensor  # type:ignore

	MetricsType = tp.Dict[
		str,
		tp.Union[float, tp.List, tp.Tuple, np.ndarray, Array, Tensor],
	]
else:
	SummaryWriter = tp.Any
	MetricsType = tp.Any
logger = get_logger(__name__)


def get_safe_arr(xs):
	if isinstance(xs, (np.generic, jax.Array)):
		if xs.size == 1:  # Only try .item() on size-1 arrays
			return xs.item()
		return xs
	return xs


# Constants
AVAILABLE_BACKENDS: tp.List[str] = ["cpu", "gpu", "tpu", None]


@etr.auto_pytree
class TrainingArguments:
	auto_shard_states: bool = field(
		default=True,
		metadata={"help": "Whether to automatically shard model states across devices."},
	)
	aux_loss_enabled: bool = field(
		default=False,
		metadata={"help": "Whether to enable the auxiliary loss."},
	)
	backend: tp.Optional[str] = field(
		default=None,
		metadata={
			"help": "The JAX backend to use (e.g., 'cpu', 'gpu', 'tpu').  If None, JAX will choose."
		},
	)
	clip_grad: tp.Optional[float] = field(
		default=None,
		metadata={"help": "The value at which to clip the gradients."},
	)
	custom_scheduler: tp.Optional[tp.Callable[[int], tp.Any]] = field(
		default=None,
		metadata={
			"help": "A custom scheduler function that takes the current step as input."
		},
	)
	dataloader_num_workers: tp.Optional[int] = field(
		default=0,
		metadata={"help": "The number of workers to use for the dataloader."},
	)
	dataloader_pin_memory: tp.Optional[bool] = field(
		default=False,
		metadata={"help": "Whether to pin memory for the dataloader."},
	)
	do_eval: bool = field(
		default=False,
		metadata={"help": "Whether to run evaluation during training."},
	)
	do_last_save: bool = field(
		default=True,
		metadata={"help": "Whether to save the model at the end of training."},
	)
	do_train: bool = field(
		default=True,
		metadata={"help": "Whether to run training."},
	)
	eval_batch_size: tp.Optional[int] = field(
		default=None,
		metadata={"help": "The batch size to use for evaluation."},
	)
	evaluation_steps: tp.Optional[int] = field(
		default=None,
		metadata={"help": "Run evaluation every X steps."},
	)
	extra_optimizer_kwargs: dict = field(
		default_factory=dict,
		metadata={"help": "Additional keyword arguments to pass to the optimizer."},
	)
	frozen_parameters: tp.Optional[str] = field(
		default=None,
		metadata={"help": "A regex pattern of parameters to freeze (not train)."},
	)
	gradient_accumulation_steps: int = field(
		default=1,
		metadata={"help": "The number of steps to accumulate gradients over."},
	)
	ids_to_pop_from_dataset: tp.Optional[tp.List[str]] = field(
		default_factory=list,
		metadata={"help": "A list of dataset columns to remove before training."},
	)
	is_fine_tuning: bool = field(
		default=True,
		metadata={"help": "Whether the training is a fine-tuning run."},
	)
	init_tx: bool = field(
		default=True,
		metadata={"help": "Whether to initialize the training state."},
	)
	jax_distributed_config: tp.Optional[dict] = field(
		default=None,
		metadata={"help": "Configuration for JAX distributed training."},
	)
	learning_rate: float = field(
		default=5e-5,
		metadata={"help": "The learning rate."},
	)
	learning_rate_end: tp.Optional[float] = field(
		default=None,
		metadata={"help": "The final learning rate for linear decay schedulers."},
	)
	log_all_workers: bool = field(
		default=False,
		metadata={
			"help": "Whether to log metrics from all workers in a distributed setup."
		},
	)
	log_grad_norms: bool = field(
		default=True,
		metadata={"help": "Whether to log gradient norms."},
	)
	report_metrics: bool = field(
		default=True,
		metadata={"help": "Whether to report metrics to a logger."},
	)
	log_steps: int = field(
		default=10,
		metadata={"help": "Log metrics every X steps."},
	)
	loss_config: tp.Optional[LossConfig] = field(
		default=None,
		metadata={"help": "Configuration for the loss function."},
	)
	low_mem_usage: bool = field(
		default=True,
		metadata={"help": "Whether to try to minimize memory usage."},
	)
	max_evaluation_steps: tp.Optional[int] = field(
		default=None,
		metadata={"help": "Maximum number of evaluation steps."},
	)
	max_sequence_length: tp.Optional[int] = field(
		default=4096,
		metadata={"help": "The maximum sequence length."},
	)
	max_training_steps: tp.Optional[int] = field(
		default=None,
		metadata={"help": "The maximum number of training steps."},
	)
	model_name: str = field(
		default="BaseTrainer",
		metadata={"help": "The name of the model."},
	)
	model_parameters: tp.Optional[dict] = field(
		default=None,
		metadata={"help": "Model architecture config"},
	)
	metrics_to_show_in_rich_pbar: tp.Optional[tp.List[str]] = field(
		default=None,
		metadata={"help": "Metrics to display in the rich progress bar."},
	)
	num_train_epochs: int = field(
		default=10,
		metadata={"help": "The number of training epochs."},
	)
	offload_dataset: bool = field(
		default=False,
		metadata={"help": "Whether to offload the dataset to CPU or disk."},
	)
	offload_device_type: str = field(
		default="cpu",
		metadata={"help": "The device type to offload the dataset to (cpu or disk)."},
	)
	offload_device_index: int = field(
		default=0,
		metadata={"help": "The device index to offload the dataset to."},
	)
	optimizer: AVAILABLE_OPTIMIZERS = field(
		default=EasyDeLOptimizers.ADAMW,
		metadata={"help": "The optimizer to use."},
	)
	performance_mode: bool = field(
		default=False,
		metadata={"help": "Whether to enable performance mode (e.g., XLA compilation)."},
	)
	pruning_module: tp.Any = field(
		default=None,
		metadata={"help": "The pruning module to use."},
	)
	process_zero_is_admin: bool = field(
		default=True,
		metadata={"help": "Whether the process with rank 0 is the admin process."},
	)
	progress_bar_type: tp.Literal["tqdm", "rich", "json"] = field(
		default="tqdm",
		metadata={"help": "The type of progress bar to use."},
	)
	remove_ckpt_after_load: bool = field(
		default=False,
		metadata={"help": "Whether to remove the checkpoint after loading it."},
	)
	remove_unused_columns: bool = field(
		default=True,
		metadata={"help": "Whether to remove unused columns from the dataset."},
	)
	report_steps: int = field(
		default=5,
		metadata={"help": "Report metrics every X steps."},
	)
	save_directory: str = field(
		default="EasyDeL-Checkpoints",
		metadata={"help": "The directory to save checkpoints to."},
	)
	save_optimizer_state: bool = field(
		default=True,
		metadata={"help": "Whether to save the optimizer state along with the model."},
	)
	save_steps: tp.Optional[int] = field(
		default=None,
		metadata={"help": "Save a checkpoint every X steps."},
	)
	save_total_limit: tp.Optional[int] = field(
		default=None,
		metadata={"help": "The maximum number of checkpoints to keep."},
	)
	scheduler: AVAILABLE_SCHEDULERS = field(
		default=EasyDeLSchedulers.NONE,
		metadata={"help": "The scheduler to use."},
	)
	sparsify_module: bool = field(
		default=False,
		metadata={"help": "Whether to sparsify the model."},
	)
	sparse_module_type: AVAILABLE_SPARSE_MODULE_TYPES = field(
		default="bcoo",
		metadata={"help": "The type of sparse module to use."},
	)
	state_apply_fn_kwarguments_to_model: tp.Optional[dict] = field(
		default=None,
		metadata={"help": "Keyword arguments to pass to the state apply function."},
	)
	step_partition_spec: PartitionSpec = field(
		default=PartitionSpec(("dp", "fsdp"), "sp"),
		metadata={"help": "The partition specification for the training step."},
	)
	step_start_point: tp.Optional[int] = field(
		default=None,
		metadata={"help": "The step to start training from (for resuming)."},
	)
	shuffle_train_dataset: bool = field(
		default=True,
		metadata={"help": "Whether to shuffle the training dataset."},
	)
	total_batch_size: int = field(
		default=32,
		metadata={"help": "The total batch size."},
	)
	training_time_limit: tp.Optional[str] = field(
		default=None,
		metadata={"help": "The maximum training time (e.g., '1d', '2h30m')."},
	)
	train_on_inputs: bool = field(
		default=True,
		metadata={"help": "Whether to train on the input data."},
	)
	truncation_mode: tp.Literal["keep_end", "keep_start"] = field(
		default="keep_end",
		metadata={"help": "The truncation mode to use."},
	)
	tx_mu_dtype: tp.Optional[jnp.dtype] = field(
		default=None,
		metadata={"help": "The dtype to use for the `tx.mu` variable."},
	)
	track_memory: bool = field(
		default=False,
		metadata={"help": "Whether to track memory usage."},
	)
	use_data_collactor: bool = field(
		default=True,
		metadata={"help": "Whether to use a data collator."},
	)
	use_wandb: bool = field(
		default=True,
		metadata={"help": "Whether to use Weights & Biases for logging."},
	)
	verbose: bool = field(
		default=True,
		metadata={"help": "Whether to print verbose output."},
	)
	wandb_entity: tp.Optional[str] = field(
		default=None,
		metadata={"help": "The Weights & Biases entity."},
	)
	warmup_steps: int = field(
		default=0,
		metadata={"help": "The number of warmup steps for the learning rate scheduler."},
	)
	weight_decay: float = field(
		default=0.01,
		metadata={"help": "The weight decay value."},
	)
	weight_distribution_pattern: str = field(
		default=r".*?(layernorm|norm).*?",
		metadata={"help": "The pattern to use to extract weight distribution."},
	)
	weight_distribution_log_steps: int = field(
		default=0,
		metadata={"help": "log weight distribution every X steps."},
	)

	@property
	def offload_device(self):
		return jax.devices(self.offload_device_type)[self.offload_device_index]

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
			"clip_grad": self.clip_grad,
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

		self.optimizer_kwargs["steps"] = steps or self.optimizer_kwargs["steps"]
		optimizer_kwargs = deepcopy(self.optimizer_kwargs)
		scheduler = optimizer_kwargs.pop("scheduler", None)
		if scheduler == "none":
			scheduler = None
		if scheduler == EasyDeLSchedulers.NONE:
			scheduler = None
		scheduler_config = SchedulerConfig(
			scheduler_type=scheduler,
			steps=optimizer_kwargs.pop("steps"),
			learning_rate=optimizer_kwargs.pop("learning_rate"),
			learning_rate_end=optimizer_kwargs.pop("learning_rate_end"),
			warmup_steps=optimizer_kwargs.pop("warmup_steps"),
			exponent=optimizer_kwargs.pop("exponent", 1),
		)
		optimizer_kwargs.pop("gradient_accumulation_steps", 0)
		optimizer, scheduler = OptimizerFactory.create(
			optimizer_type=optimizer_kwargs.pop("optimizer"),
			scheduler_config=scheduler_config,
			clip_grad=optimizer_kwargs.pop("clip_grad"),
			weight_decay=optimizer_kwargs.pop("weight_decay"),
			custom_scheduler=self.custom_scheduler,
			**optimizer_kwargs,
		)
		return optimizer, scheduler

	def get_streaming_checkpointer(self):
		"""
		Returns the checkpoint manager, responsible for saving model checkpoints.

		Returns:
		    CheckpointManager: The checkpoint manager.
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

	def log_metrics(
		self,
		metrics: MetricsType,
		step: int,
		log_as: tp.Optional[tp.Literal["summary", "config"]] = None,
	):
		"""
		Logs training metrics to Weights & Biases and/or TensorBoard.

		Args:
		  metrics (tp.Dict[str, tp.Union[float, tp.List, tp.Tuple, np.ndarray, 'jnp.ndarray', 'torch.Tensor']]):
		    A dictionary where keys are metric names and values are metric values.
		  step (int): The current training step or iteration.
		"""

		if self.report_metrics:
			filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
			metrics = {
				self._restructure_metric_name(k): get_safe_arr(v)
				for k, v in filtered_metrics.items()
			}
			self._log_to_wandb(metrics, step, log_as)
			self._log_to_tensorboard(metrics, step, log_as)

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

	def log_weight_distribution(self, state, step: int):
		if self.weight_distribution_log_steps > 0 and (
			(step % self.weight_distribution_log_steps) == 0
		):
			stats = compute_weight_stats(state.graphstate, self.weight_distribution_pattern)
			metrics = {}
			for key, value in stats.items():
				if key.endswith("/values"):
					path: str = key[:-7]
					path = path.replace("/", ".")
					metrics[f"weights/histogram/{path}"] = np.array(jax.device_get(value))
				else:
					key = key.replace("/", ".")
					metrics[f"weights/information/{key}"] = float(value)
			self.log_metrics(metrics, step)

	def _log_to_wandb(
		self,
		metrics,
		step,
		log_as: tp.Optional[tp.Literal["summary", "config"]] = None,
	):
		"""
		Log metrics to Weights & Biases (wandb).

		This method processes the given metrics and logs them to wandb if it's enabled and properly initialized.

		Args:
		  metrics (dict): A dictionary of metrics to log. Keys are metric names, values are the metric values.
		  step (int): The current step or iteration number.
		"""

		if self.use_wandb and wandb is not None:
			if log_as == "summary":
				wandb.summary.update(metrics)
			elif log_as == "config":
				wandb.config.update(metrics)
			else:
				wandb_metrics = {}
				for key, value in metrics.items():
					try:
						wandb_metrics[key] = (
							self._create_wandb_histogram(value)
							if isinstance(value, (list, tuple, np.generic, jax.Array))
							else value
						)
					except Exception as e:
						warnings.warn(f"Failed to log metric {key} to wandb: {e}", stacklevel=3)
				try:
					wandb.log(wandb_metrics, step=step)
				except Exception:
					...

	def _log_to_tensorboard(
		self,
		metrics,
		step,
		log_as: tp.Optional[tp.Literal["summary", "config"]] = None,
	):
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
			if isinstance(value, jax.Array):
				value = np.array(jax.device_get(value))
			if value.dtype in [np.bfloat16]:
				value = value.astype(np.float32)
			value = value.astype(np.float16)
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

	def __repr__(self):
		cls_name = self.__class__.__name__
		field_lines = [
			f"    {f.name}: {getattr(self, f.name)!r}".replace("\n", "\n    ")
			for f in fields(self)
		]
		return f"{cls_name}(\n" + "\n".join(field_lines) + "\n)"

	__str__ = __repr__

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

	__hash__ = hash_fn
