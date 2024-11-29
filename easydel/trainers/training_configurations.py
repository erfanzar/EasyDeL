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
import pathlib
import re
import warnings
from dataclasses import dataclass, field
from typing import (
	Any,
	Dict,
	List,
	Literal,
	Optional,
	Tuple,
	Union,
)

import jax
import jax.numpy as jnp
import numpy as np
from fjformer.lora import LoraRapture, RaptureConfig
from jax.sharding import PartitionSpec
from jax.tree_util import PyTreeDef

from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import (
	AVAILABLE_GRADIENT_CHECKPOINTS,
	AVAILABLE_OPTIMIZERS,
	AVAILABLE_PRUNING_TYPE,
	AVAILABLE_SCHEDULERS,
	AVAILABLE_SPARSE_MODULE_TYPES,
	EasyDeLGradientCheckPointers,
	EasyDeLOptimizers,
	EasyDeLSchedulers,
	get_logger,
)
from easydel.trainers.utils import JaxDistributedConfig

try:
	import wandb  # type: ignore # noqa: F821
except ImportError:
	wandb = None


import flax.metrics.tensorboard

logger = get_logger(__name__)


class LoraRaptureConfig(RaptureConfig):  # Don't Make user involved with FJFormer
	"""
	Configuration class for EasyDeL specific XRapTure settings.
	Inherits from FJFormer's RaptureConfig.

	Attributes:
	    parameters (PyTreeDef | dict): Model parameters for XRapTure.
	"""

	def __init__(self, parameters: PyTreeDef | dict, **kwargs):
		self.parameters = parameters
		super().__init__(**kwargs)


# Constants
AVAILABLE_BACKENDS: List[str] = ["cpu", "gpu", "tpu", None]


@dataclass
class TrainingArguments:
	"""
	Data class containing all the arguments for training a EasyDeL model.

	Attributes:
	    model_name (Optional[str]): Name of the model.
	    num_train_epochs (Optional[int]): Number of training epochs.
	    model_class (Optional[EasyDeLBaseModule]): The EasyDeL Flax Pretrained Model class to use.
	    total_batch_size (int): Total batch size for training.
	    eval_batch_size (int): Total batch size for evaluation of model.
	    max_training_steps (Optional[int]): Maximum number of training steps.
	    max_evaluation_steps (Optional[int]):  Maximum number of evaluation steps.
	    optimizer (AVAILABLE_OPTIMIZERS): Optimizer to use for training.
	    scheduler (AVAILABLE_SCHEDULERS): Learning rate scheduler to use.
	    learning_rate (float): The initial learning rate.
	    learning_rate_end (Optional[float]): The final learning rate for schedulers that support it.
	    gradient_accumulation_steps (int): Number of steps for gradient accumulation.
	    weight_decay (float): Weight decay applied to parameters.
	    label_smoothing_factor (float): Label smoothing factor.
	    z_loss (float): Coefficient for the z-loss term.
	    gradient_checkpointing (AVAILABLE_GRADIENT_CHECKPOINTS): Gradient checkpointing strategy to use.
	    clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.
	    max_sequence_length (Optional[int]): Maximum sequence length for model input.
	    sharding_array (Union[tuple, int]): Sharding configuration for model parameters.
	    is_fine_tuning (bool): Whether the model is being fine-tuned.
	    do_train (bool): Whether to run training.
	    do_eval  (bool): Whether to run evaluation.
	    train_on_inputs (bool): Whether to train on model inputs (as opposed to labels).
	    backend (Optional[str]): Backend to use for JAX computation.
	    extra_optimizer_kwargs (dict):  Extra keyword arguments passed to the optimizer.
	    save_steps (Optional[int]): Save checkpoints every specified number of steps.
	    save_dir (str): Directory to save checkpoints.
	    save_total_limit (Optional[int]): Total number of checkpoints to keep.
	    dtype (jnp.dtype): Data type for model computations.
	    param_dtype (jnp.dtype): Data type for model parameters.
	    fully_sharded_data_parallel (bool): Whether to use fully sharded data parallelism.
	    use_wandb (bool): Whether to use Weights & Biases for logging.
	    custom_rule (Optional[Dict[str, PartitionSpec]]):  Custom sharding rules for specific parameters.
	    extra_configs (Optional[dict]): Extra configurations passed as a dictionary.
	    ids_to_pop_from_dataset (Optional[list]): List of IDs to remove from the dataset.
	    remove_ckpt_after_load (bool): Remove checkpoint files after loading the model.
	    configs_to_initialize_model_class (Optional[dict]): Configurations used to initialize the model class.
	    do_last_save (bool): Whether to save the final model checkpoint.
	    model_parameters (Optional[dict]): Model parameters for initialization.
	    track_memory (Optional[bool]): Whether to track memory usage during training.
	    loss_re_mat (str): Regular expression for loss rematerialization.
	    loss_chunk (int): Chunk size for loss computation.
	    truncation_mode (Literal["keep_end", "keep_start"]): Truncation mode for handling long sequences.
	    warmup_steps (int): Number of warm-up steps for the scheduler.
	    init_input_shape (Tuple[int, int]): Initial input shape for model initialization.
	    step_partition_spec (PartitionSpec): Partition specification for stepping the optimizer.
	    training_time (Optional[str]): Maximum training time in the format "50min" or "23h".
	    dataloader_num_workers (Optional[int]): Number of workers for the dataloader.
	    dataloader_pin_memory (Optional[bool]): Pin memory for the dataloader.
	    jax_distributed_config (Optional[dict]): Configuration for JAX distributed training.
	    log_all_workers (bool): Log metrics from all workers (used with WandB).
	    wandb_entity (Optional[str]): WandB entity to log metrics to.
	    save_optimizer_state (bool): Save optimizer state in checkpoints.
	    step_start_point (Optional[int]): Starting step for training (resuming from checkpoint).
	    verbose (bool): Print verbose logs during training.
	    offload_device (jax.Device): Device to offload computations to.
	    rapture_config (Optional[LoraRaptureConfig]): Configuration for XRapTure (LoRA).
			pruning_module (Optional[AVAILABLE_PRUNING_TYPE]): Configuration Pruning Module.
			sparse_module_type (AVAILABLE_SPARSE_MODULE_TYPES): sparse model type to be used to prune the params.
			sparsify_module (bool): whenever to use sparse apply method for faster and better training.
	    merge_lora_rapture_parameters (bool): Merge LoRA parameters with model parameters.
	    state_apply_fn_kwarguments_to_model (Optional[dict]): Keyword arguments for the model's apply function.
	    remove_unused_columns (bool): Remove unused columns from the dataset.
	    force_batch_and_gradient_accumulation_steps_calculation (bool): Force calculation of batch and gradient accumulation steps.
	    performance_mode (bool): Enable performance mode for faster training.
	    neftune_noise_alpha (Optional[float]): NEFTune noise alpha parameter.
	    log_grad_norms (bool): Log gradient norms during training.
	    loaded_model_config_kwargs (Optional[dict]): Keyword arguments from loaded model configuration.
			num_classification_labels (Optional[int]): num classification labels for SequenceClassification Trainer
			classification_problem_type (Literal["regression", "single_label_classification", "multi_label_classification"]): num classification labels for SequenceClassification Trainer

	Methods:
	    __post_init__(): Validates configuration and sets up distributed training, optimizer, logging, and XRapTure.
	    _validate_config(): Validates the configuration settings.
	    _setup_distributed(): Sets up JAX distributed training.
	    _setup_optimizer(): Configures the optimizer and scheduler.
	    _setup_logging(): Sets up logging (TensorBoard, WandB).
	    _setup_rapture(): Sets up XRapTure if enabled.
	    _ensure_variables(): Checks and sets up variables for start.
	    _time_to_seconds(time_str: str) -> int:
	        Converts a time string ("50min", "23h") to seconds.
	    get_path(self) -> pathlib.Path:
	        Returns the path to the checkpoint directory.
	    ensure_checkpoint_path(self):
	        Creates the checkpoint directory if it doesn't exist.
	    get_mesh(self):
	        Returns the JAX device mesh for distributed training.
	    get_mesh_names():
	        Returns the names of the mesh dimensions.
	    get_optimizer_and_scheduler(self, steps: Optional[int] = None):
	        Returns the configured optimizer and scheduler.
	    get_streaming_checkpointer(self):
	        Returns the checkpoint manager for saving checkpoints.
	    get_tensorboard(self):
	        Returns the TensorBoard SummaryWriter for logging.
	    get_wandb_init(self):
	        Initializes WandB if enabled.
	    log_metrics(self, metrics: Dict[str, Union[float, List, Tuple, np.ndarray, 'jnp.ndarray', 'torch.Tensor']], step: int):
	        Logs metrics to TensorBoard and/or WandB.
	    to_dict(self) -> Dict[str, Any]:
	        Returns the configuration as a dictionary.
	    from_dict(cls, config: Dict[str, Any]) -> 'TrainingArguments':
	        Creates a TrainingArguments instance from a dictionary.
	    __str__(self): Returns a string representation of the configuration.
	"""

	model_name: str = "Model"
	num_train_epochs: int = 10
	model_class: Optional[EasyDeLBaseModule] = None
	total_batch_size: int = 32
	eval_batch_size: int = 64
	max_training_steps: Optional[int] = None
	max_evaluation_steps: Optional[int] = None
	optimizer: AVAILABLE_OPTIMIZERS = EasyDeLOptimizers.ADAMW
	scheduler: AVAILABLE_SCHEDULERS = EasyDeLSchedulers.NONE
	learning_rate: float = 5e-5
	learning_rate_end: Optional[float] = None
	gradient_accumulation_steps: int = 1
	clip_grad: Optional[float] = None
	weight_decay: float = 0.01
	label_smoothing_factor: float = 0.0
	z_loss: float = 0.0
	gradient_checkpointing: AVAILABLE_GRADIENT_CHECKPOINTS = (
		EasyDeLGradientCheckPointers.NONE
	)
	max_sequence_length: Optional[int] = 4096
	sharding_array: Union[tuple, int] = (1, -1, 1, 1)
	is_fine_tuning: bool = True
	do_train: bool = True
	do_eval: bool = False
	train_on_inputs: bool = True
	backend: Optional[str] = None
	extra_optimizer_kwargs: dict = field(default_factory=dict)
	save_steps: Optional[int] = None
	save_dir: str = "EasyDeL-Checkpoints"
	save_total_limit: Optional[int] = None
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	fully_sharded_data_parallel: bool = True
	use_wandb: bool = True
	custom_rule: Optional[Dict[str, PartitionSpec]] = None
	extra_configs: Optional[dict] = None
	ids_to_pop_from_dataset: Optional[list] = field(default_factory=list)
	remove_ckpt_after_load: bool = False
	configs_to_initialize_model_class: Optional[dict] = None
	do_last_save: bool = True
	model_parameters: Optional[dict] = None
	track_memory: Optional[bool] = None
	loss_re_mat: str = ""
	loss_chunk: int = 1024
	truncation_mode: Literal["keep_end", "keep_start"] = "keep_end"
	warmup_steps: int = 500
	init_input_shape: Tuple[int, int] = (1, 1)
	step_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp")
	training_time: Optional[str] = None
	dataloader_num_workers: Optional[int] = 0
	dataloader_pin_memory: Optional[bool] = False
	jax_distributed_config: Optional[dict] = None
	log_all_workers: bool = False
	wandb_entity: Optional[str] = None
	save_optimizer_state: bool = False
	step_start_point: Optional[int] = None
	verbose: bool = True
	offload_device: jax.Device = jax.devices("cpu")[0]
	rapture_config: Optional[LoraRaptureConfig] = None
	pruning_module: AVAILABLE_PRUNING_TYPE = None
	sparsify_module: bool = False
	sparse_module_type: AVAILABLE_SPARSE_MODULE_TYPES = "bcoo"
	merge_lora_rapture_parameters: bool = True
	state_apply_fn_kwarguments_to_model: Optional[dict] = None
	remove_unused_columns: bool = True
	force_batch_and_gradient_accumulation_steps_calculation: bool = False
	performance_mode: bool = False
	neftune_noise_alpha: Optional[float] = None
	log_grad_norms: bool = True
	loaded_model_config_kwargs: Optional[dict] = None
	num_classification_labels: Optional[int] = None
	classification_problem_type: Literal[
		"regression", "single_label_classification", "multi_label_classification"
	] = "regression"

	def __post_init__(self):
		"""
		Validates the configuration, sets up distributed training,
		initializes the optimizer, configures logging, and sets up XRapTure.
		This method is automatically called after the object is initialized.
		"""
		self._validate_config()
		self._setup_distributed()
		self._setup_optimizer()
		self._setup_logging()
		self._setup_rapture()
		self._ensure_variables()
		if self.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			warnings.warn(
				"Passing `gradient_checkpointing` in training arguments is deprecated, "
				"please pass `gradient_checkpointing` to model config_kwargs while loading or creating model",
				stacklevel=1,
			)

	def _validate_config(self):
		"""
		Performs validation checks on the provided configuration settings.
		Raises ValueError if any configuration is invalid.
		"""
		if self.backend not in AVAILABLE_BACKENDS:
			raise ValueError(
				f"Backend {self.backend} is not recognized. Available backends: {AVAILABLE_BACKENDS}"
			)

		if self.neftune_noise_alpha is not None and self.neftune_noise_alpha <= 0:
			raise ValueError("NEFTune noise alpha must be positive")

	def _setup_distributed(self):
		"""
		Sets up JAX distributed training based on the chosen backend and sharding configuration.
		Determines the number of available devices and sets up the device mesh.
		"""
		self.available_backends = len(jax.devices(self.backend))
		self.array_devices_shape = (
			jnp.ones((self.available_backends, 1)).reshape(self.sharding_array).shape
		)
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

	def _setup_rapture(self):
		"""
		Sets up XRapTure (LoRA) if enabled in the configuration.
		Initializes the XRapTure instance with the provided configuration.
		"""
		if self.rapture_config is not None:
			if self.log_grad_norms:
				warnings.warn("Gradient norm logging disabled when using LoRA", stacklevel=1)
				self.log_grad_norms = False
			self.rapture = LoraRapture(config=self.rapture_config)
		else:
			self.rapture = None

	def _ensure_variables(self):
		"""
		Checks and sets up variables for start.
		"""
		self.step_start_point = self.step_start_point or 0

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

	def get_path(self) -> pathlib.Path:
		"""
		Returns the path to the checkpoint directory.

		Returns:
		    pathlib.Path: The path to the checkpoint directory.
		"""
		return pathlib.Path(self.save_dir, self.model_name)

	def ensure_checkpoint_path(self):
		"""
		Creates the checkpoint directory if it doesn't exist.
		"""
		path = self.get_path()
		path.mkdir(parents=True, exist_ok=True)

	def get_mesh(self):
		"""
		Returns the JAX device mesh, used for distributed training.

		Returns:
		    jax.sharding.Mesh: The JAX device mesh.
		"""
		from jax.experimental.mesh_utils import create_device_mesh
		from jax.sharding import Mesh

		return Mesh(create_device_mesh(self.array_devices_shape), self.get_mesh_names())

	@staticmethod
	def get_mesh_names():
		"""
		Returns the names of the mesh dimensions.

		Returns:
		    tuple: A tuple containing the names of the mesh dimensions.
		"""
		return "dp", "fsdp", "tp", "sp"

	def get_optimizer_and_scheduler(self, steps: Optional[int] = None):
		"""
		Returns the configured optimizer and learning rate scheduler.

		Args:
		    steps (Optional[int]): The number of training steps.
		        If not provided, uses the value from `self.optimizer_kwargs`.

		Returns:
		    tuple: A tuple containing the optimizer and scheduler.
		"""
		from easydel.etils.auto_tx import get_optimizer_and_scheduler

		self.optimizer_kwargs["steps"] = steps or self.optimizer_kwargs["steps"]
		return get_optimizer_and_scheduler(**self.optimizer_kwargs)

	def get_streaming_checkpointer(self):
		"""
		Returns the checkpoint manager, responsible for saving model checkpoints.

		Returns:
		    fjformer.CheckpointManager: The checkpoint manager.
		"""
		import os.path

		from fjformer.checkpoint import CheckpointManager

		return CheckpointManager(
			os.path.join(self.save_dir, self.model_name),
			save_optimizer_state=self.save_optimizer_state,
			verbose=self.verbose,
		)

	@functools.cached_property
	def _tensorboard(self):
		return flax.metrics.tensorboard.SummaryWriter(log_dir=str(self.get_path()))

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
		    Optional[wandb.sdk.wandb_run.Run]: The WandB run object if initialized, else None.
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
			tags=["EasyDeL", "FJFormer", "OST-OpenSourceTransformers", "Jax/Flax"],
			entity=self.wandb_entity,
		)

	def ensure_training_time(self, time_passed):
		if self.training_time is not None and time_passed > self._time_to_seconds(
			self.training_time
		):
			raise EasyDeLTimerError("Time Out")

	def log_metrics(
		self,
		metrics: Dict[
			str, Union[float, List, Tuple, np.ndarray, "jnp.ndarray", "torch.Tensor"]  # type: ignore # noqa: F821
		],
		step: int,
	):
		"""
		Logs training metrics to Weights & Biases and/or TensorBoard.

		Args:
		    metrics (Dict[str, Union[float, List, Tuple, np.ndarray, 'jnp.ndarray', 'torch.Tensor']]):
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
		    - Any exceptions during logging are caught and warned about, allowing the process to continue.
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
				logger.warn(f"Failed to log metric {key} to TensorBoard: {e}")

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
		    - Any exceptions during histogram creation are caught and logged, returning None in such cases.
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

	def to_dict(self) -> Dict[str, Any]:
		"""
		Converts the TrainingArguments object into a dictionary.

		Returns:
		    Dict[str, Any]: A dictionary representation of the TrainingArguments.
		"""
		return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

	@classmethod
	def from_dict(cls, config: Dict[str, Any]) -> "TrainingArguments":
		"""
		Creates a TrainingArguments instance from a dictionary.

		Args:
		    config (Dict[str, Any]): The configuration dictionary.

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
	def _pretty_print(d: Dict[str, Any], indent: int = 0) -> str:
		"""
		Helper function for pretty-printing a dictionary.

		Args:
		    d (Dict[str, Any]): The dictionary to pretty-print.
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


class EasyDeLBaseModule:
	pass
