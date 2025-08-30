# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
import typing as tp
from abc import ABCMeta, abstractmethod

import flax
import flax.core
import jax
import numpy as np
import optax
from eformer.paths import ePathLike
from eformer.pytree import auto_pytree
from eformer.serialization import AsyncCheckpointManager
from jax.sharding import Mesh
from optax import GradientTransformation, Schedule

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.utils import CompilationTracker

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None


from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseConfig, EasyDeLBaseModule
from easydel.utils import Timers

from .metrics import BaseProgressBar, MetricsTracker, StepMetrics
from .training_configurations import MetricsType, TrainingArguments

if tp.TYPE_CHECKING:
    pass

if tp.TYPE_CHECKING:
    from datasets import Dataset
    from jax._src.pjit import JitWrapped
else:
    JitWrapped = tp.Any
    Dataset = tp.Any

logger = get_logger(__name__)


@auto_pytree
class TrainerConfigureDataloaderOutput:
    """
    Output configuration for dataloader setup.

    Contains the configured dataloaders and computed maximum steps for
    training and evaluation phases.

    Attributes:
        dataloader_train: Iterator over training batches
        max_training_steps: Total number of training steps
        dataloader_eval: Optional iterator over evaluation batches
        max_evaluation_steps: Optional total number of evaluation steps
    """

    dataloader_train: tp.Iterator[np.ndarray]
    max_training_steps: int
    dataloader_eval: tp.Iterator[np.ndarray] | None = None
    max_evaluation_steps: int | None = None


@auto_pytree
class TrainerConfigureModelOutput:
    """
    Output configuration for model setup.

    Contains the configured model, optimizer, scheduler, and model configuration.

    Attributes:
        model: The initialized EasyDeL model
        tx: Gradient transformation (optimizer) for training
        scheduler: Learning rate schedule function
        config: Optional model configuration object
    """

    model: EasyDeLBaseModule
    tx: GradientTransformation
    scheduler: Schedule
    config: EasyDeLBaseConfig | None = None


@auto_pytree
class TrainerConfigureFunctionOutput:
    """
    Output configuration for training and evaluation functions.

    Contains the compiled step functions and supporting infrastructure.

    Attributes:
        sharded_training_step_function: JIT-compiled training step function
        mesh: Device mesh for distributed computation
        checkpoint_manager: Manager for saving/loading checkpoints
        sharded_evaluation_step_function: Optional JIT-compiled evaluation function
    """

    sharded_training_step_function: JitWrapped
    mesh: Mesh
    checkpoint_manager: AsyncCheckpointManager
    sharded_evaluation_step_function: JitWrapped | None = None


@auto_pytree
class TrainerOutput:
    """
    Final output from the training process.

    Contains the final model state and checkpoint information.

    Attributes:
        state: Final model state after training
        mesh: Device mesh used during training
        last_save_file_name: Name of the last saved checkpoint file
        checkpoint_path: Full path to the last checkpoint
    """

    state: EasyDeLState
    mesh: jax.sharding.Mesh | None
    last_save_file_name: str | None = None
    checkpoint_path: str | None = None


class BaseTrainerProtocol(metaclass=ABCMeta):
    """
    Abstract base protocol defining the interface for all trainer implementations.

    This protocol ensures that all trainer implementations provide the necessary
    methods and attributes for training and evaluation workflows. It defines the
    contract that concrete trainer classes must fulfill.

    The protocol covers:
    - Initialization and configuration methods
    - Training and evaluation loops
    - Checkpoint management
    - Metrics logging and monitoring
    - Hook methods for customization

    All methods marked with @abstractmethod must be implemented by subclasses.
    """

    # Required attributes for all trainers
    arguments: TrainingArguments
    dataset_train: Dataset | None
    dataset_eval: Dataset | None
    data_collator: tp.Callable | None
    finetune: bool
    checkpoint_path: str | os.PathLike | None
    dtype: tp.Any  # jax.numpy.dtype
    param_dtype: tp.Any  # jax.numpy.dtype

    timer: Timers
    wandb_runtime: tp.Any  # wandb runtime
    dataloader_train: tp.Iterator[np.ndarray]
    dataloader_eval: tp.Iterator[np.ndarray] | None
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

    _forward_flops_per_token: int | float
    _backward_flops_per_token: int | float

    _extra_forward_flops_per_token: int | float
    _extra_backward_flops_per_token: int | float

    _train_shared_fn_static_args_: dict[str, tp.Any]
    _train_shared_fn_extra_args_: tuple[tp.Any]

    _eval_shared_fn_static_args_: dict[str, tp.Any]
    _eval_shared_fn_extra_args_: tuple[tp.Any]

    _resumed_from_checkpoint: bool
    state_shardings: tp.Any
    _training_time_start: float | None
    _evaluation_time_start: float | None

    @abstractmethod
    def __init__(
        self,
        arguments: TrainingArguments | None = None,
        model_state: EasyDeLState | None = None,
        model: tp.type[EasyDeLBaseModule] | None = None,
        dataset_train: Dataset | None = None,
        dataset_eval: Dataset | None = None,
        data_collator: tp.Callable | None = None,
        finetune: bool = True,
        **deprecated_kwargs,
    ):
        """
        Initialize the trainer with model and training configuration.

        Args:
            arguments: Training configuration and hyperparameters
            model_state: Pre-initialized model state (exclusive with model)
            model: Model class to initialize (exclusive with model_state)
            dataset_train: Training dataset
            dataset_eval: Evaluation dataset
            data_collator: Function to collate batch data
            finetune: Whether this is a fine-tuning run
            **deprecated_kwargs: Backward compatibility arguments
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

    @property
    @abstractmethod
    def is_process_zero(self): ...

    @property
    @abstractmethod
    def is_enable(self): ...

    @abstractmethod
    def _initialize_attributes(self):
        """
        Initialize all trainer attributes with default values.

        This method ensures all required attributes are properly initialized,
        preventing AttributeError during training. It sets up timers, trackers,
        compilation managers, and other essential components.
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
    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Creates a function to collect and process batches of data for training or evaluation.
        """
        ...

    @abstractmethod
    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Creates a function to collect and process batches of data for training or evaluation.
        """
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
        Configure and JIT-compile training and evaluation step functions.

        This method prepares the computational graph for training and evaluation,
        including setting up sharding specifications, compiling functions with
        appropriate static arguments, and initializing the checkpoint manager.

        Returns:
            TrainerConfigureFunctionOutput: Compiled functions and infrastructure
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
    def save_information(self, output_path: str | ePathLike) -> None:
        """
        Save the generated information to a markdown file.
        """
        ...

    @abstractmethod
    def save_pretrained(
        self,
        state: EasyDeLState,
        save_directory: str | None = None,
        gather_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None = None,
        to_torch: bool = False,
        easystate_to_huggingface_model_kwargs: dict | None = None,
        torch_save_pretrained_kwargs: dict | None = None,
    ):
        """
        Saves the model state as a checkpoint file or to a Torch compatible directory.
        """
        ...

    @abstractmethod
    def _save_to_torch(
        self,
        state: EasyDeLState,
        save_directory: str | os.PathLike,
        easystate_to_huggingface_model_kwargs: dict | None = None,
        torch_save_pretrained_kwargs: dict | None = None,
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
        run_exception: Exception | None = None,
    ):
        """Prepare training output after training loop completion."""
        ...

    @abstractmethod
    def _handle_training_interruption(
        self,
        state: EasyDeLState,
        exception: Exception,
        shard_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None,
        gather_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None,
    ):
        """Handle training interruption gracefully."""
        ...

    @abstractmethod
    def _setup_initial_metrics(self, state):
        """Setup initial metrics logging."""
        ...

    @abstractmethod
    def _get_next_batch(self, data_iter, dataloader):
        """Get next batch from iterator, reinitializing if needed."""
        ...

    @abstractmethod
    def create_progress_bar(
        self,
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
        metrics: dict[str, float],
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
        shard_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None,
        gather_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None,
    ):
        """
        Execute the core training loop.

        This method implements the main training iteration, processing batches,
        computing gradients, updating parameters, and tracking metrics across
        multiple epochs.

        Args:
            state: Initial model state
            metrics_tracker: Tracker for accumulating metrics
            step_metrics: Calculator for per-step metrics
            start_time: Training start timestamp
            shard_fns: Functions for sharding data
            gather_fns: Functions for gathering sharded data

        Returns:
            Tuple of final output and any exception encountered
        """
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
        train_dataset,
        train_iter,
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
        eval_dataset,
        eval_iter,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
        pbar: BaseProgressBar,
    ):
        """Handles training for a single epoch."""
        ...

    @property
    @abstractmethod
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any]: ...
    @property
    @abstractmethod
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any]: ...
    @property
    @abstractmethod
    def _train_shared_fn_static_args(self) -> dict[str, tp.Any]: ...
    @property
    @abstractmethod
    def _eval_shared_fn_static_args(self) -> dict[str, tp.Any]: ...

    @abstractmethod
    def _configure_grain_dataloader(self) -> TrainerConfigureDataloaderOutput:
        """Configure Grain dataloader."""
        ...

    @abstractmethod
    def _configure_tfds_dataloader(self) -> TrainerConfigureDataloaderOutput:
        """Configure TensorFlow dataloader."""
        ...

    @abstractmethod
    def _execute_eval_step(self, state, batch) -> LossMetrics:
        """Execute a single eval step."""
        ...

    @abstractmethod
    def _execute_train_step(self, state, batch) -> tuple[EasyDeLState, LossMetrics, Exception]:
        """Execute a single train step."""
        ...

    @abstractmethod
    def _finalize_training(self, output, run_exception):
        """Finalize training and prepare output."""
        ...

    @abstractmethod
    def train(
        self,
        model_parameters: flax.core.FrozenDict | None = None,
        state: EasyDeLState | None = None,
    ) -> tp.Any:
        """
        Execute the complete training process.

        This is the main entry point for training. It orchestrates the entire
        training workflow including initialization, training loops, evaluation,
        checkpointing, and finalization.

        Args:
            model_parameters: Optional model parameters to use
            state: Optional model state to use instead of self.model_state

        Returns:
            TrainerOutput or similar object containing final state and metrics
        """
        ...

    @abstractmethod
    def eval(self, model_state: EasyDeLState) -> tp.Iterator[dict]:
        """
        Evaluate the model on the evaluation dataset.

        This method runs the model in evaluation mode, computing metrics
        without updating parameters. It yields metrics for each evaluation
        step, allowing for streaming evaluation and progress monitoring.

        Args:
            model_state: The model state to evaluate

        Yields:
            dict: Evaluation metrics for each step
        """
        ...

    @abstractmethod
    def start_training_hook(self):
        """Hook to run before training starts."""

    @abstractmethod
    def start_evaluation_hook(self):
        """Hook to run before evaluation starts."""
        ...

    @abstractmethod
    def _setup_static_metrics(self):
        """Setup static metrics for logging."""
        ...

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
    ) -> tuple[EasyDeLState, MetricsType]:
        """hook process to call in start of the step."""
        ...

    @abstractmethod
    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """hook call before passing data to function (called in `_execute` functions)"""

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

    @abstractmethod
    def _try_resume_from_checkpoint(self, current_state: EasyDeLState) -> EasyDeLState | None:
        """
        Try to resume from the latest checkpoint if available.

        Args:
            current_state: The current model state (used as fallback if no checkpoint found)

        Returns:
            The resumed state if a checkpoint was found and loaded, None otherwise
        """
        ...

    @abstractmethod
    def calculate_number_total_flops(self, params, is_training=True):
        """Calculate total FLOPs for the model."""
        ...

    @classmethod
    @abstractmethod
    def load_trainer_state(
        cls,
        load_directory: str | os.PathLike,
        dataset_train: Dataset | None = None,
        dataset_eval: Dataset | None = None,
        data_collator: tp.Callable | None = None,
        device: tp.Any | None = "cpu",
        dtype: tp.Any = None,
        param_dtype: tp.Any = None,
        precision: tp.Any | None = None,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: tp.Any | None = None,
        shard_attention_computation: bool = True,
        shard_fns: tp.Mapping[tuple, tp.Callable] | dict | None = None,
        backend: tp.Any | None = None,
        platform: tp.Any | None = None,
        config_kwargs: tp.Any | None = None,
        model_task: tp.Any = None,
        auto_shard_model: bool = True,
        partition_rules: tuple[tuple[str, tp.Any], ...] | None = None,
        quantization_platform: tp.Any | None = None,
        quantization_method: tp.Any | None = None,
        quantization_block_size: int = 128,
        quantization_pattern: str | None = None,
        quantize_tensors: bool = True,
        verbose: bool = True,
        base_state: type[EasyDeLState] | None = None,
        trainer_init_arguments: dict[str, tp.Any] | None = None,
        **kwargs,
    ):
        """
        Load a trainer state from a saved checkpoint.
        """
        ...
