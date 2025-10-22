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

        Note:
            Must be called during trainer initialization to set up:
            - Timer and wandb runtime
            - Dataloaders and max steps
            - Optimizer and scheduler
            - Flops tracking metrics
            - Checkpoint manager
            - Model state and sharding
            - Compilation trackers
        """
        ...

    @abstractmethod
    def _initialize_memory_tracking(self):
        """
        Initialize memory tracking for GPU/TPU memory monitoring.

        This method sets up memory monitoring to track memory usage
        during training when track_memory is enabled in arguments.

        Note:
            Only initialized when performance_mode is False.
            Uses SMPMemoryMonitor with configurable sampling interval.
        """
        ...

    @abstractmethod
    def initialize_trainer_utils(self):
        """
        Initialize all trainer utilities in the correct order.

        This orchestration method sets up all trainer components:
        1. Weights & Biases logging (if enabled)
        2. Training timer for performance monitoring
        3. Dataloaders for training and evaluation
        4. Model, optimizer, and learning rate scheduler
        5. Model state sharding across devices
        6. Compiled training and evaluation functions

        Note:
            The initialization order is critical as later steps
            depend on earlier ones being completed.
        """
        ...

    @abstractmethod
    def _initialize_wandb(self):
        """
        Initialize Weights & Biases logging integration.

        Sets up W&B runtime for experiment tracking and metrics
        logging when use_wandb is enabled in training arguments.

        Note:
            Only initialized if arguments.use_wandb is True.
            Uses arguments.get_wandb_init() for configuration.
        """
        ...

    @abstractmethod
    def _initialize_timer(self):
        """
        Initialize the timer for performance monitoring.

        Sets up a Timers instance for tracking execution time
        of various training operations with optional TensorBoard
        integration for visualization.

        Note:
            TensorBoard writer is obtained from arguments.get_tensorboard().
        """
        ...

    @abstractmethod
    def _configure_dataloaders(self):
        """
        Configure dataloaders for training and evaluation.

        Sets up data loading pipelines using either Grain or TensorFlow
        datasets based on configuration. Handles:
        - Dataset offloading to specified devices if enabled
        - Calculation of maximum training and evaluation steps
        - Proper batch size and epoch configuration

        Note:
            Results are stored as dataloader_train, dataloader_eval,
            max_training_steps, and max_evaluation_steps attributes.
        """
        ...

    @abstractmethod
    def _configure_model(self):
        """
        Configure model, optimizer, scheduler, and configuration.

        Retrieves and sets up:
        - Model instance
        - Gradient transformation (optimizer)
        - Learning rate scheduler
        - Model configuration

        Note:
            Results are stored as _model, tx, scheduler, and config attributes.
            Time taken for configuration is logged for performance monitoring.
        """
        ...

    @abstractmethod
    def _configure_functions(self):
        """
        Configure and JIT-compile training and evaluation step functions.

        Sets up:
        - JIT-compiled training step function
        - JIT-compiled evaluation step function (if applicable)
        - Device mesh for distributed computation
        - Checkpoint manager for saving/loading

        Note:
            Functions are compiled with appropriate static arguments
            for optimal performance. Results stored as instance attributes.
        """
        ...

    @abstractmethod
    def _configure_state(self):
        """
        Configure and shard model state across devices.

        Handles:
        - Optimizer state initialization or reinitialization after checkpoint resumption
        - Creation of sharding specifications based on partition rules
        - Distribution of the model state across the device mesh

        Note:
            Ensures proper optimizer state initialization whether starting fresh
            or resuming from checkpoint. Uses model's partition rules for sharding.
        """
        ...

    @abstractmethod
    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Create a Grain data collection function for batch processing.

        Args:
            max_sequence_length: Maximum allowed sequence length for padding/truncation.
            truncation_mode: How to truncate sequences exceeding max length:
                - "keep_end": Keep the end of the sequence
                - "keep_start": Keep the beginning of the sequence

        Returns:
            Callable that processes batches for Grain dataloader.

        Note:
            Function should handle padding, truncation, and data format conversion
            compatible with Grain's data pipeline.
        """
        ...

    @abstractmethod
    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Create a TensorFlow Dataset collection function for batch processing.

        Args:
            max_sequence_length: Maximum allowed sequence length for padding/truncation.
            truncation_mode: How to truncate sequences exceeding max length:
                - "keep_end": Keep the end of the sequence
                - "keep_start": Keep the beginning of the sequence

        Returns:
            Callable that processes batches for TensorFlow datasets.

        Note:
            Function should handle padding, truncation, and data format conversion
            compatible with tf.data API.
        """
        ...

    @abstractmethod
    def create_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Create a generic data collection function for batch processing.

        Args:
            max_sequence_length: Maximum allowed sequence length for padding/truncation.
            truncation_mode: How to truncate sequences exceeding max length:
                - "keep_end": Keep the end of the sequence
                - "keep_start": Keep the beginning of the sequence

        Returns:
            Callable that processes batches of data.

        Note:
            This is a generic version that can be used with any dataloader type.
            Implementations should handle padding, truncation, and format conversion.
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
        Configure dataloaders for training and evaluation.

        Creates training and evaluation dataloaders using the provided
        datasets and data collator. Determines the maximum number of
        training and evaluation steps based on dataset sizes and arguments.

        Returns:
            TrainerConfigureDataloaderOutput containing:
            - dataloader_train: Training data iterator
            - max_training_steps: Total training steps
            - dataloader_eval: Optional evaluation data iterator
            - max_evaluation_steps: Optional total evaluation steps

        Note:
            Automatically selects between Grain and TensorFlow dataloaders
            based on arguments.use_grain setting.
        """
        ...

    @abstractmethod
    def configure_model(self) -> TrainerConfigureModelOutput:
        """
        Configure model, optimizer, scheduler, and configuration.

        Retrieves model configuration from the model state and creates
        the optimizer and scheduler using training arguments.

        Returns:
            TrainerConfigureModelOutput containing:
            - model: The EasyDeL model instance
            - tx: Gradient transformation (optimizer)
            - scheduler: Learning rate schedule
            - config: Optional model configuration

        Note:
            If pruning_module is set, it wraps the optimizer for
            structured pruning support.
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
        """
        Count total number of model parameters.

        Args:
            prm: Model parameters (can be frozen or unfrozen PyTree).

        Returns:
            int: Total number of parameters in the model.

        Note:
            Handles both frozen and unfrozen Flax parameter dictionaries.
        """
        ...

    @abstractmethod
    def apply_training_hooks(self, metrics: LossMetrics) -> LossMetrics:
        """
        Apply training hooks to check for issues and enforce limits.

        Args:
            metrics: Current training metrics including loss.

        Returns:
            LossMetrics: Potentially modified metrics.

        Raises:
            EasyDeLBreakRequest: If NaN loss detected and break_on_nan is True.
            EasyDeLTimerError: If training time limit exceeded.

        Note:
            Checks for NaN losses and training time limits based on
            configuration in training arguments.
        """
        ...

    @abstractmethod
    def _should_run_evaluation(self, current_step):
        """
        Determine if evaluation should be run at current step.

        Args:
            current_step: The current training step number.

        Returns:
            bool: True if evaluation should be run, False otherwise.

        Note:
            Based on evaluation_steps configuration in training arguments.
            Only evaluates if current_step > 0 and divisible by evaluation_steps.
        """
        ...

    @abstractmethod
    def _prepare_training_output(
        self,
        state: EasyDeLState,
        run_exception: Exception | None = None,
    ):
        """
        Prepare final training output after training completion.

        Args:
            state: Final model state after training.
            run_exception: Optional exception that interrupted training.

        Returns:
            TrainerOutput: Contains final state, mesh, and checkpoint path.

        Note:
            Handles different exception types appropriately:
            - KeyboardInterrupt: Saves state and exits gracefully
            - EasyDeLTimerError: Saves state when time limit reached
            - StopIteration: Normal completion
            - Other exceptions: Re-raised as RuntimeError
        """
        ...

    @abstractmethod
    def _handle_training_interruption(
        self,
        state: EasyDeLState,
        exception: Exception,
        shard_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None,
        gather_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None,
    ):
        """
        Handle training interruption gracefully.

        Args:
            state: Current model state at interruption.
            exception: The exception that caused interruption.
            shard_fns: Functions for sharding data.
            gather_fns: Functions for gathering sharded data.

        Returns:
            TrainerOutput or similar containing saved state.

        Note:
            Saves current state before exiting to allow resumption.
            Handles KeyboardInterrupt and EasyDeLTimerError specially.
        """
        ...

    @abstractmethod
    def _setup_initial_metrics(self, state):
        """
        Setup initial metrics logging at training start.

        Args:
            state: Model state for extracting parameter count.

        Note:
            Logs:
            - Model parameter count
            - Device and platform information
            - JAX configuration flags
            - Other system configuration
        """
        ...

    @abstractmethod
    def _get_next_batch(self, data_iter, dataloader):
        """
        Get next batch from iterator, reinitializing if needed.

        Args:
            data_iter: Current data iterator.
            dataloader: The dataloader to reinitialize from if exhausted.

        Returns:
            tuple: (batch, updated_data_iter) where batch is the next data
                  and updated_data_iter is the potentially reinitialized iterator.

        Note:
            Automatically reinitializes iterator when StopIteration occurs.
            Removes IDs specified in arguments.ids_to_pop_from_dataset.
        """
        ...

    @abstractmethod
    def create_progress_bar(
        self,
        total: int,
        desc: str = "",
        disabled: bool = False,
    ) -> BaseProgressBar:
        """
        Create a progress bar of the specified type.

        Args:
            total: Total number of steps for the progress bar.
            desc: Description text to display.
            disabled: Whether to disable the progress bar.

        Returns:
            BaseProgressBar: Progress bar instance of the configured type.

        Note:
            Type is determined by arguments.progress_bar_type:
            - "tqdm": Standard tqdm progress bar
            - "rich": Rich library progress bar with metrics
            - "json": JSON-formatted progress output
            - disabled=True returns NullProgressBar
        """

    @abstractmethod
    def log_weight_distribution(self, state: EasyDeLState, step: int):
        """
        Log weight distribution statistics.

        Args:
            state: Model state containing parameters.
            step: Current training step.

        Note:
            Logs statistics like mean, std, min, max of weights
            for monitoring training stability.
        """

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        pbar: BaseProgressBar,
        step: int,
        mode: str = "train",
    ) -> None:
        """
        Log metrics and update progress bar.

        Args:
            metrics: Dictionary of metric names and values.
            pbar: Progress bar instance to update.
            step: Current step number.
            mode: "train" or "eval" to prefix metrics.

        Note:
            - Updates progress bar every log_steps
            - Logs to wandb/tensorboard every report_steps
            - Filters out internal metrics (mlperf, grad_norm)
        """

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
        """
        Execute the core evaluation loop.

        Args:
            state: Current model state.
            metrics_tracker: Tracker for accumulating metrics.
            step_metrics: Calculator for per-step metrics.
            start_time: Evaluation start timestamp.

        Returns:
            tuple: (final_state, metrics) containing evaluation results.

        Note:
            Runs evaluation on the entire eval dataset without updating parameters.
        """
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
        """
        Handle training for a single epoch.

        Args:
            state: Current model state.
            train_dataset: Training dataset.
            train_iter: Training data iterator.
            metrics_tracker: Metrics accumulator.
            step_metrics: Per-step metrics calculator.
            pbar: Progress bar to update.
            epoch: Current epoch number.

        Returns:
            tuple: (updated_state, updated_train_iter, exception_or_none)

        Note:
            Processes all batches in the epoch, updating parameters
            and tracking metrics at each step.
        """
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
        """
        Handle evaluation for a single epoch.

        Args:
            state: Current model state.
            eval_dataset: Evaluation dataset.
            eval_iter: Evaluation data iterator.
            metrics_tracker: Metrics accumulator.
            step_metrics: Per-step metrics calculator.
            pbar: Progress bar to update.

        Returns:
            tuple: (final_state, updated_eval_iter)

        Note:
            Processes all batches without updating parameters,
            only computing and tracking evaluation metrics.
        """
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
        """
        Configure Grain dataloaders for training and evaluation.

        Returns:
            TrainerConfigureDataloaderOutput containing:
            - Grain DataLoader for training
            - Maximum training steps
            - Optional Grain DataLoader for evaluation
            - Optional maximum evaluation steps

        Note:
            Grain is Google's data loading library optimized for JAX.
            Handles sharding, batching, and preprocessing efficiently.
        """
        ...

    @abstractmethod
    def _configure_tfds_dataloader(self) -> TrainerConfigureDataloaderOutput:
        """
        Configure TensorFlow Dataset dataloaders.

        Returns:
            TrainerConfigureDataloaderOutput containing:
            - TensorFlow Dataset for training
            - Maximum training steps
            - Optional TensorFlow Dataset for evaluation
            - Optional maximum evaluation steps

        Raises:
            ImportError: If tensorflow is not installed.

        Note:
            Uses tf.data API with automatic batching and prefetching.
            Disables GPU devices to prevent TensorFlow from using them.
        """
        ...

    @abstractmethod
    def _execute_eval_step(self, state, batch) -> LossMetrics:
        """
        Execute a single evaluation step.

        Args:
            state: Current model state.
            batch: Input batch data.

        Returns:
            LossMetrics: Evaluation metrics for this batch.

        Note:
            Should not update model parameters, only compute metrics.
        """
        ...

    @abstractmethod
    def _execute_train_step(self, state, batch) -> tuple[EasyDeLState, LossMetrics, Exception | None]:
        """
        Execute a single training step.

        Args:
            state: Current model state.
            batch: Input batch data.

        Returns:
            tuple containing:
            - Updated model state with new parameters
            - Training metrics for this batch
            - Exception if any occurred, None otherwise

        Note:
            Computes gradients and updates model parameters.
        """
        ...

    @abstractmethod
    def _finalize_training(self, output, run_exception):
        """
        Finalize training and prepare output.

        Args:
            output: Training output to finalize.
            run_exception: Exception that occurred during training, if any.

        Returns:
            Final training output with any necessary cleanup.

        Note:
            Performs cleanup tasks like closing progress bars,
            finishing wandb runs, and preparing final checkpoint.
        """
        ...

    @abstractmethod
    def train(self) -> tp.Any:
        """
        Execute the complete training process.

        This is the main entry point for training. It orchestrates the entire
        training workflow including initialization, training loops, evaluation,
        checkpointing, and finalization.

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
    def calculate_number_total_flops(self, params, is_training=True):
        """Calculate total FLOPs for the model."""
        ...
