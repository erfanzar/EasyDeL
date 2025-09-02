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
import pprint
import re
import time
import typing as tp
from abc import abstractmethod
from functools import cached_property

import contextlib2
import flax
import flax.core
import flax.nnx
import grain.python as grain
import jax
import jax.extend
import numpy as np
from eformer.escale import PartitionAxis
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from flax import nnx as nn
from flax.core import unfreeze
from jax import numpy as jnp
from jax._src.stages import Compiled
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel import __version__
from easydel.infra.base_config import EasyDeLBaseConfigDict
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms, EasyDeLQuantizationMethods
from easydel.infra.factory import TaskType
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.utils import CompilationTracker
from easydel.utils import Timers, readme_generator
from easydel.utils.lazy_import import is_package_available
from easydel.utils.traversals import specs_to_name_sharding

from .metrics import BaseProgressBar, JSONProgressBar, NullProgressBar, RichProgressBar, TqdmProgressBar
from .trainer_protocol import (
    BaseTrainerProtocol,
    TrainerConfigureDataloaderOutput,
    TrainerConfigureFunctionOutput,
    TrainerConfigureModelOutput,
    TrainerOutput,
)
from .training_configurations import MetricsType, TrainingArguments
from .utils import CollateMapTransform, HFDataSource, ToNumpy

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


logger = get_logger(__name__)
DEFAULT_ARGS_JSON_NAME = "easydel-training-arguments.json"


class BaseTrainer(BaseTrainerProtocol):
    """
    Base trainer class implementing core training functionality for EasyDeL models.

    This class provides the foundation for training and evaluation workflows, including:
    - Checkpoint management and resumption
    - Dataloader configuration (Grain and TensorFlow datasets)
    - Model state initialization and sharding
    - Training and evaluation step compilation
    - Metrics logging and monitoring
    - Memory tracking and performance optimization

    The trainer handles distributed training across multiple devices using JAX's
    sharding capabilities and supports various optimization strategies.

    Attributes:
        arguments: Training configuration arguments
        model_state: Current state of the model including parameters and optimizer state
        dataset_train: Training dataset
        dataset_eval: Evaluation dataset
        data_collator: Function to collate batch data
        finetune: Whether this is a fine-tuning run
        mesh: Device mesh for distributed computation
        checkpoint_manager: Manager for saving/loading checkpoints
    """

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
        Initialize the BaseTrainer.

        Args:
            arguments: Training configuration and hyperparameters
            model_state: Pre-initialized model state (mutually exclusive with model)
            model: Model class to initialize (mutually exclusive with model_state)
            dataset_train: Training dataset
            dataset_eval: Evaluation dataset
            data_collator: Function to collate batches of data
            finetune: Whether this is a fine-tuning run (affects initialization)
            **deprecated_kwargs: Deprecated keyword arguments for backward compatibility

        Raises:
            ValueError: If both model and model_state are provided, or if neither is provided
            AssertionError: If arguments is None
        """
        assert arguments is not None, "training argument must be passed to Trainers."
        if model_state is not None and model is not None:
            raise ValueError("Either model or model_state should be passed, not both.")
        elif model_state is None and model is None:
            raise ValueError("Either model or model_state should be passed.")
        elif model_state is None:
            model_state = model.to_state()
        if arguments.model_name is None:
            arguments.model_name = getattr(model_state.model, "_model_type", "module")
        self.arguments = arguments

        self._resumed_from_checkpoint = False
        if self.arguments.resume_if_possible:
            try:
                resumed_state = self._try_resume_from_checkpoint(model_state)
            except Exception:
                logger.warning("Resuming from checkpoint failed. Starting fresh training.")
                resumed_state = None
            if resumed_state is not None:
                model_state = resumed_state
                self._resumed_from_checkpoint = True

        self.model_state = model_state
        self._model = flax.nnx.eval_shape(lambda: self.model_state.model)
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval
        self.data_collator = data_collator
        self.finetune = finetune
        self._initialize_attributes()
        self.initialize_trainer_utils()

        if self.arguments.track_memory and self.arguments.track_memory > 0:
            self._initialize_memory_tracking()

    def load_trainer_state(
        cls,
        load_directory: str | os.PathLike,
        dataset_train: Dataset | None = None,
        dataset_eval: Dataset | None = None,
        data_collator: tp.Callable | None = None,
        device: jax.Device | None = "cpu",  # type:ignore
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        shard_attention_computation: bool = True,
        shard_fns: tp.Mapping[tuple, tp.Callable] | dict | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        config_kwargs: EasyDeLBaseConfigDict | None = None,
        model_task: TaskType = TaskType.AUTO_BIND,
        auto_shard_model: bool = True,
        partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
        quantization_platform: EasyDeLPlatforms | None = None,
        quantization_method: EasyDeLQuantizationMethods | None = None,
        quantization_block_size: int = 128,
        quantization_pattern: str | None = None,
        quantize_tensors: bool = True,
        verbose: bool = True,
        base_state: type[EasyDeLState] | None = None,
        trainer_init_arguments: dict[str, tp.Any] | None = None,
        **kwargs,
    ):
        """
        Load a trainer from a saved checkpoint directory.

        This class method reconstructs a trainer instance from a previously saved
        checkpoint, including the model state and training arguments.

        Args:
            load_directory: Path to the checkpoint directory
            dataset_train: Training dataset to use with the loaded trainer
            dataset_eval: Evaluation dataset to use with the loaded trainer
            data_collator: Data collation function
            device: Device to load the model on (default: "cpu")
            dtype: Data type for computations
            param_dtype: Data type for model parameters
            precision: JAX precision level for matrix multiplications
            sharding_axis_dims: Dimensions for sharding axes (dp, fsdp, ep, tp, sp)
            sharding_dcn_axis_dims: DCN-specific sharding dimensions
            sharding_axis_names: Names for the sharding axes
            partition_axis: Custom partition axis configuration
            shard_attention_computation: Whether to shard attention computations
            shard_fns: Custom sharding functions for specific operations
            backend: Computation backend (CPU, GPU, TPU)
            platform: Platform-specific configuration
            config_kwargs: Additional configuration parameters
            model_task: Task type for the model
            auto_shard_model: Whether to automatically shard the model
            partition_rules: Rules for partitioning model parameters
            quantization_platform: Platform for quantization
            quantization_method: Method for quantization
            quantization_block_size: Block size for quantization
            quantization_pattern: Pattern for selective quantization
            quantize_tensors: Whether to quantize tensors
            verbose: Whether to print loading progress
            base_state: Base state class to use for loading
            trainer_init_arguments: Additional arguments for trainer initialization
            **kwargs: Additional keyword arguments

        Returns:
            BaseTrainer: A new trainer instance loaded from the checkpoint
        """
        if base_state is None:
            base_state = EasyDeLState
        model_state = base_state.load_state(
            load_directory=load_directory,
            device=device,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            shard_attention_computation=shard_attention_computation,
            shard_fns=shard_fns,
            backend=backend,
            platform=platform,
            config_kwargs=config_kwargs,
            model_task=model_task,
            auto_shard_model=auto_shard_model,
            partition_rules=partition_rules,
            quantization_platform=quantization_platform,
            quantization_method=quantization_method,
            quantization_block_size=quantization_block_size,
            quantization_pattern=quantization_pattern,
            quantize_tensors=quantize_tensors,
            verbose=verbose,
            **kwargs,
        )

        load_args_path = ePath(load_directory) / DEFAULT_ARGS_JSON_NAME
        arguments = TrainingArguments.load_arguments(load_args_path)
        if trainer_init_arguments is None:
            trainer_init_arguments = {}
        return cls(
            arguments=arguments,
            dataset_eval=dataset_eval,
            dataset_train=dataset_train,
            data_collator=data_collator,
            model_state=model_state,
            **trainer_init_arguments,
        )

    @property
    def model(self):
        return self._model

    @property
    def mesh(self):
        return self.model.mesh

    @mesh.setter
    def mesh(self, val):
        return val

    @property
    def training_batch_size(self):
        return self.arguments.total_batch_size * self.arguments.gradient_accumulation_steps

    @cached_property
    def is_process_zero(self):
        return self.arguments.is_process_zero

    @cached_property
    def is_enable(self):
        enable = True
        if self.arguments.process_zero_is_admin and not self.arguments.is_process_zero:
            enable = False
        return enable

    @property
    def evaluation_batch_size(self):
        return self.arguments.eval_batch_size

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return self._train_shared_fn_extra_args_

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return self._eval_shared_fn_extra_args_

    @property
    def _train_shared_fn_static_args(self) -> tuple[tp.Any]:
        return self._train_shared_fn_static_args_

    @property
    def _eval_shared_fn_static_args(self) -> tuple[tp.Any]:
        return self._eval_shared_fn_static_args_

    @_train_shared_fn_static_args.setter
    def _train_shared_fn_static_args(self, val):
        self._train_shared_fn_static_args_ = val

    @_eval_shared_fn_static_args.setter
    def _eval_shared_fn_static_args(self, val):
        self._eval_shared_fn_static_args_ = val

    @_train_shared_fn_extra_args.setter
    def _train_shared_fn_extra_args(self, val):
        self._train_shared_fn_extra_args_ = val

    @_eval_shared_fn_extra_args.setter
    def _eval_shared_fn_extra_args(self, val):
        self._eval_shared_fn_extra_args_ = val

    def _initialize_attributes(self):
        # Initialize all attributes with default values
        self.timer = getattr(self, "timer", None)
        self.wandb_runtime = getattr(self, "wandb_runtime", None)
        self.dataloader_train = getattr(self, "dataloader_train", None)
        self.dataloader_eval = getattr(self, "dataloader_eval", None)
        self.max_training_steps = getattr(self, "max_training_steps", None)
        self.max_evaluation_steps = getattr(self, "max_evaluation_steps", None)

        self.scheduler = getattr(self, "scheduler", None)
        self.tx = getattr(self, "tx", None)

        self._forward_flops_per_token = getattr(
            self,
            "_forward_flops_per_token",
            self.model_state.model.flops_per_token(include_loss=True, include_backward=False),
        )
        self._backward_flops_per_token = getattr(
            self,
            "_backward_flops_per_token",
            self.model_state.model.flops_per_token(include_loss=True, include_backward=True),
        )
        self._extra_forward_flops_per_token = getattr(self, "_extra_forward_flops_per_token", 0)
        self._extra_backward_flops_per_token = getattr(self, "_extra_backward_flops_per_token", 0)

        self.checkpoint_manager = getattr(self, "checkpoint_manager", None)
        self.pruning_module = getattr(self.arguments, "pruning_module", None)
        self.memory_monitor = getattr(self.arguments, "memory_monitor", None)

        self._model = getattr(self, "_model", None)
        self.config = getattr(self, "config", None)

        self.state_shardings = getattr(self, "state_shardings", None)
        self.model_state = getattr(self, "model_state", None)

        self._training_time_start = getattr(self, "_training_time_start", None)
        self._evaluation_time_start = getattr(self, "_evaluation_time_start", None)

        self.sharded_training_step_function = getattr(
            self,
            "sharded_training_step_function",
            None,
        )
        self.sharded_evaluation_step_function = getattr(
            self,
            "sharded_evaluation_step_function",
            None,
        )

        self.train_tracker = getattr(self, "train_tracker", CompilationTracker())
        self.evalu_tracker = getattr(self, "evalu_tracker", CompilationTracker())

        self._train_shared_fn_static_args_ = getattr(
            self,
            "_train_shared_fn_static_args_",
            {},
        )
        self._eval_shared_fn_static_args_ = getattr(
            self,
            "_eval_shared_fn_static_args_",
            {},
        )

        self._train_shared_fn_extra_args_ = getattr(
            self,
            "_train_shared_fn_extra_args",
            (),
        )
        self._eval_shared_fn_extra_args_ = getattr(
            self,
            "_eval_shared_fn_extra_args_",
            (),
        )

    def _try_resume_from_checkpoint(self, current_state: EasyDeLState) -> EasyDeLState | None:
        """
        Try to resume from the latest checkpoint if available.

        This method searches for existing checkpoints in the save directory and
        attempts to load the most recent one. It handles multiple checkpoint
        directories and falls back gracefully if loading fails.

        Args:
            current_state: The current model state (used as fallback if no checkpoint found)

        Returns:
            The resumed state if a checkpoint was found and loaded, None otherwise

        Note:
            - Checkpoints are expected to be in directories named "run-{step}"
            - The method will try multiple checkpoints if the most recent fails
            - Sets step_start_point in arguments if resuming is successful
            - ePath handles both local and cloud storage (gs://, s3://) transparently
        """
        try:
            checkpoint_dir = self.arguments._get_save_directory(create=False)
            all_paths = list(checkpoint_dir.glob("run-*"))

            if not all_paths:
                logger.info("No checkpoints found in save directory. Starting fresh training.")
                return None

            run_dir_names = set()
            for path in all_paths:
                parts = str(path).split("/")
                for i, part in enumerate(parts):
                    if part.startswith("run-") and part[4:].split("-")[0].isdigit():
                        run_dir_path = "/".join(parts[: i + 1])
                        run_dir_names.add(run_dir_path)
                        break

            if not run_dir_names:
                logger.info("No valid run directories found. Starting fresh training.")
                return None
            sorted_run_dirs = sorted(
                run_dir_names,
                key=lambda x: int(x.split("run-")[-1].split("/")[0])
                if x.split("run-")[-1].split("/")[0].isdigit()
                else 0,
                reverse=True,
            )

            for run_dir_path in sorted_run_dirs:
                try:
                    model = current_state.model
                    logger.info(f"Attempting to resume from checkpoint: {run_dir_path}")
                    run_dir = ePath(run_dir_path)

                    resumed_state = EasyDeLState.load_state(
                        load_directory=run_dir,
                        dtype=model.dtype,
                        param_dtype=model.param_dtype,
                        precision=model.precision,
                        auto_shard_model=True,
                        partition_axis=model.config.partition_axis,
                        sharding_axis_names=model.config.sharding_axis_names,
                        sharding_axis_dims=model.config.sharding_axis_dims,
                        sharding_dcn_axis_dims=model.config.sharding_dcn_axis_dims,
                        model_task=model.model_task,
                        verbose=True,
                    )

                    actual_step = int(jax.device_get(resumed_state.step))

                    logger.info(f"Successfully resumed from checkpoint at step {actual_step}")

                    if self.arguments.step_start_point is None:
                        self.arguments.step_start_point = actual_step
                        logger.info(f"Set step_start_point to {actual_step}")

                    return resumed_state

                except Exception as e:
                    logger.warning(f"Failed to load checkpoint from {run_dir_path}: {e}")
                    continue

            logger.warning("Could not load any checkpoints. Starting fresh training.")
            return None

        except Exception as e:
            logger.warning(f"Error while trying to resume from checkpoint: {e}")
            return None

    def _initialize_memory_tracking(self):
        if not self.arguments.performance_mode:
            import easydel

            interval = 1.0 if self.arguments.track_memory is True else self.arguments.track_memory
            self.memory_monitor = easydel.utils.analyze_memory.SMPMemoryMonitor(interval)

    def __repr__(self):
        return pprint.pformat(self.__dict__, indent=2)

    __str__ = __repr__

    @staticmethod
    def finish():
        if wandb is not None:
            try:
                wandb.finish()
            except Exception:
                ...

    def on_step_start(
        self,
        state: EasyDeLState,
        step: int,
    ) -> EasyDeLState:
        """hook process to call in start of the step."""
        return state

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """hook process to call in start of the step."""
        return state, metrics

    def _preprocess_batch_input(
        self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        return batch, {}

    def _ensure_functions_compiled(self):
        self.compile_aot()

    def initialize_trainer_utils(self):
        """
        Initializes various utilities used by the trainer.

        This method orchestrates the initialization of all trainer components in the
        correct order. It sets up:
        1. Weights & Biases logging (if enabled)
        2. Training timer for performance monitoring
        3. Dataloaders for training and evaluation
        4. Model, optimizer, and learning rate scheduler
        5. Model state sharding across devices
        6. Compiled training and evaluation functions

        The initialization order is important as later steps depend on earlier ones.
        For example, the optimizer configuration depends on the number of training
        steps determined during dataloader configuration.
        """

        self._initialize_wandb()
        self._initialize_timer()
        self._configure_dataloaders()
        self._configure_model()
        self._configure_state()
        self._configure_functions()

    def _initialize_wandb(self):
        if self.arguments.use_wandb:
            self.wandb_runtime = self.arguments.get_wandb_init()

    def _initialize_timer(self):
        self.timer = Timers(use_wandb=False, tensorboard_writer=self.arguments.get_tensorboard)

    def _configure_dataloaders(self):
        """
        Configures the dataloaders for training and evaluation.

        This method sets up data loading pipelines using either Grain or TensorFlow
        datasets based on the configuration. It handles:
        - Dataset offloading to specified devices if enabled
        - Calculation of maximum training and evaluation steps
        - Proper batch size and epoch configuration

        The configured dataloaders are stored as instance attributes and the
        time taken for configuration is logged for performance monitoring.
        """
        with self.timer("configure dataloaders"):
            manager = (
                jax.default_device(jax.local_devices(backend=self.arguments.offload_device)[-1])
                if self.arguments.offload_dataset
                else contextlib2.nullcontext()
            )
            with manager:
                dataset_configurations = self.configure_dataloaders()
                self.dataloader_train = dataset_configurations.dataloader_train
                self.max_training_steps = dataset_configurations.max_training_steps
                self.dataloader_eval = dataset_configurations.dataloader_eval
                self.max_evaluation_steps = dataset_configurations.max_evaluation_steps
        self.timer.log("configure dataloaders")

    def _configure_model(self):
        """
        Configures the model, optimizer, scheduler, and configuration.

        This method retrieves the model, optimizer, scheduler, and configuration from
        the `configure_model` method and configures LoRA (if enabled). It also logs
        the time taken for this configuration.
        """
        with self.timer("configure Model, Optimizer, Scheduler and Config"):
            model_configurations = self.configure_model()
            self._model = model_configurations.model
            self.tx = model_configurations.tx
            self.scheduler = model_configurations.scheduler
            self.config = model_configurations.config

        self.timer.log("configure Model, Optimizer, Scheduler and Config")

    def _configure_functions(self):
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method retrieves the configured functions from the `configure_functions`
        method, sets up the mesh, checkpoint manager, and state initialization
        function, and logs the time taken for this configuration.
        """
        with self.timer("configure functions and sharding them"):
            functions = self.configure_functions()
            sharded_training_step_function = functions.sharded_training_step_function
            sharded_evaluation_step_function = functions.sharded_evaluation_step_function

            self.sharded_training_step_function = sharded_training_step_function
            self.sharded_evaluation_step_function = sharded_evaluation_step_function
            self.mesh = functions.mesh
            self.checkpoint_manager = functions.checkpoint_manager
        self.timer.log("configure functions and sharding them")

    def _configure_state(self):
        """
        Configures and shards the model state across devices.

        This method handles:
        - Optimizer state initialization or reinitialization after checkpoint resumption
        - Creation of sharding specifications based on partition rules
        - Distribution of the model state across the device mesh

        The method ensures that the optimizer state is properly initialized whether
        starting fresh training or resuming from a checkpoint. It uses the model's
        partition rules to determine how parameters should be distributed across devices.
        """
        with self.timer("configure sharded state"):
            from eformer.escale import match_partition_rules

            with self.model.mesh:
                if self._resumed_from_checkpoint and self.model_state.opt_state is not None:
                    current_step = self.model_state.step
                    self.model_state = self.model_state.replace(tx=self.tx)
                    self.model_state = self.model_state.replace(step=current_step)

                    logger.info(f"resumed training at step {jax.device_get(current_step)}")
                elif self.arguments.init_tx and self.model_state.opt_state is None:
                    self.model_state = self.model_state.init_tx(self.tx)
                elif self.model_state.opt_state is None and self.model_state.tx is None:
                    self.model_state = self.model_state.replace(tx=self.tx)
                elif self.model_state.opt_state is not None and self.model_state.tx is None:
                    self.model_state = self.model_state.replace(tx=self.tx)

                shape = nn.eval_shape(lambda: self.model_state)
                rules = self.model.config.get_partition_rules()
                state_shardings = specs_to_name_sharding(match_partition_rules(rules, shape))

                self.state_shardings = state_shardings
                self.model_state = self.model_state.shard_with_shape(state_shardings)

        self.timer.log("configure sharded state")

    @abstractmethod
    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Creates a function to collect and process batches of data for training or evaluation.

        This function handles padding or truncating sequences to the specified `max_sequence_length`
        based on the chosen `truncation_mode`.

        Args:
            max_sequence_length (int): The maximum allowed sequence length.
            truncation_mode (typing.tp.Literal["keep_end", "keep_start"], optional):
                The truncation mode. Defaults to "keep_end".

        Returns:
            tp.Callable: A function that takes a batch of data and returns a processed batch.
        """

    @abstractmethod
    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Creates a function to collect and process batches of data for training or evaluation.

        This function handles padding or truncating sequences to the specified `max_sequence_length`
        based on the chosen `truncation_mode`.

        Args:
            max_sequence_length (int): The maximum allowed sequence length.
            truncation_mode (typing.tp.Literal["keep_end", "keep_start"], optional):
                The truncation mode. Defaults to "keep_end".

        Returns:
            tp.Callable: A function that takes a batch of data and returns a processed batch.
        """

    @abstractmethod
    def create_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Creates a function to collect and process batches of data for training or evaluation.

        This function handles padding or truncating sequences to the specified `max_sequence_length`
        based on the chosen `truncation_mode`.

        Args:
            max_sequence_length (int): The maximum allowed sequence length.
            truncation_mode (typing.tp.Literal["keep_end", "keep_start"], optional):
                The truncation mode. Defaults to "keep_end".

        Returns:
            tp.Callable: A function that takes a batch of data and returns a processed batch.
        """

    @abstractmethod
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
        raise NotImplementedError

    def _configure_grain_dataloader(self):
        def _create_grain_dataloader(dataset: Dataset | IterableDataset, is_train: bool) -> grain.DataLoader:
            """Creates a Grain DataLoader from a Hugging Face Dataset."""

            if is_train:
                batch_size = self.training_batch_size
                shuffle = self.arguments.shuffle_train_dataset
                num_epochs = self.arguments.num_train_epochs
            else:
                batch_size = self.evaluation_batch_size
                shuffle = False
                num_epochs = 1
            shard_options = grain.ShardOptions(
                shard_index=self.arguments.grain_shard_index or 0,
                shard_count=self.arguments.grain_shard_count or 1,
                drop_remainder=True,
            )
            from datasets import IterableDataset

            if is_train and hasattr(self, "model_state") and self.model_state is not None:
                current_step = int(jax.device_get(self.model_state.step))
                if current_step > 0:
                    logger.info(f"Note: Grain dataloader will start fresh, but model continues from step {current_step}")

            if isinstance(dataset, IterableDataset):
                data_source = HFDataSource(dataset=dataset, shard_options=shard_options, num_threads=1)
                sampler = grain.IndexSampler(
                    num_records=len(data_source),
                    shard_options=shard_options,
                    seed=self.arguments.shuffle_seed_train if is_train else 0,
                    num_epochs=num_epochs,
                    shuffle=shuffle,
                )
            else:
                data_source = grain.MapDataset.source(dataset)
                sampler = grain.IndexSampler(
                    num_records=len(data_source),
                    shard_options=shard_options,
                    seed=self.arguments.shuffle_seed_train if is_train else 0,
                    num_epochs=num_epochs,
                    shuffle=shuffle,
                )
            collate_fn = self.create_grain_collect_function(
                max_sequence_length=self.arguments.max_sequence_length,
                truncation_mode=self.arguments.truncation_mode,
            )
            return grain.DataLoader(
                data_source=data_source,
                sampler=sampler,
                operations=[
                    ToNumpy(),
                    CollateMapTransform(collate_fn=collate_fn),
                    grain.Batch(batch_size=batch_size, drop_remainder=True),
                ],
                worker_count=0,
                worker_buffer_size=1,
                read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=128),
            )

        def calculate_steps(dataset, is_train: bool) -> int:
            if hasattr(dataset, "__len__"):
                total_data_len = len(dataset)
            else:
                total_data_len = (
                    self.arguments.per_epoch_training_steps if is_train else self.arguments.per_epoch_evaluation_steps
                )
                if total_data_len is None:
                    raise ValueError(
                        f"Specify the number of per epoch {'training' if is_train else 'evaluation'} "
                        "steps for a generator/streaming dataset."
                    )
            batch_size = self.arguments.total_batch_size if is_train else self.evaluation_batch_size
            num_steps = (
                (total_data_len + batch_size - 1) // batch_size * (self.arguments.num_train_epochs if is_train else 1)
            )
            forced_steps = self.arguments.max_training_steps if is_train else self.arguments.max_evaluation_steps
            steps = forced_steps if forced_steps is not None else num_steps

            if is_train:
                steps = steps // self.arguments.gradient_accumulation_steps
            return steps

        max_training_steps = calculate_steps(self.dataset_train, is_train=True)
        dataloader_train = _create_grain_dataloader(self.dataset_train, is_train=True)

        dataloader_eval, max_evaluation_steps = None, 0
        if self.dataset_eval is not None and self.arguments.do_eval:
            max_evaluation_steps = calculate_steps(self.dataset_eval, is_train=False)
            dataloader_eval = _create_grain_dataloader(self.dataset_eval, is_train=False)

        return TrainerConfigureDataloaderOutput(
            dataloader_train=dataloader_train,
            max_training_steps=max_training_steps,
            dataloader_eval=dataloader_eval,
            max_evaluation_steps=max_evaluation_steps,
        )

    def _configure_tfds_dataloader(self):
        if not is_package_available("tensorflow"):
            raise ImportError("Please install `tensorflow` to use the `tensorflow-datasets` conversion.")
        import tensorflow as tf  # type:ignore

        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], "GPU")
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != "GPU"
        except RuntimeError as e:
            # Invalid device or cannot modify virtual devices once initialized.
            logger.error(f"Failed to disable GPU devices: {e}")
        except AssertionError:
            logger.warning("TensorFlow may be hogging GPU memory.")

        def create_tf_dataset(dataset: Dataset, is_train: bool) -> tp.Iterator[np.ndarray]:
            """
            Creates a TensorFlow dataset from a Hugging Face Dataset.

            Args:
                dataset (Dataset): The Hugging Face Dataset.
                is_train (bool): Whether the dataset is for training.

            Returns:
                tp.Iterator[np.ndarray]: The TensorFlow dataset iterator.
            """

            batch_size = self.training_batch_size if is_train else self.evaluation_batch_size

            return (
                dataset.to_tf_dataset(
                    collate_fn=self.create_tfds_collect_function(
                        max_sequence_length=self.arguments.max_sequence_length,
                        truncation_mode=self.arguments.truncation_mode,
                    ),
                    batch_size=batch_size,
                    drop_remainder=True,
                    shuffle=is_train and self.arguments.shuffle_train_dataset,
                    num_workers=self.arguments.dataloader_num_workers,
                )
                .repeat(self.arguments.num_train_epochs if is_train else 1)
                .prefetch(tf.data.AUTOTUNE)
                .as_numpy_iterator()
            )

        def create_tf_dataset_from_iterable(dataset: IterableDataset, is_train: bool) -> tp.Iterator[np.ndarray]:
            """
            Creates a TensorFlow dataset from an iterable Hugging Face Dataset.

            Args:
                dataset (IterableDataset): The iterable Hugging Face Dataset.
                is_train (bool): Whether the dataset is for training.

            Returns:
                tp.Iterator[np.ndarray]: The TensorFlow dataset iterator.
            """

            batch_size = self.training_batch_size if is_train else self.evaluation_batch_size
            tf_data_mapping = {
                "float16": tf.float16,
                "float32": tf.float32,
                "float64": tf.float64,
                "int16": tf.int16,
                "int32": tf.int32,
                "int64": tf.int64,
                "bool": tf.bool,
            }
            return (
                tf.data.Dataset.from_generator(
                    lambda: dataset,
                    output_signature={
                        col: tf.TensorSpec(
                            shape=vals.shape[1:]
                            if len(vals.shape) > 1 and vals.shape[0] == 1  # auto remove batch dim
                            else vals.shape,
                            dtype=tf_data_mapping[str(vals.dtype)],
                        )
                        for col, vals in next(iter(dataset)).items()
                        if hasattr(vals, "shape")
                    },
                )
                .repeat(self.arguments.num_train_epochs if is_train else 1)
                .batch(batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
                .as_numpy_iterator()
            )

        def calculate_steps(dataset: Dataset | IterableDataset, is_train: bool) -> int:
            """
            Calculates the number of training or evaluation steps based on dataset length and arguments.

            Args:
              dataset (tp.Union[Dataset, IterableDataset]): The dataset to calculate steps for.
              is_train (bool): Whether the dataset is for training.

            Returns:
              int: The number of steps.

            Raises:
              ValueError: If the dataset is a generator/streaming dataset and the number of steps is not specified.
            """
            if hasattr(dataset, "__len__"):
                total_data_len = len(dataset)
            else:
                total_data_len = (
                    self.arguments.per_epoch_training_steps if is_train else self.arguments.per_epoch_evaluation_steps
                )
                if total_data_len is None:
                    raise ValueError(
                        f"Specify the number of per epoch {'training' if is_train else 'evaluation'} "
                        "steps for a generator/streaming dataset."
                    )
            batch_size = self.arguments.total_batch_size if is_train else self.evaluation_batch_size
            num_steps = (
                (total_data_len + batch_size - 1) // batch_size * (self.arguments.num_train_epochs if is_train else 1)
            )
            forced_steps = self.arguments.max_training_steps if is_train else self.arguments.max_evaluation_steps
            steps = forced_steps if forced_steps is not None else num_steps

            if is_train:
                steps = steps // self.arguments.gradient_accumulation_steps
            return steps

        def to_tf_dataloader(dataset: Dataset | IterableDataset, is_train: bool) -> tp.Iterator[np.ndarray]:
            """
            Converts a Hugging Face Dataset to a TensorFlow dataloader.

            Args:
                dataset (tp.Union[Dataset, IterableDataset]): The Hugging Face Dataset.
                is_train (bool): Whether the dataset is for training.

            Returns:
                tp.Iterator[np.ndarray]: The TensorFlow dataloader iterator.
            """
            if hasattr(dataset, "__len__"):
                return create_tf_dataset(dataset, is_train)
            else:
                return create_tf_dataset_from_iterable(dataset, is_train)

        max_training_steps = calculate_steps(self.dataset_train, is_train=True)

        dataloader_train = to_tf_dataloader(self.dataset_train, is_train=True)

        if self.dataset_eval is not None and self.arguments.do_eval:
            max_evaluation_steps = calculate_steps(self.dataset_eval, is_train=False)
            dataloader_eval = to_tf_dataloader(self.dataset_eval, is_train=False)
        else:
            dataloader_eval, max_evaluation_steps = None, 0

        return TrainerConfigureDataloaderOutput(
            dataloader_train=dataloader_train,
            max_training_steps=max_training_steps,
            dataloader_eval=dataloader_eval,
            max_evaluation_steps=max_evaluation_steps,
        )

    def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
        """
        Configures the dataloaders for training and evaluation.

        This method creates the training and evaluation dataloaders using the provided
        datasets and data collator. It also determines the maximum number of training
        and evaluation steps based on the dataset sizes and training arguments.

        Returns:
            TrainerConfigureDataloaderOutput: An object containing the configured dataloaders and the
                                            maximum number of training and evaluation steps.
        """
        if self.arguments.use_grain:
            return self._configure_grain_dataloader()
        else:
            return self._configure_tfds_dataloader()

    def configure_model(self) -> TrainerConfigureModelOutput:
        """
        Configures the model, optimizer, scheduler, and configuration.

        This method retrieves the model configuration from the model state, creates
        the optimizer and scheduler using the training arguments, and returns an
        object containing the configured model, optimizer, scheduler, and configuration.

        Returns:
            TrainerConfigureModelOutput: An object containing the configured
                model, optimizer, scheduler, and configuration.
        """

        # Always create the optimizer and scheduler according to config
        # If we resumed from checkpoint, the optimizer state will be reinitialized in _configure_state
        tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_training_steps)
        if self.pruning_module is not None:
            tx = self.pruning_module.wrap_optax(tx)

        return TrainerConfigureModelOutput(
            model=self.model,
            tx=tx,
            scheduler=scheduler,
            config=self.model.config,
        )

    def _save_state(self, state: EasyDeLState, *args, **kwargs) -> str:
        """
        Save the current model state to a checkpoint.

        This method handles the complete checkpoint saving process including:
        - Managing checkpoint limits (removing old checkpoints if needed)
        - Creating checkpoint directories with proper naming
        - Saving training arguments alongside the model
        - Generating README documentation for the checkpoint
        - Saving the actual model state with optional optimizer state

        Args:
            state: The model state to save
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            str: Path to the saved checkpoint directory
        """
        step = self._get_current_step(state)
        self._manage_checkpoint_limit(self.arguments._get_save_directory())
        directory_name = self.arguments._get_save_directory_milestone(step=step, create=True)
        logger.info(f"saving state {directory_name}.")
        directory_name.mkdir(exist_ok=True)
        self.arguments.save_arguments(directory_name / DEFAULT_ARGS_JSON_NAME)
        self._save_readme(directory_name)
        state.save_state(
            save_directory=directory_name,
            float_dtype=self.model.param_dtype,
            save_optimizer=self.arguments.save_optimizer_state,
        )

        return str(directory_name)

    def _get_current_step(self, state):
        step = int(jax.device_get(state.step))
        if self.arguments.step_start_point is not None:
            step += self.arguments.step_start_point
        return step

    def _manage_checkpoint_limit(self, save_directory):
        """
        Manage the checkpoint limit by removing old checkpoints.

        This method enforces the save_total_limit by removing the oldest
        checkpoints when the limit is exceeded. It safely handles directory
        removal and ensures only valid checkpoint directories are deleted.

        Args:
            save_directory: Base directory containing checkpoints

        Note:
            - Only removes directories matching the pattern "run-{number}"
            - Sorts checkpoints by modification time to identify oldest
            - Handles errors gracefully to prevent training interruption
        """

        def _remove_directory_recursive(path):
            """Recursively remove directory using ePath methods"""
            if not path.exists():
                return

            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    for item in path.iterdir():
                        _remove_directory_recursive(item)
                    path.rmdir()
                except Exception as e:
                    logger.warning(f"Error removing directory {path}: {e}")

        def _operate():
            try:
                save_path = ePath(save_directory)
                checkpoint_dirs = []
                try:
                    all_run_paths = list(save_path.glob("run-*"))
                    checkpoint_dirs = [p for p in all_run_paths if p.is_dir()]

                    if not checkpoint_dirs:
                        return

                    run_pattern = re.compile(r"^run-\d+$")
                    checkpoint_dirs = [p for p in checkpoint_dirs if run_pattern.match(p.name)]

                except Exception as e:
                    logger.warning(f"Error listing checkpoint directories in {save_path}: {e}")
                    return

                if not checkpoint_dirs:
                    return

                def get_mtime(path):
                    try:
                        return path.stat().get("mtime", 0)
                    except Exception:
                        return 0

                checkpoint_dirs.sort(key=get_mtime)

                if self.arguments.save_total_limit == 0:
                    _do_dele = checkpoint_dirs
                else:
                    _do_dele = checkpoint_dirs[: -self.arguments.save_total_limit]

                for old_save_directory in _do_dele:
                    try:
                        # Double-check it's a directory and matches pattern before removal
                        if old_save_directory.is_dir() and old_save_directory.name.startswith("run-"):
                            _remove_directory_recursive(old_save_directory)
                            logger.info(f"Removed old checkpoint directory: {old_save_directory}")
                        else:
                            logger.warning(f"Skipping non-directory or invalid pattern: {old_save_directory}")
                    except Exception as e:
                        logger.error(f"Failed to remove directory {old_save_directory}: {e}")

            except Exception as e:
                logger.error(f"Error in checkpoint limit management: {e}")

        if self.arguments.save_total_limit is not None:
            _operate()

    def _save_readme(self, save_directory):
        dst = ePath(save_directory) / "README.md"
        dst.write_text(self._get_information())

    def _format_partition_rules(self) -> str:
        """Format partition rules with proper indentation and formatting."""
        try:
            return pprint.pformat(self.model.config.get_partition_rules(), indent=2, width=80)
        except Exception as e:
            logger.error(f"Error formatting partition rules: {e!s}")
            return "Error retrieving partition rules"

    def _get_device_info(self) -> dict:
        """Get information about available devices."""
        try:
            return {
                "platform": jax.local_devices()[0].platform.upper(),
                "device_count": jax.device_count(),
                "host_device_count": jax.local_device_count(),
            }
        except Exception as e:
            logger.error(f"Error getting device info: {e!s}")
            return {"platform": "UNKNOWN", "device_count": 0}

    def _get_information(self) -> str:
        """
        Generate formatted information about the model and training setup.

        This method creates a comprehensive README document containing:
        - Model architecture and configuration details
        - Training hyperparameters and settings
        - Device and platform information
        - Partition rules for distributed training
        - Data types and precision settings

        The method attempts to use Jinja2 templates for rich formatting but
        falls back to a basic format if Jinja2 is not available.

        Returns:
            str: Formatted markdown string containing model and training information

        Note:
            - Requires Jinja2 for full formatting capabilities
            - Handles missing configuration gracefully
            - Includes EasyDeL version information
        """

        if self.config is None:
            logger.warning(
                "Model config is not available for README generation. Ensure `_configure_model` has been called."
            )
            return (
                f"# Partial Training Information for {self.arguments.model_name}\n"
                f"Model configuration was not available at the time of README generation.\n"
                f"Trained with EasyDeL v{__version__}."
            )

        device_info = self._get_device_info()
        partition_rules_str = self._format_partition_rules()

        _TASK_TYPE_TO_AUTO_CLASS_KEY = {
            TaskType.CAUSAL_LM: "CausalLM",
            TaskType.SEQUENCE_CLASSIFICATION: "SequenceClassification",
            TaskType.IMAGE_TEXT_TO_TEXT: "ImageTextToText",
            TaskType.VISION_LM: "ImageTextToText",
            TaskType.DIFFUSION_LM: "DiffusionLM",
            TaskType.BASE_MODULE: "",
            TaskType.BASE_VISION: "",
            TaskType.SEQUENCE_TO_SEQUENCE: "sequence-to-sequence",
            TaskType.SPEECH_SEQUENCE_TO_SEQUENCE: "SpeechSeq2Seq",
            TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION: "ZeroShotImageClassification",
            TaskType.AUDIO_CLASSIFICATION: "",
            TaskType.IMAGE_CLASSIFICATION: "",
            TaskType.AUTO_BIND: "",
        }
        model_actual_task_type = getattr(self.config, "task_type", TaskType.CAUSAL_LM)
        if not isinstance(model_actual_task_type, TaskType):
            model_actual_task_type = TaskType.CAUSAL_LM
        model_task_for_template = _TASK_TYPE_TO_AUTO_CLASS_KEY.get(model_actual_task_type, "")

        attn_config_value = getattr(self.config, "attn_mechanism", "vanilla")
        if hasattr(attn_config_value, "value"):
            attn_mechanism_for_template = str(attn_config_value.value)
        else:
            attn_mechanism_for_template = str(attn_config_value)

        model_data = {
            "name": self.arguments.model_name,
            "architecture": str(getattr(self.config, "model_type", "N/A")),
            "device_info": device_info,
            "partition_rules_str": partition_rules_str,
            "dtype_str": str(self.model.dtype) if self.model else "N/A",
            "param_dtype_str": str(self.model.param_dtype) if self.model else "N/A",
            "model_task_str": model_task_for_template,
            "attn_mechanism_str": attn_mechanism_for_template,
        }

        context = {
            "arguments": self.arguments,
            "config": self.config,
            "model": model_data,
            "easydel_version": __version__,
        }
        try:
            from jinja2 import Environment, FileSystemLoader

            env = Environment(
                loader=FileSystemLoader(os.path.dirname(__file__)),
                autoescape=False,
            )
            template = env.from_string(readme_generator.EASYDEL_TRAINER_README_TEMPLATE)
            out = template.render(context).strip()
            return out
        except ImportError:
            logger.warning(
                "Jinja2 not installed. Falling back to basic README generation. "
                "Install Jinja2 for a richer README: `pip install jinja2`"
            )

            return f"""
# {self.arguments.model_name} - Trained with EasyDeL v{__version__}

## Training Configuration (Basic)
- Model Architecture: {model_data["architecture"]}
- Platform: {device_info["platform"]} ({device_info["device_count"]} devices)
- Learning Rate: {self.arguments.learning_rate}
- Optimizer: {self.arguments.optimizer}
- Epochs: {self.arguments.num_train_epochs}
- Batch Size: {self.arguments.total_batch_size}
- Sequence Length: {self.arguments.max_sequence_length}
- Dtype: {model_data["dtype_str"]}
- Params Dtype: {model_data["param_dtype_str"]}

*Install Jinja2 for a more detailed README.*
"""
        except Exception as e:
            logger.error(f"Error generating README with Jinja2: {e!s}")
            return f"# Error during README generation for {self.arguments.model_name}"

    def save_information(self, output_path: str | ePathLike) -> None:
        """
        Save the generated information to a markdown file.

        Args:
            output_path: Path where the markdown file should be saved
        """
        try:
            output_path = ePath(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            output_path.write_text(self._get_information())

            logger.info(f"Information saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving information: {e!s}")
            raise

    def save_pretrained(
        self,
        state: EasyDeLState,
        save_directory: str | None = None,
        gather_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None = None,
        to_torch: bool = False,
        easystate_to_huggingface_model_kwargs: dict | None = None,
        torch_save_pretrained_kwargs: dict | None = None,
    ):
        save_directory = save_directory or self.arguments.get_path()
        save_directory = ePath(save_directory)
        if to_torch:
            return self._save_to_torch(
                state=state,
                save_directory=save_directory,
                easystate_to_huggingface_model_kwargs=easystate_to_huggingface_model_kwargs,
                torch_save_pretrained_kwargs=torch_save_pretrained_kwargs,
            )
        else:
            return self._save_state(
                state=state,
                gather_fns=gather_fns,
                save_directory=save_directory,
            )

    def _save_to_torch(
        self,
        state: EasyDeLState,
        save_directory: str | os.PathLike,
        easystate_to_huggingface_model_kwargs: dict | None = None,
        torch_save_pretrained_kwargs: dict | None = None,
    ):
        easystate_to_huggingface_model_kwargs = easystate_to_huggingface_model_kwargs or {}
        torch_save_pretrained_kwargs = torch_save_pretrained_kwargs or {}
        hf_model = state.model.to_torch(**easystate_to_huggingface_model_kwargs)
        self._save_readme(save_directory)
        hf_model.save_pretrained(save_directory, **torch_save_pretrained_kwargs)
        return hf_model

    def _create_hf_model_config(
        self,
        state: EasyDeLState,
        model_config,
        model_type,
    ):
        from transformers import AutoConfig

        hf_model_config = AutoConfig.for_model(model_type=model_type)
        unsafe_dict = state.unsafe_dict(model_config.__dict__)
        blocked_statics = ["torch_dtype"]

        for k, v in unsafe_dict.items():
            if not k.startswith("_") and k in hf_model_config.__dict__ and k not in blocked_statics:
                if isinstance(v, str) and v.isnumeric():
                    v = int(float(v)) if float(v).is_integer() else float(v)
                setattr(hf_model_config, k, v)

        return hf_model_config

    def specs_to_name_sharding(self, tree, mesh=None):
        mesh = mesh or self.mesh or self.model.mesh
        return specs_to_name_sharding(tree, mesh)

    def calculate_number_total_flops(self, params, is_training=True):
        return 6 * sum(x.size for x in jax.tree_util.tree_flatten(unfreeze(params))[0])

    @staticmethod
    def count_model_parameters(prm):
        """Prints the number of model parameters in billions."""
        return sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(prm))[0])

    def apply_training_hooks(self, metrics: LossMetrics) -> LossMetrics:
        if self.arguments.loss_config is not None and self.arguments.loss_config.break_on_nan:
            if jnp.isnan(metrics.loss):
                info = "Prevent Running Model Due to NaN Loss"
                logger.info(info)
                raise EasyDeLBreakRequest(info)
        if (
            self.arguments.training_time_seconds is not None
            and time.time() > self.arguments.training_time_seconds + self._training_time_start
        ):
            info = "Prevent Running Model Due to Time Limit"
            logger.info(info)
            raise EasyDeLTimerError(info)
        return metrics

    def start_training_hook(self):
        self._setup_static_metrics()
        self._training_time_start = time.time()

    def start_evaluation_hook(self):
        self._setup_static_metrics()
        self._evaluation_time_start = time.time()

    def _setup_static_metrics(self): ...

    def compile_aot(self) -> bool:
        """
        Compile training and evaluation functions ahead-of-time.

        This method performs AOT (Ahead-Of-Time) compilation of the training
        and evaluation step functions using JAX's JIT compilation. This improves
        performance by compiling the functions once before the training loop.

        Returns:
            bool: True if any functions were compiled, False otherwise

        Note:
            - Compilation happens automatically on first call if not done AOT
            - AOT compilation can reduce first-step latency
            - Uses actual data batches to determine compilation shapes
        """
        compiled = False

        def compile_function(function, dataloader, state, tag):
            if not isinstance(function, Compiled):
                logger.info("Compiling function: %s", tag)
                return function.lower(state, next(iter(dataloader))).compile()
            return function

        if self.dataloader_train is not None:
            self.sharded_training_step_function = compile_function(
                self.sharded_training_step_function,
                self.dataloader_train,
                self.model_state,
                "trainer.sharded_training_step_function",
            )
            compiled = True

        if self.dataloader_eval is not None:
            self.sharded_evaluation_step_function = compile_function(
                self.sharded_evaluation_step_function,
                self.dataloader_eval,
                self.model_state,
                "trainer.sharded_evaluation_step_function",
            )
            compiled = True

        return compiled

    def _should_save_checkpoint(self, current_step):
        """
        Determine if checkpoint should be saved at current step.

        Args:
            current_step: The current training step

        Returns:
            bool: True if a checkpoint should be saved at this step

        Note:
            Based on save_steps configuration in training arguments
        """
        return (
            self.arguments.save_steps is not None and current_step > 0 and current_step % self.arguments.save_steps == 0
        )

    def _should_run_evaluation(self, current_step):
        """
        Determine if evaluation should be run at current step.

        Args:
            current_step: The current training step

        Returns:
            bool: True if evaluation should be run at this step

        Note:
            Based on evaluation_steps configuration in training arguments
        """
        return (
            self.arguments.evaluation_steps is not None
            and current_step > 0
            and (current_step % self.arguments.evaluation_steps) == 0
        )

    def _prepare_training_output(
        self,
        state: EasyDeLState,
        run_exception: Exception | None = None,
    ):
        if run_exception is not None:
            if isinstance(run_exception, KeyboardInterrupt):
                logger.warning("KeyboardInterrupt: Training interrupted. Saving current state...")
            elif isinstance(run_exception, EasyDeLTimerError):
                logger.warning("Training reached maximum time limit. Saving current state...")
            elif isinstance(run_exception, StopIteration):
                ...  # simply just pass
            else:
                raise RuntimeError("EasyDeL Runtime dumped") from run_exception
        checkpoint_path = "SAVING_SKIPPED"
        filename = None

        dire = ePath(self.arguments.save_directory)
        if self.arguments.do_last_save:
            filename = self._save_state(state=state, milestone=False, save_directory=dire)
            if self.arguments.save_directory is not None:
                checkpoint_path = dire / filename

        return TrainerOutput(
            state=state,
            mesh=self.mesh,
            checkpoint_path=str(checkpoint_path),
            last_save_file_name=filename,
        )

    def _handle_training_interruption(
        self,
        state: EasyDeLState,
        exception: Exception,
        shard_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None,
        gather_fns: tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable] | None,
    ):
        """Handle training interruption gracefully."""
        if isinstance(exception, KeyboardInterrupt):
            logger.warning("KeyboardInterrupt: Training interrupted. Saving current state...")
        elif isinstance(exception, EasyDeLTimerError):
            logger.warning("Training reached maximum time limit. Saving current state...")
        else:
            raise RuntimeError("EasyDeL Runtime dumped") from exception
        return self._prepare_training_output(
            state=state,
            checkpoint_manager=self.checkpoint_manager,
            shard_fns=shard_fns,
            gather_fns=gather_fns,
            run_exception=None,
        )

    def _setup_initial_metrics(self, state):
        """Setup initial metrics logging."""
        # Calculate and log model size
        model_size = self.count_model_parameters(state.graphstate)
        self.arguments.log_metrics(
            {
                "Number of Model Parameters (Billion)": model_size,
                "process_count": jax.process_count(),
                "device_count": jax.device_count(),
                "local_device_count": jax.local_device_count(),
                "platform": jax.extend.backend.get_backend().platform,
                "XLA_FLAGS": os.getenv("XLA_FLAGS", ""),
                "LIBTPU_INIT_ARGS": os.getenv("LIBTPU_INIT_ARGS", ""),
            },
            step=0,
            log_as="config",
        )

    def _get_next_batch(self, data_iter, dataloader):
        """Get next batch from iterator, reinitializing if needed."""
        try:
            batch = next(data_iter)
        except (StopIteration, IndexError):
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Remove specified ids from batch if needed
        for id_to_pop in self.arguments.ids_to_pop_from_dataset:
            _ = batch.pop(id_to_pop, None)

        return batch, data_iter

    def create_progress_bar(
        self,
        total: int,
        desc: str = "",
        disabled: bool = False,
    ) -> BaseProgressBar:
        """Create a progress bar of the specified type."""
        if disabled:
            return NullProgressBar()
        rpr = self.arguments.progress_bar_type
        if rpr == "tqdm":
            ncols = int(os.getenv("TQDM_NCOLS", "0"))
            return TqdmProgressBar(
                tqdm(
                    total=total,
                    desc=desc,
                    disable=disabled,
                    ncols=ncols if ncols > 0 else None,
                )
            )
        elif rpr == "rich":  # rich
            from rich.progress import Progress

            if hasattr(self, "_hidden_rich_pbar"):
                progress = self._hidden_rich_pbar
            else:
                from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

                from .trainer_protocol import MetricsColumn

                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    MetricsColumn(metrics_to_show=self.arguments.metrics_to_show_in_rich_pbar),
                    expand=True,
                    refresh_per_second=10,
                    disable=disabled,
                )
                progress.start()
                self._hidden_rich_pbar = progress
            task_id = progress.add_task(desc, total=total)
            return RichProgressBar(progress, task_id)
        elif rpr == "json":
            return JSONProgressBar(desc=desc)
        else:
            raise NotImplementedError(f"Progress Bar type {rpr}'s not supported.")

    def log_weight_distribution(self, state: EasyDeLState, step: int):
        return self.arguments.log_weight_distribution(state=state, step=step)

    def log_metrics(
        self,
        metrics: MetricsType,
        pbar: BaseProgressBar,
        step: int,
        mode: str = "train",
    ):
        """Log metrics and update progress bar."""

        if step % self.arguments.log_steps == 0:
            if step == 0:
                pbar.reset()
            display_metrics = {
                k.replace("train/", "").replace("eval/", ""): v
                for k, v in metrics.items()
                if not (
                    k.startswith("train-mlperf/")
                    or k.startswith("eval-mlperf/")
                    or k.startswith("mlperf/")
                    or k.startswith("train/grad_norm")
                    or k.startswith("eval/grad_norm")
                )
            }
            # Update progress bar
            pbar.set_postfix(**display_metrics)
            update_size = 0 if step == 0 else self.arguments.log_steps
            pbar.update(update_size)
        if step % self.arguments.report_steps == 0:
            self.arguments.log_metrics(metrics=metrics, step=step)
