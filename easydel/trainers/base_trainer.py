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

import copy
import os
import pprint
import time
import typing as tp
from abc import abstractmethod
from functools import cached_property

import contextlib2
import flax
import flax.nnx
import grain.python as grain
import jax
import jax.extend
import numpy as np
from eformer import common_types
from eformer.escale import with_sharding_constraint
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from flax import nnx as nn
from jax import numpy as jnp
from jax._src.stages import Compiled
from jax.sharding import NamedSharding, PartitionSpec
from tqdm.autonotebook import tqdm
from transformers import GenerationConfig

from easydel import __version__
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.factory import TaskType
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.utils import CompilationTracker
from easydel.utils import Timers, readme_generator
from easydel.utils.compiling_utils import ejit
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

log_debug_maybe = logger.debug

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
                from eformer.serialization.checkpointer import find_latest_checkpoint

                checkpoint_path = find_latest_checkpoint(str(self.arguments._get_save_directory(create=False)))
                if checkpoint_path is not None:
                    logger.info(f"Found latest checkpoint: {checkpoint_path}")
                    model = model_state.model
                    resumed_state = EasyDeLState.load_state(
                        load_directory=checkpoint_path,
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
                        tx_template=arguments.get_tx_template(),
                    )
                    actual_step = int(jax.device_get(resumed_state.step))
                    logger.info(f"Successfully resumed from checkpoint at step {actual_step}")

                    if self.arguments.step_start_point is None:
                        self.arguments.step_start_point = actual_step
                        logger.info(f"Set step_start_point to {actual_step}")

                    model_state = resumed_state
                    self._resumed_from_checkpoint = True
                else:
                    logger.info("No checkpoints found. Starting fresh training.")
            except Exception as e:
                logger.warning(f"Resuming from checkpoint failed: {e}. Starting fresh training.")

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

    @property
    def model(self):
        """Get the model instance.

        Returns:
            The model instance used for training
        """
        return self._model

    @property
    def mesh(self):
        """Get the device mesh for distributed computation.

        Returns:
            The device mesh used for sharding computations
        """
        return self.model.mesh

    @mesh.setter
    def mesh(self, val):
        return val

    @property
    def training_batch_size(self):
        """Calculate the effective training batch size.

        Returns:
            The effective batch size including gradient accumulation

        Notes
        -----
        The effective batch size is calculated as:
        total_batch_size * gradient_accumulation_steps
        """
        return self.arguments.total_batch_size * self.arguments.gradient_accumulation_steps

    @cached_property
    def is_process_zero(self):
        """Check if this is the main process (rank 0).

        Returns:
            True if this is the main process, False otherwise
        """
        return self.arguments.is_process_zero

    @cached_property
    def is_enable(self):
        """Check if operations are enabled for this process.

        Returns:
            True if operations are enabled, False if restricted to main process only

        Notes
        -----
        When process_zero_is_admin is True, only the main process
        will have operations enabled.
        """
        enable = True
        if self.arguments.process_zero_is_admin and not self.arguments.is_process_zero:
            enable = False
        return enable

    @property
    def evaluation_batch_size(self):
        """Get the evaluation batch size.

        Returns:
            The batch size used for evaluation
        """
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
        """Initialize all trainer attributes with default values.

        Notes:
            This method initializes various trainer components including:
            - Timer and wandb runtime
            - Dataloaders for training and evaluation
            - Maximum training/evaluation steps
            - Optimizer (tx) and scheduler
            - Flops tracking for performance monitoring
            - Checkpoint manager and pruning module
            - Model state and sharding configurations
            - Compilation trackers for training and evaluation
        """
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
        self.generate_function = getattr(self, "generate_function", None)
        self.latest_generation_samples = getattr(self, "latest_generation_samples", [])
        rng = getattr(self, "_generation_rng", None)
        seed = getattr(self.arguments, "generation_seed", None)
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        self._generation_rng = rng

    def _initialize_memory_tracking(self):
        """Initialize memory monitoring for tracking GPU/TPU memory usage.

        Notes:
            Only initializes when performance_mode is False.
            Uses SMPMemoryMonitor with configurable interval.
        """
        if not self.arguments.performance_mode:
            import easydel

            interval = 1.0 if self.arguments.track_memory is True else self.arguments.track_memory
            self.memory_monitor = easydel.utils.analyze_memory.SMPMemoryMonitor(interval)

    def __repr__(self):
        """Return string representation of the trainer.

        Returns:
            str: Pretty-formatted string of trainer attributes.
        """
        return pprint.pformat(self.__dict__, indent=2)

    __str__ = __repr__

    @staticmethod
    def finish():
        """Clean up resources and finish any active logging sessions.

        Notes:
            Currently only finishes wandb session if active.
            Safe to call even if wandb is not initialized.
        """
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
        """Hook method called at the start of each training step.

        Parameters
        ----------
        state : EasyDeLState
            The current model state
        step : int
            The current training step number

        Returns
        -------
        EasyDeLState
            The potentially modified model state

        Notes
        -----
        This method can be overridden in subclasses to implement
        custom logic at the beginning of each training step.
        """
        return state

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """Hook method called at the end of each training step.

        Parameters
        ----------
        state : EasyDeLState
            The current model state
        metrics : MetricsType
            The metrics computed for this step
        step : int
            The current training step number

        Returns
        -------
        tuple[EasyDeLState, MetricsType]
            The potentially modified model state and metrics

        Notes
        -----
        This method can be overridden in subclasses to implement
        custom logic at the end of each training step, such as
        custom logging or state modifications.
        """
        return state, metrics

    def _preprocess_batch_input(
        self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Preprocess a batch of input data before feeding to the model.

        Parameters
        ----------
        state : EasyDeLState
            The current model state
        batch : dict[str, jax.Array]
            The input batch data
        is_train : bool
            Whether this is a training batch (True) or evaluation batch (False)

        Returns
        -------
        tuple[dict[str, jax.Array], dict[str, float | int | str]]
            A tuple containing:
            - The preprocessed batch data
            - Additional metadata or metrics from preprocessing

        Notes
        -----
        This method can be overridden to implement custom preprocessing
        such as data augmentation, masking, or format conversion.
        """
        return batch, {}

    def _ensure_functions_compiled(self):
        """Ensure training and evaluation functions are compiled.

        Notes
        -----
        This method triggers ahead-of-time compilation of the
        training and evaluation step functions to improve performance.
        """
        self.compile_aot()

    def _configure_generation_function(self):
        """Prepare a default generation function when the model supports generation."""
        if self.generate_function is not None:
            return
        state = getattr(self, "model_state", None)
        if state is None or not hasattr(state, "model"):
            return
        model = state.model
        if not hasattr(model, "generate"):
            return
        try:
            kwargs = self._default_generation_kwargs()
            config_overrides = self._default_generation_config_overrides()
            shard_inputs = getattr(self.arguments, "generation_shard_inputs", True)
            self.generate_function = self.create_generate_function(
                shard_inputs=shard_inputs,
                config_overrides=config_overrides,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - generation is optional
            log_debug_maybe(f"Skipping default generation function setup due to: {exc}")

    def _prepare_generation_config(
        self,
        generation_config: GenerationConfig | None,
    ) -> GenerationConfig | None:
        """Return a copy of the generation config to avoid mutating shared references."""
        if GenerationConfig is None:
            return generation_config
        if generation_config is not None:
            return copy.deepcopy(generation_config)
        model = self.model_state.model
        base_config = getattr(model, "generation_config", None)
        if base_config is None:
            return None
        return copy.deepcopy(base_config)

    def _default_generation_kwargs(self) -> dict[str, tp.Any]:
        """Assemble default keyword arguments for `model.generate` based on training arguments."""
        args = self.arguments
        if args is None:
            return {}

        def _maybe_insert(target: dict[str, tp.Any], key: str, value: tp.Any) -> None:
            if value is not None:
                target[key] = value

        kwargs: dict[str, tp.Any] = {}
        _maybe_insert(kwargs, "top_p", args.generation_top_p)
        _maybe_insert(kwargs, "top_k", args.generation_top_k)
        _maybe_insert(kwargs, "temperature", args.generation_temperature)
        _maybe_insert(kwargs, "do_sample", args.generation_do_sample)
        _maybe_insert(kwargs, "num_return_sequences", args.generation_num_return_sequences)
        _maybe_insert(kwargs, "max_new_tokens", args.generation_max_new_tokens)

        if "max_new_tokens" not in kwargs:
            fallback = args.max_completion_length
            if fallback is not None:
                kwargs["max_new_tokens"] = fallback

        if "num_return_sequences" not in kwargs:
            fallback = args.num_return_sequences
            if fallback is not None:
                kwargs["num_return_sequences"] = fallback

        if "do_sample" not in kwargs:
            if any(key in kwargs for key in ("top_p", "top_k", "temperature")):
                kwargs["do_sample"] = True
        if "eos_token_id" not in kwargs:
            proc = self._get_processing_class()
            eos_token_id = getattr(proc, "eos_token_id", None)
            if eos_token_id is not None:
                kwargs["eos_token_id"] = eos_token_id
            else:
                kwargs["eos_token_id"] = getattr(getattr(proc, "tokenizer", None), "eos_token_id", None)
        extra = args.generation_extra_kwargs
        if extra:
            kwargs.update(extra)

        return kwargs

    def _default_generation_config_overrides(self) -> dict[str, tp.Any] | None:
        overrides = self.arguments.generation_config_overrides
        if not overrides:
            return None
        return dict(overrides)

    def create_generate_function(
        self,
        generation_config: GenerationConfig | None = None,
        *,
        shard_inputs: bool = True,
        config_overrides: dict[str, tp.Any] | None = None,
        **generate_kwargs,
    ) -> tp.Callable[[EasyDeLState, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
        """
        Build and return a compiled generation function that mirrors the model's `generate`.

        Parameters
        ----------
        generation_config:
            Optional generation configuration. If omitted, the model's default configuration is used.
        shard_inputs:
            Whether to shard the prompt tensors using the model's partition manager before generation.
        config_overrides:
            Optional attribute overrides applied to the copied generation configuration.
        generate_kwargs:
            Extra keyword arguments forwarded to `module.generate`.
        """
        state = getattr(self, "model_state", None)
        if state is None:
            raise RuntimeError("Model state is not initialized; cannot create generation function.")
        if not hasattr(state.model, "generate"):
            raise NotImplementedError("Attached model does not provide a `generate` method.")

        effective_generate_kwargs = dict(generate_kwargs)
        config_copy = self._prepare_generation_config(generation_config)
        if config_copy is not None and config_overrides:
            for key, value in config_overrides.items():
                setattr(config_copy, key, value)
        if config_copy is not None:
            for key, value in effective_generate_kwargs.items():
                if hasattr(config_copy, key):
                    setattr(config_copy, key, value)

        mesh = self.model.mesh
        empty_sharding = jax.sharding.NamedSharding(spec=jax.sharding.PartitionSpec(), mesh=mesh)

        @ejit(
            in_shardings=(self.state_shardings, empty_sharding, empty_sharding),
            out_shardings=(empty_sharding, empty_sharding, empty_sharding),
        )
        def generate(state: EasyDeLState, input_ids: jax.Array, attention_mask: jax.Array):
            module = state.model
            with module.mesh:
                shard_input_ids = input_ids
                shard_attention_mask = attention_mask
                if shard_inputs:
                    partition_manager = getattr(module.config, "partition_manager", None)
                    if partition_manager is not None:
                        axes = [common_types.BATCH, common_types.SEQUENCE_PARALLEL]
                        shard_input_ids = partition_manager.shard(
                            shard_input_ids,
                            axes=axes,
                            mode=common_types.MODE_PREFILL,
                        )
                        shard_attention_mask = partition_manager.shard(
                            shard_attention_mask,
                            axes=axes,
                            mode=common_types.MODE_PREFILL,
                        )

                if config_copy is None:
                    outputs = module.generate(
                        input_ids=shard_input_ids,
                        attention_mask=shard_attention_mask,
                        **effective_generate_kwargs,
                    )
                else:
                    outputs = module.generate(
                        input_ids=shard_input_ids,
                        attention_mask=shard_attention_mask,
                        generation_config=config_copy,
                        **effective_generate_kwargs,
                    )

                sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
                return sequences, shard_input_ids, shard_attention_mask

        return generate

    def generate(
        self,
        input_ids: jax.Array | np.ndarray,
        attention_mask: jax.Array | np.ndarray | None = None,
        *,
        state: EasyDeLState | None = None,
        generation_config: GenerationConfig | None = None,
        shard_inputs: bool | None = None,
        config_overrides: dict[str, tp.Any] | None = None,
        return_metadata: bool = False,
        all_gather: bool = False,
        **generate_kwargs,
    ):
        """
        Convenience wrapper around the compiled generation function.

        Parameters
        ----------
        input_ids:
            Prompt token ids.
        attention_mask:
            Optional attention mask; defaults to all ones.
        state:
            Optional model state; defaults to the trainer's current state.
        generation_config:
            Optional configuration applied for this invocation only.
        shard_inputs:
            Whether to shard inputs before generation (default True).
        config_overrides:
            Attribute overrides applied to the copied generation configuration.
        return_metadata:
            When True, returns `(sequences, prompt_ids, attention_mask)`; otherwise only sequences.
        all_gather:
            Gather results back to host replicas when `_all_gather` is available.
        generate_kwargs:
            Additional keyword arguments forwarded to `module.generate`.
        """
        if state is None:
            state = self.model_state
        if state is None:
            raise RuntimeError("Model state is not initialized; call after trainer setup.")

        if shard_inputs is None:
            shard_inputs = getattr(self.arguments, "generation_shard_inputs", True)

        input_ids = jnp.asarray(input_ids)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype=jnp.int32)
        else:
            attention_mask = jnp.asarray(attention_mask)

        needs_custom_fn = bool(generation_config or config_overrides or generate_kwargs)
        if needs_custom_fn:
            merged_kwargs = self._default_generation_kwargs()
            merged_kwargs.update(generate_kwargs)
            merged_overrides = self._default_generation_config_overrides() or {}
            if config_overrides:
                merged_overrides.update(config_overrides)
            generate_fn = self.create_generate_function(
                generation_config=generation_config,
                shard_inputs=shard_inputs,
                config_overrides=merged_overrides or None,
                **merged_kwargs,
            )
        else:
            if self.generate_function is None:
                default_kwargs = self._default_generation_kwargs()
                default_overrides = self._default_generation_config_overrides()
                self.generate_function = self.create_generate_function(
                    shard_inputs=shard_inputs,
                    config_overrides=default_overrides,
                    **default_kwargs,
                )
            generate_fn = self.generate_function

        sequences, prompt_ids, prompt_mask = generate_fn(state, input_ids, attention_mask)

        if all_gather:
            sequences = self._all_gather(sequences)
            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)

        if return_metadata:
            return sequences, prompt_ids, prompt_mask
        return sequences

    def _get_processing_class(self):
        proc = getattr(self, "processing_class", None)
        if proc is not None:
            return proc
        proc = getattr(self, "tokenizer", None)
        if proc is not None:
            return proc
        state = getattr(self, "model_state", None)
        if state is not None:
            model = state.model
            candidate = getattr(model, "processing_class", None)
            if candidate is not None:
                return candidate
            candidate = getattr(model, "tokenizer", None)
            if candidate is not None:
                return candidate
        return None

    def _batch_decode_tokens(self, token_ids: tp.Any) -> list[str] | None:
        processor = self._get_processing_class()
        if processor is None or not hasattr(processor, "batch_decode"):
            return None
        array = np.asarray(token_ids)
        if array.ndim == 1:
            array = array[None, :]
        try:
            return processor.batch_decode(array, skip_special_tokens=True)
        except Exception as exc:  # pragma: no cover - best effort decoding
            log_debug_maybe(f"Failed to decode generation tokens: {exc}")
            return None

    def _collect_generation_prompts(self) -> list[tp.Any]:
        args = self.arguments
        if args is None:
            return []
        configured_prompts = list(args.generation_prompts)
        target = args.generation_num_prompts
        prompts = configured_prompts[: target or len(configured_prompts)]
        remaining = max(target - len(prompts), 0)
        if remaining > 0 and args.generation_use_train_prompts:
            prompts.extend(self._sample_prompts_from_dataset(remaining))
        return prompts

    def _sample_prompts_from_dataset(self, expected: int) -> list[tp.Any]:
        dataset = getattr(self, "dataset_train", None)
        if dataset is None or expected <= 0:
            return []
        try:
            dataset_len = len(dataset)
        except Exception as exc:  # pragma: no cover - some datasets are not sized
            log_debug_maybe(f"Cannot sample prompts from training dataset: {exc}")
            return []
        if dataset_len == 0:
            return []
        count = min(expected, dataset_len)
        indices = self._generation_rng.choice(dataset_len, size=count, replace=False)
        prompts: list[tp.Any] = []
        for idx in np.atleast_1d(indices):
            try:
                sample = dataset[int(idx)]
            except Exception as exc:  # pragma: no cover - ignore invalid rows
                log_debug_maybe(f"Failed to sample dataset prompt at index {idx}: {exc}")
                continue
            prompts.append(sample)
        return prompts

    def _prepare_generation_input(self, prompt: tp.Any) -> dict[str, tp.Any] | None:
        processor = self._get_processing_class()
        prompt_text: str | None = None
        encode_kwargs = dict(
            truncation=True,
            truncation_side="left",
            tokenize=True,
            padding="max_length",
            max_length=self.arguments.max_sequence_length,
            return_attention_mask=True,
            return_tensors="jax",
            return_dict=True,
            padding_side="left",
            add_generation_prompt=True,
        )
        if isinstance(prompt, dict):
            if "input_ids" in prompt:
                input_ids = prompt["input_ids"]
                attention = prompt.get("attention_mask")
                prompt_text = prompt.get("prompt") or prompt.get("text")
            else:
                field = getattr(self.arguments, "generation_dataset_prompt_field", None)
                if field and field in prompt:
                    return self._prepare_generation_input(prompt[field])
                for key in ("prompt", "text"):
                    if key in prompt:
                        return self._prepare_generation_input(prompt[key])
                log_debug_maybe("Dataset sample missing `input_ids`/`prompt` keys for preview generation; skipping")
                return None
        elif isinstance(prompt, str):
            if processor is None:
                logger.warn("No tokenizer/processor available; cannot tokenize prompt text.")
                return None
            prompt_text = prompt

            try:
                processor.padding_side = "left"
                encoded = processor.apply_chat_template([{"role": "user", "content": prompt}], **encode_kwargs)
            except Exception as exc:  # pragma: no cover - tokenizer issues
                log_debug_maybe(f"Failed to tokenize generation prompt: {exc}")
                return None
            input_ids = encoded["input_ids"]
            attention = encoded.get("attention_mask")
        else:
            log_debug_maybe(f"Unsupported prompt type for preview generation: {type(prompt)}")
            return None

        input_array = np.asarray(input_ids)
        input_array = input_array.astype(np.int32, copy=False)
        if input_array.ndim == 1:
            input_array = input_array[None, :]
        input_ids = jnp.asarray(input_array, dtype=jnp.int32)

        if attention is None:
            attention_array = np.ones_like(input_array, dtype=np.int32)
        else:
            attention_array = np.asarray(attention)
            if attention_array.ndim == 1:
                attention_array = attention_array[None, :]
            attention_array = attention_array.astype(np.int32, copy=False)
        attention_mask = jnp.asarray(attention_array, dtype=jnp.int32)

        if prompt_text is None:
            decoded = self._batch_decode_tokens(input_ids)
            if decoded:
                prompt_text = decoded[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_text": prompt_text,
        }

    def maybe_generate(
        self,
        state: EasyDeLState,
        step: int,
        metrics: MetricsType | None = None,
    ) -> None:
        """Optionally run preview generation to monitor training progress."""

        args = self.arguments
        if args is None:
            return

        interval = args.generation_interval

        if interval is None:
            return

        if interval <= 0 or step % interval != 0:
            return

        prompts = self._collect_generation_prompts()
        if not prompts:
            return

        results: list[dict[str, tp.Any]] = []
        for prompt in prompts:
            prepared = self._prepare_generation_input(prompt)
            if prepared is None:
                continue
            input_ids = prepared["input_ids"]
            attention_mask = prepared["attention_mask"]
            prompt_text = prepared.get("prompt_text")

            try:
                sequences = self.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    state=state,
                    shard_inputs=args.generation_shard_inputs,
                )
            except Exception as exc:  # pragma: no cover - preview should not break training
                log_debug_maybe(f"Preview generation failed: {exc}")
                continue

            sequences = np.asarray(jax.device_get(sequences))
            if sequences.ndim == 1:
                sequences = sequences[None, :]

            if attention_mask is not None:
                attn_np = np.asarray(jax.device_get(attention_mask))
                if attn_np.ndim == 1:
                    attn_np = attn_np[None, :]
                prompt_len = int(attn_np[0].sum())
            else:
                prompt_len = input_ids.shape[-1]

            completion_tokens = sequences[:, prompt_len:]
            completions = self._batch_decode_tokens(completion_tokens)
            if completions is None:
                completions = [completion_tokens[i].tolist() for i in range(completion_tokens.shape[0])]

            if prompt_text is None:
                decoded_prompt = self._batch_decode_tokens(input_ids)
                if decoded_prompt:
                    prompt_text = decoded_prompt[0]

            results.append({"prompt": prompt_text, "completions": completions, "step": step})

        if not results:
            return

        self.latest_generation_samples = results

        for record in results:
            prompt_repr = record["prompt"] if record["prompt"] is not None else "<prompt tokens>"

        if wandb is not None and args.use_wandb and args.can_log_metrics and args.generation_log_to_wandb:
            table = wandb.Table(columns=["step", "prompt", "completion_id", "completion"])
            for record in results:
                prompt_repr = record["prompt"] if record["prompt"] is not None else "<prompt tokens>"
                for idx, completion in enumerate(record["completions"]):
                    table.add_data(step, prompt_repr, idx, completion)
            wandb.log({"preview_generations": table}, step=step)
        if args.generation_preview_print:
            logger.info(f"[preview step {step}] prompt: {prompt_repr}")
            for idx, completion in enumerate(record["completions"]):
                logger.info(f"  completion[{idx}]: {completion}")

    def _one_to_all(self, arr: jax.Array) -> jax.Array:
        """Distribute array from one device to all devices.

        Args:
            arr: Array to distribute.

        Returns:
            Array distributed across devices.
        """
        with self.mesh:
            arr = with_sharding_constraint(arr, PartitionSpec(None))
        return arr

    def _all_gather(self, arr: jax.Array) -> jax.Array:
        """Gather array from all devices to a single replicated array.

        Args:
            arr: Array to gather across devices.

        Returns:
            Array replicated across all devices.
        """
        return jax.device_put(arr, NamedSharding(self.model.mesh, PartitionSpec()))

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
        """Initialize Weights & Biases logging if enabled.

        Notes
        -----
        This method sets up the Weights & Biases runtime for experiment
        tracking and logging when use_wandb is True in arguments.
        """
        if self.arguments.use_wandb:
            self.wandb_runtime = self.arguments.get_wandb_init()

    def _initialize_timer(self):
        """Initialize the timer for performance monitoring.

        Notes
        -----
        Sets up a timer instance for tracking training and evaluation
        performance metrics, with optional TensorBoard integration.
        """
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
            self.checkpointer = self._create_checkpointer()
        self.timer.log("configure functions and sharding them")
        self._configure_generation_function()

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
        """Configure Grain dataloaders for training and evaluation.

        Returns:
            tuple: A tuple containing:
                - dataloader_train: Grain DataLoader for training
                - max_training_steps: Maximum number of training steps
                - dataloader_eval: Grain DataLoader for evaluation (or None)
                - max_evaluation_steps: Maximum evaluation steps (or None)

        Notes:
            Grain is Google's data loading library optimized for JAX.
            Handles dataset sharding, batching, and preprocessing.
        """

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
        """Configure TensorFlow Dataset dataloaders for training and evaluation.

        Returns:
            tuple: A tuple containing:
                - dataloader_train: TensorFlow Dataset for training
                - max_training_steps: Maximum number of training steps
                - dataloader_eval: TensorFlow Dataset for evaluation (or None)
                - max_evaluation_steps: Maximum evaluation steps (or None)

        Raises:
            ImportError: If tensorflow is not installed.

        Notes:
            Uses TensorFlow's tf.data API for data loading.
            Supports automatic sharding across devices.
        """
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

    def _save_state(self, state: EasyDeLState, save_directory: str | None = None, *args, **kwargs) -> str:
        """
        Save the current model state to a checkpoint.

        This method handles the complete checkpoint saving process including:
        - Creating checkpoint directories with proper naming
        - Saving training arguments alongside the model
        - Generating README documentation for the checkpoint
        - Saving the actual model state with optional optimizer state

        Args:
            state: The model state to save
            save_directory: Optional override for save directory. If None, uses
                default directory based on current step.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            str: Path to the saved checkpoint directory
        """
        if save_directory is None:
            step = self._get_current_step(state)
            directory_name = self.arguments._get_save_directory_milestone(step=step, create=True)
        else:
            directory_name = ePath(save_directory)

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

    def _create_checkpointer(self):
        """Create and configure the Checkpointer instance.

        Returns:
            Checkpointer: Configured checkpointer with policies from training arguments.

        Note:
            - Uses save_steps from TrainingArguments to create checkpoint policies
            - No time-based saving by default (can be extended via save_interval_timedelta)
            - Checkpoints are saved to the base output directory
        """
        return self.arguments.get_streaming_checkpointer()

    def _cleanup_old_checkpoints(self):
        """Clean up old permanent checkpoints based on save_total_limit.

        Only affects permanent checkpoints (run-based saves). Temporary checkpoints
        are managed automatically by Checkpointer.

        Uses Checkpointer's async deletion queue for non-blocking cleanup.

        Note:
            - Reads metadata.json to identify permanent vs temporary checkpoints
            - Keeps the N most recent permanent checkpoints (N = save_total_limit)
            - Sorts by modification time to determine which are oldest
            - Queues deletion asynchronously (non-blocking)
        """
        if self.arguments.save_total_limit is None:
            return

        from eformer.serialization.checkpointer import _read_checkpoint_metadata

        save_dir = ePath(self.arguments._get_save_directory())
        if not save_dir.exists():
            return

        # Find all checkpoint directories with metadata
        checkpoint_dirs = []
        for path in save_dir.glob("run-*"):
            if path.is_dir() and (path / "metadata.json").exists():
                try:
                    metadata = _read_checkpoint_metadata(str(path))
                    # Only consider permanent checkpoints
                    if not metadata.get("is_temporary", False):
                        checkpoint_dirs.append(path)
                except Exception:
                    continue

        if len(checkpoint_dirs) <= self.arguments.save_total_limit:
            return

        # Sort by modification time (oldest first)
        def get_mtime(path):
            try:
                return path.stat().get("mtime", 0)
            except Exception:
                return 0

        checkpoint_dirs.sort(key=get_mtime)

        # Queue oldest checkpoints for async deletion
        to_delete = checkpoint_dirs[: -self.arguments.save_total_limit]
        for old_checkpoint in to_delete:
            logger.info(f"Queueing old permanent checkpoint for deletion: {old_checkpoint}")
            self.checkpointer._queue_checkpoint_removal(str(old_checkpoint))

    def _get_current_step(self, state):
        """Get the current training step from state.

        Args:
            state: The model state containing the step counter.

        Returns:
            int: Current step number, adjusted by step_start_point if set.
        """
        step = int(jax.device_get(state.step))
        if self.arguments.step_start_point is not None:
            step += self.arguments.step_start_point
        return step

    def _save_readme(self, save_directory):
        """Save training information as README.md in checkpoint directory.

        Args:
            save_directory: Directory where README.md will be saved.
        """
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
        """Convert partition specs to named sharding.

        Args:
            tree: PyTree structure with partition specs.
            mesh: Device mesh for sharding (uses trainer's mesh if None).

        Returns:
            PyTree with named sharding specifications.
        """
        mesh = mesh or self.mesh or self.model.mesh
        return specs_to_name_sharding(tree, mesh)

    def calculate_number_total_flops(self, params, is_training=True):
        """Calculate total FLOPs for the model.

        Args:
            params: Model parameters.
            is_training: Whether calculating for training (includes backward pass).

        Returns:
            int: Total FLOPs count.
        """
        return 6 * sum(x.size for x in jax.tree_util.tree_flatten(params)[0])

    @staticmethod
    def count_model_parameters(prm):
        """Count total number of model parameters.

        Args:
            prm: Model parameters (can be frozen or unfrozen).

        Returns:
            int: Total number of parameters.
        """
        return sum(n.size for n in jax.tree_util.tree_flatten(prm)[0])

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
        """Hook called at the start of training.

        Notes:
            Sets up static metrics and records training start time.
        """
        self._setup_static_metrics()
        self._training_time_start = time.time()

    def start_evaluation_hook(self):
        """Hook called at the start of evaluation.

        Notes:
            Sets up static metrics and records evaluation start time.
        """
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
            # Use checkpointer.on_step with force=True for final save
            current_step = int(jax.device_get(state.step))

            # Track the saved directory in the callback
            saved_directory = [None]  # Use list for mutability in closure

            def save_callback(dest, mesh, meta, s=state):
                full_path = str(self.arguments._get_save_directory() / dest)
                saved_directory[0] = self._save_state(state=s, save_directory=full_path)
                # Clean up old permanent checkpoints if save_total_limit is set
                self._cleanup_old_checkpoints()

            self.checkpointer.on_step(
                mesh=self.mesh,
                pytree=None,
                step=current_step,
                force=True,  # Force final checkpoint save
                true_callbacks=[save_callback],
            )

            if saved_directory[0] is not None:
                filename = saved_directory[0]
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
        """Setup initial metrics logging.

        Args:
            state: The model state for extracting parameter count.

        Notes:
            Logs model size, parameter count, and training configuration.
        """
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
        """Get next batch from iterator, reinitializing if needed.

        Args:
            data_iter: Current data iterator.
            dataloader: The dataloader to reinitialize from if needed.

        Returns:
            tuple: (batch, updated_data_iter) where batch is the next data batch
                  and updated_data_iter is the potentially reinitialized iterator.

        Raises:
            StopIteration: If dataloader is exhausted and cannot be reinitialized.
        """
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
        """Log weight distribution statistics.

        Args:
            state: Model state containing parameters.
            step: Current training step.

        Notes:
            Delegates to arguments.log_weight_distribution method.
        """
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
