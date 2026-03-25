# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import collections.abc
import copy
import gc
import itertools
import json
import operator
import os
import pprint
import time
import typing as tp
from abc import abstractmethod
from functools import cached_property
from typing import NamedTuple

import contextlib2
import flax
import flax.nnx
import grain.python as grain  # pyright: ignore[reportMissingTypeStubs]
import jax
import jax.extend
import numpy as np
from eformer import common_types
from eformer.escale import with_sharding_constraint
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from jax import numpy as jnp
from jax._src.stages import Compiled
from jax.sharding import NamedSharding, PartitionSpec
from tqdm.autonotebook import tqdm
from transformers import GenerationConfig, ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from easydel import __version__

# Import ShardedDataSource for trainer integration
from easydel.data.core.protocols import ShardedDataSource
from easydel.data.sources.hf_wrapper import wrap_hf_dataset
from easydel.inference import SamplingParams
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.elarge.benchmarking import (
    flatten_benchmark_metrics,
    normalize_benchmark_configs,
    run_lm_eval_with_esurge,
)
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLPreemptionSignal, EasyDeLTimerError
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
from .training_utils import (
    GENERATION_MODEL_INPUT_KEYS,
    compact_generation_model_kwargs,
    filter_kwargs_for_callable,
    normalize_generation_model_kwargs,
    prepare_generation_model_kwargs_for_call,
    resolve_total_steps,
)
from .utils import CollateMapTransform, HFDataSource, ToNumpy

try:
    import wandb
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.transforms.base import Transform
    from easydel.inference.esurge.esurge_engine import RequestOutput

logger = get_logger(__name__)

log_debug_maybe = logger.debug

DEFAULT_ARGS_JSON_NAME = "easydel-training-arguments.json"


class GenerationResults(NamedTuple):
    """Results from unified generation containing both text and token representations.

    Attributes:
        generation_results: The generation results from engine
        prompt_ids: Token IDs for the prompt (batch_size, max_seq_len) - left-padded
        prompt_mask: Attention mask for the prompt (batch_size, max_seq_len)
        sequences: Complete generated sequences including prompt (batch_size, max_seq_len + max_new_tokens)
        completion_ids: Token IDs for only the generated completions (batch_size, max_new_tokens) - right-padded
        completion_mask: Attention mask for completions (batch_size, max_new_tokens)
        text: Parsed visible completion text aligned with generated completions.
        reasoning: Parsed reasoning text aligned with generated completions.
        tool_calls: Parsed tool call payloads aligned with generated completions.
        raw_text: Raw unsplit completion text aligned with generated completions,
            before reasoning/tool separation and without special-token stripping.
        completion_prompts: Optional prompt objects (text or chat dicts) aligned one-to-one with completions.
    """

    generation_results: str | list[str]
    prompt_ids: jax.Array
    prompt_mask: jax.Array
    sequences: jax.Array
    completion_ids: jax.Array
    completion_mask: jax.Array
    decoded_prompts: str | list[str]
    completion_prompts: list[str | list[dict[str, str]]] | None = None
    text: str | list[str] | None = None
    reasoning: list[str | None] | None = None
    tool_calls: list[list | None] | None = None
    raw_text: str | list[str] | None = None


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
        _train_source: Internal ShardedDataSource for training data
        _eval_source: Internal ShardedDataSource for evaluation data
    """

    # Type annotations for internal data sources
    arguments: TrainingArguments | None
    model_state: EasyDeLState | None
    _train_source: ShardedDataSource | None
    _eval_source: ShardedDataSource | None
    _RUNTIME_MODEL_OVERRIDE_STATE_ATTRS: tp.ClassVar[frozenset[str]] = frozenset(
        {
            "model_state",
            "reference_state",
            "ref_state",
            "teacher_state",
        }
    )

    def __setattr__(self, name: str, value: tp.Any) -> None:
        """Set an attribute, applying runtime model config overrides for state attributes.

        Intercepts assignments to model/reference/teacher state attributes and
        automatically applies any runtime configuration overrides from the
        training arguments.

        Args:
            name: The attribute name being set.
            value: The value to assign.
        """
        object.__setattr__(self, name, value)
        if name not in self._RUNTIME_MODEL_OVERRIDE_STATE_ATTRS or value is None:
            return

        arguments = self.__dict__.get("arguments")
        if arguments is None:
            return

        self._apply_runtime_model_config_overrides_to_state(value, arguments)

    def __init__(
        self,
        arguments: TrainingArguments | None = None,
        model_state: EasyDeLState | None = None,
        model: tp.type[EasyDeLBaseModule] | None = None,
        dataset_train: Dataset | IterableDataset | ShardedDataSource | None = None,
        dataset_eval: Dataset | IterableDataset | ShardedDataSource | None = None,
        data_collator: tp.Callable | None = None,
        finetune: bool = True,
        processing_class: PreTrainedTokenizerBase | None = None,
        **deprecated_kwargs,
    ):
        """
        Initialize the BaseTrainer.

        Args:
            arguments: Training configuration and hyperparameters
            model_state: Pre-initialized model state (mutually exclusive with model)
            model: Model class to initialize (mutually exclusive with model_state)
            dataset_train: Training dataset (HF Dataset, IterableDataset, or ShardedDataSource)
            dataset_eval: Evaluation dataset (HF Dataset, IterableDataset, or ShardedDataSource)
            data_collator: Function to collate batches of data
            finetune: Whether this is a fine-tuning run (affects initialization)
            processing_class: Tokenizer or processor for handling text encoding/decoding
            **deprecated_kwargs: Deprecated keyword arguments for backward compatibility

        Raises:
            ValueError: If both model and model_state are provided, or if neither is provided
            ValueError: If arguments is None
        """
        if arguments is None:
            raise ValueError("training argument must be passed to Trainers.")
        if model_state is not None and model is not None:
            raise ValueError("Either model or model_state should be passed, not both.")
        elif model_state is None and model is None:
            raise ValueError("Either model or model_state should be passed.")
        elif model_state is None:
            model_state = model.to_state()
        if arguments.model_name is None:
            arguments.model_name = getattr(model_state.model, "_model_type", "module")
        self.arguments = arguments
        self._preemption_checkpoint_path = None
        self._tpu_preemption_sync_available = None

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

                    model_state = resumed_state
                    self._resumed_from_checkpoint = True
                    self._maybe_remove_loaded_checkpoint(checkpoint_path)
                else:
                    logger.info("No checkpoints found. Starting fresh training.")
            except Exception as e:
                logger.warning(f"Resuming from checkpoint failed: {e}. Starting fresh training.")

        self.model_state = model_state
        self._apply_runtime_model_config_overrides()
        self._apply_step_start_point()
        self._model = flax.nnx.eval_shape(lambda: self.model_state.model)
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval
        self.data_collator = data_collator
        self.finetune = finetune
        self.processing_class = processing_class

        if self.data_collator is None and getattr(self.arguments, "use_data_collator", True):
            base_collator = self.create_collect_function(
                max_sequence_length=self.arguments.max_length,
                truncation_mode=self.arguments.truncation_mode,
            )

            def _stack_per_example_outputs(per_example: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
                """Stack per-example collator outputs into batched arrays.

                Concatenates or stacks JAX arrays from individually collated
                examples into a single batched dictionary. Arrays with a leading
                dimension of 1 are concatenated; others are stacked.

                Args:
                    per_example: List of dicts, each from collating one example.

                Returns:
                    A dict mapping keys to batched JAX arrays, or empty dict on failure.
                """
                if not per_example or not isinstance(per_example[0], dict):
                    return {}
                stacked: dict[str, tp.Any] = {}
                for key in per_example[0].keys():
                    values = [item.get(key) for item in per_example]
                    if any(v is None for v in values):
                        continue
                    if key == "tools":
                        stacked[key] = values
                        continue
                    try:
                        arrays = [jnp.asarray(v) for v in values]
                    except Exception:
                        continue
                    first = arrays[0]
                    if (
                        getattr(first, "ndim", 0) >= 1
                        and first.shape[0] == 1
                        and all(getattr(a, "shape", None) == first.shape for a in arrays)
                    ):
                        stacked[key] = jnp.concatenate(arrays, axis=0)
                    else:
                        stacked[key] = jnp.stack(arrays, axis=0)
                return stacked

            def _auto_data_collator(batch):
                """Automatically collate a batch using the base collator.

                Tries the base collator on the full batch first. If that fails
                (e.g. for datasets that yield non-standard formats), falls back
                to collating each example individually and stacking results.

                Args:
                    batch: A list/tuple of examples or an already-collated batch.

                Returns:
                    The collated batch as a dict of arrays, or the original input
                    if collation is not applicable.
                """
                if not isinstance(batch, (list, tuple)):
                    return batch
                batch_list = list(batch)
                if not batch_list:
                    return batch_list

                try:
                    return base_collator(batch_list)
                except (TypeError, AttributeError):
                    per_example = [base_collator(example) for example in batch_list]
                    stacked = _stack_per_example_outputs(per_example)
                    return stacked if stacked else batch_list

            self.data_collator = _auto_data_collator

        # Convert datasets to ShardedDataSource for unified internal handling
        self._train_source = self._to_sharded_source(dataset_train)
        self._eval_source = self._to_sharded_source(dataset_eval)

        # Apply trainer-specific preprocessing transform if available
        self._apply_preprocess_transforms()

        self._initialize_attributes()
        self.initialize_trainer_utils()

    @staticmethod
    def _apply_runtime_model_config_overrides_to_state(
        state: EasyDeLState | None,
        arguments: TrainingArguments,
    ) -> None:
        """Propagate training argument overrides onto a model state's config.

        Certain training arguments (e.g. ``lmhead_chunksize``) need to be
        reflected in the model configuration attached to a given state so
        that the model forward pass picks them up at runtime.  This static
        method writes those values into ``state.model.config`` in-place.

        Args:
            state: The model state whose config should be updated.  If
                ``None``, the call is a no-op.
            arguments: The training arguments containing the override values.
                Currently inspects ``arguments.lmhead_chunksize``.

        Returns:
            None.  The state's config is mutated in-place.
        """
        if state is None:
            return

        lmhead_chunksize = getattr(arguments, "lmhead_chunksize", None)
        if lmhead_chunksize is None:
            return

        model = getattr(state, "model", None)
        config = getattr(model, "config", None)
        if config is None:
            return

        config.lmhead_chunksize = int(lmhead_chunksize)

    def _apply_runtime_model_config_overrides(self) -> None:
        """Apply runtime model config overrides to all tracked model states.

        Iterates over the primary ``model_state`` as well as any auxiliary
        states that may exist on the trainer (``reference_state``,
        ``ref_state``, ``teacher_state``) and delegates to
        :meth:`_apply_runtime_model_config_overrides_to_state` for each.
        This ensures that training-argument-driven config values such as
        ``lmhead_chunksize`` are consistently propagated to every model
        that participates in the training loop.
        """
        self._apply_runtime_model_config_overrides_to_state(self.model_state, self.arguments)
        for attr_name in self._RUNTIME_MODEL_OVERRIDE_STATE_ATTRS - {"model_state"}:
            self._apply_runtime_model_config_overrides_to_state(getattr(self, attr_name, None), self.arguments)

    def _apply_step_start_point(self) -> None:
        """Initialize a fresh training state from ``step_start_point`` when requested."""
        requested_step_value = self.arguments.step_start_point
        if self.model_state is None or requested_step_value is None:
            return

        requested_step = int(requested_step_value)
        current_step = int(jax.device_get(self.model_state.step))
        step_dtype = getattr(self.model_state.step, "dtype", jnp.int32)
        normalized_step = jnp.asarray(requested_step, dtype=step_dtype)
        if current_step == requested_step:
            if not isinstance(self.model_state.step, jax.Array):
                self.model_state = self.model_state.replace(step=normalized_step)
            return
        if current_step != 0 and not self._resumed_from_checkpoint:
            logger.warning(
                f"Ignoring step_start_point={requested_step} because model_state.step is already "
                f"{current_step}. Use a fresh state or checkpoint resume instead."
            )
            return

        self.model_state = self.model_state.replace(step=normalized_step)
        if self._resumed_from_checkpoint:
            logger.info(
                f"Overrode resumed checkpoint step from {current_step} to {requested_step} via step_start_point."
            )
        else:
            logger.info(f"Initialized model_state.step to {requested_step} from step_start_point.")

    def _apply_step_start_point_to_optimizer_state(self) -> None:
        """Align optimizer and scheduler counters with ``step_start_point``."""
        requested_step_value = getattr(self.arguments, "step_start_point", None)
        if (
            self.model_state is None
            or requested_step_value is None
            or getattr(self.model_state, "opt_state", None) is None
        ):
            return

        requested_step = int(requested_step_value)
        if int(jax.device_get(self.model_state.step)) != requested_step:
            return

        updated = {"changed": False}

        def _seed_count(path, leaf):
            """Replace optimizer 'count' leaves with the requested step value.

            Used as a ``tree_map_with_path`` function to walk the optimizer
            state tree and set any leaf named ``count`` to the target step,
            aligning scheduler/optimizer counters with ``step_start_point``.

            Args:
                path: The pytree path to the current leaf.
                leaf: The current leaf value in the optimizer state tree.

            Returns:
                The leaf replaced with ``requested_step`` if it is a count
                leaf, otherwise the original leaf unchanged.
            """
            if not path:
                return leaf
            key = path[-1]
            key_name = getattr(key, "name", None)
            key_name = key_name if key_name is not None else getattr(key, "key", None)
            if key_name != "count":
                return leaf
            updated["changed"] = True
            leaf_dtype = getattr(leaf, "dtype", getattr(self.model_state.step, "dtype", jnp.int32))
            return jnp.asarray(requested_step, dtype=leaf_dtype)

        opt_state = jax.tree_util.tree_map_with_path(_seed_count, self.model_state.opt_state)
        if updated["changed"]:
            self.model_state = self.model_state.replace(opt_state=opt_state)
            logger.info(f"Aligned optimizer/scheduler counters to step_start_point={requested_step}.")

    @staticmethod
    def _is_memory_oom_exception(exc: BaseException) -> bool:
        """Determine whether an exception represents an out-of-memory error.

        The method converts the exception's type name and message to
        lowercase and checks for the presence of known OOM marker strings
        that are emitted by JAX, XLA, CUDA, and cuDNN runtimes.

        Args:
            exc: The exception instance to inspect.

        Returns:
            ``True`` if any recognised OOM marker is found in the
            exception's string representation, ``False`` otherwise.
        """
        message = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "resource_exhausted",
            "compiletimehbmoom",
            "compile time hbm oom",
            "memory space hbm",
            "exceeded hbm capacity",
            "ran out of memory",
            "out of memory",
            "cuda out of memory",
            "cudnn_status_alloc_failed",
        )
        has_oom_marker = any(marker in message for marker in markers)
        jax_runtime_error = getattr(jax.errors, "JaxRuntimeError", None)
        if jax_runtime_error is not None and isinstance(exc, jax_runtime_error):
            return has_oom_marker
        return has_oom_marker

    def _memory_optimization_trainer_name(self) -> str:
        """Return a human-readable trainer name for memory optimization messages.

        If a ``trainer_prefix`` is set in the training arguments, it is
        used as-is (after stripping whitespace).  Otherwise, the concrete
        class name of the trainer instance is returned.

        Returns:
            A string identifying the trainer, suitable for inclusion in
            user-facing hint messages.
        """
        prefix = self.arguments.trainer_prefix
        if isinstance(prefix, str) and prefix.strip():
            return prefix.strip()
        return type(self).__name__

    def _memory_optimization_hints(self) -> list[str]:
        """Inspect training arguments and build a list of memory optimization hints.

        Each hint is a human-readable string describing a chunking or
        capping parameter in :class:`TrainingArguments` that, if adjusted,
        could reduce peak memory usage.  The method examines parameters
        such as ``logprob_vocab_chunk_size``, ``lmhead_chunksize``,
        ``logits_chunk_size``, ``ref_logps_chunk_size``,
        ``completion_chunk_size``, and ``max_loss_completion_tokens``.
        For each parameter that is currently set, a suggestion to lower
        it is produced; for each parameter that is disabled, a suggestion
        to enable it is produced.

        Returns:
            A list of hint strings.  An empty list is returned when no
            actionable suggestions can be made (e.g. ``arguments`` is
            ``None``).
        """
        args = self.arguments
        if args is None:
            return []

        hints: list[str] = []

        def _add(text: str) -> None:
            """Append a hint string if it is not already present in the list."""
            if text not in hints:
                hints.append(text)

        def _int_attr(name: str) -> int | None:
            """Get an integer attribute from the training arguments by name.

            Args:
                name: The attribute name to look up on ``args``.

            Returns:
                The attribute value cast to int, or None if not set.
            """
            value = getattr(args, name, None)
            if value is None:
                return None
            return int(value)

        logprob_chunk_raw = getattr(args, "logprob_vocab_chunk_size", None)
        if hasattr(args, "logprob_vocab_chunk_size"):
            if logprob_chunk_raw is not None and int(logprob_chunk_raw) > 0:
                logprob_vocab_chunk_size = int(logprob_chunk_raw)
                lowered = max(logprob_vocab_chunk_size // 2, 1)
                _add(
                    "`logprob_vocab_chunk_size` "
                    f"(current: {logprob_vocab_chunk_size}): lower it to `{lowered}` or `512` "
                    "to reduce peak memory during vocab-side log-prob and entropy computation."
                )
            else:
                _add(
                    "`logprob_vocab_chunk_size` (current: disabled): enable it with a value like "
                    "`1024` or `2048` to chunk vocab-side log-prob and entropy computation."
                )

        lmhead_chunksize = getattr(args, "lmhead_chunksize", None)
        if lmhead_chunksize is not None:
            lowered = max(int(lmhead_chunksize) // 2, 1)
            _add(
                "`lmhead_chunksize` "
                f"(current: {lmhead_chunksize}): lower it to `{lowered}` or `512` "
                "to chunk the LM-head projection over the sequence dimension."
            )
        elif hasattr(args, "lmhead_chunksize"):
            _add(
                "`lmhead_chunksize` (current: disabled): enable it with a value like `2048` or `4096` "
                "to chunk the LM-head projection over the sequence dimension."
            )

        if hasattr(args, "logits_chunk_size"):
            logits_chunk_size = _int_attr("logits_chunk_size")
            if logits_chunk_size is not None and logits_chunk_size > 0:
                lowered = max(logits_chunk_size // 2, 1)
                _add(
                    "`logits_chunk_size` "
                    f"(current: {logits_chunk_size}): lower it to `{lowered}` or `2048` "
                    "to compute distillation KL loss over smaller token chunks."
                )
            else:
                _add(
                    "`logits_chunk_size` (current: disabled): enable it with a value like `2048` or `4096` "
                    "to avoid materializing full `[batch, seq, vocab]` distillation logits."
                )

        if hasattr(args, "ref_logps_chunk_size"):
            ref_logps_chunk_size = _int_attr("ref_logps_chunk_size")
            if ref_logps_chunk_size is not None and ref_logps_chunk_size > 0:
                lowered = max(ref_logps_chunk_size // 2, 1)
                _add(
                    "`ref_logps_chunk_size` "
                    f"(current: {ref_logps_chunk_size}): lower it to `{lowered}` to chunk the reference-model "
                    "log-prob pass over smaller batches."
                )
            else:
                _add(
                    "`ref_logps_chunk_size` (current: disabled): enable it with a value like `2` or `4` "
                    "to chunk reference-model log-prob computation."
                )

        if hasattr(args, "completion_chunk_size"):
            completion_chunk_size = _int_attr("completion_chunk_size")
            if completion_chunk_size is not None and completion_chunk_size > 0:
                lowered = max(completion_chunk_size // 2, 1)
                _add(
                    "`completion_chunk_size` "
                    f"(current: {completion_chunk_size}): lower it to `{lowered}` to process completion-loss "
                    "batches in smaller chunks."
                )
            else:
                _add(
                    "`completion_chunk_size` (current: disabled): enable it with a value like `2` or `4` "
                    "to chunk completion-loss computation."
                )

        if hasattr(args, "max_loss_completion_tokens"):
            max_loss_completion_tokens = _int_attr("max_loss_completion_tokens")
            if max_loss_completion_tokens is not None and max_loss_completion_tokens > 0:
                lowered = max(max_loss_completion_tokens // 2, 1)
                _add(
                    "`max_loss_completion_tokens` "
                    f"(current: {max_loss_completion_tokens}): lower it to `{lowered}` "
                    "to cap the completion tokens that participate in the loss."
                )
            else:
                _add(
                    "`max_loss_completion_tokens` (current: disabled): set it to a cap like `2048` or `4096` "
                    "to truncate the loss-bearing completion window."
                )

        total_batch_size = _int_attr("total_batch_size")
        if total_batch_size is not None:
            lowered = max(total_batch_size // 2, 1)
            _add(
                "`total_batch_size` "
                f"(current: {total_batch_size}): lower it to `{lowered}` "
                "to shrink per-step activation and temporary-buffer memory."
            )

        gradient_accumulation_steps = _int_attr("gradient_accumulation_steps")
        if gradient_accumulation_steps is not None:
            next_steps = max(gradient_accumulation_steps + 1, 2)
            _add(
                "`gradient_accumulation_steps` "
                f"(current: {gradient_accumulation_steps}): raise it to `{next_steps}` or higher after lowering "
                "`total_batch_size` if you need to preserve the effective batch size."
            )

        prompt_length = _int_attr("max_prompt_length")
        if prompt_length is not None:
            lowered = max(prompt_length // 2, 1)
            _add(
                "`max_prompt_length` "
                f"(current: {prompt_length}): lower it to `{lowered}` to reduce prompt-side attention and activation memory."
            )

        completion_length = _int_attr("max_completion_length")
        if completion_length is not None:
            lowered = max(completion_length // 2, 1)
            _add(
                "`max_completion_length` "
                f"(current: {completion_length}): lower it to `{lowered}` to reduce completion-side attention, logits, and loss memory."
            )

        max_length = _int_attr("max_length")
        if max_length is not None and prompt_length is None and completion_length is None:
            lowered = max(max_length // 2, 1)
            _add(
                "`max_length` "
                f"(current: {max_length}): lower it to `{lowered}` to cut sequence length and attention memory."
            )

        max_new_tokens = _int_attr("max_new_tokens")
        if max_new_tokens is not None:
            lowered = max(max_new_tokens // 2, 1)
            _add(
                f"`max_new_tokens` (current: {max_new_tokens}): lower it to `{lowered}` to shorten generated sequences."
            )

        num_return_sequences = _int_attr("num_return_sequences")
        if num_return_sequences is not None and num_return_sequences > 1:
            lowered = max(num_return_sequences // 2, 1)
            _add(
                "`num_return_sequences` "
                f"(current: {num_return_sequences}): lower it to `{lowered}` or `1` "
                "to generate fewer completions per prompt."
            )

        num_generations_per_prompt = _int_attr("num_generations_per_prompt")
        if num_generations_per_prompt is not None and num_generations_per_prompt > 1:
            lowered = max(num_generations_per_prompt // 2, 1)
            _add(
                "`num_generations_per_prompt` "
                f"(current: {num_generations_per_prompt}): lower it to `{lowered}` or `1` "
                "to generate fewer completions per prompt."
            )

        return hints

    def _format_memory_optimization_hints(self) -> str | None:
        """Format memory optimization hints into a single multiline message.

        Calls :meth:`_memory_optimization_hints` and, if any hints are
        available, assembles them into a bulleted list preceded by a
        header that includes the trainer name.

        Returns:
            A formatted string ready for display to the user, or ``None``
            if there are no hints to show.
        """
        hints = self._memory_optimization_hints()
        if not hints:
            return None
        lines = [f"Memory optimization techniques available for trainer `{self._memory_optimization_trainer_name()}`:"]
        lines.extend(f"- {hint}" for hint in hints)
        return "\n".join(lines)

    def _augment_memory_oom_exception(self, exc: BaseException) -> RuntimeError:
        """Wrap an OOM exception with actionable memory optimization hints.

        Creates a new :class:`jax.errors.JaxRuntimeError` whose message
        contains the original exception text followed by the formatted
        hint block produced by :meth:`_format_memory_optimization_hints`.
        This gives users immediate guidance on which training arguments
        to tune when they encounter an out-of-memory error.

        Args:
            exc: The original OOM exception to augment.

        Returns:
            A ``JaxRuntimeError`` containing the original message plus
            any available memory optimization suggestions.
        """
        hint_text = self._format_memory_optimization_hints()
        message = [str(exc)]
        if hint_text is not None:
            message.extend(["", hint_text])
        return jax.errors.JaxRuntimeError("\n".join(message))

    @staticmethod
    def _normalize_esurge_prompts(
        prompts: tp.Any,
        apply_chat_template: bool,
    ) -> list[str | list[dict[str, str]]]:
        """Normalize user-provided prompts into strings or chat conversations."""

        def _normalize_single(item: tp.Any) -> str | list[dict[str, str]]:
            """Normalize a single prompt item into a string or chat message list.

            Handles dicts (with ``role``/``content`` or ``prompt``/``text`` keys),
            lists of messages, and plain strings. When ``apply_chat_template`` is
            True, bare strings are wrapped as a user chat message.

            Args:
                item: A prompt in any supported format (str, dict, or list).

            Returns:
                A plain string or a list of chat-message dicts.
            """
            if isinstance(item, list):
                if not item:
                    return ""
                if isinstance(item[0], dict):
                    return item
                if len(item) == 1 and isinstance(item[0], list):
                    return _normalize_single(item[0])
                return str(item)

            if isinstance(item, dict):
                if "role" in item and "content" in item:
                    return [item]
                for key in ("prompt", "text", "content"):
                    if key in item:
                        return _normalize_single(item[key])
                return str(item)

            if apply_chat_template:
                return [{"role": "user", "content": str(item)}]
            return str(item)

        if isinstance(prompts, list):
            if not prompts:
                return []
            if isinstance(prompts[0], dict):
                return [_normalize_single(prompts)]
            return [_normalize_single(p) for p in prompts]
        return [_normalize_single(prompts)]

    @staticmethod
    def _decode_prompt_batch(
        processor: PreTrainedTokenizerBase | None,
        input_ids: jax.Array | np.ndarray,
        skip_special_tokens: bool = True,
        pad_token_id: int | None = None,
        pop_pad_tokens: bool = False,
        attention_mask: jax.Array | np.ndarray | None = None,
    ) -> list[str]:
        """Decode a batch of token IDs into prompt strings.

        Args:
            processor: Tokenizer or processor with a ``decode`` method.
            input_ids: Token ID array of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            skip_special_tokens: Whether to strip special tokens during decoding.
            pad_token_id: Explicit pad token ID for stripping; inferred from
                ``processor`` if not provided and ``pop_pad_tokens`` is True.
            pop_pad_tokens: If True, remove padding tokens before decoding.
            attention_mask: Optional mask used to identify real (non-pad) tokens.

        Returns:
            A list of decoded prompt strings, one per sequence in the batch.

        Raises:
            ValueError: If ``processor`` is None or lacks a ``decode`` method.
        """
        if processor is None or not hasattr(processor, "decode"):
            raise ValueError("Cannot decode input_ids to prompts without a valid processor")
        array = np.asarray(input_ids)
        if array.ndim == 1:
            array = array[None, :]

        mask = None
        if attention_mask is not None:
            mask = np.asarray(attention_mask)
            if mask.ndim == 1:
                mask = mask[None, :]

        # Get pad_token_id if not provided
        if pop_pad_tokens and pad_token_id is None:
            pad_token_id = (
                getattr(processor, "pad_token_id", None)
                or getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
                or 0
            )
        prompts: list[str] = []
        for i, seq in enumerate(array):
            if pop_pad_tokens:
                if mask is not None:
                    # Use attention_mask to reliably extract real tokens
                    seq = seq[mask[i].astype(bool)]
                elif pad_token_id is not None:
                    seq = seq[seq != pad_token_id]
            prompts.append(processor.decode(seq, skip_special_tokens=skip_special_tokens))
        return prompts

    @staticmethod
    def _coerce_generation_texts(
        values: str | collections.abc.Sequence[tp.Any] | None,
        *,
        fallback: str | collections.abc.Sequence[tp.Any] | None = None,
    ) -> list[str]:
        """Normalize generation text outputs into a list of strings."""
        source = values if values is not None else fallback
        if source is None:
            return []
        if isinstance(source, str):
            return [source]
        if isinstance(source, collections.abc.Sequence):
            return [item if isinstance(item, str) else str(item) for item in source]
        return [str(source)]

    @staticmethod
    def _coerce_optional_generation_texts(
        values: str | collections.abc.Sequence[tp.Any] | None,
        *,
        target_len: int,
    ) -> list[str | None]:
        """Normalize optional generation text metadata to a fixed-length list."""
        if values is None:
            return [None] * target_len
        if isinstance(values, str):
            normalized: list[str | None] = [values]
        elif isinstance(values, collections.abc.Sequence):
            normalized = [item if isinstance(item, str) else (None if item is None else str(item)) for item in values]
        else:
            normalized = [str(values)]
        if len(normalized) < target_len:
            normalized.extend([None] * (target_len - len(normalized)))
        return normalized[:target_len]

    @staticmethod
    def _coerce_generation_metadata_list(
        values: collections.abc.Sequence[tp.Any] | tp.Any | None,
        *,
        target_len: int,
    ) -> list[tp.Any | None]:
        """Normalize non-text generation metadata to a fixed-length list."""
        if values is None:
            return [None] * target_len
        if isinstance(values, collections.abc.Sequence) and not isinstance(values, (str, bytes)):
            normalized = list(values)
        else:
            normalized = [values]
        if len(normalized) < target_len:
            normalized.extend([None] * (target_len - len(normalized)))
        return normalized[:target_len]

    @staticmethod
    def _coerce_mapping_like(value: tp.Any) -> tp.Any:
        """Coerce JSON-string payloads into mapping-like objects when possible."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    @classmethod
    def _normalize_tool_call_payloads(cls, tool_calls: tp.Any) -> list[dict[str, tp.Any]]:
        """Normalize structured tool-call payloads for chat-template rendering."""
        if not isinstance(tool_calls, collections.abc.Sequence) or isinstance(tool_calls, (str, bytes)):
            return []

        normalized_calls: list[dict[str, tp.Any]] = []
        for raw_call in tool_calls:
            if isinstance(raw_call, dict):
                call = dict(raw_call)
            elif hasattr(raw_call, "model_dump"):
                try:
                    call = dict(raw_call.model_dump(exclude_none=True))
                except Exception:
                    continue
            else:
                function_payload = getattr(raw_call, "function", None)
                call = {}
                call_id = getattr(raw_call, "id", None)
                call_type = getattr(raw_call, "type", None)
                if call_id is not None:
                    call["id"] = call_id
                if call_type is not None:
                    call["type"] = call_type
                if function_payload is not None:
                    if hasattr(function_payload, "model_dump"):
                        try:
                            call["function"] = dict(function_payload.model_dump(exclude_none=True))
                        except Exception:
                            pass
                    else:
                        function_dict: dict[str, tp.Any] = {}
                        function_name = getattr(function_payload, "name", None)
                        function_arguments = getattr(function_payload, "arguments", None)
                        if function_name is not None:
                            function_dict["name"] = function_name
                        if function_arguments is not None:
                            function_dict["arguments"] = function_arguments
                        if function_dict:
                            call["function"] = function_dict
                if not call:
                    continue

            function_payload = call.get("function")
            if isinstance(function_payload, dict):
                function_dict = dict(function_payload)
                arguments = cls._coerce_mapping_like(function_dict.get("arguments"))
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {"value": str(arguments)}
                function_dict["arguments"] = arguments
                call["function"] = function_dict
            elif isinstance(function_payload, str):
                coerced = cls._coerce_mapping_like(function_payload)
                if isinstance(coerced, dict):
                    call["function"] = coerced

            normalized_calls.append(call)
        return normalized_calls

    def _build_structured_assistant_messages(
        self,
        contents: list[str],
        *,
        tool_calls: list[tp.Any | None] | None = None,
    ) -> list[list[dict[str, tp.Any]]]:
        """Build assistant message payloads with normalized tool calls when present."""
        if tool_calls is None:
            tool_call_records = [None] * len(contents)
        else:
            tool_call_records = self._coerce_generation_metadata_list(tool_calls, target_len=len(contents))

        messages: list[list[dict[str, tp.Any]]] = []
        for content, tool_call_payload in zip(contents, tool_call_records, strict=False):
            message: dict[str, tp.Any] = {"role": "assistant", "content": content}
            normalized_tool_calls = self._normalize_tool_call_payloads(tool_call_payload)
            if normalized_tool_calls:
                message["tool_calls"] = normalized_tool_calls
            messages.append([message])
        return messages

    def _reward_chat_template_tools(self) -> list[dict[str, tp.Any]] | None:
        """Resolve optional tool schemas for reward-side chat-template rendering."""
        arguments = getattr(self, "arguments", None)
        if arguments is None:
            return None
        tool_schemas = getattr(arguments, "tool_schemas", None)
        return tool_schemas if isinstance(tool_schemas, list) else None

    def _build_reward_call_kwargs(
        self,
        reward_func: tp.Callable[..., tp.Any],
        *,
        prompts: tp.Any,
        completions: tp.Any,
        max_length: int,
        raw_completions: tp.Any | None = None,
        prompt_texts: list[str] | None = None,
        completion_texts: list[str] | None = None,
        raw_text: list[str] | None = None,
        reasoning: list[str | None] | None = None,
        tool_calls: list[tp.Any | None] | None = None,
        batch: dict[str, tp.Any] | None = None,
        **extra_kwargs: tp.Any,
    ) -> dict[str, tp.Any]:
        """Build filtered kwargs for callable reward functions."""
        return filter_kwargs_for_callable(
            reward_func,
            {
                "prompts": prompts,
                "completions": completions,
                "raw_completions": raw_completions,
                "prompt_texts": prompt_texts,
                "completion_texts": completion_texts,
                "raw_text": raw_text,
                "reasoning": reasoning,
                "tool_calls": tool_calls,
                "max_length": max_length,
                "batch": batch,
                **extra_kwargs,
            },
        )

    @staticmethod
    def _extract_reward_batch_sidechannels(batch: tp.Any) -> dict[str, tp.Any]:
        """Preserve non-numeric batch metadata needed by callable reward functions."""
        if isinstance(batch, dict):
            if "tools" not in batch:
                return {}
            tools = batch["tools"]
            if isinstance(tools, tuple):
                tools = list(tools)
            elif isinstance(tools, (np.ndarray, jax.Array)):
                tools = np.asarray(jax.device_get(tools), dtype=object).tolist()
            return {"tools": tools}

        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
            tools = [example.get("tools") for example in batch]
            if any(tool is not None for tool in tools):
                return {"tools": tools}
        return {}

    @staticmethod
    def _sanitize_text_prompt(prompt: str, processor: PreTrainedTokenizerBase | None) -> str:
        """Remove pad token occurrences from a decoded text prompt.

        Args:
            prompt: The decoded text string to sanitize.
            processor: Tokenizer or processor to look up the pad token from.

        Returns:
            The prompt string with all pad token substrings removed.
        """
        pad_token = None
        if processor is not None:
            pad_token = getattr(processor, "pad_token", None) or getattr(
                getattr(processor, "tokenizer", None), "pad_token", None
            )
        if pad_token:
            prompt = prompt.replace(pad_token, "")
        return prompt

    @staticmethod
    def _peek_first_example(dataset: tp.Any) -> tp.Any | None:
        """Retrieve the first example from a dataset without consuming it.

        Supports HF Datasets (indexing), iterables, dicts of datasets, and
        ``ShardedDataSource`` objects. Returns None if the dataset is empty
        or the first example cannot be retrieved.

        Args:
            dataset: Any dataset-like object to peek into.

        Returns:
            The first example, or None if unavailable.
        """
        if dataset is None:
            return None
        if isinstance(dataset, dict):
            for item in dataset.values():
                return BaseTrainer._peek_first_example(item)
            return None
        try:
            return dataset[0]
        except Exception:
            pass
        try:
            return next(iter(dataset))
        except Exception:
            pass
        try:
            shard_names = getattr(dataset, "shard_names", None)
            open_shard = getattr(dataset, "open_shard", None)
            if shard_names and open_shard:
                return next(iter(open_shard(shard_names[0])))
        except Exception:
            pass
        return None

    def _initialize_conversational_flags(self, train_dataset: tp.Any, eval_dataset: tp.Any) -> None:
        """Detect whether train and eval datasets use conversational format.

        Peeks at the first example of each dataset and sets
        ``self.train_is_conversational`` and ``self.eval_is_conversational``
        accordingly.

        Args:
            train_dataset: The training dataset to inspect.
            eval_dataset: The evaluation dataset to inspect.
        """
        from .prompt_transforms import is_conversational

        self.train_is_conversational = False
        self.eval_is_conversational = False

        train_sample = self._peek_first_example(train_dataset)
        if train_sample is not None:
            self.train_is_conversational = is_conversational(train_sample)

        eval_sample = self._peek_first_example(eval_dataset)
        if eval_sample is not None:
            self.eval_is_conversational = is_conversational(eval_sample)

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
        """No-op setter for the mesh property.

        The mesh is derived from the model and cannot be set directly.
        Assignments are silently ignored.

        Args:
            val: The value to set (ignored).
        """
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
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any, ...]:
        """Extra arguments passed to the shared training function at each step."""
        return self._train_shared_fn_extra_args_

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any, ...]:
        """Extra arguments passed to the shared evaluation function at each step."""
        return self._eval_shared_fn_extra_args_

    @property
    def _train_shared_fn_static_args(self) -> tuple[tp.Any, ...]:
        """Static (compile-time constant) arguments for the shared training function."""
        return self._train_shared_fn_static_args_  # pyright: ignore[reportReturnType]

    @property
    def _eval_shared_fn_static_args(self) -> tuple[tp.Any, ...]:
        """Static (compile-time constant) arguments for the shared evaluation function."""
        return self._eval_shared_fn_static_args_  # pyright: ignore[reportReturnType]

    @_train_shared_fn_static_args.setter
    def _train_shared_fn_static_args(self, val):
        """Set static arguments for the shared training function."""
        self._train_shared_fn_static_args_ = val

    @_eval_shared_fn_static_args.setter
    def _eval_shared_fn_static_args(self, val):
        """Set static arguments for the shared evaluation function."""
        self._eval_shared_fn_static_args_ = val

    @_train_shared_fn_extra_args.setter
    def _train_shared_fn_extra_args(self, val):
        """Set extra arguments for the shared training function."""
        self._train_shared_fn_extra_args_ = val

    @_eval_shared_fn_extra_args.setter
    def _eval_shared_fn_extra_args(self, val):
        """Set extra arguments for the shared evaluation function."""
        self._eval_shared_fn_extra_args_ = val

    @cached_property
    def _pad_token_id(self):
        """Resolve the pad token ID from the processing class.

        Falls back to the first EOS token ID if no pad token is defined.

        Returns:
            The integer pad token ID.
        """
        if isinstance(self.processing_class, ProcessorMixin):
            pad_token_id = self.processing_class.tokenizer.pad_token_id
        else:
            pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is not None:
            return pad_token_id
        else:
            return self.eos_token_id[0]

    @cached_property
    def _eos_token_id(self) -> list[int]:
        """Collect all unique EOS token IDs from the processor and model config.

        Merges EOS IDs from the processing class and the model's generation
        config (if available), returning a deduplicated list.

        Returns:
            A list of unique EOS token IDs.
        """
        eos_ids = []
        if isinstance(self.processing_class, ProcessorMixin):
            proc_eos_token_id = self.processing_class.tokenizer.eos_token_id
        else:
            proc_eos_token_id = self.processing_class.eos_token_id
        if isinstance(proc_eos_token_id, int):
            proc_eos_token_id = [proc_eos_token_id]
        eos_ids = eos_ids + proc_eos_token_id
        if hasattr(self.model, "generation_config"):
            conf_eos = self.model.generation_config.eos_token_id
            if isinstance(conf_eos, int):
                conf_eos = [conf_eos]
            eos_ids = eos_ids + conf_eos
        return list(set(eos_ids))

    def _make_attn_mask(self, arr):
        """Build a causal attention mask that masks positions after the first EOS token.

        For each sequence in the batch, all positions up to and including the
        first EOS token are marked as 1 (attended); positions after are 0.
        If no EOS token is found in a sequence, all positions are attended.

        Args:
            arr: Token ID array of shape ``(batch_size, seq_len)``.

        Returns:
            An int32 attention mask of shape ``(batch_size, seq_len)``.
        """
        is_eos = jnp.isin(arr, jnp.asarray(self._eos_token_id).reshape(-1))
        return (
            (jnp.arange(is_eos.shape[1])[None, :].repeat(is_eos.shape[0], axis=0))
            <= jnp.where(
                is_eos.any(axis=1),
                jnp.argmax(is_eos.astype(jnp.int32), axis=1),
                jnp.full((is_eos.shape[0],), is_eos.shape[1]),
            )[:, None]
        ).astype(jnp.int32)

    def _to_sharded_source(
        self,
        dataset: "Dataset | IterableDataset | ShardedDataSource | None",
    ) -> ShardedDataSource | None:
        """Convert any dataset type to ShardedDataSource.

        This enables trainers to work with a unified data interface internally
        while accepting various input types from users.

        Args:
            dataset: Input dataset (HF Dataset, IterableDataset, or ShardedDataSource).

        Returns:
            ShardedDataSource wrapping the input, or None if input is None.

        Raises:
            TypeError: If dataset type is not supported.
        """
        if dataset is None:
            return None
        if isinstance(dataset, ShardedDataSource):
            return dataset
        # Use wrap_hf_dataset for HF datasets
        return wrap_hf_dataset(dataset)

    def _apply_preprocess_transforms(self) -> None:
        """Apply preprocessing transforms to data sources.

        Gets the trainer-specific transform via _get_preprocess_transform()
        and applies it to both train and eval sources if available.
        """
        transform = self._get_preprocess_transform()
        if transform is None:
            return

        if self._train_source is not None:
            self._train_source = self._train_source.transform(transform)
        if self._eval_source is not None:
            self._eval_source = self._eval_source.transform(transform)

    def _get_preprocess_transform(self) -> Transform | None:
        """Get trainer-specific preprocessing transform.

        Override in subclasses to return a Transform that will be applied
        to the ShardedDataSource during data loading.

        Returns:
            Transform instance or None if no preprocessing needed.
            Return None if data is already preprocessed (e.g., has input_ids).
        """
        return None

    def _is_pretokenized(self) -> bool:
        """Check if the training dataset is already tokenized.

        Returns:
            True if dataset already contains 'input_ids' field.
        """
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

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
        self.pruning_module = self.arguments.pruning_module
        self.memory_monitor = getattr(self.arguments, "memory_monitor", None)
        self._preemption_checkpoint_path = getattr(self, "_preemption_checkpoint_path", None)
        self._tpu_preemption_sync_available = getattr(self, "_tpu_preemption_sync_available", None)

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
            (),
        )
        self._eval_shared_fn_static_args_ = getattr(
            self,
            "_eval_shared_fn_static_args_",
            (),
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
        self.generate_function_with_model_kwargs = getattr(self, "generate_function_with_model_kwargs", None)
        self.latest_generation_samples = getattr(self, "latest_generation_samples", [])
        self.latest_benchmark_results = getattr(self, "latest_benchmark_results", {})
        self.preview_log_table = getattr(self, "preview_log_table", None)
        self.training_generation_log_table = getattr(self, "training_generation_log_table", None)
        self.benchmark_log_table = getattr(self, "benchmark_log_table", None)
        rng = getattr(self, "_generation_rng", None)
        seed = self.arguments.generation_seed
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
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
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
        The batch is automatically purified to remove non-array fields.
        """
        # Purify batch to keep only JAX-compatible array fields
        batch = self._purify_batch(batch)
        return batch, {}

    def _purify_batch(self, batch: dict) -> dict:
        """Remove non-JAX-compatible fields from a batch.

        Filters out fields that cannot be passed to JAX JIT-compiled functions,
        such as strings, lists of strings, or other non-array types.

        Parameters
        ----------
        batch : dict
            The batch dictionary to purify

        Returns
        -------
        dict
            Purified batch with only JAX-compatible array fields
        """

        # Handle list of dicts (uncollated batch)
        if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], dict):
            # Collate list of dicts into dict of arrays with padding
            collated = {}
            for key in batch[0].keys():
                values = [example.get(key) for example in batch]
                # Skip None values
                if any(v is None for v in values):
                    continue
                try:
                    arrays = [np.asarray(v) for v in values]
                    # Check if arrays have same shape
                    if all(arr.shape == arrays[0].shape for arr in arrays):
                        collated[key] = np.stack(arrays)
                    else:
                        # Pad sequences to same length (for 1D arrays like input_ids)
                        if all(arr.ndim == 1 for arr in arrays):
                            max_len = max(len(arr) for arr in arrays)
                            # Determine pad value (0 for input_ids, typically)
                            pad_value = 0
                            padded = []
                            for arr in arrays:
                                if len(arr) < max_len:
                                    padded.append(np.pad(arr, (0, max_len - len(arr)), constant_values=pad_value))
                                else:
                                    padded.append(arr)
                            collated[key] = np.stack(padded)
                        else:
                            # Can't handle multi-dimensional arrays with different shapes
                            pass
                except (ValueError, TypeError):
                    pass  # Skip non-stackable values
            batch = collated

        purified = {}
        for key, value in batch.items():
            # Keep only numeric arrays (numpy or JAX)
            if isinstance(value, (np.ndarray, jax.Array)):
                # Check if it's a numeric dtype (not object/string)
                if hasattr(value, "dtype") and np.issubdtype(value.dtype, np.number):
                    purified[key] = value
                elif hasattr(value, "dtype") and value.dtype == np.bool_:
                    purified[key] = value
            elif isinstance(value, (list, tuple)):
                # Try to convert to array - will fail for strings
                try:
                    arr = np.asarray(value)
                    if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
                        purified[key] = arr
                except (ValueError, TypeError):
                    pass  # Skip non-convertible values
        return purified

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
            shard_inputs = self.arguments.generation_shard_inputs
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
        _gen_config_cls: type | None = globals().get("GenerationConfig")
        if _gen_config_cls is None:
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
            """Insert a key-value pair into target dict only if value is not None.

            Args:
                target: Dictionary to conditionally insert into.
                key: Key to set in the dictionary.
                value: Value to set; insertion is skipped when None.
            """
            if value is not None:
                target[key] = value

        kwargs: dict[str, tp.Any] = {}
        _maybe_insert(kwargs, "top_p", args.generation_top_p)
        _maybe_insert(kwargs, "top_k", args.generation_top_k)
        _maybe_insert(kwargs, "presence_penalty", getattr(args, "generation_presence_penalty", None))
        _maybe_insert(kwargs, "frequency_penalty", getattr(args, "generation_frequency_penalty", None))
        _maybe_insert(kwargs, "repetition_penalty", getattr(args, "generation_repetition_penalty", None))
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
        """Return a copy of user-specified generation config overrides, or None if empty.

        Returns:
            A dict of generation config attribute overrides, or None if no
            overrides are configured.
        """
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
        accept_model_kwargs: bool = False,
        **generate_kwargs,
    ) -> tp.Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
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
        accept_model_kwargs:
            Whether the compiled function should accept an additional ``model_kwargs``
            dictionary (e.g. multimodal inputs) alongside ``input_ids`` and ``attention_mask``.
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

        if accept_model_kwargs:
            model_kwargs_sharding = {key: None for key in GENERATION_MODEL_INPUT_KEYS}

            @ejit(  # pyright: ignore[reportUntypedFunctionDecorator]
                in_shardings=(self.state_shardings, empty_sharding, empty_sharding, model_kwargs_sharding),
                out_shardings=(empty_sharding, empty_sharding, empty_sharding),
            )
            def generate(
                state: EasyDeLState,
                input_ids: jax.Array,
                attention_mask: jax.Array,
                model_kwargs: dict[str, tp.Any],
            ):
                """Run model generation with additional model kwargs (e.g., multimodal inputs).

                Args:
                    state: Current model state containing parameters and model.
                    input_ids: Tokenized input prompt IDs.
                    attention_mask: Attention mask for the input sequence.
                    model_kwargs: Extra model inputs such as pixel values for
                        multimodal models.

                Returns:
                    Tuple of (generated sequences, sharded input_ids,
                    sharded attention_mask).
                """
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
                    call_model_kwargs = compact_generation_model_kwargs(model_kwargs)
                    call_model_kwargs = prepare_generation_model_kwargs_for_call(
                        call_model_kwargs,
                        target_sequence_length=shard_attention_mask.shape[-1],
                        flatten_grouped_multimodal=False,
                    )
                    generate_inputs = {
                        "input_ids": shard_input_ids,
                        "attention_mask": shard_attention_mask,
                    }

                    if config_copy is None:
                        outputs = module.generate(
                            **generate_inputs,
                            **call_model_kwargs,
                            **effective_generate_kwargs,
                        )
                    else:
                        outputs = module.generate(
                            **generate_inputs,
                            generation_config=config_copy,
                            **call_model_kwargs,
                            **effective_generate_kwargs,
                        )

                    sequences: jax.Array = outputs.sequences if hasattr(outputs, "sequences") else outputs
                    return sequences, shard_input_ids, shard_attention_mask

            return generate

        @ejit(  # pyright: ignore[reportUntypedFunctionDecorator]
            in_shardings=(self.state_shardings, empty_sharding, empty_sharding),
            out_shardings=(empty_sharding, empty_sharding, empty_sharding),
        )
        def generate(state: EasyDeLState, input_ids: jax.Array, attention_mask: jax.Array):
            """Run model generation from input_ids and attention_mask only.

            Args:
                state: Current model state containing parameters and model.
                input_ids: Tokenized input prompt IDs.
                attention_mask: Attention mask for the input sequence.

            Returns:
                Tuple of (generated sequences, sharded input_ids,
                sharded attention_mask).
            """
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

                sequences: jax.Array = outputs.sequences if hasattr(outputs, "sequences") else outputs
                return sequences, shard_input_ids, shard_attention_mask

        return generate

    def generate_aio(
        self,
        input_ids: jax.Array | np.ndarray,
        attention_mask: jax.Array | np.ndarray | None = None,
        *,
        model_kwargs: dict[str, tp.Any] | None = None,
        state: EasyDeLState | None = None,
        generation_config: GenerationConfig | None = None,
        shard_inputs: bool | None = None,
        config_overrides: dict[str, tp.Any] | None = None,
        return_metadata: bool = False,
        all_gather: bool = False,
        **generate_kwargs,
    ):
        """Convenience wrapper around the compiled generation function.

        Handles generation configuration merging, function creation/caching,
        and optional all-gather across devices.

        Args:
            input_ids: Token IDs for the prompt.
            attention_mask: Optional attention mask. Defaults to all ones.
            model_kwargs: Optional dictionary of extra model inputs (e.g. multimodal
                tensors like ``pixel_values``). Keys must be supported by the model's
                ``prepare_inputs_for_generation``.
            state: Model state to use. Defaults to self.model_state.
            generation_config: Optional generation configuration override.
            shard_inputs: Whether to shard inputs across devices.
            config_overrides: Dictionary of generation config attribute overrides.
            return_metadata: If True, returns (sequences, prompt_ids, prompt_mask).
            all_gather: Whether to gather results from all devices.
            **generate_kwargs: Additional kwargs passed to generate.

        Returns:
            jax.Array: Generated sequences if return_metadata is False.
            tuple[jax.Array, jax.Array, jax.Array]: (sequences, prompt_ids, prompt_mask)
                if return_metadata is True.

        Raises:
            RuntimeError: If model state is not initialized.
        """

        if state is None:
            state = self.model_state
        if state is None:
            raise RuntimeError("Model state is not initialized; call after trainer setup.")

        if shard_inputs is None:
            shard_inputs = self.arguments.generation_shard_inputs

        input_ids = jnp.asarray(input_ids)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype=jnp.int32)
        else:
            attention_mask = jnp.asarray(attention_mask)
        normalized_model_kwargs = None
        has_model_kwargs = False
        if model_kwargs:
            normalized_model_kwargs, unsupported_model_kwargs = self._normalize_supported_generation_model_kwargs(
                state,
                model_kwargs,
            )
            if unsupported_model_kwargs:
                unsupported = ", ".join(unsupported_model_kwargs)
                raise ValueError(
                    "Compiled generation received raw model kwargs that are not supported by "
                    f"`prepare_inputs_for_generation`: {unsupported}."
                )
            normalized_model_kwargs = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x) if x is not None else None,
                normalized_model_kwargs,
            )
            has_model_kwargs = bool(compact_generation_model_kwargs(normalized_model_kwargs))

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
                accept_model_kwargs=has_model_kwargs,
                **merged_kwargs,
            )
        else:
            default_kwargs = self._default_generation_kwargs()
            default_overrides = self._default_generation_config_overrides()
            if has_model_kwargs:
                if self.generate_function_with_model_kwargs is None:
                    self.generate_function_with_model_kwargs = self.create_generate_function(
                        shard_inputs=shard_inputs,
                        config_overrides=default_overrides,
                        accept_model_kwargs=True,
                        **default_kwargs,
                    )
                generate_fn = self.generate_function_with_model_kwargs
            else:
                if self.generate_function is None:
                    self.generate_function = self.create_generate_function(
                        shard_inputs=shard_inputs,
                        config_overrides=default_overrides,
                        **default_kwargs,
                    )
                generate_fn = self.generate_function

        if has_model_kwargs:
            sequences, prompt_ids, prompt_mask = generate_fn(
                state,
                input_ids,
                attention_mask,
                normalized_model_kwargs,
            )
        else:
            sequences, prompt_ids, prompt_mask = generate_fn(state, input_ids, attention_mask)

        if all_gather:
            sequences = self._all_gather(sequences)
            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)

        if return_metadata:
            return sequences, prompt_ids, prompt_mask
        return sequences

    def release_generation_runtime(
        self,
        *,
        state: EasyDeLState | None = None,
        clear_esurge_compiled_cache: bool = False,
    ) -> None:
        """Drop cached generation runtimes so rollout HBM is reclaimed before scoring."""

        self.generate_function = None
        self.generate_function_with_model_kwargs = None

        state = state or self.model_state
        if state is not None:
            try:
                state.model.pause_esurge(
                    release_model_state=True,
                    clear_compiled_cache=clear_esurge_compiled_cache,
                )
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                log_debug_maybe(f"Failed to release generation runtime: {exc}")

        gc.collect()

    def _normalize_supported_generation_model_kwargs(
        self,
        state: EasyDeLState,
        model_kwargs: dict[str, tp.Any] | None,
    ) -> tuple[dict[str, tp.Any] | None, tuple[str, ...]]:
        """Normalize raw generation kwargs against the model's generation entrypoint."""

        if not model_kwargs:
            return None, ()

        requested_model_kwargs = compact_generation_model_kwargs(
            normalize_generation_model_kwargs(model_kwargs),
        )
        if not requested_model_kwargs:
            return None, ()

        normalized_model_kwargs = normalize_generation_model_kwargs(
            model_kwargs,
            model_callable=state.model.prepare_inputs_for_generation,
        )
        unsupported_model_kwargs = tuple(
            sorted(
                set(requested_model_kwargs.keys()) - set(compact_generation_model_kwargs(normalized_model_kwargs).keys())
            )
        )
        return normalized_model_kwargs, unsupported_model_kwargs

    def maybe_release_generation_runtime(
        self,
        results: GenerationResults,
        *,
        state: EasyDeLState | None = None,
        release_runtime_after_generation: bool = True,
        clear_esurge_compiled_cache_after_generation: bool = False,
    ) -> GenerationResults:
        """Optionally release generation runtime after materializing outputs.

        This is intentionally opt-in per generation call. Some trainers perform
        multiple back-to-back generations (for example policy then reference
        rollouts), so unconditional teardown inside `generate_unified` would
        introduce avoidable resume/rebuild churn.
        """
        if not release_runtime_after_generation:
            return results

        logger.info_once("Releasing generation runtime after generation step.")
        jax.block_until_ready(results.sequences)
        jax.block_until_ready(results.completion_ids)
        self.release_generation_runtime(
            state=state,
            clear_esurge_compiled_cache=clear_esurge_compiled_cache_after_generation,
        )
        return results

    def generate_unified(
        self,
        input_ids: jax.Array | np.ndarray | None = None,
        attention_mask: jax.Array | np.ndarray | None = None,
        prompts: str | list[str] | None = None,
        *,
        model_kwargs: dict[str, tp.Any] | None = None,
        state: EasyDeLState | None = None,
        use_esurge: bool | None = None,
        apply_chat_template: bool = False,
        generation_config: GenerationConfig | None = None,
        shard_inputs: bool | None = None,
        config_overrides: dict[str, tp.Any] | None = None,
        release_runtime_after_generation: bool = True,
        clear_esurge_compiled_cache_after_generation: bool = False,
        all_gather: bool = False,
        **generate_kwargs,
    ) -> GenerationResults:
        """Unified generation interface supporting both compiled and eSurge generation.

        This method provides a single interface for generation that automatically handles:
        - Conversion between text prompts and token IDs
        - Selection between eSurge and compiled generation based on configuration
        - Consistent output format regardless of generation method
        - Optional chat template application

        Args:
            input_ids: Optional token IDs for the prompt. If None, must provide prompts.
            attention_mask: Optional attention mask for input_ids.
            prompts: Optional text prompt(s). If None, must provide input_ids.
            state: Model state to use for generation. Defaults to self.model_state.
            use_esurge: Whether to use eSurge generation. Defaults to self.arguments.use_esurge_generation.
            apply_chat_template: Whether to apply chat template to prompts. Default False.
                If True and prompts is a string, wraps it in [{"role": "user", "content": prompt}].
            generation_config: Optional generation configuration.
            shard_inputs: Whether to shard inputs across devices.
            config_overrides: Optional overrides for generation config.
            model_kwargs: Optional dictionary of extra model inputs (e.g. multimodal
                tensors). Keys must be supported by the model's
                ``prepare_inputs_for_generation``.
            release_runtime_after_generation: Whether to release the eSurge runtime
                after generation completes. Defaults to True.
            clear_esurge_compiled_cache_after_generation: Whether to clear the eSurge
                compiled function cache after generation. Defaults to False.
            all_gather: Whether to gather results from all devices.
            **generate_kwargs: Additional kwargs passed to generation.

        Returns:
            GenerationResults: NamedTuple containing:
                - generation_results: The result text
                - prompt_ids: Token IDs for the prompt
                - prompt_mask: Attention mask for the prompt
                - sequences: Complete generated sequences (including prompt)

        Raises:
            ValueError: If neither input_ids nor prompts are provided.

        Example:
            >>> # Generate from token IDs (GRPO-style, no chat template)
            >>> results = trainer.generate_unified(
            ...     input_ids=prompt_ids,
            ...     attention_mask=mask,
            ...     apply_chat_template=False  # Raw generation
            ... )
            >>>
            >>> # Generate with chat template (preview generation)
            >>> results = trainer.generate_unified(
            ...     prompts="Explain quantum computing",
            ...     apply_chat_template=True  # Applies chat template
            ... )
            >>>
            >>> # eSurge with chat format
            >>> results = trainer.generate_unified(
            ...     prompts=[{"role": "user", "content": "Hello"}],
            ...     use_esurge=True
            ... )
        """
        if input_ids is None and prompts is None:
            raise ValueError("Must provide either input_ids or prompts")

        state = state or self.model_state
        args = self.arguments
        processor = self._get_processing_class()
        normalized_model_kwargs, unsupported_model_kwargs = self._normalize_supported_generation_model_kwargs(
            state,
            model_kwargs,
        )
        if unsupported_model_kwargs:
            unsupported = ", ".join(unsupported_model_kwargs)
            raise ValueError(
                "Raw generation model kwargs are not supported by this model's "
                f"`prepare_inputs_for_generation`: {unsupported}."
            )
        has_model_kwargs = bool(compact_generation_model_kwargs(normalized_model_kwargs))

        # Determine whether to use eSurge
        if use_esurge is None:
            use_esurge = args.use_esurge_generation
        if has_model_kwargs and prompts is not None and input_ids is None:
            raise ValueError(
                "Raw model kwargs require pretokenized `input_ids`/`attention_mask`. "
                "Use eSurge with multimodal messages or pass tokenized prompts.",
            )
        if use_esurge and has_model_kwargs:
            logger.warning_once(
                "Disabling eSurge generation because raw model kwargs were provided; "
                "falling back to compiled generation.",
            )
            use_esurge = False

        pad_token_id = self._pad_token_id
        max_tokens = args.generation_max_new_tokens
        if max_tokens is None:
            max_tokens = getattr(args, "max_completion_length", None)
        if max_tokens is None:
            max_tokens = 1024
        sampling_params = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=args.generation_temperature or 0.7,
            top_p=args.generation_top_p or 0.95,
            top_k=args.generation_top_k or 64,
            presence_penalty=float(getattr(args, "generation_presence_penalty", 0.0) or 0.0),
            frequency_penalty=float(getattr(args, "generation_frequency_penalty", 0.0) or 0.0),
            repetition_penalty=float(getattr(args, "generation_repetition_penalty", 1.0) or 1.0),
            n=args.generation_num_return_sequences or 1,
        )
        if config_overrides:
            max_new_tokens = config_overrides.get("max_new_tokens")
            if max_new_tokens is not None:
                sampling_params.max_tokens = int(max_new_tokens)
            temperature = config_overrides.get("temperature")
            if temperature is not None:
                sampling_params.temperature = float(temperature)
            top_p = config_overrides.get("top_p")
            if top_p is not None:
                sampling_params.top_p = float(top_p)
            top_k = config_overrides.get("top_k")
            if top_k is not None:
                sampling_params.top_k = int(top_k)
            num_return_sequences = config_overrides.get("num_return_sequences")
            if num_return_sequences is not None:
                sampling_params.n = int(num_return_sequences)
            presence_penalty = config_overrides.get("presence_penalty")
            if presence_penalty is not None:
                sampling_params.presence_penalty = float(presence_penalty)
            frequency_penalty = config_overrides.get("frequency_penalty")
            if frequency_penalty is not None:
                sampling_params.frequency_penalty = float(frequency_penalty)
            repetition_penalty = config_overrides.get("repetition_penalty")
            if repetition_penalty is not None:
                sampling_params.repetition_penalty = float(repetition_penalty)

        # Handle eSurge generation path
        if use_esurge:
            # When the caller provides tokenized prompts, respect their prompt length for
            # eSurge padding/sequence construction. Using `args.max_length` here can lead
            # to sequences where completions are truncated away when `args.max_length`
            # exceeds the provided `input_ids` length (common in RL trainers where
            # `input_ids` is padded to `max_prompt_length`, not `max_length`).
            prompt_seq_len = None
            if input_ids is not None:
                try:
                    prompt_seq_len = int(input_ids.shape[-1])
                except Exception:  # pragma: no cover - defensive: odd prompt containers
                    prompt_seq_len = None

            if prompts is None:
                decoded_prompts = self._decode_prompt_batch(
                    processor,
                    input_ids,
                    False,
                    pad_token_id,
                    True,
                    attention_mask,
                )
                prompts = self._normalize_esurge_prompts(decoded_prompts, apply_chat_template)
            else:
                prompts = self._normalize_esurge_prompts(prompts, apply_chat_template)
            return_prompts = prompts
            if not prompts:
                raise ValueError("No prompts available for eSurge generation")

            esurge_kwargs = self._esurge_init_kwargs()
            effective_prompt_len = prompt_seq_len if prompt_seq_len is not None else (args.max_length or 2048)
            # eSurge reserves a few tokens from the context budget (defaults to
            # `reserve_tokens = max_num_seqs`). When we tightly pack
            # `prompt_len + max_new_tokens == max_model_len` (common in PPO/GRPO
            # rollouts where prompt is padded to max_prompt_length), eSurge will
            # cap max_new_tokens by the reserve amount. Include the reserve in
            # the requested max_model_len so the user-visible generation length
            # stays consistent with `max_new_tokens`.
            reserve_tokens = esurge_kwargs.get("reserve_tokens")
            if reserve_tokens is None:
                reserve_tokens = esurge_kwargs.get("max_num_seqs", 0)
            esurge_kwargs["max_model_len"] = sampling_params.max_tokens + effective_prompt_len + int(reserve_tokens or 0)  # pyright: ignore[reportOptionalOperand]

            _log_kwargs = {k: v for k, v in esurge_kwargs.items() if k != "tokenizer"}
            logger.info_once(f"Creating eSurge {pprint.pformat(_log_kwargs)}")
            logger.info_once(
                f"SamplingParams(max_tokens={sampling_params.max_tokens},"
                f" temperature={sampling_params.temperature},"
                f" top_p={sampling_params.top_p},"
                f" top_k={sampling_params.top_k},"
                f" presence_penalty={sampling_params.presence_penalty},"
                f" frequency_penalty={sampling_params.frequency_penalty},"
                f" repetition_penalty={sampling_params.repetition_penalty},"
                f" n={sampling_params.n})"
            )
            esurge_engine = None
            try:
                esurge_engine = state.model.get_esurge(**esurge_kwargs)
                # Use the resolved engine directly to avoid a second get_esurge()/refresh pass.
                outputs: list[RequestOutput] = state.model._call_esurge_engine(
                    esurge_engine,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    stream=False,
                    use_tqdm=args.esurge_use_tqdm,
                )
            except Exception:
                if esurge_engine is None:
                    try:
                        # If setup failed before returning an engine handle, fall back to
                        # model-level pause to clean any partially initialized cached engine.
                        state.model.pause_esurge()
                    except Exception as cleanup_exc:  # pragma: no cover - best-effort resource cleanup
                        log_debug_maybe(f"Failed to pause eSurge engine(s) after setup failure: {cleanup_exc}")
                raise
            finally:
                if esurge_engine is not None:
                    try:
                        esurge_engine.pause()
                        if hasattr(esurge_engine, "release_model_state"):
                            esurge_engine.release_model_state(clear_compiled_cache=False)
                    except Exception as exc:  # pragma: no cover - best-effort resource cleanup
                        log_debug_maybe(f"Failed to pause/release eSurge engine after generation: {exc}")

            # Build padded token arrays from eSurge outputs to ensure consistent shapes
            max_seq_len = prompt_seq_len if prompt_seq_len is not None else (args.max_length or 2048)
            max_new_tokens = sampling_params.max_tokens if sampling_params.max_tokens is not None else 1024
            max_total_len = max_seq_len + max_new_tokens

            # Track prompt arrays once per request
            prompt_id_rows: list[list[int]] = []
            prompt_mask_rows: list[list[int]] = []
            sequence_rows: list[list[int]] = []
            completion_id_rows: list[list[int]] = []
            completion_mask_rows: list[list[int]] = []
            output_records: list[str] = []
            reasoning_records: list[str | None] = []
            tool_call_records: list[list | None] = []
            raw_output_records: list[str] = []
            completion_prompts: list[str | list[dict[str, str]]] = []
            prompt_indices: list[int | None] = []

            def _strip_pad(tokens: list[int] | np.ndarray) -> tuple[int, ...]:
                """Remove pad tokens for matching prompts across shuffled eSurge outputs."""
                arr = np.asarray(tokens, dtype=np.int64).tolist()
                return tuple(int(t) for t in arr if pad_token_id is None or t != pad_token_id)

            # If caller supplied tokenized prompts, keep them as-is for return values
            if input_ids is not None:
                base_prompt_ids = np.asarray(input_ids, dtype=np.int32)
                if base_prompt_ids.ndim == 1:
                    base_prompt_ids = base_prompt_ids[None, :]
                base_prompt_mask = (
                    np.ones_like(base_prompt_ids, dtype=np.int32)
                    if attention_mask is None
                    else np.asarray(attention_mask, dtype=np.int32)
                )
                for row in base_prompt_ids:
                    prompt_id_rows.append(list(row))
                for row in base_prompt_mask:
                    prompt_mask_rows.append(list(row))
                prompt_signature_map: dict[tuple[int, ...], list[int]] = {}
                for idx, row in enumerate(base_prompt_ids):
                    sig = _strip_pad(row)
                    prompt_signature_map.setdefault(sig, []).append(idx)
            else:
                prompt_signature_map = {}

            # When n>1, each RequestOutput has multiple CompletionOutput objects
            for output_idx, output in enumerate(outputs):
                # Flatten and truncate prompt tokens
                flattened_prompt_tokens: list[int] = []
                if output.prompt_token_ids:
                    for segment in output.prompt_token_ids:
                        flattened_prompt_tokens.extend(segment)
                base_prompt_tokens = flattened_prompt_tokens[:max_seq_len]
                base_prompt_len = len(base_prompt_tokens)
                prompt_padding = max_seq_len - base_prompt_len

                # Left-pad prompts for RL training
                padded_prompt_ids = [pad_token_id] * prompt_padding + base_prompt_tokens
                padded_prompt_mask = [0] * prompt_padding + [1] * base_prompt_len
                if input_ids is None:
                    # When prompts were strings, this represents the canonical prompt ids/masks.
                    prompt_id_rows.append(padded_prompt_ids)
                    prompt_mask_rows.append(padded_prompt_mask)

                mapped_prompt_idx = None
                if prompt_signature_map:
                    sig = _strip_pad(base_prompt_tokens)
                    if prompt_signature_map.get(sig):
                        mapped_prompt_idx = prompt_signature_map[sig].pop(0)
                        if not prompt_signature_map[sig]:
                            prompt_signature_map.pop(sig, None)

                source_prompt = getattr(output, "prompt", return_prompts[output_idx] if return_prompts else None)

                # Process each completion (handles n>1 sampling)
                for completion in output.outputs:
                    completion_prompts.append(source_prompt)
                    prompt_indices.append(mapped_prompt_idx)
                    # Add prompt arrays
                    completion_text = getattr(completion, "text", None)
                    if not isinstance(completion_text, str) or completion_text == "":
                        output_get_text = getattr(output, "get_text", None)
                        output_fallback_text = output_get_text() if callable(output_get_text) else ""
                        completion_text = getattr(output, "accumulated_text", "") or output_fallback_text
                    output_records.append(completion_text)

                    completion_reasoning = getattr(completion, "reasoning_content", None)
                    reasoning_records.append(completion_reasoning if isinstance(completion_reasoning, str) else None)

                    completion_tool_calls = getattr(completion, "tool_calls", None)
                    tool_call_records.append(completion_tool_calls if completion_tool_calls is not None else None)

                    completion_raw_text = getattr(completion, "raw_text", None)
                    if not isinstance(completion_raw_text, str) or completion_raw_text == "":
                        output_get_text = getattr(output, "get_text", None)
                        output_fallback_text = output_get_text() if callable(output_get_text) else ""
                        completion_raw_text = (
                            getattr(output, "raw_accumulated_text", "")
                            or getattr(output, "accumulated_text", "")
                            or output_fallback_text
                        )
                    raw_output_records.append(completion_raw_text)

                    # Extract completion tokens
                    completion_tokens: list[int] = []
                    if completion:
                        completion_tokens = list(getattr(completion, "token_ids", []) or [])

                    # Truncate completion if too long
                    if len(completion_tokens) > max_new_tokens:
                        completion_tokens = completion_tokens[:max_new_tokens]

                    # Right-pad completions
                    completion_len = len(completion_tokens)
                    completion_padding = max_new_tokens - completion_len
                    padded_completion_ids = completion_tokens + [pad_token_id] * completion_padding
                    padded_completion_mask = [1] * completion_len + [0] * completion_padding

                    completion_id_rows.append(padded_completion_ids)
                    completion_mask_rows.append(padded_completion_mask)

                    # Build full sequence: [left-padded prompt] + [generated tokens] + [right padding]
                    sequence_tokens = [pad_token_id] * prompt_padding + base_prompt_tokens + completion_tokens
                    remaining_padding = max_total_len - len(sequence_tokens)
                    sequence_rows.append(sequence_tokens + [pad_token_id] * remaining_padding)

            if not sequence_rows:
                raise RuntimeError("eSurge generation returned no completions")
            if not prompt_id_rows or not prompt_mask_rows:
                raise RuntimeError("Could not determine prompt token metadata for eSurge generation")

            # Reorder completions to align with original prompt order when possible.
            if prompt_indices:
                num_prompts = len(prompt_id_rows)
                per_prompt: list[list[int]] = [[] for _ in range(num_prompts)]
                unmatched: list[int] = []
                for i, pidx in enumerate(prompt_indices):
                    if pidx is None or pidx < 0 or pidx >= num_prompts:
                        unmatched.append(i)
                    else:
                        per_prompt[pidx].append(i)
                new_order = [idx for group in per_prompt for idx in group] + unmatched
                completion_id_rows = [completion_id_rows[i] for i in new_order]
                completion_mask_rows = [completion_mask_rows[i] for i in new_order]
                completion_prompts = [completion_prompts[i] for i in new_order]
                output_records = [output_records[i] for i in new_order]
                reasoning_records = [reasoning_records[i] for i in new_order]
                tool_call_records = [tool_call_records[i] for i in new_order]
                raw_output_records = [raw_output_records[i] for i in new_order]
                sequence_rows = [sequence_rows[i] for i in new_order]

            # Use original prompt ids/masks (not per-completion duplicates); repeat only if needed to align shapes.
            base_prompt_ids = jnp.array(np.asarray(prompt_id_rows, dtype=np.int32))
            base_prompt_mask = jnp.array(np.asarray(prompt_mask_rows, dtype=np.int32))
            prompt_ids = base_prompt_ids
            prompt_mask = base_prompt_mask

            sequences = jnp.array(np.asarray(sequence_rows, dtype=np.int32))
            completion_ids = jnp.array(np.asarray(completion_id_rows, dtype=np.int32))
            completion_mask = jnp.array(np.asarray(completion_mask_rows, dtype=np.int32))
            total_seq_len = prompt_ids.shape[-1] + completion_ids.shape[-1]
            sequences = sequences[:, :total_seq_len]

            if all_gather:
                sequences = self._all_gather(sequences)
                prompt_ids = self._all_gather(prompt_ids)
                prompt_mask = self._all_gather(prompt_mask)
                completion_ids = self._all_gather(completion_ids)
                completion_mask = self._all_gather(completion_mask)
            return self.maybe_release_generation_runtime(
                GenerationResults(
                    generation_results=output_records,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    sequences=sequences,
                    completion_ids=completion_ids,
                    completion_mask=completion_mask,
                    decoded_prompts=return_prompts,
                    completion_prompts=completion_prompts,
                    text=output_records,
                    reasoning=reasoning_records,
                    tool_calls=tool_call_records,
                    raw_text=raw_output_records,
                ),
                state=state,
                release_runtime_after_generation=release_runtime_after_generation,
                clear_esurge_compiled_cache_after_generation=clear_esurge_compiled_cache_after_generation,
            )

        # Handle compiled generation path
        else:
            prompt_text_records: list[str] = []

            if input_ids is None:
                if prompts is None:
                    raise ValueError("Must provide prompts when input_ids is None")
                if processor is None or not hasattr(processor, "encode"):
                    raise ValueError("Cannot tokenize prompts without a valid processor")

                normalized_prompts = self._normalize_esurge_prompts(prompts, apply_chat_template)
                if not normalized_prompts:
                    raise ValueError("No prompts provided for generation")

                max_seq_len = args.max_length or 2048

                # Ensure left-padding for RL training (prompts should align at the right)
                original_padding_side = getattr(processor, "padding_side", None)
                if hasattr(processor, "padding_side"):
                    processor.padding_side = "left"

                encoded_ids: list[np.ndarray] = []
                encoded_masks: list[np.ndarray] = []

                for normalized in normalized_prompts:
                    if isinstance(normalized, list):
                        if not apply_chat_template or not hasattr(processor, "apply_chat_template"):
                            raise ValueError(
                                "Chat prompts require apply_chat_template=True when using compiled generation"
                            )
                        encoding = processor.apply_chat_template(
                            normalized,
                            return_tensors="np",
                            padding="max_length",
                            max_length=max_seq_len,
                            truncation=True,
                            add_generation_prompt=True,
                            return_dict=True,
                        )
                        prompt_text_records.append(str(normalized))
                    else:
                        encoding = processor(
                            normalized,
                            return_tensors="np",
                            padding="max_length",
                            max_length=max_seq_len,
                            truncation=True,
                            add_special_tokens=True,
                        )
                        prompt_text_records.append(normalized)

                    ids = np.asarray(encoding["input_ids"], dtype=np.int32)
                    mask = encoding.get("attention_mask")
                    if mask is None:
                        mask = np.ones_like(ids, dtype=np.int32)
                    else:
                        mask = np.asarray(mask, dtype=np.int32)
                    encoded_ids.append(ids)
                    encoded_masks.append(mask)

                # Restore original padding side
                if hasattr(processor, "padding_side") and original_padding_side is not None:
                    processor.padding_side = original_padding_side

                input_ids = jnp.asarray(np.concatenate(encoded_ids, axis=0), dtype=jnp.int32)
                attention_mask = jnp.asarray(np.concatenate(encoded_masks, axis=0), dtype=jnp.int32)

            # Use compiled generation (internal call, deprecation warning suppressed)
            sequences, prompt_ids, prompt_mask = self.generate_aio(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model_kwargs=normalized_model_kwargs if has_model_kwargs else None,
                state=state,
                generation_config=generation_config,
                shard_inputs=shard_inputs,
                config_overrides=config_overrides,
                return_metadata=True,
                all_gather=all_gather,
                **generate_kwargs,
            )
            # Extract completion tokens from sequences
            max_new_tokens = sampling_params.max_tokens
            prompt_len = prompt_ids.shape[1]
            completion_ids = sequences[:, prompt_len : prompt_len + max_new_tokens]
            completion_mask = self._make_attn_mask(completion_ids)
            decoded_prompt_texts = self._decode_prompt_batch(processor, prompt_ids, False, pad_token_id, True)

            # Build per-completion prompt list aligned with generated rows.
            completion_prompts: list[str | list[dict[str, str]]] = []
            repeat_factor = completion_ids.shape[0] // max(len(decoded_prompt_texts), 1)
            repeat_factor = max(repeat_factor, 1)
            for text_prompt in decoded_prompt_texts:
                completion_prompts.extend([text_prompt] * repeat_factor)
            if len(completion_prompts) < completion_ids.shape[0] and decoded_prompt_texts:
                completion_prompts.extend(
                    [decoded_prompt_texts[-1]] * (completion_ids.shape[0] - len(completion_prompts))
                )
            completion_prompts = completion_prompts[: completion_ids.shape[0]]

            generated_texts = self._decode_prompt_batch(
                processor,
                sequences,
                skip_special_tokens=True,
                pad_token_id=pad_token_id,
                pop_pad_tokens=True,
            )
            completion_texts = self._decode_prompt_batch(
                processor,
                completion_ids,
                skip_special_tokens=True,
                pad_token_id=pad_token_id,
                pop_pad_tokens=True,
            )
            raw_completion_texts = self._decode_prompt_batch(
                processor,
                completion_ids,
                skip_special_tokens=False,
                pad_token_id=pad_token_id,
                pop_pad_tokens=True,
            )

            return self.maybe_release_generation_runtime(
                GenerationResults(
                    generation_results=generated_texts,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    sequences=sequences,
                    completion_ids=completion_ids,
                    completion_mask=completion_mask,
                    decoded_prompts=decoded_prompt_texts,
                    completion_prompts=completion_prompts or None,
                    text=completion_texts,
                    reasoning=[None] * len(completion_texts),
                    tool_calls=[None] * len(completion_texts),
                    raw_text=raw_completion_texts,
                ),
                state=state,
                release_runtime_after_generation=release_runtime_after_generation,
                clear_esurge_compiled_cache_after_generation=clear_esurge_compiled_cache_after_generation,
            )

    def _get_processing_class(self):
        """Resolve the tokenizer or processor associated with the trainer.

        Checks ``self.processing_class``, ``self.tokenizer``, and the model
        state in order, returning the first non-None result.

        Returns:
            The tokenizer/processor instance, or None if none is available.
        """
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
        """Decode a batch of token ID arrays into human-readable strings.

        Args:
            token_ids: Array-like of token IDs with shape ``(batch, seq)`` or
                ``(seq,)``.

        Returns:
            List of decoded strings, or None if no processor is available or
            decoding fails.
        """
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
        """Collect prompts for preview generation during training.

        Gathers prompts from ``arguments.generation_prompts`` first, then
        supplements with randomly sampled training dataset entries if more
        prompts are needed.

        Returns:
            List of prompt objects (strings, dicts, or chat message lists).
        """
        args = self.arguments
        if args is None:
            return []
        configured_prompts = list(args.generation_prompts)
        target = args.generation_num_prompts
        prompts = configured_prompts[: target or len(configured_prompts)]
        remaining = max((target or 0) - len(prompts), 0)
        if remaining > 0 and args.generation_use_train_prompts:
            prompts.extend(self._sample_prompts_from_dataset(remaining))
        return prompts

    def _sample_prompts_from_dataset(self, expected: int) -> list[tp.Any]:
        """Randomly sample prompt entries from the training dataset.

        Args:
            expected: Maximum number of samples to draw.

        Returns:
            List of raw dataset samples (up to ``expected`` items). Returns
            an empty list if the dataset is unavailable or empty.
        """
        dataset = getattr(self, "dataset_train", None)
        if dataset is None or expected <= 0:
            return []
        if isinstance(dataset, ShardedDataSource):
            return self._sample_prompts_from_sharded_source(dataset, expected)
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

    def _sample_random_example_from_shard(
        self,
        source: ShardedDataSource[tp.Any],
        shard_name: str,
        *,
        row_index: int | None = None,
        shard_rows: int | None = None,
    ) -> tp.Any | None:
        """Sample a single example from one shard.

        Prefers direct row seeks when a row index or row count is available.
        Falls back to reservoir sampling over ``open_shard`` when shard sizes
        are unknown.
        """
        try:
            if row_index is not None:
                return next(source.open_shard_at_row(shard_name, int(row_index)), None)
            if shard_rows is not None and shard_rows > 0:
                random_row = int(self._generation_rng.integers(shard_rows))
                return next(source.open_shard_at_row(shard_name, random_row), None)
            sampled = None
            for seen, example in enumerate(source.open_shard(shard_name), start=1):
                if int(self._generation_rng.integers(seen)) == 0:
                    sampled = example
            return sampled
        except Exception as exc:  # pragma: no cover - best effort sampling for previews
            log_debug_maybe(f"Failed to sample preview prompt from shard '{shard_name}': {exc}")
            return None

    def _sample_prompts_from_sharded_source(
        self,
        source: ShardedDataSource[tp.Any],
        expected: int,
    ) -> list[tp.Any]:
        """Randomly sample raw examples from a :class:`ShardedDataSource`.

        When shard row counts are available, samples globally across the full
        source and uses ``open_shard_at_row`` for efficient random access.
        Otherwise falls back to per-shard sampling with reservoir selection.
        """
        shard_names = list(source.shard_names)
        if not shard_names or expected <= 0:
            return []

        shard_rows: list[int | None] = []
        all_shard_sizes_known = True
        for shard_name in shard_names:
            row_count: int | None = None
            try:
                info = source.get_shard_info(shard_name)
            except Exception as exc:  # pragma: no cover - metadata fetch can fail on remote sources
                log_debug_maybe(f"Failed to fetch shard info for '{shard_name}': {exc}")
                info = None
            if info is not None and info.num_rows is not None:
                row_count = max(int(info.num_rows), 0)
            else:
                all_shard_sizes_known = False
            shard_rows.append(row_count)

        prompts: list[tp.Any] = []
        if all_shard_sizes_known:
            total_rows = int(sum(row_count for row_count in shard_rows if row_count is not None))
            if total_rows <= 0:
                return []
            count = min(expected, total_rows)
            global_indices = np.asarray(self._generation_rng.choice(total_rows, size=count, replace=False))
            cumulative_rows = np.cumsum(np.asarray([int(row_count or 0) for row_count in shard_rows], dtype=np.int64))
            previous_cumulative_rows = np.concatenate((np.asarray([0], dtype=np.int64), cumulative_rows[:-1]))

            for global_idx in np.atleast_1d(global_indices):
                shard_idx = int(np.searchsorted(cumulative_rows, int(global_idx), side="right"))
                local_row = int(int(global_idx) - int(previous_cumulative_rows[shard_idx]))
                sampled = self._sample_random_example_from_shard(
                    source,
                    shard_names[shard_idx],
                    row_index=local_row,
                    shard_rows=shard_rows[shard_idx],
                )
                if sampled is not None:
                    prompts.append(sampled)
            return prompts

        nonempty_weights = np.asarray(
            [float(row_count) if row_count is not None and row_count > 0 else 0.0 for row_count in shard_rows],
            dtype=np.float64,
        )
        sampling_probs: np.ndarray | None = None
        if nonempty_weights.sum() > 0:
            sampling_probs = nonempty_weights / nonempty_weights.sum()

        attempts = 0
        max_attempts = max(expected * 4, len(shard_names))
        while len(prompts) < expected and attempts < max_attempts:
            attempts += 1
            if sampling_probs is None:
                shard_idx = int(self._generation_rng.integers(len(shard_names)))
            else:
                shard_idx = int(self._generation_rng.choice(len(shard_names), p=sampling_probs))
            sampled = self._sample_random_example_from_shard(
                source,
                shard_names[shard_idx],
                shard_rows=shard_rows[shard_idx],
            )
            if sampled is not None:
                prompts.append(sampled)

        return prompts

    def _prepare_generation_input(
        self,
        prompt: tp.Any,
        *,
        tools: tp.Any | None = None,
    ) -> dict[str, tp.Any] | None:
        """Tokenize and pad a single prompt into model-ready input arrays.

        Handles raw strings, chat-format message lists, and dict samples that
        may already contain ``input_ids``. Uses left-padding and left-truncation
        so the generation continuation is always at the right edge.

        Args:
            prompt: A string, list of chat messages, or dict with ``input_ids``
                or a text field.

        Returns:
            Dict with ``input_ids``, ``attention_mask`` (numpy arrays), and an
            optional ``prompt_text`` string, or None if the prompt could not be
            processed.
        """
        processor = self._get_processing_class()
        prompt_text: str | None = None
        encode_kwargs = dict(
            truncation=True,
            truncation_side="left",
            tokenize=True,
            padding="max_length",
            max_length=self.arguments.max_length,
            return_attention_mask=True,
            return_tensors="np",
            return_dict=True,
            padding_side="left",
            add_generation_prompt=True,
        )

        def _apply_chat_template(messages: list[dict[str, tp.Any]], **kwargs) -> tp.Any:
            if tools is not None:
                try:
                    return processor.apply_chat_template(messages, tools=tools, **kwargs)
                except TypeError as exc:
                    if "tools" not in str(exc):
                        raise
            return processor.apply_chat_template(messages, **kwargs)

        if isinstance(prompt, dict):
            sample_tools = tools if tools is not None else prompt.get("tools")
            if "input_ids" in prompt:
                input_ids = prompt["input_ids"]
                attention = prompt.get("attention_mask")
                prompt_text = prompt.get("prompt") or prompt.get("text")
            else:
                field = getattr(self.arguments, "generation_dataset_prompt_field", None)
                if field and field in prompt:
                    return self._prepare_generation_input(prompt[field], tools=sample_tools)
                for key in ("prompt", "text"):
                    if key in prompt:
                        return self._prepare_generation_input(prompt[key], tools=sample_tools)
                log_debug_maybe("Dataset sample missing `input_ids`/`prompt` keys for preview generation; skipping")
                return None
        elif isinstance(prompt, list):
            if processor is None or not hasattr(processor, "apply_chat_template"):
                logger.warning("No tokenizer/processor available; cannot tokenize chat prompt.")
                return None
            if prompt and not isinstance(prompt[0], dict):
                log_debug_maybe(f"Unsupported prompt list format for preview generation: {type(prompt[0])}")
                return None

            try:
                processor.padding_side = "left"
                encoded = _apply_chat_template(prompt, **encode_kwargs)
                try:
                    prompt_text = _apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                except Exception:  # pragma: no cover - best effort prompt display
                    prompt_text = str(prompt)
            except Exception as exc:  # pragma: no cover - tokenizer issues
                log_debug_maybe(f"Failed to tokenize generation chat prompt: {exc}")
                return None
            input_ids = encoded["input_ids"]
            attention = encoded.get("attention_mask")
        elif isinstance(prompt, str):
            if processor is None:
                logger.warning("No tokenizer/processor available; cannot tokenize prompt text.")
                return None
            prompt_text = prompt

            try:
                processor.padding_side = "left"
                messages = [{"role": "user", "content": prompt}]
                encoded = _apply_chat_template(messages, **encode_kwargs)
                if tools is not None:
                    try:
                        prompt_text = _apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception:  # pragma: no cover - best effort prompt display
                        prompt_text = prompt
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
        """Optionally run preview generation to monitor training progress.

        Uses `generate_unified` for consistent generation across both eSurge and compiled modes.
        """

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
        no_reasoning_message = "No reasoning content ..."
        no_tools_message = "No tools were called ..."

        def _preview_reasoning_entries(
            values: str | collections.abc.Sequence[tp.Any] | None,
            *,
            target_len: int,
        ) -> list[str]:
            normalized = self._coerce_optional_generation_texts(values, target_len=target_len)
            return [value if value not in (None, "") else no_reasoning_message for value in normalized]

        def _preview_tool_call_entries(
            values: collections.abc.Sequence[tp.Any] | tp.Any | None,
            *,
            target_len: int,
        ) -> list[str]:
            normalized = self._coerce_generation_metadata_list(values, target_len=target_len)
            entries: list[str] = []
            for value in normalized:
                if value in (None, "", []):
                    entries.append(no_tools_message)
                else:
                    entries.append(pprint.pformat(value, compact=True))
            return entries

        def _finalize_preview_results(records: list[dict[str, tp.Any]]) -> None:
            """Log and store completed preview generation results.

            Saves records to ``self.latest_generation_samples``, optionally
            logs them to Weights & Biases, and prints completions when
            ``generation_preview_print`` is enabled.

            Args:
                records: List of dicts, each containing ``prompt``,
                    ``completions``, and ``step`` keys.
            """
            if not records:
                return
            self.latest_generation_samples = records

            prompt_repr = "<prompt tokens>"
            record: dict[str, tp.Any] = {}
            for record in records:
                prompt_repr = record["prompt"] if record["prompt"] is not None else "<prompt tokens>"

            if wandb is not None and args.use_wandb and args.can_log_metrics and args.generation_log_to_wandb:
                if self.preview_log_table is None:
                    self.preview_log_table = wandb.Table(
                        columns=["step", "prompt", "completion_id", "completion", "reasoning", "tool_calls"],
                        log_mode="INCREMENTAL",
                    )
                for record in records:
                    prompt_repr = record["prompt"] if record["prompt"] is not None else "<prompt tokens>"
                    reasoning = record.get("reasoning", [])
                    tool_calls = record.get("tool_calls", [])
                    for idx, completion in enumerate(record["completions"]):
                        reasoning_entry = reasoning[idx] if idx < len(reasoning) else no_reasoning_message
                        tool_calls_entry = tool_calls[idx] if idx < len(tool_calls) else no_tools_message
                        self.preview_log_table.add_data(
                            step,
                            prompt_repr,
                            idx,
                            completion,
                            reasoning_entry,
                            tool_calls_entry,
                        )
                wandb.log({"preview_generations": self.preview_log_table}, step=step)
            if args.generation_preview_print:
                logger.info(f"[preview step {step}] prompt: {prompt_repr}")
                reasoning = record.get("reasoning", [])
                tool_calls = record.get("tool_calls", [])
                for idx, completion in enumerate(record["completions"]):
                    reasoning_entry = reasoning[idx] if idx < len(reasoning) else no_reasoning_message
                    tool_calls_entry = tool_calls[idx] if idx < len(tool_calls) else no_tools_message
                    logger.info(f"  completion[{idx}]: {completion}")
                    logger.info(f"  reasoning[{idx}]: {reasoning_entry}")
                    logger.info(f"  tool_calls[{idx}]: {tool_calls_entry}")

        prepared_prompts: list[dict[str, tp.Any]] = []
        for prompt in prompts:
            try:
                prepared = self._prepare_generation_input(prompt)
            except Exception as exc:  # pragma: no cover - preview should not break training
                log_debug_maybe(f"Preview generation failed while preparing prompt: {exc}")
                continue
            if prepared is not None:
                prepared_prompts.append(prepared)

        if not prepared_prompts:
            return

        pad_token_id = self._pad_token_id
        prompt_row_counts: list[int] = []
        prompt_row_offsets: list[int] = []
        input_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        max_prompt_len = 0
        running_offset = 0
        valid_prepared_prompts: list[dict[str, tp.Any]] = []

        for prepared in prepared_prompts:
            try:
                input_np = np.asarray(prepared["input_ids"], dtype=np.int32)
                mask_np = np.asarray(prepared["attention_mask"], dtype=np.int32)

                if input_np.ndim == 1:
                    input_np = input_np[None, :]
                if mask_np.ndim == 1:
                    mask_np = mask_np[None, :]

                if input_np.shape != mask_np.shape:
                    raise ValueError(
                        f"Prompt input_ids/attention_mask shape mismatch: {input_np.shape} vs {mask_np.shape}"
                    )
            except Exception as exc:  # pragma: no cover - preview should not break training
                log_debug_maybe(f"Skipping malformed preview prompt while batching: {exc}")
                continue

            rows = int(input_np.shape[0])
            prompt_row_counts.append(rows)
            prompt_row_offsets.append(running_offset)
            running_offset += rows
            max_prompt_len = max(max_prompt_len, int(input_np.shape[1]))
            input_rows.append(input_np)
            mask_rows.append(mask_np)
            valid_prepared_prompts.append(prepared)

        if not valid_prepared_prompts:
            return
        prepared_prompts = valid_prepared_prompts

        try:
            batched_input_rows: list[np.ndarray] = []
            batched_mask_rows: list[np.ndarray] = []
            for input_np, mask_np in zip(input_rows, mask_rows, strict=False):
                prompt_len = int(input_np.shape[1])
                pad_len = max_prompt_len - prompt_len
                if pad_len > 0:
                    input_np = np.pad(
                        input_np,
                        ((0, 0), (pad_len, 0)),
                        mode="constant",
                        constant_values=pad_token_id,
                    )
                    mask_np = np.pad(mask_np, ((0, 0), (pad_len, 0)), mode="constant", constant_values=0)
                batched_input_rows.append(input_np)
                batched_mask_rows.append(mask_np)

            batched_input_ids = jnp.asarray(np.concatenate(batched_input_rows, axis=0), dtype=jnp.int32)
            batched_attention_mask = jnp.asarray(np.concatenate(batched_mask_rows, axis=0), dtype=jnp.int32)
        except Exception as exc:  # pragma: no cover - preview should not break training
            log_debug_maybe(f"Preview generation failed while batching prompts: {exc}")
            return

        try:
            gen_results = self.generate_unified(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_mask,
                state=state,
                use_esurge=args.use_esurge_generation,
                apply_chat_template=False,  # Prompts are already formatted
                shard_inputs=args.generation_shard_inputs,
                all_gather=False,  # Keep on device for now
            )
        except Exception as exc:  # pragma: no cover - preview should not break training
            log_debug_maybe(f"Preview generation failed: {exc}")
            for prepared in prepared_prompts:
                try:
                    single_results = self.generate_unified(
                        input_ids=prepared["input_ids"],
                        attention_mask=prepared["attention_mask"],
                        state=state,
                        use_esurge=args.use_esurge_generation,
                        apply_chat_template=False,  # Prompts are already formatted
                        shard_inputs=args.generation_shard_inputs,
                        all_gather=False,  # Keep on device for now
                    )
                except Exception as single_exc:  # pragma: no cover - preview should not break training
                    log_debug_maybe(f"Preview generation failed for one prompt: {single_exc}")
                    continue

                single_completions = self._coerce_generation_texts(
                    single_results.text,
                    fallback=single_results.generation_results,
                )
                single_reasoning = _preview_reasoning_entries(
                    single_results.reasoning,
                    target_len=len(single_completions),
                )
                single_tool_calls = _preview_tool_call_entries(
                    single_results.tool_calls,
                    target_len=len(single_completions),
                )

                prompt_text = prepared.get("prompt_text")
                if prompt_text is None:
                    decoded = self._batch_decode_tokens(jax.device_get(single_results.prompt_ids))
                    if decoded:
                        prompt_text = decoded[0]

                results.append(
                    {
                        "prompt": prompt_text,
                        "completions": single_completions,
                        "reasoning": single_reasoning,
                        "tool_calls": single_tool_calls,
                        "step": step,
                    }
                )

            _finalize_preview_results(results)
            return

        all_completions = self._coerce_generation_texts(gen_results.text, fallback=gen_results.generation_results)
        all_reasoning = _preview_reasoning_entries(
            gen_results.reasoning,
            target_len=len(all_completions),
        )
        all_tool_calls = _preview_tool_call_entries(
            gen_results.tool_calls,
            target_len=len(all_completions),
        )

        num_return_sequences = max(int(args.generation_num_return_sequences or 1), 1)
        expected_total = sum(count * num_return_sequences for count in prompt_row_counts)
        if len(all_completions) != expected_total:
            log_debug_maybe(
                f"Preview generation completion count mismatch: expected {expected_total}, got {len(all_completions)}"
            )

        needs_decoding = any(prepared.get("prompt_text") is None for prepared in prepared_prompts)
        decoded_prompts: list[str] | None = None
        if needs_decoding:
            decoded_prompts = self._batch_decode_tokens(jax.device_get(gen_results.prompt_ids))

        completion_cursor = 0
        for prepared, rows, row_offset in zip(prepared_prompts, prompt_row_counts, prompt_row_offsets, strict=False):
            prompt_text = prepared.get("prompt_text")
            if prompt_text is None and decoded_prompts is not None and row_offset < len(decoded_prompts):
                prompt_text = decoded_prompts[row_offset]

            completion_count = rows * num_return_sequences
            completions = all_completions[completion_cursor : completion_cursor + completion_count]
            reasoning = all_reasoning[completion_cursor : completion_cursor + completion_count]
            tool_calls = all_tool_calls[completion_cursor : completion_cursor + completion_count]
            completion_cursor += completion_count
            results.append(
                {
                    "prompt": prompt_text,
                    "completions": completions,
                    "reasoning": reasoning,
                    "tool_calls": tool_calls,
                    "step": step,
                }
            )

        _finalize_preview_results(results)

    def _esurge_init_kwargs(self, *, max_num_seqs: int | None = None) -> dict[str, tp.Any]:
        """Build keyword arguments for initializing an eSurge inference runtime.

        Reads eSurge-related settings from ``self.arguments`` and assembles
        them into a kwargs dict suitable for constructing an eSurge engine.

        Args:
            max_num_seqs: Optional override for the maximum number of
                concurrent sequences. Falls back to trainer arguments or a
                computed default.

        Returns:
            Dict of eSurge initialization keyword arguments.
        """
        args = self.arguments
        esurge_kwargs: dict[str, tp.Any] = {}
        if args.esurge_hbm_utilization is not None:
            esurge_kwargs["hbm_utilization"] = args.esurge_hbm_utilization
        if args.esurge_max_num_seqs is not None:
            esurge_kwargs["max_num_seqs"] = args.esurge_max_num_seqs
        elif max_num_seqs is not None:
            esurge_kwargs["max_num_seqs"] = int(max_num_seqs)
        else:
            max_num_seqs = (args.generation_num_return_sequences or 1) * args.total_batch_size
            esurge_kwargs["max_num_seqs"] = max_num_seqs
        if args.esurge_min_input_pad is not None:
            esurge_kwargs["min_input_pad"] = args.esurge_min_input_pad
        if args.esurge_page_size is not None:
            esurge_kwargs["page_size"] = args.esurge_page_size
        if hasattr(args, "esurge_silent_mode"):
            esurge_kwargs["silent_mode"] = args.esurge_silent_mode
        if hasattr(args, "esurge_runner_verbose"):
            esurge_kwargs["runner_verbose"] = args.esurge_runner_verbose
        if args.esurge_max_num_batched_tokens is not None:
            esurge_kwargs["max_num_batched_tokens"] = args.esurge_max_num_batched_tokens
        else:
            esurge_kwargs["max_num_batched_tokens"] = min(args.max_length, 4096)
        if args.esurge_enable_prefix_caching is not None:
            esurge_kwargs["enable_prefix_caching"] = args.esurge_enable_prefix_caching
        if args.esurge_data_parallelism_axis is not None:
            esurge_kwargs["data_parallelism_axis"] = args.esurge_data_parallelism_axis
        if args.esurge_max_num_seq_buckets is not None:
            esurge_kwargs["max_num_seq_buckets"] = [int(v) for v in args.esurge_max_num_seq_buckets]
        processor = self._get_processing_class()
        if processor is not None:
            esurge_kwargs["tokenizer"] = processor
        return esurge_kwargs

    def maybe_benchmark(
        self,
        state: EasyDeLState,
        step: int,
    ) -> None:
        """Optionally run configured lm-eval benchmark suites during training."""

        args = self.arguments
        if args is None:
            return

        interval = args.benchmark_interval
        if interval is None:
            return
        if interval <= 0 or step % interval != 0:
            return

        benchmark_cfgs = normalize_benchmark_configs(args.benchmarks)
        if not benchmark_cfgs:
            return

        processor = self._get_processing_class()
        if processor is None:
            logger.warning("Skipping benchmark hook: no tokenizer/processing_class is attached to the trainer.")
            return

        model_config = getattr(state.model, "config", None)
        max_length = getattr(model_config, "granted_freq_max_position_embedding", None)
        if max_length is None:
            max_length = getattr(model_config, "max_position_embeddings", None)
        if max_length is None:
            max_length = args.max_length or 8192

        esurge_engine = None
        benchmark_results: dict[str, dict[str, tp.Any]] = {}
        flat_metrics: dict[str, float] = {}

        try:
            esurge_engine = state.model.get_esurge(**self._esurge_init_kwargs())
            fallback_batch_size = args.esurge_max_num_seqs or args.eval_batch_size or args.total_batch_size
            for benchmark in benchmark_cfgs:
                logger.info(f"[benchmark step {step}] running {benchmark.name}: {benchmark.tasks}")
                result = run_lm_eval_with_esurge(
                    surge=esurge_engine,
                    processor=processor,
                    tasks=benchmark.tasks,
                    max_length=int(max_length),
                    fallback_batch_size=fallback_batch_size,
                    eval_config=benchmark.eval_kwargs,
                    stop_engine=False,
                    summary_logger=logger,
                )
                benchmark_results[benchmark.name] = result
                flat_metrics.update(flatten_benchmark_metrics(benchmark.name, result))
        finally:
            try:
                state.model.pause_esurge(release_model_state=True, clear_compiled_cache=False)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                log_debug_maybe(f"Failed to release benchmark runtime: {exc}")

        if not benchmark_results:
            return

        self.latest_benchmark_results = benchmark_results
        self._log_benchmark_results_to_wandb(step=step, benchmark_results=benchmark_results)
        if flat_metrics:
            self.arguments.log_metrics(metrics=flat_metrics, step=step)

    def _log_benchmark_results_to_wandb(
        self,
        *,
        step: int,
        benchmark_results: dict[str, dict[str, tp.Any]],
    ) -> None:
        """Log benchmark summaries to W&B as an incremental table."""

        args = self.arguments
        if args is None or wandb is None or not args.use_wandb or not args.can_log_metrics:
            return

        if self.benchmark_log_table is None:
            self.benchmark_log_table = wandb.Table(
                columns=["step", "benchmark", "task", "metric", "value"],
                log_mode="INCREMENTAL",
            )

        rows_added = False
        for benchmark_name, result in benchmark_results.items():
            result_metrics = result.get("results", {})
            if not isinstance(result_metrics, collections.abc.Mapping):
                continue
            for task_name, metrics in result_metrics.items():
                if not isinstance(metrics, collections.abc.Mapping):
                    continue
                for metric_name, value in metrics.items():
                    numeric_value: float | None = None
                    if isinstance(value, bool):
                        numeric_value = float(value)
                    elif isinstance(value, int | float):
                        numeric_value = float(value)
                    if numeric_value is None:
                        continue
                    self.benchmark_log_table.add_data(
                        step, benchmark_name, str(task_name), str(metric_name), numeric_value
                    )
                    rows_added = True

        if rows_added:
            wandb.log({"benchmark_results": self.benchmark_log_table}, step=step)

    @staticmethod
    def _wandb_stringify_generation_value(value: tp.Any) -> str | None:
        """Convert prompt/completion metadata into a W&B-table-friendly string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    def _log_training_generations_to_wandb(
        self,
        *,
        state: EasyDeLState,
        prompts: tp.Any,
        completions: list[str] | tuple[str, ...] | None = None,
        prompt_mask: jax.Array | np.ndarray | None = None,
        completion_ids: jax.Array | np.ndarray | None = None,
        completion_mask: jax.Array | np.ndarray | None = None,
        completion_lengths: jax.Array | np.ndarray | None = None,
        generation_time: float | None = None,
        reasoning: list[tp.Any] | tuple[tp.Any, ...] | None = None,
        tool_calls: list[tp.Any] | tuple[tp.Any, ...] | None = None,
        source: str = "policy",
    ) -> None:
        """Log rollout generations used for training to an incremental W&B table."""
        args = self.arguments
        if (
            args is None
            or wandb is None
            or not args.use_wandb
            or not args.can_log_metrics
            or not args.log_training_generations_to_wandb
        ):
            return

        if self.training_generation_log_table is None:
            self.training_generation_log_table = wandb.Table(
                columns=[
                    "step",
                    "source",
                    "sample_idx",
                    "prompt",
                    "completion",
                    "completion_length",
                    "generation_time",
                    "reasoning",
                    "tool_calls",
                ],
                log_mode="INCREMENTAL",
            )

        if hasattr(prompts, "shape"):
            prompt_ids = np.asarray(jax.device_get(prompts))
            prompt_attention_mask = None if prompt_mask is None else np.asarray(jax.device_get(prompt_mask), dtype=np.int32)
            prompt_rows = self._decode_prompt_batch(
                self.processing_class,
                prompt_ids,
                skip_special_tokens=True,
                pad_token_id=self._pad_token_id,
                pop_pad_tokens=True,
                attention_mask=prompt_attention_mask,
            )
        elif isinstance(prompts, str):
            prompt_rows = [prompts]
        else:
            prompt_rows = [self._wandb_stringify_generation_value(prompt) or "<prompt>" for prompt in prompts]

        if completions is None:
            if completion_ids is None:
                return
            host_completion_ids = np.asarray(jax.device_get(completion_ids))
            host_completion_mask = (
                None if completion_mask is None else np.asarray(jax.device_get(completion_mask), dtype=np.int32)
            )
            completion_rows = self._decode_prompt_batch(
                self.processing_class,
                host_completion_ids,
                skip_special_tokens=True,
                pad_token_id=self._pad_token_id,
                pop_pad_tokens=True,
                attention_mask=host_completion_mask,
            )
        elif isinstance(completions, str):
            completion_rows = [completions]
        else:
            completion_rows = [
                self._wandb_stringify_generation_value(completion) or "<completion>" for completion in completions
            ]
        if not completion_rows and completion_ids is not None:
            host_completion_ids = np.asarray(jax.device_get(completion_ids))
            host_completion_mask = (
                None if completion_mask is None else np.asarray(jax.device_get(completion_mask), dtype=np.int32)
            )
            completion_rows = self._decode_prompt_batch(
                self.processing_class,
                host_completion_ids,
                skip_special_tokens=True,
                pad_token_id=self._pad_token_id,
                pop_pad_tokens=True,
                attention_mask=host_completion_mask,
            )

        if completion_lengths is None and completion_mask is not None:
            completion_lengths = jnp.sum(jnp.asarray(completion_mask), axis=-1)
        length_rows = None if completion_lengths is None else np.asarray(jax.device_get(completion_lengths)).reshape(-1)

        cur_step = int(jax.device_get(state.step))
        reasoning = list(reasoning or [])
        tool_calls = list(tool_calls or [])

        for idx, (prompt, completion) in enumerate(zip(prompt_rows, completion_rows, strict=False)):
            reason = reasoning[idx] if idx < len(reasoning) else None
            tools = tool_calls[idx] if idx < len(tool_calls) else None
            completion_length = None if length_rows is None or idx >= len(length_rows) else float(length_rows[idx])
            self.training_generation_log_table.add_data(
                cur_step,
                source,
                idx,
                prompt,
                completion,
                completion_length,
                generation_time,
                self._wandb_stringify_generation_value(reason),
                self._wandb_stringify_generation_value(tools),
            )
        wandb.log({"training_generations": self.training_generation_log_table}, step=cur_step)

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

                self._apply_step_start_point_to_optimizer_state()
                rules = self.model._get_partition_rules(None)
                self.model_state = self.model_state.shard_state(partition_rules=rules, mesh=self.model.mesh)
                self.state_shardings = self.model_state.shardings

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

    def _create_dataloader_from_source(
        self,
        source: "ShardedDataSource",
        batch_size: int,
        is_train: bool = True,
        shuffle: bool = False,
        num_epochs: int = 1,
        drop_remainder: bool = True,
    ) -> collections.abc.Iterator:
        """Create dataloader iterator from ShardedDataSource.

        Iterates over the transformed source (with tokenization applied).
        Yields lists of examples (batches) that will be collated by the
        training loop's data_collator.

        Args:
            source: ShardedDataSource to iterate over.
            batch_size: Batch size for batching examples.
            is_train: Whether this is for training (affects logging).
            shuffle: Whether to shuffle (currently not implemented for sources).
            num_epochs: Number of epochs to iterate.
            drop_remainder: Whether to drop the last incomplete batch.

        Yields:
            Lists of examples (pre-tokenized dicts) to be collated.
        """
        for _ in range(num_epochs):
            batch = []
            for shard_name in source.shard_names:
                for example in source.open_shard(shard_name):
                    batch.append(example)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            # Handle remainder
            if batch and not drop_remainder:
                yield batch

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
            from datasets import IterableDataset  # pyright: ignore[reportMissingTypeStubs]

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
                max_sequence_length=self.arguments.max_length,
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
            """Compute the total number of training or evaluation steps.

            Uses forced step counts from arguments when available, otherwise
            derives the count from dataset length, batch size, and epoch count.

            Args:
                dataset: The training or evaluation dataset.
                is_train: Whether the calculation is for training steps.

            Returns:
                Total number of optimizer steps for the run.

            Raises:
                ValueError: If dataset length cannot be determined and no
                    per-epoch step count is configured.
            """
            forced_steps = self.arguments.max_training_steps if is_train else self.arguments.max_evaluation_steps
            total_data_len: int | None = None
            if forced_steps is None:
                try:
                    total_data_len = len(dataset)
                except TypeError as e:
                    total_data_len = (
                        self.arguments.per_epoch_training_steps
                        if is_train
                        else self.arguments.per_epoch_evaluation_steps
                    )
                    if total_data_len is None:
                        raise ValueError(
                            f"Specify the number of per epoch {'training' if is_train else 'evaluation'} "
                            "steps for a generator/streaming dataset."
                        ) from e

            batch_size = self.arguments.total_batch_size if is_train else self.evaluation_batch_size
            num_epochs = self.arguments.num_train_epochs if is_train else 1
            return resolve_total_steps(
                forced_steps=forced_steps,
                total_data_len=total_data_len,
                batch_size=batch_size,
                num_epochs=num_epochs,
                gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
                is_train=is_train,
            )

        max_training_steps = calculate_steps(self.dataset_train, is_train=True)

        # Use _train_source if available (has transforms applied)
        if self._train_source is not None:
            dataloader_train = self._create_dataloader_from_source(
                source=self._train_source,
                batch_size=self.training_batch_size,
                is_train=True,
                shuffle=self.arguments.shuffle_train_dataset,
                num_epochs=self.arguments.num_train_epochs,
                drop_remainder=True,
            )
        else:
            dataloader_train = _create_grain_dataloader(self.dataset_train, is_train=True)

        dataloader_eval, max_evaluation_steps = None, 0
        if self.dataset_eval is not None and self.arguments.do_eval:
            max_evaluation_steps = calculate_steps(self.dataset_eval, is_train=False)
            # Use _eval_source if available (has transforms applied)
            if self._eval_source is not None:
                dataloader_eval = self._create_dataloader_from_source(
                    source=self._eval_source,
                    batch_size=self.evaluation_batch_size,
                    is_train=False,
                    shuffle=False,
                    num_epochs=1,
                    drop_remainder=True,
                )
            else:
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
                if device.device_type == "GPU":
                    logger.warning("TensorFlow may be hogging GPU memory.")
                    break
        except RuntimeError as e:
            # Invalid device or cannot modify virtual devices once initialized.
            logger.error(f"Failed to disable GPU devices: {e}")

        def create_tf_dataset(dataset: Dataset, is_train: bool) -> collections.abc.Iterator[np.ndarray]:
            """
            Creates a TensorFlow dataset from a Hugging Face Dataset.

            Args:
                dataset (Dataset): The Hugging Face Dataset.
                is_train (bool): Whether the dataset is for training.

            Returns:
                collections.abc.Iterator[np.ndarray]: The TensorFlow dataset iterator.
            """

            batch_size = self.training_batch_size if is_train else self.evaluation_batch_size

            return (
                dataset.to_tf_dataset(
                    collate_fn=self.create_tfds_collect_function(
                        max_sequence_length=self.arguments.max_length,
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

        def create_tf_dataset_from_iterable(
            dataset: IterableDataset, is_train: bool
        ) -> collections.abc.Iterator[np.ndarray]:
            """
            Creates a TensorFlow dataset from an iterable Hugging Face Dataset.

            Args:
                dataset (IterableDataset): The iterable Hugging Face Dataset.
                is_train (bool): Whether the dataset is for training.

            Returns:
                collections.abc.Iterator[np.ndarray]: The TensorFlow dataset iterator.
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
                            shape=(
                                vals.shape[1:]
                                if len(vals.shape) > 1 and vals.shape[0] == 1  # auto remove batch dim
                                else vals.shape
                            ),
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
            forced_steps = self.arguments.max_training_steps if is_train else self.arguments.max_evaluation_steps
            total_data_len: int | None = None
            if forced_steps is None:
                try:
                    total_data_len = len(dataset)
                except TypeError as e:
                    total_data_len = (
                        self.arguments.per_epoch_training_steps
                        if is_train
                        else self.arguments.per_epoch_evaluation_steps
                    )
                    if total_data_len is None:
                        raise ValueError(
                            f"Specify the number of per epoch {'training' if is_train else 'evaluation'} "
                            "steps for a generator/streaming dataset."
                        ) from e

            batch_size = self.arguments.total_batch_size if is_train else self.evaluation_batch_size
            num_epochs = self.arguments.num_train_epochs if is_train else 1
            return resolve_total_steps(
                forced_steps=forced_steps,
                total_data_len=total_data_len,
                batch_size=batch_size,
                num_epochs=num_epochs,
                gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
                is_train=is_train,
            )

        def to_tf_dataloader(dataset: Dataset | IterableDataset, is_train: bool) -> collections.abc.Iterator[np.ndarray]:
            """
            Converts a Hugging Face Dataset to a TensorFlow dataloader.

            Args:
                dataset (tp.Union[Dataset, IterableDataset]): The Hugging Face Dataset.
                is_train (bool): Whether the dataset is for training.

            Returns:
                collections.abc.Iterator[np.ndarray]: The TensorFlow dataloader iterator.
            """
            if hasattr(dataset, "__len__"):
                return create_tf_dataset(dataset, is_train)
            else:
                return create_tf_dataset_from_iterable(dataset, is_train)

        max_training_steps = calculate_steps(self.dataset_train, is_train=True)

        # Use _train_source if available (has transforms applied)
        if self._train_source is not None:
            dataloader_train = self._create_dataloader_from_source(
                source=self._train_source,
                batch_size=self.training_batch_size,
                is_train=True,
                shuffle=self.arguments.shuffle_train_dataset,
                num_epochs=self.arguments.num_train_epochs,
                drop_remainder=True,
            )
        else:
            dataloader_train = to_tf_dataloader(self.dataset_train, is_train=True)

        if self.dataset_eval is not None and self.arguments.do_eval:
            max_evaluation_steps = calculate_steps(self.dataset_eval, is_train=False)
            # Use _eval_source if available (has transforms applied)
            if self._eval_source is not None:
                dataloader_eval = self._create_dataloader_from_source(
                    source=self._eval_source,
                    batch_size=self.evaluation_batch_size,
                    is_train=False,
                    shuffle=False,
                    num_epochs=1,
                    drop_remainder=True,
                )
            else:
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

    def _sync_checkpointer_after_callback_save(self, step: int) -> None:
        """Mirror checkpointer bookkeeping for callback-driven saves.

        EasyDeL uses ``Checkpointer.on_step(..., pytree=None, true_callbacks=[...])``
        so the external checkpointer decides *when* to save while the trainer callback
        performs the actual state serialization. In the pinned eformer implementation,
        the internal ``_last_save_step`` / ``_last_save_time`` fields are only updated
        when ``save_checkpoint`` is invoked with a real pytree. Keep those fields in
        sync here so step/time policies continue to behave correctly.
        """
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None:
            return
        try:
            if hasattr(checkpointer, "_last_save_step"):
                checkpointer._last_save_step = int(step)
            now_factory = getattr(checkpointer, "_dt_now_injection", None)
            if callable(now_factory) and hasattr(checkpointer, "_last_save_time"):
                checkpointer._last_save_time = now_factory()
        except Exception as exc:
            logger.warning(f"Failed to synchronize checkpointer bookkeeping after save: {exc}")

    def _save_checkpoint_for_step(
        self,
        state: EasyDeLState,
        *,
        step: int,
        force: bool = False,
    ) -> str | None:
        """Run checkpointer policy evaluation and serialize the current trainer state."""
        saved_directory = [None]

        def save_callback(dest, mesh, meta, s=state):
            """Checkpointer callback that serializes trainer state to disk.

            Args:
                dest: Relative checkpoint destination path.
                mesh: Device mesh (unused, required by checkpointer API).
                meta: Checkpoint metadata dict (unused, required by API).
                s: Model state to save (defaults to outer ``state``).
            """
            full_path = str(self.arguments._get_save_directory() / dest)
            saved_directory[0] = self._save_state(state=s, save_directory=full_path)
            self._cleanup_old_checkpoints()

        self.checkpointer.on_step(
            mesh=self.mesh,
            pytree=None,
            step=step,
            force=force,
            true_callbacks=[save_callback],
        )
        if saved_directory[0] is not None:
            self._sync_checkpointer_after_callback_save(step=step)
        return saved_directory[0]

    def _maybe_remove_loaded_checkpoint(self, checkpoint_path: str) -> None:
        """Delete a checkpoint after load when explicitly requested."""
        if not getattr(self.arguments, "remove_ckpt_after_load", False):
            return
        try:
            if jax.process_count() > 1:
                from jax.experimental import multihost_utils as mh

                mh.sync_global_devices("easydel.remove_ckpt_after_load.before")
            if jax.process_index() == 0:
                import fsspec

                fs, plain_path = fsspec.core.url_to_fs(str(checkpoint_path))
                logger.info(f"Removing checkpoint after load: {checkpoint_path}")
                fs.rm(plain_path, recursive=True)
            if jax.process_count() > 1:
                from jax.experimental import multihost_utils as mh

                mh.sync_global_devices("easydel.remove_ckpt_after_load.after")
        except Exception as exc:
            logger.warning(f"Failed to remove checkpoint after load {checkpoint_path}: {exc}")

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
            """Return the modification time of a checkpoint directory.

            Args:
                path: Path-like object pointing to a checkpoint directory.

            Returns:
                Modification timestamp as a float, or 0 on failure.
            """
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
            int: Current step number from the training state.
        """
        return int(jax.device_get(state.step))

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
            return pprint.pformat(self.model._get_partition_rules(None), indent=2, width=80)
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
- Sequence Length: {self.arguments.max_length}
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
        to_torch: bool = False,
        easystate_to_huggingface_model_kwargs: dict | None = None,
        torch_save_pretrained_kwargs: dict | None = None,
    ):
        """Save the model in either EasyDeL state format or PyTorch format.

        Args:
            state: The model state to save.
            save_directory: Directory to save to. If None, uses default from arguments.
            to_torch: If True, converts and saves as a HuggingFace PyTorch model.
            easystate_to_huggingface_model_kwargs: Extra kwargs for EasyDeL-to-HF conversion.
            torch_save_pretrained_kwargs: Extra kwargs for HF save_pretrained.

        Returns:
            str or HuggingFace model: Path to saved checkpoint (EasyDeL) or HF model instance (PyTorch).
        """
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
                save_directory=save_directory,
            )

    def _save_to_torch(
        self,
        state: EasyDeLState,
        save_directory: str | os.PathLike,
        easystate_to_huggingface_model_kwargs: dict | None = None,
        torch_save_pretrained_kwargs: dict | None = None,
    ):
        """Convert the JAX model to a PyTorch HuggingFace model and save it.

        Args:
            state: Current EasyDeL model state.
            save_directory: Filesystem path where the model will be saved.
            easystate_to_huggingface_model_kwargs: Extra kwargs passed to
                ``model.to_torch()``.
            torch_save_pretrained_kwargs: Extra kwargs passed to the HF
                model's ``save_pretrained``.

        Returns:
            The converted HuggingFace PyTorch model.
        """
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
        """Create a HuggingFace ``AutoConfig`` populated from the EasyDeL model config.

        Copies compatible attributes from the EasyDeL config onto a fresh HF
        config instance, converting numeric strings to numbers.

        Args:
            state: Current EasyDeL model state.
            model_config: The EasyDeL model configuration object.
            model_type: HuggingFace model type identifier string.

        Returns:
            A HuggingFace ``PretrainedConfig`` instance.
        """
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
        """Apply training hooks to check for stopping conditions.

        Checks for NaN loss (if configured) and time limits, raising appropriate
        exceptions to interrupt training when conditions are met.

        Args:
            metrics: The loss metrics from the current training step.

        Returns:
            LossMetrics: The unmodified metrics if no stopping condition is triggered.

        Raises:
            EasyDeLBreakRequest: If NaN loss is detected and break_on_nan is enabled.
            EasyDeLTimerError: If training has exceeded the configured time limit.
        """
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

    def _setup_static_metrics(self):
        """Initialize static training/evaluation metrics.

        Subclasses should override this to set up metric accumulators
        before training or evaluation begins.
        """
        ...

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
            """Lower and AOT-compile a step function if not already compiled.

            Args:
                function: The JIT-wrapped step function to compile.
                dataloader: Dataloader providing a sample batch for shape inference.
                state: Current model state passed as the first compilation arg.
                tag: Human-readable label used in log messages.

            Returns:
                The compiled function, or the original if already compiled.
            """
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
        """Finalize training by handling exceptions and saving the last checkpoint.

        Processes any exception that terminated the training loop (e.g.,
        ``KeyboardInterrupt``, timer limit), performs a final checkpoint save
        if configured, and assembles the ``TrainerOutput``.

        Args:
            state: The model state at the end of training.
            run_exception: Exception that stopped training, or None for a
                clean finish.

        Returns:
            A ``TrainerOutput`` containing the final state and checkpoint path.

        Raises:
            RuntimeError: If the exception is not a recognized graceful
                interruption.
        """
        if run_exception is not None:
            if isinstance(run_exception, KeyboardInterrupt):
                logger.warning("KeyboardInterrupt: Training interrupted. Saving current state...")
            elif isinstance(run_exception, EasyDeLTimerError):
                logger.warning("Training reached maximum time limit. Saving current state...")
            elif isinstance(run_exception, EasyDeLPreemptionSignal):
                ...  # simply just pass
            elif self._is_memory_oom_exception(run_exception):
                raise run_exception
            else:
                raise RuntimeError(f"EasyDeL Runtime dumped due to {run_exception!s}") from run_exception
        checkpoint_path = "SAVING_SKIPPED"
        filename = None

        if self._preemption_checkpoint_path is not None:
            checkpoint_path = str(self._preemption_checkpoint_path)
            filename = str(self._preemption_checkpoint_path)
        elif self.arguments.do_last_save:
            dire = ePath(self.arguments.save_directory)
            current_step = int(jax.device_get(state.step))
            filename = self._save_checkpoint_for_step(
                state=state,
                step=current_step,
                force=True,
            )
            if filename is not None:
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
            RuntimeError: If the dataloader is empty and cannot provide batches.
        """
        try:
            batch = next(data_iter)
        except (StopIteration, IndexError):
            data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration as exc:
                raise RuntimeError("Dataloader is empty and cannot provide batches.") from exc

        # Remove specified ids from batch if needed
        for id_to_pop in self.arguments.ids_to_pop_from_dataset or []:
            _ = batch.pop(id_to_pop, None)

        return batch, data_iter

    def _fast_forward_batches(self, data_iter, dataloader, num_batches: int):
        """Advance an iterator by discarding a fixed number of batches.

        This mirrors the normal training-time iterator semantics, including
        automatic reinitialization when a finite dataloader is exhausted.
        """
        num_batches = max(int(num_batches), 0)
        if num_batches > 10_000:
            logger.warning(
                f"Fast-forwarding dataloader by {num_batches} batches. This may take a while for large step counts."
            )
        if num_batches == 0:
            return data_iter
        builtin_sequence_iterators = (type(iter([])), type(iter(())), type(iter(range(0))))
        if isinstance(dataloader, collections.abc.Sequence) and isinstance(data_iter, builtin_sequence_iterators):
            total_batches = len(dataloader)
            if total_batches == 0:
                raise RuntimeError("Dataloader is empty and cannot be fast-forwarded.")
            consumed_batches = total_batches - max(operator.length_hint(data_iter), 0)
            target_index = (consumed_batches + num_batches) % total_batches
            return itertools.islice(iter(dataloader), target_index, None)
        remaining = num_batches
        while remaining > 0:
            skipped = sum(1 for _ in itertools.islice(data_iter, remaining))
            remaining -= skipped
            if remaining <= 0:
                break
            data_iter = iter(dataloader)
            try:
                next(data_iter)
            except StopIteration as exc:
                raise RuntimeError("Dataloader is empty and cannot be fast-forwarded.") from exc
            remaining -= 1
        return data_iter

    def _should_enable_tpu_preemption_checkpointing(self) -> bool:
        """Return whether TPU preemption-triggered checkpointing is enabled."""
        if not self.arguments.save_tpu_preemption_checkpoints:
            return False
        try:
            return jax.default_backend() == "tpu"
        except Exception:
            return False

    def _should_save_tpu_preemption_checkpoint(self, step: int) -> bool:
        """Return True when JAX's preemption sync service reaches a safe save step."""
        if not self._should_enable_tpu_preemption_checkpointing():
            return False
        if self._tpu_preemption_sync_available is False:
            return False
        try:
            from jax.experimental import multihost_utils

            should_save = bool(multihost_utils.reached_preemption_sync_point(int(step)))
            self._tpu_preemption_sync_available = True
            return should_save
        except RuntimeError as exc:
            self._tpu_preemption_sync_available = False
            logger.warning_once(
                "TPU preemption checkpointing requested but JAX preemption sync is unavailable. "
                "Ensure `jax_enable_preemption_service` is enabled before distributed initialization. "
                f"Disabling feature for this run. Original error: {exc}"
            )
            return False

    def _save_tpu_preemption_checkpoint(self, state: EasyDeLState, step: int) -> str | None:
        """Save a coordinated TPU preemption checkpoint using standard trainer naming."""
        from jax.experimental import multihost_utils

        sync_prefix = f"tpu-preemption-save-{int(step)}"
        multihost_utils.sync_global_devices(sync_prefix + "-start")
        saved_path = self._save_checkpoint_for_step(
            state=state,
            step=step,
            force=True,
        )
        multihost_utils.sync_global_devices(sync_prefix + "-done")
        if saved_path is not None:
            self._preemption_checkpoint_path = saved_path
        return saved_path

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

    def log_watchers(self, state: EasyDeLState, step: int):
        """Run registered LogWatcher instances and log their metrics.

        Args:
            state: Model state containing parameters.
            step: Current training step.

        Notes:
            Delegates to arguments.log_watchers method.
        """
        return self.arguments.log_watchers(state=state, step=step)

    def log_metrics(
        self,
        metrics: MetricsType,
        pbar: BaseProgressBar,
        step: int,
        mode: str = "train",
    ):
        """Log metrics to configured backends and update the progress bar.

        Metrics are logged at intervals defined by log_steps (for progress bar)
        and report_steps (for W&B/TensorBoard). MLPerf and grad_norm metrics
        are filtered from the progress bar display.

        Args:
            metrics: Dictionary of metric names to values.
            pbar: Progress bar instance to update.
            step: Current training/evaluation step.
            mode: Either 'train' or 'eval' to prefix metrics.
        """

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
