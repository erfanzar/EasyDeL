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

import ast
import collections.abc
import datetime
import functools
import json
import os
import re
import traceback
import typing as tp
import warnings
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, fields

import flax.nnx
import jax
import jax.numpy as jnp
import numpy as np
from eformer.loggings import get_logger
from eformer.optimizers import OptimizerFactory, SchedulerConfig
from eformer.paths import ePath, ePathLike
from eformer.serialization import Checkpointer
from jax.sharding import PartitionSpec
from optax import GradientTransformation  # pyright: ignore[reportMissingTypeStubs]

from easydel.infra.elarge.benchmarking import normalize_benchmark_configs
from easydel.infra.elarge.types import BenchmarkConfig
from easydel.infra.errors import EasyDeLTimerError
from easydel.infra.etils import (
    AVAILABLE_OPTIMIZERS,
    AVAILABLE_SCHEDULERS,
    AVAILABLE_SPARSE_MODULE_TYPES,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
)
from easydel.infra.loss_utils import LossConfig
from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from .metrics import LogWatcher, MetricsHistogram, compute_weight_stats, run_watchers
from .utils import JaxDistributedConfig

try:
    import wandb
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from flax.metrics.tensorboard import SummaryWriter
    from jax import Array
    from torch import Tensor
else:
    Array, Tensor = [tp.Any] * 2


MetricsType = dict[str, float | list | tuple | np.ndarray | Array | Tensor | None]
logger = get_logger(__name__)

QuantizationMode = tp.Literal[
    "nf4",
    "affine",
    "mxfp8",
    "nvfp8",
    "mxfp4",
    "nvfp4",
]
STE_QAT_QUANTIZATION_MODES: tuple[QuantizationMode, ...] = tp.get_args(QuantizationMode)
STE_QAT_QUANTIZATION_MODES_DOC = ", ".join(f"'{mode}'" for mode in STE_QAT_QUANTIZATION_MODES)
AFFINE_SUPPORTED_BITS = frozenset({2, 3, 4, 5, 6, 7, 8})
FIXED_QUANTIZATION_BITS_BY_MODE: dict[QuantizationMode, int] = {
    "nf4": 4,
    "mxfp4": 4,
    "nvfp4": 4,
    "mxfp8": 8,
    "nvfp8": 8,
}


def get_safe_arr(xs):
    """Convert single-element arrays to Python scalars for safe logging.

    Args:
        xs: Value to normalize. Can be a numpy/JAX array or scalar.

    Returns:
        Python scalar if input is a size-1 array, otherwise the original value.
    """
    if isinstance(xs, np.generic | jax.Array):
        if xs.size == 1:  # Only try .item() on size-1 arrays
            return xs.item()
        return xs
    return xs


def _normalize_partition_spec_entry(value: tp.Any) -> tp.Any:
    """Recursively convert lists to tuples within a partition spec entry.

    Args:
        value: A partition spec entry that may contain nested lists or tuples.

    Returns:
        The entry with all lists converted to tuples, preserving other types as-is.
    """
    if isinstance(value, list):
        return tuple(_normalize_partition_spec_entry(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_partition_spec_entry(item) for item in value)
    return value


def _parse_partition_spec(value: tp.Any) -> PartitionSpec:
    """Parse a value into a JAX PartitionSpec.

    Handles PartitionSpec instances, None, string representations (including
    ``"PartitionSpec(...)"`` format), and list/tuple inputs.

    Args:
        value: The value to parse. Can be a PartitionSpec, None, a string
            representation, or a list/tuple of axis names.

    Returns:
        A PartitionSpec constructed from the parsed value.

    Raises:
        ValueError: If a string value cannot be parsed into a valid spec.
    """
    if isinstance(value, PartitionSpec):
        return value
    if value is None:
        return PartitionSpec()

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return PartitionSpec()
        try:
            if stripped.startswith("PartitionSpec(") and stripped.endswith(")"):
                inner = stripped[len("PartitionSpec(") : -1].strip()
                parsed = [] if not inner else ast.literal_eval(f"[{inner}]")
            else:
                parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid `step_partition_spec` value: {value!r}") from exc

    if isinstance(parsed, PartitionSpec):
        return parsed
    if isinstance(parsed, (list, tuple)):
        return PartitionSpec(*tuple(_normalize_partition_spec_entry(item) for item in parsed))
    return PartitionSpec(_normalize_partition_spec_entry(parsed))


def _apply_training_args_legacy_aliases(data: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Rewrite deprecated field names in a config dict to their current equivalents.

    Translates ``quantization_block`` to ``quantization_group_size`` when the
    new name is not already present, then removes the deprecated key.

    Args:
        data: A configuration dictionary, typically from deserialized JSON.

    Returns:
        A new dictionary with deprecated aliases replaced by current field names.
    """
    data = dict(data)
    if "quantization_block" in data:
        if "quantization_group_size" not in data:
            data["quantization_group_size"] = data["quantization_block"]
        # Drop deprecated alias for constructor calls built from serialized configs.
        data.pop("quantization_block", None)
    return data


# Constants
AVAILABLE_BACKENDS: list[str] = ["cpu", "gpu", "tpu", None]


@Registry.register("trainer-arguments", "base")
@dataclass
class TrainingArguments:
    """
    Comprehensive configuration class for training and evaluation.

    This class encapsulates all training hyperparameters, optimization settings,
    data loading configuration, logging preferences, and hardware-specific options.
    It provides a centralized way to manage the complex configuration required for
    distributed training of large models.

    The configuration covers:
    - Training hyperparameters (learning rate, batch size, epochs)
    - Optimization settings (optimizer, scheduler, gradient clipping)
    - Data loading (dataset configuration, batch collation)
    - Checkpointing (save frequency, checkpoint limits)
    - Logging (WandB, TensorBoard, metrics reporting)
    - Hardware configuration (sharding, precision, device placement)
    - Performance optimization (compilation, memory tracking)

    Example:
        >>> args = TrainingArguments(
        ...     learning_rate=1e-4,
        ...     num_train_epochs=3,
        ...     total_batch_size=32,
        ...     save_steps=1000,
        ...     use_wandb=True
        ... )
    """

    auto_shard_states: bool = field(
        default=True,
        metadata={"help": "Whether to automatically shard model states across devices."},
    )
    aux_loss_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable the auxiliary loss."},
    )
    backend: str | None = field(
        default=None,
        metadata={"help": "The JAX backend to use (e.g., 'cpu', 'gpu', 'tpu').  If None, JAX will choose."},
    )
    clip_grad: float | None = field(
        default=None,
        metadata={"help": "The value at which to clip the gradients."},
    )
    custom_scheduler: tp.Callable[[int], tp.Any] | None = field(
        default=None,
        metadata={"help": "A custom scheduler function that takes the current step as input."},
    )
    dataloader_num_workers: int | None = field(
        default=0,
        metadata={"help": "The number of workers to use for the dataloader."},
    )
    dataloader_pin_memory: bool | None = field(
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
    eval_batch_size: int | None = field(
        default=None,
        metadata={"help": "The batch size to use for evaluation."},
    )
    evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "Run evaluation every X steps."},
    )
    extra_optimizer_kwargs: dict | None = field(
        default_factory=dict,
        metadata={"help": "Additional keyword arguments to pass to the optimizer."},
    )
    frozen_parameters: str | None = field(
        default=None,
        metadata={"help": "A regex pattern of parameters to freeze (not train)."},
    )
    grain_shard_index: int | None = field(
        default=None,
        metadata={"help": "sharding index to be used for grain dataloaders in both train and eval steps."},
    )
    grain_shard_count: int | None = field(
        default=None,
        metadata={"help": "sharding count to be used for grain dataloaders in both train and eval steps."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "The number of steps to accumulate gradients over."},
    )
    ids_to_pop_from_dataset: list[str] | None = field(
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
    jax_distributed_config: dict | None = field(
        default=None,
        metadata={"help": "Configuration for JAX distributed training."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The learning rate."},
    )
    learning_rate_end: float | None = field(
        default=None,
        metadata={"help": "The final learning rate for linear decay schedulers."},
    )
    log_all_workers: bool = field(
        default=False,
        metadata={"help": "Whether to log metrics from all workers in a distributed setup."},
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
    loss_config: LossConfig | None = field(
        default=None,
        metadata={"help": "Configuration for the loss function."},
    )
    lmhead_chunksize: int | None = field(
        default=None,
        metadata={"help": "Optional token chunk size for LM-head projection during training/inference."},
    )
    quantization_mode: QuantizationMode | None = field(
        default=None,
        metadata={
            "help": (
                "Quantization mode for quantization-aware training (QAT). "
                f"Supported values: {STE_QAT_QUANTIZATION_MODES_DOC}. "
                "When set (or when a straight-through callable is provided), trainers can apply a straight-through "
                "estimator (STE) transform to `state.graphstate` for the forward pass without permanently modifying "
                "the stored parameters."
            )
        },
    )
    quantization_group_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Quantization group size for group-wise quantizers (e.g. NF4/AFFINE/MXFP*/NVFP*). "
                "If None, the default group size for the selected quantization mode is used."
            )
        },
    )
    quantization_bits: int | None = field(
        default=None,
        metadata={
            "help": (
                "Quantization bit-width for QAT/STE quantizers. For `affine`, supported bits are 2..8. "
                "For fixed-width formats (`nf4`, `mxfp4`, `nvfp4`, `mxfp8`, `nvfp8`), when provided it must "
                "match the format width."
            )
        },
    )
    tensor_straight_through: tp.Callable[[jax.Array], jax.Array] | None = field(
        default=None,
        metadata={
            "help": (
                "Per-tensor straight-through transform used for QAT (e.g. tensor -> quantize(mode) -> dequantize "
                "with identity gradients). If `straight_through_emulator` is not provided, this callable can be "
                "mapped over `state.graphstate` via `jax.tree_util.tree_map`."
            )
        },
    )
    straight_through_emulator: tp.Callable[[flax.nnx.GraphState], flax.nnx.GraphState] | None = field(
        default=None,
        metadata={
            "help": (
                "Callable that maps a graphstate pytree -> new graphstate pytree for the forward pass. "
                "If None, and quantization is enabled, trainers can default to applying "
                "`jax.tree_util.tree_map(tensor_straight_through, graphstate)`."
            )
        },
    )
    low_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether to try to minimize memory usage."},
    )
    max_evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "Maximum number of evaluation steps."},
    )
    max_length: int | None = field(
        default=None,
        metadata={"help": "The maximum sequence length."},
    )
    max_sequence_length: InitVar[int | None] = None
    quantization_block: InitVar[int | None] = None
    max_training_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of training steps."},
    )
    per_epoch_training_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of training step per each epoch."},
    )
    per_epoch_evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of evaluation step per each epoch."},
    )
    model_name: str | None = field(
        default=None,
        metadata={"help": "The name of the model."},
    )
    model_parameters: dict | None = field(
        default=None,
        metadata={"help": "Model architecture config"},
    )
    metrics_to_show_in_rich_pbar: list[str] | None = field(
        default=None,
        metadata={"help": "Metrics to display in the rich progress bar."},
    )
    generation_top_p: float | None = field(
        default=None,
        metadata={"help": "Default nucleus sampling threshold used for preview generations."},
    )
    generation_top_k: int | None = field(
        default=None,
        metadata={"help": "Default top-k sampling threshold used for preview generations."},
    )
    generation_presence_penalty: float | None = field(
        default=None,
        metadata={"help": "Default presence penalty used for preview generations."},
    )
    generation_frequency_penalty: float | None = field(
        default=None,
        metadata={"help": "Default frequency penalty used for preview generations."},
    )
    generation_repetition_penalty: float | None = field(
        default=None,
        metadata={"help": "Default repetition penalty used for preview generations."},
    )
    generation_temperature: float | None = field(
        default=None,
        metadata={"help": "Default sampling temperature for preview generations."},
    )
    generation_do_sample: bool | None = field(
        default=None,
        metadata={"help": "Whether to enable sampling when generating previews (auto-inferred when None)."},
    )
    generation_num_return_sequences: int | None = field(
        default=None,
        metadata={"help": "Number of completions to sample per prompt for preview generations."},
    )
    generation_max_new_tokens: int | None = field(
        default=None,
        metadata={"help": "Maximum number of newly generated tokens for previews."},
    )
    generation_shard_inputs: bool = field(
        default=True,
        metadata={"help": "Whether generation previews should reuse the model's sharding plan."},
    )
    generation_interval: int | None = field(
        default=None,
        metadata={"help": "Run preview generation every X training steps (disabled when None)."},
    )
    generation_prompts: list[str | dict[str, tp.Any]] = field(
        default_factory=list,
        metadata={"help": "Static prompts (text or tokenized dicts) to sample during preview generation."},
    )
    generation_use_train_prompts: bool = field(
        default=False,
        metadata={"help": "When True, sample additional prompts from the training dataset for previews."},
    )
    generation_num_prompts: int | None = field(
        default=1,
        metadata={"help": "Number of prompts to use per preview generation call."},
    )
    generation_dataset_prompt_field: str | None = field(
        default="prompt",
        metadata={"help": "Dataset field to treat as prompt text when sampling from the training set."},
    )
    generation_extra_kwargs: dict[str, tp.Any] | None = field(
        default=None,
        metadata={"help": "Additional kwargs forwarded to `model.generate` for previews."},
    )
    generation_config_overrides: dict[str, tp.Any] | None = field(
        default=None,
        metadata={"help": "Attribute overrides applied to the copied generation config for previews."},
    )
    generation_seed: int | None = field(
        default=None,
        metadata={"help": "Seed for preview prompt sampling (None uses a random seed)."},
    )
    generation_preview_print: bool = field(
        default=False,
        metadata={"help": "Whether to print preview generations to Terminal."},
    )
    generation_log_to_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to log preview generations to WandB when available."},
    )
    log_training_generations_to_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to log rollout/training generations to WandB when available."},
    )
    benchmark_interval: int | None = field(
        default=None,
        metadata={"help": "Run configured lm-eval benchmark suites every X training steps (disabled when None)."},
    )
    benchmarks: list[BenchmarkConfig] | BenchmarkConfig | None = field(
        default_factory=list,
        metadata={"help": "Benchmark suite definitions executed during training when benchmark_interval is set."},
    )
    use_esurge_generation: bool = field(
        default=True,
        metadata={"help": "Whether to use eSurge engine for preview generation instead of compiled functions."},
    )
    esurge_use_tqdm: bool = field(
        default=True,
        metadata={"help": "Whether to use tqdm progress bars for eSurge generations."},
    )
    esurge_hbm_utilization: float | None = field(
        default=0.45,
        metadata={"help": "HBM memory utilization target for eSurge engine (0.0-1.0). None uses eSurge default."},
    )
    esurge_max_num_seqs: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of concurrent sequences for eSurge batch processing. None uses eSurge default."
        },
    )
    esurge_max_num_seq_buckets: list[int] | None = field(
        default=None,
        metadata={"help": "Optional explicit sequence-capacity buckets for eSurge runner compilation."},
    )
    esurge_min_input_pad: int | None = field(
        default=None,
        metadata={"help": "Minimum input padding for eSurge sequences. None uses eSurge default."},
    )
    esurge_page_size: int | None = field(
        default=32,
        metadata={"help": "Page size for eSurge KV cache management. None uses eSurge default."},
    )
    esurge_silent_mode: bool = field(
        default=True,
        metadata={"help": "Silence eSurge info logs (engine start/stop/resume, cache events)."},
    )
    esurge_runner_verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose eSurge runner performance logs (including `[perf]` lines)."},
    )
    esurge_max_num_batched_tokens: int | None = field(
        default=None,
        metadata={"help": "Maximum number of tokens to batch together for eSurge generation. None uses eSurge default."},
    )
    esurge_enable_prefix_caching: bool | None = field(
        default=None,
        metadata={"help": "Enable/disable eSurge prefix caching. None keeps engine default behavior."},
    )
    esurge_data_parallelism_axis: str | None = field(
        default=None,
        metadata={"help": "Mesh axis name used by eSurge as the data-parallel KV-page axis (e.g. 'dp')."},
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
    save_interval_minutes: float | None = field(
        default=None,
        metadata={"help": "Interval Minutes to save the checkpoint for state."},
    )
    save_directory: str | None = field(
        default="EasyDeL-Checkpoints",
        metadata={"help": "The directory to save checkpoints to."},
    )
    save_optimizer_state: bool = field(
        default=True,
        metadata={"help": "Whether to save the optimizer state along with the model."},
    )
    save_steps: int | None = field(
        default=None,
        metadata={"help": "Save a checkpoint every X steps."},
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of permanent checkpoints to keep. Older checkpoints are deleted. "
                "Note: Temporary checkpoints (time-based) are managed separately by Checkpointer."
            )
        },
    )
    save_tpu_preemption_checkpoints: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to force-save a standard EasyDeL checkpoint when JAX's TPU preemption "
                "sync service reaches a safe save step."
            )
        },
    )
    scheduler: AVAILABLE_SCHEDULERS = field(
        default=EasyDeLSchedulers.NONE,
        metadata={"help": "The scheduler to use."},
    )
    shuffle_seed_train: int = field(
        default=64871,
        metadata={"help": "seed used for trainer dataloader shuffle."},
    )
    sparsify_module: bool = field(
        default=False,
        metadata={"help": "Whether to sparsify the model."},
    )
    sparse_module_type: AVAILABLE_SPARSE_MODULE_TYPES = field(
        default="bcoo",
        metadata={"help": "The type of sparse module to use."},
    )
    state_apply_fn_kwarguments_to_model: dict | None = field(
        default=None,
        metadata={"help": "Keyword arguments to pass to the state apply function."},
    )
    step_partition_spec: PartitionSpec = field(
        default=PartitionSpec(("dp", "fsdp"), "sp"),
        metadata={"help": "The partition specification for the training step."},
    )
    step_start_point: int | None = field(
        default=None,
        metadata={"help": "The step to start training from (for resuming)."},
    )
    force_step_start_point: bool = field(
        default=False,
        metadata={
            "help": (
                "Force `step_start_point` onto a loaded nonzero state as well. "
                "When enabled, the trainer overwrites both `state.step` and optimizer/scheduler "
                "count leaves even if the incoming state was loaded manually rather than resumed "
                "through the trainer's auto-resume path."
            )
        },
    )
    resume_if_possible: bool = field(
        default=True,
        metadata={"help": "Automatically resume from the latest checkpoint if available."},
    )
    shuffle_train_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    total_batch_size: int = field(
        default=32,
        metadata={"help": "The total batch size."},
    )
    training_time_limit: str | None = field(
        default=None,
        metadata={"help": "The maximum training time (e.g., '1d', '2h30m')."},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "Whether to train on the input data."},
    )
    trainer_prefix: str | None = field(
        default=None,
        metadata={"help": "default prefix name for trainer."},
    )
    truncation_mode: tp.Literal["keep_end", "keep_start"] = field(
        default="keep_end",
        metadata={"help": "The truncation mode to use."},
    )
    tx_mu_dtype: jnp.dtype | None = field(
        default=None,
        metadata={"help": "The dtype to use for the `tx.mu` variable."},
    )
    track_memory: bool | float = field(
        default=False,
        metadata={"help": "Whether to track memory usage. If a float, it sets the memory tracking interval in seconds."},
    )
    use_data_collator: bool = field(
        default=True,
        metadata={"help": "Whether to use a data collator."},
    )
    use_grain: bool = field(
        default=True,
        metadata={"help": "Whether to use grain instead of `tensorflow-datasets`."},
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to use Weights & Biases for logging."},
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print verbose output."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "The Weights & Biases entity."},
    )
    wandb_name: str | None = field(
        default=None,
        metadata={"help": "The Weights & Biases run name."},
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "The number of warmup steps for the learning rate scheduler."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "The weight decay value."},
    )
    watchers: list[LogWatcher] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of LogWatcher instances that define custom per-parameter "
                "metrics to log at independent intervals during training."
            ),
        },
    )
    weight_distribution_pattern: str = field(
        default=r".*",
        metadata={"help": "The pattern to use to extract weight distribution."},
    )
    weight_distribution_log_steps: int = field(
        default=500,
        metadata={"help": "log weight distribution every X steps."},
    )

    _can_log_metrics: bool | None = None
    _im_a_hidden_checkpoint_manager: Checkpointer | None = None

    @property
    def can_log_metrics(self):
        """Whether this process should log metrics.

        Returns ``False`` when the process is not rank-zero and
        ``log_all_workers`` is disabled. Can be overridden via the setter.

        Returns:
            bool: True if metrics logging is enabled for this process.
        """
        if self._can_log_metrics is None:
            if not self.is_process_zero and not self.log_all_workers:
                return False
            return self.report_metrics
        return self._can_log_metrics

    @can_log_metrics.setter
    def can_log_metrics(self, val):
        """Override the automatic metrics-logging decision.

        Args:
            val: Explicit boolean to force metrics logging on or off,
                or None to revert to the default behavior.
        """
        self._can_log_metrics = val

    @property
    def offload_device(self):
        """Return the JAX device used for parameter offloading.

        Resolves the device from ``offload_device_type`` and
        ``offload_device_index``.

        Returns:
            jax.Device: The target offload device.
        """
        return jax.devices(self.offload_device_type)[self.offload_device_index]

    @property
    def training_time_seconds(self) -> int | None:
        """Convert ``training_time_limit`` to total seconds.

        Returns:
            The training time limit in seconds, or None if no limit is set.
        """
        if self.training_time_limit is None:
            return None
        return self._time_to_seconds(self.training_time_limit)

    @functools.cached_property
    def is_process_zero(self):
        """Whether the current process is the rank-zero (main) process.

        Returns:
            bool: True if ``jax.process_index()`` is 0.
        """
        return jax.process_index() == 0

    def _handle_deprecated_max_sequence_length(self, max_sequence_length: int | None) -> None:
        """Migrate the deprecated ``max_sequence_length`` value to ``max_length``.

        Emits a FutureWarning and sets ``max_length`` when it has not been
        explicitly provided. If both are set to conflicting values,
        ``max_length`` takes precedence and a warning is emitted.

        Args:
            max_sequence_length: The deprecated sequence-length value, or None
                if it was not supplied.
        """
        if max_sequence_length is None:
            return
        warnings.warn(
            "`max_sequence_length` is deprecated; use `max_length` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if self.max_length is None:
            self.max_length = max_sequence_length
            return
        if self.max_length == max_sequence_length:
            return

        # If `max_length` is still at the class default, treat the deprecated alias as the user's intent.
        max_length_default = None
        for field_obj in fields(self):
            if field_obj.name == "max_length":
                max_length_default = field_obj.default
                break

        if self.max_length == max_length_default:
            self.max_length = max_sequence_length
            return

        warnings.warn(
            f"Both `max_length` ({self.max_length}) and `max_sequence_length` ({max_sequence_length}) are set; "
            "ignoring `max_sequence_length`.",
            FutureWarning,
            stacklevel=2,
        )

    def _handle_deprecated_quantization_block(self, quantization_block: int | None) -> None:
        """Migrate the deprecated ``quantization_block`` value to ``quantization_group_size``.

        Emits a FutureWarning and sets ``quantization_group_size`` when it has
        not been explicitly provided. If both are set to conflicting values,
        ``quantization_group_size`` takes precedence and a warning is emitted.

        Args:
            quantization_block: The deprecated quantization block size, or None
                if it was not supplied.
        """
        if quantization_block is None:
            return
        warnings.warn(
            "`quantization_block` is deprecated; use `quantization_group_size` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if self.quantization_group_size is None:
            self.quantization_group_size = quantization_block
            return
        if self.quantization_group_size == quantization_block:
            return

        warnings.warn(
            f"Both `quantization_group_size` ({self.quantization_group_size}) and "
            f"`quantization_block` ({quantization_block}) are set; ignoring `quantization_block`.",
            FutureWarning,
            stacklevel=2,
        )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """
        Post-initialization setup and validation.

        This method is automatically called after dataclass initialization.
        It performs:
        1. Configuration validation to catch invalid settings early
        2. Distributed training setup based on backend and platform
        3. Optimizer and scheduler configuration
        4. Logging infrastructure initialization
        5. Variable normalization and default value setting

        Raises:
            ValueError: If configuration validation fails
            ValueError: If required conditions are not met
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        self._handle_deprecated_quantization_block(quantization_block)
        if self.max_length is None:
            self.max_length = 4096
        self._ensure_variables()
        self._validate_config()
        self._setup_distributed()
        self._setup_optimizer()
        self._setup_logging()

    def _validate_config(self):
        """
        Validate configuration settings for correctness and compatibility.

        This method checks:
        - Gradient accumulation steps are positive
        - Backend is supported (CPU, GPU, TPU)
        - Other configuration constraints are met

        Raises:
            ValueError: If gradient_accumulation_steps < 1 or backend is not recognized
        """
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("`gradient_accumulation_steps` can't be lower than 1.")

        if self.backend not in AVAILABLE_BACKENDS:
            raise ValueError(f"Backend {self.backend} is not recognized. Available backends: {AVAILABLE_BACKENDS}")
        if self.lmhead_chunksize is not None and self.lmhead_chunksize <= 0:
            raise ValueError("`lmhead_chunksize` must be > 0 when specified.")
        if self.quantization_group_size is not None and self.quantization_group_size <= 0:
            raise ValueError("`quantization_group_size` must be > 0 when specified.")
        if self.quantization_bits is not None and self.quantization_bits <= 0:
            raise ValueError("`quantization_bits` must be > 0 when specified.")
        if self.quantization_mode == "affine" and self.quantization_bits is not None:
            if self.quantization_bits not in AFFINE_SUPPORTED_BITS:
                bits_values = ", ".join(str(v) for v in sorted(AFFINE_SUPPORTED_BITS))
                raise ValueError(
                    f"`quantization_bits` for `affine` must be one of {{{bits_values}}}, got {self.quantization_bits}."
                )
        if self.quantization_mode in FIXED_QUANTIZATION_BITS_BY_MODE and self.quantization_bits is not None:
            expected_bits = FIXED_QUANTIZATION_BITS_BY_MODE[self.quantization_mode]
            if self.quantization_bits != expected_bits:
                raise ValueError(
                    f"`quantization_bits` for `{self.quantization_mode}` must be {expected_bits}, "
                    f"got {self.quantization_bits}."
                )

    def _setup_distributed(self):
        """
        Configure JAX for distributed training.

        This method initializes the JAX distributed configuration which handles:
        - Multi-host setup for distributed training
        - Device mesh creation for model parallelism
        - Communication backend configuration
        - Process coordination setup

        The actual implementation is delegated to JaxDistributedConfig.
        """

        JaxDistributedConfig.initialize(self.jax_distributed_config)

    def _setup_optimizer(self):
        """
        Configure optimizer and learning rate scheduler settings.

        This method prepares the optimizer_kwargs dictionary with all necessary
        parameters for optimizer creation, including:
        - Learning rate and schedule parameters
        - Weight decay and gradient clipping
        - Optimizer-specific settings from extra_optimizer_kwargs
        - Data type specifications for optimizer states

        The actual optimizer creation is deferred until training setup.
        """
        extra_optimizer_kwargs = self.extra_optimizer_kwargs if self.extra_optimizer_kwargs is not None else {}
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
        Configure logging infrastructure for training monitoring.

        This method handles the setup of various logging backends and ensures
        compatibility with performance mode:
        - Disables WandB in performance mode to reduce overhead
        - Disables metrics reporting in performance mode
        - Configures appropriate logging levels and destinations

        Performance mode prioritizes speed over detailed monitoring.
        """
        if self.use_wandb and self.performance_mode:
            logger.info("WandB logging disabled due to performance mode")
            self.use_wandb = False
        if self.report_metrics and self.performance_mode:
            logger.info("Metrics reporting disabled due to performance mode")
            self._can_log_metrics = False

    def _ensure_variables(self):
        """
        Ensure all configuration variables are properly initialized.

        This method:
        - Converts string representations to proper types (e.g., PartitionSpec)
        - Sets default values for optional parameters
        - Initializes complex configuration objects (e.g., LossConfig)
        - Normalizes configuration values for consistency

        This ensures the configuration is ready for use by the trainer.
        """

        def _coerce_float(value: tp.Any) -> tp.Any:
            """Best-effort coercion of a value to a Python float.

            Handles numeric types, NumPy scalars, and whitespace-stripped
            strings. Returns the original value unchanged if conversion is
            not possible (e.g., non-numeric strings, booleans, None).

            Args:
                value: The value to coerce.

            Returns:
                A Python float if conversion succeeds, otherwise the
                original value.
            """
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (float, int, np.floating, np.integer)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return value
                try:
                    return float(stripped)
                except ValueError:
                    return value
            return value

        def _coerce_int(value: tp.Any) -> tp.Any:
            """Best-effort coercion of a value to a Python int.

            Handles int/float types, NumPy scalars, and whitespace-stripped
            strings. Floats are converted only when they represent whole
            numbers. Returns the original value unchanged if conversion is
            not possible.

            Args:
                value: The value to coerce.

            Returns:
                A Python int if conversion succeeds, otherwise the original
                value.
            """
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return value
                try:
                    return int(stripped)
                except ValueError:
                    try:
                        casted = float(stripped)
                    except ValueError:
                        return value
                    return int(casted) if casted.is_integer() else value
            return value

        self.learning_rate = _coerce_float(self.learning_rate)
        self.learning_rate_end = _coerce_float(self.learning_rate_end)
        self.weight_decay = _coerce_float(self.weight_decay)
        self.clip_grad = _coerce_float(self.clip_grad)
        self.warmup_steps = _coerce_int(self.warmup_steps)
        self.gradient_accumulation_steps = _coerce_int(self.gradient_accumulation_steps)
        self.lmhead_chunksize = _coerce_int(self.lmhead_chunksize)
        self.quantization_group_size = _coerce_int(self.quantization_group_size)
        self.quantization_bits = _coerce_int(self.quantization_bits)

        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name, None)
            if not isinstance(value, (float, int, np.floating, np.integer)):
                raise TypeError(f"`{name}` must be a number, got {type(value).__name__}: {value!r}")

        if self.learning_rate_end is not None and not isinstance(
            self.learning_rate_end, (float, int, np.floating, np.integer)
        ):
            raise TypeError(
                "`learning_rate_end` must be a number when provided, got "
                f"{type(self.learning_rate_end).__name__}: {self.learning_rate_end!r}"
            )

        if self.clip_grad is not None and not isinstance(self.clip_grad, (float, int, np.floating, np.integer)):
            raise TypeError(
                f"`clip_grad` must be a number when provided, got {type(self.clip_grad).__name__}: {self.clip_grad!r}"
            )

        if not isinstance(self.warmup_steps, (int, np.integer)):
            raise TypeError(
                f"`warmup_steps` must be an int, got {type(self.warmup_steps).__name__}: {self.warmup_steps!r}"
            )

        if not isinstance(self.gradient_accumulation_steps, (int, np.integer)):
            raise TypeError(
                "`gradient_accumulation_steps` must be an int, got "
                f"{type(self.gradient_accumulation_steps).__name__}: {self.gradient_accumulation_steps!r}"
            )
        if self.lmhead_chunksize is not None and not isinstance(self.lmhead_chunksize, (int, np.integer)):
            raise TypeError(
                "`lmhead_chunksize` must be an int when provided, got "
                f"{type(self.lmhead_chunksize).__name__}: {self.lmhead_chunksize!r}"
            )
        if self.quantization_group_size is not None and not isinstance(self.quantization_group_size, (int, np.integer)):
            raise TypeError(
                "`quantization_group_size` must be an int when provided, got "
                f"{type(self.quantization_group_size).__name__}: {self.quantization_group_size!r}"
            )
        if self.quantization_bits is not None and not isinstance(self.quantization_bits, (int, np.integer)):
            raise TypeError(
                "`quantization_bits` must be an int when provided, got "
                f"{type(self.quantization_bits).__name__}: {self.quantization_bits!r}"
            )

        if isinstance(self.quantization_mode, str):
            quantization_mode = self.quantization_mode.strip().lower()
            self.quantization_mode = tp.cast(QuantizationMode | None, quantization_mode or None)
        if self.quantization_mode is not None and self.quantization_mode not in STE_QAT_QUANTIZATION_MODES:
            raise ValueError(
                f"`quantization_mode` must be one of {STE_QAT_QUANTIZATION_MODES_DOC}, got {self.quantization_mode!r}."
            )

        self.step_partition_spec = _parse_partition_spec(self.step_partition_spec)

        self.eval_batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.total_batch_size
        if self.loss_config is None:
            self.loss_config = LossConfig()
        if isinstance(self.loss_config, dict):
            self.loss_config = LossConfig(**self.loss_config)
        if self.generation_interval is not None and self.generation_interval <= 0:
            logger.warning("`generation_interval` must be positive; disabling preview generation.")
            self.generation_interval = None
        if self.benchmark_interval is not None and self.benchmark_interval <= 0:
            logger.warning("`benchmark_interval` must be positive; disabling benchmark hooks.")
            self.benchmark_interval = None
        if self.generation_num_prompts is not None:
            self.generation_num_prompts = max(1, int(self.generation_num_prompts))
        if self.benchmarks is None:
            self.benchmarks = []
        elif isinstance(self.benchmarks, collections.abc.Mapping):
            self.benchmarks = [dict(self.benchmarks)]
        elif isinstance(self.benchmarks, collections.abc.Sequence) and not isinstance(self.benchmarks, (str, bytes)):
            self.benchmarks = [dict(benchmark) for benchmark in self.benchmarks]
        else:
            raise TypeError("`benchmarks` must be a BenchmarkConfig or a sequence of BenchmarkConfig values.")
        normalize_benchmark_configs(self.benchmarks)  # early validation; resolved again at runtime
        if self.esurge_max_num_seq_buckets is not None:
            self.esurge_max_num_seq_buckets = [int(v) for v in self.esurge_max_num_seq_buckets]
        if self.esurge_data_parallelism_axis is not None:
            self.esurge_data_parallelism_axis = str(self.esurge_data_parallelism_axis).strip()
            if not self.esurge_data_parallelism_axis:
                raise ValueError("`esurge_data_parallelism_axis` must be a non-empty string when provided.")

        def _inherit_generation_attr(attr, fallback_name):
            """Copy a fallback attribute into a generation attribute if unset.

            Sets ``self.<attr>`` to ``self.<fallback_name>`` when the
            generation attribute is None and the fallback holds a
            non-None, non-False value.

            Args:
                attr: Name of the generation attribute to populate.
                fallback_name: Name of the legacy/alternative attribute to
                    read from.
            """
            current = getattr(self, attr, None)
            if current is None and hasattr(self, fallback_name):
                fallback_value = getattr(self, fallback_name)
                if fallback_value is not None and fallback_value is not False:
                    setattr(self, attr, fallback_value)

        _inherit_generation_attr("generation_num_return_sequences", "num_generations_per_prompt")
        _inherit_generation_attr("generation_num_return_sequences", "num_return_sequences")
        _inherit_generation_attr("generation_top_p", "top_p")
        _inherit_generation_attr("generation_top_k", "top_k")
        _inherit_generation_attr("generation_presence_penalty", "presence_penalty")
        _inherit_generation_attr("generation_frequency_penalty", "frequency_penalty")
        _inherit_generation_attr("generation_repetition_penalty", "repetition_penalty")
        _inherit_generation_attr("generation_temperature", "temperature_sampling")
        _inherit_generation_attr("generation_max_new_tokens", "max_completion_length")

        self.generation_top_p = _coerce_float(self.generation_top_p)
        self.generation_temperature = _coerce_float(self.generation_temperature)
        self.generation_top_k = _coerce_int(self.generation_top_k)
        self.generation_presence_penalty = _coerce_float(self.generation_presence_penalty)
        self.generation_frequency_penalty = _coerce_float(self.generation_frequency_penalty)
        self.generation_repetition_penalty = _coerce_float(self.generation_repetition_penalty)
        self.generation_num_return_sequences = _coerce_int(self.generation_num_return_sequences)
        self.generation_max_new_tokens = _coerce_int(self.generation_max_new_tokens)

        if self.generation_extra_kwargs is not None and not isinstance(
            self.generation_extra_kwargs, collections.abc.Mapping
        ):
            raise TypeError(
                "`generation_extra_kwargs` must be a mapping when provided, got "
                f"{type(self.generation_extra_kwargs).__name__}: {self.generation_extra_kwargs!r}"
            )
        generation_extra_kwargs = dict(self.generation_extra_kwargs or {})

        generation_kwargs = getattr(self, "generation_kwargs", None)
        if generation_kwargs is not None:
            if not isinstance(generation_kwargs, collections.abc.Mapping):
                raise TypeError(
                    "`generation_kwargs` must be a mapping when provided, got "
                    f"{type(generation_kwargs).__name__}: {generation_kwargs!r}"
                )
            generation_extra_kwargs.update(generation_kwargs)

        for attr_name, kwarg_name in (("min_p", "min_p"),):
            value = getattr(self, attr_name, None)
            if value is not None and kwarg_name not in generation_extra_kwargs:
                generation_extra_kwargs[kwarg_name] = value
        for attr_name, kwarg_name in (
            ("presence_penalty", "presence_penalty"),
            ("frequency_penalty", "frequency_penalty"),
            ("repetition_penalty", "repetition_penalty"),
        ):
            value = getattr(self, attr_name, None)
            generation_value = getattr(self, f"generation_{attr_name}", None)
            if value is not None and generation_value is None and kwarg_name not in generation_extra_kwargs:
                generation_extra_kwargs[kwarg_name] = value
        self.generation_extra_kwargs = generation_extra_kwargs or None

        if self.generation_do_sample is None:
            if any(
                getattr(self, name, None) is not None
                for name in ("generation_top_p", "generation_top_k", "generation_temperature")
            ):
                self.generation_do_sample = True
        if self.generation_do_sample is None and hasattr(self, "do_sample"):
            do_sample = getattr(self, "do_sample", None)
            if do_sample is not None:
                self.generation_do_sample = do_sample

    @staticmethod
    def _time_to_seconds(time_str: str) -> int:
        """
        Convert a human-readable time string to seconds.

        Supports various time formats:
        - Hours: "23h", "2hour", "3hours"
        - Minutes: "50min", "30m", "45minutes"
        - Seconds: "30s", "120sec", "60seconds"

        Args:
            time_str: The time string to convert

        Returns:
            int: The equivalent time in seconds

        Raises:
            ValueError: If the time format is not recognized

        Example:
            >>> _time_to_seconds("2h30min")  # Would need parsing enhancement
            >>> _time_to_seconds("90min")
            5400
        """
        match = re.match(r"(\d+)\s*(h|hour|hours|min|m|minutes|s|sec|seconds)", time_str.lower())
        if not match:
            raise ValueError("Invalid time format. Use `50min` for minutes, `23h` for hours, or `30s` for seconds.")
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

    def get_path(self) -> ePathLike:
        """Get the path to the checkpoint directory.

        Returns:
            ePathLike: The path to the checkpoint directory, combining
                      save_directory and model_name.

        Note:
            Creates a model-specific subdirectory within the main save directory.
        """
        return ePath(self.save_directory) / self.model_name  # pyright: ignore[reportReturnType]

    def ensure_checkpoint_path(self):
        """Create the checkpoint directory if it doesn't exist.

        Ensures the full checkpoint path including parent directories
        exists on the filesystem. Safe to call multiple times.

        Note:
            Uses mkdir with parents=True to create full directory tree.
        """
        path = self.get_path()
        path.mkdir(parents=True, exist_ok=True)

    def get_tx_template(self, possible_max: int | None = None) -> GradientTransformation:
        """Get the optimizer transformation without a specific step count.

        Useful for creating a template optimizer when the total number of
        training steps is not yet known (e.g., during checkpoint loading).

        Args:
            possible_max: Maximum step count to use. Defaults to 2^63-1.

        Returns:
            GradientTransformation: The configured Optax optimizer chain.
        """
        if possible_max is None:
            possible_max = 2**63 - 1
        optimizer = self.get_optimizer_and_scheduler(possible_max)[0]
        if self.pruning_module is not None:
            optimizer = self.pruning_module.wrap_optax(optimizer)
        return optimizer

    def get_optimizer_and_scheduler(self, steps: int | None = None):
        """
        Create and return the optimizer and learning rate scheduler.

        This method uses the OptimizerFactory to create the configured optimizer
        and scheduler based on the training arguments. It handles:
        - Standard optimizers (AdamW, SGD, etc.)
        - Learning rate schedules (linear, cosine, constant)
        - Gradient clipping and weight decay
        - Custom optimizers and schedulers

        Args:
            steps: Optional override for the number of training steps.
                   If not provided, uses the value from self.optimizer_kwargs.

        Returns:
            tuple: A tuple of (optimizer, scheduler) where:
                - optimizer: Optax GradientTransformation
                - scheduler: Learning rate schedule function

        Note:
            The optimizer is an Optax transformation chain that may include
            gradient clipping, weight decay, and other transformations.
        """

        optimizer_kwargs = deepcopy(self.optimizer_kwargs)
        optimizer_kwargs["steps"] = steps or optimizer_kwargs["steps"]
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
        """Get the asynchronous checkpoint manager.

        Returns:
            AsyncCheckpointManager: The checkpoint manager for handling
                                   asynchronous model checkpointing.

        Note:
            The AsyncCheckpointManager allows non-blocking checkpoint
            saves during training, improving training efficiency.
        """
        if self._im_a_hidden_checkpoint_manager is not None:
            return self._im_a_hidden_checkpoint_manager

        self._im_a_hidden_checkpoint_manager = Checkpointer(
            base_path=str(self._get_save_directory()),
            save_interval=self.get_save_interval_timedelta(),
            step_policies=self.get_checkpoint_policies(),
        )

        return self._im_a_hidden_checkpoint_manager

    def get_checkpoint_policies(self):
        """Convert save_steps configuration to CheckpointInterval policies.

        Returns:
            list[CheckpointInterval]: List of checkpoint interval policies.
                Returns empty list if save_steps is None.

        Example:
            >>> args = TrainingArguments(save_steps=1000)
            >>> policies = args.get_checkpoint_policies()
            >>> # Returns: [CheckpointInterval(every=1000, until=None)]
        """
        from eformer.serialization.checkpointer import CheckpointInterval

        if self.save_steps is None:
            return []
        return [CheckpointInterval(every=self.save_steps, until=None)]

    def get_save_interval_timedelta(self):
        """Get time-based checkpoint save interval as timedelta.

        Returns:
            timedelta | None: Time interval for temporary checkpoints,
                or None if no time-based saving is configured.

        Note:
            Currently returns None. Can be extended to support
            time-based checkpoint saving via new TrainingArguments field.
        """
        if self.save_interval_minutes is not None:
            return datetime.timedelta(minutes=self.save_interval_minutes)
        return None

    @functools.cached_property
    def _tensorboard(self):
        """Lazy initialization of TensorBoard writer.

        Returns:
            SummaryWriter | None: TensorBoard writer instance, or None if:
                - Path is None
                - Path is on Google Cloud Storage (gs://)
                - TensorBoard is not installed

        Note:
            Cached property to avoid multiple initializations.
            TensorBoard doesn't support cloud storage paths directly.
        """
        from flax.metrics.tensorboard import SummaryWriter

        path = self._get_save_directory(create=True)
        if path is None:
            return None
        if str(path).startswith("gs://"):
            return None
        return SummaryWriter(log_dir=str(path))

    def get_tensorboard(self) -> SummaryWriter | None:
        """Get the TensorBoard SummaryWriter for logging metrics.

        Returns:
            SummaryWriter | None: The TensorBoard writer instance, or None if
                                 TensorBoard is not available or not configured.

        Note:
            Handles ModuleNotFoundError gracefully if TensorBoard is not installed.
            Uses cached property internally for efficiency.
        """
        try:
            return self._tensorboard
        except ModuleNotFoundError:
            return None

    def get_wandb_init(self):
        """
        Initialize Weights & Biases for experiment tracking.

        This method creates a new WandB run with appropriate configuration:
        - Project name based on model name and optional prefix
        - Run name with timestamp if not specified
        - Configuration dictionary from training arguments
        - Standard tags for EasyDeL experiments

        The method handles process-level initialization, ensuring only the
        main process creates the WandB run in distributed settings.

        Returns:
            wandb.Run | None: The initialized WandB run object, or None if:
                - WandB is not installed
                - use_wandb is False
                - Not the main process and log_all_workers is False

        Note:
            WandB initialization is skipped in performance mode to reduce overhead.
        """
        if self.can_log_metrics:
            if not self.use_wandb or wandb is None:
                warnings.warn(
                    "you have used `use_wandb=True` but you haven't install wandb.",
                    stacklevel=1,
                )
                return None
            wandb_name = self.wandb_name
            prefix = (self.trainer_prefix or "").strip()
            project_suffix = f"-{prefix}" if prefix else ""
            resolved_model_name = self.model_name if isinstance(self.model_name, str) and self.model_name else "model"
            safe_model_name = resolved_model_name.lower().replace("/", "-")
            if wandb_name is None:
                wandb_name = self.build_wandb_run_name(safe_model_name)
            else:
                wandb_name = wandb_name + "-" + safe_model_name
            return wandb.init(
                project=f"EasyDeL{project_suffix}",
                config=self.to_dict(),
                save_code=True,
                name=wandb_name,
                tags=["EasyDeL", "Jax", "Train", "LLM", "VLM"],
                entity=self.wandb_entity,
            )
        return None

    @staticmethod
    def _wandb_token(value: tp.Any, fallback: str = "na") -> str:
        """Normalize arbitrary config values into safe W&B name tokens."""
        if value is None:
            return fallback
        if hasattr(value, "value"):
            value = value.value
        token = str(value).strip().lower()
        token = token.replace("/", "-").replace(" ", "")
        token = re.sub(r"[^a-z0-9._-]", "-", token)
        token = re.sub(r"-{2,}", "-", token).strip("-")
        return token or fallback

    @staticmethod
    def _wandb_float_token(value: float | int | None, fallback: str = "none") -> str:
        """Format a numeric value as a compact string token for W&B run names.

        Args:
            value: The numeric value to format, or None.
            fallback: String to return when ``value`` is None.

        Returns:
            A compact string representation (e.g., ``"5e-05"``) or the
            fallback string.
        """
        if value is None:
            return fallback
        number = float(value)
        token = f"{number:g}".replace("+", "")
        return token

    def build_wandb_run_name(
        self,
        model_name: str,
        size_in_billion: float | None = None,
    ) -> str:
        """Build a structured default W&B run name.

        Format (when size_in_billion is provided):
            ``{model_name}-{size}b-b{batch}-lr{lr}-tx-{optimizer}``
        Format (when size_in_billion is None):
            ``{model_name}-b{batch}-lr{lr}-tx-{optimizer}``
        """
        batch = int(self.total_batch_size)
        lr = self._wandb_float_token(self.learning_rate, fallback="none")
        optimizer = self._wandb_token(self.optimizer, fallback="NA")
        if size_in_billion is not None:
            size = self._wandb_float_token(size_in_billion)
            return f"{model_name}-{size}b-b{batch}-lr{lr}-tx-{optimizer}"
        return f"{model_name}-b{batch}-lr{lr}-tx-{optimizer}"

    def ensure_training_time_limit(self, time_passed):
        """Check if training has exceeded the configured time limit.

        Args:
            time_passed: Elapsed training time in seconds.

        Raises:
            EasyDeLTimerError: If elapsed time exceeds training_time_limit.
        """
        if self.training_time_limit is not None and time_passed > self._time_to_seconds(self.training_time_limit):
            raise EasyDeLTimerError("Time Out")

    def log_metrics(
        self,
        metrics: MetricsType,
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
    ):
        """
        Log metrics to configured logging backends.

        This method handles logging to multiple backends (WandB, TensorBoard)
        and supports various metric types including scalars, histograms, and
        distributions. It automatically filters and formats metrics for each
        backend's requirements.

        Args:
            metrics: Dictionary of metric names to values. Values can be:
                - Scalars (float, int)
                - Arrays (numpy, JAX arrays)
                - Histograms (tuple of bin_counts and bin_edges)
                - Tensors (automatically converted)
            step: The current training/evaluation step
            log_as: Special logging mode:
                - None: Regular step-based logging
                - "summary": Log as final summary (WandB only)
                - "config": Log as configuration (WandB only)

        Note:
            - Metrics are automatically filtered for None values
            - Array metrics are converted to appropriate formats
            - Gradient norm metrics are restructured for clarity
            - Logging only occurs if can_log_metrics is True
        """

        if self.can_log_metrics:
            filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
            metrics = {self._restructure_metric_name(k): get_safe_arr(v) for k, v in filtered_metrics.items()}
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
        """
        Log weight distribution histograms and statistics.

        This method computes and logs detailed statistics about model weights,
        including histograms and summary statistics (mean, std, min, max).
        It's useful for monitoring training stability and detecting issues
        like gradient explosion or vanishing.

        Args:
            state: Model state containing parameters to analyze
            step: Current training step for logging

        Note:
            - Only logs at intervals defined by weight_distribution_log_steps
            - Uses weight_distribution_pattern to filter parameters
            - Computes statistics across all processes in distributed training
            - Logs both histograms and scalar statistics for each parameter
        """
        try:
            if self.weight_distribution_log_steps > 0 and ((step % self.weight_distribution_log_steps) == 0):
                stats = compute_weight_stats(state.graphstate, self.weight_distribution_pattern)
                stats = jax.device_get(stats)

                metrics = {}
                for key, histogram in stats.items():
                    try:
                        if isinstance(histogram, MetricsHistogram):
                            path = key.replace("/", ".")
                            metrics[f"weights-histogram/{path}"] = (
                                histogram.bin_counts.tolist(),
                                histogram.bin_edges.tolist(),
                            )

                            base_path = path.replace("/histogram", "")
                            metrics[f"weights-information/{base_path}/mean"] = float(histogram.mean)
                            metrics[f"weights-information/{base_path}/std"] = histogram.std.item()
                            metrics[f"weights-information/{base_path}/min"] = histogram.min.item()
                            metrics[f"weights-information/{base_path}/max"] = histogram.max.item()
                        else:
                            path = key.replace("/", ".")
                            metrics[f"weights-information/{path}"] = histogram
                    except Exception as e:
                        traceback.print_exc()
                        raise e

                self.log_metrics(metrics, step)

        except Exception as e:
            logger.warning(f"Failed to log weight distribution {e}...")

    def log_watchers(self, state, step: int):
        """Run all registered ``LogWatcher`` instances and log their metrics.

        Each watcher is only evaluated when ``step`` is a multiple of its
        ``interval``. Results are sent through the standard
        ``log_metrics`` pipeline (wandb / TensorBoard).

        Args:
            state: Model state whose ``graphstate`` contains the parameters.
            step: Current training step.
        """
        if not self.watchers:
            return
        try:
            metrics = run_watchers(self.watchers, state.graphstate, step)
            if metrics:
                metrics = jax.device_get(metrics)
                self.log_metrics(metrics, step)
        except Exception as e:
            logger.warning(f"Failed to log watchers: {e}...")

    def _log_to_wandb(
        self,
        metrics: dict[str, tp.Any],
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
    ):
        """
        Log metrics to Weights & Biases (wandb).

        This method processes the given metrics and logs them to wandb if it's enabled and properly initialized.

        Args:
            metrics: A dictionary of metrics to log. Keys are metric names, values are the metric values.
            step: The current step or iteration number.
            log_as: How to log the metrics
                (None for regular logging, "summary" for wandb.summary, "config" for wandb.config)
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
                        if isinstance(value, tuple) and len(value) == 2:
                            bin_counts, bin_edges = value
                            if isinstance(bin_counts, list | jax.Array) and isinstance(bin_edges, list | jax.Array):
                                bin_counts = np.array(bin_counts).reshape(-1)
                                bin_edges = np.array(bin_edges).reshape(-1)
                                np_histogram = (bin_counts, bin_edges)

                                wandb_metrics[key] = wandb.Histogram(np_histogram=np_histogram)
                                continue

                        wandb_metrics[key] = (
                            self._create_wandb_histogram(value)
                            if isinstance(value, float | int | list | tuple | np.generic | jax.Array)
                            else value
                        )

                    except Exception as e:
                        warnings.warn(f"Failed to log metric {key} to wandb: {e}", stacklevel=3)

                wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}

                try:
                    wandb.log(wandb_metrics, step=step)
                except Exception as e:
                    warnings.warn(f"Failed to log metrics to wandb: {e}", stacklevel=3)

    def _log_to_tensorboard(
        self,
        metrics: dict[str, tp.Any],
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
    ):
        """
        Log metrics to TensorBoard.

        Args:
            metrics: A dictionary of metrics to log
            step: The current step or iteration number
            log_as: Currently not used for TensorBoard
        """
        summary_writer = self.get_tensorboard()
        if summary_writer is not None:
            for key, value in metrics.items():
                try:
                    if isinstance(value, float | int):
                        summary_writer.scalar(key, value, step)
                    elif isinstance(value, tuple) and len(value) == 2:
                        bin_counts, bin_edges = value
                        if isinstance(bin_counts, list | jax.Array) and isinstance(bin_edges, list | jax.Array):
                            bin_counts = np.array(bin_counts)
                            bin_edges = np.array(bin_edges)
                            values = []
                            for i, count in enumerate(bin_counts):
                                if i < len(bin_edges) - 1:
                                    bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                                    values.extend([bin_center] * int(count))

                            if values:
                                summary_writer.histogram(key, np.array(values), step)
                    elif isinstance(value, list | np.ndarray | jnp.ndarray):
                        summary_writer.histogram(key, np.array(value), step)
                except Exception as e:
                    warnings.warn(f"Failed to log metric {key} to TensorBoard: {e}", stacklevel=1)
                finally:
                    summary_writer.flush()

    def _create_wandb_histogram(self, value):
        """Create a wandb.Histogram object from the given value.

        Args:
            value: The value to convert into a wandb.Histogram. Can be:
                  - JAX array
                  - NumPy array or generic
                  - Other numeric types

        Returns:
            wandb.Histogram | None: A wandb.Histogram object if successful,
                                   None if an error occurs.

        Note:
            Handles dtype conversion for bfloat16 to float32/float16 for
            compatibility with wandb. Automatically moves JAX arrays to CPU.
        """
        try:
            if isinstance(value, jax.Array | np.generic):
                value = np.array(jax.device_get(value))
                if value.dtype in [np.bfloat16]:
                    value = value.astype(np.float32)
                value = value.astype(np.float16)
                return wandb.Histogram(value)
            return value
        except Exception as e:
            warnings.warn(f"Failed to create wandb histogram: {e}", stacklevel=1)
            return None

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        """Load a JSON file and return its contents as a dictionary.

        Args:
            json_file: Path to the JSON configuration file.

        Returns:
            dict: The parsed JSON contents.
        """
        return json.loads(ePath(json_file).read_text())

    def to_dict(self) -> dict[str, tp.Any]:
        """Serializes this instance to a dictionary.

        Returns:
            dict[str, tp.Any]: A dictionary containing all serializable fields.
        """
        result = {}
        for field_obj in fields(self):
            value = getattr(self, field_obj.name)
            if value is Ellipsis:
                continue
            if field_obj.name.startswith("_"):
                continue
            if isinstance(value, tuple):
                result[field_obj.name] = list(value)
            elif value is None:
                result[field_obj.name] = None
            elif hasattr(value, "to_dict") and callable(value.to_dict):
                result[field_obj.name] = value.to_dict()
            else:
                try:
                    json.dumps(value)
                    result[field_obj.name] = value
                except (TypeError, OverflowError):
                    result[field_obj.name] = str(value)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]):
        """Deserializes a dictionary into a TrainingArguments instance.

        Args:
            data: Dictionary containing field names and values.

        Returns:
            TrainingArguments: A new instance created from the dictionary.
        """
        data = _apply_training_args_legacy_aliases(data)

        processed_data = {}
        type_hints = tp.get_type_hints(cls)

        for field_obj in fields(cls):
            field_name = field_obj.name
            if field_name not in data:
                continue

            value = data[field_name]
            field_type = type_hints.get(field_name)

            if (
                value is not None
                and isinstance(value, list)
                and field_type is not None
                and tp.get_origin(field_type) is tuple
            ):
                processed_data[field_name] = tuple(value)
            else:
                processed_data[field_name] = value

        return cls(**processed_data)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        config_dict["trainer_config_class"] = self.__class__.__name__
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def load_arguments(cls, json_file: str | os.PathLike):
        """
        Load training arguments from a JSON file.

        This class method reconstructs a TrainingArguments instance from a
        previously saved JSON configuration file. It handles class resolution
        and proper type conversion.

        Args:
            json_file: Path to the JSON file containing saved arguments

        Returns:
            TrainingArguments: Reconstructed configuration object with all
                              settings from the saved file

        Note:
            The JSON file should contain a 'trainer_config_class' field
            for proper class resolution when using subclasses.
        """

        config_dict = cls._dict_from_json_file(json_file)
        return cls.load_from_json(config_dict)

    @classmethod
    def load_from_json(cls, config_dict):
        """Reconstruct a TrainingArguments instance from a parsed JSON dictionary.

        Args:
            config_dict: Dictionary parsed from a JSON configuration file.
                May contain a 'trainer_config_class' key for subclass resolution.

        Returns:
            TrainingArguments: Reconstructed configuration object.

        Raises:
            ValueError: If the trainer config class cannot be resolved.
        """
        config_dict = _apply_training_args_legacy_aliases(config_dict)
        if "trainer_config_class" in config_dict.keys():
            import easydel as ed

            cls = getattr(ed, config_dict.pop("trainer_config_class"))
            if cls is None:
                raise ValueError("We couldn't clarify the trainer config class from provided json.")
        return cls(**config_dict)

    def save_arguments(self, json_file_path: str | os.PathLike | ePathLike):
        """
        Save training arguments to a JSON file.

        This method serializes the current configuration to a JSON file,
        preserving all settings for later reconstruction. The saved file
        includes class information for proper deserialization.

        Args:
            json_file_path: Path where the JSON file will be saved.
                           Parent directories are created if needed.

        Note:
            The saved JSON includes a 'trainer_config_class' field to
            ensure proper class resolution when loading.
        """
        ePath(json_file_path).write_text(self.to_json_string())

    def _get_save_directory(self, create: bool = True) -> ePathLike | None:
        """Return the base checkpoint save directory.

        Args:
            create: If True, ensure the directory exists before returning.

        Returns:
            The checkpoint directory path, or None if no path is configured.
        """
        if create:
            self.ensure_checkpoint_path()
        return self.get_path()

    def _get_save_directory_milestone(self, step, create: bool = True) -> ePathLike:
        """Return the checkpoint directory for a specific training step.

        Creates a subdirectory named ``run-{step}`` under the base save
        directory.

        Args:
            step: The training step number used to name the subdirectory.
            create: If True, create the directory (and parents) if it does
                not exist.

        Returns:
            The milestone checkpoint directory path.
        """
        directory_name = f"run-{step}"
        savedir = self._get_save_directory(create=create)
        if savedir is None:
            return ePath("/dev/null")  # pyright: ignore[reportReturnType]
        save_directory = savedir / directory_name
        if create:
            save_directory.mkdir(exist_ok=True, parents=True)
        return save_directory

    __hash__ = hash_fn


def _get_max_sequence_length(self: TrainingArguments) -> int | None:
    """Getter for the deprecated ``max_sequence_length`` property.

    Returns:
        The current ``max_length`` value.
    """
    return self.max_length


def _set_max_sequence_length(self: TrainingArguments, value: int | None) -> None:
    """Setter for the deprecated ``max_sequence_length`` property.

    Emits a FutureWarning directing users to ``max_length``, then
    forwards the value.

    Args:
        value: The sequence length to set.
    """
    warnings.warn(
        "`max_sequence_length` is deprecated; use `max_length` instead.",
        FutureWarning,
        stacklevel=2,
    )
    self.max_length = value


TrainingArguments.max_sequence_length = property(
    _get_max_sequence_length,
    _set_max_sequence_length,
    doc="Deprecated alias for `max_length`.",
)


def _get_quantization_block(self: TrainingArguments) -> int | None:
    """Getter for the deprecated ``quantization_block`` property.

    Returns:
        The current ``quantization_group_size`` value.
    """
    return self.quantization_group_size


def _set_quantization_block(self: TrainingArguments, value: int | None) -> None:
    """Setter for the deprecated ``quantization_block`` property.

    Emits a FutureWarning directing users to ``quantization_group_size``,
    then forwards the value.

    Args:
        value: The quantization block size to set.
    """
    warnings.warn(
        "`quantization_block` is deprecated; use `quantization_group_size` instead.",
        FutureWarning,
        stacklevel=2,
    )
    self.quantization_group_size = value


TrainingArguments.quantization_block = property(
    _get_quantization_block,
    _set_quantization_block,
    doc="Deprecated alias for `quantization_group_size`.",
)
