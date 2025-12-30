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


"""eLargeModel - Easy Large Models master class for EasyDeL.

This module provides a unified interface for working with large language models
in the EasyDeL framework, combining configuration management, model building,
training orchestration, and inference engine initialization.

Key Features:
    - Unified configuration management for models, training, and inference
    - Automatic model and tokenizer initialization from HuggingFace or local paths
    - Support for multiple training paradigms (SFT, DPO, ORPO, GRPO, distillation)
    - Integration with the eSurge inference engine
    - Built-in evaluation with lm-evaluation-harness
    - Flexible dataset mixture configuration
    - Model sharding and quantization support
"""

from __future__ import annotations

import json
import os
import pprint
import typing
from collections.abc import Mapping
from typing import Any, NotRequired, Unpack

from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from transformers import AutoTokenizer

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.factory import TaskType
from easydel.trainers.training_configurations import TrainingArguments

from .builders import (
    build_dataset,
    build_esurge,
    build_model,
    build_sharded_source,
    to_data_mixture_kwargs,
    to_esurge_kwargs,
    to_from_pretrained_kwargs,
)
from .normalizer import materialize_base_config, normalize, resolve_task, validate
from .trainer_types import get_trainer_class, get_training_arguments_class, normalize_trainer_config
from .types import ELMConfig
from .utils import load_elm_config, save_elm_config

if typing.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource
    from easydel.inference import eSurge
    from easydel.trainers import Trainer

    from .types import TextDatasetInformCfg, VisualDatasetInformCfg
logger = get_logger("eLargeModel")


class BuildTrainerKws(typing.TypedDict, total=False):
    """Type hints for optional keyword arguments when building trainers.

    Attributes:
        data_collator: Custom data collator for batching examples
        formatting_func: Function to format examples for SFT training
        reward_processing_classes: Processing classes for reward models in GRPO
        data_tokenize_fn: Custom tokenization function for data preprocessing
        reference_model: Reference model for DPO/preference optimization
        reward_model: Reward model for GRPO training
        teacher_model: Teacher model for distillation training
        reward_funcs: Custom reward functions for GRPO
    """

    data_collator: NotRequired[typing.Callable]
    formatting_func: NotRequired[typing.Callable]
    reward_processing_classes: NotRequired[list[typing.Callable]]
    data_tokenize_fn: NotRequired[typing.Callable]
    reference_model: NotRequired[EasyDeLBaseModule | None]
    reward_model: NotRequired[EasyDeLBaseModule | None]
    teacher_model: NotRequired[EasyDeLBaseModule | None]
    reward_funcs: NotRequired[Any | None]


class eLargeModel:
    """Master class for Easy Large Models (ELM) in EasyDeL.

    This class provides a unified interface for:
    - Configuration management (load, save, create)
    - Model building and initialization (including teacher/reference models)
    - Training orchestration with multiple paradigms (SFT, DPO, ORPO, etc.)
    - eSurge inference engine integration
    - Tokenizer management
    - Dataset mixture configuration
    - Model evaluation with lm-evaluation-harness

    Attributes:
        config: The normalized ELM configuration dictionary
        model_name: The model name or path from configuration
        task: The resolved task type (auto-detected or specified)
        teacher_model_name: Teacher model name for distillation (if configured)
        reference_model_name: Reference model name for DPO/ORPO (if configured)

    Example:
        Basic model loading:
        >>> elm = eLargeModel({"model": {"name_or_path": "meta-llama/Llama-2-7b"}})
        >>> model = elm.build_model()

        From pretrained with configuration:
        >>> elm = eLargeModel.from_pretrained(
        ...     "meta-llama/Llama-2-7b",
        ...     task="causal-lm"
        ... )
        >>> elm.set_dtype("bf16")
        >>> elm.set_sharding(axis_dims=(1, 2, 1, -1))

        Loading from JSON configuration:
        >>> elm = eLargeModel.from_json("config.json")
        >>> esurge_engine = elm.build_esurge()

        Training with SFT:
        >>> elm.set_trainer("sft", learning_rate=2e-5, num_train_epochs=3)
        >>> elm.add_dataset("train.json", dataset_type="json", content_field="text")
        >>> results = elm.train()

        Evaluation:
        >>> results = elm.eval(["hellaswag", "mmlu"], engine="esurge")
    """

    def __init__(self, config: ELMConfig | Mapping[str, Any] | str | os.PathLike | ePathLike | None = None):
        """Initialize eLargeModel with configuration.

        Args:
            config: Can be:
                - ELMConfig or dict with configuration
                - Path to JSON configuration file
                - None to create empty configuration
        """
        if config is None:
            self._config = normalize({"model": {"name_or_path": ""}})
        elif isinstance(config, str | os.PathLike) or hasattr(config, "__fspath__"):
            self._config = load_elm_config(config)
        else:
            self._config = normalize(config)

        self._model = None
        self._tokenizer = None

    @classmethod
    def from_json(cls, json_path: str | os.PathLike | ePathLike) -> eLargeModel:
        """Create eLargeModel from JSON configuration file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            eLargeModel instance
        """
        return cls(load_elm_config(json_path))

    @classmethod
    def from_yaml(cls, yaml_path: str | os.PathLike | ePathLike) -> eLargeModel:
        """Create eLargeModel from a YAML configuration file.

        The YAML file may either contain the ELM config directly, or wrap it under
        a `config`/`elarge_model`/`elm` key. If the YAML file also contains a
        top-level `actions` key (e.g. used by `easydel.scripts.elarge`), that key
        is ignored when loading the ELM configuration.
        """
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PyYAML is required for YAML configs. Install with: pip install pyyaml") from e

        yaml_path = ePath(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(f"YAML root must be a mapping/dict, got {type(raw).__name__}.")

        config = None
        for key in ("config", "elarge_model", "elm"):
            if key in raw:
                config = raw[key]
                break
        if config is None:
            config = {k: v for k, v in raw.items() if k != "actions"}
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"Config must be a mapping/dict, got {type(config).__name__}.")
        return cls(config)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        task: TaskType | str | None = None,
        **kwargs,
    ) -> eLargeModel:
        """Create eLargeModel from pretrained model name or path.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            task: Optional task type (auto-detected if not provided or AUTO_BIND)
            **kwargs: Additional configuration options

        Returns:
            eLargeModel instance with configuration
        """
        from easydel.modules.auto.auto_configuration import infer_task_from_hf_config

        from .utils import normalize_task

        # Auto-detect task if None or AUTO_BIND
        if task is None or task == TaskType.AUTO_BIND or task == "auto-bind":
            inferred_task = infer_task_from_hf_config(model_name_or_path)
            if inferred_task is not None:
                task = inferred_task
                logger.info(f"Auto-detected task type: {task.value}")
        else:
            normalized = normalize_task(task)
            if normalized is not None:
                task = normalized

        config = {
            "model": {
                "name_or_path": model_name_or_path,
                **({"task": task} if task else {}),
            },
            **kwargs,
        }
        return cls(config)

    @property
    def config(self) -> ELMConfig:
        """Get the normalized configuration dictionary.

        Returns:
            The full ELM configuration including model, loader, sharding,
            quantization, training, and inference settings.
        """
        return self._config

    @property
    def model_name(self) -> str:
        """Get the model name or path.

        Returns:
            The HuggingFace model ID or local path to the model.
        """
        return self._config["model"]["name_or_path"]

    @property
    def task(self) -> TaskType:
        """Get the resolved task type.

        Returns:
            The task type (e.g., TaskType.CAUSAL_LM) either explicitly
            configured or auto-detected from the model.
        """
        return resolve_task(self._config)

    @property
    def teacher_model_name(self) -> str | None:
        """Get the teacher model name or path for distillation.

        Returns:
            The teacher model path if configured, None otherwise.
        """
        return self._config.get("teacher_model", {}).get("name_or_path")

    @property
    def reference_model_name(self) -> str | None:
        """Get the reference model name or path for DPO/ORPO.

        Returns:
            The reference model path if configured, None otherwise.
        """
        return self._config.get("reference_model", {}).get("name_or_path")

    def update_config(self, updates: Mapping[str, Any]) -> eLargeModel:
        """Update configuration with new values.

        Performs a deep merge of the updates into the existing configuration,
        preserving nested structures. The configuration is re-normalized after
        updating to ensure consistency.

        Args:
            updates: Dictionary with configuration updates. Can include nested
                structures like {"model": {"dtype": "bf16"}, "esurge": {"max_num_seqs": 32}}

        Returns:
            Self for method chaining

        Example:
            >>> elm.update_config({
            ...     "loader": {"dtype": "bf16"},
            ...     "esurge": {"max_model_len": 4096}
            ... })
        """
        from .utils import deep_merge

        self._config = normalize(deep_merge(self._config, updates))
        return self

    def set_model(self, model_name_or_path: str) -> eLargeModel:
        """Set the model name or path.

        Updates the primary model configuration. This will clear any cached
        model instance to ensure the new model is loaded on next build.

        Args:
            model_name_or_path: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")
                or local path to model directory

        Returns:
            Self for method chaining

        Example:
            >>> elm.set_model("meta-llama/Llama-2-7b-hf")
            >>> elm.set_model("/path/to/local/model")
        """
        self._config["model"]["name_or_path"] = model_name_or_path
        return self

    def set_teacher_model(self, model_name_or_path: str) -> eLargeModel:
        """Set the teacher model name or path for distillation training.

        Configures a teacher model used for knowledge distillation. The teacher
        model is typically a larger, more capable model that guides the training
        of the student (primary) model.

        Args:
            model_name_or_path: HuggingFace model ID or local path for teacher model.
                Should be a model compatible with the student model's architecture.

        Returns:
            Self for method chaining

        Example:
            >>> elm.set_model("meta-llama/Llama-2-7b")  # Student model
            >>> elm.set_teacher_model("meta-llama/Llama-2-13b")  # Teacher model
            >>> elm.set_trainer("distillation", temperature=3.0)
        """
        if "teacher_model" not in self._config:
            self._config["teacher_model"] = {}
        self._config["teacher_model"]["name_or_path"] = model_name_or_path
        return self

    def set_reference_model(self, model_name_or_path: str) -> eLargeModel:
        """Set the reference model name or path for preference optimization.

        Configures a reference model used in DPO (Direct Preference Optimization)
        and similar preference-based training methods. The reference model provides
        a baseline for computing preference losses.

        Args:
            model_name_or_path: HuggingFace model ID or local path for reference model.
                Often the same as the base model before fine-tuning.

        Returns:
            Self for method chaining

        Example:
            >>> elm.set_model("meta-llama/Llama-2-7b-hf")  # Model to train
            >>> elm.set_reference_model("meta-llama/Llama-2-7b-hf")  # Reference
            >>> elm.set_trainer("dpo", beta=0.1)
        """
        if "reference_model" not in self._config:
            self._config["reference_model"] = {}
        self._config["reference_model"]["name_or_path"] = model_name_or_path
        return self

    def set_dtype(self, dtype: str) -> eLargeModel:
        """Set the data type for model loading.

        Configures both the computation dtype and parameter dtype for the model.
        This affects memory usage and computation speed.

        Args:
            dtype: Data type string. Supported values:
                - "bf16": BFloat16 (recommended for TPU, modern GPUs)
                - "fp16": Float16 (good for older GPUs)
                - "fp32": Float32 (highest precision, most memory)
                - "fp8": Float8 (experimental, requires compatible hardware)

        Returns:
            Self for method chaining

        Example:
            >>> elm.set_dtype("bf16")  # Use bfloat16 for training/inference
        """
        self._config.setdefault("loader", {})["dtype"] = dtype
        self._config["loader"]["param_dtype"] = dtype
        return self

    def set_sharding(
        self,
        axis_dims: tuple[int, ...] | None = None,
        axis_names: tuple[str, ...] | None = None,
        **kwargs,
    ) -> eLargeModel:
        """Configure model sharding for distributed training/inference.

        Sets up model parallelism by specifying how to shard model parameters
        and computations across devices. Essential for training large models
        that don't fit on a single device.

        Args:
            axis_dims: Sharding axis dimensions as a tuple. Common patterns:
                - (1, 1, 1, -1): Data parallel only
                - (2, 1, 1, -1): 2-way tensor parallel
                - (1, 2, 1, -1): 2-way pipeline parallel
                - (2, 2, 1, -1): 2-way tensor + 2-way pipeline parallel
            axis_names: Sharding axis names (e.g., ("dp", "tp", "pp", "sp"))
                - dp: Data parallel
                - tp: Tensor parallel
                - pp: Pipeline parallel
                - sp: Sequence parallel
            **kwargs: Additional sharding options.

        Returns:
            Self for method chaining

        Example:
            >>> # 2-way tensor parallel, 2-way data parallel
            >>> elm.set_sharding(
            ...     axis_dims=(2, 2, 1, -1),
            ...     axis_names=("dp", "tp", "pp", "sp")
            ... )
        """
        sharding = self._config.setdefault("sharding", {})
        if axis_dims is not None:
            sharding["axis_dims"] = axis_dims
        if axis_names is not None:
            sharding["axis_names"] = axis_names
        sharding.update(kwargs)
        return self

    def set_quantization(
        self,
        method: str | None = None,
        block_size: int = 128,
        **kwargs,
    ) -> eLargeModel:
        """Configure quantization settings.

        Enables model quantization to reduce memory usage and potentially
        improve inference speed at the cost of some accuracy.

        Args:
            method: Quantization method.
            block_size: Quantization block size (default: 128).
                Smaller blocks = better accuracy but more overhead.
            **kwargs: Additional quantization options:
                - platform: Target platform ("cpu", "cuda", "tpu")
                - compute_dtype: Dtype for computation (e.g., "fp16")
                - double_quant: Enable double quantization for 4bit

        Returns:
            Self for method chaining

        Example:
            >>> elm.set_quantization("nf4", block_size=64)
            >>> elm.set_quantization("a8bit")
        """
        quant = self._config.setdefault("quantization", {})
        if method is not None:
            quant["method"] = method
        quant["block_size"] = block_size
        quant.update(kwargs)
        return self

    def set_operation_configs(
        self,
        configs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> eLargeModel:
        """Configure ejkernel operation overrides.

        Allows overriding ejkernel's autotune behavior for specific attention
        operations by providing explicit configuration objects. When a config
        is provided, it's passed directly to the operation instead of using
        ejkernel's autotune.

        Args:
            configs: Dictionary mapping operation names to config objects.
                Valid operation names (must match OperationRegistry):
                - "flash_attn2": Flash attention 2
                - "ring": Ring attention
                - "blocksparse": Block sparse attention
                - "ragged_page_attention_v2": Ragged page attention v2
                - "ragged_page_attention_v3": Ragged page attention v3
                - "sdpa": Scaled dot product attention
                - "vanilla": Vanilla attention
            **kwargs: Individual operation configs as keyword arguments.
                These are merged with the configs dict.

        Returns:
            Self for method chaining

        Example:
            >>> from easydel import FlashAttentionConfig, RingAttentionConfig
            >>>
            >>> # Using dict
            >>> elm.set_operation_configs({
            ...     "flash_attn2": FlashAttentionConfig(platform="triton"),
            ...     "ring": RingAttentionConfig(),
            ... })
            >>>
            >>> # Using kwargs
            >>> elm.set_operation_configs(
            ...     flash_attn2=FlashAttentionConfig(platform="pallas"),
            ... )
        """
        base_cfg = self._config.setdefault("base_config", {})
        op_configs = base_cfg.setdefault("operation_configs", {})

        if configs is not None:
            op_configs.update(configs)
        op_configs.update(kwargs)
        return self

    def set_esurge(
        self,
        max_model_len: int | None = None,
        max_num_seqs: int = 16,
        hbm_utilization: float = 0.85,
        **kwargs,
    ) -> eLargeModel:
        """Configure eSurge inference settings.

        eSurge is a high-performance batch inference engine optimized for
        throughput. It uses PagedAttention for efficient memory management.

        Args:
            max_model_len: Maximum model sequence length (input + output tokens).
                If None, uses model's default max position embeddings.
            max_num_seqs: Maximum number of sequences to process concurrently.
                Higher values increase throughput but require more memory.
            hbm_utilization: HBM memory utilization ratio (0.0-1.0).
                Controls how much device memory to use for KV cache.
            **kwargs: Additional eSurge options:
                - page_size: PagedAttention page size (default: 128)
                - enable_prefix_caching: Enable prefix caching optimization
                - kv_cache_dtype: Dtype for KV cache (None = auto)
                - decoding_engine: "ring" or "triton" (default: auto)

        Returns:
            Self for method chaining

        Example:
            >>> elm.set_esurge(
            ...     max_model_len=8192,
            ...     max_num_seqs=64,
            ...     hbm_utilization=0.9,
            ...     enable_prefix_caching=True
            ... )
        """
        esurge = self._config.setdefault("esurge", {})
        if max_model_len is not None:
            esurge["max_model_len"] = max_model_len
        esurge["max_num_seqs"] = max_num_seqs
        esurge["hbm_utilization"] = hbm_utilization
        esurge.update(kwargs)
        return self

    def set_mixture(
        self,
        informs: list[TextDatasetInformCfg | VisualDatasetInformCfg] | None = None,
        batch_size: int = 32,
        streaming: bool = True,
        use_fast_loader: bool = True,
        **kwargs,
    ) -> eLargeModel:
        """Configure data mixture settings for training/evaluation.

        Sets up a mixture of datasets that can be combined and sampled from
        during training. Supports multiple data sources and formats.

        Args:
            informs: List of dataset configurations. Each dict should contain:
                - type: Dataset type ("json", "parquet", "csv", "text", or HF dataset ID)
                - data_files: Path or pattern to data files
                - content_field: Field name containing the text content
                - split: Dataset split to use (default: "train")
                - weight: Sampling weight for this dataset (optional)
            batch_size: Batch size for data loading (default: 32)
            streaming: Use streaming mode for large datasets (default: True).
                Reduces memory usage but may be slower.
            use_fast_loader: Enable fast loading with fsspec (default: True).
                Provides optimized loading for remote/cloud storage.
            **kwargs: Additional mixture options:
                - max_samples: Maximum samples per dataset
                - shuffle: Whether to shuffle data
                - seed: Random seed for shuffling

        Returns:
            Self for method chaining

        Example:
            >>> elm.set_mixture(
            ...     informs=[
            ...         {"type": "json", "data_files": "train.json", "content_field": "text", "weight": 0.7},
            ...         {"type": "parquet", "data_files": "valid/*.parquet", "content_field": "content", "weight": 0.3}
            ...     ],
            ...     batch_size=32,
            ...     streaming=True,
            ...     shuffle=True,
            ...     seed=42
            ... )
        """
        mixture = self._config.setdefault("mixture", {})
        if informs is not None:
            mixture["informs"] = informs
        mixture["batch_size"] = batch_size
        mixture["streaming"] = streaming
        mixture["use_fast_loader"] = use_fast_loader
        mixture.update(kwargs)
        return self

    def add_dataset(
        self,
        data_files: str | list[str],
        dataset_type: str | None = None,
        content_field: str = "content",
        split: str = "train",
        **kwargs,
    ) -> eLargeModel:
        """Add a dataset to the mixture configuration.

        Appends a new dataset to the existing mixture. Multiple datasets
        can be added and will be combined during training.

        Args:
            data_files: Path(s) to data files. Can be:
                - Single file: "data.json"
                - Multiple files: ["data1.json", "data2.json"]
                - Glob pattern: "data/*.parquet"
                - Remote URL: "https://example.com/data.json"
            dataset_type: Dataset type or format. Options:
                - File formats: "json", "jsonl", "parquet", "csv", "text"
                - HuggingFace dataset ID: "imdb", "squad", etc.
                - None: Auto-detect from file extension
            content_field: Field name containing the text content (default: "content").
                For chat data, might be "messages" or "conversations".
            split: Dataset split to use (default: "train").
                Common values: "train", "validation", "test".
            **kwargs: Additional dataset options:
                - weight: Sampling weight for this dataset
                - max_samples: Maximum samples to use
                - filter_fn: Function to filter samples
                - map_fn: Function to transform samples

        Returns:
            Self for method chaining

        Example:
            >>> # Add a JSON dataset
            >>> elm.add_dataset("train.json", dataset_type="json", content_field="text")
            >>>
            >>> # Add a HuggingFace dataset
            >>> elm.add_dataset("imdb", dataset_type="imdb", split="train")
            >>>
            >>> # Add multiple Parquet files with sampling weight
            >>> elm.add_dataset(
            ...     "data/*.parquet",
            ...     dataset_type="parquet",
            ...     content_field="content",
            ...     weight=0.5
            ... )
        """
        mixture = self._config.setdefault("mixture", {})
        informs = mixture.setdefault("informs", [])

        inform = {
            "data_files": data_files,
            "content_field": content_field,
            "split": split,
            **kwargs,
        }

        if dataset_type:
            inform["type"] = dataset_type

        informs.append(inform)
        return self

    def set_eval(
        self,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        batch_size: int | None = None,
        use_tqdm: bool = True,
        **kwargs,
    ) -> eLargeModel:
        """Configure evaluation settings for lm-evaluation-harness.

        Sets default parameters for model evaluation on standard benchmarks.
        These settings apply when using the eval() method.

        Args:
            max_new_tokens: Maximum tokens to generate per evaluation sample
                (default: 2048). Lower values speed up evaluation.
            temperature: Sampling temperature (default: 0.0 for greedy decoding).
                0.0 = deterministic/greedy, higher = more random.
            top_p: Top-p (nucleus) sampling parameter (default: 0.95).
                Only used when temperature > 0.
            batch_size: Evaluation batch size (default: engine-specific).
                Higher values increase throughput but use more memory.
            use_tqdm: Show progress bar during evaluation (default: True)
            **kwargs: Additional evaluation options:
                - top_k: Top-k sampling parameter
                - repetition_penalty: Penalty for repeated tokens
                - num_beams: Beam search width (1 = greedy)
                - do_sample: Whether to use sampling
                - early_stopping: Stop generation at first EOS

        Returns:
            Self for method chaining

        Example:
            >>> # Configure for deterministic evaluation
            >>> elm.set_eval(
            ...     max_new_tokens=512,
            ...     temperature=0.0,
            ...     batch_size=64
            ... )
            >>>
            >>> # Configure for sampling-based evaluation
            >>> elm.set_eval(
            ...     temperature=0.7,
            ...     top_p=0.9,
            ...     top_k=50
            ... )
        """
        eval_cfg = self._config.setdefault("eval", {})
        eval_cfg["max_new_tokens"] = max_new_tokens
        eval_cfg["temperature"] = temperature
        eval_cfg["top_p"] = top_p
        if batch_size is not None:
            eval_cfg["batch_size"] = batch_size
        eval_cfg["use_tqdm"] = use_tqdm
        eval_cfg.update(kwargs)
        return self

    def validate(self) -> None:
        """Validate the current configuration.

        Checks that all required fields are present and have valid values.
        This is automatically called before training or building engines.

        Raises:
            ValueError: If configuration is invalid (e.g., missing model name,
                invalid dtype, incompatible settings)
        """
        validate(self._config)

    def to_json(self, json_path: str | os.PathLike | ePathLike) -> None:
        """Save configuration to JSON file.

        Exports the current configuration to a JSON file that can be loaded
        later with from_json() or shared with others.

        Args:
            json_path: Path where the JSON configuration file will be saved.
                Will create parent directories if they don't exist.

        Example:
            >>> elm.to_json("config.json")
            >>> # Later or on another machine:
            >>> elm2 = eLargeModel.from_json("config.json")
        """
        save_elm_config(self._config, json_path)

    def to_yaml(self, yaml_path: str | os.PathLike | ePathLike) -> None:
        """Save configuration to a YAML file."""
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PyYAML is required for YAML configs. Install with: pip install pyyaml") from e

        def to_yamlable(obj: Any) -> Any:
            if obj is None or isinstance(obj, str | int | float | bool):
                return obj
            if isinstance(obj, dict):
                return {str(k): to_yamlable(v) for k, v in obj.items()}
            if isinstance(obj, list | tuple | set):
                return [to_yamlable(v) for v in obj]
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return to_yamlable(obj.to_dict())
            if hasattr(obj, "value") and not isinstance(obj, type):  # enums
                return to_yamlable(obj.value)
            if isinstance(obj, os.PathLike) or hasattr(obj, "__fspath__"):
                return os.fspath(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not YAML serializable")

        data = to_yamlable(self._config)

        yaml_path = ePath(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary.

        Returns a copy of the full configuration dictionary that can be
        modified without affecting the eLargeModel instance.

        Returns:
            Configuration dictionary with all settings

        Example:
            >>> config_dict = elm.to_dict()
            >>> print(config_dict["model"]["name_or_path"])
            >>> # Modify the dict without affecting elm
            >>> config_dict["model"]["dtype"] = "fp16"
        """
        return dict(self._config)

    def get_from_pretrained_kwargs(self) -> dict[str, Any]:
        """Get kwargs for model.from_pretrained() calls.

        Extracts and formats the configuration options that should be passed
        to the model's from_pretrained() method, including dtype, sharding,
        and quantization settings.

        Returns:
            Dictionary of from_pretrained arguments ready to use with
            EasyDeL model loading functions

        Example:
            >>> kwargs = elm.get_from_pretrained_kwargs()
            >>> # Can be used directly:
            >>> model = LlamaForCausalLM.from_pretrained(
            ...     "meta-llama/Llama-2-7b",
            ...     **kwargs
            ... )
        """
        return to_from_pretrained_kwargs(self._config)

    def get_esurge_kwargs(self) -> dict[str, Any]:
        """Get kwargs for eSurge initialization.

        Extracts and formats the configuration options for creating an
        eSurge engine instance.

        Returns:
            Dictionary of eSurge arguments including max_model_len,
            max_num_seqs, hbm_utilization, and other engine settings

        Example:
            >>> kwargs = elm.get_esurge_kwargs()
            >>> # Can be used directly:
            >>> from easydel.inference import eSurge
            >>> engine = eSurge(model, **kwargs)
        """
        return to_esurge_kwargs(self._config)

    def get_base_config(self, prefer: str = "base") -> dict[str, Any]:
        """Get materialized base configuration.

        Resolves the configuration hierarchy, materializing shared base
        settings across different configuration sections.

        Args:
            prefer: Resolution preference when conflicts exist:
                - "base": Prefer values from base configuration
                - "section": Prefer values from specific sections

        Returns:
            Base configuration dictionary with resolved values

        Example:
            >>> # Get configuration with base values taking precedence
            >>> base_config = elm.get_base_config(prefer="base")
            >>> print(base_config["dtype"])  # Shows the base dtype setting
        """
        return materialize_base_config(self._config, prefer)

    def build_model(self, force_rebuild: bool = False) -> EasyDeLBaseModule:
        """Build the EasyDeL model from configuration.

        Loads the model using the configured settings including dtype,
        sharding, and quantization. The model is cached after first build
        unless force_rebuild is True.

        Args:
            force_rebuild: Force rebuilding even if model is already cached.
                Useful when configuration has changed.

        Returns:
            EasyDeLBaseModule instance ready for training or inference

        Raises:
            ValueError: If model name/path is not set
            RuntimeError: If model loading fails

        Example:
            >>> elm = eLargeModel.from_pretrained("meta-llama/Llama-2-7b")
            >>> elm.set_dtype("bf16")
            >>> model = elm.build_model()
        """
        if self._model is None or force_rebuild:
            if not self.model_name:
                raise ValueError("Model name/path must be set before building")
            self._model = build_model(self._config)
        return self._model

    def build_tokenizer(self, force_rebuild: bool = False) -> AutoTokenizer:
        """Build or get the tokenizer for the model.

        Loads the tokenizer from the model path or a separately specified
        tokenizer path. The tokenizer is cached after first build.

        Args:
            force_rebuild: Force rebuilding even if tokenizer is already cached.
                Useful when switching between different tokenizers.

        Returns:
            AutoTokenizer instance configured for the model

        Raises:
            ValueError: If tokenizer path cannot be determined

        Example:
            >>> tokenizer = elm.build_tokenizer()
            >>> tokens = tokenizer("Hello world", return_tensors="np")
        """
        if self._tokenizer is None or force_rebuild:
            tok_path = self._config["model"].get("tokenizer", self.model_name)
            if not tok_path:
                raise ValueError("Tokenizer path must be set")
            trust_remote_code = bool(self._config.get("loader", {}).get("trust_remote_code", False))
            self._tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=trust_remote_code)
        return self._tokenizer

    def build_esurge(self) -> "eSurge":
        """Build the eSurge inference engine.

        Creates an eSurge engine instance configured with the current settings.
        Automatically builds the model if not already built.

        Returns:
            eSurge instance ready for batch inference

        Example:
            >>> elm.set_esurge(max_num_seqs=32, hbm_utilization=0.9)
            >>> engine = elm.build_esurge()
            >>> # Use engine for batch inference
            >>> results = engine.generate(prompts, max_tokens=100)
        """
        self.build_model()
        return build_esurge(self._config, self._model)

    def build_teacher_model(self) -> EasyDeLBaseModule | None:
        """Build the teacher model for distillation training.

        Loads the teacher model using the same loader configuration as the
        student model (dtype, sharding, etc.) but with the teacher model path.

        Returns:
            EasyDeLBaseModule instance for the teacher model, or None if no
            teacher model is configured

        Example:
            >>> elm.set_teacher_model("meta-llama/Llama-2-13b")
            >>> teacher = elm.build_teacher_model()
            >>> # Teacher model will be used automatically in distillation training
        """
        if "teacher_model" not in self._config:
            return None

        teacher_config = dict(self._config)
        teacher_config["model"] = self._config["teacher_model"]
        return build_model(teacher_config)

    def build_reference_model(self) -> EasyDeLBaseModule | None:
        """Build the reference model for preference optimization (DPO, etc.).

        Loads the reference model using the same loader configuration as the
        primary model. The reference model provides a baseline for computing
        preference losses in DPO, ORPO, and similar methods.

        Returns:
            EasyDeLBaseModule instance for the reference model, or None if no
            reference model is configured

        Example:
            >>> elm.set_reference_model("meta-llama/Llama-2-7b-hf")
            >>> reference = elm.build_reference_model()
            >>> # Reference model will be used automatically in DPO training
        """
        if "reference_model" not in self._config:
            return None

        reference_config = dict(self._config)
        reference_config["model"] = self._config["reference_model"]
        return build_model(reference_config)

    def build_reward_model(self) -> EasyDeLBaseModule | None:
        """Build the reward model for RLHF trainers (GRPO/GFPO/GSPO/XPO/Nash-MD).

        Loads the reward model using the same loader configuration as the
        primary model, but with the `reward_model` configuration block.

        Returns:
            EasyDeLBaseModule instance for the reward model, or None if no
            reward model is configured.

        Notes:
            Reward models are typically sequence-classification heads with `num_labels=1`.
            In YAML, set for example:
                reward_model:
                  name_or_path: Qwen/Qwen2.5-0.5B-Instruct
                  task: sequence-classification
                  extra_kwargs:
                    num_labels: 1
        """
        if "reward_model" not in self._config:
            return None

        reward_config = dict(self._config)
        reward_config["model"] = self._config["reward_model"]
        return build_model(reward_config)

    def build_dataset(self) -> "Dataset | IterableDataset | None":
        """Build dataset from mixture configuration.

        Creates a dataset from the configured mixture of data sources.
        Supports multiple formats (JSON, Parquet, CSV) and can combine
        multiple data sources into a single dataset.

        Returns:
            Dataset: The loaded and processed dataset ready for training,
                or None if no mixture is configured

        Example:
            >>> elm = eLargeModel()
            >>> elm.add_dataset("train.json", dataset_type="json", content_field="text")
            >>> elm.add_dataset("valid/*.parquet", dataset_type="parquet", content_field="content")
            >>> dataset = elm.build_dataset()
            >>> print(f"Dataset size: {len(dataset)}")
        """
        return build_dataset(self._config)

    def build_sharded_source(self) -> "ShardedDataSource | None":
        """Build dataset as ShardedDataSource for use with new data pipeline.

        Creates a ShardedDataSource from the configured mixture of data sources.
        This uses the new data architecture that supports lazy transforms,
        efficient streaming, and better integration with trainers.

        Returns:
            ShardedDataSource: The data source ready for training, or None
                if no mixture is configured

        Example:
            >>> elm = eLargeModel()
            >>> elm.add_dataset("train.json", dataset_type="json", content_field="text")
            >>> elm.set_mixture(use_sharded_source=True)
            >>> source = elm.build_sharded_source()
            >>> for batch in source.open_shard(source.shard_names[0]):
            ...     process(batch)
        """
        return build_sharded_source(self._config)

    def get_train_source(self) -> "ShardedDataSource | Dataset | IterableDataset | None":
        """Get training data as ShardedDataSource or Dataset.

        Automatically selects the appropriate data format based on the
        `use_sharded_source` configuration option.

        Returns:
            ShardedDataSource if use_sharded_source=True in mixture config,
            otherwise HuggingFace Dataset. Returns None if no mixture configured.

        Example:
            >>> elm = eLargeModel()
            >>> elm.add_dataset("train.json", dataset_type="json")
            >>> elm.set_mixture(use_sharded_source=True)  # Use new pipeline
            >>> data = elm.get_train_source()  # Returns ShardedDataSource
            >>>
            >>> elm.set_mixture(use_sharded_source=False)  # Use legacy pipeline
            >>> data = elm.get_train_source()  # Returns HF Dataset
        """
        mixture_cfg = self._config.get("mixture", {})
        if mixture_cfg.get("use_sharded_source", True):
            return self.build_sharded_source()
        return self.build_dataset()

    def get_data_mixture_kwargs(self) -> dict[str, Any]:
        """Get kwargs for DatasetMixture initialization.

        Extracts and formats the mixture configuration for use with
        the DatasetMixture class.

        Returns:
            Dictionary of DatasetMixture arguments including informs,
            batch_size, streaming settings, and other mixture options
        """
        return to_data_mixture_kwargs(self._config)

    def clear_cache(self) -> None:
        """Clear cached model, tokenizer, and inference engine instances.

        This is useful when you want to reload models with different
        configurations or free memory after model operations.
        """
        self._model = None
        self._tokenizer = None

    def set_trainer(self, trainer_type: str, **kwargs) -> eLargeModel:
        """Configure trainer settings.

        Sets the training paradigm and associated hyperparameters.

        Args:
            trainer_type: Type of trainer to use:
                - "sft": Supervised Fine-Tuning
                - "dpo": Direct Preference Optimization
                - "orpo": Odds Ratio Preference Optimization
                - "grpo": Group Relative Policy Optimization
                - "reward": Reward model training
                - "distillation": Knowledge distillation
                - "base": Basic trainer for custom training loops
            **kwargs: Trainer-specific configuration options:
                Common options:
                - learning_rate: Learning rate (default: 5e-5)
                - num_train_epochs: Number of training epochs
                - per_device_train_batch_size: Batch size per device
                - gradient_accumulation_steps: Gradient accumulation steps
                - warmup_steps: Number of warmup steps
                - output_dir: Directory to save checkpoints
                DPO-specific:
                - beta: KL regularization coefficient
                - loss_type: "sigmoid", "ipo", "hinge"
                Distillation-specific:
                - temperature: Distillation temperature
                - alpha: Weight for distillation loss

        Returns:
            Self for method chaining

        Example:
            >>> # SFT training
            >>> elm.set_trainer(
            ...     "sft",
            ...     learning_rate=2e-5,
            ...     num_train_epochs=3,
            ...     per_device_train_batch_size=4
            ... )
            >>>
            >>> # DPO training
            >>> elm.set_trainer(
            ...     "dpo",
            ...     beta=0.1,
            ...     learning_rate=1e-6
            ... )
        """
        trainer_cfg = self._config.setdefault("trainer", {})
        trainer_cfg["trainer_type"] = trainer_type
        trainer_cfg.update(kwargs)
        return self

    def get_trainer_config(self) -> dict[str, Any]:
        """Get normalized trainer configuration.

        This method processes the raw trainer configuration and applies
        defaults and normalization for the specified trainer type.

        Returns:
            Normalized trainer configuration dictionary with all required
            fields populated with defaults where necessary.
        """
        raw_config = self._config.get("trainer", {})
        return normalize_trainer_config(raw_config)

    def train(
        self,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        base_state_class: type[EasyDeLState] | None = None,
        args_class: type[TrainingArguments] | None = None,
        trainer_class: type[Trainer] | None = None,
        **build_kwargs: Unpack[BuildTrainerKws],
    ):
        """Train the model with the configured settings.

        This is a high-level convenience method that orchestrates the entire
        training pipeline:
        1. Validates configuration
        2. Builds the model if not already built
        3. Creates the dataset from mixture configuration if not provided
        4. Builds reference/teacher models if needed
        5. Creates the appropriate trainer
        6. Runs training and returns results

        Args:
            train_dataset: Optional training dataset (Dataset or ShardedDataSource).
                If None, will build from mixture configuration.
            eval_dataset: Optional evaluation dataset for validation during training.
            base_state_class: Optional custom EasyDeLState class for model state
                management. Use for custom model implementations.
            args_class: Optional custom TrainingArguments class. If None, will
                auto-select based on trainer_type.
            trainer_class: Optional custom Trainer class. If None, will auto-select
                based on trainer_type.
            **build_kwargs: Additional kwargs for trainer building:
                - data_collator: Custom data collator function
                - formatting_func: Function to format examples (SFT)
                - reward_processing_classes: Processing classes for rewards (GRPO)
                - data_tokenize_fn: Custom tokenization function
                - reference_model: Override reference model
                - reward_model: Override reward model
                - teacher_model: Override teacher model
                - reward_funcs: Custom reward functions

        Returns:
            Training results from the trainer, including metrics and final model state

        Example:
            Basic SFT training:
            >>> elm = eLargeModel.from_pretrained("meta-llama/Llama-2-7b")
            >>> elm.add_dataset("train.json", dataset_type="json")
            >>> elm.set_trainer("sft", learning_rate=2e-5, num_train_epochs=3)
            >>> results = elm.train()

            DPO training with custom datasets:
            >>> train_data = load_dataset("preference_data", split="train")
            >>> eval_data = load_dataset("preference_data", split="test")
            >>> elm.set_trainer("dpo", beta=0.1)
            >>> elm.set_reference_model("meta-llama/Llama-2-7b")
            >>> results = elm.train(train_dataset=train_data, eval_dataset=eval_data)

            Custom trainer with formatting function:
            >>> def format_fn(examples):
            ...     return [f"Question: {q}\nAnswer: {a}"
            ...             for q, a in zip(examples["question"], examples["answer"])]
            >>> results = elm.train(formatting_func=format_fn)
        """
        self.validate()

        trainer = self.build_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            base_state_class=base_state_class,
            args_class=args_class,
            trainer_class=trainer_class,
            **build_kwargs,
        )

        logger.info("Starting training with configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Trainer Arguments:\n{pprint.pformat(self._config.get('trainer', {}))}")

        logger.info("Beginning training...")
        results = trainer.train()
        logger.info("Training completed successfully")
        self._model = trainer.model
        return results

    def build_training_arguments(self, args_class: TrainingArguments | None = None, **overrides):
        """Build TrainingArguments for the configured trainer.

        Args:
            args_class: Optional custom TrainingArguments class. If not provided,
                will automatically select based on trainer_type.
            **overrides: Override specific configuration values

        Returns:
            TrainingArguments instance for the configured trainer type
            (e.g., DPOConfig for DPO training, SFTConfig for SFT)
        """
        trainer_cfg = self.get_trainer_config()
        trainer_cfg.update(overrides)

        if args_class is None:
            trainer_type = trainer_cfg.get("trainer_type", "sft")
            args_class = get_training_arguments_class(trainer_type)

        config_for_args = {k: v for k, v in trainer_cfg.items() if k != "trainer_type"}

        try:
            return args_class(**config_for_args)
        except TypeError:
            import inspect

            sig = inspect.signature(args_class.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
            filtered_config = {k: v for k, v in config_for_args.items() if k in valid_params}
            return args_class(**filtered_config)

    def build_trainer(
        self,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        reference_model: EasyDeLBaseModule | None = None,
        reward_model: EasyDeLBaseModule | None = None,
        teacher_model: EasyDeLBaseModule | None = None,
        reward_funcs: Any | None = None,
        base_state_class: type[EasyDeLState] | None = None,
        args_class: type[TrainingArguments] | None = None,
        trainer_class: type[Trainer] | None = None,
        **kwargs,
    ) -> Trainer:
        """Build a trainer instance with the configured settings.

        Creates and configures a trainer based on the trainer_type setting.
        Automatically builds required models and datasets if not provided.

        Args:
            train_dataset: Training dataset (Dataset or ShardedDataSource).
                If None, builds from mixture config using get_train_source().
            eval_dataset: Evaluation dataset for validation metrics.
            reference_model: Reference model for DPO/ORPO. If None, builds from
                reference_model configuration if present.
            reward_model: Reward model for GRPO. If None, builds from config.
            teacher_model: Teacher model for distillation. If None, builds from
                teacher_model configuration if present.
            reward_funcs: Custom reward functions for GRPO. Alternative to reward_model.
            base_state_class: Custom EasyDeLState class for model state management.
            args_class: Custom TrainingArguments class. Auto-selected if None.
            trainer_class: Custom Trainer class. Auto-selected if None.
            **kwargs: Additional trainer configuration overrides

        Returns:
            Configured trainer instance ready for training

        Raises:
            ValueError: If required models or datasets are not configured

        Example:
            >>> # Build trainer with auto-configuration
            >>> trainer = elm.build_trainer()
            >>>
            >>> # Build trainer with custom dataset
            >>> custom_data = load_dataset("custom_data")
            >>> trainer = elm.build_trainer(train_dataset=custom_data)
            >>>
            >>> # Build DPO trainer with custom reference model
            >>> ref_model = elm.build_reference_model()
            >>> trainer = elm.build_trainer(
            ...     trainer_type="dpo",
            ...     reference_model=ref_model
            ... )
        """
        from easydel.infra.base_state import EasyDeLState

        trainer_cfg = self.get_trainer_config()
        trainer_type = trainer_cfg.get("trainer_type", "sft")

        if self._model is None:
            self.build_model()

        if self._tokenizer is None:
            self.build_tokenizer()

        if trainer_class is None:
            trainer_class = get_trainer_class(trainer_type)

        training_args = self.build_training_arguments(args_class=args_class, **kwargs)

        if train_dataset is None and "mixture" in self._config:
            # Use new get_train_source() which auto-selects based on use_sharded_source
            train_dataset = self.get_train_source()

        trainer_kwargs = {}
        model = self._model

        if base_state_class is not None:
            model = model.to_state(base_state_class)
        if trainer_type == "base":
            trainer_kwargs["arguments"] = training_args
            if isinstance(model, EasyDeLState):
                trainer_kwargs["model_state"] = model
            else:
                trainer_kwargs["model"] = model
            trainer_kwargs["dataset_train"] = train_dataset
            trainer_kwargs["dataset_eval"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "dpo":
            if reference_model is None:
                reference_model = self.build_reference_model()

            if reference_model is not None and base_state_class is not None:
                reference_model = reference_model.to_state(base_state_class)

            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["reference_model"] = reference_model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type in {"bco", "kto"}:
            if reference_model is None:
                reference_model = self.build_reference_model()

            if reference_model is not None and base_state_class is not None:
                reference_model = reference_model.to_state(base_state_class)

            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["reference_model"] = reference_model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "orpo":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type in {"grpo", "gfpo", "gspo"}:
            if reward_funcs is None and reward_model is None:
                reward_model = self.build_reward_model()

            resolved_reward = reward_funcs if reward_funcs is not None else reward_model
            if resolved_reward is None:
                raise ValueError(
                    f"{trainer_type} training requires `reward_model` (config key) or `reward_funcs` (runtime kwarg)."
                )

            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["reward_funcs"] = resolved_reward
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["reward_processing_classes"] = kwargs.get("reward_processing_classes", None)
            trainer_kwargs["data_tokenize_fn"] = kwargs.get("data_tokenize_fn", None)

        elif trainer_type == "ppo":
            if reward_funcs is None and reward_model is None:
                reward_model = self.build_reward_model()

            resolved_reward = reward_funcs if reward_funcs is not None else reward_model
            if resolved_reward is None:
                raise ValueError(
                    "ppo training requires `reward_model` (config key) or `reward_funcs` (runtime kwarg)."
                )

            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["reward_funcs"] = resolved_reward
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["reward_processing_classes"] = kwargs.get("reward_processing_classes", None)
            trainer_kwargs["data_tokenize_fn"] = kwargs.get("data_tokenize_fn", None)

        elif trainer_type in {"xpo", "nash-md", "nash_md"}:
            if reward_funcs is None and reward_model is None:
                reward_model = self.build_reward_model()

            resolved_reward = reward_funcs if reward_funcs is not None else reward_model
            if resolved_reward is None:
                raise ValueError(
                    f"{trainer_type} training requires `reward_model` (config key) or `reward_funcs` (runtime kwarg)."
                )

            if reference_model is None:
                reference_model = self.build_reference_model()

            if reference_model is not None and base_state_class is not None:
                reference_model = reference_model.to_state(base_state_class)

            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["reward_funcs"] = resolved_reward
            trainer_kwargs["reference_model"] = reference_model
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["reward_processing_classes"] = kwargs.get("reward_processing_classes", None)

        elif trainer_type == "sft":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["model"] = model
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["formatting_func"] = kwargs.get("formatting_func", None)
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "reward":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "gkd":
            if teacher_model is None:
                teacher_model = self.build_teacher_model()

            if teacher_model is not None and base_state_class is not None:
                teacher_model = teacher_model.to_state(base_state_class)

            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["teacher_model"] = teacher_model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["formatting_func"] = kwargs.get("formatting_func", None)
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "distillation":
            if teacher_model is None:
                teacher_model = self.build_teacher_model()

            if teacher_model is not None and base_state_class is not None:
                teacher_model = teacher_model.to_state(base_state_class)

            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["student_model"] = model
            trainer_kwargs["teacher_model"] = teacher_model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        else:
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset

        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

        for key, value in kwargs.items():
            if key not in trainer_kwargs and value is not None:
                if key not in ["data_collator", "formatting_func", "reward_processing_classes", "data_tokenize_fn"]:
                    trainer_kwargs[key] = value

        return trainer_class(**trainer_kwargs)

    def eval(
        self,
        tasks: str | list[str],
        engine: typing.Literal["esurge", "auto"] | Any = "auto",
        num_fewshot: int = 0,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Run evaluation on specified tasks using lm-evaluation-harness.

        This method provides a unified interface for evaluating models using the
        eSurge engine with the lm-evaluation-harness framework.

        Args:
            tasks: Task name(s) to evaluate on. Can be a single task string or list of tasks.
                Common tasks include:
                - Language understanding: "hellaswag", "winogrande", "piqa", "arc_easy", "arc_challenge"
                - Math: "gsm8k", "math", "minerva_math"
                - Knowledge: "mmlu", "triviaqa", "naturalquestions"
                - Reasoning: "bbh", "boolq", "copa"
                - Truthfulness: "truthfulqa_mc1", "truthfulqa_mc2"
                - Coding: "humaneval", "mbpp"
                Full list: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
            engine: Inference engine to use. Options:
                - "esurge": Use eSurge engine (high throughput)
                - "auto": Automatically select based on configuration (default)
                - An existing eSurge instance for custom configuration
            num_fewshot: Number of few-shot examples to use (default: 0 for zero-shot).
                Different tasks may have different recommended values:
                - MMLU: typically 5-shot
                - GSM8K: typically 8-shot
                - HellaSwag: typically 0-shot
            output_path: Optional path to save evaluation results as JSON.
                Results include detailed metrics, task versions, and configuration.

        Returns:
            Dictionary containing evaluation results with structure:
            {
                "results": {
                    task_name: {
                        metric_name: value,  # e.g., "acc": 0.85, "acc_stderr": 0.02
                        ...
                    },
                    ...
                },
                "versions": {task_name: version_string, ...},
                "config": {"model": ..., "num_fewshot": ..., ...}
            }

        Example:
            Basic zero-shot evaluation:
            >>> elm = eLargeModel.from_pretrained("meta-llama/Llama-2-7b")
            >>> results = elm.eval("hellaswag")
            >>> print(f"HellaSwag accuracy: {results['results']['hellaswag']['acc']:.2%}")

            Few-shot evaluation with multiple tasks:
            >>> elm.set_esurge(max_num_seqs=64, hbm_utilization=0.9)
            >>> results = elm.eval(
            ...     ["gsm8k", "mmlu", "truthfulqa_mc1"],
            ...     engine="esurge",
            ...     num_fewshot=5,
            ...     output_path="eval_results.json"
            ... )
            >>> for task, metrics in results["results"].items():
            ...     print(f"{task}: {metrics.get('acc', metrics.get('exact_match')):.2%}")

            Evaluation with custom settings:
            >>> elm.set_eval(
                ...     max_new_tokens=512,
            ...     temperature=0.0,  # Greedy decoding
            ...     batch_size=32
            ... )
            >>> results = elm.eval(["humaneval", "mbpp"])

        Raises:
            ImportError: If lm-eval is not installed (install with: pip install lm-eval)
            ValueError: If invalid engine type or model not configured
            RuntimeError: If evaluation fails during execution

        Note:
            The evaluation uses settings from set_eval() for generation parameters.
            Default settings are optimized for deterministic evaluation (temperature=0).
        """
        try:
            from lm_eval import evaluator  # type:ignore
        except ImportError as e:
            raise ImportError("lm-eval is required for evaluation. Install with: pip install lm-eval") from e

        if isinstance(tasks, str):
            tasks = [tasks]

        if self._tokenizer is None:
            self.build_tokenizer()

        eval_config = self._config.get("eval", {}).copy()
        batch_size = eval_config.pop("batch_size", None)
        max_new_tokens = eval_config.pop("max_new_tokens", 2048)
        temperature = eval_config.pop("temperature", 0.0)
        top_p = eval_config.pop("top_p", 0.95)

        eval_adapter = None
        engine_instance = None

        if isinstance(engine, str):
            if engine == "auto":
                engine = "esurge"

            if engine == "esurge":
                from easydel.inference.evaluations import eSurgeLMEvalAdapter

                engine_instance = self.build_esurge()

                if batch_size is None:
                    batch_size = self._config.get("esurge", {}).get("max_num_seqs", 32)

                eval_adapter = eSurgeLMEvalAdapter(
                    surge=engine_instance,
                    processor=self._tokenizer,
                    max_length=self._config.get("esurge", {}).get("max_model_len", 8192),
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                raise ValueError(f"Unknown engine type: {engine}")

        else:
            engine_type = type(engine).__name__

            if "eSurge" in engine_type:
                from easydel.inference.evaluations import eSurgeLMEvalAdapter

                if batch_size is None:
                    batch_size = getattr(engine, "max_num_seqs", 32)

                eval_adapter = eSurgeLMEvalAdapter(
                    surge=engine,
                    processor=self._tokenizer,
                    max_length=getattr(engine, "max_model_len", 8192),
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                    temperature=temperature,
                    top_p=top_p,
                )

            else:
                raise ValueError(f"Unknown engine instance type: {engine_type}")

        if eval_adapter is None:
            raise RuntimeError("Failed to create evaluation adapter")

        try:
            logger.info(f"Starting evaluation on tasks: {tasks}")
            logger.info(f"Using {engine if isinstance(engine, str) else type(engine).__name__} engine")
            logger.info(f"Batch size: {batch_size}, Few-shot: {num_fewshot}")

            results = evaluator.simple_evaluate(
                model=eval_adapter,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device="cpu",
                **eval_config,
            )

            if output_path:
                ePath(output_path).write_text(json.dumps(results, indent=2))
                logger.info(f"eval results saved to: {output_path}")

            logger.info("evaluation summary:")

            for task, metrics in results.get("results", {}).items():
                logger.info(f"{task}:")
                for metric, value in metrics.items():
                    if isinstance(value, int | float):
                        logger.info(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

            return results

        finally:
            if hasattr(eval_adapter, "stop"):
                eval_adapter.stop()

    def __repr__(self) -> str:
        """Developer-friendly string representation of eLargeModel.

        Returns:
            A concise representation showing key configuration like model,
            task, dtype, quantization, and sharding.
        """
        task_str = f"TaskType.{self.task.name}" if self.task else "None"
        dtype_str = repr(self._config.get("loader", {}).get("dtype", "default"))

        quant = self._config.get("quantization", {})
        quant_str = f", quantization={quant['method']!r}" if quant.get("method") else ""

        sharding = self._config.get("sharding", {})
        axis_dims = sharding.get("axis_dims")
        shard_str = f", sharding={axis_dims}" if axis_dims and axis_dims != (1, 1, 1, -1, 1) else ""

        return f"eLargeModel(model={self.model_name!r}, task={task_str}, dtype={dtype_str}{quant_str}{shard_str})"

    def __str__(self) -> str:
        """Human-readable string representation with formatted configuration.

        Returns:
            A nicely formatted multi-line string showing all configured
            components including model, loading options, sharding,
            quantization, training, and inference settings.
        """
        w = 53  # inner width (content area)

        def _fmt(val: Any) -> str:
            """Format a value for display, handling enums and special types."""
            if val is None:
                return "none"
            if hasattr(val, "value"):
                return str(val.value) if val.value else "none"
            if hasattr(val, "name"):
                return val.name.lower()
            return str(val)

        def _line(content: str) -> str:
            """Create a padded line within the box."""
            return f"║ {content:<{w}}║"

        def _sep() -> str:
            """Create a separator line."""
            return f"╟{'─' * (w + 1)}╢"

        lines = []
        lines.append(f"╔{'═' * (w + 1)}╗")
        lines.append(f"║{'eLargeModel':^{w + 1}}║")
        lines.append(f"║{self.model_name or 'not set':^{w + 1}}║")
        lines.append(f"╠{'═' * (w + 1)}╣")

        # Loader & Task
        loader = self._config.get("loader", {})
        dtype_str = loader.get("dtype", "default")
        prec_str = loader.get("precision", "default")
        task_str = self.task.name.lower() if self.task else "auto"
        lines.append(_line(f"▸ dtype: {dtype_str:<12} ▸ task: {task_str}"))
        lines.append(_line(f"▸ precision: {prec_str}"))

        # Sharding
        sharding = self._config.get("sharding", {})
        if sharding.get("axis_dims"):
            dims = sharding["axis_dims"]
            auto = "auto" if sharding.get("auto_shard_model") else "manual"
            dims_str = ",".join(str(d) for d in dims)
            lines.append(_line(f"▸ shard: ({dims_str}) {auto}"))

        lines.append(_sep())

        # Config section
        base_cfg = self._config.get("base_config", {}).get("values", {})
        if base_cfg:
            if "attn_mechanism" in base_cfg:
                lines.append(_line(f"▸ attn: {_fmt(base_cfg['attn_mechanism'])}"))
            if "moe_method" in base_cfg:
                lines.append(_line(f"▸ moe:  {_fmt(base_cfg['moe_method'])}"))
            if "gradient_checkpointing" in base_cfg:
                gc = base_cfg["gradient_checkpointing"]
                gc_str = _fmt(gc) or "disabled"
                lines.append(_line(f"▸ grad_ckpt: {gc_str}"))

        # Quantization
        quant = self._config.get("quantization", {})
        if quant.get("method"):
            lines.append(_line(f"▸ quant: {quant['method']} (block:{quant.get('block_size', 128)})"))

        # eSurge
        esurge = self._config.get("esurge", {})
        if esurge and any(esurge.get(k) for k in ["max_model_len", "max_num_seqs", "hbm_utilization"]):
            lines.append(_sep())
            lines.append(_line("eSurge"))

            ctx = esurge.get("max_model_len", 0)
            seqs = esurge.get("max_num_seqs", 0)
            hbm = esurge.get("hbm_utilization", 0)
            lines.append(_line(f"▸ {ctx:,} context x {seqs} sequences"))

            parts = []
            if hbm:
                parts.append(f"{hbm:.0%} HBM")
            if esurge.get("page_size"):
                parts.append(f"page:{esurge['page_size']}")
            if esurge.get("enable_prefix_caching"):
                parts.append("prefix_cache")
            if esurge.get("min_input_pad"):
                parts.append(f"pad:{esurge['min_input_pad']}")
            if parts:
                lines.append(_line(f"▸ {' │ '.join(parts)}"))

        # Training
        trainer = self._config.get("trainer", {})
        if trainer.get("trainer_type"):
            lines.append(_sep())
            lines.append(_line(f"Training: {trainer['trainer_type'].upper()}"))
            parts = []
            if trainer.get("learning_rate"):
                parts.append(f"lr:{trainer['learning_rate']:.2e}")
            if trainer.get("num_train_epochs"):
                parts.append(f"epochs:{trainer['num_train_epochs']}")
            if trainer.get("total_batch_size"):
                parts.append(f"batch:{trainer['total_batch_size']}")
            if parts:
                lines.append(_line(f"▸ {' │ '.join(parts)}"))

        # Status
        lines.append(_sep())
        model_icon = "●" if self._model is not None else "○"
        tok_icon = "●" if self._tokenizer is not None else "○"
        model_status = "loaded" if self._model is not None else "not loaded"
        tok_status = "loaded" if self._tokenizer is not None else "not loaded"
        lines.append(_line(f"{model_icon} model: {model_status}   {tok_icon} tokenizer: {tok_status}"))
        lines.append(f"╚{'═' * (w + 1)}╝")

        return "\n".join(lines)
