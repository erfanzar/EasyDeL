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
and inference engine initialization.
"""

from __future__ import annotations

import json
import os
import pprint
import typing
from collections.abc import Mapping
from typing import Any, NotRequired, Unpack

import jax
from eformer.loggings import get_logger
from eformer.paths import ePathLike
from transformers import AutoTokenizer

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType

from .builders import (
    build_dataset,
    build_esurge,
    build_model,
    build_vsurge,
    to_data_mixture_kwargs,
    to_esurge_kwargs,
    to_from_pretrained_kwargs,
    to_vsurge_kwargs,
)
from .normalizer import materialize_base_config, normalize, resolve_task, validate
from .trainer_types import (
    get_trainer_class,
    get_training_arguments_class,
    normalize_trainer_config,
)
from .types import ELMConfig
from .utils import load_elm_config, save_elm_config

if typing.TYPE_CHECKING:
    from datasets import Dataset

    from easydel.trainers import Trainer
logger = get_logger("eLargeModel")


class BuildTrainerKws(typing.TypedDict, total=False):
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
    - Model building and initialization
    - eSurge inference engine integration
    - Tokenizer management

    Example:
        >>>
        >>> elm = eLargeModel({"model": {"name_or_path": "meta-llama/Llama-2-7b"}})
        >>> model = elm.build_model()
        >>>
        >>>
        >>> elm = eLargeModel.from_json("config.json")
        >>> esurge_engine = elm.build_esurge()
        >>>
        >>>
        >>> elm.to_json("my_config.json")
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
    def from_pretrained(
        cls,
        model_name_or_path: str,
        task: TaskType | str | None = None,
        **kwargs,
    ) -> eLargeModel:
        """Create eLargeModel from pretrained model name or path.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            task: Optional task type (auto-detected if not provided)
            **kwargs: Additional configuration options

        Returns:
            eLargeModel instance with configuration
        """
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
        """Get the normalized configuration."""
        return self._config

    @property
    def model_name(self) -> str:
        """Get the model name or path."""
        return self._config["model"]["name_or_path"]

    @property
    def task(self) -> TaskType:
        """Get the resolved task type."""
        return resolve_task(self._config)

    def update_config(self, updates: Mapping[str, Any]) -> eLargeModel:
        """Update configuration with new values.

        Args:
            updates: Dictionary with configuration updates

        Returns:
            Self for method chaining
        """
        from .utils import deep_merge

        self._config = normalize(deep_merge(self._config, updates))
        return self

    def set_model(self, model_name_or_path: str) -> eLargeModel:
        """Set the model name or path.

        Args:
            model_name_or_path: HuggingFace model ID or local path

        Returns:
            Self for method chaining
        """
        self._config["model"]["name_or_path"] = model_name_or_path
        return self

    def set_dtype(self, dtype: str) -> eLargeModel:
        """Set the data type for model loading.

        Args:
            dtype: Data type (e.g., "bf16", "fp16", "fp32")

        Returns:
            Self for method chaining
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

        Args:
            axis_dims: Sharding axis dimensions
            axis_names: Sharding axis names
            **kwargs: Additional sharding options

        Returns:
            Self for method chaining
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

        Args:
            method: Quantization method
            block_size: Quantization block size
            **kwargs: Additional quantization options

        Returns:
            Self for method chaining
        """
        quant = self._config.setdefault("quantization", {})
        if method is not None:
            quant["method"] = method
        quant["block_size"] = block_size
        quant.update(kwargs)
        return self

    def set_esurge(
        self,
        max_model_len: int | None = None,
        max_num_seqs: int = 16,
        hbm_utilization: float = 0.85,
        **kwargs,
    ) -> eLargeModel:
        """Configure eSurge inference settings.

        Args:
            max_model_len: Maximum model sequence length
            max_num_seqs: Maximum number of sequences
            hbm_utilization: HBM memory utilization ratio
            **kwargs: Additional eSurge options

        Returns:
            Self for method chaining
        """
        esurge = self._config.setdefault("esurge", {})
        if max_model_len is not None:
            esurge["max_model_len"] = max_model_len
        esurge["max_num_seqs"] = max_num_seqs
        esurge["hbm_utilization"] = hbm_utilization
        esurge.update(kwargs)
        return self

    def set_vsurge(
        self,
        max_concurrent_decodes: int | None = None,
        max_concurrent_prefill: int = 1,
        bytecode_decode: bool = False,
        **kwargs,
    ) -> eLargeModel:
        """Configure vSurge inference settings.

        Args:
            max_concurrent_decodes: Maximum concurrent decode calls
            max_concurrent_prefill: Maximum concurrent prefill steps
            bytecode_decode: Enable bytecode decoding
            **kwargs: Additional vSurge options

        Returns:
            Self for method chaining
        """
        vsurge = self._config.setdefault("vsurge", {})
        if max_concurrent_decodes is not None:
            vsurge["max_concurrent_decodes"] = max_concurrent_decodes
        vsurge["max_concurrent_prefill"] = max_concurrent_prefill
        vsurge["bytecode_decode"] = bytecode_decode
        vsurge.update(kwargs)
        return self

    def set_mixture(
        self,
        informs: list[dict] | None = None,
        batch_size: int = 32,
        streaming: bool = True,
        use_fast_loader: bool = True,
        **kwargs,
    ) -> eLargeModel:
        """Configure data mixture settings for training/evaluation.

        Args:
            informs: List of dataset configurations
            batch_size: Batch size for data loading
            streaming: Use streaming mode for large datasets
            use_fast_loader: Enable fast loading with fsspec
            **kwargs: Additional mixture options

        Returns:
            Self for method chaining

        Example:
            >>> elm.set_mixture(
            ...     informs=[
            ...         {"type": "json", "data_files": "train.json", "content_field": "text"},
            ...         {"type": "parquet", "data_files": "valid/*.parquet", "content_field": "content"}
            ...     ],
            ...     batch_size=32,
            ...     use_fast_loader=True
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

        Args:
            data_files: Path(s) to data files
            dataset_type: Dataset type (json, parquet, csv, etc.) or HF dataset ID
            content_field: Field containing text content
            split: Dataset split to use
            **kwargs: Additional dataset options

        Returns:
            Self for method chaining
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

        Args:
            max_new_tokens: Maximum tokens to generate (default: 2048)
            temperature: Sampling temperature (default: 0.0 for greedy)
            top_p: Top-p sampling parameter (default: 0.95)
            batch_size: Evaluation batch size (default: engine-specific)
            use_tqdm: Show progress bar (default: True)
            **kwargs: Additional evaluation options (see EvalKwargs)

        Returns:
            Self for method chaining
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

        Raises:
            ValueError: If configuration is invalid
        """
        validate(self._config)

    def to_json(self, json_path: str | os.PathLike | ePathLike) -> None:
        """Save configuration to JSON file.

        Args:
            json_path: Path to save JSON configuration
        """
        save_elm_config(self._config, json_path)

    def to_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return dict(self._config)

    def get_from_pretrained_kwargs(self) -> dict[str, Any]:
        """Get kwargs for model.from_pretrained() calls.

        Returns:
            Dictionary of from_pretrained arguments
        """
        return to_from_pretrained_kwargs(self._config)

    def get_esurge_kwargs(self) -> dict[str, Any]:
        """Get kwargs for eSurge initialization.

        Returns:
            Dictionary of eSurge arguments
        """
        return to_esurge_kwargs(self._config)

    def get_vsurge_kwargs(self) -> dict[str, Any]:
        """Get kwargs for vSurge initialization.

        Returns:
            Dictionary of vSurge arguments
        """
        return to_vsurge_kwargs(self._config)

    def get_base_config(self, prefer: str = "base") -> dict[str, Any]:
        """Get materialized base configuration.

        Args:
            prefer: Preference for base values or section values

        Returns:
            Base configuration dictionary
        """
        return materialize_base_config(self._config, prefer)

    def build_model(self, force_rebuild: bool = False) -> EasyDeLBaseModule:
        """Build the EasyDeL model from configuration.

        Args:
            force_rebuild: Force rebuilding even if model exists

        Returns:
            EasyDeLBaseModule instance
        """
        if self._model is None or force_rebuild:
            if not self.model_name:
                raise ValueError("Model name/path must be set before building")
            self._model = build_model(self._config)
        return self._model

    def build_tokenizer(self, force_rebuild: bool = False) -> AutoTokenizer:
        """Build or get the tokenizer for the model.

        Args:
            force_rebuild: Force rebuilding even if tokenizer exists

        Returns:
            AutoTokenizer instance
        """
        if self._tokenizer is None or force_rebuild:
            tok_path = self._config["model"].get("tokenizer", self.model_name)
            if not tok_path:
                raise ValueError("Tokenizer path must be set")
            self._tokenizer = AutoTokenizer.from_pretrained(tok_path)
        return self._tokenizer

    def build_esurge(self):
        """Build the eSurge inference engine.

        Returns:
            eSurge instance
        """
        self.build_model()
        return build_esurge(self._config, self._model)

    def build_vsurge(self):
        """Build the vSurge inference engine.

        Returns:
            vSurge instance
        """

        self.build_model()
        return build_vsurge(self._config, self._model)

    def build_dataset(self):
        """Build dataset from mixture configuration.

        Returns:
            Dataset: The loaded and processed dataset, or None if no mixture configured

        Example:
            >>> elm = eLargeModel()
            >>> elm.add_dataset("train.json", path="json", content_field="text")
            >>> dataset = elm.build_dataset()
        """
        return build_dataset(self._config)

    def get_data_mixture_kwargs(self) -> dict[str, Any]:
        """Get kwargs for DatasetMixture initialization.

        Returns:
            Dictionary of DatasetMixture arguments
        """
        return to_data_mixture_kwargs(self._config)

    def clear_cache(self) -> None:
        """Clear cached model, tokenizer, and inference engine instances."""
        self._model = None
        self._tokenizer = None

    def set_trainer(self, trainer_type: str, **kwargs) -> eLargeModel:
        """Configure trainer settings.

        Args:
            trainer_type: Type of trainer (dpo, orpo, grpo, sft, reward, distillation)
            **kwargs: Trainer-specific configuration options

        Returns:
            Self for method chaining
        """
        trainer_cfg = self._config.setdefault("trainer", {})
        trainer_cfg["trainer_type"] = trainer_type
        trainer_cfg.update(kwargs)
        return self

    def get_trainer_config(self) -> dict[str, Any]:
        """Get normalized trainer configuration.

        Returns:
            Normalized trainer configuration dictionary
        """
        raw_config = self._config.get("trainer", {})
        return normalize_trainer_config(raw_config)

    def train(
        self,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **build_kwargs: Unpack[BuildTrainerKws],
    ):
        """Train the model with the configured settings.

        This is a high-level convenience method that:
        1. Builds the model if not already built
        2. Creates the dataset from mixture configuration if not provided
        3. Creates the appropriate trainer
        4. Runs training

        Args:
            train_dataset: Optional training dataset, will use mixture if not provided
            eval_dataset: Optional evaluation dataset
            **trainer_kwargs: Additional kwargs to override trainer configuration

        Returns:
            Training results from the trainer

        Example:
            >>> elm = eLargeModel({
            ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
            ...     "mixture": {
            ...         "informs": [
            ...             {"type": "json", "data_files": "train.json", "content_field": "text"}
            ...         ]
            ...     },
            ...     "trainer": {
            ...         "trainer_type": "sft",
            ...         "learning_rate": 2e-5,
            ...         "num_train_epochs": 3,
            ...     }
            ... })
            >>> results = elm.train()
        """
        self.validate()

        trainer = self.build_trainer(train_dataset=train_dataset, eval_dataset=eval_dataset, **build_kwargs)

        logger.info("Starting training with configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Trainer Arguments:\n{pprint.pformat(self._config.get('trainer', {}))}")

        logger.info("Beginning training...")
        results = trainer.train()
        logger.info("Training completed successfully")
        self._model = trainer.model
        return results

    def build_training_arguments(self, **overrides):
        """Build TrainingArguments for the configured trainer.

        Args:
            **overrides: Override specific configuration values

        Returns:
            TrainingArguments instance for the configured trainer type
        """
        trainer_cfg = self.get_trainer_config()
        trainer_cfg.update(overrides)

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
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        reference_model: EasyDeLBaseModule | None = None,
        reward_model: EasyDeLBaseModule | None = None,
        teacher_model: EasyDeLBaseModule | None = None,
        reward_funcs: Any | None = None,
        **kwargs,
    ) -> Trainer:
        """Build a trainer instance with the configured settings.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            reference_model: Reference model for DPO/ORPO (optional)
            reward_model: Reward model for GRPO (optional)
            teacher_model: Teacher model for distillation (optional)
            reward_funcs: Reward functions for GRPO (optional)
            **kwargs: Additional trainer arguments

        Returns:
            Configured trainer instance
        """
        from easydel.infra.base_state import EasyDeLState

        trainer_cfg = self.get_trainer_config()
        trainer_type = trainer_cfg.get("trainer_type", "sft")

        if self._model is None:
            self.build_model()

        if self._tokenizer is None:
            self.build_tokenizer()

        trainer_class = get_trainer_class(trainer_type)
        training_args = self.build_training_arguments(**kwargs)

        if train_dataset is None and "mixture" in self._config:
            train_dataset = self.build_dataset()

        trainer_kwargs = {}

        if trainer_type == "base":
            trainer_kwargs["arguments"] = training_args
            if isinstance(self._model, EasyDeLState):
                trainer_kwargs["model_state"] = self._model
            else:
                trainer_kwargs["model"] = self._model
            trainer_kwargs["dataset_train"] = train_dataset
            trainer_kwargs["dataset_eval"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "dpo":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = self._model
            trainer_kwargs["reference_model"] = reference_model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "orpo":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = self._model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "grpo":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = self._model
            trainer_kwargs["reward_funcs"] = reward_funcs if reward_funcs is not None else reward_model
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["reward_processing_classes"] = kwargs.get("reward_processing_classes", None)
            trainer_kwargs["data_tokenize_fn"] = kwargs.get("data_tokenize_fn", None)

        elif trainer_type == "sft":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["model"] = self._model
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["formatting_func"] = kwargs.get("formatting_func", None)
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "reward":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = self._model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        elif trainer_type == "distillation":
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["student_model"] = self._model
            trainer_kwargs["teacher_model"] = teacher_model
            trainer_kwargs["processing_class"] = self._tokenizer
            trainer_kwargs["train_dataset"] = train_dataset
            trainer_kwargs["eval_dataset"] = eval_dataset
            trainer_kwargs["data_collator"] = kwargs.get("data_collator", None)

        else:
            trainer_kwargs["arguments"] = training_args
            trainer_kwargs["model"] = self._model
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
        engine: typing.Literal["esurge", "vsurge", "auto"] | Any = "auto",
        num_fewshot: int = 0,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Run evaluation on specified tasks using lm-evaluation-harness.

        This method provides a unified interface for evaluating models using either
        eSurge or vSurge engines with the lm-evaluation-harness framework.

        Args:
            tasks: Task name(s) to evaluate on. Can be a single task string or list of tasks.
                Common tasks include: 'gsm8k', 'hellaswag', 'mmlu', 'truthfulqa', etc.
            engine: Inference engine to use. Options:
                - "esurge": Use eSurge engine (better for large batches)
                - "vsurge": Use vSurge engine (better for streaming)
                - "auto": Automatically select based on configuration
                - An existing eSurge/vSurge instance
            num_fewshot: Number of few-shot examples to use (default: 0 for zero-shot)
            batch_size: Evaluation batch size. If None, uses engine's default
            output_path: Optional path to save evaluation results JSON
            use_tqdm: Show progress bar during evaluation
            **eval_kwargs: Additional arguments passed to the evaluator

        Returns:
            Dictionary containing evaluation results with structure:
            {
                "results": {task_name: {metric: value, ...}, ...},
                "versions": {task_name: version, ...},
                "config": {...evaluation config...}
            }

        Example:
            >>>
            >>> elm = eLargeModel.from_pretrained("meta-llama/Llama-2-7b")
            >>> results = elm.eval("gsm8k", num_fewshot=5)
            >>> print(results["results"]["gsm8k"]["acc"])

            >>>
            >>> elm.set_esurge(max_num_seqs=64, bytecode_decode=True)
            >>> results = elm.eval(
            ...     ["hellaswag", "mmlu", "truthfulqa"],
            ...     engine="esurge",
            ...     batch_size=32,
            ...     output_path="eval_results.json"
            ... )

            >>>
            >>> surge = elm.build_vsurge()
            >>> results = elm.eval("arc_easy", engine=surge, num_fewshot=25)

        Raises:
            ImportError: If lm-eval is not installed
            ValueError: If invalid engine type or model not configured
            RuntimeError: If evaluation fails
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
                if self._config.get("esurge"):
                    engine = "esurge"
                elif self._config.get("vsurge"):
                    engine = "vsurge"
                else:
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

            elif engine == "vsurge":
                from easydel.inference.evaluations import vSurgeLMEvalAdapter

                engine_instance = self.build_vsurge()

                if batch_size is None:
                    batch_size = self._config.get("vsurge", {}).get("max_concurrent_decodes", jax.device_count())

                eval_adapter = vSurgeLMEvalAdapter(
                    surge=engine_instance,
                    processor=self._tokenizer,
                    max_length=self._model.config.granted_mask_max_position_embedding if self._model else 8192,
                    max_new_tokens=max_new_tokens,
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

            elif "vSurge" in engine_type:
                from easydel.inference.evaluations import vSurgeLMEvalAdapter

                if batch_size is None:
                    batch_size = getattr(engine.driver, "max_concurrent_decodes", jax.device_count())

                eval_adapter = vSurgeLMEvalAdapter(
                    surge=engine,
                    processor=self._tokenizer,
                    max_length=getattr(engine.driver.engine.model.config, "granted_mask_max_position_embedding", 8192),
                    max_new_tokens=max_new_tokens,
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
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
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
        """String representation of eLargeModel."""
        task_str = f"TaskType.{self.task.name}" if self.task else "None"
        dtype_str = repr(self._config.get("loader", {}).get("dtype", "default"))

        quant = self._config.get("quantization", {})
        quant_str = f", quantization={quant['method']!r}" if quant.get("method") else ""

        sharding = self._config.get("sharding", {})
        axis_dims = sharding.get("axis_dims")
        shard_str = f", sharding={axis_dims}" if axis_dims and axis_dims != (1, 1, 1, -1, 1) else ""

        return f"eLargeModel(model={self.model_name!r}, task={task_str}, dtype={dtype_str}{quant_str}{shard_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = ["╭─── eLargeModel Configuration ───╮"]

        lines.append("│ 📦 Model")
        lines.append(f"│   • Name: {self.model_name or 'Not set'}")
        lines.append(f"│   • Task: {self.task.name if self.task else 'Auto-detect'}")
        tokenizer = self._config.get("model", {}).get("tokenizer")
        if tokenizer and tokenizer != self.model_name:
            lines.append(f"│   • Tokenizer: {tokenizer}")

        loader = self._config.get("loader", {})
        if loader:
            lines.append("│ ⚙️  Loading")
            lines.append(f"│   • Dtype: {loader.get('dtype', 'default')}")
            if loader.get("param_dtype") and loader["param_dtype"] != loader.get("dtype"):
                lines.append(f"│   • Param dtype: {loader['param_dtype']}")
            if loader.get("precision"):
                lines.append(f"│   • Precision: {loader['precision']}")
            if loader.get("from_torch") is not None:
                lines.append(f"│   • From PyTorch: {loader['from_torch']}")

        sharding = self._config.get("sharding", {})
        if sharding.get("axis_dims") and sharding["axis_dims"] != (1, 1, 1, -1, 1):
            lines.append("│ 🔀 Sharding")
            lines.append(f"│   • Dimensions: {sharding['axis_dims']}")
            if sharding.get("axis_names"):
                lines.append(f"│   • Axis names: {sharding['axis_names']}")
            if sharding.get("shard_attention_computation") is False:
                lines.append(f"│   • Shard attention: {sharding['shard_attention_computation']}")

        quant = self._config.get("quantization", {})
        if quant.get("method"):
            lines.append("│ 📉 Quantization")
            lines.append(f"│   • Method: {quant['method']}")
            if quant.get("block_size"):
                lines.append(f"│   • Block size: {quant['block_size']}")
            if quant.get("platform"):
                lines.append(f"│   • Platform: {quant['platform']}")

        esurge = self._config.get("esurge", {})
        if esurge:
            has_esurge_config = any(
                esurge.get(k) is not None for k in ["max_model_len", "max_num_seqs", "hbm_utilization"]
            )
            if has_esurge_config:
                lines.append("│ 🚀 eSurge")
                if esurge.get("max_model_len"):
                    lines.append(f"│   • Max length: {esurge['max_model_len']:,}")
                if esurge.get("max_num_seqs"):
                    lines.append(f"│   • Max sequences: {esurge['max_num_seqs']}")
                if esurge.get("hbm_utilization"):
                    lines.append(f"│   • HBM utilization: {esurge['hbm_utilization']:.0%}")
                if esurge.get("page_size") and esurge["page_size"] != 128:
                    lines.append(f"│   • Page size: {esurge['page_size']}")

        trainer = self._config.get("trainer", {})
        if trainer.get("trainer_type"):
            lines.append("│ 🎯 Training")
            lines.append(f"│   • Type: {trainer['trainer_type'].upper()}")
            if trainer.get("learning_rate"):
                lines.append(f"│   • Learning rate: {trainer['learning_rate']:.2e}")
            if trainer.get("num_train_epochs"):
                lines.append(f"│   • Epochs: {trainer['num_train_epochs']}")
            if trainer.get("total_batch_size"):
                lines.append(f"│   • Batch size: {trainer['total_batch_size']}")
            if trainer.get("output_dir"):
                lines.append(f"│   • Output: {trainer['output_dir']}")

        platform = self._config.get("platform", {})
        if platform:
            if platform.get("backend") or platform.get("platform"):
                lines.append("│ 💻 Platform")
                if platform.get("backend"):
                    lines.append(f"│   • Backend: {platform['backend']}")
                if platform.get("platform"):
                    lines.append(f"│   • Hardware: {platform['platform']}")

        vsurge = self._config.get("vsurge", {})
        if vsurge:
            has_vsurge_config = any(
                vsurge.get(k) is not None for k in ["max_concurrent_decodes", "bytecode_decode", "interleaved_mode"]
            )
            if has_vsurge_config:
                lines.append("│ ⚡ vSurge")
                if vsurge.get("max_concurrent_decodes"):
                    lines.append(f"│   • Max decodes: {vsurge['max_concurrent_decodes']}")
                if vsurge.get("max_concurrent_prefill"):
                    lines.append(f"│   • Max prefill: {vsurge['max_concurrent_prefill']}")
                if vsurge.get("bytecode_decode"):
                    lines.append(f"│   • Bytecode decode: {vsurge['bytecode_decode']}")
                if vsurge.get("interleaved_mode"):
                    lines.append(f"│   • Interleaved mode: {vsurge['interleaved_mode']}")

        lines.append("│ 📊 Status")
        lines.append(f"│   • Model loaded: {'✓' if self._model is not None else '✗'}")
        lines.append(f"│   • Tokenizer loaded: {'✓' if self._tokenizer is not None else '✗'}")

        lines.append("╰────────────────────────────────╯")

        return "\n".join(lines)
