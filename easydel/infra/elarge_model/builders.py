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


"""Builder functions for creating models and inference engines from ELM configurations.

This module provides high-level functions to build EasyDeL models and eSurge inference
engines from ELM configuration dictionaries.
"""

from __future__ import annotations

import pathlib
import typing as tp
from collections.abc import Mapping
from typing import Any

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizerBase

    from easydel.data.core.protocols import ShardedDataSource

from easydel.inference.esurge.esurge_engine import DEFAULT_DETOKENIZER_MAX_STATES
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType
from easydel.layers.quantization.quantizers import EasyDeLQuantizationConfig
from easydel.modules.auto import (
    AutoEasyDeLAnyToAnyModel,
    AutoEasyDeLModel,
    AutoEasyDeLModelForCausalLM,
    AutoEasyDeLModelForDiffusionLM,
    AutoEasyDeLModelForImageTextToText,
    AutoEasyDeLModelForSeq2SeqLM,
    AutoEasyDeLModelForSequenceClassification,
    AutoEasyDeLModelForSpeechSeq2Seq,
    AutoEasyDeLModelForZeroShotImageClassification,
)

from .normalizer import materialize_base_config, normalize, resolve_task
from .types import ELMConfig
from .utils import coerce_dtype, coerce_precision


def to_from_pretrained_kwargs(cfg_like: ELMConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Convert ELM configuration to kwargs for model.from_pretrained() calls.

    Extracts and transforms configuration values from various sections into
    the format expected by EasyDeL's from_pretrained methods.

    Args:
        cfg_like: ELM configuration dictionary or mapping

    Returns:
        Dictionary of keyword arguments for from_pretrained() methods

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "loader": {"dtype": "bf16"},
        ...     "sharding": {"axis_dims": (1, 1, 1, -1, 1)}
        ... }
        >>> kwargs = to_from_pretrained_kwargs(cfg)
        >>> model = AutoEasyDeLModelForCausalLM.from_pretrained(**kwargs)
    """
    cfg = normalize(cfg_like)
    model = cfg["model"]
    loader = cfg.get("loader", {})
    sharding = cfg.get("sharding", {})
    platform = cfg.get("platform", {})
    quant = cfg.get("quantization", {})

    config_kwargs = materialize_base_config(cfg, prefer="base")

    config_kwargs.pop("partition_axis", None)
    config_kwargs.pop("backend", None)
    config_kwargs.pop("platform", None)
    quant_model = quant.get("model")
    if quant_model is not None:
        quant_model = EasyDeLQuantizationConfig(**quant_model)
    return dict(
        pretrained_model_name_or_path=model["name_or_path"],
        device=loader.get("device"),
        dtype=coerce_dtype(loader.get("dtype")),
        param_dtype=coerce_dtype(loader.get("param_dtype")),
        precision=coerce_precision(loader.get("precision")),
        sharding_axis_dims=tuple(sharding.get("axis_dims", (1, 1, 1, -1, 1))),
        sharding_dcn_axis_dims=tuple(sharding["dcn_axis_dims"]) if sharding.get("dcn_axis_dims") else None,
        sharding_axis_names=tuple(sharding.get("axis_names", ("dp", "fsdp", "ep", "tp", "sp"))),
        partition_axis=sharding.get("partition_axis"),
        shard_fns=sharding.get("shard_fns"),
        backend=platform.get("backend"),
        platform=platform.get("platform"),
        config_kwargs=config_kwargs,
        auto_shard_model=bool(sharding.get("auto_shard_model", True)),
        partition_rules=sharding.get("partition_rules"),
        quantization_config=quant_model,
        quantize_tensors=bool(quant.get("quantize_tensors", False)),
        verbose=bool(loader.get("verbose", True)),
        from_torch=loader.get("from_torch"),
        trust_remote_code=loader.get("trust_remote_code", False),
        **(model.get("extra_kwargs") or {}),
    )


def build_model(cfg_like: ELMConfig | Mapping[str, Any]) -> EasyDeLBaseModule:
    """Build an EasyDeL model from ELM configuration.

    Automatically selects the appropriate model class based on the task type
    specified in the configuration.

    Args:
        cfg_like: ELM configuration dictionary or mapping

    Returns:
        EasyDeLBaseModule: The loaded model instance

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b", "task": "causal_lm"},
        ...     "loader": {"dtype": "bf16"}
        ... }
        >>> model = build_model(cfg)
        >>>
    """
    kw = to_from_pretrained_kwargs(cfg_like)
    task = resolve_task(normalize(cfg_like))
    if task == TaskType.CAUSAL_LM:
        return AutoEasyDeLModelForCausalLM.from_pretrained(**kw)
    if task == TaskType.SEQUENCE_TO_SEQUENCE:
        return AutoEasyDeLModelForSeq2SeqLM.from_pretrained(**kw)
    if task == TaskType.SPEECH_SEQUENCE_TO_SEQUENCE:
        return AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(**kw)
    if task == TaskType.IMAGE_TEXT_TO_TEXT:
        return AutoEasyDeLModelForImageTextToText.from_pretrained(**kw)
    if task == TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION:
        return AutoEasyDeLModelForZeroShotImageClassification.from_pretrained(**kw)
    if task == TaskType.DIFFUSION_LM:
        return AutoEasyDeLModelForDiffusionLM.from_pretrained(**kw)
    if task == TaskType.SEQUENCE_CLASSIFICATION:
        return AutoEasyDeLModelForSequenceClassification.from_pretrained(**kw)
    if task == TaskType.ANY_TO_ANY:
        return AutoEasyDeLAnyToAnyModel.from_pretrained(**kw)
    return AutoEasyDeLModel.from_pretrained(**kw)


def to_esurge_kwargs(cfg_like: ELMConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Convert ELM configuration to kwargs for eSurge initialization.

    Extracts eSurge-specific configuration values and infers defaults from
    base configuration when needed.

    Args:
        cfg_like: ELM configuration dictionary or mapping

    Returns:
        Dictionary of keyword arguments for eSurge initialization

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "esurge": {"max_model_len": 4096, "max_num_seqs": 32}
        ... }
        >>> kwargs = to_esurge_kwargs(cfg)
        >>> kwargs["max_model_len"]
        4096
    """
    cfg = normalize(cfg_like)
    es = cfg.get("esurge", {})
    base_vals = dict(cfg.get("base_config", {}).get("values", {}) or {})
    max_model_len = (
        es.get("max_model_len")
        or base_vals.get("mask_max_position_embeddings")
        or base_vals.get("freq_max_position_embeddings")
        or 8192
    )
    min_input_pad_val = es.get("min_input_pad")
    min_token_pad_val = es.get("min_token_pad")
    max_num_seqs_val = es.get("max_num_seqs")
    max_num_seq_buckets_val = es.get("max_num_seq_buckets")
    page_size_val = es.get("page_size")
    hbm_utilization_val = es.get("hbm_utilization")
    use_aot_forward_val = es.get("use_aot_forward")
    enable_prefix_caching_val = es.get("enable_prefix_caching")
    auto_shard_model_val = es.get("auto_shard_model")
    compile_runner_val = es.get("compile_runner")
    overlap_execution_val = es.get("overlap_execution")
    sampler_metrics_val = es.get("sampler_metrics")
    auto_truncate_prompt_val = es.get("auto_truncate_prompt")
    auto_cap_new_tokens_val = es.get("auto_cap_new_tokens")
    strict_context_val = es.get("strict_context")
    prefer_preserve_prompt_val = es.get("prefer_preserve_prompt")
    decode_truncated_prompt_val = es.get("decode_truncated_prompt")
    destroy_pages_on_pause_val = es.get("destroy_pages_on_pause")
    silent_mode_val = es.get("silent_mode")

    sharding_axis_dims_val = es.get("sharding_axis_dims", (1, 1, 1, -1, 1))
    sharding_axis_dims = tuple(sharding_axis_dims_val) if sharding_axis_dims_val is not None else None

    max_num_batched_tokens = es.get("max_num_batched_tokens")
    if max_num_batched_tokens is not None:
        max_num_batched_tokens = int(max_num_batched_tokens)

    reserve_tokens = es.get("reserve_tokens")
    if reserve_tokens is not None:
        reserve_tokens = int(reserve_tokens)

    detokenizer_max_states = es.get("detokenizer_max_states", DEFAULT_DETOKENIZER_MAX_STATES)
    if detokenizer_max_states is not None:
        detokenizer_max_states = int(detokenizer_max_states)

    extra_eos_token_ids = es.get("extra_eos_token_ids")
    if extra_eos_token_ids is not None:
        extra_eos_token_ids = list(extra_eos_token_ids)

    runner_verbose = bool(es.get("runner_verbose", es.get("verbose", False)))
    truncate_mode = es.get("truncate_mode", "left")

    max_num_seq_buckets = None
    if max_num_seq_buckets_val is not None:
        max_num_seq_buckets = [int(v) for v in max_num_seq_buckets_val]

    return dict(
        max_model_len=int(max_model_len),
        min_input_pad=int(min_input_pad_val) if min_input_pad_val is not None else 16,
        min_token_pad=int(min_token_pad_val) if min_token_pad_val is not None else None,
        max_num_seqs=int(max_num_seqs_val) if max_num_seqs_val is not None else 256,
        max_num_seq_buckets=max_num_seq_buckets,
        max_num_batched_tokens=max_num_batched_tokens,
        hbm_utilization=float(hbm_utilization_val) if hbm_utilization_val is not None else 0.85,
        page_size=int(page_size_val) if page_size_val is not None else 128,
        use_aot_forward=True if use_aot_forward_val is None else bool(use_aot_forward_val),
        enable_prefix_caching=True if enable_prefix_caching_val is None else bool(enable_prefix_caching_val),
        auto_shard_model=True if auto_shard_model_val is None else bool(auto_shard_model_val),
        sharding_axis_dims=sharding_axis_dims,
        compile_runner=True if compile_runner_val is None else bool(compile_runner_val),
        runner_verbose=runner_verbose,
        overlap_execution=False if overlap_execution_val is None else bool(overlap_execution_val),
        sampler_metrics=False if sampler_metrics_val is None else bool(sampler_metrics_val),
        esurge_name=es.get("esurge_name"),
        reserve_tokens=reserve_tokens,
        auto_truncate_prompt=True if auto_truncate_prompt_val is None else bool(auto_truncate_prompt_val),
        auto_cap_new_tokens=True if auto_cap_new_tokens_val is None else bool(auto_cap_new_tokens_val),
        strict_context=False if strict_context_val is None else bool(strict_context_val),
        truncate_mode=truncate_mode,
        prefer_preserve_prompt=True if prefer_preserve_prompt_val is None else bool(prefer_preserve_prompt_val),
        decode_truncated_prompt=True if decode_truncated_prompt_val is None else bool(decode_truncated_prompt_val),
        destroy_pages_on_pause=True if destroy_pages_on_pause_val is None else bool(destroy_pages_on_pause_val),
        detokenizer_max_states=detokenizer_max_states,
        tokenizer_endpoint=es.get("tokenizer_endpoint"),
        detokenizer_endpoint=es.get("detokenizer_endpoint"),
        sampling_params_callback=es.get("sampling_params_callback"),
        extra_eos_token_ids=extra_eos_token_ids,
        silent_mode=False if silent_mode_val is None else bool(silent_mode_val),
    )


def build_esurge(cfg_like: ELMConfig | Mapping[str, Any], model: EasyDeLBaseModule | None = None):
    """Build an eSurge inference engine from ELM configuration.

    Creates an eSurge instance with the model, tokenizer, and inference
    configuration specified in the ELM config.

    Args:
        cfg_like: ELM configuration dictionary or mapping

    Returns:
        eSurge: Configured eSurge inference engine

    Raises:
        NotImplementedError: If the task type is not supported by eSurge

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "esurge": {"max_model_len": 4096, "max_num_seqs": 32}
        ... }
        >>> engine = build_esurge(cfg)
        >>>
    """
    from transformers import AutoProcessor, AutoTokenizer

    from easydel.inference import eSurge

    cfg = normalize(cfg_like)
    task = resolve_task(cfg)
    if task not in [
        TaskType.CAUSAL_LM,
        TaskType.IMAGE_TEXT_TO_TEXT,
        TaskType.ANY_TO_ANY,
        getattr(TaskType, "VISION_LM", None),
    ]:
        raise NotImplementedError(f"eSurge supports [CAUSAL_LM, IMAGE_TEXT_TO_TEXT, ANY_TO_ANY, VISION_LM]; got {task}")
    trust_remote_code = bool(cfg["loader"].get("trust_remote_code", False))
    proc_path = (
        cfg["model"].get("processor")
        or cfg["model"].get("tokenizer")
        or cfg["model"].get("name_or_path")
        or cfg["model"]["name_or_path"]
    )
    if model is None:
        model = build_model(cfg)
    processor = None

    is_vlm_task = task in [TaskType.IMAGE_TEXT_TO_TEXT, TaskType.ANY_TO_ANY, getattr(TaskType, "VISION_LM", None)]

    # For VLM tasks, try to load a ProcessorMixin; if it doesn't exist in the
    # installed Transformers version, fall back to the tokenizer as the
    # "processor" (eSurge has a tokenizer-only multimodal fallback for patchified VLMs).
    if is_vlm_task:
        try:
            processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=trust_remote_code)
        except Exception:
            processor = None

    if processor is None:
        processor = AutoTokenizer.from_pretrained(proc_path, trust_remote_code=trust_remote_code)
    return eSurge(
        model=model,
        processor=processor,
        **to_esurge_kwargs(cfg),
    )


def to_data_mixture_kwargs(cfg_like: ELMConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Convert ELM configuration to kwargs for DatasetMixture creation.

    Transforms the mixture configuration section into the format expected
    by the DatasetMixture and DataManager classes. Supports all modern
    features including token packing and block-deterministic mixing.

    Args:
        cfg_like: ELM configuration dictionary or mapping

    Returns:
        Dictionary of keyword arguments for DatasetMixture initialization

    Example:
        >>> cfg = {
        ...     "mixture": {
        ...         "informs": [
        ...             {"type": "json", "data_files": "train.json", "content_field": "text"}
        ...         ],
        ...         "batch_size": 32,
        ...         "block_mixture": True,
        ...         "pack_tokens": True,
        ...         "pack_seq_length": 2048
        ...     }
        ... }
        >>> kwargs = to_data_mixture_kwargs(cfg)
        >>> mixture = DatasetMixture(**kwargs)
    """
    from easydel.data import TextDatasetInform, VisualDatasetInform

    cfg = normalize(cfg_like)
    mixture_cfg = cfg.get("mixture", {})

    if not mixture_cfg:
        return {}

    informs = []
    for inform_cfg in mixture_cfg.get("informs", []):
        data_files = inform_cfg.get("data_files")
        source_type = inform_cfg.get("type")

        if data_files is None:
            if isinstance(source_type, str):
                candidate = source_type.strip()
                known_types = {
                    "json",
                    "jsonl",
                    "parquet",
                    "csv",
                    "arrow",
                    "tsv",
                    "txt",
                    "text",
                    "huggingface",
                    "hf",
                }
                if candidate and candidate.lower() not in known_types:
                    # Backward-compatible shorthand: allow HF dataset id in `type` without `data_files`.
                    data_files = candidate
                    source_type = "hf"
                else:
                    raise ValueError("mixture.informs[].data_files is required")
            else:
                raise ValueError("mixture.informs[].data_files is required")

        if "pixel_field" in inform_cfg:
            inform = VisualDatasetInform(
                type=source_type,
                data_files=data_files,
                dataset_split_name=inform_cfg.get("dataset_split_name", None),
                split=inform_cfg.get("split", "train"),
                pixel_field=inform_cfg.get("pixel_field", "images"),
                content_field=inform_cfg.get("content_field"),
                image_size=tuple(inform_cfg["image_size"]) if inform_cfg.get("image_size") else None,
                num_rows=inform_cfg.get("num_rows"),
                format_callback=inform_cfg.get("format_callback"),
                format_fields=inform_cfg.get("format_fields"),
                preprocessing_fn=inform_cfg.get("preprocessing_fn"),
            )
        else:
            inform = TextDatasetInform(
                type=source_type,
                data_files=data_files,
                dataset_split_name=inform_cfg.get("dataset_split_name", None),
                split=inform_cfg.get("split", "train"),
                content_field=inform_cfg.get("content_field", "content"),
                additional_fields=inform_cfg.get("additional_fields"),
                num_rows=inform_cfg.get("num_rows"),
                format_callback=inform_cfg.get("format_callback"),
                format_fields=inform_cfg.get("format_fields"),
                preprocessing_fn=inform_cfg.get("preprocessing_fn"),
            )
        informs.append(inform)

    kwargs = dict(
        informs=informs,
        cache_dir=mixture_cfg.get("cache_dir", f"{pathlib.Path.home()}/.cache/easydel"),
        streaming=mixture_cfg.get("streaming", True),
        text_target_field=mixture_cfg.get("text_target_field", "text"),
        image_target_field=mixture_cfg.get("image_target_field", "image"),
        batch_size=mixture_cfg.get("batch_size", 1),
        shuffle_buffer_size=mixture_cfg.get("shuffle_buffer_size"),
        seed=mixture_cfg.get("seed", 42),
        prefetch_workers=mixture_cfg.get("prefetch_workers", 2),
        prefetch_buffer_size=mixture_cfg.get("prefetch_buffer_size", 4),
        cloud_max_retries=mixture_cfg.get("cloud_max_retries", 3),
        cloud_retry_delay=mixture_cfg.get("cloud_retry_delay", 0.1),
        cache_remote_files=mixture_cfg.get("cache_remote_files", True),
        cache_expiry_seconds=mixture_cfg.get("cache_expiry_seconds", 86400),
    )

    if "pack_tokens" in mixture_cfg:
        kwargs["pack_tokens"] = mixture_cfg["pack_tokens"]
    if "tokens_field_name" in mixture_cfg:
        kwargs["tokens_field_name"] = mixture_cfg["tokens_field_name"]
    if "pack_seq_length" in mixture_cfg:
        kwargs["pack_seq_length"] = mixture_cfg["pack_seq_length"]
    if "pack_eos_token_id" in mixture_cfg:
        kwargs["pack_eos_token_id"] = mixture_cfg["pack_eos_token_id"]
    if "pack_shuffle" in mixture_cfg:
        kwargs["pack_shuffle"] = mixture_cfg["pack_shuffle"]
    if "pack_shuffle_buffer_factor" in mixture_cfg:
        kwargs["pack_shuffle_buffer_factor"] = mixture_cfg["pack_shuffle_buffer_factor"]
    if "dask_storage_options" in mixture_cfg:
        kwargs["dask_storage_options"] = mixture_cfg["dask_storage_options"]

    if "pack_on_the_fly" in mixture_cfg:
        kwargs["pack_on_the_fly"] = mixture_cfg["pack_on_the_fly"]
    if "tokenize_callback" in mixture_cfg:
        kwargs["tokenize_callback"] = mixture_cfg["tokenize_callback"]

    if "block_mixture" in mixture_cfg:
        kwargs["block_mixture"] = mixture_cfg["block_mixture"]
    if "mixture_block_size" in mixture_cfg:
        kwargs["mixture_block_size"] = mixture_cfg["mixture_block_size"]
    if "stop_strategy" in mixture_cfg:
        kwargs["stop_strategy"] = mixture_cfg["stop_strategy"]
    if "mixture_weights" in mixture_cfg:
        kwargs["mixture_weights"] = mixture_cfg["mixture_weights"]

    return kwargs


def build_dataset(cfg_like: ELMConfig | Mapping[str, Any]) -> Dataset | IterableDataset | None:
    """Build a dataset from ELM configuration with data mixture.

    Creates a unified dataset from the mixture configuration using the
    new DatasetMixture.build() method. Supports all modern features including
    token packing, block-deterministic mixing, and streaming.

    Args:
        cfg_like: ELM configuration dictionary or mapping

    Returns:
        Dataset or IterableDataset: The loaded and processed dataset

    Example:
        >>> cfg = {
        ...     "mixture": {
        ...         "informs": [
        ...             {"type": "json", "data_files": "data.json", "content_field": "text"}
        ...         ],
        ...         "block_mixture": True,
        ...         "pack_tokens": True,
        ...         "pack_seq_length": 2048
        ...     }
        ... }
        >>> dataset = build_dataset(cfg)
    """
    from easydel.data import DatasetMixture

    cfg = normalize(cfg_like)
    mixture_cfg = cfg.get("mixture", {})

    if not mixture_cfg or not mixture_cfg.get("informs"):
        return None

    mixture_kwargs = to_data_mixture_kwargs(cfg)
    mixture = DatasetMixture(**mixture_kwargs)
    return mixture.build()


def tokenize_dataset(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    text_field: str = "text",
    output_field: str = "tokens",
    max_length: int = 2048,
    truncation: bool = True,
    padding: bool | str = False,
    add_special_tokens: bool = True,
    return_attention_mask: bool = True,
    num_proc: int | None = None,
    batched: bool = True,
    batch_size: int = 1000,
    remove_columns: list[str] | None = None,
    keep_in_memory: bool = False,
) -> Dataset | IterableDataset:
    """Tokenize a dataset using the provided tokenizer.

    Args:
        dataset: HuggingFace Dataset or IterableDataset to tokenize
        tokenizer: HuggingFace tokenizer instance
        text_field: Field name containing text to tokenize (default: "text")
        output_field: Field name for tokenized output (default: "tokens")
        max_length: Maximum sequence length (default: 2048)
        truncation: Whether to truncate sequences (default: True)
        padding: Padding strategy (default: False)
        add_special_tokens: Add special tokens like BOS/EOS (default: True)
        return_attention_mask: Return attention masks (default: True)
        num_proc: Number of processes for parallel tokenization (default: None)
        batched: Process examples in batches (default: True)
        batch_size: Batch size for batched processing (default: 1000)
        remove_columns: Columns to remove after tokenization (default: None)
        keep_in_memory: Keep processed dataset in memory (default: False)

    Returns:
        Tokenized dataset with token IDs in the output_field

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        >>> tokenized = tokenize_dataset(dataset, tokenizer, text_field="content")
    """
    from easydel.data.utils import is_streaming

    def tokenize_fn(examples):
        # Handle both batched and single examples
        texts = examples[text_field]
        if isinstance(texts, str):
            texts = [texts]

        outputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
        )

        result = {output_field: outputs["input_ids"]}
        if return_attention_mask:
            result["attention_mask"] = outputs["attention_mask"]

        return result

    # Determine columns to remove
    if remove_columns is None:
        if hasattr(dataset, "column_names"):
            remove_columns = dataset.column_names
        else:
            remove_columns = []

    # Handle streaming vs non-streaming datasets
    if is_streaming(dataset):
        return dataset.map(
            tokenize_fn,
            batched=batched,
            remove_columns=remove_columns,
        )
    else:
        return dataset.map(
            tokenize_fn,
            batched=batched,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
        )


def save_dataset(
    dataset: Dataset | IterableDataset,
    output_path: str,
    format: str = "parquet",  # noqa: A002
    num_shards: int | None = None,
    compression: str | None = "snappy",
    max_shard_size: str | int = "500MB",
    overwrite: bool = False,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    hub_private: bool = False,
    hub_token: str | None = None,
) -> str:
    """Save a dataset to disk or HuggingFace Hub.

    Args:
        dataset: HuggingFace Dataset to save
        output_path: Path to save the dataset
        format: Output format - "parquet", "arrow", "json", "jsonl" (default: "parquet")
        num_shards: Number of shards (default: None, auto-detect)
        compression: Compression algorithm (default: "snappy")
        max_shard_size: Maximum shard size (default: "500MB")
        overwrite: Whether to overwrite existing files (default: False)
        push_to_hub: Push to HuggingFace Hub (default: False)
        hub_repo_id: Hub repository ID (required if push_to_hub=True)
        hub_private: Make Hub repo private (default: False)
        hub_token: HuggingFace token (default: None)

    Returns:
        Path to saved dataset or Hub URL if pushed

    Example:
        >>> save_dataset(tokenized_dataset, "output/tokenized", format="parquet")
        >>> # Or push to hub
        >>> save_dataset(tokenized_dataset, "output/tokenized",
        ...              push_to_hub=True, hub_repo_id="username/my-dataset")
    """
    import os

    from easydel.data.utils import is_streaming

    # Check if output exists
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output path '{output_path}' already exists. Set overwrite=True to replace.")

    # Handle streaming datasets - materialize first
    if is_streaming(dataset):
        from datasets import Dataset

        # Convert iterable dataset to regular dataset
        dataset = Dataset.from_generator(lambda: (ex for ex in dataset))

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save based on format
    if format == "parquet":
        dataset.to_parquet(
            os.path.join(output_path, "data.parquet"),
            compression=compression,
        )
    elif format == "arrow":
        dataset.save_to_disk(output_path, num_shards=num_shards, max_shard_size=max_shard_size)
    elif format in ("json", "jsonl"):
        dataset.to_json(
            os.path.join(output_path, "data.jsonl"),
            lines=(format == "jsonl"),
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet', 'arrow', 'json', or 'jsonl'.")

    # Push to Hub if requested
    if push_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id is required when push_to_hub=True")
        dataset.push_to_hub(
            repo_id=hub_repo_id,
            private=hub_private,
            token=hub_token,
        )
        return f"https://huggingface.co/datasets/{hub_repo_id}"

    return output_path


def build_tokenized_dataset(
    cfg_like: ELMConfig | Mapping[str, Any],
    save: bool = True,
) -> Dataset | IterableDataset | tuple[Dataset | IterableDataset, str]:
    """Build, tokenize, and optionally save a dataset from ELM configuration.

    This is the main entry point for the tokenization pipeline. It:
    1. Loads the dataset from the mixture configuration
    2. Tokenizes using the specified tokenizer
    3. Optionally saves to disk or HuggingFace Hub

    Args:
        cfg_like: ELM configuration dictionary or mapping
        save: Whether to save the tokenized dataset (default: True)

    Returns:
        Tuple of (tokenized_dataset, save_path) if save=True, else tokenized_dataset

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "mixture": {
        ...         "informs": [
        ...             {"type": "json", "data_files": "data.json", "content_field": "text"}
        ...         ],
        ...         "streaming": False,  # Must be False for saving
        ...         "tokenization": {
        ...             "max_length": 2048,
        ...             "text_field": "text",
        ...             "output_field": "tokens",
        ...             "num_proc": 4
        ...         },
        ...         "save": {
        ...             "output_path": "tokenized_data",
        ...             "format": "parquet"
        ...         }
        ...     }
        ... }
        >>> dataset, path = build_tokenized_dataset(cfg)
    """
    from transformers import AutoTokenizer

    cfg = normalize(cfg_like)
    mixture_cfg = cfg.get("mixture", {})

    if not mixture_cfg or not mixture_cfg.get("informs"):
        raise ValueError("mixture.informs is required for tokenization")

    # Get tokenization config
    tok_cfg = mixture_cfg.get("tokenization", {})
    save_cfg = mixture_cfg.get("save", {})

    # Determine tokenizer path
    tokenizer_path = tok_cfg.get("tokenizer")
    if tokenizer_path is None:
        model_cfg = cfg.get("model", {})
        tokenizer_path = model_cfg.get("tokenizer", model_cfg.get("name_or_path"))

    if not tokenizer_path:
        raise ValueError("Tokenizer not specified. Set mixture.tokenization.tokenizer or model.name_or_path")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Build dataset (ensure non-streaming for save)
    if save and mixture_cfg.get("streaming", True):
        # Override streaming for save operation
        cfg_copy = dict(cfg)
        cfg_copy["mixture"] = dict(mixture_cfg)
        cfg_copy["mixture"]["streaming"] = False
        dataset = build_dataset(cfg_copy)
    else:
        dataset = build_dataset(cfg)

    if dataset is None:
        raise ValueError("Failed to build dataset from configuration")

    # Tokenize
    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
        text_field=tok_cfg.get("text_field", mixture_cfg.get("text_target_field", "text")),
        output_field=tok_cfg.get("output_field", "tokens"),
        max_length=tok_cfg.get("max_length", 2048),
        truncation=tok_cfg.get("truncation", True),
        padding=tok_cfg.get("padding", False),
        add_special_tokens=tok_cfg.get("add_special_tokens", True),
        return_attention_mask=tok_cfg.get("return_attention_mask", True),
        num_proc=tok_cfg.get("num_proc"),
        batched=tok_cfg.get("batched", True),
        batch_size=tok_cfg.get("batch_size", 1000),
        remove_columns=tok_cfg.get("remove_columns"),
        keep_in_memory=tok_cfg.get("keep_in_memory", False),
    )

    # Save if requested
    if save and save_cfg:
        output_path = save_cfg.get("output_path")
        if not output_path:
            raise ValueError("mixture.save.output_path is required when save=True")

        saved_path = save_dataset(
            tokenized,
            output_path=output_path,
            format=save_cfg.get("format", "parquet"),
            num_shards=save_cfg.get("num_shards"),
            compression=save_cfg.get("compression", "snappy"),
            max_shard_size=save_cfg.get("max_shard_size", "500MB"),
            overwrite=save_cfg.get("overwrite", False),
            push_to_hub=save_cfg.get("push_to_hub", False),
            hub_repo_id=save_cfg.get("hub_repo_id"),
            hub_private=save_cfg.get("hub_private", False),
            hub_token=save_cfg.get("hub_token"),
        )
        return tokenized, saved_path

    return tokenized


def _extract_dataset_name(inform_cfg: Mapping[str, Any], fallback_index: int = 0) -> str:
    """Extract a meaningful dataset name from inform configuration.

    Uses the following priority:
    1. Explicit 'name' field if provided
    2. HuggingFace repo name from 'data_files' (e.g., "LDJnr/Puffin" -> "Puffin")
    3. GCS/S3/cloud path - extract bucket or meaningful path segment
    4. File/directory name from 'data_files' path
    5. Fallback to "dataset_{index}"

    Args:
        inform_cfg: Dataset inform configuration dictionary.
        fallback_index: Index to use in fallback name.

    Returns:
        Extracted dataset name string.

    Examples:
        - "LDJnr/Puffin" -> "Puffin"
        - "gs://my-bucket/datasets/alpaca/*.parquet" -> "alpaca"
        - "s3://bucket/data/train.json" -> "train"
        - "/local/path/to/data.parquet" -> "data"
        - "hf://datasets/tatsu-lab/alpaca" -> "alpaca"
    """
    import os
    import re

    # Check for explicit name
    if inform_cfg.get("name"):
        return inform_cfg["name"]

    data_files = inform_cfg.get("data_files", "")
    candidate: str | None = None

    if not isinstance(data_files, str):
        # Handle list of files - use first file
        if isinstance(data_files, list) and data_files:
            data_files = data_files[0]
        else:
            return f"dataset_{fallback_index}"

    # Strip whitespace
    data_files = data_files.strip()

    # Handle cloud storage paths (gs://, s3://, az://, hf://)
    cloud_match = re.match(r"^(gs|s3|az|gcs|https?|hf)://(.+)$", data_files, re.IGNORECASE)
    if cloud_match:
        path_part = cloud_match.group(2)
        # Remove bucket name for gs/s3/az, keep meaningful path
        # e.g., "my-bucket/datasets/alpaca/*.parquet" -> extract "alpaca"
        # e.g., "datasets/tatsu-lab/alpaca" -> extract "alpaca"
        parts = [p for p in path_part.split("/") if p and not p.startswith("*")]

        # Walk backwards to find a meaningful name (skip bucket, globs, extensions)
        for part in reversed(parts):
            # Skip if it's just a file extension pattern
            if part.startswith("*."):
                continue
            # Clean glob patterns and extensions
            clean = part.rstrip("*").rstrip("/")
            name, _ = os.path.splitext(clean)
            # Skip if empty or looks like a bucket name (too generic)
            if name and name not in ("data", "train", "test", "val", "dataset", "datasets"):
                return name
            if name:
                # Use it even if generic, but keep looking
                candidate = name

        # Use the last candidate found
        if candidate:
            return candidate

        # Fallback to last non-empty part
        if parts:
            name, _ = os.path.splitext(parts[-1].rstrip("*"))
            if name:
                return name

    if "/" in data_files and not data_files.startswith(("/", ".", "~")):
        parts = data_files.split("/")
        if len(parts) == 2 and not any(
            p.endswith(
                (
                    ".json",
                    ".parquet",
                    ".arrow",
                    ".csv",
                    ".txt",
                    ".jsonl",
                )
            )
            for p in parts
        ):
            return parts[-1]

    clean_path = data_files.rstrip("*").rstrip("/")
    base_name = os.path.basename(clean_path)
    if base_name:
        # Remove extension if present
        name, _ = os.path.splitext(base_name)
        if name:
            return name

    # Fallback
    return f"dataset_{fallback_index}"


def _create_source_from_inform(
    inform_cfg: Mapping[str, Any],
    mixture_cfg: Mapping[str, Any],
) -> "ShardedDataSource":
    """Create a ShardedDataSource from an inform configuration.

    Maps the inform config to the appropriate ShardedDataSource type
    based on the dataset type (JSON, Parquet, Arrow, CSV, HF, etc.).

    Args:
        inform_cfg: Dataset inform configuration dictionary.
        mixture_cfg: Parent mixture configuration dictionary.

    Returns:
        ShardedDataSource: The created data source.

    Raises:
        ValueError: If the dataset type is not supported.
    """
    from easydel.data.sources import (
        ArrowShardedSource,
        CsvShardedSource,
        HuggingFaceShardedSource,
        JsonShardedSource,
        ParquetShardedSource,
        TextShardedSource,
        expand_data_files,
    )

    data_files = inform_cfg.get("data_files")
    source_type_raw = inform_cfg.get("type") or ""

    if data_files is None:
        if isinstance(source_type_raw, str):
            candidate = source_type_raw.strip()
            known_types = {
                "json",
                "jsonl",
                "parquet",
                "csv",
                "arrow",
                "tsv",
                "txt",
                "text",
                "huggingface",
                "hf",
            }
            if candidate and candidate.lower() not in known_types:
                # Backward-compatible shorthand: allow HF dataset id in `type` without `data_files`.
                data_files = candidate
                source_type_raw = ""
            else:
                raise ValueError("mixture.informs[].data_files is required")
        else:
            raise ValueError("mixture.informs[].data_files is required")
    source_type = source_type_raw.lower() if isinstance(source_type_raw, str) else str(source_type_raw).lower()
    split = inform_cfg.get("split", "train")
    dataset_split_name = inform_cfg.get("dataset_split_name")

    # Check if it's a HuggingFace dataset
    if source_type in ("huggingface", "hf"):
        if not isinstance(data_files, str):
            raise TypeError("mixture.informs[].data_files must be a string for HuggingFace datasets")
        streaming = inform_cfg.get("streaming")
        if streaming is None:
            streaming = mixture_cfg.get("streaming", True)
        return HuggingFaceShardedSource(
            dataset_name=data_files,
            split=split,
            subset=dataset_split_name,
            streaming=streaming,
            cache_dir=mixture_cfg.get("cache_dir"),
        )

    # Expand files
    try:
        files = expand_data_files(data_files)
    except FileNotFoundError:
        # Might be a HuggingFace dataset if type wasn't specified
        if isinstance(data_files, str) and not source_type:
            streaming = inform_cfg.get("streaming")
            if streaming is None:
                streaming = mixture_cfg.get("streaming", True)
            return HuggingFaceShardedSource(
                dataset_name=data_files,
                split=split,
                subset=dataset_split_name,
                streaming=streaming,
                cache_dir=mixture_cfg.get("cache_dir"),
            )
        raise

    if not files:
        raise ValueError(f"No files found for pattern: {data_files}")

    # Infer type from first file if not specified
    if not source_type:
        first_file = files[0]
        if first_file.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
            source_type = "json"
        elif first_file.endswith(".parquet"):
            source_type = "parquet"
        elif first_file.endswith(".arrow"):
            source_type = "arrow"
        elif first_file.endswith((".csv", ".tsv")):
            source_type = "csv"
        elif first_file.endswith(".txt"):
            source_type = "txt"

    # Create appropriate source
    if source_type in ("json", "jsonl"):
        return JsonShardedSource(files)
    elif source_type == "parquet":
        return ParquetShardedSource(files)
    elif source_type == "arrow":
        return ArrowShardedSource(files)
    elif source_type in ("csv", "tsv"):
        return CsvShardedSource(files)
    elif source_type in ("txt", "text"):
        return TextShardedSource(files)
    else:
        raise ValueError(f"Unsupported dataset type: {source_type}")


def build_sharded_source(cfg_like: ELMConfig | Mapping[str, Any]) -> "ShardedDataSource | None":
    """Build a ShardedDataSource from ELM configuration.

    Uses the new ShardedDataSource architecture for efficient streaming
    and lazy transforms. Supports mixing, packing, and field transforms.

    This function creates a unified ShardedDataSource from the mixture
    configuration, optionally applying:
    - Field renaming via transforms
    - Dataset mixing via MixedShardedSource
    - Sequence packing via PackedShardedSource

    Args:
        cfg_like: ELM configuration dictionary or mapping containing
            a 'mixture' section with dataset configurations.

    Returns:
        ShardedDataSource if mixture is configured, None otherwise.

    Example:
        >>> cfg = {
        ...     "mixture": {
        ...         "informs": [
        ...             {"type": "json", "data_files": "data.json", "content_field": "text"}
        ...         ],
        ...         "use_sharded_source": True,
        ...         "pack_tokens": True,
        ...         "pack_seq_length": 2048
        ...     }
        ... }
        >>> source = build_sharded_source(cfg)
        >>> for batch in source.open_shard(source.shard_names[0]):
        ...     process(batch)
    """
    from easydel.data.transforms import (
        MapTransform,
        MixedShardedSource,
        RenameFields,
    )
    from easydel.data.transforms.pack import PackedShardedSource

    cfg = normalize(cfg_like)
    mixture_cfg = cfg.get("mixture", {})

    if not mixture_cfg or not mixture_cfg.get("informs"):
        return None

    # Build ShardedDataSource for each inform
    sources: dict[str, "ShardedDataSource"] = {}
    content_target = mixture_cfg.get("text_target_field", "text")

    for i, inform_cfg in enumerate(mixture_cfg.get("informs", [])):
        # Extract meaningful name from config
        name = _extract_dataset_name(inform_cfg, fallback_index=i)
        source = _create_source_from_inform(inform_cfg, mixture_cfg)

        # Apply format_callback if specified (custom transformation function)
        format_callback = inform_cfg.get("format_callback")
        if format_callback is not None:
            source = source.transform(MapTransform(format_callback))

        # Apply field renaming if format_fields is specified
        if inform_cfg.get("format_fields"):
            source = source.transform(RenameFields(inform_cfg["format_fields"]))

        # Rename content_field to target field
        content_field = inform_cfg.get("content_field", "content")
        if content_field != content_target:
            source = source.transform(RenameFields({content_field: content_target}))

        sources[name] = source

    # Mix if multiple sources
    if len(sources) > 1:
        weights = mixture_cfg.get("mixture_weights")

        source = MixedShardedSource(
            sources=sources,
            weights=weights,
            block_size=mixture_cfg.get("mixture_block_size", 2048),
            seed=mixture_cfg.get("seed", 42),
            stop_strategy=mixture_cfg.get("stop_strategy", "restart"),
        )
    else:
        source = next(iter(sources.values()))

    # Apply packing if enabled
    if mixture_cfg.get("pack_tokens"):
        source = PackedShardedSource(
            source=source,
            seq_length=mixture_cfg.get("pack_seq_length", 2048),
            eos_token_id=mixture_cfg.get("pack_eos_token_id", 0),
            pad_token_id=mixture_cfg.get("pack_eos_token_id", 0),
            strategy="greedy",
            input_field=mixture_cfg.get("tokens_field_name", "input_ids"),
            shuffle=mixture_cfg.get("pack_shuffle", True),
            shuffle_buffer_factor=mixture_cfg.get("pack_shuffle_buffer_factor", 16),
            seed=mixture_cfg.get("seed", 42),
        )

    return source
