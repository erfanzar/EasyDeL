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
from collections.abc import Mapping
from typing import Any

from easydel.inference.esurge.esurge_engine import DEFAULT_DETOKENIZER_MAX_STATES
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType
from easydel.modules.auto import (
    AutoEasyDeLModel,
    AutoEasyDeLModelForCausalLM,
    AutoEasyDeLModelForDiffusionLM,
    AutoEasyDeLModelForImageTextToText,
    AutoEasyDeLModelForSeq2SeqLM,
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
        quantization_platform=quant.get("platform"),
        quantization_method=quant.get("method"),
        quantization_block_size=int(quant.get("block_size", 128)),
        quantization_pattern=quant.get("pattern"),
        quantize_tensors=bool(quant.get("quantize_tensors", True)),
        verbose=bool(loader.get("verbose", True)),
        from_torch=loader.get("from_torch"),
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
    max_num_seqs_val = es.get("max_num_seqs")
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

    return dict(
        max_model_len=int(max_model_len),
        min_input_pad=int(min_input_pad_val) if min_input_pad_val is not None else 16,
        max_num_seqs=int(max_num_seqs_val) if max_num_seqs_val is not None else 256,
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
    from transformers import AutoTokenizer

    from easydel.inference import eSurge

    cfg = normalize(cfg_like)
    task = resolve_task(cfg)
    if task not in [TaskType.CAUSAL_LM, TaskType.IMAGE_TEXT_TO_TEXT, getattr(TaskType, "VISION_LM", None)]:
        raise NotImplementedError(f"eSurge supports [CAUSAL_LM, IMAGE_TEXT_TO_TEXT, VISION_LM]; got {task}")
    tok_path = cfg["model"].get("tokenizer", cfg["model"]["name_or_path"])
    if model is None:
        model = build_model(cfg)
    return eSurge(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained(tok_path),
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
    from easydel.utils.data_managers import TextDatasetInform, VisualDatasetInform

    cfg = normalize(cfg_like)
    mixture_cfg = cfg.get("mixture", {})

    if not mixture_cfg:
        return {}

    informs = []
    for inform_cfg in mixture_cfg.get("informs", []):
        if "pixel_field" in inform_cfg:
            inform = VisualDatasetInform(
                type=inform_cfg.get("type"),
                data_files=inform_cfg["data_files"],
                dataset_split_name=inform_cfg.get("dataset_split_name", None),
                split=inform_cfg.get("split", "train"),
                pixel_field=inform_cfg.get("pixel_field", "images"),
                content_field=inform_cfg.get("content_field"),
                image_size=tuple(inform_cfg["image_size"]) if inform_cfg.get("image_size") else None,
                num_rows=inform_cfg.get("num_rows"),
                format_callback=inform_cfg.get("format_callback"),
                format_fields=inform_cfg.get("format_fields"),
            )
        else:
            inform = TextDatasetInform(
                type=inform_cfg.get("type"),
                data_files=inform_cfg["data_files"],
                dataset_split_name=inform_cfg.get("dataset_split_name", None),
                split=inform_cfg.get("split", "train"),
                content_field=inform_cfg.get("content_field", "content"),
                additional_fields=inform_cfg.get("additional_fields"),
                num_rows=inform_cfg.get("num_rows"),
                format_callback=inform_cfg.get("format_callback"),
                format_fields=inform_cfg.get("format_fields"),
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


def build_dataset(cfg_like: ELMConfig | Mapping[str, Any]):
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
    from easydel.utils.data_managers import DatasetMixture

    cfg = normalize(cfg_like)
    mixture_cfg = cfg.get("mixture", {})

    if not mixture_cfg or not mixture_cfg.get("informs"):
        return None

    mixture_kwargs = to_data_mixture_kwargs(cfg)
    mixture = DatasetMixture(**mixture_kwargs)
    return mixture.build()
