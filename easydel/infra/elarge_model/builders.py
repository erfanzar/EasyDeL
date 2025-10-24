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

    config_kwargs.pop("shard_attention_computation", None)
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
        shard_attention_computation=bool(sharding.get("shard_attention_computation", True)),
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
    return dict(
        max_model_len=int(max_model_len),
        min_input_pad=int(es.get("min_input_pad", 16)),
        max_num_seqs=int(es.get("max_num_seqs", 16)),
        hbm_utilization=float(es.get("hbm_utilization", 0.85)),
        page_size=int(es.get("page_size", 128)),
        enable_prefix_caching=bool(es.get("enable_prefix_caching", True)),
        use_aot_forward=bool(es.get("use_aot_forward", True)),
        runner_verbose=bool(es.get("verbose", False)),
    )


def to_vsurge_kwargs(cfg_like: ELMConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Convert ELM configuration to kwargs for vSurge initialization.

    Extracts vSurge-specific configuration values and infers defaults from
    base configuration when needed.

    Args:
        cfg_like: ELM configuration dictionary or mapping

    Returns:
        Dictionary of keyword arguments for vSurge initialization

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "vsurge": {"max_concurrent_decodes": 8, "bytecode_decode": True}
        ... }
        >>> kwargs = to_vsurge_kwargs(cfg)
        >>> kwargs["bytecode_decode"]
        True
    """
    cfg = normalize(cfg_like)
    vs = cfg.get("vsurge", {})
    base_vals = dict(cfg.get("base_config", {}).get("values", {}) or {})

    max_length = (
        vs.get("max_length")
        or base_vals.get("mask_max_position_embeddings")
        or base_vals.get("freq_max_position_embeddings")
        or 8192
    )

    return dict(
        max_concurrent_decodes=vs.get("max_concurrent_decodes"),
        max_concurrent_prefill=int(vs.get("max_concurrent_prefill", 1)),
        prefill_lengths=vs.get("prefill_lengths"),
        max_prefill_length=int(vs.get("max_prefill_length", max_length // 2)),
        max_length=int(max_length),
        interleaved_mode=bool(vs.get("interleaved_mode", False)),
        slot_clear_steps=int(vs.get("slot_clear_steps", 0)),
        bytecode_decode=bool(vs.get("bytecode_decode", False)),
        verbose=bool(vs.get("verbose", True)),
        seed=int(vs.get("seed", 894)),
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


def build_vsurge(cfg_like: ELMConfig | Mapping[str, Any], model: EasyDeLBaseModule | None = None):
    """Build a vSurge inference engine from ELM configuration.

    Creates a vSurge instance with the model, processor, and inference
    configuration specified in the ELM config.

    Args:
        cfg_like: ELM configuration dictionary or mapping

    Returns:
        vSurge: Configured vSurge inference engine

    Raises:
        NotImplementedError: If the task type is not supported by vSurge

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "vsurge": {"max_concurrent_decodes": 8, "bytecode_decode": True}
        ... }
        >>> engine = build_vsurge(cfg)
        >>>
    """
    from transformers import AutoTokenizer

    from easydel.inference import vSurge

    cfg = normalize(cfg_like)
    task = resolve_task(cfg)
    if task not in [TaskType.CAUSAL_LM, TaskType.IMAGE_TEXT_TO_TEXT, getattr(TaskType, "VISION_LM", None)]:
        raise NotImplementedError(f"vSurge supports [CAUSAL_LM, IMAGE_TEXT_TO_TEXT, VISION_LM]; got {task}")

    tok_path = cfg["model"].get("tokenizer", cfg["model"]["name_or_path"])
    if model is None:
        model = build_model(cfg)
    return vSurge.from_model(
        model=model,
        processor=AutoTokenizer.from_pretrained(tok_path),
        **to_vsurge_kwargs(cfg),
    )
