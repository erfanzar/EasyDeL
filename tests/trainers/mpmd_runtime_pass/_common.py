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

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
from spectrax.runtime import (
    DualPipeV,
    Eager1F1B,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    Schedule,
    Std1F1B,
    ZeroBubbleH1,
)

import easydel as ed

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from tests.trainers import _common as _runtime_common
from tests.trainers._common import (
    MAX_COMPLETION_LENGTH,
    MAX_PROMPT_LENGTH,
    MAX_TOTAL_LENGTH,
    MAX_TRAINING_STEP,
    MODEL_REPO,
    PREFERENCE_DATASET,
    PREFERENCE_SPLIT,
    SAVE_ROOT,
    build_lm_dataset,
    build_sft_text_dataset,
    get_logger,
    get_tokenizer,
    load_preference_dataset,
    run_trainer,
)

SUPPORTED_MPMD_SCHEDULERS = (
    "gpipe",
    "std1f1b",
    "eager1f1b",
    "zerobubbleh1",
    "interleaved_gpipe",
    "interleaved_h1",
    "interleaved_1f1b_plus_one",
    "kimi_k2",
    "dualpipe_v",
)


def make_mpmd_scheduler() -> Schedule:
    """Build the schedule used by the MPMD runtime pass suite.

    Defaults are intentionally conservative for broad trainer smoke coverage.
    Override with:

    * ``EASYDEL_MPMD_RUNTIME_SCHEDULER``: one of ``SUPPORTED_MPMD_SCHEDULERS``.
    * ``EASYDEL_MPMD_MICROBATCHES``: schedule microbatch count.
    * ``EASYDEL_MPMD_VIRTUAL_STAGES``: virtual stages for interleaved/Kimi.
    * ``EASYDEL_MPMD_STAGE_LAYOUT``: contiguous, interleaved, or loop.
    * ``EASYDEL_MPMD_EXTRA_WARMUP``: KimiK2 extra warmup.
    """

    name = _normalize_scheduler_name(_env_text("EASYDEL_MPMD_RUNTIME_SCHEDULER", "gpipe"))
    microbatches = _env_int("EASYDEL_MPMD_MICROBATCHES", 2)
    lazy_bwd_batching = _env_flag("EASYDEL_MPMD_LAZY_BWD_BATCHING", False)
    virtual_stages = _env_int("EASYDEL_MPMD_VIRTUAL_STAGES", 2)
    stage_layout = _env_text("EASYDEL_MPMD_STAGE_LAYOUT", "loop")
    extra_warmup = _env_int("EASYDEL_MPMD_EXTRA_WARMUP", 1)
    microbatches = max(microbatches, _minimum_microbatches_for_scheduler(name, virtual_stages))

    flat_kwargs = {"microbatches": microbatches, "lazy_bwd_batching": lazy_bwd_batching}
    virtual_kwargs = {
        **flat_kwargs,
        "virtual_stages": virtual_stages,
        "stage_layout": stage_layout,
    }
    factories: dict[str, Callable[[], Schedule]] = {
        "gpipe": lambda: GPipe(**flat_kwargs),
        "std1f1b": lambda: Std1F1B(**flat_kwargs),
        "eager1f1b": lambda: Eager1F1B(**flat_kwargs),
        "zerobubbleh1": lambda: ZeroBubbleH1(**flat_kwargs),
        "interleaved_gpipe": lambda: InterleavedGPipe(**virtual_kwargs),
        "interleaved_h1": lambda: InterleavedH1(**virtual_kwargs),
        "interleaved_1f1b_plus_one": lambda: Interleaved1F1BPlusOne(**virtual_kwargs),
        "kimi_k2": lambda: KimiK2(**virtual_kwargs, extra_warmup=extra_warmup),
        "dualpipe_v": lambda: DualPipeV(**flat_kwargs),
    }
    try:
        return factories[name]()
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_MPMD_SCHEDULERS)
        raise ValueError(f"Unsupported EASYDEL_MPMD_RUNTIME_SCHEDULER={name!r}. Expected one of: {supported}") from exc


def make_config(
    config_cls,
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> Any:
    mpmd_overrides = dict(overrides or {})
    scheduler = make_mpmd_scheduler()
    mpmd_overrides["mpmd_scheduler"] = scheduler
    mpmd_overrides["use_wandb"] = False
    mpmd_overrides["do_last_save"] = False
    mpmd_overrides["save_optimizer_state"] = False
    mpmd_overrides.setdefault("esurge_max_num_batched_tokens", 256)
    mpmd_overrides.setdefault("esurge_max_num_seqs", 2)
    mpmd_overrides.setdefault("esurge_max_num_seq_buckets", [1, 2])
    mpmd_overrides.setdefault("esurge_min_input_pad", 1)
    max_training_steps = os.environ.get("EASYDEL_MPMD_MAX_TRAINING_STEPS")
    if max_training_steps:
        mpmd_overrides["max_training_steps"] = _env_int("EASYDEL_MPMD_MAX_TRAINING_STEPS", 1)
    _normalize_batching_for_microbatches(mpmd_overrides, scheduler.microbatches)

    scheduler_name = type(scheduler).__name__.lower()
    return _runtime_common.make_config(config_cls, f"mpmd-{scheduler_name}-{name}", overrides=mpmd_overrides)


def mpmd_generation_length_overrides(
    *,
    prompt: int = 64,
    completion: int = 32,
    max_batched_tokens: int | None = None,
) -> dict[str, Any]:
    """Small generation defaults for MPMD trainer runtime-pass tests."""

    prompt = _env_int("EASYDEL_MPMD_MAX_PROMPT_LENGTH", prompt)
    completion = _env_int("EASYDEL_MPMD_MAX_COMPLETION_LENGTH", completion)
    max_length = _env_int("EASYDEL_MPMD_MAX_LENGTH", prompt + completion)
    if max_length < prompt + completion:
        max_length = prompt + completion
    if max_batched_tokens is None:
        max_batched_tokens = max(16, min(128, max_length))
    return {
        "max_prompt_length": prompt,
        "max_completion_length": completion,
        "max_length": max_length,
        "esurge_max_num_batched_tokens": _env_int("EASYDEL_MPMD_ESURGE_MAX_BATCHED_TOKENS", max_batched_tokens),
    }


def build_reward_dataset(split: str = PREFERENCE_SPLIT):
    """Reward preprocessing can extract prompts from chosen/rejected chats."""
    return load_preference_dataset(split)


def load_causal_lm_model(model_repo: str | None = None) -> ed.AutoEasyDeLModelForCausalLM:
    repo = model_repo or MODEL_REPO
    tokenizer = get_tokenizer(repo)
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(repo, **_load_mpmd_model_kwargs())
    model.config.pad_token_id = tokenizer.pad_token_id
    return model  # pyright: ignore[reportReturnType]


def load_embedding_model(model_repo: str | None = None) -> ed.AutoEasyDeLModelForEmbedding:
    repo = model_repo or MODEL_REPO
    tokenizer = get_tokenizer(repo)
    model = ed.AutoEasyDeLModelForEmbedding.from_pretrained(repo, **_load_mpmd_model_kwargs())
    model.config.pad_token_id = tokenizer.pad_token_id
    return model  # pyright: ignore[reportReturnType]


def load_sequence_classifier_model(
    model_repo: str | None = None,
) -> ed.AutoEasyDeLModelForSequenceClassification:
    repo = model_repo or MODEL_REPO
    tokenizer = get_tokenizer(repo)
    model = ed.AutoEasyDeLModelForSequenceClassification.from_pretrained(
        repo,
        num_labels=1,
        **_load_mpmd_model_kwargs(),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model  # pyright: ignore[reportReturnType]


def _load_mpmd_model_kwargs() -> dict[str, Any]:
    kwargs = _runtime_common._load_model_kwargs()
    kwargs["sharding_axis_dims"] = _mpmd_sharding_axis_dims()
    kwargs["auto_shard_model"] = True
    return kwargs


def _mpmd_sharding_axis_dims() -> tuple[int, int, int, int, int, int]:
    pp = _mpmd_physical_stages()
    return (pp, 1, 1, 1, -1, 1)


def _mpmd_physical_stages() -> int:
    device_count = jax.device_count()
    requested_pp = _env_int("EASYDEL_MPMD_PP", min(2, max(device_count, 1)))
    allow_single = _env_flag("EASYDEL_ALLOW_SINGLE_DEVICE_MPMD", False)
    if device_count < 2 and requested_pp > 1 and not allow_single:
        raise RuntimeError(
            "MPMD runtime pass requires at least 2 JAX devices. Set EASYDEL_ALLOW_SINGLE_DEVICE_MPMD=1 "
            "only when intentionally smoke-testing the wiring on a single device."
        )
    pp = 1 if device_count < 2 else min(requested_pp, device_count)
    return pp


def _minimum_microbatches_for_scheduler(name: str, virtual_stages: int) -> int:
    pp = _mpmd_physical_stages()
    if name == "dualpipe_v":
        return 2 * pp
    if name in {"interleaved_h1", "interleaved_1f1b_plus_one", "kimi_k2"}:
        return pp * virtual_stages
    if name in {"std1f1b", "eager1f1b", "zerobubbleh1"}:
        return pp
    return 1


def _normalize_batching_for_microbatches(overrides: dict[str, Any], microbatches: int) -> None:
    if "gradient_accumulation_steps" not in overrides:
        overrides["gradient_accumulation_steps"] = 1
    total_batch_size = int(overrides.get("total_batch_size", 4))
    if total_batch_size < microbatches:
        total_batch_size = microbatches
    remainder = total_batch_size % microbatches
    if remainder:
        total_batch_size += microbatches - remainder
    overrides["total_batch_size"] = total_batch_size


def _env_text(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _normalize_scheduler_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    aliases = {
        "1f1b": "std1f1b",
        "std_1f1b": "std1f1b",
        "eager_1f1b": "eager1f1b",
        "zero_bubble_h1": "zerobubbleh1",
        "interleavedgpipe": "interleaved_gpipe",
        "interleavedh1": "interleaved_h1",
        "interleaved1f1bplusone": "interleaved_1f1b_plus_one",
        "interleaved_1f1b_plusone": "interleaved_1f1b_plus_one",
        "kimik2": "kimi_k2",
        "dualpipev": "dualpipe_v",
    }
    return aliases.get(normalized, normalized)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


__all__ = (
    "MAX_COMPLETION_LENGTH",
    "MAX_PROMPT_LENGTH",
    "MAX_TOTAL_LENGTH",
    "MAX_TRAINING_STEP",
    "MODEL_REPO",
    "PREFERENCE_DATASET",
    "PREFERENCE_SPLIT",
    "SAVE_ROOT",
    "SUPPORTED_MPMD_SCHEDULERS",
    "build_lm_dataset",
    "build_reward_dataset",
    "build_sft_text_dataset",
    "get_logger",
    "get_tokenizer",
    "load_causal_lm_model",
    "load_embedding_model",
    "load_preference_dataset",
    "load_sequence_classifier_model",
    "make_config",
    "make_mpmd_scheduler",
    "mpmd_generation_length_overrides",
    "run_trainer",
)
