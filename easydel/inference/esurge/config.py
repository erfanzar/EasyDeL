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

"""Configuration classes for the eSurge inference engine.

Each class is a ``TypedDict`` decorated with :func:`easydel.typings.typed_config`.
That keeps a single source of truth for the schema:

- Type checkers see a ``TypedDict`` (so ``Unpack[eSurgeRuntimeConfig]`` works in ``**kwargs``).
- At runtime, ``Cls.from_dict(**kwargs)`` returns a ``ConfigDict`` instance — a
  ``dict`` subclass with attribute access, ``to_dict()``, and ``replace(**)``.

Example:
    >>> from easydel.inference.esurge.config import eSurgeRuntimeConfig
    >>> runtime = eSurgeRuntimeConfig.from_dict(max_num_seqs=16, max_model_len=8192)
    >>> runtime.max_num_seqs       # attribute access
    16
    >>> runtime["max_num_seqs"]    # dict access
    16
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeAlias, TypedDict, Unpack, cast

from spectrax.common_types import NOT_GIVEN, _Empty

from easydel.operations.kernels._gdn_policy import KernelTilePolicy, normalize_kernel_tile_policy
from easydel.typings import typed_config

if TYPE_CHECKING:
    from easydel.infra.etils import MpMdSchedulers
else:
    MpMdSchedulers = object

LONG_PREFILL_TRS: int = 2048

PipelineInferenceMode: TypeAlias = Literal["auto", "on", "off"]
PIPELINE_INFERENCE_MODES: frozenset[str] = frozenset(("auto", "on", "off"))


def normalize_pipeline_inference_mode(mode: str | None) -> PipelineInferenceMode:
    normalized = "auto" if mode is None else str(mode).lower()
    if normalized not in PIPELINE_INFERENCE_MODES:
        raise ValueError(f"pipeline_inference must be one of 'auto', 'on', or 'off'; got {mode!r}.")
    return cast(PipelineInferenceMode, normalized)


def _validate_esurge_runtime_config(self):
    if self.max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")
    if self.min_input_pad <= 0:
        raise ValueError(f"min_input_pad must be positive, got {self.min_input_pad}")
    if self.min_token_pad is not None and self.min_token_pad <= 0:
        raise ValueError(f"min_token_pad must be positive when specified, got {self.min_token_pad}")
    if self.max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")
    if self.max_num_batched_tokens is not NOT_GIVEN and self.max_num_batched_tokens is not None:
        if self.max_num_batched_tokens <= 0:
            raise ValueError(f"max_num_batched_tokens must be positive, got {self.max_num_batched_tokens}")
    if self.long_prefill_token_threshold is not None and self.long_prefill_token_threshold < 0:
        raise ValueError(f"long_prefill_token_threshold must be non-negative, got {self.long_prefill_token_threshold}")
    self.pipeline_inference = normalize_pipeline_inference_mode(self.pipeline_inference)
    self.kernel_tile_policy = normalize_kernel_tile_policy(self.kernel_tile_policy)


@typed_config(
    defaults={
        "max_model_len": 8192,
        "esurge_name": None,
        "pipeline_inference": "auto",
        "kernel_tile_policy": "auto",
        "min_input_pad": 16,
        "min_token_pad": None,
        "max_num_seqs": 256,
        "max_num_seq_buckets": None,
        "async_scheduling": True,
        "max_num_batched_tokens": NOT_GIVEN,
        "use_aot_forward": True,
        "bind_graphstate_for_aot": False,
        "compile_runner": True,
        "runner_verbose": False,
        "overlap_execution": True,
        "sampler_metrics": False,
        "long_prefill_token_threshold": None,
        "enable_window_aware_runtime_cap": False,
        "mpmd_scheduler": None,
    },
    post_init=_validate_esurge_runtime_config,
)
class eSurgeRuntimeConfig(TypedDict, total=False):
    esurge_name: NotRequired[str | None]
    pipeline_inference: NotRequired[PipelineInferenceMode]
    kernel_tile_policy: NotRequired[KernelTilePolicy]
    max_model_len: NotRequired[int]
    min_input_pad: NotRequired[int]
    min_token_pad: NotRequired[int | None]
    max_num_seqs: NotRequired[int]
    max_num_seq_buckets: NotRequired[list[int] | tuple[int, ...] | None]
    async_scheduling: NotRequired[bool]
    max_num_batched_tokens: NotRequired[int | None | _Empty]
    use_aot_forward: NotRequired[bool]
    bind_graphstate_for_aot: NotRequired[bool]
    compile_runner: NotRequired[bool]
    runner_verbose: NotRequired[bool]
    overlap_execution: NotRequired[bool]
    sampler_metrics: NotRequired[bool]
    long_prefill_token_threshold: NotRequired[int | None]
    enable_window_aware_runtime_cap: NotRequired[bool]
    mpmd_scheduler: NotRequired[MpMdSchedulers | None]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeRuntimeConfig"],
    ) -> "eSurgeRuntimeConfig": ...


def _validate_esurge_cache_runtime_config(self):
    if not (0.0 < float(self.hbm_utilization) <= 1.0):
        raise ValueError(f"hbm_utilization must be in (0, 1], got {self.hbm_utilization}")
    if self.page_size <= 0:
        raise ValueError(f"page_size must be positive, got {self.page_size}")
    if self.max_cache_tokens is not None and self.max_cache_tokens <= 0:
        raise ValueError(f"max_cache_tokens must be positive when specified, got {self.max_cache_tokens}")
    if not (0.0 < float(self.cache_capacity_margin) <= 1.0):
        raise ValueError(f"cache_capacity_margin must be in (0, 1], got {self.cache_capacity_margin}")


@typed_config(
    defaults={
        "hbm_utilization": 0.85,
        "page_size": 128,
        "enable_prefix_caching": True,
        "max_cache_tokens": None,
        "cache_capacity_margin": 0.92,
        "data_parallelism_axis": "dp",
        "destroy_pages_on_pause": True,
    },
    post_init=_validate_esurge_cache_runtime_config,
)
class eSurgeCacheRuntimeConfig(TypedDict, total=False):
    hbm_utilization: NotRequired[float]
    page_size: NotRequired[int]
    enable_prefix_caching: NotRequired[bool]
    max_cache_tokens: NotRequired[int | None]
    cache_capacity_margin: NotRequired[float]
    data_parallelism_axis: NotRequired[str]
    destroy_pages_on_pause: NotRequired[bool]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeCacheRuntimeConfig"],
    ) -> "eSurgeCacheRuntimeConfig": ...


@typed_config(
    defaults={
        "reserve_tokens": None,
        "auto_truncate_prompt": True,
        "auto_cap_new_tokens": True,
        "strict_context": False,
        "truncate_mode": "left",
        "prefer_preserve_prompt": True,
        "decode_truncated_prompt": True,
    },
)
class eSurgeContextConfig(TypedDict, total=False):
    reserve_tokens: NotRequired[int | None]
    auto_truncate_prompt: NotRequired[bool]
    auto_cap_new_tokens: NotRequired[bool]
    strict_context: NotRequired[bool]
    truncate_mode: NotRequired[Literal["left", "right", "middle"]]
    prefer_preserve_prompt: NotRequired[bool]
    decode_truncated_prompt: NotRequired[bool]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeContextConfig"],
    ) -> "eSurgeContextConfig": ...


@typed_config(
    defaults={
        "detokenizer_max_states": 1 << 16,
        "tokenizer_endpoint": None,
        "detokenizer_endpoint": None,
        "worker_startup_timeout": None,
        "max_request_outputs": 1000,
        "idle_reset_seconds": None,
        "idle_reset_min_interval": 60.0,
    },
)
class eSurgeWorkerConfig(TypedDict, total=False):
    detokenizer_max_states: NotRequired[int]
    tokenizer_endpoint: NotRequired[str | None]
    detokenizer_endpoint: NotRequired[str | None]
    worker_startup_timeout: NotRequired[float | None]
    max_request_outputs: NotRequired[int | None]
    idle_reset_seconds: NotRequired[float | None]
    idle_reset_min_interval: NotRequired[float]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeWorkerConfig"],
    ) -> "eSurgeWorkerConfig": ...


@typed_config(
    defaults={
        "sampling_params_callback": None,
        "extra_eos_token_ids": None,
        "extra_stops": None,
        "ignore_stop_strings_in_reasoning": True,
        "silent_mode": False,
        "tool_parser": None,
        "reasoning_parser": None,
    },
)
class eSurgeParsingConfig(TypedDict, total=False):
    sampling_params_callback: NotRequired[Any]
    extra_eos_token_ids: NotRequired[list[int] | None]
    extra_stops: NotRequired[str | list[str] | None]
    ignore_stop_strings_in_reasoning: NotRequired[bool]
    silent_mode: NotRequired[bool]
    tool_parser: NotRequired[Any | None]
    reasoning_parser: NotRequired[Any | None]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeParsingConfig"],
    ) -> "eSurgeParsingConfig": ...


@typed_config(
    defaults={
        "resolution_buckets": None,
        "vision_cache_capacity_mb": 1024,
    },
)
class eSurgeVisionConfig(TypedDict, total=False):
    resolution_buckets: NotRequired[list[tuple[int, int]] | None]
    vision_cache_capacity_mb: NotRequired[int]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeVisionConfig"],
    ) -> "eSurgeVisionConfig": ...


@typed_config(
    defaults={
        "distributed_mode": False,
        "distributed_role": "auto",
        "distributed_service_name": None,
        "distributed_world_size": None,
        "distributed_rank": None,
        "distributed_control_port": 19666,
        "distributed_control_bind_host": "0.0.0.0",
        "distributed_advertise_addr": None,
        "distributed_auth_token": None,
        "distributed_step_timeout_s": 30.0,
        "distributed_connect_timeout_s": 15.0,
        "distributed_verify_sampling_digest": True,
    },
)
class eSurgeDistributedConfig(TypedDict, total=False):
    distributed_mode: NotRequired[bool]
    distributed_role: NotRequired[Literal["auto", "leader", "worker"]]
    distributed_service_name: NotRequired[str | None]
    distributed_world_size: NotRequired[int | None]
    distributed_rank: NotRequired[int | None]
    distributed_control_port: NotRequired[int]
    distributed_control_bind_host: NotRequired[str]
    distributed_advertise_addr: NotRequired[str | None]
    distributed_auth_token: NotRequired[str | None]
    distributed_step_timeout_s: NotRequired[float]
    distributed_connect_timeout_s: NotRequired[float]
    distributed_verify_sampling_digest: NotRequired[bool]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeDistributedConfig"],
    ) -> "eSurgeDistributedConfig": ...
