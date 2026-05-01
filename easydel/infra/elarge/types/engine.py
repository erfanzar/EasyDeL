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

"""BaseCfg and eSurgeCfg TypedDicts."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from easydel.inference.esurge.config import (
    eSurgeCacheRuntimeConfig,
    eSurgeContextConfig,
    eSurgeDistributedConfig,
    eSurgeParsingConfig,
    eSurgeRuntimeConfig,
    eSurgeVisionConfig,
    eSurgeWorkerConfig,
)
from easydel.infra.base_config import EasyDeLBaseConfigDict

from .model import OperationConfigsDict


class BaseCfg(TypedDict, total=False):
    """Container for base model configuration values and ejkernel operation overrides.

    Attributes:
        values: Base model configuration dictionary. Accepts an
            ``EasyDeLBaseConfigDict`` or a plain ``dict`` of key-value
            overrides applied during model initialization.
        operation_configs: Per-operation kernel configuration overrides.
            Maps operation names to their custom settings. ``None`` keeps
            the default kernel configurations.
    """

    values: NotRequired[EasyDeLBaseConfigDict | dict[str, Any]]
    operation_configs: NotRequired[OperationConfigsDict | None]


class eSurgeCfg(TypedDict, total=False):
    """Configuration for the eSurge inference engine, organized by sub-system.

    Each field is a dict matching the corresponding eSurge sub-config schema —
    the keys you author here mirror the engine's actual sectioned API one-to-one.
    Spread the result of :func:`to_esurge_kwargs` directly into
    :class:`~easydel.inference.esurge.eSurge`::

        eSurge(model=..., **to_esurge_kwargs(cfg))

    Attributes:
        runtime: Runtime / scheduler / runner settings (max_model_len,
            max_num_seqs, async_scheduling, compile_runner, AOT options, ...).
        cache: KV-cache / paging settings (page_size, hbm_utilization,
            enable_prefix_caching, cache_capacity_margin, ...).
        context: Prompt / context-window handling (auto_truncate_prompt,
            truncate_mode, reserve_tokens, strict_context, ...).
        workers: Tokenizer / detokenizer worker settings
            (detokenizer_max_states, worker_startup_timeout, idle reset, ...).
        parsing: Generation parsing & stop-handling (extra_eos_token_ids,
            extra_stops, tool_parser, reasoning_parser, silent_mode).
        vision: Multimodal / vision settings (resolution_buckets,
            vision_cache_capacity_mb).
        distributed: Multi-host serving (distributed_mode, distributed_role,
            world_size, rank, control plane settings).
    """

    runtime: NotRequired[eSurgeRuntimeConfig]
    cache: NotRequired[eSurgeCacheRuntimeConfig]
    context: NotRequired[eSurgeContextConfig]
    workers: NotRequired[eSurgeWorkerConfig]
    parsing: NotRequired[eSurgeParsingConfig]
    vision: NotRequired[eSurgeVisionConfig]
    distributed: NotRequired[eSurgeDistributedConfig]
