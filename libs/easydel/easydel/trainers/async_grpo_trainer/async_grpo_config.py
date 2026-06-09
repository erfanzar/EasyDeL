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
"""Async GRPO configuration for local eSurge-backed rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "async_grpo")
@dataclass
class AsyncGRPOConfig(GRPOConfig):
    """Configuration for AsyncGRPO with local eSurge async execution.

    AsyncGRPO keeps GRPO's compiled loss/update path but requires rollout
    generation to use EasyDeL/eSurge with async scheduler token handling and
    overlap execution enabled. This preserves the no-server EasyDeL contract:
    models are initialized EasyDeL modules/states, not string IDs, and no
    external inference-server compatibility surface is exposed.
    """

    trainer_prefix: str | None = field(default="AsyncGRPO")
    learning_rate: float = field(default=1e-6)
    request_timeout: float = field(default=120.0)
    max_inflight_tasks: int = field(default=32)
    max_staleness: int = field(default=1)
    queue_maxsize: int = field(default=0)
    weight_sync_steps: int = field(default=1)
    heartbeat_stale_after_s: float = field(default=120.0)
    log_completions: bool = field(default=False)
    num_completions_to_print: int | None = field(default=None)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Normalize GRPO settings and validate local async execution knobs.

        AsyncGRPO is only backed by local eSurge in EasyDeL. The inherited
        eSurge runtime flags are therefore forced on here instead of relying on
        engine defaults, so runtime logs and tests can verify that async
        scheduler execution is active.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        self.use_esurge_generation = True
        self.esurge_async_scheduling = True
        self.esurge_overlap_execution = True
        if self.max_inflight_tasks <= 0:
            raise ValueError("`max_inflight_tasks` must be positive.")
        if self.max_staleness < 0:
            raise ValueError("`max_staleness` must be non-negative.")
        if self.queue_maxsize < 0:
            raise ValueError("`queue_maxsize` must be non-negative.")
        if self.weight_sync_steps <= 0:
            raise ValueError("`weight_sync_steps` must be positive.")
        if self.request_timeout <= 0:
            raise ValueError("`request_timeout` must be positive.")
        if self.heartbeat_stale_after_s <= 0:
            raise ValueError("`heartbeat_stale_after_s` must be positive.")

    __hash__ = hash_fn
