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

import typing as tp
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
    async_mode: bool = field(default=True)
    environment: str | dict[str, tp.Any] | None = field(
        default=None,
        metadata={
            "help": "Optional registered environment name, 'auto', or config mapping resolved via AutoEnvironment."
        },
    )
    environment_backend: str = field(
        default="kiri",
        metadata={"help": "Sandbox backend for environment rollouts: 'kiri', 'auto', 'docker', or 'local'."},
    )
    environment_endpoints: list[str] | str | None = field(
        default=None,
        metadata={
            "help": "Optional Kiri/Docker endpoints, or 'shm:N' / 'pod-shm:N' to auto-boot RAM-backed daemons."
        },
    )

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Normalize GRPO settings and validate local async execution knobs."""
        self.async_mode = True
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if self.environment_backend not in {"local", "auto", "kiri", "docker"}:
            raise ValueError(
                "`environment_backend` must be one of 'local', 'auto', 'kiri', or 'docker'; "
                f"got {self.environment_backend!r}."
            )

    __hash__ = hash_fn
