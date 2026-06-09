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
"""RLOO trainer config aliases backed by GRPO."""

from __future__ import annotations

from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from .rloo_config import RLOOConfig


@Registry.register("trainer", "rloo")
class RLOOTrainer(GRPOTrainer):
    """GRPO trainer alias configured for leave-one-out advantages.

    The implementation is inherited from :class:`GRPOTrainer`; this class keeps
    a dedicated registry key and typed config so RLOO runs are configured and
    logged separately from standard GRPO.
    """

    arguments: RLOOConfig
