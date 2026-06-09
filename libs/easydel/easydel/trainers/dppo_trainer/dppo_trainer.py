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
"""DPPO config and trainer surface backed by GRPO."""

from __future__ import annotations

from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer


@Registry.register("trainer", "dppo")
class DPPOTrainer(GRPOTrainer):
    """DPPO trainer using EasyDeL's native GRPO rollout path.

    DPPO reuses GRPO generation, reward scoring, and batching, then selects the
    DPPO loss branch configured by :class:`DPPOConfig`. The current EasyDeL
    implementation supports sampled-token binary divergence masks and does not
    rely on external inference servers.
    """
