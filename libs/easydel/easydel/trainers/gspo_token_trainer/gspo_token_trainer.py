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
"""GSPO token-level importance-sampling aliases."""

from __future__ import annotations

from easydel.utils import Registry

from ..group_sequence_policy_optimization import GSPOTrainer
from .gspo_token_config import GSPOTokenConfig


@Registry.register("trainer", "gspo_token")
class GSPOTokenTrainer(GSPOTrainer):
    """GSPO trainer alias with token-aware importance sampling config.

    The class does not override GSPO runtime behavior. Its purpose is registry
    separation: ``gspo_token`` resolves to the same trainer machinery with a
    config that pins the expected token-aware importance-sampling mode.
    """

    arguments: GSPOTokenConfig
