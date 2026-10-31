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

"""Detection of per-example text metadata that online-RL reward functions read from ``batch``."""

from __future__ import annotations

import typing as tp


def is_reward_sideband(value: tp.Any, depth: int = 0) -> bool:
    """Whether ``value`` is JSON-like metadata containing text (strings, or lists/dicts holding strings).

    Bare numeric columns are excluded so nothing new reaches the purified, model-facing batch.

    Args:
        value: Column value for one example.
        depth: Current nesting depth; values nested deeper than 4 levels are rejected.

    Returns:
        ``True`` when the value should travel to reward functions as Python data.
    """
    if depth > 4:
        return False
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        return any(is_reward_sideband(v, depth + 1) for v in value) and all(
            v is None or isinstance(v, (str, int, float, bool, list, tuple, dict)) for v in value
        )
    if isinstance(value, dict):
        return all(isinstance(k, str) for k in value) and any(is_reward_sideband(v, depth + 1) for v in value.values())
    return False
