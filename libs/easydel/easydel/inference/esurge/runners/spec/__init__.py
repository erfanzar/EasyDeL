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

"""Speculative-decoding support for the eSurge runner.

The runner-native drafter path is the production speculative-decoding
implementation. It is packaged as a pluggable strategy driven by
:class:`~easydel.inference.esurge.runners.model_runner.eSurgeRunner`:

- :class:`SpeculativeStrategy` / :class:`NullSpeculation` (interface.py) —
  the strategy protocol and the inert no-drafter implementation.
- :class:`DrafterSpeculation` (strategy.py) — the drafter-backed
  draft/verify/commit machinery.
- support.py — shared, model-agnostic helpers.
"""

from .interface import NullSpeculation, SpeculativeStrategy
from .strategy import DrafterSpeculation
from .support import SpecDecodeStats, build_target_kv_pairs, default_assistant_layer_mapping

__all__ = (
    "DrafterSpeculation",
    "NullSpeculation",
    "SpecDecodeStats",
    "SpeculativeStrategy",
    "build_target_kv_pairs",
    "default_assistant_layer_mapping",
)
