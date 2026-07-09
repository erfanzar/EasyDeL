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

The runner-native drafter path (draft/verify/commit inside
:class:`~easydel.inference.esurge.runners.model_runner.eSurgeRunner`) is the
production speculative-decoding implementation. This package holds its
shared, model-agnostic helpers.
"""

from .support import SpecDecodeStats, build_target_kv_pairs, default_assistant_layer_mapping

__all__ = (
    "SpecDecodeStats",
    "build_target_kv_pairs",
    "default_assistant_layer_mapping",
)
