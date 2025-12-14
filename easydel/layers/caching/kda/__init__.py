# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""KDA Cache module for Key-Driven Attention linear attention layers.

This module provides caching for KDA-style linear attention (used in Kimi)
which uses separate convolution states for Q, K, V projections.
"""

from .cache import (
    KDACache,
    KDACacheConfig,
    KDACacheView,
    KDAMetadata,
)

__all__ = (
    "KDACache",
    "KDACacheConfig",
    "KDACacheView",
    "KDAMetadata",
)
