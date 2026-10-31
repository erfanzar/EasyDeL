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

"""TurboQuant-compressed ragged page KV cache.

Stores KV caches in TurboQuant format using separate page tensors for
codebook indices, QJL signs, and norms. Achieves ~3.8x compression
at 4-bit vs bf16 while preserving attention accuracy.
"""

from .cache import (
    TurboQuantRaggedPagesCache,
    TurboQuantRaggedPagesCacheConfig,
    TurboQuantRaggedPagesCacheView,
)

__all__ = (
    "TurboQuantRaggedPagesCache",
    "TurboQuantRaggedPagesCacheConfig",
    "TurboQuantRaggedPagesCacheView",
)
