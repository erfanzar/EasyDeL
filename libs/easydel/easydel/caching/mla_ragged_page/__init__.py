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

"""Multi-Latent Attention (MLA) ragged page cache.

Provides a paged KV-cache optimized for MLA architectures (e.g. DeepSeek-V3)
where key and value projections share a single low-rank compressed latent.
Pages are laid out in the format expected by the ``ejkernel`` attention backend:
``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``.

Public API:
    - :class:`MLARaggedPagesCacheConfig` -- cache geometry and HBM budget.
    - :class:`MLARaggedPagesCacheView`   -- per-layer page buffer.
    - :class:`MLARaggedPagesCache`       -- multi-layer container.
"""

from .cache import MLARaggedPagesCache, MLARaggedPagesCacheConfig, MLARaggedPagesCacheView

__all__ = (
    "MLARaggedPagesCache",
    "MLARaggedPagesCacheConfig",
    "MLARaggedPagesCacheView",
)
