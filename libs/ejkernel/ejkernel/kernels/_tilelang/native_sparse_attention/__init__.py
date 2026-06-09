# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""TileLang Native Sparse Attention (NSA) kernel package.

Exports two registered GPU kernels:
- ``apply_native_sparse_attention``: raw selected-block sparse attention
  without compression or gating.
- ``native_sparse_attention``: higher-level wrapper with optional output
  gating via ``g_slc``.

Both are forward+backward capable.  Compressed/top-k attention (``g_cmp``)
and packed/ragged inputs are not yet implemented.
"""

from ._interface import apply_native_sparse_attention, native_sparse_attention

__all__ = ["apply_native_sparse_attention", "native_sparse_attention"]
