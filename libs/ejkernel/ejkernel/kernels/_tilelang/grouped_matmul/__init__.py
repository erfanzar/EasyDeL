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

"""TileLang backend for grouped matrix multiplication (v1 and v2).

Both ``grouped_matmul`` and ``grouped_matmulv2`` are registered here and share
the same underlying kernel via :mod:`._interface`.

Exports:
    grouped_matmul: Grouped matmul (v1).
    grouped_matmulv2: Grouped matmul (v2 — identical kernel to v1).
"""

from ._interface import grouped_matmul

__all__ = ["grouped_matmul"]
