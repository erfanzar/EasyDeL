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

"""Tile-lang ``all_gather_matmul`` — single-device subset (v0).

When ``tp_size == 1`` (or ``None``) the collective is a no-op and the
operation reduces to a single :func:`dense_matmul_tilelang` call.
Multi-device (``tp_size > 1``) is not yet implemented.
"""

from ._interface import all_gather_matmul

__all__ = ["all_gather_matmul"]
