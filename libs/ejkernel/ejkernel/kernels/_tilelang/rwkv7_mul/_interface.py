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

"""Tile-lang ``rwkv7_mul`` — same DPLR scan as ``rwkv7``.

The actual registration lives in ``ejkernel/kernels/_tilelang/rwkv7/_interface.py``
(both rwkv7 and rwkv7_mul are registered there, both pointing at the same
native tile-lang kernel). This module exists so the package layout
mirrors the XLA tree.
"""

from ..rwkv7._interface import rwkv7_mul

__all__ = ["rwkv7_mul"]
