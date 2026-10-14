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


"""Pallas TPU backend for Grouped Matrix Multiplication.

This submodule provides TPU-optimized grouped matrix multiplication
using Pallas with Mosaic backend.

Key Features:
    - Efficient batched processing of variable-size matmuls
    - Optimized tiling for TPU Matrix Units
    - Support for group indices and offsets
    - Incremental accumulation support
"""

from ._interface import grouped_matmul

__all__ = ("grouped_matmul",)
