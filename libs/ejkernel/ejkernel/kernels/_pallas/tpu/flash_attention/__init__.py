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


"""Pallas TPU backend for Flash Attention.

This submodule provides TPU-optimized Flash Attention using Pallas with
Mosaic backend for memory-efficient exact attention computation.

Key Features:
    - Memory-efficient O(N) computation using tiled approach
    - Custom forward and backward Pallas kernels
    - Support for causal masking and sliding window
    - Optimized for TPU Matrix Units
"""

from ._interface import _flash_attention_bwd as flash_attention_tpu_bwd
from ._interface import _flash_attention_fwd as flash_attention_tpu_fwd
from ._interface import flash_attention

__all__ = (
    "flash_attention",
    "flash_attention_tpu_bwd",
    "flash_attention_tpu_fwd",
)
