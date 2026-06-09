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


"""Triton backend for Lightning Attention.

This submodule provides GPU-optimized Lightning Attention using Triton
kernels with layer-adaptive exponential decay.

Key Features:
    - O(N) complexity with linear attention formulation
    - Layer-dependent decay rates for varied temporal receptive fields
    - Chunked processing for efficient training
    - State persistence for incremental inference
"""

from ._interface import lightning_attn

__all__ = ("lightning_attn",)
