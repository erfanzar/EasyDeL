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


"""Triton backend for Flash Attention.

This submodule provides GPU-optimized Flash Attention using Triton kernels.
Achieves O(N) memory complexity through tiled computation with online softmax.

Key Features:
    - Memory-efficient exact attention computation
    - Custom forward and backward Triton kernels
    - Support for causal masking, sliding window, and bias
    - GQA/MQA head configurations
"""

from ._interface import _jax_bwd_attention_call as flash_attention_gpu_bwd
from ._interface import _jax_fwd_attention_call as flash_attention_gpu_fwd
from ._interface import flash_attention

__all__ = (
    "flash_attention",
    "flash_attention_gpu_bwd",
    "flash_attention_gpu_fwd",
)
