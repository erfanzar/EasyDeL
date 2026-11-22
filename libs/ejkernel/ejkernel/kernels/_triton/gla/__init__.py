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


"""Triton backend for Gated Linear Attention (GLA).

This submodule provides GPU-optimized GLA using Triton kernels with
recurrent formulation for O(N) complexity.

Key Features:
    - Linear complexity through recurrent state updates
    - Gated key-value updates for selective memory
    - Chunked computation for training efficiency
    - State caching for autoregressive generation
"""

from ._interface import recurrent_gla

__all__ = ("recurrent_gla",)
