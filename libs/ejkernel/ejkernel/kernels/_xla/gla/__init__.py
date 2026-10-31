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


"""XLA backend for Gated Linear Attention (GLA).

This submodule provides XLA-optimized implementation of Gated Linear Attention,
a linear-complexity attention mechanism with learnable gates.

Key Features:
    - O(N) complexity through recurrent formulation
    - Gated key-value updates for selective memory
    - Chunked computation for training efficiency
    - State caching for autoregressive generation

Reference:
    Gated Linear Attention Transformers with Hardware-Efficient Training
    https://arxiv.org/abs/2312.06635
"""

from ._interface import recurrent_gla

__all__ = ["recurrent_gla"]
