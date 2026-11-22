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

"""Pallas TPU backend for Flash Multi-head Latent Attention.

This submodule provides a TPU-optimised Flash MLA implementation using
Pallas with Mosaic backend.  The kernel tiles the attention computation
and projects the compressed KV latent on-the-fly inside each tile to
avoid materialising the full K/V tensors in HBM.

Key Features:
    - On-the-fly KV reconstruction from low-rank latent
    - Memory-efficient O(N) tiled attention
    - Three RoPE modes (none / fused / decoupled)
    - GQA support (q_heads > kv_heads)
    - Causal masking and sliding window
    - Logits soft-capping
"""

from ._interface import flash_mla

__all__ = ("flash_mla",)
