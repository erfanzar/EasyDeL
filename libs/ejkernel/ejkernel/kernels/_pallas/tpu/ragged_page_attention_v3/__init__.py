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


"""Pallas TPU backend for Ragged Paged Attention v3.

This submodule provides TPU-optimized paged attention v3 for mixed prefill
and decode operations. Based on Google/vLLM implementation with extensions.

Key Features:
    - Mixed prefill and decode in single batch
    - Attention sink support for head_dim==128
    - Integrated KV cache update
    - Sliding window attention support
"""

from ._interface import ragged_page_attention_v3

__all__ = ("ragged_page_attention_v3",)
