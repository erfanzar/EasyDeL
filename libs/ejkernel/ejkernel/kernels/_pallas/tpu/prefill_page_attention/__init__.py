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


"""Pallas TPU backend for Prefill Paged Attention.

This submodule provides TPU-optimized paged attention for the prefill
phase using Pallas with Mosaic backend.

Key Features:
    - Paged KV cache management during prefill
    - Support for variable-length prompt processing
    - Efficient page table lookups
    - Sliding window attention support
"""

from ._interface import prefill_page_attention

__all__ = ("prefill_page_attention",)
