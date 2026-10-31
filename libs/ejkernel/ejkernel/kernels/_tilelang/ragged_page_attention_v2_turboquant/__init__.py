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

"""TileLang backend for TurboQuant ragged paged attention v2.

Exposes :func:`ragged_page_attention_v2_turboquant`, which performs inference
attention over a KV cache compressed with the TurboQuant VQ + residual-sign
scheme.  Keys are stored as 4-bit codebook indices plus binary sign bits for
a residual QJL projection; values are stored as 4-bit codebook indices with
a scalar norm.
"""

from ._interface import ragged_page_attention_v2_turboquant

__all__ = ["ragged_page_attention_v2_turboquant"]
