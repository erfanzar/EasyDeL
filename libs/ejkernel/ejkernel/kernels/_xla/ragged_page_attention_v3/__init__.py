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


"""XLA backend for Ragged Page Attention V3.

Fused update-and-attend paged attention for mixed prefill/decode batches.

Key advances over V2:
    - **In-place KV cache writes**: new key/value tokens are written into the
      paged cache before attention, enabling self-attention within prefill.
    - **Merged K/V storage**: keys and values are interleaved in a packed
      layout for cache-friendly access.
    - **Distribution-based dispatch**: a 3-element ``distribution`` array
      describes the prefill/decode breakdown; only ``distribution[2]``
      (total sequences) is used by the XLA backend.
    - **Quantisation scales**: optional ``q_scale``, ``k_scale``, ``v_scale``
      for FP8 / INT8 inference modes.
"""

from ._interface import ragged_page_attention_v3

__all__ = ("ragged_page_attention_v3",)
