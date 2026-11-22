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

"""XLA backend for Multi-Latent Attention (MLA) ragged paged attention v2.

Version 2 extends the v1 API with per-case (decode / prefill / mixed) block-size
tuning: ``num_kv_pages_per_block`` and ``num_queries_per_block`` may now be
supplied as either a scalar or a 3-tuple ``(decode_size, prefill_size,
mixed_size)``.  The XLA fallback normalises these to a scalar (taking the
mixed-case value at index 2) and delegates to the v1 XLA kernel.

This submodule's public interface is otherwise identical to v1.
"""

from ._interface import multi_latent_ragged_page_attention_v2

__all__ = ("multi_latent_ragged_page_attention_v2",)
