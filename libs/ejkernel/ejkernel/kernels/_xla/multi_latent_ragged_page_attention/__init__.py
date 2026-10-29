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

"""XLA backend for Multi-Latent Attention (MLA) ragged paged attention.

This submodule provides a pure-JAX/XLA reference implementation of the
MLA ragged paged-attention kernel used by DeepSeek-style models.

The operation combines two distinct steps in a single fused kernel:

1. **KV cache write**: new ``keys_values`` and ``keys_pe`` tokens are packed and
   written into the paged KV cache at the positions indicated by
   ``kv_lens`` and ``block_tables``.

2. **Ragged attention**: for each sequence in the ragged batch, online-softmax
   scaled dot-product attention is computed over all cached tokens up to the
   current ``kv_lens`` length, producing an output per query token.

Tensor layout for the KV cache::

    kv_cache[num_pages, page_size / kv_packing, kv_packing, kv_dim_padded]

where ``kv_packing = 32 // bitwidth`` and ``kv_dim_padded`` is the padded
combined nope+pe dimension (each component padded to a multiple of 128).

The implementation is backend-agnostic XLA and serves as the numerical
reference that TPU/GPU Pallas kernels must match.
"""

from ._interface import multi_latent_ragged_page_attention

__all__ = ("multi_latent_ragged_page_attention",)
