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


"""EJKernel Triton Kernels Module.

This module provides optimized Triton kernel implementations for various
attention mechanisms and neural network operations. All kernels are designed
for efficient GPU execution using JAX and Triton.

Available kernels:
- blocksparse_attention: Block-sparse attention with custom sparsity patterns
- chunked_prefill_paged_decode: Hybrid prefill + paged decode for KV-cache inference
- decode_attention: Paged decode-time attention (vLLM-style)
- flash_attention: Memory-efficient exact attention with O(N) memory complexity
- flash_mla: Flash Multi-head Latent Attention stub (not yet implemented)
- recurrent: Linear-time recurrent attention mechanisms
- recurrent_gla: Gated Linear Attention implementation
- lightning_attn: Lightning Attention with layer-adaptive decay
- mean_pooling: Efficient mean pooling over sequence dimension
- native_sparse_attention: Sparse attention with dynamic block selection
- page_attention: Standard paged attention
- quantized_matmul: Packed uint32 quantized matmul
- ragged_decode_attention: Ragged-batch decode attention
- ragged_page_attention_v2: Ragged paged attention (v2)
- ragged_page_attention_v3: Ragged paged attention (v3)
- ring_attention: Ring-buffer attention for long contexts
- rwkv4, rwkv6, rwkv7: RWKV recurrent model kernels
- unified_attention: Unified paged attention supporting mixed prefill/decode

All implementations support:
- Variable-length sequences via cumulative sequence lengths
- Custom backward passes for gradient computation where applicable
- Automatic kernel tuning for optimal performance
"""

from .blocksparse_attention import blocksparse_attention
from .chunked_prefill_paged_decode import chunked_prefill_paged_decode
from .decode_attention import decode_attention
from .flash_attention import flash_attention
from .gla import recurrent_gla
from .lightning_attn import lightning_attn
from .mean_pooling import mean_pooling
from .native_sparse_attention import apply_native_sparse_attention, native_sparse_attention
from .page_attention import page_attention
from .quantized_matmul import quantized_matmul
from .ragged_decode_attention import ragged_decode_attention
from .ragged_page_attention_v2 import ragged_page_attention_v2
from .ragged_page_attention_v3 import ragged_page_attention_v3
from .recurrent import recurrent
from .ring_attention import ring_attention
from .rwkv4 import rwkv4
from .rwkv6 import rwkv6
from .rwkv7 import rwkv7, rwkv7_mul
from .unified_attention import unified_attention

__all__ = (
    "apply_native_sparse_attention",
    "blocksparse_attention",
    "chunked_prefill_paged_decode",
    "decode_attention",
    "flash_attention",
    "lightning_attn",
    "mean_pooling",
    "native_sparse_attention",
    "page_attention",
    "quantized_matmul",
    "ragged_decode_attention",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "recurrent",
    "recurrent_gla",
    "ring_attention",
    "rwkv4",
    "rwkv6",
    "rwkv7",
    "rwkv7_mul",
    "unified_attention",
)
