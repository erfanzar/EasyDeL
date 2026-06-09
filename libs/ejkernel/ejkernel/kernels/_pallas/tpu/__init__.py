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


"""Pallas TPU kernel implementations (Mosaic backend).

Provides TPU-optimized kernels compiled through JAX Pallas with the Mosaic
(XLA extension) backend.  Kernels exploit the TPU Matrix Multiply Unit (MXU),
the VMEM (on-chip vector memory) / HBM hierarchy, and the TPU's DMA engine for
asynchronous data movement.

Available kernels:
    all_gather_matmul: Bidirectional ring all-gather fused with matmul; uses
        TPU DMA for peer-to-peer data movement across devices.
    blocksparse_attention: Block-sparse (Splash) attention with support for
        causal, local, chunked-causal, and fully-custom sparsity patterns.
    deepseek_attn: DeepSeek-style sparse attention combining MLA with a
        lightning indexer for KV cache selection.
    flash_attention: Memory-efficient exact attention (FlashAttention-TPU).
    flash_mla: Multi-head latent attention forward pass.
    fused_cross_entropy: Fused sparse cross-entropy with analytic backward.
    fused_kl_divergence: Fused KL divergence with analytic student backward.
    gated_delta_rule: Gated delta-rule linear attention recurrence.
    grouped_matmul: Grouped (expert) matrix multiplication (v1).
    grouped_matmulv2: Grouped matmul v2 with improved tiling.
    grouped_matmulv3: Grouped matmul v3 with further optimizations.
    multi_latent_ragged_page_attention: MLA ragged paged attention (v1).
    multi_latent_ragged_page_attention_v2: MLA ragged paged attention v2.
    page_attention: Standard paged KV-cache attention.
    prefill_page_attention: Prefill-phase paged attention.
    quantized_matmul: Packed-weight quantized matrix multiplication.
    ragged_decode_attention: Decode-phase attention for ragged batches.
    ragged_gated_delta_rule: Ragged-batch gated delta-rule recurrence.
    ragged_page_attention_v2: Variable-length paged attention v2.
    ragged_page_attention_v3: Variable-length paged attention v3 (VMEM-safe).
    reduce_scatter_matmul: Matmul fused with reduce-scatter collective.
    ring_attention: Distributed attention using ring all-reduce collectives.
"""

from .all_gather_matmul import all_gather_matmul
from .blocksparse_attention import blocksparse_attention as blocksparse_attention
from .deepseek_attn import deepseek_attn
from .flash_attention import flash_attention
from .flash_mla import flash_mla
from .fused_cross_entropy import fused_cross_entropy
from .fused_kl_divergence import fused_kl_divergence
from .gated_delta_rule import gated_delta_rule
from .grouped_matmul import grouped_matmul
from .grouped_matmulv2 import grouped_matmulv2
from .grouped_matmulv3 import grouped_matmulv3
from .multi_latent_ragged_page_attention import multi_latent_ragged_page_attention
from .multi_latent_ragged_page_attention_v2 import multi_latent_ragged_page_attention_v2
from .page_attention import page_attention
from .prefill_page_attention import prefill_page_attention
from .quantized_matmul import quantized_matmul
from .ragged_decode_attention import ragged_decode_attention
from .ragged_gated_delta_rule import ragged_gated_delta_rule as ragged_gated_delta_rule
from .ragged_page_attention_v2 import ragged_page_attention_v2
from .ragged_page_attention_v3 import ragged_page_attention_v3
from .reduce_scatter_matmul import reduce_scatter_matmul
from .ring_attention import ring_attention

__all__ = (
    "all_gather_matmul",
    "blocksparse_attention",
    "deepseek_attn",
    "flash_attention",
    "flash_mla",
    "fused_cross_entropy",
    "fused_kl_divergence",
    "gated_delta_rule",
    "grouped_matmul",
    "grouped_matmulv2",
    "grouped_matmulv3",
    "multi_latent_ragged_page_attention",
    "multi_latent_ragged_page_attention_v2",
    "page_attention",
    "prefill_page_attention",
    "quantized_matmul",
    "ragged_decode_attention",
    "ragged_gated_delta_rule",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "reduce_scatter_matmul",
    "ring_attention",
)
