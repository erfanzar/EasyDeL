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


"""XLA-based kernel implementations for attention and related operations.

Pure JAX/XLA implementations of attention mechanisms and related kernels.
These kernels compile to XLA and run on any XLA-supported backend (GPU, TPU,
CPU) without platform-specific dependencies.  They serve as the canonical
numerical reference that all other backends (Triton, Pallas, CUDA) must match.

XLA kernels provide:
    - Cross-platform compatibility (GPU/TPU/CPU)
    - Automatic gradient computation via JAX autodiff (or custom VJP)
    - XLA compilation and device-placement optimisations
    - No platform-specific dependencies

Available Operations:
    Dense Attention:
        - attention: Standard multi-head / GQA / MQA attention (BSHD layout)
        - flash_attention: Memory-efficient chunked attention (O(N) memory)
        - flash_mla: Multi-head Latent Attention (MLA)
        - deepseek_attn: DeepSeek Sparse Attention (MLA + Lightning Indexer)
        - blocksparse_attention: Block-sparse / packed-sequence attention
        - native_sparse_attention: Sparse attention with flexible patterns
        - ring_attention: Distributed ring attention for long sequences
        - scaled_dot_product_attention: Basic scaled dot-product attention
        - unified_attention: Unified paged-decode attention API

    Serving / Paged Attention:
        - page_attention: Single-batch paged KV-cache attention
        - prefill_page_attention: Paged attention for prefill phase
        - decode_attention: Single-token decode over a paged KV buffer
        - chunked_prefill_paged_decode: Mixed prefill+decode with paged cache
        - ragged_decode_attention: Ragged-batch decode attention
        - ragged_page_attention_v2/v3: Variable-length paged attention
        - ragged_page_attention_v2_turboquant: v2 with turboquant KV cache
        - ragged_page_attention_v3_turboquant: v3 with turboquant KV cache
        - multi_latent_ragged_page_attention: MLA ragged paged attention
        - multi_latent_ragged_page_attention_v2: MLA ragged paged attention v2

    Linear / Recurrent Attention:
        - recurrent_gla: Gated linear attention (recurrent form)
        - gated_delta_rule: Gated delta rule linear attention
        - ragged_gated_delta_rule: Ragged-batch gated delta rule
        - lightning_attn: Lightning attention with exponential decay
        - recurrent: General recurrent state-space attention
        - kernel_delta_attention (kda / kda_decay): Delta-rule linear attention

    State Space Models:
        - state_space_v1: Mamba1-style SSM
        - state_space_v2: Mamba2-style SSM
        - rwkv4 / rwkv6 / rwkv7 / rwkv7_mul: RWKV recurrent architectures

    Matrix Operations:
        - all_gather_matmul: All-gather + matmul across a device mesh
        - reduce_scatter_matmul: Matmul + reduce-scatter across a device mesh
        - grouped_matmul: Grouped (MoE-style) matrix multiplication
        - grouped_matmulv3: Grouped matmul v3
        - quantized_matmul: Packed-uint32 quantized matmul

    Utilities:
        - mean_pooling: Masked sequence mean pooling

Note:
    XLA implementations are the fallback when platform-specific kernels
    (Triton, Pallas, CUDA) are unavailable for the current hardware.  They
    are also used as the numerical gold standard in test suites.
"""

from .all_gather_matmul import all_gather_matmul
from .attention import attention
from .blocksparse_attention import blocksparse_attention
from .chunked_prefill_paged_decode import chunked_prefill_paged_decode
from .decode_attention import decode_attention
from .deepseek_attn import deepseek_attn
from .flash_attention import flash_attention
from .flash_mla import flash_mla
from .fused_cross_entropy import fused_cross_entropy
from .fused_kl_divergence import fused_kl_divergence
from .gated_delta_rule import gated_delta_rule
from .gla import recurrent_gla
from .grouped_matmul import grouped_matmul
from .grouped_matmulv3 import grouped_matmulv3
from .kernel_delta_attention import kda, kda_decay, kernel_delta_attention
from .lightning_attn import lightning_attn
from .mean_pooling import mean_pooling
from .multi_latent_ragged_page_attention import multi_latent_ragged_page_attention
from .multi_latent_ragged_page_attention_v2 import multi_latent_ragged_page_attention_v2
from .native_sparse_attention import apply_native_sparse_attention
from .page_attention import page_attention
from .prefill_page_attention import prefill_page_attention
from .quantized_matmul import quantized_matmul
from .ragged_decode_attention import ragged_decode_attention
from .ragged_gated_delta_rule import ragged_gated_delta_rule as ragged_gated_delta_rule
from .ragged_page_attention_v2 import ragged_page_attention_v2
from .ragged_page_attention_v2_turboquant import ragged_page_attention_v2_turboquant
from .ragged_page_attention_v3 import ragged_page_attention_v3
from .ragged_page_attention_v3_turboquant import ragged_page_attention_v3_turboquant
from .recurrent import recurrent
from .reduce_scatter_matmul import reduce_scatter_matmul
from .ring_attention import ring_attention
from .rwkv4 import rwkv4
from .rwkv6 import rwkv6
from .rwkv7 import rwkv7, rwkv7_mul
from .scaled_dot_product_attention import scaled_dot_product_attention
from .state_space_v1 import state_space_v1
from .state_space_v2 import state_space_v2
from .unified_attention import unified_attention

__all__ = [
    "all_gather_matmul",
    "apply_native_sparse_attention",
    "attention",
    "blocksparse_attention",
    "chunked_prefill_paged_decode",
    "decode_attention",
    "deepseek_attn",
    "flash_attention",
    "flash_mla",
    "fused_cross_entropy",
    "fused_kl_divergence",
    "gated_delta_rule",
    "grouped_matmul",
    "grouped_matmulv3",
    "kda",
    "kda_decay",
    "kernel_delta_attention",
    "lightning_attn",
    "mean_pooling",
    "multi_latent_ragged_page_attention",
    "multi_latent_ragged_page_attention_v2",
    "page_attention",
    "prefill_page_attention",
    "quantized_matmul",
    "ragged_decode_attention",
    "ragged_gated_delta_rule",
    "ragged_page_attention_v2",
    "ragged_page_attention_v2_turboquant",
    "ragged_page_attention_v3",
    "ragged_page_attention_v3_turboquant",
    "recurrent",
    "recurrent_gla",
    "reduce_scatter_matmul",
    "ring_attention",
    "rwkv4",
    "rwkv6",
    "rwkv7",
    "rwkv7_mul",
    "scaled_dot_product_attention",
    "state_space_v1",
    "state_space_v2",
    "unified_attention",
]
