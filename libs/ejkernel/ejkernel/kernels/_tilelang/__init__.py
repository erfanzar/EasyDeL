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

"""TileLang DSL kernel implementations for NVIDIA GPUs.

Each subpackage exposes one algorithm implemented natively in the TileLang
DSL and wired to JAX through :mod:`ejkernel.callib._tilelang_ffi`.
Implementations register against
:class:`ejkernel.kernels._registry.Platform.TILELANG` via the
``@kernel_registry.register`` decorator so that they are automatically
selected by :func:`ejkernel.kernels._registry.get_kernel` when the caller
targets the ``TILELANG`` platform.

Importing this package eagerly imports every subpackage so that all
``@kernel_registry.register`` hooks run at import time.

Differentiability:

* Native tile-lang forward **and** backward (``jax.custom_vjp``):
  ``flash_attention``, ``attention``, ``blocksparse_attention``,
  ``mean_pooling``, ``quantized_matmul``, ``deepseek_attn``,
  ``flash_mla``, ``recurrent`` (and the ``gla`` / ``lightning_attn`` /
  ``gated_delta_rule`` / ``kernel_delta_attention`` wrappers that share
  its kernels), ``state_space_v1`` / ``state_space_v2`` (and the
  ``mamba1`` / ``mamba2`` / ``ssm1`` / ``ssm2`` aliases), ``rwkv4`` /
  ``rwkv6`` / ``rwkv7`` / ``rwkv7_mul``, ``ragged_gated_delta_rule``,
  ``grouped_matmul`` / ``grouped_matmulv2`` / ``grouped_matmulv3``.
* Native tile-lang forward, **inference-only** (no backward implemented
  here or in the XLA reference): all paged / decode / MLA / chunked-prefill
  attention variants.
"""

from . import (
    all_gather_matmul,
    attention,
    blocksparse_attention,
    chunked_prefill_paged_decode,
    decode_attention,
    deepseek_attn,
    flash_attention,
    flash_mla,
    fused_cross_entropy,
    fused_kl_divergence,
    gated_delta_rule,
    gla,
    grouped_matmul,
    grouped_matmulv2,
    grouped_matmulv3,
    kernel_delta_attention,
    lightning_attn,
    mamba1,
    mamba2,
    mean_pooling,
    multi_latent_ragged_page_attention,
    multi_latent_ragged_page_attention_v2,
    native_sparse_attention,
    page_attention,
    prefill_page_attention,
    quantized_matmul,
    ragged_decode_attention,
    ragged_gated_delta_rule,
    ragged_page_attention_v2,
    ragged_page_attention_v2_turboquant,
    ragged_page_attention_v3,
    ragged_page_attention_v3_turboquant,
    recurrent,
    reduce_scatter_matmul,
    ring_attention,
    rwkv4,
    rwkv6,
    rwkv7,
    rwkv7_mul,
    scaled_dot_product_attention,
    ssm1,
    ssm2,
    state_space_v1,
    state_space_v2,
    unified_attention,
)

__all__ = [
    "all_gather_matmul",
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
    "gla",
    "grouped_matmul",
    "grouped_matmulv2",
    "grouped_matmulv3",
    "kernel_delta_attention",
    "lightning_attn",
    "mamba1",
    "mamba2",
    "mean_pooling",
    "multi_latent_ragged_page_attention",
    "multi_latent_ragged_page_attention_v2",
    "native_sparse_attention",
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
    "reduce_scatter_matmul",
    "ring_attention",
    "rwkv4",
    "rwkv6",
    "rwkv7",
    "rwkv7_mul",
    "scaled_dot_product_attention",
    "ssm1",
    "ssm2",
    "state_space_v1",
    "state_space_v2",
    "unified_attention",
]
