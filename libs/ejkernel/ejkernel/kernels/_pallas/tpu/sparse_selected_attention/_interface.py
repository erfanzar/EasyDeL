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

"""Registry entry for the TPU Pallas mask-fused selected (top-k mask) attention."""

from __future__ import annotations

from jaxtyping import Array, Bool, Float

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.sparse_selected_attention._xla_impl import sparse_selected_attention_impl
from ._pallas_impl import sparse_selected_attention_pallas


@kernel_registry.register("sparse_selected_attention", Platform.PALLAS, Backend.TPU, priority=10)
def sparse_selected_attention(
    query: Float[Array, "batch seq_len num_heads qk_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_dim"],
    mask: Bool[Array, "batch seq_len seq_len"],
    softmax_scale: float | None = None,
    causal: bool = True,
    block_q: int = 512,
    block_k: int = 512,
) -> Float[Array, "batch seq_len num_heads v_dim"]:
    """Same contract as the XLA entry; Pallas forward + backward on TPU.

    Default tiles are 512x512: on TPU v5 (glm_moe_dsa shapes, Dk=192 Dv=128) fwd+bwd is
    2.65 ms at 4k / 33.1 ms at 16k vs 4.32 / 58.6 ms at 256; 1024 overflows scoped VMEM.

    Head counts that are not a GQA multiple route to the XLA implementation.
    """
    if softmax_scale is None:
        softmax_scale = float(query.shape[-1]) ** -0.5
    if query.shape[2] % key.shape[2]:
        return sparse_selected_attention_impl(
            query, key, value, mask, float(softmax_scale), bool(causal), int(block_q), int(block_k)
        )
    return sparse_selected_attention_pallas(
        query, key, value, mask, float(softmax_scale), bool(causal), int(block_q), int(block_k)
    )


__all__ = ("sparse_selected_attention",)
