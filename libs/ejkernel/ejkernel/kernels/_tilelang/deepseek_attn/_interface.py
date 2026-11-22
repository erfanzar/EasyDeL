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

"""Tile-lang DeepSeek Sparse Attention (DSA) — native fused kernels.

The whole forward runs on two natively-authored ``@T.prim_func`` kernels: a
Lightning-Indexer kernel and a fused sparse-MLA attention kernel that
reconstructs ``K`` / ``V`` from the compressed latent inside the attention
loop. See :mod:`._kernel`. No ``jnp.einsum`` / ``jax.vmap`` carries the
attention math — only the ``top_k`` selection between the two kernels is a
host-side op.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._impl import deepseek_attn_tilelang


@kernel_registry.register("deepseek_attn", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def deepseek_attn(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
    key_index: Float[Array, "batch seq_len index_head_dim"],
    index_weights: Float[Array, "batch seq_len index_heads"],
    index_topk: int = 2048,
    softmax_scale: float | None = None,
    index_softmax_scale: float | None = None,
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    causal: bool = True,
    *,
    gemm_block: int = 128,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """DeepSeek Sparse Attention — native tile-lang Lightning-Indexer + fused
    sparse-MLA attention kernels (forward).

    GQA/MQA (``kv_heads`` dividing ``q_heads``), RoPE score tails
    (``b_q`` / ``b_k``) and different value head dimensions are handled by
    native TileLang pack/pad kernels before the FlashAttention core.

    Raises:
        EjkernelRuntimeError: if the MLA dimensions are inconsistent or if
            ``q_heads`` is not a multiple of ``kv_heads``.
    """
    q_head_dim = query.shape[-1]
    kv_heads = w_kc.shape[1]
    q_heads = query.shape[2]
    qk_nope = w_kc.shape[-1]
    v_head_dim = w_vc.shape[-1]

    if b_k is None and q_head_dim != qk_nope:
        raise EjkernelRuntimeError(f"tile-lang deepseek_attn without b_k requires query dim {q_head_dim} == {qk_nope}.")
    if b_k is not None:
        rope_dim = b_k.shape[-1]
        if b_q is None and q_head_dim != qk_nope + rope_dim:
            raise EjkernelRuntimeError(
                "tile-lang deepseek_attn with b_k requires query dim to equal qk_nope_head_dim + qk_rope_head_dim "
                f"(got {q_head_dim}, {qk_nope} + {rope_dim})."
            )
        if b_q is not None and q_head_dim != qk_nope:
            raise EjkernelRuntimeError(
                "tile-lang deepseek_attn with b_q/b_k requires query dim to equal qk_nope_head_dim "
                f"(got {q_head_dim}, {qk_nope})."
            )
    if v_head_dim <= 0:
        raise EjkernelRuntimeError("tile-lang deepseek_attn requires a positive value head dimension.")
    if q_heads % kv_heads != 0:
        raise EjkernelRuntimeError(
            f"tile-lang deepseek_attn requires kv_heads ({kv_heads}) to divide q_heads ({q_heads})."
        )

    return deepseek_attn_tilelang(
        query,
        key_value,
        w_kc,
        w_vc,
        query_index,
        key_index,
        index_weights,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        index_softmax_scale=index_softmax_scale,
        b_q=b_q,
        b_k=b_k,
        causal=causal,
        gemm_block=int(gemm_block),
    )


__all__ = ["deepseek_attn"]
