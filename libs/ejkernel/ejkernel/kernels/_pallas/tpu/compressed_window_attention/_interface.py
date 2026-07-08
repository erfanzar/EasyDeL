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

"""Compressed-window full-sequence attention interface (Pallas TPU)."""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.ops import FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import compressed_window_attention_tpu


@kernel_registry.register("compressed_window_attention", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def compressed_window_attention(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len kv_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    window: int = 0,
    token_kv_len: int = 0,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Tiled-flash sink-augmented shared-KV attention (Pallas TPU, differentiable).

    Registered under ``"compressed_window_attention"`` for ``Platform.PALLAS`` /
    ``Backend.TPU``. Signature matches the XLA reference (which it must match in
    value and gradient); see that impl for argument semantics. Streams the KV
    axis in tiles via ``pltpu.make_async_copy`` double-buffering with an online
    softmax, and is differentiable via a ``custom_vjp`` (dense-reference
    backward this pass).

    Args:
        query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``).
        bias: Additive fp32 mask ``[batch, q_len, kv_len]`` (head-broadcast).
        softmax_aux: Optional per-head sink logits ``[num_heads]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        fwd_params: Optional tiling hints (``q_blocksize`` / ``kv_blocksize``).
        window: Sliding-window size for band-skip (``0`` scans the whole KV).
        token_kv_len: Length of the sliding-window token region (entries follow).

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    return compressed_window_attention_tpu(
        query=query,
        kv=kv,
        bias=bias,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        window=window,
        token_kv_len=token_kv_len,
    )
