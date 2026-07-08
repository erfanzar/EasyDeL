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

"""Compressed-window attention interface (XLA reference, differentiable)."""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.ops import FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import compressed_window_attention_xla


@kernel_registry.register("compressed_window_attention", Platform.XLA, Backend.ANY)
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
    """Sink-augmented shared-KV full-sequence attention with a bias (XLA).

    Registered under ``"compressed_window_attention"`` for ``Platform.XLA`` /
    ``Backend.ANY`` (priority 0). This is the mandatory fallback and the
    numerical + gradient reference for DeepSeek-V4's training / prefill forward;
    the Pallas TPU kernel must match it in both value and gradient. Plain dense
    XLA math, so ``jax.grad`` differentiates it natively.

    Args:
        query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
        kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``): the
            sliding-window token KVs concatenated with the compressed entries.
        bias: Additive fp32 mask ``[batch, q_len, kv_len]`` broadcast over heads
            (sliding-window causal, compressed-entry causal, indexer gating).
        softmax_aux: Optional per-head sink logits ``[num_heads]``. ``None``
            disables sinks.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        fwd_params: Unused tiling hint (interface parity; the reference is dense).
        window: Accepted and ignored — the dense reference masks via ``bias``
            regardless, so it fully supports the band-skip call signature (it is
            deliberately NOT ``del``'d, which would mark it unsupported).
        token_kv_len: Accepted and ignored (interface parity).

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    del fwd_params  # window / token_kv_len are supported no-ops for the dense path
    return compressed_window_attention_xla(
        query=query,
        kv=kv,
        bias=bias,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
    )
