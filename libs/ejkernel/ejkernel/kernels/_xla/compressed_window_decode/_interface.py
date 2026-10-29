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

"""Compressed-window decode attention interface (XLA reference)."""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.ops import FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import compressed_window_decode_xla


@kernel_registry.register("compressed_window_decode", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def compressed_window_decode(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len kv_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Shared-KV sink-augmented decode attention with an explicit bias (XLA).

    Registered under ``"compressed_window_decode"`` for ``Platform.XLA`` /
    ``Backend.ANY``. This is the numerical reference for the DeepSeek-V4
    compressed-window decode step; the Pallas TPU implementation must match it.

    Args:
        query: Rotated queries ``[batch, num_heads, q_len, head_dim]``. During
            serving decode ``q_len == 1``; the reference also handles ``q_len > 1``.
        kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``): the
            sliding-window token ring concatenated with the compressed entries.
        bias: Additive fp32 mask ``[batch, q_len, kv_len]`` broadcast over heads
            (ring validity, compressed-entry causal thresholds, indexer gating).
        softmax_aux: Optional per-head sink logits ``[num_heads]``. ``None``
            disables sinks.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        fwd_params: Unused tiling hint (interface parity; the reference is dense).

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    del fwd_params
    return compressed_window_decode_xla(
        query=query,
        kv=kv,
        bias=bias,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
    )
