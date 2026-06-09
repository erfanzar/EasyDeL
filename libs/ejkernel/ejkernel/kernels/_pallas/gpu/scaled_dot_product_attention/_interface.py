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
"""Public interface and kernel-registry registration for GPU SDPA.

Registers ``scaled_dot_product_attention`` under both ``Platform.PALLAS`` and
``Platform.TRITON`` (both with ``Backend.GPU``) so the kernel registry can
dispatch to this cuDNN implementation from either platform selector.

The function signature is type-checked at call time via ``beartype`` +
``jaxtyping``.  The actual computation delegates to
:func:`._pallas_impl_fwd.scaled_dot_product_attention`.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ejkernel.ops import BwdParams, FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import Array, Bool, Callable, Float, Int
from ._pallas_impl_fwd import scaled_dot_product_attention as _scaled_dot_product_attention_impl


@kernel_registry.register("scaled_dot_product_attention", Platform.PALLAS, Backend.GPU)
@kernel_registry.register("scaled_dot_product_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def scaled_dot_product_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    cum_seqlens_q: Int[Array, "batch"] | None = None,
    cum_seqlens_k: Int[Array, "batch"] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Registered entry point for GPU scaled dot-product attention via cuDNN.

    Thin wrapper that adds ``beartype``/``jaxtyping`` runtime type-checking
    and registers the function in the kernel registry under both
    ``Platform.PALLAS / Backend.GPU`` and ``Platform.TRITON / Backend.GPU``.
    All arguments and semantics are identical to
    :func:`._pallas_impl_fwd.scaled_dot_product_attention`; refer to that
    function for full parameter documentation.

    Args:
        query: ``[batch, seq_len, num_q_heads, head_dim]``.
        key: ``[batch, kv_len, num_kv_heads, head_dim]``.
        value: ``[batch, kv_len, num_kv_heads, head_dim]``.
        attention_mask: Optional boolean mask
            ``[batch, num_heads_or_1, seq_len, kv_len]``.
        bias: Optional additive bias ``[batch, num_heads, seq_len, kv_len]``.
        init_bias: Optional lazy bias factory; called when ``bias`` is None.
        softmax_scale: Logit scale; defaults to ``1/sqrt(head_dim)``.
        causal: Enable causal (lower-triangular) masking.
        sliding_window: Local attention window (int or ``(left, right)``).
        cum_seqlens_q: Cumulative query lengths for packed sequences
            ``[batch+1]``.
        cum_seqlens_k: Cumulative key/value lengths for packed sequences
            ``[batch+1]``.
        fwd_params: Operation-level tuning hint accepted for API parity; ignored by cuDNN.
        bwd_params: Operation-level tuning hint accepted for API parity; ignored by cuDNN.

    Returns:
        Attention output of shape ``[batch, seq_len, num_q_heads, head_dim]``.
    """
    _ = fwd_params, bwd_params
    return _scaled_dot_product_attention_impl(
        query,
        key,
        value,
        attention_mask,
        bias,
        init_bias,
        softmax_scale,
        causal,
        sliding_window,
        cum_seqlens_q,
        cum_seqlens_k,
    )


__all__ = ("scaled_dot_product_attention",)
