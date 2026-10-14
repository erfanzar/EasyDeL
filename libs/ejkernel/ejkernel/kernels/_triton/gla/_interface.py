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
"""Public interface and kernel-registry entry for Gated Linear Attention (GLA).

This module re-exports :func:`recurrent_gla` and registers it under the
``"gla"`` key for the ``Platform.TRITON / Backend.GPU`` combination.  The
actual computation is delegated to the ``recurrent`` kernel via
:mod:`ejkernel.kernels._triton.recurrent`.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import Array, Float, Int
from ._triton_impl_fwd import recurrent_gla as _recurrent_gla_impl


@kernel_registry.register("gla", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def recurrent_gla(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Compute Gated Linear Attention (GLA) in a recurrent, linear-time manner.

    This is the kernel-registry entry point for GLA on the Triton/GPU platform.
    It delegates to :func:`ejkernel.kernels._triton.recurrent.recurrent` with
    the gate tensor ``g`` applied.

    Supports both padded-batch mode and variable-length (packed) mode via
    ``cu_seqlens``.

    Args:
        query: Query tensor [batch, seq_len, num_heads, qk_head_dim].
            When ``cu_seqlens`` is used, shape must be [1, total_tokens, num_heads, qk_head_dim].
        key: Key tensor [batch, seq_len, num_kv_heads, qk_head_dim].
        value: Value tensor [batch, seq_len, num_kv_heads, v_head_dim].
        g: Optional gate tensor [batch, seq_len, num_heads, qk_head_dim].
            Element-wise gates applied inside the recurrent update.
        g_gamma: Optional per-head decay factors with a trailing ``num_heads``
            dimension, e.g. shape ``[num_heads]`` or ``[batch, num_heads]``.
        softmax_scale: Scaling factor applied to queries before the recurrent
            computation. Defaults to ``1 / sqrt(qk_head_dim)``.
        initial_state: Initial recurrent hidden state
            [(...,) num_heads, qk_head_dim, v_head_dim].
            Useful for chunked long-sequence processing.
        reverse: If True, process the sequence in reverse temporal order.
        cu_seqlens: Cumulative sequence lengths [num_seqs + 1] for packed
            variable-length inputs. When provided, batch dimension of all
            input tensors must be 1.

    Returns:
        A 2-tuple:
            - output: Attention output, same shape as ``query``.
            - final_state: Final recurrent state
              [(...,) num_heads, qk_head_dim, v_head_dim], usable as
              ``initial_state`` for the next chunk.

    Raises:
        ValueError: If ``cu_seqlens`` is provided and ``query.shape[0] != 1``.
        ValueError: If ``cu_seqlens`` is provided and ``initial_state.shape[0]``
            does not equal ``len(cu_seqlens) - 1``.
    """
    return _recurrent_gla_impl(
        query,
        key,
        value,
        g,
        g_gamma,
        softmax_scale,
        initial_state,
        reverse,
        cu_seqlens,
        block_k,
        block_v,
        num_warps,
        num_stages,
    )


__all__ = ("recurrent_gla",)
