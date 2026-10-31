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

"""TileLang interface for Lightning Attention.

Registers ``lightning_attn`` under ``Platform.TILELANG / Backend.GPU``.
The decay rate per head is derived from the layer position using the formula::

    gamma_slope = -(8 / num_heads) * (1 - layer_idx / num_layers)

This is passed as a static scalar to the shared recurrent kernel
(:func:`ejkernel.kernels._tilelang.recurrent._impl.recurrent_tilelang`)
via the ``static_gamma_slope`` argument.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ..recurrent._impl import recurrent_tilelang


@kernel_registry.register("lightning_attn", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def lightning_attn(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    layer_idx: int,
    num_layers: int,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """Run Lightning Attention on GPU via TileLang.

    Computes recurrent linear attention with a layer-dependent, head-specific
    exponential decay.  The decay slope is determined entirely by
    ``layer_idx`` and ``num_layers``::

        gamma_slope = -(8 / num_heads) * (1 - layer_idx / num_layers)

    No per-token or per-head learnable decay parameters are used; the decay
    is a fixed function of layer depth and head index (as in the original
    Lightning Attention paper).

    Args:
        query: ``[batch, seq_len, num_heads, qk_head_dim]`` float tensor.
        key: ``[batch, seq_len, num_kv_heads, qk_head_dim]`` float tensor.
            Multi-query attention (``num_kv_heads < num_heads``) is supported.
        value: ``[batch, seq_len, num_kv_heads, v_head_dim]`` float tensor.
        layer_idx: Zero-based index of the current transformer layer.  Used
            to compute the decay slope.
        num_layers: Total number of transformer layers.  Must be > 0.
        softmax_scale: Optional scalar attention scale.  Defaults to
            ``1 / sqrt(qk_head_dim)`` inside the recurrent kernel.
        initial_state: Optional ``[..., num_heads, qk_head_dim, v_head_dim]``
            float32 initial hidden state.
        reverse: If ``True``, process the sequence in reverse time order.
        cu_seqlens: Optional ``int32`` cumulative-length vector of shape
            ``[num_seqs + 1]`` for packed variable-length sequences.
            Requires ``batch == 1`` and, when ``initial_state`` is provided,
            ``initial_state.shape[0] == num_seqs``.
        block_k: Accepted for operation-level config parity with Triton; the
            current TileLang recurrent kernel uses its internal tile schedule.
        block_v: Accepted for operation-level config parity with Triton; the
            current TileLang recurrent kernel uses its internal tile schedule.
        num_warps: Accepted for operation-level config parity with Triton.
        num_stages: Accepted for operation-level config parity with Triton.

    Returns:
        A 2-tuple ``(output, final_state)`` where:

        - ``output``: ``[batch, seq_len, num_heads, v_head_dim]``.
        - ``final_state``: ``[..., num_heads, qk_head_dim, v_head_dim]``
          float32.

    Raises:
        EjkernelRuntimeError: If ``num_layers <= 0`` or packed-sequence
            constraints are violated.
    """
    if num_layers <= 0:
        raise EjkernelRuntimeError("tile-lang lightning_attn requires num_layers > 0.")
    if cu_seqlens is not None:
        if query.shape[0] != 1:
            raise EjkernelRuntimeError("tile-lang lightning_attn packed cu_seqlens mode expects batch size 1.")
        if cu_seqlens.dtype.name != "int32":
            raise EjkernelRuntimeError("tile-lang lightning_attn packed cu_seqlens must be int32.")
        num_seqs = cu_seqlens.shape[0] - 1
        if initial_state is not None and initial_state.shape[0] != num_seqs:
            raise EjkernelRuntimeError("tile-lang lightning_attn packed initial_state must have one state per sequence.")
    num_heads = query.shape[2]
    gamma_slope = -(8.0 / num_heads) * (1.0 - float(layer_idx) / float(num_layers))
    return recurrent_tilelang(
        query,
        key,
        value,
        initial_state=initial_state,
        softmax_scale=softmax_scale,
        static_gamma_slope=gamma_slope,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["lightning_attn"]
