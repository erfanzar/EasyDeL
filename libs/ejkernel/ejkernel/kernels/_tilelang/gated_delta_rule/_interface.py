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

"""TileLang interface for the Gated Delta Rule (GDR) recurrent kernel.

Registers ``gated_delta_rule`` under ``Platform.TILELANG / Backend.GPU`` in the
ejkernel registry.  The underlying implementation is
:func:`ejkernel.kernels._tilelang.gated_delta_rule._impl.delta_rule_tilelang`.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._impl import delta_rule_tilelang


@kernel_registry.register("gated_delta_rule", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def gated_delta_rule(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_heads v_head_dim"],
    beta: Float[Array, "batch seq_len num_heads"],
    decay: Float[Array, "batch seq_len num_heads"] | None = None,
    *,
    chunk_size: int = 256,
    initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
    use_qk_l2norm: bool = True,
    use_chunked: bool = True,
    seg_ids: Int[Array, "batch seq_len"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "batch num_heads qk_head_dim v_head_dim"],
]:
    """Run the Gated Delta Rule recurrent scan on GPU via TileLang.

    Implements the per-timestep update::

        h_t = decay_t * h_{t-1} + k_t * (beta_t * (v_t - k_t^T h_{t-1}))
        o_t = h_t^T q_t

    where ``decay_t = exp(decay[t])`` when ``decay`` is provided, otherwise 1.

    Both forward and backward passes are executed by native TileLang kernels.
    Accumulation is done in float32 regardless of the input dtype; inputs and
    outputs are cast to the query dtype.

    Args:
        query: Query tensor of shape ``[batch, seq_len, num_heads, qk_head_dim]``.
        key: Key tensor of shape ``[batch, seq_len, num_heads, qk_head_dim]``.
        value: Value tensor of shape ``[batch, seq_len, num_heads, v_head_dim]``.
        beta: Per-timestep update gate of shape ``[batch, seq_len, num_heads]``.
            Controls how much of the new (key, value) outer-product is written
            into the hidden state.
        decay: Optional per-timestep log-decay of shape
            ``[batch, seq_len, num_heads]``.  The state is multiplied by
            ``exp(decay[t])`` before the update.  Pass ``None`` to disable
            decay (equivalent to ``decay = 0``).
        chunk_size: Accepted for API compatibility with the XLA backend; has
            no effect on the TileLang kernel — the recurrent scan always
            processes one timestep at a time.
        initial_state: Optional initial hidden state of shape
            ``[batch, num_heads, qk_head_dim, v_head_dim]`` in float32.
            Defaults to a zero state computed on device.
        use_qk_l2norm: When ``True`` (default), L2-normalise query and key
            vectors before computing the delta update.  The normalisation
            is folded into the kernel and is differentiable.
        use_chunked: Accepted for API compatibility with the XLA backend;
            has no effect on the TileLang kernel.
        seg_ids: Accepted for signature parity with the XLA backend (sequence
            packing). The TileLang kernel does not implement segment-aware
            recurrence, so a non-``None`` value raises ``NotImplementedError``;
            packed training routes through the XLA backend instead.

    Returns:
        A 2-tuple ``(output, final_state)`` where:

        - ``output``: float tensor of shape
          ``[batch, seq_len, num_heads, v_head_dim]``.
        - ``final_state``: float32 tensor of shape
          ``[batch, num_heads, qk_head_dim, v_head_dim]`` holding the
          recurrent state after the last timestep.

    Raises:
        EjkernelRuntimeError: If ``chunk_size <= 0`` or ``use_chunked`` is not
            a bool.
        NotImplementedError: If ``seg_ids`` is provided (sequence packing is
            unsupported on the TileLang backend; use the XLA backend instead).
    """
    # ``seg_ids`` (sequence packing) is accepted for signature parity with the XLA backend
    # (the kernel registry validates matching signatures across platforms). This TileLang path
    # does not implement segment-aware recurrence; packed training routes through the XLA
    # chunked path instead, so seg_ids should never be non-None here.
    if seg_ids is not None:
        raise NotImplementedError(
            "TileLang GDR does not support sequence packing (seg_ids); the packed path uses the XLA backend."
        )
    if chunk_size <= 0:
        raise EjkernelRuntimeError("tile-lang gated_delta_rule requires chunk_size > 0.")
    if not isinstance(use_chunked, bool):
        raise EjkernelRuntimeError("tile-lang gated_delta_rule requires use_chunked to be a bool.")
    return delta_rule_tilelang(
        query,
        key,
        value,
        beta,
        decay,
        initial_state=initial_state,
        softmax_scale=None,
        use_qk_l2norm=use_qk_l2norm,
    )


__all__ = ["gated_delta_rule"]
