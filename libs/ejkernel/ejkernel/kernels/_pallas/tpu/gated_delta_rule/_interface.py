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

"""TPU Pallas interface for Gated Delta Rule (GDR)."""

from __future__ import annotations

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.gated_delta_rule._xla_impl_fwd import (
    _recurrent_gdr_fwd,
)
from ._pallas_impl_fwd import _chunk_gdr_fwd, _chunk_gdr_grouped_fwd, _single_step_gdr_fwd


@kernel_registry.register("gated_delta_rule", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def gated_delta_rule(
    query: Float[Array, "batch seq_len num_qk_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_qk_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_value_heads v_head_dim"],
    beta: Float[Array, "batch seq_len num_value_heads"],
    decay: Float[Array, "batch seq_len num_value_heads"] | None = None,
    *,
    chunk_size: int = 256,
    initial_state: Float[Array, "batch num_value_heads qk_head_dim v_head_dim"] | None = None,
    use_qk_l2norm: bool = True,
    use_chunked: bool = True,
    use_input_dtype_phase1_outputs: bool = False,
    use_input_dtype_state: bool = False,
    seg_ids: Int[Array, "batch seq_len"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_value_heads v_head_dim"],
    Float[Array, "batch num_value_heads qk_head_dim v_head_dim"],
]:
    """Gated Delta Rule linear attention using TPU Pallas kernels.

    Inputs are accepted in the public ``(batch, seq_len, num_heads, head_dim)`` layout and
    transposed internally to the heads-first ``(batch, num_heads, seq_len, head_dim)`` layout
    expected by the underlying implementations; the output is transposed back before being
    returned. Dispatch selects one of four implementations based on ``seg_ids``, the input
    shape, and the ``use_chunked`` flag, in this priority order:

    * **Segment-packed recurrent** (``seg_ids is not None``): delegates to the XLA chunked
      recurrent implementation with segment ids so that the recurrent state is reset at
      document boundaries (sequence packing). Takes precedence over all other branches.
    * **Single-step** (``seq_len == 1`` with ``initial_state`` provided): a Pallas
      single-step kernel for autoregressive decoding.
    * **Chunked** (``use_chunked=True``, default): a vectorized Pallas kernel that uses
      a parallel exact strict-lower solve mapped to MXU-heavy matmuls, with the scan body
      keeping data in VMEM.
    * **Recurrent fallback** (``use_chunked=False``): falls back to the XLA step-by-step
      recurrent implementation.

    All paths optionally apply L2 normalisation to queries and keys when ``use_qk_l2norm`` is
    ``True`` (the default) and scale queries by ``1 / sqrt(qk_head_dim)``.

    Args:
        query: Query tensor of shape ``(batch, seq_len, num_qk_heads, qk_head_dim)``.
        key: Key tensor of shape ``(batch, seq_len, num_qk_heads, qk_head_dim)``.
        value: Value tensor of shape ``(batch, seq_len, num_value_heads, v_head_dim)``.
        beta: Per-token learning-rate (gate) of shape ``(batch, seq_len, num_value_heads)``.
            Controls how strongly each token's delta update is applied to the
            recurrent state.
        decay: Optional log-space decay gate of shape ``(batch, seq_len, num_heads)``.
            When provided, the recurrent state is multiplied by ``exp(decay)``
            at each step before the delta update. ``None`` means no decay
            (equivalent to all-zeros).
        chunk_size: Number of tokens per chunk for the blocked-recurrent kernel.
            Must be a positive integer; the sequence length is padded up to a
            multiple of ``chunk_size`` internally.
        initial_state: Optional initial recurrent state of shape
            ``(batch, num_heads, qk_head_dim, v_head_dim)``. Defaults to zeros.
        use_qk_l2norm: If ``True``, L2-normalise queries and keys before
            computing attention.
        use_chunked: If ``True`` (default), use the chunked Pallas kernel.
            Set to ``False`` to use the XLA recurrent fallback. Ignored when
            ``seg_ids`` is provided or when the single-step path is taken.
        use_input_dtype_phase1_outputs: If ``True``, the forward-only chunked
            Pallas path writes its large phase-1 handoff tensors in the input
            dtype instead of fp32. This is ignored by the segmented, single-step,
            recurrent fallback, and custom-VJP backward-residual paths.
        use_input_dtype_state: If ``True``, the forward-only chunked Pallas
            path carries the recurrent state in the input dtype instead of fp32.
            Leave disabled for callers that consume the returned state as cache.
        seg_ids: Optional per-token segment (document) ids of shape
            ``(batch, seq_len)``. When provided, dispatch is forced to the XLA
            recurrent path, which uses these ids to reset the recurrent state at
            segment boundaries so that packed sequences do not leak state across
            documents. ``None`` (default) treats the whole sequence as one segment.

    Returns:
        A tuple ``(output, final_state)`` where:

        * ``output`` has shape ``(batch, seq_len, num_value_heads, v_head_dim)`` —
          the attention output for every position.
        * ``final_state`` has shape ``(batch, num_value_heads, qk_head_dim, v_head_dim)``
          — the recurrent state after processing the full sequence, which can
          be fed as ``initial_state`` to a subsequent call.
    """

    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    b = beta.transpose(0, 2, 1)
    d = decay.transpose(0, 2, 1) if decay is not None else None
    grouped_heads = q.shape[1] != v.shape[1]
    if grouped_heads and v.shape[1] % q.shape[1] != 0:
        raise ValueError(
            f"grouped GDR requires value heads ({v.shape[1]}) to be a multiple of query heads ({q.shape[1]})"
        )

    if seg_ids is not None:
        if grouped_heads:
            expand_ratio = v.shape[1] // q.shape[1]
            q = jnp.repeat(q, expand_ratio, axis=1)
            k = jnp.repeat(k, expand_ratio, axis=1)
        out, final_state = _recurrent_gdr_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
            chunk_size=chunk_size,
            seg_ids=seg_ids,
        )
    elif query.shape[1] == 1 and initial_state is not None:
        if grouped_heads:
            expand_ratio = v.shape[1] // q.shape[1]
            q = jnp.repeat(q, expand_ratio, axis=1)
            k = jnp.repeat(k, expand_ratio, axis=1)
        out, final_state = _single_step_gdr_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            recurrent_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
        )
    elif use_chunked and grouped_heads:
        out, final_state = _chunk_gdr_grouped_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            chunk_size=chunk_size,
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
            use_input_dtype_phase1_outputs=use_input_dtype_phase1_outputs,
            use_input_dtype_state=use_input_dtype_state,
        )
    elif use_chunked:
        out, final_state = _chunk_gdr_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            chunk_size=chunk_size,
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
            use_input_dtype_phase1_outputs=use_input_dtype_phase1_outputs,
            use_input_dtype_state=use_input_dtype_state,
        )
    else:
        if grouped_heads:
            expand_ratio = v.shape[1] // q.shape[1]
            q = jnp.repeat(q, expand_ratio, axis=1)
            k = jnp.repeat(k, expand_ratio, axis=1)
        out, final_state = _recurrent_gdr_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
        )

    return out.transpose(0, 2, 1, 3), final_state
