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

"""Pallas TPU registry entry for ragged per-channel-decay KDA (chunk rows in one kernel, short rows per token)."""

from __future__ import annotations

import jax
from jaxtyping import Array, Bool, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.kda_chunk._xla_impl import pairwise_sub_block
from ...._xla.ragged_kda._xla_impl import ragged_kda_xla, resolve_inverse, validate_options, validate_shapes
from ._pallas_impl import ragged_kda_pallas, supported


@kernel_registry.register("ragged_kda", Platform.PALLAS, Backend.TPU, priority=10)
def ragged_kda(
    query: Float[Array, "num_tokens num_heads head_dim"],
    key: Float[Array, "num_tokens num_heads head_dim"],
    value: Float[Array, "num_tokens num_heads v_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads head_dim"],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    query_start_loc: Int[Array, "num_rows_plus_1"],
    state_indices: Int[Array, "num_rows"] | None,
    has_initial_state: Bool[Array, "num_rows"] | None = None,
    read_offset: Int[Array, "num_rows"] | None = None,
    commit_lens: Int[Array, "num_rows"] | None = None,
    q_conv_weight: Float[Array, "key_dim d_conv"] | None = None,
    k_conv_weight: Float[Array, "key_dim d_conv"] | None = None,
    v_conv_weight: Float[Array, "value_dim d_conv"] | None = None,
    q_conv_state: Float[Array, "num_slots key_dim d_conv"] | None = None,
    k_conv_state: Float[Array, "num_slots key_dim d_conv"] | None = None,
    v_conv_state: Float[Array, "num_slots value_dim d_conv"] | None = None,
    *,
    short_row_max: int = 1,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    decay_lower_bound: float | None = None,
    precision: str = "highest",
    inverse: str = "auto",
    pairwise: str = "auto",
) -> tuple[Array, ...]:
    """Ragged (flat-token) KDA: fused causal conv, per-channel decay, per-token state capture (Pallas TPU).

    Same contract as the XLA reference (``ejkernel.kernels._xla.ragged_kda``). Chunk rows (more than
    ``short_row_max`` tokens) run in one Pallas kernel over their live chunks (state in VMEM, pool slots read at
    ``g + a`` and written at ``g`` in place). Short rows run token by token: with the conv arrays each token's
    recurrence is the Pallas decode kernel (the serving decode step's arithmetic), on post-conv inputs the reference's
    decode step (the round-4 ``ragged_kda`` arithmetic). Options the kernel does not lower (the dense pairwise form,
    i.e. no usable ``decay_lower_bound``; the exact ``"solve"`` inverse; head sizes that are not multiples of 128;
    chunk sizes outside the power-of-two range ``[8, 128]``) and non-TPU backends run the XLA reference instead.

    Args:
        query: Queries ``[T, H, K]``: raw (pre-conv) with the six conv arrays, post-conv without them.
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid already applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``), ideally fp32.
        recurrent_state: State pool ``[S, H, K, V]``; the recurrence always runs in fp32.
        query_start_loc: Cumulative row offsets ``[R + 1]``; tokens at or past ``query_start_loc[R]`` are padding.
        state_indices: Base slot of every row ``[R]`` (rows whose slots fall out of range are padding rows), or
            ``None`` (row ``r``'s base slot is ``r``). Rows must own disjoint slot groups.
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from zero windows and a zero state.
        read_offset: Optional ``[R]`` read offsets (zero when ``None``).
        commit_lens: Optional ``[R]`` commit lengths (``lens`` when ``None``).
        q_conv_weight: Depthwise causal conv taps ``[H * K, d_conv]``, or ``None`` (post-conv inputs).
        k_conv_weight: ``[H * K, d_conv]`` or ``None``.
        v_conv_weight: ``[H * V, d_conv]`` or ``None``.
        q_conv_state: Conv-window pool ``[S, H * K, d_conv]``, or ``None``.
        k_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        v_conv_state: ``[S, H * V, d_conv]`` or ``None``.
        short_row_max: ``W``: rows of at most ``W`` tokens take the short-row path (per-token capture).
        chunk_size: Chunk length of the multi-token path.
        use_qk_l2norm: L2-normalise queries and keys first.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound``; a tight enough bound selects
            the factored MXU pairwise decay, which the kernel requires.
        precision: ``"highest"`` or ``"default"`` for the non-inverse chunk dots.
        inverse: Chunk inverse: ``"auto"``, ``"masked"`` or ``"solve"`` (XLA reference only).
        pairwise: ``"auto"`` or ``"dense"`` (XLA reference only).

    Returns:
        With the conv arrays ``(out [T, H, V] fp32, q_conv_state, k_conv_state, v_conv_state, recurrent_state)``,
        without them ``(out, recurrent_state)``: zero output at padding tokens; the pools keep their dtypes, and slots
        no live row writes are unchanged.
    """
    arrays = (
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        query_start_loc,
        state_indices,
        has_initial_state,
        read_offset,
        commit_lens,
        q_conv_weight,
        k_conv_weight,
        v_conv_weight,
        q_conv_state,
        k_conv_state,
        v_conv_state,
    )
    validate_options(chunk_size, precision, inverse, pairwise, decay_lower_bound, short_row_max)
    validate_shapes(*arrays)
    cs = int(chunk_size)
    sub = pairwise_sub_block(decay_lower_bound, cs) if pairwise == "auto" else 0
    kwargs = dict(
        short_row_max=int(short_row_max),
        chunk_size=cs,
        use_qk_l2norm=use_qk_l2norm,
        decay_lower_bound=decay_lower_bound,
        precision=precision,
        inverse=inverse,
        pairwise=pairwise,
    )
    if jax.default_backend() != "tpu" or not supported(
        query.shape[-1], value.shape[-1], cs, sub, resolve_inverse(inverse, cs)
    ):
        return ragged_kda_xla(*arrays, **kwargs)
    return ragged_kda_pallas(*arrays, **kwargs)


__all__ = ("ragged_kda",)
