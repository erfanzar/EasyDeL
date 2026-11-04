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

"""XLA registry entry (priority 0 fallback and reference) for ragged per-channel-decay KDA."""

from __future__ import annotations

from jaxtyping import Array, Bool, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import ragged_kda_xla, validate_options, validate_shapes


@kernel_registry.register("ragged_kda", Platform.XLA, Backend.ANY, priority=0)
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
    """Ragged (flat-token) KDA: fused causal conv, per-channel decay, per-token state capture (XLA reference).

    Row ``r`` owns tokens ``[query_start_loc[r], query_start_loc[r + 1])`` and the slot group starting at its base slot
    ``g = state_indices[r]``. It reads its conv windows and state from slot ``g + read_offset[r]``; rows of at most
    ``short_row_max`` tokens write the windows and state after token ``commit_lens[r] - 1 + j`` into slot ``g + j``
    (``j = 0 .. lens - commit``), longer rows their final windows and state into slot ``g``. See :mod:`._xla_impl`.

    Args:
        query: Queries ``[T, H, K]``: raw (pre-conv) with the six conv arrays, post-conv without them.
        key: Keys ``[T, H, K]``.
        value: Values ``[T, H, V]``.
        beta: Write strengths ``[T, H]`` (sigmoid already applied).
        decay: Per-channel log-decay ``[T, H, K]`` (``<= 0``), ideally fp32.
        recurrent_state: State pool ``[S, H, K, V]`` (the recurrence always runs in fp32).
        query_start_loc: Cumulative row offsets ``[R + 1]``; tokens at or past ``query_start_loc[R]`` are padding.
        state_indices: Base slot of every row ``[R]`` (rows whose slots fall out of range are padding rows), or
            ``None`` (row ``r``'s base slot is ``r``).
        has_initial_state: Optional ``[R]`` flags; ``False`` rows start from zero windows and a zero state.
        read_offset: Optional ``[R]`` read offsets (zero when ``None``).
        commit_lens: Optional ``[R]`` commit lengths (``lens`` when ``None``).
        q_conv_weight: Depthwise causal conv taps ``[H * K, d_conv]`` (the last tap is the current token), or
            ``None`` (post-conv inputs).
        k_conv_weight: ``[H * K, d_conv]`` or ``None``.
        v_conv_weight: ``[H * V, d_conv]`` or ``None``.
        q_conv_state: Conv-window pool ``[S, H * K, d_conv]`` (oldest tap first), or ``None``.
        k_conv_state: ``[S, H * K, d_conv]`` or ``None``.
        v_conv_state: ``[S, H * V, d_conv]`` or ``None``.
        short_row_max: ``W``: rows of at most ``W`` tokens take the short-row path (per-token capture).
        chunk_size: Chunk length of the multi-token path.
        use_qk_l2norm: L2-normalise queries and keys first.
        decay_lower_bound: Promise that every log-decay is ``>= decay_lower_bound`` (e.g. a safe-gate bound); with
            ``pairwise="auto"`` a tight enough bound selects the factored MXU pairwise decay. ``None`` keeps the
            dense ``[cs, cs, K]`` form.
        precision: ``"highest"`` or ``"default"`` for the non-inverse chunk dots.
        inverse: Chunk inverse: ``"auto"``, ``"masked"`` (block-recursive, matmul only) or ``"solve"`` (exact
            triangular solve).
        pairwise: ``"auto"`` or ``"dense"``.

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
    return ragged_kda_xla(
        *arrays,
        short_row_max=int(short_row_max),
        chunk_size=int(chunk_size),
        use_qk_l2norm=use_qk_l2norm,
        decay_lower_bound=decay_lower_bound,
        precision=precision,
        inverse=inverse,
        pairwise=pairwise,
    )


__all__ = ("ragged_kda",)
