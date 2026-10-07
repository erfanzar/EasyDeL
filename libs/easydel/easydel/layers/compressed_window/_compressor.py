# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Softmax-gated KV compressor with an incremental (decode) step."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import jax
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from easydel.layers.linears import ColumnParallelLinear
from easydel.layers.norms import RMSNorm

__all__ = (
    "BlockGroupLayout",
    "GatedPoolCompressor",
    "GatedPoolCompressorConfig",
    "block_group_layout",
    "gather_group_members",
    "last_closed_overlap",
    "overlap_group_slots",
    "record_stream_trace",
    "refresh_group_state",
    "trace_compressor_stream",
)


class BlockGroupLayout(tp.NamedTuple):
    """How a block of new tokens per row maps onto ``ratio``-token compression groups (see :func:`block_group_layout`).

    Attributes:
        from_state: ``[B, G * ratio]`` bool; ``True`` where a group member was written by an earlier step (it is read
            from the open-group state) rather than by this block.
        state_index: ``[B, G * ratio]`` open-group state slot of each member (used where ``from_state``).
        block_index: ``[B, G * ratio]`` block column of each member (used elsewhere).
        closed: ``[B, G]`` bool; the groups that complete inside the block (a prefix per row).
        entry: ``[B, G]`` entry index each group occupies.
        refresh: ``[B, ratio]`` bool; open-group state slots this block overwrites.
        refresh_source: ``[B, ratio]`` block column whose value lands in each refreshed slot.
    """

    from_state: Array
    state_index: Array
    block_index: Array
    closed: Array
    entry: Array
    refresh: Array
    refresh_source: Array


def block_group_layout(
    positions: Int[Array, "batch block"],
    valid: Bool[Array, "batch block"],
    ratio: int,
) -> BlockGroupLayout:
    """Group layout of a block of tokens per row for an incremental ``ratio``-token compressor.

    Each row's valid cells are a prefix of the block at consecutive stream positions starting at ``positions[:, 0]``;
    position ``p`` lives in open-group state slot ``p % ratio``. The layout reproduces a per-token step run once per
    valid column: groups whose first members arrived earlier take them from the state, and afterwards each state slot
    holds the latest projection at a position congruent to it. Shared by every compressor stream that keeps an
    open-group buffer (DeepSeek-V4 single/two-series streams, :class:`GatedPoolCompressor`).

    Args:
        positions: Stream positions ``[B, blk]`` (garbage at invalid cells).
        valid: ``[B, blk]`` bool, a prefix per row.
        ratio: Tokens per group (static, ``>= 1``).

    Returns:
        The :class:`BlockGroupLayout` with ``G = ceil((ratio - 1 + blk) / ratio)`` groups.
    """
    rows, block = positions.shape
    positions = positions.astype(jnp.int32)
    n_valid = jnp.sum(valid, axis=1).astype(jnp.int32)
    pos0 = positions[:, 0]
    offset = pos0 % ratio  # members of the open group already in the state
    groups = -(-(ratio - 1 + block) // ratio)
    virtual = jnp.arange(groups * ratio, dtype=jnp.int32)[None, :]
    from_state = jnp.broadcast_to(virtual < offset[:, None], (rows, groups * ratio))
    state_index = jnp.broadcast_to(jnp.clip(virtual, 0, ratio - 1), (rows, groups * ratio))
    block_index = jnp.clip(virtual - offset[:, None], 0, block - 1)
    group = jnp.arange(groups, dtype=jnp.int32)[None, :]
    closed = ((group * ratio + ratio - 1) < (offset + n_valid)[:, None]) & (n_valid > 0)[:, None]
    entry = pos0[:, None] // ratio + group
    last = pos0 + n_valid - 1
    member = last[:, None] - ((last[:, None] - jnp.arange(ratio, dtype=jnp.int32)[None, :]) % ratio)
    refresh = (member >= pos0[:, None]) & (n_valid > 0)[:, None]
    refresh_source = jnp.clip(member - pos0[:, None], 0, block - 1)
    return BlockGroupLayout(from_state, state_index, block_index, closed, entry, refresh, refresh_source)


def gather_group_members(
    layout: BlockGroupLayout,
    state: Float[Array, "batch ratio dim"],
    fresh: Float[Array, "batch block dim"],
) -> Float[Array, "batch groups ratio dim"]:
    """Members of every group touched by the block: earlier members from ``state``, new ones from ``fresh``."""
    rows = fresh.shape[0]
    ratio = state.shape[1]
    old = jnp.take_along_axis(state, layout.state_index[..., None], axis=1)
    new = jnp.take_along_axis(fresh, layout.block_index[..., None], axis=1)
    return jnp.where(layout.from_state[..., None], old, new).reshape(rows, -1, ratio, fresh.shape[-1])


def overlap_group_slots(
    kv_members: Float[Array, "batch groups ratio feat"],
    gate_members: Float[Array, "batch groups ratio feat"],
    overlap_kv: Float[Array, "batch ratio dim"],
    overlap_gate: Float[Array, "batch ratio dim"],
    head_dim: int,
) -> tuple[Array, Array]:
    """Pooling slots of a two-series (Ca/Cb) stream for every group of a block.

    Each member carries two series in ``feat = 2 * head_dim`` features: Ca (``[..., :head_dim]``) feeds the *next*
    group's entry, Cb (``[..., head_dim:]``) the group's own. Group ``g`` pools ``[Ca of group g - 1 | Cb of group
    g]`` along the slot axis; group 0 takes its Ca half from the carried overlap state (the last group closed before
    the block). Same slot order as the per-token two-series step.

    Args:
        kv_members: Group members ``[B, G, ratio, 2 * head_dim]`` (see :func:`gather_group_members`).
        gate_members: Their gates, position bias already added, fp32, same shape.
        overlap_kv: Carried Ca kv ``[B, ratio, head_dim]``.
        overlap_gate: Carried biased fp32 Ca gate ``[B, ratio, head_dim]`` (``-inf`` before the first close).
        head_dim: Per-series width.

    Returns:
        ``(slot_kv, slot_gate)`` of shape ``[B, G, 2 * ratio, head_dim]``.
    """
    prev_kv = jnp.concatenate([overlap_kv.astype(kv_members.dtype)[:, None], kv_members[:, :-1, :, :head_dim]], axis=1)
    prev_gate = jnp.concatenate(
        [overlap_gate.astype(gate_members.dtype)[:, None], gate_members[:, :-1, :, :head_dim]], axis=1
    )
    slot_kv = jnp.concatenate([prev_kv, kv_members[..., head_dim:]], axis=2)
    slot_gate = jnp.concatenate([prev_gate, gate_members[..., head_dim:]], axis=2)
    return slot_kv, slot_gate


def last_closed_overlap(
    layout: BlockGroupLayout,
    kv_members: Float[Array, "batch groups ratio feat"],
    gate_members: Float[Array, "batch groups ratio feat"],
    overlap_kv: Float[Array, "batch ratio dim"],
    overlap_gate: Float[Array, "batch ratio dim"],
    head_dim: int,
) -> tuple[Array, Array]:
    """Two-series overlap state after a block: the Ca half of the last group that closed in it (unchanged if none).

    Args:
        layout: The block's :class:`BlockGroupLayout` (closed groups form a prefix per row).
        kv_members: Group members ``[B, G, ratio, 2 * head_dim]``.
        gate_members: Their biased fp32 gates, same shape.
        overlap_kv: Current carried Ca kv ``[B, ratio, head_dim]``.
        overlap_gate: Current carried biased fp32 Ca gate ``[B, ratio, head_dim]``.
        head_dim: Per-series width.

    Returns:
        ``(overlap_kv, overlap_gate)`` in the input dtypes.
    """
    n_closed = jnp.sum(layout.closed, axis=1).astype(jnp.int32)  # [B]
    last = jnp.clip(n_closed - 1, 0, kv_members.shape[1] - 1)[:, None, None, None]
    ca_kv = jnp.take_along_axis(kv_members, last, axis=1)[:, 0, :, :head_dim]
    ca_gate = jnp.take_along_axis(gate_members, last, axis=1)[:, 0, :, :head_dim]
    any_closed = (n_closed > 0)[:, None, None]
    return (
        jnp.where(any_closed, ca_kv.astype(overlap_kv.dtype), overlap_kv),
        jnp.where(any_closed, ca_gate.astype(overlap_gate.dtype), overlap_gate),
    )


def refresh_group_state(
    layout: BlockGroupLayout,
    state: Float[Array, "batch ratio dim"],
    fresh: Float[Array, "batch block dim"],
) -> Float[Array, "batch ratio dim"]:
    """Open-group state after the block, as the per-token writes would leave it."""
    src = jnp.take_along_axis(fresh, layout.refresh_source[..., None], axis=1)
    return jnp.where(layout.refresh[..., None], src, state)


def record_stream_trace(
    trace_kv: Array | None,
    trace_gate: Array | None,
    trace_overlap_kv: Array | None,
    trace_overlap_gate: Array | None,
    *,
    buffer_kv: Array,
    buffer_gate: Array,
    kv_blk: Array,
    gate_blk: Array,
    positions: Int[Array, "batch block"],
    valid: Bool[Array, "batch block"] | None,
    rate: int,
    head_dim: int,
    position_bias: Array | None = None,
) -> tuple[Array | None, Array | None, Array | None, Array | None]:
    """Record a block of one compressor stream's tokens for replay-free speculative rollback.

    Must see the stream's open-group buffers *before* the block is applied. Writes each valid cell's raw kv / gate row
    (in buffer dtype: exactly what the buffer stores) at trace slot ``position % L``; for two-series streams also the
    Ca half of every window that closes inside the block (kv in overlap dtype, gate with ``position_bias`` added in
    fp32: exactly the overlap state the step leaves) at slot ``window % L``. Read back by
    :func:`easydel.caching.compressed_window.commit_speculative_window`; a single-token decode step is a block of one.

    Args:
        trace_kv: Raw kv trace ``[B, L, feat]`` (``None``: nothing is recorded).
        trace_gate: Raw gate trace ``[B, L, feat]``.
        trace_overlap_kv: Closed-window Ca kv trace ``[B, L, rate, head_dim]`` (``None`` for single-series streams).
        trace_overlap_gate: Closed-window biased fp32 Ca gate trace, same shape.
        buffer_kv: Open-window kv buffer before the block ``[B, rate, feat]``.
        buffer_gate: Open-window gate buffer before the block ``[B, rate, feat]``.
        kv_blk: Raw kv projections of the block ``[B, blk, feat]``.
        gate_blk: Raw gate projections of the block ``[B, blk, feat]`` (position bias NOT applied).
        positions: Stream positions ``[B, blk]`` (garbage at invalid cells).
        valid: ``[B, blk]`` bool prefix per row (``None``: every cell is valid).
        rate: Compression rate.
        head_dim: Per-series width of a two-series stream.
        position_bias: Learned within-window bias ``[rate, feat]`` (required with overlap traces).

    Returns:
        The four updated traces.
    """
    if trace_kv is None:
        return trace_kv, trace_gate, trace_overlap_kv, trace_overlap_gate
    positions = positions.astype(jnp.int32)
    if valid is None:
        valid = jnp.ones(positions.shape, dtype=bool)
    rows = jnp.arange(positions.shape[0])[:, None]
    length = trace_kv.shape[1]
    fresh_kv = kv_blk.astype(buffer_kv.dtype)
    fresh_gate = gate_blk.astype(buffer_gate.dtype)
    slot = jnp.where(valid, positions % length, length)
    trace_kv = trace_kv.at[rows, slot].set(fresh_kv.astype(trace_kv.dtype), mode="drop")
    trace_gate = trace_gate.at[rows, slot].set(fresh_gate.astype(trace_gate.dtype), mode="drop")
    if trace_overlap_kv is not None:
        if position_bias is None:
            raise ValueError("record_stream_trace needs position_bias to record two-series overlaps.")
        layout = block_group_layout(positions, valid, rate)
        kv_members = gather_group_members(layout, buffer_kv, fresh_kv)  # [B, G, rate, feat]
        gate_members = gather_group_members(layout, buffer_gate, fresh_gate).astype(jnp.float32) + position_bias.astype(
            jnp.float32
        )
        windows = trace_overlap_kv.shape[1]
        window_slot = jnp.where(layout.closed, layout.entry % windows, windows)  # [B, G]
        trace_overlap_kv = trace_overlap_kv.at[rows, window_slot].set(
            kv_members[..., :head_dim].astype(trace_overlap_kv.dtype), mode="drop"
        )
        trace_overlap_gate = trace_overlap_gate.at[rows, window_slot].set(
            gate_members[..., :head_dim].astype(trace_overlap_gate.dtype), mode="drop"
        )
    return trace_kv, trace_gate, trace_overlap_kv, trace_overlap_gate


def trace_compressor_stream(
    view: tp.Any,
    prefix: str,
    *,
    kv: Array,
    gate: Array,
    positions: Int[Array, "batch block"],
    valid: Bool[Array, "batch block"] | None,
    rate: int,
    head_dim: int,
    position_bias: Array | None = None,
) -> tp.Any:
    """Apply :func:`record_stream_trace` to the ``<prefix>_*`` stream of a cache view (no-op without traces).

    Call with the view *before* the stream's step updates its buffers. Reads ``<prefix>_buffer_kv/gate`` and updates
    ``<prefix>_trace_kv``, ``<prefix>_trace_gate`` and (two-series) ``<prefix>_trace_overlap_kv/gate``.
    """
    if getattr(view, f"{prefix}_trace_kv", None) is None:
        return view
    names = (f"{prefix}_trace_kv", f"{prefix}_trace_gate", f"{prefix}_trace_overlap_kv", f"{prefix}_trace_overlap_gate")
    traces = record_stream_trace(
        *(getattr(view, name, None) for name in names),
        buffer_kv=getattr(view, f"{prefix}_buffer_kv"),
        buffer_gate=getattr(view, f"{prefix}_buffer_gate"),
        kv_blk=kv,
        gate_blk=gate,
        positions=positions,
        valid=valid,
        rate=rate,
        head_dim=head_dim,
        position_bias=position_bias,
    )
    return view.replace(**{name: value for name, value in zip(names, traces, strict=True) if value is not None})


@dataclass(frozen=True)
class GatedPoolCompressorConfig:
    """Scalar settings of :class:`GatedPoolCompressor`, independent of model configs.

    Args:
        hidden_size: Input hidden width.
        head_dim: Width of the compressed latent.
        ratio: Source tokens pooled into one entry (``1`` is a plain projection).
        rms_norm_eps: Epsilon of the latent RMSNorm.
        initializer_range: Standard deviation of the projection initializers.
    """

    hidden_size: int
    head_dim: int
    ratio: int
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {self.ratio}.")


class GatedPoolCompressor(spx.Module):
    """Pools ``ratio`` consecutive tokens into one pre-RoPE KV latent.

    Ratio 1 is ``kv_norm(kv_proj(x))``. Ratio > 1 runs in fp32: ``sum(kv * softmax(gate, over the group))``, then
    ``kv_norm``. Tokens of an incomplete trailing group produce no latent in the stateless forward; the incremental
    :meth:`decode_step` keeps the open group's fp32 projections in caller-owned state.

    Parameter leaves: ``kv_proj.weight``, ``gate_proj.weight`` (ratio > 1 only) and ``kv_norm.weight``.
    """

    def __init__(
        self,
        config: GatedPoolCompressorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the compressor.

        Args:
            config: Scalar compressor settings.
            dtype: Activation dtype (the projections run in fp32 when ``ratio > 1``).
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators (consumed in kv / gate / norm order).
        """
        self.ratio = config.ratio
        pool_dtype = jnp.float32 if self.ratio > 1 else dtype
        kwargs = dict(
            use_bias=False,
            dtype=pool_dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.kv_proj = ColumnParallelLinear(config.hidden_size, config.head_dim, **kwargs)
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.head_dim, **kwargs) if self.ratio > 1 else None
        self.kv_norm = RMSNorm(
            dim=config.head_dim, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def project(self, hidden_states: Float[Array, "batch seq hidden"]) -> tuple[Array, Array]:
        """fp32 kv and gate-score projections ``[B, S, head_dim]`` (ratio > 1 only)."""
        x = hidden_states.astype(jnp.float32)
        return self.kv_proj(x).astype(jnp.float32), self.gate_proj(x).astype(jnp.float32)

    def pool(self, kv: Array, score: Array, dtype: jnp.dtype) -> Array:
        """Softmax-gated pooling over the group axis (``-2``) of fp32 ``[..., ratio, D]``, then ``kv_norm``."""
        pooled = jnp.sum(kv * jax.nn.softmax(score, axis=-2), axis=-2)
        return self.kv_norm(pooled.astype(dtype))

    def forward(self, hidden_states: Float[Array, "batch seq hidden"]) -> Array | None:
        """Return the latent ``[B, S // ratio, head_dim]`` (pre-RoPE), or ``None`` if no group is complete."""
        batch, seq, _ = hidden_states.shape
        ratio = self.ratio
        groups = seq // ratio
        if groups == 0:
            return None
        if ratio == 1:
            return self.kv_norm(self.kv_proj(hidden_states))
        kv, score = self.project(hidden_states[:, : groups * ratio])
        return self.pool(
            kv.reshape(batch, groups, ratio, -1), score.reshape(batch, groups, ratio, -1), hidden_states.dtype
        )

    def decode_step(
        self,
        hidden_states: Float[Array, "batch 1 hidden"],
        kv_state: Float[Array, "batch ratio dim"] | None,
        score_state: Float[Array, "batch ratio dim"] | None,
        position: Int[Array, " batch"],
        projections: tuple[Array, Array] | None = None,
    ) -> tuple[Array, Array | None, Array | None, Bool[Array, " batch"], Int[Array, " batch"]]:
        """Advance the compressor by one token per row.

        For ``ratio > 1`` the token's fp32 projections are written into slot ``position % ratio`` of the open-group
        state and the (possibly still partial) group is pooled; the result is only meaningful as an entry on rows
        whose group just closed. Ratio 1 is stateless: every token closes its own group.

        Args:
            hidden_states: New token hidden states ``[B, 1, hidden]``.
            kv_state: Open-group fp32 kv state ``[B, ratio, head_dim]`` (``None`` for ratio 1).
            score_state: Open-group fp32 gate-score state ``[B, ratio, head_dim]`` (``None`` for ratio 1).
            position: Stream position of the new token per row ``[B]``.
            projections: Optional precomputed :meth:`project` output ``(kv [B, 1, D], score [B, 1, D])`` (ratio > 1);
                ``None`` projects ``hidden_states`` here.

        Returns:
            ``(latent [B, 1, head_dim], kv_state, score_state, closed [B], entry [B])`` where ``closed`` marks rows
            whose group completed with this token and ``entry`` is the index of the entry that group occupies.
        """
        batch = hidden_states.shape[0]
        if self.ratio == 1:
            latent = self(hidden_states)  # [B, 1, D]
            return latent, kv_state, score_state, jnp.ones((batch,), dtype=bool), position
        rows = jnp.arange(batch)
        kv_t, score_t = self.project(hidden_states) if projections is None else projections
        slot = position % self.ratio
        kv_state = kv_state.at[rows, slot].set(kv_t[:, 0])
        score_state = score_state.at[rows, slot].set(score_t[:, 0])
        latent = self.pool(kv_state, score_state, hidden_states.dtype)[:, None]  # [B, 1, D]
        closed = (position + 1) % self.ratio == 0
        return latent, kv_state, score_state, closed, position // self.ratio

    def extend_step(
        self,
        hidden_states: Float[Array, "batch block hidden"],
        kv_state: Float[Array, "batch ratio dim"] | None,
        score_state: Float[Array, "batch ratio dim"] | None,
        positions: Int[Array, "batch block"],
        valid: Bool[Array, "batch block"],
        projections: tuple[Array, Array] | None = None,
    ) -> tuple[Array, Array | None, Array | None, Bool[Array, "batch groups"], Int[Array, "batch groups"]]:
        """Advance the compressor by a block of tokens per row in one pass (batched packed prefill).

        Equivalent to calling :meth:`decode_step` once per valid column. Each row's valid cells form a prefix of the
        block at consecutive positions starting at ``positions[:, 0]``. A group whose first members arrived in earlier
        steps takes them from the open-group state; afterwards the state holds the latest projection per in-group
        slot, exactly as the per-token writes would leave it.

        Args:
            hidden_states: Block hidden states ``[B, blk, hidden]``.
            kv_state: Open-group fp32 kv state ``[B, ratio, head_dim]`` (``None`` for ratio 1).
            score_state: Open-group fp32 gate-score state ``[B, ratio, head_dim]`` (``None`` for ratio 1).
            positions: Stream positions ``[B, blk]`` (garbage at invalid cells).
            valid: ``[B, blk]`` bool, a prefix per row.
            projections: Optional precomputed :meth:`project` output ``(kv [B, blk, D], score [B, blk, D])``
                (ratio > 1); ``None`` projects ``hidden_states`` here.

        Returns:
            ``(latent [B, G, head_dim], kv_state, score_state, closed [B, G], entry [B, G])``: ``closed`` marks the
            groups that complete inside the block and ``entry`` the entry index each group occupies. For ratio 1,
            ``G == blk`` and every valid token closes its own group.
        """
        ratio = self.ratio
        dtype = hidden_states.dtype
        positions = positions.astype(jnp.int32)
        if ratio == 1:
            return self(hidden_states), kv_state, score_state, valid, positions
        kv_t, score_t = self.project(hidden_states) if projections is None else projections  # fp32 [B, blk, D]
        layout = block_group_layout(positions, valid, ratio)
        latent = self.pool(
            gather_group_members(layout, kv_state, kv_t), gather_group_members(layout, score_state, score_t), dtype
        )  # [B, G, D]
        kv_state = refresh_group_state(layout, kv_state, kv_t)
        score_state = refresh_group_state(layout, score_state, score_t)
        return latent, kv_state, score_state, layout.closed, layout.entry
