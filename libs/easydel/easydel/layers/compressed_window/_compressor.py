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
from easydel.operations.kernels.gated_pool_compress import (
    BlockGroupLayout,
    block_group_layout,
    gather_group_members,
    last_closed_overlap,
    overlap_group_slots,
    refresh_group_state,
)

__all__ = (
    "BlockGroupLayout",
    "GatedPoolCompressor",
    "GatedPoolCompressorConfig",
    "advance_compressor_stream",
    "block_group_layout",
    "compress_windows",
    "gather_group_members",
    "last_closed_overlap",
    "overlap_group_slots",
    "record_stream_trace",
    "refresh_group_state",
    "trace_compressor_stream",
)


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


def _gated_pool_op():
    """The registered ``gated_pool_compress`` operation (imported lazily: operations import layers)."""
    from easydel.operations.kernels.gated_pool_compress import gated_pool_compress_op

    return gated_pool_compress_op()


def compress_windows(
    kv: Float[Array, "batch length feat"],
    gate: Float[Array, "batch length feat"],
    rate: int,
    *,
    position_bias: Array | None = None,
    series_dim: int | None = None,
):
    """Pool the complete ``rate``-token groups of a prefill through the ``gated_pool_compress`` operation.

    Args:
        kv: Raw kv projections ``[B, G * rate, F]``.
        gate: Raw gate projections, same shape (position bias not applied).
        rate: Tokens per group.
        position_bias: Optional learned within-group bias ``[rate, F]``.
        series_dim: Per-series width ``D`` of a two-series (Ca/Cb) stream (``F == 2 * D``); ``None`` otherwise.

    Returns:
        ``GatedPoolCompressOutput`` with ``pooled [B, G, D]`` (pre-norm) and, for two-series streams, the last group's
        Ca ``overlap_kv`` / biased fp32 ``overlap_gate`` ``[B, rate, D]``.
    """
    return _gated_pool_op()("windows", kv, gate, rate, position_bias=position_bias, series_dim=series_dim)


def advance_compressor_stream(
    *,
    kv: Array,
    gate: Array,
    buffer_kv: Array,
    buffer_gate: Array,
    entries: Array,
    rate: int,
    finalize: tp.Callable[[Array, Array], Array],
    position: Int[Array, " batch"] | None = None,
    positions: Int[Array, "batch block"] | None = None,
    valid: Array | None = None,
    position_bias: Array | None = None,
    overlap_kv: Array | None = None,
    overlap_gate: Array | None = None,
) -> tuple[Array, Array, Array, Array | None, Array | None]:
    """Advance a cached compressor stream and write every entry that closes.

    One token per row when ``position`` is given (decode), else a block per row at ``positions`` (batched packed
    prefill; equal to the per-token step run once per valid column). Pooling and the open-group / overlap state machine
    run in the ``gated_pool_compress`` operation; ``finalize(pooled [B, G, D], entry [B, G]) -> [B, G, D]`` applies the
    architecture's entry norm and RoPE. Rows whose group did not close leave ``entries`` untouched (their writes are
    steered out of bounds and dropped, so the entry buffer is never read back).

    Args:
        kv: Raw kv projections ``[B, 1, F]`` (decode) or ``[B, blk, F]`` (extend).
        gate: Raw gate projections, same shape (position bias not applied).
        buffer_kv: Open-group kv buffer ``[B, rate, F]``.
        buffer_gate: Open-group raw gate buffer ``[B, rate, F]``.
        entries: Padded entry buffer ``[B, n_slots, D]``.
        rate: Tokens per group.
        finalize: Entry norm + RoPE.
        position: Decode: stream position of the token per row ``[B]``.
        positions: Extend: stream positions ``[B, blk]``.
        valid: Decode ``[B]`` (optional) or extend ``[B, blk]`` (prefix per row) cell mask.
        position_bias: Optional learned within-group bias ``[rate, F]``.
        overlap_kv: Two-series only: carried Ca kv ``[B, rate, D]``.
        overlap_gate: Two-series only: carried biased fp32 Ca gate ``[B, rate, D]``.

    Returns:
        ``(entries, buffer_kv, buffer_gate, overlap_kv, overlap_gate)`` (overlaps ``None`` for single-series streams).
    """
    decode = positions is None
    out = _gated_pool_op()(
        "decode" if decode else "extend",
        kv,
        gate,
        rate,
        position_bias=position_bias,
        buffer_kv=buffer_kv,
        buffer_gate=buffer_gate,
        position=position,
        positions=positions,
        valid=valid,
        overlap_kv=overlap_kv,
        overlap_gate=overlap_gate,
    )
    pooled, closed, entry = out.pooled, out.closed, out.entry
    if decode:
        pooled, closed, entry = pooled[:, None], closed[:, None], entry[:, None]
    candidate = finalize(pooled, entry)
    n_slots = entries.shape[1]
    target = jnp.where(closed, jnp.clip(entry, 0, n_slots - 1), n_slots)
    rows = jnp.arange(entries.shape[0])[:, None]
    entries = entries.at[rows, target].set(candidate.astype(entries.dtype), mode="drop")
    return entries, out.buffer_kv, out.buffer_gate, out.overlap_kv, out.overlap_gate


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

    def forward(self, hidden_states: Float[Array, "batch seq hidden"]) -> Array | None:
        """Return the latent ``[B, S // ratio, head_dim]`` (pre-RoPE), or ``None`` if no group is complete."""
        batch, seq, _ = hidden_states.shape
        ratio = self.ratio
        groups = seq // ratio
        if groups == 0:
            return None
        if ratio == 1:
            return self.kv_norm(self.kv_proj(hidden_states))
        del batch
        kv, score = self.project(hidden_states[:, : groups * ratio])
        return self.kv_norm(compress_windows(kv, score, ratio).pooled.astype(hidden_states.dtype))

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
        kv_t, score_t = self.project(hidden_states) if projections is None else projections
        out = _gated_pool_op()(
            "decode", kv_t, score_t, self.ratio, buffer_kv=kv_state, buffer_gate=score_state, position=position
        )
        latent = self.kv_norm(out.pooled.astype(hidden_states.dtype))[:, None]  # [B, 1, D]
        return latent, out.buffer_kv, out.buffer_gate, out.closed, out.entry

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
        out = _gated_pool_op()(
            "extend", kv_t, score_t, ratio, buffer_kv=kv_state, buffer_gate=score_state, positions=positions, valid=valid
        )
        latent = self.kv_norm(out.pooled.astype(dtype))  # [B, G, D]
        return latent, out.buffer_kv, out.buffer_gate, out.closed, out.entry
