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

"""Softmax-gated KV pooling for compressed-window attention, as a registered operation.

A compressor stream pools every ``rate`` consecutive token projections into one entry:
``sum(kv * softmax(gate + position_bias, over the group))``. Two-series streams (DeepSeek-V4 CSA and its indexer)
carry ``2 * D`` features per token: the first half (Ca) feeds the *next* group's entry, the second half (Cb) the
group's own, so entry ``g`` pools ``[Ca of group g - 1 | Cb of group g]`` over ``2 * rate`` slots.

The operation owns the pooling arithmetic and the incremental state machine (open-group buffers, the two-series
overlap carry, which groups close and which entry each occupies) in three modes:

- ``"windows"``: complete groups of a prefill, stateless.
- ``"decode"``: one token per row against the open-group buffers.
- ``"extend"``: a block of tokens per row in one pass (batched packed prefill); equal to ``"decode"`` per column.

Pooling runs in ``compute_dtype`` (default: the dtype of the projections) with fp32 softmax. Projections, the entry
norm, RoPE and cache writes stay with the calling layer (see :mod:`easydel.layers.compressed_window`), which reaches
this operation through :func:`gated_pool_compress_op`.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from jaxtyping import Array

from .._operation_impl import OperationImpl, OperationOutput, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder

__all__ = (
    "BlockGroupLayout",
    "GatedPoolCompressOp",
    "GatedPoolCompressOutput",
    "block_group_layout",
    "gated_pool_compress_op",
    "gather_group_members",
    "last_closed_overlap",
    "overlap_group_slots",
    "refresh_group_state",
)


class BlockGroupLayout(tp.NamedTuple):
    """Fixed-shape mapping of new tokens to compression groups and refreshed state slots.

    Attributes:
        from_state: Members sourced from the open-group buffer, ``[B, G * ratio]``.
        state_index: Open-group slot for each member.
        block_index: Block column for each new member.
        closed: Groups completed by the block, ``[B, G]``.
        entry: Stream entry indices, ``[B, G]``.
        refresh: Buffer slots overwritten by this block, ``[B, ratio]``.
        refresh_source: Block column whose value lands in each refreshed slot.
    """

    from_state: Array
    state_index: Array
    block_index: Array
    closed: Array
    entry: Array
    refresh: Array
    refresh_source: Array


def block_group_layout(positions: Array, valid: Array, ratio: int) -> BlockGroupLayout:
    """Map consecutive stream positions (valid prefixes per row) to ``ratio``-token groups.

    Args:
        positions: Stream positions ``[B, block]``; invalid cells are ignored.
        valid: Boolean prefix mask of the same shape.
        ratio: Positive, static compression group size.

    Returns:
        Layout with ``ceil((ratio - 1 + block) / ratio)`` group slots per row.
    """
    rows, block = positions.shape
    positions = positions.astype(jnp.int32)
    n_valid = jnp.sum(valid, axis=1).astype(jnp.int32)
    pos0 = positions[:, 0]
    offset = pos0 % ratio
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


def gather_group_members(layout: BlockGroupLayout, state: Array, fresh: Array) -> Array:
    """Gather ``[B, groups, ratio, features]`` members from earlier state and new block tokens."""
    rows, ratio = fresh.shape[0], state.shape[1]
    old = jnp.take_along_axis(state, layout.state_index[..., None], axis=1)
    new = jnp.take_along_axis(fresh, layout.block_index[..., None], axis=1)
    return jnp.where(layout.from_state[..., None], old, new).reshape(rows, -1, ratio, fresh.shape[-1])


def overlap_group_slots(
    kv_members: Array, gate_members: Array, overlap_kv: Array, overlap_gate: Array, head_dim: int
) -> tuple[Array, Array]:
    """Return two-series pool slots ``[Ca of previous group | Cb of current group]``."""
    prev_kv = jnp.concatenate([overlap_kv.astype(kv_members.dtype)[:, None], kv_members[:, :-1, :, :head_dim]], axis=1)
    prev_gate = jnp.concatenate(
        [overlap_gate.astype(gate_members.dtype)[:, None], gate_members[:, :-1, :, :head_dim]], axis=1
    )
    return (
        jnp.concatenate([prev_kv, kv_members[..., head_dim:]], axis=2),
        jnp.concatenate([prev_gate, gate_members[..., head_dim:]], axis=2),
    )


def last_closed_overlap(
    layout: BlockGroupLayout,
    kv_members: Array,
    gate_members: Array,
    overlap_kv: Array,
    overlap_gate: Array,
    head_dim: int,
) -> tuple[Array, Array]:
    """Carry the last closed group's Ca half forward, leaving rows with no closed group unchanged."""
    n_closed = jnp.sum(layout.closed, axis=1).astype(jnp.int32)
    last = jnp.clip(n_closed - 1, 0, kv_members.shape[1] - 1)[:, None, None, None]
    ca_kv = jnp.take_along_axis(kv_members, last, axis=1)[:, 0, :, :head_dim]
    ca_gate = jnp.take_along_axis(gate_members, last, axis=1)[:, 0, :, :head_dim]
    any_closed = (n_closed > 0)[:, None, None]
    return (
        jnp.where(any_closed, ca_kv.astype(overlap_kv.dtype), overlap_kv),
        jnp.where(any_closed, ca_gate.astype(overlap_gate.dtype), overlap_gate),
    )


def refresh_group_state(layout: BlockGroupLayout, state: Array, fresh: Array) -> Array:
    """Apply the latest block write for each open-group buffer slot."""
    src = jnp.take_along_axis(fresh, layout.refresh_source[..., None], axis=1)
    return jnp.where(layout.refresh[..., None], src, state)


@auto_pytree
class GatedPoolCompressOutput(OperationOutput):
    """Result of one :class:`GatedPoolCompressOp` call.

    Attributes:
        pooled: Pooled (pre-norm) entries: ``[B, G, D]`` for ``"windows"`` / ``"extend"``, ``[B, D]`` for
            ``"decode"`` (meaningful only where ``closed``; ``"windows"`` entries are all complete).
        closed: ``[B]`` (decode) or ``[B, G]`` (extend) bool, groups that complete with this call.
        entry: Entry index each pooled group occupies, same shape as ``closed``.
        buffer_kv: Updated open-group kv buffer ``[B, rate, F]`` (decode / extend).
        buffer_gate: Updated open-group gate buffer ``[B, rate, F]`` (raw gates, position bias not applied).
        overlap_kv: Two-series only: carried Ca kv ``[B, rate, D]`` (windows: the last group's).
        overlap_gate: Two-series only: carried biased fp32 Ca gate ``[B, rate, D]``.
    """

    pooled: Array | None = None
    closed: Array | None = None
    entry: Array | None = None
    buffer_kv: Array | None = None
    buffer_gate: Array | None = None
    overlap_kv: Array | None = None
    overlap_gate: Array | None = None


def _biased(gate: Array, position_bias: Array | None) -> Array:
    """fp32 gate with the learned within-group bias added (when the stream has one)."""
    gate = gate.astype(jnp.float32)
    return gate if position_bias is None else gate + position_bias.astype(jnp.float32)


def _pool(slot_kv: Array, slot_gate: Array, compute_dtype: jnp.dtype) -> Array:
    """``sum(kv * softmax(gate))`` over the slot axis (``-2``), in ``compute_dtype`` with fp32 softmax.

    Buffers may be stored in fp8; they are dequantized to ``compute_dtype`` here (JAX does not promote fp8 with bf16,
    and casting the probabilities to fp8 would destroy precision).
    """
    weights = jax.nn.softmax(slot_gate, axis=-2).astype(compute_dtype)
    return jnp.sum(slot_kv.astype(compute_dtype) * weights, axis=-2)


def _windows(
    kv: Array,
    gate: Array,
    rate: int,
    position_bias: Array | None,
    series_dim: int | None,
    compute_dtype: jnp.dtype | None,
) -> GatedPoolCompressOutput:
    """Pool complete groups of ``kv`` / ``gate`` ``[B, G * rate, F]`` (prefill from an empty stream)."""
    batch, length, feat = kv.shape
    groups = length // rate
    chunk_kv = kv.reshape(batch, groups, rate, feat)
    chunk_gate = _biased(gate.reshape(batch, groups, rate, feat), position_bias)
    overlap_kv = overlap_gate = None
    if series_dim is None:
        slot_kv, slot_gate = chunk_kv, chunk_gate
    else:
        dim = series_dim
        # Group 0 has no predecessor: its Ca half is zero kv with a -inf gate.
        prev_kv = jnp.concatenate(
            [jnp.zeros((batch, 1, rate, dim), dtype=chunk_kv.dtype), chunk_kv[:, :-1, :, :dim]], axis=1
        )
        prev_gate = jnp.concatenate(
            [jnp.full((batch, 1, rate, dim), -jnp.inf, dtype=jnp.float32), chunk_gate[:, :-1, :, :dim]], axis=1
        )
        slot_kv = jnp.concatenate([prev_kv, chunk_kv[..., dim:]], axis=2)
        slot_gate = jnp.concatenate([prev_gate, chunk_gate[..., dim:]], axis=2)
        overlap_kv, overlap_gate = chunk_kv[:, -1, :, :dim], chunk_gate[:, -1, :, :dim]
    pooled = _pool(slot_kv, slot_gate, kv.dtype if compute_dtype is None else compute_dtype)
    return GatedPoolCompressOutput(pooled=pooled, overlap_kv=overlap_kv, overlap_gate=overlap_gate)


def _decode(
    kv: Array,
    gate: Array,
    buffer_kv: Array,
    buffer_gate: Array,
    position: Array,
    rate: int,
    position_bias: Array | None,
    overlap_kv: Array | None,
    overlap_gate: Array | None,
    valid: Array | None,
    compute_dtype: jnp.dtype | None,
) -> GatedPoolCompressOutput:
    """Advance by one token per row: write slot ``position % rate``, pool the (possibly partial) group."""
    batch = kv.shape[0]
    rows = jnp.arange(batch)
    p = position.astype(jnp.int32)
    fill = jnp.mod(p, rate)
    new_kv_row = kv[:, 0].astype(buffer_kv.dtype)
    new_gate_row = gate[:, 0].astype(buffer_gate.dtype)
    if valid is not None:
        new_kv_row = jnp.where(valid[:, None], new_kv_row, buffer_kv[rows, fill])
        new_gate_row = jnp.where(valid[:, None], new_gate_row, buffer_gate[rows, fill])
    buffer_kv = buffer_kv.at[rows, fill].set(new_kv_row)
    buffer_gate = buffer_gate.at[rows, fill].set(new_gate_row)
    closed = fill == (rate - 1)
    if valid is not None:
        closed = closed & valid

    biased_gate = _biased(buffer_gate, position_bias)
    if overlap_kv is None:
        slot_kv, slot_gate = buffer_kv, biased_gate
    else:
        dim = overlap_kv.shape[-1]
        slot_kv = jnp.concatenate([overlap_kv.astype(buffer_kv.dtype), buffer_kv[..., dim:]], axis=1)
        slot_gate = jnp.concatenate([overlap_gate.astype(jnp.float32), biased_gate[..., dim:]], axis=1)
        overlap_kv = jnp.where(closed[:, None, None], buffer_kv[..., :dim].astype(overlap_kv.dtype), overlap_kv)
        overlap_gate = jnp.where(closed[:, None, None], biased_gate[..., :dim], overlap_gate)
    pooled = _pool(slot_kv, slot_gate, kv.dtype if compute_dtype is None else compute_dtype)
    return GatedPoolCompressOutput(
        pooled=pooled,
        closed=closed,
        entry=p // rate,
        buffer_kv=buffer_kv,
        buffer_gate=buffer_gate,
        overlap_kv=overlap_kv,
        overlap_gate=overlap_gate,
    )


def _extend(
    kv: Array,
    gate: Array,
    buffer_kv: Array,
    buffer_gate: Array,
    positions: Array,
    valid: Array,
    rate: int,
    position_bias: Array | None,
    overlap_kv: Array | None,
    overlap_gate: Array | None,
    compute_dtype: jnp.dtype | None,
) -> GatedPoolCompressOutput:
    """Advance by a block per row; every group closing inside the block is pooled in one pass."""
    layout = block_group_layout(positions, valid, rate)
    fresh_kv = kv.astype(buffer_kv.dtype)
    fresh_gate = gate.astype(buffer_gate.dtype)
    kv_members = gather_group_members(layout, buffer_kv, fresh_kv)  # [B, G, rate, F]
    gate_members = _biased(gather_group_members(layout, buffer_gate, fresh_gate), position_bias)
    if overlap_kv is None:
        slot_kv, slot_gate = kv_members, gate_members
    else:
        dim = overlap_kv.shape[-1]
        slot_kv, slot_gate = overlap_group_slots(kv_members, gate_members, overlap_kv, overlap_gate, dim)
        overlap_kv, overlap_gate = last_closed_overlap(layout, kv_members, gate_members, overlap_kv, overlap_gate, dim)
    pooled = _pool(slot_kv, slot_gate, kv.dtype if compute_dtype is None else compute_dtype)
    return GatedPoolCompressOutput(
        pooled=pooled,
        closed=layout.closed,
        entry=layout.entry,
        buffer_kv=refresh_group_state(layout, buffer_kv, fresh_kv),
        buffer_gate=refresh_group_state(layout, buffer_gate, fresh_gate),
        overlap_kv=overlap_kv,
        overlap_gate=overlap_gate,
    )


@OperationRegistry.register
class GatedPoolCompressOp(OperationImpl):
    """Softmax-gated KV pooling with an incremental state machine (DeepSeek-V4 / V4.1 compressors and indexers).

    Pure-JAX on every backend (the arithmetic is small elementwise work XLA fuses well). Cache-agnostic: callers pass
    and receive the open-group buffers and overlap state explicitly.
    """

    @classmethod
    def get_impl_name(cls) -> str:
        """Return the operation registry key."""
        return "gated_pool_compress"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """No metadata fields and no engine cache view: compressor state is owned by the caller's cache."""
        del mode
        return (
            RequirementsBuilder("gated_pool_compress").require_metadata(MetadataField.NONE).no_cache_required().build()
        )

    @jax.named_scope("easydel-gated_pool_compress")
    def forward_native(
        self,
        mode: tp.Literal["windows", "decode", "extend"],
        kv: Array,
        gate: Array,
        rate: int,
        *,
        position_bias: Array | None = None,
        series_dim: int | None = None,
        buffer_kv: Array | None = None,
        buffer_gate: Array | None = None,
        position: Array | None = None,
        positions: Array | None = None,
        valid: Array | None = None,
        overlap_kv: Array | None = None,
        overlap_gate: Array | None = None,
        compute_dtype: jnp.dtype | None = None,
        **ignores,
    ) -> GatedPoolCompressOutput:
        """Run one compression step.

        Args:
            mode: ``"windows"`` (stateless complete groups), ``"decode"`` (one token per row) or ``"extend"`` (a
                block per row).
            kv: Raw kv projections: ``[B, G * rate, F]`` (windows), ``[B, 1, F]`` (decode), ``[B, blk, F]`` (extend).
            gate: Raw gate projections, same shape (position bias not applied).
            rate: Tokens per group (static).
            position_bias: Optional learned within-group bias ``[rate, F]`` added to the fp32 gates.
            series_dim: Windows only: per-series width ``D`` of a two-series stream (``F == 2 * D``); ``None`` for
                single-series. Decode / extend infer it from ``overlap_kv``.
            buffer_kv: Decode / extend: open-group kv buffer ``[B, rate, F]`` (any storage dtype).
            buffer_gate: Decode / extend: open-group raw gate buffer ``[B, rate, F]``.
            position: Decode: stream position of the new token per row ``[B]``.
            positions: Extend: stream positions ``[B, blk]`` (garbage at invalid cells).
            valid: Decode: optional ``[B]`` row mask (``False`` rows leave state untouched). Extend: ``[B, blk]``,
                a prefix per row.
            overlap_kv: Two-series decode / extend: carried Ca kv ``[B, rate, D]``.
            overlap_gate: Two-series decode / extend: carried biased fp32 Ca gate ``[B, rate, D]`` (``-inf`` before
                the first close).
            compute_dtype: Pooling dtype; defaults to ``kv.dtype``.
            **ignores: Ignored (interface parity).

        Returns:
            GatedPoolCompressOutput: pooled pre-norm entries and the updated state.

        Raises:
            ValueError: For an unknown ``mode``.
        """
        del ignores
        if mode == "windows":
            return _windows(kv, gate, rate, position_bias, series_dim, compute_dtype)
        if mode == "decode":
            return _decode(
                kv,
                gate,
                buffer_kv,
                buffer_gate,
                position,
                rate,
                position_bias,
                overlap_kv,
                overlap_gate,
                valid,
                compute_dtype,
            )
        if mode == "extend":
            return _extend(
                kv,
                gate,
                buffer_kv,
                buffer_gate,
                positions,
                valid,
                rate,
                position_bias,
                overlap_kv,
                overlap_gate,
                compute_dtype,
            )
        raise ValueError(f"gated_pool_compress: unknown mode {mode!r}.")


@functools.lru_cache(maxsize=1)
def gated_pool_compress_op() -> GatedPoolCompressOp:
    """Process-wide :class:`GatedPoolCompressOp` (stateless; built without a mesh so any layer may call it)."""
    from .._operation_meta import OperationMetadata

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=None,
        platform=None,
        backend=None,
    )
    return GatedPoolCompressOp(metadata)
