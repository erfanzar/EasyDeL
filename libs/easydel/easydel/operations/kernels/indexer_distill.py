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

"""Sparse-attention indexer warm-up objective as a registered operation.

The DSA lightning-indexer objective ``KL(stop_grad(p) || softmax(indexer scores))``: ``p`` is the layer's dense
attention distribution over the indexer's candidate entries (summed over attention heads, L1-normalized per query),
and only the indexer scores receive gradient.

The attention target is a sink-augmented softmax over ``[window tokens | entries | sink]``. Both the sliding-window
token region and the sink are optional, so one operation covers:

- compressed-window attention (DeepSeek-V4.1 style): window tokens + compressed entries + per-head sinks;
- plain token DSA (DeepSeek-V3.2 / GLM-DSA style): entries are the tokens themselves, no window, no sink.

Entry reachability is read from the scores: an entry with a ``-inf`` score is unreachable for that query (causality,
padding and segment masks are the caller's job). Queries that reach no entry contribute nothing; the loss is the mean
KL over the queries that do.

The computation is tiled over queries (``jax.lax.map`` + ``jax.checkpoint``): no ``[B, H, S, S]`` tensor exists, and
with lazy scores (any object implementing :class:`LazyIndexerScores`) no ``[B, S, T]`` score tensor exists in either
AD direction. Pure JAX on every backend; the registered operation gives a fused kernel a dispatch slot.
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
    "IndexerDistillKLOp",
    "IndexerDistillKLOutput",
    "LazyIndexerScores",
    "indexer_distill_kl_op",
)

_MASK_MIN = float(jnp.finfo(jnp.float32).min)


@tp.runtime_checkable
class LazyIndexerScores(tp.Protocol):
    """Indexer scores evaluated per query tile instead of as one ``[B, S, T]`` tensor.

    ``pad_rows(pad)`` returns a source with ``pad`` extra query rows that reach nothing (all ``-inf``);
    ``score_rows(start, size)`` returns the ``[B, size, T]`` scores of rows ``start .. start + size`` (``start`` may
    be traced). See :class:`easydel.layers.indexer.IndexerScoreInputs`.
    """

    def pad_rows(self, pad: int) -> LazyIndexerScores: ...

    def score_rows(self, start: Array, size: int) -> Array: ...


@auto_pytree
class IndexerDistillKLOutput(OperationOutput):
    """Result of one :class:`IndexerDistillKLOp` call.

    Attributes:
        loss: Scalar fp32 mean KL over the queries that reach at least one entry (``0`` when none do).
        kl_sum: Scalar fp32 sum of the per-query KL terms.
        num_rows: Scalar fp32 number of queries that reach at least one entry.
    """

    loss: Array | None = None
    kl_sum: Array | None = None
    num_rows: Array | None = None


def _head_logits(q: Array, keys: Array, precision: jax.lax.PrecisionLike) -> Array:
    """``[B, H, Sq, Sk]`` fp32 logits for shared keys ``[B, Sk, D]`` or per-head keys ``[B, H, Sk, D]``."""
    spec = "bhsd,btd->bhst" if keys.ndim == 3 else "bhsd,bhtd->bhst"
    return jnp.einsum(spec, q, keys, precision=precision).astype(jnp.float32)


def _distill_kl(
    q: Array,
    entries: Array,
    scores: Array | LazyIndexerScores,
    window_kv: Array | None,
    sinks: Array | None,
    token_valid: Array | None,
    window: int | None,
    scaling: float,
    precision: jax.lax.PrecisionLike,
    block_q: int,
    entry_group: int = 1,
    entry_valid: Array | None = None,
    q_positions: Array | None = None,
    entry_positions: Array | None = None,
    q_segment_ids: Array | None = None,
    entry_segment_ids: Array | None = None,
) -> IndexerDistillKLOutput:
    """Tiled KL; see :meth:`IndexerDistillKLOp.forward_native` for the argument contract."""
    # Imported lazily: ``easydel.layers.compressed_window`` imports ``easydel.operations`` at module load.
    from easydel.layers.compressed_window import window_visible

    entry_group = int(entry_group)
    if entry_group < 1:
        raise ValueError(f"indexer_distill_kl: entry_group must be >= 1; got {entry_group}.")
    if (q_positions is None) != (entry_positions is None):
        raise ValueError("indexer_distill_kl: q_positions and entry_positions must be given together.")
    if (q_segment_ids is None) != (entry_segment_ids is None):
        raise ValueError("indexer_distill_kl: q_segment_ids and entry_segment_ids must be given together.")
    if q.ndim != 4:
        raise ValueError(f"indexer_distill_kl: q must be [B, H, S, D]; got shape {q.shape}.")
    if entries.ndim not in (3, 4):
        raise ValueError(f"indexer_distill_kl: entries must be [B, T, D] or [B, H, T, D]; got {entries.shape}.")
    q = jax.lax.stop_gradient(q)
    has_window = window is not None
    if has_window:
        window = int(window)
        if window < 1:
            raise ValueError(f"indexer_distill_kl: window must be >= 1; got {window}.")
        if window_kv is None:
            raise ValueError("indexer_distill_kl: window_kv is required when window is set.")
        if window_kv.ndim not in (3, 4):
            raise ValueError(f"indexer_distill_kl: window_kv must be [B, S, D] or [B, H, S, D]; got {window_kv.shape}.")
        window_kv = jax.lax.stop_gradient(window_kv)
        entries = jax.lax.stop_gradient(entries).astype(window_kv.dtype)
    else:
        entries = jax.lax.stop_gradient(entries)
    batch, heads, seq, _ = q.shape
    num_entries = entries.shape[-2]
    tile = max(1, min(int(block_q), seq))
    tiles = -(-seq // tile)
    pad = tiles * tile - seq
    q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
    span = min(tile + window - 1, seq) if has_window else 0
    if has_window:
        valid = jnp.ones((batch, seq), dtype=bool) if token_valid is None else token_valid.astype(bool)
        kv_axis = window_kv.ndim - 2
    sink = None if sinks is None else jax.lax.stop_gradient(sinks).astype(jnp.float32).reshape(1, heads, 1, 1)
    num_groups = -(-num_entries // entry_group)
    if not isinstance(scores, LazyIndexerScores) and scores.shape[-1] != num_groups:
        raise ValueError(
            f"indexer_distill_kl: scores cover {scores.shape[-1]} candidates; expected ceil({num_entries} / "
            f"{entry_group}) = {num_groups}."
        )
    ent_valid = None if entry_valid is None else entry_valid.astype(bool)
    qpos_padded = None
    if q_positions is not None:
        qpos_padded = jnp.pad(q_positions.astype(jnp.int32), ((0, 0), (0, pad)), constant_values=-1)
        entry_positions = entry_positions.astype(jnp.int32)
    qseg_padded = None
    if q_segment_ids is not None:
        qseg_padded = jnp.pad(q_segment_ids.astype(jnp.int32), ((0, 0), (0, pad)), constant_values=-1)
        entry_segment_ids = entry_segment_ids.astype(jnp.int32)

    if isinstance(scores, LazyIndexerScores):
        padded_scores = scores.pad_rows(pad)

        def _tile_scores(index):
            return padded_scores.score_rows(index * tile, tile)

    else:
        dense_padded = jnp.pad(scores, ((0, 0), (0, pad), (0, 0)), constant_values=-jnp.inf)

        def _tile_scores(index):
            return jax.lax.dynamic_slice_in_dim(dense_padded, index * tile, tile, axis=1)

    @jax.checkpoint
    def _tile_kl(index):
        """``(sum of per-query KL, number of queries reaching an entry)`` for queries in tile ``index``.

        The target is normalized per query row, so the KL is complete inside the tile; the tile is rematerialized in
        the backward pass.
        """
        start = index * tile
        q_t = jax.lax.dynamic_slice_in_dim(q_padded, start, tile, axis=2)
        scores_t = _tile_scores(index)  # [B, tile, T], carries the indexer gradient
        reach_t = jax.lax.stop_gradient(scores_t) > -jnp.inf
        parts = []
        if has_window:
            lo = jnp.clip(start - window + 1, 0, seq - span)
            k_t = jax.lax.dynamic_slice_in_dim(window_kv, lo, span, axis=kv_axis)
            key_ok = jax.lax.dynamic_slice_in_dim(valid, lo, span, axis=1)
            q_pos = start + jnp.arange(tile, dtype=jnp.int32)[:, None]
            k_pos = lo + jnp.arange(span, dtype=jnp.int32)[None, :]
            allowed = window_visible(q_pos, k_pos, window)[None] & key_ok[:, None, :]  # [B, tile, span]
            window_logits = _head_logits(q_t, k_t, precision) * scaling
            parts.append(jnp.where(allowed[:, None], window_logits, _MASK_MIN))
        # Attention-side entry visibility: the indexer's reach (expanded from groups to entries) narrowed by the
        # optional token-level validity, causal and segment masks.
        ent_vis = reach_t if entry_group == 1 else jnp.repeat(reach_t, entry_group, axis=-1)[..., :num_entries]
        if ent_valid is not None:
            ent_vis = ent_vis & ent_valid[:, None, :]
        if qpos_padded is not None:
            qpos_t = jax.lax.dynamic_slice_in_dim(qpos_padded, start, tile, axis=1)
            ent_vis = ent_vis & (entry_positions[:, None, :] <= qpos_t[:, :, None])
        if qseg_padded is not None:
            qseg_t = jax.lax.dynamic_slice_in_dim(qseg_padded, start, tile, axis=1)
            ent_vis = ent_vis & (entry_segment_ids[:, None, :] == qseg_t[:, :, None])
        entry_logits = _head_logits(q_t, entries, precision)
        parts.append(jnp.where(ent_vis[:, None], entry_logits * scaling, _MASK_MIN))
        if sink is not None:
            parts.append(jnp.broadcast_to(sink, (batch, heads, tile, 1)))
        combined = jnp.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
        probs = jax.nn.softmax(combined - jnp.max(combined, axis=-1, keepdims=True), axis=-1)
        target = jnp.sum(probs[..., span : span + num_entries], axis=1)  # [B, tile, T], summed over heads
        if entry_group > 1:
            target = jnp.pad(target, ((0, 0), (0, 0), (0, num_groups * entry_group - num_entries)))
            target = target.reshape(batch, tile, num_groups, entry_group).sum(axis=-1)  # [B, tile, G]
        target = jax.lax.stop_gradient(target / jnp.maximum(jnp.sum(target, axis=-1, keepdims=True), 1e-30))
        log_pred = jax.nn.log_softmax(jnp.where(reach_t, scores_t, _MASK_MIN).astype(jnp.float32), axis=-1)
        positive = target > 0
        kl = jnp.sum(jnp.where(positive, target * (jnp.log(jnp.where(positive, target, 1.0)) - log_pred), 0.0), axis=-1)
        row_valid = (jnp.any(reach_t, axis=-1) & jnp.any(ent_vis, axis=-1)).astype(jnp.float32)
        return jnp.sum(kl * row_valid), jnp.sum(row_valid)

    kl_sums, counts = jax.lax.map(_tile_kl, jnp.arange(tiles, dtype=jnp.int32))
    kl_sum, num_rows = jnp.sum(kl_sums), jnp.sum(counts)
    return IndexerDistillKLOutput(loss=kl_sum / jnp.maximum(num_rows, 1.0), kl_sum=kl_sum, num_rows=num_rows)


@OperationRegistry.register
class IndexerDistillKLOp(OperationImpl):
    """Indexer warm-up KL against the dense attention distribution (DSA lightning-indexer training).

    Pure JAX on every backend. Cache-agnostic and stateless: a training-time auxiliary objective over explicit
    tensors.
    """

    @classmethod
    def get_impl_name(cls) -> str:
        """Return the operation registry key."""
        return "indexer_distill_kl"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """No metadata fields and no cache view: every input is passed explicitly."""
        del mode
        return RequirementsBuilder("indexer_distill_kl").require_metadata(MetadataField.NONE).no_cache_required().build()

    @jax.named_scope("easydel-indexer_distill_kl")
    def forward_native(
        self,
        q: Array,
        entries: Array,
        scores: Array | LazyIndexerScores,
        *,
        scaling: float,
        window_kv: Array | None = None,
        sinks: Array | None = None,
        token_valid: Array | None = None,
        window: int | None = None,
        precision: jax.lax.PrecisionLike = None,
        block_q: int = 64,
        entry_group: int = 1,
        entry_valid: Array | None = None,
        q_positions: Array | None = None,
        entry_positions: Array | None = None,
        q_segment_ids: Array | None = None,
        entry_segment_ids: Array | None = None,
        **ignores,
    ) -> IndexerDistillKLOutput:
        """Compute the indexer distillation KL.

        Args:
            q: Attention queries (already roped) ``[B, H, S, D]``.
            entries: Attention keys of the indexer's candidates: shared ``[B, T, D]`` or per-head ``[B, H, T, D]``
                (compressed entries, or the tokens themselves for token DSA).
            scores: Indexer scores ``[B, S, T]`` with ``-inf`` at unreachable entries, or a
                :class:`LazyIndexerScores` evaluated per query tile.
            scaling: Attention logit scale.
            window_kv: Sliding-window token keys ``[B, S, D]`` or ``[B, H, S, D]``; required when ``window`` is set.
            sinks: Optional per-head attention sinks ``[H]`` (an extra logit column per head).
            token_valid: Optional per-key validity of the window region ``[B, S]``; key ``k`` is visible to query
                ``s`` iff ``s - window < k <= s`` and ``token_valid[k]``.
            window: Sliding-window size; ``None`` drops the token region (the softmax runs over entries [+ sink]).
            precision: Matmul precision.
            block_q: Queries per tile.
            entry_group: The indexer scores groups of ``entry_group`` consecutive entries (pooled or block
                indexers): ``scores`` has ``ceil(T / entry_group)`` columns, a group's reach opens all its entries
                on the attention side, and the target is the attention mass summed per group.
            entry_valid: Optional per-entry validity ``[B, T]`` (padding) on the attention side.
            q_positions: Optional query positions ``[B, S]``; with ``entry_positions`` ``[B, T]`` the attention
                side sees entry ``t`` only when ``entry_positions[t] <= q_positions[s]`` (token-level causality
                inside a partially reachable group).
            entry_positions: See ``q_positions``.
            q_segment_ids: Optional packed-sequence ids ``[B, S]``; with ``entry_segment_ids`` ``[B, T]`` the
                attention side sees only entries of the query's segment.
            entry_segment_ids: See ``q_segment_ids``.
            **ignores: Ignored (interface parity).

        Returns:
            IndexerDistillKLOutput: the mean KL and its sum / row count (for weighting across layers). A query
            counts when it reaches an indexer candidate and the attention side sees at least one entry.

        Raises:
            ValueError: On malformed shapes, ``window < 1``, ``window`` without ``window_kv``, a score width that
                does not match ``entry_group``, or half of a positions / segment-ids pair.
        """
        del ignores
        return _distill_kl(
            q,
            entries,
            scores,
            window_kv,
            sinks,
            token_valid,
            window,
            scaling,
            precision,
            block_q,
            entry_group=entry_group,
            entry_valid=entry_valid,
            q_positions=q_positions,
            entry_positions=entry_positions,
            q_segment_ids=q_segment_ids,
            entry_segment_ids=entry_segment_ids,
        )


@functools.lru_cache(maxsize=1)
def indexer_distill_kl_op() -> IndexerDistillKLOp:
    """Process-wide :class:`IndexerDistillKLOp` (stateless; built without a mesh so any layer may call it)."""
    from .._operation_meta import OperationMetadata

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=None,
        platform=None,
        backend=None,
    )
    return IndexerDistillKLOp(metadata)
