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

"""History penalties (presence / frequency / repetition) on the greedy speculative verify path.

A history penalty is a deterministic function of the already-emitted prefix, so greedy verification stays lossless
under it: when every verify row is penalised with the count state of its own prefix (committed history plus the
drafts that precede that row), the emitted stream is exactly the target's penalised greedy stream. Only the
acceptance rate changes.

Sampled verification is deliberately not covered. Rejection sampling reads the drafter's own proposal
distribution, and penalising the target while that distribution stays unpenalised biases the acceptance ratio, so
sampled requests that carry a penalty keep the plain decode path.

Parity with the non-speculative sampler is by construction: the device kernel gathers only the penalised logit
rows, rebuilds their dense count rows from sparse ``(token, count)`` entries and calls
:func:`~easydel.inference.esurge.core.sampler.apply_history_penalties_from_counts`, the same function and dtypes
(uint32 counts cast to the logits dtype, float32 penalties cast to the logits dtype) the sampler applies, then takes
the same ``argmax``. The logits never leave the device: only the sparse entries travel host-to-device and only one
int32 per logit row comes back.

Costs:
    * Host: :class:`HistoryCountCache` keeps each penalised request's committed-history counts incrementally, so a
      step costs O(new committed tokens) to update plus O(distinct history tokens) to pack the sparse entries
      (one entry list per request, shared by all of its verify rows; drafts add O(k^2) entries).
    * Device: transient ``[request groups, vocab]`` and ``[penalised rows, vocab]`` uint32 count matrices inside the
      kernel (both bucketed to powers of two), no larger than the verify logits already materialised.
    * Compiles: one kernel variant per (logit-row count, group bucket, row bucket, entry bucket, draft bucket).
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp

from ...core.sampler import apply_history_penalties_from_counts

__all__ = (
    "HistoryCountCache",
    "HistoryCounts",
    "VerifyPenaltyRows",
    "history_penalty_params",
    "penalized_verify_argmax",
)


def history_penalty_params(sampling_params: typing.Any) -> tuple[float, float, float] | None:
    """Return ``(presence, frequency, repetition)`` when any history penalty is active.

    Mirrors the sampler's ``need_history_penalties`` predicate: presence and frequency are additive (``0`` is
    inactive) and repetition is multiplicative (``1`` is inactive).

    Args:
        sampling_params: Request sampling parameters (``SamplingParams`` or any object with the penalty attributes).

    Returns:
        The three penalties as floats, or ``None`` when none of them changes the logits.
    """
    presence = float(getattr(sampling_params, "presence_penalty", 0.0) or 0.0)
    frequency = float(getattr(sampling_params, "frequency_penalty", 0.0) or 0.0)
    repetition_raw = getattr(sampling_params, "repetition_penalty", 1.0)
    repetition = 1.0 if repetition_raw is None else float(repetition_raw)
    if presence == 0.0 and frequency == 0.0 and repetition == 1.0:
        return None
    return presence, frequency, repetition


class HistoryCounts:
    """Append-only token-count state of one token history.

    Attributes:
        length: Number of history positions consumed so far (negative ids are consumed but not counted).
    """

    __slots__ = ("_counts", "_ids", "_index", "_size", "length")

    def __init__(self, capacity: int = 64) -> None:
        """Create an empty history.

        Args:
            capacity: Initial number of distinct ids the backing arrays hold before growing.
        """
        capacity = max(1, int(capacity))
        self._ids = np.empty((capacity,), dtype=np.int32)
        self._counts = np.empty((capacity,), dtype=np.int32)
        self._index: dict[int, int] = {}
        self._size = 0
        self.length = 0

    def extend(self, tokens: Iterable[int]) -> None:
        """Consume the next history positions.

        Args:
            tokens: Token ids at the following positions, in order.
        """
        for token in tokens:
            self.length += 1
            tok = int(token)
            if tok < 0:
                continue
            slot = self._index.get(tok)
            if slot is not None:
                self._counts[slot] += 1
                continue
            if self._size == int(self._ids.shape[0]):
                grown = 2 * self._size
                self._ids = np.resize(self._ids, (grown,))
                self._counts = np.resize(self._counts, (grown,))
            self._ids[self._size] = tok
            self._counts[self._size] = 1
            self._index[tok] = self._size
            self._size += 1

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return snapshots of the distinct ids and their counts.

        The snapshots are copies (O(distinct ids), the same cost as packing them), so later :meth:`extend` calls
        never mutate arrays a caller still holds.

        Returns:
            ``(token_ids, counts)`` int32 arrays of equal length, in first-seen order.
        """
        return self._ids[: self._size].copy(), self._counts[: self._size].copy()


class HistoryCountCache:
    """Per-request :class:`HistoryCounts`, advanced incrementally across steps."""

    def __init__(self) -> None:
        """Create an empty cache."""
        self._entries: dict[str, HistoryCounts] = {}

    def counts(self, req_id: str, token_at: Callable[[int], int], upto: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the counts of a request's history positions ``[0, upto)``.

        Positions already consumed are reused; a shorter ``upto`` than the cached length (a recomputed request)
        restarts the history from scratch.

        Args:
            req_id: Request identifier.
            token_at: Maps an absolute sequence position to its committed token id.
            upto: Exclusive end of the history.

        Returns:
            ``(token_ids, counts)`` as returned by :meth:`HistoryCounts.arrays`.
        """
        upto = max(0, int(upto))
        entry = self._entries.get(req_id)
        if entry is None or entry.length > upto:
            entry = HistoryCounts()
            self._entries[req_id] = entry
        if entry.length < upto:
            entry.extend(token_at(pos) for pos in range(entry.length, upto))
        return entry.arrays()

    def drop(self, req_id: str) -> None:
        """Forget one request's history.

        Args:
            req_id: Request identifier.
        """
        self._entries.pop(req_id, None)

    def clear(self) -> None:
        """Forget every history."""
        self._entries.clear()

    def __contains__(self, req_id: object) -> bool:
        """Whether ``req_id`` has a cached history."""
        return req_id in self._entries


@dataclass(frozen=True)
class VerifyPenaltyRows:
    """Penalty state of one request's consecutive verify rows.

    Logit row ``offset + i`` is penalised with ``history`` plus ``draft_tokens[: draft_prefix + i]``.

    Attributes:
        offset: First logit row of the request.
        num_rows: Number of consecutive logit rows.
        history_ids: Distinct committed-history token ids.
        history_counts: Count of each id in ``history_ids``.
        draft_tokens: Drafts following the committed history, in order.
        presence_penalty: Additive penalty for every token with a non-zero count.
        frequency_penalty: Additive penalty per occurrence.
        repetition_penalty: Sign-dependent multiplicative penalty; ``1`` disables it.
        draft_prefix: Number of drafts already preceding the first row.
    """

    offset: int
    num_rows: int
    history_ids: np.ndarray
    history_counts: np.ndarray
    draft_tokens: tuple[int, ...]
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    draft_prefix: int = 0


def _bucket(n: int) -> int:
    """Return the smallest power of two that is ``>= max(n, 1)``."""
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


@partial(jax.jit, static_argnames=("num_groups",))
def _penalized_argmax_kernel(
    logits: jax.Array,
    penalized_rows: jax.Array,
    row_group: jax.Array,
    history_group: jax.Array,
    history_token: jax.Array,
    history_count: jax.Array,
    draft_row: jax.Array,
    draft_token: jax.Array,
    presence: jax.Array,
    frequency: jax.Array,
    repetition: jax.Array,
    active: jax.Array,
    *,
    num_groups: int,
) -> jax.Array:
    """Greedy argmax of every row, with the listed rows penalised exactly as the sampler would.

    Only the penalised rows are gathered; their dense count rows are rebuilt from sparse entries and fed to
    :func:`apply_history_penalties_from_counts`. Padding entries carry an out-of-range group, row or logit-row index
    and are dropped by the scatters.

    Args:
        logits: Verify logits ``[rows, vocab]``.
        penalized_rows: Logit row of every penalised slot ``[slots]``.
        row_group: Request group of every penalised slot ``[slots]``.
        history_group: Group of every committed-history entry ``[entries]``.
        history_token: Token id of every committed-history entry ``[entries]``.
        history_count: Count of every committed-history entry ``[entries]``.
        draft_row: Penalised slot of every draft entry ``[drafts]``.
        draft_token: Token id of every draft entry ``[drafts]``.
        presence: Presence penalty per slot ``[slots]``.
        frequency: Frequency penalty per slot ``[slots]``.
        repetition: Repetition penalty per slot ``[slots]``.
        active: Slots that hold a real penalised row ``[slots]``.
        num_groups: Static group-axis size.

    Returns:
        ``[rows]`` int32 argmax ids.
    """
    num_rows, vocab = logits.shape
    plain = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    selected = logits[jnp.clip(penalized_rows, 0, num_rows - 1)]
    group_counts = (
        jnp.zeros((num_groups, vocab), dtype=jnp.uint32)
        .at[history_group, history_token]
        .add(history_count.astype(jnp.uint32), mode="drop")
    )
    slot_counts = group_counts[jnp.clip(row_group, 0, num_groups - 1)]
    slot_counts = slot_counts.at[draft_row, draft_token].add(jnp.ones_like(draft_row, dtype=jnp.uint32), mode="drop")
    penalized = apply_history_penalties_from_counts(
        selected,
        token_counts=slot_counts,
        active_mask=active,
        presence_penalties=presence,
        frequency_penalties=frequency,
        repetition_penalties=repetition,
    )
    penalized_tokens = jnp.argmax(penalized, axis=-1).astype(jnp.int32)
    return plain.at[penalized_rows].set(penalized_tokens, mode="drop")


def penalized_verify_argmax(logits: jax.Array, rows: Sequence[VerifyPenaltyRows]) -> np.ndarray:
    """Greedy argmax of every logit row, penalising the rows covered by ``rows``.

    Rows not covered by any entry keep their plain argmax.

    Args:
        logits: Verify logits ``[total_rows, vocab]`` (device array; never copied to host).
        rows: Penalty state per penalised request; row ranges must not overlap.

    Returns:
        ``[total_rows]`` int64 token ids.

    Raises:
        ValueError: If ``logits`` is not rank 2, or a row range falls outside it or overlaps another.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [rows, vocab], got shape {tuple(logits.shape)}")
    total_rows, vocab = int(logits.shape[0]), int(logits.shape[1])
    num_groups = _bucket(len(rows))
    num_slots = sum(max(0, int(spec.num_rows)) for spec in rows)
    slots_pad = _bucket(num_slots)

    # Padding slots point past the logits (dropped by the final scatter) and stay inactive.
    slot_row = np.full((slots_pad,), total_rows, dtype=np.int32)
    slot_group = np.full((slots_pad,), num_groups, dtype=np.int32)
    active = np.zeros((slots_pad,), dtype=np.bool_)
    presence = np.zeros((slots_pad,), dtype=np.float32)
    frequency = np.zeros((slots_pad,), dtype=np.float32)
    repetition = np.ones((slots_pad,), dtype=np.float32)
    covered = np.zeros((total_rows,), dtype=np.bool_)
    hist_group: list[np.ndarray] = []
    hist_token: list[np.ndarray] = []
    hist_count: list[np.ndarray] = []
    draft_slots: list[int] = []
    draft_tokens: list[int] = []

    slot = 0
    for group, spec in enumerate(rows):
        start, stop = int(spec.offset), int(spec.offset) + int(spec.num_rows)
        if start < 0 or stop > total_rows or spec.num_rows <= 0:
            raise ValueError(f"penalty rows [{start}, {stop}) fall outside {total_rows} logit rows")
        if bool(covered[start:stop].any()):
            raise ValueError(f"penalty rows [{start}, {stop}) overlap another request")
        if int(spec.draft_prefix) + int(spec.num_rows) - 1 > len(spec.draft_tokens):
            raise ValueError("penalty rows need more draft tokens than were provided")
        covered[start:stop] = True

        ids = np.asarray(spec.history_ids, dtype=np.int64).reshape(-1)
        counts = np.asarray(spec.history_counts, dtype=np.int64).reshape(-1)
        if ids.shape != counts.shape:
            raise ValueError(f"history ids {ids.shape} and counts {counts.shape} must match")
        keep = (ids >= 0) & (ids < vocab) & (counts > 0)
        hist_group.append(np.full((int(keep.sum()),), group, dtype=np.int32))
        hist_token.append(ids[keep].astype(np.int32))
        hist_count.append(counts[keep].astype(np.uint32))

        for i in range(int(spec.num_rows)):
            slot_row[slot] = start + i
            slot_group[slot] = group
            active[slot] = True
            presence[slot] = spec.presence_penalty
            frequency[slot] = spec.frequency_penalty
            repetition[slot] = spec.repetition_penalty
            for tok in spec.draft_tokens[: int(spec.draft_prefix) + i]:
                if 0 <= int(tok) < vocab:
                    draft_slots.append(slot)
                    draft_tokens.append(int(tok))
            slot += 1

    num_entries = sum(int(part.shape[0]) for part in hist_group)
    entries_pad = _bucket(num_entries)
    h_group = np.full((entries_pad,), num_groups, dtype=np.int32)
    h_token = np.zeros((entries_pad,), dtype=np.int32)
    h_count = np.zeros((entries_pad,), dtype=np.uint32)
    if num_entries:
        h_group[:num_entries] = np.concatenate(hist_group)
        h_token[:num_entries] = np.concatenate(hist_token)
        h_count[:num_entries] = np.concatenate(hist_count)

    drafts_pad = _bucket(len(draft_slots))
    d_slot = np.full((drafts_pad,), slots_pad, dtype=np.int32)
    d_token = np.zeros((drafts_pad,), dtype=np.int32)
    d_slot[: len(draft_slots)] = draft_slots
    d_token[: len(draft_tokens)] = draft_tokens

    out = _penalized_argmax_kernel(
        logits,
        slot_row,
        slot_group,
        h_group,
        h_token,
        h_count,
        d_slot,
        d_token,
        presence,
        frequency,
        repetition,
        active,
        num_groups=num_groups,
    )
    return np.asarray(jax.device_get(out)).astype(np.int64)
