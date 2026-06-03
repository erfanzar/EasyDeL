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

"""Row-aware best-fit window packer for the precomputed "embeds-only" VLM data pack.

This is the PACKING half of the packed-embeds training path; the MATERIALIZER half is
:func:`easydel.data.transforms.collators.collate_packed_embeds`. The packer decides *which rows
go in which window*; the materializer turns one such assignment into a static ``(n_windows, seq_len)``
batch (ids / labels / attention_mask / segment_ids / 3D M-RoPE position_ids + the image embed
side-channel). Keeping them separate means the placement policy can change without touching the
deterministic shape/M-RoPE/embed-scatter logic, and vice versa.

WHY A NEW PACKER (vs. the token packers in ``pack.py``)
-------------------------------------------------------
``pack.py``'s :class:`GreedyPacker` / :class:`PoolPacker` / :class:`FirstFitPacker` are
TOKEN-ALIGNED: they concatenate token streams and slice every ``seq_length`` tokens, cutting
*across* example boundaries. That is fine for plain text, but the embeds pack carries a per-image
side-channel -- ``image_embeds`` / ``image_grid_thw`` / ``embed_n_tok`` -- whose alignment to the
``<image>`` placeholder RUN inside ``input_ids`` is load-bearing (the materializer asserts
``#placeholder-tokens == #decoded-embed-rows`` and scatters the k-th embed onto the k-th
placeholder). Slicing a row mid-stream would cut a placeholder run and desynchronise that scatter.

So this packer is ROW-AWARE: it never splits a row. A row either fits a window whole or is carried
to a later batch. Rows are the atomic unit, exactly as the materializer needs.

PLACEMENT POLICY (best-fit, online, carry across emits)
-------------------------------------------------------
Each :meth:`emit` opens ``n_windows`` empty bins of capacity ``seq_len`` and makes one pass over the
carry, placing each row into the bin with the LEAST remaining token-space that still fits it
(best-fit -- tighter packing than first-fit, deterministic lowest-index tie-break). A row that fits
no bin in this batch stays in the carry for the next emit. :meth:`flush` drains the carry at the end
(its final batch underfills -> the materializer whole-pads the trailing windows; that all-pad row is
proven backward-finite). Rows are NEVER dropped.

TWO capacity axes, not one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A materialized batch has BOTH a ``(n_windows, seq_len)`` token grid AND a ``(max_embed_rows, H)``
embed buffer, and a row consumes both. Binning on tokens alone could emit an image-dense batch whose
total embed rows exceed ``max_embed_rows`` -- the materializer would then raise mid-training. So the
packer also tracks a per-BATCH embed budget: a row joins the current emit only if both (a) some bin
has token room AND (b) the batch's running embed-row total + this row's embed rows <= ``max_embed_rows``.
This keeps the packer/materializer contract sound (the materializer never rejects a packer batch).
A row's embed-row count is ``sum(embed_n_tok)`` (== its image-placeholder count == its decoded embed
rows, by the pack invariant).

ADMISSION GUARDS (fail loud at push, not silently in carry)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A row whose token length > ``seq_len``, or whose embed rows > ``max_embed_rows``, can NEVER be placed
in any batch and would otherwise starve in the carry forever. :meth:`push` rejects both up front with
the offending magnitude, forcing an explicit upstream fix (raise the ceiling / route to a larger
bucket / truncate TEXT). ``seq_len`` must therefore be >= the largest single-row length present.

Only ``"best_fit"`` is implemented. The ``strategy`` knob is a seam for a future
first-fit-decreasing / lookahead policy; it is intentionally a fail-fast stub (raises) rather than a
half-built path, because the call on whether tighter packing is worth the buffering latency is a
*measure padding waste first* decision, not a speculative build.

USAGE
-----
::

    packer = EmbedsWindowPacker(n_windows=N, seq_len=S, max_embed_rows=E)
    for row in source:                      # or packer.push(chunk_of_rows)
        packer.push([row])
        if len(packer) >= enough_to_fill_a_batch:   # caller-owned cadence
            yield collate_packed_embeds(packer.emit(), pad_id, S, E, N)
    for packs in packer.flush():            # drain the tail (trailing windows whole-padded)
        yield collate_packed_embeds(packs, pad_id, S, E, N)

The caller owns the emit cadence: calling :meth:`emit` on a small carry produces a sparse batch
(few real windows, the rest inert pad) -- still correct and statically shaped, just lower MFU. Emit
when the carry is large enough to fill the windows; reserve sparse batches for :meth:`flush`.
"""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _row_tokens(row: dict) -> int:
    """Token length a row occupies in a window (== ``len(input_ids)``)."""
    return len(row["input_ids"])


def _row_embed_rows(row: dict) -> int:
    """Image-embed rows a row contributes to the batch embed buffer (== ``sum(embed_n_tok)``).

    By the pack invariant this equals the row's ``<image>`` placeholder count and its decoded embed
    row count, so it is exactly what the materializer concatenates into the ``(max_embed_rows, H)``
    buffer. Text-only rows have no ``embed_n_tok`` -> 0.
    """
    return int(sum(int(x) for x in row.get("embed_n_tok", [])))


class EmbedsWindowPacker:
    """Stateful, row-aware best-fit packer feeding :func:`collate_packed_embeds`.

    Maintains a carry buffer of not-yet-placed rows across :meth:`emit` calls. Each emit packs the
    carry into one static batch's worth of windows (best-fit into ``n_windows`` bins of capacity
    ``seq_len`` tokens, subject to a per-batch ``max_embed_rows`` embed budget) and returns the
    per-window row assignments; rows that do not fit stay carried. :meth:`flush` drains the carry.
    Rows are never split and never dropped. See the module docstring for the full rationale.
    """

    def __init__(self, n_windows: int, seq_len: int, max_embed_rows: int, strategy: str = "best_fit"):
        """Bind the compiled-bucket geometry and select the placement policy.

        Args:
            n_windows: Number of windows per emitted batch (the static leading dim the materializer
                pins). ``emit`` opens this many bins; underfilled batches whole-pad the rest.
            seq_len: Per-window token capacity (the bucket's sequence ceiling). Must be >= the
                largest single-row token length that will be pushed.
            max_embed_rows: Per-batch image-embed-row budget (the materializer's embed buffer height).
                Must be >= the largest single-row embed-row count that will be pushed.
            strategy: Placement policy. Only ``"best_fit"`` is implemented; any other value raises
                ``NotImplementedError`` (reserved seam for first-fit-decreasing / lookahead).
        """
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if max_embed_rows < 0:
            raise ValueError(f"max_embed_rows must be >= 0, got {max_embed_rows}")
        if strategy != "best_fit":
            raise NotImplementedError(
                f"packing strategy {strategy!r} is not implemented; only 'best_fit' is available. "
                "first_fit_decreasing / lookahead are a reserved seam -- measure best-fit padding "
                "waste before building a tighter policy."
            )
        self.n_windows = n_windows
        self.seq_len = seq_len
        self.max_embed_rows = max_embed_rows
        self.strategy = strategy
        self._carry: list[dict] = []

    def __len__(self) -> int:
        """Number of rows currently buffered in the carry (not yet emitted)."""
        return len(self._carry)

    def push(self, rows: Iterable[dict]) -> None:
        """Admit rows into the carry, rejecting any that can never be packed.

        A row longer than ``seq_len`` tokens, or carrying more than ``max_embed_rows`` embed rows,
        cannot fit any batch and would starve forever -- so it is rejected here with the offending
        magnitude rather than silently held. Order is preserved.

        Raises:
            ValueError: If a row's token length exceeds ``seq_len`` or its embed-row count exceeds
                ``max_embed_rows``.
        """
        for r in rows:
            tok = _row_tokens(r)
            if tok > self.seq_len:
                raise ValueError(
                    f"unpackable row: input_ids length {tok} exceeds seq_len {self.seq_len} -- an "
                    "oversized row can never fit a window and would starve in carry forever; raise "
                    "seq_len, route to a larger bucket, or truncate TEXT before packing"
                )
            emb = _row_embed_rows(r)
            if emb > self.max_embed_rows:
                raise ValueError(
                    f"unpackable row: {emb} image-embed rows exceed max_embed_rows "
                    f"{self.max_embed_rows} -- the row's embeds can never fit one batch's embed "
                    "budget; raise max_embed_rows for this bucket"
                )
            self._carry.append(r)

    def emit(self) -> list[list[dict]]:
        """Pack the carry into one batch's per-window row assignments (best-fit); carry the rest.

        Opens ``n_windows`` empty bins and makes a single pass over the carry: each row is placed in
        the bin with the least remaining token-space that still fits it, provided the batch embed
        budget is not exceeded; otherwise the row is carried forward. Returns only the NON-EMPTY
        windows (length ``M <= n_windows``); the materializer whole-pads the trailing
        ``n_windows - M``. Returns ``[]`` when the carry is empty.
        """
        bins: list[list[dict]] = [[] for _ in range(self.n_windows)]
        used_tok = [0] * self.n_windows
        batch_embed = 0
        leftover: list[dict] = []

        for r in self._carry:
            tok = _row_tokens(r)
            emb = _row_embed_rows(r)
            # Per-batch embed budget gate (independent of which bin): keep the batch within the
            # materializer's (max_embed_rows, H) buffer so it never rejects this emit.
            if batch_embed + emb > self.max_embed_rows:
                leftover.append(r)
                continue
            # Best-fit by token room: tightest bin that still fits; strict `<` keeps the lowest
            # index on ties (so empty bins fill in order).
            best = -1
            best_rem = self.seq_len + 1
            for bi in range(self.n_windows):
                rem = self.seq_len - used_tok[bi]
                if rem >= tok and rem < best_rem:
                    best, best_rem = bi, rem
            if best < 0:
                leftover.append(r)
                continue
            bins[best].append(r)
            used_tok[best] += tok
            batch_embed += emb

        self._carry = leftover
        return [b for b in bins if b]

    def flush(self) -> Iterator[list[list[dict]]]:
        """Drain the carry, yielding one batch's per-window assignments at a time until empty.

        Normal end-of-epoch use yields a single (typically underfilled) final batch, but a carry
        larger than one batch's capacity yields several. Each emit places at least the first carried
        row (its token/embed fit is guaranteed by :meth:`push`), so the carry strictly shrinks and
        this terminates. After draining, the packer is empty and reusable.
        """
        while self._carry:
            before = len(self._carry)
            packs = self.emit()
            if len(self._carry) >= before:
                raise RuntimeError(
                    "flush made no progress draining the carry -- this should be impossible given "
                    "the push admission guards; report as a packer bug"
                )
            yield packs
