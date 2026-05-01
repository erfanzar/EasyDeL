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

"""Token packing utilities for efficient training.

This module provides:
- Greedy packing: Simple concatenation-based packing
- Pool-based packing: Multiple packers for efficient bin-packing
- First-fit packing: Bin-packing with first-fit decreasing
- Segment IDs for attention masking
"""

from __future__ import annotations

import logging
import random
import typing as tp
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..core.config import PackStageConfig
from ..core.protocols import BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)


@dataclass
class PackedSequence:
    """One fixed-length output produced by a packer, plus per-token bookkeeping.

    Carries the packed token sequence together with the metadata
    downstream consumers need to undo the packing for attention and
    loss computation: the attention mask (which positions are valid
    after padding), the segment ids (which packed example each
    position belongs to, so attention can be masked across segment
    boundaries), and the source ids (which constituent dataset each
    segment came from, for per-source metrics). Constructed by
    :class:`GreedyPacker`, :class:`PoolPacker`, and
    :class:`FirstFitPacker` and consumed by
    :class:`PackedShardedSource`.

    Attributes:
        input_ids (np.ndarray): Token id sequence of shape
            ``(seq_length,)`` and dtype ``int32``.
        attention_mask (np.ndarray | None): Same-shape mask with
            ``1`` for valid tokens and ``0`` for padding; ``None``
            for fully-filled sequences that did not require padding.
        segment_ids (np.ndarray | None): Same-shape array assigning
            each token to a packed segment; segment boundaries are
            consumed by attention masks so model layers do not
            cross-attend between concatenated examples. ``None``
            when the parent packer had ``include_segment_ids=False``.
        source_ids (list[str] | None): Per-segment list of source
            identifiers (typically dataset names from
            :class:`MixedShardedSource`'s ``"__source__"`` metadata).
            ``None`` when no sources were attached.
        num_segments (int): Count of original examples concatenated
            into this packed window.
    """

    input_ids: np.ndarray
    attention_mask: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    source_ids: list[str] | None = None
    num_segments: int = 0

    def to_dict(self) -> dict[str, np.ndarray]:
        """Render the packed sequence as a dict suitable for batch collation.

        Always includes ``"input_ids"``; ``"attention_mask"`` and
        ``"segment_ids"`` are only added when the corresponding
        attributes are non-``None`` so consumers can distinguish "not
        present" from "present but all-ones".

        Returns:
            dict[str, np.ndarray]: Dict with the packer's
            stack-friendly arrays. Used as the row payload yielded by
            :class:`PackedShardedSource`.
        """
        result = {"input_ids": self.input_ids}
        if self.attention_mask is not None:
            result["attention_mask"] = self.attention_mask
        if self.segment_ids is not None:
            result["segment_ids"] = self.segment_ids
        return result


class GreedyPacker:
    """Streaming first-come-first-served packer that fills one window at a time.

    Maintains a single rolling buffer; each call to :meth:`add`
    appends the new tokens (plus an EOS separator) and emits a
    completed :class:`PackedSequence` whenever the buffer reaches
    ``seq_length``. Leftover tokens past the boundary roll into the
    next window. Final non-empty leftovers are surfaced via
    :meth:`flush_final` (with padding and an attention mask).

    This is the cheapest strategy — a single buffer means no fitting
    overhead — but produces sub-optimal packing density relative to
    :class:`PoolPacker` and :class:`FirstFitPacker` on highly skewed
    length distributions.
    """

    def __init__(
        self,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        include_segment_ids: bool = True,
    ):
        """Capture packing settings and initialise empty rolling buffers.

        Args:
            seq_length: Number of tokens per emitted window.
            eos_token_id: Token id appended between concatenated
                examples; consumed by attention masking on the model
                side.
            pad_token_id: Token id used to pad incomplete windows
                produced by :meth:`flush_final`.
            include_segment_ids: When ``True``, track segment ids
                alongside ``input_ids`` so attention can be masked
                across segment boundaries.
        """
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.include_segment_ids = include_segment_ids

        # Current buffer
        self._buffer: list[int] = []
        self._segment_ids: list[int] = []
        self._current_segment = 0
        self._source_ids: list[str] = []

    def add(self, tokens: list[int], source_id: str | None = None) -> PackedSequence | None:
        """Append a tokenized example, emitting a packed window if one is ready.

        After adding, an EOS separator and segment-id bump are
        recorded so the boundary between this and the next example
        is preserved. A full window is flushed via :meth:`_flush`
        whenever the rolling buffer reaches ``seq_length``.

        Args:
            tokens: Token id list for one upstream example.
            source_id: Optional source label used to populate
                :attr:`PackedSequence.source_ids`. Pass the dataset
                name from :class:`MixedShardedSource`'s
                ``"__source__"`` field for per-source visibility.

        Returns:
            PackedSequence | None: A packed window when this call
            filled the buffer to ``seq_length``; otherwise ``None``
            (the tokens were absorbed into the buffer for the next
            call).
        """
        result = None

        # Add tokens to buffer
        for tok in tokens:
            self._buffer.append(tok)
            if self.include_segment_ids:
                self._segment_ids.append(self._current_segment)

            # Check if we have a full sequence
            if len(self._buffer) >= self.seq_length:
                result = self._flush()

        # Add EOS and update segment
        if len(self._buffer) > 0:
            self._buffer.append(self.eos_token_id)
            if self.include_segment_ids:
                self._segment_ids.append(self._current_segment)
            self._current_segment += 1
            if source_id:
                self._source_ids.append(source_id)

        # Check if we hit the target length
        if len(self._buffer) >= self.seq_length:
            result = self._flush()

        return result

    def _flush(self) -> PackedSequence:
        """Emit a packed sequence and retain any leftover tokens.

        Returns:
            ``PackedSequence`` populated from the first ``seq_length``
            tokens of the buffer; remaining tokens are kept for the next
            packing round.
        """
        # Take exactly seq_length tokens
        input_ids = np.array(self._buffer[: self.seq_length], dtype=np.int32)

        segment_ids = None
        if self.include_segment_ids:
            segment_ids = np.array(self._segment_ids[: self.seq_length], dtype=np.int32)

        result = PackedSequence(
            input_ids=input_ids,
            segment_ids=segment_ids,
            source_ids=self._source_ids.copy() if self._source_ids else None,
            num_segments=self._current_segment,
        )

        # Keep remainder
        self._buffer = self._buffer[self.seq_length :]
        if self.include_segment_ids:
            self._segment_ids = self._segment_ids[self.seq_length :]
        self._source_ids = []
        self._current_segment = 0

        return result

    def flush_final(self) -> PackedSequence | None:
        """Emit the trailing partial window (padded) when iteration ends.

        Pads the remaining tokens to ``seq_length`` with
        :attr:`pad_token_id`, builds an attention mask that is ``1``
        for the original tokens and ``0`` for padding, and resets the
        internal buffers so the packer can be reused. No-op when the
        buffer is empty.

        Returns:
            PackedSequence | None: The padded trailing window with
            ``attention_mask`` set; ``None`` when there's nothing
            left to flush.
        """
        if not self._buffer:
            return None

        # Pad to seq_length
        pad_len = self.seq_length - len(self._buffer)
        input_ids = np.array(self._buffer + [self.pad_token_id] * pad_len, dtype=np.int32)

        attention_mask = np.ones(self.seq_length, dtype=np.int32)
        attention_mask[len(self._buffer) :] = 0

        segment_ids = None
        if self.include_segment_ids:
            padded_segments = self._segment_ids + [self._current_segment] * pad_len
            segment_ids = np.array(padded_segments, dtype=np.int32)

        result = PackedSequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            source_ids=self._source_ids.copy() if self._source_ids else None,
            num_segments=self._current_segment + 1,
        )

        self._buffer = []
        self._segment_ids = []
        self._source_ids = []
        self._current_segment = 0

        return result


class PoolPacker:
    """Pool of independent :class:`GreedyPacker` instances using best-fit dispatch.

    Maintains ``num_packers`` separate greedy packers; each incoming
    example is routed to the packer that would have the least free
    space *after* accepting it (best-fit decreasing). This trims
    padding compared with a single greedy packer at the cost of
    extra book-keeping. Output windows from any packer are surfaced
    in completion order.
    """

    def __init__(
        self,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        num_packers: int = 4,
        include_segment_ids: bool = True,
    ):
        """Allocate ``num_packers`` underlying :class:`GreedyPacker` instances.

        Args:
            seq_length: Target window length, propagated to every
                inner packer.
            eos_token_id: EOS separator id, propagated to every
                inner packer.
            pad_token_id: Padding id used during final flushes.
            num_packers: Pool size — larger values trade memory for
                better packing density. ``1`` collapses to greedy
                packing.
            include_segment_ids: Whether the inner packers track
                segment ids.
        """
        self.seq_length = seq_length
        self.num_packers = num_packers
        self._packers = [
            GreedyPacker(seq_length, eos_token_id, pad_token_id, include_segment_ids) for _ in range(num_packers)
        ]

    def add(self, tokens: list[int], source_id: str | None = None) -> list[PackedSequence]:
        """Route the tokens to the best-fit inner packer and surface any completed windows.

        Best fit is computed as the smallest non-negative
        ``seq_length - (current_buffer_len + token_len + 1_for_EOS)``;
        ties pick the lowest-index packer. The chosen packer's
        return is forwarded as a list (length 0 or 1) for API
        symmetry with :class:`FirstFitPacker`.

        Args:
            tokens: Token id list for one upstream example.
            source_id: Optional source label propagated through to
                the inner packer.

        Returns:
            list[PackedSequence]: Completed packed windows produced
            by this call. Empty when no packer rolled over.
        """
        results = []
        token_len = len(tokens)

        # Find the packer with the best fit (least remaining space after adding)
        best_idx = 0
        best_fit = float("inf")

        for i, packer in enumerate(self._packers):
            current_len = len(packer._buffer)
            remaining_after = self.seq_length - (current_len + token_len + 1)  # +1 for EOS

            if 0 <= remaining_after < best_fit:
                best_fit = remaining_after
                best_idx = i

        # Add to best packer
        result = self._packers[best_idx].add(tokens, source_id)
        if result is not None:
            results.append(result)

        return results

    def flush_all(self) -> list[PackedSequence]:
        """Drain trailing partial windows from every inner packer.

        Calls :meth:`GreedyPacker.flush_final` on each packer in the
        pool and aggregates the non-``None`` results. Suitable for
        end-of-iteration cleanup.

        Returns:
            list[PackedSequence]: One trailing window per inner
            packer that still had data. Empty when every packer was
            already drained.
        """
        results = []
        for packer in self._packers:
            result = packer.flush_final()
            if result is not None:
                results.append(result)
        return results


class FirstFitPacker:
    """Buffered first-fit-decreasing bin packer for higher packing density.

    Buffers ``buffer_size`` examples before producing windows, then
    sorts them in decreasing length and applies the classic
    first-fit-decreasing bin-packing heuristic to arrange them into
    ``seq_length``-sized bins. Each bin becomes one
    :class:`PackedSequence` with attention mask and segment ids
    populated. Higher density than :class:`PoolPacker` at the cost
    of buffering latency (output is delayed until each
    ``buffer_size`` window of upstream rows arrives).
    """

    def __init__(
        self,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        include_segment_ids: bool = True,
        buffer_size: int = 1000,
    ):
        """Capture packing settings and initialise the pending buffer.

        Args:
            seq_length: Target bin/window length in tokens.
            eos_token_id: Token id appended after each example
                inside a bin.
            pad_token_id: Token id used to pad bins that did not
                completely fill.
            include_segment_ids: Whether to record segment ids per
                token.
            buffer_size: Number of upstream examples accumulated
                before a packing pass runs. Larger buffers give
                better density but more latency and memory use.
        """
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.include_segment_ids = include_segment_ids
        self.buffer_size = buffer_size

        self._pending: list[tuple[list[int], str | None]] = []

    def add(self, tokens: list[int], source_id: str | None = None) -> list[PackedSequence]:
        """Buffer one example, triggering a packing pass when the buffer fills.

        Args:
            tokens: Token id list for one upstream example.
            source_id: Optional source label associated with this
                example; aggregated into
                :attr:`PackedSequence.source_ids` when the bin is
                emitted.

        Returns:
            list[PackedSequence]: Newly packed bins when this call
            tipped the buffer over ``buffer_size`` and triggered
            :meth:`_pack_buffer`; an empty list otherwise.
        """
        self._pending.append((tokens, source_id))

        if len(self._pending) >= self.buffer_size:
            return self._pack_buffer()

        return []

    def _pack_buffer(self) -> list[PackedSequence]:
        """Pack the pending buffer using first-fit decreasing.

        Returns:
            List of ``PackedSequence`` objects, padded to ``seq_length``,
            consuming all currently buffered sequences.
        """
        if not self._pending:
            return []

        # Sort by length (decreasing)
        sorted_pending = sorted(self._pending, key=lambda x: len(x[0]), reverse=True)

        # Bins: list of (tokens, segment_ids, source_ids)
        bins: list[tuple[list[int], list[int], list[str]]] = []

        for tokens, source_id in sorted_pending:
            token_len = len(tokens) + 1  # +1 for EOS
            placed = False

            # Find first bin that fits
            for _i, (bin_tokens, bin_segments, bin_sources) in enumerate(bins):
                if len(bin_tokens) + token_len <= self.seq_length:
                    # Add to this bin
                    segment_id = max(bin_segments) + 1 if bin_segments else 0
                    bin_tokens.extend(tokens)
                    bin_tokens.append(self.eos_token_id)
                    bin_segments.extend([segment_id] * (len(tokens) + 1))
                    if source_id:
                        bin_sources.append(source_id)
                    placed = True
                    break

            if not placed:
                # Create new bin
                new_tokens = [*tokens, self.eos_token_id]
                new_segments = [0] * len(new_tokens)
                new_sources = [source_id] if source_id else []
                bins.append((new_tokens, new_segments, new_sources))

        # Convert bins to PackedSequences
        results = []
        for bin_tokens, bin_segments, bin_sources in bins:
            # Pad if needed
            pad_len = self.seq_length - len(bin_tokens)
            input_ids = np.array(bin_tokens + [self.pad_token_id] * pad_len, dtype=np.int32)

            attention_mask = np.ones(self.seq_length, dtype=np.int32)
            attention_mask[len(bin_tokens) :] = 0

            segment_ids = None
            if self.include_segment_ids:
                max_seg = max(bin_segments) if bin_segments else 0
                padded_segments = bin_segments + [max_seg + 1] * pad_len
                segment_ids = np.array(padded_segments, dtype=np.int32)

            results.append(
                PackedSequence(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    source_ids=bin_sources if bin_sources else None,
                    num_segments=max(bin_segments) + 1 if bin_segments else 0,
                )
            )

        self._pending = []
        return results

    def flush_all(self) -> list[PackedSequence]:
        """Run a final packing pass over whatever is still buffered.

        Returns:
            list[PackedSequence]: Bins produced from the residual
            buffer; may be empty when the buffer was already drained.
        """
        return self._pack_buffer()


class PackedShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` adapter that packs an upstream tokenized source.

    Reads tokenized rows from an underlying
    :class:`ShardedDataSource`, drives a configurable packer
    (:class:`GreedyPacker`, :class:`PoolPacker`, or
    :class:`FirstFitPacker`), and yields packed dict rows ready for
    training. The ``"__source__"`` metadata produced by
    :class:`MixedShardedSource` is forwarded to the packer as
    ``source_id`` so per-source segment provenance is preserved.

    An optional reservoir-sampling shuffle (sized as
    ``shuffle_buffer_factor * 100``) acts on the packed rows so
    consecutive output windows do not all come from the same packer
    in pool/first-fit modes.
    """

    def __init__(
        self,
        source: ShardedDataSource[dict],
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        strategy: str = "greedy",
        num_packers: int = 4,
        include_segment_ids: bool = True,
        input_field: str = "input_ids",
        shuffle: bool = True,
        shuffle_buffer_factor: int = 10,
        seed: int | None = None,
    ):
        """Capture packer selection and shuffle settings without iterating.

        Args:
            source: Upstream tokenized :class:`ShardedDataSource`
                whose rows carry token ids in ``input_field``.
            seq_length: Window size produced by the packer.
            eos_token_id: EOS separator token id used between
                packed examples.
            pad_token_id: Padding token id for trailing partial
                windows.
            strategy: ``"greedy"`` (single rolling buffer),
                ``"pool"`` (best-fit dispatch across multiple
                buffers), or ``"first_fit"`` (buffered FFD bin
                packing).
            num_packers: Pool size when ``strategy == "pool"``;
                ignored otherwise.
            include_segment_ids: Whether the packer tracks segment
                ids.
            input_field: Row key from which token ids are read.
            shuffle: Run reservoir-sampling shuffle on packed rows
                before yielding.
            shuffle_buffer_factor: Multiplier on a reference batch
                size (currently 100) controlling the shuffle
                reservoir capacity.
            seed: Optional RNG seed for the shuffle reservoir.
        """
        self._source = source
        self._seq_length = seq_length
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._strategy = strategy
        self._num_packers = num_packers
        self._include_segment_ids = include_segment_ids
        self._input_field = input_field
        self._shuffle = shuffle
        self._shuffle_buffer_factor = shuffle_buffer_factor
        self._seed = seed

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return a single synthetic shard for the packed source.

        Returns:
            One-element list ``["packed_shard_0"]``.
        """
        return ["packed_shard_0"]

    def num_shards(self) -> int:
        """Return the constant shard count of one.

        Returns:
            Always ``1``.
        """
        return 1

    def _create_packer(self):
        """Instantiate a packer matching ``self._strategy``.

        Returns:
            A ``GreedyPacker``, ``PoolPacker``, or ``FirstFitPacker``
            configured from this source's settings.
        """
        if self._strategy == "pool":
            return PoolPacker(
                self._seq_length,
                self._eos_token_id,
                self._pad_token_id,
                self._num_packers,
                self._include_segment_ids,
            )
        elif self._strategy == "first_fit":
            return FirstFitPacker(
                self._seq_length,
                self._eos_token_id,
                self._pad_token_id,
                self._include_segment_ids,
            )
        else:  # greedy
            return GreedyPacker(
                self._seq_length,
                self._eos_token_id,
                self._pad_token_id,
                self._include_segment_ids,
            )

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Drive the upstream source through the packer and yield packed dict rows.

        Walks every shard of :attr:`_source`, hands each row's
        tokens to the configured packer, and re-emits any packed
        windows the packer produces. Empty token lists are skipped.
        After the upstream is exhausted, drains the packer's
        residual buffer (via :meth:`flush_all` for batch-style
        packers or :meth:`flush_final` for the greedy packer). When
        shuffling is enabled, the residual reservoir is fully
        shuffled and drained at the end.

        Args:
            shard_name: Ignored — :class:`PackedShardedSource`
                exposes a single synthetic shard.

        Yields:
            dict: Each packed row as produced by
            :meth:`PackedSequence.to_dict` — at minimum
            ``{"input_ids": ndarray}`` and possibly
            ``"attention_mask"`` / ``"segment_ids"`` depending on
            packer state.
        """
        if self._seed is not None:
            random.seed(self._seed)

        packer = self._create_packer()
        shuffle_buffer = []
        max_buffer = self._shuffle_buffer_factor * 100  # Approximate batch size

        def emit(packed: PackedSequence):
            """Inline closure: feed a packed window through the optional shuffle reservoir.

            Captures ``self._shuffle``, ``shuffle_buffer``, and
            ``max_buffer`` from the enclosing :meth:`open_shard`
            scope. With shuffling disabled, simply renders the
            packed sequence as a dict. With shuffling on, fills the
            reservoir up to ``max_buffer`` (returning ``None`` while
            warming up); once full, swaps the new entry for a
            randomly selected existing one and yields the evicted
            entry.

            Args:
                packed: The newly produced packed window.

            Returns:
                dict | None: Dict ready to yield, or ``None`` while
                the shuffle reservoir is still being filled (the
                caller skips ``None`` returns).
            """
            result = packed.to_dict()
            if self._shuffle:
                if len(shuffle_buffer) < max_buffer:
                    shuffle_buffer.append(result)
                    return None
                else:
                    idx = random.randrange(0, max_buffer)
                    out = shuffle_buffer[idx]
                    shuffle_buffer[idx] = result
                    return out
            return result

        # Iterate through source
        for source_shard in self._source.shard_names:
            for example in self._source.open_shard(source_shard):
                tokens = example.get(self._input_field, [])
                if not tokens:
                    continue

                source_id = example.get("__source__")

                if isinstance(packer, (PoolPacker, FirstFitPacker)):
                    results = packer.add(list(tokens), source_id)
                    for packed in results:
                        out = emit(packed)
                        if out is not None:
                            yield out
                else:
                    result = packer.add(list(tokens), source_id)
                    if result is not None:
                        out = emit(result)
                        if out is not None:
                            yield out

        # Flush packer
        if isinstance(packer, (PoolPacker, FirstFitPacker)):
            for packed in packer.flush_all():
                out = emit(packed)
                if out is not None:
                    yield out
        else:
            final = packer.flush_final()
            if final is not None:
                out = emit(final)
                if out is not None:
                    yield out

        # Emit remaining shuffle buffer
        if self._shuffle:
            random.shuffle(shuffle_buffer)
            yield from shuffle_buffer

    def __len__(self) -> int:
        """Coarse estimate of the packed-row count, derived from upstream length.

        Assumes ~70% packing efficiency / ~1.4 source examples per
        packed window — accurate enough for progress bars but not
        for exact counts. Real counts depend on the token-length
        distribution and the chosen strategy.

        Returns:
            int: Estimate of ``len(source) / 1.4`` floored to at
            least ``1``.

        Raises:
            TypeError: If the wrapped source does not support
                ``len()`` (i.e. is streaming).
        """
        # Estimate based on average sequence length ratio
        # Assume ~70% packing efficiency as a rough heuristic
        source_len = len(self._source)
        # Rough estimate: each packed sequence contains ~1.4 original sequences on average
        return max(1, int(source_len / 1.4))

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"PackedShardedSource(seq_length=N, strategy='...', source=...)"``.
        """
        return (
            f"PackedShardedSource(seq_length={self._seq_length}, strategy={self._strategy!r}, source={self._source!r})"
        )


class PackStage(BaseStage):
    """Pipeline stage that wraps each dataset in a :class:`PackedShardedSource`.

    Activated by :attr:`PackStageConfig.enabled`; when off, the
    stage is a pass-through. When on, every entry of the rolling
    source dict is replaced by a :class:`PackedShardedSource`
    configured from :class:`PackStageConfig`. The trainer then sees
    fixed-length packed windows instead of variable-length
    tokenized rows, dramatically improving GPU/TPU utilisation for
    short-sequence datasets.
    """

    def __init__(self, config: PackStageConfig | None = None):
        """Capture the pack configuration and forward to the base stage.

        Args:
            config: :class:`PackStageConfig` controlling enable
                flag, sequence length, EOS/pad ids, packing
                strategy, and shuffle behaviour. ``None`` produces
                a default disabled config so the stage is
                constructible without arguments.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or PackStageConfig()

    @property
    def name(self) -> str:
        """Stage identifier used in metric and log namespaces.

        Returns:
            str: Constant string ``"pack"``.
        """
        return "pack"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        """Replace each source in ``data`` with a packed version.

        No-ops when packing is disabled in the stage config, so
        callers can chain ``pack()`` unconditionally. The
        per-source RNG seed is taken from ``context.seed`` so the
        whole pipeline run is reproducible.

        Args:
            data: Rolling ``{dataset_name: ShardedDataSource}`` dict
                from the previous stage.
            context: Shared :class:`PipelineContext`; only
                ``context.seed`` is consulted.

        Returns:
            dict[str, ShardedDataSource]: ``data`` itself when the
            stage is disabled; otherwise a same-keyed dict whose
            values are :class:`PackedShardedSource` instances.
        """
        if not self._stage_config.enabled:
            return data

        result = {}
        for ds_name, source in data.items():
            packed = PackedShardedSource(
                source=source,
                seq_length=self._stage_config.seq_length,
                eos_token_id=self._stage_config.eos_token_id,
                pad_token_id=self._stage_config.pad_token_id,
                strategy=self._stage_config.strategy,
                num_packers=self._stage_config.num_packers,
                include_segment_ids=self._stage_config.include_segment_ids,
                shuffle=self._stage_config.shuffle_packed,
                shuffle_buffer_factor=self._stage_config.shuffle_buffer_factor,
                seed=context.seed,
            )
            result[ds_name] = packed
            logger.info(f"Packed dataset '{ds_name}' with strategy={self._stage_config.strategy}")

        return result


def pack_pre_tokenized(stream, seq_length: int, eos_token_id: int, batch_size: int, shuffle: bool, buffer_factor: int):
    """Pack pre-tokenized sequences into constant-length chunks.

    Takes a stream of pre-tokenized examples and packs them into fixed-length
    sequences for efficient training. Sequences are concatenated and split
    at the specified sequence length, with EOS tokens inserted as needed.

    Args:
        stream: Iterator of dictionaries containing 'tokens' field.
        seq_length: Target length for packed sequences.
        eos_token_id: Token ID to use for padding/separation.
        batch_size: Batch size (used for shuffle buffer calculation).
        shuffle: Whether to shuffle the packed sequences.
        buffer_factor: Multiplier for shuffle buffer size (batch_size * buffer_factor).

    Returns:
        Generator yielding dictionaries with 'input_ids' as JAX arrays.
    """

    def gen():
        """Inline closure that runs the pre-tokenized constant-length packer.

        Captures ``stream``, ``seq_length``, ``eos_token_id``,
        ``batch_size``, ``shuffle``, and ``buffer_factor`` from
        :func:`pack_pre_tokenized`. Maintains a numpy buffer plus
        an optional reservoir-shuffle list of size
        ``batch_size * buffer_factor``; slices ``seq_length``
        windows out of the buffer and either yields them directly
        or pushes them through the reservoir.

        Yields:
            dict: ``{"input_ids": jax.Array}`` per packed window.
            The trailing reservoir contents are shuffled and drained
            after the upstream stream is exhausted.
        """
        buf = np.array([], dtype=np.int32)
        eos = np.array([eos_token_id], dtype=np.int32)
        shuffle_buf = []
        max_buf = batch_size * buffer_factor

        for sample in stream:
            toks = sample["tokens"]
            # Use asarray to avoid unnecessary copy if already int32 ndarray
            toks = np.asarray(toks, dtype=np.int32)
            buf = np.concatenate([buf, toks], axis=0)
            if len(buf) % seq_length != 0:
                buf = np.concatenate([buf, eos], axis=0)
            while len(buf) >= seq_length:
                ex = {"input_ids": jnp.array(buf[:seq_length])}
                buf = buf[seq_length:]
                if shuffle:
                    if len(shuffle_buf) < max_buf:
                        shuffle_buf.append(ex)
                    else:
                        i = random.randrange(0, max_buf)
                        yield shuffle_buf[i]
                        shuffle_buf[i] = ex
                else:
                    yield ex
        random.shuffle(shuffle_buf)
        for ex in shuffle_buf:
            yield ex

    return gen


def pack_constant_length(
    stream,
    tokenize_fn,
    seq_length: int,
    eos_token_id: int,
    batch_size: int,
    shuffle: bool,
    buffer_factor: int,
):
    """Pack sequences with on-the-fly tokenization into constant-length chunks.

    Combines tokenization and packing in a single pipeline. Takes raw examples,
    tokenizes them using the provided function, and packs the results into
    fixed-length sequences.

    Args:
        stream: Iterator of raw examples to tokenize.
        tokenize_fn: Function that takes an example and returns token IDs.
        seq_length: Target length for packed sequences.
        eos_token_id: Token ID to use for padding/separation.
        batch_size: Batch size (used for shuffle buffer calculation).
        shuffle: Whether to shuffle the packed sequences.
        buffer_factor: Multiplier for shuffle buffer size (batch_size * buffer_factor).

    Returns:
        Generator yielding dictionaries with 'input_ids' as JAX arrays.
    """

    def token_iter():
        """Inline closure: lazily tokenise the upstream stream into a ``tokens`` shape.

        Captures ``stream`` and ``tokenize_fn`` from
        :func:`pack_constant_length`. Each upstream example is
        passed through ``tokenize_fn`` and re-emitted as
        ``{"tokens": <ids>}`` so :func:`pack_pre_tokenized` can
        consume it.

        Yields:
            dict: One ``{"tokens": list[int]}`` per upstream example.
        """
        for ex in stream:
            toks = tokenize_fn(ex)
            yield {"tokens": toks}

    return pack_pre_tokenized(token_iter(), seq_length, eos_token_id, batch_size, shuffle, buffer_factor)
