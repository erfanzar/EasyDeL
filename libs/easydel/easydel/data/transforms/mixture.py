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

"""Dataset mixing utilities with dynamic weight scheduling.

This module provides:
- Block-based deterministic mixing for reproducible training
- Dynamic weight scheduling that changes weights during training
- ShardedDataSource-based mixing for new pipeline
- Linear and cosine weight interpolation
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import typing as tp
import zlib
from dataclasses import dataclass

import numpy as np

from ..core.config import MixStageConfig, WeightSchedulePoint
from ..core.protocols import BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

# Sentinel returned by probe draws in `_empty_constituents`; mixed rows are dicts,
# so a private object() can never collide with a real example.
_PROBE_EMPTY = object()

STOP_STRATEGIES: tuple[str, ...] = ("restart", "first_exhausted", "all_exhausted", "error")

DATASET_ID_FIELD = "_dataset_id"

ASSISTANT_ONLY_FIELD = "_assistant_only"

DATASET_ID_TOKEN_FIELD = "dataset_ids"

ASSISTANT_ONLY_TOKEN_FIELD = "assistant_only_mask"

DEFAULT_DRAW_LOG_INTERVAL = 262_144


def largest_remainder_counts(weights: np.ndarray, block_size: int) -> np.ndarray:
    """Split ``block_size`` slots across datasets by largest-remainder rounding.

    Floors each ``weight * block_size`` and hands the leftover slots to the largest fractional
    parts (stable ties by index), so the counts always sum to ``block_size`` and no dataset is
    systematically over-sampled. Shared by :class:`MixedShardedSource` and
    :func:`block_mixture_interleave` so both mixers round identically.

    Args:
        weights: Normalized (sum to 1) per-dataset weights.
        block_size: Total number of examples per mixing block.

    Returns:
        np.ndarray: Integer per-dataset counts summing to ``block_size``.
    """
    exact = weights * block_size
    counts = np.floor(exact).astype(int)
    remainder = int(block_size - counts.sum())
    if remainder > 0:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


class WeightScheduler:
    """Step-indexed interpolator that produces curriculum mixing weights on demand.

    Wraps a list of :class:`WeightSchedulePoint` knots and answers
    "what should the weights be at step N?" via :meth:`get_weights`.
    Three interpolation modes are supported: ``"step"`` returns the
    weights of the previous knot unchanged, ``"linear"`` blends
    linearly between adjacent knots, and ``"cosine"`` uses a cosine
    annealing curve. Linearly interpolated values are renormalised
    so they always sum to ``1.0`` even when the source vectors do.

    Two invariants are enforced at construction:

    1. The schedule must be non-empty.
    2. Every knot must declare weights for the same set of dataset
       names; otherwise interpolation between knots would be
       under-specified.

    The schedule is sorted by step internally so callers can pass
    knots in any order.

    Example:
        >>> schedule = [
        ...     WeightSchedulePoint(step=0, weights={"code": 0.3, "text": 0.7}),
        ...     WeightSchedulePoint(step=10000, weights={"code": 0.5, "text": 0.5}),
        ... ]
        >>> scheduler = WeightScheduler(schedule, interpolation="linear")
        >>> weights = scheduler.get_weights(step=5000)
        >>> # {"code": 0.4, "text": 0.6}
    """

    def __init__(
        self,
        schedule: list[WeightSchedulePoint],
        interpolation: str = "step",
    ):
        """Validate and store the schedule, sorted by step.

        Args:
            schedule: One or more :class:`WeightSchedulePoint` knots.
                Order does not matter — the constructor sorts by
                step. Must be non-empty.
            interpolation: One of ``"step"`` (piecewise-constant),
                ``"linear"`` (linear blending), or ``"cosine"``
                (cosine-annealed blending). Unknown values fall
                through to linear.

        Raises:
            ValueError: When ``schedule`` is empty or when knots
                declare different dataset name sets.
        """
        if not schedule:
            raise ValueError("Schedule must have at least one point")

        self._schedule = sorted(schedule, key=lambda p: p.step)
        self._interpolation = interpolation
        self._dataset_names = list(self._schedule[0].weights.keys())

        for point in self._schedule:
            if set(point.weights.keys()) != set(self._dataset_names):
                raise ValueError("All schedule points must have the same dataset names")

    def get_weights(self, step: int) -> dict[str, float]:
        """Resolve the active mixing weights at ``step`` per the configured policy.

        Steps before the first knot return that knot's weights;
        steps after the last knot return the last knot's weights.
        Steps in between are interpolated according to
        :attr:`interpolation`. Output is always renormalised to sum
        to 1.0.

        Args:
            step: Schedule position (0-indexed) at which to evaluate the schedule. NOTE: when driven by
                ``MixedShardedSource`` this is the cumulative count of EMITTED EXAMPLES, not optimizer steps
                (the data mixer has no batch size); a point for optimizer step ``N`` is at ``N * batch_size``.

        Returns:
            dict[str, float]: Mapping from dataset name to its weight
            at ``step``. The dataset name set is identical across
            steps and equals
            :attr:`WeightSchedulePoint.weights`'s keys.
        """
        if step <= self._schedule[0].step:
            return self._schedule[0].weights.copy()

        if step >= self._schedule[-1].step:
            return self._schedule[-1].weights.copy()

        for i in range(len(self._schedule) - 1):
            if self._schedule[i].step <= step < self._schedule[i + 1].step:
                start = self._schedule[i]
                end = self._schedule[i + 1]
                break
        else:
            return self._schedule[-1].weights.copy()

        if self._interpolation == "step":
            return start.weights.copy()

        progress = (step - start.step) / (end.step - start.step)

        if self._interpolation == "cosine":
            progress = (1 - math.cos(progress * math.pi)) / 2

        # Linear interpolation (or cosine-adjusted linear)
        result = {}
        for name in self._dataset_names:
            start_w = start.weights[name]
            end_w = end.weights[name]
            result[name] = start_w + progress * (end_w - start_w)

        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    @property
    def dataset_names(self) -> list[str]:
        """Names of the datasets the schedule applies to, in declaration order.

        Returns:
            list[str]: Fresh copy of the canonical name list — mutating
            it does not affect the scheduler.
        """
        return self._dataset_names.copy()


@dataclass
class MixedShardState:
    """Resumable position marker for an in-progress :class:`MixedShardedSource` iteration.

    Captured (and later restored) by checkpointing so a long mixed
    run can be paused and continued without restarting from the
    beginning. Most fields are bookkeeping for the mixer's
    block-based interleaver — together they identify which source
    we were drawing from, where inside that source, and how many
    blocks of the schedule we've completed.

    Attributes:
        source_index (int): Position within
            :attr:`MixedShardedSource._names` of the source currently
            being drawn from.
        shard_index (int): Index into the active source's
            :attr:`ShardedDataSource.shard_names` list.
        row_index (int): Row offset inside the active shard.
        block_index (int): Number of mixing blocks that have been
            fully consumed so far; drives the curriculum scheduler
            and per-block RNG seed for reproducibility.
        examples_yielded (int): Total examples produced across all
            blocks since iteration began. Useful for progress
            display and for debugging schedule resumption.
        seed (int | None): Effective RNG seed captured by the mixer, including
            an automatically chosen seed for an initially unseeded mix.
    """

    source_index: int
    shard_index: int
    row_index: int
    block_index: int = 0
    examples_yielded: int = 0
    seed: int | None = None


class MixedShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` adapter implementing block-based weighted mixing.

    Wraps several named :class:`ShardedDataSource` constituents into
    a single virtual source. Each output "block" of ``block_size``
    examples contains exactly the configured number of rows from
    each constituent (rounded by largest-remainder, ensuring sums
    match block size); the rows inside the block are shuffled with
    a per-block deterministic RNG so cross-source ordering is
    reproducible across restarts.

    Behaviour at end-of-source is controlled by ``stop_strategy``:
    ``"restart"`` recycles the constituent, ``"first_exhausted"``
    halts the entire mix, ``"all_exhausted"`` keeps drawing from
    the surviving positive-weight sources until each is exhausted.
    Zero-weight sources are excluded; a schedule stops when no currently
    eligible source can yield, since its emitted-row counter cannot then
    advance to a future schedule knot. ``"error"`` raises
    :class:`RuntimeError` on the first exhaustion, for protocols where
    repeating data means the configuration is wrong. Optional
    :class:`WeightScheduler` integration replaces the static weights with
    curriculum-driven weights at each block.

    With ``reshuffle_on_restart=True`` every restart of a constituent visits
    its shards in a fresh deterministic permutation seeded by
    ``(seed, crc32(name), epoch)`` instead of replaying the identical
    order; row order inside a shard is unchanged, so pair it with a
    downstream shuffle buffer for row-level decorrelation. Epoch 0 always
    uses canonical shard order and seeked resume reproduces the permuted
    epochs exactly.

    The realized per-source draw counts and restart counts of the most
    recently started stream are readable live as :attr:`last_draw_counts`
    and :attr:`last_epochs`.
    """

    def __init__(
        self,
        sources: dict[str, ShardedDataSource],
        weights: dict[str, float] | None = None,
        block_size: int = 1000,
        seed: int | None = None,
        stop_strategy: str = "restart",
        weight_scheduler: WeightScheduler | None = None,
        reshuffle_on_restart: bool = False,
        attach_dataset_ids: bool = False,
        draw_log_interval: int = DEFAULT_DRAW_LOG_INTERVAL,
    ):
        """Validate inputs, normalise weights, and capture state without iterating.

        Args:
            sources: ``{dataset_name: ShardedDataSource}`` mapping
                whose order defines the canonical name list.
            weights: Static per-dataset weights; must have the
                exact same key set as ``sources``. ``None`` falls
                back to uniform mixing.
            block_size: Number of examples produced per mixing
                block; larger blocks improve throughput at the cost
                of mixing granularity.
            seed: Master RNG seed; per-block seeds are derived from
                ``seed + block_idx`` so iteration is deterministic
                yet uncorrelated across blocks. ``None`` uses NumPy
                default randomness (non-deterministic).
            stop_strategy: One of :data:`STOP_STRATEGIES` (``"restart"``,
                ``"first_exhausted"``, ``"all_exhausted"``, ``"error"``);
                see class docstring.
            weight_scheduler: Optional :class:`WeightScheduler`. When
                set, ``weights`` are recomputed per block from the
                scheduler instead of using the static map. Its dataset
                names must equal the source names exactly.
            reshuffle_on_restart: Visit a restarted constituent's shards in
                a fresh deterministic per-epoch permutation. Only meaningful
                with ``stop_strategy="restart"``. Off by default: trainer resume
                restores the mixture from an emitted-row count only, so turning it
                on for a run that was checkpointed without it would replay a
                different order past the first wrap. An unseeded mixer draws its
                seed once at construction and records it in
                :class:`MixedShardState`, so resume stays exact.
            attach_dataset_ids: Stamp :data:`DATASET_ID_FIELD` (the
                constituent's position in ``sources``) on every emitted
                row, next to ``"__source__"``.
            draw_log_interval: Emit an INFO line with the realized
                per-source draw counts every this many emitted rows;
                ``0`` disables it.

        Raises:
            ValueError: When ``weights`` keys don't match ``sources`` keys,
                a weight is negative, the weights sum to a non-positive
                value, ``stop_strategy`` is unknown, the scheduler names
                differ from the sources, or a positive-weight source would
                receive zero rows per block (starvation).
        """
        if not sources:
            raise ValueError("No datasets to mix")
        if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if stop_strategy not in STOP_STRATEGIES:
            raise ValueError(f"stop_strategy must be one of {STOP_STRATEGIES}, got {stop_strategy!r}")
        if weight_scheduler is not None and set(weight_scheduler.dataset_names) != set(sources.keys()):
            raise ValueError(
                "weight_scheduler dataset names must match source names exactly; schedule has "
                f"{sorted(weight_scheduler.dataset_names)}, sources are {sorted(sources.keys())}."
            )
        if int(draw_log_interval) < 0:
            raise ValueError("draw_log_interval must be >= 0")
        self._sources = sources
        self._names = list(sources.keys())
        self._block_size = block_size
        # Draw one seed per source, not one per block traversal, so even an
        # initially unseeded mix can be checkpointed and replayed exactly.
        self._seed = seed if seed is not None else int(np.random.SeedSequence().entropy)
        self._stop_strategy = stop_strategy
        self._weight_scheduler = weight_scheduler
        self._reshuffle_on_restart = bool(reshuffle_on_restart)
        self._attach_dataset_ids = bool(attach_dataset_ids)
        self._dataset_ids = {name: index for index, name in enumerate(self._names)}
        self._draw_log_interval = int(draw_log_interval)
        self.last_draw_counts: dict[str, int] = dict.fromkeys(self._names, 0)
        self.last_epochs: dict[str, int] = dict.fromkeys(self._names, 0)

        if weights is None:
            n = len(self._names)
            self._weights = {name: 1.0 / n for name in self._names}
        else:
            missing = set(self._names) - set(weights.keys())
            if missing:
                raise ValueError(f"Weight keys must match source names. Missing: {sorted(missing)}")
            extra = set(weights.keys()) - set(self._names)
            if extra:
                raise ValueError(f"Weight keys must match source names. Extra: {sorted(extra)}")
            negative = {name: weights[name] for name in self._names if float(weights[name]) < 0}
            if negative:
                # A negative weight can pass a positive-sum check but corrupts block arithmetic.
                raise ValueError(f"Mixture weights must be non-negative; got {negative}.")
            total = float(sum(weights[name] for name in self._names))
            if total <= 0:
                raise ValueError("Sum of mixture weights must be > 0.")
            self._weights = {name: float(weights[name]) / total for name in self._names}

        # Starvation only excludes a source permanently under the recycling strategies.
        base_counts = self._compute_counts(self._weights)
        if self._stop_strategy in ("restart", "error"):
            self._raise_if_starved(self._weights, base_counts, what="static weights")
            if weight_scheduler is not None:
                # User-specified knots are validated; transient interpolated values between knots are not.
                for point in weight_scheduler._schedule:
                    self._raise_if_starved(point.weights, self._compute_counts(point.weights), what=f"knot {point.step}")
        for name in self._names if weight_scheduler is None else ():
            exact = self._weights[name] * float(block_size)
            if exact > 0:
                rel_bias = abs(base_counts[name] - exact) / exact
                if rel_bias > 0.02:
                    logger.warning(
                        "Mixture rounding bias: source %r gets %d rows/block vs exact %.3f (%.1f%% relative, "
                        "same sign every block). Raise block_size to shrink it.",
                        name,
                        base_counts[name],
                        exact,
                        100.0 * rel_bias,
                    )

    @property
    def shard_names(self) -> "Sequence[str]":
        """Return a single synthetic shard name for the virtual mix.

        Returns:
            One-element list ``["mixed_shard_0"]``.
        """
        # Return a synthetic shard name since this is a virtual mix
        return ["mixed_shard_0"]

    def num_shards(self) -> int:
        """Return the constant shard count of one.

        Returns:
            Always ``1``.
        """
        return 1

    def _get_weights_for_step(self, step: int) -> dict[str, float]:
        """Resolve mixing weights for a training step.

        Args:
            step: Training step index.

        Returns:
            Dictionary mapping dataset name to weight, sourced from the
            scheduler when available, else the static weights.
        """
        if self._weight_scheduler is not None:
            return self._weight_scheduler.get_weights(step)
        return self._weights

    def _compute_counts(self, weights: dict[str, float]) -> dict[str, int]:
        """Convert normalized weights to integer per-dataset block counts.

        Args:
            weights: Per-dataset weights (re-normalized internally).

        Returns:
            Dictionary mapping dataset name to integer count whose sum
            equals ``self._block_size``.
        """
        ws = np.array([weights.get(name, 0) for name in self._names], dtype=np.float64)
        if not np.all(np.isfinite(ws)) or np.any(ws < 0) or ws.sum() <= 0:
            raise ValueError("Mixture weights must be finite, nonnegative and have a positive sum")
        ws = ws / ws.sum()

        # Largest-remainder rounding: giving the whole remainder to argmax(ws)
        # would over-sample the heaviest dataset by up to len(sources)-1 rows per block.
        counts_arr = largest_remainder_counts(ws, self._block_size)

        return {name: int(counts_arr[i]) for i, name in enumerate(self._names)}

    def _raise_if_starved(self, weights: dict[str, float], counts: dict[str, int], *, what: str) -> None:
        """Refuse a weight vector under which a positive-weight source gets zero rows per block.

        Largest-remainder rounding gives a source with ``weight * block_size < 1`` zero rows in every
        block, which under a recycling strategy is silent, permanent exclusion of that dataset.

        Args:
            weights: Per-dataset weights (any positive scale).
            counts: The rounded per-block counts for ``weights``.
            what: Label for the error message (``"static weights"`` or a schedule knot).

        Raises:
            ValueError: When at least one positive-weight source rounds to zero rows.
        """
        starved = [name for name in self._names if float(weights.get(name, 0)) > 0 and counts.get(name, 0) == 0]
        if starved:
            raise ValueError(
                f"Mixture starvation ({what}): sources {starved} would receive ZERO rows per block "
                f"(weight * block_size < 1 at block_size={self._block_size}). Raise block_size, raise the "
                "weight, or drop the source."
            )

    def _block_source_ids(self, block_idx: int, examples_emitted: int, excluded: set[str] | None = None) -> list[str]:
        """Build a deterministic block, redistributing finite mixes over live sources.

        Zero-weight sources are excluded. If every currently eligible source is
        exhausted, return an empty block rather than spinning: an emitted-row
        schedule cannot advance to a future knot without emitting another row.
        """
        weights = self._get_weights_for_step(examples_emitted)
        if excluded:
            weights = {name: weight for name, weight in weights.items() if name not in excluded}
        if not any(weights.get(name, 0) > 0 for name in self._names):
            return []
        counts = self._compute_counts(weights)

        ids = []
        for name, count in counts.items():
            ids.extend([name] * count)

        block_seed = self._seed + block_idx if self._seed is not None else None
        block_rng = np.random.default_rng(block_seed)
        block_rng.shuffle(ids)
        return ids

    def _source_offsets_for_row(self, row: int) -> dict[str, int]:
        """Return per-source rows consumed before virtual mixed row ``row``."""
        row = max(int(row), 0)
        offsets = {name: 0 for name in self._names}
        full_blocks, partial = divmod(row, self._block_size)

        for block_idx in range(full_blocks):
            block_start = block_idx * self._block_size
            counts = self._compute_counts(self._get_weights_for_step(block_start))
            for name, count in counts.items():
                offsets[name] += count

        if partial:
            block_start = full_blocks * self._block_size
            for name in self._block_source_ids(full_blocks, block_start)[:partial]:
                offsets[name] += 1

        return offsets

    def resume_state(self, row: int) -> MixedShardState:
        """Capture the next emitted position after ``row`` mixed examples.

        Replays the source to discover exact delivered shard/row positions, including
        empty constituents, filtering readers and finite stop strategies. The marker
        includes the effective seed so it also resumes on a newly constructed mixer
        when the original mix was unseeded. Replay costs O(row) reads.

        Args:
            row: Number of mixed examples already emitted.

        Returns:
            MixedShardState: Marker for the next example. An exhausted stream has
            ``source_index == -1`` and resumes as an empty iterator.
        """
        row = max(int(row), 0)
        stream = self._iter_positioned()
        try:
            for _ in range(row):
                if next(stream, None) is None:
                    return MixedShardState(-1, 0, 0, examples_yielded=row, seed=self._seed)
            item = next(stream, None)
            if item is None:
                return MixedShardState(-1, 0, 0, examples_yielded=row, seed=self._seed)
            return item[1]
        finally:
            stream.close()

    def open_shard_from_state(self, state: MixedShardState) -> "Iterator[dict]":
        """Continue the mixed stream from a captured :class:`MixedShardState`.

        Args:
            state: Marker captured by :meth:`resume_state`; its emitted count
                and effective seed identify the replay position.

        Returns:
            Iterator[dict]: Mixed rows starting at the marked position.
        """
        if state.source_index == -1:
            return iter(())
        resumed = copy.copy(self)
        if state.seed is not None:
            resumed._seed = state.seed
        return resumed.open_shard_at_row(self.shard_names[0], state.examples_yielded)

    def _empty_constituents(self) -> set[str]:
        """Identify constituents that can never yield a single row.

        Under ``stop_strategy="restart"`` a constituent is permanently
        dead exactly when it is empty, and that is decided by its very
        first draw, so the dead set can be resolved up front:
        sources with trustworthy positive metadata row counts are
        provably live, everything else gets one probe draw on a
        fresh iterator (no shared iteration state is touched).

        Returns:
            set[str]: Names of constituents with no rows at all.
        """
        empty: set[str] = set()
        for name, source in self._sources.items():
            known = self._known_source_rows(source, list(source.shard_names))
            if getattr(source, "delivers_all_metadata_rows", False) and known is not None and known[1] > 0:
                continue
            probe = self._chain_shards(source)
            try:
                if next(probe, _PROBE_EMPTY) is _PROBE_EMPTY:
                    empty.add(name)
            finally:
                probe.close()
        return empty

    def _known_shard_rows(self, source: ShardedDataSource, shard_name: str) -> int | None:
        """Best-effort exact row count for one constituent shard."""
        try:
            info = source.get_shard_info(shard_name)
        except Exception:
            return None
        if info is None or info.num_rows is None:
            return None
        return max(int(info.num_rows), 0)

    def _known_source_rows(
        self, source: ShardedDataSource, shard_names: list[str]
    ) -> tuple[list[tuple[str, int]], int] | None:
        """Return exact per-shard and total row counts when all are known."""
        rows_by_shard = []
        total = 0
        for shard_name in shard_names:
            rows = self._known_shard_rows(source, shard_name)
            if rows is None:
                return None
            rows_by_shard.append((shard_name, rows))
            total += rows
        return rows_by_shard, total

    def open_shard(self, _shard_name: str) -> "Iterator[dict]":
        """Drive the block-mixer until either ``stop_strategy`` halts it or all sources die.

        Maintains one chained iterator per constituent (built by
        :meth:`_chain_shards`) and walks them in block-randomised
        order. Each emitted example is augmented with a
        ``"__source__"`` key naming the dataset of origin, so
        downstream consumers can apply per-source logic if they
        need to. Per-block deterministic RNG means re-running with
        the same seed produces an identical interleaving.

        Args:
            _shard_name: Ignored — :class:`MixedShardedSource`
                exposes only one synthetic shard.

        Yields:
            dict: Mixed rows with the extra ``"__source__"`` field.
            Iteration ends per the configured ``stop_strategy``.
        """
        stream = self._iter_positioned(capture_state=False)
        try:
            for example, _ in stream:
                yield example
        finally:
            stream.close()

    def _iter_positioned(self, capture_state: bool = True) -> "Iterator[tuple[dict, MixedShardState | None]]":
        """Drive the mixer with exact delivered positions for checkpoint capture.

        Ordinary iteration disables marker allocation; checkpoint capture enables it.
        """

        def constituent(source, name, epoch):
            canonical = {shard_name: index for index, shard_name in enumerate(source.shard_names)}
            for shard_name in self._epoch_shard_order(source, name=name, epoch=epoch):
                shard_index = canonical[shard_name]
                rows = iter(source.open_shard(shard_name))
                try:
                    for row_index, example in enumerate(rows):
                        yield example, shard_index, row_index
                finally:
                    close = getattr(rows, "close", None)
                    if close is not None:
                        close()

        epochs: dict[str, int] = dict.fromkeys(self._names, 0)
        draw_counts: dict[str, int] = dict.fromkeys(self._names, 0)
        if not capture_state:
            # Live, per-stream accounting: readable while the stream runs, describing the most recently
            # started consumer stream of this instance (checkpoint replays do not overwrite it).
            self.last_draw_counts = draw_counts
            self.last_epochs = epochs
        next_log_rows = self._draw_log_interval if not capture_state else 0
        iters = {name: constituent(source, name, 0) for name, source in self._sources.items()}
        block_idx = 0
        examples_emitted = 0
        dead: set[str] = set()
        source_indices = {name: index for index, name in enumerate(self._names)}
        try:
            while True:
                # Schedules use emitted examples, not planned ids: dead sources
                # and finite stop strategies can make a block emit fewer rows.
                ids = self._block_source_ids(block_idx, examples_emitted, excluded=dead)
                if not ids:
                    return
                for name in ids:
                    if name in dead:
                        continue
                    item = next(iters[name], None)
                    if item is None:
                        if self._stop_strategy == "first_exhausted":
                            return
                        if self._stop_strategy == "error":
                            raise RuntimeError(
                                f"Mixture constituent '{name}' exhausted after {examples_emitted} emitted examples "
                                "(stop_strategy='error'). If repeating this dataset is intended, use "
                                "stop_strategy='restart'."
                            )
                        if self._stop_strategy == "restart":
                            iters[name].close()
                            epochs[name] += 1
                            iters[name] = constituent(self._sources[name], name, epochs[name])
                            item = next(iters[name], None)
                        if item is None:
                            dead.add(name)
                            continue
                    example, shard_index, row_index = item
                    state = (
                        MixedShardState(
                            source_index=source_indices[name],
                            shard_index=shard_index,
                            row_index=row_index,
                            block_index=block_idx,
                            examples_yielded=examples_emitted,
                            seed=self._seed,
                        )
                        if capture_state
                        else None
                    )
                    # Counted before the yield so a consumer stopping mid-block still sees exact counts.
                    draw_counts[name] += 1
                    yield self._tag(example, name), state
                    examples_emitted += 1
                if len(dead) == len(self._names):
                    return
                block_idx += 1
                if next_log_rows and examples_emitted >= next_log_rows:
                    next_log_rows = examples_emitted + self._draw_log_interval
                    self._log_draws(block_idx, examples_emitted, draw_counts, epochs)
        finally:
            for stream in iters.values():
                stream.close()

    def _tag(self, example: dict, name: str) -> dict:
        """Return a copy of ``example`` carrying its provenance (``__source__`` and optionally the dataset id)."""
        tagged = {**example, "__source__": name}
        if self._attach_dataset_ids:
            tagged[DATASET_ID_FIELD] = self._dataset_ids[name]
        return tagged

    @staticmethod
    def _log_draws(block_idx: int, rows: int, draw_counts: dict[str, int], epochs: dict[str, int]) -> None:
        """Log the realized per-source draw counts and restart counts of a running stream."""
        logger.info(
            "mixture draws after %d blocks (%d rows): %s | source epochs: %s",
            block_idx,
            rows,
            dict(sorted(draw_counts.items())),
            {k: v for k, v in sorted(epochs.items()) if v},
        )

    def open_shard_at_row(self, _shard_name: str, row: int) -> "Iterator[dict]":
        """Open the virtual mixed shard after ``row`` emitted examples."""
        row = max(int(row), 0)
        if row == 0:
            yield from self.open_shard(_shard_name)
            return

        if self._stop_strategy != "restart":
            logger.warning(
                "Seeked resume for MixedShardedSource is only exact for stop_strategy='restart'; "
                "falling back to sequential skip of %d mixed examples.",
                row,
            )
            yield from itertools.islice(self.open_shard(_shard_name), row, None)
            return

        empty = self._empty_constituents()
        if empty:
            # The block-offset fast path below assumes every planned id yields exactly
            # one row. With a dead (empty) constituent the fresh path skips those ids
            # without emitting, so an emitted-example count no longer lines up with
            # block boundaries and the fast path would replay rows that were already
            # served. Replay sequentially instead: it walks the exact fresh path.
            logger.warning(
                "MixedShardedSource resume falls back to sequential skip of %d mixed examples "
                "because constituent(s) %s are empty.",
                row,
                sorted(empty),
            )
            yield from itertools.islice(self.open_shard(_shard_name), row, None)
            return

        # For an inexact source without finite shard metadata we cannot
        # determine its restart period. Replaying the mixed stream preserves
        # both its per-source offset and the weight schedule's emitted count.
        if any(
            not getattr(source, "delivers_all_metadata_rows", False)
            and self._known_source_rows(source, list(source.shard_names)) is None
            for source in self._sources.values()
        ):
            yield from itertools.islice(self.open_shard(_shard_name), row, None)
            return

        offsets = self._source_offsets_for_row(row)
        # Each constituent's seek records the epoch (restart count) it landed in, so post-resume restarts
        # continue the epoch sequence (and, with reshuffle_on_restart, the permutation sequence).
        epochs: dict[str, int] = dict.fromkeys(self._names, 0)
        iters = {
            name: self._chain_shards_at_row(self._sources[name], offsets[name], name=name, epochs=epochs)
            for name in self._names
        }

        block_idx, partial = divmod(row, self._block_size)
        dead: set[str] = set()

        while True:
            block_start = block_idx * self._block_size
            ids = self._block_source_ids(block_idx, block_start)
            if partial:
                ids = ids[partial:]
                partial = 0

            for name in ids:
                if name in dead:
                    continue
                try:
                    example = next(iters[name])
                    yield self._tag(example, name)
                except StopIteration:
                    epochs[name] += 1
                    iters[name] = self._chain_epoch(self._sources[name], name=name, epoch=epochs[name])
                    try:
                        example = next(iters[name])
                        yield self._tag(example, name)
                    except StopIteration:
                        logger.warning(f"Dataset '{name}' is empty")
                        dead.add(name)

            if len(dead) == len(self._names):
                return

            block_idx += 1

    def _epoch_shard_order(self, source: ShardedDataSource, *, name: str, epoch: int) -> list[str]:
        """Shard visit order for one epoch (restart count) of one constituent.

        Epoch 0 always uses canonical order. Later epochs return a deterministic permutation seeded by
        ``(seed, crc32(name), epoch)`` when ``reshuffle_on_restart`` is set, canonical order otherwise.

        Args:
            source: The constituent source.
            name: Constituent name (part of the permutation seed).
            epoch: 0-based restart count.

        Returns:
            list[str]: Shard names in visit order.
        """
        shard_names = list(source.shard_names)
        if not self._reshuffle_on_restart or epoch <= 0 or len(shard_names) < 2:
            return shard_names
        rng = np.random.default_rng((int(self._seed), zlib.crc32(name.encode("utf-8")), int(epoch)))
        rng.shuffle(shard_names)
        return shard_names

    def _chain_epoch(self, source: ShardedDataSource, *, name: str, epoch: int) -> "Iterator[dict]":
        """Chain ``source`` for one epoch, in that epoch's shard order.

        Canonical-order epochs go through :meth:`_chain_shards` so subclasses overriding it keep
        working; permuted epochs open the shards in :meth:`_epoch_shard_order`.
        """
        if not self._reshuffle_on_restart or epoch <= 0 or len(source.shard_names) < 2:
            return self._chain_shards(source)
        return self._chain_ordered(source, self._epoch_shard_order(source, name=name, epoch=epoch))

    @staticmethod
    def _chain_ordered(source: ShardedDataSource, order: list[str]) -> "Iterator[dict]":
        """Yield every row of the shards in ``order``, one shard after another."""
        for shard_name in order:
            yield from source.open_shard(shard_name)

    def _chain_shards(self, source: ShardedDataSource) -> "Iterator[dict]":
        """Chain every shard of ``source`` into one continuous iterator.

        Args:
            source: Source whose shards should be flattened.

        Yields:
            Examples concatenated across all shards in ``source.shard_names``
            order.
        """
        for shard_name in source.shard_names:
            yield from source.open_shard(shard_name)

    def _chain_shards_at_row(
        self,
        source: ShardedDataSource,
        row: int,
        *,
        name: str = "",
        epochs: dict[str, int] | None = None,
    ) -> "Iterator[dict]":
        """Chain ``source`` shards starting at a global (possibly wrapped) row offset.

        Args:
            source: The constituent source.
            row: Rows of this constituent consumed before the resume point (may exceed one pass).
            name: Constituent name; seeds per-epoch shard permutations.
            epochs: Optional dict updated with ``epochs[name] = <epoch the seek landed in>`` once known.
        """
        row = max(int(row), 0)
        shard_names = list(source.shard_names)

        def _record(epoch: int) -> None:
            if epochs is not None:
                epochs[name] = int(epoch)

        if row == 0:
            _record(0)
            yield from self._chain_shards(source)
            return

        known_rows = self._known_source_rows(source, shard_names)
        if known_rows is None:
            logger.warning(
                "Cannot seek mixed constituent %s by metadata; falling back to sequential skip of %d rows.",
                type(source).__name__,
                row,
            )

        # Without metadata, or when metadata may over-count delivered rows, skip lazily. Only this
        # restart-mode path calls here, so a skip that crosses the end of the constituent must wrap
        # to ``row % period`` (r11 S13-1: a finite islice restarted the cycle from its first row).
        if known_rows is None or not getattr(source, "delivers_all_metadata_rows", False):
            # Skip delivered rows lazily: ordinary seeks must not count the
            # entire dataset just to learn a restart period they never cross.
            _record(0)
            stream = self._chain_shards(source)
            consumed = 0
            try:
                while consumed < row:
                    try:
                        next(stream)
                    except StopIteration:
                        if consumed == 0:
                            return
                        # We crossed the end of one delivered cycle. Only now
                        # is its true length known; restart at the residual
                        # offset (in that epoch's shard order) rather than
                        # trusting physical row metadata.
                        stream.close()
                        epoch, residual = divmod(row, consumed)
                        _record(epoch)
                        stream = self._chain_epoch(source, name=name, epoch=epoch)
                        for _ in range(residual):
                            next(stream)
                        break
                    consumed += 1
                yield from stream
            finally:
                stream.close()
            return

        rows_by_shard, total_rows = known_rows
        if total_rows <= 0:
            _record(0)
            return

        epoch = 0
        if self._stop_strategy == "restart":
            epoch, row = divmod(row, total_rows)
        _record(epoch)

        # Apply this epoch's shard order to both the seek arithmetic and the visit order, so a resume
        # into a reshuffled epoch lands on exactly the row the original stream would produce next.
        rows_map = dict(rows_by_shard)
        order = self._epoch_shard_order(source, name=name, epoch=epoch)
        remaining = row
        started = False
        for shard_name, rows in ((shard, rows_map[shard]) for shard in order):
            if not started:
                if remaining >= rows:
                    remaining -= rows
                    continue
                yield from source.open_shard_at_row(shard_name, remaining)
                started = True
                continue
            yield from source.open_shard(shard_name)

    def __len__(self) -> int:
        """Sum of each constituent's length — note this is a notional length only.

        With ``stop_strategy="restart"`` actual iteration is
        infinite (constituents are recycled), so this value should
        be treated as the "one-pass" length rather than the
        iteration cap.

        Returns:
            int: ``sum(len(source) for source in self._sources.values())``.

        Raises:
            TypeError: If any constituent does not support
                ``len()`` (i.e. is streaming).
        """
        return sum(len(source) for source in self._sources.values())

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"MixedShardedSource(sources=N, weights=[...], block_size=B)"``.
        """
        weights_str = ", ".join(f"{k}={v:.2f}" for k, v in self._weights.items())
        return (
            f"MixedShardedSource(sources={len(self._sources)}, weights=[{weights_str}], block_size={self._block_size})"
        )


class MixStage(BaseStage):
    """Pipeline stage that collapses ``{name: source}`` into a single mixed source.

    On a multi-dataset pipeline, instantiates a
    :class:`MixedShardedSource` configured with the static weights
    or the dynamic schedule from :class:`MixStageConfig` and replaces
    the rolling source dict with ``{"mixed": mixed_source}``. On a
    single-dataset pipeline the stage is a no-op so the same code
    path can run with or without mixing.
    """

    def __init__(self, config: MixStageConfig | None = None):
        """Capture the mix configuration; defaulted to a fresh :class:`MixStageConfig` when omitted.

        Args:
            config: :class:`MixStageConfig` controlling weights,
                block size, stop strategy, RNG seed, and optional
                schedule. ``None`` produces a default config so the
                stage is constructible without arguments in tests.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or MixStageConfig()

    @property
    def name(self) -> str:
        """Stage identifier used in metric and log namespaces.

        Returns:
            str: Constant string ``"mix"``.
        """
        return "mix"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        """Build a :class:`MixedShardedSource` from the input dataset map.

        When ``data`` already contains a single entry the stage is
        a no-op (the input dict is returned unchanged) so callers
        can invoke ``mix()`` unconditionally. When the
        :class:`MixStageConfig.weight_schedule` is set, a
        :class:`WeightScheduler` is built and attached to the mixer.
        The mixer's RNG seed defaults to
        :attr:`MixStageConfig.seed`, falling back to
        ``context.seed`` when unset.

        Args:
            data: Rolling ``{dataset_name: ShardedDataSource}`` dict
                from the previous stage.
            context: Shared :class:`PipelineContext` whose
                ``seed`` is used as the fallback RNG seed.

        Returns:
            dict[str, ShardedDataSource]: A single-key dict
            ``{"mixed": MixedShardedSource}`` for multi-source
            inputs, or ``data`` itself when there's only one source.

        Raises:
            ValueError: When ``data`` is empty.
        """
        if len(data) == 0:
            raise ValueError("No datasets to mix")

        if len(data) == 1:
            # Single dataset, no mixing needed
            return data

        weight_scheduler = None
        if self._stage_config.weight_schedule:
            weight_scheduler = WeightScheduler(
                schedule=self._stage_config.weight_schedule,
                interpolation=self._stage_config.weight_schedule_type,
            )

        mixed = MixedShardedSource(
            sources=data,
            weights=self._stage_config.weights,
            block_size=self._stage_config.block_size,
            seed=self._stage_config.seed if self._stage_config.seed is not None else context.seed,
            stop_strategy=self._stage_config.stop_strategy,
            weight_scheduler=weight_scheduler,
            reshuffle_on_restart=self._stage_config.reshuffle_on_restart,
            attach_dataset_ids=self._stage_config.attach_dataset_ids,
        )

        logger.info(f"Mixed {len(data)} datasets with block_size={self._stage_config.block_size}")
        return {"mixed": mixed}


def block_mixture_interleave(
    datasets: dict[str, tp.Any],
    weights: dict[str, float] | None = None,
    block_size: int = 1000,
    seed: int | None = 42,
    stop: str = "restart",
    attach_dataset_ids: bool = False,
):
    """Create a deterministic block-based mixture of multiple datasets.

    This function implements a block-based dataset mixing strategy where examples
    from different datasets are interleaved in fixed-size blocks with specified
    proportions. This approach ensures deterministic and restart-friendly data
    loading for distributed training.

    Args:
        datasets: Dict mapping names to datasets: {"code": code_ds, "text": text_ds}
        weights: Dict mapping names to weights: {"code": 0.3, "text": 0.7}
            Keys must match dataset keys. If None, datasets are mixed equally.
        block_size: Number of examples per block.
        seed: Random seed for deterministic shuffling within blocks.
        stop: Strategy when a dataset is exhausted: ``"restart"`` loops it forever (ratios preserved);
            ``"first_exhausted"`` ends the stream at the first exhaustion; ``"all_exhausted"`` keeps
            recycling exhausted datasets until every dataset has finished at least once; ``"error"``
            raises :class:`RuntimeError` so a run can never silently train off-ratio.
        attach_dataset_ids: Stamp :data:`DATASET_ID_FIELD` (the dataset's position in ``datasets``)
            on every yielded dict row.

    Returns:
        IterableDataset that yields examples from the mixed datasets.

    Raises:
        ValueError: If no datasets are provided, weight keys don't match dataset keys, a weight is
            negative, the weights do not have a positive finite sum, ``stop`` is unknown, or a
            positive-weight dataset would receive zero rows per block (starvation).

    Example:
        >>> mixed = block_mixture_interleave(
        ...     datasets={"code": code_ds, "text": text_ds},
        ...     weights={"code": 0.3, "text": 0.7},
        ...     block_size=1000,
        ...     seed=42,
        ...     stop="restart"
        ... )
        >>>
        >>> # Equal weights
        >>> mixed = block_mixture_interleave(
        ...     datasets={"code": code_ds, "text": text_ds},
        ...     weights=None,  # Equal 50/50
        ... )
    """
    from datasets import IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    if not isinstance(datasets, dict):
        raise TypeError(f"datasets must be a dict, got {type(datasets).__name__}")

    dataset_names = list(datasets.keys())
    datasets_list = list(datasets.values())
    n = len(datasets_list)

    if n == 0:
        raise ValueError("No datasets to mix")
    if stop not in STOP_STRATEGIES:
        raise ValueError(f"stop must be one of {STOP_STRATEGIES}, got {stop!r}")

    if weights is not None:
        if not isinstance(weights, dict):
            raise TypeError(f"weights must be a dict or None, got {type(weights).__name__}")
        missing = set(dataset_names) - set(weights.keys())
        if missing:
            raise ValueError(f"Weight keys must match dataset keys. Missing: {missing}")
        extra = set(weights.keys()) - set(dataset_names)
        if extra:
            raise ValueError(f"Weight keys must match dataset keys. Extra: {extra}")
        ws = np.array([weights[name] for name in dataset_names], dtype=np.float64)
        if (ws < 0).any() or not np.isfinite(ws).all() or ws.sum() <= 0:
            # Negative weights make blocks emit != block_size rows; zero/NaN sums floor-cast to garbage.
            raise ValueError(f"Mixture weights must be non-negative with a positive sum; got {weights}.")
        ws = ws / ws.sum()
    else:
        ws = np.ones(n, dtype=np.float64) / n

    # Largest-remainder rounding, shared with MixedShardedSource._compute_counts.
    counts = largest_remainder_counts(ws, block_size)
    starved = [dataset_names[i] for i in range(n) if ws[i] > 0 and counts[i] == 0]
    if starved:
        # weight * block_size < 1 gives ZERO rows in every block: silent, permanent exclusion.
        raise ValueError(
            f"Mixture starvation: datasets {starved} would receive ZERO rows per block "
            f"(weight * block_size < 1 at block_size={block_size}). Raise block_size, raise the weight, "
            "or drop the dataset."
        )
    # Zero-weight datasets never draw, so they cannot gate all_exhausted termination.
    positive = {i for i in range(n) if counts[i] > 0}

    def gen():
        """Inline closure that drives the deterministic block-based interleaver.

        Captures ``datasets_list``, ``counts``, ``seed``, and
        ``stop`` from :func:`block_mixture_interleave`'s scope.
        Produces blocks of ``block_size`` examples by picking
        ``counts[i]`` rows from each input dataset, shuffling the
        per-block ordering with a seed of ``seed + block_idx`` so
        each block is reproducible independently. End-of-source is
        handled per the ``stop`` flag: ``"restart"`` recycles the
        constituent forever; ``"all_exhausted"`` recycles it but ends
        the stream once every drawn constituent has exhausted at least
        once; ``"error"`` raises; ``"first_exhausted"`` ends the stream
        at the first exhaustion.

        Yields:
            Any: Each yielded example is whatever the underlying
            datasets produced (typically dict rows).

        Raises:
            ValueError: When a recycling strategy (``"restart"`` /
                ``"all_exhausted"``) meets a constituent with zero
                examples (it cannot legitimately be restarted).
            RuntimeError: When ``stop="error"`` and any constituent
                exhausts.
        """
        iters = [iter(ds) for ds in datasets_list]
        exhausted: set[int] = set()
        block_idx = 0

        def _tag(example, i):
            if attach_dataset_ids and isinstance(example, dict):
                return {**example, DATASET_ID_FIELD: i}
            return example

        def _recycle(i):
            iters[i] = iter(datasets_list[i])
            try:
                return next(iters[i])
            except StopIteration as e:
                raise ValueError(f"Dataset at index {i} is empty and cannot be restarted") from e

        while True:
            block_rng = np.random.default_rng(seed + block_idx if seed is not None else None)
            ids = []
            for i, c in enumerate(counts):
                ids.extend([i] * c)
            block_rng.shuffle(ids)
            for i in ids:
                try:
                    yield _tag(next(iters[i]), i)
                except StopIteration:
                    if stop == "restart":
                        yield _tag(_recycle(i), i)
                    elif stop == "error":
                        raise RuntimeError(
                            f"stop_strategy='error': dataset {dataset_names[i]!r} exhausted during block "
                            f"{block_idx}; refusing to continue with a drifted mixture."
                        ) from None
                    elif stop == "all_exhausted":
                        exhausted.add(i)
                        if exhausted >= positive:
                            return
                        yield _tag(_recycle(i), i)
                    else:  # first_exhausted
                        return
            block_idx += 1

    return IterableDataset.from_generator(gen)
