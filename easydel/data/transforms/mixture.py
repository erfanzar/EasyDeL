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

import logging
import math
import typing as tp
from dataclasses import dataclass

import numpy as np

from ..core.config import MixStageConfig, WeightSchedulePoint
from ..core.protocols import BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)


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

        # Sort by step
        self._schedule = sorted(schedule, key=lambda p: p.step)
        self._interpolation = interpolation
        self._dataset_names = list(self._schedule[0].weights.keys())

        # Validate all points have same dataset names
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
            step: Training step (0-indexed) at which to evaluate the
                schedule.

        Returns:
            dict[str, float]: Mapping from dataset name to its weight
            at ``step``. The dataset name set is identical across
            steps and equals
            :attr:`WeightSchedulePoint.weights`'s keys.
        """
        # Find surrounding schedule points
        if step <= self._schedule[0].step:
            return self._schedule[0].weights.copy()

        if step >= self._schedule[-1].step:
            return self._schedule[-1].weights.copy()

        # Find the interval
        for i in range(len(self._schedule) - 1):
            if self._schedule[i].step <= step < self._schedule[i + 1].step:
                start = self._schedule[i]
                end = self._schedule[i + 1]
                break
        else:
            return self._schedule[-1].weights.copy()

        # Interpolate
        if self._interpolation == "step":
            return start.weights.copy()

        # Calculate interpolation factor
        progress = (step - start.step) / (end.step - start.step)

        if self._interpolation == "cosine":
            # Cosine annealing
            progress = (1 - math.cos(progress * math.pi)) / 2

        # Linear interpolation (or cosine-adjusted linear)
        result = {}
        for name in self._dataset_names:
            start_w = start.weights[name]
            end_w = end.weights[name]
            result[name] = start_w + progress * (end_w - start_w)

        # Normalize to ensure sum is 1
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
    """

    source_index: int
    shard_index: int
    row_index: int
    block_index: int = 0
    examples_yielded: int = 0


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
    the surviving sources until every constituent has been
    exhausted at least once. Optional :class:`WeightScheduler`
    integration replaces the static weights with curriculum-driven
    weights at each block.
    """

    def __init__(
        self,
        sources: dict[str, ShardedDataSource],
        weights: dict[str, float] | None = None,
        block_size: int = 1000,
        seed: int | None = None,
        stop_strategy: str = "restart",
        weight_scheduler: WeightScheduler | None = None,
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
            stop_strategy: ``"restart"``, ``"first_exhausted"``, or
                ``"all_exhausted"``; see class docstring.
            weight_scheduler: Optional :class:`WeightScheduler`. When
                set, ``weights`` are recomputed per block from the
                scheduler instead of using the static map.

        Raises:
            ValueError: When ``weights`` keys don't match
                ``sources`` keys, or the weights sum to a non-positive
                value.
        """
        self._sources = sources
        self._names = list(sources.keys())
        self._block_size = block_size
        self._seed = seed
        self._stop_strategy = stop_strategy
        self._weight_scheduler = weight_scheduler

        # Validate and normalize weights
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
            total = float(sum(weights[name] for name in self._names))
            if total <= 0:
                raise ValueError("Sum of mixture weights must be > 0.")
            self._weights = {name: float(weights[name]) / total for name in self._names}

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
        ws = ws / ws.sum()

        counts_arr = np.floor(ws * self._block_size).astype(int)
        remainder = self._block_size - counts_arr.sum()
        if remainder > 0:
            counts_arr[np.argmax(ws)] += remainder

        return {name: int(counts_arr[i]) for i, name in enumerate(self._names)}

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
        # Create iterators for all sources (from all their shards)
        iters = {}
        for name, source in self._sources.items():
            iters[name] = self._chain_shards(source)

        block_idx = 0
        global_step = 0

        while True:
            # Get weights for current step (for dynamic scheduling)
            weights = self._get_weights_for_step(global_step)
            counts = self._compute_counts(weights)

            # Create deterministic RNG per block
            block_seed = self._seed + block_idx if self._seed is not None else None
            block_rng = np.random.default_rng(block_seed)

            # Build block indices
            ids = []
            for name, count in counts.items():
                ids.extend([name] * count)
            block_rng.shuffle(ids)

            # Yield examples from the block
            exhausted_count = 0
            for name in ids:
                try:
                    example = next(iters[name])
                    # Add source metadata
                    example["__source__"] = name
                    yield example
                    global_step += 1
                except StopIteration:
                    if self._stop_strategy == "restart":
                        iters[name] = self._chain_shards(self._sources[name])
                        try:
                            example = next(iters[name])
                            example["__source__"] = name
                            yield example
                            global_step += 1
                        except StopIteration:
                            # Empty dataset
                            logger.warning(f"Dataset '{name}' is empty")
                            exhausted_count += 1
                    elif self._stop_strategy == "first_exhausted":
                        return
                    else:  # all_exhausted
                        exhausted_count += 1

            if exhausted_count == len(self._names):
                return

            block_idx += 1

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

        # Create weight scheduler if schedule provided
        weight_scheduler = None
        if self._stage_config.weight_schedule:
            weight_scheduler = WeightScheduler(
                schedule=self._stage_config.weight_schedule,
                interpolation=self._stage_config.weight_schedule_type,
            )

        # Create mixed source
        mixed = MixedShardedSource(
            sources=data,
            weights=self._stage_config.weights,
            block_size=self._stage_config.block_size,
            seed=self._stage_config.seed or context.seed,
            stop_strategy=self._stage_config.stop_strategy,
            weight_scheduler=weight_scheduler,
        )

        logger.info(f"Mixed {len(data)} datasets with block_size={self._stage_config.block_size}")
        return {"mixed": mixed}


def block_mixture_interleave(
    datasets: dict[str, tp.Any],
    weights: dict[str, float] | None = None,
    block_size: int = 1000,
    seed: int | None = 42,
    stop: str = "restart",
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
        stop: Strategy when a dataset is exhausted - "restart" to loop the dataset
            or "first_exhausted" to stop iteration.

    Returns:
        IterableDataset that yields examples from the mixed datasets.

    Raises:
        ValueError: If no datasets are provided or weight keys don't match dataset keys.

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
        ws = ws / ws.sum()
    else:
        ws = np.ones(n, dtype=np.float64) / n

    counts = np.floor(ws * block_size).astype(int)
    remainder = block_size - counts.sum()
    if remainder > 0:
        counts[np.argmax(ws)] += remainder

    def gen():
        """Inline closure that drives the deterministic block-based interleaver.

        Captures ``datasets_list``, ``counts``, ``seed``, and
        ``stop`` from :func:`block_mixture_interleave`'s scope.
        Produces blocks of ``block_size`` examples by picking
        ``counts[i]`` rows from each input dataset, shuffling the
        per-block ordering with a seed of ``seed + block_idx`` so
        each block is reproducible independently. End-of-source is
        handled per the ``stop`` flag: ``"restart"`` recycles the
        constituent (raising ``ValueError`` if it's truly empty),
        anything else terminates the whole generator.

        Yields:
            Any: Each yielded example is whatever the underlying
            datasets produced (typically dict rows).

        Raises:
            ValueError: When ``stop="restart"`` is requested and one
                of the constituent datasets has zero examples (so it
                cannot legitimately be restarted).
        """
        iters = [iter(ds) for ds in datasets_list]
        block_idx = 0
        while True:
            # Create deterministic RNG per block for reproducible checkpoint resume
            block_rng = np.random.default_rng(seed + block_idx if seed is not None else None)
            ids = []
            for i, c in enumerate(counts):
                ids.extend([i] * c)
            block_rng.shuffle(ids)
            for i in ids:
                try:
                    yield next(iters[i])
                except StopIteration:
                    if stop == "restart":
                        iters[i] = iter(datasets_list[i])
                        try:
                            yield next(iters[i])
                        except StopIteration as e:
                            raise ValueError(f"Dataset at index {i} is empty and cannot be restarted") from e
                    else:
                        return
            block_idx += 1

    return IterableDataset.from_generator(gen)
