# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
    """Dynamic weight scheduler for dataset mixing.

    Supports step-function, linear, and cosine interpolation between
    weight checkpoints during training.

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
        """Initialize WeightScheduler.

        Args:
            schedule: List of weight schedule points, sorted by step.
            interpolation: Interpolation type - "step", "linear", or "cosine".
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
        """Get weights for a specific training step.

        Args:
            step: Current training step.

        Returns:
            Dictionary mapping dataset names to weights.
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
        """Get the list of dataset names in the schedule."""
        return self._dataset_names.copy()


@dataclass
class MixedShardState:
    """State for tracking position in mixed iteration."""

    source_index: int
    shard_index: int
    row_index: int
    block_index: int = 0
    examples_yielded: int = 0


class MixedShardedSource(ShardedDataSource[dict]):
    """Sharded source that mixes multiple sources with weights.

    Implements block-based deterministic mixing for reproducibility.
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
        """Initialize MixedShardedSource.

        Args:
            sources: Dictionary mapping dataset names to sources.
            weights: Static weights per dataset (if no scheduler).
            block_size: Number of examples per mixing block.
            seed: Random seed for deterministic mixing.
            stop_strategy: What to do when exhausted - "restart", "first_exhausted", "all_exhausted".
            weight_scheduler: Optional dynamic weight scheduler.
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
            total = sum(weights.values())
            self._weights = {k: v / total for k, v in weights.items()}

    @property
    def shard_names(self) -> "Sequence[str]":
        # Return a synthetic shard name since this is a virtual mix
        return ["mixed_shard_0"]

    def num_shards(self) -> int:
        return 1

    def _get_weights_for_step(self, step: int) -> dict[str, float]:
        """Get weights for a specific step (uses scheduler if available)."""
        if self._weight_scheduler is not None:
            return self._weight_scheduler.get_weights(step)
        return self._weights

    def _compute_counts(self, weights: dict[str, float]) -> dict[str, int]:
        """Compute per-dataset counts for a block."""
        ws = np.array([weights.get(name, 0) for name in self._names], dtype=np.float64)
        ws = ws / ws.sum()

        counts_arr = np.floor(ws * self._block_size).astype(int)
        remainder = self._block_size - counts_arr.sum()
        if remainder > 0:
            counts_arr[np.argmax(ws)] += remainder

        return {name: int(counts_arr[i]) for i, name in enumerate(self._names)}

    def open_shard(self, _shard_name: str) -> "Iterator[dict]":
        """Open the mixed shard and iterate."""
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
        """Chain all shards of a source into a single iterator."""
        for shard_name in source.shard_names:
            yield from source.open_shard(shard_name)

    def __len__(self) -> int:
        """Return total number of examples across all sources.

        Note: For 'restart' stop_strategy, this returns the sum of all source lengths.
        Actual iteration may be infinite with 'restart' strategy.
        """
        return sum(len(source) for source in self._sources.values())

    def __repr__(self) -> str:
        weights_str = ", ".join(f"{k}={v:.2f}" for k, v in self._weights.items())
        return (
            f"MixedShardedSource(sources={len(self._sources)}, weights=[{weights_str}], block_size={self._block_size})"
        )


class MixStage(BaseStage):
    """Pipeline stage for mixing multiple datasets.

    Supports static weights and dynamic weight scheduling.
    """

    def __init__(self, config: MixStageConfig | None = None):
        """Initialize MixStage.

        Args:
            config: Mixing stage configuration.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or MixStageConfig()

    @property
    def name(self) -> str:
        return "mix"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        """Mix multiple datasets into one.

        Args:
            data: Dictionary mapping dataset names to sources.
            context: Pipeline context.

        Returns:
            Dictionary with single "mixed" source.
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
    seed: int = 42,
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
    from datasets import IterableDataset

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
