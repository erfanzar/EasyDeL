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
"""GRPO replay-buffer trainer extension."""

from __future__ import annotations

import heapq

import jax
import numpy as np


class _ReplayBuffer:
    """Score-prioritized host-side replay buffer for grouped GRPO batches.

    The buffer keeps the highest-scored groups up to ``max_size`` using a heap.
    Sampling is host-side and score-weighted when scores are finite and
    positive, with uniform fallback for degenerate score distributions.
    """

    def __init__(self, max_size: int, seed: int | None = None) -> None:
        """Create an empty replay heap.

        Args:
            max_size: Maximum number of replay groups retained. Values less
                than or equal to zero disable storage while keeping method calls
                valid.
            seed: Optional NumPy RNG seed used for deterministic sampling in
                tests or reproducible trainer runs.
        """
        self.max_size = int(max_size)
        self._heap: list[tuple[float, int, dict[str, jax.Array]]] = []
        self._counter = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of replay groups currently retained.

        The value reflects stored groups, not individual token rows. It is used
        by trainer-side scheduling and tests to decide whether replay sampling
        can contribute examples to the next GRPO batch.
        """
        return len(self._heap)

    def add(self, score: float, data: dict[str, jax.Array]) -> None:
        """Insert one replay group, evicting the lowest-scored item if full.

        A disabled buffer (``max_size <= 0``) ignores inserts. The counter is
        stored with each item to make heap ordering stable when two groups have
        the same score.
        """
        if self.max_size <= 0:
            return
        item = (float(score), self._counter, data)
        self._counter += 1
        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, item)
        elif item[0] > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def sample(self, num_samples: int) -> list[dict[str, jax.Array]]:
        """Sample replay groups, weighting by stored score when scores are usable.

        If the requested count exceeds the number of stored groups, sampling is
        performed with replacement. Non-finite, all-zero, or negative-only score
        sets fall back to NumPy's uniform sampling behavior.
        """
        if not self._heap or num_samples <= 0:
            return []
        scores = np.asarray([max(item[0], 0.0) for item in self._heap], dtype=np.float64)
        if not np.isfinite(scores).all() or float(scores.sum()) <= 0.0:
            probabilities = None
        else:
            probabilities = scores / scores.sum()
        replace = num_samples > len(self._heap)
        indices = self._rng.choice(len(self._heap), size=num_samples, replace=replace, p=probabilities)
        return [self._heap[int(index)][2] for index in np.atleast_1d(indices)]
