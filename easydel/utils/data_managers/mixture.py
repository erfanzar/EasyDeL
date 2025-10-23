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

"""Dataset mixing utilities.

This module provides functions for mixing multiple datasets with
deterministic block-based strategies for reproducible training.
"""

from __future__ import annotations

import numpy as np


def block_mixture_interleave(datasets_list, weights: dict[str, float] | None, block_size: int, seed: int, stop: str):
    """Create a deterministic block-based mixture of multiple datasets.

    This function implements a block-based dataset mixing strategy where examples
    from different datasets are interleaved in fixed-size blocks with specified
    proportions. This approach ensures deterministic and restart-friendly data
    loading for distributed training.

    Args:
        datasets_list: List of datasets to mix.
        weights: Optional dictionary mapping dataset names to their weights.
            If None, datasets are mixed equally.
        block_size: Number of examples per block.
        seed: Random seed for deterministic shuffling within blocks.
        stop: Strategy when a dataset is exhausted - "restart" to loop the dataset
            or "first_exhausted" to stop iteration.

    Returns:
        IterableDataset that yields examples from the mixed datasets.

    Raises:
        ValueError: If no datasets are provided.

    Example:
        >>> from datasets import IterableDataset
        >>> datasets = [dataset1, dataset2, dataset3]
        >>> weights = {"ds1": 0.5, "ds2": 0.3, "ds3": 0.2}
        >>> mixed = block_mixture_interleave(
        ...     datasets,
        ...     weights=weights,
        ...     block_size=1000,
        ...     seed=42,
        ...     stop="restart"
        ... )
        >>> for example in mixed:
        ...     process(example)
    """
    from datasets import IterableDataset

    n = len(datasets_list)
    if n == 0:
        raise ValueError("No datasets to mix")

    if weights is None or len(weights) != n:
        ws = np.ones(n, dtype=np.float64) / n
    else:
        ws = np.array(list(weights.values()), dtype=np.float64)
        ws = ws / ws.sum()

    counts = np.floor(ws * block_size).astype(int)
    remainder = block_size - counts.sum()
    if remainder > 0:
        counts[np.argmax(ws)] += remainder

    rng = np.random.default_rng(seed)

    def gen():
        iters = [iter(ds) for ds in datasets_list]
        while True:
            ids = []
            for i, c in enumerate(counts):
                ids.extend([i] * c)
            rng.shuffle(ids)
            for i in ids:
                try:
                    yield next(iters[i])
                except StopIteration:
                    if stop == "restart":
                        iters[i] = iter(datasets_list[i])
                        yield next(iters[i])
                    else:
                        return

    return IterableDataset.from_generator(gen)
