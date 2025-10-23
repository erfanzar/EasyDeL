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

"""Data loading and iteration utilities.

This module provides functions for creating efficient data iterators
with support for batching, shuffling, and prefetching.
"""

from __future__ import annotations

import typing as tp
from concurrent.futures import ThreadPoolExecutor


def create_data_iterator(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    prefetch: bool = True,
    shuffle_buffer: int | None = None,
    seed: int | None = None,
) -> tp.Iterator:
    """Create an efficient data iterator with optional batching and prefetching.

    Creates an iterator over a dataset with support for shuffling, batching,
    and thread-based prefetching for improved performance.

    Args:
        dataset: Dataset to iterate over (Dataset or IterableDataset).
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle the data (default: True).
        drop_last: Whether to drop the last incomplete batch (default: False).
        prefetch: Whether to enable thread-based prefetching (default: True).
        shuffle_buffer: Buffer size for streaming datasets (default: 10000).
        seed: Random seed for shuffling (default: None).

    Returns:
        Iterator yielding batches of data or individual examples if batch_size=1.

    Example:
        >>> iterator = create_data_iterator(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     prefetch=True
        ... )
        >>> for batch in iterator:
        ...     process_batch(batch)
    """
    if shuffle:
        try:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer or 10000, seed=seed)
        except TypeError:
            dataset = dataset.shuffle(seed=seed)

    def _batched(it, bs):
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) >= bs:
                yield batch
                batch = []
        if batch and not drop_last:
            yield batch

    it = iter(dataset)
    if batch_size and batch_size > 1:
        it = _batched(it, batch_size)

    if prefetch:
        executor = ThreadPoolExecutor(max_workers=1)

        def _gen():
            fut = executor.submit(next, it, StopIteration)
            while True:
                res = fut.result()
                if res is StopIteration:
                    break
                fut = executor.submit(next, it, StopIteration)
                yield res
            executor.shutdown(wait=False)

        return _gen()

    return it
