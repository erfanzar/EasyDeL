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

"""Regression tests for the 2026-07-03 data-pipeline correctness audit.

Every test here reproduces a confirmed defect: packers losing windows or
truncating long documents, the mixer terminating early (or never), biased
remainder allocation, global-RNG pollution, and prefetch-thread hangs/leaks.
"""

from __future__ import annotations

import random
import time

import numpy as np

from easydel.data.execution.loader import AsyncDataLoader, PrefetchIterator
from easydel.data.transforms.mixture import MixedShardedSource
from easydel.data.transforms.pack import FirstFitPacker, GreedyPacker, PackedShardedSource, PoolPacker


class ListSource:
    """Minimal ShardedDataSource over a list of rows."""

    def __init__(self, rows, name="s0"):
        self._rows = rows
        self.shard_names = [name]

    def num_shards(self):
        return 1

    def open_shard(self, _name):
        yield from ({**r} for r in self._rows)

    def __len__(self):
        return len(self._rows)


class TestLongDocumentPacking:
    def test_greedy_returns_every_completed_window(self):
        packer = GreedyPacker(seq_length=8, eos_token_id=99)
        windows = packer.add(list(range(20)))
        assert len(windows) == 2
        assert windows[0].input_ids.tolist() == list(range(8))
        assert windows[1].input_ids.tolist() == list(range(8, 16))
        tail = packer.flush_final()
        assert tail.input_ids.tolist()[:5] == [16, 17, 18, 19, 99]

    def test_pool_returns_every_completed_window(self):
        packer = PoolPacker(seq_length=8, eos_token_id=99, num_packers=2)
        windows = packer.add(list(range(20)))
        assert len(windows) == 2

    def test_packed_source_preserves_all_tokens_of_long_doc(self):
        src = ListSource([{"input_ids": list(range(50))}])
        packed = PackedShardedSource(src, seq_length=8, eos_token_id=99, shuffle=False)
        got = [t for row in packed.open_shard("packed_shard_0") for t in row["input_ids"].tolist()]
        for tok in range(50):
            assert tok in got, f"token {tok} lost in packing"

    def test_first_fit_splits_oversized_instead_of_truncating(self):
        packer = FirstFitPacker(seq_length=8, eos_token_id=99, buffer_size=4)
        windows = packer.add(list(range(20)))
        windows += packer.flush_all()
        got = [t for w in windows for t, m in zip(w.input_ids.tolist(), w.attention_mask.tolist(), strict=False) if m]
        for tok in range(20):
            assert tok in got, f"token {tok} truncated away"


class TestMixtureTermination:
    def test_empty_source_does_not_kill_restart_mix(self):
        healthy = ListSource([{"x": i} for i in range(10)], name="h")
        empty = ListSource([], name="e")
        mixed = MixedShardedSource(
            {"healthy": healthy, "empty": empty},
            weights={"healthy": 0.5, "empty": 0.5},
            block_size=4,
            seed=0,
            stop_strategy="restart",
        )
        import itertools

        rows = list(itertools.islice(mixed.open_shard("mixed_shard_0"), 12))
        # the old event-counter aliasing stopped the whole mix as soon as the
        # empty source was drawn len(sources)=2 times inside one block
        assert len(rows) == 12
        assert all(r["__source__"] == "healthy" for r in rows)

    def test_all_exhausted_terminates_with_large_blocks(self):
        a = ListSource([{"x": i} for i in range(3)], name="a")
        b = ListSource([{"y": i} for i in range(5)], name="b")
        mixed = MixedShardedSource(
            {"a": a, "b": b},
            block_size=64,  # >> len(sources): old counter never matched -> infinite loop
            seed=0,
            stop_strategy="all_exhausted",
        )
        rows = list(mixed.open_shard("mixed_shard_0"))
        assert len(rows) == 8  # every row from both sources, then clean stop

    def test_largest_remainder_rounding(self):
        sources = {n: ListSource([{"x": 0}], name=n) for n in ("a", "b", "c", "d")}
        mixed = MixedShardedSource(sources, weights={"a": 0.5, "b": 0.17, "c": 0.17, "d": 0.16}, block_size=10)
        counts = mixed._compute_counts(mixed._weights)
        assert sum(counts.values()) == 10
        # floors are [5,1,1,1]; the two leftover slots go to the largest
        # fractional parts (b and c at .7), not both to the heaviest dataset
        assert counts == {"a": 5, "b": 2, "c": 2, "d": 1}


class TestShuffleRng:
    def test_loader_shuffle_no_global_pollution_and_epoch_variation(self):
        rows = [{"input_ids": np.array([i])} for i in range(64)]
        loader = AsyncDataLoader(
            ListSource(rows),
            batch_size=1,
            prefetch_enabled=False,
            shuffle_buffer_size=16,
            seed=1234,
            drop_last=False,
        )
        random.seed(7)
        global_before = random.getstate()
        epoch1 = [int(b["input_ids"][0][0]) for b in loader]
        assert random.getstate() == global_before, "loader shuffle mutated the global random state"
        epoch2 = [int(b["input_ids"][0][0]) for b in loader]
        assert sorted(epoch1) == sorted(epoch2) == list(range(64))
        assert epoch1 != epoch2, "every epoch replayed the identical shuffle order"

        # same seed, fresh loader -> reproducible first epoch
        loader2 = AsyncDataLoader(
            ListSource(rows),
            batch_size=1,
            prefetch_enabled=False,
            shuffle_buffer_size=16,
            seed=1234,
            drop_last=False,
        )
        assert [int(b["input_ids"][0][0]) for b in loader2] == epoch1

    def test_packed_source_epoch_variation(self):
        rows = [{"input_ids": [i] * 5} for i in range(400)]
        packed = PackedShardedSource(
            ListSource(rows), seq_length=16, eos_token_id=0, shuffle=True, shuffle_buffer_factor=1, seed=3
        )
        e1 = [r["input_ids"].tolist() for r in packed.open_shard("packed_shard_0")]
        e2 = [r["input_ids"].tolist() for r in packed.open_shard("packed_shard_0")]
        assert e1 != e2


class TestPrefetchTeardown:
    def test_close_unblocks_producer_stuck_on_full_queue(self):
        def slow_source():
            yield from range(1000)

        it = PrefetchIterator(slow_source(), buffer_size=1)
        assert next(it) == 0  # starts the producer; it then blocks on the full queue
        time.sleep(0.2)
        it.close()
        deadline = time.time() + 5
        while it._worker.is_alive() and time.time() < deadline:
            time.sleep(0.05)
        assert not it._worker.is_alive(), "producer thread leaked after close()"
