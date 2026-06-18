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

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

from easydel.data.core.config import DatasetConfig, PackStageConfig, PipelineConfig
from easydel.data.core.protocols import PipelineContext, ShardedDataSource
from easydel.data.transforms.pack import (
    FirstFitPacker,
    GreedyPacker,
    PackedShardedSource,
    PackStage,
    PoolPacker,
    report_packing_stats,
)


class _ListShardedSource(ShardedDataSource[dict]):
    """In-memory sharded source for testing packers."""

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    @property
    def shard_names(self) -> list[str]:
        return ["shard_0"]

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        yield from self._rows


def _make_rows(lengths: list[int]) -> list[dict[str, Any]]:
    return [{"input_ids": [1] * length} for length in lengths]


def test_greedy_packer_perfect_fill_stats():
    """When rows exactly fill windows, efficiency should be 1.0."""
    packer = GreedyPacker(seq_length=6, eos_token_id=2)
    # Each row is 5 tokens; with one EOS separator it fills the window exactly.
    for _ in range(3):
        packer.add([1, 2, 3, 4, 5])
    packer.flush_final()

    assert packer.stats.efficiency == 1.0
    assert packer.stats.total_emitted_windows == 3
    assert packer.stats.perfect_fills == 3
    assert packer.stats.total_eos_separators == 3


def test_greedy_packer_skewed_efficiency_below_one():
    """Skewed lengths leave padding, so efficiency < 1.0."""
    packer = GreedyPacker(seq_length=20, eos_token_id=2)
    lengths = [5, 3, 8, 2, 6, 4, 7, 3, 5, 4]
    for length in lengths:
        packer.add([1] * length)
    packer.flush_final()

    assert 0.0 < packer.stats.efficiency < 1.0
    assert packer.stats.total_input_tokens == sum(lengths)
    assert packer.stats.total_emitted_windows > 0


def test_pool_and_first_fit_at_least_as_efficient_as_greedy():
    """Pool / first-fit should not be less efficient than greedy on the same data."""
    lengths = [5, 3, 8, 2, 6, 4, 7, 3, 5, 4]

    greedy = GreedyPacker(seq_length=20, eos_token_id=2)
    for length in lengths:
        greedy.add([1] * length)
    greedy.flush_final()

    pool = PoolPacker(seq_length=20, eos_token_id=2, num_packers=4)
    for length in lengths:
        pool.add([1] * length)
    pool.flush_all()

    first_fit = FirstFitPacker(seq_length=20, eos_token_id=2, buffer_size=100)
    for length in lengths:
        first_fit.add([1] * length)
    first_fit.flush_all()

    assert pool.stats.efficiency >= greedy.stats.efficiency
    assert first_fit.stats.efficiency >= greedy.stats.efficiency


def test_packed_sharded_source_exposes_stats():
    """PackedShardedSource should expose packing_stats after iteration."""
    rows = _make_rows([5, 3, 8, 2, 6, 4, 7, 3, 5, 4])
    source = _ListShardedSource(rows)
    packed = PackedShardedSource(
        source=source,
        seq_length=20,
        eos_token_id=2,
        strategy="greedy",
        shuffle=False,
        return_stats=True,
    )

    assert packed.packing_stats is None
    for _ in packed.open_shard(packed.shard_names[0]):
        pass

    stats = packed.packing_stats
    assert stats is not None
    assert stats.total_input_tokens == sum([5, 3, 8, 2, 6, 4, 7, 3, 5, 4])
    assert 0.0 < stats.efficiency <= 1.0


def test_packed_sharded_source_no_stats_when_disabled():
    """When return_stats=False, packing_stats should stay None."""
    rows = _make_rows([5, 3, 8])
    source = _ListShardedSource(rows)
    packed = PackedShardedSource(
        source=source,
        seq_length=20,
        eos_token_id=2,
        strategy="greedy",
        shuffle=False,
        return_stats=False,
    )

    for _ in packed.open_shard(packed.shard_names[0]):
        pass

    assert packed.packing_stats is None


def test_report_packing_stats_helper():
    """report_packing_stats should return stats without mutating the source."""
    lengths = [5, 3, 8, 2, 6, 4, 7, 3, 5, 4]
    rows = _make_rows(lengths)
    source = _ListShardedSource(rows)

    stats = report_packing_stats(source, seq_length=20, strategy="greedy")

    assert stats.total_input_tokens == sum(lengths)
    assert 0.0 < stats.efficiency <= 1.0
    assert stats.total_emitted_windows > 0


def test_report_packing_stats_respects_max_rows():
    """report_packing_stats should only consume up to max_rows upstream rows."""
    lengths = [5] * 100
    rows = _make_rows(lengths)
    source = _ListShardedSource(rows)

    stats = report_packing_stats(source, seq_length=20, strategy="greedy", max_rows=10)

    assert stats.total_input_tokens == 5 * 10


def test_pack_stage_records_context_metrics_on_iteration():
    """PackStage records packing stats into PipelineContext after the source is iterated."""
    rows = _make_rows([4] * 50)
    source = _ListShardedSource(rows)

    config = PackStageConfig(
        enabled=True,
        seq_length=20,
        eos_token_id=2,
        strategy="greedy",
        return_stats=True,
        record_context_stats=True,
        shuffle_packed=False,
    )
    stage = PackStage(config)
    context = PipelineContext(
        config=PipelineConfig(datasets=[DatasetConfig(data_files="dummy")]),
        seed=0,
    )
    result = stage.process({"ds1": source}, context)

    # Metrics are recorded lazily; process() itself should not populate them.
    assert "pack" not in context.get_metrics()

    # Iterate the packed source to trigger the callback.
    packed = result["ds1"]
    for _ in packed.iter_shards():
        pass

    metrics = context.get_metrics()
    assert "pack" in metrics
    assert "ds1_packing_stats" in metrics["pack"]
    stats_dict = metrics["pack"]["ds1_packing_stats"]
    assert stats_dict["seq_length"] == 20
    assert stats_dict["total_emitted_windows"] > 0
    assert 0.0 < stats_dict["efficiency"] <= 1.0


def test_pack_stage_respects_record_context_stats_false():
    """When record_context_stats=False, no metric is recorded even after iteration."""
    rows = _make_rows([4] * 50)
    source = _ListShardedSource(rows)

    config = PackStageConfig(
        enabled=True,
        seq_length=20,
        eos_token_id=2,
        strategy="greedy",
        return_stats=True,
        record_context_stats=False,
        shuffle_packed=False,
    )
    stage = PackStage(config)
    context = PipelineContext(
        config=PipelineConfig(datasets=[DatasetConfig(data_files="dummy")]),
        seed=0,
    )
    result = stage.process({"ds1": source}, context)

    for _ in result["ds1"].iter_shards():
        pass

    assert "pack" not in context.get_metrics()
