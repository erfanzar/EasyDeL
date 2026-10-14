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

import pytest

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

from easydel.data import DatasetProfiler, profile_dataset
from easydel.data.core.protocols import ShardedDataSource


class _ListShardedSource(ShardedDataSource[dict]):
    """In-memory sharded source for profiling tests."""

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    @property
    def shard_names(self) -> list[str]:
        return ["shard_0"]

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        yield from self._rows


def _make_rows(lengths: list[int], source: str = "test") -> list[dict[str, Any]]:
    rows = []
    for length in lengths:
        rows.append(
            {
                "input_ids": [1] * length,
                "attention_mask": [1] * length,
                "labels": [1] * length,
                "__source__": source,
            }
        )
    return rows


def test_profiler_collects_length_histograms():
    """The profiler should collect exact length histograms for token fields."""
    rows = [
        {"input_ids": [1] * 10, "attention_mask": [1] * 10, "labels": [1] * 10},
        {"input_ids": [1] * 20, "attention_mask": [1] * 20, "labels": [1] * 20},
        {"input_ids": [1] * 10, "attention_mask": [1] * 10, "labels": [1] * 10},
    ]
    source = _ListShardedSource(rows)
    profiler = DatasetProfiler(max_rows=10)
    profile = profiler.profile(source)

    assert profile.num_rows_sampled == 3
    assert profile.length_histogram["input_ids"] == {10: 2, 20: 1}
    assert profile.length_histogram["attention_mask"] == {10: 2, 20: 1}
    assert profile.length_histogram["labels"] == {10: 2, 20: 1}


def test_profiler_counts_fields_and_sources():
    """Field presence and __source__ distribution should be counted."""
    rows = [
        {"input_ids": [1] * 5, "a": 1, "__source__": "src_a"},
        {"input_ids": [1] * 5, "b": 2, "__source__": "src_b"},
        {"input_ids": [1] * 5, "a": 3, "__source__": "src_a"},
    ]
    source = _ListShardedSource(rows)
    profile = DatasetProfiler(max_rows=10).profile(source)

    assert profile.field_counts["input_ids"] == 3
    assert profile.field_counts["a"] == 2
    assert profile.field_counts["b"] == 1
    assert profile.source_distribution == {"src_a": 2, "src_b": 1}
    assert profile.source_token_counts == {"src_a": 10, "src_b": 5}
    assert profile.total_input_tokens == 15


def test_profiler_detects_padding_and_truncation():
    """Padding and explicit truncation fields should be reflected in rates."""
    rows = [
        {"input_ids": [1] * 10, "attention_mask": [1] * 10, "truncated": True},
        {"input_ids": [1] * 10, "attention_mask": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]},
        {"input_ids": [1] * 10, "attention_mask": [1] * 10},
    ]
    source = _ListShardedSource(rows)
    profile = DatasetProfiler(max_rows=10).profile(source)

    assert profile.truncation_rate == pytest.approx(1 / 3)
    assert profile.padding_rate == pytest.approx(1 / 3)


def test_profiler_computes_length_percentiles():
    """Length percentiles should summarize the distribution per field."""
    rows = [
        {"input_ids": [1] * 10, "attention_mask": [1] * 10},
        {"input_ids": [1] * 20, "attention_mask": [1] * 20},
        {"input_ids": [1] * 30, "attention_mask": [1] * 30},
        {"input_ids": [1] * 40, "attention_mask": [1] * 40},
    ]
    source = _ListShardedSource(rows)
    profile = DatasetProfiler(max_rows=10).profile(source)

    percentiles = profile.length_percentiles["input_ids"]
    assert percentiles["p50"] == pytest.approx(25.0)
    assert percentiles["p90"] == pytest.approx(37.0)
    assert percentiles["p99"] == pytest.approx(39.7)


def test_profiler_chat_template_fallback_rate():
    """Rows with explicit chat-template fallback markers should be counted."""
    rows = [
        {"input_ids": [1] * 5, "chat_template_applied": True},
        {"input_ids": [1] * 5, "chat_template_applied": False},
        {"input_ids": [1] * 5, "template_fallback": True},
        {"input_ids": [1] * 5},
    ]
    source = _ListShardedSource(rows)
    profile = DatasetProfiler(max_rows=10).profile(source)

    assert profile.chat_template_fallback_rate == pytest.approx(2 / 4)


def test_profiler_estimated_efficiency_decreases_with_skew():
    """Skewed lengths should yield lower estimated packing efficiency than uniform."""
    # Length 4 packs perfectly into seq_length=20 windows (4+EOS pairs sum to 20),
    # while a mixed 3/8 distribution leaves more padding waste.
    uniform_lengths = [4] * 100
    skewed_lengths = [3, 8] * 50

    uniform_profile = DatasetProfiler(seq_length=20).profile(_ListShardedSource(_make_rows(uniform_lengths)))
    skewed_profile = DatasetProfiler(seq_length=20).profile(_ListShardedSource(_make_rows(skewed_lengths)))

    assert uniform_profile.estimated_packing_efficiency["greedy"] == pytest.approx(1.0)
    assert uniform_profile.estimated_packing_efficiency["greedy"] > skewed_profile.estimated_packing_efficiency["greedy"]
    for strategy in ("greedy", "pool", "first_fit"):
        assert 0.0 < skewed_profile.estimated_packing_efficiency[strategy] <= 1.0


def test_profiler_packing_efficiency_gap():
    """Packing-efficiency gap should show smarter packers vs greedy."""
    # Short, varied sequences are where pool/first-fit visibly outperform greedy.
    rows = _make_rows([2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
    source = _ListShardedSource(rows)
    profile = DatasetProfiler(seq_length=20).profile(source)

    assert "pool_vs_greedy" in profile.packing_efficiency_gap
    assert "first_fit_vs_greedy" in profile.packing_efficiency_gap
    assert profile.packing_efficiency_gap["pool_vs_greedy"] >= 0.0
    assert profile.packing_efficiency_gap["first_fit_vs_greedy"] >= 0.0


def test_profiler_respects_max_rows():
    """The profiler should stop sampling after max_rows."""
    rows = _make_rows([10] * 100)
    source = _ListShardedSource(rows)
    profile = DatasetProfiler(max_rows=7).profile(source)

    assert profile.num_rows_sampled == 7
    assert profile.total_input_tokens == 7 * 10


def test_profiler_no_packing_estimation_without_seq_length():
    """When seq_length is not provided, packing efficiency should be empty."""
    rows = _make_rows([10, 20, 30])
    source = _ListShardedSource(rows)
    profile = DatasetProfiler(max_rows=10).profile(source)

    assert profile.estimated_packing_efficiency == {}
    assert profile.packing_efficiency_gap == {}
    assert profile.seq_length is None


def test_profiler_handles_empty_source():
    """Profiling an empty source should return zeroed/empty metrics."""
    source = _ListShardedSource([])
    profile = DatasetProfiler(seq_length=20).profile(source)

    assert profile.num_rows_sampled == 0
    assert profile.total_input_tokens == 0
    assert profile.length_histogram == {
        "input_ids": {},
        "attention_mask": {},
        "labels": {},
    }
    assert profile.length_percentiles == {
        "input_ids": {"p50": 0.0, "p90": 0.0, "p99": 0.0},
        "attention_mask": {"p50": 0.0, "p90": 0.0, "p99": 0.0},
        "labels": {"p50": 0.0, "p90": 0.0, "p99": 0.0},
    }
    assert profile.estimated_packing_efficiency == {}
    assert profile.packing_efficiency_gap == {}


def test_profiler_handles_non_dict_rows():
    """Non-dict rows should be skipped without crashing."""

    class _BadSource(ShardedDataSource[Any]):
        @property
        def shard_names(self) -> list[str]:
            return ["shard_0"]

        def num_shards(self) -> int:
            return 1

        def open_shard(self, shard_name: str) -> Iterator[Any]:
            yield None
            yield "not-a-dict"
            yield {"input_ids": [1] * 5}

    profile = DatasetProfiler(max_rows=10).profile(_BadSource())
    assert profile.num_rows_sampled == 3
    assert profile.total_input_tokens == 5
    assert profile.length_histogram["input_ids"] == {5: 1}


def test_profile_dataset_prints_and_returns(capsys):
    """profile_dataset should print the profile and return it."""
    rows = _make_rows([10, 20])
    source = _ListShardedSource(rows)
    profile = profile_dataset(source, seq_length=100, max_rows=10)

    captured = capsys.readouterr()
    assert "DatasetProfile" in captured.out
    assert profile.num_rows_sampled == 2


def test_profiler_pipeline_integration():
    """Pipeline.profile() should return per-source profiles without mutation."""
    from easydel.data import DatasetConfig, Pipeline, PipelineConfig

    config = PipelineConfig(
        datasets=[
            DatasetConfig(
                name="ds1",
                data_files="unused",
            )
        ]
    )
    pipeline = Pipeline.from_config(config)
    # Bypass real source loading by injecting a test source manually.
    pipeline._data = {"ds1": _ListShardedSource(_make_rows([10, 20, 30]))}

    profiles = pipeline.profile(seq_length=100, max_rows=10)

    assert set(profiles.keys()) == {"ds1"}
    assert profiles["ds1"].num_rows_sampled == 3
    assert "greedy" in profiles["ds1"].estimated_packing_efficiency
    assert pipeline.get_stages() == []
