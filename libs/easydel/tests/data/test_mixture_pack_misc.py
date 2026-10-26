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

"""Regression tests: MixStage seed 0, mixer resume state, packer field alignment, embed packer None rows."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from easydel.data.core.config import DatasetConfig, MixStageConfig, PipelineConfig
from easydel.data.core.protocols import PipelineContext
from easydel.data.transforms.base import Transform
from easydel.data.transforms.mixture import (
    MixedShardedSource,
    MixStage,
    block_mixture_interleave,
    largest_remainder_counts,
)
from easydel.data.transforms.pack import FirstFitPacker, GreedyPacker, PackedShardedSource
from easydel.data.transforms.packer import EmbedsWindowPacker
from easydel.data.utils import infer_builder_from_ext

from ._common import ListSource

pytestmark = pytest.mark.cpu_ok


def _sources():
    return {
        "a": ListSource({"s": [{"v": f"a{i}"} for i in range(40)]}),
        "b": ListSource({"s": [{"v": f"b{i}"} for i in range(40)]}),
    }


def _take(source, n):
    return [r["v"] for r in itertools.islice(source.open_shard(source.shard_names[0]), n)]


def test_mix_stage_honours_explicit_zero_seed():
    config = PipelineConfig(datasets=[DatasetConfig(data_files="x.jsonl", name="a")])
    stage = MixStage(MixStageConfig(block_size=10, seed=0))
    mixed = stage.process(_sources(), PipelineContext(config=config, seed=5))["mixed"]

    seed0 = MixedShardedSource(_sources(), block_size=10, seed=0)
    seed5 = MixedShardedSource(_sources(), block_size=10, seed=5)
    assert _take(mixed, 30) == _take(seed0, 30)
    assert _take(mixed, 30) != _take(seed5, 30)


def test_mix_stage_falls_back_to_context_seed_when_unset():
    config = PipelineConfig(datasets=[DatasetConfig(data_files="x.jsonl", name="a")])
    stage = MixStage(MixStageConfig(block_size=10))
    mixed = stage.process(_sources(), PipelineContext(config=config, seed=5))["mixed"]
    assert _take(mixed, 30) == _take(MixedShardedSource(_sources(), block_size=10, seed=5), 30)


def test_largest_remainder_counts_sum_and_ties():
    counts = largest_remainder_counts(np.array([0.5, 0.3, 0.2]), 7)
    assert counts.sum() == 7
    assert list(largest_remainder_counts(np.array([1 / 3] * 3), 10)) == [4, 3, 3]


def test_block_mixture_interleave_uses_same_rounding():
    ds = {"a": [{"v": "a"}] * 100, "b": [{"v": "b"}] * 100, "c": [{"v": "c"}] * 100}
    mixed = block_mixture_interleave(ds, weights=None, block_size=10, seed=1)
    first_block = [r["v"] for r in itertools.islice(iter(mixed), 10)]
    assert sorted(first_block) == sorted(["a"] * 4 + ["b"] * 3 + ["c"] * 3)


@pytest.mark.parametrize("row", [0, 7, 10, 23])
def test_mixer_resume_state_roundtrip(row):
    mixer = MixedShardedSource(_sources(), weights={"a": 0.7, "b": 0.3}, block_size=10, seed=3)
    state = mixer.resume_state(row)
    assert state.examples_yielded == row
    assert state.block_index == row // 10
    expected = list(itertools.islice(mixer.open_shard("mixed_shard_0"), row, row + 15))
    resumed = list(itertools.islice(mixer.open_shard_from_state(state), 15))
    assert resumed == expected


@pytest.mark.parametrize("strategy", ["restart", "first_exhausted", "all_exhausted"])
@pytest.mark.parametrize("seed", [None, 0])
def test_mixer_state_restores_exact_positions_on_new_mixer(strategy, seed):
    shards = {"s0": [{"v": "a0"}, {"v": "a1"}], "s1": [{"v": "a2"}]}
    sources = {"a": ListSource(shards), "b": ListSource({"s0": []})}
    mixer = MixedShardedSource(sources, block_size=5, seed=seed, stop_strategy=strategy)
    expected = list(itertools.islice(mixer.open_shard("mixed_shard_0"), 18))
    for row in range(len(expected) + 1):
        state = mixer.resume_state(row)
        restored = MixedShardedSource(sources, block_size=5, stop_strategy=strategy)
        continuation = list(itertools.islice(restored.open_shard_from_state(state), len(expected) - row))
        assert continuation == expected[row:]
        if row < len(expected):
            assert state.source_index == 0
            shard = list(shards)[state.shard_index]
            assert shards[shard][state.row_index]["v"] == expected[row]["v"]
        elif strategy != "restart":
            assert state.source_index == -1
            assert list(restored.open_shard_from_state(state)) == []


def test_greedy_packer_aligns_missing_and_short_extra_fields():
    packer = GreedyPacker(seq_length=4, eos_token_id=99, extra_field_pad_values={"labels": -100})
    assert packer.add([1, 2], extra_fields={"labels": [5]}) == []
    out = packer.add([3], extra_fields={"labels": [7]})
    assert len(out) == 1
    assert out[0].input_ids.tolist() == [1, 2, 99, 3]
    assert out[0].extra_fields["labels"].tolist() == [5, -100, -100, 7]


def test_greedy_packer_separator_override():
    packer = GreedyPacker(
        seq_length=4,
        eos_token_id=99,
        extra_field_pad_values={"labels": -100},
        extra_field_separator_values={"labels": -1},
    )
    packer.add([1, 2], extra_fields=None)
    out = packer.add([3], extra_fields={"labels": [7]})
    assert out[0].extra_fields["labels"].tolist() == [-100, -100, -1, 7]


def test_first_fit_packer_aligns_extra_fields():
    packer = FirstFitPacker(seq_length=4, eos_token_id=99, buffer_size=10, extra_field_pad_values={"labels": -100})
    packer.add([1, 2], extra_fields={"labels": [5]})
    windows = packer.flush_all()
    assert len(windows) == 1
    assert windows[0].input_ids.tolist()[:3] == [1, 2, 99]
    assert windows[0].extra_fields["labels"].tolist() == [5, -100, -100, -100]


@pytest.mark.parametrize("strategy", ["greedy", "first_fit", "pool"])
def test_packed_source_shares_null_padding_and_truncation_for_extra_fields(strategy):
    raw = ListSource(
        {
            "s0": [
                {"input_ids": [1, 2], "labels": None},
                {"input_ids": [3], "labels": [7, 8]},
            ]
        }
    )
    source = PackedShardedSource(
        raw,
        seq_length=4,
        eos_token_id=99,
        strategy=strategy,
        num_packers=1,
        extra_field_pad_values={"labels": -100},
        extra_field_separator_values={"labels": -1},
        warn_on_padded_input=False,
        shuffle=False,
    )
    windows = list(source.open_shard("packed_shard_0"))
    aligned = [
        (int(token), int(label))
        for window in windows
        for token, label in zip(window["input_ids"], window["labels"], strict=True)
        if token != 0
    ]
    assert [(token, label) for token, label in aligned if token != 99] == [(1, -100), (2, -100), (3, 7)]
    separators = [label for token, label in aligned if token == 99]
    assert separators and all(label == -1 for label in separators)


def test_embeds_window_packer_counts_none_embed_rows_as_zero():
    packer = EmbedsWindowPacker(n_windows=2, seq_len=8, max_embed_rows=4)
    packer.push(
        [
            {"input_ids": [1, 2, 3], "embed_n_tok": None},
            {"input_ids": [4, 5], "embed_n_tok": [2, 1]},
            {"input_ids": [6], "embed_n_tok": np.array([], dtype=np.int64)},
            {"input_ids": [7]},
        ]
    )
    windows = packer.emit()
    placed = [row["input_ids"] for window in windows for row in window]
    assert sorted(map(tuple, placed)) == [(1, 2, 3), (4, 5), (6,), (7,)]
    assert len(packer) == 0


def test_embeds_window_packer_still_enforces_embed_budget():
    packer = EmbedsWindowPacker(n_windows=1, seq_len=8, max_embed_rows=2)
    with pytest.raises(ValueError, match="image-embed rows"):
        packer.push([{"input_ids": [1], "embed_n_tok": [2, 1]}])


def test_chain_via_rshift_stays_flat_and_ordered():
    class Append(Transform):
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, example):
            return {**example, "trace": example.get("trace", "") + self.tag}

    chain = Append("a") >> Append("b") >> Append("c")
    assert chain({})["trace"] == "abc"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("x.arrow", "arrow"),
        ("x.csv", "csv"),
        ("x.jsonl", "json"),
        ("gs://b/x.pq", "parquet"),
        ("x.parquet", "parquet"),
        ("x.bin", None),
    ],
)
def test_infer_builder_from_ext(path, expected):
    assert infer_builder_from_ext(path) == expected
