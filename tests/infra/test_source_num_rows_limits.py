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

from collections.abc import Iterator, Sequence

from easydel.data.core.protocols import ShardedDataSource
from easydel.data.transforms import LimitedShardedSource
from easydel.infra.elarge_model.builders import _create_source_from_inform


class UnknownSizeSource(ShardedDataSource[dict]):
    def __init__(self, shards: dict[str, list[int]]):
        self._shards = shards
        self.open_counts = {name: 0 for name in shards}

    @property
    def shard_names(self) -> Sequence[str]:
        return tuple(self._shards)

    def num_shards(self) -> int:
        return len(self._shards)

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        self.open_counts[shard_name] += 1
        for value in self._shards[shard_name]:
            yield {"value": value}


class DummyHFSource(ShardedDataSource[dict]):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        streaming: bool = True,
        cache_dir: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.streaming = streaming
        self.cache_dir = cache_dir

    @property
    def shard_names(self) -> Sequence[str]:
        return ("hf:0",)

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        yield {"value": 0}


def test_limited_sharded_source_does_not_prescan_unknown_shards():
    source = UnknownSizeSource({"first": list(range(5)), "second": list(range(5, 10))})

    limited = LimitedShardedSource(source, 2)

    assert source.open_counts == {"first": 0, "second": 0}
    assert [row["value"] for row in limited.open_shard("first")] == [0, 1]


def test_limited_sharded_source_carries_remaining_budget_to_later_shards():
    source = UnknownSizeSource({"first": [1], "second": [2, 3, 4]})
    limited = LimitedShardedSource(source, 2)

    assert [row["value"] for row in limited.open_shard("first")] == [1]
    assert [row["value"] for row in limited.open_shard("second")] == [2]


def test_create_source_from_inform_keeps_num_rows_for_implicit_hf_fallback(monkeypatch):
    import easydel.data.sources as sources_mod

    def _raise_file_not_found(_data_files):
        raise FileNotFoundError("missing local files")

    monkeypatch.setattr(sources_mod, "HuggingFaceShardedSource", DummyHFSource)
    monkeypatch.setattr(sources_mod, "expand_data_files", _raise_file_not_found)

    source = _create_source_from_inform(
        {"data_files": "org/dataset", "split": "train", "num_rows": 3},
        {"streaming": True, "cache_dir": "/tmp/easydel-tests"},
    )

    assert isinstance(source, LimitedShardedSource)
    assert isinstance(source._source, DummyHFSource)
    assert source._max_rows == 3
    assert source._source.dataset_name == "org/dataset"
