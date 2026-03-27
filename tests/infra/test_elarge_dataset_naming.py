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

import pytest

from easydel.data.core.protocols import ShardedDataSource
from easydel.data.transforms.mixture import MixedShardedSource
from easydel.infra.elarge.builders import _extract_dataset_name


class _TinySource(ShardedDataSource[dict]):
    def __init__(self, values: list[int]):
        self._values = values

    @property
    def shard_names(self) -> Sequence[str]:
        return ("tiny:0",)

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        del shard_name
        for value in self._values:
            yield {"value": value}

    def __len__(self) -> int:
        return len(self._values)


def test_extract_dataset_name_skips_generic_train_glob_segment():
    assert (
        _extract_dataset_name({"data_files": "gs://uscentral1stuff/data/qwen35-toolcalling/data/train-*.parquet"})
        == "qwen35-toolcalling"
    )


def test_mixed_sharded_source_rejects_weight_key_mismatches():
    with pytest.raises(ValueError, match="Weight keys must match source names"):
        MixedShardedSource(
            sources={"alpha": _TinySource([1]), "beta": _TinySource([2])},
            weights={"alpha": 1.0},
        )
