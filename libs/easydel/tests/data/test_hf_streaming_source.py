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

"""Public HF source iteration contracts across streaming and materialized loads."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from easydel.data.core.types import DatasetMixture, TextDatasetInform
from easydel.data.sources.base import HuggingFaceShardedSource, load_for_inform


@pytest.fixture(params=["parquet", "jsonl"])
def local_dataset(request, tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
    rows = [{"text": f"example {i}", "value": i} for i in range(5)]
    path = directory / f"train.{request.param}"
    if request.param == "parquet":
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    else:
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return directory, path, rows


@pytest.mark.parametrize("streaming", [True, False])
def test_hf_source_rows_resume_and_reiteration(local_dataset, tmp_path, streaming):
    directory, _, rows = local_dataset
    source = HuggingFaceShardedSource(str(directory), streaming=streaming, cache_dir=str(tmp_path / "cache"))
    shard = source.shard_names[0]
    partial = source.open_shard(shard)
    assert next(partial) == rows[0]
    partial.close()
    assert list(source.open_shard_at_row(shard, 2)) == rows[2:]
    assert list(source.open_shard(shard)) == rows
    if not streaming:
        assert len(source) == len(rows)


@pytest.mark.parametrize("streaming", [True, False])
def test_hf_mixture_file_source_preserves_row_limit(local_dataset, tmp_path, streaming):
    _, path, rows = local_dataset
    inform = TextDatasetInform(data_files=str(path), num_rows=3)
    mixture = DatasetMixture(informs=[inform], streaming=streaming, cache_dir=str(tmp_path / "cache"))
    dataset = load_for_inform(inform, mixture)
    assert list(dataset) == rows[:3]
    assert list(dataset) == rows[:3]


def test_hf_streaming_state_resume(local_dataset, tmp_path):
    _, path, rows = local_dataset
    inform = TextDatasetInform(data_files=str(path))
    mixture = DatasetMixture(informs=[inform], streaming=True, cache_dir=str(tmp_path / "cache"))
    dataset = load_for_inform(inform, mixture)
    iterator = iter(dataset)
    assert next(iterator) == rows[0]
    assert next(iterator) == rows[1]
    state = dataset.state_dict()
    iterator.close()
    restored = load_for_inform(inform, mixture)
    restored.load_state_dict(state)
    assert list(restored) == rows[2:]


@pytest.mark.parametrize("row_groups,expected", [(None, ["1", "2", "3", "4"]), ([1], ["2", "3"]), ([], [])])
def test_synchronous_parquet_projection_filter_and_cast(tmp_path, row_groups, expected):
    from datasets import DatasetInfo, DownloadConfig, Features, Value
    from datasets.packaged_modules.parquet.parquet import ParquetConfig
    from easydel.data.sources._hf_parquet import SynchronousParquetTables

    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"value": range(5), "keep": [False, True, True, True, True]}), path, row_group_size=2)
    features = Features({"value": Value("string")})
    config = ParquetConfig(columns=["value"], features=features, filters=[("keep", "==", True)], batch_size=1)
    producer = SynchronousParquetTables(config, DatasetInfo(features=features), DownloadConfig())
    rows = [row for _, table in producer([str(path)], [row_groups]) for row in table.to_pylist()]
    assert rows == [{"value": value} for value in expected]


def test_synchronous_parquet_empty_file(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    pq.write_table(pa.table({"text": pa.array([], type=pa.string())}), directory / "train.parquet")
    source = HuggingFaceShardedSource(str(directory), streaming=True)
    assert list(source.open_shard(source.shard_names[0])) == []
