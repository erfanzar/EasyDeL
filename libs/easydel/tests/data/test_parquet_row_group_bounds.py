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

"""Every Parquet writer bounds its row groups (explicit row count or a byte target)."""

from __future__ import annotations

import glob

import pyarrow.parquet as pq
import pytest
from easydel.data.core.config import SaveStageConfig
from easydel.data.execution.save import (
    DEFAULT_ROW_GROUP_TARGET_BYTES,
    ParquetWriter,
    SaveStage,
    create_writer,
    save_dataset,
    save_iterator,
)
from easydel.data.sources.base import ParquetShardedSource

pytestmark = pytest.mark.cpu_ok


class _ListSource:
    """Minimal reopenable sharded source over in-memory rows."""

    def __init__(self, rows):
        self._rows = rows

    @property
    def shard_names(self):
        return ["s0"]

    def num_shards(self):
        return 1

    def open_shard(self, _name):
        return iter(list(self._rows))

    def open_shard_at_row(self, _name, row):
        return iter(list(self._rows)[row:])


def _rows(n, width=100):
    return [{"rid": i, "text": "x" * width} for i in range(n)]


def _layout(path):
    out = []
    for f in sorted(glob.glob(f"{path}/*.parquet")):
        md = pq.ParquetFile(f).metadata
        out.append([md.row_group(i).num_rows for i in range(md.num_row_groups)])
    return out


def _all_rids(path):
    source = ParquetShardedSource(f"{path}/*.parquet")
    return sorted(int(r["rid"]) for n in source.shard_names for r in source.open_shard(n))


def test_explicit_row_group_size_size_rotated_writer(tmp_path):
    out = str(tmp_path / "o")
    stats = save_dataset(_ListSource(_rows(25)), out, row_group_size=10)
    assert stats.num_examples == 25
    assert _layout(out) == [[10, 10, 5]]
    assert _all_rids(out) == list(range(25))


def test_byte_target_derives_rows_per_group(tmp_path):
    """~108 estimated bytes per row and a 1 KiB target -> 9 rows per group."""
    out = str(tmp_path / "o")
    save_dataset(_ListSource(_rows(40)), out, row_group_target_bytes=1024)
    (groups,) = _layout(out)
    assert len(groups) > 1 and max(groups) <= 10 and sum(groups) == 40


def test_default_target_keeps_small_rows_in_one_group(tmp_path):
    out = str(tmp_path / "o")
    save_dataset(_ListSource(_rows(50)), out)
    assert _layout(out) == [[50]]
    assert DEFAULT_ROW_GROUP_TARGET_BYTES == 64 * 1024 * 1024


def test_partitioned_writer_is_bounded_too(tmp_path):
    out = str(tmp_path / "o")
    writer = create_writer(out, num_shards=2, row_group_size=4)
    stats = writer.write(_ListSource(_rows(20)))
    assert stats.num_shards == 2
    layout = _layout(out)
    assert all(max(groups) <= 4 for groups in layout) and sum(map(sum, layout)) == 20
    assert _all_rids(out) == list(range(20))


def test_save_iterator_forwards_row_group_size(tmp_path):
    out = str(tmp_path / "o")
    save_iterator(iter(_rows(9)), out, row_group_size=4)
    assert _layout(out) == [[4, 4, 1]]


def test_save_stage_reads_config_fields(tmp_path):
    from easydel.data.core.config import DatasetConfig, PipelineConfig
    from easydel.data.core.protocols import PipelineContext

    data = tmp_path / "in"
    data.mkdir()
    pq.write_table(__import__("pyarrow").table({"rid": list(range(12))}), data / "a.parquet")
    cfg = PipelineConfig(datasets=[DatasetConfig(data_files=str(data / "a.parquet"), name="d")])
    stage = SaveStage(SaveStageConfig(enabled=True, output_dir=str(tmp_path / "out"), row_group_size=5))
    stage.process({"d": ParquetShardedSource(str(data / "a.parquet"))}, PipelineContext(cfg))
    assert _layout(str(tmp_path / "out" / "d")) == [[5, 5, 2]]


@pytest.mark.parametrize("kw", [{"row_group_size": 0}, {"row_group_target_bytes": 0}])
def test_non_positive_bounds_are_refused(tmp_path, kw):
    with pytest.raises(ValueError):
        ParquetWriter(str(tmp_path / "o"), **kw)
