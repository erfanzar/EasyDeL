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
"""eLarge sharded-source lane: per-dataset sideband stamps, reshuffle forwarding and parquet reader selection."""

from __future__ import annotations

import json

import pytest
from easydel.data import ASSISTANT_ONLY_FIELD, DATASET_ID_FIELD
from easydel.data.sources import ParquetRandomAccessSource, ParquetShardedSource, RowGroupShuffledParquetSource
from easydel.infra.elarge.builders import _create_parquet_source, build_sharded_source

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = pytest.mark.cpu_ok


def _jsonl(path, prefix, n=6):
    path.write_text("".join(json.dumps({"content": f"{prefix}{i}"}) + "\n" for i in range(n)))
    return str(path)


def _rows(source):
    for shard in source.shard_names:
        yield from source.open_shard(shard)


def _cfg(informs, **mixture):
    return {"model": {"name_or_path": "unused"}, "mixture": {"informs": informs, "shuffle_buffer_size": 0, **mixture}}


def test_assistant_only_is_stamped_tristate_on_every_inform_with_dataset_ids(tmp_path):
    cfg = _cfg(
        [
            {"name": "a", "type": "json", "data_files": _jsonl(tmp_path / "a.jsonl", "a")},
            {"name": "b", "type": "json", "data_files": _jsonl(tmp_path / "b.jsonl", "b"), "assistant_only": True},
            {"name": "c", "type": "json", "data_files": _jsonl(tmp_path / "c.jsonl", "c"), "assistant_only": False},
        ],
        attach_dataset_ids=True,
        mixture_block_size=6,
        seed=3,
    )
    expected = {"a": (-1, 0), "b": (1, 1), "c": (0, 2)}
    seen = set()
    for row, _ in zip(_rows(build_sharded_source(cfg)), range(18), strict=False):
        prefix = row["text"][0]
        seen.add(prefix)
        assert (row[ASSISTANT_ONLY_FIELD], row[DATASET_ID_FIELD]) == expected[prefix]
    assert seen == {"a", "b", "c"}


def test_no_sideband_columns_without_configuration(tmp_path):
    cfg = _cfg([{"type": "json", "data_files": _jsonl(tmp_path / "a.jsonl", "a")}])
    row = next(_rows(build_sharded_source(cfg)))
    assert ASSISTANT_ONLY_FIELD not in row and DATASET_ID_FIELD not in row


def test_reshuffle_on_restart_routes_a_single_inform_through_the_mixer(tmp_path):
    files = []
    for i in range(4):
        files.append(_jsonl(tmp_path / f"s{i}.jsonl", f"{i}-", n=2))
    cfg = _cfg([{"type": "json", "data_files": files}], reshuffle_on_restart=True, mixture_block_size=4, seed=5)
    source = build_sharded_source(cfg)
    assert type(source).__name__ == "MixedShardedSource"
    stream = [row["text"] for row, _ in zip(_rows(source), range(24), strict=False)]
    epoch0, epoch1, epoch2 = stream[:8], stream[8:16], stream[16:24]
    assert sorted(epoch0) == sorted(epoch1) == sorted(epoch2)  # every epoch visits every row once
    assert len({tuple(epoch1), tuple(epoch2), tuple(epoch0)}) > 1  # restarted epochs are reordered


def _parquet(tmp_path):
    path = tmp_path / "d.parquet"
    pq.write_table(pa.Table.from_pylist([{"content": str(i)} for i in range(32)]), str(path), row_group_size=4)
    return [str(path)]


@pytest.mark.parametrize(
    ("access", "expected"),
    [
        ("stream", ParquetShardedSource),
        ("random_access", ParquetRandomAccessSource),
        ("rowgroup_shuffle", RowGroupShuffledParquetSource),
    ],
)
def test_parquet_access_selects_the_reader_and_rows_are_complete(tmp_path, access, expected):
    files = _parquet(tmp_path)
    source = _create_parquet_source(files, None, {"name": "d"}, {"parquet_access": access, "seed": 1})
    assert type(source) is expected
    assert sorted(int(r["content"]) for r in _rows(source)) == list(range(32))


def test_rowgroup_shuffle_permutes_and_inform_overrides_mixture(tmp_path):
    files = _parquet(tmp_path)
    source = _create_parquet_source(
        files, None, {"name": "d", "parquet_access": "rowgroup_shuffle"}, {"parquet_access": "stream", "seed": 1}
    )
    order = [int(r["content"]) for r in _rows(source)]
    assert order != list(range(32))
    with pytest.raises(ValueError, match="parquet_access"):
        _create_parquet_source(files, None, {}, {"parquet_access": "bogus"})


def test_metadata_cache_dir_writes_sidecars_with_explicit_process_index(tmp_path):
    files = _parquet(tmp_path)
    cache = tmp_path / "meta"
    source = _create_parquet_source(
        files, None, {}, {"parquet_access": "random_access", "metadata_cache_dir": str(cache)}
    )
    assert len(source) == 32 and source.locate(5)[1:] == (1, 1)
    assert any(cache.rglob("*")), "the row-group index sidecar should be written under metadata_cache_dir"
