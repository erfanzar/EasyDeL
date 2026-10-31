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

"""RowGroupShuffledParquetSource: global permutation, determinism, exact seek, pool, sidecars."""

from __future__ import annotations

import pathlib
import threading
import time
from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from easydel.data.sources import RowGroupShuffledParquetSource
from easydel.data.sources import rowgroup_shuffle as rgs
from easydel.data.sources.base import ParquetShardedSource
from easydel.data.transforms.mixture import MixedShardedSource

pytestmark = pytest.mark.cpu_ok

_SHARD = "rg_shuffled_0"


def _write_corpus(root: pathlib.Path) -> str:
    """3 files x 4 row groups x 5 rows = 60 rows with globally unique ids (``rid // 5`` = group)."""
    root.mkdir(parents=True, exist_ok=True)
    next_id = 0
    for f in range(3):
        ids = list(range(next_id, next_id + 20))
        next_id += 20
        pq.write_table(
            pa.table({"rid": ids, "text": [f"row-{i}" for i in ids]}), root / f"part-{f}.parquet", row_group_size=5
        )
    return str(root / "*.parquet")


@pytest.fixture
def corpus(tmp_path):
    # Function scope: the sidecar tests rewrite files in place.
    return _write_corpus(tmp_path / "data")


def _ids(rows):
    return [int(r["rid"]) for r in rows]


def _stream(pattern, **kw):
    return _ids(RowGroupShuffledParquetSource(pattern, **kw).open_shard(_SHARD))


def test_epoch_covers_corpus_exactly_once(corpus):
    src = RowGroupShuffledParquetSource(corpus, seed=3)
    assert sorted(_ids(src.open_shard(src.shard_names[0]))) == list(range(60))
    assert len(src) == 60
    assert src.get_shard_info(_SHARD).num_rows == 60


def test_order_is_shuffled_and_seed_deterministic(corpus):
    a1, a2, b = _stream(corpus, seed=3), _stream(corpus, seed=3), _stream(corpus, seed=4)
    assert a1 == a2
    assert a1 != list(range(60))
    assert a1 != b and sorted(a1) == sorted(b)


def test_within_group_shuffle_toggle(corpus):
    rows = _stream(corpus, seed=3, shuffle_within_group=False)
    for k in range(0, 60, 5):
        window = rows[k : k + 5]
        assert window == sorted(window) and len({i // 5 for i in window}) == 1


@pytest.mark.parametrize("split", [0, 3, 5, 17, 30, 59, 60, 75])
def test_seek_matches_full_stream(corpus, split):
    src = RowGroupShuffledParquetSource(corpus, seed=7)
    full = _ids(src.open_shard(_SHARD))
    assert _ids(src.open_shard_at_row(_SHARD, split)) == full[split:]


def test_seek_reads_no_skipped_group(corpus, monkeypatch):
    src = RowGroupShuffledParquetSource(corpus, seed=7, shuffle_within_group=False)
    reads = []
    real = src._read_group_rows
    monkeypatch.setattr(
        src, "_read_group_rows", lambda p, rg, expected_rows=None: reads.append((p, rg)) or real(p, rg, expected_rows)
    )
    next(iter(src.open_shard_at_row(_SHARD, 52)))
    assert len(reads) == 1  # groups 0..9 skipped on counts alone


def test_under_mixer_restart_and_seek(corpus):
    def build():
        return MixedShardedSource(
            sources={
                "rg": RowGroupShuffledParquetSource(corpus, seed=5),
                "rg2": RowGroupShuffledParquetSource(corpus, seed=9, group_pool_size=4),
            },
            weights={"rg": 1.0, "rg2": 1.0},
            block_size=8,
            seed=11,
            stop_strategy="restart",
        )

    full = list(islice(build().open_shard("mixed_shard_0"), 200))
    for split in (10, 60, 130):
        assert list(islice(build().open_shard_at_row("mixed_shard_0", split), 200 - split)) == full[split:]


def test_num_shards_is_callable_and_iter_shards_works(corpus):
    src = RowGroupShuffledParquetSource(corpus, seed=3)
    assert src.num_shards() == 1
    assert sorted(_ids(src.iter_shards())) == list(range(60))


def test_emit_original_index_matches_file_order(corpus):
    for pool in (1, 4):
        rows = list(
            RowGroupShuffledParquetSource(corpus, seed=3, emit_original_index=True, group_pool_size=pool).open_shard(
                _SHARD
            )
        )
        assert all(r["__orig_row__"] == r["rid"] for r in rows)


def test_projection_of_missing_column_does_not_silently_empty(tmp_path):
    pq.write_table(
        pa.table({"rid": list(range(12)), "text": [f"t{i}" for i in range(12)]}),
        tmp_path / "p.parquet",
        row_group_size=4,
    )
    path = str(tmp_path / "p.parquet")
    assert len(list(RowGroupShuffledParquetSource(path, columns=["not_a_column"], seed=1).open_shard(_SHARD))) == 12
    rows = list(RowGroupShuffledParquetSource(path, columns=["rid", "not_a_column"], seed=1).open_shard(_SHARD))
    assert sorted(_ids(rows)) == list(range(12)) and all(set(r) == {"rid"} for r in rows)


def test_file_mutation_after_indexing_raises_not_misaligns(tmp_path):
    p = tmp_path / "m.parquet"
    pq.write_table(pa.table({"rid": list(range(20))}), p, row_group_size=10)
    src = RowGroupShuffledParquetSource(str(p), seed=2)
    assert len(src) == 20
    pq.write_table(pa.table({"rid": list(range(7))}), p, row_group_size=10)
    with pytest.raises(RuntimeError, match="metadata promised"):
        list(src.open_shard(_SHARD))


def test_oversized_group_raises_instead_of_skipping(corpus):
    src = RowGroupShuffledParquetSource(corpus, seed=1, max_row_group_bytes=1)
    assert src.delivers_all_metadata_rows is True
    with pytest.raises(MemoryError, match="max_row_group_bytes"):
        next(iter(src.open_shard(_SHARD)))


def test_transient_read_error_is_retried(corpus, monkeypatch):
    monkeypatch.setattr(rgs.time, "sleep", lambda *_: None)
    src = RowGroupShuffledParquetSource(corpus, seed=1)
    len(src)
    real, calls = src._open_file, []

    def flaky(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("connection reset")
        return real(path)

    src._open_file = flaky
    assert sorted(_ids(src.open_shard(_SHARD))) == list(range(60))
    assert len(calls) > 1


class TestGroupPoolInterleave:
    def test_pool_epoch_exact_coverage(self, corpus):
        assert sorted(_stream(corpus, seed=3, group_pool_size=8)) == list(range(60))

    def test_pool_seed_deterministic(self, corpus):
        a1, a2 = _stream(corpus, seed=3, group_pool_size=8), _stream(corpus, seed=3, group_pool_size=8)
        b = _stream(corpus, seed=4, group_pool_size=8)
        assert a1 == a2 and a1 != b and sorted(a1) == sorted(b)

    def test_pool_one_is_historical_order(self, corpus):
        assert _stream(corpus, seed=7, group_pool_size=1) == _stream(corpus, seed=7)

    @pytest.mark.parametrize("split", [0, 1, 7, 29, 42, 59])
    def test_pool_seek_matches_full_stream(self, corpus, split):
        src = RowGroupShuffledParquetSource(corpus, seed=7, group_pool_size=8)
        full = _ids(src.open_shard(_SHARD))
        assert _ids(src.open_shard_at_row(_SHARD, split)) == full[split:]

    def test_pool_breaks_group_drain_locality(self, corpus):
        def mean_distinct_groups(ids, w=10):
            wins = [ids[k : k + w] for k in range(0, len(ids) - w + 1, w)]
            return sum(len({i // 5 for i in win}) for win in wins) / len(wins)

        assert mean_distinct_groups(_stream(corpus, seed=3)) <= 3.0
        assert mean_distinct_groups(_stream(corpus, seed=3, group_pool_size=8)) >= 5.0

    def test_pool_larger_than_deck_is_safe(self, corpus):
        assert sorted(_stream(corpus, seed=3, group_pool_size=500)) == list(range(60))


class TestRowGroupIndexSidecar:
    def test_sidecar_roundtrip_and_hit(self, corpus, tmp_path, monkeypatch):
        cache = tmp_path / "rgcache"
        baseline = rgs.enumerate_parquet_row_groups(ParquetShardedSource(corpus), cache_dir=str(cache))
        assert len(list(cache.glob("rg_index_v*.json.gz"))) == 1

        scans = []
        orig = rgs._scan_one_file
        monkeypatch.setattr(rgs, "_scan_one_file", lambda s, p: scans.append(p) or orig(s, p))
        again = rgs.enumerate_parquet_row_groups(ParquetShardedSource(corpus), cache_dir=str(cache))
        assert again == baseline and scans == []

    def test_source_metadata_cache_dir_is_the_default_root(self, corpus, tmp_path):
        cache = tmp_path / "meta"
        RowGroupShuffledParquetSource(corpus, seed=0, metadata_cache_dir=str(cache)).__len__()
        assert len(list(cache.glob("rg_index_v*.json.gz"))) == 1

    def test_changed_file_rescans_only_that_file(self, corpus, tmp_path, monkeypatch):
        cache = str(tmp_path / "rgcache2")
        rgs.enumerate_parquet_row_groups(ParquetShardedSource(corpus), cache_dir=cache)
        victim = sorted(pathlib.Path(corpus).parent.glob("part-*.parquet"))[1]
        pq.write_table(pa.table({"rid": list(range(900, 906)), "text": ["x"] * 6}), victim, row_group_size=3)

        scans = []
        orig = rgs._scan_one_file
        monkeypatch.setattr(rgs, "_scan_one_file", lambda s, p: scans.append(p) or orig(s, p))
        fresh = rgs.enumerate_parquet_row_groups(ParquetShardedSource(corpus), cache_dir=cache)
        assert [pathlib.Path(p).name for p in scans] == [victim.name]
        by_file = {}
        for path, _rg, rows in fresh:
            by_file.setdefault(pathlib.Path(path).name, []).append(rows)
        assert by_file[victim.name] == [3, 3]

    def test_corrupt_sidecar_falls_back_to_full_scan(self, corpus, tmp_path):
        cache = tmp_path / "rgcache3"
        baseline = rgs.enumerate_parquet_row_groups(ParquetShardedSource(corpus), cache_dir=str(cache))
        next(cache.glob("rg_index_v*.json.gz")).write_bytes(b"not gzip")
        assert rgs.enumerate_parquet_row_groups(ParquetShardedSource(corpus), cache_dir=str(cache)) == baseline

    def test_unset_cache_writes_nothing(self, corpus, tmp_path):
        assert len(rgs.enumerate_parquet_row_groups(ParquetShardedSource(corpus))) == 12
        assert not list(tmp_path.glob("**/rg_index_*"))

    def test_bytes_preflight_is_cached_too(self, corpus, tmp_path, monkeypatch):
        cache = str(tmp_path / "rgb")
        src = ParquetShardedSource(corpus)
        first = rgs.max_row_group_bytes(src, src.shard_names, cache_dir=cache)
        monkeypatch.setattr(rgs, "_read_metadata", lambda *_a: (_ for _ in ()).throw(AssertionError("rescanned")))
        assert rgs.max_row_group_bytes(src, src.shard_names, cache_dir=cache) == first
        assert all(v > 0 for v in first.values())


class TestFooterScanParallelism:
    def test_parallel_equals_serial_and_is_bounded(self, corpus, monkeypatch):
        src = ParquetShardedSource(corpus)
        serial = rgs.enumerate_parquet_row_groups(src, scan_workers=1)
        in_flight, peak, lock = [0], [0], threading.Lock()
        orig = rgs._scan_one_file

        def slow(s, p):
            with lock:
                in_flight[0] += 1
                peak[0] = max(peak[0], in_flight[0])
            time.sleep(0.05)
            try:
                return orig(s, p)
            finally:
                with lock:
                    in_flight[0] -= 1

        monkeypatch.setattr(rgs, "_scan_one_file", slow)
        assert rgs.enumerate_parquet_row_groups(src, scan_workers=2) == serial
        assert peak[0] == 2  # concurrent, and never above the cap
        peak[0] = 0
        rgs.enumerate_parquet_row_groups(src, scan_workers=1)
        assert peak[0] == 1

    def test_worker_exception_propagates(self, corpus, monkeypatch):
        def boom(_s, _p):
            raise ValueError("bad footer")

        monkeypatch.setattr(rgs, "_scan_one_file", boom)
        with pytest.raises(ValueError, match="bad footer"):
            rgs.enumerate_parquet_row_groups(ParquetShardedSource(corpus), scan_workers=4)


class TestSharedSidecarWriter:
    def test_remote_sidecar_is_written_by_process_zero_only(self):
        uri = "gs://bucket/rg_index_cache/rg_index_v1_abc.json.gz"
        assert rgs._sidecar_writer(uri, 0)
        assert rgs._sidecar_writer(uri, None)
        assert not rgs._sidecar_writer(uri, 7)
        assert not rgs._sidecar_writer("s3://bucket/x.json.gz", 3)

    def test_local_sidecar_is_written_by_every_process(self, tmp_path):
        assert rgs._sidecar_writer(str(tmp_path / "x.json.gz"), 7)
        assert rgs._sidecar_writer("file://" + str(tmp_path / "x.json.gz"), 7)

    def test_non_zero_process_never_writes_remote_sidecar(self, monkeypatch):
        opened = []
        monkeypatch.setattr(
            rgs, "_fs_for", lambda uri, so: opened.append(uri) or (_ for _ in ()).throw(AssertionError(uri))
        )
        rgs._write_sidecar("gs://bucket/rg_index_v1_x.json.gz", {"a": {}}, None, 3, "rg_index")
        assert opened == []
