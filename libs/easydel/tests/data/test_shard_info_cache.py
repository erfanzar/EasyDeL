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

"""The durable parquet shard-info cache behind ``ParquetShardedSource.get_shard_info``.

Pinned properties: a warm process serves shard info with zero footer opens; the in-memory dict stays
the first-level cache; the cache is off unless ``metadata_cache_dir`` is set; ``shard_id`` is never
persisted or served; ``(size, mtime, generation)`` validation refuses stale entries per shard;
entries with no discriminator or non-count values are never served; every failure fails open; the
stat baseline predates the first footer read; flushes merge and evict proven-stale records without
making them servable; only process 0 (or an unknown index) writes.
"""

from __future__ import annotations

import io
import json
import os
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from easydel.data import shard_info_cache as sic
from easydel.data.shard_info_cache import ShardInfoDiskCache, prewarm
from easydel.data.sources.base import ParquetShardedSource

pytestmark = pytest.mark.cpu_ok

_ROWS = (7, 11, 5)


def _write_shards(root: pathlib.Path, counts=_ROWS) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, n in enumerate(counts):
        p = root / f"part-{i:03d}.parquet"
        pq.write_table(pa.table({"x": list(range(n))}), p)
        paths.append(str(p))
    return paths


@pytest.fixture
def shards(tmp_path):
    return _write_shards(tmp_path / "data")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "cache"
    r.mkdir()
    return r


def _glob(shards):
    return str(pathlib.Path(shards[0]).parent / "*.parquet")


def _src(shards, root, **kw):
    return ParquetShardedSource(_glob(shards), metadata_cache_dir=str(root), **kw)


def _warm(shards, root) -> ParquetShardedSource:
    cold = _src(shards, root)
    for name in shards:
        cold.get_shard_info(name)
    assert cold._shard_info_disk.flush(force_writer=True), "fixture failed to write the cache"
    return cold


def _object(root) -> pathlib.Path:
    (obj,) = [p for p in root.glob("shardinfo-*.json")]
    return obj


class _NoOpens:
    """Turn any real footer read into a test failure."""

    def __init__(self, source):
        self.source = source

    def __enter__(self):
        def _boom(path):
            raise AssertionError(f"footer read for {path}: the cache did not serve it")

        self._orig = self.source._open_file
        self.source._open_file = _boom
        return self

    def __exit__(self, *exc):
        self.source._open_file = self._orig
        return False


def test_a_warm_process_serves_shard_info_without_opening_files(shards, root):
    cold = _warm(shards, root)
    want = {n: (cold.get_shard_info(n).num_rows, cold.get_shard_info(n).num_row_groups) for n in shards}
    warm = _src(shards, root)
    with _NoOpens(warm):
        got = {n: (warm.get_shard_info(n).num_rows, warm.get_shard_info(n).num_row_groups) for n in shards}
        assert len(warm) == sum(_ROWS)
    assert got == want


def test_the_in_memory_dict_stays_the_first_level_cache(shards, root):
    _warm(shards, root)
    warm = _src(shards, root)
    assert warm.get_shard_info(shards[0]).num_rows == _ROWS[0]

    def _boom(_name):
        raise AssertionError("durable cache consulted twice for one shard")

    warm._shard_info_disk.get = _boom
    assert warm.get_shard_info(shards[0]).num_rows == _ROWS[0]


def test_the_cache_is_off_without_a_root(shards, tmp_path):
    source = ParquetShardedSource(_glob(shards))
    assert source._shard_info_disk.enabled is False
    assert source._shard_info_disk.get(shards[0]) is None
    assert source.get_shard_info(shards[0]).num_rows == _ROWS[0]
    assert source._shard_info_disk.flush(force_writer=True) is False
    assert not list(tmp_path.glob("**/shardinfo-*"))


def test_create_source_forwards_the_source_stage_fields(shards, root):
    from easydel.data.core.config import DatasetConfig, SourceStageConfig
    from easydel.data.sources.base import create_source

    src = create_source(
        DatasetConfig(data_files=_glob(shards)),
        SourceStageConfig(metadata_cache_dir=str(root), process_index=3),
    )
    assert src._shard_info_disk.enabled and src._shard_info_disk._process_index == 3


def test_shard_id_is_recomputed_never_served(shards, root):
    full = _warm(shards, root)
    assert full.get_shard_info(shards[1]).shard_id == 1
    subset = ParquetShardedSource([shards[1], shards[2]], metadata_cache_dir=str(root))
    assert subset.get_shard_info(shards[1]).shard_id == 0
    assert subset.get_shard_info(shards[1]).num_rows == _ROWS[1]


def test_shard_id_is_absent_from_the_persisted_object(shards, root):
    _warm(shards, root)
    blob = json.loads(_object(root).read_text())
    assert set(blob["shards"]) == set(shards)
    for rec in blob["shards"].values():
        assert set(rec) == {"size", "mtime", "generation", "num_rows", "num_row_groups"}


def test_the_object_is_keyed_on_the_file_list(shards, root, tmp_path):
    a = ShardInfoDiskCache(sorted(shards), root=str(root))
    b = ShardInfoDiskCache(list(reversed(shards)), root=str(root))
    assert a.uri == b.uri
    grown = ShardInfoDiskCache([*shards, str(tmp_path / "data" / "part-999.parquet")], root=str(root))
    assert grown.uri != a.uri


def test_a_rewritten_shard_is_not_served_stale_and_neighbours_survive(shards, root):
    _warm(shards, root)
    pq.write_table(pa.table({"x": list(range(999))}), shards[0])
    warm = _src(shards, root)
    assert warm.get_shard_info(shards[0]).num_rows == 999
    with _NoOpens(warm):
        assert warm.get_shard_info(shards[1]).num_rows == _ROWS[1]


def test_an_mtime_change_alone_invalidates(shards, root):
    _warm(shards, root)
    size_before = os.path.getsize(shards[0])
    st = os.stat(shards[0])
    os.utime(shards[0], (st.st_atime, st.st_mtime + 1000))
    assert os.path.getsize(shards[0]) == size_before
    warm = _src(shards, root)
    assert warm._shard_info_disk.get(shards[0]) is None
    with _NoOpens(warm):
        assert warm.get_shard_info(shards[1]).num_rows == _ROWS[1]


def test_a_size_change_alone_invalidates(shards, root):
    _warm(shards, root)
    st = os.stat(shards[0])
    pq.write_table(pa.table({"x": list(range(999))}), shards[0])
    os.utime(shards[0], ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(shards[0]).st_mtime_ns == st.st_mtime_ns
    warm = _src(shards, root)
    assert warm._shard_info_disk.get(shards[0]) is None
    assert warm.get_shard_info(shards[0]).num_rows == 999


def test_an_untouched_shard_still_serves(shards, root):
    _warm(shards, root)
    st = os.stat(shards[0])
    os.utime(shards[0], ns=(st.st_atime_ns, st.st_mtime_ns))
    warm = _src(shards, root)
    with _NoOpens(warm):
        assert warm.get_shard_info(shards[0]).num_rows == _ROWS[0]


def _pad(stats):
    return {name: tuple(key) + (None,) * (3 - len(key)) for name, key in stats.items()}


def _bank_with_stats(shards, root, stats):
    cache = ShardInfoDiskCache(shards, root=str(root))
    cache._stats = _pad(stats)
    cache._loaded = True
    for i, name in enumerate(shards):
        cache.record(name, num_rows=_ROWS[i], num_row_groups=1)
    assert cache.flush(force_writer=True)


def _read_with_stats(shards, root, stats):
    cache = ShardInfoDiskCache(shards, root=str(root))
    cache._stats = _pad(stats)
    cache.load()
    return cache


@pytest.mark.parametrize(
    "banked,live,served",
    [
        ((10, None), (10, None), True),  # absent on both sides compares equal; size still guards
        ((10, None), (10, "t1"), False),  # an mtime that appears invalidates
        ((10, "t1"), (10, None), False),  # an mtime that disappears invalidates
        ((10, "t1"), (10, "t2"), False),  # an mtime that changes invalidates
        ((10, "t1"), (11, "t1"), False),  # a size change alone invalidates
        ((10, "t1", "g1"), (10, "t1", "g2"), False),  # a generation change alone invalidates
    ],
)
def test_stat_component_rules(shards, root, banked, live, served):
    _bank_with_stats(shards, root, {n: banked for n in shards})
    got = _read_with_stats(shards, root, {n: live for n in shards}).get(shards[0])
    assert (got == (_ROWS[0], 1)) is served


def test_an_unlistable_shard_is_neither_stored_nor_served(shards, root):
    cache = ShardInfoDiskCache(shards, root=str(root))
    cache._stats = {}
    cache._loaded = True
    cache.record(shards[0], num_rows=7, num_row_groups=1)
    assert cache._learned == {} and cache.flush(force_writer=True) is False


def test_stat_normalisation_across_backends():
    import datetime

    assert sic._stat_key({"size": 10}) == (10, None, None)
    assert sic._stat_key({"size": "10", "mtime": "t"}) == (10, "t", None)
    assert sic._stat_key({"size": 10, "updated": "t"}) == (10, "t", None)
    assert sic._stat_key({"size": 10, "LastModified": "t"}) == (10, "t", None)
    assert sic._stat_key({"size": 10, "mtime": "t", "generation": 17}) == (10, "t", "17")
    assert sic._stat_key({"size": 10, "mtime": "t", "ETag": "abc"}) == (10, "t", "abc")
    assert sic._stat_key({"size": 10, "mtime": 0}) == (10, 0, None)  # 0 is a real mtime
    stamp = datetime.datetime(2026, 9, 11, tzinfo=datetime.UTC)
    assert sic._stat_key({"size": 10, "mtime": stamp}) == (10, str(stamp), None)
    json.dumps(sic._stat_key({"size": 10, "mtime": stamp}))


def _mutate_record(root, name, **fields):
    obj = _object(root)
    blob = json.loads(obj.read_text())
    blob["shards"][name].update(fields)
    obj.write_text(json.dumps(blob))


@pytest.mark.parametrize("bad", [True, False, -1, 7.0, "7", None, [7], {"n": 7}])
def test_a_num_rows_that_is_not_a_count_is_refused(shards, root, bad):
    _warm(shards, root)
    _mutate_record(root, shards[0], num_rows=bad)
    warm = _src(shards, root)
    assert warm._shard_info_disk.get(shards[0]) is None
    assert warm.get_shard_info(shards[0]).num_rows == _ROWS[0]


@pytest.mark.parametrize("bad", [True, -1, 0])
def test_a_num_row_groups_that_cannot_hold_rows_is_refused(shards, root, bad):
    _warm(shards, root)
    _mutate_record(root, shards[0], num_row_groups=bad)
    assert _src(shards, root)._shard_info_disk.get(shards[0]) is None


def test_an_entry_with_no_discriminator_is_not_served(shards, root):
    _warm(shards, root)
    _mutate_record(root, shards[0], size=None, mtime=None, generation=None)
    assert _src(shards, root)._shard_info_disk.get(shards[0]) is None


@pytest.mark.parametrize("blob", ["{ not json", "[1, 2, 3]", '"a string"', "null", "7"])
def test_a_corrupt_or_non_object_cache_falls_back(shards, root, blob):
    _warm(shards, root)
    _object(root).write_text(blob)
    warm = _src(shards, root)
    assert warm._shard_info_disk.get(shards[0]) is None
    assert warm.get_shard_info(shards[0]).num_rows == _ROWS[0]


@pytest.mark.parametrize("field", [["a", "b"], "nope", 7, 1.5, True])
def test_a_shards_field_that_is_not_a_mapping_falls_back(shards, root, field):
    _warm(shards, root)
    obj = _object(root)
    blob = json.loads(obj.read_text())
    blob["shards"] = field
    obj.write_text(json.dumps(blob))
    warm = _src(shards, root)
    assert warm.get_shard_info(shards[0]).num_rows == _ROWS[0]


def test_a_schema_bump_ignores_old_objects(shards, root):
    _warm(shards, root)
    obj = _object(root)
    blob = json.loads(obj.read_text())
    blob["schema"] = 999
    obj.write_text(json.dumps(blob))
    assert _src(shards, root)._shard_info_disk.get(shards[0]) is None


@pytest.mark.parametrize("rec", ["not-a-record", 7, None, ["size", 1]])
def test_a_single_malformed_record_does_not_cost_its_neighbours(shards, root, rec):
    _warm(shards, root)
    obj = _object(root)
    blob = json.loads(obj.read_text())
    blob["shards"][shards[0]] = rec
    obj.write_text(json.dumps(blob))
    warm = _src(shards, root)
    assert warm._shard_info_disk.get(shards[0]) is None
    with _NoOpens(warm):
        assert warm.get_shard_info(shards[1]).num_rows == _ROWS[1]


def test_an_unwritable_root_never_raises(shards):
    source = ParquetShardedSource(_glob(shards), metadata_cache_dir="/proc/definitely/not/writable")
    assert source.get_shard_info(shards[0]).num_rows == _ROWS[0]
    assert source._shard_info_disk.flush(force_writer=True) is False


def test_a_backend_whose_listing_raises_never_raises_here(shards, root, monkeypatch):
    def _denied(*_a, **_k):
        raise PermissionError("403")

    monkeypatch.setattr(sic, "_fs_for", _denied)
    source = _src(shards, root)
    assert source._shard_info_disk.get(shards[0]) is None
    assert source.get_shard_info(shards[0]).num_rows == _ROWS[0]
    assert source._shard_info_disk.flush(force_writer=True) is False


class _ProxyFS:
    """Real fsspec filesystem except for the detail listing, which :meth:`rows` rewrites."""

    def __init__(self, fs):
        self._fs = fs

    def __getattr__(self, item):
        return getattr(self._fs, item)

    def rows(self, listing):
        raise NotImplementedError

    def ls(self, path, detail=False, **kw):
        rows = self.rows(self._fs.ls(path, detail=True, **kw))
        return rows if detail else [r["name"] for r in rows]


def _patch_backend(monkeypatch, wrapper):
    original = sic._fs_for

    def _wrapped(path, storage_options):
        fs, stripped = original(path, storage_options)
        return wrapper(fs), stripped

    monkeypatch.setattr(sic, "_fs_for", _wrapped)


class _OneMalformedRow(_ProxyFS):
    def rows(self, listing):
        return [{"name": "/nowhere/unreadable.parquet", "size": object()}, *listing]


def test_one_malformed_listing_row_costs_one_shard_not_the_directory(shards, root, monkeypatch):
    _patch_backend(monkeypatch, _OneMalformedRow)
    stats = ShardInfoDiskCache(shards, root=str(root))._current_stats()
    assert [n for n in shards if n in stats] == shards


class _NoSizeNoMtime(_ProxyFS):
    def rows(self, listing):
        return [{"name": e["name"], "size": None, "type": e.get("type", "file")} for e in listing]


def test_a_listing_with_no_discriminator_is_never_banked(shards, root, monkeypatch):
    _patch_backend(monkeypatch, _NoSizeNoMtime)
    cold = _src(shards, root)
    for name in shards:
        cold.get_shard_info(name)
    assert cold._shard_info_disk.flush(force_writer=True) is False


class _SwapDuringRead:
    """A concurrent job replaces the shard while its footer is being read (the read sees old bytes)."""

    def __init__(self, source, path, rows):
        self.source, self.path, self.rows = source, path, rows

    def __enter__(self):
        self._orig = self.source._open_file

        def _open(path):
            if path != self.path:
                return self._orig(path)
            blob = pathlib.Path(path).read_bytes()
            pq.write_table(pa.table({"x": list(range(self.rows))}), path)
            return io.BytesIO(blob)

        self.source._open_file = _open
        return self

    def __exit__(self, *exc):
        self.source._open_file = self._orig
        return False


def test_a_shard_replaced_during_its_footer_read_is_never_banked_as_fresh(shards, root):
    cold = _src(shards, root)
    with _SwapDuringRead(cold, shards[0], 999):
        assert cold.get_shard_info(shards[0]).num_rows == _ROWS[0]  # the pre-swap footer
    for name in shards[1:]:
        cold.get_shard_info(name)
    assert cold._shard_info_disk.flush(force_writer=True)

    warm = _src(shards, root)
    assert warm.get_shard_info(shards[0]).num_rows == 999
    with _NoOpens(warm):  # the untouched shards of that walk are still banked
        assert warm.get_shard_info(shards[1]).num_rows == _ROWS[1]


def test_a_flush_merges_with_the_object_already_on_disk(shards, root):
    _warm(shards, root)
    late = ShardInfoDiskCache(shards, root=str(root))
    late._current_stats()
    late._loaded = True  # the object itself was never read
    late.record(shards[0], num_rows=_ROWS[0], num_row_groups=1)
    assert late.flush(force_writer=True)
    blob = json.loads(_object(root).read_text())
    assert [blob["shards"][n]["num_rows"] for n in shards] == list(_ROWS)


def test_two_writers_do_not_erase_each_others_entries(shards, root):
    first, second = ShardInfoDiskCache(shards, root=str(root)), ShardInfoDiskCache(shards, root=str(root))
    for cache in (first, second):
        cache._current_stats()
        cache._loaded = True
    first.record(shards[0], num_rows=_ROWS[0], num_row_groups=1)
    second.record(shards[1], num_rows=_ROWS[1], num_row_groups=1)
    assert first.flush(force_writer=True) and second.flush(force_writer=True)
    blob = json.loads(_object(root).read_text())
    assert {n: blob["shards"][n]["num_rows"] for n in blob["shards"]} == {shards[0]: 7, shards[1]: 11}


def test_a_record_the_listing_proves_stale_is_not_written_back(shards, root):
    _warm(shards, root)
    pq.write_table(pa.table({"x": list(range(999))}), shards[0])
    writer = _src(shards, root)
    writer.get_shard_info(shards[1])  # a cache hit; nothing learned yet
    writer._shard_info_disk.record(shards[2], _ROWS[2], 1)
    assert writer._shard_info_disk.flush(force_writer=True)
    blob = json.loads(_object(root).read_text())
    assert shards[0] not in blob["shards"]  # evicted, not re-persisted
    assert blob["shards"][shards[1]]["num_rows"] == _ROWS[1]


def test_a_record_the_listing_cannot_cover_survives_the_merge(shards, root):
    _warm(shards, root)
    cache = ShardInfoDiskCache(shards, root=str(root))
    cache._stats = {n: s for n, s in cache._current_stats().items() if not n.endswith("part-000.parquet")}
    cache._loaded = True
    cache.record(shards[1], _ROWS[1], 1)
    assert cache.flush(force_writer=True)
    assert shards[0] in json.loads(_object(root).read_text())["shards"]


def test_a_flush_does_not_make_a_refused_record_servable(shards, root):
    _warm(shards, root)
    _mutate_record(root, shards[0], num_rows=12345, mtime=-1.0)  # a stale record the listing refutes
    cache = ShardInfoDiskCache(shards, root=str(root))
    cache.load()
    assert cache.get(shards[0]) is None
    cache.record(shards[1], _ROWS[1], 1)
    cache.flush(force_writer=True)
    assert cache.get(shards[0]) is None


def test_an_in_place_regeneration_is_never_served_stale_past_the_flush_boundary(tmp_path, root):
    counts = (3, 4, 5, 6, 7, 8)
    shards = _write_shards(tmp_path / "many", counts)
    _warm(shards, root)
    new_counts = tuple(c * 10 for c in counts)
    _write_shards(tmp_path / "many", new_counts)

    run = _src(shards, root)
    run._shard_info_disk._flush_every = 2
    for name in shards[:3]:  # a partial walk that crosses a flush boundary
        run.get_shard_info(name)
    later = _src(shards, root)
    assert [later.get_shard_info(n).num_rows for n in shards] == list(new_counts)


@pytest.mark.parametrize("index,writes", [(None, True), (0, True), (1, False), (7, False)])
def test_only_process_zero_or_unknown_writes(shards, root, index, writes):
    source = _src(shards, root, process_index=index)
    for name in shards:
        source.get_shard_info(name)
    assert source._shard_info_disk.flush() is writes
    assert bool(list(root.glob("shardinfo-*.json"))) is writes


def test_a_non_writer_does_not_grow(shards, root):
    source = _src(shards, root, process_index=5)
    source._shard_info_disk._flush_every = 1
    for name in shards:
        source.get_shard_info(name)
    assert source._shard_info_disk._learned == {}


def test_a_non_writer_still_reads(shards, root):
    _warm(shards, root)
    reader = _src(shards, root, process_index=9)
    with _NoOpens(reader):
        assert reader.get_shard_info(shards[2]).num_rows == _ROWS[2]


def test_the_staging_name_carries_the_host(monkeypatch):
    monkeypatch.setattr(sic.socket, "gethostname", lambda: "host/with:odd chars")
    assert sic._tmp_suffix().startswith("host-with-odd-chars-")

    def _explode():
        raise OSError("no hostname")

    monkeypatch.setattr(sic.socket, "gethostname", _explode)
    assert sic._tmp_suffix().startswith("nohost-")


def test_the_exit_hook_is_registered_once(shards, root, monkeypatch):
    calls = []
    monkeypatch.setattr(sic, "_ATEXIT_REGISTERED", False)
    monkeypatch.setattr(sic.atexit, "register", lambda fn: calls.append(fn))
    for _ in range(4):
        _src(shards, root)
    assert len(calls) == 1


def test_prewarm_banks_every_shard_and_reports_the_servable_count(shards, root):
    assert prewarm([_glob(shards)], str(root)) == len(shards)
    warm = _src(shards, root, process_index=3)
    with _NoOpens(warm):
        assert [warm.get_shard_info(n).num_rows for n in shards] == list(_ROWS)


def test_prewarm_needs_a_root(shards):
    with pytest.raises(ValueError):
        prewarm([_glob(shards)], "")
