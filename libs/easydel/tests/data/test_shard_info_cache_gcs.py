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

"""Object-store behaviour of the durable metadata caches, against a fake gcsfs (no network).

gcsfs listings differ from local ones in three ways that matter here: they carry ``generation``
(a string, the exact object identity), their mtime is a tz-aware ``datetime`` under ``updated``
(not JSON-serialisable), and an in-place rewrite changes the generation even when size and
timestamp do not. The fake backend reproduces those listing rows over local files; the cache
objects themselves live on fsspec's ``memory://`` filesystem, which counts as a remote root.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from easydel.data import shard_info_cache as sic
from easydel.data.sources import rowgroup_shuffle as rgs
from easydel.data.sources.base import ParquetShardedSource

pytestmark = pytest.mark.cpu_ok

_ROWS = (7, 11, 5)


class _FakeGCSListing:
    """Local filesystem whose ``ls(detail=True)`` rows look like gcsfs rows.

    ``generation`` comes from :attr:`generations` (bumped by tests to simulate an in-place rewrite
    that preserves size and mtime); ``mtime`` is replaced by a tz-aware ``updated`` datetime.
    """

    generations: dict[str, int] = {}  # noqa: RUF012 - shared test state, reset per test

    def __init__(self, fs):
        self._fs = fs

    def __getattr__(self, item):
        return getattr(self._fs, item)

    def ls(self, path, detail=False, **kw):
        rows = []
        for e in self._fs.ls(path, detail=True, **kw):
            name = e["name"]
            st = os.stat(name) if os.path.exists(name) else None
            row = {"name": name, "size": e.get("size"), "type": e.get("type", "file")}
            if st is not None:
                row["updated"] = datetime.datetime.fromtimestamp(int(st.st_mtime), tz=datetime.UTC)
            row["generation"] = str(self.generations.get(name, 1_000_000_000_000_000))
            rows.append(row)
        return rows if detail else [r["name"] for r in rows]


@pytest.fixture
def fake_gcs(monkeypatch):
    _FakeGCSListing.generations = {}
    original = sic._fs_for

    def _wrapped(path, storage_options):
        fs, stripped = original(path, storage_options)
        if getattr(fs, "protocol", None) in ("file", ("file", "local"), ("file", "local", "local")):
            return _FakeGCSListing(fs), stripped
        return fs, stripped

    monkeypatch.setattr(sic, "_fs_for", _wrapped)
    monkeypatch.setattr(rgs, "_fs_for", _wrapped)
    return _FakeGCSListing


@pytest.fixture
def remote_root():
    # No test-node name here: parametrized ids carry ``[...]``, which fsspec treats as glob magic.
    import uuid

    root = f"memory://cache-{uuid.uuid4().hex}"
    yield root
    fs = fsspec.filesystem("memory")
    path = fs._strip_protocol(root)
    if fs.exists(path):
        fs.rm(path, recursive=True)


@pytest.fixture
def shards(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    out = []
    for i, n in enumerate(_ROWS):
        p = root / f"part-{i:03d}.parquet"
        pq.write_table(pa.table({"x": list(range(n))}), p)
        out.append(str(p))
    return out


def _glob(shards):
    return str(pathlib.Path(shards[0]).parent / "*.parquet")


def _src(shards, root, **kw):
    return ParquetShardedSource(_glob(shards), metadata_cache_dir=root, **kw)


def _bank(shards, root):
    cold = _src(shards, root)
    for n in shards:
        cold.get_shard_info(n)
    assert cold._shard_info_disk.flush(force_writer=True)
    return cold


def test_a_gcs_listing_carries_all_three_discriminators(shards, fake_gcs, remote_root):
    cache = sic.ShardInfoDiskCache(shards, root=remote_root)
    size, mtime, generation = cache._stat_of(shards[0])
    assert size == os.path.getsize(shards[0])
    assert isinstance(mtime, str) and "+00:00" in mtime  # tz-aware datetime, stringified
    assert isinstance(generation, str)


def test_a_datetime_mtime_survives_the_round_trip(shards, fake_gcs, remote_root):
    cold = _bank(shards, remote_root)
    fs, path = fsspec.core.url_to_fs(cold._shard_info_disk.uri)
    blob = json.loads(fs.cat(path))  # valid JSON despite datetime mtimes
    assert all(isinstance(r["mtime"], str) for r in blob["shards"].values())
    warm = _src(shards, remote_root)
    warm._open_file = lambda p: (_ for _ in ()).throw(AssertionError(p))
    assert [warm.get_shard_info(n).num_rows for n in shards] == list(_ROWS)


def test_a_generation_change_alone_is_refused_per_shard(shards, fake_gcs, remote_root):
    """Same size, same mtime second, new generation: that shard rereads, its neighbours still serve."""
    _bank(shards, remote_root)
    fake_gcs.generations[shards[0]] = 1_000_000_000_000_001
    warm = _src(shards, remote_root)
    assert warm._shard_info_disk.get(shards[0]) is None
    assert warm._shard_info_disk.get(shards[1]) == (_ROWS[1], 1)


def test_an_in_place_rewrite_is_refused(shards, fake_gcs, remote_root):
    _bank(shards, remote_root)
    pq.write_table(pa.table({"x": list(range(999))}), shards[0])
    fake_gcs.generations[shards[0]] = 2
    warm = _src(shards, remote_root)
    assert warm.get_shard_info(shards[0]).num_rows == 999
    assert warm._shard_info_disk.get(shards[2]) == (_ROWS[2], 1)


def test_publish_leaves_no_staging_objects(shards, fake_gcs, remote_root):
    cold = _bank(shards, remote_root)
    fs, path = fsspec.core.url_to_fs(cold._shard_info_disk.uri)
    parent = path.rsplit("/", 1)[0]
    names = [p.rsplit("/", 1)[-1] for p in fs.ls(parent, detail=False)]
    assert names == [path.rsplit("/", 1)[-1]]


@pytest.mark.parametrize("index,writes", [(0, True), (None, True), (3, False)])
def test_remote_shard_info_object_is_written_by_process_zero_only(shards, fake_gcs, remote_root, index, writes):
    src = _src(shards, remote_root, process_index=index)
    for n in shards:
        src.get_shard_info(n)
    assert src._shard_info_disk.flush() is writes
    fs, path = fsspec.core.url_to_fs(src._shard_info_disk.uri)
    assert fs.exists(path) is writes


@pytest.mark.parametrize("index,writes", [(0, True), (5, False)])
def test_remote_rg_index_sidecar_is_written_by_process_zero_only(shards, fake_gcs, remote_root, index, writes):
    groups = rgs.enumerate_parquet_row_groups(
        ParquetShardedSource(_glob(shards)), cache_dir=remote_root, process_index=index
    )
    assert [g[2] for g in groups] == list(_ROWS)
    fs = fsspec.filesystem("memory")
    base = fs._strip_protocol(remote_root)
    found = fs.glob(f"{base}/rg_index_v*.json.gz") if fs.exists(base) else []
    assert bool(found) is writes


def test_a_remote_rg_index_sidecar_is_served_and_invalidated_by_generation(shards, fake_gcs, remote_root, monkeypatch):
    src = ParquetShardedSource(_glob(shards))
    baseline = rgs.enumerate_parquet_row_groups(src, cache_dir=remote_root, process_index=0)
    scans = []
    orig = rgs._scan_one_file
    monkeypatch.setattr(rgs, "_scan_one_file", lambda s, p: scans.append(p) or orig(s, p))
    assert rgs.enumerate_parquet_row_groups(src, cache_dir=remote_root, process_index=0) == baseline
    assert scans == []
    fake_gcs.generations[shards[1]] = 42
    rgs.enumerate_parquet_row_groups(src, cache_dir=remote_root, process_index=0)
    assert scans == [shards[1]]
