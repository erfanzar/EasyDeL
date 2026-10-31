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

"""ParquetRandomAccessSource: index correctness, coverage, guards, pickling, LRU, sidecar reuse."""

from __future__ import annotations

import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from easydel.data.sources import ParquetRandomAccessSource

pytestmark = pytest.mark.cpu_ok


def _write_corpus(root):
    """3 files x 4 row groups x 5 rows = 60 rows; ``rid`` == global row index in file order."""
    next_id = 0
    for f in range(3):
        ids = list(range(next_id, next_id + 20))
        next_id += 20
        table = pa.table({"rid": ids, "text": [f"row-{i}" for i in ids]})
        pq.write_table(table, root / f"part-{f}.parquet", row_group_size=5)
    return str(root / "*.parquet")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return _write_corpus(tmp_path_factory.mktemp("racorpus"))


def test_len_matches_corpus(corpus):
    assert len(ParquetRandomAccessSource(corpus)) == 60


def test_getitem_exact_rows_across_group_boundaries(corpus):
    src = ParquetRandomAccessSource(corpus)
    # Bisect edges: corpus first/last, adjacent groups (4|5), file boundary (19|20), interior.
    for idx in (0, 4, 5, 19, 20, 37, 59):
        row = src[idx]
        assert row["rid"] == idx and row["text"] == f"row-{idx}"
    assert src[-1]["rid"] == 59
    with pytest.raises(IndexError):
        src[60]


def test_locate_maps_global_rows_without_reading_data(corpus, monkeypatch):
    import easydel.data.sources.parquet_random_access as pra

    src = ParquetRandomAccessSource(corpus)
    len(src)  # build the index

    def _boom(*_a, **_k):
        raise AssertionError("locate decoded a row group")

    monkeypatch.setattr(pra, "read_parquet_row_group_table", _boom)
    path, rg, off = src.locate(37)
    assert path.endswith("part-1.parquet") and (rg, off) == (3, 2)
    assert src.locate(-1)[1:] == (3, 4)


def test_permutation_covers_every_row_exactly_once(corpus):
    src = ParquetRandomAccessSource(corpus)
    order = np.random.default_rng(11).permutation(60)
    assert sorted(src[int(i)]["rid"] for i in order) == list(range(60))

    grain = pytest.importorskip("grain.python")
    ds = grain.MapDataset.source(ParquetRandomAccessSource(corpus)).shuffle(seed=3)
    assert sorted(ds[i]["rid"] for i in range(len(ds))) == list(range(60))


def test_oversized_group_guard_raises_eagerly_at_index_time(corpus):
    """Every offending file is named while indexing, before any decode."""
    src = ParquetRandomAccessSource(corpus, max_row_group_bytes=1)
    with pytest.raises(ValueError, match="decode guard") as exc:
        len(src)
    msg = str(exc.value)
    assert "largest rg" in msg and "row_group_size" in msg and "max_row_group_bytes" in msg
    assert msg.count("part-") == 3


def test_decode_time_guard_still_backstops_the_preflight(corpus):
    """A cap that appears after indexing is enforced at decode time."""
    src = ParquetRandomAccessSource(corpus)
    assert len(src) == 60
    src._max_row_group_bytes = 1
    with pytest.raises(MemoryError, match="max_row_group_bytes"):
        src[0]


def test_zero_cap_disables_both_guards(corpus):
    src = ParquetRandomAccessSource(corpus, max_row_group_bytes=0)
    assert len(src) == 60
    assert src[0]["rid"] == 0


def test_picklable_before_and_after_index_build(corpus):
    thawed = pickle.loads(pickle.dumps(ParquetRandomAccessSource(corpus)))
    assert thawed[7]["rid"] == 7 and len(thawed) == 60

    built = ParquetRandomAccessSource(corpus)
    assert built[42]["rid"] == 42
    thawed2 = pickle.loads(pickle.dumps(built))
    assert not thawed2._group_cache  # decoded tables never travel in the pickle
    assert thawed2._group_index is not None  # the cheap index does
    assert thawed2[42]["rid"] == 42 and thawed2[0]["rid"] == 0


def test_lru_cache_bound_and_hit_behavior(corpus, monkeypatch):
    import easydel.data.sources.parquet_random_access as pra

    reads: list[tuple[str, int]] = []
    real = pra.read_parquet_row_group_table

    def counting(source, path, rg, expected_rows=None, parquet_file=None):
        reads.append((path, rg))
        return real(source, path, rg, expected_rows=expected_rows, parquet_file=parquet_file)

    monkeypatch.setattr(pra, "read_parquet_row_group_table", counting)
    src = ParquetRandomAccessSource(corpus, cache_groups=2)

    for idx in range(5):  # one group, sequential -> a single decode
        assert src[idx]["rid"] == idx
    assert len(reads) == 1

    src[5], src[10]  # two more groups; group 0 evicted (bound = 2)
    assert len(reads) == 3 and len(src._group_cache) <= 2
    src[11]  # still cached
    assert len(reads) == 3
    src[0]  # evicted -> re-decode
    assert len(reads) == 4 and len(src._group_cache) <= 2


def test_cache_groups_must_be_positive(corpus):
    with pytest.raises(ValueError, match="cache_groups"):
        ParquetRandomAccessSource(corpus, cache_groups=0)


def test_returned_rows_are_mutation_safe(corpus):
    src = ParquetRandomAccessSource(corpus)
    src[3]["rid"] = -1
    assert src[3]["rid"] == 3


def test_projection_including_unknown_column_keeps_known_values(tmp_path):
    pq.write_table(pa.table({"rid": list(range(6)), "text": ["t"] * 6}), tmp_path / "p.parquet", row_group_size=3)
    src = ParquetRandomAccessSource(str(tmp_path / "p.parquet"), columns=["rid", "missing"])
    assert [src[i] for i in range(6)] == [{"rid": i} for i in range(6)]


def test_warm_index_from_metadata_cache_opens_no_footer(tmp_path, monkeypatch):
    """With ``metadata_cache_dir`` both the index and the byte-size preflight are served from sidecars."""
    import easydel.data.sources.rowgroup_shuffle as rgs

    (tmp_path / "data").mkdir()
    pattern = _write_corpus(tmp_path / "data")
    cache = str(tmp_path / "meta")
    cold = ParquetRandomAccessSource(pattern, metadata_cache_dir=cache)
    assert len(cold) == 60

    def _boom(_source, path):
        raise AssertionError(f"footer read for {path} on a warm start")

    monkeypatch.setattr(rgs, "_read_metadata", _boom)
    warm = ParquetRandomAccessSource(pattern, metadata_cache_dir=cache)
    assert len(warm) == 60
    assert warm[37]["rid"] == 37
