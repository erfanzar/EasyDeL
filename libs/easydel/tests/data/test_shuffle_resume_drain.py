from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
from easydel.data import ShuffledShardedSource
from easydel.data.core.protocols import ShardedDataSource, ShardInfo

pytestmark = pytest.mark.cpu_ok

_SHARD = "shuffled_shard_0"


class _MetaSource(ShardedDataSource[dict]):
    """In-memory sharded source with optional shard row metadata."""

    def __init__(self, n: int, nshards: int = 1, *, with_metadata: bool = True, reported_delta: int = 0):
        self._rows = [{"id": i} for i in range(n)]
        self._nshards = nshards
        self._with_metadata = with_metadata
        self._reported_delta = reported_delta

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"s{i}" for i in range(self._nshards)]

    def num_shards(self) -> int:
        return self._nshards

    def _shard_rows(self, shard_name: str) -> list[dict]:
        index = int(shard_name[1:])
        return self._rows[index :: self._nshards]

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        yield from self._shard_rows(shard_name)

    def get_shard_info(self, shard_name: str) -> ShardInfo | None:
        if not self._with_metadata:
            return None
        count = len(self._shard_rows(shard_name))
        if shard_name == "s0":
            count += self._reported_delta
        return ShardInfo(shard_id=int(shard_name[1:]), shard_name=shard_name, num_rows=count)

    @property
    def delivers_all_metadata_rows(self) -> bool:
        return True


def _ids(rows) -> list[int]:
    return [row["id"] for row in rows]


_COMBOS = [
    # (n, buffer, rows): steady = max(n - buffer, 0)
    (50, 16, (1, 20, 33, 34, 35, 40, 49, 50, 53)),
    (5, 16, (1, 3, 5, 7)),
    (16, 16, (1, 8, 15, 16, 17)),
    (17, 16, (1, 2, 16, 17)),
    (10, 3, (1, 6, 7, 8, 10)),
    (1, 1, (1, 2)),
    (7, 1, (1, 5, 6, 7)),
]


@pytest.mark.parametrize("nshards", [1, 3])
@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize(("n", "buffer", "rows"), _COMBOS)
def test_resume_is_exact(n, buffer, rows, seed, nshards):
    source = _MetaSource(n, nshards=nshards)
    shuffled = ShuffledShardedSource(source, buffer_size=buffer, seed=seed)
    full = _ids(shuffled.open_shard(_SHARD))
    assert sorted(full) == list(range(n))
    for row in rows:
        assert _ids(shuffled.open_shard_at_row(_SHARD, row)) == full[row:], (n, buffer, row)


@pytest.mark.parametrize("shuffle_shards", [True, False])
def test_resume_in_drain_phase_respects_shard_order(shuffle_shards):
    source = _MetaSource(30, nshards=4)
    shuffled = ShuffledShardedSource(source, buffer_size=8, seed=3, shuffle_shards=shuffle_shards)
    full = _ids(shuffled.open_shard(_SHARD))
    for row in (22, 23, 29, 30):
        assert _ids(shuffled.open_shard_at_row(_SHARD, row)) == full[row:]


def test_unknown_total_rows_stays_exact():
    shuffled = ShuffledShardedSource(_MetaSource(20, with_metadata=False), buffer_size=16, seed=1)
    full = _ids(shuffled.open_shard(_SHARD))
    assert _ids(shuffled.open_shard_at_row(_SHARD, 10)) == full[10:]


def test_metadata_undercount_stays_exact():
    # Metadata claims 18 rows while the stream holds 20: drain reconstruction would be wrong.
    shuffled = ShuffledShardedSource(_MetaSource(20, reported_delta=-2), buffer_size=16, seed=1)
    full = _ids(shuffled.open_shard(_SHARD))
    assert sorted(full) == list(range(20))
    assert _ids(shuffled.open_shard_at_row(_SHARD, 18)) == full[18:]


def test_failing_shard_info_lookup_stays_exact():
    class RaisingInfo(_MetaSource):
        def get_shard_info(self, shard_name: str) -> ShardInfo | None:
            raise OSError("metadata unavailable")

    shuffled = ShuffledShardedSource(RaisingInfo(12), buffer_size=16, seed=2)
    full = _ids(shuffled.open_shard(_SHARD))
    assert _ids(shuffled.open_shard_at_row(_SHARD, 5)) == full[5:]
