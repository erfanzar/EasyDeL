"""Regression coverage for the ``_DropFieldsShardedSource`` compatibility wrapper.

The private pipeline wrapper is a constructor adapter over the shared
:meth:`~easydel.data.core.protocols.ShardedDataSource.drop_fields`
implementation. These tests pin its observable behaviour.
"""

from collections.abc import Iterator, Sequence

import pytest
from easydel.data.core.protocols import ShardedDataSource
from easydel.data.execution.pipeline import _DropFieldsShardedSource

pytestmark = pytest.mark.cpu_ok


class _Rows(ShardedDataSource):
    """Minimal in-memory source with two shards."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    @property
    def shard_names(self) -> Sequence[str]:
        return ["s0", "s1"]

    def num_shards(self) -> int:
        return 2

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        index = int(shard_name[1:])
        yield from self._rows[index::2]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        yield from list(self.open_shard(shard_name))[row:]

    def __len__(self) -> int:
        return len(self._rows)


def test_compat_wrapper_matches_shared_drop_fields_behaviour():
    rows = [{"id": i, "x": "drop-me", "text": f"row-{i}"} for i in range(6)]
    source = _Rows(rows)
    wrapped = _DropFieldsShardedSource(source, frozenset({"x"}))
    shared = source.drop_fields(["x"])

    assert wrapped.shard_names == shared.shard_names == ["s0", "s1"]
    assert wrapped.num_shards() == shared.num_shards() == 2
    assert len(wrapped) == len(shared) == 6
    for shard in wrapped.shard_names:
        expected = [{"id": row["id"], "text": row["text"]} for row in rows if rows.index(row) % 2 == int(shard[1:])]
        assert list(wrapped.open_shard(shard)) == expected
        assert list(wrapped.open_shard(shard)) == list(shared.open_shard(shard))
        assert list(wrapped.open_shard_at_row(shard, 1)) == expected[1:]


def test_compat_wrapper_never_mutates_caller_rows():
    row = {"id": 1, "x": 2}
    wrapped = _DropFieldsShardedSource(_Rows([row]), ["x", "absent-key"])
    assert list(wrapped.open_shard("s0")) == [{"id": 1}]
    assert row == {"id": 1, "x": 2}


def test_compat_wrapper_forwards_metadata_exactness_flag():
    exact = _Rows([{"id": 1}])
    assert _DropFieldsShardedSource(exact, ["x"]).delivers_all_metadata_rows is True

    class Inexact(_Rows):
        @property
        def delivers_all_metadata_rows(self) -> bool:
            return False

    assert _DropFieldsShardedSource(Inexact([{"id": 1}]), ["x"]).delivers_all_metadata_rows is False
