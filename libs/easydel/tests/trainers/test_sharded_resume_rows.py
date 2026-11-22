"""Pure-Python trainer shard-resume contract tests."""

from collections.abc import Iterator, Sequence

from easydel.data.core.protocols import ShardedDataSource, ShardInfo
from easydel.trainers.base_trainer import _iter_sharded_source_from_example_offset


class _DeliveredRowsSource(ShardedDataSource[dict]):
    """A decoder that may drop rows relative to its shard metadata."""

    def __init__(self, *, all_rows_delivered: bool):
        self.all_rows_delivered = all_rows_delivered
        self.seeks: list[tuple[str, int]] = []

    @property
    def shard_names(self) -> Sequence[str]:
        return ["first", "second"]

    @property
    def delivers_all_metadata_rows(self) -> bool:
        return self.all_rows_delivered

    def num_shards(self) -> int:
        return 2

    def get_shard_info(self, shard_name: str) -> ShardInfo:
        return ShardInfo(shard_id=self.shard_names.index(shard_name), shard_name=shard_name, num_rows=4)

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        rows = ["a", "b"] if shard_name == "first" and not self.all_rows_delivered else ["a", "b", "c", "d"]
        if shard_name == "second":
            rows = ["e", "f", "g", "h"]
        yield from ({"text": value} for value in rows)

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        self.seeks.append((shard_name, row))
        yield from list(self.open_shard(shard_name))[row:]


def test_resume_uses_delivered_rows_not_metadata_for_multi_shard_source():
    source = _DeliveredRowsSource(all_rows_delivered=False)
    all_rows = [r["text"] for name in source.shard_names for r in source.open_shard(name)]
    assert all_rows == ["a", "b", "e", "f", "g", "h"]
    for offset in (1, 2, 3, 5, 6):
        actual = [r["text"] for r in _iter_sharded_source_from_example_offset(source, offset)]
        assert actual == all_rows[offset:]
    assert source.seeks == []  # no shard-local seek can repair a wrong metadata boundary


def test_resume_through_row_limit_wrapper_uses_delivered_rows():
    from easydel.data.transforms.source import LimitedShardedSource

    source = LimitedShardedSource(_DeliveredRowsSource(all_rows_delivered=False), max_rows=100)
    for offset, expected in ((3, ["f", "g", "h"]), (5, ["h"])):
        actual = [r["text"] for r in _iter_sharded_source_from_example_offset(source, offset)]
        assert actual == expected


def test_resume_keeps_fast_seek_when_metadata_counts_delivered_rows():
    source = _DeliveredRowsSource(all_rows_delivered=True)
    actual = [r["text"] for r in _iter_sharded_source_from_example_offset(source, 5)]
    assert actual == ["f", "g", "h"]
    assert source.seeks == [("second", 1)]
