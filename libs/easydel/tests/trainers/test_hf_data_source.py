"""Regression tests for ``HFDataSource`` sharding and exhaustion semantics."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_ok


def _iterable_dataset(num_rows: int = 16, num_shards: int = 4):
    from datasets import Dataset

    return Dataset.from_dict({"x": list(range(num_rows))}).to_iterable_dataset(num_shards=num_shards)


def _drain(source, limit: int = 1000) -> list[int]:
    out = []
    for _ in range(limit):
        try:
            out.append(source[0]["x"])
        except IndexError:
            break
    return out


@pytest.mark.parametrize("num_shards", [1, 3, 4])
def test_grain_shards_read_disjoint_slices(monkeypatch, num_shards):
    from easydel.trainers.utils import HFDataSource
    from grain import python as pygrain

    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "0")
    ds = _iterable_dataset(num_shards=num_shards)
    seen = []
    for index in range(2):
        options = pygrain.ShardOptions(shard_index=index, shard_count=2, drop_remainder=False)
        if num_shards < 2:
            with pytest.warns(UserWarning, match="Dataset has 1 shards"):
                source = HFDataSource(ds, options, num_threads=1)
        else:
            source = HFDataSource(ds, options, num_threads=1)
        seen.append(set(_drain(source)))
    assert seen[0] and seen[1]
    assert seen[0].isdisjoint(seen[1])
    assert seen[0] | seen[1] == set(range(16))


def test_exhausted_source_raises_index_error_when_nonstop_disabled(monkeypatch):
    from easydel.trainers.utils import HFDataSource
    from grain import python as pygrain

    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "0")
    options = pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False)
    source = HFDataSource(_iterable_dataset(num_rows=3, num_shards=1), options, num_threads=1)
    assert [source[0]["x"] for _ in range(3)] == [0, 1, 2]
    with pytest.raises(IndexError):
        source[0]


def test_exhausted_source_restarts_when_nonstop_enabled(monkeypatch):
    from easydel.trainers.utils import HFDataSource
    from grain import python as pygrain

    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "1")
    options = pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False)
    source = HFDataSource(_iterable_dataset(num_rows=3, num_shards=1), options, num_threads=1)
    assert [source[0]["x"] for _ in range(5)] == [0, 1, 2, 0, 1]


def test_grain_shards_and_threads_partition_all_rows(monkeypatch):
    from threading import Thread

    from easydel.trainers.utils import HFDataSource
    from grain import python as pygrain

    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "0")
    seen = [[] for _ in range(4)]
    for grain_index in range(2):
        options = pygrain.ShardOptions(shard_index=grain_index, shard_count=2, drop_remainder=False)
        source = HFDataSource(_iterable_dataset(), options, num_threads=2)
        for worker in range(2):

            def read_worker(slot=grain_index * 2 + worker, source=source):
                seen[slot].extend(_drain(source))

            thread = Thread(target=read_worker, name=f"loader_{worker}")
            thread.start()
            thread.join()
    assert all(seen)
    combined = [value for partition in seen for value in partition]
    assert sorted(combined) == list(range(16))


def test_empty_source_raises_index_error_even_in_nonstop_mode(monkeypatch):
    from datasets import IterableDataset
    from easydel.trainers.utils import HFDataSource
    from grain import python as pygrain

    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "1")
    dataset = IterableDataset.from_generator(lambda: iter(()))
    options = pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False)
    source = HFDataSource(dataset, options)
    with pytest.raises(IndexError):
        source[0]


def test_nonstop_restart_does_not_rewind_other_workers(monkeypatch):
    from threading import Thread

    from easydel.trainers.utils import HFDataSource
    from grain import python as pygrain

    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "1")
    options = pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False)
    source = HFDataSource(_iterable_dataset(num_rows=6, num_shards=2), options, num_threads=2)
    other_worker = []

    def read_other_worker():
        other_worker.append(source[0]["x"])

    thread = Thread(target=read_other_worker, name="loader_1")
    thread.start()
    thread.join()
    assert [source[0]["x"] for _ in range(4)] == [0, 1, 2, 0]
    thread = Thread(target=read_other_worker, name="loader_1")
    thread.start()
    thread.join()
    assert other_worker == [3, 4]


def test_dataset_without_shard_metadata_is_supported(monkeypatch):
    import easydel.trainers.utils as utils
    from grain import python as pygrain

    # Stand-in for an iterable dataset implementation without optional n_shards metadata.
    dataset = [{"x": 1}, {"x": 2}]
    monkeypatch.setattr(utils, "split_dataset_by_node", lambda dataset, **kwargs: dataset)
    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "0")
    options = pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False)
    assert _drain(utils.HFDataSource(dataset, options)) == [1, 2]
