from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import ClassVar

import pytest
from easydel.data import ShuffledShardedSource
from easydel.data.core.config import WeightSchedulePoint
from easydel.data.core.protocols import ShardedDataSource, ShardInfo
from easydel.data.transforms.base import ExpandTransform, Transform
from easydel.data.transforms.mixture import MixedShardedSource, WeightScheduler
from easydel.data.transforms.source import TransformedShardedSource

pytestmark = pytest.mark.cpu_ok


class ListSource(ShardedDataSource[dict]):
    """Minimal in-memory source: ``nshards`` shards, round-robin row assignment."""

    def __init__(self, rows: list[dict], nshards: int = 1):
        self._rows = rows
        self._nshards = nshards

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"s{i}" for i in range(self._nshards)]

    def num_shards(self) -> int:
        return self._nshards

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        i = int(shard_name[1:])
        yield from self._rows[i :: self._nshards]

    def __len__(self) -> int:
        return len(self._rows)


def _drain(source: ShardedDataSource) -> list:
    out: list = []
    for name in source.shard_names:
        out.extend(source.open_shard(name))
    return out


def test_shuffle_preserves_all_rows_and_reorders():
    src = ListSource([{"id": i} for i in range(50)])
    shuffled = src.shuffle(buffer_size=16, seed=42)
    order = [r["id"] for r in _drain(shuffled)]
    assert sorted(order) == list(range(50))  # nothing lost / duplicated
    assert order != list(range(50))  # actually shuffled


def test_shuffle_is_deterministic_for_fixed_seed():
    src = ListSource([{"id": i} for i in range(50)])
    a = [r["id"] for r in ShuffledShardedSource(src, buffer_size=16, seed=7).open_shard("shuffled_shard_0")]
    b = [r["id"] for r in ShuffledShardedSource(src, buffer_size=16, seed=7).open_shard("shuffled_shard_0")]
    assert a == b


def test_shuffle_seed_changes_order():
    src = ListSource([{"id": i} for i in range(50)])
    a = [r["id"] for r in src.shuffle(buffer_size=16, seed=7).open_shard("shuffled_shard_0")]
    b = [r["id"] for r in src.shuffle(buffer_size=16, seed=8).open_shard("shuffled_shard_0")]
    assert a != b


def test_open_shard_at_row_resumes_from_tail():
    """Resume offset must reproduce the exact tail (fast-forward correctness)."""
    src = ListSource([{"id": i} for i in range(50)])
    sh = src.shuffle(buffer_size=16, seed=42)
    full = [r["id"] for r in sh.open_shard("shuffled_shard_0")]
    for k in (0, 5, 16, 33):
        resumed = [r["id"] for r in sh.open_shard_at_row("shuffled_shard_0", k)]
        assert resumed == full[k:]


def test_shuffle_collapses_to_single_shard():
    src = ListSource([{"id": i} for i in range(50)], nshards=4)
    sh = src.shuffle(buffer_size=8, seed=1)
    assert list(sh.shard_names) == ["shuffled_shard_0"]
    assert sh.num_shards() == 1
    assert sorted(r["id"] for r in _drain(sh)) == list(range(50))


def test_buffer_size_must_be_positive():
    src = ListSource([{"id": 0}])
    try:
        ShuffledShardedSource(src, buffer_size=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("buffer_size=0 should raise ValueError")


def test_mix_then_shuffle_preserves_weights_and_source_tags():
    a = ListSource([{"text": f"A{i}"} for i in range(200)])
    b = ListSource([{"text": f"B{i}"} for i in range(200)])
    mixed = MixedShardedSource(
        {"A": a, "B": b},
        weights={"A": 1, "B": 9},
        block_size=20,
        seed=7,
        stop_strategy="first_exhausted",
    )
    shuffled = mixed.shuffle(buffer_size=64, seed=7)
    rows = list(shuffled.open_shard("shuffled_shard_0"))

    # __source__ tagging from the mixer survives the shuffle.
    assert all("__source__" in r for r in rows)

    # Weighted mix is preserved (B drawn ~9x as often as A).
    n_a = sum(1 for r in rows if r["text"].startswith("A"))
    n_b = sum(1 for r in rows if r["text"].startswith("B"))
    assert n_b > n_a * 5

    # Output is decorrelated from file order (the bug being fixed: mixed-but-not-shuffled).
    texts = [r["text"] for r in rows]
    assert texts[:8] != [f"B{i}" for i in range(8)]


def test_mix_resume_exact_with_scheduler_and_dead_source():
    """Resuming must reproduce the continuous stream even with an empty constituent."""
    live = ListSource([{"text": f"L{i:03d}"} for i in range(200)])
    hole = ListSource([])
    scheduler = WeightScheduler(
        [
            WeightSchedulePoint(step=0, weights={"live": 0.5, "hole": 0.5}),
            WeightSchedulePoint(step=8, weights={"live": 1.0, "hole": 0.0}),
        ],
        interpolation="step",
    )
    mix = MixedShardedSource(
        {"live": live, "hole": hole},
        block_size=10,
        seed=123,
        stop_strategy="restart",
        weight_scheduler=scheduler,
    )
    continuous = [r["text"] for _, r in zip(range(45), mix.open_shard("mixed_shard_0"), strict=False)]
    resumed = [r["text"] for _, r in zip(range(20), mix.open_shard_at_row("mixed_shard_0", 25), strict=False)]
    assert resumed == continuous[25:]
    # Guard against the old behaviour re-serving already-emitted rows.
    assert not set(continuous[:25]) & set(resumed[:5])


def test_mix_resume_exact_with_static_weights_and_dead_source():
    """Emitted-count resume must not rewind across dead ids even without a scheduler."""
    live = ListSource([{"text": f"L{i:03d}"} for i in range(200)])
    mix = MixedShardedSource(
        {"live": live, "hole": ListSource([])},
        weights={"live": 1.0, "hole": 1.0},
        block_size=10,
        seed=123,
        stop_strategy="restart",
    )
    continuous = [r["text"] for _, r in zip(range(45), mix.open_shard("mixed_shard_0"), strict=False)]
    resumed = [r["text"] for _, r in zip(range(20), mix.open_shard_at_row("mixed_shard_0", 25), strict=False)]
    assert resumed == continuous[25:]


def test_mix_resume_probe_closes_stream_before_opening_resume_stream():
    class StreamingSource(ShardedDataSource[dict]):
        def __init__(self):
            self.open_handles = 0
            self.streams = []  # Keep references: leaked iterators cannot be cleaned up by reference counting.

        @property
        def shard_names(self) -> Sequence[str]:
            return ["stream"]

        def num_shards(self) -> int:
            return 1

        def open_shard(self, shard_name: str) -> Iterator[dict]:
            def stream():
                self.open_handles += 1
                try:
                    yield {"text": "one"}
                    yield {"text": "two"}
                finally:
                    self.open_handles -= 1

            iterator = stream()
            self.streams.append(iterator)
            yield from iterator

    class RetainingMixedSource(MixedShardedSource):
        def _chain_shards(self, source):
            stream = super()._chain_shards(source)
            self.streams.append(stream)  # Simulate a streaming owner retaining iterator references.
            return stream

    source = StreamingSource()
    mixed = RetainingMixedSource({"only": source}, block_size=1, seed=0, stop_strategy="restart")
    mixed.streams = []
    resumed = mixed.open_shard_at_row("mixed_shard_0", 1)
    try:
        assert next(resumed)["text"] == "two"
        assert source.open_handles == 1  # Only the actual resume stream, not the probe.
    finally:
        resumed.close()
        for stream in mixed.streams:
            stream.close()
        for stream in source.streams:
            stream.close()


def test_mix_resume_fast_path_still_used_for_live_sources():
    """All-live resume stays exact (weights drift under the scheduler must not appear)."""
    a = ListSource([{"text": f"A{i:03d}"} for i in range(200)])
    b = ListSource([{"text": f"B{i:03d}"} for i in range(200)])
    scheduler = WeightScheduler(
        [
            WeightSchedulePoint(step=0, weights={"a": 0.5, "b": 0.5}),
            WeightSchedulePoint(step=8, weights={"a": 1.0, "b": 0.0}),
        ],
        interpolation="step",
    )
    mix = MixedShardedSource(
        {"a": a, "b": b},
        block_size=10,
        seed=123,
        stop_strategy="restart",
        weight_scheduler=scheduler,
    )
    continuous = [r["text"] for _, r in zip(range(45), mix.open_shard("mixed_shard_0"), strict=False)]
    resumed = [r["text"] for _, r in zip(range(20), mix.open_shard_at_row("mixed_shard_0", 25), strict=False)]
    assert resumed == continuous[25:]


class _GuardSkippingSource(ShardedDataSource[dict]):
    """Two-shard source whose iteration delivers fewer rows than ``num_rows``.

    Models a parquet shard whose decode-memory guard silently skips oversized
    row groups: metadata promises 4+2 rows but shard ``s1`` delivers only one.
    """

    _ROWS: ClassVar[dict[str, list[str]]] = {"s0": ["a", "b", "c", "d"], "s1": ["e"]}

    @property
    def shard_names(self) -> Sequence[str]:
        return ["s0", "s1"]

    def num_shards(self) -> int:
        return 2

    @property
    def delivers_all_metadata_rows(self) -> bool:
        return False

    def get_shard_info(self, shard_name: str) -> ShardInfo:
        return ShardInfo(
            shard_id=int(shard_name[1]),
            shard_name=shard_name,
            num_rows={"s0": 4, "s1": 2}[shard_name],
        )

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        yield from ({"text": t} for t in self._ROWS[shard_name])


def test_shuffle_resume_exact_when_source_delivers_less_than_metadata():
    """Metadata-driven shard-boundary seeks must not be used for sources that
    can deliver fewer rows than num_rows promises: resume would start mid-stream
    at the wrong row and replay or drop rows."""
    src = _GuardSkippingSource()
    assert src.delivers_all_metadata_rows is False
    sh = src.shuffle(buffer_size=3, seed=5)
    full = [r["text"] for r in sh.open_shard("shuffled_shard_0")]
    assert sorted(full) == ["a", "b", "c", "d", "e"]  # delivered rows only
    for k in (1, 3):
        resumed = [r["text"] for r in sh.open_shard_at_row("shuffled_shard_0", k)]
        assert resumed == full[k:]


class _MetadataRowsSource(ShardedDataSource[dict]):
    """Finite shards with independently specified physical and delivered counts."""

    def __init__(self, rows_by_shard: dict[str, list[int]], reported_rows: dict[str, int] | None = None):
        self.rows_by_shard = rows_by_shard
        self.reported_rows = reported_rows or {name: len(rows) for name, rows in rows_by_shard.items()}

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self.rows_by_shard)

    @property
    def delivers_all_metadata_rows(self) -> bool:
        return all(len(rows) == self.reported_rows[name] for name, rows in self.rows_by_shard.items())

    def num_shards(self) -> int:
        return len(self.rows_by_shard)

    def get_shard_info(self, shard_name: str) -> ShardInfo:
        return ShardInfo(
            shard_id=self.shard_names.index(shard_name),
            shard_name=shard_name,
            num_rows=self.reported_rows[shard_name],
        )

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        yield from ({"id": row} for row in self.rows_by_shard[shard_name])


def test_shuffled_filtered_source_resumes_by_delivered_rows():
    source = _MetadataRowsSource({"s0": list(range(4)), "s1": list(range(4, 8))})
    filtered = source.filter(lambda row: row["id"] % 2 == 1)
    assert filtered.delivers_all_metadata_rows is False
    assert [row["id"] for row in filtered.open_shard_at_row("s0", 1)] == [3]
    shuffled = filtered.shuffle(buffer_size=1, seed=2, shuffle_shards=False)
    full = [row["id"] for row in shuffled.open_shard(shuffled.shard_names[0])]
    assert full == [1, 3, 5, 7]
    assert [row["id"] for row in shuffled.open_shard_at_row(shuffled.shard_names[0], 2)] == full[2:]


def test_restart_mixture_filtered_source_resumes_by_delivered_rows():
    source = _MetadataRowsSource({"s0": list(range(4)), "s1": list(range(4, 8))})
    filtered = source.filter(lambda row: row["id"] % 2 == 1)
    mixed = MixedShardedSource({"only": filtered}, block_size=1, seed=2, stop_strategy="restart")
    full = [row["id"] for row in itertools.islice(mixed.open_shard(mixed.shard_names[0]), 12)]
    assert full == [1, 3, 5, 7] * 3
    resumed = [row["id"] for row in itertools.islice(mixed.open_shard_at_row(mixed.shard_names[0], 2), 8)]
    assert resumed == full[2:10]


def test_restart_mixture_resumes_past_actual_row_cycle_not_metadata_row_cycle():
    source = _MetadataRowsSource({"s0": [10, 11]}, reported_rows={"s0": 4})
    mixed = MixedShardedSource({"only": source}, block_size=2, seed=3, stop_strategy="restart")
    full = [row["id"] for row in itertools.islice(mixed.open_shard(mixed.shard_names[0]), 8)]
    resumed = [row["id"] for row in itertools.islice(mixed.open_shard_at_row(mixed.shard_names[0], 3), 5)]
    assert resumed == full[3:8] == [11, 10, 11, 10, 11]


def test_restart_mixture_probes_positive_metadata_when_all_rows_are_skipped():
    live = _MetadataRowsSource({"s0": list(range(100))})
    skipped = _MetadataRowsSource({"s0": []}, reported_rows={"s0": 4})
    mixed = MixedShardedSource({"live": live, "skipped": skipped}, block_size=4, seed=0, stop_strategy="restart")
    full = [row["id"] for row in itertools.islice(mixed.open_shard(mixed.shard_names[0]), 18)]
    resumed = [row["id"] for row in itertools.islice(mixed.open_shard_at_row(mixed.shard_names[0], 7), 8)]
    assert resumed == full[7:15]


def test_limited_source_counts_delivered_rows_across_shard_boundaries():
    from easydel.data.transforms.source import LimitedShardedSource

    source = _MetadataRowsSource({"s0": [0], "s1": [4, 5, 6, 7]}, reported_rows={"s0": 4, "s1": 4})
    limited = LimitedShardedSource(source, max_rows=3)
    assert len(limited) == 3
    assert [row["id"] for shard in limited.shard_names for row in limited.open_shard(shard)] == [0, 4, 5]
    assert limited.get_shard_info("s0").num_rows == 1
    assert [row["id"] for row in limited.open_shard_at_row("s1", 1)] == [5]


def test_expanding_transform_resumes_by_output_row_not_input_row():
    class DuplicateRows(ExpandTransform):
        def __call__(self, example: dict) -> Iterator[dict]:
            yield example
            yield {"id": example["id"] + 100}

    source = _MetadataRowsSource({"s0": [0, 1], "s1": [2]})
    expanded = source.transform(DuplicateRows())
    assert expanded.delivers_all_metadata_rows is False
    assert [row["id"] for row in expanded.open_shard_at_row("s0", 1)] == [100, 1, 101]
    shuffled = expanded.shuffle(buffer_size=1, seed=2, shuffle_shards=False)
    full = [row["id"] for row in shuffled.open_shard(shuffled.shard_names[0])]
    assert [row["id"] for row in shuffled.open_shard_at_row(shuffled.shard_names[0], 3)] == full[3:]


def test_restart_mixture_inexact_resume_does_not_scan_entire_source():
    class CountingInexactSource(_MetadataRowsSource):
        @property
        def delivers_all_metadata_rows(self) -> bool:
            return False  # Like guarded Parquet, even when all physical rows happen to arrive.

        def __init__(self):
            super().__init__({"s0": list(range(10_000))})
            self.reads = 0

        def open_shard(self, shard_name: str) -> Iterator[dict]:
            for row in super().open_shard(shard_name):
                self.reads += 1
                yield row

    source = CountingInexactSource()
    mixed = MixedShardedSource({"only": source}, block_size=1, seed=0, stop_strategy="restart")
    resumed = mixed.open_shard_at_row(mixed.shard_names[0], 1)
    try:
        assert next(resumed)["id"] == 1
        assert source.reads <= 4  # One emptiness probe, one skipped row, one yielded row.
    finally:
        resumed.close()


def test_map_batch_native_seek_requires_explicit_row_count_opt_in():
    class NativeSeekSource(_MetadataRowsSource):
        def __init__(self):
            super().__init__({"s0": list(range(100))})
            self.reads = 0
            self.seek_rows: list[int] = []

        def open_shard(self, shard_name: str) -> Iterator[dict]:
            for row in super().open_shard(shard_name):
                self.reads += 1
                yield row

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
            self.seek_rows.append(row)
            return iter({"id": value} for value in self.rows_by_shard[shard_name][row:])

    class BatchOneToOne(Transform):
        def __call__(self, example: dict) -> dict:
            return example

        def map_batch(self, examples: list[dict]) -> list[dict]:
            return [dict(example) for example in examples]

    class OptedInBatchOneToOne(BatchOneToOne):
        preserves_row_count = True

    native = NativeSeekSource()
    opted_in = native.transform(OptedInBatchOneToOne())
    assert opted_in.delivers_all_metadata_rows is True
    assert next(opted_in.open_shard_at_row("s0", 90))["id"] == 90
    assert native.seek_rows == [64]  # Default map-batch size is 32; 90 aligns down to 64.
    assert native.reads == 0

    replay = NativeSeekSource()
    conservative = replay.transform(BatchOneToOne())
    assert conservative.delivers_all_metadata_rows is False
    assert next(conservative.open_shard_at_row("s0", 90))["id"] == 90
    assert replay.seek_rows == []
    assert replay.reads >= 91


def test_shuffle_resume_exact_when_short_shard_comes_first():
    """Order-independent variant: with shard order fixed (no shard shuffle) and the
    short shard FIRST, a metadata-driven seek is guaranteed to land in the wrong
    shard, so the sequential fallback is what keeps resume exact."""
    source = _MetadataRowsSource({"s0": [0], "s1": [4, 5, 6, 7]}, reported_rows={"s0": 4, "s1": 4})
    assert source.delivers_all_metadata_rows is False
    shuffled = source.shuffle(buffer_size=1, seed=3, shuffle_shards=False)
    name = shuffled.shard_names[0]
    full = [row["id"] for row in shuffled.open_shard(name)]
    assert sorted(full) == [0, 4, 5, 6, 7]
    for k in range(len(full) + 1):
        assert [row["id"] for row in shuffled.open_shard_at_row(name, k)] == full[k:], k


class _SeekCountingSource(_MetadataRowsSource):
    """Exact source that records native seeks and rows decoded by sequential iteration."""

    def __init__(self, n: int):
        super().__init__({"s0": list(range(n))})
        self.sequential_reads = 0
        self.seek_rows: list[int] = []

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        for row in super().open_shard(shard_name):
            self.sequential_reads += 1
            yield row

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        self.seek_rows.append(row)
        return iter({"id": value} for value in self.rows_by_shard[shard_name][row:])


def test_mix_resume_uses_native_seek_for_exact_sources():
    """An exact constituent is resumed through ``open_shard_at_row``, not by decoding
    and discarding the skipped prefix."""
    source = _SeekCountingSource(1000)
    assert source.delivers_all_metadata_rows is True
    mixed = MixedShardedSource({"only": source}, block_size=1, seed=0, stop_strategy="restart")
    resumed = mixed.open_shard_at_row(mixed.shard_names[0], 900)
    try:
        assert [next(resumed)["id"] for _ in range(5)] == [900, 901, 902, 903, 904]
    finally:
        resumed.close()
    assert source.seek_rows == [900]
    assert source.sequential_reads == 0


class _KeepOdd(Transform):
    """Filtering transform used to prove wrappers do not claim exactness they lack."""

    def __call__(self, example: dict) -> dict | None:
        return example if example["id"] % 2 else None

    @property
    def is_filter(self) -> bool:
        return True


class _Identity(Transform):
    def __call__(self, example: dict) -> dict:
        return example


def _wrapper_factories(source: ShardedDataSource) -> dict[str, ShardedDataSource]:
    """Every row-preserving wrapper around ``source`` that forwards the exactness flag."""
    from easydel.data.core.protocols import MappedShardedDataSource
    from easydel.data.execution.pipeline import (
        _ArrayFieldsOnlyShardedSource,
        _ParallelTransformedShardedSource,
        _ProgressBarShardedSource,
    )
    from easydel.data.sources.base import CompositeShardedSource

    return {
        "drop_fields": source.drop_fields(["x"]),
        "array_fields_only": _ArrayFieldsOnlyShardedSource(source),
        "progress_bar": _ProgressBarShardedSource(source, update_interval=10),
        "parallel_identity": _ParallelTransformedShardedSource(source, _Identity(), num_workers=1),
        "mapped": MappedShardedDataSource(source, lambda row: row),
        "composite": CompositeShardedSource([source]),
    }


def test_row_preserving_wrappers_forward_metadata_exactness_flag():
    """Wrappers that keep row counts must report the wrapped source's exactness, both ways."""
    exact = _MetadataRowsSource({"s0": [0, 1, 2]})
    inexact = _MetadataRowsSource({"s0": [0, 1, 2]}, reported_rows={"s0": 5})
    assert exact.delivers_all_metadata_rows is True
    assert inexact.delivers_all_metadata_rows is False
    for name, wrapper in _wrapper_factories(exact).items():
        assert wrapper.delivers_all_metadata_rows is True, name
    for name, wrapper in _wrapper_factories(inexact).items():
        assert wrapper.delivers_all_metadata_rows is False, name


def test_composite_source_is_inexact_if_any_member_is_inexact():
    from easydel.data.sources.base import CompositeShardedSource

    exact = _MetadataRowsSource({"s0": [0, 1]})
    inexact = _MetadataRowsSource({"s1": [2, 3]}, reported_rows={"s1": 9})
    assert CompositeShardedSource([exact, inexact]).delivers_all_metadata_rows is False
    assert CompositeShardedSource([exact]).delivers_all_metadata_rows is True


def test_parallel_transformed_source_filter_is_inexact_and_resumes_by_output_row():
    """A filtering transform must neither claim exactness nor seek upstream by INPUT row."""
    from easydel.data.execution.pipeline import _ParallelTransformedShardedSource

    source = _MetadataRowsSource({"s0": list(range(6))})
    parallel = _ParallelTransformedShardedSource(source, _KeepOdd(), num_workers=1, batch_size=2)
    assert parallel.delivers_all_metadata_rows is False
    full = [row["id"] for row in parallel.open_shard("s0")]
    assert full == [1, 3, 5]
    for k in range(len(full) + 1):
        assert [row["id"] for row in parallel.open_shard_at_row("s0", k)] == full[k:]


def test_row_preserving_batch_transform_resume_keeps_original_padding_groups():
    lengths = [8, 1, 3, 2, 5, 1, 4, 2, 7, 1, 1]

    class LongestPaddedBatch(Transform):
        preserves_row_count = True

        def __call__(self, example: dict) -> dict:
            raise AssertionError("batched transform must use map_batch")

        def map_batch(self, examples: list[dict]) -> list[dict]:
            max_length = max(lengths[example["id"]] for example in examples)
            return [
                {
                    "id": example["id"],
                    "input_ids": [example["id"] + 1] * lengths[example["id"]]
                    + [0] * (max_length - lengths[example["id"]]),
                    "attention_mask": [1] * lengths[example["id"]] + [0] * (max_length - lengths[example["id"]]),
                }
                for example in examples
            ]

    source = _MetadataRowsSource({"s0": list(range(len(lengths)))})
    transformed = TransformedShardedSource(source, LongestPaddedBatch(), map_batch_size=4)
    full = list(transformed.open_shard("s0"))
    assert len(full[1]["input_ids"]) == 8  # Batch 0-3 is padded to row 0.
    assert full[1]["attention_mask"] == [1, 0, 0, 0, 0, 0, 0, 0]
    for offset in (0, 1, 3, 4, 5, 8, 9, len(lengths)):
        assert list(transformed.open_shard_at_row("s0", offset)) == full[offset:]
