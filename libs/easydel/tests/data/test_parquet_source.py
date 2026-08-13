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

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

from easydel.data.sources.base import ParquetShardedSource


def test_parquet_source_retries_nested_projection_error_single_threaded(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    data_path = tmp_path / "train.parquet"
    table = pa.table(
        {
            "messages": [["hello"], ["world"]],
            "unused_blob": ["x" * 1024, "y" * 1024],
        }
    )
    pq.write_table(table, data_path, row_group_size=1)

    original_read_row_group = pq.ParquetFile.read_row_group
    read_calls: list[tuple[int, tuple[str, ...] | None, bool]] = []

    def flaky_read_row_group(self, i, columns=None, *args, **kwargs):
        use_threads = kwargs.get("use_threads", True)
        read_calls.append((i, tuple(columns) if columns is not None else None, use_threads))
        if columns is not None and use_threads:
            raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")
        if args:
            return original_read_row_group(self, i, columns, *args, **kwargs)
        return original_read_row_group(self, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", flaky_read_row_group)

    source = ParquetShardedSource(str(data_path), columns=["messages"])

    rows = list(source.open_shard(source.shard_names[0]))

    assert rows == [{"messages": ["hello"]}, {"messages": ["world"]}]
    assert read_calls == [
        (0, ("messages",), True),
        (0, ("messages",), False),
        (1, ("messages",), False),
    ]


def test_parquet_source_falls_back_to_projected_batches(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    data_path = tmp_path / "train.parquet"
    table = pa.table(
        {
            "messages": [["hello"], ["world"]],
            "unused_blob": ["x" * 1024, "y" * 1024],
        }
    )
    pq.write_table(table, data_path, row_group_size=1)

    projected_calls: list[tuple[int, tuple[str, ...], bool]] = []
    full_calls: list[int] = []

    def failing_projected_read_row_group(self, i, columns=None, *args, **kwargs):
        if columns is not None:
            projected_calls.append((i, tuple(columns), kwargs.get("use_threads", True)))
            raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")
        full_calls.append(i)
        raise AssertionError("projected batch fallback should not read full row groups")

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", failing_projected_read_row_group)

    source = ParquetShardedSource(str(data_path), columns=["messages"])

    rows = list(source.open_shard(source.shard_names[0]))

    assert rows == [{"messages": ["hello"]}, {"messages": ["world"]}]
    assert projected_calls == [
        (0, ("messages",), True),
        (0, ("messages",), False),
        (1, ("messages",), False),
    ]
    assert full_calls == []


def test_parquet_source_projection_fallback_respects_start_row(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    data_path = tmp_path / "train.parquet"
    table = pa.table(
        {
            "messages": [["hello"], ["world"], ["again"]],
            "unused_blob": ["x" * 1024, "y" * 1024, "z" * 1024],
        }
    )
    pq.write_table(table, data_path, row_group_size=2)

    original_read_row_group = pq.ParquetFile.read_row_group

    def flaky_read_row_group(self, i, columns=None, *args, **kwargs):
        if columns is not None:
            raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")
        if args:
            return original_read_row_group(self, i, columns, *args, **kwargs)
        return original_read_row_group(self, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", flaky_read_row_group)

    source = ParquetShardedSource(str(data_path), columns=["messages"])

    rows = list(source.open_shard_at_row(source.shard_names[0], 1))

    assert rows == [{"messages": ["world"]}, {"messages": ["again"]}]


def test_parquet_source_never_falls_back_to_unprojected_batches(tmp_path, monkeypatch):
    # Unprojected full-row-group decodes are the memory bomb for embed-heavy
    # files: if even single-column projected reads fail, the error must
    # propagate instead of silently decoding every column.
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    data_path = tmp_path / "train.parquet"
    table = pa.table(
        {
            "messages": [["hello"], ["world"]],
            "unused_blob": ["x" * 1024, "y" * 1024],
        }
    )
    pq.write_table(table, data_path, row_group_size=1)

    unprojected_calls: list[tuple] = []
    original_iter_batches = pq.ParquetFile.iter_batches

    def failing_read_row_group(self, i, columns=None, *args, **kwargs):
        raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")

    def flaky_iter_batches(self, *args, **kwargs):
        if kwargs.get("columns") is not None:
            raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")
        unprojected_calls.append((args, kwargs))
        return original_iter_batches(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", failing_read_row_group)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", flaky_iter_batches)

    source = ParquetShardedSource(str(data_path), columns=["messages"])

    with pytest.raises(pa.ArrowNotImplementedError):
        list(source.open_shard(source.shard_names[0]))

    assert unprojected_calls == []


def test_parquet_source_falls_back_to_projected_column_batches(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    data_path = tmp_path / "train.parquet"
    table = pa.table(
        {
            "messages": [["hello"], ["world"]],
            "tools": [["search"], ["calc"]],
            "unused_blob": ["x" * 1024, "y" * 1024],
        }
    )
    pq.write_table(table, data_path, row_group_size=2)

    original_iter_batches = pq.ParquetFile.iter_batches
    batch_calls: list[tuple[tuple[str, ...] | None, int | None]] = []

    def failing_read_row_group(self, i, columns=None, *args, **kwargs):
        raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")

    def flaky_iter_batches(self, *args, **kwargs):
        columns = kwargs.get("columns")
        batch_calls.append((tuple(columns) if columns is not None else None, kwargs.get("batch_size")))
        if columns is None or len(columns) > 1:
            raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")
        return original_iter_batches(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", failing_read_row_group)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", flaky_iter_batches)

    source = ParquetShardedSource(str(data_path), columns=["messages", "tools"])

    rows = list(source.open_shard(source.shard_names[0]))

    assert rows == [
        {"messages": ["hello"], "tools": ["search"]},
        {"messages": ["world"], "tools": ["calc"]},
    ]
    # The byte-budgeted ladder retries the multi-column projection at
    # descending batch sizes before resorting to per-column reads.
    assert batch_calls == [
        (("messages", "tools"), 256),
        (("messages", "tools"), 64),
        (("messages", "tools"), 16),
        (("messages", "tools"), 4),
        (("messages", "tools"), 1),
        (("messages",), 256),
        (("tools",), 256),
    ]


def test_null_element_inside_list_takes_python_path_not_float64_nan():
    """A null ELEMENT inside a non-null list survives arr.null_count == 0 (which
    only counts null lists); pyarrow's to_numpy() on the flattened child then
    silently upcasts the ENTIRE column chunk to float64 with NaN in place of
    the null -- every row in the batch, not just the offending one, and a
    downstream int cast turns the NaN into a garbage token id with no error
    anywhere. The fast path must detect element-level nulls and route that
    column through the python materialization instead."""
    pa = pytest.importorskip("pyarrow")
    np = pytest.importorskip("numpy")

    table = pa.table(
        {
            "input_ids": pa.array([[101, 202, None, 404], [11, 22, 33, 44]], type=pa.list_(pa.int64())),
            "clean_ids": pa.array([[7, 8], [9, 10]], type=pa.list_(pa.int64())),
            "label": pa.array([1, 0]),
        }
    )
    rows = list(ParquetShardedSource._table_to_rows(table))
    assert len(rows) == 2

    # The nulled column keeps exact values with None preserved -- no float64, no NaN.
    assert rows[0]["input_ids"] == [101, 202, None, 404]
    assert rows[1]["input_ids"] == [11, 22, 33, 44]
    assert all(isinstance(v, int) for v in rows[1]["input_ids"])

    # Columns without nulls anywhere keep the numpy fast path.
    assert isinstance(rows[0]["clean_ids"], np.ndarray)
    assert rows[0]["clean_ids"].dtype == np.int64
    np.testing.assert_array_equal(rows[0]["clean_ids"], [7, 8])


def test_numpy_path_failure_falls_back_before_first_row_no_duplicates():
    """All fallible numpy-path work must happen at call time, BEFORE any row is
    yielded: a failure after k yielded rows would fall back to the python path
    and re-yield from row 0, silently duplicating the k rows already consumed.
    Simulated via a table whose second column raises during materialization --
    the fallback must produce every row exactly once."""
    pa = pytest.importorskip("pyarrow")

    real = pa.table({"ok": [1, 2, 3], "bad": ["a", "b", "c"]})

    class PoisonedTable:
        num_rows = real.num_rows
        column_names = real.column_names

        @staticmethod
        def column(name):
            if name == "bad":
                raise RuntimeError("materialization failure")
            return real.column(name)

        @staticmethod
        def to_pydict():
            return real.to_pydict()

    rows = list(ParquetShardedSource._table_to_rows(PoisonedTable()))
    assert [r["ok"] for r in rows] == [1, 2, 3], "fallback duplicated or dropped rows"
    assert [r["bad"] for r in rows] == ["a", "b", "c"]


def test_midstream_iteration_failure_propagates_instead_of_duplicating():
    """The regression this guards: pre-fix, _table_to_rows wrapped the WHOLE
    numpy-path iteration in try/except, so an exception thrown after k rows
    were already yielded fell back to the python path and re-yielded from row
    0 -- duplicating the k consumed rows with no error surfaced. Post-fix,
    only call-time materialization may fall back; a mid-iteration failure
    must propagate, never duplicate."""
    pa = pytest.importorskip("pyarrow")

    real = pa.table({"ok": [1, 2, 3]})

    class BadPylist:
        def __getitem__(self, i):
            if i >= 1:
                raise RuntimeError("row 1 explodes")
            return "a"

    class BadColumn:
        type = pa.string()  # not list / not primitive -> routed to the pylist path

        @staticmethod
        def to_pylist():
            return BadPylist()

    class MidstreamPoisonedTable:
        num_rows = real.num_rows
        column_names = ("ok", "bad")

        @staticmethod
        def column(name):
            return BadColumn() if name == "bad" else real.column(name)

        @staticmethod
        def to_pydict():
            return {"ok": [1, 2, 3], "bad": ["a", "b", "c"]}

    collected = []
    with pytest.raises(RuntimeError, match="row 1 explodes"):
        for row in ParquetShardedSource._table_to_rows(MidstreamPoisonedTable()):
            collected.append(int(row["ok"]))
    assert collected == [1], f"rows re-yielded after mid-stream failure: {collected}"
