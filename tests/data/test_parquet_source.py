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
        return original_read_row_group(self, i, columns=columns, *args, **kwargs)

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
        return original_read_row_group(self, i, columns=columns, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", flaky_read_row_group)

    source = ParquetShardedSource(str(data_path), columns=["messages"])

    rows = list(source.open_shard_at_row(source.shard_names[0], 1))

    assert rows == [{"messages": ["world"]}, {"messages": ["again"]}]


def test_parquet_source_falls_back_to_unprojected_batches(tmp_path, monkeypatch):
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

    original_iter_batches = pq.ParquetFile.iter_batches

    def failing_read_row_group(self, i, columns=None, *args, **kwargs):
        raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")

    def flaky_iter_batches(self, *args, **kwargs):
        if kwargs.get("columns") is not None:
            raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")
        return original_iter_batches(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", failing_read_row_group)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", flaky_iter_batches)

    source = ParquetShardedSource(str(data_path), columns=["messages"])

    rows = list(source.open_shard(source.shard_names[0]))

    assert rows == [{"messages": ["hello"]}, {"messages": ["world"]}]


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
    assert batch_calls == [
        (("messages", "tools"), None),
        (("messages", "tools"), 1),
        (("messages",), 1),
        (("tools",), 1),
    ]
