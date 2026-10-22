"""Regression coverage for writer overwrite, compression, and partition contracts."""

import gzip
import json

import pyarrow.ipc as ipc
import pytest
from easydel.data import (
    ArrowShardedSource,
    ArrowWriter,
    DatasetConfig,
    HFDatasetShardedSource,
    JsonlWriter,
    PipelineConfig,
    PipelineContext,
    SaveStage,
    SaveStageConfig,
    create_writer,
    pretokenize,
)

pytestmark = pytest.mark.cpu_ok


def source(rows):
    return HFDatasetShardedSource(rows)


@pytest.mark.parametrize("old_codec", [None, "gzip"])
@pytest.mark.parametrize("new_codec", [None, "gzip"])
def test_jsonl_detects_both_suffixes_and_replaces_stale_tail(tmp_path, old_codec, new_codec):
    rows = [{"text": str(index)} for index in range(4)]
    JsonlWriter(str(tmp_path), max_shard_size=1, compression=old_codec).write(source(rows))
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        JsonlWriter(str(tmp_path), compression=new_codec).write(source(rows[:1]))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original
    result = JsonlWriter(str(tmp_path), compression=new_codec, overwrite=True).write(source(rows[:1]))
    assert result.num_shards == 1
    paths = list(tmp_path.iterdir())
    assert len(paths) == 1
    text = gzip.decompress(paths[0].read_bytes()).decode() if new_codec else paths[0].read_text()
    assert [json.loads(line) for line in text.splitlines()] == rows[:1]


@pytest.mark.parametrize("codec", ["zstd", "lz4"])
def test_arrow_compression_roundtrip_and_size(tmp_path, codec):
    rows = [{"text": "a predictable repeated payload " * 300} for _ in range(100)]
    plain = ArrowWriter(str(tmp_path / "plain")).write(source(rows))
    compressed = ArrowWriter(str(tmp_path / codec), compression=codec).write(source(rows))
    assert compressed.total_bytes < plain.total_bytes / 3
    reader = ArrowShardedSource(compressed.output_paths)
    assert list(reader.open_shard(reader.shard_names[0])) == rows
    with ipc.open_file(compressed.output_paths[0]) as file:
        assert file.read_all().to_pylist() == rows


@pytest.mark.parametrize("format", ["jsonl", "arrow", "parquet"])
@pytest.mark.parametrize("key", [None, "group"])
def test_save_stage_honors_dataset_partition_settings(tmp_path, format, key):  # noqa: A002
    rows = [{"text": str(index), "group": index % 3} for index in range(12)]
    dataset = DatasetConfig("unused", name="demo", shard_column=key, num_shards=3)
    config = PipelineConfig(
        datasets=[dataset],
        save=SaveStageConfig(enabled=True, format=format, output_dir=str(tmp_path), max_shard_size=100),
    )
    stage = SaveStage(config.save)
    stage.process({"demo": source(rows)}, PipelineContext(config))
    paths = sorted((tmp_path / "demo").iterdir())
    assert len(paths) == 3
    partitions = []
    for path in paths:
        if format == "jsonl":
            partitions.append([json.loads(line) for line in path.read_text().splitlines()])
        elif format == "arrow":
            with ipc.open_file(str(path)) as file:
                partitions.append(file.read_all().to_pylist())
        else:
            import pyarrow.parquet as pq

            partitions.append(pq.read_table(path).to_pylist())
    assert sorted((row for part in partitions for row in part), key=lambda row: int(row["text"])) == rows
    if key is None:
        assert [[row["text"] for row in part] for part in partitions] == [
            [str(index) for index in range(offset, 12, 3)] for offset in range(3)
        ]
    else:
        locations = {}
        for index, part in enumerate(partitions):
            for row in part:
                assert locations.setdefault(row["group"], index) == index


def test_pretokenize_drop_fields_does_not_mutate_input_and_preserves_other_fields(tmp_path):
    rows = [{"text": "a", "keep": 1}, {"text": "b", "keep": 2}]
    stats = pretokenize(
        source(rows),
        lambda row: dict(row, input_ids=[row["keep"]]),
        str(tmp_path),
        output_format="jsonl",
        drop_fields=["text", "absent"],
        arrays_only=False,
        show_progress=False,
        log_process=False,
    )
    assert rows == [{"text": "a", "keep": 1}, {"text": "b", "keep": 2}]
    with open(stats.output_paths[0]) as file:
        assert [json.loads(line) for line in file] == [{"keep": 1, "input_ids": [1]}, {"keep": 2, "input_ids": [2]}]


def test_invalid_arrow_codec_does_not_remove_existing_output(tmp_path):
    existing = tmp_path / "shard-00000.arrow"
    existing.write_bytes(b"keep me")
    with pytest.raises(ValueError):
        create_writer(str(tmp_path), format="arrow", compression="gzip", overwrite=True).write(source([{"id": 1}]))
    assert existing.read_bytes() == b"keep me"


@pytest.mark.parametrize("format", ["arrow", "parquet"])
@pytest.mark.parametrize("num_shards", [1, 3, 8])
@pytest.mark.parametrize("shard_column", [None, "group"])
def test_partitioned_token_rows_share_schema_across_flushes_and_empty_partitions(
    tmp_path,
    format,  # noqa: A002
    num_shards,
    shard_column,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {"id": 0, "group": 0, "input_ids": None, "score": None, "meta": {"early": 1}},
        {"id": 1, "group": 0, "input_ids": [], "meta": None},
        {"id": 2, "group": 0, "input_ids": [1], "score": 1, "late": "present"},
        {"id": 3, "group": 0, "input_ids": [2, 3], "score": 1.5, "meta": {"later": "yes"}},
    ]
    result = create_writer(
        str(tmp_path), format=format, num_shards=num_shards, shard_column=shard_column, max_shard_size=1
    ).write(source(rows))
    tables = []
    for path in result.output_paths:
        if format == "arrow":
            with ipc.open_file(path) as file:
                tables.append(file.read_all())
        else:
            tables.append(pq.read_table(path))
    assert result.num_shards == num_shards
    assert result.num_examples == len(rows)
    assert all(table.schema == tables[0].schema for table in tables)
    assert pa.types.is_list(tables[0].schema.field("input_ids").type)
    actual = sorted((row for table in tables for row in table.to_pylist()), key=lambda row: row["id"])
    assert [row["input_ids"] for row in actual] == [None, [], [1], [2, 3]]
    assert [row["score"] for row in actual] == [None, None, 1.0, 1.5]
    assert [row["late"] for row in actual] == [None, None, "present", None]
    assert [row["meta"] for row in actual] == [
        {"early": 1, "later": None},
        None,
        None,
        {"early": None, "later": "yes"},
    ]
    if num_shards > len(rows) or (shard_column and num_shards > 1):
        assert any(table.num_rows == 0 for table in tables)


@pytest.mark.parametrize("format", ["arrow", "parquet"])
def test_partitioned_variable_length_tokens_roundtrip(tmp_path, format):  # noqa: A002
    import pyarrow.parquet as pq

    rows = [{"input_ids": [1]}, {"input_ids": [2, 3]}]
    stats = create_writer(str(tmp_path), format=format, num_shards=1, max_shard_size=1).write(source(rows))
    if format == "arrow":
        with ipc.open_file(stats.output_paths[0]) as file:
            actual = file.read_all().to_pylist()
    else:
        actual = pq.read_table(stats.output_paths[0]).to_pylist()
    assert actual == rows


@pytest.mark.parametrize("format", ["arrow", "parquet"])
@pytest.mark.parametrize("max_shard_size", [1, 1024])
def test_partition_schema_preserves_heterogeneous_metadata_fallback(tmp_path, format, max_shard_size):  # noqa: A002
    import pyarrow.parquet as pq

    rows = [{"metadata": value} for value in [{"value": 1}, {"value": "two"}, 3, None]]
    stats = create_writer(str(tmp_path), format=format, num_shards=2, max_shard_size=max_shard_size).write(source(rows))
    actual = []
    for path in stats.output_paths:
        if format == "arrow":
            with ipc.open_file(path) as file:
                actual.extend(file.read_all().to_pylist())
        else:
            actual.extend(pq.read_table(path).to_pylist())
    # Each partition receives alternating rows; metadata stays JSON rather than losing values.
    assert [json.loads(row["metadata"]) if row["metadata"] is not None else None for row in actual] == [
        {"value": 1},
        3,
        {"value": "two"},
        None,
    ]


@pytest.mark.parametrize("format", ["arrow", "parquet"])
def test_partitioned_empty_dataset_produces_readable_empty_files(tmp_path, format):  # noqa: A002
    import pyarrow.parquet as pq

    stats = create_writer(str(tmp_path), format=format, num_shards=3, max_shard_size=1).write(source([]))
    assert stats.num_examples == 0
    assert len(stats.output_paths) == 3
    for path in stats.output_paths:
        if format == "arrow":
            with ipc.open_file(path) as file:
                assert file.read_all().to_pylist() == []
        else:
            assert pq.read_table(path).to_pylist() == []
