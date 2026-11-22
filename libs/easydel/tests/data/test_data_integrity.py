"""CPU-only public data-integrity regressions: readers, writers, disk cache, sharding and profiler."""

import json

import jax
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pytest
from easydel.data import (
    CacheMetadata,
    DatasetMixture,
    DiskCache,
    HFDatasetShardedSource,
    ShardingSpec,
    TextDatasetInform,
    create_writer,
)
from easydel.data.execution.save import _rows_to_arrow_table
from easydel.data.profiling import DatasetProfiler
from easydel.data.sources.base import CsvShardedSource, load_for_inform
from jax.sharding import Mesh, PartitionSpec

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("use_fast_loader", [False, True])
def test_late_optional_columns_survive_raw_loading_and_mixture_build(tmp_path, use_fast_loader):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"content":"first"}\n{"content":"second","label":7}\n')
    mixture = DatasetMixture(
        informs=[TextDatasetInform(data_files=str(path), additional_fields=["label"])],
        streaming=False,
        use_fast_loader=use_fast_loader,
        cache_dir=str(tmp_path / "cache"),
        stop_strategy="first_exhausted",
    )
    assert list(load_for_inform(mixture.informs[0], mixture)) == [
        {"content": "first", "label": None},
        {"content": "second", "label": 7},
    ]
    assert list(mixture.build()) == [{"text": "first", "label": None}, {"text": "second", "label": 7}]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("delimiter", [",", "\t"])
def test_csv_fast_and_standard_readers_preserve_numeric_null_types(tmp_path, streaming, delimiter):
    path = tmp_path / ("rows.csv" if delimiter == "," else "rows.tsv")
    path.write_text("text,label,score,nullable,flag\nfirst,7,0.5,3,True\nsecond,8,0.75,,False\n".replace(",", delimiter))
    inform = TextDatasetInform(data_files=str(path), content_field="text")
    outputs = []
    for use_fast_loader in (False, True):
        mixture = DatasetMixture(
            informs=[], streaming=streaming, use_fast_loader=use_fast_loader, cache_dir=str(tmp_path / "cache")
        )
        outputs.append(list(load_for_inform(inform, mixture)))
    expected = [
        {"text": "first", "label": 7, "score": 0.5, "nullable": 3.0, "flag": True},
        {"text": "second", "label": 8, "score": 0.75, "nullable": None, "flag": False},
    ]
    assert outputs == [expected, expected]
    assert all(type(row["label"]) is int and type(row["score"]) is float for rows in outputs for row in rows)
    # The opt-in legacy reader must not alter the old pipeline source API.
    source = CsvShardedSource(str(path), delimiter=delimiter)
    assert next(iter(source.open_shard(source.shard_names[0])))["label"] == "7"


@pytest.mark.parametrize("format", ["arrow", "parquet", "jsonl"])
@pytest.mark.parametrize("fixed_count", [False, True])
def test_one_shot_source_survives_partition_discovery(tmp_path, format, fixed_count):  # noqa: A002
    rows = [{"input_ids": [1, 2], "group": "a"}, {"input_ids": [3], "group": "b"}]
    result = create_writer(
        str(tmp_path),
        format=format,
        num_shards=3 if fixed_count else None,
        shard_column="group",
        max_shard_size=30,
    ).write(HFDatasetShardedSource(iter(rows)))
    assert result.num_examples == 2
    if format == "jsonl":
        actual = []
        for path in result.output_paths:
            with open(path) as file:
                actual.extend(json.loads(line) for line in file)
    else:
        actual = (
            pads.dataset(result.output_paths, format="ipc" if format == "arrow" else "parquet").to_table().to_pylist()
        )
    assert sorted(actual, key=lambda row: row["group"]) == rows
    if fixed_count:
        assert result.num_shards == 3


@pytest.mark.parametrize("format", ["arrow", "parquet"])
def test_size_rotated_variable_length_lists_form_one_readable_dataset(tmp_path, format):  # noqa: A002
    rows = [{"ids": [1]}, {"ids": [2, 3]}]
    result = create_writer(str(tmp_path), format=format, max_shard_size=1).write(HFDatasetShardedSource(rows))
    assert result.num_shards == 2
    table = pads.dataset(result.output_paths, format="ipc" if format == "arrow" else "parquet").to_table()
    assert pa.types.is_list(table.schema.field("ids").type)
    assert table.to_pylist() == rows


@pytest.mark.parametrize("format", ["arrow", "parquet"])
@pytest.mark.parametrize(
    "values",
    [
        np.array([-(2**40), 2**40 + 1], dtype=np.int64),
        np.array([-(2**31) - 1, 2**31], dtype=np.int64),
        np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64),
        np.array([-(2**31), 2**31 - 1], dtype=np.int64),
    ],
)
def test_arbitrary_integer_lists_roundtrip_without_narrowing(tmp_path, format, values):  # noqa: A002
    rows = [{"ids": values}]
    result = create_writer(str(tmp_path), format=format).write(HFDatasetShardedSource(rows))
    table = pads.dataset(result.output_paths, format="ipc" if format == "arrow" else "parquet").to_table()
    expected = [{"ids": values.tolist()}]
    assert table.to_pylist() == expected
    # Explicit fixed-width conversion remains lossless too, including uint64 max.
    assert _rows_to_arrow_table(rows, pa).to_pylist() == expected


def test_disk_overwrite_without_metadata_clears_previous_provenance(tmp_path):
    cache = DiskCache(str(tmp_path))
    cache.put("entry", "old", CacheMetadata(config_hash="old-config"))
    cache.put("entry", "new")
    assert cache.get("entry") == ("new", None)
    assert DiskCache(str(tmp_path)).get("entry") == ("new", None)


@pytest.mark.parametrize("ambient", [False, True])
def test_sharding_spec_uses_explicit_mesh_without_device_computation(monkeypatch, ambient):
    from contextlib import nullcontext

    import spectrax as spx

    devices = np.asarray(jax.devices())
    requested = Mesh(devices, ("requested",))
    other = Mesh(devices, ("other",))
    # Inspect the placement requested by the public API without dispatching JAX numerics.
    monkeypatch.setattr(jax, "device_put", lambda array, sharding: sharding)
    with spx.sharding.use_mesh(other) if ambient else nullcontext():
        result = ShardingSpec(requested, {"ids": PartitionSpec("requested")}).apply({"ids": np.zeros((8,))})
        assert result["ids"].mesh.axis_names == ("requested",)
        assert result["ids"].spec == PartitionSpec("requested")
        if ambient:
            assert spx.get_incontext_mesh().axis_names == ("other",)


@pytest.mark.parametrize("max_rows", [0, 1])
def test_profiler_honors_exact_sample_budget_and_closes_upstream(max_rows):
    class Upstream:
        def __init__(self):
            self.reads = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            self.reads += 1
            if self.reads > max_rows:
                raise RuntimeError("read beyond sample budget")
            return {"input_ids": [1, 2]}

        def close(self):
            self.closed = True

    upstream = Upstream()

    class Source:
        def iter_shards(self):
            return upstream

    result = DatasetProfiler(max_rows=max_rows).profile(Source())
    assert result.num_rows_sampled == max_rows
    assert result.total_input_tokens == max_rows * 2
    assert upstream.reads == max_rows
    assert upstream.closed


def test_profiler_rejects_negative_sample_budget():
    with pytest.raises(ValueError, match="max_rows must be non-negative"):
        DatasetProfiler(max_rows=-1)
