"""CPU-only regressions for legacy mixture filtering, caps, and reader selection."""

import gzip
import inspect
import json

import pytest
from datasets import Dataset, IterableDataset
from easydel.data import DatasetMixture, TextDatasetInform, VisualDatasetInform
from easydel.data.sources.base import load_for_inform

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("use_fast_loader", [False, True])
def test_filter_then_cap_before_formatting_and_mixing(tmp_path, streaming, use_fast_loader):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"content": str(i), "id": i}) + "\n" for i in range(8)))
    inform = TextDatasetInform(
        data_files=str(path),
        filter_fn=lambda row: row["id"] % 2 == 1,
        max_samples=2,
        format_callback=lambda row: {"content": "kept-" + row["content"]},
    )
    mixture = DatasetMixture(
        informs=[inform],
        streaming=streaming,
        use_fast_loader=use_fast_loader,
        cache_dir=str(tmp_path / "cache"),
        stop_strategy="first_exhausted",
    )
    assert list(mixture.build()) == [{"text": "kept-1"}, {"text": "kept-3"}]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("max_samples", [0, 3])
def test_empty_filtered_stream_supports_format_callback(tmp_path, streaming, max_samples):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"content": "dropped"}\n')
    mixture = DatasetMixture(
        informs=[
            TextDatasetInform(
                data_files=str(path),
                filter_fn=lambda row: False,
                max_samples=max_samples,
                format_callback=lambda row: {"content": row["content"].upper()},
            )
        ],
        cache_dir=str(tmp_path / "cache"),
        streaming=streaming,
        stop_strategy="first_exhausted",
    )
    assert list(mixture.build()) == []


@pytest.mark.parametrize("streaming", [False, True])
def test_raw_row_limit_remains_distinct_from_filtered_sample_limit(tmp_path, streaming):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"content": str(i), "id": i}) + "\n" for i in range(8)))
    mixture = DatasetMixture(
        informs=[
            TextDatasetInform(
                data_files=str(path),
                num_rows=3,
                filter_fn=lambda row: row["id"] % 2 == 1,
                max_samples=2,
            )
        ],
        cache_dir=str(tmp_path / "cache"),
        streaming=streaming,
        stop_strategy="first_exhausted",
    )
    assert list(mixture.build()) == [{"text": "1"}]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize("file_type", ["jsonl", "json", "csv", "tsv", "txt", "parquet", "arrow"])
def test_reader_modes_preserve_public_rows(tmp_path, streaming, use_fast_loader, file_type):
    rows = [{"text": "first"}, {"text": "second"}]
    path = tmp_path / f"rows.{file_type}"
    if file_type == "jsonl":
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif file_type == "json":
        path.write_text(json.dumps(rows))
    elif file_type in {"csv", "tsv"}:
        path.write_text("text\nfirst\nsecond\n")
    elif file_type == "txt":
        path.write_text("first\nsecond\n")
    elif file_type == "parquet":
        Dataset.from_list(rows).to_parquet(str(path))
    else:
        import pyarrow as pa

        with (
            pa.OSFile(str(path), "wb") as output,
            pa.ipc.new_stream(output, pa.schema([("text", pa.string())])) as writer,
        ):
            writer.write_table(pa.Table.from_pylist(rows))
    mixture = DatasetMixture(
        informs=[],
        cache_dir=str(tmp_path / "cache"),
        streaming=streaming,
        use_fast_loader=use_fast_loader,
    )
    loaded = load_for_inform(TextDatasetInform(data_files=str(path), content_field="text"), mixture)
    assert isinstance(loaded, IterableDataset if streaming else Dataset)
    assert list(loaded) == rows


@pytest.mark.parametrize("file_type", ["jsonl", "json", "tsv", "txt"])
@pytest.mark.parametrize("use_fast_loader", [False, True])
def test_reader_modes_support_compressed_files(tmp_path, file_type, use_fast_loader):
    path = tmp_path / f"rows.{file_type}.gz"
    payloads = {
        "jsonl": '{"text": "first"}\n{"text": "second"}\n',
        "json": '[{"text": "first"}, {"text": "second"}]',
        "tsv": "text\tid\nfirst\ta\nsecond\tb\n",
        "txt": "first\nsecond\n",
    }
    with gzip.open(path, "wt") as output:
        output.write(payloads[file_type])
    mixture = DatasetMixture(informs=[], cache_dir=str(tmp_path / "cache"), use_fast_loader=use_fast_loader)
    inform = TextDatasetInform(data_files=str(path), type=file_type, content_field="text")
    assert [row["text"] for row in load_for_inform(inform, mixture)] == ["first", "second"]


def test_fast_reader_preserves_arrow_file_container_support(tmp_path):
    import pyarrow as pa
    from easydel.data.sources.base import ArrowShardedSource

    path = tmp_path / "rows.arrow"
    table = pa.Table.from_pylist([{"text": "file-container"}])
    with pa.OSFile(str(path), "wb") as output, pa.ipc.new_file(output, table.schema) as writer:
        writer.write_table(table)
    source = ArrowShardedSource(str(path))
    assert list(source.open_shard(source.shard_names[0])) == [{"text": "file-container"}]


def test_fast_reader_works_without_huggingface_file_builder(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"text": "direct"}\n')

    def unavailable(*args, **kwargs):
        raise RuntimeError("standard file builder disabled")

    monkeypatch.setattr("easydel.data.sources.base._load_hf_dataset", unavailable)
    mixture = DatasetMixture(informs=[], cache_dir=str(tmp_path / "cache"), use_fast_loader=True)
    inform = TextDatasetInform(data_files=str(path), content_field="text")
    assert list(load_for_inform(inform, mixture)) == [{"text": "direct"}]
    mixture.use_fast_loader = False
    with pytest.raises(RuntimeError, match="standard file builder disabled"):
        load_for_inform(inform, mixture)


def test_new_options_do_not_shift_existing_positional_inform_arguments():
    for cls in (TextDatasetInform, VisualDatasetInform):
        parameters = inspect.signature(cls).parameters
        assert parameters["filter_fn"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["max_samples"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(DatasetMixture).parameters["use_fast_loader"].kind is inspect.Parameter.KEYWORD_ONLY


def test_negative_post_filter_cap_is_rejected():
    with pytest.raises(ValueError, match="max_samples must be >= 0"):
        TextDatasetInform(type="json", max_samples=-1)


@pytest.mark.parametrize("compression", ["xz", "bz2", "zst"])
@pytest.mark.parametrize("reader", ["pipeline", "legacy-fast", "legacy-standard"])
def test_compressed_tsv_autodetection_preserves_columns(tmp_path, compression, reader):
    import bz2
    import lzma

    import zstandard
    from easydel.data import DatasetConfig, create_source

    path = tmp_path / f"rows.tsv.{compression}"
    payload = b"text\tlabel\nfirst\ta\nsecond\tb\n"
    compress = {"xz": lzma.compress, "bz2": bz2.compress, "zst": zstandard.ZstdCompressor().compress}
    path.write_bytes(compress[compression](payload))
    if reader == "pipeline":
        source = create_source(DatasetConfig(data_files=str(path)))
        actual = [row for name in source.shard_names for row in source.open_shard(name)]
    else:
        mixture = DatasetMixture(informs=[], cache_dir=str(tmp_path / "cache"), use_fast_loader=reader == "legacy-fast")
        actual = list(load_for_inform(TextDatasetInform(data_files=str(path), content_field="text"), mixture))
    assert actual == [{"text": "first", "label": "a"}, {"text": "second", "label": "b"}]


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("local_cap", [None, 3])
def test_elarge_forwards_filter_cap_and_independent_raw_limit(visual, local_cap):
    from easydel.infra.elarge.builders import to_data_mixture_kwargs

    def predicate(row):
        return row["id"] >= 3

    inform_config = {"type": "json", "data_files": "unused.jsonl", "filter_fn": predicate, "num_rows": 9}
    if visual:
        inform_config["pixel_field"] = "images"
    if local_cap is not None:
        inform_config["max_samples"] = local_cap
    kwargs = to_data_mixture_kwargs(
        {"model": {"name_or_path": "unused"}, "mixture": {"informs": [inform_config], "max_samples": 2}}
    )
    inform = kwargs["informs"][0]
    assert isinstance(inform, VisualDatasetInform if visual else TextDatasetInform)
    assert inform.num_rows == 9
    assert inform.max_samples == (local_cap if local_cap is not None else 2)
    assert not inform.filter_fn({"id": 0})
    assert inform.filter_fn({"id": 3})


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("use_fast_loader", [False, True])
def test_elarge_legacy_rejected_early_rows_do_not_consume_sample_cap(tmp_path, streaming, use_fast_loader):
    from easydel.infra.elarge.builders import build_dataset

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"content": str(i), "id": i}) + "\n" for i in range(9)))
    cfg = {
        "model": {"name_or_path": "unused"},
        "mixture": {
            "informs": [
                {
                    "type": "json",
                    "data_files": str(path),
                    "num_rows": 7,
                    "filter_fn": lambda row: row["id"] >= 3 and row["id"] % 2 == 1,
                }
            ],
            "max_samples": 2,
            "streaming": streaming,
            "use_fast_loader": use_fast_loader,
            "shuffle": False,
            "stop_strategy": "first_exhausted",
            "cache_dir": str(tmp_path / "cache"),
        },
    }
    assert list(build_dataset(cfg)) == [{"text": "3"}, {"text": "5"}]


@pytest.mark.parametrize("use_fast_loader", [None, False, True])
def test_elarge_legacy_reader_selection_including_documented_default(tmp_path, monkeypatch, use_fast_loader):
    from easydel.infra.elarge.builders import build_dataset, to_data_mixture_kwargs

    path = tmp_path / "rows.jsonl"
    path.write_text('{"content":"direct"}\n')
    mixture_config = {
        "informs": [{"type": "json", "data_files": str(path)}],
        "shuffle": False,
        "stop_strategy": "first_exhausted",
        "cache_dir": str(tmp_path / "cache"),
    }
    if use_fast_loader is not None:
        mixture_config["use_fast_loader"] = use_fast_loader
    cfg = {"model": {"name_or_path": "unused"}, "mixture": mixture_config}
    assert to_data_mixture_kwargs(cfg)["use_fast_loader"] is (use_fast_loader is not False)

    def standard_reader_unavailable(*args, **kwargs):
        raise RuntimeError("standard file reader unavailable")

    monkeypatch.setattr("easydel.data.sources.base._load_hf_dataset", standard_reader_unavailable)
    if use_fast_loader is False:
        with pytest.raises(RuntimeError, match="standard file reader unavailable"):
            list(build_dataset(cfg))
    else:
        assert list(build_dataset(cfg)) == [{"text": "direct"}]
