"""Text-only data builder regression coverage; no JAX computation."""

import json
from itertools import islice

import pytest
from datasets import Dataset
from easydel.infra.elarge.builders import (
    build_dataset,
    build_sharded_source,
    build_tokenized_dataset,
    save_dataset,
    to_data_mixture_kwargs,
    tokenize_dataset,
)

pytestmark = pytest.mark.cpu_ok


def config(informs, **mixture):
    return {"model": {"name_or_path": "unused"}, "mixture": {"informs": informs, "shuffle_buffer_size": 0, **mixture}}


def rows(source):
    for shard in source.shard_names:
        yield from source.open_shard(shard)


@pytest.mark.parametrize("use_fast_loader", [True, False])
def test_per_dataset_transforms_caps_and_shuffle_control(tmp_path, use_fast_loader):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"content": str(i), "number": i}) + "\n" for i in range(6)))
    cfg = config(
        [
            {
                "type": "json",
                "data_files": str(path),
                "filter_fn": lambda row: row["number"] % 2 == 0,
                "map_fn": lambda row: {**row, "content": "item-" + row["content"]},
                "max_samples": 2,
            }
        ],
        use_fast_loader=use_fast_loader,
        shuffle=False,
        shuffle_buffer_size=100,
    )
    assert [row["text"] for row in rows(build_sharded_source(cfg))] == ["item-0", "item-2"]


def test_global_max_samples_applies_per_source(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"content": str(i)}) + "\n" for i in range(6)))
    cfg = config([{"type": "json", "data_files": str(path)}], max_samples=2)
    assert [row["text"] for row in rows(build_sharded_source(cfg))] == ["0", "1"]
    inform = to_data_mixture_kwargs(cfg)["informs"][0]
    assert inform.max_samples == 2
    assert inform.num_rows is None


def test_inform_weights_control_observable_mixing(tmp_path):
    paths = []
    for label in ("a", "b"):
        path = tmp_path / f"{label}.jsonl"
        path.write_text(json.dumps({"content": label}) + "\n")
        paths.append(path)
    informs = [
        {"name": label, "type": "json", "data_files": str(path), "weight": weight}
        for label, path, weight in zip(("a", "b"), paths, (1.0, 0.0), strict=True)
    ]
    cfg = config(informs, mixture_block_size=4, stop_strategy="restart")
    assert [row["text"] for row in islice(rows(build_sharded_source(cfg)), 8)] == ["a"] * 8
    assert to_data_mixture_kwargs(cfg)["mixture_weights"] == {"a": 1.0, "b": 0.0}


def test_legacy_mixture_honors_map_caps_and_shuffle_flag(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"content": str(i)}) + "\n" for i in range(4)))
    cfg = config(
        [
            {
                "type": "json",
                "data_files": str(path),
                "max_samples": 2,
                "map_fn": lambda row: {"content": "x" + row["content"]},
            }
        ],
        streaming=False,
        shuffle=False,
        shuffle_buffer_size=100,
    )
    assert list(build_dataset(cfg)["text"]) == ["x0", "x1"]


def test_hf_type_shorthand_matches_explicit_location():
    shorthand = to_data_mixture_kwargs(config([{"type": "owner/dataset"}]))["informs"][0]
    explicit = to_data_mixture_kwargs(config([{"type": "hf", "data_files": "owner/dataset"}]))["informs"][0]
    assert shorthand.data_files == explicit.data_files == "owner/dataset"
    assert shorthand.get_str_type() == explicit.get_str_type()


@pytest.mark.parametrize("file_format", ["jsonl", "parquet", "arrow"])
def test_public_save_dataset_preserves_rows_and_overwrite_guard(tmp_path, file_format):
    destination = tmp_path / file_format
    dataset = Dataset.from_list([{"text": "hello"}, {"text": "world"}])
    assert save_dataset(dataset, str(destination), format=file_format) == str(destination)
    if file_format == "jsonl":
        restored = [json.loads(line) for line in (destination / "data.jsonl").read_text().splitlines()]
    elif file_format == "parquet":
        restored = Dataset.from_parquet(str(destination / "data.parquet")).to_list()
    else:
        restored = Dataset.load_from_disk(str(destination)).to_list()
    assert restored == dataset.to_list()
    with pytest.raises(FileExistsError):
        save_dataset(dataset, str(destination), format=file_format)


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        ids = [[ord(char) for char in text] for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def test_documented_public_tokenization_chain_remains_usable(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    dataset = Dataset.from_list([{"text": "ab"}])
    assert tokenize_dataset(dataset, TinyTokenizer())[0]["tokens"] == [97, 98]
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"content": "ab"}) + "\n")
    cfg = config(
        [{"type": "json", "data_files": str(path)}],
        streaming=False,
        save={"output_path": str(tmp_path / "output"), "format": "jsonl"},
    )
    tokenized, saved_path = build_tokenized_dataset(cfg)
    assert tokenized[0]["tokens"] == [97, 98]
    assert json.loads((tmp_path / "output" / "data.jsonl").read_text())["tokens"] == [97, 98]
    assert saved_path == str(tmp_path / "output")
