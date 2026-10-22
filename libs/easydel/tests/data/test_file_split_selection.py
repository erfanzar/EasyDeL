"""CPU-only split-selection parity for the legacy file reader modes."""

import json

import pytest
from datasets import Dataset, IterableDataset
from easydel.data import DatasetMixture, TextDatasetInform
from easydel.data.sources.base import load_for_inform

pytestmark = pytest.mark.cpu_ok


def _inform_and_mixture(tmp_path, use_fast_loader, split, *, streaming=False, num_rows=None):
    files = []
    for shard in range(2):
        path = tmp_path / f"rows-{shard}.jsonl"
        path.write_text("".join(json.dumps({"text": str(i)}) + "\n" for i in range(shard * 5, (shard + 1) * 5)))
        files.append(str(path))
    return (
        TextDatasetInform(data_files=files, split=split, content_field="text", num_rows=num_rows),
        DatasetMixture(
            informs=[],
            streaming=streaming,
            use_fast_loader=use_fast_loader,
            cache_dir=str(tmp_path / "cache"),
        ),
    )


@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize(
    ("split", "expected"),
    [
        ("train[:1]", [0]),
        ("train[3:7]", [3, 4, 5, 6]),
        ("train[-2:]", [8, 9]),
        ("train[:20%]", [0, 1]),
        ("train[20%:60%]", [2, 3, 4, 5]),
        ("train[-40%:]", [6, 7, 8, 9]),
        ("train[:1]+train[-2:]", [0, 8, 9]),
        ("train[2:4]+train[1:3]", [2, 3, 1, 2]),
        ("train[8:100]", [8, 9]),
        ("train[7:3]", []),
        # HF's Arrow reader treats a take of -1 as "through the end".
        ("train[3:2]", list(range(3, 10))),
        ("train[30%:20%]", list(range(3, 10))),
        ("train[-3:-4]", [7, 8, 9]),
        ("train[3:2]+train[:2]", [3, 4, 5, 6, 7, 8, 9, 0, 1]),
        ("train[70%:30%]", []),
        ("train[-2:-6]", []),
        ("train[7:3]+train[:0]", []),
        ("train[7:3]+train[:2]", [0, 1]),
        ("all", list(range(10))),
        ("train", list(range(10))),
    ],
)
def test_file_reader_honors_split_instruction(tmp_path, use_fast_loader, split, expected):
    inform, mixture = _inform_and_mixture(tmp_path, use_fast_loader, split)
    loaded = load_for_inform(inform, mixture)
    assert isinstance(loaded, Dataset)
    assert list(loaded) == [{"text": str(i)} for i in expected]


@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize("split", ["train[4:9]", "train[4:3]"])
def test_raw_limit_applies_after_split_selection(tmp_path, use_fast_loader, split):
    inform, mixture = _inform_and_mixture(tmp_path, use_fast_loader, split, num_rows=2)
    assert list(load_for_inform(inform, mixture)) == [{"text": "4"}, {"text": "5"}]


@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize("split", ["validation", "validation[:1]", "train+validation", "train[nope:]"])
def test_file_reader_rejects_unavailable_or_invalid_splits(tmp_path, use_fast_loader, split):
    inform, mixture = _inform_and_mixture(tmp_path, use_fast_loader, split)
    with pytest.raises(ValueError):
        load_for_inform(inform, mixture)


@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize("split", ["train[:0]", "train[10:]", "train[3:3]", "train[30%:30%]", "train[:0]+train[3:3]"])
def test_file_reader_rejects_empty_split_instructions(tmp_path, use_fast_loader, split):
    inform, mixture = _inform_and_mixture(tmp_path, use_fast_loader, split)
    with pytest.raises(ValueError):
        load_for_inform(inform, mixture)


@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize("split", ["validation", "train[:2]", "train[:20%]", "all"])
def test_streaming_file_reader_preserves_huggingface_split_validation(tmp_path, use_fast_loader, split):
    inform, mixture = _inform_and_mixture(tmp_path, use_fast_loader, split, streaming=True)
    with pytest.raises(ValueError):
        load_for_inform(inform, mixture)


@pytest.mark.parametrize("use_fast_loader", [False, True])
def test_streaming_train_split_remains_lazy_and_row_limited(tmp_path, use_fast_loader):
    inform, mixture = _inform_and_mixture(tmp_path, use_fast_loader, "train", streaming=True, num_rows=2)
    loaded = load_for_inform(inform, mixture)
    assert isinstance(loaded, IterableDataset)
    assert list(loaded) == [{"text": "0"}, {"text": "1"}]


@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize(
    ("split", "expected"),
    [("train[:15%]", list(range(16))), ("train[:15%](pct1_dropremainder)", list(range(15)))],
)
def test_file_reader_uses_huggingface_percentage_rounding(tmp_path, use_fast_loader, split, expected):
    path = tmp_path / "rounding.jsonl"
    path.write_text("".join(json.dumps({"text": str(i)}) + "\n" for i in range(107)))
    mixture = DatasetMixture(
        informs=[], use_fast_loader=use_fast_loader, streaming=False, cache_dir=str(tmp_path / "cache")
    )
    loaded = load_for_inform(TextDatasetInform(data_files=str(path), split=split), mixture)
    assert list(loaded) == [{"text": str(i)} for i in expected]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("split", ["train[2:5]", "train[2:1]", "train[5:2]", "validation"])
def test_nested_parquet_fallback_does_not_bypass_split_selection(tmp_path, monkeypatch, streaming, split):
    path = tmp_path / "rows.parquet"
    Dataset.from_list([{"text": str(i)} for i in range(7)]).to_parquet(str(path))

    def nested_features_unavailable(*args, **kwargs):
        raise ValueError("Feature type 'List' not found")

    monkeypatch.setattr("easydel.data.sources.base._load_hf_dataset", nested_features_unavailable)
    mixture = DatasetMixture(informs=[], streaming=streaming, cache_dir=str(tmp_path / "cache"))
    inform = TextDatasetInform(data_files=str(path), split=split, num_rows=2)
    if streaming or split == "validation":
        with pytest.raises(ValueError):
            load_for_inform(inform, mixture)
    else:
        expected = [] if split == "train[5:2]" else [{"text": "2"}, {"text": "3"}]
        assert list(load_for_inform(inform, mixture)) == expected


@pytest.mark.parametrize("use_fast_loader", [None, False, True])
def test_elarge_legacy_reader_selection_preserves_split_expression(tmp_path, use_fast_loader):
    from easydel.infra.elarge.builders import build_dataset

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"content": str(i)}) + "\n" for i in range(8)))
    mixture_config = {
        "informs": [{"data_files": str(path), "split": "train[3:7]", "num_rows": 2}],
        "streaming": False,
        "shuffle": False,
        "stop_strategy": "first_exhausted",
        "cache_dir": str(tmp_path / "cache"),
    }
    if use_fast_loader is not None:
        mixture_config["use_fast_loader"] = use_fast_loader
    cfg = {"model": {"name_or_path": "unused"}, "mixture": mixture_config}
    assert list(build_dataset(cfg)) == [{"text": "3"}, {"text": "4"}]
