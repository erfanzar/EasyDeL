"""CPU-only regression tests for weighted legacy map-style mixtures."""

import json
from collections import Counter

import pytest
from datasets import Dataset
from easydel.data import DatasetMixture, TextDatasetInform
from easydel.data.execution.pipeline import build_dataset

pytestmark = pytest.mark.cpu_ok


def _mixture(tmp_path, weights, *, sizes=(1000, 1000), **kwargs):
    informs = []
    for name, size in zip(("a", "b"), sizes, strict=True):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"text": f"{name}:{i}"}) + "\n" for i in range(size)))
        informs.append(TextDatasetInform(name=name, data_files=str(path), content_field="text"))
    return DatasetMixture(
        informs=informs,
        mixture_weights=weights,
        streaming=False,
        use_fast_loader=True,
        batch_size=1,
        shuffle_buffer_size=0,
        seed=42,
        cache_dir=str(tmp_path / "cache"),
        **kwargs,
    )


@pytest.mark.parametrize("block_mixture", [False, True])
@pytest.mark.parametrize("stop_strategy", ["first_exhausted", "all_exhausted", "restart"])
def test_materialized_weighted_sampling_is_seeded_and_honors_proportions(tmp_path, block_mixture, stop_strategy):
    mixture = _mixture(tmp_path, {"a": 3.0, "b": 1.0}, block_mixture=block_mixture, stop_strategy=stop_strategy)
    result = build_dataset(mixture)
    assert isinstance(result, Dataset)
    rows = list(result)
    assert rows == list(build_dataset(mixture))
    counts = Counter(row["text"].split(":")[0] for row in rows)
    assert 0.70 < counts["a"] / len(rows) < 0.80
    for name in ("a", "b"):
        indices = [int(row["text"].split(":")[1]) for row in rows if row["text"].startswith(name + ":")]
        if stop_strategy == "first_exhausted":
            assert indices == list(range(len(indices)))
        else:
            assert set(indices) == set(range(1000))
            assert indices == [i % 1000 for i in range(len(indices))]


@pytest.mark.parametrize("stop_strategy", ["first_exhausted", "all_exhausted", "restart"])
def test_zero_weight_source_is_excluded_before_exhaustion_tracking(tmp_path, stop_strategy):
    mixture = _mixture(tmp_path, {"a": 1.0, "b": 0.0}, sizes=(3, 1), stop_strategy=stop_strategy)
    assert list(build_dataset(mixture)) == [{"text": f"a:{i}"} for i in range(3)]


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"a": 1.0},
        {"a": 1.0, "typo": 1.0},
        {"a": -1.0, "b": 2.0},
        {"a": 0.0, "b": 0.0},
        {"a": float("nan"), "b": 1.0},
        {"a": float("inf"), "b": 1.0},
    ],
)
def test_materialized_weights_are_validated(tmp_path, weights):
    with pytest.raises(ValueError, match="mixture_weights"):
        build_dataset(_mixture(tmp_path, weights, sizes=(3, 3)))


@pytest.mark.parametrize("stop_strategy", ["first_exhausted", "all_exhausted", "restart"])
@pytest.mark.parametrize("all_empty", [False, True])
def test_weighted_sources_emptied_by_filtering_do_not_leak_other_source_rows(tmp_path, stop_strategy, all_empty):
    mixture = _mixture(tmp_path, {"a": 1.0, "b": 1.0}, sizes=(3, 3), stop_strategy=stop_strategy)
    mixture.informs[0].max_samples = 0
    if all_empty:
        mixture.informs[1].max_samples = 0
    result = build_dataset(mixture)
    assert isinstance(result, Dataset)
    expected = [] if all_empty or stop_strategy == "first_exhausted" else [{"text": f"b:{i}"} for i in range(3)]
    assert list(result) == expected


@pytest.mark.parametrize("use_fast_loader", [False, True])
@pytest.mark.parametrize("empty_reason", ["max_samples", "filter_fn"])
@pytest.mark.parametrize("empty_index", [0, 1])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("stop_strategy", ["first_exhausted", "all_exhausted", "restart"])
def test_empty_sources_do_not_remove_canonical_text_or_metadata(
    tmp_path, use_fast_loader, empty_reason, empty_index, weighted, stop_strategy
):
    informs = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps({"content": f"{name}:{i}", "metadata": f"m{name}{i}"}) + "\n" for i in range(2))
        )
        informs.append(
            TextDatasetInform(name=name, data_files=str(path), content_field="content", additional_fields=["metadata"])
        )
    if empty_reason == "max_samples":
        informs[empty_index].max_samples = 0
    else:
        informs[empty_index].filter_fn = lambda row: False
    mixture = DatasetMixture(
        informs=informs,
        mixture_weights={"a": 1.0, "b": 1.0} if weighted else None,
        streaming=False,
        use_fast_loader=use_fast_loader,
        batch_size=1,
        shuffle_buffer_size=0,
        seed=42,
        cache_dir=str(tmp_path / "cache"),
        stop_strategy=stop_strategy,
    )
    result = build_dataset(mixture)
    assert isinstance(result, Dataset)
    assert set(result.column_names) == {"text", "metadata"}
    surviving_name = "b" if empty_index == 0 else "a"
    expected = [{"text": f"{surviving_name}:{i}", "metadata": f"m{surviving_name}{i}"} for i in range(2)]
    if weighted and stop_strategy == "first_exhausted":
        expected = []
    assert list(result) == expected


def test_omitting_weights_preserves_legacy_concatenation(tmp_path):
    mixture = _mixture(tmp_path, None, sizes=(2, 3))
    assert list(build_dataset(mixture)) == [
        {"text": "a:0"},
        {"text": "a:1"},
        {"text": "b:0"},
        {"text": "b:1"},
        {"text": "b:2"},
    ]


@pytest.mark.parametrize("weight_source", ["inform", "mixture"])
def test_elarge_materialized_mixture_uses_weights(tmp_path, weight_source):
    from easydel.infra.elarge.builders import build_dataset as build_elarge_dataset

    mixture = _mixture(tmp_path, {"a": 1.0, "b": 0.0}, sizes=(2, 2))
    informs = [
        {"name": inform.name, "data_files": inform.data_files, "content_field": "text"} for inform in mixture.informs
    ]
    config = {
        "model": {"name_or_path": "unused"},
        "mixture": {
            "informs": informs,
            "streaming": False,
            "shuffle": False,
            "cache_dir": str(tmp_path / "cache"),
            "stop_strategy": "first_exhausted",
        },
    }
    if weight_source == "inform":
        informs[0]["weight"] = 1.0
        informs[1]["weight"] = 0.0
    else:
        config["mixture"]["mixture_weights"] = mixture.mixture_weights
    assert list(build_elarge_dataset(config)) == [{"text": "a:0"}, {"text": "a:1"}]
