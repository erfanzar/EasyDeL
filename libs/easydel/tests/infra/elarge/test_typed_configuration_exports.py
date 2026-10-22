"""CPU-only configuration and text-reader integration regressions."""

from dataclasses import dataclass
from threading import Lock

import pytest
from datasets import Dataset
from easydel.infra import factory
from easydel.infra.elarge import processing
from easydel.infra.elarge.builders import (
    build_dataset,
    build_sharded_source,
    normalize_esurge_sections,
    to_data_mixture_kwargs,
    to_esurge_kwargs,
)
from easydel.infra.elarge.model import eLargeModel
from easydel.layers.quantization import QuantizationConfig

pytestmark = pytest.mark.cpu_ok


def config(**sections):
    return {"model": {"name_or_path": "unused", "task": "causal_lm"}, **sections}


def source_rows(source):
    return [row for shard in source.shard_names for row in source.open_shard(shard)]


def test_typed_configuration_exports_are_independent_and_runtime_handles_shared():
    @dataclass
    class CustomConfig:
        values: list
        runtime: object

    lock = Lock()
    original = QuantizationConfig(dtype="nf4", group_size=32)
    elm = eLargeModel(config(quantization={"model": original, "kv_cache": original}, custom=CustomConfig([1], lock)))
    exported = elm.to_dict()
    exported["quantization"]["model"].group_size = 999
    exported["quantization"]["kv_cache"].group_size = 7
    exported["custom"].values.append(2)
    again = elm.to_dict()
    assert again["quantization"]["model"].group_size == 32
    assert again["quantization"]["kv_cache"].group_size == original.group_size == 32
    assert again["custom"].values == [1]
    assert again["custom"].runtime is lock


def test_elarge_task_entry_point_is_the_shared_implementation():
    assert processing.normalize_task is factory.normalize_task
    assert processing.TASK_ALIASES is factory.TASK_ALIASES


@pytest.mark.parametrize(
    "task, expected",
    [
        ("causal-language-model", factory.TaskType.CAUSAL_LM),
        ("lm", factory.TaskType.CAUSAL_LM),
        ("any_to_any", factory.TaskType.ANY_TO_ANY),
        ("seq2seq", factory.TaskType.SEQUENCE_TO_SEQUENCE),
        ("  IMAGE_TEXT_TO_TEXT  ", factory.TaskType.IMAGE_TEXT_TO_TEXT),
        ("unknown", None),
        (None, None),
    ],
)
def test_elarge_task_normalization_matches_public_values(task, expected):
    assert processing.normalize_task(task) == expected


def test_engine_options_follow_public_schema():
    cfg = config(
        esurge={
            "runtime": {"chunked_prefill_enabled": True, "include_finished_set": True, "token_safety_margin": 17},
            "distributed": {"distributed_admit_timeout_s": 23.0},
        }
    )
    normalized = normalize_esurge_sections(cfg["esurge"])
    assert normalized["runtime"]["token_safety_margin"] == 17
    kwargs = to_esurge_kwargs(cfg)
    assert kwargs["runtime"].chunked_prefill_enabled is True
    assert kwargs["runtime"].include_finished_set is True
    assert kwargs["distributed"].distributed_admit_timeout_s == 23.0


@pytest.mark.parametrize("use_fast_loader", [True, False])
def test_legacy_filter_then_cap_preserves_raw_reader_limit(tmp_path, use_fast_loader):
    path = tmp_path / "data.parquet"
    Dataset.from_list([{"content": str(i), "keep": i % 2 == 1} for i in range(8)]).to_parquet(str(path))

    def predicate(row):
        return row["keep"]

    cfg = config(
        mixture={
            "informs": [
                {
                    "type": "parquet",
                    "data_files": str(path),
                    "content_field": "content",
                    "filter_fn": predicate,
                    "max_samples": 2,
                    "num_rows": 5,
                }
            ],
            "use_fast_loader": use_fast_loader,
            "streaming": False,
            "shuffle": False,
        }
    )
    kwargs = to_data_mixture_kwargs(cfg)
    assert kwargs["use_fast_loader"] is use_fast_loader
    assert kwargs["informs"][0].filter_fn is predicate
    assert kwargs["informs"][0].num_rows == 5
    assert kwargs["informs"][0].max_samples == 2
    assert list(build_dataset(cfg)["text"]) == ["1", "3"]
    assert [row["text"] for row in source_rows(build_sharded_source(cfg))] == ["1", "3"]


@pytest.mark.parametrize("callback", ["filter_fn", "map_fn", "format_callback"])
def test_parquet_callbacks_can_read_non_content_columns(tmp_path, callback):
    path = tmp_path / "data.parquet"
    Dataset.from_list([{"content": "a", "keep": True}, {"content": "b", "keep": False}]).to_parquet(str(path))
    transform = (
        (lambda row: row["keep"])
        if callback == "filter_fn"
        else (lambda row: {**row, "content": row["content"] + str(row["keep"])})
    )
    cfg = config(
        mixture={
            "informs": [{"type": "parquet", "data_files": str(path), "content_field": "content", callback: transform}],
            "shuffle": False,
        }
    )
    actual = [row["text"] for row in source_rows(build_sharded_source(cfg))]
    assert actual == (["a"] if callback == "filter_fn" else ["aTrue", "bFalse"])


def test_slow_sharded_sources_keep_global_weights_out_of_single_reader(tmp_path):
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for label, path in zip(("a", "b"), paths, strict=True):
        path.write_text('{"content": "' + label + '"}\n')
    cfg = config(
        mixture={
            "informs": [{"type": "json", "data_files": str(path)} for path in paths],
            "use_fast_loader": False,
            "streaming": False,
            "shuffle": False,
            "mixture_weights": [0.5, 0.5],
            "stop_strategy": "all_exhausted",
            "mixture_block_size": 2,
        }
    )
    assert sorted(row["text"] for row in source_rows(build_sharded_source(cfg))) == ["a", "b"]


def test_quantization_mapping_and_typed_defaults_match():
    typed = QuantizationConfig(dtype="nf4")
    mapping = eLargeModel(config(quantization={"model": {"dtype": "nf4"}})).get_from_pretrained_kwargs()
    obj = eLargeModel(config(quantization={"model": typed})).get_from_pretrained_kwargs()
    assert mapping["quantization_config"].to_dict() == obj["quantization_config"].to_dict() == typed.to_dict()
    raw = eLargeModel(config(quantization={"method": "nf4"})).get_from_pretrained_kwargs()["quantization_config"]
    setter = eLargeModel(config()).set_quantization("nf4").get_from_pretrained_kwargs()["quantization_config"]
    assert raw.to_dict() == setter.to_dict()
    assert raw.group_size == 128
    assert raw.pattern == typed.pattern


def test_explicit_linear_overrides_reach_typed_configuration():
    original = QuantizationConfig(dtype="nf4", group_size=32)
    elm = eLargeModel(config(quantization={"model": original, "linear_pattern": "q_proj", "linear_group_size": 16}))
    actual = elm.get_from_pretrained_kwargs()["quantization_config"]
    assert actual.pattern == "q_proj"
    assert actual.group_size == 16
    assert original.group_size == 32
