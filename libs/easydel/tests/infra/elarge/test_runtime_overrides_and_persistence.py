"""Array-free eLarge routing, state ownership, and dataset persistence regressions."""

import gzip
import importlib
import inspect
import json
from types import SimpleNamespace

import datasets
import pytest
from easydel.inference.esurge.config import eSurgeRuntimeConfig
from easydel.infra.elarge import builders, processing
from easydel.infra.elarge.model import eLargeModel
from easydel.infra.elarge.types import get_trainer_defaults, normalize_trainer_config
from spectrax.common_types import NOT_GIVEN

pytestmark = pytest.mark.cpu_ok


def config(**sections):
    return {"model": {"name_or_path": "unused-local-model", "task": "causal_lm"}, **sections}


def prepared_elm():
    elm = eLargeModel(config())
    elm._model = object()
    elm._tokenizer = object()
    return elm


class Arguments:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


@pytest.mark.parametrize("catchall_arguments", [False, True])
def test_custom_runtime_extension_reaches_only_the_trainer(catchall_arguments):
    elm = prepared_elm()
    runtime = object()

    class CatchallArguments:
        def __init__(self, **kwargs):
            self.values = kwargs

    class CustomTrainer:
        def __init__(self, *, model, arguments, processing_class, train_dataset, custom_runtime):
            self.arguments = arguments
            self.custom_runtime = custom_runtime

    trainer = elm.build_trainer(
        train_dataset=[{"text": "a"}],
        trainer_class=CustomTrainer,
        args_class=CatchallArguments if catchall_arguments else Arguments,
        custom_runtime=runtime,
        learning_rate=0.25,
    )
    assert trainer.custom_runtime is runtime
    if catchall_arguments:
        assert trainer.arguments.values["learning_rate"] == 0.25
        assert "custom_runtime" not in trainer.arguments.values
    else:
        assert trainer.arguments.learning_rate == 0.25


def test_training_argument_override_binds_real_sft_constructor_once():
    from easydel.trainers.supervised_fine_tuning_trainer import SFTTrainer

    signature = inspect.signature(SFTTrainer)

    def boundary(**kwargs):
        signature.bind(**kwargs)
        return SimpleNamespace(**kwargs)

    boundary.__signature__ = signature
    trainer = prepared_elm().build_trainer(
        train_dataset=[{"text": "a"}], trainer_class=boundary, args_class=Arguments, learning_rate=0.125
    )
    assert trainer.arguments.learning_rate == 0.125
    assert not hasattr(trainer, "learning_rate")


def test_unknown_override_is_rejected_instead_of_silently_dropped():
    class CustomTrainer:
        def __init__(self, *, model, arguments, processing_class, train_dataset):
            pass

    with pytest.raises(TypeError, match="lerning_rate"):
        prepared_elm().build_trainer(
            train_dataset=[], trainer_class=CustomTrainer, args_class=Arguments, lerning_rate=0.1
        )


def test_training_caches_authoritative_final_state_for_the_next_stage(monkeypatch):
    elm = prepared_elm()
    restored = SimpleNamespace(model=object(), step=10, optimizer="restored")
    final = SimpleNamespace(model=object(), step=20, optimizer="trained")

    class StateLoader:
        @staticmethod
        def load_state(**kwargs):
            return restored

    assert elm.build_state(state_class=StateLoader) is restored
    monkeypatch.setattr(elm, "validate", lambda: None)

    class Trainer:
        def __init__(self, *, model, arguments, processing_class, train_dataset):
            self.input_state = model
            # Deliberately differ from final.state.model: the output state is
            # authoritative even when a plugin retains a stale model attribute.
            self.model = object()

        def train(self):
            return SimpleNamespace(state=final)

    result = elm.train(train_dataset=[], trainer_class=Trainer, args_class=Arguments)
    assert result.state is final
    assert elm.build_model() is final.model
    assert elm.build_state() is final
    next_trainer = elm.build_trainer(train_dataset=[], trainer_class=Trainer, args_class=Arguments)
    assert next_trainer.input_state is final
    assert next_trainer.input_state.step == 20
    assert next_trainer.input_state.optimizer == "trained"


def test_force_model_rebuild_invalidates_previously_restored_state(monkeypatch):
    model_module = importlib.import_module("easydel.infra.elarge.model")
    elm = prepared_elm()
    elm._state = SimpleNamespace(model=elm._model, step=10)
    replacement = object()
    monkeypatch.setattr(model_module, "build_model", lambda cfg: replacement)
    assert elm.build_model(force_rebuild=True) is replacement
    assert elm._state is None


@pytest.mark.parametrize("trainer_type", ["sft", "speculative_decoding"])
def test_injected_defaults_are_owned_by_each_configuration(trainer_type):
    first = normalize_trainer_config({"trainer_type": trainer_type})
    fresh = normalize_trainer_config({"trainer_type": trainer_type})
    first["benchmarks"].append({"tasks": ["changed"]})
    if "metrics_to_show_in_rich_pbar" in first:
        first["metrics_to_show_in_rich_pbar"].append("changed")
    assert normalize_trainer_config({"trainer_type": trainer_type}) == fresh
    direct = get_trainer_defaults(trainer_type)
    direct["benchmarks"].append({"tasks": ["direct-change"]})
    assert get_trainer_defaults(trainer_type)["benchmarks"] == fresh["benchmarks"]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("dialect", ["pretty-json", "compact-json", "jsonl", "tsv"])
def test_public_sharded_source_resolves_row_dialects(tmp_path, explicit, compressed, dialect):
    rows = [{"content": "a", "number": "1"}, {"content": "b", "number": "2"}]
    if dialect == "tsv":
        extension, content = "tsv", "content\tnumber\na\t1\nb\t2\n"
    elif dialect == "jsonl":
        extension, content = "jsonl", "\n".join(json.dumps(row) for row in rows)
    else:
        extension = "json"
        content = json.dumps(rows, indent=2 if dialect == "pretty-json" else None)
    path = tmp_path / (f"rows.{extension}" + (".gz" if compressed else ""))
    path.write_bytes(gzip.compress(content.encode()) if compressed else content.encode())
    inform = {"data_files": str(path), "content_field": "content"}
    if explicit:
        inform["type"] = extension
    source = builders.build_sharded_source(config(mixture={"informs": [inform], "shuffle": False}))
    actual = [row for shard in source.shard_names for row in source.open_shard(shard)]
    assert [row["text"] for row in actual] == ["a", "b"]


class TextTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": [ord(char) for char in texts], "attention_mask": [1] * len(texts)}
        return {
            "input_ids": [[ord(char) for char in text] for text in texts],
            "attention_mask": [[1] * len(text) for text in texts],
        }


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_scalar_and_batched_tokenization_have_rank_one_rows(streaming, batched):
    rows = [{"text": "ab"}, {"text": "c"}]
    source = (
        datasets.IterableDataset.from_generator(lambda: iter(rows)) if streaming else datasets.Dataset.from_list(rows)
    )
    actual = list(builders.tokenize_dataset(source, TextTokenizer(), batched=batched))
    assert [{key: row[key] for key in ("tokens", "attention_mask")} for row in actual] == [
        {"tokens": [97, 98], "attention_mask": [1, 1]},
        {"tokens": [99], "attention_mask": [1]},
    ]


@pytest.mark.parametrize("builder", [False, True])
def test_explicit_arrow_shard_count_round_trips(tmp_path, monkeypatch, builder):
    output = str(tmp_path / "output")
    if builder:
        from transformers import AutoTokenizer

        path = tmp_path / "input.jsonl"
        path.write_text('{"content": "ab"}\n{"content": "c"}\n')
        monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TextTokenizer())
        original, saved = builders.build_tokenized_dataset(
            config(
                mixture={
                    "informs": [{"type": "jsonl", "data_files": str(path)}],
                    "shuffle": False,
                    "save": {"output_path": output, "format": "arrow", "num_shards": 2},
                }
            )
        )
        assert saved == output
    else:
        original = datasets.Dataset.from_list([{"tokens": [97, 98]}, {"tokens": [99]}])
        assert builders.save_dataset(original, output, format="arrow", num_shards=2) == output
    restored = datasets.load_from_disk(output)
    assert list(restored) == list(original)
    assert len(list((tmp_path / "output").glob("*.arrow"))) == 2


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le"])
def test_remote_config_writes_publish_complete_encoded_objects_without_local_io(monkeypatch, encoding):
    objects = {}

    class RemotePath:
        def __init__(self, path):
            self.path = path

        def __str__(self):
            return self.path

        def write_bytes(self, data):
            objects[self.path] = data

    monkeypatch.setattr(processing, "ePath", RemotePath)
    monkeypatch.setattr(processing.tempfile, "mkstemp", lambda **kwargs: pytest.fail("remote path sent to mkstemp"))
    monkeypatch.setattr(processing.os, "replace", lambda *args: pytest.fail("remote path sent to os.replace"))
    processing.write_text_atomic("gs://test-bucket/config.yaml", "name: café\n", encoding=encoding)
    assert objects == {"gs://test-bucket/config.yaml": "name: café\n".encode(encoding)}


def test_local_atomic_config_write_replaces_complete_file(tmp_path):
    path = tmp_path / "nested" / "config.yaml"
    processing.write_text_atomic(path, "old")
    processing.write_text_atomic(path, "new: café\n")
    assert path.read_text() == "new: café\n"
    assert list(path.parent.iterdir()) == [path]


def test_materialized_runtime_mapping_retains_auto_sizing_sentinel_and_options():
    runtime = eSurgeRuntimeConfig.from_dict(spec_debug_max_traces=7, spec_adaptive_conf_threshold=0.75)
    raw = {"runtime": dict(runtime)}
    normalized = builders.normalize_esurge_sections(raw)
    assert normalized["runtime"]["max_num_batched_tokens"] is NOT_GIVEN
    elm = eLargeModel(config(esurge=raw))
    kwargs = builders.to_esurge_kwargs(elm.to_dict())
    assert kwargs["runtime"].max_num_batched_tokens is NOT_GIVEN
    assert kwargs["runtime"].spec_debug_max_traces == 7
    assert kwargs["runtime"].spec_adaptive_conf_threshold == 0.75
