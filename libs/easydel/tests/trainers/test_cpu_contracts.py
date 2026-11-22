"""Pure-Python regression coverage for trainer contracts (no JAX numerics)."""

import json
from types import SimpleNamespace

import pytest
from datasets import Dataset, IterableDataset
from easydel.data.core.protocols import ShardedDataSource
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from easydel.trainers.grpo_replay_buffer_trainer import GRPOWithReplayBufferConfig
from easydel.trainers.gspo_token_trainer import GSPOTokenConfig
from easydel.trainers.kto_trainer.kto_trainer import KTOTrainer
from easydel.trainers.merge_callback import MergeConfig, MergeModelCallback, register_merge_method
from easydel.trainers.minillm_trainer import MiniLLMConfig
from easydel.trainers.nemo_gym_trainer import NeMoGymTrainer, load_nemo_gym_jsonl
from easydel.trainers.training_utils import get_scheduled_loss_adapter

pytestmark = pytest.mark.cpu_ok


class RowsSource(ShardedDataSource):
    def __init__(self, rows):
        self.rows = rows

    @property
    def shard_names(self):
        return ["rows"]

    def num_shards(self):
        return 1

    def open_shard(self, shard_name):
        yield from self.rows


@pytest.mark.parametrize("kind", ["dataset", "iterable", "source"])
def test_kto_preprocesses_paired_rows_and_preserves_metadata(kind):
    rows = [{"prompt": "Q", "chosen": "yes", "rejected": "no", "task": "math"}]
    if kind == "dataset":
        dataset = Dataset.from_list(rows)
    elif kind == "iterable":
        dataset = IterableDataset.from_generator(lambda: iter(rows))
    else:
        dataset = RowsSource(rows)
    result = KTOTrainer._preprocess_kto_dataset(
        dataset, None, SimpleNamespace(skip_apply_chat_template=True, dataset_num_proc=None)
    )
    actual = list(result.open_shard("rows")) if kind == "source" else list(result)
    assert [(row["prompt"], row["completion"], row["label"]) for row in actual] == [
        ("Q", "yes", True),
        ("Q", "no", False),
    ]
    if kind != "dataset":
        assert [row["task"] for row in actual] == ["math", "math"]
    if kind == "source":
        assert result.num_shards() == 1
        assert list(result.open_shard_at_row("rows", 1)) == actual[1:]


def test_kto_pretokenized_stream_passes_through():
    rows = [{"prompt_input_ids": [1], "completion_input_ids": [1, 2], "label": True}]
    source = KTOTrainer._preprocess_kto_dataset(RowsSource(rows), None, SimpleNamespace())
    assert list(source.open_shard("rows")) == rows


@pytest.mark.parametrize("signature", ["keyword", "positional", "empty"])
def test_nemo_factory_signature_variants(signature):
    metadata = {"task": "demo"}
    if signature == "keyword":

        def factory(*, metadata, agent_ref, request_timeout):
            return metadata, agent_ref, request_timeout

        expected = (metadata, "agent", 3)
    elif signature == "positional":

        def factory(context, /):
            return context

        expected = metadata
    else:

        def factory():
            return "env"

        expected = "env"
    trainer = object.__new__(NeMoGymTrainer)
    trainer.arguments = SimpleNamespace(request_timeout=3)
    trainer.environment_factory = factory
    assert trainer._make_nemo_environment(metadata, "agent") == expected


def test_nemo_factory_body_typeerror_propagates_once():
    calls = []

    def factory(metadata=None):
        calls.append(metadata)
        raise TypeError("factory body failed")

    trainer = object.__new__(NeMoGymTrainer)
    trainer.arguments = SimpleNamespace(request_timeout=3)
    trainer.environment_factory = factory
    with pytest.raises(TypeError, match="factory body failed"):
        trainer._make_nemo_environment({"task": "demo"}, None)
    assert calls == [{"task": "demo"}]


def test_nemo_jsonl_loads_local_tasks(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('\n{"prompt":"hello","agent_ref":"worker"}\n{"responses_create_params":{"input":"next"}}\n')
    rows = list(load_nemo_gym_jsonl(path))
    assert [row["prompt"] for row in rows] == ["hello", "next"]
    assert rows[0]["agent_ref"] == "worker"
    assert json.loads(rows[1]["metadata"])["responses_create_params"]["input"] == "next"


def test_custom_merge_method_is_configurable():
    register_merge_method("review_trainers4_custom", lambda policy, target, context: policy)
    config = MergeConfig(method="review_trainers4_custom")
    assert config.create()["method"] == "review_trainers4_custom"
    with pytest.raises(ValueError):
        MergeConfig(method="review_trainers4_unknown")


@pytest.mark.parametrize("every_checkpoint", [False, True])
def test_merge_hooks_publish_selected_event(every_checkpoint):
    published = []
    callback = MergeModelCallback(
        merge_at_every_checkpoint=every_checkpoint, push_to_hub=True, publish_fn=published.append
    )
    # Non-array leaves exercise callback lifecycle without any array computation.
    policy = {"name": "policy"}
    target = {"name": "target"}
    callback.on_save(policy_state=policy, target_state=target)
    assert len(published) == int(every_checkpoint)
    callback.on_train_end(policy_state=policy, target_state=target)
    assert published == [policy]
    assert callback.last_merged_state == policy


@pytest.mark.parametrize("config_cls", [GRPOWithReplayBufferConfig, MiniLLMConfig, GSPOTokenConfig])
def test_derived_configs_retain_hash_contract(config_cls):
    # Hashing empty config instances is pure Python and needs no model/devices.
    config = object.__new__(config_cls)
    assert isinstance(hash(config), int)
    assert hash(config) == hash(config)


def test_scheduled_dppo_cache_keys_include_divergence_and_cap():
    adapter = get_scheduled_loss_adapter(grpo_step)

    def key(divergence, cap):
        values = {"dppo_divergence_type": divergence, "dppo_clip_ratio_c": cap}
        return adapter.make_cache_key(
            SimpleNamespace(get=values.get, state=SimpleNamespace(graphdef=None, graphother=None))
        )

    assert key("binary_tv", 20.0) != key("binary_kl", 20.0)
    assert key("binary_tv", 20.0) != key("binary_tv", 5.0)
    assert key("binary_tv", 20.0) == key("binary_tv", 20.0)


def test_completion_parquet_logging_uses_remote_path_backend(monkeypatch):
    import pandas as pd
    from easydel.trainers.group_relative_policy_optimization import grpo_trainer as module

    stored = {}

    class MemoryPath:
        def __init__(self, path):
            self.path = str(path)

        def __str__(self):
            return self.path

        def __truediv__(self, name):
            return MemoryPath(self.path + "/" + name)

        def mkdir(self, **kwargs):
            pass

        def write_bytes(self, content):
            stored[self.path] = content

    monkeypatch.setattr(module, "ePath", MemoryPath)
    monkeypatch.setattr(module.jax, "process_index", lambda: 0)
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = SimpleNamespace(
        log_completions=True,
        log_completions_hub_repo=None,
        _get_save_directory=lambda **kwargs: "gs://example/run",
    )
    result = trainer._write_completion_log_rows([{"prompt": "question", "completion": "answer"}], step=7)
    assert str(result) == "gs://example/run/completions/step-00000007-process-00000.parquet"
    import io

    frame = pd.read_parquet(io.BytesIO(stored[str(result)]))
    assert frame.to_dict("records") == [{"step": 7, "process_index": 0, "prompt": "question", "completion": "answer"}]


def test_nemo_jsonl_supports_gcs_backend(monkeypatch):
    from easydel.trainers.nemo_gym_trainer import _fn as module

    def path_factory(path):
        assert path == "gs://example/tasks.jsonl"
        return SimpleNamespace(read_text=lambda: '{"prompt":"remote"}\n')

    monkeypatch.setattr(module, "ePath", path_factory)
    assert next(iter(load_nemo_gym_jsonl("gs://example/tasks.jsonl")))["prompt"] == "remote"
