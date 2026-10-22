"""Ray wiring tests use an in-process scheduler; never start a real Ray/TPU runtime."""

import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest
from easydel.data import (
    DatasetConfig,
    ObservabilityConfig,
    Pipeline,
    PipelineConfig,
    RayConfig,
    ShardedDataSource,
    TokenizeStageConfig,
)
from easydel.data.distributed import ray_utils
from easydel.data.transforms.tokenize import TokenizerManager

pytestmark = pytest.mark.cpu_ok


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        ids = [[ord(char) for char in text][: kwargs["max_length"]] for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


class FakeRay:
    def __init__(self):
        self.started = False
        self.actors = []
        self.peak_actors = 0
        self.init_options = {}
        self.actor_options = []

    def is_initialized(self):
        return self.started

    def init(self, **kwargs):
        self.started = True
        self.init_options = kwargs

    def remote(self, cls):
        runtime = self

        class ActorFactory:
            def __init__(self):
                self.settings = {}

            def options(self, **kwargs):
                self.settings = kwargs
                return self

            def remote(self, *args, **kwargs):
                actor = cls(*args, **kwargs)
                runtime.actors.append(actor)
                runtime.actor_options.append(self.settings)
                runtime.peak_actors = max(runtime.peak_actors, len(runtime.actors))
                return SimpleNamespace(
                    actor=actor,
                    process=SimpleNamespace(remote=lambda *a, **k: lambda: actor.process(*a, **k)),
                    next_chunk=SimpleNamespace(remote=lambda: actor.next_chunk),
                )

        return ActorFactory()

    def get(self, future):
        return future()

    def kill(self, handle):
        handle.actor.close()
        self.actors.remove(handle.actor)

    def cancel(self, future):
        pass


def test_distributed_module_does_not_eagerly_import_ray(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "ray" or name.startswith("ray."):
            raise AssertionError("Ray imported before a distributed operation was requested")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded)
        importlib.reload(ray_utils)


def test_ray_config_runs_normal_tokenization_and_releases_workers(tmp_path, monkeypatch):
    runtime = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", runtime)
    monkeypatch.setattr(TokenizerManager, "get_tokenizer", lambda *args, **kwargs: FakeTokenizer())
    paths = []
    for index in range(5):
        path = tmp_path / f"data-{index}.jsonl"
        path.write_text(f'{{"body":"abc{index}","label":{index}}}\n')
        paths.append(str(path))
    config = PipelineConfig(
        datasets=[
            DatasetConfig(
                paths,
                name="sample",
                tokenizer="fake",
                content_field="body",
                additional_fields=["label"],
                cache_enabled=False,
            )
        ],
        tokenize=TokenizeStageConfig(max_length=3, num_workers=1, cache_tokenized=False),
        ray=RayConfig(
            enabled=True,
            num_workers=2,
            object_store_memory=1024,
            resources_per_worker={"CPU": 2, "memory": 128, "custom": 1},
        ),
        observability=ObservabilityConfig(progress_enabled=False),
    )
    pipeline = Pipeline.from_config(config).source().tokenize()
    assert not runtime.started
    source = pipeline.build()
    rows = list(source.open_shard(source.shard_names[0]))
    assert [row["label"] for row in rows] == list(range(5))
    assert [row["input_ids"] for row in rows] == [[97, 98, 99]] * 5
    assert runtime.actors == []
    assert runtime.peak_actors == 2
    assert runtime.init_options == {"object_store_memory": 1024}
    assert all(
        options == {"num_cpus": 2, "memory": 128, "resources": {"custom": 1}} for options in runtime.actor_options
    )
    assert list(source.open_shard_at_row(source.shard_names[0], 3)) == rows[3:]
    assert runtime.actors == []
    iterator = source.open_shard(source.shard_names[0])
    assert next(iterator) == rows[0]
    iterator.close()
    assert runtime.actors == []


def test_ray_pipeline_propagates_worker_errors_and_releases_actors(tmp_path, monkeypatch):
    runtime = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", runtime)

    class FailedTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise ValueError("tokenization failed")

    monkeypatch.setattr(TokenizerManager, "get_tokenizer", lambda *args, **kwargs: FailedTokenizer())
    path = tmp_path / "rows.jsonl"
    path.write_text('{"text":"abc"}\n')
    source = (
        Pipeline.from_config(
            PipelineConfig(
                datasets=[DatasetConfig(str(path), tokenizer="fake")],
                tokenize=TokenizeStageConfig(num_workers=1, cache_tokenized=False),
                ray=RayConfig(enabled=True, num_workers=2),
                observability=ObservabilityConfig(progress_enabled=False),
            )
        )
        .source()
        .tokenize()
        .build()
    )
    with pytest.raises(ValueError, match="tokenization failed"):
        list(source.open_shard(source.shard_names[0]))
    assert runtime.actors == []


def test_ray_delivers_first_chunk_without_exhausting_unbounded_shard(monkeypatch):
    runtime = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", runtime)
    monkeypatch.setattr(TokenizerManager, "get_tokenizer", lambda *args, **kwargs: FakeTokenizer())
    state = {"consumed": 0, "closed": False}

    class EndlessSource(ShardedDataSource):
        @property
        def shard_names(self):
            return ["whole-hub-split"]

        def num_shards(self):
            return 1

        def open_shard(self, shard_name):
            try:
                while True:
                    if state["consumed"] >= 10:
                        raise AssertionError("Worker exhausted the upstream shard before yielding")
                    index = state["consumed"]
                    state["consumed"] += 1
                    yield {"text": "abc", "id": index}
            finally:
                state["closed"] = True

    config = PipelineConfig(
        datasets=[DatasetConfig("unused", name="stream", tokenizer="fake", additional_fields=["id"])],
        tokenize=TokenizeStageConfig(batch_size=2, num_workers=1, cache_tokenized=False),
        ray=RayConfig(enabled=True, num_workers=2),
    )
    source = ray_utils.RayTokenizedShardedSource(EndlessSource(), config, config.tokenize, "stream")
    iterator = source.open_shard(source.shard_names[0])
    try:
        assert next(iterator)["id"] == 0
        assert state["consumed"] == 2
        assert next(iterator)["id"] == 1
        assert state["consumed"] == 2  # No further consumption until the chunk is drained.
        assert next(iterator)["id"] == 2
        assert state["consumed"] == 4
    finally:
        iterator.close()
    assert state["closed"]
    assert runtime.actors == []


def test_ray_chunks_preserve_multi_shard_order_and_resume(monkeypatch):
    runtime = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", runtime)
    monkeypatch.setattr(TokenizerManager, "get_tokenizer", lambda *args, **kwargs: FakeTokenizer())
    rows = [{"text": "ab", "id": i} for i in range(13)]
    config = PipelineConfig(
        datasets=[DatasetConfig("unused", name="rows", tokenizer="fake", additional_fields=["id"])],
        tokenize=TokenizeStageConfig(batch_size=2, num_workers=1, cache_tokenized=False),
        ray=RayConfig(enabled=True, num_workers=2),
    )

    class MultiShardSource(ShardedDataSource):
        @property
        def shard_names(self):
            return ["0", "1", "2"]

        def num_shards(self):
            return 3

        def open_shard(self, shard_name):
            start = int(shard_name) * 5
            yield from rows[start : start + 5]

    source = ray_utils.RayTokenizedShardedSource(MultiShardSource(), config, config.tokenize, "rows")
    results = list(source.open_shard(source.shard_names[0]))
    assert [row["id"] for row in results] == list(range(13))
    assert all(row["input_ids"] == [97, 98] for row in results)
    assert list(source.open_shard_at_row(source.shard_names[0], 5)) == results[5:]
    assert runtime.actors == []
