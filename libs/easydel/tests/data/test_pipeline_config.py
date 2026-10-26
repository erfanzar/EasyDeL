"""Pure-Python configuration, source, cache, and pipeline state regressions."""

from collections import UserDict
from threading import Event

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from easydel.data import (
    CacheStageConfig,
    DatasetConfig,
    HFDatasetShardedSource,
    LoadStageConfig,
    ObservabilityConfig,
    Pipeline,
    PipelineConfig,
    PipelineContext,
    ShardedDataSource,
    SourceStageConfig,
    TokenizerConfig,
    TreeCacheManager,
    create_data_iterator,
    create_source,
)
from easydel.data.sources.base import ParquetShardedSource, RetryingShardedSource

pytestmark = pytest.mark.cpu_ok


def quiet():
    return ObservabilityConfig(progress_enabled=False)


def test_dict_config_normalizes_all_stages_and_nested_values():
    pipeline = Pipeline.from_config(
        {
            "datasets": [{"data_files": "unused", "name": "demo", "tokenizer": {"name_or_path": "fake"}}],
            "source": {"streaming": False},
            "cache": {"cache_type": "memory"},
            "mix": {"weight_schedule": [{"step": 0, "weights": {"demo": 1.0}}]},
            "ray": {"enabled": True, "num_workers": 2},
            "observability": {"metrics_enabled": False},
        }
    )
    config = pipeline.get_context().config
    assert config.validate() == []
    assert isinstance(config.source, SourceStageConfig)
    assert isinstance(config.cache, CacheStageConfig)
    assert isinstance(config.datasets[0].tokenizer, TokenizerConfig)
    assert config.ray.enabled and config.ray.num_workers == 2
    assert not config.observability.metrics_enabled
    assert config.mix.weight_schedule[0].weights == {"demo": 1.0}
    assert config.load == LoadStageConfig()
    defaults = Pipeline.from_config({"datasets": [{"data_files": "unused"}], "mix": None}).get_context().config
    assert defaults.validate() == []
    assert defaults.source == SourceStageConfig()
    assert defaults.cache == CacheStageConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"datasets": [DatasetConfig("unused", name="duplicate"), DatasetConfig("unused", name="duplicate")]},
        {"datasets": [DatasetConfig("unused")], "mix": {"weights": {"unknown": 1.0}}},
    ],
)
def test_pipeline_rejects_invalid_cross_field_config(kwargs):
    with pytest.raises(ValueError, match="Invalid pipeline configuration"):
        Pipeline.from_config(kwargs)


@pytest.mark.parametrize("global_streaming,stage_streaming", [(False, True), (True, False)])
def test_streaming_false_materializes_before_downstream_stages(tmp_path, global_streaming, stage_streaming):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text":"original"}\n')
    pipeline = Pipeline.from_config(
        PipelineConfig(
            datasets=[DatasetConfig(str(path))],
            streaming=global_streaming,
            source=SourceStageConfig(streaming=stage_streaming),
            observability=quiet(),
        )
    ).source()
    path.write_text('{"text":"changed"}\n')
    source = pipeline.build()
    assert list(source.open_shard(source.shard_names[0])) == [{"text": "original"}]
    assert pipeline.get_context().get_metrics()["consume"]["dataset_0_rows"] == 1


def test_hub_source_honors_storage_options_streaming_and_retry_policy(monkeypatch):
    def load_dataset(_name, **kwargs):
        return [{"streaming": kwargs["streaming"], "credential": kwargs["storage_options"]["credential"]}]

    monkeypatch.setattr("easydel.data.sources.base._load_hf_dataset", load_dataset)
    source = create_source(
        DatasetConfig("org/dataset", type="hf"),
        SourceStageConfig(streaming=False, cloud_max_retries=0, dask_storage_options={"credential": "test"}),
    )
    assert list(source.open_shard(source.shard_names[0])) == [{"streaming": False, "credential": "test"}]
    assert len(source) == 1


def test_parquet_guard_is_per_source_and_available_in_dataset_config(tmp_path):
    path = tmp_path / "data.parquet"
    rows = [{"text": "payload" * 10}, {"text": "different" * 10}]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    guarded = create_source(DatasetConfig(str(path), max_row_group_bytes=1))
    assert list(guarded.open_shard(guarded.shard_names[0])) == rows
    assert list(guarded.open_shard_at_row(guarded.shard_names[0], 1)) == rows[1:]
    assert guarded.delivers_all_metadata_rows
    unguarded = ParquetShardedSource(str(path), max_row_group_bytes=0)
    assert list(unguarded.open_shard(unguarded.shard_names[0])) == rows
    assert list(unguarded.open_shard_at_row(unguarded.shard_names[0], 1)) == rows[1:]
    assert unguarded.delivers_all_metadata_rows


class FlakySource(ShardedDataSource):
    """A source which loses its first connection after two delivered rows."""

    def __init__(self):
        self.failed = False

    @property
    def shard_names(self):
        return ["one"]

    def num_shards(self):
        return 1

    def open_shard(self, shard_name):
        yield from self.open_shard_at_row(shard_name, 0)

    def open_shard_at_row(self, shard_name, row):
        for index in range(row, 5):
            if index == 2 and not self.failed:
                self.failed = True
                raise OSError("connection lost")
            yield {"id": index}


def test_retry_policy_resumes_without_duplicates_and_zero_retries_is_respected():
    source = RetryingShardedSource(FlakySource(), max_retries=1, retry_delay=0)
    assert list(source.open_shard("one")) == [{"id": index} for index in range(5)]
    iterator = RetryingShardedSource(FlakySource(), max_retries=0, retry_delay=0).open_shard("one")
    assert next(iterator) == {"id": 0}
    assert next(iterator) == {"id": 1}
    with pytest.raises(OSError, match="connection lost"):
        next(iterator)


@pytest.mark.parametrize("cache_type", ["memory", "disk", "hierarchical"])
def test_context_cache_honors_layer_options_and_dataset_overrides(tmp_path, cache_type):
    context = PipelineContext(
        PipelineConfig(
            datasets=[
                DatasetConfig("unused", cache_path=str(tmp_path / "override")),
                DatasetConfig("disabled", cache_enabled=False),
            ],
            cache=CacheStageConfig(
                cache_dir=str(tmp_path / "global"),
                cache_type=cache_type,
                memory_cache_size=1,
                compression="gzip",
                disk_cache_expiry=0,
            ),
        )
    )
    manager = context.get_cache_manager("dataset_0")
    manager.put("one", {"payload": 1})
    manager.put("two", {"payload": 2})
    assert manager.stats["memory"]["size"] <= 1
    assert context.get_cache_manager("dataset_1") is None
    if cache_type == "memory":
        assert manager.get("one") is None
        assert manager.get("two")[0] == {"payload": 2}
        assert not (tmp_path / "override").exists()
    else:
        assert manager.stats["disk"]["compression"] == "gzip"
        assert manager.stats["disk"]["cache_dir"] == str(tmp_path / "override")
        fresh = TreeCacheManager(tmp_path / "override", disk_expiry=0, cache_type="disk", compression="gzip")
        assert fresh.get("two") is None
    assert not (tmp_path / "global").exists()


def test_metrics_switch_and_loader_step_epoch_state(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(f'{{"id":{index}}}' for index in range(3)))
    pipeline = (
        Pipeline.from_config(
            PipelineConfig(
                datasets=[DatasetConfig(str(path))],
                load=LoadStageConfig(
                    batch_size=2,
                    drop_last=False,
                    prefetch_enabled=False,
                    collate_fn=lambda rows: {"ids": [row["id"] for row in rows]},
                ),
                observability=quiet(),
            )
        )
        .source()
        .load()
    )
    assert list(pipeline.build()) == [{"ids": [0, 1]}, {"ids": [2]}]
    context = pipeline.get_context()
    assert (context.step, context.epoch) == (2, 1)
    assert context.get_metrics()["load"]["dataset_0_rows"] == 2
    disabled = PipelineContext(
        PipelineConfig(
            datasets=[DatasetConfig("unused")],
            observability=ObservabilityConfig(metrics_enabled=False, progress_enabled=False),
        )
    )
    assert list(disabled.observe_rows(iter([1, 2]), "test", "data")) == [1, 2]
    disabled.record_metric("manual", "key", 1)
    assert disabled.get_metrics() == {}


def test_hf_source_public_metadata_and_mapping_normalization():
    source = HFDatasetShardedSource([UserDict(id=1), UserDict(id=2)])
    assert source.estimated_length == 2
    assert not source.is_streaming
    assert list(source.open_shard(source.shard_names[0])) == [{"id": 1}, {"id": 2}]
    assert list(source.open_shard_at_row(source.shard_names[0], 1)) == [{"id": 2}]
    streaming = HFDatasetShardedSource(iter([UserDict(id=3)]))
    assert streaming.is_streaming
    assert streaming.estimated_length is None
    assert list(streaming.open_shard(streaming.shard_names[0])) == [{"id": 3}]


def test_create_data_iterator_preserves_batching_and_propagates_errors():
    assert list(create_data_iterator(range(5), 2, shuffle=False, prefetch=True)) == [[0, 1], [2, 3], [4]]

    def failed():
        yield {"id": 1}
        raise ValueError("upstream error")

    iterator = create_data_iterator(failed(), 1, shuffle=False)
    assert next(iterator) == {"id": 1}
    with pytest.raises(ValueError, match="upstream error"):
        next(iterator)
    iterator.close()


@pytest.mark.parametrize("abandon", [False, True])
def test_create_data_iterator_closes_abandoned_full_queue_worker(abandon):
    filled = Event()
    released = Event()

    def rows():
        try:
            for index in range(10000):
                if index >= 2:
                    filled.set()
                yield index
        finally:
            released.set()

    iterator = create_data_iterator(rows(), 1, shuffle=False, prefetch_buffer=1)
    assert next(iterator) == 0
    assert filled.wait(2)
    if abandon:
        import gc

        del iterator
        gc.collect()
    else:
        iterator.close()
    assert released.wait(2)
