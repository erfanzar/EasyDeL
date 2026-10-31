"""Documented public helpers remain available even without internal callers."""

import json

import pytest
from easydel.data import (
    AsyncDataLoader,
    DatasetConfig,
    HFDatasetShardedSource,
    LoadStage,
    ObservabilityConfig,
    Pipeline,
    PipelineConfig,
    ResumeState,
    ShardingSpec,
    create_data_iterator,
)
from easydel.data.core.protocols import AsyncDatasetProtocol, PipelineStage

pytestmark = pytest.mark.cpu_ok


def test_resume_state_roundtrips_json_and_partial_old_checkpoints():
    state = ResumeState(shard_index=3, row_index=12, epoch=2, step=100, dataset_states={"code": {"row": 8}})
    restored = ResumeState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored == state
    assert ResumeState.from_dict({"row_index": 4}).to_dict() == {
        "shard_index": 0,
        "row_index": 4,
        "epoch": 0,
        "step": 0,
        "dataset_states": {},
    }


def test_sharding_disabled_leaves_batch_unchanged_and_protocols_are_structural():
    batch = {"ids": [[1, 2], [3, 4]]}
    assert ShardingSpec().apply(batch) is batch
    assert isinstance(LoadStage(), PipelineStage)
    loader = AsyncDataLoader(
        HFDatasetShardedSource([{"id": 1}]),
        batch_size=1,
        prefetch_enabled=False,
        collate_fn=lambda rows: {"ids": [row["id"] for row in rows]},
    )
    assert isinstance(loader, AsyncDatasetProtocol)
    assert list(loader) == [{"ids": [1]}]


def test_pipeline_multisource_build_keeps_all_sources_accessible(tmp_path):
    paths = [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]
    for index, path in enumerate(paths):
        path.write_text(json.dumps({"id": index}) + "\n")
    pipeline = Pipeline.from_config(
        PipelineConfig(
            datasets=[DatasetConfig(str(path), name=f"source_{index}") for index, path in enumerate(paths)],
            observability=ObservabilityConfig(progress_enabled=False),
        )
    ).source()
    first = pipeline.build()
    assert list(first.open_shard(first.shard_names[0])) == [{"id": 0}]
    assert set(pipeline.get_data()) == {"source_0", "source_1"}
    second = pipeline.get_data()["source_1"]
    assert list(second.open_shard(second.shard_names[0])) == [{"id": 1}]


def test_finished_prefetch_iterator_remains_exhausted():
    iterator = create_data_iterator([1, 2], batch_size=1, shuffle=False)
    assert list(iterator) == [1, 2]
    with pytest.raises(StopIteration):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
