"""Hardware-only regression for the device-prefetch configuration switch."""

import jax
from easydel.data import (
    DatasetConfig,
    HFDatasetShardedSource,
    LoadStage,
    LoadStageConfig,
    ObservabilityConfig,
    PipelineConfig,
    PipelineContext,
)


def test_load_stage_prefetches_batches_to_device():
    source = HFDatasetShardedSource([{"ids": [1, 2]}, {"ids": [3, 4]}])
    config = PipelineConfig(
        datasets=[DatasetConfig("unused")],
        load=LoadStageConfig(batch_size=2, prefetch_to_device=True),
        observability=ObservabilityConfig(progress_enabled=False),
    )
    loader = LoadStage(config.load).process({"dataset_0": source}, PipelineContext(config))["dataset_0"]
    iterator = iter(loader)
    batch = next(iterator)
    assert isinstance(batch["ids"], jax.Array)
    assert batch["ids"].shape == (2, 2)
    assert jax.device_get(batch["ids"]).tolist() == [[1, 2], [3, 4]]
    iterator.close()
