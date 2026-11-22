"""Host-only regressions for source batch filtering and deterministic shuffling."""

from types import SimpleNamespace

import pytest
from datasets import Dataset, IterableDataset
from easydel.data.sources.hf_wrapper import HFDatasetShardedSource
from easydel.trainers.base_trainer import _ReiterableDataLoader
from easydel.trainers.trainer.trainer import Trainer

pytestmark = pytest.mark.cpu_ok


def make_trainer():
    trainer = object.__new__(Trainer)
    trainer.arguments = SimpleNamespace(ids_to_pop_from_dataset=["metadata", "missing"], shuffle_seed_train=17)
    return trainer


@pytest.mark.parametrize("container", [list, tuple])
def test_raw_examples_drop_columns_without_mutating_source(container):
    trainer = make_trainer()
    rows = [{"input_ids": [1], "metadata": "a"}, {"input_ids": [2], "metadata": "b"}]
    loader = [container(rows)]
    batch, iterator = trainer._get_next_batch(iter(loader), loader)
    assert batch == [{"input_ids": [1]}, {"input_ids": [2]}]
    assert rows[0]["metadata"] == "a"
    # Exhaustion restarts the loader and applies the same filtering.
    restarted, _ = trainer._get_next_batch(iterator, loader)
    assert restarted == batch


def test_collated_mapping_still_drops_columns():
    trainer = make_trainer()
    loader = [{"input_ids": [[1], [2]], "metadata": ["a", "b"]}]
    batch, _ = trainer._get_next_batch(iter(loader), loader)
    assert batch == {"input_ids": [[1], [2]]}


def test_shuffled_source_replays_and_changes_epoch_order():
    trainer = make_trainer()
    source = HFDatasetShardedSource(Dataset.from_dict({"id": list(range(40))}))

    def batches(**kwargs):
        return list(trainer._create_dataloader_from_source(source, batch_size=4, **kwargs))

    ordered = batches(shuffle=False)
    shuffled = batches(shuffle=True)
    assert shuffled != ordered
    assert shuffled == batches(shuffle=True)
    assert sorted(row["id"] for batch in shuffled for row in batch) == list(range(40))
    assert batches(shuffle=True, skip_batches=3) == shuffled[3:]
    epochs = batches(shuffle=True, num_epochs=2)
    assert epochs[:10] == shuffled
    assert epochs[10:] != shuffled
    assert sorted(row["id"] for batch in epochs[10:] for row in batch) == list(range(40))


@pytest.mark.parametrize("num_rows", [0, 2, 40, 42])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("drop_remainder", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_multi_epoch_resume_matches_uninterrupted_batches(num_rows, streaming, drop_remainder, shuffle):
    trainer = make_trainer()
    dataset = Dataset.from_dict({"id": list(range(num_rows))})
    if streaming:
        dataset = IterableDataset.from_generator(lambda: ({"id": value} for value in range(num_rows)))
    source = HFDatasetShardedSource(dataset)

    def factory(skip_batches=0):
        return trainer._create_dataloader_from_source(
            source,
            batch_size=4,
            num_epochs=3,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            skip_batches=skip_batches,
        )

    uninterrupted = list(factory())
    for skip_batches in [0, 3, 10, 11, 13, 22, 30, 35]:
        assert list(factory(skip_batches)) == uninterrupted[skip_batches:]


def test_fast_forward_seek_carries_offset_into_second_shuffled_epoch():
    trainer = make_trainer()
    source = HFDatasetShardedSource(Dataset.from_dict({"id": list(range(40))}))

    def factory(skip_batches=0):
        return trainer._create_dataloader_from_source(
            source, batch_size=4, shuffle=True, num_epochs=2, skip_batches=skip_batches
        )

    loader = _ReiterableDataLoader(factory, skip_factory=factory)
    resumed = trainer._fast_forward_batches(iter(loader), loader, 13)
    assert list(resumed) == list(loader)[13:]
