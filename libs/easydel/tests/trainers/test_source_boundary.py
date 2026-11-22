"""CPU-only source-boundary regressions for trainer preprocessing."""

from types import SimpleNamespace

import pytest
from datasets import Dataset, IterableDataset
from easydel.data.core.protocols import ShardedDataSource
from easydel.data.sources.hf_wrapper import HFDatasetShardedSource
from easydel.trainers.odds_ratio_preference_optimization_trainer import ORPOTrainer
from easydel.trainers.prm_trainer import PRMTrainer
from easydel.trainers.prompt_transforms import KTOPreprocessTransform, preprocess_dataset_with_num_proc
from easydel.trainers.proximal_policy_optimization_trainer import PPOTrainer

from .test_preprocessing import TextTokenizer

pytestmark = pytest.mark.cpu_ok


class CustomSource(ShardedDataSource):
    def __init__(self, rows):
        self.rows = rows
        self.opens = 0

    @property
    def shard_names(self):
        return ["custom"]

    def num_shards(self):
        return 1

    def open_shard(self, shard_name):
        self.opens += 1
        yield from self.rows


def make_source(rows, kind):
    if kind == "custom":
        return CustomSource(rows)
    if kind == "streaming":
        return HFDatasetShardedSource(IterableDataset.from_generator(lambda: iter(rows)))
    return HFDatasetShardedSource(Dataset.from_list(rows))


@pytest.mark.parametrize("kind", ["wrapped_hf", "streaming", "custom"])
def test_num_proc_helper_accepts_sources_and_preserves_lazy_inputs(kind):
    source = make_source([{"prompt": "p", "completion": "a", "label": True}], kind)
    result = preprocess_dataset_with_num_proc(source, KTOPreprocessTransform(TextTokenizer()), 2)
    if kind == "custom":
        assert source.opens == 0
    assert next(iter(result.iter_shards()))["completion_labels"] == [-100, 97, 2]
    if kind == "wrapped_hf":
        assert isinstance(result, HFDatasetShardedSource)
        assert not result.is_streaming


@pytest.mark.parametrize("kind", ["wrapped_hf", "streaming", "custom"])
def test_prm_num_proc_preserves_independent_train_eval_policies(kind):
    trainer = object.__new__(PRMTrainer)
    trainer.processing_class = TextTokenizer()
    trainer.arguments = SimpleNamespace(
        dataset_num_proc=2,
        step_separator="\n",
        max_length=8,
        max_completion_length=None,
        train_on_last_step_only=True,
        pad_to_multiple_of=None,
        sequence_packing=False,
    )
    rows = [{"prompt": "p", "completions": ["a", "b"], "labels": [0, 1]}] * 2
    trainer._train_source = make_source(rows, kind)
    trainer._eval_source = make_source(rows, kind)
    trainer._apply_preprocess_transforms()
    train = list(trainer._train_source.iter_shards())
    evaluation = list(trainer._eval_source.iter_shards())
    assert len(train) == len(evaluation) == 2
    assert [label for label in train[0]["labels"] if label != -100] == [1]
    assert [label for label in evaluation[0]["labels"] if label != -100] == [0, 1]
    if kind == "wrapped_hf":
        assert isinstance(trainer._train_source, HFDatasetShardedSource)
        assert isinstance(trainer._eval_source, HFDatasetShardedSource)


@pytest.mark.parametrize("kind", ["wrapped_hf", "streaming", "custom"])
@pytest.mark.parametrize("trainer_class", [ORPOTrainer, PPOTrainer])
def test_preference_trainers_use_shared_num_proc_source_boundary(kind, trainer_class):
    trainer = object.__new__(trainer_class)
    trainer.processing_class = TextTokenizer()
    trainer.arguments = SimpleNamespace(
        dataset_num_proc=2,
        max_prompt_length=8,
        max_completion_length=8,
        label_pad_token_id=-100,
        skip_apply_chat_template=True,
        chat_template_kwargs=None,
        sequence_packing=False,
    )
    rows = [{"prompt": "p", "chosen": "a", "rejected": "b"}] * 2
    trainer._train_source = make_source(rows, kind)
    trainer._eval_source = make_source(rows, kind)
    trainer._apply_preprocess_transforms()
    train = list(trainer._train_source.iter_shards())
    evaluation = list(trainer._eval_source.iter_shards())
    assert train == evaluation
    assert len(train) == 2
    if trainer_class is ORPOTrainer:
        assert train[0]["prompt_input_ids"] == [0] * 7 + [112]
        assert train[0]["prompt_attention_mask"] == [0] * 7 + [1]
        assert train[0]["chosen_labels"] == [-100, 97, 2] + [-100] * 13
        assert train[0]["rejected_labels"] == [-100, 98, 2] + [-100] * 13
    else:
        assert train[0]["input_ids"] == [0] * 7 + [112]
        assert train[0]["attention_mask"] == [0] * 7 + [1]
    if kind == "wrapped_hf":
        assert isinstance(trainer._train_source, HFDatasetShardedSource)
        assert isinstance(trainer._eval_source, HFDatasetShardedSource)
