"""Host-only BCO threshold, checkpoint, and multiprocessing preprocessing regressions."""

from types import SimpleNamespace

import pytest
from datasets import Dataset
from easydel.data.sources.hf_wrapper import HFDatasetShardedSource
from easydel.trainers.binary_classifier_optimization_trainer._fn import RunningMoments
from easydel.trainers.binary_classifier_optimization_trainer.bco_trainer import BCOTrainer
from easydel.trainers.prompt_transforms import BCOPreprocessTransform
from easydel.trainers.trainer.trainer import Trainer
from eformer.paths import ePath

pytestmark = pytest.mark.cpu_ok


class TextTokenizer:
    """Tiny text-only tokenizer suitable for pickling into Dataset.map workers."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        ids = [ord(char) for char in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_threshold_tracks_all_rewards_with_accumulation():
    trainer = object.__new__(BCOTrainer)
    trainer.arguments = SimpleNamespace(gradient_accumulation_steps=2)
    trainer.running = RunningMoments()
    state = object()
    # Rewards [1, 3, 5, 7] reduced by a two-microbatch mean.
    metrics = SimpleNamespace(
        other_metrics={"reward_moments/sum": 8.0, "reward_moments/sumsq": 42.0, "reward_moments/count": 2.0}
    )
    assert trainer.on_step_end(state, metrics, 1) == (state, metrics)
    assert trainer.running.as_dict() == pytest.approx({"mean": 4.0, "var": 5.0, "count": 4.0})
    # A smaller second batch must be weighted by its actual row count.
    metrics.other_metrics = {
        "reward_moments/sum": -2.0,
        "reward_moments/sumsq": 4.0,
        "reward_moments/count": 1.0,
    }
    trainer.on_step_end(state, metrics, 2)
    assert trainer.running.mean == pytest.approx(2.0)
    assert trainer.running.var == pytest.approx(34.0 / 3.0)
    assert trainer.running.count == pytest.approx(6.0)


def test_bco_threshold_roundtrips_with_policy_checkpoint(tmp_path, monkeypatch):
    def save_policy(self, state, save_directory=None, *args, **kwargs):
        ePath(save_directory).mkdir(parents=True, exist_ok=True)
        return str(save_directory)

    monkeypatch.setattr(Trainer, "_save_state", save_policy)
    trainer = object.__new__(BCOTrainer)
    trainer.running = RunningMoments()
    trainer.running.update_from_moments(3.0, 2.0, 7)
    directory = trainer._save_state(object(), str(tmp_path / "checkpoint"))
    restored = object.__new__(BCOTrainer)
    restored.running = RunningMoments()
    restored._load_auxiliary_states(directory)
    assert restored.running.as_dict() == trainer.running.as_dict()
    # Checkpoints without the threshold sidecar remain loadable.
    restored._load_auxiliary_states(str(tmp_path / "legacy"))
    assert restored.running.as_dict() == trainer.running.as_dict()


def test_parallel_preprocessing_unpairs_and_matches_lazy_transform():
    dataset = Dataset.from_dict(
        {"prompt": ["a", "b"], "chosen": ["yes", "good"], "rejected": ["no", "bad"], "sample_id": [7, 9]}
    )
    transform = BCOPreprocessTransform(tokenizer=TextTokenizer(), max_prompt_length=8, max_completion_length=8)
    expected = [row for example in dataset for row in transform(example)]
    processed = BCOTrainer._preprocess_dataset_parallel(dataset, transform, num_proc=2)
    assert list(processed) == expected
    assert [row["label"] for row in processed] == [True, False, True, False]
    assert processed[0]["prompt_input_ids"] == [ord("a")]
    assert processed[2]["prompt_input_ids"] == [ord("b")]
    assert BCOTrainer._preprocess_dataset_parallel(processed, transform, num_proc=2) is processed


@pytest.mark.parametrize("eval_kind", ["iterable", "sharded", "pretokenized"])
def test_eager_training_keeps_independent_lazy_eval_preprocessing(eval_kind):
    tokenizer = TextTokenizer()
    dataset = Dataset.from_dict({"prompt": ["a"], "chosen": ["yes"], "rejected": ["no"]})
    transform = BCOPreprocessTransform(tokenizer=tokenizer, max_prompt_length=8, max_completion_length=8)
    processed_train = BCOTrainer._preprocess_dataset_parallel(dataset, transform, num_proc=2)
    expected = list(processed_train)
    raw_eval = dataset.to_iterable_dataset()
    if eval_kind == "sharded":
        raw_eval = HFDatasetShardedSource(raw_eval)
    elif eval_kind == "pretokenized":
        raw_eval = processed_train.to_iterable_dataset()
    # Match constructor source setup: eager mapping leaves streaming/sharded
    # evaluation unchanged, then BaseTrainer wraps and transforms both sources.
    processed_eval = BCOTrainer._preprocess_dataset_parallel(raw_eval, transform, num_proc=2)
    assert processed_eval is raw_eval
    trainer = object.__new__(BCOTrainer)
    trainer.arguments = SimpleNamespace(
        max_prompt_length=8,
        max_completion_length=8,
        label_pad_token_id=-100,
        sequence_packing=False,
    )
    trainer.processing_class = tokenizer
    trainer.embedding_tokenizer = None
    trainer._train_source = trainer._to_sharded_source(processed_train)
    trainer._eval_source = trainer._to_sharded_source(processed_eval)
    trainer._apply_preprocess_transforms()

    def rows(source):
        return [row for shard in source.shard_names for row in source.open_shard(shard)]

    assert rows(trainer._train_source) == expected
    assert rows(trainer._eval_source) == expected
    assert [row["label"] for row in rows(trainer._eval_source)] == [True, False]
