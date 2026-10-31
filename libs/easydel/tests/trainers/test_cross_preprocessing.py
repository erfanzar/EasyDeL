"""Host-only preprocessing regressions for configured process workers."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset, IterableDataset
from easydel.data.sources.hf_wrapper import HFDatasetShardedSource
from easydel.data.transforms.base import ExpandTransform, Transform
from easydel.trainers.prompt_transforms import (
    CPOPreprocessTransform,
    DPOPreprocessTransform,
    EmbeddingPreprocessTransform,
    RewardPreprocessTransform,
    SFTPreprocessTransform,
)
from easydel.trainers.trainer.trainer import Trainer

pytestmark = pytest.mark.cpu_ok


class TextTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    eos_token = ""
    chat_template = None

    def __call__(self, text, *, max_length=None, padding=False, truncation=False, **kwargs):
        if isinstance(text, list):
            rows = [self(item, max_length=max_length, padding=padding, truncation=truncation, **kwargs) for item in text]
            return {key: [row[key] for row in rows] for key in rows[0]}
        ids = [ord(char) for char in text]
        if truncation and max_length:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length" and max_length:
            mask += [0] * max(0, max_length - len(ids))
            ids += [0] * max(0, max_length - len(ids))
        return {"input_ids": ids, "attention_mask": mask}


class ProcessTransform(Transform):
    def __call__(self, example):
        return {"value": example["value"] + 1, "worker_pid": os.getpid()}


class ExpandRows(ExpandTransform):
    def __call__(self, example):
        if example["value"] < 32:
            return
        yield {"value": example["value"] * 2}
        yield {"value": example["value"] * 2 + 1}


def make_trainer(workers, train=None, evaluation=None):
    trainer = object.__new__(Trainer)
    trainer.arguments = SimpleNamespace(dataset_num_proc=workers, sequence_packing=False)
    trainer._train_source = train
    trainer._eval_source = evaluation
    return trainer


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("kind", ["sft", "dpo", "cpo", "embedding", "reward"])
def test_real_text_transforms_match_lazy_outputs_and_resume(workers, kind):
    tokenizer = TextTokenizer()
    if kind == "sft":
        transform = SFTPreprocessTransform(tokenizer, max_length=8, padding=False, add_eos=False)
        rows = [{"text": value} for value in ("abc", "def", "ghi", "jkl")]
    elif kind == "embedding":
        transform = EmbeddingPreprocessTransform(tokenizer, max_length=8)
        rows = [{"query": value, "positive": value.upper(), "metadata": "drop me"} for value in ("ab", "cd", "ef", "gh")]
    else:
        rows = [{"prompt": "Q", "chosen": value, "rejected": "N"} for value in ("ab", "cd", "ef", "gh")]
        if kind == "reward":
            transform = RewardPreprocessTransform(tokenizer, max_length=8)
        else:
            cls = DPOPreprocessTransform if kind == "dpo" else CPOPreprocessTransform
            transform = cls(tokenizer, max_prompt_length=2, max_completion_length=4)
    source = HFDatasetShardedSource(Dataset.from_list(rows), name="test")
    expected = [
        {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
        for row in source.transform(transform).iter_shards()
    ]
    mapped = make_trainer(workers)._preprocess_source(source, transform)
    assert isinstance(mapped, HFDatasetShardedSource)
    assert mapped.shard_names == ["test:0"]
    assert list(mapped.iter_shards()) == expected
    assert list(mapped.open_shard_at_row("test:0", 2)) == expected[2:]
    if kind == "embedding":
        assert "metadata" not in expected[0]
        assert expected[0]["query_input_ids"] == [97, 98, 0, 0, 0, 0, 0, 0]


def test_two_workers_really_execute_in_processes_once_and_keep_order():
    source = HFDatasetShardedSource(Dataset.from_dict({"value": list(range(80))}))
    mapped = make_trainer(2)._preprocess_source(source, ProcessTransform())
    rows = list(mapped.iter_shards())
    assert [row["value"] for row in rows] == list(range(1, 81))
    assert len({row["worker_pid"] for row in rows}) == 2
    assert all(row["worker_pid"] != os.getpid() for row in rows)
    assert list(mapped.iter_shards()) == rows


@pytest.mark.parametrize("workers", [1, 2])
def test_filtered_empty_batches_and_expansion_preserve_delivered_offsets(workers):
    source = HFDatasetShardedSource(Dataset.from_dict({"value": list(range(80))}))
    mapped = make_trainer(workers)._preprocess_source(source, ExpandRows())
    expected = [{"value": value} for value in range(64, 160)]
    assert list(mapped.iter_shards()) == expected
    assert list(mapped.open_shard_at_row(mapped.shard_names[0], 7)) == expected[7:]


@pytest.mark.parametrize("workers", [None, 2])
def test_streaming_stays_lazy(workers):
    source = HFDatasetShardedSource(IterableDataset.from_generator(lambda: ({"value": value} for value in range(4))))
    mapped = make_trainer(workers)._preprocess_source(source, ProcessTransform())
    assert [row["value"] for row in mapped.iter_shards()] == [1, 2, 3, 4]
    assert [row["worker_pid"] for row in mapped.iter_shards()] == [os.getpid()] * 4


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_invalid_worker_counts_rejected_even_without_datasets(workers):
    with pytest.raises(ValueError, match="dataset_num_proc"):
        make_trainer(workers)._apply_preprocess_transforms()


@pytest.mark.parametrize("raw_train", [False, True])
def test_train_and_eval_schema_are_resolved_independently(raw_train):
    raw = HFDatasetShardedSource(Dataset.from_dict({"text": ["abc", "def"]}))
    tokenized = HFDatasetShardedSource(Dataset.from_dict({"input_ids": [[7], [8]]}))
    trainer = make_trainer(2, raw if raw_train else tokenized, tokenized if raw_train else raw)
    transform = SFTPreprocessTransform(TextTokenizer(), max_length=8, padding=False, add_eos=False)
    trainer._get_preprocess_transform = lambda: None if trainer._is_pretokenized() else transform
    trainer._apply_preprocess_transforms()
    expected_raw = [
        {"input_ids": [97, 98, 99], "attention_mask": [1, 1, 1]},
        {"input_ids": [100, 101, 102], "attention_mask": [1, 1, 1]},
    ]
    actual_raw = trainer._train_source if raw_train else trainer._eval_source
    actual_tokenized = trainer._eval_source if raw_train else trainer._train_source
    assert list(actual_raw.iter_shards()) == expected_raw
    assert list(actual_tokenized.iter_shards()) == [{"input_ids": [7]}, {"input_ids": [8]}]
