"""Non-numeric dataset columns reach online-RL reward functions, one row per completion (host-only)."""

import numpy as np
import pytest
from easydel.trainers.base_trainer import BaseTrainer

pytestmark = pytest.mark.cpu_ok


def _examples():
    return [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "answer": "72", "tools": None},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "answer": "10", "tools": [{"name": "calc"}]},
    ]


def test_string_columns_survive_and_numeric_columns_do_not():
    side = BaseTrainer._extract_reward_batch_sidechannels(_examples())
    assert side == {"answer": ["72", "10"], "tools": [None, [{"name": "calc"}]]}


def test_dict_batch_keeps_string_arrays_and_drops_numeric_arrays():
    batch = {"input_ids": np.ones((2, 3), np.int32), "answer": np.asarray(["72", "10"], dtype=object)}
    assert BaseTrainer._extract_reward_batch_sidechannels(batch) == {"answer": ["72", "10"]}


def test_sidechannels_repeat_in_prompt_major_completion_order():
    side = BaseTrainer._extract_reward_batch_sidechannels(_examples())
    purified = {"input_ids": np.zeros((2, 3), np.int32)}
    aligned = BaseTrainer._completion_aligned_reward_batch(
        {**purified, **side}, side, prompt_count=2, completion_count=6
    )
    assert aligned["answer"] == ["72", "72", "72", "10", "10", "10"]
    assert len(aligned["tools"]) == 6
    assert aligned["input_ids"].shape == (2, 3)


def test_shipped_accuracy_reward_scores_against_aligned_answers():
    def accuracy(completions, *, batch):
        return [float(c == gold) for c, gold in zip(completions, batch["answer"], strict=True)]

    side = BaseTrainer._extract_reward_batch_sidechannels(_examples())
    batch = BaseTrainer._completion_aligned_reward_batch(side, side, prompt_count=2, completion_count=4)
    assert accuracy(["72", "1", "10", "10"], batch=batch) == [1.0, 0.0, 1.0, 1.0]
