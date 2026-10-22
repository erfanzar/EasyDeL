"""Host-only coverage of SFT/speculative preprocessing integration."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers.speculative_decoding_trainer import SpeculativeDecodingTrainer
from easydel.trainers.supervised_fine_tuning_trainer import SFTTrainer
from easydel.trainers.trainer import Trainer

pytestmark = pytest.mark.cpu_ok

TRAINERS = [SFTTrainer, SpeculativeDecodingTrainer]


@pytest.fixture
def preprocess(monkeypatch):
    infos = {"source": "base-preprocessor"}
    monkeypatch.setattr(
        Trainer,
        "_preprocess_batch_input",
        lambda self, *, state, batch, is_train: (batch, infos),
    )

    def run(trainer_cls, batch):
        trainer = object.__new__(trainer_cls)
        result, result_infos = trainer._preprocess_batch_input(state=None, batch=batch, is_train=True)
        assert result is batch
        assert result_infos is infos
        return result

    return run


@pytest.mark.parametrize("trainer_cls", TRAINERS)
@pytest.mark.parametrize("empty_completion_mask", [False, True])
def test_caller_normalizes_assistant_masks_without_changing_empty_mask_policy(
    preprocess, trainer_cls, empty_completion_mask
):
    input_ids = np.asarray([[10, 11, 12, 0]], dtype=np.int64)
    attention = np.asarray([[1, 1, 1, 0]], dtype=np.int16)
    assistant = np.asarray([[0, 1, 1, 1]], dtype=np.int32)
    batch = {"input_ids": input_ids, "attention_mask": attention, "assistant_masks": assistant}
    if empty_completion_mask:
        batch["completion_mask"] = np.zeros_like(assistant)
    result = preprocess(trainer_cls, batch)
    keep_empty = empty_completion_mask and trainer_cls is SpeculativeDecodingTrainer
    assert result["completion_mask"].tolist() == ([[0, 0, 0, 0]] if keep_empty else [[0, 1, 1, 0]])
    assert result["labels"].tolist() == ([[-100] * 4] if keep_empty else [[-100, 11, 12, -100]])
    assert result["completion_mask"].dtype == np.int16
    assert result["labels"].dtype == np.int32
    assert "assistant_masks" not in result
    assert input_ids.tolist() == [[10, 11, 12, 0]]
    assert assistant.tolist() == [[0, 1, 1, 1]]
    assert attention.tolist() == [[1, 1, 1, 0]]


@pytest.mark.parametrize("trainer_cls", TRAINERS)
def test_caller_preserves_explicit_completion_mask_and_existing_labels(preprocess, trainer_cls):
    labels = np.asarray([[-100, 21, 22, -100]])
    result = preprocess(
        trainer_cls,
        {
            "input_ids": [[20, 21, 22, 0]],
            "labels": labels,
            "attention_mask": [[1, 1, 1, 0]],
            "completion_mask": [[0, 1, 0, 1]],
            "assistant_masks": [[1, 0, 1, 0]],
        },
    )
    assert result["completion_mask"].tolist() == [[0, 1, 0, 0]]
    assert result["labels"] is labels
    assert labels.tolist() == [[-100, 21, 22, -100]]
    assert "assistant_masks" not in result


@pytest.mark.parametrize("trainer_cls", TRAINERS)
@pytest.mark.parametrize("has_ignored_label", [False, True])
def test_caller_recovers_mask_only_from_ignored_labels(preprocess, trainer_cls, has_ignored_label):
    labels = np.asarray([[-100 if has_ignored_label else 20, 21, 22, 0]], dtype=np.int32)
    result = preprocess(trainer_cls, {"labels": labels, "attention_mask": [[1, 1, 1, 0]]})
    if has_ignored_label:
        assert result["completion_mask"].tolist() == [[0, 1, 1, 0]]
    else:
        assert "completion_mask" not in result
    assert result["labels"] is labels


class _TextTokenizer:
    eos_token = "!"
    truncation_side = "right"

    def __call__(self, text, *, truncation=False, max_length=None, padding=False, **kwargs):
        ids = [ord(char) for char in text]
        if truncation:
            ids = ids[:max_length]
        attention = [1] * len(ids)
        if padding == "max_length":
            padding_length = max_length - len(ids)
            ids += [0] * padding_length
            attention += [0] * padding_length
        return {"input_ids": ids, "attention_mask": attention}


@pytest.mark.parametrize("trainer_cls", TRAINERS)
@pytest.mark.parametrize(
    "assistant_only, completion_only, expected_mask",
    [(False, None, False), (True, None, True), (True, False, False), (False, True, True)],
)
def test_caller_prompt_mask_resolution_controls_emitted_tokens(
    trainer_cls, assistant_only, completion_only, expected_mask
):
    trainer = object.__new__(trainer_cls)
    trainer.arguments = SimpleNamespace(
        assistant_only_loss=assistant_only,
        completion_only_loss=completion_only,
        max_length=6,
        dataset_text_field="text",
        dataset_kwargs=None,
        packing=False,
        padding_free=False,
        sequence_packing=False,
        truncation_mode="keep_start",
        pad_to_multiple_of=None,
    )
    trainer.processing_class = _TextTokenizer()
    trainer._train_source = None
    trainer._dataset_text_field = "text"
    trainer._formatting_func = None
    trainer._user_data_collator = False
    transform = trainer._get_preprocess_transform()
    result = transform({"prompt": "ab", "completion": "c"})
    assert list(result["input_ids"]) == [97, 98, 99, 33, 0, 0]
    assert list(result["attention_mask"]) == [1, 1, 1, 1, 0, 0]
    if expected_mask:
        assert list(result["completion_mask"]) == [0, 0, 1, 1, 0, 0]
    else:
        assert "completion_mask" not in result
