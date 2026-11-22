# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-dataset assistant-only supervision reaches the SFT / distillation / speculative loss masks.

The assembled ``completion_mask`` itself is asserted for hand-built two-dataset batches (one dataset opts
in, the other out), on unpacked (per-row carrier) and packed (per-token carrier) batches. The parent
preprocess is stubbed to a passthrough so only the mask assembly runs; no model, no tokenizer download.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from easydel.trainers.assistant_only_mask import (
    AssistantSpanIds,
    assistant_mask_from_token_ids,
    derive_assistant_span_ids,
)

pytestmark = pytest.mark.cpu_ok

IM_START, ROLE, NL, END = 90, 91, 92, 99
SPAN = AssistantSpanIds(header_ids=(IM_START, ROLE, NL), end_id=END, newline_family=(NL,))
# 0 filler | 1..4 user turn | 5..7 assistant header | 8..10 content | 11 terminator | 12..15 filler
DOC = [50, IM_START, 51, 52, END, IM_START, ROLE, NL, 53, 54, 55, END, 56, 57, 58, 59]
ASSISTANT = np.zeros(len(DOC), np.int32)
ASSISTANT[[8, 9, 10, 11]] = 1
FULL = np.ones(len(DOC), np.int32)


def _trainer(monkeypatch, trainer_path: str, **arguments):
    import importlib

    from easydel.trainers.trainer import Trainer

    module_name, class_name = trainer_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    monkeypatch.setattr(Trainer, "_preprocess_batch_input", lambda self, state, batch, is_train: (batch, {}))
    tr = object.__new__(cls)
    tr.arguments = types.SimpleNamespace(assistant_only_loss=False, **arguments)
    tr._cached_assistant_span_ids = SPAN
    return tr


TRAINERS = [
    "easydel.trainers.supervised_fine_tuning_trainer.sft_trainer.SFTTrainer",
    "easydel.trainers.distillation_trainer.distillation_trainer.DistillationTrainer",
    "easydel.trainers.speculative_decoding_trainer.spec_trainer.SpeculativeDecodingTrainer",
]


def _unpacked(flags):
    return {
        "input_ids": np.asarray([DOC] * len(flags), np.int32),
        "attention_mask": np.ones((len(flags), len(DOC)), np.int32),
        "_assistant_only": np.asarray(flags, np.int32),
        "_dataset_id": np.arange(len(flags), dtype=np.int32),
    }


def _packed(flags):
    n = len(flags)
    return {
        "input_ids": np.asarray([DOC * n], np.int32),
        "attention_mask": np.ones((1, n * len(DOC)), np.int32),
        "segment_ids": np.asarray([np.repeat(np.arange(n), len(DOC))], np.int32),
        "assistant_only_mask": np.asarray([np.repeat(flags, len(DOC))], np.int32),
        "dataset_ids": np.asarray([np.repeat(np.arange(n), len(DOC))], np.int32),
    }


@pytest.mark.parametrize("trainer_path", TRAINERS)
def test_unpacked_mixture_masks_only_the_opted_in_dataset(monkeypatch, trainer_path):
    tr = _trainer(monkeypatch, trainer_path)
    batch, infos = tr._preprocess_batch_input(state=None, batch=_unpacked([1, 0]), is_train=True)
    cm = np.asarray(batch["completion_mask"])
    np.testing.assert_array_equal(cm[0], ASSISTANT)
    np.testing.assert_array_equal(cm[1], FULL)
    labels = np.asarray(batch["labels"])
    assert (labels[0][ASSISTANT == 0] == -100).all() and (labels[1] != -100).all()
    for carrier in ("_assistant_only", "_dataset_id", "assistant_only_mask", "dataset_ids"):
        assert carrier not in batch
    assert infos["data/documents"] == 2.0 and infos["data/zero_supervision_docs"] == 0.0


@pytest.mark.parametrize("trainer_path", TRAINERS)
def test_packed_mixture_masks_per_document(monkeypatch, trainer_path):
    tr = _trainer(monkeypatch, trainer_path)
    batch, _ = tr._preprocess_batch_input(state=None, batch=_packed([1, 0]), is_train=True)
    cm = np.asarray(batch["completion_mask"])[0]
    np.testing.assert_array_equal(cm[: len(DOC)], ASSISTANT)
    np.testing.assert_array_equal(cm[len(DOC) :], FULL)
    assert "assistant_only_mask" not in batch and "dataset_ids" not in batch


def test_global_flag_on_respects_a_dataset_opting_out(monkeypatch):
    tr = _trainer(monkeypatch, TRAINERS[0])
    tr.arguments.assistant_only_loss = True
    batch, _ = tr._preprocess_batch_input(state=None, batch=_unpacked([-1, 0]), is_train=True)
    cm = np.asarray(batch["completion_mask"])
    np.testing.assert_array_equal(cm[0], ASSISTANT)
    np.testing.assert_array_equal(cm[1], FULL)


def test_explicit_completion_mask_beats_the_span_scan(monkeypatch):
    tr = _trainer(monkeypatch, TRAINERS[0])
    batch = _unpacked([1, 1])
    explicit = np.zeros((2, len(DOC)), np.int32)
    explicit[:, -2:] = 1
    batch["completion_mask"] = explicit
    out, _ = tr._preprocess_batch_input(state=None, batch=batch, is_train=True)
    np.testing.assert_array_equal(np.asarray(out["completion_mask"]), explicit)


def test_segment_aware_labels_do_not_zero_window_neighbours(monkeypatch):
    tr = _trainer(monkeypatch, TRAINERS[0])
    batch = _packed([1, -1])
    labels = np.asarray(batch["input_ids"]).copy()
    labels[0, :4] = -100  # only doc 0 carries supervision-bearing labels
    batch["labels"] = labels
    out, _ = tr._preprocess_batch_input(state=None, batch=batch, is_train=True)
    cm = np.asarray(out["completion_mask"])[0]
    assert cm[:4].sum() == 0 and cm[4 : len(DOC)].sum() == len(DOC) - 4  # labels win over the scan for doc 0
    np.testing.assert_array_equal(cm[len(DOC) :], FULL)  # inheriting doc without -100 keeps full supervision


@pytest.mark.parametrize("trainer_path", TRAINERS)
def test_no_carriers_keeps_the_legacy_labels_rule(monkeypatch, trainer_path):
    tr = _trainer(monkeypatch, trainer_path)
    batch = _unpacked([1, 0])
    batch.pop("_assistant_only")
    batch.pop("_dataset_id")
    labels = np.asarray(batch["input_ids"]).copy()
    labels[0, :4] = -100
    batch["labels"] = labels
    out, infos = tr._preprocess_batch_input(state=None, batch=batch, is_train=True)
    cm = np.asarray(out["completion_mask"])
    assert cm[0, :4].sum() == 0 and cm[0, 4:].sum() == len(DOC) - 4
    assert cm[1].sum() == len(DOC)
    assert "data/documents" not in infos


def test_zero_supervision_abort_and_report_thresholds(monkeypatch):
    tr = _trainer(monkeypatch, TRAINERS[0], zero_supervision_abort_fraction=0.4)
    # A document with role structure but no complete header fails safe to zero supervision.
    broken = [50, IM_START, 51, END] * 4
    batch = {
        "input_ids": np.asarray([broken, broken], np.int32),
        "attention_mask": np.ones((2, 16), np.int32),
        "_assistant_only": np.asarray([1, 1], np.int32),
    }
    with pytest.raises(ValueError, match="zero supervised tokens"):
        tr._preprocess_batch_input(state=None, batch=dict(batch), is_train=True)
    tr.arguments.zero_supervision_abort_fraction = None
    out, infos = tr._preprocess_batch_input(state=None, batch=dict(batch), is_train=True)
    assert infos["data/zero_supervision_frac"] == 1.0
    assert np.asarray(out["completion_mask"]).sum() == 0


def test_incoherent_carriers_raise(monkeypatch):
    tr = _trainer(monkeypatch, TRAINERS[0])
    batch = _packed([1, 0])
    batch["_assistant_only"] = np.asarray([1], np.int32)
    with pytest.raises(ValueError, match="both per-row"):
        tr._preprocess_batch_input(state=None, batch=batch, is_train=True)


def test_non_consuming_trainer_drops_carriers_before_the_model():
    from easydel.trainers.trainer import Trainer

    tr = object.__new__(Trainer)
    tr.arguments = types.SimpleNamespace()
    batch, _ = tr._preprocess_batch_input(state=None, batch=_packed([1, 0]), is_train=True)
    assert "assistant_only_mask" not in batch and "dataset_ids" not in batch
    assert "input_ids" in batch


def test_scan_fails_safe_on_decapitated_segments_and_trains_plain_text():
    ids = np.asarray([[50, 51, 52, 53, IM_START, 51, END, 54]], np.int32)
    seg = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1]], np.int32)
    mask = assistant_mask_from_token_ids(ids, SPAN, segment_ids=seg)
    np.testing.assert_array_equal(mask[0], [1, 1, 1, 1, 0, 0, 0, 0])


def _chat_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "<u>", "<a>", "<e>", "hi", "there"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="<e>")
    tok.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'assistant' %}<a> {{ m['content'] }} <e> "
        "{% else %}<u> {{ m['content'] }} {% endif %}{% endfor %}{% if add_generation_prompt %}<a> {% endif %}"
    )
    return tok, words


def test_span_markers_derive_from_the_chat_template_or_explicit_text():
    tok, words = _chat_tokenizer()
    span = derive_assistant_span_ids(tok)
    assert span.header_ids == (words.index("<a>"),) and span.end_id == words.index("<e>")
    explicit = derive_assistant_span_ids(tok, header_text="<u>", end_text="<e>")
    assert explicit.header_ids == (words.index("<u>"),)
    ids = np.asarray([[2, 5, 3, 6, 6, 4, 2, 5]], np.int32)  # <u> hi <a> there there <e> <u> hi
    np.testing.assert_array_equal(assistant_mask_from_token_ids(ids, span)[0], [0, 0, 0, 1, 1, 1, 0, 0])


def test_span_derivation_without_template_or_markers_raises():
    tok, _ = _chat_tokenizer()
    tok.chat_template = None
    with pytest.raises(RuntimeError, match="assistant_header_text"):
        derive_assistant_span_ids(tok)
