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

import pytest

try:
    from easydel.trainers.prompt_transforms import DPOPreprocessTransform
    from easydel.trainers.utils import (
        DataCollatorForPreferenceGrain,
        DataCollatorForPreferenceTFDS,
        DPODataCollatorWithPaddingGrain,
        DPODataCollatorWithPaddingTFDS,
    )
except ImportError as exc:
    pytest.skip(f"Preference collators unavailable: {exc}", allow_module_level=True)


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) % 100 for char in text]}


def _full_sequence_feature():
    return {
        "prompt_input_ids": [11, 12],
        "prompt_attention_mask": [1, 1],
        "chosen_input_ids": [11, 12, 21, 22],
        "chosen_attention_mask": [1, 1, 1, 1],
        "chosen_labels": [-100, -100, 21, 22],
        "rejected_input_ids": [11, 12, 31],
        "rejected_attention_mask": [1, 1, 1],
        "rejected_labels": [-100, -100, 31],
    }


def test_tfds_preference_collator_extracts_completion_tokens():
    collator = DataCollatorForPreferenceTFDS(
        max_prompt_length=2,
        max_completion_length=3,
        pad_token_id=0,
        label_pad_token_id=-100,
    )

    batch = collator([_full_sequence_feature()])

    assert batch["prompt_input_ids"].tolist() == [[11, 12]]
    assert batch["prompt_attention_mask"].tolist() == [[1, 1]]
    assert batch["chosen_input_ids"].tolist() == [[21, 22, 0]]
    assert batch["chosen_attention_mask"].tolist() == [[1, 1, 0]]
    assert batch["rejected_input_ids"].tolist() == [[31, 0, 0]]
    assert batch["rejected_attention_mask"].tolist() == [[1, 0, 0]]


def test_grain_preference_collator_extracts_completion_tokens():
    collator = DataCollatorForPreferenceGrain(
        max_prompt_length=2,
        max_completion_length=3,
        pad_token_id=0,
        label_pad_token_id=-100,
    )

    batch = collator(_full_sequence_feature())

    assert batch["prompt_input_ids"].tolist() == [11, 12]
    assert batch["prompt_attention_mask"].tolist() == [1, 1]
    assert batch["chosen_input_ids"].tolist() == [21, 22, 0]
    assert batch["chosen_attention_mask"].tolist() == [1, 1, 0]
    assert batch["rejected_input_ids"].tolist() == [31, 0, 0]
    assert batch["rejected_attention_mask"].tolist() == [1, 0, 0]


def test_tfds_preference_collator_preserves_tools():
    collator = DataCollatorForPreferenceTFDS(
        max_prompt_length=2,
        max_completion_length=3,
        pad_token_id=0,
        label_pad_token_id=-100,
    )

    batch = collator(
        [
            {
                **_full_sequence_feature(),
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            }
        ]
    )

    assert batch["tools"] == [[{"type": "function", "function": {"name": "lookup"}}]]


def test_grain_preference_collator_preserves_tools():
    collator = DataCollatorForPreferenceGrain(
        max_prompt_length=2,
        max_completion_length=3,
        pad_token_id=0,
        label_pad_token_id=-100,
    )

    batch = collator(
        {
            **_full_sequence_feature(),
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
    )

    assert batch["tools"] == [{"type": "function", "function": {"name": "lookup"}}]


def test_dpo_padding_collators_preserve_tools():
    tfds_collator = DPODataCollatorWithPaddingTFDS(
        max_prompt_length=2,
        max_completion_length=4,
        pad_token_id=0,
        prepadded=False,
    )
    grain_collator = DPODataCollatorWithPaddingGrain(
        max_prompt_length=2,
        max_completion_length=4,
        pad_token_id=0,
        prepadded=False,
    )
    feature = {
        **_full_sequence_feature(),
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }

    tfds_batch = tfds_collator([feature])
    grain_batch = grain_collator(feature)

    assert tfds_batch["tools"] == [[{"type": "function", "function": {"name": "lookup"}}]]
    assert grain_batch["tools"] == [{"type": "function", "function": {"name": "lookup"}}]


def test_dpo_transform_preserves_precomputed_reference_columns():
    transform = DPOPreprocessTransform(
        tokenizer=MockTokenizer(),
        max_prompt_length=16,
        max_completion_length=8,
    )

    result = transform(
        {
            "prompt": "Why?",
            "chosen": "Because.",
            "rejected": "No.",
            "ref_chosen_logps": 0.75,
            "ref_rejected_logps": -0.5,
        }
    )

    assert result["ref_chosen_logps"] == pytest.approx(0.75)
    assert result["ref_rejected_logps"] == pytest.approx(-0.5)
