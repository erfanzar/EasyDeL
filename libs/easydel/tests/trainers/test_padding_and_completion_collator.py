"""Regression tests for ``pad_to_length``, ``pad_sequence`` and ``DataCollatorForCompletionOnlyLM``."""

from __future__ import annotations

import numpy as np
import pytest
from easydel.trainers.utils import (
    DataCollatorForCompletionOnlyLM,
    DPODataCollatorWithPaddingGrain,
    DPODataCollatorWithPaddingTFDS,
    pad_sequence,
    pad_to_length,
)
from jax import numpy as jnp


class _FakeTokenizer:
    _pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 99
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


def test_pad_to_length_truncates_along_requested_axis():
    x = jnp.arange(12).reshape(4, 3)
    assert pad_to_length(x, 2, 0, axis=0).shape == (2, 3)
    assert pad_to_length(x, 2, 0, axis=1).shape == (4, 2)
    x3 = jnp.arange(24).reshape(2, 3, 4)
    out = pad_to_length(x3, 2, 0, axis=1)
    assert out.shape == (2, 2, 4)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x3)[:, :2])
    assert pad_to_length(x, 5, -1, axis=0).shape == (5, 3)


def test_pad_sequence_keeps_integer_dtype_and_batch_shape():
    seqs = [jnp.asarray([1, 2, 3], dtype=jnp.int32), jnp.asarray([4], dtype=jnp.int32)]
    right = pad_sequence(seqs, batch_first=True, padding_value=-100)
    left = pad_sequence(seqs, batch_first=False, padding_value=-100)
    assert right.shape == left.shape == (2, 3)
    assert jnp.issubdtype(right.dtype, jnp.integer)
    np.testing.assert_array_equal(np.asarray(right), [[1, 2, 3], [4, -100, -100]])
    np.testing.assert_array_equal(np.asarray(left), [[1, 2, 3], [-100, -100, 4]])


def test_dpo_grain_encoder_decoder_path_pads_integers():
    collator = DPODataCollatorWithPaddingGrain(
        max_prompt_length=4, max_completion_length=4, pad_token_id=0, is_encoder_decoder=True
    )
    out = collator(
        {
            "prompt_input_ids": [[1, 2, 3], [4, 5, 6]],
            "chosen_labels": [[7, 8, 9], [1, 2, 3]],
        }
    )
    assert out["prompt_input_ids"].shape == (2, 3)
    assert jnp.issubdtype(out["prompt_input_ids"].dtype, jnp.integer)
    assert jnp.issubdtype(out["chosen_labels"].dtype, jnp.integer)


def test_completion_only_collator_accepts_tokenizer_instance_and_masks_prompt():
    collator = DataCollatorForCompletionOnlyLM(_FakeTokenizer(), response_template=[7, 8], mlm=False)
    batch = collator([[1, 2, 7, 8, 3, 4], [1, 7, 8, 5]])
    np.testing.assert_array_equal(np.asarray(batch["input_ids"]), [[1, 2, 7, 8, 3, 4], [1, 7, 8, 5, 0, 0]])
    np.testing.assert_array_equal(
        np.asarray(batch["labels"]),
        [[-100, -100, -100, -100, 3, 4], [-100, -100, -100, 5, -100, -100]],
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("tokenizer_name", [None, "local-test-tokenizer"])
def test_completion_only_collator_resolves_string_template_with_tokenizer_instance(monkeypatch, tokenizer_name):
    from transformers import AutoTokenizer

    tokenizer = _FakeTokenizer()
    if tokenizer_name is not None:
        monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda name: tokenizer)
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer if tokenizer_name is None else tokenizer_name,
        response_template="7 8",
        instruction_template="6",
    )
    assert collator.response_token_ids == [7, 8]
    assert collator.instruction_token_ids == [6]
    assert collator.mlm is False


@pytest.mark.parametrize("kind", ["grain", "tfds"])
def test_dpo_encoder_decoder_ragged_batch_has_integer_labels(kind):
    kwargs = dict(max_prompt_length=4, max_completion_length=4, pad_token_id=0, is_encoder_decoder=True)
    columns = {"prompt_input_ids": [[1, 2, 3], [4]], "chosen_labels": [[7, 8], [9]]}
    if kind == "grain":
        output = DPODataCollatorWithPaddingGrain(**kwargs)(columns)
    else:
        features = [{key: values[row] for key, values in columns.items()} for row in range(2)]
        output = DPODataCollatorWithPaddingTFDS(**kwargs)(features)
    np.testing.assert_array_equal(np.asarray(output["prompt_input_ids"]), [[1, 2, 3], [0, 0, 4]])
    np.testing.assert_array_equal(np.asarray(output["chosen_labels"]), [[7, 8], [-100, 9]])
    assert output["prompt_input_ids"].dtype == output["chosen_labels"].dtype == jnp.int32


@pytest.mark.parametrize("left_pad", [False, True])
def test_completion_only_preserves_real_eos_when_pad_equals_eos(left_pad):
    tokenizer = _FakeTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left" if left_pad else "right"
    collator = DataCollatorForCompletionOnlyLM(tokenizer, response_template=[7, 8])
    result = collator([[1, 7, 8, 3, 99], [7, 8, 99]])
    expected = [[-100, -100, -100, 3, 99], [-100, -100, -100, -100, 99] if left_pad else [-100, -100, 99, -100, -100]]
    np.testing.assert_array_equal(np.asarray(result["labels"]), expected)


def test_completion_only_honors_existing_attention_mask_and_tools():
    collator = DataCollatorForCompletionOnlyLM(_FakeTokenizer(), response_template=[7, 8], ignore_index=-1)
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    result = collator([{"input_ids": [1, 7, 8, 3, 0], "attention_mask": [1, 1, 1, 1, 0], "tools": tools}])
    np.testing.assert_array_equal(np.asarray(result["labels"]), [[-1, -1, -1, 3, -1]])
    assert result["tools"] == [tools]


def test_completion_only_masks_multiple_instruction_turns():
    collator = DataCollatorForCompletionOnlyLM(_FakeTokenizer(), response_template=[7, 8], instruction_template=[6])
    result = collator([[6, 1, 7, 8, 3, 6, 2, 7, 8, 4, 6, 5]])
    np.testing.assert_array_equal(
        np.asarray(result["labels"]), [[-100, -100, -100, -100, 3, -100, -100, -100, -100, 4, -100, -100]]
    )


def test_completion_only_missing_template_ignores_example():
    collator = DataCollatorForCompletionOnlyLM(_FakeTokenizer(), response_template=[7, 8])
    with pytest.warns(UserWarning):
        result = collator([[1, 2, 3]])
    np.testing.assert_array_equal(np.asarray(result["labels"]), [[-100, -100, -100]])


def test_completion_only_mlm_uses_original_boundaries_and_zero_padded_masks(monkeypatch):
    import random

    class Tokenizer(_FakeTokenizer):
        pad_token_id = 50
        mask_token_id = 49

        def _convert_id_to_token(self, token):
            return str(token)

        def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
            return [token in (50, 99) for token in ids]

        def __len__(self):
            return 100

    # Whole-word selection picks the final tokens, leaving the response template's labels ignored.
    monkeypatch.setattr(random, "shuffle", lambda candidates: candidates.reverse())
    collator = DataCollatorForCompletionOnlyLM(Tokenizer(), response_template=[7, 8], mlm=True, ignore_index=-1)
    inputs = [[1, 7, 8, *range(10, 27)], [1, 7, 8, *range(10, 17)]]
    with pytest.warns(UserWarning):
        result = collator(inputs)
    expected = [[-1] * 17 + [24, 25, 26], [-1] * 8 + [15, 16] + [-1] * 10]
    np.testing.assert_array_equal(np.asarray(result["labels"]), expected)


def test_pad_sequence_explicit_maximum_truncates_and_pads_consistently():
    sequences = [jnp.asarray([1, 2, 3, 4]), jnp.asarray([5])]
    np.testing.assert_array_equal(np.asarray(pad_sequence(sequences, max_len=2)), [[3, 4], [0, 5]])
    np.testing.assert_array_equal(np.asarray(pad_sequence(sequences, batch_first=True, max_len=2)), [[1, 2], [5, 0]])
