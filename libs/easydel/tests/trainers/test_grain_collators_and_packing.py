"""Regression coverage for per-example Grain columns and ragged padding."""

from __future__ import annotations

import numpy as np
import pytest
from easydel.trainers.utils import (
    BCODataCollatorGrain,
    EmbeddingDataCollatorGrain,
    create_constant_length_dataset,
    np_pad,
    pad,
)
from jax import numpy as jnp

from .test_grain_dpo_loader import _load_batch


@pytest.mark.cpu_ok
def test_bco_grain_scalar_columns_get_only_one_batch_axis():
    examples = [
        {
            "prompt_input_ids": [token],
            "prompt_attention_mask": [1],
            "completion_input_ids": [token, 7],
            "completion_attention_mask": [1, 1],
            "completion_labels": [-100, 7],
            "label": label,
            "reference_logps": logp,
        }
        for token, label, logp in [(1, True, -0.25), (2, False, -2.0)]
    ]
    result = _load_batch(
        examples,
        BCODataCollatorGrain(
            max_prompt_length=2,
            max_completion_length=2,
            pad_token_id=0,
            label_pad_token_id=-100,
            is_encoder_decoder=False,
        ),
    )
    assert result["label"].shape == result["reference_logps"].shape == (2,)
    np.testing.assert_array_equal(result["label"], [True, False])
    np.testing.assert_array_equal(result["reference_logps"], [-0.25, -2.0])
    assert result["label"].dtype == np.bool_
    assert result["reference_logps"].dtype == np.float32


@pytest.mark.cpu_ok
@pytest.mark.parametrize("has_negatives", [False, True])
def test_embedding_grain_maps_individual_examples_before_batch(has_negatives):
    examples = [
        {
            f"{name}_{kind}": [token, token + 1] if kind == "input_ids" else [1, 1]
            for name in ("query", "positive", "negative")
            for kind in ("input_ids", "attention_mask")
        }
        for token in (3, 8)
    ]
    result = _load_batch(examples, EmbeddingDataCollatorGrain(has_negatives=has_negatives))
    for name in ("query", "positive", "negative") if has_negatives else ("query", "positive"):
        np.testing.assert_array_equal(result[f"{name}_input_ids"], [[3, 4], [8, 9]])
        np.testing.assert_array_equal(result[f"{name}_attention_mask"], [[1, 1], [1, 1]])
    assert ("negative_input_ids" in result) == has_negatives
    assert all(value.shape == (2, 2) and value.dtype == np.int32 for value in result.values())


def test_embedding_grain_keeps_legacy_batched_list_interface():
    result = EmbeddingDataCollatorGrain()([{"query_input_ids": [1, 2]}, {"query_input_ids": [3, 4]}])
    np.testing.assert_array_equal(np.asarray(result["query_input_ids"]), [[1, 2], [3, 4]])


def _expected_padding(rows, length, side):
    length = max(len(row) for row in rows) if length is None else length
    expected = []
    for row in rows:
        retained = row[max(len(row) - length, 0) :] if side == "left" else row[:length]
        padding = [-1] * (length - len(retained))
        expected.append(padding + retained if side == "left" else retained + padding)
    return np.array(expected, dtype=np.int32)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("length", [None, 0, 2, 5])
@pytest.mark.parametrize("reverse", [False, True])
def test_numpy_pad_ragged_inputs_and_truncation(side, length, reverse):
    rows = [[1], [2, 3, 4]]
    if reverse:
        rows.reverse()
    result = np_pad([np.array(row, dtype=np.int32) for row in rows], length, -1, side)
    np.testing.assert_array_equal(result, _expected_padding(rows, length, side))
    assert result.dtype == np.int32


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("length", [None, 0, 2, 5])
@pytest.mark.parametrize("reverse", [False, True])
def test_jax_pad_ragged_inputs_and_truncation(side, length, reverse):
    rows = [[1], [2, 3, 4]]
    if reverse:
        rows.reverse()
    result = pad([jnp.array(row, dtype=jnp.int32) for row in rows], length, -1, side)
    np.testing.assert_array_equal(np.asarray(result), _expected_padding(rows, length, side))
    assert result.dtype == jnp.int32


@pytest.mark.cpu_ok
@pytest.mark.parametrize("side", ["left", "right"])
def test_numpy_pad_uses_last_axis_for_multidimensional_rows(side):
    rows = [np.array([[1], [2]], dtype=np.int32), np.array([[3, 4, 5], [6, 7, 8]], dtype=np.int32)]
    result = np_pad(rows, 2, -1, side)
    expected = [[[-1, 1], [-1, 2]], [[4, 5], [7, 8]]] if side == "left" else [[[1, -1], [2, -1]], [[3, 4], [6, 7]]]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("side", ["left", "right"])
def test_jax_pad_uses_last_axis_for_multidimensional_rows(side):
    rows = [jnp.array([[1], [2]], dtype=jnp.int32), jnp.array([[3, 4, 5], [6, 7, 8]], dtype=jnp.int32)]
    result = pad(rows, 2, -1, side)
    expected = [[[-1, 1], [-1, 2]], [[4, 5], [7, 8]]] if side == "left" else [[[1, -1], [2, -1]], [[3, 4], [6, 7]]]
    np.testing.assert_array_equal(np.asarray(result), expected)


class _TextTokenizer:
    eos_token_id = 99

    def __call__(self, *, text, **kwargs):
        ids = [[int(token) for token in row.split()] for row in text]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


@pytest.mark.cpu_ok
@pytest.mark.parametrize("texts", [[], ["1"], [""]])
@pytest.mark.parametrize("shuffle", [False, True])
def test_constant_length_empty_or_short_finite_input_ends_cleanly(texts, shuffle):
    generator = create_constant_length_dataset(
        _TextTokenizer(),
        [{"text": text} for text in texts],
        dataset_text_field="text",
        seq_length=4,
        append_concat_token=False,
        shuffle=shuffle,
    )
    assert list(generator()) == []


@pytest.mark.cpu_ok
def test_constant_length_empty_infinite_input_does_not_spin():
    generator = create_constant_length_dataset(_TextTokenizer(), [], dataset_text_field="text", infinite=True)
    assert list(generator()) == []


def test_constant_length_retains_tokens_between_small_buffers():
    generator = create_constant_length_dataset(
        _TextTokenizer(),
        [{"text": str(token)} for token in range(1, 6)],
        dataset_text_field="text",
        seq_length=4,
        chars_per_token=0.25,
        num_of_sequences=1,
        append_concat_token=False,
        shuffle=False,
    )
    result = list(generator())
    assert len(result) == 1
    np.testing.assert_array_equal(np.asarray(result[0]["input_ids"]), [1, 2, 3, 4])
    np.testing.assert_array_equal(np.asarray(result[0]["attention_mask"]), [1, 1, 1, 1])


def test_constant_length_infinite_restarts_at_exact_buffer_boundary():
    from itertools import islice

    generator = create_constant_length_dataset(
        _TextTokenizer(),
        [{"text": "1"}],
        dataset_text_field="text",
        infinite=True,
        seq_length=4,
        chars_per_token=0.25,
        num_of_sequences=1,
        append_concat_token=False,
        shuffle=False,
    )
    with pytest.warns(UserWarning, match="iterator is reset"):
        result = list(islice(generator(), 2))
    assert len(result) == 2
    for example in result:
        np.testing.assert_array_equal(np.asarray(example["input_ids"]), [1, 1, 1, 1])
        np.testing.assert_array_equal(np.asarray(example["attention_mask"]), [1, 1, 1, 1])
