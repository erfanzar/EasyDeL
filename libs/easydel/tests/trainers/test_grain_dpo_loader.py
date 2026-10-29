"""Host-side regression coverage for Grain's map-before-batch DPO collation."""

from __future__ import annotations

import numpy as np
import pytest
from easydel.trainers.utils import CollateMapTransform, DPODataCollatorWithPaddingGrain, ToNumpy
from grain import python as grain

pytestmark = pytest.mark.cpu_ok


def _load_batch(examples, collator):
    """Use the same operation ordering as BaseTrainer's Grain DataLoader."""
    loader = grain.DataLoader(
        data_source=grain.MapDataset.source(examples),
        sampler=grain.IndexSampler(
            num_records=len(examples),
            shard_options=grain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
            num_epochs=1,
            shuffle=False,
            seed=0,
        ),
        operations=[ToNumpy(), CollateMapTransform(collate_fn=collator), grain.Batch(batch_size=2, drop_remainder=True)],
        worker_count=0,
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=2),
    )
    return next(iter(loader))


@pytest.mark.parametrize("is_encoder_decoder", [False, True])
@pytest.mark.parametrize("prepadded", [False, True])
def test_grain_map_before_batch_preserves_token_axes(is_encoder_decoder, prepadded):
    collator = DPODataCollatorWithPaddingGrain(
        max_prompt_length=4,
        max_completion_length=3,
        is_encoder_decoder=is_encoder_decoder,
        pad_token_id=0,
        prepadded=prepadded,
    )
    examples = [
        {"prompt_input_ids": [1, 2], "prompt_attention_mask": [1, 1], "chosen_labels": [7, 8, 9]},
        {"prompt_input_ids": [3, 4], "prompt_attention_mask": [1, 1], "chosen_labels": [10, 11, 12]},
    ]
    result = _load_batch(examples, collator)
    expected_prompt = [[1, 2], [3, 4]] if prepadded else [[0, 0, 1, 2], [0, 0, 3, 4]]
    expected_mask = [[1, 1], [1, 1]] if prepadded else [[0, 0, 1, 1], [0, 0, 1, 1]]
    np.testing.assert_array_equal(result["prompt_input_ids"], expected_prompt)
    np.testing.assert_array_equal(result["prompt_attention_mask"], expected_mask)
    np.testing.assert_array_equal(result["chosen_labels"], [[7, 8, 9], [10, 11, 12]])
    assert result["prompt_input_ids"].shape == (2, 2 if prepadded else 4)
    assert result["chosen_labels"].shape == (2, 3)
    assert all(isinstance(value, np.ndarray) and value.dtype == np.int32 for value in result.values())


def test_grain_encoder_decoder_pads_variable_examples_before_batch():
    collator = DPODataCollatorWithPaddingGrain(
        max_prompt_length=4, max_completion_length=3, is_encoder_decoder=True, prepadded=False
    )
    result = _load_batch(
        [
            {"prompt_input_ids": [1, 2], "prompt_attention_mask": [1, 1], "chosen_labels": [7]},
            {"prompt_input_ids": [3], "prompt_attention_mask": [1], "chosen_labels": [8, 9]},
        ],
        collator,
    )
    np.testing.assert_array_equal(result["prompt_input_ids"], [[0, 0, 1, 2], [0, 0, 0, 3]])
    np.testing.assert_array_equal(result["prompt_attention_mask"], [[0, 0, 1, 1], [0, 0, 0, 1]])
    np.testing.assert_array_equal(result["chosen_labels"], [[-100, -100, 7], [-100, 8, 9]])
    assert all(value.ndim == 2 and value.dtype == np.int32 for value in result.values())


@pytest.mark.parametrize("container", [list, np.asarray])
@pytest.mark.parametrize("prepadded", [False, True])
def test_grain_encoder_decoder_single_sequence_keeps_one_axis(container, prepadded):
    collator = DPODataCollatorWithPaddingGrain(
        max_prompt_length=4,
        max_completion_length=3,
        is_encoder_decoder=True,
        prepadded=prepadded,
        label_pad_token_id=-1,
    )
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    result = collator(
        {
            "prompt_input_ids": container([1, 2]),
            "chosen_labels": container([7]),
            "chosen_logps": -2.5,
            "tools": tools,
            "text": "drop non-array fields",
        }
    )
    np.testing.assert_array_equal(result["prompt_input_ids"], [1, 2] if prepadded else [0, 0, 1, 2])
    np.testing.assert_array_equal(result["chosen_labels"], [7] if prepadded else [-1, -1, 7])
    assert result["prompt_input_ids"].ndim == result["chosen_labels"].ndim == 1
    assert result["chosen_logps"].shape == ()
    assert result["tools"] == tools
    assert "text" not in result


@pytest.mark.parametrize("prepadded", [False, True])
def test_grain_encoder_decoder_batched_ragged_columns_remain_supported(prepadded):
    collator = DPODataCollatorWithPaddingGrain(
        max_prompt_length=4, max_completion_length=3, is_encoder_decoder=True, prepadded=prepadded
    )
    result = collator({"prompt_input_ids": [[1, 2, 3], [4]], "chosen_labels": [[7, 8], [9]]})
    expected_prompt = [[1, 2, 3], [0, 0, 4]] if prepadded else [[0, 1, 2, 3], [0, 0, 0, 4]]
    expected_labels = [[7, 8], [-100, 9]] if prepadded else [[-100, 7, 8], [-100, -100, 9]]
    np.testing.assert_array_equal(result["prompt_input_ids"], expected_prompt)
    np.testing.assert_array_equal(result["chosen_labels"], expected_labels)
    assert all(isinstance(value, np.ndarray) and value.dtype == np.int32 for value in result.values())


def test_grain_encoder_decoder_empty_single_sequence_is_not_a_batch():
    collator = DPODataCollatorWithPaddingGrain(
        max_prompt_length=4, max_completion_length=3, is_encoder_decoder=True, prepadded=False
    )
    result = collator({"prompt_input_ids": [], "chosen_labels": []})
    np.testing.assert_array_equal(result["prompt_input_ids"], [0, 0, 0, 0])
    np.testing.assert_array_equal(result["chosen_labels"], [-100, -100, -100])


def test_grain_encoder_decoder_requires_prompt_padding_token():
    collator = DPODataCollatorWithPaddingGrain(
        max_prompt_length=4, max_completion_length=3, is_encoder_decoder=True, pad_token_id=None, prepadded=False
    )
    with pytest.raises(ValueError, match="padding token"):
        collator({"prompt_input_ids": [1, 2]})
