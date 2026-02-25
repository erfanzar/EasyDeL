from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from easydel.trainers.binary_classifier_optimization_trainer._fn import (
    concatenated_forward as bco_concatenated_forward,
)
from easydel.trainers.contrastive_preference_optimization_trainer._fn import (
    concatenated_forward as cpo_concatenated_forward,
)
from easydel.trainers.direct_preference_optimization_trainer._fn import (
    concatenated_forward as dpo_concatenated_forward,
)
from easydel.trainers.odds_ratio_preference_optimization_trainer._fn import (
    concatenated_forward as orpo_concatenated_forward,
)


class _StrictModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask, labels=None):
        self.calls += 1
        batch_size, sequence_length = input_ids.shape
        vocab_size = 32
        logits = jnp.zeros((batch_size, sequence_length, vocab_size), dtype=jnp.float32)
        return SimpleNamespace(logits=logits)


class _StrictEncoderModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        batch_size, sequence_length = input_ids.shape
        vocab_size = 32
        logits = jnp.zeros((batch_size, sequence_length, vocab_size), dtype=jnp.float32)
        return SimpleNamespace(logits=logits)


def _preference_batch(batch_size: int = 2, prompt_len: int = 3, completion_len: int = 3):
    prompt_input_ids = (jnp.arange(batch_size * prompt_len).reshape(batch_size, prompt_len) % 7) + 1
    prompt_attention_mask = jnp.ones_like(prompt_input_ids)
    chosen_input_ids = (jnp.arange(batch_size * completion_len).reshape(batch_size, completion_len) % 7) + 1
    rejected_input_ids = ((chosen_input_ids + 1) % 7) + 1
    chosen_attention_mask = jnp.ones_like(chosen_input_ids)
    rejected_attention_mask = jnp.ones_like(rejected_input_ids)
    return {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "pixel_values": jnp.ones((batch_size, 1, 1, 1), dtype=jnp.float32),
        "pixel_attention_mask": jnp.ones((batch_size, 1), dtype=jnp.int32),
        "image_sizes": jnp.ones((batch_size, 2), dtype=jnp.int32),
    }


def test_dpo_forward_filters_unsupported_model_kwargs():
    model = _StrictModel()
    batch = _preference_batch()
    outputs = dpo_concatenated_forward(
        model=model,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )
    assert model.calls == 1
    assert outputs["chosen_logps"].shape == (2,)
    assert outputs["rejected_logps"].shape == (2,)


def test_cpo_forward_filters_unsupported_model_kwargs():
    model = _StrictModel()
    batch = _preference_batch()
    outputs = cpo_concatenated_forward(
        model=model,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )
    assert model.calls == 1
    assert outputs["chosen_logps"].shape == (2,)
    assert outputs["rejected_logps"].shape == (2,)


def test_bco_forward_filters_unsupported_model_kwargs():
    model = _StrictModel()
    batch_size, sequence_length = 2, 4
    completion_input_ids = (jnp.arange(batch_size * sequence_length).reshape(batch_size, sequence_length) % 7) + 1
    completion_attention_mask = jnp.ones_like(completion_input_ids)
    batch = {
        "prompt_input_ids": completion_input_ids,
        "prompt_attention_mask": completion_attention_mask,
        "completion_input_ids": completion_input_ids,
        "completion_attention_mask": completion_attention_mask,
        "completion_labels": completion_input_ids,
        "pixel_values": jnp.ones((batch_size, 1, 1, 1), dtype=jnp.float32),
        "pixel_attention_mask": jnp.ones((batch_size, 1), dtype=jnp.int32),
        "image_sizes": jnp.ones((batch_size, 2), dtype=jnp.int32),
    }
    outputs = bco_concatenated_forward(
        model=model,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )
    assert model.calls == 1
    assert outputs["completion_logps"].shape == (2,)


def test_orpo_forward_filters_encoder_decoder_kwargs():
    model = _StrictEncoderModel()
    pair_batch_size, sequence_length = 1, 6
    chosen_input_ids = (jnp.arange(pair_batch_size * sequence_length).reshape(pair_batch_size, sequence_length) % 7) + 1
    rejected_input_ids = ((chosen_input_ids + 1) % 7) + 1
    attention_mask = jnp.ones_like(chosen_input_ids)
    prompt_input_ids = (jnp.arange(2 * 3).reshape(2, 3) % 7) + 1
    prompt_attention_mask = jnp.ones_like(prompt_input_ids)
    batch = {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": attention_mask,
        "chosen_labels": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": attention_mask,
        "rejected_labels": rejected_input_ids,
    }
    state = SimpleNamespace(model=model)
    outputs = orpo_concatenated_forward(
        state=state,
        batch=batch,
        is_encoder_decoder=True,
        label_pad_token_id=-100,
        padding_value=0,
    )
    assert model.calls == 1
    assert len(outputs) == 6
