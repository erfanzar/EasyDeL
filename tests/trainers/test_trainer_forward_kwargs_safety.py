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

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from easydel.trainers.binary_classifier_optimization_trainer._fn import (
    concatenated_forward as bco_concatenated_forward,
)
from easydel.trainers.contrastive_preference_optimization_trainer._fn import (
    concatenated_forward as cpo_concatenated_forward,
)
from easydel.trainers.direct_preference_optimization_trainer._fn import (
    concatenated_forward as dpo_concatenated_forward,
)
from easydel.trainers.direct_preference_optimization_trainer._fn import (
    concatenated_inputs as dpo_concatenated_inputs,
)
from easydel.trainers.distillation_trainer.distillation_trainer import DistillationTrainer
from easydel.trainers.group_relative_policy_optimization._fn import get_per_token_logps
from easydel.trainers.odds_ratio_preference_optimization_trainer._fn import (
    concatenated_forward as orpo_concatenated_forward,
)
from easydel.trainers.proximal_policy_optimization_trainer._fn import get_per_token_logps_values_entropies
from easydel.trainers.seq_kd_trainer.seq_kd_trainer import SeqKDTrainer
from easydel.trainers.supervised_fine_tuning_trainer.sft_config import SFTConfig
from easydel.trainers.supervised_fine_tuning_trainer.sft_trainer import SFTTrainer
from easydel.trainers.trainer import Trainer


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


class _KnownLogitsModel:
    def __call__(self, input_ids, attention_mask, labels=None):
        del attention_mask, labels
        batch_size, sequence_length = input_ids.shape
        vocab_size = 11
        logits = jnp.arange(
            batch_size * sequence_length * vocab_size,
            dtype=jnp.float32,
        ).reshape(batch_size, sequence_length, vocab_size)
        return SimpleNamespace(logits=logits / 13.0)


class _HeadlessChunkedKnownLogitsModel:
    def __init__(self):
        self.config = SimpleNamespace(lmhead_chunksize=2)
        self._projection = (jnp.arange(4 * 11, dtype=jnp.float32).reshape(4, 11) / 17.0) - 0.3

    def __call__(self, input_ids, attention_mask, labels=None, apply_lm_head=True, output_hidden_states=False):
        del output_hidden_states
        hidden_states = jax.nn.one_hot(
            input_ids % self._projection.shape[0], self._projection.shape[0], dtype=jnp.float32
        )
        logits = self.compute_lm_logits(hidden_states) if apply_lm_head else None
        return SimpleNamespace(logits=logits, last_hidden_state=hidden_states)

    def prepare_lm_head_inputs(self, hidden_states):
        return hidden_states

    def compute_lm_logits(self, hidden_states):
        return jnp.einsum("bsh,hv->bsv", hidden_states, self._projection)


class _HeadlessChunkedValueHeadModel:
    def __init__(self):
        self.model = _HeadlessChunkedKnownLogitsModel()
        self.config = self.model.config

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def value_head(self, hidden_states):
        return jnp.sum(hidden_states, axis=-1, keepdims=True) / 7.0


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
        logprob_vocab_chunk_size=2048,
    )
    assert model.calls == 1
    assert outputs["chosen_logps"].shape == (2,)
    assert outputs["rejected_logps"].shape == (2,)


def test_dpo_forward_matches_dense_log_softmax_reference():
    model = _KnownLogitsModel()
    batch = _preference_batch(batch_size=2, prompt_len=2, completion_len=2)

    outputs = dpo_concatenated_forward(
        model=model,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=2048,
    )

    concatenated_batch = dpo_concatenated_inputs(batch=batch, padding_value=0)
    input_ids = jnp.concatenate(
        [concatenated_batch["prompt_input_ids"], concatenated_batch["completion_input_ids"]],
        axis=1,
    )
    attention_mask = jnp.concatenate(
        [
            concatenated_batch["prompt_attention_mask"],
            concatenated_batch["completion_attention_mask"],
        ],
        axis=1,
    )
    loss_mask = jnp.concatenate(
        [
            jnp.zeros_like(concatenated_batch["prompt_attention_mask"]),
            concatenated_batch["completion_attention_mask"],
        ],
        axis=1,
    )

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    labels = jnp.roll(input_ids, shift=-1, axis=1)
    loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype(bool)
    labels = jnp.where(loss_mask, labels, 0)

    dense_log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered_dense = jnp.take_along_axis(
        dense_log_probs,
        labels[..., None],
        axis=-1,
    )[..., 0]
    expected_logps = jnp.roll(
        jnp.where(loss_mask, gathered_dense, 0.0),
        shift=1,
        axis=1,
    ).sum(-1)
    expected_mean_chosen_logits = jnp.where(loss_mask[:2], logits[:2].sum(axis=-1), 0.0).sum() / jnp.maximum(
        loss_mask[:2].sum(), 1
    )
    expected_mean_rejected_logits = jnp.where(loss_mask[2:], logits[2:].sum(axis=-1), 0.0).sum() / jnp.maximum(
        loss_mask[2:].sum(), 1
    )

    assert jnp.allclose(outputs["chosen_logps"], expected_logps[:2], atol=1e-6)
    assert jnp.allclose(outputs["rejected_logps"], expected_logps[2:], atol=1e-6)
    assert jnp.allclose(outputs["mean_chosen_logits"], expected_mean_chosen_logits, atol=1e-6)
    assert jnp.allclose(outputs["mean_rejected_logits"], expected_mean_rejected_logits, atol=1e-6)


def test_dpo_forward_matches_dense_reference_with_headless_chunked_lm_head():
    model = _HeadlessChunkedKnownLogitsModel()
    batch = _preference_batch(batch_size=2, prompt_len=2, completion_len=2)

    outputs = dpo_concatenated_forward(
        model=model,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=3,
    )

    concatenated_batch = dpo_concatenated_inputs(batch=batch, padding_value=0)
    input_ids = jnp.concatenate(
        [concatenated_batch["prompt_input_ids"], concatenated_batch["completion_input_ids"]],
        axis=1,
    )
    attention_mask = jnp.concatenate(
        [
            concatenated_batch["prompt_attention_mask"],
            concatenated_batch["completion_attention_mask"],
        ],
        axis=1,
    )
    loss_mask = jnp.concatenate(
        [
            jnp.zeros_like(concatenated_batch["prompt_attention_mask"]),
            concatenated_batch["completion_attention_mask"],
        ],
        axis=1,
    )

    logits = model(input_ids=input_ids, attention_mask=attention_mask, apply_lm_head=True).logits
    labels = jnp.roll(input_ids, shift=-1, axis=1)
    loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype(bool)
    labels = jnp.where(loss_mask, labels, 0)

    dense_log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered_dense = jnp.take_along_axis(
        dense_log_probs,
        labels[..., None],
        axis=-1,
    )[..., 0]
    expected_logps = jnp.roll(
        jnp.where(loss_mask, gathered_dense, 0.0),
        shift=1,
        axis=1,
    ).sum(-1)
    expected_mean_chosen_logits = jnp.where(loss_mask[:2], logits[:2].sum(axis=-1), 0.0).sum() / jnp.maximum(
        loss_mask[:2].sum(), 1
    )
    expected_mean_rejected_logits = jnp.where(loss_mask[2:], logits[2:].sum(axis=-1), 0.0).sum() / jnp.maximum(
        loss_mask[2:].sum(), 1
    )

    assert jnp.allclose(outputs["chosen_logps"], expected_logps[:2], atol=1e-6)
    assert jnp.allclose(outputs["rejected_logps"], expected_logps[2:], atol=1e-6)
    assert jnp.allclose(outputs["mean_chosen_logits"], expected_mean_chosen_logits, atol=1e-6)
    assert jnp.allclose(outputs["mean_rejected_logits"], expected_mean_rejected_logits, atol=1e-6)


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


def test_cpo_forward_matches_dense_reference_with_headless_chunked_lm_head():
    model = _HeadlessChunkedKnownLogitsModel()
    batch = _preference_batch(batch_size=2, prompt_len=2, completion_len=2)

    outputs = cpo_concatenated_forward(
        model=model,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=3,
    )

    concatenated_batch = dpo_concatenated_inputs(batch=batch, padding_value=0)
    input_ids = jnp.concatenate(
        [concatenated_batch["prompt_input_ids"], concatenated_batch["completion_input_ids"]],
        axis=1,
    )
    attention_mask = jnp.concatenate(
        [concatenated_batch["prompt_attention_mask"], concatenated_batch["completion_attention_mask"]],
        axis=1,
    )
    loss_mask = jnp.concatenate(
        [jnp.zeros_like(concatenated_batch["prompt_attention_mask"]), concatenated_batch["completion_attention_mask"]],
        axis=1,
    )
    logits = model(input_ids=input_ids, attention_mask=attention_mask, apply_lm_head=True).logits
    labels = jnp.roll(input_ids, shift=-1, axis=1)
    loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype(bool)
    labels = jnp.where(loss_mask, labels, 0)
    dense_log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered_dense = jnp.take_along_axis(dense_log_probs, labels[..., None], axis=-1)[..., 0]
    expected_per_token = jnp.roll(jnp.where(loss_mask, gathered_dense, 0.0), shift=1, axis=1)
    expected_sum_logps = expected_per_token.sum(-1)
    expected_lengths = jnp.maximum(loss_mask.sum(-1), 1)
    expected_mean_chosen_logits = jnp.where(loss_mask[:2], logits[:2].sum(axis=-1), 0.0).sum() / jnp.maximum(
        loss_mask[:2].sum(), 1
    )
    expected_mean_rejected_logits = jnp.where(loss_mask[2:], logits[2:].sum(axis=-1), 0.0).sum() / jnp.maximum(
        loss_mask[2:].sum(), 1
    )

    assert jnp.allclose(outputs["chosen_logps"], expected_sum_logps[:2], atol=1e-6)
    assert jnp.allclose(outputs["rejected_logps"], expected_sum_logps[2:], atol=1e-6)
    assert jnp.allclose(outputs["chosen_logps_raw"], expected_sum_logps[:2], atol=1e-6)
    assert jnp.allclose(outputs["rejected_logps_raw"], expected_sum_logps[2:], atol=1e-6)
    assert jnp.array_equal(outputs["chosen_lengths"], expected_lengths[:2])
    assert jnp.array_equal(outputs["rejected_lengths"], expected_lengths[2:])
    assert jnp.allclose(outputs["mean_chosen_logits"], expected_mean_chosen_logits, atol=1e-6)
    assert jnp.allclose(outputs["mean_rejected_logits"], expected_mean_rejected_logits, atol=1e-6)


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


def test_bco_forward_matches_dense_reference_with_headless_chunked_lm_head():
    model = _HeadlessChunkedKnownLogitsModel()
    batch_size, sequence_length = 2, 4
    completion_input_ids = (jnp.arange(batch_size * sequence_length).reshape(batch_size, sequence_length) % 7) + 1
    completion_attention_mask = jnp.ones_like(completion_input_ids)
    batch = {
        "prompt_input_ids": completion_input_ids,
        "prompt_attention_mask": completion_attention_mask,
        "completion_input_ids": completion_input_ids,
        "completion_attention_mask": completion_attention_mask,
        "completion_labels": completion_input_ids,
    }

    outputs = bco_concatenated_forward(
        model=model,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=3,
    )

    logits = model(
        input_ids=completion_input_ids,
        attention_mask=completion_attention_mask,
        apply_lm_head=True,
    ).logits
    logits_shifted = logits[:, :-1, :]
    labels_shifted = completion_input_ids[:, 1:]
    loss_mask = labels_shifted != -100
    dense_log_probs = jax.nn.log_softmax(logits_shifted, axis=-1)
    gathered_dense = jnp.take_along_axis(dense_log_probs, labels_shifted[..., None], axis=-1)[..., 0]
    expected_completion_logps = jnp.where(loss_mask, gathered_dense, 0.0).sum(axis=1)
    expected_mean_logits = jnp.where(loss_mask, logits_shifted.sum(axis=-1), 0.0).sum() / jnp.maximum(loss_mask.sum(), 1)

    assert jnp.allclose(outputs["completion_logps"], expected_completion_logps, atol=1e-6)
    assert jnp.allclose(outputs["mean_completion_logits"], expected_mean_logits, atol=1e-6)


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


def test_orpo_forward_matches_dense_reference_with_headless_chunked_lm_head():
    model = _HeadlessChunkedKnownLogitsModel()
    pair_batch_size, sequence_length = 2, 5
    chosen_input_ids = (jnp.arange(pair_batch_size * sequence_length).reshape(pair_batch_size, sequence_length) % 7) + 1
    rejected_input_ids = ((chosen_input_ids + 1) % 7) + 1
    attention_mask = jnp.ones_like(chosen_input_ids)
    batch = {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": attention_mask,
        "chosen_labels": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": attention_mask,
        "rejected_labels": rejected_input_ids,
    }
    state = SimpleNamespace(model=model)

    headless_outputs = orpo_concatenated_forward(
        state=state,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=3,
    )

    model.config.lmhead_chunksize = None
    dense_outputs = orpo_concatenated_forward(
        state=state,
        batch=batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=3,
    )

    concatenated_input_ids = jnp.concatenate([chosen_input_ids, rejected_input_ids], axis=0)
    concatenated_attention_mask = jnp.concatenate([attention_mask, attention_mask], axis=0)
    labels = jnp.where(concatenated_attention_mask == 1, concatenated_input_ids, -100)
    logits = model(
        input_ids=concatenated_input_ids,
        attention_mask=concatenated_attention_mask,
        apply_lm_head=True,
    ).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    safe_labels = jnp.where(shifted_labels == -100, 0, shifted_labels)
    loss_mask = shifted_labels != -100
    dense_log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
    gathered_dense = jnp.take_along_axis(dense_log_probs, safe_labels[..., None], axis=-1)[..., 0]
    expected_mean_logps = jnp.where(loss_mask, gathered_dense, 0.0).sum(axis=1) / jnp.maximum(loss_mask.sum(axis=1), 1)
    expected_nll = -jnp.where(loss_mask[:pair_batch_size], gathered_dense[:pair_batch_size], 0.0).sum() / jnp.maximum(
        loss_mask[:pair_batch_size].sum(),
        1,
    )
    expected_accuracy = jnp.where(
        loss_mask[:pair_batch_size],
        (jnp.argmax(shifted_logits[:pair_batch_size], axis=-1) == safe_labels[:pair_batch_size]).astype(jnp.float32),
        0.0,
    ).sum() / jnp.maximum(loss_mask[:pair_batch_size].sum(), 1)
    expected_mean_logits = jnp.where(loss_mask, shifted_logits.sum(axis=-1), 0.0).sum(axis=1) / jnp.maximum(
        loss_mask.sum(axis=1), 1
    )

    for outputs in (headless_outputs, dense_outputs):
        assert jnp.allclose(outputs[0], expected_mean_logps[:pair_batch_size], atol=1e-6)
        assert jnp.allclose(outputs[1], expected_mean_logps[pair_batch_size:], atol=1e-6)
        assert jnp.allclose(outputs[2], expected_mean_logits[:pair_batch_size], atol=1e-6)
        assert jnp.allclose(outputs[3], expected_mean_logits[pair_batch_size:], atol=1e-6)
        assert jnp.allclose(outputs[4], expected_nll, atol=1e-6)
        assert jnp.allclose(outputs[5], expected_accuracy, atol=1e-6)


def test_grpo_logprob_helper_matches_dense_reference_with_headless_chunked_lm_head():
    model = _HeadlessChunkedKnownLogitsModel()
    input_ids = (jnp.arange(2 * 5).reshape(2, 5) % 7) + 1
    attention_mask = jnp.ones_like(input_ids)
    prompt_length = 2

    token_logps = get_per_token_logps(
        model,
        input_ids,
        attention_mask,
        prompt_length,
        logprob_vocab_chunk_size=3,
    )

    logits = model(input_ids=input_ids, attention_mask=attention_mask, apply_lm_head=True).logits
    logits = logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    targets = input_ids[:, prompt_length:]
    dense_log_probs = jax.nn.log_softmax(logits, axis=-1)
    expected = jnp.take_along_axis(dense_log_probs, targets[..., None], axis=-1)[..., 0]

    assert jnp.allclose(token_logps, expected, atol=1e-6)


def test_ppo_logprob_value_entropy_helper_matches_dense_reference_with_headless_chunked_lm_head():
    model = _HeadlessChunkedValueHeadModel()
    input_ids = (jnp.arange(2 * 5).reshape(2, 5) % 7) + 1
    attention_mask = jnp.ones_like(input_ids)
    prompt_length = 2

    token_logps, values, entropies = get_per_token_logps_values_entropies(
        model,
        input_ids,
        attention_mask,
        prompt_length,
        logprob_vocab_chunk_size=3,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        apply_lm_head=True,
        output_hidden_states=True,
    )
    logits = outputs.logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    targets = input_ids[:, prompt_length:]
    dense_log_probs = jax.nn.log_softmax(logits, axis=-1)
    expected_logps = jnp.take_along_axis(dense_log_probs, targets[..., None], axis=-1)[..., 0]
    expected_entropies = -jnp.sum(jnp.exp(dense_log_probs) * dense_log_probs, axis=-1)
    hidden_states = outputs.last_hidden_state
    expected_values = model.value_head(hidden_states).squeeze(-1)[:, prompt_length - 1 : -1]

    assert jnp.allclose(token_logps, expected_logps, atol=1e-6)
    assert jnp.allclose(values, expected_values, atol=1e-6)
    assert jnp.allclose(entropies, expected_entropies, atol=1e-6)


def test_distillation_preprocess_converts_and_drops_assistant_masks(monkeypatch):
    trainer = DistillationTrainer.__new__(DistillationTrainer)

    def _fake_base_preprocess(self, state, batch, is_train):
        del self, state, is_train
        return dict(batch), {}

    monkeypatch.setattr(Trainer, "_preprocess_batch_input", _fake_base_preprocess)

    batch = {
        "input_ids": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "attention_mask": jnp.array([[1, 1, 1]], dtype=jnp.int32),
        "assistant_masks": jnp.array([[0, 1, 1]], dtype=jnp.int32),
    }
    processed, _ = DistillationTrainer._preprocess_batch_input(
        trainer,
        state=None,
        batch=batch,
        is_train=True,
    )

    assert "assistant_masks" not in processed
    assert "completion_mask" in processed
    assert jnp.array_equal(processed["completion_mask"], batch["assistant_masks"])


def test_sft_preprocess_converts_and_drops_assistant_masks(monkeypatch):
    trainer = SFTTrainer.__new__(SFTTrainer)

    def _fake_base_preprocess(self, state, batch, is_train):
        del self, state, is_train
        return dict(batch), {}

    monkeypatch.setattr(Trainer, "_preprocess_batch_input", _fake_base_preprocess)

    batch = {
        "input_ids": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "attention_mask": jnp.array([[1, 1, 1]], dtype=jnp.int32),
        "assistant_masks": jnp.array([[0, 1, 1]], dtype=jnp.int32),
    }
    processed, _ = SFTTrainer._preprocess_batch_input(
        trainer,
        state=None,
        batch=batch,
        is_train=True,
    )

    assert "assistant_masks" not in processed
    assert "completion_mask" in processed
    assert jnp.array_equal(processed["completion_mask"], batch["assistant_masks"])


def test_sft_transform_uses_assistant_only_loss():
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.processing_class = SimpleNamespace(
        chat_template="plain chat template",
        pad_token_id=0,
        eos_token_id=2,
    )
    trainer.arguments = SFTConfig(max_length=16, assistant_only_loss=True)
    trainer._formatting_func = None
    trainer._dataset_text_field = None
    trainer._is_pretokenized = lambda: False

    transform = SFTTrainer._get_preprocess_transform(trainer)

    assert transform is not None
    assert transform._mask_prompt is True


def test_seq_kd_transform_passes_tools_to_chat_template():
    class _ToolAwareTokenizer:
        def __init__(self):
            self.chat_template = "tool-chat-template"
            self.calls = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
            del kwargs
            self.calls.append(
                {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "tools": tools,
                }
            )
            return "rendered prompt"

        def __call__(
            self,
            text,
            padding=False,
            max_length=None,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=True,
            **kwargs,
        ):
            del text, padding, max_length, truncation, add_special_tokens, kwargs
            result = {"input_ids": [1, 2, 3]}
            if return_attention_mask:
                result["attention_mask"] = [1, 1, 1]
            return result

    tokenizer = _ToolAwareTokenizer()
    trainer = SeqKDTrainer.__new__(SeqKDTrainer)
    trainer.processing_class = tokenizer
    trainer.arguments = SimpleNamespace(
        max_prompt_length=32,
        skip_apply_chat_template=False,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    trainer._is_pretokenized = lambda: False

    transform = SeqKDTrainer._get_preprocess_transform(trainer)

    assert transform is not None
    transformed = transform({"prompt": [{"role": "user", "content": "search"}]})

    assert transformed["input_ids"][-3:] == [1, 2, 3]
    assert transformed["attention_mask"][-3:] == [1, 1, 1]
    assert tokenizer.calls
    assert tokenizer.calls[-1]["tools"] == trainer.arguments.tools


def test_seq_kd_transform_uses_example_tools_when_present():
    class _ToolAwareTokenizer:
        def __init__(self):
            self.chat_template = "tool-chat-template"
            self.calls = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
            del kwargs
            self.calls.append(
                {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "tools": tools,
                }
            )
            return "rendered prompt"

        def __call__(
            self,
            text,
            padding=False,
            max_length=None,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=True,
            **kwargs,
        ):
            del text, padding, max_length, truncation, add_special_tokens, kwargs
            result = {"input_ids": [4, 5]}
            if return_attention_mask:
                result["attention_mask"] = [1, 1]
            return result

    tokenizer = _ToolAwareTokenizer()
    trainer = SeqKDTrainer.__new__(SeqKDTrainer)
    trainer.processing_class = tokenizer
    trainer.arguments = SimpleNamespace(
        max_prompt_length=16,
        skip_apply_chat_template=False,
        tools=None,
    )
    trainer._is_pretokenized = lambda: False

    transform = SeqKDTrainer._get_preprocess_transform(trainer)
    example_tools = [{"type": "function", "function": {"name": "search"}}]
    transformed = transform(
        {
            "prompt": [{"role": "user", "content": "search"}],
            "tools": example_tools,
        }
    )

    assert transformed["input_ids"][-2:] == [4, 5]
    assert transformed["attention_mask"][-2:] == [1, 1]
    assert tokenizer.calls
    assert tokenizer.calls[-1]["tools"] == example_tools


def test_seq_kd_teacher_fn_repeats_prompts_for_multi_generation():
    class _Tokenizer:
        def batch_decode(self, prompt_ids, skip_special_tokens=True):
            del skip_special_tokens
            return [f"prompt-{idx}" for idx in range(prompt_ids.shape[0])]

        def __call__(
            self,
            texts,
            padding="max_length",
            max_length=None,
            truncation=True,
            return_tensors="np",
            add_special_tokens=False,
        ):
            del padding, truncation, return_tensors, add_special_tokens
            rows = []
            masks = []
            for idx, _text in enumerate(texts):
                tokens = [idx + 1, idx + 11][: max_length or 2]
                pad_len = max((max_length or len(tokens)) - len(tokens), 0)
                rows.append(tokens + [0] * pad_len)
                masks.append([1] * len(tokens) + [0] * pad_len)
            return {"input_ids": rows, "attention_mask": masks}

    trainer = SeqKDTrainer.__new__(SeqKDTrainer)
    trainer.processing_class = _Tokenizer()
    trainer.arguments = SimpleNamespace(
        max_completion_length=4,
        generation_num_return_sequences=3,
        num_generations_per_prompt=3,
    )
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._make_attn_mask = lambda ids: (ids != 0).astype(jnp.int32)

    captured = {}

    def teacher_fn(prompts):
        captured["prompts"] = list(prompts)
        return [f"completion-{idx}" for idx, _ in enumerate(prompts)]

    trainer.teacher_fn = teacher_fn

    batch, _ = SeqKDTrainer._preprocess_batch_input(
        trainer,
        state=SimpleNamespace(),
        batch={
            "input_ids": jnp.asarray([[1, 2, 0], [3, 4, 0]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1, 0], [1, 1, 0]], dtype=jnp.int32),
        },
        is_train=True,
    )

    assert captured["prompts"] == [
        "prompt-0",
        "prompt-0",
        "prompt-0",
        "prompt-1",
        "prompt-1",
        "prompt-1",
    ]
    assert batch["input_ids"].shape[0] == 6
    assert batch["attention_mask"].shape[0] == 6
    assert batch["labels"].shape[0] == 6


def test_seq_kd_teacher_fn_validates_multi_generation_length():
    class _Tokenizer:
        def batch_decode(self, prompt_ids, skip_special_tokens=True):
            del skip_special_tokens
            return [f"prompt-{idx}" for idx in range(prompt_ids.shape[0])]

        def __call__(
            self,
            texts,
            padding="max_length",
            max_length=None,
            truncation=True,
            return_tensors="np",
            add_special_tokens=False,
        ):
            del texts, padding, max_length, truncation, return_tensors, add_special_tokens
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    trainer = SeqKDTrainer.__new__(SeqKDTrainer)
    trainer.processing_class = _Tokenizer()
    trainer.arguments = SimpleNamespace(
        max_completion_length=4,
        generation_num_return_sequences=2,
        num_generations_per_prompt=2,
    )
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._make_attn_mask = lambda ids: (ids != 0).astype(jnp.int32)
    trainer.teacher_fn = lambda prompts: ["only-one"] * (len(prompts) - 1)

    with pytest.raises(ValueError, match="must return exactly one completion per prompt"):
        SeqKDTrainer._preprocess_batch_input(
            trainer,
            state=SimpleNamespace(),
            batch={
                "input_ids": jnp.asarray([[1, 2, 0]], dtype=jnp.int32),
                "attention_mask": jnp.asarray([[1, 1, 0]], dtype=jnp.int32),
            },
            is_train=True,
        )
