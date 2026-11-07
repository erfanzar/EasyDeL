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

"""Trainer rollouts send pretokenized prompts to eSurge as token ids (accelerator test).

A tiny float32 Llama is served through ``BaseTrainer.rollout``. The prompt ids are random vocabulary ids, which in
general do not survive a decode/re-encode round trip, so the greedy parity check fails if the engine conditions on
anything but the caller's exact ids.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.trainers.trainer.trainer import Trainer
from easydel.trainers.training_configurations import TrainingArguments
from jax import numpy as jnp

from tests.inference.esurge.runtime_pass._tiny_runtime_common import (
    assert_cached_decode_matches_uncached_forward,
    load_test_tokenizer,
)

MAX_LENGTH = 64
MAX_NEW_TOKENS = 8


def _tiny_llama(tokenizer):
    import easydel as ed

    config = ed.LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128,
        kvdtype=jnp.float32,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sharding_axis_dims=(1, 1, 1, 1, jax.device_count(), 1),
    )
    with config.mesh:
        return ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tokenizer = load_test_tokenizer()
    model = _tiny_llama(tokenizer)
    trainer = object.__new__(Trainer)
    trainer.arguments = TrainingArguments(
        save_directory=str(tmp_path_factory.mktemp("rollout")),
        max_length=MAX_LENGTH,
        total_batch_size=2,
        generation_max_new_tokens=MAX_NEW_TOKENS,
        generation_do_sample=False,
        generation_num_return_sequences=1,
        esurge_max_num_seqs=2,
        esurge_page_size=16,
        esurge_max_cache_tokens=1024,
        esurge_max_num_batched_tokens=64,
        esurge_silent_mode=True,
        esurge_async_scheduling=False,
        esurge_overlap_execution=False,
    )
    trainer.processing_class = tokenizer
    return trainer, model, model.to_state(), tokenizer


def _prompt_batch(tokenizer, width: int, lengths: list[int], seed: int):
    rng = np.random.default_rng(seed)
    ids = np.full((len(lengths), width), tokenizer.pad_token_id, dtype=np.int32)
    mask = np.zeros((len(lengths), width), dtype=np.int32)
    for row, length in enumerate(lengths):
        ids[row, width - length :] = rng.integers(100, 5000, size=length)
        mask[row, width - length :] = 1
    return ids, mask


def test_token_id_rollout_keeps_prompt_rows_and_matches_greedy_reference(setup):
    trainer, model, state, tokenizer = setup
    input_ids, attention_mask = _prompt_batch(tokenizer, width=12, lengths=[12, 7], seed=0)

    results = trainer.rollout(input_ids=input_ids, attention_mask=attention_mask, state=state)

    np.testing.assert_array_equal(np.asarray(results.prompt_ids), input_ids)
    np.testing.assert_array_equal(np.asarray(results.prompt_mask), attention_mask)
    sequences = np.asarray(results.sequences)
    np.testing.assert_array_equal(sequences[:, : input_ids.shape[1]], input_ids)
    completion_ids = np.asarray(results.completion_ids)
    completion_mask = np.asarray(results.completion_mask).astype(bool)
    assert completion_ids.shape == (2, MAX_NEW_TOKENS)
    np.testing.assert_array_equal(sequences[:, input_ids.shape[1] :], completion_ids)
    for row in range(2):
        prompt = input_ids[row][attention_mask[row].astype(bool)].tolist()
        generated = completion_ids[row][completion_mask[row]].tolist()
        assert generated, "the engine generated no token"
        assert_cached_decode_matches_uncached_forward(model, prompt, generated)


def test_rollouts_with_different_prompt_widths_and_budgets_share_one_engine(setup):
    trainer, _, state, tokenizer = setup
    for width, budget, seed in ((12, None, 1), (20, 4, 2)):
        input_ids, attention_mask = _prompt_batch(tokenizer, width=width, lengths=[width, width - 3], seed=seed)
        overrides = None if budget is None else {"max_new_tokens": budget}
        results = trainer.rollout(
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            config_overrides=overrides,
        )
        assert np.asarray(results.completion_ids).shape[1] == (budget or MAX_NEW_TOKENS)
    assert len(state.model.list_esurge_engines()) == 1
