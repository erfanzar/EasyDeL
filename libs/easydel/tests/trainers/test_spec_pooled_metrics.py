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

"""Speculative-decoding metrics pool ratios over gradient-accumulation microbatches."""

from __future__ import annotations

from types import SimpleNamespace

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers import SpeculativeDecodingConfig, SpeculativeDecodingTrainer
from easydel.trainers.speculative_decoding_trainer._fn import (
    _deepspec_block_loss,
    _finalize_pooled_metrics,
    _loss_mask_from_batch,
    speculative_decoding_step,
    speculative_draft_loss,
)
from easydel.trainers.training_utils import minibatch_call

_BIG = 30.0


def _one_hot_logits(ids: np.ndarray, vocab: int) -> np.ndarray:
    return np.where(np.eye(vocab, dtype=np.float32)[ids] > 0, _BIG, -_BIG).astype(np.float32)


def _block_batch():
    """Four rows of one 2-slot block; rows 0-1 form microbatch 0, rows 2-3 microbatch 1.

    Slot 1 always matches the verifier. Slot 2 is supervised on rows 0, 2, 3 and
    matches only on row 0, so the pooled slot-2 rate is 1/3 while the mean of the
    per-microbatch rates is (1/1 + 0/2) / 2 = 1/2.
    """
    vocab = 3
    verifier = np.array([[[0, 1]], [[1, 2]], [[2, 0]], [[0, 2]]], dtype=np.int32)
    draft = verifier.copy()
    draft[2, 0, 1] = (verifier[2, 0, 1] + 1) % vocab
    draft[3, 0, 1] = (verifier[3, 0, 1] + 1) % vocab
    eval_mask = np.array([[[1, 1]], [[1, 0]], [[1, 1]], [[1, 1]]], dtype=np.float32)
    return {
        "block_logits": jnp.asarray(_one_hot_logits(draft, vocab)),
        "aligned_target_logits": jnp.asarray(_one_hot_logits(verifier, vocab)),
        "target_ids": jnp.asarray(verifier),
        "eval_mask": jnp.asarray(eval_mask),
        "block_keep_mask": jnp.ones((4, 1), dtype=jnp.bool_),
    }


_BLOCK_CONFIG = SimpleNamespace(ce_loss_alpha=1.0, l1_loss_alpha=0.5, confidence_head_alpha=0.0, loss_decay_gamma=None)


def _block_loss_fn(tree, minibatch):
    outputs = SimpleNamespace(**minibatch)
    outputs.block_logits = outputs.block_logits + tree["w"]
    return _deepspec_block_loss(outputs, _BLOCK_CONFIG)


def _run_ga(batch, minibatch_size):
    state = SimpleNamespace(graphstate={"w": jnp.zeros((), jnp.float32)})
    _, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(_block_loss_fn, has_aux=True),
    )
    return _finalize_pooled_metrics(metrics)


def test_ga2_block_metrics_are_pooled_not_microbatch_means():
    metrics = _run_ga(_block_batch(), minibatch_size=2)
    other = metrics.other_metrics
    assert not [key for key in other if key.startswith("_pooled")]
    np.testing.assert_allclose(float(other["draft_step2_accept_rate"]), 1.0 / 3.0, atol=1e-6)
    np.testing.assert_allclose(float(other["greedy_conditional_step2"]), 1.0 / 3.0, atol=1e-6)
    np.testing.assert_allclose(float(other["greedy_survival_step2"]), 1.0 / 3.0, atol=1e-6)
    np.testing.assert_allclose(float(other["greedy_conditional_step1"]), 1.0, atol=1e-6)
    # Supervised slots: 2 + 1 + 2 + 2 = 7, of which 5 match.
    np.testing.assert_allclose(float(other["greedy_top1_match"]), 5.0 / 7.0, atol=1e-6)
    np.testing.assert_allclose(float(other["draft_accept_rate"]), 5.0 / 7.0, atol=1e-6)
    np.testing.assert_allclose(float(metrics.accuracy), 5.0 / 7.0, atol=1e-6)
    # Accepted lengths 2, 1, 1, 1 -> tau = mean(len + 1).
    np.testing.assert_allclose(float(other["tau_greedy"]), 9.0 / 4.0, atol=1e-6)
    np.testing.assert_allclose(float(other["tau_probabilistic"]), 9.0 / 4.0, atol=1e-5)
    np.testing.assert_allclose(float(metrics.weight_sum), 7.0)


def test_ga2_pooled_equals_ga1_and_differs_from_microbatch_mean():
    batch = _block_batch()
    pooled = _run_ga(batch, minibatch_size=2).other_metrics
    single = _run_ga(batch, minibatch_size=4).other_metrics
    halves = [
        _run_ga({k: v[rows] for k, v in batch.items()}, minibatch_size=2).other_metrics
        for rows in (slice(0, 2), slice(2, 4))
    ]
    mean_of_ratios = 0.5 * sum(float(h["draft_step2_accept_rate"]) for h in halves)
    np.testing.assert_allclose(mean_of_ratios, 0.5, atol=1e-6)
    for key in ("draft_step2_accept_rate", "greedy_conditional_step2", "greedy_top1_match", "tau_greedy"):
        np.testing.assert_allclose(float(pooled[key]), float(single[key]), atol=1e-6)


def test_empty_step_denominator_is_nan():
    batch = _block_batch()
    batch["eval_mask"] = batch["eval_mask"].at[:, :, 1].set(0.0)
    other = _run_ga(batch, minibatch_size=2).other_metrics
    assert np.isnan(float(other["draft_step2_accept_rate"]))
    assert np.isnan(float(other["greedy_conditional_step2"]))
    np.testing.assert_allclose(float(other["draft_step1_accept_rate"]), 1.0, atol=1e-6)


def test_loss_mask_falls_back_to_all_ones_for_input_ids_only():
    input_ids = jnp.arange(6, dtype=jnp.int32).reshape(2, 3)
    mask = _loss_mask_from_batch({"input_ids": input_ids})
    np.testing.assert_array_equal(np.asarray(mask), np.ones((2, 3), np.float32))
    assert _loss_mask_from_batch({}) is None


def _tiny_llama(seed: int, *, tie: bool = False):
    cfg = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        mlp_bias=False,
        tie_word_embeddings=tie,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )
    model = ed.LlamaForCausalLM(config=cfg, rngs=spx.Rngs(seed), dtype=jnp.float32, param_dtype=jnp.float32)
    return cfg, model


def _lm_batch():
    rng = np.random.default_rng(0)
    batch_size, seq_len = 4, 8
    completion_mask = np.ones((batch_size, seq_len), np.int32)
    completion_mask[2:, :-2] = 0
    return {
        "input_ids": jnp.asarray(rng.integers(0, 64, (batch_size, seq_len)), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "completion_mask": jnp.asarray(completion_mask),
    }


def test_spec_step_pools_over_ga_and_matches_unsharded_reference():
    cfg, drafter = _tiny_llama(0)
    _, target = _tiny_llama(1)
    batch = _lm_batch()
    with jax.default_matmul_precision("highest"), cfg.mesh:
        drafter_state = EasyDeLState.create(model=drafter, tx=optax.sgd(0.0), init_opt_state=True)
        target_state = target.to_state()
        lr_fn = optax.constant_schedule(0.0)

        @ed.ejit(static_argnums=(3, 4, 5, 6))  # pyright: ignore[reportUntypedFunctionDecorator]
        def step(ds, b, ts, lr, ps, ga, train):
            return speculative_decoding_step(ds, b, ts, None, lr, ps, ga, train)

        eval_metrics = step(drafter_state, batch, target_state, lr_fn, None, 1, False)
        _, ga1 = step(drafter_state, batch, target_state, lr_fn, None, 1, True)
        _, ga2 = step(drafter_state, batch, target_state, lr_fn, None, 2, True)
        halves = [
            step(drafter_state, {k: v[rows] for k, v in batch.items()}, target_state, lr_fn, None, 1, False)
            for rows in (slice(0, 2), slice(2, 4))
        ]

        draft_logits = drafter(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        target_logits = target(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        ref_loss, ref_metrics = speculative_draft_loss(
            draft_logits,
            target_logits,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["completion_mask"],
        )

    ref_rate = float(ref_metrics.other_metrics["draft_accept_rate"])
    np.testing.assert_allclose(float(eval_metrics.loss), float(ref_loss), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(eval_metrics.other_metrics["draft_accept_rate"]), ref_rate, rtol=1e-4)
    for metrics in (eval_metrics, ga1, ga2):
        assert not [key for key in metrics.other_metrics if key.startswith("_pooled")]
    for key in ("draft_accept_rate", "target_prob_of_draft", "draft_step1_acceptance_match"):
        np.testing.assert_allclose(float(ga2.other_metrics[key]), float(ga1.other_metrics[key]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(ga2.accuracy), float(ga1.accuracy), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(ga2.weight_sum), 18.0)

    half_rates = [float(h.other_metrics["draft_accept_rate"]) for h in halves]
    assert abs(half_rates[0] - half_rates[1]) > 1e-3, half_rates
    mean_of_ratios = 0.5 * (half_rates[0] + half_rates[1])
    assert abs(mean_of_ratios - float(ga2.other_metrics["draft_accept_rate"])) > 1e-4


def test_pre_split_drafter_state_must_match_selector(tmp_path):
    cfg = ed.DSparkConfig(
        vocab_size=32,
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        target_layer_ids=(-1, 0),
        block_size=3,
        num_anchors=2,
        mask_token_id=31,
        markov_rank=4,
        freeze_embeddings_and_lm_head=True,
    )
    model = ed.DSparkModel(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    args = SpeculativeDecodingConfig(save_directory=str(tmp_path / "spec"))
    expected = SpeculativeDecodingTrainer._expected_drafter_selector(args, model)

    SpeculativeDecodingTrainer._validate_pre_split_drafter_state(
        drafter_state=model.to_state(), trainable_selector=expected
    )
    with pytest.raises(ValueError, match="does not match `trainable_selector`"):
        SpeculativeDecodingTrainer._validate_pre_split_drafter_state(
            drafter_state=model.to_state(trainable_selector="parameters"), trainable_selector=expected
        )
