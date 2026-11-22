"""Tiny accelerator-only regressions; do not run these on CPU."""

import math
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers.direct_preference_optimization_trainer.dpo_trainer import DPOTrainer
from easydel.trainers.self_distillation_policy_optimization._fn import sdpo_step
from easydel.trainers.sparse_distillation_trainer._fn import partial_kl_distillation_loss
from easydel.trainers.tpo_trainer import TPOConfig, TPOTrainer
from jax import numpy as jnp


def test_sparse_distillation_weights_next_token_completion_mask():
    logits = jnp.asarray([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-5.0, 5.0]]])
    indices = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    logprobs = jnp.zeros((1, 4, 1))
    mask = jnp.asarray([[0, 0, 1, 1]])
    loss, _ = partial_kl_distillation_loss(logits, indices, logprobs, loss_mask=mask, temperature=1.0, alpha=1.0)
    expected = (math.log1p(math.exp(-2.0)) + math.log1p(math.exp(2.0))) / 2
    assert float(loss) == pytest.approx(expected, abs=1e-5)


def test_sparse_hard_labels_use_same_next_token_alignment():
    logits = jnp.asarray([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-5.0, 5.0]]])
    loss, metrics = partial_kl_distillation_loss(
        logits,
        jnp.zeros((1, 4, 1), dtype=jnp.int32),
        jnp.zeros((1, 4, 1)),
        loss_mask=jnp.asarray([[0, 0, 1, 1]]),
        labels=jnp.asarray([[-100, -100, 1, 0]]),
        use_hard_labels=True,
        temperature=1.0,
        alpha=0.0,
    )
    expected = math.log1p(math.exp(2.0))
    assert float(loss) == pytest.approx(expected, abs=1e-5)
    assert float(metrics["ce_loss"]) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_skipped_completion_tokens_remain_visible_to_student(chunk_size):
    class CumulativeContextModel:
        mesh = None
        config = SimpleNamespace(lmhead_chunksize=None)

        def __call__(self, input_ids, attention_mask, **kwargs):
            context = jnp.cumsum(input_ids * attention_mask, axis=1).astype(jnp.float32)
            return SimpleNamespace(logits=jnp.stack([jnp.zeros_like(context), context], axis=-1))

    model = CumulativeContextModel()
    state = SimpleNamespace(model=model, graphstate={}, merge=lambda tree: model)
    batch = {
        "prompt_ids": jnp.ones((2, 1), dtype=jnp.int32),
        "prompt_mask": jnp.ones((2, 1), dtype=jnp.int32),
        "completion_ids": jnp.ones((2, 3), dtype=jnp.int32),
        "completion_mask": jnp.ones((2, 3), dtype=jnp.int32),
        "teacher_ids": jnp.ones((2, 5), dtype=jnp.int32),
        "teacher_mask": jnp.ones((2, 5), dtype=jnp.int32),
        "num_items_in_batch": jnp.asarray(6.0),
    }
    metrics = sdpo_step(
        state,
        batch,
        num_generations=1,
        teacher_prompt_length=2,
        beta=0.0,
        distillation_type="kl",
        distillation_weight=1.0,
        logprob_vocab_chunk_size=None,
        max_loss_completion_tokens=None,
        completion_chunk_size=chunk_size,
        is_training=False,
        num_loss_tokens_to_skip=1,
    )
    expected_student_logps = -(math.log1p(math.exp(-2.0)) + math.log1p(math.exp(-3.0))) / 2
    assert float(metrics.other_metrics["sdpo/student_logps"]) == pytest.approx(expected_student_logps, abs=1e-5)


@pytest.mark.parametrize("backend", ["grain", "tfds"])
def test_tpo_loader_captures_reference_collator_during_parent_initialization(monkeypatch, backend):
    captured = {}

    def initialize(self, *, arguments, **kwargs):
        self.arguments = arguments
        self.padding_value = 0
        self.input_data_collator_grain = object()
        self.input_data_collator_tfds = object()
        captured["collate"] = getattr(self, f"create_{backend}_collect_function")(arguments.max_length)

    monkeypatch.setattr(DPOTrainer, "__init__", initialize)
    config = TPOConfig(max_prompt_length=2, max_completion_length=2)
    TPOTrainer(config, model=object())
    row = {
        "prompt_input_ids": [1],
        "prompt_attention_mask": [1],
        **{f"{branch}_input_ids": [1, token] for branch, token in [("chosen", 2), ("rejected", 3), ("reference", 4)]},
        **{f"{branch}_attention_mask": [1, 1] for branch in ["chosen", "rejected", "reference"]},
        **{f"{branch}_labels": [-100, token] for branch, token in [("chosen", 2), ("rejected", 3), ("reference", 4)]},
    }
    output = captured["collate"](row if backend == "grain" else [row])
    assert "reference_input_ids" in output
    np.testing.assert_array_equal(np.asarray(jax.device_get(output["reference_input_ids"])).reshape(-1)[:1], [4])
    assert np.asarray(jax.device_get(output["reference_attention_mask"])).sum() == 1
