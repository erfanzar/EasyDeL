"""Tiny accelerator-only BCO scheduled-QAT and pipeline-forward regressions.

Do not run these tests on CPU; they execute JAX array computation.
"""

import importlib
import math
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers.binary_classifier_optimization_trainer import _fn
from easydel.trainers.binary_classifier_optimization_trainer.bco_trainer import BCOTrainer
from jax import numpy as jnp


class ScheduledCall:
    def __init__(self, batch, **kwargs):
        self.batch = batch
        self.schedule = None
        self.kwargs = kwargs

    def get(self, key, default=None):
        return self.kwargs.get(key, default)


def test_scheduled_loss_emulates_weights_and_reports_reward_moments(monkeypatch):
    batch = {
        "label": jnp.asarray([True, False]),
        "reference_logps": jnp.zeros(2),
        "running_mean": jnp.asarray(0.0),
    }

    def forward(weight, call_batch, vocab_shard_stage=None):
        return {"completion_logps": weight * jnp.asarray([1.0, -1.0])}

    def emulator(weight):
        return weight + jax.lax.stop_gradient(0.8 - weight)

    monkeypatch.setattr(_fn, "bind_scheduled_module", lambda call, tree: tree)
    monkeypatch.setattr(_fn, "constrain_scheduled_batch", lambda module, value, spec: value)
    monkeypatch.setattr(_fn, "_scheduled_terminal_stage_rank", lambda module, schedule: None)
    call = ScheduledCall(batch, beta=0.5, concatenated_forward_fn=forward, straight_through_emulator=emulator)
    loss_fn = _fn._make_bco_scheduled_loss(call)
    (loss, moments), gradient = jax.value_and_grad(loss_fn, has_aux=True)(jnp.asarray(0.4), batch)
    # Both labels receive a positive signed reward of 0.4 after emulation.
    assert float(loss) == pytest.approx(math.log1p(math.exp(-0.4)), rel=1e-5)
    assert float(gradient) == pytest.approx(-0.5 / (1.0 + math.exp(0.4)), rel=1e-5)
    assert float(moments["reward_moments/count"]) == pytest.approx(2.0)
    assert float(moments["reward_moments/sum"]) == pytest.approx(0.0, abs=1e-6)
    assert float(moments["reward_moments/sumsq"]) == pytest.approx(0.32, rel=1e-5)


def test_configured_forward_accepts_terminal_stage_and_scores_tokens(monkeypatch):
    trainer_module = importlib.import_module("easydel.trainers.binary_classifier_optimization_trainer.bco_trainer")

    class UniformModel:
        mesh = None
        output_router_logits = False

        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(logits=jnp.zeros((*input_ids.shape, 4)))

    model = UniformModel()
    trainer = object.__new__(BCOTrainer)
    trainer.model_state = SimpleNamespace(model=model)
    trainer._model = model  # BaseTrainer initializes the cached model during construction.
    trainer.reference_state = None
    trainer.state_shardings = None
    trainer.scheduler = None
    trainer.padding_value = 0
    trainer._runtime_trace = lambda *args, **kwargs: None
    trainer.arguments = SimpleNamespace(
        quantization_mode=None,
        quantization_group_size=64,
        quantization_bits=8,
        tensor_straight_through=False,
        straight_through_emulator=None,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        max_length=4,
        truncation_mode="keep_end",
        logprob_vocab_chunk_size=None,
        beta=0.5,
        loss_config=None,
        step_partition_spec=None,
        gradient_accumulation_steps=1,
        mpmd_scheduler=None,
        get_streaming_checkpointer=lambda: None,
    )
    monkeypatch.setattr(trainer_module, "replicated_named_sharding", lambda mesh: None)
    monkeypatch.setattr(trainer_module, "resolve_straight_through_emulator", lambda **kwargs: None)
    monkeypatch.setattr(trainer_module, "compile_trainer_auxiliary", lambda fn, **kwargs: fn)
    monkeypatch.setattr(trainer_module, "compile_trainer_step", lambda fn, **kwargs: fn)
    trainer.configure_functions()
    batch = {
        "prompt_input_ids": jnp.asarray([[1]]),
        "prompt_attention_mask": jnp.ones((1, 1)),
        "completion_input_ids": jnp.asarray([[1, 2, 3]]),
        "completion_attention_mask": jnp.ones((1, 3)),
        "completion_labels": jnp.asarray([[-100, 2, 3]]),
    }
    outputs = trainer.concatenated_forward(model, batch, vocab_shard_stage=0)
    np.testing.assert_allclose(outputs["completion_logps"], [-2.0 * math.log(4)], rtol=1e-5)


def test_encoder_decoder_scores_only_unmasked_completion_targets():
    class UniformEncoderDecoder:
        def __call__(self, input_ids, attention_mask, labels):
            return SimpleNamespace(logits=jnp.zeros((*labels.shape, 4), dtype=jnp.float32))

    batch = {
        "prompt_input_ids": jnp.asarray([[1]]),
        "prompt_attention_mask": jnp.asarray([[1]]),
        "completion_input_ids": jnp.asarray([[1, 2, 3, 0]]),
        "completion_attention_mask": jnp.asarray([[1, 1, 1, 0]]),
        "completion_labels": jnp.asarray([[-100, 2, 3, -100]]),
    }
    outputs = _fn.concatenated_forward(
        UniformEncoderDecoder(),
        batch,
        is_encoder_decoder=True,
        label_pad_token_id=-100,
        padding_value=0,
        max_length=4,
    )
    np.testing.assert_allclose(outputs["completion_logps"], [-2.0 * math.log(4)], rtol=1e-5)
