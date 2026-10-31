"""Accelerator regressions using the real configured SDPO/SSD steps."""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers.self_distillation_policy_optimization import SDPOTrainer
from easydel.trainers.self_distillation_policy_optimization import sdpo_trainer as sdpo_module
from easydel.trainers.self_distillation_policy_optimization._fn import sdpo_step
from easydel.trainers.ssd_trainer import SSDTrainer
from easydel.trainers.ssd_trainer import ssd_trainer as ssd_module
from jax import numpy as jnp


class _ContextModel:
    mesh = None

    def __init__(self, weight, context):
        self.weight = weight
        self.config = SimpleNamespace(lmhead_chunksize=None, max_position_embeddings=context)

    def __call__(self, input_ids, attention_mask, **kwargs):
        context = jnp.cumsum(input_ids * attention_mask, axis=1).astype(jnp.float32)
        return SimpleNamespace(logits=jnp.stack([jnp.zeros_like(context), self.weight * context], axis=-1))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _State:
    weight: jax.Array
    step: jax.Array
    context: int = 32

    @property
    def graphstate(self):
        return {"weight": self.weight}

    @property
    def model(self):
        return self.merge(self.graphstate)

    graphdef = None
    shardings = SimpleNamespace(graphstate=None, graphother=None)

    def merge(self, tree):
        return _ContextModel(tree["weight"], self.context)

    def apply_gradients(self, *, grads):
        return _State(self.weight - 0.1 * grads["weight"], self.step + 1, self.context)

    def tree_flatten(self):
        return (self.weight, self.step), self.context

    @classmethod
    def tree_unflatten(cls, context, children):
        return cls(*children, context=context)


def _arguments(**overrides):
    values = dict(
        max_prompt_length=4,
        max_completion_length=3,
        max_feedback_length=8,
        max_reprompt_len=6,
        max_length=32,
        beta=0.0,
        distillation_type="kl",
        distillation_weight=1.0,
        logprob_vocab_chunk_size=None,
        max_loss_completion_tokens=None,
        completion_chunk_size=None,
        loss_config=None,
        step_partition_spec=None,
        gradient_accumulation_steps=1,
        quantization_mode=None,
        quantization_group_size=None,
        quantization_bits=None,
        tensor_straight_through=None,
        straight_through_emulator=None,
        full_logit_distillation=False,
        distillation_topk=None,
        distillation_is_clip=None,
        distillation_add_tail=False,
        mpmd_scheduler=None,
        ensure_checkpoint_path=lambda: None,
        get_streaming_checkpointer=lambda: None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _trainer(trainer_cls, arguments, context=10):
    trainer = object.__new__(trainer_cls)
    trainer.arguments = arguments
    trainer.model_state = _State(jnp.asarray(1.0), jnp.asarray(0), context)
    trainer._model = trainer.model_state.model
    trainer.state_shardings = None
    trainer.scheduler = None
    trainer.num_generations = 1
    trainer._runtime_trace = lambda *args, **kwargs: None
    return trainer


def _configure_without_compiling(monkeypatch, module):
    # Only replace the compilation boundary: execute the actual production steps,
    # including shape alignment, autodiff, minibatching, QAT and state updates.
    monkeypatch.setattr(module, "replicated_named_sharding", lambda mesh: None)

    def compile_wrapper(fn, **kwargs):
        def configured(*args, **call_kwargs):
            return fn(*args, **call_kwargs)

        return configured

    monkeypatch.setattr(module, "compile_trainer_step", compile_wrapper)


@pytest.mark.parametrize("context, cap, expected", [(10, 6, 7), (32, 2, 6)])
def test_sdpo_configured_step_uses_capped_teacher_prefix(monkeypatch, context, cap, expected):
    _configure_without_compiling(monkeypatch, sdpo_module)
    trainer = _trainer(SDPOTrainer, _arguments(max_reprompt_len=cap), context=context)
    trainer.teacher_prompt_length = 12  # Uncapped constructor value.
    configured = trainer.configure_functions()
    batch = {
        "prompt_ids": jnp.ones((4, 4), dtype=jnp.int32),
        "prompt_mask": jnp.ones((4, 4), dtype=jnp.int32),
        "completion_ids": jnp.ones((4, 3), dtype=jnp.int32),
        "completion_mask": jnp.ones((4, 3), dtype=jnp.int32),
        "teacher_ids": jnp.ones((4, expected + 3), dtype=jnp.int32),
        "teacher_mask": jnp.ones((4, expected + 3), dtype=jnp.int32),
        "num_items_in_batch": jnp.asarray(12.0),
    }
    state, train_metrics = configured.sharded_training_step_function(
        trainer.model_state, batch, *trainer._train_shared_fn_static_args
    )
    eval_metrics = configured.sharded_evaluation_step_function(
        trainer.model_state, batch, *trainer._eval_shared_fn_static_args
    )
    student = -np.logaddexp(0, -np.arange(4, 7, dtype=np.float64))
    teacher = -np.logaddexp(0, -np.arange(expected, expected + 3, dtype=np.float64))
    expected_loss = np.mean((student - teacher) * student)
    expected_grad = np.mean((student - teacher) * np.arange(4, 7) / (1 + np.exp(np.arange(4, 7))))
    for metrics in [train_metrics, eval_metrics]:
        assert float(metrics.loss) == pytest.approx(expected_loss, abs=2e-6)
        assert float(metrics.other_metrics["sdpo/teacher_logps"]) == pytest.approx(teacher.mean(), abs=2e-6)
    assert float(state.weight) == pytest.approx(1 - 0.1 * expected_grad, abs=2e-6)


def test_ssd_configured_train_step_receives_explicit_emulator(monkeypatch):
    _configure_without_compiling(monkeypatch, ssd_module)

    def emulator(tree):
        return {"weight": tree["weight"] + 1.0}

    trainer = _trainer(SSDTrainer, _arguments(straight_through_emulator=emulator))
    configured = trainer.configure_functions()
    batch = {
        "prompt_ids": jnp.ones((4, 2), dtype=jnp.int32),
        "prompt_mask": jnp.ones((4, 2), dtype=jnp.int32),
        "completion_ids": jnp.ones((4, 3), dtype=jnp.int32),
        "completion_mask": jnp.ones((4, 3), dtype=jnp.int32),
    }
    state, train_metrics = configured.sharded_training_step_function(
        trainer.model_state, batch, *trainer._train_shared_fn_static_args
    )
    eval_metrics = configured.sharded_evaluation_step_function(
        trainer.model_state, batch, *trainer._eval_shared_fn_static_args
    )
    assert float(train_metrics.loss) == pytest.approx(sum(math.log1p(math.exp(-2 * x)) for x in [2, 3, 4]) / 3, abs=2e-6)
    assert float(eval_metrics.loss) == pytest.approx(sum(math.log1p(math.exp(-x)) for x in [2, 3, 4]) / 3, abs=2e-6)
    expected_grad = -sum(x / (1 + math.exp(2 * x)) for x in [2, 3, 4]) / 3
    assert float(state.weight) == pytest.approx(1 - 0.1 * expected_grad, abs=2e-6)


@pytest.mark.parametrize("chunk_size, full_vocab", [(None, False), (1, False), (None, True)])
def test_sdpo_multigeneration_accumulation_matches_independent_update(chunk_size, full_vocab):
    prompts = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int32)
    batch = {
        "prompt_ids": jnp.asarray(prompts),
        "prompt_mask": jnp.ones((4, 2), dtype=jnp.int32),
        "completion_ids": jnp.ones((8, 3), dtype=jnp.int32),
        "completion_mask": jnp.ones((8, 3), dtype=jnp.int32),
        "teacher_ids": jnp.ones((8, 7), dtype=jnp.int32),
        "teacher_mask": jnp.ones((8, 7), dtype=jnp.int32),
        "num_items_in_batch": jnp.asarray(24.0),
    }
    contexts = np.repeat(prompts.sum(axis=1), 2)[:, None] + np.arange(3)[None, :]
    teacher_contexts = np.arange(4, 7)[None, :]
    if full_vocab:
        expected_grad = np.mean((1 / (1 + np.exp(-contexts)) - 1 / (1 + np.exp(-teacher_contexts))) * contexts)
    else:
        student = -np.logaddexp(0, -contexts)
        teacher = -np.logaddexp(0, -teacher_contexts)
        expected_grad = np.mean((student - teacher) * contexts / (1 + np.exp(contexts)))
    states = []
    for accumulation in [1, 2]:
        state, _ = sdpo_step(
            _State(jnp.asarray(1.0), jnp.asarray(0)),
            batch,
            num_generations=2,
            teacher_prompt_length=4,
            beta=0.0,
            distillation_type="kl",
            distillation_weight=1.0,
            logprob_vocab_chunk_size=None,
            max_loss_completion_tokens=None,
            completion_chunk_size=chunk_size,
            gradient_accumulation_steps=accumulation,
            full_logit_distillation=full_vocab,
        )
        assert float(state.weight) == pytest.approx(1 - 0.1 * expected_grad, abs=2e-6)
        states.append(state)
    np.testing.assert_allclose(states[0].weight, states[1].weight, atol=2e-6, rtol=0)
