"""Distillation loss numerics; accelerator-only."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.infra.loss_utils import LossConfig
from easydel.trainers.direct_preference_optimization_trainer._fn import get_loss_function
from easydel.trainers.distillation_trainer import _fn as kd
from easydel.trainers.generalized_knowledge_distillation_trainer import _fn as gkd
from jax import numpy as jnp


def _logsoftmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=-1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=-1, keepdims=True))


def test_dense_bf16_loss_reduces_and_mixes_in_float32():
    student = jnp.asarray([[[0.0, 1.0, -1.0], [2.0, -0.5, 0.5]]] * 129, dtype=jnp.bfloat16)
    teacher = jnp.asarray([[[1.0, -1.0, 0.0], [-0.5, 0.5, 2.0]]] * 129, dtype=jnp.bfloat16)
    labels = jnp.asarray([[1, 2]] * 129)
    mask = jnp.ones((129, 2), dtype=jnp.float32)
    loss, metrics = kd.distillation_loss(
        student,
        teacher,
        labels=labels,
        attention_mask=mask,
        use_hard_labels=True,
        alpha=0.9,
        temperature=1.7,
        shift_tokens=False,
    )
    s = np.asarray(student, dtype=np.float32)
    t = np.asarray(teacher, dtype=np.float32)
    log_s = _logsoftmax(s / 1.7)
    log_t = _logsoftmax(t / 1.7)
    expected_kl = np.mean(np.sum(np.exp(log_t) * (log_t - log_s), axis=-1)) * 1.7**2
    expected_ce = -np.take_along_axis(_logsoftmax(s), np.asarray(labels)[..., None], axis=-1).mean()
    assert loss.dtype == jnp.float32
    np.testing.assert_allclose(loss, 0.9 * expected_kl + 0.1 * expected_ce, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(metrics["ce_loss"], expected_ce, rtol=2e-5)


class Model:
    mesh = None

    def __init__(self, logits, aux_loss=None):
        self.logits = logits
        self.aux_loss = aux_loss

    def __call__(self, input_ids, attention_mask=None, apply_lm_head=True):
        return SimpleNamespace(logits=self.logits, last_hidden_state=self.logits, aux_loss=self.aux_loss)

    def make_lm_head_fn(self):
        return lambda hidden: hidden

    def eval(self):
        return self


class Call:
    schedule = None

    def __init__(self, **values):
        self.values = values
        self.bound_arguments = values
        self.batch = values.get("batch", {})

    def get(self, name, default=None):
        return self.values.get(name, default)


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_scheduled_distillation_includes_aux_and_z_loss(monkeypatch, chunk_size):
    logits = jnp.asarray([[[0.0, 1.0, 2.0], [2.0, -1.0, 0.0], [1.0, 0.0, -2.0]]])
    student = Model(logits, jnp.float32(0.375))
    teacher = Model(logits)
    teacher_state = SimpleNamespace(graphstate={}, merge=lambda tree: teacher)
    monkeypatch.setattr(kd, "bind_scheduled_module", lambda call, tree: student)
    monkeypatch.setattr(kd.spx, "get_current_stage_mesh", lambda *args, **kwargs: None)
    call = Call(
        teacher_state=teacher_state,
        temperature=1.0,
        alpha=1.0,
        logits_chunk_size=chunk_size,
        loss_config=LossConfig(z_loss=0.1),
    )
    batch = {"input_ids": jnp.asarray([[0, 1, 2]]), "attention_mask": jnp.asarray([[1, 1, 0]])}
    loss = kd._make_distillation_scheduled_loss(call)({}, batch)
    first = np.asarray(logits, dtype=np.float64)[0, 0]
    lse = np.log(np.exp(first).sum())
    np.testing.assert_allclose(loss, 0.375 + 0.1 * lse**2, rtol=2e-5, atol=2e-5)


def test_cached_gkd_loss_uses_current_batch_teacher(monkeypatch):
    student = Model(jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]]))
    monkeypatch.setattr(gkd, "bind_scheduled_module", lambda call, tree: student)
    monkeypatch.setattr(gkd, "_gkd_forward_logits", lambda model, batch: model.logits)
    monkeypatch.setattr(gkd.spx, "jit", lambda fn, **kwargs: fn)
    teacher = Model(student.logits)
    kwargs = dict(
        beta=0.0,
        temperature=1.0,
        loss_config=LossConfig(shift_tokens=False),
        teacher_state=SimpleNamespace(model=teacher),
        batch={"input_ids": jnp.asarray([[0, 1]])},
    )
    first_call = Call(**kwargs)
    first_batch = gkd._prepare_gkd_scheduled_batch(first_call)
    loss_fn = jax.jit(gkd._make_gkd_scheduled_loss(first_call))
    np.testing.assert_allclose(loss_fn({}, first_batch), 0.0, atol=1e-6)
    teacher.logits = -student.logits
    second_batch = gkd._prepare_gkd_scheduled_batch(Call(**kwargs))
    expected = np.sum(
        np.exp(_logsoftmax(np.asarray(-student.logits)))
        * (_logsoftmax(np.asarray(-student.logits)) - _logsoftmax(np.asarray(student.logits))),
        axis=-1,
    ).mean()
    np.testing.assert_allclose(loss_fn({}, second_batch), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("variant", ["sigmoid", "kto", "hinge"])
def test_dpo_compatibility_closure_preserves_objectives(variant):
    chosen = jnp.asarray([0.5, -0.5])
    rejected = jnp.asarray([-0.5, 0.5])
    zeros = jnp.zeros_like(chosen)
    loss = get_loss_function(variant, 0.2, 0.1)(chosen, rejected, zeros, zeros, 0.2, 0.1)
    z = np.asarray([0.2, -0.2])
    expected = np.maximum(1 - z, 0) if variant == "hinge" else 0.9 * np.logaddexp(0, -z) + 0.1 * np.logaddexp(0, z)
    np.testing.assert_allclose(loss, expected, rtol=1e-6)
