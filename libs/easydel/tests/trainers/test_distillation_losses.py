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
    def _student_forward(model, batch):
        del model
        x = batch["input_ids"].astype(jnp.float32)
        return jnp.stack((x, -x), axis=-1), None

    def _teacher_forward(model, batch):
        del model
        x = batch["input_ids"].astype(jnp.float32)
        return jnp.stack((x * 2.0 - 1.0, 1.0 - x * 2.0), axis=-1)

    student = Model(jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]]))
    teacher = Model(student.logits)
    teacher_state = SimpleNamespace(model=teacher, graphstate={}, merge=lambda tree: teacher)
    monkeypatch.setattr(gkd, "bind_scheduled_module", lambda call, tree: student)
    monkeypatch.setattr(gkd, "_gkd_forward_outputs", _student_forward)
    monkeypatch.setattr(gkd, "_gkd_forward_logits", _teacher_forward)

    first_batch_in = {"input_ids": jnp.asarray([[1, 0]], dtype=jnp.int32)}
    second_batch_in = {"input_ids": jnp.asarray([[0, 1]], dtype=jnp.int32)}
    kwargs = dict(
        beta=0.0,
        temperature=1.0,
        loss_config=LossConfig(shift_tokens=False),
        teacher_state=teacher_state,
        batch=first_batch_in,
    )
    first_call = Call(**kwargs)
    first_batch = gkd._prepare_gkd_scheduled_batch(first_call)
    assert "teacher_logits" not in first_batch
    loss_fn = jax.jit(gkd._make_gkd_scheduled_loss(first_call))
    s1, _ = _student_forward(student, first_batch_in)
    t1 = _teacher_forward(teacher, first_batch_in)
    expected1 = np.sum(
        np.exp(_logsoftmax(np.asarray(t1))) * (_logsoftmax(np.asarray(t1)) - _logsoftmax(np.asarray(s1))),
        axis=-1,
    ).mean()
    np.testing.assert_allclose(loss_fn({}, first_batch), expected1, rtol=2e-5, atol=2e-5)

    second_batch = gkd._prepare_gkd_scheduled_batch(Call(**{**kwargs, "batch": second_batch_in}))
    s2, _ = _student_forward(student, second_batch_in)
    t2 = _teacher_forward(teacher, second_batch_in)
    expected2 = np.sum(
        np.exp(_logsoftmax(np.asarray(t2))) * (_logsoftmax(np.asarray(t2)) - _logsoftmax(np.asarray(s2))),
        axis=-1,
    ).mean()
    np.testing.assert_allclose(loss_fn({}, second_batch), expected2, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("variant", ["sigmoid", "hinge"])
def test_dpo_compatibility_closure_preserves_objectives(variant):
    chosen = jnp.asarray([0.5, -0.5])
    rejected = jnp.asarray([-0.5, 0.5])
    zeros = jnp.zeros_like(chosen)
    loss = get_loss_function(variant, 0.2, 0.1)(chosen, rejected, zeros, zeros, 0.2, 0.1)
    z = np.asarray([0.2, -0.2])
    expected = np.maximum(1 - z, 0) if variant == "hinge" else 0.9 * np.logaddexp(0, -z) + 0.1 * np.logaddexp(0, z)
    np.testing.assert_allclose(loss, expected, rtol=1e-6)


def test_dpo_compatibility_closure_rejects_removed_kto_variant():
    with pytest.raises(ValueError, match="not valid"):
        get_loss_function("kto", 0.2, 0.1)
