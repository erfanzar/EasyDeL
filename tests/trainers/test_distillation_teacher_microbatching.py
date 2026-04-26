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

from easydel.trainers.distillation_trainer import _fn as distill_fn
from easydel.trainers.generalized_knowledge_distillation_trainer import _fn as gkd_fn


class _TeacherModule:
    def __init__(self, call_shapes: list[tuple[int, ...]]):
        self._call_shapes = call_shapes

    def __call__(
        self,
        input_ids,
        attention_mask,
        apply_lm_head=True,
        output_hidden_states=False,
        output_attentions=False,
    ):
        self._call_shapes.append(tuple(int(dim) for dim in input_ids.shape))
        x = input_ids.astype(jnp.float32)
        logits = jnp.stack((x * 0.5 + 0.1, x * -0.25 - 0.2), axis=-1)
        hidden = jnp.stack((x, x + 1.0), axis=-1)
        attention = (
            attention_mask.astype(jnp.float32)[:, None, :, None] * attention_mask.astype(jnp.float32)[:, None, None, :]
        )
        return SimpleNamespace(
            logits=logits,
            last_hidden_state=hidden,
            hidden_states=(hidden,) if output_hidden_states else None,
            attentions=(attention,) if output_attentions else None,
        )

    def make_lm_head_fn(self):
        def lm_head(hidden):
            h0 = hidden[..., 0]
            h1 = hidden[..., 1]
            return jnp.stack((h0 + h1, h0 - h1), axis=-1)

        return lm_head


class _TeacherState:
    def __init__(self, call_shapes: list[tuple[int, ...]]):
        self.model = _TeacherModule(call_shapes)
        self.graphstate = jnp.asarray(0.0, dtype=jnp.float32)

    def merge(self, _tree):
        return self.model


class _StudentModule:
    def __init__(self, scale):
        self._scale = scale

    def __call__(
        self,
        input_ids,
        attention_mask,
        apply_lm_head=True,
        output_hidden_states=False,
        output_attentions=False,
    ):
        x = input_ids.astype(jnp.float32)
        logits = jnp.stack((x * self._scale + 0.3, x * self._scale - 0.4), axis=-1)
        hidden = jnp.stack((x * self._scale, x * self._scale + 0.5), axis=-1)
        attention = (
            attention_mask.astype(jnp.float32)[:, None, :, None] * attention_mask.astype(jnp.float32)[:, None, None, :]
        )
        return SimpleNamespace(
            logits=logits,
            last_hidden_state=hidden,
            hidden_states=(hidden,) if output_hidden_states else None,
            attentions=(attention,) if output_attentions else None,
        )

    def make_lm_head_fn(self):
        def lm_head(hidden):
            h0 = hidden[..., 0]
            h1 = hidden[..., 1]
            return jnp.stack((h0 + h1, h0 - h1), axis=-1)

        return lm_head


class _StudentState:
    def __init__(self):
        self.model = SimpleNamespace(mesh=None)
        self.graphstate = jnp.asarray(1.0, dtype=jnp.float32)
        self.step = jnp.asarray(0, dtype=jnp.int32)

    def merge(self, tree):
        return _StudentModule(tree)


def _run_single_minibatch(state, batch, minibatch_size, grad_fn):
    assert "_teacher_logits" not in batch
    assert "_teacher_hidden_state" not in batch
    batch_size = int(batch["input_ids"].shape[0])
    minibatch = jax.tree_util.tree_map(
        lambda x: x[:minibatch_size] if hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == batch_size else x,
        batch,
    )
    (_, metrics), grads = grad_fn(state.graphstate, minibatch)
    return grads, metrics


def test_distillation_teacher_forward_uses_minibatches(monkeypatch):
    teacher_call_shapes: list[tuple[int, ...]] = []
    student_state = _StudentState()
    teacher_state = _TeacherState(teacher_call_shapes)
    batch = {
        "input_ids": jnp.arange(12, dtype=jnp.int32).reshape(4, 3),
        "attention_mask": jnp.ones((4, 3), dtype=jnp.int32),
    }

    monkeypatch.setattr(distill_fn, "minibatch_call", _run_single_minibatch)
    monkeypatch.setattr(distill_fn, "update_state_respectfully", lambda state, gradients, loss_config, metrics: state)
    monkeypatch.setattr(distill_fn, "update_metrics", lambda metrics, learning_rate_fn, step, gradients: metrics)
    monkeypatch.setattr(distill_fn, "with_sharding_constraint", lambda batch, sharding, **kwargs: batch)

    _state, metrics = distill_fn.distillation_step(
        student_state=student_state,
        batch=batch,
        teacher_state=teacher_state,
        gradient_accumulation_steps=2,
        is_training=True,
        temperature=1.0,
        alpha=1.0,
        logits_chunk_size=2,
    )

    assert teacher_call_shapes
    assert max(shape[0] for shape in teacher_call_shapes) == 2
    assert metrics.loss.shape == ()


def test_gkd_teacher_forward_uses_minibatches(monkeypatch):
    teacher_call_shapes: list[tuple[int, ...]] = []
    student_state = _StudentState()
    teacher_state = _TeacherState(teacher_call_shapes)
    batch = {
        "input_ids": jnp.arange(12, dtype=jnp.int32).reshape(4, 3),
        "attention_mask": jnp.ones((4, 3), dtype=jnp.int32),
    }

    monkeypatch.setattr(gkd_fn, "minibatch_call", _run_single_minibatch)
    monkeypatch.setattr(gkd_fn, "update_state_respectfully", lambda state, gradients, loss_config, metrics: state)
    monkeypatch.setattr(gkd_fn, "update_metrics", lambda metrics, learning_rate_fn, step, gradients: metrics)
    monkeypatch.setattr(gkd_fn, "with_sharding_constraint", lambda batch, sharding, **kwargs: batch)

    _state, metrics = gkd_fn.gkd_step(
        student_state=student_state,
        batch=batch,
        teacher_state=teacher_state,
        gradient_accumulation_steps=2,
        is_training=True,
        beta=0.5,
        temperature=1.0,
    )

    assert teacher_call_shapes
    assert max(shape[0] for shape in teacher_call_shapes) == 2
    assert metrics.loss.shape == ()
