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

import dataclasses

import jax
import jax.numpy as jnp

from easydel.trainers import training_utils
from easydel.trainers.training_utils import compile_trainer_step, make_assertions_and_get_sizes, minibatch_call


@dataclasses.dataclass
class _DummyState:
    graphstate: jax.Array


def test_make_assertions_skips_scalar_leaves_for_batch_size():
    batch = {"running_mean": jnp.asarray(0.0, dtype=jnp.float32), "x": jnp.zeros((4, 3), dtype=jnp.float32)}
    batch_size, minibatch_size, _ = make_assertions_and_get_sizes(batch=batch, gradient_accumulation_steps=2)
    assert batch_size == 4
    assert minibatch_size == 2


def test_make_assertions_uses_dominant_leading_dimension():
    batch = {
        "shared_bias": jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32),
        "teacher_hidden": jnp.zeros((4, 5, 8), dtype=jnp.float32),
        "x": jnp.zeros((4, 3), dtype=jnp.float32),
    }
    batch_size, minibatch_size, _ = make_assertions_and_get_sizes(batch=batch, gradient_accumulation_steps=2)
    assert batch_size == 4
    assert minibatch_size == 2


def test_minibatch_call_matches_full_batch_gradients():
    state = _DummyState(graphstate=jnp.asarray(2.0, dtype=jnp.float32))
    batch = {"x": jnp.asarray([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)}

    def loss_fn(param, minibatch):
        loss = jnp.mean(param * minibatch["x"])
        return loss, loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (full_loss, _), full_grads = grad_fn(state.graphstate, batch)

    grads, metrics = minibatch_call(state=state, batch=batch, minibatch_size=2, grad_fn=grad_fn)

    assert jnp.allclose(grads, full_grads)
    assert jnp.allclose(metrics, full_loss)


def test_minibatch_call_preserves_non_batch_leaves_across_accumulation():
    state = _DummyState(graphstate=jnp.asarray(1.5, dtype=jnp.float32))
    batch = {
        "x": jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3),
        "teacher_hidden": jnp.arange(160, dtype=jnp.float32).reshape(4, 5, 8),
        "global_scale": jnp.asarray(0.25, dtype=jnp.float32),
        "shared_bias": jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32),
    }

    def loss_fn(param, minibatch):
        token_term = jnp.mean(minibatch["x"]) * param
        teacher_term = jnp.mean(minibatch["teacher_hidden"] + minibatch["shared_bias"]) * minibatch["global_scale"]
        loss = token_term + teacher_term
        return loss, loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (full_loss, _), full_grads = grad_fn(state.graphstate, batch)

    grads, metrics = minibatch_call(state=state, batch=batch, minibatch_size=2, grad_fn=grad_fn)

    assert jnp.allclose(grads, full_grads)
    assert jnp.allclose(metrics, full_loss)


def test_compile_trainer_step_delegates_to_spx_jit(monkeypatch):
    class _Mesh:
        is_mpmd = True

    captured = {}

    def fake_jit(fn, **kwargs):
        captured.update(kwargs)

        def raw(*args, **kwargs):
            return fn(*args, **kwargs)

        return raw

    monkeypatch.setattr(training_utils.spx, "jit", fake_jit)

    def fn(x, scale):
        return {"loss": x * scale, "aux": x + scale}

    compiled = compile_trainer_step(fn, mesh=_Mesh(), static_argnums=(1,), in_shardings=("ignored",))
    out = compiled(jnp.asarray(2.0), 5.0)

    assert captured["mesh"].is_mpmd
    assert captured["in_shardings"] == ("ignored",)
    assert set(out) == {"loss", "aux"}
    assert float(out["aux"]) == 7.0
    assert float(out["loss"]) == 10.0
