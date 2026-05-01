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


def test_compile_trainer_step_does_not_schedule_unregistered_steps(monkeypatch):
    class _Mesh:
        is_mpmd = True

    captured = {}
    schedule = object()

    def fake_jit(fn, **kwargs):
        captured.update(kwargs)
        return fn

    monkeypatch.setattr(training_utils.spx, "jit", fake_jit)

    def fn(state, batch):
        return state, batch

    compile_trainer_step(fn, mesh=_Mesh(), schedule=schedule)

    assert captured["mesh"].is_mpmd
    assert "schedule" not in captured
    assert "batch_argnums" not in captured


def test_compile_trainer_step_does_not_schedule_full_trainer_steps(monkeypatch):
    class _Mesh:
        is_mpmd = True

    captured = {}
    schedule = object()

    def fake_jit(fn, **kwargs):
        captured.update(kwargs)
        return fn

    monkeypatch.setattr(training_utils.spx, "jit", fake_jit)

    def training_step(state, batch):
        return state, batch

    compile_trainer_step(training_step, mesh=_Mesh(), schedule=schedule)

    assert "schedule" not in captured
    assert "batch_argnums" not in captured


def test_compile_trainer_step_uses_scheduled_wrapper_for_base_training_step(monkeypatch):
    from easydel.trainers.trainer._fn import training_step

    class _Mesh:
        is_mpmd = True

    captured = {}
    schedule = object()

    def fake_jit(fn, **kwargs):
        captured.update(kwargs)
        return fn

    monkeypatch.setattr(training_utils.spx, "jit", fake_jit)

    compiled = compile_trainer_step(
        training_step,
        mesh=_Mesh(),
        schedule=schedule,
        static_argnums=(2, 3, 4, 5, 6),
        donate_argnums=(0,),
    )

    assert captured == {}
    assert compiled.static_argnums_ == (2, 3, 4, 5, 6)


def test_registered_scheduled_adapter_uses_shared_scheduled_wrapper(monkeypatch):
    class _Mesh:
        is_mpmd = True

    @dataclasses.dataclass
    class _State:
        graphstate: jax.Array

    def custom_training_step(
        state,
        batch,
        loss_config=None,
        learning_rate_fn=None,
        partition_spec=None,
        gradient_accumulation_steps=1,
        custom_scale=1.0,
    ):
        return state, batch

    seen = {}

    def make_loss(call):
        seen["custom_scale"] = call.get("custom_scale")
        seen["partition_spec"] = call.get("partition_spec")

        def loss_fn(tree, batch):
            return tree * batch["x"].mean() * call.get("custom_scale")

        return loss_fn

    adapter = training_utils.ScheduledLossAdapter(
        name="custom",
        make_loss=make_loss,
        make_cache_key=lambda call: (
            call.get("custom_scale"),
            training_utils.scheduled_cache_token(call.get("partition_spec")),
        ),
    )
    training_utils.register_scheduled_loss_adapter(custom_training_step, adapter)
    jit_calls = []

    def fake_jit(fn, **kwargs):
        jit_calls.append(kwargs)
        return fn

    monkeypatch.setattr(training_utils.spx, "jit", fake_jit)
    monkeypatch.setattr(
        training_utils.spx,
        "sxvalue_and_grad",
        lambda fn, argnums=0: lambda tree, batch: (fn(tree, batch), (tree + 1,)),
    )
    monkeypatch.setattr(
        training_utils,
        "_apply_stage_local_gradients",
        lambda **kwargs: (kwargs["state"], kwargs["loss"]),
    )

    try:
        schedule = object()
        compiled = compile_trainer_step(
            custom_training_step,
            mesh=_Mesh(),
            schedule=schedule,
            static_argnums=(2, 3, 4, 5, 6),
        )
        state, loss = compiled(
            _State(graphstate=jnp.asarray(2.0, dtype=jnp.float32)),
            {"x": jnp.ones((4,), dtype=jnp.float32)},
            None,
            None,
            None,
            2,
            3.0,
        )
    finally:
        training_utils._SCHEDULED_LOSS_ADAPTERS.pop(training_utils._scheduled_step_key(custom_training_step), None)
        if hasattr(custom_training_step, "__easydel_scheduled_loss_adapter__"):
            delattr(custom_training_step, "__easydel_scheduled_loss_adapter__")

    assert float(jax.device_get(state.graphstate)) == 2.0
    assert float(jax.device_get(loss)) == 6.0
    assert jit_calls[0]["schedule"] is schedule
    assert jit_calls[0]["batch_argnums"] == (1,)
    assert seen["custom_scale"] == 3.0
    assert seen["partition_spec"] is not None
    assert compiled.static_argnums_ == (2, 3, 4, 5, 6)
