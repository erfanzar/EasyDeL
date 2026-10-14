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
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from easydel.infra.loss_utils import ForCausalLMLoss, LossConfig, LossMetrics
from easydel.modules._base.causal_lm_module import BaseCausalLMModule
from easydel.trainers import training_utils
from easydel.trainers.trainer._fn import (
    _CE_TOKEN_SCALE_KEY,
    _ce_token_scale_leaf,
    _make_base_spmd_weight,
    _split_ce_aux_objective,
    _valid_token_count_fn,
)
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


def test_minibatch_call_token_weighted_matches_full_batch_token_mean():
    # Microbatches hold different numbers of valid tokens (3 vs 1). A uniform average of
    # per-microbatch token means differs from the full-batch token mean; weighting each
    # microbatch by its valid-token count must reproduce the full-batch value and gradient.
    state = _DummyState(graphstate=jnp.asarray(1.5, dtype=jnp.float32))
    batch = {
        "x": jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
        "mask": jnp.asarray([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
    }

    def loss_fn(param, minibatch):
        per_token = (param * minibatch["x"]) ** 2
        loss = jnp.sum(per_token * minibatch["mask"]) / jnp.maximum(jnp.sum(minibatch["mask"]), 1.0)
        return loss, loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (full_loss, _), full_grads = grad_fn(state.graphstate, batch)

    uniform_grads, uniform_metrics = minibatch_call(state=state, batch=batch, minibatch_size=2, grad_fn=grad_fn)
    assert not jnp.allclose(uniform_metrics, full_loss)
    assert not jnp.allclose(uniform_grads, full_grads)

    grads, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=2,
        grad_fn=grad_fn,
        microbatch_weight_fn=lambda _tree, mb: jnp.sum(mb["mask"]),
    )
    assert jnp.allclose(metrics, full_loss, rtol=1e-6)
    assert jnp.allclose(grads, full_grads, rtol=1e-6)


def test_minibatch_call_sums_weight_sum_in_uniform_and_weighted_branches():
    state = _DummyState(graphstate=jnp.asarray(1.5, dtype=jnp.float32))
    batch = {
        "x": jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
        "mask": jnp.asarray([[1, 1], [1, 0], [0, 0], [0, 1]], dtype=jnp.float32),
    }

    def loss_fn(param, mb):
        count = jnp.sum(mb["mask"])
        loss = jnp.sum(param * mb["x"] * mb["mask"]) / jnp.maximum(count, 1.0)
        return loss, LossMetrics(loss=loss, weight_sum=count, accuracy=loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, full_metrics), full_grads = grad_fn(state.graphstate, batch)
    uniform_grads, uniform_metrics = minibatch_call(state, batch, 2, grad_fn)
    weighted_grads, weighted_metrics = minibatch_call(
        state, batch, 2, grad_fn, microbatch_weight_fn=lambda _tree, mb: jnp.sum(mb["mask"])
    )
    assert jnp.allclose(full_metrics.weight_sum, 4.0)
    assert jnp.allclose(uniform_metrics.weight_sum, full_metrics.weight_sum)
    assert jnp.allclose(weighted_metrics.weight_sum, full_metrics.weight_sum)
    assert jnp.allclose(weighted_grads, full_grads)
    assert jnp.allclose(weighted_metrics.loss, full_metrics.loss)
    assert jnp.allclose(weighted_metrics.accuracy, full_metrics.accuracy)
    assert not jnp.allclose(uniform_grads, full_grads)


def test_aux_capable_causal_lm_uses_uniform_microbatch_gradients():
    # The inherited causal-LM forward can add router aux through this feature.
    # Avoid constructing/forwarding a real model here; the gating decision is
    # made from the same static model shape before tracing the training step.
    class _AuxCapableCausalLM(BaseCausalLMModule):
        loss_function = property(lambda self: ForCausalLMLoss)
        _router_aux_loss_feature = object()

    model = object.__new__(_AuxCapableCausalLM)

    # The same inherited forward with its router feature absent is provably
    # CE-only and retains the token-mean weighting improvement.
    class _PureCausalLM(BaseCausalLMModule):
        loss_function = property(lambda self: ForCausalLMLoss)
        _router_aux_loss_feature = None

    pure_call = {"state": SimpleNamespace(model=object.__new__(_PureCausalLM)), "loss_config": LossConfig()}
    assert _make_base_spmd_weight(pure_call) is not None
    call = {"state": SimpleNamespace(model=model), "loss_config": LossConfig(), "loss_type": "nll"}
    weight_fn = _make_base_spmd_weight(call)
    assert weight_fn is None

    state = _DummyState(graphstate=jnp.asarray(2.0, dtype=jnp.float32))
    batch = {
        "x": jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        "mask": jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32),
    }

    def loss_fn(param, mb):
        count = jnp.sum(mb["mask"])
        ce = jnp.sum(param * mb["x"] * mb["mask"]) / jnp.maximum(count, 1.0)
        aux = jnp.mean(param * mb["x"])
        loss = ce + aux
        return loss, LossMetrics(loss=loss, weight_sum=count, other_metrics={"aux_loss": aux})

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, first_metrics), first_grad = grad_fn(state.graphstate, {key: val[:2] for key, val in batch.items()})
    (_, second_metrics), second_grad = grad_fn(state.graphstate, {key: val[2:] for key, val in batch.items()})
    grads, metrics = minibatch_call(state, batch, 2, grad_fn, microbatch_weight_fn=weight_fn)
    assert jnp.allclose(grads, (first_grad + second_grad) / 2)
    assert jnp.allclose(
        metrics.other_metrics["aux_loss"],
        (first_metrics.other_metrics["aux_loss"] + second_metrics.other_metrics["aux_loss"]) / 2,
    )
    assert jnp.allclose(metrics.weight_sum, 2.0)
    # If token weights were incorrectly applied, the first aux gradient would
    # disappear with its zero valid-label count.
    incorrectly_weighted_grads, _ = minibatch_call(
        state, batch, 2, grad_fn, microbatch_weight_fn=lambda _tree, mb: jnp.sum(mb["mask"])
    )
    assert not jnp.allclose(grads, incorrectly_weighted_grads)


def test_ce_aux_split_matches_full_batch_ce_plus_mean_aux():
    """Uniform mean of split objectives == full-batch valid-token CE + mean aux (S13-4).

    Microbatch 0 has no valid labels: its CE must contribute nothing while its aux still does.
    """
    state = _DummyState(graphstate=jnp.asarray(1.5, dtype=jnp.float32))
    x = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, -2.0, 3.0]], dtype=jnp.float32)
    labels = jnp.asarray([[-100, -100, -100], [-100, -100, -100], [0, 1, 2], [0, 1, -100]], dtype=jnp.int32)
    count = _valid_token_count_fn(LossConfig(), "nll")
    batch = {"x": x, "labels": labels}
    batch[_CE_TOKEN_SCALE_KEY] = _ce_token_scale_leaf(count, batch)

    def parts(param, mb):
        valid = (mb["labels"][:, 1:] != -100).astype(jnp.float32)
        per_tok = param * mb["x"][:, 1:]
        ce = jnp.sum(per_tok * valid) / jnp.maximum(jnp.sum(valid), 1.0)
        aux = jnp.mean(param * mb["x"])
        return ce, aux

    def loss_fn(param, mb):
        mb = dict(mb)
        scale = mb.pop(_CE_TOKEN_SCALE_KEY)
        ce, aux = parts(param, mb)
        valid = (mb["labels"][:, 1:] != -100).astype(jnp.float32)
        # A token-normalized accuracy stand-in: fraction of valid tokens with positive score.
        hits = ((param * mb["x"][:, 1:]) > 2.0).astype(jnp.float32)
        accuracy = jnp.sum(hits * valid) / jnp.maximum(jnp.sum(valid), 1.0)
        metrics = LossMetrics(loss=ce + aux, accuracy=accuracy, other_metrics={"aux_loss": aux})
        return _split_ce_aux_objective(ce + aux, metrics, scale, count, mb)

    grads, metrics = minibatch_call(state, batch, 2, jax.value_and_grad(loss_fn, has_aux=True))

    def expected(param):
        valid = (labels[:, 1:] != -100).astype(jnp.float32)
        ce_full = jnp.sum(param * x[:, 1:] * valid) / jnp.sum(valid)
        aux_mean = (jnp.mean(param * x[:2]) + jnp.mean(param * x[2:])) / 2
        return ce_full + aux_mean

    want = jax.grad(expected)(state.graphstate)
    assert jnp.allclose(grads, want, rtol=1e-6), (grads, want)
    # r11 S13-2: the reported loss is the differentiated objective, and the token-normalized
    # accuracy is the full-batch valid-token value, not the uniform mean of per-microbatch values.
    assert jnp.allclose(metrics.loss, expected(state.graphstate), rtol=1e-6), metrics.loss
    valid = (labels[:, 1:] != -100).astype(jnp.float32)
    hits = ((state.graphstate * x[:, 1:]) > 2.0).astype(jnp.float32)
    want_acc = jnp.sum(hits * valid) / jnp.sum(valid)
    assert jnp.allclose(metrics.accuracy, want_acc, rtol=1e-6), (metrics.accuracy, want_acc)
    assert jnp.allclose(
        metrics.other_metrics["aux_loss"], (jnp.mean(1.5 * x[:2]) + jnp.mean(1.5 * x[2:])) / 2, rtol=1e-6
    )

    # Negative control: the old uniform mean of unsplit (CE_i + aux_i) is a different gradient.
    def unsplit(param, mb):
        mb = {k: v for k, v in mb.items() if k != _CE_TOKEN_SCALE_KEY}
        ce, aux = parts(param, mb)
        return ce + aux, LossMetrics(loss=ce + aux)

    old, _ = minibatch_call(state, batch, 2, jax.value_and_grad(unsplit, has_aux=True))
    assert not jnp.allclose(old, want, rtol=1e-3)


def test_minibatch_call_all_zero_weights_fall_back_to_uniform_average():
    state = _DummyState(graphstate=jnp.asarray(2.0, dtype=jnp.float32))
    batch = {"x": jnp.asarray([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)}

    def loss_fn(param, minibatch):
        loss = jnp.mean(param * minibatch["x"])
        return loss, loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    uniform_grads, uniform_metrics = minibatch_call(state=state, batch=batch, minibatch_size=2, grad_fn=grad_fn)
    grads, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=2,
        grad_fn=grad_fn,
        microbatch_weight_fn=lambda _tree, _mb: jnp.float32(0.0),
    )
    assert jnp.allclose(grads, uniform_grads)
    assert jnp.allclose(metrics, uniform_metrics)


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


def test_physical_stage_config_context_restores_virtual_stage_settings():
    class _Config:
        pipeline_virtual_stages = 4

    class _Child:
        config = _Config()

    class _Module:
        config = _Config()
        model = _Child()

    module = _Module()

    with training_utils.sync_module_physical_stage_config(module):
        assert module.config.pipeline_virtual_stages == 1
        assert module.model.config.pipeline_virtual_stages == 1

    assert module.config.pipeline_virtual_stages == 4
    assert module.model.config.pipeline_virtual_stages == 4


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
        lambda fn, argnums=0, has_aux=False, microbatch_weight_fn=None: (
            lambda tree, batch: (fn(tree, batch), (tree + 1,))
        ),
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
