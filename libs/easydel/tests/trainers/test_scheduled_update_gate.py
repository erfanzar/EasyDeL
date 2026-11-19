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

"""Update gate of the scheduled (MPMD pipeline) training step.

Host-orchestration tests (``cpu_ok``) use NumPy stand-ins for the scheduled value-and-grad; the
pipeline tests run a real pp=2 schedule on an accelerator, and the GRPO tests check the scheduled
gate against the SPMD ``grpo_step`` gate on the same batch.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import easydel as ed
import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from easydel.trainers import training_utils
from easydel.trainers.group_relative_policy_optimization import _fn as grpo_fn
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from easydel.trainers.trainer._fn import training_step
from easydel.trainers.training_utils import SCHEDULED_UPDATE_GATE_KEY
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec

# --------------------------------------------------------------------------------------------
# Host orchestration (no array computation)
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _HostState:
    tx: object
    graphstate: object = dataclasses.field(default_factory=lambda: np.asarray(1.0, dtype=np.float32))
    opt_state: str = "optimizer"
    step: int = 0

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def _host_gate_step_open(state, batch, loss_config=None, learning_rate_fn=None):
    raise AssertionError("Expected the registered scheduled implementation")


def _host_gate_step_closed(state, batch, loss_config=None, learning_rate_fn=None):
    raise AssertionError("Expected the registered scheduled implementation")


def _host_gate_step_absent(state, batch, loss_config=None, learning_rate_fn=None):
    raise AssertionError("Expected the registered scheduled implementation")


def _host_compile(monkeypatch, step_fn, gate):
    monkeypatch.setattr(training_utils, "jnp", np)
    monkeypatch.setattr(training_utils.spx, "jit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(training_utils.jax, "process_count", lambda: 1)
    monkeypatch.setattr(training_utils, "_SCHED_SECTION_LOG_BUDGET", [0])

    def host_scheduled_vag(fn, *, argnums, has_aux, microbatch_weight_fn):
        def evaluate(tree, batch):
            return fn(tree, batch), (np.asarray(0.25, np.float32),)

        return evaluate

    monkeypatch.setattr(training_utils.spx, "sxvalue_and_grad", host_scheduled_vag)

    def loss_fn(tree, batch):
        aux = {"accuracy": np.asarray(0.5, np.float32)}
        if gate is not None:
            aux[SCHEDULED_UPDATE_GATE_KEY] = np.asarray(gate, np.float32)
        return np.asarray(2.0, np.float32) * tree, aux

    adapter = training_utils.ScheduledLossAdapter(
        name="host_gate",
        make_loss=lambda call: loss_fn,
        make_cache_key=lambda call: (),
        has_aux=True,
    )
    training_utils.register_scheduled_loss_adapter(step_fn, adapter)
    return training_utils.compile_trainer_step(step_fn, mesh=SimpleNamespace(is_mpmd=True), schedule=object())


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("step_fn", "gate", "applied", "skipped"),
    [
        (_host_gate_step_closed, 0.0, False, 1.0),
        (_host_gate_step_open, 1.0, True, 0.0),
        (_host_gate_step_absent, None, True, None),
    ],
)
def test_host_scheduled_step_honours_update_gate(monkeypatch, step_fn, gate, applied, skipped):
    received = []

    def apply_gradients_stage_local(**kwargs):
        received.append(float(kwargs["grads"]))
        return kwargs["params"] - kwargs["grads"], "updated optimizer"

    state = _HostState(tx=SimpleNamespace(apply_gradients_stage_local=apply_gradients_stage_local))
    compiled = _host_compile(monkeypatch, step_fn, gate)
    new_state, metrics = compiled(state, {"input_ids": np.zeros((2, 3), np.int32)})

    other = dict(metrics.other_metrics or {})
    assert SCHEDULED_UPDATE_GATE_KEY not in other
    assert float(metrics.accuracy) == pytest.approx(0.5)
    if applied:
        assert received == [0.25]
        assert new_state.step == 1
        assert float(new_state.graphstate) == pytest.approx(0.75)
        assert new_state.opt_state == "updated optimizer"
    else:
        assert received == []
        assert new_state is state
    if skipped is None:
        assert "step_skipped" not in other
    else:
        assert float(other["step_skipped"]) == skipped


@pytest.mark.cpu_ok
@pytest.mark.parametrize(("value", "expected"), [(1.0, True), (0.9995, True), (0.5, False), (0.0, False)])
def test_gate_value_decision_and_nonfinite_fail_closed(value, expected):
    aux, gate = training_utils._resolve_scheduled_update_gate({"a": 1.0, SCHEDULED_UPDATE_GATE_KEY: value})
    assert gate is expected
    assert aux == {"a": 1.0}
    assert training_utils._resolve_scheduled_update_gate({SCHEDULED_UPDATE_GATE_KEY: float("nan")})[1] is False
    assert training_utils._resolve_scheduled_update_gate({"a": 1.0}) == ({"a": 1.0}, None)


@pytest.mark.cpu_ok
def test_grpo_gate_knobs_are_accepted_under_mpmd_scheduler():
    from spectrax.runtime.schedules import Std1F1B

    cfg = GRPOConfig(
        mpmd_scheduler=Std1F1B(microbatches=2),
        off_policy_min_acceptance=0.5,
        startup_behavior_k3_max=0.5,
    )
    assert cfg.off_policy_min_acceptance == 0.5 and cfg.startup_behavior_k3_max == 0.5


def _finalize(monkeypatch, sums, **bound):
    monkeypatch.setattr(grpo_fn, "jnp", np)
    batch = bound.pop("batch", {})
    call = SimpleNamespace(get=lambda name, default=None: bound.get(name, default), batch=batch)
    aux = {grpo_fn._GRPO_GATE_AUX_PREFIX + key: np.float32(sums.get(key, 0.0)) for key in grpo_fn._GRPO_GATE_SUM_KEYS}
    return grpo_fn._grpo_scheduled_finalize_aux(call, aux)


@pytest.mark.cpu_ok
def test_grpo_finalize_without_behaviour_logprobs(monkeypatch):
    # Acceptance gate without behaviour logprobs fails closed.
    out = _finalize(monkeypatch, {}, off_policy_min_acceptance=0.5)
    assert float(out[SCHEDULED_UPDATE_GATE_KEY]) == 0.0 and float(out["off_policy/step_skipped"]) == 1.0
    # Parity-only gate skips exactly the flagged first step.
    flagged = {"behavior_parity_gate": np.asarray(1, np.int32)}
    out = _finalize(monkeypatch, {}, startup_behavior_k3_max=0.1, batch=flagged)
    assert float(out[SCHEDULED_UPDATE_GATE_KEY]) == 0.0
    cleared = {"behavior_parity_gate": np.asarray(0, np.int32)}
    out = _finalize(monkeypatch, {}, startup_behavior_k3_max=0.1, batch=cleared)
    assert float(out[SCHEDULED_UPDATE_GATE_KEY]) == 1.0


@pytest.mark.cpu_ok
def test_grpo_finalize_ratios_from_reduced_sums(monkeypatch):
    # Sums averaged over 2 microbatches (scale 1/2): ratios are scale invariant.
    sums = {"has_behavior": 1.0, "eligible_rows": 2.0, "accepted_rows": 1.5, "tokens": 4.5, "k3_sum": 0.45}
    out = _finalize(monkeypatch, sums, off_policy_min_acceptance=0.75)
    assert float(out["off_policy/accepted_fraction"]) == pytest.approx(0.75)
    assert float(out["off_policy/behavior_k3"]) == pytest.approx(0.1)
    assert float(out[SCHEDULED_UPDATE_GATE_KEY]) == 1.0
    out = _finalize(monkeypatch, sums, off_policy_min_acceptance=0.8)
    assert float(out[SCHEDULED_UPDATE_GATE_KEY]) == 0.0


# --------------------------------------------------------------------------------------------
# GRPO: scheduled gate == SPMD gate on the same batch (accelerator)
# --------------------------------------------------------------------------------------------

VOCAB = 5


def _stub_logits(input_ids, bias, xp):
    tilt = 0.1 * input_ids[:, :, None].astype(xp.float32) * xp.arange(VOCAB, dtype=xp.float32)[None, None, :]
    return bias[None, : input_ids.shape[1], :] + tilt


class _StubModel:
    def __init__(self, bias, mesh):
        self.bias = bias
        self.mesh = mesh

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=_stub_logits(input_ids, self.bias, jnp))


@jax.tree_util.register_pytree_node_class
class _StubState:
    def __init__(self, graphstate, opt_state, step, tx, mesh):
        self.graphstate, self.opt_state, self.step, self.tx, self.mesh = graphstate, opt_state, step, tx, mesh

    def tree_flatten(self):
        return (self.graphstate, self.opt_state, self.step), (self.tx, self.mesh)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    @property
    def model(self):
        return _StubModel(self.graphstate, self.mesh)

    def merge(self, tree):
        return _StubModel(tree, self.mesh)

    def replace(self, graphstate):
        return _StubState(graphstate, self.opt_state, self.step, self.tx, self.mesh)

    def apply_gradients(self, grads):
        updates, opt_state = self.tx.update(grads, self.opt_state, self.graphstate)
        return _StubState(optax.apply_updates(self.graphstate, updates), opt_state, self.step + 1, self.tx, self.mesh)


def _grpo_batch(behavior_shift):
    rng = np.random.default_rng(1)
    prompt_ids = np.array([[1], [3]], dtype=np.int32)
    completion_ids = np.array([[0, 2, 4], [1, 1, 3], [4, 0, 2], [3, 2, 1]], dtype=np.int32)
    mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=np.float32)
    bias = rng.normal(size=(4, VOCAB)).astype(np.float32)
    input_ids = np.concatenate([np.repeat(prompt_ids, 2, 0), completion_ids], axis=1)
    logits = _stub_logits(input_ids, bias, np).astype(np.float64)[:, prompt_ids.shape[1] - 1 : -1]
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    policy = np.take_along_axis(log_probs, completion_ids[..., None], -1)[..., 0]
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.ones_like(jnp.asarray(prompt_ids)),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(mask),
        "advantages": jnp.asarray([1.5, -0.7, 0.4, -1.2], dtype=jnp.float32),
        "behavior_per_token_logps": jnp.asarray(policy + np.asarray(behavior_shift), dtype=jnp.float32),
    }
    return batch, jnp.asarray(bias)


@pytest.mark.parametrize(
    ("knobs", "shift_row", "parity_flag"),
    [
        ({"off_policy_k3_threshold": 0.5, "off_policy_min_acceptance": 1.0}, 2.0, None),
        ({"off_policy_k3_threshold": 0.5, "off_policy_min_acceptance": 0.5}, 2.0, None),
        ({"startup_behavior_k3_max": 0.01}, 0.5, 1),
        ({"startup_behavior_k3_max": 0.01}, 0.5, 0),
        ({"startup_behavior_k3_max": 10.0, "off_policy_min_acceptance": 0.5}, 0.0, 1),
    ],
)
def test_grpo_scheduled_gate_matches_spmd_gate(knobs, shift_row, parity_flag):
    shift = np.zeros((4, 3))
    shift[1] = shift_row
    batch, bias = _grpo_batch(shift)
    if parity_flag is not None:
        batch["behavior_parity_gate"] = jnp.asarray(parity_flag, dtype=jnp.int32)
    tx = optax.adamw(1e-2)
    state = _StubState(bias, tx.init(bias), jnp.asarray(0), tx, Mesh(np.array(jax.devices()[:1]), ("dp",)))
    common = dict(partition_spec=PartitionSpec(), loss_type="dapo", **knobs)

    _, spmd_metrics = grpo_step(state, batch, 2, 0.0, is_training=True, **common)
    spmd_skipped = float(spmd_metrics.other_metrics["off_policy/step_skipped"])

    # Bind exactly as the scheduled step does (signature defaults applied).
    signature = inspect.signature(grpo_step)
    bound_args = signature.bind(state, batch, 2, 0.0, **common)
    bound_args.apply_defaults()
    bound = dict(bound_args.arguments)
    call = training_utils.ScheduledStepCall(
        step_fn=grpo_step,
        state=state,
        batch=batch,
        args=(),
        kwargs={},
        bound_arguments=bound,
        schedule=None,
    )
    adapter = training_utils.get_scheduled_loss_adapter(grpo_step)
    assert adapter.resolve_has_aux(call)
    loss_fn = adapter.make_loss(call)
    pipelined = adapter.prepare_batch(call)
    assert "behavior_parity_gate" not in pipelined
    # Two pipeline microbatches (one prompt each) reduced with the runtime's uniform 1/m scale.
    auxes = []
    for i in range(2):
        mb = {
            key: (value[i : i + 1] if key.startswith("prompt") else value[2 * i : 2 * i + 2])
            for key, value in pipelined.items()
        }
        _, aux = loss_fn(state.graphstate, mb)
        auxes.append(aux)
    reduced = jax.tree_util.tree_map(lambda a, b: (a + b) / 2.0, *auxes)
    out = adapter.finalize_aux(call, reduced)

    assert float(out["off_policy/step_skipped"]) == spmd_skipped
    assert float(out[SCHEDULED_UPDATE_GATE_KEY]) == 1.0 - spmd_skipped
    for key in ("off_policy/accepted_fraction", "off_policy/behavior_k3", "off_policy/tail_fraction"):
        np.testing.assert_allclose(float(out[key]), float(spmd_metrics.other_metrics[key]), rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------------------------
# Real pp=2 pipeline (accelerator)
# --------------------------------------------------------------------------------------------

_FORCED_GATE: dict[str, float] = {}


def _gate_closed_step(
    state,
    batch,
    loss_config=None,
    learning_rate_fn=None,
    partition_spec=None,
    gradient_accumulation_steps=1,
    straight_through_emulator=None,
    loss_type="nll",
):
    raise AssertionError("Expected the registered scheduled implementation")


def _gate_open_step(
    state,
    batch,
    loss_config=None,
    learning_rate_fn=None,
    partition_spec=None,
    gradient_accumulation_steps=1,
    straight_through_emulator=None,
    loss_type="nll",
):
    raise AssertionError("Expected the registered scheduled implementation")


def _register_forced_gate(step_fn, gate: float) -> None:
    """Base causal-LM scheduled loss that additionally reports a constant update gate."""
    base = training_utils.get_scheduled_loss_adapter(training_step)

    def make_loss(call):
        inner = base.make_loss(call)

        def loss(tree, batch):
            value, aux = inner(tree, batch)
            # Anchored on the loss so the runtime computes it in the terminal stage.
            anchored = jnp.asarray(gate, jnp.float32) + 0.0 * jax.lax.stop_gradient(value).astype(jnp.float32)
            return value, {**aux, SCHEDULED_UPDATE_GATE_KEY: anchored}

        return loss

    training_utils.register_scheduled_loss_adapter(
        step_fn,
        training_utils.ScheduledLossAdapter(
            name=f"forced_gate_{int(gate)}",
            make_loss=make_loss,
            make_cache_key=lambda call: (*base.make_cache_key(call), gate),
            has_aux=True,
            prepare_batch=base.prepare_batch,
            make_microbatch_weight=base.make_microbatch_weight,
        ),
    )


_register_forced_gate(_gate_closed_step, 0.0)
_register_forced_gate(_gate_open_step, 1.0)


@pytest.fixture(scope="module")
def pp2_model():
    if jax.device_count() < 2:
        pytest.skip("pp=2 pipeline tests need at least 2 devices")
    config = ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sharding_axis_dims=(2, 1, 1, -1, 1, 1),
        pipeline_stage_regions=True,
        scan_layers=False,
        attn_mechanism="vanilla",
        pad_token_id=0,
    )
    model = ed.LlamaForCausalLM.sequential_init(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    ).shard_model()
    assert bool(getattr(model.mesh, "is_mpmd", False))
    return model


def _fresh_state(model):
    from eformer.optimizers import OptimizerFactory, SchedulerConfig

    tx, scheduler = OptimizerFactory.create("adamw", SchedulerConfig(learning_rate=1e-2))
    return model.to_state().init_tx(tx), scheduler


def _host_tree(tree):
    return [np.array(jax.device_get(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]


def _run_pp2_step(model, step_fn):
    state, scheduler = _fresh_state(model)
    before = (_host_tree(state.graphstate), _host_tree(state.opt_state), int(jax.device_get(state.step)))
    ids = np.random.default_rng(0).integers(1, 128, size=(4, 16)).astype(np.int32)
    batch = {
        "input_ids": jnp.asarray(ids),
        "attention_mask": jnp.ones_like(jnp.asarray(ids)),
        "labels": jnp.asarray(ids),
    }
    compiled = training_utils.compile_trainer_step(
        step_fn, mesh=model.mesh, schedule=spx.GPipe(microbatches=2), arguments=None
    )
    new_state, metrics = compiled(state, batch, learning_rate_fn=scheduler)
    after = (_host_tree(new_state.graphstate), _host_tree(new_state.opt_state), int(jax.device_get(new_state.step)))
    return before, after, metrics


def test_pp2_closed_gate_leaves_params_and_optimizer_state_bit_identical(pp2_model):
    before, after, metrics = _run_pp2_step(pp2_model, _gate_closed_step)
    assert float(metrics.other_metrics["step_skipped"]) == 1.0
    assert np.isfinite(float(metrics.loss))
    assert after[2] == before[2]
    for group_before, group_after in zip(before[:2], after[:2], strict=True):
        assert len(group_before) == len(group_after)
        for a, b in zip(group_before, group_after, strict=True):
            np.testing.assert_array_equal(a, b)


def test_pp2_open_gate_matches_ungated_update(pp2_model):
    before, gated, gated_metrics = _run_pp2_step(pp2_model, _gate_open_step)
    _, ungated, ungated_metrics = _run_pp2_step(pp2_model, training_step)

    assert float(gated_metrics.other_metrics["step_skipped"]) == 0.0
    assert "step_skipped" not in (ungated_metrics.other_metrics or {})
    assert SCHEDULED_UPDATE_GATE_KEY not in gated_metrics.other_metrics
    assert gated[2] == ungated[2] == before[2] + 1
    assert any(not np.array_equal(a, b) for a, b in zip(before[0], gated[0], strict=True))
    np.testing.assert_allclose(float(gated_metrics.loss), float(ungated_metrics.loss), rtol=1e-6)
    for a, b in zip(gated[0], ungated[0], strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
