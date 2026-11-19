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

"""Fail-closed off-policy contract: diagnostics, sequence rejection and the update gate."""

from types import SimpleNamespace

import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
from easydel.trainers._shared.off_policy import (
    behavior_policy_terms,
    check_startup_behavior_parity,
    summarize_behavior_sums,
)
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec

VOCAB = 5


def _stub_logits(input_ids, bias, xp):
    tilt = 0.1 * input_ids[:, :, None].astype(xp.float32) * xp.arange(VOCAB, dtype=xp.float32)[None, None, :]
    return bias[None, : input_ids.shape[1], :] + tilt


class _Model:
    def __init__(self, bias, mesh):
        self.bias = bias
        self.mesh = mesh

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=_stub_logits(input_ids, self.bias, jnp))


@jax.tree_util.register_pytree_node_class
class _TrainState:
    """Minimal optax-backed state exposing the surface ``grpo_step`` uses."""

    def __init__(self, graphstate, opt_state, step, tx, mesh):
        self.graphstate = graphstate
        self.opt_state = opt_state
        self.step = step
        self.tx = tx
        self.mesh = mesh

    def tree_flatten(self):
        return (self.graphstate, self.opt_state, self.step), (self.tx, self.mesh)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    @property
    def model(self):
        return _Model(self.graphstate, self.mesh)

    def merge(self, tree):
        return _Model(tree, self.mesh)

    def apply_gradients(self, grads):
        updates, opt_state = self.tx.update(grads, self.opt_state, self.graphstate)
        return _TrainState(optax.apply_updates(self.graphstate, updates), opt_state, self.step + 1, self.tx, self.mesh)


def _mesh():
    return Mesh(np.array(jax.devices()[:1]), ("dp",))


def _policy_logps(prompt_ids, completion_ids, bias, num_generations):
    input_ids = np.concatenate([np.repeat(prompt_ids, num_generations, 0), completion_ids], axis=1)
    logits = _stub_logits(input_ids, np.asarray(bias), np).astype(np.float64)
    logits = logits[:, prompt_ids.shape[1] - 1 : -1]
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    return np.take_along_axis(log_probs, completion_ids[..., None], -1)[..., 0]


def _batch(behavior_shift):
    rng = np.random.default_rng(1)
    prompt_ids = np.array([[1], [3]], dtype=np.int32)
    completion_ids = np.array([[0, 2, 4], [1, 1, 3], [4, 0, 2], [3, 2, 1]], dtype=np.int32)
    mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=np.float32)
    bias = rng.normal(size=(4, VOCAB)).astype(np.float32)
    policy = _policy_logps(prompt_ids, completion_ids, bias, 2)
    behavior = policy + np.asarray(behavior_shift, dtype=np.float64)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.ones_like(jnp.asarray(prompt_ids)),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(mask),
        "advantages": jnp.asarray([1.5, -0.7, 0.4, -1.2], dtype=jnp.float32),
        "behavior_per_token_logps": jnp.asarray(behavior, dtype=jnp.float32),
    }
    return batch, jnp.asarray(bias), mask


def test_behavior_diagnostics_match_numpy_reference():
    learner = np.array([[-1.0, -0.5, -2.0, -0.1], [-0.3, -1.2, -0.7, -4.0]], dtype=np.float32)
    behavior = np.array([[-1.1, -0.2, -2.6, 9.0], [-0.3, -1.0, -0.9, 9.0]], dtype=np.float32)
    mask = np.array([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=np.float32)
    terms = behavior_policy_terms(jnp.asarray(learner), jnp.asarray(behavior), jnp.asarray(mask))
    summary = summarize_behavior_sums(terms.sums)

    lr = (learner.astype(np.float64) - behavior)[mask > 0]
    w = np.exp(lr)
    k3 = np.expm1(lr) - lr
    np.testing.assert_allclose(float(summary["off_policy/behavior_k3"]), k3.mean(), rtol=2e-5)
    np.testing.assert_allclose(float(summary["off_policy/ess"]), w.sum() ** 2 / (w.size * (w**2).sum()), rtol=2e-5)
    np.testing.assert_allclose(float(summary["off_policy/chi2"]), ((w - 1) ** 2).mean(), rtol=2e-5)
    np.testing.assert_allclose(float(summary["off_policy/tail_fraction"]), (np.abs(lr) > 0.25).mean(), rtol=1e-6)
    seq_k3 = (np.expm1(learner - behavior) - (learner - behavior)) * mask
    np.testing.assert_allclose(np.asarray(terms.sequence_k3[:, 0]), seq_k3.sum(-1) / mask.sum(-1), rtol=2e-5)
    assert float(summary["off_policy/accepted_fraction"]) == 1.0
    assert float(summary["off_policy/nonfinite_tokens"]) == 0.0


def test_nonfinite_behavior_token_is_counted_and_rejected():
    learner = jnp.asarray([[-1.0, -0.5], [-0.3, -0.2]], dtype=jnp.float32)
    behavior = jnp.asarray([[-1.0, jnp.nan], [-0.3, -0.2]], dtype=jnp.float32)
    mask = jnp.ones((2, 2), dtype=jnp.float32)
    terms = behavior_policy_terms(learner, behavior, mask)
    summary = summarize_behavior_sums(terms.sums)

    np.testing.assert_array_equal(np.asarray(terms.accept[:, 0]), [False, True])
    assert float(summary["off_policy/nonfinite_tokens"]) == 1.0
    assert float(summary["off_policy/accepted_fraction"]) == 0.5
    # The non-finite token is neutralised (lr = 0), so every diagnostic stays finite.
    assert all(np.isfinite(float(v)) for v in summary.values())


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_rejected_sequence_keeps_the_token_normaliser(chunk_size):
    # Row 1 drifts far from the behaviour policy; rows 0, 2, 3 are on-policy.
    shift = np.zeros((4, 3))
    shift[1] = 2.0
    batch, bias, mask = _batch(shift)
    state = _TrainState(bias, None, jnp.asarray(0), None, _mesh())
    metrics = grpo_step(
        state,
        batch,
        2,
        0.0,
        partition_spec=PartitionSpec(),
        is_training=False,
        loss_type="dapo",
        completion_chunk_size=chunk_size,
        off_policy_k3_threshold=0.5,
    )
    # On-policy surrogate is -A per token; row 1 is zeroed but its 2 tokens stay in the 9-token denominator.
    advantages = np.asarray(batch["advantages"], np.float64)
    tokens = mask.sum(-1)
    keep = np.array([1.0, 0.0, 1.0, 1.0])
    expected = -(advantages * tokens * keep).sum() / mask.sum()
    np.testing.assert_allclose(float(metrics.loss), expected, rtol=1e-5)
    np.testing.assert_allclose(float(metrics.other_metrics["off_policy/accepted_fraction"]), 0.75, rtol=1e-6)


@pytest.mark.parametrize(("min_acceptance", "skipped"), [(1.0, True), (0.5, False)])
def test_acceptance_gate_leaves_params_and_optimizer_state_bit_identical(min_acceptance, skipped):
    shift = np.zeros((4, 3))
    shift[1] = 2.0
    batch, bias, _ = _batch(shift)
    tx = optax.adamw(1e-2)
    # Warm the optimizer once so its moments are non-trivial.
    opt_state = tx.init(bias)
    _, opt_state = tx.update(jnp.ones_like(bias), opt_state, bias)
    state = _TrainState(bias, opt_state, jnp.asarray(3), tx, _mesh())
    before = jax.tree_util.tree_map(np.asarray, (state.graphstate, state.opt_state, state.step))

    new_state, metrics = grpo_step(
        state,
        batch,
        2,
        0.0,
        partition_spec=PartitionSpec(),
        is_training=True,
        loss_type="dapo",
        off_policy_k3_threshold=0.5,
        off_policy_min_acceptance=min_acceptance,
    )
    after = jax.tree_util.tree_map(np.asarray, (new_state.graphstate, new_state.opt_state, new_state.step))

    assert float(metrics.other_metrics["off_policy/step_skipped"]) == float(skipped)
    leaves_before = jax.tree_util.tree_leaves(before)
    leaves_after = jax.tree_util.tree_leaves(after)
    if skipped:
        for a, b in zip(leaves_before, leaves_after, strict=True):
            np.testing.assert_array_equal(a, b)
    else:
        assert int(after[2]) == 4
        assert not np.array_equal(before[0], after[0])


def test_startup_parity_gate_skips_first_update_when_behaviour_drifts():
    batch, bias, _ = _batch(np.full((4, 3), 0.5))
    batch["behavior_parity_gate"] = jnp.asarray(1, dtype=jnp.int32)
    tx = optax.adamw(1e-2)
    state = _TrainState(bias, tx.init(bias), jnp.asarray(0), tx, _mesh())
    new_state, metrics = grpo_step(
        state,
        batch,
        2,
        0.0,
        partition_spec=PartitionSpec(),
        is_training=True,
        loss_type="dapo",
        startup_behavior_k3_max=0.01,
    )
    assert float(metrics.other_metrics["off_policy/step_skipped"]) == 1.0
    np.testing.assert_array_equal(np.asarray(new_state.graphstate), np.asarray(bias))
    with pytest.raises(RuntimeError, match="not in sync"):
        check_startup_behavior_parity(metrics.other_metrics, k3_max=0.01, tail_fraction_max=0.01)

    # Once the host check has run (flag 0) the same drift no longer blocks updates.
    batch["behavior_parity_gate"] = jnp.asarray(0, dtype=jnp.int32)
    new_state, metrics = grpo_step(
        state,
        batch,
        2,
        0.0,
        partition_spec=PartitionSpec(),
        is_training=True,
        loss_type="dapo",
        startup_behavior_k3_max=0.01,
    )
    assert float(metrics.other_metrics["off_policy/step_skipped"]) == 0.0
    assert not np.array_equal(np.asarray(new_state.graphstate), np.asarray(bias))


@pytest.mark.cpu_ok
def test_off_policy_config_validation():
    cfg = GRPOConfig(off_policy_k3_threshold=0.1, off_policy_min_acceptance=0.5, startup_behavior_k3_max=0.02)
    assert (cfg.off_policy_k3_threshold, cfg.off_policy_min_acceptance) == (0.1, 0.5)
    assert GRPOConfig().discard_unscorable_groups is None
    for knob in ("off_policy_k3_threshold", "off_policy_min_acceptance", "startup_behavior_k3_max"):
        with pytest.raises(ValueError, match=knob):
            GRPOConfig(**{knob: 0.0})
        with pytest.raises(ValueError, match="esurge_importance_sampling_correction"):
            GRPOConfig(**{knob: 0.5}, esurge_importance_sampling_correction=False)
    with pytest.raises(ValueError, match="off_policy_min_acceptance"):
        GRPOConfig(off_policy_min_acceptance=1.5)
    with pytest.raises(ValueError, match="startup_behavior_tail_fraction_max"):
        GRPOConfig(startup_behavior_tail_fraction_max=0.0)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("knob", ["off_policy_min_acceptance", "startup_behavior_k3_max"])
def test_update_gates_are_rejected_under_mpmd_scheduler(knob):
    from spectrax.runtime.schedules import Std1F1B

    # The scheduled step applies every update, so an update gate there would be silently skipped.
    with pytest.raises(ValueError, match=f"{knob}.*SPMD"):
        GRPOConfig(mpmd_scheduler=Std1F1B(microbatches=2), **{knob: 0.5})
    # Row rejection runs inside the loss and stays supported under the scheduler.
    cfg = GRPOConfig(mpmd_scheduler=Std1F1B(microbatches=2), off_policy_k3_threshold=0.5)
    assert cfg.off_policy_k3_threshold == 0.5
    assert GRPOConfig(**{knob: 0.5}).mpmd_scheduler is None
