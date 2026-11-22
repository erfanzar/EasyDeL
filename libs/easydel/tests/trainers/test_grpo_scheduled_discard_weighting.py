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

"""Scheduled (pp=2 pipeline) GRPO with ``difficulty_weights`` matches the SPMD step.

Discarded groups and dataset weights must weight pipeline and accumulation microbatches by their
loss denominators, so a fully discarded microbatch does not dilute the gradient, and the gated
update's ``off_policy/*`` diagnostics stay whole-batch values.
"""

from __future__ import annotations

import easydel as ed
import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from easydel.trainers import training_utils
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import _apply_discarded_groups
from jax import numpy as jnp
from jax.sharding import PartitionSpec

ROWS = 8
PROMPT_LEN = 4
COMPLETION_LEN = 6


def _config(pipeline: bool):
    return ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sharding_axis_dims=(2, 1, 1, -1, 1, 1) if pipeline else (1, 1, -1, 1, 1, 1),
        pipeline_stage_regions=pipeline,
        scan_layers=False,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        pad_token_id=0,
    )


def _init(config):
    return ed.LlamaForCausalLM.sequential_init(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    ).shard_model()


@pytest.fixture(scope="module")
def models():
    if jax.device_count() < 2:
        pytest.skip("pp=2 pipeline tests need at least 2 devices")
    # The pipeline runtime compiles stages on worker threads, which do not see a thread-local
    # ``jax.default_matmul_precision`` context; set full precision process-wide for this module.
    previous = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    pipeline = _init(_config(True))
    assert bool(getattr(pipeline.mesh, "is_mpmd", False))
    reference = _init(_config(False))
    source = {path: np.asarray(jax.device_get(var.value)) for path, var in spx.iter_variables(pipeline)}
    targets = dict(spx.iter_variables(reference))
    assert targets.keys() == source.keys()
    for path, var in targets.items():
        var.value = jax.device_put(source[path], var.value.sharding)
    yield pipeline, reference
    jax.config.update("jax_default_matmul_precision", previous)


def _batch(weights=None, discarded=None, behavior=False):
    """One prompt row per completion so pipeline and accumulation microbatches slice on one axis.

    No ``old_per_token_logps``: the ratio is ``1``, so no token sits at a clip boundary where tiny
    pipeline/SPMD numerical differences could flip a clip decision.
    """
    rng = np.random.default_rng(3)
    prompt_ids = rng.integers(1, 127, (ROWS, PROMPT_LEN)).astype(np.int32)
    lengths = np.array([6, 3, 5, 2, 6, 4, 1, 5])
    mask = (np.arange(COMPLETION_LEN)[None, :] < lengths[:, None]).astype(np.float32)
    completion_ids = np.where(mask > 0, rng.integers(1, 127, (ROWS, COMPLETION_LEN)), 0).astype(np.int32)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.ones_like(jnp.asarray(prompt_ids)),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(mask),
        "advantages": jnp.asarray(rng.standard_normal(ROWS).astype(np.float32)),
    }
    if behavior:
        # Around the tiny model's ~log(1/128) log-probs, so some rows pass and some fail the K3 bound.
        batch["behavior_per_token_logps"] = jnp.asarray(rng.uniform(-5.5, -4.2, mask.shape).astype(np.float32))
    if weights is not None:
        batch["difficulty_weights"] = jnp.asarray(weights, jnp.float32)
    if discarded is not None:
        flags = np.zeros(ROWS, bool)
        flags[list(discarded)] = True
        batch, _ = _apply_discarded_groups(batch, jnp.asarray(flags), 2)
    batch.pop("num_items_in_batch", None)
    return batch


def _leaves(tree):
    return [np.asarray(jax.device_get(leaf), np.float64) for leaf in jax.tree_util.tree_leaves(tree)]


def _spmd_gradient(model, batch, **kwargs):
    """Full-batch (K=1) SPMD ``grpo_step`` gradient via one SGD(lr=1) step."""
    with jax.default_matmul_precision("highest"), model.config.mesh:
        state = model.to_state().init_tx(optax.sgd(1.0))
        before = _leaves(state.graphstate)
        new_state, metrics = grpo_step(state, batch, 1, 0.0, partition_spec=PartitionSpec(), is_training=True, **kwargs)
        after = _leaves(new_state.graphstate)
    return [b - a for b, a in zip(before, after, strict=True)], metrics


class _RecordingTx:
    """Stage-local optimizer stand-in that records the gradients the scheduled step applies."""

    def __init__(self):
        self.gradients = None

    def apply_gradients_stage_local(self, *, params, grads, opt_state, **kwargs):
        self.gradients = _leaves(grads)
        return params, opt_state


def _scheduled_gradient(model, batch, accumulation, **kwargs):
    tx = _RecordingTx()
    state = model.to_state().replace(tx=tx, opt_state=())
    compiled = training_utils.compile_trainer_step(
        grpo_step, mesh=model.mesh, schedule=spx.GPipe(microbatches=2), arguments=None
    )
    with jax.default_matmul_precision("highest"):
        _, metrics = compiled(
            state,
            batch,
            1,
            0.0,
            learning_rate_fn=optax.constant_schedule(1.0),
            gradient_accumulation_steps=accumulation,
            **kwargs,
        )
    assert tx.gradients is not None, "the scheduled step did not apply gradients"
    return tx.gradients, metrics


def _assert_gradients_match(scheduled, reference):
    assert len(scheduled) == len(reference)
    scale = max(float(np.abs(leaf).max()) for leaf in reference)
    assert scale > 1e-6
    for got, want in zip(scheduled, reference, strict=True):
        assert got.shape == want.shape
        # Measured on TPU with fp32 attention: max |scheduled - SPMD| <= 3.3e-7 * scale over every case.
        # The defect this guards against (a discarded microbatch diluting the update) changes the
        # gradient by tens of percent.
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * scale)


# (accumulation steps, discarded rows, dataset weights). GPipe(2) splits each accumulation chunk in two.
_WEIGHTED_CASES = {
    # Pipeline microbatch 0 (rows 0-3) fully discarded.
    "k1_first_microbatch_discarded": (1, (0, 1, 2, 3), None),
    # Accumulation chunk 0 has a fully discarded pipeline microbatch next to a kept one.
    "k2_first_pipeline_microbatch_discarded": (2, (0, 1), None),
    # Accumulation chunk 0 fully discarded (its pipeline call falls back to uniform, scale 0 outside).
    "k2_first_chunk_discarded": (2, (0, 1, 2, 3), None),
    # Dataset weights with a discarded group.
    "k2_dataset_weights": (2, (4, 5), [1.0, 2.5, 1.5, 0.5, 1.0, 1.0, 3.0, 2.0]),
}


@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "dr_grpo"])
@pytest.mark.parametrize("case", list(_WEIGHTED_CASES))
def test_scheduled_weighted_grpo_matches_spmd_gradient(models, loss_type, case):
    pipeline, reference = models
    accumulation, discarded, weights = _WEIGHTED_CASES[case]
    batch = _batch(weights=weights, discarded=discarded)
    assert "difficulty_weights" in batch
    kwargs = dict(loss_type=loss_type, epsilon=0.3, epsilon_high=0.3)
    want, spmd_metrics = _spmd_gradient(reference, batch, **kwargs)
    got, metrics = _scheduled_gradient(pipeline, batch, accumulation, **kwargs)
    _assert_gradients_match(got, want)
    np.testing.assert_allclose(float(metrics.loss), float(spmd_metrics.loss), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("accumulation", [1, 2])
def test_scheduled_unweighted_grpo_matches_spmd_gradient(models, accumulation):
    pipeline, reference = models
    batch = _batch()
    assert "difficulty_weights" not in batch
    kwargs = dict(loss_type="grpo", epsilon=0.3, epsilon_high=0.3)
    want, spmd_metrics = _spmd_gradient(reference, batch, **kwargs)
    got, metrics = _scheduled_gradient(pipeline, batch, accumulation, **kwargs)
    _assert_gradients_match(got, want)
    np.testing.assert_allclose(float(metrics.loss), float(spmd_metrics.loss), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("accumulation", [1, 2])
def test_scheduled_gated_weighted_grpo_reports_whole_batch_diagnostics(models, accumulation):
    pipeline, reference = models
    batch = _batch(weights=[1.0, 2.5, 1.5, 0.5, 1.0, 1.0, 3.0, 2.0], discarded=(0, 1), behavior=True)
    kwargs = dict(loss_type="dapo", off_policy_k3_threshold=0.07, off_policy_min_acceptance=0.0)
    want, spmd_metrics = _spmd_gradient(reference, batch, **kwargs)
    got, metrics = _scheduled_gradient(pipeline, batch, accumulation, **kwargs)
    spmd_other = spmd_metrics.other_metrics
    other = metrics.other_metrics
    assert float(other["off_policy/step_skipped"]) == float(spmd_other["off_policy/step_skipped"]) == 0.0
    # Microbatches differ in K3, so a weighted (rather than additive) reduction would move these values.
    assert float(spmd_other["off_policy/behavior_k3"]) > 0.0
    for key in ("off_policy/accepted_fraction", "off_policy/behavior_k3", "off_policy/tail_fraction"):
        np.testing.assert_allclose(float(other[key]), float(spmd_other[key]), rtol=1e-5, atol=1e-7, err_msg=key)
    _assert_gradients_match(got, want)
