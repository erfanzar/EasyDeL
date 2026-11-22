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

"""Discarded groups must not dilute the gradient under gradient accumulation."""

from types import SimpleNamespace

import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import _apply_discarded_groups
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec

VOCAB = 5


def _stub_logits(input_ids, bias):
    tilt = 0.1 * input_ids[:, :, None].astype(jnp.float32) * jnp.arange(VOCAB, dtype=jnp.float32)[None, None, :]
    return bias[None, : input_ids.shape[1], :] + tilt


class _Model:
    def __init__(self, bias, mesh):
        self.bias = bias
        self.mesh = mesh

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=_stub_logits(input_ids, self.bias))


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


def _batch():
    """Two prompt groups of two completions; one prompt row per completion so microbatches slice cleanly."""
    rng = np.random.default_rng(7)
    prompt_ids = np.array([[1], [1], [3], [3]], dtype=np.int32)
    completion_ids = np.array([[0, 2, 4], [1, 1, 3], [4, 0, 2], [3, 2, 1]], dtype=np.int32)
    mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=np.float32)
    old = rng.uniform(-2.5, -0.5, size=mask.shape).astype(np.float32)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.ones_like(jnp.asarray(prompt_ids)),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(mask),
        "advantages": jnp.asarray([1.5, -0.7, 0.4, -1.2], dtype=jnp.float32),
        "old_per_token_logps": jnp.asarray(old),
        "num_items_in_batch": jnp.asarray(mask.sum()),
    }
    bias = jnp.asarray(rng.normal(size=(4, VOCAB)).astype(np.float32))
    return batch, bias


def _update(bias, batch, accumulation, loss_type):
    """Parameter delta of one SGD(lr=1) step, i.e. minus the step's gradient."""
    tx = optax.sgd(1.0)
    state = _TrainState(bias, tx.init(bias), jnp.asarray(0), tx, Mesh(np.array(jax.devices()[:1]), ("dp",)))
    new_state, _ = grpo_step(
        state,
        batch,
        1,
        0.0,
        partition_spec=PartitionSpec(),
        gradient_accumulation_steps=accumulation,
        is_training=True,
        loss_type=loss_type,
        epsilon=0.3,
        epsilon_high=0.3,
    )
    return np.asarray(new_state.graphstate) - np.asarray(bias)


@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "bnpo", "dr_grpo", "luspo"])
def test_fully_discarded_microbatch_does_not_dilute_accumulated_gradient(loss_type):
    batch, bias = _batch()
    # Group 0 (rows 0-1, exactly microbatch 0 at K=2) is unscorable.
    discarded, _ = _apply_discarded_groups(batch, jnp.asarray([True, True, False, False]), 2)
    kept = {key: value[2:] for key, value in batch.items() if key != "num_items_in_batch"}
    kept["num_items_in_batch"] = jnp.sum(kept["completion_mask"])

    accumulated = _update(bias, discarded, 2, loss_type)
    reference = _update(bias, kept, 1, loss_type)

    assert np.abs(reference).max() > 1e-4
    np.testing.assert_allclose(accumulated, reference, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("loss_type", ["grpo", "dapo"])
def test_uneven_discard_matches_full_batch_gradient_under_accumulation(loss_type):
    batch, bias = _batch()
    # Row 1 is zero-weighted (one microbatch is half discarded): K=2 must equal K=1 on the same batch.
    discarded, _ = _apply_discarded_groups(batch, jnp.asarray([False, True, False, False]), 1)
    np.testing.assert_allclose(
        _update(bias, discarded, 2, loss_type), _update(bias, discarded, 1, loss_type), rtol=1e-5, atol=1e-7
    )
