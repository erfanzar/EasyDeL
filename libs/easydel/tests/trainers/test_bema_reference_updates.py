# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BEMA reference-update math and snapshot-ownership tests.

BEMA keeps three parameter-sized snapshots (``theta0``, ``ema``, ``bema``). These
tests pin the smoothing math against an independent computation and assert the
ownership rules that keep the snapshot count at three: the stored snapshots never
alias the live policy graphstate (whose buffers the compiled step donates), while
the reference model shares the BEMA snapshot instead of copying it.
"""

from __future__ import annotations

import jax
from easydel.trainers.bema_trainer.bema_config import BEMACallback
from jax import numpy as jnp


def _graphstate(value: float):
    """Build a minimal graphstate-shaped pytree with one array and one metadata leaf."""
    return {"w": jnp.full((4, 2), value, dtype=jnp.float32), "name": "layer"}


def _reference_state(graphstate):
    """Minimal stand-in exposing the ``replace``/``graphstate`` surface BEMA uses."""

    class _State:
        def __init__(self, graphstate):
            self.graphstate = graphstate

        def replace(self, *, graphstate):
            return _State(graphstate)

    return _State(graphstate)


class _PolicyState:
    """Policy stand-in carrying only the graphstate BEMA reads."""

    def __init__(self, graphstate):
        self.graphstate = graphstate


def test_first_eligible_step_snapshots_do_not_alias_the_policy():
    callback = BEMACallback(update_after=0, update_freq=1, min_ema_multiplier=0.0)
    policy = _graphstate(1.0)

    bema = callback.update_graphstate(policy, step=0)

    assert jnp.array_equal(bema["w"], policy["w"])
    # theta0/ema must own their arrays: the policy's buffers are donated by the
    # compiled train step and would be invalid on the next update otherwise.
    assert callback._theta0_graphstate["w"] is not policy["w"]
    assert callback._ema_graphstate["w"] is not policy["w"]
    assert bema["w"] is not policy["w"]


def test_second_update_matches_independent_bema_math():
    callback = BEMACallback(update_after=0, update_freq=1, min_ema_multiplier=0.0)
    theta0_value, current_value = 1.0, 3.0

    callback.update_graphstate(_graphstate(theta0_value), step=0)
    ema_before = callback._ema_graphstate["w"]

    step = 1
    beta = (callback.lag + callback.multiplier * step) ** (-callback.ema_power)
    alpha = (callback.lag + callback.multiplier * step) ** (-callback.bias_power)
    current = _graphstate(current_value)

    bema = callback.update_graphstate(current, step=step)

    expected_ema = (1.0 - beta) * ema_before + beta * current["w"]
    expected_bema = expected_ema + alpha * (current["w"] - theta0_value)
    assert jnp.allclose(callback._ema_graphstate["w"], expected_ema, atol=1e-6)
    assert jnp.allclose(bema["w"], expected_bema, atol=1e-6)
    # Snapshots stay independent of the policy arrays after the in-place-free update.
    assert callback._ema_graphstate["w"] is not current["w"]
    assert bema["w"] is not current["w"]
    assert jnp.array_equal(callback._theta0_graphstate["w"], jnp.full((4, 2), theta0_value))
    assert callback._theta0_graphstate["name"] == "layer"


def test_reference_shares_the_bema_snapshot_instead_of_copying_it():
    callback = BEMACallback(
        update_after=0,
        update_freq=1,
        update_ref_model=True,
        ref_model_update_after=0,
        ref_model_update_freq=1,
        min_ema_multiplier=0.0,
    )
    reference = _reference_state(_graphstate(0.0))

    updated = callback.update_reference_state(_PolicyState(_graphstate(1.0)), reference, step=0)

    assert updated is not reference
    for path, leaf in jax.tree_util.tree_leaves_with_path(updated.graphstate):
        del path
        assert any(leaf is snapshot for snapshot in jax.tree_util.tree_leaves(callback._bema_graphstate)), (
            "reference must share the BEMA snapshot's arrays, not a fourth copy"
        )

    # A later update rebinds both the snapshot and the reference; the previous
    # reference keeps the weights it was given.
    previous_graphstate = updated.graphstate
    refreshed = callback.update_reference_state(_PolicyState(_graphstate(5.0)), updated, step=1)
    assert refreshed.graphstate["w"] is not previous_graphstate["w"]
    assert jnp.array_equal(previous_graphstate["w"], jnp.full((4, 2), 1.0))


def test_reference_is_untouched_outside_the_update_cadence():
    callback = BEMACallback(
        update_after=0,
        update_freq=1,
        update_ref_model=True,
        ref_model_update_after=0,
        ref_model_update_freq=4,
        min_ema_multiplier=0.0,
    )
    reference = _reference_state(_graphstate(0.0))

    same = callback.update_reference_state(_PolicyState(_graphstate(1.0)), reference, step=1)

    assert same is reference
