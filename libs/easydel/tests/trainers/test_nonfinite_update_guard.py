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

"""``LossConfig.skip_nonfinite_updates`` makes ``update_state_respectfully`` skip non-finite steps."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers import training_utils


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _FakeState:
    step: jax.Array
    params: jax.Array

    def apply_gradients(self, grads):
        return _FakeState(step=self.step + 1, params=self.params - grads["w"])


def _state():
    return _FakeState(step=jnp.asarray(0, jnp.int32), params=jnp.ones((3,), jnp.float32))


def _run(grads, loss, config, *, fast_compile):
    metrics = LossMetrics(loss=jnp.asarray(loss, jnp.float32))
    original = training_utils.FAST_COMPILE
    training_utils.FAST_COMPILE = fast_compile
    try:
        return training_utils.update_state_respectfully(_state(), grads, config, metrics)
    finally:
        training_utils.FAST_COMPILE = original


@pytest.mark.parametrize("fast_compile", [True, False])
@pytest.mark.parametrize("bad", [jnp.inf, jnp.nan])
def test_guard_skips_update_on_nonfinite_gradient(fast_compile, bad):
    grads = {"w": jnp.asarray([0.5, bad, 0.5], jnp.float32)}
    new = _run(grads, 1.0, LossConfig(skip_nonfinite_updates=True), fast_compile=fast_compile)

    assert int(new.step) == 0
    assert bool(jnp.all(new.params == 1.0))


@pytest.mark.parametrize("fast_compile", [True, False])
def test_guard_skips_update_on_infinite_loss(fast_compile):
    grads = {"w": jnp.full((3,), 0.5, jnp.float32)}
    new = _run(grads, float("inf"), LossConfig(skip_nonfinite_updates=True), fast_compile=fast_compile)

    assert int(new.step) == 0


@pytest.mark.parametrize("fast_compile", [True, False])
def test_guard_applies_finite_update(fast_compile):
    grads = {"w": jnp.full((3,), 0.5, jnp.bfloat16)}
    new = _run(grads, 1.0, LossConfig(skip_nonfinite_updates=True), fast_compile=fast_compile)

    assert int(new.step) == 1
    assert bool(jnp.allclose(new.params, 0.5))


def test_break_on_nan_skips_nan_loss_but_not_infinite_loss():
    """Without the opt-in guard, ``break_on_nan`` keeps its NaN-only meaning."""
    grads = {"w": jnp.full((3,), 0.5, jnp.float32)}
    config = LossConfig(break_on_nan=True)

    assert int(_run(grads, float("nan"), config, fast_compile=False).step) == 0
    assert int(_run(grads, float("inf"), config, fast_compile=False).step) == 1


def test_fast_compile_without_flag_keeps_unguarded_path():
    grads = {"w": jnp.asarray([0.5, jnp.inf, 0.5], jnp.float32)}
    new = _run(grads, 1.0, LossConfig(), fast_compile=True)

    assert int(new.step) == 1
