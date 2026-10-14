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

"""Regression tests for two GRPO-trainer correctness fixes.

1. ``RewardProtocol.reduction`` now defaults to ``"mean"`` (the standard GRPO
   group baseline, matching the plain-callable path) instead of ``"sum"``.
2. ``_global_token_normalizer`` divides the full-batch token count by
   ``gradient_accumulation_steps`` so the ``dapo``/``cispo``/``vespo``/``dppo``
   loss types are not silently scaled by ``1/K`` under gradient accumulation.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from easydel.trainers.group_relative_policy_optimization._fn import _global_token_normalizer
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import (
    GRPOTrainer,
    _compute_rewards_and_advantages,
)
from easydel.trainers.reward_protocol import RewardProtocol


# Bug 1: RewardProtocol.reduction defaults to "mean"
def test_default_reduction_is_mean():
    """A RewardProtocol subclass that does not set ``reduction`` uses ``"mean"``.

    The default used to be ``"sum"``, which made the group baseline the group
    *sum* (non-centered, and blowing up under low reward variance once divided
    by the group std). The standard GRPO baseline — and the plain-callable
    default — is the group mean.
    """

    class DefaultReward(RewardProtocol):
        def compute(self, **kwargs):
            return 0.0

    assert DefaultReward().reduction == "mean"
    assert GRPOTrainer._resolve_group_reduction([DefaultReward()]) == "mean"


def test_default_reduction_gives_centered_advantages():
    """With the (new) default reduction, advantages are zero-centered per group."""

    class DefaultReward(RewardProtocol):
        def compute(self, **kwargs):
            return 0.0

    rewards_per_func = jnp.array([[1.0], [2.0], [3.0], [4.0]], dtype="f4")
    _, advantages, _, _ = _compute_rewards_and_advantages(
        rewards_per_func=rewards_per_func,
        reward_weights=jnp.array([1.0], dtype="f4"),
        generation_factor=4,
        scale_rewards="none",
        multi_objective_aggregation="sum_then_normalize",
        arguments=SimpleNamespace(advantage_estimator="group_mean", reward_clip_range=None),
        group_reduction=GRPOTrainer._resolve_group_reduction([DefaultReward()]),
    )
    adv = np.asarray(advantages)
    # mean baseline of [1,2,3,4] is 2.5 -> centered advantages summing to 0.
    np.testing.assert_allclose(adv, np.array([1.0, 2.0, 3.0, 4.0]) - 2.5, rtol=0, atol=1e-6)
    np.testing.assert_allclose(adv.sum(), 0.0, atol=1e-5)


# Bug 2: global-token normalizer is accumulation-aware
def test_global_token_normalizer_divides_by_accum_steps():
    """The global normalizer is divided by K so ``÷num_accum_steps`` recombines correctly."""
    num_items = jnp.asarray(32.0)
    local = jnp.asarray(8.0)

    # K == 1: no averaging in minibatch_call -> normalizer unchanged.
    assert float(_global_token_normalizer(num_items, local, False, 1)) == 32.0
    # K == 4: minibatch_call averages by 1/4 -> global count pre-divided by 4.
    assert float(_global_token_normalizer(num_items, local, False, 4)) == 8.0


def test_global_token_normalizer_falls_back_to_local():
    """Truncated windows / missing global count use the microbatch-local count (never divided)."""
    local = jnp.asarray(8.0)
    # truncated -> local, even with K>1.
    assert float(_global_token_normalizer(jnp.asarray(32.0), local, True, 4)) == 8.0
    # missing num_items_in_batch -> local, not divided.
    assert float(_global_token_normalizer(None, local, False, 4)) == 8.0


def test_global_token_normalizer_recombines_under_accumulation():
    """K microbatch losses normalized with the fix average back to the full-batch loss.

    Emulates ``minibatch_call``: each of K microbatches computes
    ``sum(local tokens) / normalizer`` and the gradients/metrics are averaged
    by ``1/K``. The fixed normalizer must make that average equal the
    single-shot full-batch value ``S_total / N_global``.
    """
    per_microbatch_token_sums = [3.0, 5.0, 2.0, 6.0]  # S_k
    n_global = float(sum(per_microbatch_token_sums))  # 16 tokens total
    k = len(per_microbatch_token_sums)

    full_batch_loss = n_global / n_global  # S_total / N_global == 1.0 (uniform per-token loss)

    normalizer = float(_global_token_normalizer(jnp.asarray(n_global), jnp.asarray(0.0), False, k))
    accumulated = sum(s_k / normalizer for s_k in per_microbatch_token_sums) / k  # ÷K average
    np.testing.assert_allclose(accumulated, full_batch_loss, atol=1e-6)
