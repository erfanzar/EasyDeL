# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Shared auxiliary-loss-free top-k expert selection.

Ten families implemented this routing pipeline independently. Three of them —
``glm4_moe``, ``glm4_moe_lite``, ``glm_moe_dsa`` — had drifted into dropping
``e_score_correction_bias`` entirely: their select hook took ``pre_bias_logits``
and immediately ``del``'d it, so the trained load-balancing term never reached
routing. ``hy_v3`` and ``minimax_m3_vl`` threaded the same term correctly, which
is what marks it as drift rather than design.

The load-bearing invariant is that the bias steers *selection only*. Weights are
gathered from the unbiased scores, which is what makes the routing
auxiliary-loss-free: expert load moves without perturbing the combine.
"""

import jax
import numpy as np
import pytest
from easydel.layers.moe import moe_group_topk_select
from jax import numpy as jnp

EXPERTS = 16
TOKENS = 12
GROUPED = dict(
    n_routed_experts=EXPERTS,
    score_fn="sigmoid",
    n_group=4,
    topk_group=2,
    group_topk_k=2,
    group_score="topk_sum",
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
)


def _logits(seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((TOKENS, EXPERTS)), jnp.float32)


def _bias(seed=1, scale=2.0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((EXPERTS,)) * scale, jnp.float32)


def test_zero_bias_matches_no_bias():
    """A zero bias must be indistinguishable from passing none.

    This is the no-regression guarantee for the migrated families: a freshly
    initialised model routes exactly as it did before the fix.
    """
    logits = _logits()
    zero = jnp.zeros((EXPERTS,), jnp.float32)

    w_none, i_none = moe_group_topk_select(logits, None, 4, e_score_correction_bias=None, **GROUPED)
    w_zero, i_zero = moe_group_topk_select(logits, None, 4, e_score_correction_bias=zero, **GROUPED)

    assert jnp.array_equal(i_none, i_zero)
    assert jnp.array_equal(w_none, w_zero)


def test_nonzero_bias_changes_selection():
    """The regression the GLM families carried: a live bias must matter."""
    logits = _logits()
    _, unbiased = moe_group_topk_select(logits, None, 4, e_score_correction_bias=None, **GROUPED)
    _, biased = moe_group_topk_select(logits, None, 4, e_score_correction_bias=_bias(), **GROUPED)

    assert not jnp.array_equal(unbiased, biased)


def test_weights_come_from_unbiased_scores():
    """Selection may follow the bias; the returned weights may not include it."""
    logits = _logits()
    bias = _bias()
    weights, indices = moe_group_topk_select(logits, None, 4, e_score_correction_bias=bias, **GROUPED)

    scores = jax.nn.sigmoid(logits.astype(jnp.float32))
    expected = jnp.take_along_axis(scores, indices, axis=-1)
    expected = expected / (jnp.sum(expected, axis=-1, keepdims=True) + 1e-20)
    expected = expected * GROUPED["routed_scaling_factor"]

    assert jnp.allclose(weights, expected)
    # And emphatically NOT the biased scores.
    biased = jnp.take_along_axis(scores + bias, indices, axis=-1)
    biased = biased / (jnp.sum(biased, axis=-1, keepdims=True) + 1e-20) * GROUPED["routed_scaling_factor"]
    assert not jnp.allclose(weights, biased)


def test_group_restriction_confines_picks_to_surviving_groups():
    """Selected experts must all fall inside the kept groups."""
    logits = _logits(3)
    k = 4
    _, indices = moe_group_topk_select(logits, None, k, e_score_correction_bias=None, **GROUPED)

    group_size = EXPERTS // GROUPED["n_group"]
    groups_used = np.unique(np.asarray(indices) // group_size, axis=-1)
    # Each token may draw from at most `topk_group` distinct groups.
    for row in np.asarray(indices):
        assert len(set(row // group_size)) <= GROUPED["topk_group"]
    assert groups_used.size > 0


def test_single_group_disables_the_group_stage():
    """``n_group=1`` must reduce to a plain flat top-k over all experts."""
    logits = _logits(5)
    flat = dict(GROUPED, n_group=1, topk_group=1)
    weights, indices = moe_group_topk_select(logits, None, 4, e_score_correction_bias=None, **flat)

    scores = jax.nn.sigmoid(logits.astype(jnp.float32))
    expected_idx = jax.lax.top_k(scores, k=4)[1]
    assert jnp.array_equal(indices, expected_idx)
    assert weights.shape == (TOKENS, 4)


def test_max_group_score_differs_from_topk_sum():
    """DeepSeek-V2 scores a group by its best member, not a top-2 sum."""
    logits = jnp.asarray([[9.0, -10.0, 6.0, 5.0]])
    config = dict(GROUPED, n_routed_experts=4, score_fn="none", n_group=2, topk_group=1)
    by_sum = moe_group_topk_select(logits, None, 1, e_score_correction_bias=None, **config)[1]
    by_max = moe_group_topk_select(logits, None, 1, e_score_correction_bias=None, **dict(config, group_score="max"))[1]
    np.testing.assert_array_equal(by_sum, [[2]])  # 6 + 5 > 9 - 10
    np.testing.assert_array_equal(by_max, [[0]])  # 9 > 6


@pytest.mark.parametrize("norm_eps", [0.0, 1e-20])
def test_norm_eps_is_respected(norm_eps):
    """``minimax_m3_vl`` normalises with a bare sum; others add 1e-20."""
    logits = _logits(9)
    weights, indices = moe_group_topk_select(
        logits, None, 4, e_score_correction_bias=None, **dict(GROUPED, norm_eps=norm_eps)
    )
    scores = jax.nn.sigmoid(logits.astype(jnp.float32))
    raw = jnp.take_along_axis(scores, indices, axis=-1)
    expected = raw / (jnp.sum(raw, axis=-1, keepdims=True) + norm_eps) * GROUPED["routed_scaling_factor"]
    assert jnp.array_equal(weights, expected)


def test_norm_topk_prob_off_leaves_weights_unnormalised():
    logits = _logits(11)
    weights, indices = moe_group_topk_select(
        logits, None, 4, e_score_correction_bias=None, **dict(GROUPED, norm_topk_prob=False)
    )
    scores = jax.nn.sigmoid(logits.astype(jnp.float32))
    expected = jnp.take_along_axis(scores, indices, axis=-1) * GROUPED["routed_scaling_factor"]
    assert jnp.array_equal(weights, expected)


def test_rejects_unknown_score_fn():
    with pytest.raises(ValueError, match="unknown score_fn"):
        moe_group_topk_select(_logits(), None, 2, n_routed_experts=EXPERTS, score_fn="tanh")


def test_rejects_unknown_group_score():
    with pytest.raises(ValueError, match="unknown group_score"):
        moe_group_topk_select(_logits(), None, 2, n_routed_experts=EXPERTS, n_group=4, topk_group=2, group_score="mean")


MIGRATED = ["glm4_moe", "glm4_moe_lite", "glm_moe_dsa", "hy_v3", "minimax_m3_vl"]


@pytest.mark.parametrize("family", MIGRATED)
def test_family_routes_through_the_shared_selector(family):
    """No migrated family may carry its own copy of the selector again."""
    import importlib

    module = importlib.import_module(f"easydel.modules.{family}.modeling_{family}")
    source = open(module.__file__).read()
    assert "_select_experts_static" not in source
    assert "moe_group_topk_select" in source


@pytest.mark.parametrize("family", ["glm4_moe", "glm4_moe_lite", "glm_moe_dsa"])
def test_glm_families_bind_the_hook_per_call(family):
    """The three drifted families must read the bias at call time.

    Binding ``e_score_correction_bias`` once in ``__init__`` freezes the
    zero-initialised tensor — that is precisely how the term stopped reaching
    routing. ``hy_v3`` and ``minimax_m3_vl`` already rebound in ``forward``, so
    only the GLM trio needed the dedicated accessor.
    """
    import importlib

    module = importlib.import_module(f"easydel.modules.{family}.modeling_{family}")
    source = open(module.__file__).read()
    assert "self._select_hook()" in source


@pytest.mark.parametrize("family", ["hy_v3", "minimax_m3_vl"])
def test_already_correct_families_still_thread_the_live_bias(family):
    """These two were never broken; the migration must not regress them."""
    import importlib

    module = importlib.import_module(f"easydel.modules.{family}.modeling_{family}")
    source = open(module.__file__).read()
    assert "e_score_correction_bias=self" in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_excluded_negative_score_group_cannot_win():
    """Masking must work even when every valid routing score is negative."""
    logits = jnp.asarray([[-1.0, -2.0, -3.0, -4.0]], dtype=jnp.float32)
    _, indices = moe_group_topk_select(
        logits,
        None,
        1,
        n_routed_experts=4,
        score_fn="none",
        n_group=2,
        topk_group=1,
        group_score="max",
        norm_topk_prob=False,
    )
    assert np.asarray(indices).tolist() == [[0]]


@pytest.mark.parametrize("family", ["glm4_moe", "glm4_moe_lite", "glm_moe_dsa"])
def test_family_select_hook_observes_changed_live_bias(family):
    """Execute the actual family hook, not a source-string surrogate."""
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module(f"easydel.modules.{family}.modeling_{family}")
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == module.__name__ and "_select_hook" in value.__dict__
    ]
    assert len(classes) == 1
    bias_parameter = SimpleNamespace(value=jnp.zeros((EXPERTS,)))
    owner = SimpleNamespace(
        n_routed_experts=EXPERTS,
        group_topk_k=2,
        config=SimpleNamespace(n_group=4, topk_group=2, norm_topk_prob=True, routed_scaling_factor=2.5),
        gate=SimpleNamespace(e_score_correction_bias=bias_parameter),
    )
    logits = _logits(13)
    _, original = classes[0]._select_hook(owner)(logits, None, 4)
    bias_parameter.value = _bias(14, scale=10.0)
    weights, selected = classes[0]._select_hook(owner)(logits, None, 4)
    assert not np.array_equal(original, selected)
    unbiased = jnp.take_along_axis(jax.nn.sigmoid(logits), selected, axis=-1)
    expected = unbiased / jnp.sum(unbiased, axis=-1, keepdims=True) * 2.5
    np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-6)
