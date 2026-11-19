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

"""CPU tests for ordering-safe reward composition (pure Python, no array computation)."""

from __future__ import annotations

import numpy as np
import pytest
from easydel.environments import (
    RewardShaper,
    compose_outcome_reward,
    correct_answer_floor,
    count_think_tokens,
    gate_invalid,
    lexicographic_blend,
    structural_validity,
    think_budget_penalty,
    validate_structural_config,
)

pytestmark = pytest.mark.cpu_ok

EOS, TURN_END, THINK_CLOSE, RESTART = 2, 3, 4, 5


def test_correct_answer_floor_is_midpoint_and_validates():
    assert correct_answer_floor(0.5, 1.0) == pytest.approx(0.75)
    assert correct_answer_floor(-1.0, 3.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        correct_answer_floor(1.0, 1.0)
    with pytest.raises(ValueError):
        compose_outcome_reward(True, (), correct_reward=1.0, incorrect_ceiling=2.0)


def test_compose_outcome_reward_floor_and_ceiling():
    assert compose_outcome_reward(True, [-0.1]) == pytest.approx(0.9)
    assert compose_outcome_reward(True, [-5.0, None]) == pytest.approx(0.75)
    assert compose_outcome_reward(False, [0.3]) == pytest.approx(0.3)
    assert compose_outcome_reward(False, [5.0]) == pytest.approx(0.5)
    assert compose_outcome_reward(False, [-0.4]) == pytest.approx(-0.4)


def test_compose_outcome_reward_any_correct_outranks_every_incorrect():
    rng = np.random.default_rng(1234)
    for _ in range(300):
        ceiling = float(rng.uniform(-2.0, 2.0))
        correct_reward = ceiling + float(rng.uniform(1e-3, 3.0))
        correct = [
            compose_outcome_reward(
                True,
                rng.normal(0, 10, size=int(rng.integers(0, 5))).tolist(),
                correct_reward=correct_reward,
                incorrect_ceiling=ceiling,
            )
            for _ in range(8)
        ]
        incorrect = [
            compose_outcome_reward(
                False,
                rng.normal(0, 10, size=int(rng.integers(0, 5))).tolist(),
                correct_reward=correct_reward,
                incorrect_ceiling=ceiling,
                incorrect_reward=float(rng.uniform(-3, 3)),
            )
            for _ in range(8)
        ]
        assert min(correct) > max(incorrect)


def _validity(ids, **overrides):
    kwargs = dict(
        eos_ids=(EOS,),
        max_completion_length=8,
        forbidden_ids=(RESTART,),
        turn_end_ids=(TURN_END,),
        max_turn_ends=1,
        think_close_ids=(THINK_CLOSE,),
    )
    kwargs.update(overrides)
    return structural_validity(ids, **kwargs)


def test_structural_validity_flags():
    ok = _validity([7, THINK_CLOSE, 8, TURN_END, EOS])
    assert ok.valid and ok.has_eos and not ok.truncated and ok.think_ok

    truncated = _validity([7] * 8)
    assert truncated.truncated and not truncated.has_eos and not truncated.valid

    short_no_eos = _validity([7, 7])
    assert not short_no_eos.has_eos and not short_no_eos.truncated and not short_no_eos.valid

    restarted = _validity([7, RESTART, 8, EOS])
    assert restarted.restarted and not restarted.valid

    extra = _validity([7, TURN_END, 8, TURN_END, EOS])
    assert extra.extra_turn_ends and not extra.valid

    two_closes = _validity([THINK_CLOSE, 7, THINK_CLOSE, EOS])
    assert not two_closes.think_ok and not two_closes.valid

    no_close_lenient = _validity([7, EOS])
    assert no_close_lenient.think_ok and no_close_lenient.valid
    no_close_strict = _validity([7, EOS], require_single_think_close=True)
    assert not no_close_strict.think_ok and not no_close_strict.valid


def test_validate_structural_config_rejects_strict_without_ids():
    with pytest.raises(ValueError):
        validate_structural_config(eos_ids=(EOS,), max_completion_length=8, require_single_think_close=True)
    with pytest.raises(ValueError):
        validate_structural_config(eos_ids=(), max_completion_length=8)
    with pytest.raises(ValueError):
        validate_structural_config(eos_ids=(EOS,), max_completion_length=0)


def test_gate_invalid_keeps_only_downside():
    assert gate_invalid(0.9, True) == pytest.approx(0.9)
    assert gate_invalid(0.9, False) == 0.0
    assert gate_invalid(-0.3, False) == pytest.approx(-0.3)
    assert gate_invalid(0.9, False, cap=-1.0) == pytest.approx(-1.0)
    assert gate_invalid(-2.0, False, cap=-1.0) == pytest.approx(-2.0)


def test_lexicographic_blend_values():
    assert lexicographic_blend(7.0, [10.0, 0.0], primary_scale=1.0, secondary_max=10.0) == pytest.approx(7.25)
    assert lexicographic_blend(3.0, [], primary_scale=1.0) == pytest.approx(3.0)
    assert lexicographic_blend(2.0, [1.0, 0.0], [3.0, 1.0], primary_scale=2.0, band=0.5) == pytest.approx(2.75)
    with pytest.raises(ValueError):
        lexicographic_blend(1.0, [0.5], [1.0, 2.0])
    with pytest.raises(ValueError):
        lexicographic_blend(1.0, [0.5], band=1.0)


def test_lexicographic_blend_secondaries_never_outweigh_one_primary_unit():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        scale = float(rng.uniform(0.1, 5.0))
        band = float(rng.uniform(0.0, 0.999))
        k = int(rng.integers(1, 5))
        weights = rng.uniform(0.01, 3.0, size=k).tolist()
        primary_b = float(rng.uniform(-10, 10))
        primary_a = primary_b + scale * float(rng.uniform(1.0, 3.0))
        sec_a = rng.uniform(-2.0, 12.0, size=k).tolist()
        sec_b = rng.uniform(-2.0, 12.0, size=k).tolist()
        blend_a = lexicographic_blend(primary_a, sec_a, weights, primary_scale=scale, secondary_max=10.0, band=band)
        blend_b = lexicographic_blend(primary_b, sec_b, weights, primary_scale=scale, secondary_max=10.0, band=band)
        assert blend_a > blend_b
        assert blend_a >= primary_a and blend_b >= primary_b
        assert blend_b - primary_b <= band * scale + 1e-12
        # Equal primaries: the better secondary profile wins the tie.
        hi = lexicographic_blend(primary_b, [10.0] * k, weights, primary_scale=scale, secondary_max=10.0, band=band)
        lo = lexicographic_blend(primary_b, [0.0] * k, weights, primary_scale=scale, secondary_max=10.0, band=band)
        assert hi >= lo


def test_think_budget_penalty_piecewise():
    kw = dict(budget=100, min_fraction=0.1, tolerance=0.5, weight=0.2)
    assert think_budget_penalty(50, **kw) == 0.0
    assert think_budget_penalty(10, **kw) == 0.0
    assert think_budget_penalty(100, **kw) == 0.0
    assert think_budget_penalty(5, **kw) == pytest.approx(-0.05)
    assert think_budget_penalty(0, **kw) == pytest.approx(-0.2)
    assert think_budget_penalty(125, **kw) == pytest.approx(-0.1)
    assert think_budget_penalty(1000, **kw) == pytest.approx(-0.2)
    with pytest.raises(ValueError):
        think_budget_penalty(5, budget=0)


def test_count_think_tokens():
    assert count_think_tokens([7, 7, 7, THINK_CLOSE, 8], (THINK_CLOSE,)) == 3
    assert count_think_tokens([7, 7], (THINK_CLOSE,)) == 2


def test_reward_shaper_preserves_outcome_order_and_unscorable():
    def accuracy(completions, answer, **kwargs):
        return [None if a is None else float(c == a) for c, a in zip(completions, answer, strict=True)]

    def formatting(completions, **kwargs):
        # Pathological shaping: rewards the wrong answers enormously.
        return [10.0 if c == "wrong" else -10.0 for c in completions]

    shaper = RewardShaper(accuracy, shaping_funcs=[formatting])
    rewards = shaper(
        prompts=["p"] * 4,
        completions=["right", "wrong", "right", "x"],
        answer=["right", "right", "right", None],
    )
    assert rewards[0] == pytest.approx(0.75)
    assert rewards[1] == pytest.approx(0.5)
    assert rewards[2] == pytest.approx(0.75)
    assert rewards[3] is None
    assert shaper.__name__ == "shaped_accuracy"


def test_reward_shaper_structural_gate_and_think_budget():
    def always_correct(completions, **kwargs):
        return [1.0] * len(completions)

    shaper = RewardShaper(
        always_correct,
        structural={"eos_ids": (EOS,), "max_completion_length": 16, "think_close_ids": (THINK_CLOSE,)},
        think_budget={"budget": 4, "min_fraction": 0.5, "tolerance": 1.0, "weight": 0.2},
        think_close_ids=(THINK_CLOSE,),
    )
    ids = [
        [7, 7, 7, THINK_CLOSE, 8, EOS],  # 3 thinking tokens: inside [2, 4], valid
        [7, 7, 7, 7, 7, 7, THINK_CLOSE, EOS],  # 6 tokens: overshoot 2/4 -> -0.1
        [7, 7, THINK_CLOSE, 8],  # no EOS -> invalid, upside removed
    ]
    rewards = shaper(prompts=["p"] * 3, completions=["a", "b", "c"], completion_ids=ids)
    assert rewards == pytest.approx([1.0, 0.9, 0.0])
    with pytest.raises(ValueError):
        shaper(prompts=["p"], completions=["a"])


def test_compose_outcome_reward_rejects_non_finite_shaping():
    for bad in (float("nan"), np.float32("nan"), float("inf"), -np.inf):
        with pytest.raises(ValueError):
            compose_outcome_reward(True, [0.1, bad])


def test_reward_shaper_treats_numpy_nan_as_unscorable_or_zero_shaping():
    def base(completions, **kwargs):
        return [np.float32(1.0), np.float32("nan"), np.float64(0.0)]

    def shaping(completions, **kwargs):
        return [np.float32("nan"), np.float32(0.1), np.float32(0.2)]

    rewards = RewardShaper(base, shaping_funcs=[shaping])(prompts=["p"] * 3, completions=["a", "b", "c"])
    assert rewards[0] == pytest.approx(1.0)  # NaN shaping contributes 0, the correct reward stays finite
    assert rewards[1] is None  # NaN base reward -> unscorable
    assert rewards[2] == pytest.approx(0.2)


def test_reward_shaper_structural_gate_never_lifts_invalid_above_valid_incorrect():
    rng = np.random.default_rng(99)
    for _ in range(300):
        ceiling = float(rng.uniform(-3.0, 1.0))
        correct_reward = ceiling + float(rng.uniform(1e-3, 3.0))
        incorrect_reward = float(rng.uniform(-4.0, 2.0))
        n = 8
        correctness = rng.integers(0, 2, size=n).astype(float).tolist()
        shaping_vals = rng.normal(0, 2, size=n).tolist()
        valid_mask = rng.integers(0, 2, size=n).astype(bool).tolist()
        ids = [[7, EOS] if ok else [7, 7] for ok in valid_mask]

        shaper = RewardShaper(
            lambda completions, _vals=correctness, **kw: _vals,
            shaping_funcs=[lambda completions, _vals=shaping_vals, **kw: _vals],
            correct_reward=correct_reward,
            incorrect_ceiling=ceiling,
            incorrect_reward=incorrect_reward,
            structural={"eos_ids": (EOS,), "max_completion_length": 16},
        )
        rewards = shaper(prompts=["p"] * n, completions=["c"] * n, completion_ids=ids)
        zero_shaping_incorrect = min(incorrect_reward, ceiling)
        for i in range(n):
            if valid_mask[i]:
                continue
            assert rewards[i] <= zero_shaping_incorrect
            for j in range(n):
                if not valid_mask[j]:
                    continue
                if correctness[j] >= 0.5 or shaping_vals[j] >= 0:
                    assert rewards[i] <= rewards[j]


def test_reward_shaper_rejects_bad_configs():
    with pytest.raises(ValueError):
        RewardShaper(lambda completions, **kw: [0.0], correct_reward=0.0, incorrect_ceiling=0.0)
    with pytest.raises(ValueError):
        RewardShaper(lambda completions, **kw: [0.0], think_budget={"budget": 10})
    with pytest.raises(ValueError):
        RewardShaper(
            lambda completions, **kw: [0.0],
            structural={"eos_ids": (EOS,), "max_completion_length": 4, "require_single_think_close": True},
        )
