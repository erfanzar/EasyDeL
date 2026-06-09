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
"""Unit tests for the class-based :class:`RewardProtocol` abstraction."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from easydel.trainers.group_relative_policy_optimization.grpo_trainer import (
    GRPOTrainer,
    _compute_rewards_and_advantages,
)
from easydel.trainers.reward_protocol import RewardProtocol, as_messages


def test_per_example_compute_with_signature_filtering():
    """compute() receives only the kwargs it declares and runs once per completion."""

    class KeywordReward(RewardProtocol):
        def compute(self, *, completion, **kwargs):
            return 1.0 if "yes" in completion else 0.0

    reward = KeywordReward()
    out = reward(prompts=["p1", "p2"], completions=["yes a", "no b"], max_length=8, batch={"g": 1})
    assert out == [1.0, 0.0]


def test_conversational_items_are_flattened_to_str_for_compute():
    """compute() receives plain strings even when the batch holds chat message lists."""
    seen: dict = {}

    class Inspect(RewardProtocol):
        def compute(self, *, prompt, completion, messages, **kwargs):
            seen.update(prompt=prompt, completion=completion, messages=messages)
            return 0.0

    Inspect()(
        prompts=[[{"role": "user", "content": "hi"}]],
        completions=[[{"role": "assistant", "content": "hello"}]],
    )
    assert isinstance(seen["prompt"], str) and seen["prompt"] == "hi"
    assert isinstance(seen["completion"], str) and seen["completion"] == "hello"
    # structured form is still available via messages
    assert seen["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_none_is_passed_through_for_unscored_examples():
    """Returning None marks an example unscored (trainer maps it to NaN)."""

    class MaybeReward(RewardProtocol):
        def compute(self, *, completion, **kwargs):
            return None if completion == "skip" else 1.0

    assert MaybeReward()(prompts=[None, None], completions=["ok", "skip"]) == [1.0, None]


def test_compute_batch_override_is_honored():
    """A vectorized subclass overrides compute_batch directly."""

    class LengthReward(RewardProtocol):
        def compute_batch(self, *, prompts, completions, **kwargs):
            return [float(len(c)) for c in completions]

    assert LengthReward()(prompts=["a"], completions=["abc"]) == [3.0]


def test_per_item_fields_are_indexed_and_renamed():
    """List-aligned batch fields are split per example; scalars/dicts pass through."""

    seen = {}

    class InspectReward(RewardProtocol):
        def compute(self, *, completion, completion_text, reasoning, batch, max_length, **kwargs):
            seen.update(
                completion=completion,
                completion_text=completion_text,
                reasoning=reasoning,
                batch=batch,
                max_length=max_length,
            )
            return 0.0

    InspectReward()(
        prompts=["p0", "p1"],
        completions=["c0", "c1"],
        completion_texts=["t0", "t1"],  # renamed -> completion_text (per item)
        reasoning=["r0", "r1"],
        max_length=16,  # scalar -> passthrough
        batch={"gold": [1, 2]},  # dict -> passthrough
    )
    # last example (index 1)
    assert seen["completion"] == "c1"
    assert seen["completion_text"] == "t1"
    assert seen["reasoning"] == "r1"
    assert seen["max_length"] == 16
    assert seen["batch"] == {"gold": [1, 2]}


def test_generation_signals_are_indexed_per_example():
    """finish_reason / truncated / completion_length / completion_ids reach compute() per example."""
    seen: list[tuple] = []

    class SignalReward(RewardProtocol):
        def compute(self, *, completion, finish_reason, truncated, completion_length, completion_ids, **kwargs):
            seen.append((completion, finish_reason, truncated, completion_length, completion_ids))
            return 0.0

    SignalReward()(
        prompts=[None, None],
        completions=["a", "b"],
        finish_reason=["stop", "length"],
        truncated=[False, True],
        completion_length=[3, 7],
        completion_ids=[[1, 2, 3], [4, 5, 6, 7, 8, 9, 10]],
    )
    assert seen[0] == ("a", "stop", False, 3, [1, 2, 3])
    assert seen[1] == ("b", "length", True, 7, [4, 5, 6, 7, 8, 9, 10])


def test_as_messages_text_and_conversational():
    assert as_messages("hello", "world") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    # conversational: prompt is already a message list, completion an assistant turn list
    prompt_msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    completion_msgs = [{"role": "assistant", "content": "a"}]
    assert as_messages(prompt_msgs, completion_msgs) == prompt_msgs + completion_msgs


def test_name_defaults_to_class_name_and_honors_override():
    class Plain(RewardProtocol):
        def compute(self, **kwargs):
            return 0.0

    class Named(RewardProtocol):
        name = "custom_metric"

        def compute(self, **kwargs):
            return 0.0

    assert Plain().__name__ == "Plain"
    assert Named().__name__ == "custom_metric"


def test_subclass_validation():
    with pytest.raises(ValueError):

        class BadReduction(RewardProtocol):
            reduction = "median"

            def compute(self, **kwargs):
                return 0.0

    with pytest.raises(TypeError):

        class NoScoringMethod(RewardProtocol):
            pass


@pytest.mark.parametrize(
    ("reduction", "baseline"),
    [("mean", 2.5), ("sum", 10.0)],
)
def test_group_reduction_baseline_math(reduction, baseline):
    """`reduction` selects the per-group baseline subtracted to form advantages."""
    rewards_per_func = jnp.array([[1.0], [2.0], [3.0], [4.0]], dtype="f4")
    reward_weights = jnp.array([1.0], dtype="f4")
    arguments = SimpleNamespace(advantage_estimator="group_mean", reward_clip_range=None)

    _, advantages, _, _ = _compute_rewards_and_advantages(
        rewards_per_func=rewards_per_func,
        reward_weights=reward_weights,
        generation_factor=4,
        scale_rewards="none",
        multi_objective_aggregation="sum_then_normalize",
        arguments=arguments,
        group_reduction=reduction,
    )
    expected = np.array([1.0, 2.0, 3.0, 4.0]) - baseline
    assert np.allclose(np.asarray(advantages), expected)


def test_resolve_group_reduction():
    class SumReward(RewardProtocol):
        reduction = "sum"

        def compute(self, **kwargs):
            return 0.0

    class MeanReward(RewardProtocol):
        reduction = "mean"

        def compute(self, **kwargs):
            return 0.0

    def plain_reward(**kwargs):
        return [0.0]

    # a protocol drives the reduction; plain rewards keep the classic "mean" baseline
    assert GRPOTrainer._resolve_group_reduction([SumReward()]) == "sum"
    assert GRPOTrainer._resolve_group_reduction([plain_reward]) == "mean"
    assert GRPOTrainer._resolve_group_reduction([SumReward(), plain_reward]) == "sum"

    with pytest.raises(ValueError):
        GRPOTrainer._resolve_group_reduction([SumReward(), MeanReward()])


def test_reward_weight_field_default_class_attr_and_instance_override():
    """`weight` defaults to 1.0, is overridable per-class and per-instance."""

    class DefaultWeight(RewardProtocol):
        def compute(self, **kwargs):
            return 1.0

    class ClassWeight(RewardProtocol):
        weight = 0.5

        def compute(self, **kwargs):
            return 1.0

    assert DefaultWeight().weight == 1.0
    assert ClassWeight().weight == 0.5
    # per-instance override wins over the class default
    assert DefaultWeight(weight=0.3).weight == 0.3
    assert ClassWeight(weight=0.25).weight == 0.25
    # plain callables expose no weight -> trainer falls back to 1.0
    assert getattr(lambda **k: [0.0], "weight", 1.0) == 1.0


def test_reward_weights_built_from_protocol_weight_field():
    """The trainer derives reward_weights from each reward's `weight` field."""

    class W(RewardProtocol):
        def compute(self, **kwargs):
            return 1.0

    def plain_reward(**kwargs):
        return [0.0]

    funcs = [W(), W(weight=0.5), plain_reward]
    # mirrors GRPOTrainer.__init__: weights come from each reward's `weight` (default 1.0)
    built = [float(getattr(func, "weight", 1.0)) for func in funcs]
    assert built == [1.0, 0.5, 1.0]
