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

from __future__ import annotations

import sys
from pathlib import Path

import easydel as ed
import pytest
from easydel.infra.elarge.model import eLargeModel
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import (
    GRPOTrainer,
    _clip_rewards_if_configured,
)
from jax import numpy as jnp

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import make_config  # type: ignore
else:
    from ._common import make_config


def _make_args(**overrides):
    return make_config(
        ed.RLVRConfig,
        "rlvr-external-rewarders",
        overrides={
            "answer_key": "answer",
            **overrides,
        },
    )


@pytest.mark.cpu_ok
def test_rlvr_trainer_supports_explicit_external_rewarders(monkeypatch):
    captured = {}

    def fake_grpo_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_grpo_init)

    def syntax_reward(**kwargs):
        del kwargs
        return [0.0]

    args = _make_args(
        format_pattern=r"<tool_call>.*?</tool_call>",
        format_reward_weight=0.25,
    )

    ed.RLVRTrainer(
        arguments=args,
        model=None,
        processing_class=None,
        external_reward_funcs=[syntax_reward],
        external_reward_processing_classes=["syntax-proc"],
        external_reward_weights=[0.7],
    )

    reward_funcs = captured["reward_funcs"]
    assert len(reward_funcs) == 3
    assert reward_funcs[-1] is syntax_reward
    assert captured["reward_processing_classes"] == [None, None, "syntax-proc"]
    assert captured["arguments"].reward_weights == [1.0, 0.25, 0.7]


@pytest.mark.cpu_ok
def test_rlvr_trainer_wires_code_verifier_when_test_key_is_set(monkeypatch):
    captured = {}

    def fake_grpo_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_grpo_init)

    args = _make_args(answer_key=None, test_key="tests")

    ed.RLVRTrainer(
        arguments=args,
        model=None,
        processing_class=None,
    )

    reward_funcs = captured["reward_funcs"]
    assert len(reward_funcs) == 1
    assert captured["reward_processing_classes"] is None
    assert captured["arguments"].reward_weights == [1.0]
    assert reward_funcs[0](
        completions=["```python\ndef add_one(x):\n    return x + 1\n```"],
        batch={"tests": ["assert add_one(1) == 2"]},
    ) == [1.0]


@pytest.mark.cpu_ok
def test_rlvr_trainer_allows_external_rewarders_without_math_verifier(monkeypatch):
    captured = {}

    def fake_grpo_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_grpo_init)

    def syntax_reward(**kwargs):
        del kwargs
        return [0.0]

    args = _make_args(
        answer_key=None,
        format_pattern=None,
        format_reward_weight=0.0,
    )

    ed.RLVRTrainer(
        arguments=args,
        model=None,
        processing_class=None,
        external_reward_funcs=[syntax_reward],
        external_reward_processing_classes=["syntax-proc"],
        external_reward_weights=[0.7],
    )

    reward_funcs = captured["reward_funcs"]
    assert reward_funcs == [syntax_reward]
    assert captured["reward_processing_classes"] == ["syntax-proc"]
    assert captured["arguments"].reward_weights == [0.7]


@pytest.mark.cpu_ok
def test_rlvr_trainer_keeps_legacy_reward_funcs_api(monkeypatch):
    captured = {}

    def fake_grpo_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_grpo_init)

    def legacy_reward(**kwargs):
        del kwargs
        return [1.0]

    args = _make_args()

    ed.RLVRTrainer(
        arguments=args,
        model=None,
        processing_class=None,
        reward_funcs=[legacy_reward],
        reward_processing_classes=[None, "legacy-proc"],
    )

    reward_funcs = captured["reward_funcs"]
    assert len(reward_funcs) == 2
    assert reward_funcs[-1] is legacy_reward
    assert captured["reward_processing_classes"] == [None, "legacy-proc"]
    assert captured["arguments"].reward_weights == [1.0, 1.0]


@pytest.mark.cpu_ok
def test_rlvr_trainer_rejects_mixed_processing_class_apis(monkeypatch):
    monkeypatch.setattr(GRPOTrainer, "__init__", lambda self, **kwargs: None)

    def outside_reward(**kwargs):
        del kwargs
        return [0.0]

    args = _make_args()

    with pytest.raises(ValueError, match="Pass either `reward_processing_classes`"):
        ed.RLVRTrainer(
            arguments=args,
            model=None,
            processing_class=None,
            external_reward_funcs=[outside_reward],
            reward_processing_classes=[None, "legacy-proc"],
            external_reward_processing_classes=["outside-proc"],
        )


@pytest.mark.cpu_ok
def test_elarge_build_trainer_forwards_rlvr_external_rewarders():
    captured = {}

    class _CaptureTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def syntax_reward(**kwargs):
        del kwargs
        return [1.0]

    elm = eLargeModel(
        {
            "model": {
                "name_or_path": "dummy-model",
            },
            "trainer": {
                "trainer_type": "rlvr",
                "total_batch_size": 1,
                "answer_key": None,
            },
        }
    )
    elm._model = object()
    elm._tokenizer = object()

    elm.build_trainer(
        trainer_class=_CaptureTrainer,
        external_reward_funcs=[syntax_reward],
        external_reward_processing_classes=["syntax-proc"],
        external_reward_weights=[0.5],
    )

    assert captured["external_reward_funcs"] == [syntax_reward]
    assert captured["external_reward_processing_classes"] == ["syntax-proc"]
    assert captured["external_reward_weights"] == [0.5]


@pytest.mark.cpu_ok
def test_rlvr_config_mask_truncated_completions_not_clobbered_by_default():
    default_cfg = ed.RLVRConfig()
    assert default_cfg.mask_truncated_completions is True
    assert default_cfg.max_len_mask is True

    explicit_false = ed.RLVRConfig(mask_truncated_completions=False)
    assert explicit_false.mask_truncated_completions is False
    assert explicit_false.max_len_mask is False

    alias_false = ed.RLVRConfig(max_len_mask=False)
    assert alias_false.mask_truncated_completions is False
    assert alias_false.max_len_mask is False

    with pytest.raises(ValueError, match="Conflicting"):
        ed.RLVRConfig(mask_truncated_completions=False, max_len_mask=True)


@pytest.mark.cpu_ok
def test_rlvr_trainer_wires_configurable_length_penalty_max_length(monkeypatch):
    captured = {}

    def fake_grpo_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_grpo_init)

    args_unbounded = _make_args(
        answer_key=None,
        length_penalty_target=6000,
        length_penalty_weight=0.5,
    )
    ed.RLVRTrainer(arguments=args_unbounded, model=None, processing_class=None)
    lp_unbounded = captured["reward_funcs"][0]
    assert lp_unbounded(completions=["x" * 6000]) == [1.0]

    args_capped = _make_args(
        answer_key=None,
        length_penalty_target=500,
        length_penalty_max_length=600,
        length_penalty_weight=0.5,
    )
    ed.RLVRTrainer(arguments=args_capped, model=None, processing_class=None)
    lp_capped = captured["reward_funcs"][0]
    assert lp_capped(completions=["x" * 500, "x" * 650]) == [1.0, 0.0]


@pytest.mark.cpu_ok
def test_rlvr_trainer_preserves_both_arguments_and_external_reward_weights(monkeypatch):
    captured = {}

    def fake_grpo_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_grpo_init)

    def ext_reward(**kwargs):
        del kwargs
        return [1.0]

    # When arguments.reward_weights weights the built-in verifiers and
    # external_reward_weights weights the external rewards, both are preserved.
    args = _make_args(answer_key="answer", reward_weights=[2.0])
    ed.RLVRTrainer(
        arguments=args,
        model=None,
        processing_class=None,
        external_reward_funcs=[ext_reward],
        external_reward_weights=[0.5],
    )
    assert captured["arguments"].reward_weights == [2.0, 0.5]

    # Conflicting ensemble-length arguments.reward_weights + external_reward_weights raises instead of dropping.
    conflicting = _make_args(answer_key="answer", reward_weights=[2.0, 0.3])
    with pytest.raises(ValueError, match="Cannot combine"):
        ed.RLVRTrainer(
            arguments=conflicting,
            model=None,
            processing_class=None,
            external_reward_funcs=[ext_reward],
            external_reward_weights=[0.5],
        )

    # Passing external_reward_weights with no external reward functions raises instead of dropping.
    with pytest.raises(ValueError, match="without any external reward functions"):
        ed.RLVRTrainer(
            arguments=_make_args(answer_key="answer"),
            model=None,
            processing_class=None,
            external_reward_weights=[0.5],
        )


def test_rlvr_reward_clip_range_is_applied_to_combined_rewards():
    args = _make_args(reward_clip_range=0.5)

    rewards = _clip_rewards_if_configured(
        jnp.array([-1.25, -0.25, 0.25, 1.25], dtype=jnp.float32),
        args,
    )

    assert rewards.tolist() == pytest.approx([-0.5, -0.25, 0.25, 0.5])
