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

import pytest

import easydel as ed
from easydel.infra.elarge.model import eLargeModel
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer

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
