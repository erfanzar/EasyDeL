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

"""CPU tests for unified `async_mode` across all rollout-capable EasyDeL trainers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from easydel.trainers import (
    AgenticMoshPitConfig,
    AgenticMoshPitTrainer,
    AsyncGRPOConfig,
    AsyncGRPOTrainer,
    AsyncRolloutMixin,
    CompactionRLTrainer,
    DPOTrainer,
    DPPOConfig,
    DPPOTrainer,
    EmbeddingTrainer,
    GFPOConfig,
    GFPOTrainer,
    GKDConfig,
    GKDTrainer,
    GRPOConfig,
    GRPOTrainer,
    GRPOWithReplayBufferConfig,
    GRPOWithReplayBufferTrainer,
    GSPOConfig,
    GSPOTokenConfig,
    GSPOTokenTrainer,
    GSPOTrainer,
    MiniLLMConfig,
    MiniLLMTrainer,
    NashMDConfig,
    NashMDTrainer,
    NeMoGymConfig,
    NeMoGymTrainer,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    OnPolicyDistillationConfig,
    OnPolicyDistillationTrainer,
    PAPOConfig,
    PAPOTrainer,
    PPOConfig,
    PPOTrainer,
    RewardTrainer,
    RLOOConfig,
    RLOOTrainer,
    RLVRConfig,
    RLVRTrainer,
    SAOTrainer,
    SDFTConfig,
    SDFTTrainer,
    SDPOConfig,
    SDPOTrainer,
    SeqKDConfig,
    SeqKDTrainer,
    SFTConfig,
    SFTTrainer,
    SSDConfig,
    SSDTrainer,
    XPOConfig,
    XPOTrainer,
)

pytestmark = pytest.mark.cpu_ok


ROLLOUT_CONFIGS = [
    GRPOConfig,
    GSPOConfig,
    GSPOTokenConfig,
    GFPOConfig,
    RLOOConfig,
    RLVRConfig,
    DPPOConfig,
    GRPOWithReplayBufferConfig,
    AgenticMoshPitConfig,
    NeMoGymConfig,
    NashMDConfig,
    XPOConfig,
    PAPOConfig,
    SDPOConfig,
    SDFTConfig,
    SSDConfig,
    MiniLLMConfig,
    PPOConfig,
    OnlineDPOConfig,
    OnPolicyDistillationConfig,
    GKDConfig,
    SeqKDConfig,
]

ROLLOUT_TRAINERS = [
    GRPOTrainer,
    GSPOTrainer,
    GSPOTokenTrainer,
    GFPOTrainer,
    RLOOTrainer,
    RLVRTrainer,
    DPPOTrainer,
    GRPOWithReplayBufferTrainer,
    AgenticMoshPitTrainer,
    NeMoGymTrainer,
    NashMDTrainer,
    XPOTrainer,
    PAPOTrainer,
    SDPOTrainer,
    SDFTTrainer,
    SSDTrainer,
    MiniLLMTrainer,
    AsyncGRPOTrainer,
    PPOTrainer,
    OnlineDPOTrainer,
    OnPolicyDistillationTrainer,
    GKDTrainer,
    SeqKDTrainer,
]


@pytest.mark.parametrize("cfg_cls", ROLLOUT_CONFIGS)
def test_all_rollout_configs_support_async_mode(cfg_cls):
    default_cfg = cfg_cls()
    assert default_cfg.async_mode is False

    async_cfg = cfg_cls(async_mode=True, max_inflight_tasks=16, max_staleness=2, weight_sync_steps=4)
    assert async_cfg.async_mode is True
    assert async_cfg.esurge_async_scheduling is True
    assert async_cfg.esurge_overlap_execution is True
    assert async_cfg.max_inflight_tasks == 16
    assert async_cfg.max_staleness == 2
    assert async_cfg.weight_sync_steps == 4


def test_async_mode_validation_and_async_grpo_default():
    ag = AsyncGRPOConfig()
    assert ag.async_mode is True
    assert ag.esurge_async_scheduling is True
    assert ag.esurge_overlap_execution is True

    with pytest.raises(ValueError, match="max_inflight_tasks"):
        GRPOConfig(async_mode=True, max_inflight_tasks=0)
    with pytest.raises(ValueError, match="max_staleness"):
        PPOConfig(async_mode=True, max_staleness=-1)
    with pytest.raises(ValueError, match="weight_sync_steps"):
        OnlineDPOConfig(async_mode=True, weight_sync_steps=0)


@pytest.mark.parametrize("trainer_cls", ROLLOUT_TRAINERS)
def test_all_rollout_trainers_enable_async_rollout_capability(trainer_cls):
    assert trainer_cls.supports_async_rollout is True
    assert issubclass(trainer_cls, AsyncRolloutMixin)


@pytest.mark.parametrize(
    "trainer_cls", [SFTTrainer, DPOTrainer, RewardTrainer, EmbeddingTrainer, SAOTrainer, CompactionRLTrainer]
)
def test_non_rollout_and_dedicated_actor_critic_trainers_reject_async_mode(trainer_cls):
    from easydel.trainers import BaseTrainer

    assert BaseTrainer.__doc__ is not None
    assert trainer_cls.supports_async_rollout is False
    dummy_state = SimpleNamespace(model=SimpleNamespace(_model_type="llama"))
    # BaseTrainer.__init__ enforces supports_async_rollout before any model/dataloader setup.
    inst = object.__new__(trainer_cls)
    with pytest.raises(ValueError, match="does not support `async_mode=True`"):
        BaseTrainer.__init__(inst, arguments=SFTConfig(async_mode=True), model_state=dummy_state)


@pytest.mark.parametrize("trainer_cls", [GRPOTrainer, AsyncGRPOTrainer, GRPOWithReplayBufferTrainer])
def test_grpo_family_honors_async_force_old_logps_and_disables_caching(trainer_cls):
    inst = object.__new__(trainer_cls)
    inst.steps_per_generation = 2
    inst.num_iterations = 2
    inst.arguments = SimpleNamespace(
        async_mode=True,
        max_inflight_tasks=8,
        max_staleness=1,
        weight_sync_steps=2,
        esurge_async_scheduling=True,
        esurge_overlap_execution=True,
    )
    inst._async_grpo_force_old_logps = True
    assert inst._generation_reuse_span() >= 2

    batch, metrics = inst._store_buffered_grpo_batch({"prompt_ids": [1]}, {})
    assert batch == {"prompt_ids": [1]}
    assert getattr(inst, "_buffered_grpo_batch", None) is None
    assert getattr(inst, "_buffered_grpo_remaining", 0) == 0
    assert metrics["generation_reused"] == 0
    assert "async_rollout/max_inflight_tasks" in metrics
