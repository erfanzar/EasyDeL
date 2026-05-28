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
"""EasyDeL trainers package.

Top-level entry point that re-exports the public trainer classes, their
configuration dataclasses, and prompt-preprocessing transforms.  The
package contains:

* The base trainer (:class:`BaseTrainer`) and the generic supervised
  :class:`Trainer` plus its arguments dataclass
  (:class:`TrainingArguments`).
* Supervised fine-tuning, reward modelling, and embedding trainers.
* Preference- and policy-optimization trainers (DPO, ORPO, CPO, KTO, BCO,
  PPO, GRPO, GSPO, GFPO, RLVR, Nash-MD, XPO, SDPO, ...).
* Distillation trainers (offline, on-policy, generalized knowledge
  distillation, sequence-level KD, sparse distillation).
* Agentic self-play training (:class:`AgenticMoshPitTrainer`).
* Utilities such as :func:`pack_sequences`, :class:`LogWatcher`, and the
  Ray-based distributed launcher (:class:`RayDistributedTrainer`).

Importing from this package is the recommended way to access trainers and
configs; sub-modules expose internal helpers that are not part of the
public API.
"""

from . import prompt_transforms
from .agentic_moshpit import AgenticMoshPitConfig, AgenticMoshPitTrainer
from .async_grpo_trainer import AsyncGRPOConfig, AsyncGRPOTrainer
from .base_trainer import BaseTrainer
from .bema_trainer import BEMACallback, BEMAConfig, BEMADPOTrainer
from .binary_classifier_optimization_trainer import BCOConfig, BCOTrainer
from .contrastive_preference_optimization_trainer import CPOConfig, CPOTrainer
from .direct_preference_optimization_trainer import DPOConfig, DPOTrainer
from .distillation_trainer import DistillationConfig, DistillationTrainer
from .dppo_trainer import DPPOConfig, DPPOTrainer
from .embedding_trainer import EmbeddingConfig, EmbeddingTrainer
from .esurge_rollout import OpenRewardSpec, eSurgeRolloutConfig, eSurgeRolloutGenerator, generate_rollout_completions
from .generalized_knowledge_distillation_trainer import GKDConfig, GKDTrainer
from .gold_trainer import GOLDConfig, GOLDTrainer
from .group_filtered_policy_optimization import GFPOConfig, GFPOTrainer
from .group_relative_policy_optimization import GRPOConfig, GRPOTrainer
from .group_sequence_policy_optimization import GSPOConfig, GSPOTrainer
from .grpo_replay_buffer_trainer import (
    GRPOReplayBufferConfig,
    GRPOReplayBufferTrainer,
    GRPOWithReplayBufferConfig,
    GRPOWithReplayBufferTrainer,
)
from .gspo_token_trainer import GSPOTokenConfig, GSPOTokenTrainer
from .kto_trainer import KTOConfig, KTOTrainer
from .merge_callback import MergeConfig, MergeModelCallback
from .metrics import LogWatcher
from .minillm_trainer import MiniLLMConfig, MiniLLMTrainer
from .nash_md_trainer import NashMDConfig, NashMDTrainer
from .nemo_gym_trainer import NeMoGymConfig, NeMoGymTrainer, load_nemo_gym_jsonl
from .odds_ratio_preference_optimization_trainer import ORPOConfig, ORPOTrainer
from .on_policy_distillation_trainer import OnPolicyDistillationConfig, OnPolicyDistillationTrainer
from .online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer
from .packer import pack_sequences
from .papo_trainer import PAPOConfig, PAPOTrainer
from .prm_trainer import PRMConfig, PRMPreprocessTransform, PRMTrainer
from .prompt_transforms import (
    BCOPreprocessTransform,
    CPOPreprocessTransform,
    DPOPreprocessTransform,
    GRPOPreprocessTransform,
    KTOPreprocessTransform,
    ORPOPreprocessTransform,
    PPOPreprocessTransform,
    RewardPreprocessTransform,
    SFTPreprocessTransform,
)
from .proximal_policy_optimization_trainer import PPOConfig, PPOTrainer
from .ray_scaler import RayDistributedTrainer
from .reward_trainer import RewardConfig, RewardTrainer
from .rloo_trainer import RLOOConfig, RLOOTrainer
from .rlvr_trainer import RLVRConfig, RLVRTrainer
from .sdft_trainer import SDFTConfig, SDFTTrainer, SelfDistillationConfig, SelfDistillationTrainer
from .self_distillation_policy_optimization import SDPOConfig, SDPOTrainer
from .seq_kd_trainer import SeqKDConfig, SeqKDTrainer
from .sparse_distillation_trainer import SparseDistillationConfig, SparseDistillationTrainer
from .ssd_trainer import SSDConfig, SSDTrainer
from .supervised_fine_tuning_trainer import SFTConfig, SFTTrainer
from .tpo_trainer import TPOConfig, TPOTrainer
from .trainer import Trainer
from .training_configurations import TrainingArguments
from .xpo_trainer import XPOConfig, XPOTrainer

__all__ = (
    "AgenticMoshPitConfig",
    "AgenticMoshPitTrainer",
    "AsyncGRPOConfig",
    "AsyncGRPOTrainer",
    "BCOConfig",
    "BCOPreprocessTransform",
    "BCOTrainer",
    "BEMACallback",
    "BEMAConfig",
    "BEMADPOTrainer",
    "BaseTrainer",
    "CPOConfig",
    "CPOPreprocessTransform",
    "CPOTrainer",
    "DPOConfig",
    "DPOPreprocessTransform",
    "DPOTrainer",
    "DPPOConfig",
    "DPPOTrainer",
    "DistillationConfig",
    "DistillationTrainer",
    "EmbeddingConfig",
    "EmbeddingTrainer",
    "GFPOConfig",
    "GFPOTrainer",
    "GKDConfig",
    "GKDTrainer",
    "GOLDConfig",
    "GOLDTrainer",
    "GRPOConfig",
    "GRPOPreprocessTransform",
    "GRPOReplayBufferConfig",
    "GRPOReplayBufferTrainer",
    "GRPOTrainer",
    "GRPOWithReplayBufferConfig",
    "GRPOWithReplayBufferTrainer",
    "GSPOConfig",
    "GSPOTokenConfig",
    "GSPOTokenTrainer",
    "GSPOTrainer",
    "KTOConfig",
    "KTOPreprocessTransform",
    "KTOTrainer",
    "LogWatcher",
    "MergeConfig",
    "MergeModelCallback",
    "MiniLLMConfig",
    "MiniLLMTrainer",
    "NashMDConfig",
    "NashMDTrainer",
    "NeMoGymConfig",
    "NeMoGymTrainer",
    "ORPOConfig",
    "ORPOPreprocessTransform",
    "ORPOTrainer",
    "OnPolicyDistillationConfig",
    "OnPolicyDistillationTrainer",
    "OnlineDPOConfig",
    "OnlineDPOTrainer",
    "OpenRewardSpec",
    "PAPOConfig",
    "PAPOTrainer",
    "PPOConfig",
    "PPOPreprocessTransform",
    "PPOTrainer",
    "PRMConfig",
    "PRMPreprocessTransform",
    "PRMTrainer",
    "RLOOConfig",
    "RLOOTrainer",
    "RLVRConfig",
    "RLVRTrainer",
    "RayDistributedTrainer",
    "RewardConfig",
    "RewardPreprocessTransform",
    "RewardTrainer",
    "SDFTConfig",
    "SDFTTrainer",
    "SDPOConfig",
    "SDPOTrainer",
    "SFTConfig",
    "SFTPreprocessTransform",
    "SFTTrainer",
    "SSDConfig",
    "SSDTrainer",
    "SelfDistillationConfig",
    "SelfDistillationTrainer",
    "SeqKDConfig",
    "SeqKDTrainer",
    "SparseDistillationConfig",
    "SparseDistillationTrainer",
    "TPOConfig",
    "TPOTrainer",
    "Trainer",
    "TrainingArguments",
    "XPOConfig",
    "XPOTrainer",
    "eSurgeRolloutConfig",
    "eSurgeRolloutGenerator",
    "generate_rollout_completions",
    "load_nemo_gym_jsonl",
    "pack_sequences",
    "prompt_transforms",
)
