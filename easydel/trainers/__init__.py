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
from .base_trainer import BaseTrainer
from .binary_classifier_optimization_trainer import BCOConfig, BCOTrainer
from .contrastive_preference_optimization_trainer import CPOConfig, CPOTrainer
from .direct_preference_optimization_trainer import DPOConfig, DPOTrainer
from .distillation_trainer import DistillationConfig, DistillationTrainer
from .embedding_trainer import EmbeddingConfig, EmbeddingTrainer
from .generalized_knowledge_distillation_trainer import GKDConfig, GKDTrainer
from .group_filtered_policy_optimization import GFPOConfig, GFPOTrainer
from .group_relative_policy_optimization import GRPOConfig, GRPOTrainer
from .group_sequence_policy_optimization import GSPOConfig, GSPOTrainer
from .kto_trainer import KTOConfig, KTOTrainer
from .metrics import LogWatcher
from .nash_md_trainer import NashMDConfig, NashMDTrainer
from .odds_ratio_preference_optimization_trainer import ORPOConfig, ORPOTrainer
from .on_policy_distillation_trainer import OnPolicyDistillationConfig, OnPolicyDistillationTrainer
from .packer import pack_sequences
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
from .rlvr_trainer import RLVRConfig, RLVRTrainer
from .self_distillation_policy_optimization import SDPOConfig, SDPOTrainer
from .seq_kd_trainer import SeqKDConfig, SeqKDTrainer
from .sparse_distillation_trainer import SparseDistillationConfig, SparseDistillationTrainer
from .supervised_fine_tuning_trainer import SFTConfig, SFTTrainer
from .trainer import Trainer
from .training_configurations import TrainingArguments
from .xpo_trainer import XPOConfig, XPOTrainer

__all__ = (
    "AgenticMoshPitConfig",
    "AgenticMoshPitTrainer",
    "BCOConfig",
    "BCOPreprocessTransform",
    "BCOTrainer",
    "BaseTrainer",
    "CPOConfig",
    "CPOPreprocessTransform",
    "CPOTrainer",
    "DPOConfig",
    "DPOPreprocessTransform",
    "DPOTrainer",
    "DistillationConfig",
    "DistillationTrainer",
    "EmbeddingConfig",
    "EmbeddingTrainer",
    "GFPOConfig",
    "GFPOTrainer",
    "GKDConfig",
    "GKDTrainer",
    "GRPOConfig",
    "GRPOPreprocessTransform",
    "GRPOTrainer",
    "GSPOConfig",
    "GSPOTrainer",
    "KTOConfig",
    "KTOPreprocessTransform",
    "KTOTrainer",
    "LogWatcher",
    "NashMDConfig",
    "NashMDTrainer",
    "ORPOConfig",
    "ORPOPreprocessTransform",
    "ORPOTrainer",
    "OnPolicyDistillationConfig",
    "OnPolicyDistillationTrainer",
    "PPOConfig",
    "PPOPreprocessTransform",
    "PPOTrainer",
    "RLVRConfig",
    "RLVRTrainer",
    "RayDistributedTrainer",
    "RewardConfig",
    "RewardPreprocessTransform",
    "RewardTrainer",
    "SDPOConfig",
    "SDPOTrainer",
    "SFTConfig",
    "SFTPreprocessTransform",
    "SFTTrainer",
    "SeqKDConfig",
    "SeqKDTrainer",
    "SparseDistillationConfig",
    "SparseDistillationTrainer",
    "Trainer",
    "TrainingArguments",
    "XPOConfig",
    "XPOTrainer",
    "pack_sequences",
    "prompt_transforms",
)
