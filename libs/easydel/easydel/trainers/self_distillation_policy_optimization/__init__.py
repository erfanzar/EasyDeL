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

"""Self-Distillation Policy Optimization (SDPO) trainer module for EasyDeL.

SDPO (arXiv:2601.20802) is a reinforcement-learning post-training method that
converts rich tokenized environment feedback into a dense learning signal
*without any external teacher model*.  The current policy acts as its own
teacher by conditioning on the feedback it just received, allowing retrospective
identification of mistakes and dense per-token credit assignment.

This module provides:
- :class:`SDPOConfig` - configuration extending :class:`GRPOConfig` with
  feedback-length and distillation-type parameters.
- :class:`SDPOTrainer` - drop-in replacement for :class:`GRPOTrainer` that
  uses self-distillation advantages instead of scalar-reward advantages.

References:
    Hübotter et al. (2026). Reinforcement Learning via Self-Distillation.
    arXiv:2601.20802.
"""

from .sdpo_config import SDPOConfig
from .sdpo_trainer import SDPOTrainer

__all__ = ("SDPOConfig", "SDPOTrainer")
