# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Proximal Policy Optimization (PPO) trainer module for EasyDeL.

This module implements PPO-style RLHF training for autoregressive language models.

The module includes:
- PPOConfig: Configuration for PPO training parameters
- PPOTrainer: Trainer implementing PPO rollouts + clipped policy/value updates
"""

from .ppo_config import PPOConfig
from .ppo_trainer import PPOTrainer

__all__ = ("PPOConfig", "PPOTrainer")

