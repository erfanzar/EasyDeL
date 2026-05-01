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

"""Proximal Policy Optimization (PPO) trainer module for EasyDeL.

PPO is an actor-critic policy gradient method that constrains the policy
update to a proximal region of the rollout policy. The objective is the
clipped surrogate

    L_pi = E_t[ min( r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t ) ]

with a value-function regression term ``L_v`` and an optional entropy bonus.
For RLHF the actor is a causal language model with a value-head adapter,
advantages are estimated via GAE on per-token rewards (typically a reward
model score plus a KL penalty against a reference policy), and rollouts are
generated outside the gradient computation.

The module exposes:
    - :class:`PPOConfig`: Hyperparameter container for PPO training.
    - :class:`PPOTrainer`: Trainer orchestrating rollouts and clipped
      policy/value updates.
"""

from .ppo_config import PPOConfig
from .ppo_trainer import PPOTrainer

__all__ = ("PPOConfig", "PPOTrainer")
