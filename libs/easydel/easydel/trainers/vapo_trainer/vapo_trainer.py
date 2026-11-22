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
"""VAPO trainer surface backed by the PPO actor-critic path."""

from __future__ import annotations

from easydel.utils import Registry

from ..proximal_policy_optimization_trainer import PPOTrainer
from .vapo_config import VAPOConfig


@Registry.register("trainer", "vapo")
class VAPOTrainer(PPOTrainer):
    """VAPO trainer (arXiv:2504.05118) built on :class:`PPOTrainer`.

    The algorithm is implemented by the PPO rollout and loss path. Decoupled
    and length-adaptive GAE live in
    :meth:`PPOTrainer._compute_advantages_and_returns`. Clip-higher,
    token-level loss, positive-example NLL and value pretraining live in
    :func:`~easydel.trainers.proximal_policy_optimization_trainer._fn.ppo_step`.
    :class:`VAPOConfig` only switches on the VAPO defaults. To change a single
    technique, override the corresponding config field or the PPO hook, for
    example ``_compute_advantages_and_returns`` for a different lambda
    schedule.

    Example:
        >>> config = VAPOConfig(per_device_train_batch_size=4, num_return_sequences=4)
        >>> trainer = VAPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=verifier,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    arguments: VAPOConfig
