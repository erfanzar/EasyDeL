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

"""Exploratory Preference Optimization (XPO) trainer module for EasyDeL.

This module wires the online variant of DPO described in Liu et al.
(2024), pairing a DPO/IPO preference loss with an explicit exploration
bonus that keeps probability mass on completions sampled from a frozen
reference policy. The trainer reuses the GRPO rollout / sampling
infrastructure but restricts the group to a single completion per
prompt; the *second* sample required for the preference pair is drawn
from the reference model and acts as a stand-in for an oracle answer.

The module includes:
- :class:`XPOConfig`: Configuration class for XPO hyperparameters and
  schedules (``loss_type``, ``beta``, ``alpha``, missing-EOS penalty).
- :class:`XPOTrainer`: Main trainer class implementing online XPO on
  top of :class:`GRPOTrainer`.

Key features:
- DPO sigmoid surrogate or IPO squared-margin variant via ``loss_type``.
- Epoch-wise schedules for both KL temperature ``beta`` and exploration
  weight ``alpha``.
- Reference-policy sync cadence inherited from GRPO.
- Optional missing-EOS penalty to discourage truncated completions.

Example:
    >>> from easydel.trainers import XPOConfig, XPOTrainer
    >>> config = XPOConfig(
    ...     loss_type="sigmoid",
    ...     beta=0.1,
    ...     alpha=1e-5,
    ...     max_length=1024,
    ... )
    >>> trainer = XPOTrainer(
    ...     arguments=config,
    ...     model=policy_model,
    ...     reference_model=reference_model,
    ...     reward_funcs=reward_fn,
    ...     train_dataset=prompt_dataset,
    ...     processing_class=tokenizer,
    ... )
    >>> trainer.train()

References:
    - Liu et al., "Exploratory Preference Optimization: Provably Sample
      Efficient Exploration in RLHF with General Function Approximation"
      (https://arxiv.org/abs/2405.21046)
"""

from .xpo_config import XPOConfig
from .xpo_trainer import XPOTrainer

__all__ = (
    "XPOConfig",
    "XPOTrainer",
)
