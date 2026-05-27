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
"""Process reward model preprocessing and trainer."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..reward_trainer import RewardConfig


@Registry.register("trainer-arguments", "prm")
@dataclass
class PRMConfig(RewardConfig):
    """Configuration for process reward model fine-tuning.

    PRM preprocessing splits completions into reasoning steps using
    ``step_separator`` and builds token labels for either all steps or only the
    final step. The remaining fields mirror reward-model training while adding
    PRM-specific completion length and multiprocessing controls.
    """

    trainer_prefix: str | None = field(default="PRM")
    max_completion_length: int | None = field(default=None)
    step_separator: str = field(default="\n")
    train_on_last_step_only: bool = field(default=False)
    dataset_num_proc: int | None = field(default=None)

    __hash__ = hash_fn
