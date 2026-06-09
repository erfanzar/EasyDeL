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

"""On-policy knowledge distillation trainer module for EasyDeL.

This module provides on-policy distillation where the student generates
completions from prompts during training and teacher knowledge is distilled
on the student's own generated sequences, keeping the training distribution
on-policy.
"""

from .on_policy_distillation_config import OnPolicyDistillationConfig
from .on_policy_distillation_trainer import OnPolicyDistillationTrainer

__all__ = ("OnPolicyDistillationConfig", "OnPolicyDistillationTrainer")
