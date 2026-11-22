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

"""Public exports for the DPPO trainer.

DPPO reuses EasyDeL's GRPO rollout path while exposing the DPPO configuration
and trainer names expected by downstream alignment scripts.
"""

from .dppo_config import DPPOConfig
from .dppo_trainer import DPPOTrainer

__all__ = ("DPPOConfig", "DPPOTrainer")
