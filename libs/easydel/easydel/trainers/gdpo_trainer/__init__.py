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
"""Group reward-Decoupled Policy Optimization (GDPO, arXiv:2601.05242) trainer entry point.

GDPO normalizes each reward function within its prompt group before summing,
then batch-normalizes the sum; it reuses GRPO's generation and loss kernels.
"""

from .gdpo_config import GDPOConfig
from .gdpo_trainer import GDPOTrainer

__all__ = ["GDPOConfig", "GDPOTrainer"]
