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
"""Geometric-Mean Policy Optimization (GMPO, arXiv:2507.20673) trainer entry point.

GMPO optimises the geometric mean of advantage-sign-aware clipped token
importance ratios per sequence; it reuses GRPO's generation and loss kernels.
"""

from .gmpo_config import GMPOConfig
from .gmpo_trainer import GMPOTrainer

__all__ = ["GMPOConfig", "GMPOTrainer"]
