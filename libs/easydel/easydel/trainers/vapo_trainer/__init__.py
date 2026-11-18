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
"""Value-model-based Augmented PPO (VAPO, arXiv:2504.05118).

VAPO is PPO with decoupled and length-adaptive GAE, clip-higher, a
token-level policy loss and a positive-example NLL term. It also supports
optional value pretraining. All of these are :class:`PPOConfig` knobs;
:class:`VAPOConfig` turns on the VAPO defaults.
"""

from .vapo_config import VAPOConfig
from .vapo_trainer import VAPOTrainer

__all__ = ("VAPOConfig", "VAPOTrainer")
