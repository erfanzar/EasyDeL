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

"""Public exports for online DPO training.

Online DPO generates candidate completions with EasyDeL/eSurge, scores them,
and converts the result into preference batches for the standard DPO loss.
"""

from .online_dpo_config import OnlineDPOConfig
from .online_dpo_trainer import OnlineDPOTrainer

__all__ = ("OnlineDPOConfig", "OnlineDPOTrainer")
