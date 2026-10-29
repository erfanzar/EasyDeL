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

"""Public exports for triple preference optimization.

TPO extends DPO-style preference training to chosen, rejected, and reference
responses with a dedicated collator, preprocessing transform, and loss.
"""

from ._fn import compute_tpo_losses, tpo_concatenated_forward, tpo_evaluation_step, tpo_training_step
from .tpo_config import TPOConfig
from .tpo_preprocess import (
    DataCollatorForTriplePreferenceGrain,
    DataCollatorForTriplePreferenceTFDS,
    TPOPreprocessTransform,
)
from .tpo_trainer import TPOTrainer

__all__ = (
    "DataCollatorForTriplePreferenceGrain",
    "DataCollatorForTriplePreferenceTFDS",
    "TPOConfig",
    "TPOPreprocessTransform",
    "TPOTrainer",
    "compute_tpo_losses",
    "tpo_concatenated_forward",
    "tpo_evaluation_step",
    "tpo_training_step",
)
