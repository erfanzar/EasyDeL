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

"""Trainer package for fitting speculative-decoding draft models.

The package exports the config and trainer used to optimize a trainable
drafter against a frozen target model.  The trainer supports plain
next-token drafters, explicit multi-token draft heads, and target-conditioned
draft modules that consume cached target logits or hidden states.
"""

from .spec_config import SpeculativeDecodingConfig
from .spec_trainer import SpeculativeDecodingTrainer

__all__ = ("SpeculativeDecodingConfig", "SpeculativeDecodingTrainer")
