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

"""EAGLE3 speculative drafter model family.

Exports the config, target-conditioned drafter model, refinement layer, and
forward-output schema for DeepSpec EAGLE3 training/inference flows.
"""

from .eagle3_configuration import Eagle3Config
from .modeling_eagle3 import Eagle3Attention, Eagle3DecoderLayer, Eagle3ForwardOutput, Eagle3Model

__all__ = ("Eagle3Attention", "Eagle3Config", "Eagle3DecoderLayer", "Eagle3ForwardOutput", "Eagle3Model")
