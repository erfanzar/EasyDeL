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

"""Public exports for BEMA reference-model update helpers.

BEMA wraps DPO with beta/exponential moving-average reference updates, exposing
both the callback and the trainer alias from one import location.
"""

from .bema_config import BEMACallback, BEMAConfig
from .bema_trainer import BEMADPOTrainer

__all__ = ("BEMACallback", "BEMAConfig", "BEMADPOTrainer")
