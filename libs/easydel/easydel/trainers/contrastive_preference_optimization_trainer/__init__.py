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
"""Contrastive Preference Optimization (CPO) trainer entry point.

CPO drops the reference model entirely and minimises a max-margin
preference loss with an auxiliary supervised log-likelihood term on the
chosen response.  This package re-exports :class:`CPOConfig` and
:class:`CPOTrainer`; algorithm internals live in :mod:`._fn`.
"""

from .cpo_config import CPOConfig
from .cpo_trainer import CPOTrainer

__all__ = ("CPOConfig", "CPOTrainer")
