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

"""DFlash drafter modules.

DFlash reuses the DSpark parallel block-drafter backbone but removes the Markov
and confidence branches.  It is the CE-only DeepSpec variant for training block
draft logits from target hidden-state context.
"""

from __future__ import annotations

from easydel.infra.factory import TaskType, register_module
from easydel.modules._drafting import DraftModelOutput
from easydel.modules.dspark.modeling_dspark import DSparkAttention, DSparkDecoderLayer, DSparkModel

from .dflash_configuration import DFlashConfig

DFlashForwardOutput = DraftModelOutput


class DFlashAttention(DSparkAttention):
    """DSpark block attention reused by the DFlash CE-only drafter."""


class DFlashDecoderLayer(DSparkDecoderLayer):
    """DSpark block-refinement layer used unchanged by DFlash."""


@register_module(TaskType.BASE_MODULE, config=DFlashConfig, model_type="dflash")
class DFlashModel(DSparkModel):
    """DeepSpec DFlash block drafter.

    The class inherits the DSpark block forward path, but `DFlashConfig`
    disables Markov bias and confidence prediction by default.  Outputs keep the
    DSpark block schema so DFlash and DSpark losses can share batching code.
    """

    _task_type = TaskType.BASE_MODULE
    _model_type = "dflash"
    _config_class = DFlashConfig


__all__ = ("DFlashAttention", "DFlashDecoderLayer", "DFlashForwardOutput", "DFlashModel")
