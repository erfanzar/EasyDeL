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

"""Configuration for DFlash.

DFlash is the DSpark block-drafter backbone with the Markov and confidence
branches disabled.  It keeps the same target-feature and block-output contract
as DSpark, but trains as a CE-only parallel block drafter.
"""

from __future__ import annotations

from easydel.infra.factory import register_config
from easydel.modules.dspark.dspark_configuration import DSparkConfig


@register_config("dflash")
class DFlashConfig(DSparkConfig):
    """DSpark-derived config for the DFlash CE-only drafter."""

    model_type = "dflash"

    def __init__(self, **kwargs):
        """Initialize DFlash with DSpark extras disabled by default.

        DFlash defaults override the parent ``DSparkConfig`` values to disable
        Markov and confidence branches, setting CE-only training mode.

        Args:
            **kwargs: Unpacked[DSparkConfig].  Forwarded to ``DSparkConfig.__init__``.
                Includes all ``DSparkConfig`` and ``EasyDeLBaseConfig`` fields.
                DFlash-specific overrides applied by default:
                markov_rank (int): Defaults to ``0`` (Markov disabled).
                confidence_head_alpha (float): Defaults to ``0.0`` (confidence disabled).
                confidence_head_with_markov (bool): Defaults to ``False``.
                ce_loss_alpha (float): Defaults to ``1.0`` (pure CE loss).
                l1_loss_alpha (float): Defaults to ``0.0`` (L1 disabled).

        Returns:
            None.
        """
        kwargs.setdefault("markov_rank", 0)
        kwargs.setdefault("confidence_head_alpha", 0.0)
        kwargs.setdefault("confidence_head_with_markov", False)
        kwargs.setdefault("ce_loss_alpha", 1.0)
        kwargs.setdefault("l1_loss_alpha", 0.0)
        super().__init__(**kwargs)


__all__ = ("DFlashConfig",)
