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
"""GSPO token-level importance-sampling aliases."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..group_sequence_policy_optimization import GSPOConfig


@Registry.register("trainer-arguments", "gspo_token")
@dataclass
class GSPOTokenConfig(GSPOConfig):
    """GSPO configuration fixed to token-aware sequence importance sampling.

    This alias exists for users who want the token-level GSPO surface while
    still using the canonical EasyDeL GSPO implementation. Validation enforces
    ``importance_sampling_level='sequence_token'`` so the alias cannot silently
    drift into another objective.
    """

    trainer_prefix: str | None = field(default="GSPOToken")
    importance_sampling_level: str = field(default="sequence_token")

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Finalize GSPO config and enforce the token-level alias contract.

        ``GSPOTokenConfig`` is a registry alias, not a new trainer algorithm.
        After the base GSPO config validates shared fields, this hook asserts
        that the importance-sampling level remains ``"sequence_token"`` so a
        ``gspo_token`` run cannot silently become standard sequence-level GSPO.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if self.importance_sampling_level != "sequence_token":
            raise ValueError("GSPOTokenConfig requires `importance_sampling_level='sequence_token'`.")
