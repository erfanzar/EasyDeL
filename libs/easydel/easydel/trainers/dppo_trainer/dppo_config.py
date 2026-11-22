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
"""DPPO config and trainer surface backed by GRPO."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "dppo")
@dataclass
class DPPOConfig(GRPOConfig):
    """Configuration for Direct Preference Policy Optimization on GRPO batches.

    EasyDeL wires DPPO into the GRPO rollout/loss path. Binary variants compare
    the sampled-token probability under the generation-time policy and the
    current policy. Top-k variants additionally store the generation-time
    top-k token support and compare the current policy on that same support.
    """

    trainer_prefix: str | None = field(default="DPPO")
    divergence_type: tp.Literal["binary_tv", "binary_kl", "topk_tv", "topk_kl"] = field(default="binary_tv")
    divergence_topk: int = field(default=20)
    clip_ratio_c: float = field(default=20.0)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Finalize DPPO-specific invariants before GRPO validation runs.

        DPPO is implemented as a GRPO loss branch, so this method pins
        ``loss_type`` to ``"dppo"`` before delegating to ``GRPOConfig``. It then
        validates the DPPO-only knobs that affect divergence computation:
        ``divergence_topk`` must select at least one vocabulary entry for top-k
        TV/KL modes, and ``clip_ratio_c`` must be positive because it bounds the
        probability-ratio correction.
        """
        self.loss_type = "dppo"
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if self.divergence_topk < 1:
            raise ValueError("`divergence_topk` must be >= 1.")
        if self.clip_ratio_c <= 0:
            raise ValueError("`clip_ratio_c` must be positive.")
