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
"""NeMo Gym style GRPO trainer backed by EasyDeL/eSurge generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "nemo_gym")
@dataclass
class NeMoGymConfig(GRPOConfig):
    """Configuration for NeMo Gym style environment training.

    Policy completions are produced by the inherited eSurge
    ``generate_unified`` path, then the configured environment factory scores
    each completion.
    """

    trainer_prefix: str | None = field(default="NeMoGym")
    use_esurge_generation: bool = field(default=True)
    shuffle_dataset: bool | None = field(default=False)
    num_generations_eval: int | None = field(default=1)
    request_timeout: float = field(default=10800.0)
    metadata_key: str = field(default="metadata")
    agent_ref_key: str = field(default="agent_ref")
    environment_reward_weight: float = field(default=1.0)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Normalize inherited GRPO fields and validate environment settings.

        NeMo Gym always uses EasyDeL/eSurge generation for local rollouts, so
        ``use_esurge_generation`` is forced after the parent config has
        normalized generation lengths and sharding settings. The method rejects
        non-positive timeout and reward-weight values because those would make
        environment scoring either impossible or silently ignored.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        self.use_esurge_generation = True
        if self.request_timeout <= 0.0:
            raise ValueError("`request_timeout` must be positive.")
        if self.environment_reward_weight <= 0.0:
            raise ValueError("`environment_reward_weight` must be positive.")

    __hash__ = hash_fn
