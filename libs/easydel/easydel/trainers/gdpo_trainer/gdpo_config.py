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
"""Configuration dataclass for the GDPO trainer.

Defines :class:`GDPOConfig` as a thin subclass of :class:`GRPOConfig`
that enables decoupled per-reward group normalization.
"""

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "gdpo")
@dataclass
class GDPOConfig(GRPOConfig):
    """Configuration class for Group reward-Decoupled Policy Optimization (GDPO).

    GDPO (arXiv:2601.05242) targets multi-reward RL. Summing raw rewards
    before GRPO's group normalization lets distinct reward combinations
    collapse onto the same advantage; GDPO instead normalizes every
    reward function within its prompt group, takes the weighted sum, and
    then normalizes the summed reward over the whole batch.

    Compared with :class:`GRPOConfig` the default is
    ``decouple_reward_normalization=True`` (which selects
    ``multi_objective_aggregation="normalize_then_sum"``).

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"GDPO"``).
        decouple_reward_normalization: Per-reward group normalization
            switch. Default ``True``.
    """

    trainer_prefix: str | None = field(
        default="GDPO",
        metadata={"help": "default prefix name for trainer."},
    )
    decouple_reward_normalization: bool = field(
        default=True,
        metadata={
            "help": (
                "Normalize each reward function within its prompt group before the weighted sum, then "
                "batch-normalize the summed reward."
            )
        },
    )

    __hash__ = hash_fn
