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
"""Configuration dataclass for the REINFORCE++ trainer.

Defines :class:`ReinforcePlusPlusConfig` as a thin subclass of
:class:`GRPOConfig` with batch-level advantage normalization and the
``k2`` KL estimator.
"""

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "reinforce_plus_plus")
@dataclass
class ReinforcePlusPlusConfig(GRPOConfig):
    """Configuration class for REINFORCE++ (arXiv:2501.03262).

    REINFORCE++ keeps the group-mean baseline but standardizes the
    group-centred advantages ``A = r - mean_group(r)`` over the *global
    batch*: ``(A - mean_batch(A)) / (std_batch(A) + eps)``. Unlike dividing
    by the std of the raw rewards, this scale excludes between-group
    variance, and unlike per-group std scaling it does not inflate easy or
    hard prompts. The KL penalty uses the ``k2``
    estimator (``0.5 * (log r)^2``), whose gradient is an unbiased
    estimate of the reverse-KL gradient.

    Compared with :class:`GRPOConfig` the defaults are:

    * ``scale_rewards = "batch_advantage"``.
    * ``kl_estimator = "k2"``.

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"REINFORCE++"``).
        scale_rewards: Reward scaling strategy. Default ``"batch_advantage"``.
        kl_estimator: KL estimator for the ``beta`` penalty. Default
            ``"k2"``.
    """

    trainer_prefix: str | None = field(
        default="REINFORCE++",
        metadata={"help": "default prefix name for trainer."},
    )
    scale_rewards: str | bool = field(
        default="batch_advantage",
        metadata={
            "help": (
                "Reward scaling strategy. REINFORCE++ standardizes group-centred advantages over the batch "
                "('batch_advantage')."
            )
        },
    )
    kl_estimator: str = field(
        default="k2",
        metadata={"help": "Per-token KL estimator for the beta penalty. REINFORCE++ uses 'k2'."},
    )

    __hash__ = hash_fn
