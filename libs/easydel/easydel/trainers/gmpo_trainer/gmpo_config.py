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
"""Configuration dataclass for the GMPO trainer.

Defines :class:`GMPOConfig` as a thin subclass of :class:`GRPOConfig`
that selects ``loss_type="gmpo"`` and the wider GMPO clip range.
"""

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "gmpo")
@dataclass
class GMPOConfig(GRPOConfig):
    """Configuration class for Geometric-Mean Policy Optimization (GMPO).

    GMPO (arXiv:2507.20673) replaces GRPO's arithmetic mean of per-token
    clipped surrogates with the geometric mean of the clipped token
    ratios of each sequence. The geometric mean is far less sensitive to
    outlier token ratios, so GMPO tolerates a wider clip range than GRPO.

    Compared with :class:`GRPOConfig` the defaults are:

    * ``loss_type = "gmpo"``.
    * ``epsilon = 0.4`` and ``epsilon_high = 0.4``.

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"GMPO"``).
        loss_type: Loss variant. Default ``"gmpo"``.
        epsilon: Lower clip offset on token ratios (``1 - epsilon``).
            Default ``0.4``; must satisfy ``0 < epsilon < 1``.
        epsilon_high: Upper clip offset on token ratios
            (``1 + epsilon_high``). Default ``0.4``.
    """

    trainer_prefix: str | None = field(
        default="GMPO",
        metadata={"help": "default prefix name for trainer."},
    )
    loss_type: str = field(
        default="gmpo",
        metadata={"help": "Loss variant to use. GMPO uses the geometric-mean 'gmpo' surrogate by default."},
    )
    epsilon: float = field(
        default=0.4,
        metadata={"help": "Lower clipping offset for token importance ratios (ratio >= 1 - epsilon)."},
    )
    epsilon_high: float | None = field(
        default=0.4,
        metadata={"help": "Upper clipping offset for token importance ratios (ratio <= 1 + epsilon_high)."},
    )

    __hash__ = hash_fn
