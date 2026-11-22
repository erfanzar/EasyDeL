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
"""Configuration dataclass for the GSPO trainer.

Defines :class:`GSPOConfig` as a thin subclass of :class:`GRPOConfig`
that switches the default ``importance_sampling_level`` to
``"sequence"`` and clamps the loss type to GSPO-friendly variants.
"""

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "gspo")
@dataclass
class GSPOConfig(GRPOConfig):
    """Configuration class for Group Sequence Policy Optimization (GSPO).

    GSPO (Qwen team, arXiv:2507.18071) is a sequence-level variant of
    GRPO that aggregates the importance-sampling ratio over the full
    completion before applying the PPO clip, instead of clipping
    per-token. This change yields markedly more stable optimisation
    on MoE backbones (where individual token ratios can be highly
    bimodal across experts) and lets GSPO operate with very tight
    clipping bounds (``epsilon ~ 3e-4``).

    Compared with the GRPO defaults inherited from
    :class:`GRPOConfig`, GSPO flips:

    * ``importance_sampling_level = "sequence"`` (vs. ``"token"``).
    * ``epsilon = 3e-4`` and ``epsilon_high = 4e-4`` (vs. ``0.2``).
    * ``beta = 0.0`` -- the canonical GSPO recipe drops the
      reference-KL penalty, relying on the tight ratio clip alone for
      regularisation.
    * ``loss_type = "grpo"`` (vs. ``"dapo"`` in :class:`GRPOConfig`).

    All other fields inherited from :class:`GRPOConfig` are kept as
    documented there. Construct with dict-literal kwargs:

    >>> cfg = GSPOConfig(num_generations=4, max_prompt_length=512,
    ...                  max_completion_length=256)

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"GSPO"``).
        importance_sampling_level: ``"sequence"`` (default) or
            ``"token"``. GSPO requires sequence-level aggregation for
            its tight clipping recipe.
        epsilon: Lower clipping bound on the sequence-level
            importance ratio. Default ``3e-4``.
        epsilon_high: Upper clipping bound. Default ``4e-4``.
        beta: KL coefficient against the reference model. Default
            ``0.0`` (no KL penalty).
        loss_type: GRPO loss variant. Default ``"grpo"``; other
            entries from the GRPO surface (``"bnpo"``, ``"dr_grpo"``,
            ``"dapo"``, ``"cispo"``) are accepted.
    """

    trainer_prefix: str | None = field(
        default="GSPO",
        metadata={"help": "default prefix name for trainer."},
    )
    importance_sampling_level: str = field(
        default="sequence",
        metadata={
            "help": "Importance sampling applied per 'token' or aggregated per 'sequence'. "
            "GSPO uses sequence-level importance sampling by default."
        },
    )
    epsilon: float = field(
        default=3e-4,
        metadata={
            "help": "Lower clipping bound for importance sampling weights. "
            "GSPO uses much smaller values than GRPO (3e-4 vs 0.2)."
        },
    )
    epsilon_high: float | None = field(
        default=4e-4,
        metadata={
            "help": "Upper clipping bound for importance sampling weights. "
            "GSPO uses much smaller values than GRPO (4e-4 vs 0.2)."
        },
    )
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL regularization coefficient. GSPO sets this to 0 by default, meaning no KL penalty is applied."
        },
    )
    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Loss variant to use. GSPO uses 'grpo' loss type by default. "
            "Options: ['grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo']."
        },
    )

    __hash__ = hash_fn
