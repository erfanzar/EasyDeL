# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "gspo")
@dataclass
class GSPOConfig(GRPOConfig):
    """Configuration class for Group Sequence Policy Optimization training.

    GSPO is a variant of GRPO that operates at the sequence level rather than
    token level, providing improved training stability and performance, especially
    for Mixture-of-Experts (MoE) models. It was developed by Alibaba Qwen team
    and contributed to the improvements in Qwen3 models.

    Key differences from GRPO:
    - Sequence-level importance ratios instead of token-level
    - Much smaller clipping bounds (3e-4 to 4e-4 vs 0.2)
    - No KL regularization by default (beta=0.0)
    - Better stability for MoE model training

    Reference:
        Group Sequence Policy Optimization (arXiv:2507.18071)

    Example:
        >>> config = GSPOConfig(
        ...     per_device_train_batch_size=4,
        ...     num_generations=4,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ...     learning_rate=1e-6,
        ... )
        >>> trainer = GSPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=reward_model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    trainer_prefix: str | None = field(
        default="gspotrainer",
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
