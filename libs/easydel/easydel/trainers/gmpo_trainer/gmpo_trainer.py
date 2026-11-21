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
"""Geometric-Mean Policy Optimization (GMPO) trainer.

Thin subclass of :class:`GRPOTrainer` that enforces a :class:`GMPOConfig`;
the geometric-mean surrogate itself lives in GRPO's ``loss_type="gmpo"``
branch.
"""

from __future__ import annotations

import typing as tp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from ..reward_protocol import RewardProtocol
from .gmpo_config import GMPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource


RewardFunc = EasyDeLBaseModule | EasyDeLState | RewardProtocol | tp.Callable[[list, list], list[float]]


@Registry.register("trainer", "gmpo")
class GMPOTrainer(GRPOTrainer):
    """Geometric-Mean Policy Optimization trainer.

    Optimises ``-A * exp(mean_t log(clip(r_t)))`` per sequence, where
    ``clip`` is advantage-sign aware: positive advantages cannot push a
    token ratio above ``1 + epsilon_high`` and negative advantages cannot
    push it below ``1 - epsilon``. Generation, rewards, and advantages
    are inherited from GRPO.

    Reference:
        Geometric-Mean Policy Optimization (arXiv:2507.20673)
    """

    arguments: GMPOConfig

    def __init__(
        self,
        arguments: GMPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        data_tokenize_fn: tp.Callable | None = None,
        tools: list[dict | str | tp.Callable] | None = None,
        environment_factory: tp.Callable[[], object] | None = None,
    ):
        """Initialize the GMPO trainer.

        Args:
            arguments: GMPO configuration.
            model: Policy module or state.
            reward_funcs: Reward function(s) or reward model(s).
            train_dataset: Prompt dataset for training.
            eval_dataset: Optional evaluation prompts.
            processing_class: Policy tokenizer or processor.
            reward_processing_classes: Optional reward processors.
            data_tokenize_fn: Optional dataset tokenizer.
            tools: Tool definitions used by the inherited rollout loop.
            environment_factory: Optional environment factory for reward feedback.

        Raises:
            TypeError: If ``arguments`` is not a :class:`GMPOConfig`.
        """
        if not isinstance(arguments, GMPOConfig):
            raise TypeError(f"arguments must be `GMPOConfig` but got {type(arguments)}.")
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
            tools=tools,
            environment_factory=environment_factory,
        )
