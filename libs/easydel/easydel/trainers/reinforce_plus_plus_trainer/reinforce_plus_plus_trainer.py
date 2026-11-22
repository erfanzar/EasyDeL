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
"""REINFORCE++ trainer.

Thin subclass of :class:`GRPOTrainer` that enforces a
:class:`ReinforcePlusPlusConfig`; batch advantage normalization and the
``k2`` KL estimator are GRPO knobs.
"""

from __future__ import annotations

import typing as tp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from ..reward_protocol import RewardProtocol
from .reinforce_plus_plus_config import ReinforcePlusPlusConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource


RewardFunc = EasyDeLBaseModule | EasyDeLState | RewardProtocol | tp.Callable[[list, list], list[float]]


@Registry.register("trainer", "reinforce_plus_plus")
class ReinforcePlusPlusTrainer(GRPOTrainer):
    """REINFORCE++ trainer.

    Uses a group-mean baseline with global-batch advantage
    standardization and a ``k2`` KL penalty against the reference
    policy. Generation, rewards, and the policy loss are inherited from
    GRPO.

    Reference:
        REINFORCE++: Stabilizing Critic-Free Policy Optimization with
        Global Advantage Normalization (arXiv:2501.03262)
    """

    arguments: ReinforcePlusPlusConfig

    def __init__(
        self,
        arguments: ReinforcePlusPlusConfig,
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
        """Initialize the REINFORCE++ trainer.

        Args:
            arguments: REINFORCE++ configuration.
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
            TypeError: If ``arguments`` is not a :class:`ReinforcePlusPlusConfig`.
        """
        if not isinstance(arguments, ReinforcePlusPlusConfig):
            raise TypeError(f"arguments must be `ReinforcePlusPlusConfig` but got {type(arguments)}.")
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
