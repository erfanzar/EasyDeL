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
"""Group reward-Decoupled Policy Optimization (GDPO) trainer.

Thin subclass of :class:`GRPOTrainer` that enforces a :class:`GDPOConfig`;
the decoupled normalization lives in GRPO's ``normalize_then_sum``
advantage path.
"""

from __future__ import annotations

import typing as tp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from ..reward_protocol import RewardProtocol
from .gdpo_config import GDPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource


RewardFunc = EasyDeLBaseModule | EasyDeLState | RewardProtocol | tp.Callable[[list, list], list[float]]


@Registry.register("trainer", "gdpo")
class GDPOTrainer(GRPOTrainer):
    """Group reward-Decoupled Policy Optimization trainer.

    Each reward function is normalized within its prompt group before
    the weighted sum, and the summed reward is then normalized over the
    batch. Everything else is inherited from GRPO.

    Reference:
        GDPO: Group reward-Decoupled Normalization Policy Optimization
        (arXiv:2601.05242)
    """

    arguments: GDPOConfig

    def __init__(
        self,
        arguments: GDPOConfig,
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
        """Initialize the GDPO trainer.

        Args:
            arguments: GDPO configuration.
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
            TypeError: If ``arguments`` is not a :class:`GDPOConfig`.
        """
        if not isinstance(arguments, GDPOConfig):
            raise TypeError(f"arguments must be `GDPOConfig` but got {type(arguments)}.")
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
