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

"""Single-Rollout Asynchronous Optimization trainer."""

from __future__ import annotations

import typing as tp

import spectrax as spx

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from .._online_rl.trainer import OnlineActorCriticTrainer, RolloutProvider
from ..proximal_policy_optimization_trainer.ppo_trainer import RewardFunc
from .sao_config import SAOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource


@Registry.register(
    "trainer",
    ["sao", "single-rollout-asynchronous-optimization", "single_rollout_asynchronous_optimization"],
)
class SAOTrainer(OnlineActorCriticTrainer):
    """Train an actor with SAO's direct double-sided importance sampling.

    The trainer consumes one rollout per prompt, updates a separate critic
    before computing actor advantages, bridges GAE across observation tokens,
    and runs one policy update using the strict DIS interval.  Behavior-policy
    ratios are detached in the score-function surrogate.

    A custom ``rollout_provider`` enables multi-turn environments and bounded
    completion-order asynchronous collection.  It must return full trajectory
    rows with an ``attention_mask`` and a target-aligned ``action_mask``.
    Without a provider, SAO reuses EasyDeL's synchronous prompt-only rollout
    path and recomputes behavior log probabilities from the rollout snapshot.

    Args:
        arguments: SAO hyperparameters.
        model: Actor module or state.
        reward_funcs: Reward functions/models.
        critic_model: Optional independently initialized critic base model.
        critic_state: Optional prebuilt value-head critic state.
        critic_trainable_selector: Exact SpectraX selector for critic
            parameters. Overrides ``critic_trainable_mode``.
        rollout_provider: Optional multi-turn/asynchronous rollout callable.
        environment_factory: Optional environment factory exposed to the
            rollout provider.
        tools: Optional tool definitions exposed to the rollout provider and
            inherited prompt chat template.
        train_dataset: Prompt-only training dataset.
        eval_dataset: Optional evaluation dataset(s).
        processing_class: Tokenizer/processor.
        reward_processing_classes: Optional reward-model processors.
        data_tokenize_fn: Optional tokenizer override.
    """

    actor_algorithm: tp.ClassVar[tp.Literal["sao", "ppo"]] = "sao"
    arguments: SAOConfig

    def __init__(
        self,
        *,
        arguments: SAOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc] | None,
        critic_model: EasyDeLBaseModule | None = None,
        critic_state: EasyDeLState | None = None,
        critic_trainable_selector: spx.SelectorSugar | None = None,
        rollout_provider: RolloutProvider | None = None,
        environment_factory: tp.Callable[[], tp.Any] | None = None,
        tools: list[dict | tp.Callable] | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        data_tokenize_fn: tp.Callable | None = None,
    ) -> None:
        """Initialize SAO actor, reference, critic, and rollout surfaces."""
        if not isinstance(arguments, SAOConfig):
            raise TypeError(f"arguments must be SAOConfig, got {type(arguments)}")
        if tools is not None:
            arguments.tools = list(tools)
        self.environment_factory = environment_factory
        self.tools = tools
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            critic_model=critic_model,
            critic_state=critic_state,
            critic_trainable_selector=critic_trainable_selector,
            rollout_provider=rollout_provider,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )


__all__ = ("SAOTrainer",)
