# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compaction Reinforcement Learning trainer."""

from __future__ import annotations

import typing as tp

import jax
import spectrax as spx

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from .._online_rl.trainer import OnlineActorCriticTrainer, RolloutProvider
from ..proximal_policy_optimization_trainer.ppo_trainer import RewardFunc
from ._fn import apply_cross_segment_gae_correction
from .compaction_rl_config import CompactionRLConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource


@Registry.register(
    "trainer",
    ["compaction_rl", "compaction-rl", "compaction_reinforcement_learning"],
)
class CompactionRLTrainer(OnlineActorCriticTrainer):
    """Train execution and summary segments with critic-based CompactionRL.

    The actor uses a clipped PPO objective, while an independent critic is
    updated first.  Provider batches may place each execution or summary
    segment in its own full-context row.  When ``trajectory_ids`` and
    ``segment_ids`` are present, local advantages are multiplied by the
    published cross-segment factor

    ``(gamma * lambda) ** number_of_action_tokens_in_later_segments``.

    Compaction policy, tool-call guarding, and environment orchestration are
    intentionally injected rather than hidden in the loss.  The reusable
    :mod:`easydel.trainers.agentic_rollout` components provide the default
    controller and two-stage guard contracts.  A rollout provider uses those
    components to generate exact token spans and returns them to this trainer.

    Without a provider, the trainer remains useful as a one-segment PPO
    canary, but no context compaction occurs.
    """

    actor_algorithm: tp.ClassVar[tp.Literal["sao", "ppo"]] = "ppo"
    arguments: CompactionRLConfig

    def __init__(
        self,
        *,
        arguments: CompactionRLConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc] | None,
        critic_model: EasyDeLBaseModule | None = None,
        critic_state: EasyDeLState | None = None,
        critic_trainable_selector: spx.SelectorSugar | None = None,
        rollout_provider: RolloutProvider | None = None,
        compaction_controller: tp.Any | None = None,
        tool_guard: tp.Any | None = None,
        environment_factory: tp.Callable[[], tp.Any] | None = None,
        tools: list[dict | tp.Callable] | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        data_tokenize_fn: tp.Callable | None = None,
    ) -> None:
        """Initialize CompactionRL actor, critic, and rollout controls."""
        if not isinstance(arguments, CompactionRLConfig):
            raise TypeError(f"arguments must be CompactionRLConfig, got {type(arguments)}")
        if tools is not None:
            arguments.tools = list(tools)
        if compaction_controller is None:
            try:
                from ..agentic_rollout import CompactionController

                compaction_controller = CompactionController(
                    context_budget=arguments.context_budget,
                    compaction_threshold=arguments.compaction_threshold,
                    recent_pairs=arguments.recent_steps,
                    max_compactions=arguments.max_compactions,
                    train_summary_tokens=arguments.train_summary_tokens,
                    summary_instruction=arguments.summary_instruction,
                    resume_template=arguments.resume_template,
                    failure_policy=arguments.compaction_failure_policy,
                )
            except ImportError:
                compaction_controller = None
        self.compaction_controller = compaction_controller
        self.tool_guard = tool_guard
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

    def _transform_policy_advantages(
        self,
        advantages: jax.Array,
        batch: dict[str, jax.Array],
        lambdas: jax.Array,
    ) -> jax.Array:
        """Apply CompactionRL's cross-segment token-count correction."""
        if not self.arguments.cross_trajectory_gae:
            return advantages
        trajectory_ids = batch.get("trajectory_ids")
        segment_ids = batch.get("segment_ids")
        if trajectory_ids is None or segment_ids is None:
            if getattr(self, "rollout_provider", None) is not None:
                raise ValueError(
                    "CompactionRL rollout-provider batches require row-level "
                    "`trajectory_ids` and `segment_ids` for cross-segment GAE."
                )
            return advantages

        return apply_cross_segment_gae_correction(
            advantages,
            trajectory_ids,
            segment_ids,
            batch["action_mask"],
            gamma=float(self.arguments.gamma),
            gae_lambdas=lambdas,
        )


__all__ = ("CompactionRLTrainer",)
