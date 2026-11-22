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
"""Group Filtered Policy Optimization using the shared GRPO rollout pipeline."""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from ..reward_protocol import RewardProtocol
from .gfpo_config import GFPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

RewardFunc = EasyDeLBaseModule | EasyDeLState | RewardProtocol | tp.Callable[[list, list], list[float]]
GroupFilterFunc = tp.Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


@Registry.register("trainer", "gfpo")
class GFPOTrainer(GRPOTrainer):
    """Generate extra rollouts and keep the most efficient samples per prompt.

    Filtering is the only GFPO-specific rollout stage. Generation reuse, tools,
    environment feedback, evaluation sampling, reward aggregation, and advantage
    estimation are inherited from GRPO and operate on the selected samples.
    """

    arguments: GFPOConfig

    def __init__(
        self,
        arguments: GFPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
        group_filter_func: GroupFilterFunc | None = None,
        tools: list[dict | str | tp.Callable] | None = None,
        environment_factory: tp.Callable | None = None,
    ):
        """Initialize GRPO with an optional custom per-completion filter score.

        Args:
            arguments: GFPO sampling, filtering, and training configuration.
            model: Initialized policy model or state.
            reward_funcs: Reward functions or reward models.
            train_dataset: Prompt dataset for training.
            eval_dataset: Optional evaluation prompts.
            processing_class: Policy tokenizer or processor.
            reward_processing_classes: Optional reward processors.
            data_tokenize_fn: Optional dataset tokenizer.
            group_filter_func: Callable taking completion ids, combined rewards,
                and loss masks, returning scores where higher means better.
            tools: Tool definitions used by the inherited rollout loop.
            environment_factory: Optional environment factory for reward feedback.
        """
        if not isinstance(arguments, GFPOConfig):
            raise TypeError(f"arguments must be `GFPOConfig` but got {type(arguments)}.")
        self.group_filter_func = group_filter_func
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

    def configure_functions(self):
        """Compile GRPO steps with the effective post-filter group sizes."""
        output = super().configure_functions()
        if self.arguments.num_remains_in_group is not None:
            effective_generations = self.arguments.num_remains_in_group
            self._train_shared_fn_static_args = (
                effective_generations,
                *self._train_shared_fn_static_args[1:],
            )
            eval_generations = self._eval_shared_fn_static_args[0]
            self._eval_shared_fn_static_args = (
                min(effective_generations, int(eval_generations)),
                *self._eval_shared_fn_static_args[1:],
            )
        return output

    def _default_filter_func(self, completion_ids, rewards, completion_mask):
        """Score completions by weighted reward-per-token and brevity."""
        lengths = jnp.sum(completion_mask, axis=-1)
        scores = jnp.zeros_like(rewards)
        if self.arguments.filter_by_efficiency:
            efficiency = rewards / jnp.maximum(lengths, 1.0)
            efficiency_norm = efficiency / (jnp.abs(efficiency).max() + 1e-8)
            scores = scores + self.arguments.efficiency_weight * efficiency_norm
        if self.arguments.filter_by_length:
            length_score = 1.0 - lengths / completion_mask.shape[-1]
            scores = scores + self.arguments.length_weight * length_score
        return scores

    def _select_rollout_indices(self, completion_ids, completion_mask, rewards, generation_factor):
        """Return top-scoring row indices, preserving prompt-group order."""
        num_remains = self.arguments.num_remains_in_group
        if num_remains is None or num_remains >= generation_factor:
            return None
        return self._rank_rollout_indices(completion_ids, completion_mask, rewards, generation_factor, num_remains)

    def _rank_rollout_indices(self, completion_ids, completion_mask, rewards, num_generations, num_remains):
        """Rank each prompt group for both the rollout hook and legacy filtering API."""
        filter_func = self.group_filter_func or self._default_filter_func
        scores = filter_func(completion_ids, rewards, completion_mask).reshape(-1, num_generations)
        _, top_indices = jax.lax.top_k(scores, num_remains)
        offsets = jnp.arange(scores.shape[0])[:, None] * num_generations
        return (offsets + top_indices).reshape(-1)

    def _filter_completions(
        self,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        ref_per_token_logps,
        rewards,
        rewards_per_func,
        completion_prompts,
        num_generations,
        num_remains,
    ):
        """Filter an already scored group (compatibility wrapper for callers)."""
        gather_indices = self._rank_rollout_indices(
            completion_ids, completion_mask, rewards, num_generations, num_remains
        )
        host_indices = jax.device_get(gather_indices).tolist()
        return (
            prompt_ids,
            prompt_mask,
            completion_ids[gather_indices],
            completion_mask[gather_indices],
            ref_per_token_logps[gather_indices],
            rewards[gather_indices],
            rewards_per_func[gather_indices],
            [completion_prompts[i] for i in host_indices],
            num_remains,
        )
