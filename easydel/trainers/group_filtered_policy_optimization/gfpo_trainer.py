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
"""Group Filtered Policy Optimization (GFPO) trainer.

GFPO subclasses :class:`GRPOTrainer` and rewrites the
``_preprocess_batch_input`` hook so that, for every prompt group, only
a "keep" subset of rollouts (chosen by length / diversity / token-level
reward) survives into the gradient computation.  This counteracts the
length-inflation pathology of large-scale RL fine-tuning.
"""

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from ..group_relative_policy_optimization import GRPOTrainer
from ..prompt_utils import apply_chat_template
from ..training_utils import (
    extract_generation_model_kwargs,
    normalize_generation_model_kwargs,
    repeat_prompt_aligned_model_kwargs,
    slice_prompt_aligned_model_kwargs,
    strip_prompt_only_scoring_model_kwargs,
    validate_prompt_aligned_generation_model_kwargs,
)
from .gfpo_config import GFPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource


RewardFunc = EasyDeLBaseModule | EasyDeLState | tp.Callable[[list, list], list[float]]
GroupFilterFunc = tp.Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


@Registry.register("trainer", "gfpo")
class GFPOTrainer(GRPOTrainer):
    """Group Filtered Policy Optimization trainer for RLHF.

    GFPO (Group Filtered Policy Optimization) is a Microsoft algorithm that reduces
    response length inflation while maintaining accuracy. It generates more samples
    per prompt during training, then filters to keep only the most efficient ones
    based on length and reward-per-token metrics.

    The key insight is: "Sample more at training time, think less at inference time."
    By training on shorter, more efficient responses, models learn to generate
    concise reasoning at inference time, reducing length inflation by 46-85%.

    Key differences from GRPO:
    - Generates more samples per prompt (num_generations)
    - Filters to keep top K samples (num_remains_in_group)
    - Filters based on response length and reward-per-token efficiency
    - Reduces verbose, repetitive text while maintaining accuracy

    Attributes:
        arguments: GFPOConfig instance with GFPO-specific hyperparameters
        group_filter_func: Custom function to compute filter scores for completions
        ref_state: Reference model state for computing log probabilities
        processing_class: Tokenizer or processor for text encoding

    Reference:
        "Sample More to Think Less: Group Filtered Policy Optimization for
        Concise Reasoning" (arXiv:2508.09726)

    Example:
        >>> config = GFPOConfig(
        ...     per_device_train_batch_size=4,
        ...     num_generations=8,           # Generate more samples
        ...     num_remains_in_group=4,      # Keep top 4 after filtering
        ...     learning_rate=1e-6,
        ... )
        >>> trainer = GFPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=reward_model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
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
    ):
        """Initialize the GFPO trainer.

        Args:
            arguments: GFPOConfig containing training hyperparameters with
                GFPO-specific settings (num_remains_in_group, filter weights).
            model: The policy model to train. Can be an EasyDeLBaseModule or
                EasyDeLState. Will be converted to state if needed.
            reward_funcs: Reward function(s) for scoring completions. Can be
                a single function/model or a list for multi-reward training.
            train_dataset: Training dataset containing prompts.
            eval_dataset: Optional evaluation dataset.
            processing_class: Tokenizer for encoding text.
            reward_processing_classes: Optional separate tokenizers for reward
                models if they differ from the policy model tokenizer.
            data_tokenize_fn: Optional custom tokenization function.
            group_filter_func: Optional custom function to compute filter scores.
                Signature: (completion_ids, rewards, completion_mask) -> scores.
                Higher scores are better. If None, uses default efficiency-based
                filtering.

        Raises:
            TypeError: If arguments is not a GFPOConfig instance.
        """
        if not isinstance(arguments, GFPOConfig):
            raise TypeError(
                f"arguments must be `GFPOConfig` but got {type(arguments)}. "
                "Use GFPOConfig for GFPO training or GRPOConfig for GRPO training."
            )

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
        )

    def configure_functions(self):
        """Configure training functions with filtered generation count.

        Overrides the parent method to use num_remains_in_group instead of
        num_generations in the static args passed to grpo_step. This ensures
        the loss function uses the correct repeat count after filtering.
        """
        output = super().configure_functions()
        straight_through_emulator = self._train_shared_fn_static_args[-1]

        # If filtering is active, replace num_generations with num_remains_in_group
        if self.arguments.num_remains_in_group is not None:
            effective_generations = self.arguments.num_remains_in_group

            self._train_shared_fn_static_args = (
                effective_generations,  # Use filtered count instead of original
                self.arguments.beta,
                self.arguments.loss_config,
                self.scheduler,
                self.arguments.step_partition_spec,
                self.arguments.gradient_accumulation_steps,
                True,  # is_train
                self.loss_type,
                self.epsilon,
                self.epsilon_high,
                self.delta,
                self.importance_sampling_level,
                self.top_entropy_quantile,
                self.arguments.completion_chunk_size,
                self.arguments.max_loss_completion_tokens,
                straight_through_emulator,
            )

            self._eval_shared_fn_static_args = (
                effective_generations,
                self.arguments.beta,
                self.arguments.loss_config,
                self.scheduler,
                self.arguments.step_partition_spec,
                self.arguments.gradient_accumulation_steps,
                False,  # is_train
                self.loss_type,
                self.epsilon,
                self.epsilon_high,
                self.delta,
                self.importance_sampling_level,
                self.top_entropy_quantile,
                self.arguments.completion_chunk_size,
                self.arguments.max_loss_completion_tokens,
                straight_through_emulator,
            )

        return output

    def _default_filter_func(
        self,
        completion_ids: jax.Array,
        rewards: jax.Array,
        completion_mask: jax.Array,
    ) -> jax.Array:
        """Default filter function based on length and reward-per-token efficiency.

        Computes a weighted score combining:
        - Negative length (shorter is better)
        - Reward-per-token efficiency (higher is better)

        Args:
            completion_ids: Token IDs of completions [batch, seq_len]
            rewards: Reward scores for each completion [batch]
            completion_mask: Attention mask for completions [batch, seq_len]

        Returns:
            Filter scores [batch] - higher scores indicate better samples to keep
        """
        lengths = jnp.sum(completion_mask, axis=-1)
        max_length = completion_mask.shape[-1]

        scores = jnp.zeros_like(rewards)

        if self.arguments.filter_by_efficiency:
            efficiency = rewards / jnp.maximum(lengths, 1.0)
            # Normalize efficiency to [0, 1] range approximately
            efficiency_norm = efficiency / (jnp.abs(efficiency).max() + 1e-8)
            scores = scores + self.arguments.efficiency_weight * efficiency_norm

        if self.arguments.filter_by_length:
            # Negative length normalized (shorter = higher score)
            length_score = 1.0 - (lengths / max_length)
            scores = scores + self.arguments.length_weight * length_score

        return scores

    def _filter_completions(
        self,
        prompt_ids: jax.Array,
        prompt_mask: jax.Array,
        completion_ids: jax.Array,
        completion_mask: jax.Array,
        ref_per_token_logps: jax.Array,
        rewards: jax.Array,
        rewards_per_func: jax.Array,
        completion_prompts: list[str],
        num_generations: int,
        num_remains: int,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        list[str],
        int,
    ]:
        """Filter completions to keep top-K samples per group.

        Args:
            prompt_ids: Prompt token IDs [num_prompts, prompt_len]
            prompt_mask: Prompt attention mask [num_prompts, prompt_len]
            completion_ids: Completion token IDs [batch, completion_len]
            completion_mask: Completion attention mask [batch, completion_len]
            ref_per_token_logps: Reference model log probs [batch, completion_len]
            rewards: Combined rewards [batch]
            rewards_per_func: Per-function rewards [batch, num_funcs]
            completion_prompts: Text prompts for each completion
            num_generations: Original number of generations per prompt
            num_remains: Number of samples to keep per group

        Returns:
            Filtered versions of all inputs with shape [num_prompts * num_remains, ...]
        """
        # Compute filter scores
        filter_func = self.group_filter_func or self._default_filter_func
        scores = filter_func(completion_ids, rewards, completion_mask)

        # Reshape to groups: [num_prompts, num_generations]
        num_prompts = prompt_ids.shape[0]
        scores_grouped = scores.reshape(num_prompts, num_generations)

        # Select top-K indices per group
        _, top_indices = jax.lax.top_k(scores_grouped, num_remains)  # [num_prompts, num_remains]

        # Create batch indices for gathering
        batch_offsets = jnp.arange(num_prompts)[:, None] * num_generations  # [num_prompts, 1]
        gather_indices = (batch_offsets + top_indices).reshape(-1)  # [num_prompts * num_remains]

        # Gather filtered samples
        filtered_completion_ids = completion_ids[gather_indices]
        filtered_completion_mask = completion_mask[gather_indices]
        filtered_ref_per_token_logps = ref_per_token_logps[gather_indices]
        filtered_rewards = rewards[gather_indices]
        filtered_rewards_per_func = rewards_per_func[gather_indices]

        # Filter completion prompts (list)
        gather_indices_np = jax.device_get(gather_indices).tolist()
        filtered_completion_prompts = [completion_prompts[i] for i in gather_indices_np]

        return (
            prompt_ids,
            prompt_mask,
            filtered_completion_ids,
            filtered_completion_mask,
            filtered_ref_per_token_logps,
            filtered_rewards,
            filtered_rewards_per_func,
            filtered_completion_prompts,
            num_remains,
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Preprocess batch with GFPO filtering.

        This method extends GRPO's preprocessing by adding a filtering step
        after reward computation to keep only the most efficient samples.
        """
        reward_batch = self._extract_reward_batch_sidechannels(batch)
        batch = self._purify_batch(batch)
        if reward_batch:
            reward_batch = {**batch, **reward_batch}
        else:
            reward_batch = batch
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]
            prompt_model_kwargs = extract_generation_model_kwargs(
                batch, model_callable=getattr(state.model, "forward", state.model)
            )
            scoring_prompt_model_kwargs = strip_prompt_only_scoring_model_kwargs(prompt_model_kwargs)
            validate_prompt_aligned_generation_model_kwargs(
                scoring_prompt_model_kwargs,
                prompt_batch_size=prompt_ids.shape[0],
            )

            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    model_kwargs=prompt_model_kwargs,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                )
                sequences = results.sequences
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids
                completion_prompts = results.completion_prompts

            generation_time = generation_time_fn()
            prompt_completion_ids = sequences

            completion_mask = self._make_attn_mask(completion_ids)
            if self.arguments.mask_truncated_completions:
                eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
                has_eos = jnp.any(jnp.isin(completion_ids, eos_tokens), axis=1)
                completion_mask = completion_mask * has_eos[:, None].astype(completion_mask.dtype)

            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            ridmask = prompt_mask.repeat(generation_factor, 0)
            repeated_prompt_model_kwargs = repeat_prompt_aligned_model_kwargs(
                scoring_prompt_model_kwargs,
                generation_factor,
                prompt_batch_size=prompt_mask.shape[0],
            )
            normalized_repeated_model_kwargs = normalize_generation_model_kwargs(
                repeated_prompt_model_kwargs,
                model_callable=getattr(self.ref_state.model, "forward", self.ref_state.model),
            )
            prompt_completion_mask = jnp.concatenate([ridmask, completion_mask], -1)

            with capture_time() as token_logps_time_fn:
                if self.ref_logps_chunk_size is not None and prompt_completion_ids.shape[0] > self.ref_logps_chunk_size:
                    ref_chunks: list[jax.Array] = []
                    full_batch_size = int(prompt_completion_ids.shape[0])
                    for start in range(0, full_batch_size, self.ref_logps_chunk_size):
                        end = min(start + self.ref_logps_chunk_size, full_batch_size)
                        ref_chunks.append(
                            self.compute_refmodel_logps(
                                self.ref_state.graphstate,
                                self.ref_state.graphother,
                                prompt_completion_ids[start:end],
                                prompt_completion_mask[start:end],
                                slice_prompt_aligned_model_kwargs(
                                    normalized_repeated_model_kwargs,
                                    start,
                                    end,
                                    prompt_batch_size=full_batch_size,
                                ),
                            )
                        )
                    ref_per_token_logps = jnp.concatenate(ref_chunks, axis=0)
                else:
                    ref_per_token_logps = self.compute_refmodel_logps(
                        self.ref_state.graphstate,
                        self.ref_state.graphother,
                        prompt_completion_ids,
                        prompt_completion_mask,
                        normalized_repeated_model_kwargs,
                    )
            token_logps_time = token_logps_time_fn()

            raw_completions_text = self._coerce_generation_texts(
                results.raw_text,
                fallback=results.text,
            )
            clean_completions_text = self._coerce_generation_texts(
                results.text,
                fallback=raw_completions_text,
            )
            if not raw_completions_text or not clean_completions_text:
                host_completion_ids = np.asarray(jax.device_get(completion_ids), dtype=np.int64)
                host_completion_mask = np.asarray(jax.device_get(completion_mask), dtype=np.int32)
                if not raw_completions_text:
                    raw_completions_text = self._decode_prompt_batch(
                        self.processing_class,
                        host_completion_ids,
                        skip_special_tokens=False,
                        pad_token_id=self._pad_token_id,
                        pop_pad_tokens=True,
                        attention_mask=host_completion_mask,
                    )
                if not clean_completions_text:
                    clean_completions_text = self._decode_prompt_batch(
                        self.processing_class,
                        host_completion_ids,
                        skip_special_tokens=True,
                        pad_token_id=self._pad_token_id,
                        pop_pad_tokens=True,
                        attention_mask=host_completion_mask,
                    )
            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational

            if is_conversational:
                raw_completions = [[{"role": "assistant", "content": completion}] for completion in raw_completions_text]
                completions = [[{"role": "assistant", "content": completion}] for completion in clean_completions_text]
            else:
                raw_completions = raw_completions_text
                completions = clean_completions_text
            target_len = len(clean_completions_text) or len(raw_completions_text) or int(completion_ids.shape[0])
            reasoning_records = self._coerce_optional_generation_texts(
                results.reasoning,
                target_len=target_len,
            )
            tool_call_records = self._coerce_generation_metadata_list(
                results.tool_calls,
                target_len=target_len,
            )
            structured_completions = (
                self._build_structured_assistant_messages(
                    clean_completions_text,
                    tool_calls=tool_call_records,
                )
                if is_conversational
                else completions
            )

            rewards_per_func = jnp.full(
                (prompt_ids.shape[0] * generation_factor, len(self.reward_funcs)),
                jnp.nan,
                dtype="f4",
            )
            with capture_time() as rewarding_time_fn:
                for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes, strict=False)
                ):
                    if isinstance(reward_func, EasyDeLState):
                        if is_conversational:
                            messages = [
                                {"messages": p + c}
                                for p, c in zip(completion_prompts, structured_completions, strict=False)
                            ]
                            texts = [
                                apply_chat_template(
                                    x,
                                    reward_processing_class,
                                    tools=self._reward_chat_template_tools(),
                                )["text"]
                                for x in messages
                            ]
                        else:
                            texts = [p + c for p, c in zip(completion_prompts, completions, strict=False)]

                        rew = reward_func.apply_fn(
                            reward_func.graphdef,
                            reward_func.graphstate,
                            reward_func.graphother,
                            dict(
                                reward_processing_class(
                                    texts,
                                    return_tensors="np",
                                    padding="max_length",
                                    padding_side="right",
                                    add_special_tokens=False,
                                    truncation=True,
                                    return_attention_mask=True,
                                    max_length=self.arguments.max_length,
                                )
                            ),
                        ).logits[:, 0]
                    else:
                        in_prompts = completion_prompts
                        reward_call_kwargs = self._build_reward_call_kwargs(
                            reward_func,
                            prompts=in_prompts,
                            completions=completions,
                            raw_completions=raw_completions,
                            completion_texts=clean_completions_text,
                            raw_text=raw_completions_text,
                            reasoning=reasoning_records,
                            tool_calls=tool_call_records,
                            max_length=self.arguments.max_length,
                            batch=reward_batch,
                        )
                        output_reward_func = reward_func(**reward_call_kwargs)
                        rew = jnp.array(
                            [val if val is not None else jnp.nan for val in output_reward_func],
                            dtype="f4",
                        )
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
            rewarding_time = rewarding_time_fn()

            # Compute rewards before filtering (for metrics)
            rewards = jnp.nansum(rewards_per_func * self.reward_weights[None, :], axis=1)
            log_completion_length = jnp.sum(completion_mask, -1)

            num_remains = self.arguments.num_remains_in_group
            if num_remains is not None and num_remains < generation_factor:
                with capture_time() as filter_time_fn:
                    (
                        prompt_ids,
                        prompt_mask,
                        completion_ids,
                        completion_mask,
                        ref_per_token_logps,
                        rewards,
                        rewards_per_func,
                        completion_prompts,
                        generation_factor,
                    ) = self._filter_completions(
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        completion_ids=completion_ids,
                        completion_mask=completion_mask,
                        ref_per_token_logps=ref_per_token_logps,
                        rewards=rewards,
                        rewards_per_func=rewards_per_func,
                        completion_prompts=completion_prompts,
                        num_generations=generation_factor,
                        num_remains=num_remains,
                    )
                filter_time = filter_time_fn()
            else:
                filter_time = 0.0

            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)
            completion_ids = self._all_gather(completion_ids)
            completion_mask = self._all_gather(completion_mask)
            ref_per_token_logps = self._all_gather(ref_per_token_logps)
            rewards_per_func = self._all_gather(rewards_per_func)
            scoring_prompt_model_kwargs = jax.tree_util.tree_map(
                lambda x: self._all_gather(x) if isinstance(x, jax.Array) else x,
                scoring_prompt_model_kwargs,
            )

            with capture_time() as grouped_comp_time_fn:
                # Recompute generation_factor after filtering and gathering
                generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
                generation_factor = max(generation_factor, 1)
                rewards = jnp.nansum(rewards_per_func * self.reward_weights[None, :], axis=1)
                mean_grouped_rewards = jnp.nanmean(rewards.reshape(-1, generation_factor), axis=-1)
                advantages = rewards - mean_grouped_rewards.repeat(generation_factor, axis=0)

                if self.scale_rewards in ("group", "none"):
                    std_rewards = jnp.nanstd(rewards.reshape(-1, generation_factor), axis=-1)
                    std_rewards = std_rewards.repeat(generation_factor, axis=0)
                elif self.scale_rewards == "batch":
                    std_rewards = jnp.nanstd(rewards)
                    std_rewards = jnp.broadcast_to(std_rewards, advantages.shape)
                else:
                    raise ValueError(
                        f"Invalid value for scale_rewards: {self.scale_rewards}. Must be 'batch', 'group', or 'none'."
                    )
                is_std_zero = jnp.isclose(std_rewards, 0.0)
                if self.scale_rewards != "none":
                    advantages = advantages / (std_rewards + 1e-4)
                advantages = jnp.nan_to_num(advantages)
            grouped_comp_time = grouped_comp_time_fn()

        preprocessing_time = preprocessing_time_fn()
        completion_length = jnp.sum(completion_mask, -1)
        metrics_dict: dict[str, float | int | str] = {
            "reward_mean": float(jnp.nanmean(rewards, -1)),
            "reward_std": float(jnp.nanmean(std_rewards)),
            "completion_length": float(jnp.mean(completion_length)),
            "grouped_comp_time": grouped_comp_time,
            "rewarding_time": rewarding_time,
            "token_logps_time": token_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
            "frac_reward_zero_std": float(jnp.mean(is_std_zero.astype(jnp.float32))),
            "filter_time": filter_time,
        }
        for i, reward_func_name in enumerate(self.reward_func_names):
            metrics_dict[reward_func_name] = float(jnp.nanmean(rewards_per_func[:, i]))
        self._log_training_generations_to_wandb(
            state=state,
            prompts=completion_prompts,
            completions=clean_completions_text,
            completion_lengths=log_completion_length,
            generation_time=generation_time,
            reasoning=reasoning_records,
            tool_calls=tool_call_records,
            source="policy",
        )

        return (
            {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "ref_per_token_logps": ref_per_token_logps,
                "advantages": advantages,
                "num_items_in_batch": jnp.sum(completion_mask),
                **scoring_prompt_model_kwargs,
            },
            metrics_dict,
        )
