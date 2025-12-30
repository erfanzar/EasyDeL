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

"""Proximal Policy Optimization (PPO) trainer for RLHF.

This module implements the PPOTrainer class for training language models with
Proximal Policy Optimization. PPO is a policy gradient method that uses clipped
surrogate objectives to ensure stable policy updates during reinforcement learning
from human feedback (RLHF).

The trainer supports:
- Online generation of completions
- Multiple reward functions with configurable weights
- Value head for advantage estimation (GAE)
- KL penalty against a frozen reference model
- Clipped policy and value function objectives
"""

from __future__ import annotations

import typing as tp
from functools import partial

import flax
import flax.nnx
import jax
from eformer.escale import with_sharding_constraint
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from transformers import AutoTokenizer

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time, get_logger
from easydel.utils.traversals import deepcopy_model

from ..prompt_transforms import GRPOPreprocessTransform, is_conversational
from ..prompt_utils import apply_chat_template
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import resolve_straight_through_emulator
from ._fn import ppo_step
from .modeling_value_head import CausalLMWithValueHead
from .ppo_config import PPOConfig

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)
RewardFunc = tp.Union[EasyDeLBaseModule, EasyDeLState, tp.Callable[[list, list], list[float]]]  # noqa


@Registry.register("trainer", "ppo")
class PPOTrainer(Trainer):
    """Proximal Policy Optimization trainer for RLHF.

    PPO is a policy gradient method that uses clipped surrogate objectives to
    ensure stable policy updates. This trainer implements online PPO where
    completions are generated on-the-fly and scored by reward functions.

    Key features:
    - Online generation with configurable sampling parameters
    - Value head for advantage estimation via GAE
    - KL penalty against a frozen reference model
    - Support for multiple weighted reward functions
    - Clipped policy and value function objectives

    The training loop:
    1. Sample prompts from the dataset
    2. Generate completions using the current policy
    3. Score completions with reward function(s)
    4. Compute advantages using GAE with the value head
    5. Update policy with clipped PPO objective

    Attributes:
        arguments: PPOConfig with training hyperparameters.
        ref_state: Frozen reference model for KL computation.
        reward_funcs: List of reward functions/models.
        reward_weights: Weights for combining multiple rewards.
        processing_class: Tokenizer for text encoding.
        num_generations: Number of completions per prompt.

    Example:
        >>> config = PPOConfig(
        ...     per_device_train_batch_size=4,
        ...     num_return_sequences=4,
        ...     kl_coef=0.05,
        ...     cliprange=0.2,
        ...     learning_rate=1e-6
        ... )
        >>> trainer = PPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=reward_model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    arguments: PPOConfig

    def __init__(
        self,
        arguments: PPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        """Initialize the PPO trainer.

        Args:
            arguments: PPOConfig instance with training hyperparameters.
            model: The policy model to train. Will be wrapped with a value head
                if not already present.
            reward_funcs: Reward function(s) for scoring completions. Can be:
                - A callable taking (prompts, completions) and returning scores
                - An EasyDeLBaseModule/EasyDeLState reward model
                - A list of multiple reward functions
            train_dataset: Training dataset with prompt examples.
            eval_dataset: Optional evaluation dataset(s).
            processing_class: Tokenizer for encoding prompts and completions.
            reward_processing_classes: Optional separate tokenizers for reward models.
            data_tokenize_fn: Optional custom tokenization function.

        Raises:
            AssertionError: If arguments is None or not a PPOConfig.
            ValueError: If model is None or reward weights don't match reward funcs.
        """
        assert arguments is not None, "PPOTrainer requires `arguments`."
        assert isinstance(arguments, PPOConfig), f"arguments type must be `PPOConfig` but got {type(arguments)}"
        assert processing_class is not None, "processing_class must be specified to tokenize PPO prompts."

        self.arguments = arguments
        self.processing_class = processing_class

        if model is None:
            raise ValueError("`model` must be provided for PPO training.")

        # Ensure we have a value head attached.
        if isinstance(model, EasyDeLState):
            module = model.model
            if not hasattr(module, "value_head"):
                model = CausalLMWithValueHead(module).to_state()
        else:
            if not hasattr(model, "value_head"):
                model = CausalLMWithValueHead(model)
            model = model.to_state()

        self.ref_state = deepcopy_model(model=model)

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.model.config._name_or_path,
                padding_side="left",
            )

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=model.model.mesh)

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=False)
        ):
            if isinstance(reward_func, EasyDeLBaseModule | EasyDeLState):
                if isinstance(reward_func, EasyDeLBaseModule):
                    reward_func = reward_func.to_state()
                    sharding = reward_func.shardings

                    @ejit(
                        static_argnums=(0,),
                        in_shardings=(sharding.graphstate, sharding.graphother, empty_sharding),
                        out_shardings=empty_sharding,
                    )
                    def apply_fn(gd, gs, gt, batch):
                        batch = with_sharding_constraint(arr=batch, sharding=self.arguments.step_partition_spec)
                        return nn.merge(gd, gs, gt)(**batch)

                    reward_func = reward_func.replace(apply_fn=apply_fn)

                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.model.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token

                reward_func.model.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
                reward_funcs[i] = reward_func

        if arguments.reward_weights is not None and len(arguments.reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Expected {len(reward_funcs)} reward weights, but got {len(arguments.reward_weights)} instead."
            )

        self.reward_weights = jnp.asarray(
            arguments.reward_weights if arguments.reward_weights is not None else [1.0] * len(reward_funcs),
            dtype="f4",
        )
        self.reward_func_names = [getattr(func, "__name__", None) or func.__class__.__name__ for func in reward_funcs]

        self.num_generations = arguments.num_generations
        self.reward_processing_classes = reward_processing_classes
        self.reward_funcs = reward_funcs

        if getattr(self.arguments, "generation_num_return_sequences", None) is None:
            self.arguments.generation_num_return_sequences = self.num_generations
        if getattr(self.arguments, "generation_top_p", None) is None:
            self.arguments.generation_top_p = self.arguments.top_p
        if getattr(self.arguments, "generation_top_k", None) is None:
            self.arguments.generation_top_k = self.arguments.top_k
        if getattr(self.arguments, "generation_temperature", None) is None:
            self.arguments.generation_temperature = self.arguments.temperature
        if getattr(self.arguments, "generation_extra_kwargs", None) is None:
            self.arguments.generation_extra_kwargs = {}
        if self.arguments.generation_kwargs is not None:
            self.arguments.generation_extra_kwargs.update(self.arguments.generation_kwargs)
        for key, value in (
            ("min_p", self.arguments.min_p),
            ("repetition_penalty", self.arguments.repetition_penalty),
        ):
            if value is not None and key not in self.arguments.generation_extra_kwargs:
                self.arguments.generation_extra_kwargs[key] = value

        def _peek_first_example(dataset):
            if dataset is None:
                return None
            if isinstance(dataset, dict):
                for item in dataset.values():
                    return _peek_first_example(item)
                return None
            try:
                return dataset[0]
            except Exception:
                pass
            try:
                return next(iter(dataset))
            except Exception:
                pass
            try:
                shard_names = getattr(dataset, "shard_names", None)
                open_shard = getattr(dataset, "open_shard", None)
                if shard_names and open_shard:
                    return next(iter(open_shard(shard_names[0])))
            except Exception:
                pass
            return None

        self.train_is_conversational = False
        self.eval_is_conversational = False
        train_sample = _peek_first_example(train_dataset)
        if train_sample is not None:
            self.train_is_conversational = is_conversational(train_sample)
        eval_sample = _peek_first_example(eval_dataset)
        if eval_sample is not None:
            self.eval_is_conversational = is_conversational(eval_sample)

        self.data_tokenize_fn = data_tokenize_fn
        log_table = None
        if self.arguments.use_wandb and self.arguments.can_log_metrics and wandb is not None:
            log_table = wandb.Table(columns=["generated_result", "input_prompt", "took", "length", "step"])
        self.log_table = log_table

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> GRPOPreprocessTransform | None:
        """Get the preprocessing transform for ShardedDataSource.

        Returns:
            GRPOPreprocessTransform if dataset needs tokenization, None otherwise.
        """
        if self._is_pretokenized():
            return None
        return GRPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            tools=getattr(self.arguments, "tools", None),
            skip_apply_chat_template=self.arguments.skip_apply_chat_template,
        )

    def _is_pretokenized(self) -> bool:
        """Check if the dataset already has tokenized fields.

        Returns:
            True if dataset contains 'input_ids', False otherwise.
        """
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create a data collator for Grain data loading.

        Args:
            max_sequence_length: Maximum sequence length (unused, kept for API).
            truncation_mode: How to truncate sequences (unused, kept for API).

        Returns:
            GRPODataCollatorGrain instance for batching prompt data.
        """
        from ..utils import GRPODataCollatorGrain

        return GRPODataCollatorGrain(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create a data collator for TFDS data loading.

        Args:
            max_sequence_length: Maximum sequence length (unused, kept for API).
            truncation_mode: How to truncate sequences (unused, kept for API).

        Returns:
            GRPODataCollatorTFDS instance for batching prompt data.
        """
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure and JIT-compile training and evaluation step functions.

        Sets up the sharded training and evaluation functions with proper
        JIT compilation and sharding specifications for distributed training.
        Also configures helper functions for computing reference model log
        probabilities and rollout statistics.

        Returns:
            TrainerConfigureFunctionOutput containing:
                - sharded_training_step_function: JIT-compiled training step
                - sharded_evaluation_step_function: JIT-compiled eval step
                - mesh: Device mesh for sharding
                - checkpoint_manager: For saving checkpoints
        """
        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_block=self.arguments.quantization_block,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        prompt_length = int(self.arguments.max_prompt_length)

        self._train_shared_fn_static_args = (
            prompt_length,
            float(self.arguments.cliprange),
            float(self.arguments.vf_coef),
            float(self.arguments.cliprange_value),
            float(self.arguments.entropy_coef),
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
            straight_through_emulator,
        )
        static_argnums = tuple(range(2, 13))
        sharded_training_step_function = ejit(
            ppo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
        )

        self._eval_shared_fn_static_args = (
            prompt_length,
            float(self.arguments.cliprange),
            float(self.arguments.vf_coef),
            float(self.arguments.cliprange_value),
            float(self.arguments.entropy_coef),
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
            straight_through_emulator,
        )
        sharded_evaluation_step_function = ejit(
            ppo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
        )

        def _compute_refmodel_logps(graphtree, graphother, ids, mask, graphdef):
            apply = flax.nnx.merge(graphdef, graphtree, graphother)
            with apply.mesh:
                ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
                mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
                # Reuse GRPO logps utility implementation via PPO step helpers (fast path).
                outputs = apply(input_ids=ids, attention_mask=mask)
                logits = outputs.logits[:, prompt_length - 1 :]
                logits = logits[:, :-1, :]
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                target_ids = ids[:, prompt_length:]
                token_log_probs = jnp.take_along_axis(log_probs, jnp.expand_dims(target_ids, axis=-1), axis=-1)
                return jnp.squeeze(token_log_probs, axis=-1)

        self.compute_refmodel_logps = ejit(
            partial(_compute_refmodel_logps, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=empty_sharding,
        )

        def _compute_rollout_logps_values(graphtree, graphother, ids, mask, graphdef):
            apply = flax.nnx.merge(graphdef, graphtree, graphother)
            with apply.mesh:
                ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
                mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
                outputs = apply(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                logits = outputs.logits[:, prompt_length - 1 :]
                logits = logits[:, :-1, :]
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                target_ids = ids[:, prompt_length:]
                token_log_probs = jnp.take_along_axis(log_probs, jnp.expand_dims(target_ids, axis=-1), axis=-1)
                token_log_probs = jnp.squeeze(token_log_probs, axis=-1)

                hidden_states = getattr(outputs, "last_hidden_state", None)
                if hidden_states is None:
                    hidden_states = getattr(outputs, "hidden_states", None)
                    if hidden_states is None:
                        raise ValueError("Model outputs do not provide hidden states; cannot compute value outputs.")
                    hidden_states = hidden_states[-1]

                values_full = apply.value_head(hidden_states).squeeze(-1)
                values = values_full[:, prompt_length - 1 : -1]
                return token_log_probs, values

        self.compute_rollout_logps_values = ejit(
            partial(_compute_rollout_logps_values, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=(empty_sharding, empty_sharding),
        )

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _masked_whiten(self, x: jax.Array, mask: jax.Array, *, shift_mean: bool) -> jax.Array:
        """Normalize array values with optional mean shifting.

        Args:
            x: Input array to normalize.
            mask: Binary mask for valid elements.
            shift_mean: Whether to subtract the mean (True for advantages).

        Returns:
            Normalized array with unit variance (and zero mean if shift_mean).
        """
        mask = mask.astype(x.dtype)
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        mean = jnp.sum(x * mask) / denom
        var = jnp.sum(jnp.square(x - mean) * mask) / denom
        std = jnp.sqrt(var + 1e-8)
        if shift_mean:
            x = x - mean
        return x / std

    def _compute_gae(
        self,
        rewards: jax.Array,
        values: jax.Array,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute Generalized Advantage Estimation (GAE).

        Computes advantages and returns using the GAE algorithm, which provides
        a balance between bias and variance in advantage estimation.

        Args:
            rewards: Per-token rewards including KL penalty and final score.
            values: Value head predictions for each position.
            mask: Binary mask for valid completion tokens.

        Returns:
            Tuple of (advantages, returns):
                - advantages: GAE-computed advantages for policy gradient.
                - returns: Target values for value function training.
        """
        rewards = rewards.astype(jnp.float32)
        values = values.astype(jnp.float32)
        mask = mask.astype(jnp.float32)

        batch_size, _gen_len = rewards.shape
        values_next = jnp.concatenate([values[:, 1:], jnp.zeros((batch_size, 1), dtype=values.dtype)], axis=1)
        mask_next = jnp.concatenate([mask[:, 1:], jnp.zeros((batch_size, 1), dtype=mask.dtype)], axis=1)

        gamma = float(self.arguments.gamma)
        lam = float(self.arguments.lam)

        def scan_fn(adv_next, inputs):
            r_t, v_t, v_next_t, m_t, m_next_t = inputs
            delta = r_t + gamma * v_next_t * m_next_t - v_t
            adv_t = delta + gamma * lam * adv_next * m_next_t
            adv_t = adv_t * m_t
            return adv_t, adv_t

        inputs = (
            rewards[:, ::-1].T,
            values[:, ::-1].T,
            values_next[:, ::-1].T,
            mask[:, ::-1].T,
            mask_next[:, ::-1].T,
        )
        _, adv_rev = jax.lax.scan(scan_fn, jnp.zeros((batch_size,), dtype=values.dtype), inputs)
        advantages = adv_rev.T[:, ::-1]
        returns = advantages + values
        return advantages, returns

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Preprocess a batch for PPO training.

        Performs the full PPO rollout preprocessing:
        1. Generate completions from prompts
        2. Compute log probabilities and values from policy
        3. Compute reference model log probabilities
        4. Score completions with reward functions
        5. Compute per-token rewards with KL penalty
        6. Compute GAE advantages and returns

        Args:
            state: Current model state for generation.
            batch: Input batch with prompt input_ids and attention_mask.
            is_train: Whether this is for training (affects conversational format).

        Returns:
            Tuple of (processed_batch, metrics):
                - processed_batch: Dictionary with all tensors needed for PPO step.
                - metrics: Dictionary with timing and reward statistics.
        """
        batch = self._purify_batch(batch)
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
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

            completion_mask = self._make_attn_mask(completion_ids)
            if self.arguments.mask_truncated_completions:
                eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
                has_eos = jnp.any(jnp.isin(completion_ids, eos_tokens), axis=1)
                completion_mask = completion_mask * has_eos[:, None].astype(completion_mask.dtype)

            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            prompt_mask_rep = prompt_mask.repeat(generation_factor, 0)
            attention_mask = jnp.concatenate([prompt_mask_rep, completion_mask], axis=1)
            input_ids = sequences

            with capture_time() as ref_logps_time_fn:
                ref_per_token_logps = self.compute_refmodel_logps(
                    self.ref_state.graphstate,
                    self.ref_state.graphother,
                    input_ids,
                    attention_mask,
                )
            ref_logps_time = ref_logps_time_fn()

            with capture_time() as rollout_stats_time_fn:
                old_logps, old_values = self.compute_rollout_logps_values(
                    state.graphstate,
                    state.graphother,
                    input_ids,
                    attention_mask,
                )
            rollout_stats_time = rollout_stats_time_fn()

            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            is_conv = self.train_is_conversational if is_train else self.eval_is_conversational
            if completion_prompts:
                first_prompt = completion_prompts[0]
                if not isinstance(first_prompt, list):
                    is_conv = False
            else:
                is_conv = False
            if is_conv:
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                completions = completions_text

            rewards_per_func = jnp.full((completion_ids.shape[0], len(self.reward_funcs)), jnp.nan, dtype="f4")
            with capture_time() as rewarding_time_fn:
                for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes, strict=False)
                ):
                    if isinstance(reward_func, EasyDeLState):
                        if is_conv:
                            messages = [
                                {"messages": p + c} for p, c in zip(completion_prompts, completions, strict=False)
                            ]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(completion_prompts, completions, strict=False)]

                        rew = reward_func.apply_fn(
                            reward_func.graphdef,
                            reward_func.graphstate,
                            reward_func.graphother,
                            dict(
                                reward_processing_class(
                                    texts,
                                    return_tensors="jax",
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
                        output_reward_func = reward_func(
                            prompts=completion_prompts,
                            completions=completions,
                            max_length=self.arguments.max_length,
                            batch=batch,
                        )
                        rew = jnp.array(
                            [val if val is not None else jnp.nan for val in output_reward_func],
                            dtype="f4",
                        )
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
            rewarding_time = rewarding_time_fn()

            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)
            completion_ids = self._all_gather(completion_ids)
            completion_mask = self._all_gather(completion_mask)
            input_ids = self._all_gather(input_ids)
            attention_mask = self._all_gather(attention_mask)
            old_logps = self._all_gather(old_logps)
            old_values = self._all_gather(old_values)
            ref_per_token_logps = self._all_gather(ref_per_token_logps)
            rewards_per_func = self._all_gather(rewards_per_func)

            scores = jnp.nansum(rewards_per_func * self.reward_weights[None, :], axis=1)
            if self.arguments.missing_eos_penalty is not None:
                eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
                has_eos = jnp.any(jnp.isin(completion_ids, eos_tokens), axis=1)
                scores = scores - (~has_eos).astype(scores.dtype) * float(self.arguments.missing_eos_penalty)

            logr = ref_per_token_logps - old_logps
            if self.arguments.kl_estimator == "k1":
                kl = -logr
            else:
                kl = jnp.exp(logr) - 1.0 - logr
            non_score_reward = -float(self.arguments.kl_coef) * kl
            rewards = non_score_reward

            lengths = jnp.sum(completion_mask, axis=1).astype(jnp.int32)
            last_idx = jnp.maximum(lengths - 1, 0)
            batch_idx = jnp.arange(rewards.shape[0])
            rewards = rewards.at[batch_idx, last_idx].add(scores.astype(rewards.dtype))
            rewards = rewards * completion_mask

            if self.arguments.whiten_rewards:
                rewards = self._masked_whiten(rewards, completion_mask, shift_mean=False)
                rewards = rewards * completion_mask

            advantages, returns = self._compute_gae(rewards, old_values, completion_mask)
            if self.arguments.whiten_advantages:
                advantages = self._masked_whiten(advantages, completion_mask, shift_mean=True)
                advantages = advantages * completion_mask

        preprocessing_time = preprocessing_time_fn()

        token_count = jnp.maximum(jnp.sum(completion_mask), 1.0)
        metrics_dict = {
            "score_mean": jnp.nanmean(scores),
            "reward_mean": jnp.sum(rewards) / token_count,
            "mean_kl": jnp.sum(kl * completion_mask) / token_count,
            "completion_length": jnp.mean(jnp.sum(completion_mask, axis=1)),
            "rewarding_time": rewarding_time,
            "rollout_stats_time": rollout_stats_time,
            "ref_logps_time": ref_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
        }
        for i, reward_func_name in enumerate(self.reward_func_names):
            metrics_dict[reward_func_name] = jnp.nanmean(rewards_per_func[:, i])

        return (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": completion_mask,
                "old_logps": old_logps,
                "old_values": old_values,
                "advantages": advantages,
                "returns": returns,
            },
            metrics_dict,
        )
