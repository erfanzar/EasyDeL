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
from ..training_configurations import MetricsType
from ..training_utils import resolve_straight_through_emulator
from ._fn import get_per_token_logps, grpo_step
from .grpo_config import GRPOConfig

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)
RewardFunc = tp.Union[EasyDeLBaseModule, EasyDeLState, tp.Callable[[list, list], list[float]]]  # noqa


def _fileaf(x):
    return isinstance(x, jax.Array)


def delete_tree(pytree):
    return jax.tree_util.tree_map(
        lambda x: x.delete() if isinstance(x, jax.Array) else None,
        pytree,
        is_leaf=_fileaf,
    )


@Registry.register("trainer", "grpo")
class GRPOTrainer(Trainer):
    """Group Relative Policy Optimization trainer for RLHF.

    GRPO is a reinforcement learning method that optimizes policies by comparing
    responses within groups, providing more stable training than standard PPO.
    It uses relative scoring within batches to reduce variance and improve
    convergence in preference-based learning tasks.

    Key features:
    - Group-based advantage normalization
    - Stable policy updates with KL regularization
    - Support for multiple reward models
    - Efficient generation and scoring pipeline

    Attributes:
        arguments: GRPOConfig instance with training hyperparameters
        ref_state: Reference model state for KL divergence computation
        processing_class: Tokenizer or processor for text encoding
        reward_processing_classes: Optional separate processors for reward models
        generation_config: Configuration for response generation
        data_tokenize_fn: Function to tokenize dataset samples

    Example:
        >>> config = GRPOConfig(
        ...     per_device_train_batch_size=4,
        ...     grpo_n_samples=4,
        ...     grpo_beta=0.1,
        ...     learning_rate=1e-6
        ... )
        >>> trainer = GRPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=reward_model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    arguments: GRPOConfig  # type hinting

    def __init__(
        self,
        arguments: GRPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        assert arguments is not None, (
            "You Have to pass `arguments` that will be used for training, but you have passed `arguments=None`"
        )
        assert isinstance(arguments, GRPOConfig), f"arguments type must be `GRPOConfig` but got {type(arguments)}"
        assert processing_class is not None, "processing_class must be specified to tokenize a DPO dataset."

        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.processing_class = processing_class
        self.loss_type = arguments.loss_type.lower() if isinstance(arguments.loss_type, str) else arguments.loss_type
        self.epsilon = arguments.epsilon
        self.epsilon_high = arguments.epsilon_high
        self.delta = arguments.delta
        self.importance_sampling_level = arguments.importance_sampling_level
        if isinstance(self.importance_sampling_level, str):
            self.importance_sampling_level = self.importance_sampling_level.lower()
        self.scale_rewards = arguments.scale_rewards
        if isinstance(self.scale_rewards, str):
            self.scale_rewards = self.scale_rewards.lower()
        self.top_entropy_quantile = arguments.top_entropy_quantile

        if not isinstance(model, EasyDeLState):
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
                    reward_model_name = reward_func.model.config._name_or_path
                    try:
                        reward_processing_class = AutoTokenizer.from_pretrained(reward_model_name)
                    except ValueError as exc:
                        if "tiktoken" in str(exc).lower():
                            reward_processing_class = AutoTokenizer.from_pretrained(reward_model_name, use_fast=False)
                        else:
                            raise
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
        self.arguments = arguments
        self.processing_class = processing_class
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

        # Check if datasets are conversational before passing to BaseTrainer
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
        """Get GRPO preprocessing transform for ShardedDataSource."""

        if self._is_pretokenized():
            return None
        return GRPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            tools=getattr(self.arguments, "tools", None),
            skip_apply_chat_template=self.arguments.skip_apply_chat_template,
        )

    def _is_pretokenized(self) -> bool:
        """Check if dataset already has tokenized fields."""
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
        """Create data collator for Grain data loading."""
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
        """Create data collator for TFDS data loading."""
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    @property
    def step_sharding(self):
        return NamedSharding(
            mesh=self.model.mesh,
            spec=self.arguments.step_partition_spec,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method sets up the necessary functions for training and evaluation, including:
            - Initialization of the model state.
            - Sharding of the model parameters and optimizer state.
            - JIT-compilation of the training and evaluation step functions.

        Returns:
            TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
        """
        mesh = self.model.mesh

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_block=self.arguments.quantization_block,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        self._train_shared_fn_static_args = (
            self.num_generations,
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
            straight_through_emulator,
        )

        static_argnames = tuple(range(2, 16))

        sharded_training_step_function = ejit(
            grpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        self._eval_shared_fn_static_args = (
            self.num_generations,
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
            straight_through_emulator,
        )

        sharded_evaluation_step_function = ejit(
            grpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        def _compute_refmodel_logps(graphtree, graphother, ids, mask, graphdef):
            apply = flax.nnx.merge(graphdef, graphtree, graphother)
            with apply.mesh:
                ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
                mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
                return get_per_token_logps(apply, ids, mask, self.arguments.max_prompt_length)

        self.compute_refmodel_logps = ejit(
            partial(_compute_refmodel_logps, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef"),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=empty_sharding,
        )

        sharded_training_step_function.static_argnums_ = static_argnames
        sharded_evaluation_step_function.static_argnums_ = static_argnames

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        # Purify batch first to handle list of dicts (uncollated batch)
        batch = self._purify_batch(batch)
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=state,
                    apply_chat_template=False,  # GRPO doesn't apply chat template to prompts
                    shard_inputs=False,  # Already sharded
                    all_gather=False,  # We'll handle gathering ourselves
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
            # Derive how many completions we have per prompt instead of trusting config-only value.
            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            ridmask = prompt_mask.repeat(generation_factor, 0)

            with capture_time() as token_logps_time_fn:
                ref_per_token_logps = self.compute_refmodel_logps(
                    self.ref_state.graphstate,
                    self.ref_state.graphother,
                    prompt_completion_ids,
                    jnp.concatenate([ridmask, completion_mask], -1),
                )
            token_logps_time = token_logps_time_fn()

            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational

            if is_conversational:
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                completions = completions_text

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
                        in_prompts = completion_prompts
                        output_reward_func = reward_func(
                            prompts=in_prompts,
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
            log_completion_ids = completion_ids
            log_completion_length = jnp.sum(completion_mask, -1)

            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)
            completion_ids = self._all_gather(completion_ids)
            completion_mask = self._all_gather(completion_mask)
            ref_per_token_logps = self._all_gather(ref_per_token_logps)
            rewards_per_func = self._all_gather(rewards_per_func)

            with capture_time() as grouped_comp_time_fn:
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
        metrics_dict = {
            "reward_mean": jnp.nanmean(rewards, -1),
            "reward_std": jnp.nanmean(std_rewards),
            "completion_length": jnp.mean(completion_length),
            "grouped_comp_time": grouped_comp_time,
            "rewarding_time": rewarding_time,
            "token_logps_time": token_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
            "frac_reward_zero_std": jnp.mean(is_std_zero.astype(jnp.float32)),
        }
        for i, reward_func_name in enumerate(self.reward_func_names):
            metrics_dict[reward_func_name] = jnp.nanmean(rewards_per_func[:, i])
        if self.log_table is not None:
            cur_step = jax.device_get(state.step)
            decoded_prompt = completion_prompts
            decoded_text = self._decode_prompt_batch(
                self.processing_class,
                jax.device_get(log_completion_ids),
                False,
                self._pad_token_id,
                True,
            )
            for decoded, prompt, length in zip(decoded_text, decoded_prompt, log_completion_length, strict=False):
                prompt_repr = prompt if isinstance(prompt, str) else str(prompt)
                self.log_table.add_data(decoded, prompt_repr, generation_time, float(jax.device_get(length)), cur_step)
            wandb.log({"generations": self.log_table}, step=cur_step)

        # i don't care who you are and what you do.
        # ill find you and ill gather u...
        return (
            {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "ref_per_token_logps": ref_per_token_logps,
                "advantages": advantages,
                "num_items_in_batch": jnp.sum(completion_mask),
            },
            metrics_dict,
        )

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """hook process to call in start of the step."""

        if (
            self.arguments.sync_ref_model
            and self.ref_state is not None
            and (step % self.arguments.ref_model_sync_steps == 0)
        ):
            alpha = self.arguments.ref_model_mixup_alpha
            new_graphstate = jax.tree_util.tree_map(
                lambda new, old: alpha * new + (1 - alpha) * old,
                deepcopy_model(state.graphstate),
                self.ref_state.graphstate,
            )
            self.ref_state = self.ref_state.replace(graphstate=new_graphstate)
        return state, metrics
