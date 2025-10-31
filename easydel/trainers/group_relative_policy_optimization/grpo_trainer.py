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
from functools import cached_property, partial

import flax
import flax.nnx
import jax
from eformer import common_types
from eformer.escale import with_sharding_constraint
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from transformers import AutoTokenizer, GenerationConfig, ProcessorMixin

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time, get_logger
from easydel.utils.traversals import deepcopy_model

from ..prompt_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, maybe_extract_prompt
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_configurations import MetricsType
from ..training_utils import compute_group_advantages, compute_length_reward, update_ema
from ._fn import get_per_token_logps, grpo_step
from .grpo_config import GRPOConfig

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

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
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        assert (
            arguments is not None
        ), "You Have to pass `arguments` that will be used for training, but you have passed `arguments=None`"
        assert isinstance(arguments, GRPOConfig), f"arguments type must be `GRPOConfig` but got {type(arguments)}"
        assert processing_class is not None, "processing_class must be specified to tokenize a DPO dataset."

        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.processing_class = processing_class

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
                        in_shardings=(
                            sharding.graphstate,
                            sharding.graphother,
                            empty_sharding,
                        ),
                        out_shardings=empty_sharding,
                    )
                    def apply_fn(gd, gs, gt, batch):
                        batch = with_sharding_constraint(
                            arr=batch,
                            sharding=self.arguments.step_partition_spec,
                        )
                        return nn.merge(gd, gs, gt)(**batch)

                    reward_func = reward_func.replace(apply_fn=apply_fn)

                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.model.config._name_or_path)

                # Type narrowing: at this point reward_processing_class is guaranteed to be non-None
                assert reward_processing_class is not None

                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token

                reward_func.model.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
                reward_funcs[i] = reward_func

        self.num_generations = arguments.num_return_sequences
        self.reward_processing_classes = reward_processing_classes
        self.reward_funcs = reward_funcs
        self.arguments = arguments
        self.processing_class = processing_class
        self.train_is_conversational = False
        self.eval_is_conversational = False
        self.data_tokenize_fn = data_tokenize_fn
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                dataset=train_dataset,
                processing_class=processing_class,
                arguments=arguments,
                dataset_name="train",
            )
        if eval_dataset is not None:
            eval_dataset = self._prepare_dataset(
                dataset=eval_dataset,
                processing_class=processing_class,
                arguments=arguments,
                dataset_name="eval",
            )
        log_table = None
        if self.arguments.use_wandb and self.arguments.can_log_metrics and wandb is not None:
            log_table = wandb.Table(columns=["generations", "took", "length", "step"])
        self.log_table = log_table

        self._kl_ema: float | None = None
        self._entropy_ema: float | None = None
        self._last_reference_reset: int = 0


        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
        )

    @cached_property
    def pad_token_id(self):
        if isinstance(self.processing_class, ProcessorMixin):
            pad_token_id = self.processing_class.tokenizer.pad_token_id
        else:
            pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is not None:
            return pad_token_id
        else:
            return self.eos_token_id[0]

    @cached_property
    def eos_token_id(self) -> list[int]:
        eos_ids = []
        if isinstance(self.processing_class, ProcessorMixin):
            proc_eos_token_id = self.processing_class.tokenizer.eos_token_id
        else:
            proc_eos_token_id = self.processing_class.eos_token_id
        if isinstance(proc_eos_token_id, int):
            proc_eos_token_id = [proc_eos_token_id]
        eos_ids = eos_ids + proc_eos_token_id
        if hasattr(self.model, "generation_config"):
            conf_eos = self.model.generation_config.eos_token_id
            if isinstance(conf_eos, int):
                conf_eos = [conf_eos]
            eos_ids = eos_ids + conf_eos
        return list(set(eos_ids))

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: ProcessingClassType,
        arguments: GRPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        map_kwargs = {"writer_batch_size": 10}
        from datasets import Dataset

        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = arguments.dataset_num_proc

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
        if dataset_name == "train":
            self.train_is_conversational = is_conversational(dataset[0])
        else:
            self.eval_is_conversational = is_conversational(dataset[0])

        dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
        if not self.arguments.skip_apply_chat_template:
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": arguments.tools},
                **map_kwargs,
            )

        def _tokenize(example):
            return processing_class(
                example["prompt"],
                return_tensors="np",
                padding="max_length",
                padding_side="left",
                tools=example.get("tools", None),
                max_length=arguments.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                return_attention_mask=True,
            )

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"tokenizing {dataset_name} dataset"
        if self.data_tokenize_fn is not None:
            dataset = dataset.map(
                self.data_tokenize_fn,
                batched=True,
                fn_kwargs={"tokenizer": processing_class, "tools": arguments.tools},
                **map_kwargs,
            )
        else:
            dataset = dataset.map(_tokenize, batched=True, **map_kwargs)
        return dataset

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

        @ejit(
            in_shardings=(self.state_shardings, empty_sharding, empty_sharding),
            out_shardings=(empty_sharding, empty_sharding, empty_sharding),
        )
        def generate(state: EasyDeLState, input_ids, attention_mask):
            module = state.model

            with module.mesh:
                input_ids = module.config.partition_manager.shard(
                    input_ids,
                    axes=[common_types.BATCH, common_types.SEQUENCE_PARALLEL],
                    mode=common_types.MODE_PREFILL,
                )
                attention_mask = module.config.partition_manager.shard(
                    attention_mask,
                    axes=[common_types.BATCH, common_types.SEQUENCE_PARALLEL],
                    mode=common_types.MODE_PREFILL,
                )
                sequences = module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=GenerationConfig(
                        top_p=self.arguments.top_p,
                        top_k=self.arguments.top_k,
                        temperature=self.arguments.temperature,
                        pad_token_id=self.pad_token_id,
                        eos_token_id=self.eos_token_id,
                        max_new_tokens=self.arguments.max_completion_length,
                        max_length=self.arguments.max_completion_length + self.arguments.max_prompt_length,
                        num_return_sequences=self.num_generations,
                        do_sample=True,
                    ),
                ).sequences
                return sequences, input_ids, attention_mask

        self.generate_function = generate

        self._train_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.epsilon_low,
            self.arguments.epsilon_high,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            self.arguments.per_token_weighting,
            True,  # is_train
        )

        static_argnames = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

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
            self.arguments.epsilon_low,
            self.arguments.epsilon_high,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            self.arguments.per_token_weighting,
            False,  # is_train
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

    def _make_attn_mask(self, arr):
        is_eos = jnp.isin(arr, jnp.asarray(self.eos_token_id).reshape(-1))
        return (
            (jnp.arange(is_eos.shape[1])[None, :].repeat(is_eos.shape[0], axis=0))
            <= jnp.where(
                is_eos.any(axis=1),
                jnp.argmax(is_eos.astype(jnp.int32), axis=1),
                jnp.full((is_eos.shape[0],), is_eos.shape[1]),
            )[:, None]
        ).astype(jnp.int32)

    def _refresh_reference_policy(self, state: EasyDeLState) -> None:
        style = self.arguments.reference_reset_style
        if style == "none" or self.ref_state is None:
            return

        if style == "hard":
            self.ref_state = self.ref_state.replace(
                graphstate=deepcopy_model(state.graphstate),
                graphother=deepcopy_model(state.graphother),
            )
        elif style == "mix":
            alpha = self.arguments.ref_model_mixup_alpha

            def _mix(ref, cur):
                return alpha * ref + (1.0 - alpha) * cur

            mixed_graphstate = jax.tree_util.tree_map(_mix, self.ref_state.graphstate, state.graphstate)
            if self.ref_state.graphother is not None and state.graphother is not None:
                mixed_graphother = jax.tree_util.tree_map(_mix, self.ref_state.graphother, state.graphother)
            else:
                mixed_graphother = self.ref_state.graphother

            self.ref_state = self.ref_state.replace(graphstate=mixed_graphstate, graphother=mixed_graphother)
        else:
            raise ValueError(f"Unknown reference reset style: {style}")

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            with capture_time() as generation_time_fn:
                sequences, prompt_ids, prompt_mask = jax.block_until_ready(
                    self.generate_function(state, prompt_ids, prompt_mask)
                )
            generation_time = generation_time_fn()
            prompt_completion_ids = sequences
            completion_ids = prompt_completion_ids[..., prompt_ids.shape[-1] :]
            completion_mask = self._make_attn_mask(completion_ids)
            ridmask = prompt_mask.repeat(self.num_generations, 0)

            with capture_time() as token_logps_time_fn:
                ref_per_token_logps = self.compute_refmodel_logps(
                    self.ref_state.graphstate,
                    self.ref_state.graphother,
                    prompt_completion_ids,
                    jnp.concatenate([ridmask, completion_mask], -1),
                )
            token_logps_time = token_logps_time_fn()
            prompts = self.processing_class.batch_decode(batch["input_ids"], skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational
            if is_conversational:
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                completions = completions_text

            rewards_per_func = jnp.zeros(
                (prompt_ids.shape[0] * self.num_generations, len(self.reward_funcs)),
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
                                for p, c in zip(prompts * self.num_generations, completions, strict=False)
                            ]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(prompts * self.num_generations, completions, strict=False)]

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
                                    max_length=self.arguments.max_sequence_length,
                                )
                            ),
                        ).logits[:, 0]
                    else:
                        in_prompts = prompts * self.num_generations
                        output_reward_func = reward_func(
                            prompts=in_prompts,
                            completions=completions,
                            max_length=self.arguments.max_sequence_length,
                            batch=batch,
                        )
                        rew = jnp.array(output_reward_func, dtype="f4")
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
            rewarding_time = rewarding_time_fn()
            with capture_time() as grouped_comp_time_fn:
                rewards = rewards_per_func.sum(axis=1)
                completion_length = jnp.sum(completion_mask, axis=1)
                length_bonus = compute_length_reward(
                    completion_length,
                    self.arguments.max_completion_length,
                    self.arguments.length_cache_tokens,
                    self.arguments.length_shaping,
                    self.arguments.length_reward_scale,
                )
                total_rewards = rewards + length_bonus
                (
                    normalized_advantages,
                    group_means,
                    group_stds,
                    collapsed_mask,
                ) = compute_group_advantages(
                    total_rewards,
                    self.num_generations,
                    epsilon=self.arguments.z_score_epsilon,
                    enforce_mixed=self.arguments.enforce_mixed_sampling,
                    jitter=self.arguments.dynamic_sampling_jitter,
                )
                raw_advantages = total_rewards - group_means

                if self.arguments.adv_estimator == "group":
                    advantages = normalized_advantages
                elif self.arguments.adv_estimator == "gae":
                    advantages = raw_advantages
                elif self.arguments.adv_estimator == "truncated":
                    decay = 1.0 - (self.arguments.gae_gamma ** self.arguments.truncated_return_k)
                    decay = jnp.maximum(decay, self.arguments.z_score_epsilon)
                    advantages = raw_advantages * decay
                else:
                    raise ValueError(f"Unsupported advantage estimator: {self.arguments.adv_estimator}")

                if self.arguments.enforce_mixed_sampling:
                    collapsed_repeated = jnp.repeat(collapsed_mask, self.num_generations)
                    advantages = jnp.where(
                        collapsed_repeated,
                        normalized_advantages,
                        advantages,
                    )

                value_targets = total_rewards.astype(jnp.float32)
            grouped_comp_time = grouped_comp_time_fn()
        preprocessing_time = preprocessing_time_fn()
        completion_length = jnp.sum(completion_mask.sum(-1), -1)
        token_weights = completion_mask.astype(jnp.float32) / jnp.maximum(
            completion_mask.sum(axis=1, keepdims=True).astype(jnp.float32),
            1.0,
        )
        token_weights = token_weights.astype(jnp.float32)
        positive_fraction = jnp.mean(
            (total_rewards.reshape(-1, self.num_generations) > self.arguments.positive_reward_threshold).astype(
                jnp.float32
            )
        )
        residual = (total_rewards - group_means).astype(jnp.float32)
        overall_var = jnp.var(total_rewards.astype(jnp.float32))
        residual_var = jnp.var(residual)
        explained_variance = jnp.where(overall_var > 1e-6, 1.0 - residual_var / overall_var, 0.0)
        mae = jnp.mean(jnp.abs(residual))
        metrics_dict = {
            "rewards": jnp.mean(rewards, -1),
            "completion_length": jnp.mean(completion_length),
            "grouped_comp_time": grouped_comp_time,
            "rewarding_time": rewarding_time,
            "token_logps_time": token_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
            "length_bonus": jnp.mean(length_bonus),
            "collapsed_groups": jnp.asarray(jnp.sum(collapsed_mask), dtype=jnp.float32),
            "positive_fraction": positive_fraction,
            "value_explained_var": explained_variance,
            "value_mae": mae,
        }
        for i, reward_func in enumerate(self.reward_funcs):
            _name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
            metrics_dict[_name] = jnp.mean(rewards_per_func[:, i])
        if self.log_table is not None:
            cur_step = jax.device_get(state.step)
            decoded_text = self.processing_class.batch_decode(jax.device_get(completion_ids))
            for text in decoded_text:
                self.log_table.add_data(text, generation_time, completion_length, cur_step)
            wandb.log({"generations": self.log_table}, step=cur_step)

        # i don't care who you are and what you do.
        # ill find you and ill gather u...
        return (
            {
                "prompt_ids": self._all_gather(prompt_ids),
                "prompt_mask": self._all_gather(prompt_mask),
                "completion_ids": self._all_gather(completion_ids),
                "completion_mask": self._all_gather(completion_mask),
                "ref_per_token_logps": self._all_gather(ref_per_token_logps),
                "advantages": advantages,
                "token_weights": self._all_gather(token_weights),
                "value_targets": value_targets,
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
        other_metrics = dict(metrics.other_metrics or {})

        reset_triggered = False

        mean_kl = other_metrics.get("mean_kl")
        entropy = other_metrics.get("entropy")

        if mean_kl is not None:
            mean_kl_scalar = float(jax.device_get(mean_kl))
            self._kl_ema = update_ema(self._kl_ema, mean_kl_scalar, self.arguments.kl_horizon)
            other_metrics["kl_ema"] = jnp.asarray(self._kl_ema, dtype=jnp.float32)

        if entropy is not None:
            entropy_scalar = float(jax.device_get(entropy))
            self._entropy_ema = update_ema(self._entropy_ema, entropy_scalar, self.arguments.kl_horizon)
            other_metrics["entropy_ema"] = jnp.asarray(self._entropy_ema, dtype=jnp.float32)

        should_reset = False
        if self.arguments.reference_reset_style != "none" and self.ref_state is not None:
            if self.arguments.reference_reset_steps and self.arguments.reference_reset_steps > 0:
                if (step - self._last_reference_reset) >= self.arguments.reference_reset_steps:
                    should_reset = True
            if (
                not should_reset
                and self.arguments.kl_target is not None
                and self._kl_ema is not None
                and self._kl_ema > self.arguments.kl_target
            ):
                should_reset = True
            if (
                not should_reset
                and self.arguments.entropy_floor is not None
                and self._entropy_ema is not None
                and self._entropy_ema < self.arguments.entropy_floor
            ):
                should_reset = True

            if should_reset:
                self._refresh_reference_policy(state)
                self._last_reference_reset = step
                reset_triggered = True

        if (
            not reset_triggered
            and self.arguments.sync_ref_model
            and self.ref_state is not None
            and (step % self.arguments.ref_model_sync_steps == 0)
        ):
            self.ref_state = self.ref_state.replace(graphstate=deepcopy_model(state.graphstate))
            reset_triggered = True

        if reset_triggered:
            other_metrics["reference_reset"] = jnp.asarray(1.0, dtype=jnp.float32)

        metrics = metrics.replace(other_metrics=other_metrics)
        return state, metrics
