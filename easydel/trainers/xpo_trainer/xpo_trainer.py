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

import jax
import numpy as np
from datasets import Dataset, IterableDataset
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time
from easydel.utils.traversals import deepcopy_model

from ..group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_configurations import MetricsType
from ..training_utils import resolve_straight_through_emulator
from ._fn import xpo_step
from .xpo_config import XPOConfig


def _ensure_state(model: EasyDeLBaseModule | EasyDeLState) -> EasyDeLState:
    """Convert a model to EasyDeLState if it isn't already.

    Args:
        model: Either an EasyDeLBaseModule or EasyDeLState instance.

    Returns:
        EasyDeLState instance.
    """
    return model if isinstance(model, EasyDeLState) else model.to_state()


@Registry.register("trainer", "xpo")
class XPOTrainer(GRPOTrainer):
    """Trainer for Exploratory Preference Optimization (XPO).

    XPO extends preference optimization by combining DPO-style learning with an
    exploratory term that encourages the policy to maintain probability mass on
    reference completions. The trainer samples completions from both the policy
    and reference models, compares their rewards, and optimizes using preference
    pairs while adding an exploratory regularization term.

    Attributes:
        arguments: XPO-specific configuration including loss type, beta, and alpha schedules.
    """

    arguments: XPOConfig

    def __init__(
        self,
        arguments: XPOConfig,
        model: EasyDeLBaseModule | EasyDeLState,
        reward_funcs: tp.Sequence[tp.Callable] | tp.Callable,
        *,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | IterableDataset | None = None,
        processing_class: ProcessingClassType,
        reward_processing_classes: ProcessingClassType | list[ProcessingClassType] | None = None,
    ):
        """Initialize the XPO trainer.

        Args:
            arguments: XPO configuration containing hyperparameters and training settings.
            model: The policy model to train (either module or state).
            reward_funcs: Reward function(s) for scoring completions. Can be callable or EasyDeLState.
            reference_model: Optional frozen reference model. If None, initialized from policy model.
            train_dataset: Training dataset containing prompts.
            eval_dataset: Optional evaluation dataset(s).
            processing_class: Tokenizer or processor for encoding text.
            reward_processing_classes: Optional processing class(es) for reward model inputs.
        """

        if reference_model is not None:
            self.ref_state = _ensure_state(reference_model)

        self._beta_schedule = arguments.beta
        self._alpha_schedule = arguments.alpha
        self.loss_type_id = 0 if arguments.loss_type == "sigmoid" else 1

        if reward_funcs is not None:
            reward_funcs_list = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
            if len(reward_funcs_list) != 1:
                raise ValueError("XPOTrainer only supports a single reward function/model.")
            reward_funcs = reward_funcs_list

        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
        )

    def _get_reward_processing_classes(self) -> list[ProcessingClassType | None]:
        """Normalize reward processing classes to a list aligned to reward functions."""
        reward_processing_classes = self.reward_processing_classes
        if reward_processing_classes is None:
            return [None] * len(self.reward_funcs)
        if isinstance(reward_processing_classes, (list, tuple)):
            if len(reward_processing_classes) == 0:
                return [None] * len(self.reward_funcs)
            if len(reward_processing_classes) == 1 and len(self.reward_funcs) > 1:
                return list(reward_processing_classes) * len(self.reward_funcs)
            return list(reward_processing_classes)
        return [reward_processing_classes] * len(self.reward_funcs)

    def _schedule_value(self, schedule: tp.Any, default: float) -> float:
        """Resolve a scheduled value based on current training progress.

        Args:
            schedule: Either a float value or sequence of floats for epoch-wise scheduling.
            default: Default value to use if schedule is empty or None.

        Returns:
            Current scheduled value based on training progress.
        """
        if isinstance(schedule, tp.Sequence):
            if not schedule:
                return default
            if self.max_training_steps is None or self.arguments.num_train_epochs <= 0:
                return float(schedule[0])
            steps_per_epoch = max(self.max_training_steps // self.arguments.num_train_epochs, 1)
            step = int(jax.device_get(self.model_state.step)) if self.model_state is not None else 0
            idx = min(step // steps_per_epoch, len(schedule) - 1)
            return float(schedule[idx])
        return float(schedule if schedule is not None else default)

    def _current_beta_value(self) -> float:
        """Get the current beta (KL penalty scaling) value based on training progress.

        Returns:
            Current beta value from schedule or default of 0.1.
        """
        return self._schedule_value(self._beta_schedule, 0.1)

    def _current_alpha_value(self) -> float:
        """Get the current alpha (exploratory weight) value based on training progress.

        Returns:
            Current alpha value from schedule or default of 1e-5.
        """
        return self._schedule_value(self._alpha_schedule, 1e-5)

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure and compile the training and evaluation step functions.

        Sets up JIT-compiled functions with appropriate sharding specifications
        for distributed training, including both training and evaluation steps.

        Returns:
            TrainerConfigureFunctionOutput containing compiled step functions,
            mesh configuration, and checkpoint manager.
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
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,
            straight_through_emulator,
        )
        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,
            straight_through_emulator,
        )

        static_argnums = (3, 4, 5, 6, 7, 8)
        sharded_training_step_function = ejit(
            xpo_step,
            in_shardings=(self.state_shardings, empty_sharding, self.ref_state.shardings),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
        )
        sharded_evaluation_step_function = ejit(
            xpo_step,
            in_shardings=(self.state_shardings, empty_sharding, self.ref_state.shardings),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
        )

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        self._train_shared_fn_extra_args = (self.ref_state,)
        self._eval_shared_fn_extra_args = (self.ref_state,)

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _score_rewards(
        self,
        prompt_ids: jax.Array,
        prompt_mask: jax.Array,
        completion_ids: jax.Array,
        completion_mask: jax.Array,
        *,
        prompt_texts: list[str] | None,
        completion_texts: list[str] | None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Score completions using configured reward functions.

        Evaluates each completion using all reward functions and aggregates
        the results. Supports both callable reward functions and EasyDeLState
        reward models.

        Args:
            prompt_ids: Token IDs for the prompt portion.
            prompt_mask: Attention mask for the prompt portion.
            completion_ids: Token IDs for the completion portion.
            completion_mask: Attention mask for the completion portion.
            prompt_texts: Optional prompt strings (needed for text-based reward funcs).
            completion_texts: Optional completion strings (needed for text-based reward funcs).

        Returns:
            Tuple of (total_rewards, reward_breakdown) where total_rewards is
            the sum across all reward functions and reward_breakdown is a dict
            mapping reward function names to their individual scores.
        """
        rewards = []
        breakdown: dict[str, jnp.ndarray] = {}
        reward_processing_classes = self._get_reward_processing_classes()
        for reward_func, reward_processing_class in zip(self.reward_funcs, reward_processing_classes, strict=False):
            name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
            if isinstance(reward_func, EasyDeLState) and (
                reward_processing_class is None or reward_processing_class is self.processing_class
            ):
                reward_inputs = {
                    "input_ids": jnp.concatenate([prompt_ids, completion_ids], axis=1),
                    "attention_mask": jnp.concatenate([prompt_mask, completion_mask], axis=1),
                }
                logits = reward_func.apply_fn(
                    reward_func.graphdef,
                    reward_func.graphstate,
                    reward_func.graphother,
                    reward_inputs,
                ).logits[:, 0]
                values = jnp.asarray(logits, dtype=jnp.float32)
            else:
                if prompt_texts is None or completion_texts is None:
                    raise ValueError(
                        "Text prompts are required for text-based reward functions or mismatched tokenizers."
                    )
                if isinstance(reward_func, EasyDeLState):
                    if reward_processing_class is None:
                        raise ValueError("Reward processing class must be provided when using EasyDeL reward models.")
                    texts = [p + c for p, c in zip(prompt_texts, completion_texts, strict=False)]
                    tokenized = reward_processing_class(
                        texts,
                        return_tensors="jax",
                        padding="max_length",
                        padding_side="right",
                        add_special_tokens=False,
                        truncation=True,
                        return_attention_mask=True,
                        max_length=self.arguments.max_length,
                    )
                    logits = reward_func.apply_fn(
                        reward_func.graphdef,
                        reward_func.graphstate,
                        reward_func.graphother,
                        dict(tokenized),
                    ).logits[:, 0]
                    values = jnp.asarray(logits, dtype=jnp.float32)
                else:
                    outputs = reward_func(
                        prompts=prompt_texts,
                        completions=completion_texts,
                        max_length=self.arguments.max_length,
                    )
                    values = jnp.asarray(np.asarray(list(outputs), dtype=np.float32))
            rewards.append(values)
            breakdown[name] = values

        if not rewards:
            zeros = jnp.zeros((prompt_ids.shape[0],), dtype=jnp.float32)
            return zeros, {}

        stacked = jnp.stack(rewards, axis=1)
        totals = stacked.sum(axis=1)
        return totals, breakdown

    def _gather_scalar(self, value: float, batch_size: int) -> jnp.ndarray:
        """Create a scalar array and gather across all devices.

        Args:
            value: Scalar value to broadcast.
            batch_size: Size of the batch dimension.

        Returns:
            Array of shape (batch_size,) filled with the value, gathered across devices.
        """
        arr = jnp.full((batch_size,), value, dtype=jnp.float32)
        return self._all_gather(arr)

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Preprocess a batch by generating completions and computing rewards.

        Generates completions from both the policy and reference models,
        scores them using reward functions, determines preference pairs,
        and prepares the batch for the XPO training step.

        Args:
            state: Current policy model state.
            batch: Input batch containing prompt token IDs and attention masks.
            is_train: Whether this is for training or evaluation.

        Returns:
            Tuple of (processed_batch, metrics_dict) where processed_batch contains
            all necessary tensors for the XPO step and metrics_dict contains
            generation times, rewards, and other preprocessing metrics.
        """
        # Purify batch first to handle list of dicts (uncollated batch)
        batch = self._purify_batch(batch)
        with capture_time() as preprocessing_time_fn:
            prompt_ids = batch["input_ids"]
            prompt_mask = batch["attention_mask"]

            with capture_time() as policy_time_fn:
                policy_results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                    config_overrides={"num_return_sequences": 1},
                )
                jax.block_until_ready(policy_results.sequences)
                prompt_ids = policy_results.prompt_ids
                prompt_mask = policy_results.prompt_mask
                policy_completion_ids = policy_results.completion_ids
                policy_completion_mask = policy_results.completion_mask
            policy_generation_time = policy_time_fn()

            with capture_time() as ref_time_fn:
                ref_results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=self.ref_state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                    config_overrides={"num_return_sequences": 1},
                )
                jax.block_until_ready(ref_results.sequences)
                ref_completion_ids = ref_results.completion_ids
                ref_completion_mask = ref_results.completion_mask
            ref_generation_time = ref_time_fn()

        preprocessing_time = preprocessing_time_fn()

        reward_processing_classes = self._get_reward_processing_classes()
        needs_text_for_rewards = any(
            not isinstance(rf, EasyDeLState) or (rpc is not None and rpc is not self.processing_class)
            for rf, rpc in zip(self.reward_funcs, reward_processing_classes, strict=False)
        )

        prompt_texts: list[str] | None = None
        policy_texts: list[str] | None = None
        ref_texts: list[str] | None = None
        if needs_text_for_rewards:
            prompt_texts = list(
                self.processing_class.batch_decode(
                    np.asarray(jax.device_get(prompt_ids)),
                    skip_special_tokens=True,
                )
            )
            policy_texts = list(
                self.processing_class.batch_decode(
                    np.asarray(jax.device_get(policy_completion_ids)),
                    skip_special_tokens=True,
                )
            )
            ref_texts = list(
                self.processing_class.batch_decode(
                    np.asarray(jax.device_get(ref_completion_ids)),
                    skip_special_tokens=True,
                )
            )

        policy_scores, reward_breakdown = self._score_rewards(
            prompt_ids,
            prompt_mask,
            policy_completion_ids,
            policy_completion_mask,
            prompt_texts=prompt_texts,
            completion_texts=policy_texts,
        )
        ref_scores, _ = self._score_rewards(
            prompt_ids,
            prompt_mask,
            ref_completion_ids,
            ref_completion_mask,
            prompt_texts=prompt_texts,
            completion_texts=ref_texts,
        )

        if self.arguments.missing_eos_penalty is not None:
            eos_id = getattr(self.processing_class, "eos_token_id", None)
            if eos_id is not None:
                eos_ids = eos_id if isinstance(eos_id, (list, tuple)) else [eos_id]
                eos_tensor = jnp.asarray(eos_ids)
                policy_eos = jnp.isin(policy_completion_ids, eos_tensor).any(axis=1)
                ref_eos = jnp.isin(ref_completion_ids, eos_tensor).any(axis=1)
                penalty = self.arguments.missing_eos_penalty
                policy_scores = policy_scores - jnp.where(policy_eos, 0.0, penalty)
                ref_scores = ref_scores - jnp.where(ref_eos, 0.0, penalty)

        chosen_mask = policy_scores >= ref_scores

        beta_value = self._current_beta_value()
        alpha_value = self._current_alpha_value()

        metrics_dict: dict[str, float | int | str] = {
            "rewards/chosen": float(jnp.mean(policy_scores)),
            "rewards/rejected": float(jnp.mean(ref_scores)),
            "rewards/margins": float(jnp.mean(policy_scores - ref_scores)),
            "policy_generation_time": policy_generation_time,
            "reference_generation_time": ref_generation_time,
            "preprocessing_time": preprocessing_time,
            "beta": beta_value,
            "alpha": alpha_value,
        }
        for name, value in reward_breakdown.items():
            metrics_dict[f"reward/{name}"] = float(jnp.mean(value))

        processed_batch = {
            "prompt_ids": self._all_gather(prompt_ids),
            "prompt_mask": self._all_gather(prompt_mask),
            "policy_completion_ids": self._all_gather(policy_completion_ids),
            "policy_completion_mask": self._all_gather(policy_completion_mask),
            "ref_completion_ids": self._all_gather(ref_completion_ids),
            "ref_completion_mask": self._all_gather(ref_completion_mask),
            "chosen_mask": self._all_gather(chosen_mask),
            "beta": self._gather_scalar(beta_value, prompt_ids.shape[0]),
            "alpha": self._gather_scalar(alpha_value, prompt_ids.shape[0]),
            "loss_type": self._gather_scalar(float(self.loss_type_id), prompt_ids.shape[0]),
        }
        return processed_batch, metrics_dict

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """Hook called at the end of each training step.

        Optionally synchronizes the reference model with the current policy
        if sync_ref_model is enabled and the sync interval has been reached.

        Args:
            state: Current model state after the step.
            metrics: Metrics collected during the step.
            step: Current training step number.

        Returns:
            Tuple of (potentially updated state, metrics).
        """
        if (
            self.arguments.sync_ref_model
            and self.ref_state is not None
            and (step % self.arguments.ref_model_sync_steps == 0)
        ):
            self.ref_state = deepcopy_model(state)
        return state, metrics
