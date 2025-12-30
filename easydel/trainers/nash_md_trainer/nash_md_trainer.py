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

import flax.nnx
import jax
import jax.nn
import numpy as np
from datasets import Dataset, IterableDataset
from eformer.escale import with_sharding_constraint
from eformer.loggings import get_logger
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time

from ..group_relative_policy_optimization._fn import get_per_token_logps
from ..group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import resolve_straight_through_emulator
from ._fn import nash_md_step
from .nash_md_config import NashMDConfig

logger = get_logger(__name__)

if tp.TYPE_CHECKING:
    pass


def _ensure_state(model: EasyDeLBaseModule | EasyDeLState | None) -> EasyDeLState | None:
    """Convert model to EasyDeLState if needed.

    Args:
        model: Model or state to convert.

    Returns:
        EasyDeLState or None.
    """
    if model is None:
        return None
    if isinstance(model, EasyDeLState):
        return model
    return model.to_state()


@Registry.register("trainer", ["nash-md", "nash_md"])
class NashMDTrainer(GRPOTrainer):
    """Nash Mirror Descent trainer for preference optimization.

    Implements the Nash-MD algorithm which optimizes language models using
    a mixture of policy and reference model generations. Extends GRPO with
    Nash equilibrium-based optimization.

    Args:
        arguments: Nash-MD specific training configuration.
        model: Policy model to train.
        reward_funcs: Reward function(s) for scoring completions.
        reference_model: Reference model for mixture generation.
        train_dataset: Training dataset with prompts.
        eval_dataset: Optional evaluation dataset.
        processing_class: Tokenizer or processor.
        reward_processing_classes: Optional processing classes for reward models.
    """

    arguments: NashMDConfig

    def __init__(
        self,
        arguments: NashMDConfig,
        model: EasyDeLBaseModule | EasyDeLState,
        reward_funcs: tp.Sequence[tp.Callable] | tp.Callable,
        *,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | IterableDataset | None = None,
        processing_class: ProcessingClassType,
        reward_processing_classes: ProcessingClassType | list[ProcessingClassType] | None = None,
    ):
        if reward_funcs is None:
            raise ValueError("NashMDTrainer requires at least one reward function.")
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
        )
        ref_state = _ensure_state(reference_model)
        if ref_state is not None:
            self.ref_state = ref_state

        self._beta_schedule = arguments.beta
        self._mixture_schedule = arguments.mixture_coef
        self.missing_eos_penalty = arguments.missing_eos_penalty
        self.num_generations = 1
        if reward_funcs is not None:
            reward_funcs_list = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
            if len(reward_funcs_list) != 1:
                raise ValueError("NashMDTrainer only supports a single reward function/model.")

    def _schedule_value(self, schedule: tp.Any, default: float) -> float:
        """Get current value from schedule based on training epoch.

        Args:
            schedule: Single value or sequence of per-epoch values.
            default: Default value if schedule is None.

        Returns:
            Current scheduled value.
        """
        if isinstance(schedule, tp.Sequence):
            if not schedule:
                return default
            if self.max_training_steps is None or self.arguments.num_train_epochs <= 0:
                return float(schedule[0])
            steps_per_epoch = max(self.max_training_steps // self.arguments.num_train_epochs, 1)
            current_step = int(jax.device_get(self.model_state.step)) if self.model_state is not None else 0
            idx = min(current_step // steps_per_epoch, len(schedule) - 1)
            return float(schedule[idx])
        if schedule is None:
            return default
        return float(schedule)

    def _current_beta_value(self) -> float:
        """Get current beta value from schedule.

        Returns:
            Current beta value.
        """
        return self._schedule_value(self._beta_schedule, 0.1)

    def _current_mixture_coef(self) -> float:
        """Get current mixture coefficient from schedule.

        Returns:
            Current mixture coefficient.
        """
        return self._schedule_value(self._mixture_schedule, 0.5)

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any, ...]:
        return (jnp.asarray(self._current_beta_value(), dtype=jnp.float32),)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any, ...]:
        return (jnp.asarray(self._current_beta_value(), dtype=jnp.float32),)

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure JIT-compiled training and evaluation functions.

        Returns:
            Configuration containing compiled step functions and mesh.
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
            nash_md_step,
            in_shardings=(self.state_shardings, empty_sharding, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
        )
        sharded_evaluation_step_function = ejit(
            nash_md_step,
            in_shardings=(self.state_shardings, empty_sharding, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
        )

        def _compute_model_logps(
            graphtree: flax.nnx.GraphState,
            graphother: flax.nnx.GraphState,
            ids: jax.Array,
            mask: jax.Array,
            graphdef: flax.nnx.GraphDef,
        ):
            apply = flax.nnx.merge(graphdef, graphtree, graphother)
            with apply.mesh:
                ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
                mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
                return get_per_token_logps(apply, ids, mask, self.arguments.max_prompt_length)

        self.compute_refmodel_logps = ejit(
            partial(_compute_model_logps, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef",),
            in_shardings=(
                self.ref_state.shardings.graphstate,
                self.ref_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=empty_sharding,
        )

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _score_rewards(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute reward scores for prompt-completion pairs.

        Args:
            prompts: List of prompt strings.
            completions: List of completion strings.

        Returns:
            Tuple of (total_rewards, reward_breakdown_dict).
        """
        rewards = []
        breakdown: dict[str, jnp.ndarray] = {}
        for reward_func, reward_processing_class in zip(
            self.reward_funcs,
            self.reward_processing_classes,
            strict=True,
        ):
            name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
            if isinstance(reward_func, EasyDeLState):
                if reward_processing_class is None:
                    raise ValueError("Reward processing class must be provided when using EasyDeL reward models.")
                texts = [p + c for p, c in zip(prompts, completions, strict=False)]
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
                    prompts=prompts,
                    completions=completions,
                    max_length=self.arguments.max_length,
                )
                values = jnp.asarray(np.asarray(outputs, dtype=np.float32))
            rewards.append(values)
            breakdown[name] = values

        if not rewards:
            zeros = jnp.zeros((len(prompts),), dtype=jnp.float32)
            return zeros, {}

        stacked = jnp.stack(rewards, axis=1)
        totals = stacked.sum(axis=1)
        return totals, breakdown

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Generate model and mixture completions, compute rewards and probabilities.

        Args:
            state: Current model state.
            batch: Input batch with prompts.
            is_train: Whether this is a training step.

        Returns:
            Processed batch with completions and rewards, and metrics dictionary.
        """
        # Purify batch first to handle list of dicts (uncollated batch)
        batch = self._purify_batch(batch)
        with capture_time() as preprocessing_time_fn:
            prompt_ids = batch["input_ids"]
            prompt_mask = batch["attention_mask"]

            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                    config_overrides={"num_return_sequences": 1},
                )
                jax.block_until_ready(results.sequences)
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids
                completion_mask = results.completion_mask
            generation_time = generation_time_fn()

            with capture_time() as mixture_time_fn:  # noqa
                mixture_results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=self.ref_state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                    config_overrides={"num_return_sequences": 1},
                )
                jax.block_until_ready(mixture_results.sequences)
                mixture_completion_ids = mixture_results.completion_ids
                mixture_completion_mask = mixture_results.completion_mask

            # Sample mixture between policy and reference completions per prompt
            mixture_coef = self._current_mixture_coef()
            step_int = int(jax.device_get(state.step))
            mix_key = jax.random.PRNGKey(step_int & 0xFFFFFFFF)
            take_policy = jax.random.uniform(mix_key, (completion_ids.shape[0],)) < mixture_coef
            take_policy = take_policy[:, None]
            mixture_completion_ids = jnp.where(take_policy, completion_ids, mixture_completion_ids)
            mixture_completion_mask = jnp.where(take_policy, completion_mask, mixture_completion_mask)

            prompt_completion_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            attention_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)

            with capture_time() as ref_logps_time_fn:
                ref_token_logps = self.compute_refmodel_logps(
                    self.ref_state.graphstate,
                    self.ref_state.graphother,
                    prompt_completion_ids,
                    attention_mask,
                )
            ref_logps_time = ref_logps_time_fn()

        preprocessing_time = preprocessing_time_fn()

        host_prompt_ids = np.asarray(jax.device_get(prompt_ids))
        host_completion_ids = np.asarray(jax.device_get(completion_ids))
        host_mixture_completion_ids = np.asarray(jax.device_get(mixture_completion_ids))

        prompts_text = list(self.processing_class.batch_decode(host_prompt_ids, skip_special_tokens=True))
        model_completions_text = list(self.processing_class.batch_decode(host_completion_ids, skip_special_tokens=True))
        mixture_completions_text = list(
            self.processing_class.batch_decode(host_mixture_completion_ids, skip_special_tokens=True)
        )

        model_scores, model_breakdown = self._score_rewards(prompts_text, model_completions_text)
        mixture_scores, _ = self._score_rewards(prompts_text, mixture_completions_text)

        penalty = self.missing_eos_penalty
        eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        if penalty is not None and eos_token_id is not None:
            eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
            eos_tensor = jnp.asarray(eos_ids)
            policy_contains_eos = jnp.isin(completion_ids, eos_tensor).any(axis=1)
            mixture_contains_eos = jnp.isin(mixture_completion_ids, eos_tensor).any(axis=1)
            model_scores = model_scores - jnp.where(policy_contains_eos, 0.0, penalty)
            mixture_scores = mixture_scores - jnp.where(mixture_contains_eos, 0.0, penalty)

        probabilities = jax.nn.sigmoid(model_scores - mixture_scores)
        completion_lengths = completion_mask.sum(axis=1)

        metrics_dict: dict[str, float | int | str] = {
            "rewards/chosen": float(jnp.mean(model_scores)),
            "rewards/rejected": float(jnp.mean(mixture_scores)),
            "rewards/margins": float(jnp.mean(model_scores - mixture_scores)),
            "rewards/probabilities": float(jnp.mean(probabilities)),
            "completion_length": float(jnp.mean(completion_lengths)),
            "generation_time": generation_time,
            "reference_logps_time": ref_logps_time,
            "preprocessing_time": preprocessing_time,
            "mixture_coef": float(self._current_mixture_coef()),
            "beta": float(self._current_beta_value()),
        }
        for name, values in model_breakdown.items():
            metrics_dict[f"reward/{name}"] = float(jnp.mean(values))

        processed_batch = {
            "prompt_ids": self._all_gather(prompt_ids),
            "prompt_mask": self._all_gather(prompt_mask),
            "completion_ids": self._all_gather(completion_ids),
            "completion_mask": self._all_gather(completion_mask),
            "ref_token_logps": self._all_gather(ref_token_logps),
            "probabilities": self._all_gather(probabilities),
        }
        return processed_batch, metrics_dict
