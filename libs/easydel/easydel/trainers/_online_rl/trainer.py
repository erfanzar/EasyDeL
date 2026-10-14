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

"""Shared separate-state actor/critic trainer implementation."""

from __future__ import annotations

import json
import typing as tp
from functools import partial

import jax
import numpy as np
import spectrax as spx
from eformer.paths import ePath
from jax import numpy as jnp
from spectrax import with_sharding_constraint
from spectrax.serialization import Checkpointer

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger
from easydel.utils.traversals import deepcopy_model

from ..proximal_policy_optimization_trainer import PPOTrainer
from ..proximal_policy_optimization_trainer.modeling_value_head import CausalLMWithValueHead
from ..trainer_protocol import TrainerConfigureFunctionOutput, TrainerConfigureModelOutput
from ..training_utils import compile_trainer_step, resolve_straight_through_emulator
from .actor_critic import (
    actor_step,
    build_critic_optimizer,
    build_critic_trainable_selector,
    critic_step,
    recover_rewards_from_gae,
    score_actor_tokens,
    score_critic_values,
)
from .config import OnlineActorCriticConfig
from .math import length_adaptive_lambda, skip_observation_gae

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource
    from easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer import RewardFunc

logger = get_logger(__name__)


class RolloutProvider(tp.Protocol):
    """Callable contract for externally orchestrated online trajectories.

    Providers may run synchronous or asynchronous agent/environment loops.
    They return target-aligned arrays; the trainer owns critic scoring,
    advantage construction, and actor/critic updates.
    """

    def __call__(
        self,
        *,
        trainer: OnlineActorCriticTrainer,
        state: EasyDeLState,
        batch: dict[str, tp.Any],
        is_train: bool,
    ) -> dict[str, tp.Any] | tuple[dict[str, tp.Any], dict[str, float | int | str]]:
        """Collect and collate trajectories for one learner batch."""
        ...


class OnlineActorCriticTrainer(PPOTrainer):
    """PPO rollout plumbing with independently optimized actor and critic.

    Public algorithms subclass this internal implementation and select the
    actor loss through :attr:`actor_algorithm`.  PPO's mature prompt
    tokenization, generation, reward routing, reference-policy KL, and logging
    are reused, but its joint policy/value loss is replaced completely:

    * ``model_state`` is the actor and advances once per learner update;
    * ``critic_state`` has its own optimizer, scheduler, step, and checkpoint;
    * masks align with ``input_ids[:, 1:]`` and keep attended observations
      separate from generated actions;
    * critic updates run before actor advantage estimation;
    * actor and critic losses use one global action-token denominator.

    A ``rollout_provider`` can supply full multi-turn or segmented agentic
    batches.  Without one, the inherited single-turn prompt rollout path is
    adapted into the same target-aligned schema.
    """

    actor_algorithm: tp.ClassVar[tp.Literal["sao", "ppo"]] = "ppo"
    requires_value_head: tp.ClassVar[bool] = False
    _RUNTIME_MODEL_OVERRIDE_STATE_ATTRS: tp.ClassVar[frozenset[str]] = PPOTrainer._RUNTIME_MODEL_OVERRIDE_STATE_ATTRS | {
        "critic_state"
    }

    arguments: OnlineActorCriticConfig
    critic_state: EasyDeLState

    def __init__(
        self,
        *,
        arguments: OnlineActorCriticConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc] | None,
        critic_model: EasyDeLBaseModule | None = None,
        critic_state: EasyDeLState | None = None,
        critic_trainable_selector: spx.SelectorSugar | None = None,
        rollout_provider: RolloutProvider | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        data_tokenize_fn: tp.Callable | None = None,
    ) -> None:
        """Initialize distinct actor, reference, and critic states.

        Args:
            arguments: Algorithm-specific online actor/critic configuration.
            model: Actor module or state.
            reward_funcs: One or more reward functions/models.
            critic_model: Optional independently initialized critic base model.
            critic_state: Optional prebuilt value-head critic state. Mutually
                exclusive with ``critic_model``.
            critic_trainable_selector: Optional exact SpectraX selector. When
                omitted, ``arguments.critic_trainable_mode`` is used.
            rollout_provider: Optional callable producing target-aligned
                multi-turn or segmented batches.
            train_dataset: Prompt-only training dataset.
            eval_dataset: Optional evaluation dataset(s).
            processing_class: Tokenizer/processor.
            reward_processing_classes: Optional reward-model processors.
            data_tokenize_fn: Optional tokenizer override.
        """
        if model is None:
            raise ValueError("Online actor/critic trainers require `model`.")
        if isinstance(model, EasyDeLState) and isinstance(model.model, CausalLMWithValueHead):
            raise ValueError(
                "Online actor/critic trainers require a plain causal-LM actor state; "
                "pass the underlying base model instead of a PPO value-head state."
            )
        if isinstance(model, CausalLMWithValueHead):
            model = model.model
        if critic_model is not None and critic_state is not None:
            raise ValueError("Pass at most one of `critic_model` and `critic_state`.")
        if critic_state is not None and not hasattr(critic_state.model, "value_head"):
            raise ValueError("`critic_state` must contain a model with a `value_head`.")
        self.arguments = arguments
        self.rollout_provider = rollout_provider
        self._critic_pretraining_complete = False
        self.critic_state = self._prepare_critic_state(
            actor=model,
            critic_model=critic_model,
            critic_state=critic_state,
            selector=critic_trainable_selector,
            arguments=arguments,
        )
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

    @staticmethod
    def _unwrap_value_model(module: EasyDeLBaseModule) -> EasyDeLBaseModule:
        """Return the causal LM inside an existing value-head wrapper."""
        if isinstance(module, CausalLMWithValueHead):
            return module.model
        return module

    @classmethod
    def _prepare_critic_state(
        cls,
        *,
        actor: EasyDeLBaseModule | EasyDeLState,
        critic_model: EasyDeLBaseModule | None,
        critic_state: EasyDeLState | None,
        selector: spx.SelectorSugar | None,
        arguments: OnlineActorCriticConfig,
    ) -> EasyDeLState:
        """Create or normalize a value-head critic before base initialization."""
        if critic_state is not None:
            return critic_state

        if critic_model is not None:
            base_critic = critic_model
        else:
            actor_module = actor.model if isinstance(actor, EasyDeLState) else actor
            base_critic = deepcopy_model(cls._unwrap_value_model(actor_module))

        if not hasattr(base_critic, "value_head"):
            base_critic = CausalLMWithValueHead(base_critic, rngs=spx.Rngs(0))
        trainable_selector = selector or build_critic_trainable_selector(arguments.critic_trainable_mode)
        return base_critic.to_state(trainable_selector=trainable_selector)

    def configure_model(self) -> TrainerConfigureModelOutput:
        """Configure independent actor and critic optimizers/schedulers."""
        actor_output = super().configure_model()
        critic_training_steps = int(self.arguments.critic_pretrain_steps)
        if self.max_training_steps is not None:
            critic_training_steps += int(self.max_training_steps) * int(self.arguments.critic_updates_per_actor_update)
        self.critic_tx, self.critic_scheduler = build_critic_optimizer(
            self.arguments,
            critic_training_steps,
        )
        return actor_output

    def _configure_state(self) -> None:
        """Initialize optimizer slots and shard both actor and critic states."""
        if self._resumed_from_checkpoint and self.critic_state.opt_state is not None:
            current_step = self.critic_state.step
            self.critic_state = self.critic_state.replace(
                tx=self.critic_tx,
                step=current_step,
            )
        elif self.critic_state.opt_state is None and self.arguments.init_tx:
            self.critic_state = self.critic_state.init_tx(self.critic_tx)
        elif self.critic_state.tx is None:
            self.critic_state = self.critic_state.replace(tx=self.critic_tx)
        super()._configure_state()
        self.critic_state_shardings = self.critic_state.shardings

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Compile policy-only actor, critic, and full-row scoring functions."""
        inherited = super().configure_functions()
        mesh = self.model.mesh
        empty_sharding = replicated_named_sharding(mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            quantization_bits=self.arguments.quantization_bits,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )
        epsilon_low = float(getattr(self.arguments, "dis_epsilon_low", 0.0))
        epsilon_high = float(getattr(self.arguments, "dis_epsilon_high", 0.0))
        entropy_coef = 0.0 if self.arguments.entropy_coef is None else float(self.arguments.entropy_coef)
        actor_static_args = (
            self.actor_algorithm,
            epsilon_low,
            epsilon_high,
            float(self.arguments.cliprange),
            entropy_coef,
            self.arguments.logprob_vocab_chunk_size,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            int(self.arguments.gradient_accumulation_steps),
            bool(self.arguments.global_token_normalization),
            True,
            straight_through_emulator,
        )
        self._train_shared_fn_static_args = actor_static_args
        actor_static_argnums = tuple(range(2, 15))
        sharded_actor_step = compile_trainer_step(
            actor_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=actor_static_argnums,
            mesh=mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )

        self._eval_shared_fn_static_args = (*actor_static_args[:-2], False, straight_through_emulator)
        sharded_actor_eval_step = compile_trainer_step(
            actor_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=actor_static_argnums,
            mesh=mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )

        self._critic_shared_fn_static_args = (
            float(self.arguments.critic_cliprange_value),
            self.arguments.loss_config,
            self.critic_scheduler,
            self.arguments.step_partition_spec,
            int(self.arguments.gradient_accumulation_steps),
            bool(self.arguments.global_token_normalization),
            True,
        )
        critic_static_argnums = tuple(range(2, 9))
        self.sharded_critic_step_function = compile_trainer_step(
            critic_step,
            in_shardings=(self.critic_state_shardings, empty_sharding),
            out_shardings=(self.critic_state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=critic_static_argnums,
            mesh=self.critic_state.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )

        def _score_values(graphtree, graphother, ids, mask, graphdef):
            """Bind a critic state and score all target-aligned values."""
            graphother = jax.tree_util.tree_map(
                lambda value: jax.lax.stop_gradient(value) if hasattr(value, "shape") else value,
                graphother,
            )
            module = spx.bind(graphdef, graphtree.merge(graphother, copy=False))
            ids = with_sharding_constraint(
                ids,
                self.arguments.step_partition_spec,
                mesh=module.mesh,
                ignore_mpmd=True,
            )
            mask = with_sharding_constraint(
                mask,
                self.arguments.step_partition_spec,
                mesh=module.mesh,
                ignore_mpmd=True,
            )
            return score_critic_values(module, ids, mask)

        self.compute_critic_values = compile_trainer_step(
            partial(_score_values, graphdef=self.critic_state.graphdef),
            mesh=self.critic_state.model.mesh,
            static_argnames=("graphdef",),
            in_shardings=(
                self.critic_state.shardings.graphstate,
                self.critic_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=empty_sharding,
        )

        def _score_logps(graphtree, graphother, ids, mask, graphdef):
            """Bind a policy state and score all target-aligned log-probabilities."""
            graphother = jax.tree_util.tree_map(
                lambda value: jax.lax.stop_gradient(value) if hasattr(value, "shape") else value,
                graphother,
            )
            module = spx.bind(graphdef, graphtree.merge(graphother, copy=False))
            ids = with_sharding_constraint(
                ids,
                self.arguments.step_partition_spec,
                mesh=module.mesh,
                ignore_mpmd=True,
            )
            mask = with_sharding_constraint(
                mask,
                self.arguments.step_partition_spec,
                mesh=module.mesh,
                ignore_mpmd=True,
            )
            logps, _ = score_actor_tokens(
                module,
                ids,
                mask,
                logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                return_entropy=False,
            )
            return logps

        self.compute_actor_logps_full = compile_trainer_step(
            partial(_score_logps, graphdef=self.model_state.graphdef),
            mesh=mesh,
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=empty_sharding,
        )
        if self.ref_state is None:
            # kl_coef == 0: PPO skips the reference model entirely, so there is
            # nothing to score against (the KL term is already gated below).
            self.compute_reference_logps_full = None
        else:
            self.compute_reference_logps_full = compile_trainer_step(
                partial(_score_logps, graphdef=self.ref_state.graphdef),
                mesh=mesh,
                static_argnames=("graphdef",),
                in_shardings=(
                    self.ref_state.shardings.graphstate,
                    self.ref_state.shardings.graphother,
                    empty_sharding,
                    empty_sharding,
                ),
                out_shardings=empty_sharding,
            )

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_actor_step,
            sharded_evaluation_step_function=sharded_actor_eval_step,
            mesh=mesh,
            checkpoint_manager=inherited.checkpoint_manager,
        )

    @staticmethod
    def _skip_observation_gae(
        rewards: jax.Array,
        values: jax.Array,
        action_mask: jax.Array,
        done_mask: jax.Array,
        *,
        gamma: float,
        gae_lambda: float | jax.Array,
        bootstrap_value: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute GAE while bridging across non-action observation tokens."""
        if bootstrap_value is None:
            bootstrap_value = jnp.zeros(rewards.shape[:-1], dtype=jnp.float32)
        transition_mask = action_mask.astype(jnp.bool_) & ~done_mask.astype(jnp.bool_)
        return skip_observation_gae(
            rewards,
            values,
            action_mask,
            gamma=gamma,
            lam=gae_lambda,
            transition_mask=transition_mask,
            bootstrap_values=bootstrap_value,
        )

    def _policy_lambdas(self, batch: dict[str, jax.Array]) -> jax.Array:
        """Resolve one fixed or length-adaptive lambda per trajectory row."""
        batch_size = int(batch["action_mask"].shape[0])
        if self.arguments.policy_gae_mode == "fixed":
            return jnp.full((batch_size,), float(self.arguments.lam), dtype=jnp.float32)

        action_counts = np.asarray(jax.device_get(jnp.sum(batch["action_mask"], axis=1)), dtype=np.float32)
        trajectory_ids = batch.get("trajectory_ids")
        if trajectory_ids is not None:
            host_ids = np.asarray(jax.device_get(trajectory_ids)).reshape(-1)
            total_by_trajectory: dict[int, float] = {}
            for trajectory_id, count in zip(host_ids.tolist(), action_counts.tolist(), strict=True):
                total_by_trajectory[int(trajectory_id)] = total_by_trajectory.get(int(trajectory_id), 0.0) + count
            action_counts = np.asarray([total_by_trajectory[int(value)] for value in host_ids], dtype=np.float32)
        return length_adaptive_lambda(
            jnp.asarray(action_counts, dtype=jnp.float32),
            alpha=float(self.arguments.length_adaptive_alpha),
        )

    def _transform_policy_advantages(
        self,
        advantages: jax.Array,
        batch: dict[str, jax.Array],
        lambdas: jax.Array,
    ) -> jax.Array:
        """Algorithm hook for post-GAE advantage transformations."""
        del batch, lambdas
        return advantages

    def _prepare_target_batch(
        self,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Score critic values and construct critic plus actor targets."""
        action_mask = batch["action_mask"].astype(jnp.float32)
        done_mask = batch.get("done_mask", jnp.zeros_like(action_mask)).astype(jnp.float32)
        old_values = self.compute_critic_values(
            self.critic_state.graphstate,
            self.critic_state.graphother,
            batch["input_ids"],
            batch["attention_mask"],
        )
        bootstrap_value = batch.get("bootstrap_value")
        policy_lambdas = self._policy_lambdas(batch)
        critic_lambdas: float | jax.Array
        if self.arguments.critic_gae_mode == "match_policy":
            critic_lambdas = policy_lambdas
        else:
            critic_lambdas = float(self.arguments.critic_gae_lambda)
        _, critic_returns = self._skip_observation_gae(
            batch["rewards"],
            old_values,
            action_mask,
            done_mask,
            gamma=float(self.arguments.gamma),
            gae_lambda=critic_lambdas,
            bootstrap_value=bootstrap_value,
        )
        advantages, _ = self._skip_observation_gae(
            batch["rewards"],
            old_values,
            action_mask,
            done_mask,
            gamma=float(self.arguments.gamma),
            gae_lambda=policy_lambdas,
            bootstrap_value=bootstrap_value,
        )
        advantages = self._transform_policy_advantages(advantages, batch, policy_lambdas)
        if self.arguments.whiten_advantages:
            advantages = self._masked_whiten(advantages, action_mask, shift_mean=True) * action_mask
        batch = {
            **batch,
            "action_mask": action_mask,
            "done_mask": done_mask,
            "old_values": old_values,
            "critic_returns": critic_returns,
            "advantages": advantages,
            "num_items_in_batch": jnp.sum(action_mask),
        }
        return batch

    def _adapt_ppo_rollout_batch(
        self,
        ppo_batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Convert the inherited contiguous PPO batch to full-row target masks."""
        input_ids = ppo_batch["input_ids"]
        batch_size, sequence_length = input_ids.shape
        completion_mask = ppo_batch["completion_mask"].astype(jnp.float32)
        completion_length = int(completion_mask.shape[1])
        prompt_length = sequence_length - completion_length
        target_length = sequence_length - 1
        start = prompt_length - 1
        action_mask = jnp.zeros((batch_size, target_length), dtype=jnp.float32)
        action_mask = action_mask.at[:, start : start + completion_length].set(completion_mask)
        behavior_logps = jnp.zeros_like(action_mask)
        behavior_logps = behavior_logps.at[:, start : start + completion_length].set(ppo_batch["old_logps"])

        contiguous_advantages = ppo_batch["returns"] - ppo_batch["old_values"]
        completion_rewards = recover_rewards_from_gae(
            contiguous_advantages,
            ppo_batch["old_values"],
            completion_mask,
            gamma=float(self.arguments.gamma),
            gae_lambda=float(self.arguments.lam),
        )
        rewards = jnp.zeros_like(action_mask)
        rewards = rewards.at[:, start : start + completion_length].set(completion_rewards)
        done_mask = jnp.zeros_like(action_mask)
        lengths = jnp.sum(completion_mask, axis=1).astype(jnp.int32)
        rows = jnp.arange(batch_size)
        last_targets = start + jnp.maximum(lengths - 1, 0)
        done_mask = done_mask.at[rows, last_targets].set((lengths > 0).astype(jnp.float32))
        return self._prepare_target_batch(
            {
                "input_ids": input_ids,
                "attention_mask": ppo_batch["attention_mask"],
                "action_mask": action_mask,
                "action_end_mask": done_mask,
                "done_mask": done_mask,
                "rewards": rewards,
                "behavior_logps": behavior_logps,
            }
        )

    def _score_rollout_policy_and_values(
        self,
        state: EasyDeLState,
        input_ids: jax.Array,
        attention_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Score a synchronous rollout with the plain actor and critic.

        Args:
            state: Exact actor snapshot used for generation.
            input_ids: Full prompt-plus-completion token IDs.
            attention_mask: Full attention mask.

        Returns:
            Completion-aligned actor log-probabilities and critic values.
        """
        logps = self.compute_actor_logps_full(
            state.graphstate,
            state.graphother,
            input_ids,
            attention_mask,
        )
        values = self.compute_critic_values(
            self.critic_state.graphstate,
            self.critic_state.graphother,
            input_ids,
            attention_mask,
        )
        completion_start = int(self.arguments.max_prompt_length) - 1
        return logps[:, completion_start:], values[:, completion_start:]

    def _normalize_provider_batch(
        self,
        provider_batch: dict[str, tp.Any],
        *,
        scoring_state: EasyDeLState,
    ) -> dict[str, jax.Array]:
        """Validate and enrich a provider-supplied target-aligned batch.

        Args:
            provider_batch: Full-token or target-aligned rollout arrays.
            scoring_state: Actor snapshot whose log-probabilities are used
                when the provider requests synchronous recomputation.

        Returns:
            A target-aligned actor/critic learner batch.
        """
        provider_batch = dict(provider_batch)
        rewards_include_kl = bool(provider_batch.pop("rewards_include_kl", False))
        required = ("input_ids", "attention_mask")
        missing = [name for name in required if name not in provider_batch]
        if "action_mask" not in provider_batch and "target_action_mask" not in provider_batch:
            missing.append("action_mask or target_action_mask")
        if missing:
            raise ValueError(f"Rollout provider batch is missing required fields: {missing}.")
        normalized: dict[str, jax.Array] = {}
        for key, value in provider_batch.items():
            if value is None:
                continue
            try:
                normalized[key] = jnp.asarray(value)
            except (TypeError, ValueError):
                continue
        input_ids = normalized["input_ids"]
        attention_mask = normalized["attention_mask"]
        if input_ids.ndim != 2:
            raise ValueError(f"`input_ids` must have shape [batch, sequence], got {input_ids.shape}.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(f"`attention_mask` must have shape {input_ids.shape}, got {attention_mask.shape}.")
        expected_shape = (input_ids.shape[0], input_ids.shape[1] - 1)
        action_mask = normalized.get("target_action_mask", normalized.get("action_mask"))
        if action_mask is None:  # pragma: no cover - guarded above
            raise AssertionError("missing action mask")
        if action_mask.shape == input_ids.shape:
            action_mask = (
                action_mask[:, 1:].astype(jnp.bool_)
                & attention_mask[:, :-1].astype(jnp.bool_)
                & attention_mask[:, 1:].astype(jnp.bool_)
            )
        action_mask = action_mask.astype(jnp.float32)
        if action_mask.shape != expected_shape:
            raise ValueError(
                f"`action_mask` must align with input_ids[:, 1:] and have shape {expected_shape}, "
                f"got {action_mask.shape}."
            )
        valid_target_pair = attention_mask[:, :-1].astype(jnp.bool_) & attention_mask[:, 1:].astype(jnp.bool_)
        if np.any(
            np.asarray(
                jax.device_get(action_mask.astype(jnp.bool_) & ~valid_target_pair),
                dtype=np.bool_,
            )
        ):
            raise ValueError("`action_mask` cannot select a target outside the attended causal context.")

        behavior_logps = normalized.get("target_behavior_logps", normalized.get("behavior_logps"))
        if behavior_logps is None:
            if self.arguments.rollout_logprob_source == "engine":
                raise ValueError(
                    "`rollout_logprob_source='engine'` requires token-aligned `behavior_logps` from the provider."
                )
            behavior_logps = self.compute_actor_logps_full(
                scoring_state.graphstate,
                scoring_state.graphother,
                input_ids,
                normalized["attention_mask"],
            )
        elif behavior_logps.shape == input_ids.shape:
            behavior_logps = behavior_logps[:, 1:]
        if behavior_logps.shape != expected_shape:
            raise ValueError(f"`behavior_logps` must have shape {expected_shape}, got {behavior_logps.shape}.")
        host_action_mask = np.asarray(jax.device_get(action_mask), dtype=np.bool_)
        host_behavior_logps = np.asarray(jax.device_get(behavior_logps))
        if not np.all(np.isfinite(host_behavior_logps[host_action_mask])):
            raise ValueError("`behavior_logps` must be finite at every optimized action token.")
        behavior_logps = jnp.where(
            action_mask.astype(jnp.bool_),
            behavior_logps.astype(jnp.float32),
            0.0,
        )

        rewards = normalized.get("target_rewards", normalized.get("rewards"))
        if rewards is None:
            rewards = jnp.zeros(expected_shape, dtype=jnp.float32)
        elif rewards.shape == input_ids.shape:
            rewards = rewards[:, 1:]
        if rewards.shape != expected_shape:
            raise ValueError(f"`rewards` must have shape {expected_shape}, got {rewards.shape}.")
        host_rewards = np.asarray(jax.device_get(rewards))
        if not np.all(np.isfinite(host_rewards[host_action_mask])):
            raise ValueError("`rewards` must be finite at every optimized action token.")
        rewards = jnp.where(action_mask.astype(jnp.bool_), rewards.astype(jnp.float32), 0.0)

        done_mask = normalized.get("done_mask", normalized.get("action_end_mask"))
        if done_mask is not None and done_mask.shape == input_ids.shape:
            done_mask = done_mask[:, 1:]
        transition_mask = normalized.get("target_transition_mask", normalized.get("transition_mask"))
        if done_mask is None:
            if transition_mask is not None:
                if transition_mask.shape == input_ids.shape:
                    transition_mask = transition_mask[:, 1:]
                if transition_mask.shape != expected_shape:
                    raise ValueError(f"`transition_mask` must have shape {expected_shape}, got {transition_mask.shape}.")
                done_mask = action_mask * (1.0 - transition_mask.astype(jnp.bool_).astype(jnp.float32))

        segment_ids = normalized.get("target_segment_ids", normalized.get("segment_ids"))
        if segment_ids is not None:
            if segment_ids.shape == input_ids.shape:
                segment_ids = segment_ids[:, 1:]
            if segment_ids.shape == expected_shape:
                host_segments = np.asarray(jax.device_get(segment_ids))
                if np.any(host_segments[host_action_mask] < 0):
                    raise ValueError("Every optimized action token must have a non-negative `segment_id`.")
                row_segment_ids = []
                for row_segments, row_actions in zip(host_segments, host_action_mask, strict=True):
                    action_segments = np.unique(row_segments[row_actions])
                    action_segments = action_segments[action_segments >= 0]
                    if action_segments.size > 1:
                        raise ValueError(
                            "Each provider row may optimize only one segment. Compacted segments use "
                            "different causal contexts and must be returned as separate full-context rows."
                        )
                    row_segment_ids.append(int(action_segments[0]) if action_segments.size else 0)
                segment_ids = jnp.asarray(row_segment_ids, dtype=jnp.int32)
            elif segment_ids.shape == (input_ids.shape[0],):
                segment_ids = segment_ids.astype(jnp.int32)
                if np.any(np.asarray(jax.device_get(segment_ids)) < 0):
                    raise ValueError("Row-level `segment_ids` must be non-negative.")
            else:
                raise ValueError(
                    "`segment_ids` must be target-token aligned with shape "
                    f"{expected_shape} or have one value per provider row; got {segment_ids.shape}."
                )
            normalized["segment_ids"] = segment_ids

        if done_mask is not None:
            if done_mask.shape != expected_shape:
                raise ValueError(f"`done_mask` must have shape {expected_shape}, got {done_mask.shape}.")
            normalized["done_mask"] = done_mask.astype(jnp.float32) * action_mask

        trajectory_ids = normalized.get("trajectory_ids")
        if trajectory_ids is not None:
            trajectory_ids = trajectory_ids.reshape(-1)
            if trajectory_ids.shape != (input_ids.shape[0],):
                raise ValueError(f"`trajectory_ids` must have one value per provider row, got {trajectory_ids.shape}.")
            normalized["trajectory_ids"] = trajectory_ids

        policy_versions = normalized.get("policy_versions")
        if policy_versions is not None:
            policy_versions = policy_versions.reshape(-1)
            if policy_versions.shape != (input_ids.shape[0],):
                raise ValueError(f"`policy_versions` must have one value per provider row, got {policy_versions.shape}.")
            host_versions = np.asarray(jax.device_get(policy_versions), dtype=np.int64)
            if np.any(host_versions < 0):
                raise ValueError("`policy_versions` must be non-negative.")
            max_staleness = getattr(self.arguments, "max_policy_staleness", None)
            if max_staleness is not None:
                if not hasattr(scoring_state, "step"):
                    raise ValueError(
                        "`max_policy_staleness` requires the scoring actor state to expose its update step."
                    )
                current_version = int(np.asarray(jax.device_get(scoring_state.step)))
                if np.any(host_versions > current_version):
                    raise ValueError(
                        "Rollout provider returned a policy version newer than the learner actor "
                        f"(current={current_version}, newest={int(host_versions.max())})."
                    )
                stale_by = current_version - host_versions
                if np.any(stale_by > int(max_staleness)):
                    raise ValueError(
                        "Rollout provider batch exceeds `max_policy_staleness`: "
                        f"current={current_version}, oldest={int(host_versions.min())}, "
                        f"allowed={int(max_staleness)}."
                    )

        scores = normalized.get("scores")
        if scores is not None:
            scores = scores.reshape(-1).astype(jnp.float32)
            if scores.shape != (input_ids.shape[0],):
                raise ValueError(f"`scores` must have one value per provider row, got {scores.shape}.")
            if not np.all(np.isfinite(np.asarray(jax.device_get(scores)))):
                raise ValueError("`scores` must contain only finite values.")
            reward_endpoints = np.zeros(expected_shape, dtype=np.bool_)
            for row, row_actions in enumerate(host_action_mask):
                positions = np.flatnonzero(row_actions)
                if positions.size:
                    reward_endpoints[row, positions[-1]] = True
            rewards = rewards + scores[:, None] * jnp.asarray(reward_endpoints, dtype=jnp.float32)

        if not rewards_include_kl and float(self.arguments.kl_coef) != 0.0:
            reference_logps = normalized.get("reference_logps")
            if reference_logps is None:
                reference_logps = self.compute_reference_logps_full(
                    self.ref_state.graphstate,
                    self.ref_state.graphother,
                    input_ids,
                    normalized["attention_mask"],
                )
            elif reference_logps.shape == input_ids.shape:
                reference_logps = reference_logps[:, 1:]
            if reference_logps.shape != expected_shape:
                raise ValueError(f"`reference_logps` must have shape {expected_shape}, got {reference_logps.shape}.")
            host_reference_logps = np.asarray(jax.device_get(reference_logps))
            if not np.all(np.isfinite(host_reference_logps[host_action_mask])):
                raise ValueError("`reference_logps` must be finite at every optimized action token.")
            reference_logps = jnp.where(
                action_mask.astype(jnp.bool_),
                reference_logps.astype(jnp.float32),
                0.0,
            )
            log_ref_over_behavior = reference_logps - behavior_logps
            if self.arguments.kl_estimator == "k1":
                kl = -log_ref_over_behavior
            else:
                kl = jnp.exp(log_ref_over_behavior) - 1.0 - log_ref_over_behavior
            rewards = rewards - float(self.arguments.kl_coef) * kl * action_mask

        if "done_mask" not in normalized:
            last_mask = jnp.zeros_like(action_mask)
            rows = jnp.arange(action_mask.shape[0])
            last_indices = jnp.maximum(
                jnp.max(
                    jnp.where(
                        action_mask.astype(jnp.bool_),
                        jnp.arange(action_mask.shape[1])[None, :],
                        -1,
                    ),
                    axis=1,
                ),
                0,
            )
            has_action = jnp.any(action_mask.astype(jnp.bool_), axis=1)
            last_mask = last_mask.at[rows, last_indices].set(has_action.astype(jnp.float32))
            normalized["done_mask"] = last_mask
        learner_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_mask": action_mask,
            "behavior_logps": behavior_logps,
            "rewards": rewards * action_mask,
            "done_mask": normalized["done_mask"],
        }
        for key in ("bootstrap_value", "trajectory_ids", "segment_ids"):
            if key in normalized:
                learner_batch[key] = normalized[key]
        return self._prepare_target_batch(learner_batch)

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Collect a rollout and build separate actor/critic targets."""
        if self.rollout_provider is None:
            ppo_batch, information = super()._preprocess_batch_input(
                state=state,
                batch=batch,
                is_train=is_train,
            )
            return self._adapt_ppo_rollout_batch(ppo_batch), information

        provider_output = self.rollout_provider(
            trainer=self,
            state=state,
            batch=batch,
            is_train=is_train,
        )
        if isinstance(provider_output, tuple):
            provider_batch, information = provider_output
        else:
            provider_batch, information = provider_output, {}
        return self._normalize_provider_batch(provider_batch, scoring_state=state), dict(information)

    def _refresh_policy_advantages(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Re-score the updated critic before the actor update."""
        values = self.compute_critic_values(
            self.critic_state.graphstate,
            self.critic_state.graphother,
            batch["input_ids"],
            batch["attention_mask"],
        )
        lambdas = self._policy_lambdas(batch)
        advantages, _ = self._skip_observation_gae(
            batch["rewards"],
            values,
            batch["action_mask"],
            batch["done_mask"],
            gamma=float(self.arguments.gamma),
            gae_lambda=lambdas,
            bootstrap_value=batch.get("bootstrap_value"),
        )
        advantages = self._transform_policy_advantages(advantages, batch, lambdas)
        if self.arguments.whiten_advantages:
            advantages = (
                self._masked_whiten(
                    advantages,
                    batch["action_mask"],
                    shift_mean=True,
                )
                * batch["action_mask"]
            )
        return {**batch, "advantages": advantages}

    def _execute_train_step(
        self,
        state: EasyDeLState,
        batch,
    ) -> tuple[EasyDeLState, LossMetrics, BaseException | None]:
        """Run the per-batch critic updates, refresh advantages, then update actor."""
        metrics = LossMetrics()
        try:
            batch, information = self._preprocess_batch_input(
                state=state,
                batch=batch,
                is_train=True,
            )
            critic_updates = int(self.arguments.critic_updates_per_actor_update)
            critic_metrics = LossMetrics()
            for _ in range(critic_updates):
                self.critic_state, critic_metrics = jax.block_until_ready(
                    self.sharded_critic_step_function(
                        self.critic_state,
                        batch,
                        *self._critic_shared_fn_static_args,
                    )
                )
            batch = self._refresh_policy_advantages(batch)

            for _ in range(int(self.arguments.num_ppo_epochs)):
                state, metrics = jax.block_until_ready(
                    self.sharded_training_step_function(
                        state,
                        batch,
                        *self._train_shared_fn_extra_args,
                        *self._train_shared_fn_static_args,
                    )
                )

            combined_metrics: dict[str, tp.Any] = dict(information)
            if critic_metrics.other_metrics is not None:
                combined_metrics.update(critic_metrics.other_metrics)
            if metrics.other_metrics is not None:
                combined_metrics.update(metrics.other_metrics)
            combined_metrics["critic/updates_this_actor_step"] = jnp.asarray(critic_updates)
            combined_metrics["critic/step"] = self.critic_state.step
            metrics = metrics.replace(other_metrics=combined_metrics)
            return state, metrics, None
        except (
            KeyboardInterrupt,
            EasyDeLTimerError,
            EasyDeLBreakRequest,
            TypeError,
        ) as run_exception:
            return state, metrics, run_exception
        except Exception as run_exception:
            if self._is_memory_oom_exception(run_exception):
                return state, metrics, self._augment_memory_oom_exception(run_exception)
            raise

    def start_training_hook(self) -> None:
        """Run an optional distinct-batch critic pretraining phase.

        ``critic_pretrain_steps`` counts value updates on separate rollout
        batches while the actor remains frozen. This is intentionally outside
        :meth:`_execute_train_step`: repeating a single first batch would not
        implement value pretraining on the training dataset.
        """
        super().start_training_hook()
        pretrain_steps = int(self.arguments.critic_pretrain_steps)
        if self._critic_pretraining_complete or pretrain_steps == 0:
            self._critic_pretraining_complete = True
            return
        if self.dataloader_train is None:
            raise ValueError("Critic pretraining requires a training dataset.")

        logger.info("Starting %d critic-only value-pretraining steps.", pretrain_steps)
        train_iter = iter(self.dataloader_train)
        data_collator = self.data_collator or (lambda batch: batch)
        for pretrain_step in range(pretrain_steps):
            raw_batch, train_iter = self._get_next_batch(train_iter, self.dataloader_train)
            batch = data_collator(raw_batch)
            prepared, _ = self._preprocess_batch_input(
                state=self.model_state,
                batch=batch,
                is_train=True,
            )
            self.critic_state, metrics = jax.block_until_ready(
                self.sharded_critic_step_function(
                    self.critic_state,
                    prepared,
                    *self._critic_shared_fn_static_args,
                )
            )
            if (pretrain_step + 1) % 10 == 0 or pretrain_step + 1 == pretrain_steps:
                logger.info(
                    "Critic value pretraining: %d/%d, loss=%s.",
                    pretrain_step + 1,
                    pretrain_steps,
                    jax.device_get(metrics.loss),
                )
        self._critic_pretraining_complete = True

    def _save_state(
        self,
        state: EasyDeLState,
        save_directory: str | None = None,
        *args,
        merge_lora_before_save: bool = False,
        **kwargs,
    ) -> str:
        """Save a paired actor checkpoint plus a nested critic checkpoint."""
        if save_directory is None:
            step = self._get_current_step(state)
            intended_directory = self.arguments._get_save_directory_milestone(step=step, create=False)
        else:
            intended_directory = save_directory
        if jax.process_index() == 0:
            intended_path = ePath(intended_directory)
            intended_path.mkdir(parents=True, exist_ok=True)
            (intended_path / "online_rl_checkpoint.json").write_text(
                json.dumps(
                    {
                        "format": "easydel-online-actor-critic-v1",
                        "actor_step": int(jax.device_get(state.step)),
                        "critic_step": None,
                        "actor_algorithm": self.actor_algorithm,
                        "complete": False,
                    },
                    indent=2,
                )
            )
        directory = super()._save_state(
            state,
            save_directory,
            *args,
            merge_lora_before_save=merge_lora_before_save,
            **kwargs,
        )
        critic_directory = ePath(directory) / "critic"
        self.critic_state.save_state(
            save_directory=critic_directory,
            float_dtype=self.critic_state.model.param_dtype,
            save_optimizer=self.arguments.save_optimizer_state,
            merge_lora_before_save=False,
        )
        if jax.process_index() == 0:
            manifest = {
                "format": "easydel-online-actor-critic-v1",
                "actor_step": int(jax.device_get(state.step)),
                "critic_step": int(jax.device_get(self.critic_state.step)),
                "actor_algorithm": self.actor_algorithm,
                "complete": True,
            }
            (ePath(directory) / "online_rl_checkpoint.json").write_text(json.dumps(manifest, indent=2))
        return directory

    def _load_auxiliary_states(self, checkpoint_path: str) -> None:
        """Restore ``checkpoint/critic`` into the existing wrapper template.

        ``CausalLMWithValueHead`` deliberately cannot be reconstructed from a
        config alone, so the generic :meth:`EasyDeLState.load_state` path is
        not valid for critics. The saved SpectraX model tree is instead loaded
        into the already constructed critic graph and repartitioned using the
        template's trainable/frozen paths.

        Args:
            checkpoint_path: Paired online-RL checkpoint directory.
        """
        manifest_path = ePath(checkpoint_path) / "online_rl_checkpoint.json"
        if not manifest_path.exists():
            raise ValueError(
                f"Online-RL checkpoint {checkpoint_path} has no completion manifest; "
                "refusing to resume the actor without its paired critic."
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "easydel-online-actor-critic-v1" or manifest.get("complete") is not True:
            raise ValueError(f"Online-RL checkpoint {checkpoint_path} has an invalid or incomplete manifest.")
        if manifest.get("actor_algorithm") != self.actor_algorithm:
            raise ValueError(
                f"Online-RL checkpoint algorithm {manifest.get('actor_algorithm')!r} does not match "
                f"trainer algorithm {self.actor_algorithm!r}."
            )
        actor_metadata_path = ePath(checkpoint_path) / "metadata.json"
        if not actor_metadata_path.exists():
            raise ValueError(f"Online-RL checkpoint {checkpoint_path} has no actor metadata.")
        actor_metadata = json.loads(actor_metadata_path.read_text())
        actor_step = int(actor_metadata.get("step", -1))
        if actor_step != int(manifest["actor_step"]):
            raise ValueError(
                f"Actor checkpoint step {actor_step} does not match paired manifest step {manifest['actor_step']}."
            )

        critic_directory = ePath(checkpoint_path) / "critic"
        if not critic_directory.exists():
            raise ValueError(
                f"Online-RL checkpoint {checkpoint_path} has no critic subdirectory; refusing a partially paired resume."
            )

        template_state = self.critic_state
        template_model = template_state.model
        critic_tx, _ = build_critic_optimizer(
            self.arguments,
            int(self.arguments.critic_pretrain_steps)
            + int(self.arguments.max_training_steps or 0) * int(self.arguments.critic_updates_per_actor_update),
        )
        full_template = template_state.graphstate.merge(template_state.graphother, copy=False)
        raw_template = jax.eval_shape(lambda: full_template.raw())
        loaded_raw, _ = Checkpointer(
            base_path=str(critic_directory),
            save_interval=None,
            step_policies=[],
        ).load_pytree(
            mesh=template_model.mesh,
            path=str(critic_directory),
            prefix="model",
            discover_latest=False,
            template=raw_template,
            sharding_rules=template_model.resolve_shardings_regex(),
        )
        loaded_full = spx.State(loaded_raw)
        expected_paths = set(full_template.paths())
        loaded_paths = set(loaded_full.paths())
        if loaded_paths != expected_paths:
            missing = sorted(expected_paths - loaded_paths)
            unexpected = sorted(loaded_paths - expected_paths)
            raise ValueError(
                f"Critic checkpoint tree does not match its template; missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}."
            )

        trainable_paths = set(template_state.graphstate.paths())
        trainable_raw: dict[str, dict[str, tp.Any]] = {}
        frozen_raw: dict[str, dict[str, tp.Any]] = {}
        for collection, path, value in loaded_full.items():
            target = trainable_raw if (collection, path) in trainable_paths else frozen_raw
            target.setdefault(collection, {})[path] = value

        loaded = template_state.replace(
            graphstate=spx.State(trainable_raw),
            graphother=spx.State(frozen_raw),
            tx=critic_tx,
            opt_state=None,
        )
        metadata_path = critic_directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get("has_optimizer_state", False):
            loaded = loaded.load_optimizer(
                load_directory=critic_directory,
                tx_template=critic_tx,
                verbose=True,
            )
        else:
            loaded = loaded.replace(
                step=jnp.asarray(int(metadata.get("step", manifest["critic_step"])), dtype=jnp.int32)
            )
        critic_step = int(jax.device_get(loaded.step))
        if critic_step != int(manifest["critic_step"]):
            raise ValueError(
                f"Critic checkpoint step {critic_step} does not match paired manifest step {manifest['critic_step']}."
            )
        self.critic_state = loaded
        self._critic_pretraining_complete = critic_step >= int(self.arguments.critic_pretrain_steps)


__all__ = ("OnlineActorCriticTrainer", "RolloutProvider")
