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
"""Group Relative Policy Optimization (GRPO) trainer.

GRPO -- DeepSeek (2024) -- replaces the PPO critic with a group-
relative advantage: rewards are mean-centred (and optionally
standardised) within each prompt group of online rollouts.  This
trainer drives the online generation pipeline, calls the registered
reward functions / models, computes reference log-probabilities for
the KL penalty, and dispatches to the JIT-compiled GRPO step.
"""

from __future__ import annotations

import inspect
import json
import typing as tp
from functools import partial
from pathlib import Path

import jax
import numpy as np
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import NamedSharding
from spectrax import with_sharding_constraint
from transformers import AutoTokenizer

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import (  # pyright: ignore[reportPrivateLocalImportUsage]
    capture_time,
    get_logger,
)
from easydel.utils.traversals import deepcopy_model

from ..agentic_moshpit.environment import ToolEnvWrapper, create_tool_call_parser
from ..agentic_moshpit.tools import function_to_json, make_tool
from ..model_loading import disable_state_dropout, reject_string_model_id
from ..prompt_transforms import GRPOPreprocessTransform
from ..prompt_utils import apply_chat_template
from ..reward_protocol import RewardProtocol
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_configurations import MetricsType
from ..training_utils import (
    compile_trainer_step,
    extract_generation_model_kwargs,
    filter_kwargs_for_callable,
    normalize_generation_model_kwargs,
    repeat_prompt_aligned_model_kwargs,
    resolve_straight_through_emulator,
    sanitize_model_call_kwargs,
    slice_prompt_aligned_model_kwargs,
    strip_prompt_only_scoring_model_kwargs,
    validate_prompt_aligned_generation_model_kwargs,
)
from ._fn import get_per_token_logps, get_per_token_logps_and_topk, grpo_step
from .grpo_config import GRPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)
RewardFunc = EasyDeLBaseModule | EasyDeLState | RewardProtocol | tp.Callable[[list, list], list[float]]


class _ToolLike(tp.Protocol):
    """Protocol for executable tools used by GRPO environment feedback.

    A tool must expose a stable name, a chat-template schema, and an execution
    method that accepts the serialized arguments emitted by the model. This is
    intentionally structural so native tool objects, agentic-moshpit tools, and
    local adapters can all be consumed without inheritance.
    """

    @property
    def name(self) -> str:
        """Return the externally visible tool name.

        The name is used both when rendering tool schemas into prompts and when
        matching model-emitted tool calls back to an executable implementation.
        """
        ...

    @property
    def chat_schema(self) -> dict[str, object]:
        """Return a serializable function schema for chat-template rendering.

        The schema should describe the callable name, description, and argument
        shape in the format expected by the tokenizer or processor chat
        template.
        """
        ...

    def execute(self, arguments: str) -> str:
        """Execute the tool with serialized model-provided arguments.

        Implementations are expected to parse the argument string according to
        their own schema and return a text observation that can be appended to
        the dialogue for another generation turn.
        """
        ...


class _CallableToolAdapter:
    """Adapt a plain Python callable to the GRPO executable-tool protocol.

    The adapter derives a chat schema from the callable signature and maps
    JSON object arguments to keyword arguments. Non-object arguments are passed
    to the first positional parameter when one exists, matching the simple
    single-input tool style used by agentic rollouts.
    """

    def __init__(self, func: tp.Callable[..., object]):
        self._func = func

    @property
    def name(self) -> str:
        """Return the callable name exposed to chat-template tool schemas.

        The function ``__name__`` is preferred for stable prompt rendering; a
        class name fallback keeps callable instances usable.
        """
        return getattr(self._func, "__name__", type(self._func).__name__)

    @property
    def chat_schema(self) -> dict[str, object]:
        """Return an OpenAI-style function schema for the wrapped callable.

        The schema is generated from the Python signature so callables can be
        passed directly to GRPO without manually creating tool dictionaries.
        """
        return function_to_json(self._func)["function"]

    def execute(self, arguments: str) -> str:
        """Parse tool-call arguments and execute the wrapped callable.

        JSON objects are expanded as keyword arguments. Other payloads are
        treated as a single value for the callable's first parameter, and the
        result is stringified for use as a model-visible observation.
        """
        try:
            parsed = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            parsed = arguments
        if isinstance(parsed, dict):
            result = self._func(**parsed)
        else:
            parameters = inspect.signature(self._func).parameters
            first_param = next(iter(parameters), None)
            result = self._func(**{first_param: parsed}) if first_param is not None else self._func()
        return str(result)


def _clip_rewards_if_configured(rewards: jax.Array, arguments: GRPOConfig) -> jax.Array:
    """Apply symmetric reward clipping when the GRPO config requests it.

    ``reward_clip_range=None`` leaves rewards unchanged. Otherwise each reward
    is clipped into ``[-reward_clip_range, reward_clip_range]`` before grouped
    advantage normalization, matching TRL-style reward clipping semantics.
    """
    reward_clip_range = getattr(arguments, "reward_clip_range", None)
    if reward_clip_range is None:
        return rewards
    return jnp.clip(rewards, -reward_clip_range, reward_clip_range)


def _compute_rewards_and_advantages(
    *,
    rewards_per_func: jax.Array,
    reward_weights: jax.Array,
    generation_factor: int,
    scale_rewards: str,
    multi_objective_aggregation: str,
    arguments: GRPOConfig,
    group_reduction: str = "mean",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Aggregate reward-function outputs and compute grouped advantages.

    Returns the scalar reward per completion, normalized advantages, per-group
    reward means, and per-group reward standard deviations used by the GRPO
    loss and logging path.

    ``group_reduction`` selects how the per-group baseline subtracted from the
    rewards is reduced over the ``num_generations`` completions: ``"mean"``
    (the classic GRPO baseline) or ``"sum"``.  It applies to the ``group_mean``
    advantage estimator only; ``leave_one_out`` keeps its own RLOO baseline.
    """
    if multi_objective_aggregation == "sum_then_normalize":
        rewards = jnp.nansum(rewards_per_func * reward_weights[None, :], axis=1)
        rewards = _clip_rewards_if_configured(rewards, arguments)
        grouped_rewards = rewards.reshape(-1, generation_factor)
        if getattr(arguments, "advantage_estimator", "group_mean") == "leave_one_out":
            if generation_factor <= 1:
                raise ValueError("RLOO leave-one-out advantages require `num_generations > 1`.")
            grouped_baseline = (jnp.nansum(grouped_rewards, axis=-1, keepdims=True) - grouped_rewards) / (
                generation_factor - 1
            )
            advantages = (grouped_rewards - grouped_baseline).reshape(-1)
        else:
            if group_reduction == "sum":
                grouped_baseline = jnp.nansum(grouped_rewards, axis=-1)
            else:
                grouped_baseline = jnp.nanmean(grouped_rewards, axis=-1)
            advantages = rewards - grouped_baseline.repeat(generation_factor, axis=0)

        if scale_rewards in ("group", "none"):
            std_rewards = jnp.nanstd(grouped_rewards, axis=-1)
            std_rewards = std_rewards.repeat(generation_factor, axis=0)
        elif scale_rewards == "batch":
            std_rewards = jnp.nanstd(rewards)
            std_rewards = jnp.broadcast_to(std_rewards, advantages.shape)
        else:
            raise ValueError(f"Invalid value for scale_rewards: {scale_rewards}. Must be 'batch', 'group', or 'none'.")
        is_std_zero = jnp.isclose(std_rewards, 0.0)
        if scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        return rewards, jnp.nan_to_num(advantages), std_rewards, is_std_zero

    if multi_objective_aggregation == "normalize_then_sum":
        grouped = rewards_per_func.reshape(-1, generation_factor, rewards_per_func.shape[-1])
        mean_per_reward = jnp.nanmean(grouped, axis=1, keepdims=True)
        std_per_reward = jnp.nanstd(grouped, axis=1, keepdims=True)
        normalized_per_reward = (grouped - mean_per_reward) / (std_per_reward + 1e-4)
        normalized_per_reward = normalized_per_reward.reshape(-1, rewards_per_func.shape[-1])
        rewards = jnp.nansum(normalized_per_reward * reward_weights[None, :], axis=1)
        rewards = _clip_rewards_if_configured(rewards, arguments)
        std_rewards = jnp.broadcast_to(jnp.nanstd(rewards), rewards.shape)
        advantages = (rewards - jnp.nanmean(rewards)) / (std_rewards + 1e-4)
        is_std_zero = jnp.isclose(std_rewards, 0.0)
        return rewards, jnp.nan_to_num(advantages), std_rewards, is_std_zero

    raise ValueError(
        "Invalid multi_objective_aggregation: "
        f"{multi_objective_aggregation}. Must be 'sum_then_normalize' or 'normalize_then_sum'."
    )


def _fileaf(x):
    """``is_leaf`` predicate that stops at JAX arrays.

    Args:
        x: Pytree leaf candidate.

    Returns:
        ``True`` when ``x`` is a JAX array.
    """
    return isinstance(x, jax.Array)


def delete_tree(pytree):
    """Eagerly free every JAX-array leaf inside ``pytree``.

    Used after a generation pass to release on-device buffers before
    the next compiled step allocates new ones.

    Args:
        pytree: A pytree mixing JAX arrays with Python objects.

    Returns:
        A pytree of ``None`` leaves, returned for caller symmetry; the
        side effect is the in-place buffer deletion.
    """
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

    supports_sequence_packing: tp.ClassVar[bool] = False  # RL/online or paired-preference: warn-and-ignore packing

    arguments: GRPOConfig  # type hinting
    reward_processing_classes: list | None

    def __init__(
        self,
        arguments: GRPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        data_tokenize_fn: tp.Callable | None = None,
        tools: list[dict | str | tp.Callable[..., object] | _ToolLike] | None = None,
        environment_factory: tp.Callable[[], object] | None = None,
    ):
        """Initialize the GRPO trainer.

        Resolves the policy state, deep-copies it into a frozen
        reference state, sets up reward modules / callables (lifting
        :class:`EasyDeLBaseModule` rewards into compiled
        :class:`EasyDeLState` apply functions when needed), and forwards
        construction to :class:`Trainer`.

        Args:
            arguments: GRPO-specific training configuration.
            model: Policy module or state.
            reward_funcs: Single reward callable / module / state, or a
                list of them.  Reward modules are converted to
                :class:`EasyDeLState` automatically.
            train_dataset: Optional dataset of prompts.
            eval_dataset: Optional evaluation dataset.
            processing_class: Tokenizer/processor; defaults to
                ``AutoTokenizer.from_pretrained(model.config._name_or_path)``.
            reward_processing_classes: Per-reward tokenizer overrides;
                falls back to ``AutoTokenizer`` based on each reward
                model's ``config._name_or_path``.
            data_tokenize_fn: Optional custom tokenization callable.

        Raises:
            ValueError: If ``arguments`` is ``None`` or the reward-
                weight count does not match the number of rewards.
            TypeError: If ``arguments`` is not a :class:`GRPOConfig`.
        """
        if arguments is None:
            raise ValueError(
                "You Have to pass `arguments` that will be used for training, but you have passed `arguments=None`"
            )
        if not isinstance(arguments, GRPOConfig):
            raise TypeError(f"arguments type must be `GRPOConfig` but got {type(arguments)}")
        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
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
        self.multi_objective_aggregation = arguments.multi_objective_aggregation
        if isinstance(self.multi_objective_aggregation, str):
            self.multi_objective_aggregation = self.multi_objective_aggregation.lower()
        self.top_entropy_quantile = arguments.top_entropy_quantile
        self.ref_logps_chunk_size = arguments.ref_logps_chunk_size
        self.num_iterations = arguments.num_iterations
        self.steps_per_generation = int(arguments.steps_per_generation or 1)
        self._buffered_grpo_batch: tuple[dict[str, jax.Array], dict[str, float | int | str]] | None = None
        self._buffered_grpo_remaining = 0
        self._completion_log_dir: Path | None = None
        self._completion_commit_scheduler: object | None = None

        if isinstance(model, str):
            reject_string_model_id(model, role="policy model")
        if not isinstance(model, EasyDeLState):
            model = model.to_state(trainable_selector=arguments.trainable_selector)

        if arguments.beta != 0.0:
            self.ref_state = deepcopy_model(model=model)
            if arguments.disable_dropout:
                model, self.ref_state = self._disable_state_dropout(model, self.ref_state)
        else:
            # beta == 0 -> no KL term. Skip the reference model entirely: no
            # deep-copy (saves ~13.6 GB/chip for a 27B) and no reference forward
            # (which would otherwise alias the donated policy buffers under the
            # async rollout overlap and crash with "Buffer has been deleted").
            if arguments.disable_dropout:
                (model,) = self._disable_state_dropout(model)
            self.ref_state = None

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.model.config._name_or_path,
                padding_side="left",
            )
        self.processing_class = processing_class
        self.environment_factory = environment_factory or arguments.environment_factory
        self._environment_tools = self._normalize_environment_tools(tools or arguments.tools)
        self._environment_tool_parser = (
            create_tool_call_parser(arguments.tool_caller, tokenizer=processing_class)
            if self._environment_tools
            else None
        )
        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_token_id is None and hasattr(self.processing_class, "tokenizer"):
            pad_token_id = getattr(self.processing_class.tokenizer, "pad_token_id", None)
        self.padding_value = 0 if pad_token_id is None else int(pad_token_id)
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

        empty_sharding = replicated_named_sharding(model.model.mesh)
        if not isinstance(reward_processing_classes, list):
            raise TypeError(f"reward_processing_classes must be a list, got {type(reward_processing_classes)}")

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=False)
        ):
            reward_func = self._resolve_reward_func_model(reward_func)
            if isinstance(reward_func, EasyDeLBaseModule | EasyDeLState):
                if isinstance(reward_func, EasyDeLBaseModule):
                    reward_func = reward_func.to_state(trainable_selector=arguments.trainable_selector)
                    sharding = reward_func.shardings

                    def apply_fn(gd, gs, gt, batch):
                        """Sharded reward-model forward used as the state's ``apply_fn``.

                        Args:
                            gd: Reward-module graphdef.
                            gs: Trainable graphstate.
                            gt: Frozen graphother (stop-gradient applied).
                            batch: Tokenized input batch.

                        Returns:
                            The reward module's output (typically a
                            ``logits`` field with the per-example score).
                        """
                        gt = jax.tree_util.tree_map(
                            lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                            gt,
                        )
                        module = spx.bind(gd, gs.merge(gt, copy=False))
                        batch = with_sharding_constraint(
                            arr=batch,
                            sharding=self.arguments.step_partition_spec,
                            mesh=module.mesh,
                            ignore_mpmd=True,
                        )
                        call_kwargs = filter_kwargs_for_callable(getattr(module, "forward", module), batch)
                        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
                        return module(**call_kwargs)

                    apply_fn = compile_trainer_step(
                        apply_fn,
                        mesh=model.model.mesh,
                        static_argnums=(0,),
                        in_shardings=(sharding.graphstate, sharding.graphother, empty_sharding),
                        out_shardings=empty_sharding,
                    )
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
                if arguments.disable_dropout:
                    (reward_func,) = self._disable_state_dropout(reward_func)
                reward_processing_classes[i] = reward_processing_class
                reward_funcs[i] = reward_func

        if arguments.reward_weights is not None and len(arguments.reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Expected {len(reward_funcs)} reward weights, but got {len(arguments.reward_weights)} instead."
            )

        # Weights come from each reward's own `weight` (RewardProtocol.weight, default
        # 1.0; plain callables/models -> 1.0). An explicit `reward_weights` in the
        # trainer config overrides them.
        self.reward_weights = jnp.asarray(
            arguments.reward_weights
            if arguments.reward_weights is not None
            else [float(getattr(func, "weight", 1.0)) for func in reward_funcs],
            dtype="f4",
        )
        self.reward_func_names = [getattr(func, "__name__", None) or func.__class__.__name__ for func in reward_funcs]
        self._group_reduction = self._resolve_group_reduction(reward_funcs)

        self.num_generations = arguments.num_generations
        self.eval_num_generations = arguments.num_generations_eval or arguments.num_generations
        self.reward_processing_classes = reward_processing_classes
        self.reward_funcs = reward_funcs
        self.arguments = arguments
        self._initialize_conversational_flags(train_dataset, eval_dataset)

        self.data_tokenize_fn = data_tokenize_fn

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )
        self._setup_completion_hub_logging()

    @staticmethod
    def _resolve_group_reduction(reward_funcs: list[RewardFunc]) -> str:
        """Resolve the group-baseline reduction used by advantage estimation.

        Reads the ``reduction`` attribute of any :class:`RewardProtocol` in
        ``reward_funcs``.  When at least one is present its reduction is used
        (all protocols must agree); otherwise the classic ``"mean"`` baseline is
        kept so plain reward callables / models are unaffected.

        Args:
            reward_funcs: The normalized list of reward functions/models.

        Returns:
            ``"sum"`` or ``"mean"``.

        Raises:
            ValueError: If two ``RewardProtocol`` instances request conflicting
                reductions.
        """
        reductions = {func.reduction for func in reward_funcs if isinstance(func, RewardProtocol)}
        if len(reductions) > 1:
            raise ValueError(
                f"Conflicting RewardProtocol.reduction values {sorted(reductions)}; "
                "all RewardProtocols passed together must use the same reduction."
            )
        return reductions.pop() if reductions else "mean"

    @staticmethod
    def _resolve_reward_func_model(reward_func: RewardFunc) -> RewardFunc:
        """Reject string reward identifiers and return a concrete reward object.

        EasyDeL trainer APIs expect initialized modules, states, or callables.
        A string would imply implicit model loading, so this helper fails early
        with a targeted message before reward setup mutates tokenizer/model
        padding state.
        """
        if isinstance(reward_func, str):
            reject_string_model_id(reward_func, role="reward model")
        return reward_func

    @staticmethod
    def _disable_state_dropout(*states: EasyDeLState | None) -> tuple[EasyDeLState | None, ...]:
        """Put GRPO policy, reference, or reward states into eval mode.

        ``None`` entries are preserved so callers can pass optional state slots
        without branching. Concrete states are routed through the shared helper
        that disables dropout on the underlying EasyDeL module graph.
        """
        return tuple(disable_state_dropout(state) for state in states)

    def _generation_reuse_span(self) -> int:
        """Return how many optimizer steps may consume one generated batch.

        GRPO can reuse a rollout across ``steps_per_generation`` and
        ``num_iterations`` update passes. The product is clamped to at least one
        so downstream cache accounting never stores a zero or negative reuse
        budget.
        """
        steps_per_generation = int(getattr(self, "steps_per_generation", 1) or 1)
        num_iterations = int(getattr(self, "num_iterations", 1) or 1)
        return max(1, steps_per_generation * num_iterations)

    def _take_buffered_grpo_batch(self) -> tuple[dict[str, jax.Array], dict[str, float | int | str]] | None:
        """Return a cached generated/scored batch when GRPO reuse is active.

        The cached batch is shallow-copied before returning so callers can add
        step-local fields without mutating the stored copy. Metrics are copied
        and annotated with ``generation_reused`` plus the remaining reuse count
        for logging and debugging.
        """
        buffered_batch = getattr(self, "_buffered_grpo_batch", None)
        buffered_remaining = int(getattr(self, "_buffered_grpo_remaining", 0))
        if buffered_batch is None or buffered_remaining <= 0:
            return None
        buffered_remaining -= 1
        self._buffered_grpo_remaining = buffered_remaining
        model_batch, metrics = buffered_batch
        reused_metrics = dict(metrics)
        reused_metrics["generation_reused"] = 1
        reused_metrics["generation_reuse_remaining"] = buffered_remaining
        return dict(model_batch), reused_metrics

    def _store_buffered_grpo_batch(
        self,
        model_batch: dict[str, jax.Array],
        metrics: dict[str, float | int | str],
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Cache a newly generated/scored GRPO batch for configured reuse.

        The method records reuse metrics on every fresh rollout. When the reuse
        span is one, any previous cache is cleared and the batch is returned as
        a normal one-shot update. When reuse is enabled, shallow copies of the
        model batch and metrics are stored for subsequent optimizer steps while
        the current fresh batch is returned unchanged.
        """
        reuse_span = self._generation_reuse_span()
        metrics = dict(metrics)
        metrics["generation_reused"] = 0
        metrics["generation_reuse_span"] = reuse_span
        if reuse_span <= 1:
            self._buffered_grpo_batch = None
            self._buffered_grpo_remaining = 0
            metrics["generation_reuse_remaining"] = 0
            return model_batch, metrics
        self._buffered_grpo_batch = (dict(model_batch), dict(metrics))
        self._buffered_grpo_remaining = reuse_span - 1
        metrics["generation_reuse_remaining"] = self._buffered_grpo_remaining
        return model_batch, metrics

    @staticmethod
    def _is_tool_like(tool: object) -> tp.TypeGuard[_ToolLike]:
        """Return whether ``tool`` implements the executable tool protocol.

        The check is deliberately structural because user-supplied tools do not
        have to inherit from EasyDeL classes. A valid tool needs a name, a chat
        schema, and a callable ``execute`` method.
        """
        return (
            hasattr(tool, "name")
            and hasattr(tool, "execute")
            and callable(tool.execute)
            and hasattr(tool, "chat_schema")
        )

    @classmethod
    def _normalize_environment_tools(
        cls,
        tools: list[dict | str | tp.Callable[..., object] | _ToolLike] | None,
    ) -> list[_ToolLike]:
        """Return executable tools for environment feedback.

        Dict-only tool schemas are useful for chat-template rendering but
        do not carry an implementation, so they are intentionally ignored
        for execution.
        """
        if not tools:
            return []
        normalized: list[_ToolLike] = []
        for tool in tools:
            if isinstance(tool, dict):
                continue
            if isinstance(tool, str):
                normalized.append(make_tool(tool))
                continue
            if cls._is_tool_like(tool):
                normalized.append(tool)
                continue
            if callable(tool):
                normalized.append(_CallableToolAdapter(tool))
        return normalized

    def _prompt_chat_template_tools(self) -> list[dict[str, object]] | list[dict | tp.Callable] | None:
        """Return tool schemas/callables passed into prompt chat templating.

        Explicit config tools take precedence because they may include schema
        dictionaries that are prompt-only. If only executable environment tools
        are present, their ``chat_schema`` values are exposed to the tokenizer.
        """
        if self.arguments.tools is not None:
            return self.arguments.tools
        if self._environment_tools:
            return [tool.chat_schema for tool in self._environment_tools]
        return None

    def _get_preprocess_transform(self) -> GRPOPreprocessTransform | None:
        """Return the GRPO preprocessing transform for sharded sources.

        Skips transform construction when the underlying dataset is
        already tokenised (i.e. carries ``input_ids``); otherwise builds
        a :class:`GRPOPreprocessTransform` wired with this trainer's
        tokenizer, ``max_prompt_length``, ``tools`` registry, and
        chat-template policy.

        Returns:
            A :class:`GRPOPreprocessTransform` instance, or ``None`` if
            the dataset is already tokenised.
        """

        if self._is_pretokenized():
            return None
        return GRPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            tools=self._prompt_chat_template_tools(),
            skip_apply_chat_template=self.arguments.skip_apply_chat_template,
            pad_to_multiple_of=self.arguments.pad_to_multiple_of,
            chat_template_kwargs=self.arguments.chat_template_kwargs,
        )

    def _is_pretokenized(self) -> bool:
        """Return ``True`` when the training source already exposes ``input_ids``.

        Peeks at the first record of the first training shard to decide
        whether GRPO preprocessing (tokenisation, chat-template
        application, prompt truncation) is required or can be skipped.

        Returns:
            ``True`` if the sample carries an ``input_ids`` field;
            ``False`` otherwise (including when the source is empty or
            missing).
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
        """Build the Grain-side collator for GRPO prompt batches.

        Args:
            max_sequence_length: Unused for GRPO collation (kept for
                interface parity with other trainers).
            truncation_mode: Unused for GRPO collation (kept for
                interface parity).

        Returns:
            A :class:`GRPODataCollatorGrain` instance configured with
            the trainer's ``max_prompt_length`` and padding value.
        """
        from ..utils import GRPODataCollatorGrain

        return GRPODataCollatorGrain(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
            pad_to_multiple_of=self.arguments.pad_to_multiple_of,
        )

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Build the TFDS-side collator for GRPO prompt batches.

        Args:
            max_sequence_length: Unused for GRPO collation (kept for
                interface parity with other trainers).
            truncation_mode: Unused for GRPO collation (kept for
                interface parity).

        Returns:
            A :class:`GRPODataCollatorTFDS` instance configured with
            the trainer's ``max_prompt_length`` and padding value.
        """
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
            pad_to_multiple_of=self.arguments.pad_to_multiple_of,
        )

    def _run_environment_feedback(
        self,
        *,
        action_texts: list[str],
        tool_calls: list[object | None],
    ) -> dict[str, list[object]] | None:
        """Step opt-in environments with generated completions.

        This is intentionally scoped to explicit ``environment_factory``
        users. It does not affect normal GRPO and keeps dynamic access
        limited to user-supplied environment objects.
        """
        if self.environment_factory is None:
            return None

        observations: list[object] = []
        rewards: list[object] = []
        terminated: list[object] = []
        truncated: list[object] = []
        infos: list[object] = []

        for idx, action_text in enumerate(action_texts):
            env = self.environment_factory()
            if self._environment_tools and not isinstance(env, ToolEnvWrapper):
                env = ToolEnvWrapper(
                    env=env,
                    tools=self._environment_tools,
                    tool_call_parser=self._environment_tool_parser,
                    max_tool_calls_per_step=self.arguments.max_tool_calls_per_step,
                )

            try:
                reset = getattr(env, "reset", None)
                if callable(reset):
                    reset()

                step_with_tool_calls = getattr(env, "step_with_tool_calls", None)
                if callable(step_with_tool_calls):
                    step_result = step_with_tool_calls(
                        action_text,
                        tool_calls=tool_calls[idx] if idx < len(tool_calls) else None,
                    )
                else:
                    step = getattr(env, "step", None)
                    if not callable(step):
                        raise TypeError(
                            "GRPO environment_factory must create environments with step(action) "
                            "or step_with_tool_calls(action, tool_calls=...)."
                        )
                    step_result = step(action_text)

                observations.append(getattr(step_result, "observation", ""))
                rewards.append(float(getattr(step_result, "reward", 0.0)))
                terminated.append(bool(getattr(step_result, "terminated", False)))
                truncated.append(bool(getattr(step_result, "truncated", False)))
                infos.append(getattr(step_result, "info", {}))
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

        return {
            "environment_observations": observations,
            "environment_rewards": rewards,
            "environment_terminated": terminated,
            "environment_truncated": truncated,
            "environment_infos": infos,
        }

    def _tool_call_loop_enabled(self) -> bool:
        """Return whether generated tool calls should be executed and appended.

        Tool-call looping requires at least one executable environment tool and
        a non-zero iteration budget. A zero budget disables the loop while still
        allowing tool schemas to be rendered into the prompt.
        """
        return bool(self._environment_tools) and self.arguments.max_tool_calling_iterations != 0

    def _encode_tool_loop_fragment(self, text: str) -> list[int]:
        """Encode one tool-observation fragment with the trainer processor.

        Tokenizers exposing ``encode`` are used directly; processor-style
        objects are called and their ``input_ids`` are normalized to Python
        integers for concatenation with generated token ids.
        """
        processor = self.processing_class
        if hasattr(processor, "encode"):
            return [int(token) for token in processor.encode(text, add_special_tokens=False)]
        tokenized = processor(text, add_special_tokens=False)
        return [int(token) for token in tokenized["input_ids"]]

    @staticmethod
    def _completion_prompt_to_text(prompt: str | list[dict[str, object]] | object) -> str:
        """Normalize string or chat-message prompts to plain text.

        Chat message lists are flattened as ``role: content`` lines so tool-loop
        observations can be logged and passed through fallback text-only paths.
        Non-chat objects are converted with ``str`` as a last resort.
        """
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            parts = []
            for message in prompt:
                if isinstance(message, dict):
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(message))
            return "\n".join(parts)
        return str(prompt)

    def _execute_tool_calls_as_observation(self, tool_calls: object | None) -> str | None:
        """Execute parsed tool calls and render textual observations for the model.

        Unknown tool names are converted into explicit error observations rather
        than raising, which lets the model continue the interaction. Execution
        is capped by ``max_tool_calls_per_step`` to keep a single completion
        from monopolizing the rollout loop.
        """
        normalized_calls = self._normalize_tool_call_payloads(tool_calls)
        if not normalized_calls:
            return None

        tool_by_name = {tool.name: tool for tool in self._environment_tools}
        observations: list[str] = []
        for call in normalized_calls[: self.arguments.max_tool_calls_per_step]:
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str):
                continue
            tool = tool_by_name.get(name)
            if tool is None:
                observations.append(f"[{name}]: {{'error': 'Tool not found.'}}")
                continue
            arguments = function.get("arguments", {})
            try:
                argument_text = json.dumps(arguments)
            except (TypeError, ValueError):
                argument_text = str(arguments)
            try:
                result = tool.execute(argument_text)
            except Exception as exc:
                result = {"error": str(exc)}
            observations.append(f"[{name}]: {result}")
        return "\n".join(observations) if observations else None

    def _generation_config_overrides_for_phase(self, is_train: bool) -> dict[str, int] | None:
        """Return generation overrides for train/eval GRPO rollout count."""
        if is_train:
            num_generations = getattr(self.arguments, "num_generations", None)
            if num_generations is None:
                return None
            return {"num_return_sequences": int(num_generations)}
        num_generations_eval = getattr(self.arguments, "num_generations_eval", None)
        if num_generations_eval is None:
            return None
        return {"num_return_sequences": int(num_generations_eval)}

    def _select_completion_log_rows(
        self,
        *,
        prompts: tp.Sequence[object],
        completions: tp.Sequence[object],
        completion_lengths: jax.Array | np.ndarray | tp.Sequence[float] | None,
    ) -> list[dict[str, object]]:
        """Select prompt/completion rows for TRL-style completion logging."""
        args = self.arguments
        length_rows = None
        if completion_lengths is not None:
            length_rows = np.asarray(jax.device_get(completion_lengths)).reshape(-1).tolist()

        rows: list[dict[str, object]] = []
        seen_prompts: set[str] = set()
        limit = getattr(args, "num_completions_to_print", None)
        log_unique_prompts = bool(getattr(args, "log_unique_prompts", False))
        for idx, (prompt, completion) in enumerate(zip(prompts, completions, strict=False)):
            prompt_text = self._wandb_stringify_generation_value(prompt) or "<prompt>"
            if log_unique_prompts:
                if prompt_text in seen_prompts:
                    continue
                seen_prompts.add(prompt_text)
            completion_length = None if length_rows is None or idx >= len(length_rows) else float(length_rows[idx])
            rows.append(
                {
                    "sample_idx": idx,
                    "prompt": prompt_text,
                    "completion": self._wandb_stringify_generation_value(completion) or "<completion>",
                    "completion_length": completion_length,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
        return rows

    def _maybe_log_grpo_completions(
        self,
        *,
        prompts: tp.Sequence[object],
        completions: tp.Sequence[object],
        completion_lengths: jax.Array | np.ndarray | tp.Sequence[float] | None,
        step: int | None = None,
    ) -> list[dict[str, object]]:
        """Log sampled GRPO completions when ``GRPOConfig.log_completions`` is enabled."""
        if not bool(getattr(self.arguments, "log_completions", False)):
            return []
        rows = self._select_completion_log_rows(
            prompts=prompts,
            completions=completions,
            completion_lengths=completion_lengths,
        )
        if rows:
            logger.info("GRPO completion samples: %s", json.dumps(rows, ensure_ascii=False, default=str))
            self._write_completion_log_rows(rows, step=step)
        return rows

    def _setup_completion_hub_logging(self) -> None:
        """Prepare local completion logs and optional Hub background uploads."""
        if not bool(getattr(self.arguments, "log_completions", False)):
            return
        if int(jax.process_index()) != 0:
            return

        save_root = self.arguments._get_save_directory(create=True)
        if save_root is None:
            return
        completion_log_dir = Path(str(save_root)) / "completions"
        completion_log_dir.mkdir(parents=True, exist_ok=True)
        self._completion_log_dir = completion_log_dir

        repo_id = self.arguments.log_completions_hub_repo
        if repo_id is None:
            return

        from huggingface_hub import CommitScheduler, create_repo

        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        self._completion_commit_scheduler = CommitScheduler(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=completion_log_dir,
            every=2,
            allow_patterns=["*.parquet"],
        )

    def _write_completion_log_rows(self, rows: list[dict[str, object]], *, step: int | None) -> Path | None:
        """Write completion rows as parquet and trigger the optional Hub scheduler."""
        if int(jax.process_index()) != 0 or not rows:
            return None
        completion_log_dir = getattr(self, "_completion_log_dir", None)
        if completion_log_dir is None:
            self._setup_completion_hub_logging()
            completion_log_dir = getattr(self, "_completion_log_dir", None)
        if completion_log_dir is None:
            return None

        import pandas as pd

        step_value = -1 if step is None else int(step)
        enriched_rows = [
            {
                "step": step_value,
                "process_index": int(jax.process_index()),
                **row,
            }
            for row in rows
        ]
        path = Path(completion_log_dir) / f"step-{step_value:08d}-process-{int(jax.process_index()):05d}.parquet"
        pd.DataFrame(enriched_rows).to_parquet(path, index=False)

        scheduler = getattr(self, "_completion_commit_scheduler", None)
        trigger = getattr(scheduler, "trigger", None)
        if callable(trigger):
            trigger()
        return path

    def _maybe_run_tool_call_generation_loop(
        self,
        *,
        state: EasyDeLState,
        results,
        completion_mask: jax.Array,
    ):
        """Execute GRPO tools and regenerate until calls stop or budget is exhausted."""
        if not self._tool_call_loop_enabled():
            return results, completion_mask

        raw_texts = self._coerce_generation_texts(results.raw_text, fallback=results.text)
        visible_texts = self._coerce_generation_texts(results.text, fallback=raw_texts)
        target_len = int(results.completion_ids.shape[0])
        tool_call_records = self._coerce_generation_metadata_list(results.tool_calls, target_len=target_len)
        if not any(tool_call_records):
            return results, completion_mask

        host_completion_ids = np.asarray(jax.device_get(results.completion_ids), dtype=np.int32)
        host_completion_mask = np.asarray(jax.device_get(completion_mask), dtype=np.int32)
        max_completion_length = int(host_completion_ids.shape[1])
        pad_token_id = int(self._pad_token_id or 0)

        completion_rows: list[list[int]] = []
        completion_mask_rows: list[list[int]] = []
        for ids, mask in zip(host_completion_ids, host_completion_mask, strict=True):
            active = [int(token) for token, keep in zip(ids.tolist(), mask.tolist(), strict=True) if keep]
            completion_rows.append(active)
            completion_mask_rows.append([1] * len(active))

        if len(raw_texts) < target_len:
            raw_texts.extend([""] * (target_len - len(raw_texts)))
        if len(visible_texts) < target_len:
            visible_texts.extend(raw_texts[len(visible_texts) : target_len])
        raw_texts = raw_texts[:target_len]
        visible_texts = visible_texts[:target_len]

        completion_prompts = results.completion_prompts or []
        if len(completion_prompts) < target_len:
            completion_prompts = [*completion_prompts, *([None] * (target_len - len(completion_prompts)))]

        active_indices = [idx for idx, calls in enumerate(tool_call_records) if calls]
        max_iterations = self.arguments.max_tool_calling_iterations
        if max_iterations is None:
            max_iterations = max_completion_length

        iteration = 0
        while active_indices and iteration < max_iterations:
            loop_prompts: list[str] = []
            loop_indices: list[int] = []
            for idx in active_indices:
                observation = self._execute_tool_calls_as_observation(tool_call_records[idx])
                if not observation:
                    continue

                observation_text = f"\n{observation}\n"
                observation_tokens = self._encode_tool_loop_fragment(observation_text)
                remaining = max_completion_length - len(completion_rows[idx])
                if remaining <= 0:
                    continue
                observation_tokens = observation_tokens[:remaining]
                completion_rows[idx].extend(observation_tokens)
                completion_mask_rows[idx].extend([0] * len(observation_tokens))
                raw_texts[idx] = f"{raw_texts[idx]}{observation_text}"
                visible_texts[idx] = f"{visible_texts[idx]}{observation_text}"

                remaining = max_completion_length - len(completion_rows[idx])
                if remaining <= 0:
                    continue
                prompt_text = self._completion_prompt_to_text(completion_prompts[idx])
                loop_prompts.append(f"{prompt_text}{raw_texts[idx]}")
                loop_indices.append(idx)

            if not loop_prompts:
                break

            max_new_tokens = max(max_completion_length - len(completion_rows[idx]) for idx in loop_indices)
            followup_results = self.generate_unified(
                prompts=loop_prompts,
                state=state,
                apply_chat_template=False,
                shard_inputs=True,
                all_gather=False,
                release_runtime_after_generation=False,
                config_overrides={
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": 1,
                },
            )
            followup_mask = np.asarray(jax.device_get(followup_results.completion_mask), dtype=np.int32)
            followup_ids = np.asarray(jax.device_get(followup_results.completion_ids), dtype=np.int32)
            followup_raw = self._coerce_generation_texts(followup_results.raw_text, fallback=followup_results.text)
            followup_visible = self._coerce_generation_texts(followup_results.text, fallback=followup_raw)
            followup_tool_calls = self._coerce_generation_metadata_list(
                followup_results.tool_calls,
                target_len=len(loop_indices),
            )

            next_active: list[int] = []
            for row, idx in enumerate(loop_indices):
                remaining = max_completion_length - len(completion_rows[idx])
                if remaining <= 0:
                    continue
                active_tokens = [
                    int(token)
                    for token, keep in zip(followup_ids[row].tolist(), followup_mask[row].tolist(), strict=True)
                    if keep
                ][:remaining]
                completion_rows[idx].extend(active_tokens)
                completion_mask_rows[idx].extend([1] * len(active_tokens))
                if row < len(followup_raw):
                    raw_texts[idx] = f"{raw_texts[idx]}{followup_raw[row]}"
                if row < len(followup_visible):
                    visible_texts[idx] = f"{visible_texts[idx]}{followup_visible[row]}"
                tool_call_records[idx] = followup_tool_calls[row]
                if followup_tool_calls[row] and len(completion_rows[idx]) < max_completion_length:
                    next_active.append(idx)

            active_indices = next_active
            iteration += 1

        padded_ids = []
        padded_masks = []
        for ids, mask in zip(completion_rows, completion_mask_rows, strict=True):
            ids = ids[:max_completion_length]
            mask = mask[:max_completion_length]
            pad_len = max_completion_length - len(ids)
            padded_ids.append([*ids, *([pad_token_id] * pad_len)])
            padded_masks.append([*mask, *([0] * pad_len)])

        completion_ids = jnp.asarray(np.asarray(padded_ids, dtype=np.int32))
        completion_mask = jnp.asarray(np.asarray(padded_masks, dtype=np.int32))
        generation_factor = completion_ids.shape[0] // max(results.prompt_ids.shape[0], 1)
        generation_factor = max(generation_factor, 1)
        prompt_rows = results.prompt_ids.repeat(generation_factor, 0)
        sequences = jnp.concatenate([prompt_rows, completion_ids], axis=-1)
        return (
            results._replace(
                sequences=sequences,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                text=visible_texts,
                raw_text=raw_texts,
                tool_calls=tool_call_records,
            ),
            completion_mask,
        )

    @property
    def step_sharding(self):
        """Return the :class:`NamedSharding` used for per-step batch tensors."""
        return NamedSharding(
            mesh=self.model.mesh,
            spec=self.arguments.step_partition_spec,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Build the JIT-compiled GRPO training/evaluation step functions.

        Resolves the optional QAT straight-through emulator, wires up
        the captured static-arg tuples for both training and
        evaluation passes (``num_generations``, ``beta``, loss type,
        clip bounds, importance-sampling level, chunk sizes, ...),
        compiles :func:`grpo_step` once for each mode under the
        active MPMD pipeline schedule, and registers the sharded
        reference-model forward callable used inside
        :meth:`_preprocess_batch_input`.

        Returns:
            ``TrainerConfigureFunctionOutput`` with the sharded
            training / evaluation step callables, the model mesh, and
            the streaming checkpoint manager.
        """
        mesh = self.model.mesh

        empty_sharding = replicated_named_sharding(mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            quantization_bits=self.arguments.quantization_bits,
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
            self.arguments.completion_chunk_size,
            self.arguments.max_loss_completion_tokens,
            self.arguments.logprob_vocab_chunk_size,
            self.arguments.sapo_temperature_pos,
            self.arguments.sapo_temperature_neg,
            self.arguments.vespo_k_pos,
            self.arguments.vespo_lambda_pos,
            self.arguments.vespo_k_neg,
            self.arguments.vespo_lambda_neg,
            self.arguments.off_policy_mask_threshold,
            self.arguments.use_bias_correction_kl,
            straight_through_emulator,
            getattr(self.arguments, "divergence_type", None) if self.loss_type == "dppo" else None,
            getattr(self.arguments, "clip_ratio_c", 20.0),
        )

        static_argnames = tuple(range(2, 29))

        self._runtime_trace("train.compile_wrapper.begin")
        sharded_training_step_function = compile_trainer_step(
            grpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        self._runtime_trace("train.compile_wrapper.end")

        self._eval_shared_fn_static_args = (
            self.eval_num_generations,
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
            self.arguments.logprob_vocab_chunk_size,
            self.arguments.sapo_temperature_pos,
            self.arguments.sapo_temperature_neg,
            self.arguments.vespo_k_pos,
            self.arguments.vespo_lambda_pos,
            self.arguments.vespo_k_neg,
            self.arguments.vespo_lambda_neg,
            self.arguments.off_policy_mask_threshold,
            self.arguments.use_bias_correction_kl,
            straight_through_emulator,
            getattr(self.arguments, "divergence_type", None) if self.loss_type == "dppo" else None,
            getattr(self.arguments, "clip_ratio_c", 20.0),
        )

        self._runtime_trace("eval.compile_wrapper.begin")
        sharded_evaluation_step_function = compile_trainer_step(
            grpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        self._runtime_trace("eval.compile_wrapper.end")

        def _compute_refmodel_logps(graphtree, graphother, ids, mask, model_kwargs=None, graphdef=None):
            """Sharded reference-model per-token log-prob forward.

            Stops gradients through the reference parameters, applies
            the trainer's step partition spec, and dispatches to
            :func:`get_per_token_logps` with the configured vocab chunk
            size.

            Args:
                graphtree: Reference trainable graphstate.
                graphother: Reference frozen graphother.
                ids: Token id array ``[batch, seq_len]``.
                mask: Attention mask ``[batch, seq_len]``.
                model_kwargs: Optional dict of additional model kwargs
                    (forwarded after normalization).
                graphdef: Reference graphdef captured via partial.

            Returns:
                ``[batch, seq_len]`` reference log-probabilities.
            """
            graphother = jax.tree_util.tree_map(
                lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                graphother,
            )
            apply = spx.bind(graphdef, graphtree.merge(graphother, copy=False))
            with apply.mesh:
                ids = with_sharding_constraint(
                    ids,
                    self.arguments.step_partition_spec,
                    mesh=apply.mesh,
                    ignore_mpmd=True,
                )
                mask = with_sharding_constraint(
                    mask,
                    self.arguments.step_partition_spec,
                    mesh=apply.mesh,
                    ignore_mpmd=True,
                )
                model_kwargs = normalize_generation_model_kwargs(
                    model_kwargs,
                    model_callable=getattr(apply, "forward", apply),
                )
                return get_per_token_logps(
                    apply,
                    ids,
                    mask,
                    self.arguments.max_prompt_length,
                    model_kwargs=model_kwargs,
                    logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                )

        if self.ref_state is not None:
            self.compute_refmodel_logps = compile_trainer_step(
                partial(_compute_refmodel_logps, graphdef=self.ref_state.graphdef),
                mesh=mesh,
                static_argnames=("graphdef"),
                in_shardings=(
                    self.ref_state.shardings.graphstate,
                    self.ref_state.shardings.graphother,
                    empty_sharding,
                    empty_sharding,
                    {key: None for key in normalize_generation_model_kwargs(None).keys()},
                ),
                out_shardings=empty_sharding,
            )
        else:
            # beta == 0: no reference model, so no reference-logps function.
            self.compute_refmodel_logps = None

        def _compute_state_logps(
            model_state: EasyDeLState,
            ids,
            mask,
            model_kwargs=None,
            *,
            prompt_length: int,
            logprob_vocab_chunk_size: int | None,
        ):
            """JIT-compiled policy/teacher per-token log-prob forward."""
            module = model_state.model
            with module.mesh:
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
                model_kwargs = normalize_generation_model_kwargs(
                    model_kwargs,
                    model_callable=getattr(module, "forward", module),
                )
                return get_per_token_logps(
                    module,
                    ids,
                    mask,
                    prompt_length,
                    model_kwargs=model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )

        def _compute_state_logps_and_topk(
            model_state: EasyDeLState,
            ids,
            mask,
            model_kwargs=None,
            *,
            prompt_length: int,
            topk: int,
        ):
            """JIT-compiled policy log-prob forward with top-k support snapshot."""
            module = model_state.model
            with module.mesh:
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
                model_kwargs = normalize_generation_model_kwargs(
                    model_kwargs,
                    model_callable=getattr(module, "forward", module),
                )
                return get_per_token_logps_and_topk(
                    module,
                    ids,
                    mask,
                    prompt_length,
                    topk,
                    model_kwargs=model_kwargs,
                )

        self.compute_state_logps = compile_trainer_step(
            _compute_state_logps,
            mesh=mesh,
            static_argnames=("prompt_length", "logprob_vocab_chunk_size"),
            out_shardings=empty_sharding,
        )
        self.compute_state_logps_and_topk = compile_trainer_step(
            _compute_state_logps_and_topk,
            mesh=mesh,
            static_argnames=("prompt_length", "topk"),
            out_shardings=(empty_sharding, empty_sharding, empty_sharding),
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
        """Run online generation, score completions, and assemble the GRPO batch.

        For every prompt this hook:

        1. Calls :meth:`generate_unified` to draw ``num_generations``
           completions per prompt.
        2. Computes the reference-model per-token log-probabilities
           via the compiled :meth:`compute_refmodel_logps`.
        3. Calls each registered reward function and combines results
           with the configured weights.
        4. Packs prompt/completion ids, masks, advantages, and reward
           breakdown into a JAX-friendly batch.

        Args:
            state: Current policy state.
            batch: Raw batch from the dataloader (containing prompt
                tokens and any per-prompt side-channel metadata).
            is_train: Whether the call is during training or eval.

        Returns:
            A ``(batch, info)`` tuple where ``batch`` is the JAX-pure
            dict consumed by ``grpo_step`` and ``info`` is a dict of
            timing / reward metrics.
        """
        if is_train:
            cached = self._take_buffered_grpo_batch()
            if cached is not None:
                return cached

        batch = self._apply_user_data_collator(batch)
        reward_batch = self._extract_reward_batch_sidechannels(batch)
        batch = self._purify_batch(batch)
        if reward_batch:
            reward_batch = {**batch, **reward_batch}
        else:
            reward_batch = batch
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]
            prompt_model_kwargs = extract_generation_model_kwargs(
                batch,
                model_callable=getattr(state.model, "forward", state.model),
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
                    apply_chat_template=False,  # GRPO doesn't apply chat template to prompts
                    shard_inputs=False,  # Already sharded
                    all_gather=False,  # We'll handle gathering ourselves
                    config_overrides=self._generation_config_overrides_for_phase(is_train),
                )
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids
                completion_prompts = results.completion_prompts

            generation_time = generation_time_fn()

            completion_mask = self._make_attn_mask(completion_ids)
            results, completion_mask = self._maybe_run_tool_call_generation_loop(
                state=state,
                results=results,
                completion_mask=completion_mask,
            )
            completion_ids = results.completion_ids
            completion_prompts = results.completion_prompts
            if self.arguments.mask_truncated_completions:
                eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
                has_eos = jnp.any(jnp.isin(completion_ids, eos_tokens), axis=1)
                completion_mask = completion_mask * has_eos[:, None].astype(completion_mask.dtype)
            # Derive how many completions we have per prompt instead of trusting config-only value.
            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            prompt_completion_ids = jnp.concatenate([prompt_ids.repeat(generation_factor, 0), completion_ids], axis=-1)
            difficulty_weights = None
            difficulty_key = getattr(self.arguments, "difficulty_key", None)
            if getattr(self.arguments, "difficulty_loss_weight", False) and difficulty_key is not None:
                difficulty_values = batch.get(difficulty_key)
                if difficulty_values is None:
                    difficulty_values = reward_batch.get(difficulty_key)
                if difficulty_values is not None:
                    difficulty_weights = jnp.asarray(difficulty_values, dtype=jnp.float32).reshape(-1)
                    difficulty_weights = difficulty_weights.repeat(generation_factor, axis=0)
            ridmask = prompt_mask.repeat(generation_factor, 0)
            repeated_prompt_model_kwargs = repeat_prompt_aligned_model_kwargs(
                scoring_prompt_model_kwargs,
                generation_factor,
                prompt_batch_size=prompt_mask.shape[0],
            )
            policy_repeated_model_kwargs = normalize_generation_model_kwargs(
                repeated_prompt_model_kwargs,
                model_callable=getattr(state.model, "forward", state.model),
            )
            prompt_completion_mask = jnp.concatenate([ridmask, completion_mask], -1)
            old_per_token_logps = None

            with capture_time() as token_logps_time_fn:
                if self.ref_state is None:
                    # beta == 0: KL term disabled -> skip the reference forward
                    # entirely. The placeholder is never read by the loss.
                    ref_per_token_logps = jnp.zeros(
                        (completion_ids.shape[0], completion_ids.shape[1]),
                        dtype=jnp.float32,
                    )
                else:
                    normalized_repeated_model_kwargs = normalize_generation_model_kwargs(
                        repeated_prompt_model_kwargs,
                        model_callable=getattr(self.ref_state.model, "forward", self.ref_state.model),
                    )
                    if (
                        self.ref_logps_chunk_size is not None
                        and prompt_completion_ids.shape[0] > self.ref_logps_chunk_size
                    ):
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
            old_token_logps_time = 0.0
            sampling_topk_indices = None
            sampling_topk_logps = None
            loss_type = getattr(self, "loss_type", getattr(self.arguments, "loss_type", None))
            dppo_divergence_type = getattr(self.arguments, "divergence_type", None) if loss_type == "dppo" else None
            needs_dppo_topk = dppo_divergence_type in {"topk_tv", "topk_kl"}
            if is_train and (self._generation_reuse_span() > 1 or needs_dppo_topk):
                with capture_time() as old_token_logps_time_fn:
                    if needs_dppo_topk:
                        (
                            old_per_token_logps,
                            sampling_topk_indices,
                            sampling_topk_logps,
                        ) = self.compute_state_logps_and_topk(
                            state,
                            prompt_completion_ids,
                            prompt_completion_mask,
                            policy_repeated_model_kwargs,
                            prompt_length=self.arguments.max_prompt_length,
                            topk=int(getattr(self.arguments, "divergence_topk", 20)),
                        )
                    else:
                        old_per_token_logps = self.compute_state_logps(
                            state,
                            prompt_completion_ids,
                            prompt_completion_mask,
                            policy_repeated_model_kwargs,
                            prompt_length=self.arguments.max_prompt_length,
                            logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                        )
                old_token_logps_time = old_token_logps_time_fn()

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
                clean_completions = [
                    [{"role": "assistant", "content": completion}] for completion in clean_completions_text
                ]
            else:
                raw_completions = raw_completions_text
                clean_completions = clean_completions_text
            target_len = len(clean_completions_text) or len(raw_completions_text) or int(completion_ids.shape[0])
            reasoning_records = self._coerce_optional_generation_texts(
                results.reasoning,
                target_len=target_len,
            )
            tool_call_records = self._coerce_generation_metadata_list(
                results.tool_calls,
                target_len=target_len,
            )
            # Per-completion generation signals exposed to reward functions:
            # finish_reason / truncated (hit the length cap), token-level length,
            # and the (padding-trimmed) completion token ids.
            finish_reason_records = self._coerce_generation_metadata_list(
                results.finish_reason,
                target_len=target_len,
            )
            truncated_records = [
                (reason == "length") if isinstance(reason, str) else None for reason in finish_reason_records
            ]
            host_completion_mask = np.asarray(jax.device_get(completion_mask), dtype=np.int32)
            host_completion_ids = np.asarray(jax.device_get(completion_ids), dtype=np.int64)
            completion_token_lengths = host_completion_mask.sum(axis=-1).tolist()
            completion_length_records = self._coerce_generation_metadata_list(
                [int(length) for length in completion_token_lengths],
                target_len=target_len,
            )
            completion_ids_records = self._coerce_generation_metadata_list(
                [
                    row[: int(length)]
                    for row, length in zip(host_completion_ids.tolist(), completion_token_lengths, strict=False)
                ],
                target_len=target_len,
            )
            structured_clean_completions = (
                self._build_structured_assistant_messages(
                    clean_completions_text,
                    tool_calls=tool_call_records,
                )
                if is_conversational
                else clean_completions
            )
            environment_feedback = self._run_environment_feedback(
                action_texts=raw_completions_text or clean_completions_text,
                tool_calls=tool_call_records,
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
                                for p, c in zip(completion_prompts, structured_clean_completions, strict=False)
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
                            texts = [p + c for p, c in zip(completion_prompts, clean_completions, strict=False)]

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
                            completions=clean_completions,
                            raw_completions=raw_completions,
                            completion_texts=clean_completions_text,
                            raw_text=raw_completions_text,
                            reasoning=reasoning_records,
                            tool_calls=tool_call_records,
                            finish_reason=finish_reason_records,
                            truncated=truncated_records,
                            completion_length=completion_length_records,
                            completion_ids=completion_ids_records,
                            max_length=self.arguments.max_length,
                            batch=reward_batch,
                            **(environment_feedback or {}),
                        )
                        output_reward_func = reward_func(**reward_call_kwargs)
                        rew = jnp.array(
                            [val if val is not None else jnp.nan for val in output_reward_func],
                            dtype="f4",
                        )
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
                if environment_feedback is not None:
                    environment_rewards = jnp.asarray(environment_feedback["environment_rewards"], dtype="f4").reshape(
                        -1, 1
                    )
                    rewards_per_func = jnp.concatenate([rewards_per_func, environment_rewards], axis=1)
            rewarding_time = rewarding_time_fn()
            log_completion_length = jnp.sum(completion_mask, -1)

            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)
            completion_ids = self._all_gather(completion_ids)
            completion_mask = self._all_gather(completion_mask)
            ref_per_token_logps = self._all_gather(ref_per_token_logps)
            if old_per_token_logps is not None:
                old_per_token_logps = self._all_gather(old_per_token_logps)
            if sampling_topk_indices is not None:
                sampling_topk_indices = self._all_gather(sampling_topk_indices)
            if sampling_topk_logps is not None:
                sampling_topk_logps = self._all_gather(sampling_topk_logps)
            rewards_per_func = self._all_gather(rewards_per_func)
            if difficulty_weights is not None:
                difficulty_weights = self._all_gather(difficulty_weights)
            scoring_prompt_model_kwargs = jax.tree_util.tree_map(
                lambda x: self._all_gather(x) if isinstance(x, jax.Array) else x,
                scoring_prompt_model_kwargs,
            )

            with capture_time() as grouped_comp_time_fn:
                generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
                generation_factor = max(generation_factor, 1)
                reward_weights = self.reward_weights
                if environment_feedback is not None:
                    reward_weights = jnp.concatenate([reward_weights, jnp.ones((1,), dtype=reward_weights.dtype)])
                rewards, advantages, std_rewards, is_std_zero = _compute_rewards_and_advantages(
                    rewards_per_func=rewards_per_func,
                    reward_weights=reward_weights,
                    generation_factor=generation_factor,
                    scale_rewards=self.scale_rewards,
                    multi_objective_aggregation=self.multi_objective_aggregation,
                    arguments=self.arguments,
                    group_reduction=getattr(self, "_group_reduction", "mean"),
                )
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
            "old_token_logps_time": old_token_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
            "frac_reward_zero_std": float(jnp.mean(is_std_zero.astype(jnp.float32))),
        }
        for i, reward_func_name in enumerate(self.reward_func_names):
            metrics_dict[reward_func_name] = float(jnp.nanmean(rewards_per_func[:, i]))
        if environment_feedback is not None:
            metrics_dict["environment_reward"] = float(jnp.nanmean(rewards_per_func[:, -1]))
        if difficulty_weights is not None:
            metrics_dict["difficulty_weight_mean"] = float(jnp.mean(difficulty_weights))
        self._maybe_log_grpo_completions(
            prompts=completion_prompts,
            completions=clean_completions_text,
            completion_lengths=log_completion_length,
            step=int(jax.device_get(state.step)),
        )
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

        # i don't care who you are and what you do.
        # ill find you and ill gather u...
        model_batch = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "num_items_in_batch": jnp.sum(completion_mask),
            **scoring_prompt_model_kwargs,
        }
        if old_per_token_logps is not None:
            model_batch["old_per_token_logps"] = old_per_token_logps
            model_batch["sampling_per_token_logps"] = old_per_token_logps
        if sampling_topk_indices is not None and sampling_topk_logps is not None:
            model_batch["sampling_topk_indices"] = sampling_topk_indices
            model_batch["sampling_topk_logps"] = sampling_topk_logps
        if difficulty_weights is not None:
            model_batch["difficulty_weights"] = difficulty_weights
        if is_train:
            return self._store_buffered_grpo_batch(model_batch, metrics_dict)
        return (model_batch, metrics_dict)

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """Post-step hook that optionally synchronizes the reference model.

        When ``sync_ref_model`` is enabled in the training arguments, this
        method performs an exponential moving average (EMA) update of the
        reference model parameters toward the current policy parameters
        every ``ref_model_sync_steps`` steps.

        Args:
            state: The current model state after the training step.
            metrics: Metrics collected during the training step.
            step: The current global training step number.

        Returns:
            tuple[EasyDeLState, MetricsType]: The (possibly unchanged) state
                and metrics, passed through for further processing.
        """

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
