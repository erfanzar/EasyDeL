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

"""Agentic MoshPit Trainer for multi-turn environment-based RL training.

This module implements the AgenticMoshPitTrainer, which extends GRPOTrainer
to support multi-turn agent-environment interactions. Instead of single-turn
generation followed by reward scoring, the trainer:

1. Runs multi-turn rollouts through environments (reset -> generate -> step -> ...)
2. Collects trajectories with per-turn prompt/response masks
3. Computes advantages using various estimators (episode, step, GiGPO)
4. Trains the policy using the same GRPO loss function

The trainer supports:
- Custom environments via ``env_factory`` callable
- Tool-calling with registered tools
- Multiple advantage estimation strategies
- Group-relative advantages across rollouts with the same seed

Inspired by Alibaba's ROLL agentic pipeline, adapted for EasyDeL's
JAX-native architecture.
"""

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from easydel.inference.reasoning.auto_detect import make_reasoning_stripper
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import capture_time, get_logger

try:
    import wandb
except ImportError:
    wandb = None

from ..group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from ..prompt_utils import apply_chat_template
from ..training_utils import (
    normalize_generation_model_kwargs,
    slice_prompt_aligned_model_kwargs,
)
from .agentic_moshpit_config import AgenticMoshPitConfig
from .env_manager import RolloutManager, turn_record_to_message
from .environment import AgenticEnvironment, ToolEnvWrapper, create_tool_call_parser
from .self_play import LocalQuestionGenerator, SelfPlayEnvironment
from .tools import Tool, make_tool
from .utils import (
    compute_advantages_episode,
    compute_advantages_gigpo,
    compute_advantages_step,
)

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)

RewardFunc = EasyDeLBaseModule | EasyDeLState | tp.Callable[[list, list], list[float]]


class _InfiniteRolloutDataset:
    """Infinite dummy dataset for environment-driven training.

    Yields empty prompt dicts forever so the training loop never
    exhausts and stays on epoch 0. The actual data comes from
    environment rollouts in ``_preprocess_batch_input``, so the
    batch content is ignored.

    Implements both the ``IterableDataset`` and ``Dataset`` interfaces
    that the base trainer's dataloader and preprocessing expect
    (``__iter__``, ``__len__``, ``__getitem__``, ``column_names``).
    """

    def __init__(self, batch_size: int):
        """Configure the dummy dataset to produce ``batch_size``-sized batches.

        Args:
            batch_size: Per-step batch size used to compute a synthetic
                length so the trainer treats one "epoch" as 10 000 steps.
        """
        self._batch_size = batch_size
        self._row = {"prompt": [{"role": "user", "content": ""}], "input_ids": [0], "attention_mask": [0]}
        self._len = batch_size * 10_000

    def __iter__(self):
        """Yield the same empty prompt forever."""
        while True:
            yield self._row

    def __len__(self):
        """Return the synthetic dataset length."""
        return self._len

    def __getitem__(self, key):
        """Return the empty prompt regardless of the requested index/slice.

        Args:
            key: Integer index or slice; both return the same canned row.

        Returns:
            The empty prompt row, or a list of identical rows for a slice.
        """
        if isinstance(key, int):
            return self._row
        if isinstance(key, slice):
            indices = range(*key.indices(self._len))
            return [self._row for _ in indices]
        return self._row

    @property
    def column_names(self) -> list[str]:
        """Return the column names exposed by the dummy row."""
        return list(self._row.keys())

    @property
    def num_rows(self) -> int:
        """Return the synthetic row count."""
        return self._len


@Registry.register("trainer", "agentic-moshpit")
class AgenticMoshPitTrainer(GRPOTrainer):
    """Agentic MoshPit Trainer for multi-turn RL training with environments.

    This trainer extends GRPOTrainer to handle multi-turn agent-environment
    interactions. The key difference is in ``_preprocess_batch_input``:
    instead of single-turn generation + reward scoring, it runs full
    episode rollouts through environments and collects trajectories.

    The trainer supports:
    - Custom environments via ``env_factory``
    - Tool-calling with registered tools
    - Multiple advantage estimators (GRPO, step, GiGPO, agentic_reinforce)
    - Dense (per-step) and sparse (terminal-only) rewards
    - Group-relative advantage computation

    Args:
        arguments: AgenticMoshPitConfig with training hyperparameters.
        model: Language model or state for the policy.
        env_factory: Factory function that creates environment instances.
            Called once per episode. Should return an AgenticEnvironment.
        reward_funcs: Optional additional reward functions (on top of
            environment rewards). Same interface as GRPOTrainer.
        tools: Optional list of Tool instances for the agent.
        train_dataset: Optional training dataset (used for env seed generation
            if env_factory is provided, or as prompts if not).
        eval_dataset: Optional evaluation dataset.
        processing_class: Tokenizer or processor.
        reward_processing_classes: Tokenizers for reward models.
        data_tokenize_fn: Optional tokenization function.

    References:
        https://open.spotify.com/track/4evMMKc2HD6fV9slMfgkMx?si=35247c70983549a1

    Example:
        >>> class MathEnv(AgenticEnvironment):
        ...     def reset(self, seed=None):
        ...         return ResetResult("Solve: 2 + 2 = ?")
        ...     def step(self, action):
        ...         correct = "4" in action
        ...         return StepResult("", 1.0 if correct else 0.0, terminated=True)
        ...
        >>> config = AgenticMoshPitConfig(
        ...     max_steps=5,
        ...     group_size=4,
        ...     num_env_groups=8,
        ...     reward_mode="episode",
        ...     advantage_estimator="grpo",
        ... )
        >>> trainer = AgenticMoshPitTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     env_factory=lambda: MathEnv(),
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    arguments: AgenticMoshPitConfig

    def __init__(
        self,
        arguments: AgenticMoshPitConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        env_factory: tp.Callable[[], AgenticEnvironment],
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        tools: list[Tool] | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        """Initialize the multi-turn agentic trainer.

        Wires up the reasoning stripper, tool registry, environment
        factory, rollout manager, and the underlying GRPO trainer.

        Args:
            arguments: Algorithm configuration (must be
                :class:`AgenticMoshPitConfig`).
            model: Model module or state used as the policy.
            env_factory: Zero-argument callable that returns a fresh
                :class:`AgenticEnvironment` per episode.
            reward_funcs: Optional auxiliary reward callables/strings
                applied on top of environment rewards.  Defaults to a
                placeholder so GRPO sees a non-empty list.
            tools: Optional pre-instantiated tool list to expose to the
                policy.
            train_dataset: Optional training dataset; defaults to an
                infinite synthetic dataset when no prompts are required.
            eval_dataset: Optional evaluation dataset.
            processing_class: Tokenizer or processor.
            reward_processing_classes: Optional tokenizer(s) for reward
                models.
            data_tokenize_fn: Optional custom tokenization callable.

        Raises:
            TypeError: If ``arguments`` is not an
                :class:`AgenticMoshPitConfig`.
        """
        if not isinstance(arguments, AgenticMoshPitConfig):
            raise TypeError(f"arguments must be AgenticMoshPitConfig, got {type(arguments)}")

        model_type = None
        _model = model.model if isinstance(model, EasyDeLState) else model
        if _model is not None and hasattr(_model, "config"):
            model_type = getattr(_model.config, "model_type", None)

        self._strip_thinking = make_reasoning_stripper(
            parser_name=arguments.reasoning_parser,
            model_type=model_type,
            tokenizer=processing_class,
        )

        self.env_factory = env_factory
        self._rollout_step = 0

        self._tools: list[Tool] = []
        if tools:
            self._tools.extend(tools)
        if arguments.tool_names:
            for tool_name in arguments.tool_names:
                self._tools.append(make_tool(tool_name))

        if arguments.tool_schemas is None and self._tools:
            arguments.tool_schemas = [tool.chat_schema for tool in self._tools]

        self._tool_call_parser = None
        if self._tools:
            self._tool_call_parser = create_tool_call_parser(
                tool_caller=arguments.tool_caller,
                tokenizer=processing_class,
            )

        if reward_funcs is None:
            reward_funcs = [self._env_reward_placeholder]

        if train_dataset is None:
            train_dataset = _InfiniteRolloutDataset(arguments.total_batch_size)

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

        self._rollout_manager = RolloutManager(
            tokenizer=self.processing_class,
            max_steps=arguments.max_steps,
            max_seq_length=arguments.max_length,
            system_prompt=arguments.system_prompt,
            tool_schemas=arguments.tool_schemas,
        )

    @staticmethod
    def _env_reward_placeholder(prompts, completions, **kwargs) -> list[float]:
        """Placeholder reward function — actual rewards come from environments."""
        return [0.0] * len(completions)

    def _make_generate_fn(
        self,
        state: EasyDeLState,
    ) -> tp.Callable[..., list[str]]:
        """Create a batched generation function for rollouts.

        Wraps ``generate_unified`` to accept a list of prompt strings
        and return a list of response strings. Supports optional
        ``temperature``, ``top_p``, ``top_k`` overrides per call so
        the questioner/verifier can use different sampling params
        than the solver.

        Args:
            state: Current model state.

        Returns:
            Function ``(prompts, *, temperature?, top_p?, top_k?) -> list[str]``
        """

        def generate_fn(
            prompts: list[str],
            *,
            temperature: float | None = None,
            top_p: float | None = None,
            top_k: int | None = None,
            num_return_sequences: int | None = None,
            strip_thinking: bool = False,
        ) -> list[str]:
            """Generate visible action text for a batch of prompts.

            Args:
                prompts: List of prompt strings (already chat-templated).
                temperature: Optional sampling temperature override.
                top_p: Optional nucleus sampling threshold override.
                top_k: Optional top-k sampling override.
                num_return_sequences: Optional number of completions per
                    prompt.
                strip_thinking: If ``True``, the configured reasoning
                    stripper is applied to each completion before return.

            Returns:
                A list of visible action strings, one per generation. The
                wrapper also stashes raw text / reasoning / tool-call
                metadata on ``generate_fn.last_generation_metadata`` for
                downstream rollout bookkeeping.
            """
            overrides: dict[str, tp.Any] = {}
            if temperature is not None:
                overrides["temperature"] = temperature
            if top_p is not None:
                overrides["top_p"] = top_p
            if top_k is not None:
                overrides["top_k"] = top_k
            if num_return_sequences is not None:
                overrides["num_return_sequences"] = num_return_sequences

            results = self.generate_unified(
                prompts=prompts,
                state=state,
                apply_chat_template=False,
                shard_inputs=True,
                release_runtime_after_generation=False,
                config_overrides=overrides or None,
            )

            raw_texts = self._coerce_generation_texts(
                results.raw_text,
                fallback=results.text,
            )
            if not raw_texts:
                raw_texts = self._coerce_generation_texts(results.generation_results)
            visible_texts = self._coerce_generation_texts(
                results.text,
                fallback=raw_texts,
            )
            action_texts = list(visible_texts)
            if strip_thinking:
                action_texts = [self._strip_thinking(text) for text in action_texts]
            reasoning_records = self._coerce_optional_generation_texts(results.reasoning, target_len=len(action_texts))
            tool_call_records = self._coerce_generation_metadata_list(results.tool_calls, target_len=len(action_texts))
            generate_fn.last_generation_metadata = [
                {
                    "text": action_texts[i],
                    "raw_text": raw_texts[i],
                    "reasoning": reasoning_records[i],
                    "tool_calls": tool_call_records[i],
                }
                for i in range(len(action_texts))
            ]
            return action_texts

        return generate_fn

    @staticmethod
    def _is_env_reward_placeholder(reward_func: tp.Any) -> bool:
        """Return ``True`` when ``reward_func`` is the env-reward placeholder.

        Args:
            reward_func: A reward callable or state object.

        Returns:
            ``True`` when the function is the placeholder injected for
            environments that score themselves.
        """
        return getattr(reward_func, "__name__", None) == "_env_reward_placeholder"

    def _score_auxiliary_rewards(
        self,
        trajectories: list,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run any non-placeholder auxiliary reward functions over rollouts.

        Builds the chat-templated prompts and assistant completions for
        each trajectory's final response turn, then dispatches to either
        a reward-model :class:`EasyDeLState` or a plain Python callable.

        Args:
            trajectories: Rollout trajectories produced by the rollout
                manager.

        Returns:
            ``(total_rewards, breakdown)`` where ``total_rewards`` is a
            ``[len(trajectories)]`` float32 vector summing every
            auxiliary reward weighted by ``reward_weights`` and
            ``breakdown`` maps reward-name to its raw per-trajectory
            scores.
        """
        active_reward_indices = [
            idx for idx, reward_func in enumerate(self.reward_funcs) if not self._is_env_reward_placeholder(reward_func)
        ]
        if not active_reward_indices:
            return np.zeros(len(trajectories), dtype=np.float32), {}

        prompt_messages: list[list[dict[str, tp.Any]]] = []
        prompt_texts: list[str] = []
        completion_texts: list[str] = []
        raw_texts: list[str] = []
        reasoning_records: list[str | None] = []
        tool_call_records: list[tp.Any | None] = []
        completion_messages: list[dict[str, tp.Any]] = []

        for traj in trajectories:
            response_indices = [i for i, turn in enumerate(traj.turns) if turn.is_response]
            if not response_indices:
                prompt_messages.append([{"role": "user", "content": ""}])
                prompt_texts.append("")
                completion_texts.append("")
                raw_texts.append("")
                reasoning_records.append(None)
                tool_call_records.append(None)
                completion_messages.append({"role": "assistant", "content": ""})
                continue

            final_idx = response_indices[-1]
            final_turn = traj.turns[final_idx]
            prefix_messages = [turn_record_to_message(turn) for turn in traj.turns[:final_idx]]
            prompt_messages.append(prefix_messages)
            try:
                prompt_texts.append(
                    self.processing_class.apply_chat_template(
                        prefix_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        tools=self.arguments.tool_schemas,
                    )
                )
            except Exception:
                prompt_texts.append(str(prefix_messages))

            raw_texts.append(final_turn.raw_content if final_turn.raw_content is not None else final_turn.content)
            completion_texts.append(
                final_turn.visible_content if final_turn.visible_content is not None else final_turn.content
            )
            reasoning_records.append(final_turn.reasoning)
            tool_call_records.append(final_turn.tool_calls)
            completion_messages.append(turn_record_to_message(final_turn))

        completions = [[dict(message)] for message in completion_messages]
        raw_completions = []
        for message, raw_text in zip(completion_messages, raw_texts, strict=False):
            raw_message = dict(message)
            raw_message["content"] = raw_text
            raw_completions.append([raw_message])

        total_rewards = np.zeros(len(trajectories), dtype=np.float32)
        reward_breakdown: dict[str, np.ndarray] = {}
        reward_weights = np.asarray(jax.device_get(self.reward_weights), dtype=np.float32)

        for idx in active_reward_indices:
            reward_func = self.reward_funcs[idx]
            reward_processing_class = self.reward_processing_classes[idx]
            name = self.reward_func_names[idx]

            if isinstance(reward_func, EasyDeLState):
                messages = [{"messages": [*p, c]} for p, c in zip(prompt_messages, completion_messages, strict=False)]
                texts = [
                    apply_chat_template(
                        x,
                        reward_processing_class,
                        tools=self._reward_chat_template_tools(),
                    )["text"]
                    for x in messages
                ]
                values = np.asarray(
                    jax.device_get(
                        reward_func.apply_fn(
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
                    ),
                    dtype=np.float32,
                )
            else:
                reward_call_kwargs = self._build_reward_call_kwargs(
                    reward_func,
                    prompts=prompt_messages,
                    completions=completions,
                    raw_completions=raw_completions,
                    prompt_texts=prompt_texts,
                    completion_texts=completion_texts,
                    raw_text=raw_texts,
                    reasoning=reasoning_records,
                    tool_calls=tool_call_records,
                    max_length=self.arguments.max_length,
                    trajectories=trajectories,
                )
                outputs = reward_func(**reward_call_kwargs)
                values = np.asarray([val if val is not None else np.nan for val in outputs], dtype=np.float32)

            reward_breakdown[name] = values
            total_rewards = total_rewards + np.nan_to_num(values) * reward_weights[idx]

        return total_rewards, reward_breakdown

    def _apply_auxiliary_rewards_to_trajectories(
        self,
        trajectories: list,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Add auxiliary rewards back into trajectory step rewards.

        Computes auxiliary rewards via :meth:`_score_auxiliary_rewards`
        and folds them into each trajectory based on the configured
        ``advantage_estimator``.  Trajectories without reward changes are
        returned unchanged.

        Args:
            trajectories: Rollout trajectories to mutate in-place.

        Returns:
            ``(aux_rewards, breakdown)`` from the underlying scorer for
            downstream logging.
        """
        aux_rewards, reward_breakdown = self._score_auxiliary_rewards(trajectories)
        if not trajectories or not np.any(np.nan_to_num(aux_rewards)):
            return aux_rewards, reward_breakdown

        estimator = self.arguments.advantage_estimator
        for idx, (traj, aux_reward) in enumerate(zip(trajectories, aux_rewards, strict=False)):
            env_reward = float(traj.episode_reward)
            aux_value = float(np.nan_to_num(aux_reward))
            traj.info["env_reward"] = env_reward
            traj.info["aux_reward"] = aux_value
            for name, values in reward_breakdown.items():
                traj.info[f"aux_reward/{name}"] = float(np.nan_to_num(values[idx]))
            traj.episode_reward = env_reward + aux_value
            traj.info["combined_reward"] = traj.episode_reward
            if estimator in ("step_reinforce", "agentic_reinforce"):
                if traj.step_rewards:
                    traj.step_rewards[-1] += aux_value
                else:
                    traj.step_rewards.append(aux_value)

        return aux_rewards, reward_breakdown

    def _wrap_env_with_tools(self, env: AgenticEnvironment) -> AgenticEnvironment:
        """Wrap environment with tool support if tools are configured."""
        if not self._tools:
            return env
        return ToolEnvWrapper(
            env=env,
            tools=self._tools,
            tool_call_parser=self._tool_call_parser,
            max_tool_calls_per_step=self.arguments.max_tool_calls_per_step,
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Run agentic rollouts and prepare training batch.

        This overrides GRPOTrainer's _preprocess_batch_input to:
        1. Run multi-turn environment rollouts (instead of single-turn generation)
        2. Collect trajectories with prompt/response masks
        3. Compute advantages using the configured estimator
        4. Compute reference model log-probs on the trajectories
        5. Return the same batch format as standard GRPO

        Args:
            state: Current model state.
            batch: Input batch (may contain seeds or prompts).
            is_train: Whether this is a training step.

        Returns:
            Tuple of (processed batch dict, metrics dict).
        """
        with capture_time() as total_time_fn:
            with capture_time() as rollout_time_fn:
                generate_fn = self._make_generate_fn(state)
                try:

                    def wrapped_env_factory():
                        """Create an environment instance with tools and self-play hooks attached.

                        Returns:
                            A fresh :class:`AgenticEnvironment` wrapped in
                            :class:`ToolEnvWrapper` when tools are
                            registered, with the local generation
                            function and reasoning stripper installed on
                            self-play question generators.
                        """
                        env = self.env_factory()
                        if isinstance(env, SelfPlayEnvironment):
                            env.set_generate_fn(generate_fn)
                            gen = env._generator
                            if isinstance(gen, LocalQuestionGenerator):
                                if gen._tokenizer is None:
                                    gen._tokenizer = self.processing_class
                                gen._strip_reasoning = self._strip_thinking
                        return self._wrap_env_with_tools(env)

                    trajectories = self._rollout_manager.run_grouped_episodes(
                        env_factory=wrapped_env_factory,
                        generate_fn=generate_fn,
                        group_size=self.arguments.group_size,
                        base_seed=self._rollout_step * self.arguments.num_env_groups,
                        num_groups=self.arguments.num_env_groups,
                    )
                    aux_rewards, aux_reward_breakdown = self._apply_auxiliary_rewards_to_trajectories(trajectories)
                    self._rollout_step += 1
                finally:
                    if self.arguments.use_esurge_generation:
                        try:
                            state.model.pause_esurge(release_model_state=True)
                        except Exception as exc:  # pragma: no cover - best-effort rollout cleanup
                            logger.debug("Failed to release eSurge runtime after agentic rollout: %s", exc)

            rollout_time = rollout_time_fn()

            collated = self._rollout_manager.collate_trajectories(
                trajectories,
                max_prompt_length=self.arguments.max_prompt_length,
                max_completion_length=self.arguments.max_completion_length,
            )

            prompt_ids = jnp.array(collated["prompt_ids"])
            prompt_mask = jnp.array(collated["prompt_mask"])
            completion_ids = jnp.array(collated["completion_ids"])
            completion_mask = jnp.array(collated["completion_mask"])
            env_rewards = jnp.array(collated["rewards"])
            step_rewards_list = collated["step_rewards_list"]
            rollout_reasoning = []
            rollout_tool_calls = []
            for traj in trajectories:
                response_turns = [turn for turn in traj.turns if turn.is_response]
                reasoning_entries = [turn.reasoning for turn in response_turns if turn.reasoning is not None]
                tool_call_entries = [turn.tool_calls for turn in response_turns if turn.tool_calls not in (None, [])]
                rollout_reasoning.append(reasoning_entries or None)
                rollout_tool_calls.append(tool_call_entries or None)

            with capture_time() as advantage_time_fn:
                group_size = self.arguments.group_size

                if self.arguments.advantage_estimator == "grpo":
                    advantages, std_rewards = compute_advantages_episode(env_rewards, group_size, self.scale_rewards)
                elif self.arguments.advantage_estimator == "gigpo":
                    advantages, std_rewards = compute_advantages_gigpo(
                        episode_rewards=env_rewards,
                        step_rewards_list=step_rewards_list,
                        group_size=group_size,
                        episode_weight=self.arguments.episode_reward_weight,
                        step_weight=self.arguments.step_reward_weight,
                        gamma=self.arguments.step_reward_gamma,
                        scale_rewards=self.scale_rewards,
                    )
                elif self.arguments.advantage_estimator in ("step_reinforce", "agentic_reinforce"):
                    step_adv = compute_advantages_step(step_rewards_list, group_size, self.arguments.step_reward_gamma)
                    advantages = jnp.array(step_adv, dtype=jnp.float32)
                    std_rewards = jnp.nanstd(env_rewards) * jnp.ones_like(advantages)
                elif self.arguments.advantage_estimator == "reinforce":
                    advantages = env_rewards - jnp.nanmean(env_rewards)
                    std = jnp.nanstd(env_rewards)
                    is_zero = jnp.isclose(std, 0.0)
                    advantages = jnp.where(is_zero, 0.0, advantages / (std + 1e-4))
                    std_rewards = std * jnp.ones_like(advantages)
                else:
                    raise ValueError(f"Unknown advantage_estimator: {self.arguments.advantage_estimator}")
            advantage_time = advantage_time_fn()

            with capture_time() as ref_logps_time_fn:
                sequences = jnp.concatenate(
                    [
                        prompt_ids.repeat(1, 0),
                        completion_ids,
                    ],
                    axis=1,
                )
                seq_mask = jnp.concatenate(
                    [
                        prompt_mask.repeat(1, 0),
                        completion_mask,
                    ],
                    axis=1,
                )

                ref_model_kwargs = normalize_generation_model_kwargs(
                    None,
                    model_callable=getattr(self.ref_state.model, "forward", self.ref_state.model),
                )

                if self.ref_logps_chunk_size is not None and sequences.shape[0] > self.ref_logps_chunk_size:
                    ref_chunks: list[jax.Array] = []
                    full_batch_size = int(sequences.shape[0])
                    for start in range(0, full_batch_size, self.ref_logps_chunk_size):
                        end = min(start + self.ref_logps_chunk_size, full_batch_size)
                        ref_chunks.append(
                            self.compute_refmodel_logps(
                                self.ref_state.graphstate,
                                self.ref_state.graphother,
                                sequences[start:end],
                                seq_mask[start:end],
                                slice_prompt_aligned_model_kwargs(
                                    ref_model_kwargs,
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
                        sequences,
                        seq_mask,
                        ref_model_kwargs,
                    )
            ref_logps_time = ref_logps_time_fn()

            self._log_training_generations_to_wandb(
                state=state,
                prompts=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                generation_time=rollout_time,
                reasoning=rollout_reasoning,
                tool_calls=rollout_tool_calls,
                source="policy",
            )

            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)
            completion_ids = self._all_gather(completion_ids)
            completion_mask = self._all_gather(completion_mask)
            ref_per_token_logps = self._all_gather(ref_per_token_logps)
            advantages = self._all_gather(advantages)

        total_time = total_time_fn()

        completion_length = jnp.sum(completion_mask, -1)
        avg_steps = float(np.mean([t.num_steps for t in trajectories]))

        metrics_dict: dict[str, float | int | str] = {
            "reward_mean": float(jnp.nanmean(env_rewards)),
            "reward_std": float(jnp.nanstd(env_rewards)),
            "completion_length": float(jnp.mean(completion_length)),
            "rollout_time": rollout_time,
            "advantage_time": advantage_time,
            "ref_logps_time": ref_logps_time,
            "preprocessing_time": total_time,
            "avg_episode_steps": avg_steps,
            "frac_reward_zero_std": float(jnp.mean(jnp.isclose(std_rewards, 0.0).astype(jnp.float32))),
        }
        if aux_reward_breakdown:
            metrics_dict["aux_reward_mean"] = float(np.nanmean(aux_rewards))
            for name, values in aux_reward_breakdown.items():
                metrics_dict[f"aux_reward/{name}"] = float(np.nanmean(values))

        self._log_trajectories_to_wandb(trajectories, state)

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

    def _log_trajectories_to_wandb(
        self,
        trajectories: list,
        state: EasyDeLState,
    ) -> None:
        """Log full trajectory conversations to Weights & Biases.

        Each trajectory is logged as a wandb.Table row containing the
        full multi-turn conversation, reward, number of steps, and
        episode metadata. This gives full visibility into what the
        agent is doing at each training step.
        """
        if (
            not self.arguments.use_wandb
            or wandb is None
            or not self.arguments.can_log_metrics
            or not self.arguments.log_training_generations_to_wandb
        ):
            return

        import json as _json

        cur_step = int(jax.device_get(state.step))
        rows = []
        for i, traj in enumerate(trajectories):
            turns = []
            for turn in traj.turns:
                row = {"role": turn.role, "content": turn.content}
                if turn.visible_content is not None:
                    row["visible_content"] = turn.visible_content
                if turn.raw_content is not None:
                    row["raw_content"] = turn.raw_content
                if turn.reasoning is not None:
                    row["reasoning"] = turn.reasoning
                if turn.tool_calls is not None:
                    row["tool_calls"] = turn.tool_calls
                turns.append(row)
            rows.append(
                [
                    cur_step,
                    i,
                    traj.episode_reward,
                    traj.num_steps,
                    float(jnp.sum(traj.response_mask)),
                    traj.info.get("question", ""),
                    traj.info.get("topic", ""),
                    _json.dumps(turns, ensure_ascii=False),
                    traj.info.get("verifier_response", ""),
                ]
            )
        table = wandb.Table(
            columns=[
                "step",
                "episode_idx",
                "reward",
                "num_turns",
                "response_tokens",
                "question",
                "topic",
                "conversation",
                "verifier_response",
            ],
            data=rows,
        )
        wandb.log({"trajectories": table}, step=cur_step)

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create a data collator for agentic training.

        Since agentic training generates data through rollouts rather
        than from a dataset, the collator is minimal — it just passes
        through seed/index information if present.
        """
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create a Grain data collator for agentic training."""
        from ..utils import GRPODataCollatorGrain

        return GRPODataCollatorGrain(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )
