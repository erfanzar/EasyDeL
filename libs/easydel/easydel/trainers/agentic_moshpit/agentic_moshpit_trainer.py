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

import math
import pickle
import typing as tp

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec

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

from .._shared.esurge_generation import (
    DEFAULT_MAX_NEW_TOKENS,
    clamp_new_tokens_to_bound,
    generation_length_bound,
    resolve_sampling,
)
from ..agent_flow_trainer.agent_flow_trainer import (
    _EYVAN_MAX_COMPLETION_GROUPS_PER_STEP,
    _EYVAN_MAX_GROUP_RESPONSES,
    _EYVAN_MAX_OBS_PREVIEW_CHARS,
    _EYVAN_MAX_ROLLOUT_STEPS,
    _EYVAN_MAX_ROLLOUTS_PER_STEP,
    _EYVAN_MAX_TEXT_CHARS,
    _clip_eyvan_text,
    _compute_icepop_active_ratios,
    _compute_per_episode_kl_numpy,
    _encode_action_labels,
    _extract_nonneg_step,
    _eyvan_log_completions,
    _eyvan_log_episode,
    _eyvan_log_histogram,
    _eyvan_log_rollout,
    _resolve_eyvan_env_name,
    _select_rollout_sample_indices,
    _to_host_numpy,
    _to_host_numpy_local_only,
)
from ..group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from ..metrics import _host_mean_float, _host_scalar_float
from ..prompt_utils import apply_chat_template
from ..reward_protocol import RewardProtocol
from ..training_configurations import _resolve_eyvan_run_name, _sanitize_eyvan_key
from ..training_utils import (
    normalize_generation_model_kwargs,
    slice_prompt_aligned_model_kwargs,
)
from .agentic_moshpit_config import AgenticMoshPitConfig
from .env_manager import RolloutManager, turn_record_to_message
from .environment import AgenticEnvironment, ToolEnvWrapper, create_tool_call_parser
from .tools import Tool, make_tool
from .utils import (
    compute_advantages_agentic,
    compute_advantages_episode,
    compute_advantages_gigpo,
    compute_advantages_step,
    normalize_rewards_batch,
)

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)

RewardFunc = EasyDeLBaseModule | EasyDeLState | RewardProtocol | tp.Callable[[list, list], list[float]]


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
        arguments.validate_agentic_knobs()

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
        self._eval_rollout_step = 0
        self._eyvan_total_turns = 0
        self._eyvan_total_action_tokens = 0
        self._eyvan_env_name = _resolve_eyvan_env_name(env_factory, fallback="agentic_moshpit")
        eyvan_tag_env = _resolve_eyvan_env_name(env_factory, fallback="") or None
        bind_eyvan = getattr(arguments, "_bind_eyvan_runtime_context", None)
        if callable(bind_eyvan):
            bind_eyvan(
                trainer=self,
                model=_model,
                env_name=eyvan_tag_env,
                extra_trainer_tags=("agentic_moshpit",),
            )
        _resolve_eyvan_run_name(arguments, model=_model, trainer=self, trainer_slug="agentic_moshpit")

        self._tools: list[Tool] = []
        if tools:
            self._tools.extend(tools)
        if arguments.tool_names:
            for tool_name in arguments.tool_names:
                self._tools.append(make_tool(tool_name))

        # Copy tool_schemas rather than mutating arguments.tool_schemas in-place.
        if arguments.tool_schemas is not None:
            self._tool_schemas: list[dict[str, tp.Any]] | None = list(arguments.tool_schemas)
        elif self._tools:
            self._tool_schemas = [tool.chat_schema for tool in self._tools]
        else:
            self._tool_schemas = None

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
            tool_schemas=self._tool_schemas,
        )

    @staticmethod
    def _env_reward_placeholder(prompts, completions, **kwargs) -> list[float]:
        """Placeholder reward function — actual rewards come from environments.

        Used as a sentinel reward callable so the underlying GRPO trainer
        sees a non-empty ``reward_funcs`` list even when all real rewards
        originate inside the environment.

        Args:
            prompts: Batch prompts (ignored).
            completions: Batch completions; used only to derive the
                output length.
            **kwargs: Unused additional context forwarded by the GRPO
                trainer's reward dispatch.

        Returns:
            A list of zeros with the same length as ``completions``.
        """
        return [0.0] * len(completions)

    @staticmethod
    def _broadcast_bytes_from_host0(data: bytes | None) -> bytes:
        """Broadcast a variable-length byte payload from process 0 to all JAX hosts."""
        if jax.process_count() <= 1:
            return data if data is not None else b""
        is_root = jax.process_index() == 0
        raw = data if is_root and data is not None else b""
        length_arr = np.array([len(raw)], dtype=np.int64)
        length = int(np.asarray(multihost_utils.broadcast_one_to_all(length_arr))[0])
        if length == 0:
            return b""
        buf = np.frombuffer(raw, dtype=np.uint8) if is_root else np.zeros(length, dtype=np.uint8)
        synced = np.asarray(multihost_utils.broadcast_one_to_all(buf), dtype=np.uint8)
        return synced.tobytes()

    @classmethod
    def _broadcast_object_from_host0(cls, obj: tp.Any) -> tp.Any:
        """Broadcast a picklable Python object from process 0 to all JAX hosts."""
        if jax.process_count() <= 1:
            return obj
        raw = pickle.dumps(obj) if jax.process_index() == 0 else None
        synced = cls._broadcast_bytes_from_host0(raw)
        return pickle.loads(synced)

    def _to_global_replicated_array(self, arr: tp.Any, *, dtype: tp.Any = None) -> jax.Array:
        """Convert host-local arrays into mesh-consistent global replicated JAX arrays.

        Avoids passing host-local ``SingleDeviceSharding`` ``jax.Array``s to
        ``BaseTrainer._all_gather`` (``jax.device_put`` with a global sharding),
        which raises on multi-host meshes.
        """
        if jax.process_count() > 1:
            mesh = getattr(getattr(self, "model", None), "mesh", None)
            if isinstance(arr, jax.Array) and not getattr(arr.sharding, "is_fully_addressable", True):
                return self._all_gather(arr)
            np_arr = np.asarray(jax.device_get(arr), dtype=dtype)
            if mesh is not None:
                return multihost_utils.host_local_array_to_global_array(np_arr, mesh, PartitionSpec())
            return jnp.asarray(np_arr, dtype=dtype) if dtype is not None else jnp.asarray(np_arr)
        jnp_arr = jnp.asarray(arr, dtype=dtype) if dtype is not None else jnp.asarray(arr)
        return self._all_gather(jnp_arr)

    def _pad_prompt_token_ids(
        self,
        prompt_token_ids: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Left-pad variable-length prompt token ID rows into rectangular arrays."""
        pad_id = getattr(self, "_pad_token_id", None)
        if pad_id is None:
            proc = getattr(self, "processing_class", None)
            pad_id = getattr(proc, "pad_token_id", None)
            if pad_id is None:
                eos_id = getattr(proc, "eos_token_id", None)
                if isinstance(eos_id, list | tuple):
                    eos_id = eos_id[0] if eos_id else None
                pad_id = eos_id
            pad_id = int(pad_id) if pad_id is not None else 0

        args = getattr(self, "arguments", None)
        max_allowed = getattr(args, "max_length", None)
        bound = generation_length_bound(args) if args is not None else 0
        max_prompt_cap: int | None = int(max_allowed) if max_allowed is not None else None
        if bound > 1:
            max_prompt_cap = bound - 1 if max_prompt_cap is None else min(max_prompt_cap, bound - 1)
        clipped_rows = [
            row[-max_prompt_cap:] if (max_prompt_cap is not None and len(row) > max_prompt_cap) else list(row)
            for row in prompt_token_ids
        ]
        width = max((len(row) for row in clipped_rows), default=1)
        width = max(width, 1)
        padded_ids = np.full((len(clipped_rows), width), pad_id, dtype=np.int32)
        padded_mask = np.zeros((len(clipped_rows), width), dtype=np.int32)
        for i, row in enumerate(clipped_rows):
            if row:
                padded_ids[i, width - len(row) :] = row
                padded_mask[i, width - len(row) :] = 1
        return padded_ids, padded_mask

    @staticmethod
    def _extract_rollout_token_entries(
        results: tp.Any,
        target_len: int,
    ) -> list[dict[str, np.ndarray | None]]:
        """Extract exact unpadded completion_ids and sampling_per_token_logps per row."""
        comp_ids = getattr(results, "completion_ids", None)
        comp_mask = getattr(results, "completion_mask", None)
        sampling_logps = getattr(results, "sampling_per_token_logps", None)
        if comp_ids is None:
            return [{"token_ids": None, "sampling_logps": None} for _ in range(target_len)]

        host_ids = np.asarray(jax.device_get(comp_ids), dtype=np.int64)
        if host_ids.ndim == 1:
            host_ids = host_ids[None, :]
        host_mask = None
        if comp_mask is not None:
            host_mask = np.asarray(jax.device_get(comp_mask), dtype=bool)
            if host_mask.ndim == 1:
                host_mask = host_mask[None, :]
        host_logps = None
        if sampling_logps is not None:
            host_logps = np.asarray(jax.device_get(sampling_logps), dtype=np.float32)
            if host_logps.ndim == 1:
                host_logps = host_logps[None, :]

        entries: list[dict[str, np.ndarray | None]] = []
        for i in range(target_len):
            if i >= host_ids.shape[0]:
                entries.append({"token_ids": None, "sampling_logps": None})
                continue
            if host_mask is not None and i < host_mask.shape[0]:
                keep = host_mask[i]
                toks = host_ids[i][keep].astype(np.int64, copy=True)
                lps = (
                    host_logps[i][keep].astype(np.float32, copy=True)
                    if host_logps is not None and i < host_logps.shape[0]
                    else None
                )
            else:
                toks = host_ids[i].astype(np.int64, copy=True)
                lps = (
                    host_logps[i].astype(np.float32, copy=True)
                    if host_logps is not None and i < host_logps.shape[0]
                    else None
                )
            entries.append({"token_ids": toks, "sampling_logps": lps})
        return entries

    def _invoke_rollout_collective(
        self,
        *,
        state: EasyDeLState,
        prompts: list[str],
        prompt_token_ids: list[list[int]] | None = None,
        config_overrides: dict[str, tp.Any] | None = None,
    ) -> tp.Any:
        """Invoke ``self.rollout`` with synchronized inputs across all hosts."""
        if prompt_token_ids is not None:
            padded_ids, padded_mask = self._pad_prompt_token_ids(prompt_token_ids)
            effective_overrides = config_overrides
            args = getattr(self, "arguments", None)
            if args is not None:
                width = int(padded_ids.shape[-1])
                resolved_sampling = resolve_sampling(args, effective_overrides)
                requested_max_new = resolved_sampling.max_tokens
                if requested_max_new is None:
                    requested_max_new = DEFAULT_MAX_NEW_TOKENS
                clamped_max_new = clamp_new_tokens_to_bound(
                    args,
                    prompt_len=width,
                    max_new_tokens=int(requested_max_new),
                )
                if clamped_max_new < int(requested_max_new):
                    effective_overrides = dict(effective_overrides or {})
                    effective_overrides.pop("max_tokens", None)
                    effective_overrides["max_new_tokens"] = clamped_max_new
            return self.rollout(
                input_ids=padded_ids,
                attention_mask=padded_mask,
                prompts=prompts,
                state=state,
                apply_chat_template=False,
                shard_inputs=True,
                release_runtime_after_generation=False,
                config_overrides=effective_overrides,
            )
        return self.rollout(
            prompts=prompts,
            state=state,
            apply_chat_template=False,
            shard_inputs=True,
            release_runtime_after_generation=False,
            config_overrides=config_overrides,
        )

    def _make_generate_fn(
        self,
        state: EasyDeLState,
        *,
        broadcast_rpc: bool = False,
    ) -> tp.Callable[..., list[str]]:
        """Create a batched generation function for rollouts.

        Wraps ``rollout`` to accept a list of prompt strings (and optional
        exact ``prompt_token_ids``) and return a list of response strings while
        preserving exact ``completion_ids`` and ``sampling_per_token_logps`` on
        ``generate_fn.last_generation_tokens``. Supports optional
        ``temperature``, ``top_p``, ``top_k`` overrides per call so
        the questioner/verifier can use different sampling params
        than the solver.

        Args:
            state: Current model state.
            broadcast_rpc: When ``True`` on multi-host rank 0, broadcasts each
                generation request to follower hosts before invoking the
                collective ``self.rollout`` call so all hosts stay in lockstep.

        Returns:
            Function ``(prompts, *, temperature?, top_p?, top_k?, ...) -> list[str]``
        """
        state_ref: list[EasyDeLState | None] = [state]
        trainer_ref: list[AgenticMoshPitTrainer | None] = [self]

        def generate_fn(
            prompts: list[str],
            *,
            temperature: float | None = None,
            top_p: float | None = None,
            top_k: int | None = None,
            num_return_sequences: int | None = None,
            strip_thinking: bool = False,
            prompt_token_ids: list[list[int]] | None = None,
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
                prompt_token_ids: Optional exact prompt token ID rows aligned
                    with ``prompts`` to eliminate cross-turn BPE drift.

            Returns:
                A list of visible action strings, one per generation. The
                wrapper also stashes raw text / reasoning / tool-call
                metadata on ``generate_fn.last_generation_metadata`` and exact
                sampled ``token_ids`` / ``sampling_logps`` on
                ``generate_fn.last_generation_tokens``.
            """
            active_trainer = trainer_ref[0]
            active_state = state_ref[0]
            if active_trainer is None or active_state is None:
                raise RuntimeError("Rollout generate_fn has already been released after step completion.")

            overrides: dict[str, tp.Any] = {}
            if temperature is not None:
                overrides["temperature"] = temperature
            if top_p is not None:
                overrides["top_p"] = top_p
            if top_k is not None:
                overrides["top_k"] = top_k
            if num_return_sequences is not None:
                overrides["num_return_sequences"] = num_return_sequences
            resolved_overrides = overrides or None

            if broadcast_rpc:
                active_trainer._broadcast_object_from_host0(
                    {
                        "op": "generate",
                        "prompts": list(prompts),
                        "prompt_token_ids": [list(row) for row in prompt_token_ids]
                        if prompt_token_ids is not None
                        else None,
                        "overrides": resolved_overrides,
                    }
                )

            results = active_trainer._invoke_rollout_collective(
                state=active_state,
                prompts=prompts,
                prompt_token_ids=prompt_token_ids,
                config_overrides=resolved_overrides,
            )

            raw_texts = active_trainer._coerce_generation_texts(
                results.raw_text,
                fallback=results.text,
            )
            if not raw_texts:
                raw_texts = active_trainer._coerce_generation_texts(results.generation_results)
            visible_texts = active_trainer._coerce_generation_texts(
                results.text,
                fallback=raw_texts,
            )
            action_texts = list(visible_texts)
            if strip_thinking:
                action_texts = [active_trainer._strip_thinking(text) for text in action_texts]
            reasoning_records = active_trainer._coerce_optional_generation_texts(
                results.reasoning, target_len=len(action_texts)
            )
            tool_call_records = active_trainer._coerce_generation_metadata_list(
                results.tool_calls, target_len=len(action_texts)
            )
            generate_fn.last_generation_metadata = [
                {
                    "text": action_texts[i],
                    "raw_text": raw_texts[i],
                    "reasoning": reasoning_records[i],
                    "tool_calls": tool_call_records[i],
                }
                for i in range(len(action_texts))
            ]
            generate_fn.last_generation_tokens = active_trainer._extract_rollout_token_entries(
                results, len(action_texts)
            )
            return action_texts

        def _release_generate_fn() -> None:
            state_ref[0] = None
            trainer_ref[0] = None
            generate_fn.last_generation_metadata = None
            generate_fn.last_generation_tokens = None

        generate_fn._accepts_prompt_token_ids = True
        generate_fn.release = _release_generate_fn
        return generate_fn

    @staticmethod
    def _detach_rollout_generation(
        configured_envs: list[tp.Any],
        generate_fn: tp.Callable[..., list[str]],
    ) -> None:
        """Detach per-step ``generate_fn`` from environments/shared generators and clear closure refs."""
        for env in configured_envs:
            try:
                clear_env = getattr(env, "clear_generation", None)
                if callable(clear_env):
                    clear_env(generate_fn)
                else:
                    generator = getattr(env, "_generator", None)
                    if generator is not None:
                        clear_gen = getattr(generator, "clear_generation", None)
                        if callable(clear_gen):
                            clear_gen(generate_fn)
                        elif getattr(generator, "_generate_fn", None) is generate_fn:
                            set_fn = getattr(generator, "set_generate_fn", None)
                            if callable(set_fn):
                                set_fn(None)
                            else:
                                generator._generate_fn = None
                if getattr(env, "_generate_fn", None) is generate_fn:
                    env._generate_fn = None
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.debug("Failed to detach rollout generate_fn from environment: %s", exc)
        release = getattr(generate_fn, "release", None)
        if callable(release):
            release()
        else:
            if hasattr(generate_fn, "last_generation_metadata"):
                generate_fn.last_generation_metadata = None
            if hasattr(generate_fn, "last_generation_tokens"):
                generate_fn.last_generation_tokens = None

    def _run_grouped_episodes_coordinated(
        self,
        *,
        state: EasyDeLState,
        base_seed: int,
    ) -> list:
        """Run grouped rollouts with rank-0 coordination on multi-host meshes.

        On multi-host meshes (``jax.process_count() > 1``), process 0 owns all
        environment creation, resets, stepping, and termination decisions while
        broadcasting every ``self.rollout`` invocation so every host enters
        every collective generation call with identical prompts, batch counts,
        and shapes. Completed trajectories are broadcast to all hosts once the
        rollout loop finishes.
        """
        is_multihost = jax.process_count() > 1
        if not is_multihost:
            generate_fn = self._make_generate_fn(state, broadcast_rpc=False)
            configured_envs: list[tp.Any] = []

            def wrapped_env_factory():
                env = self.env_factory()
                configured_envs.append(env)
                env.configure_generation(
                    generate_fn,
                    tokenizer=self.processing_class,
                    strip_reasoning_fn=self._strip_thinking,
                )
                return self._wrap_env_with_tools(env)

            try:
                return self._rollout_manager.run_grouped_episodes(
                    env_factory=wrapped_env_factory,
                    generate_fn=generate_fn,
                    group_size=self.arguments.group_size,
                    base_seed=base_seed,
                    num_groups=self.arguments.num_env_groups,
                )
            finally:
                self._detach_rollout_generation(configured_envs, generate_fn)

        if jax.process_index() == 0:
            generate_fn = self._make_generate_fn(state, broadcast_rpc=True)
            configured_envs = []

            def wrapped_env_factory():
                env = self.env_factory()
                configured_envs.append(env)
                env.configure_generation(
                    generate_fn,
                    tokenizer=self.processing_class,
                    strip_reasoning_fn=self._strip_thinking,
                )
                return self._wrap_env_with_tools(env)

            try:
                trajectories = self._rollout_manager.run_grouped_episodes(
                    env_factory=wrapped_env_factory,
                    generate_fn=generate_fn,
                    group_size=self.arguments.group_size,
                    base_seed=base_seed,
                    num_groups=self.arguments.num_env_groups,
                )
            except BaseException as exc:
                self._broadcast_object_from_host0({"op": "error", "type": type(exc).__name__, "message": str(exc)})
                raise
            finally:
                self._detach_rollout_generation(configured_envs, generate_fn)
            self._broadcast_object_from_host0({"op": "done"})
            return self._broadcast_object_from_host0(trajectories)

        while True:
            cmd = self._broadcast_object_from_host0(None)
            op = cmd.get("op")
            if op == "generate":
                self._invoke_rollout_collective(
                    state=state,
                    prompts=cmd["prompts"],
                    prompt_token_ids=cmd.get("prompt_token_ids"),
                    config_overrides=cmd.get("overrides"),
                )
            elif op == "done":
                break
            elif op == "error":
                raise RuntimeError(f"Host 0 agentic rollout failed ({cmd.get('type')}): {cmd.get('message')}")
            else:
                raise RuntimeError(f"Unexpected multi-host rollout command: {op!r}")
        return self._broadcast_object_from_host0(None)

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
                        tools=getattr(self, "_tool_schemas", self.arguments.tool_schemas),
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
            reward_processing_class = (
                self.reward_processing_classes[idx] if self.reward_processing_classes is not None else None
            )
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
                if jax.process_count() > 1:
                    if jax.process_index() == 0:
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
                    else:
                        values = None
                    values = self._broadcast_object_from_host0(values)
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
        ``advantage_estimator``. Records auxiliary reward fields into
        ``traj.info`` whenever any active auxiliary reward function ran,
        including when its score is ``0.0``.

        Args:
            trajectories: Rollout trajectories to mutate in-place.

        Returns:
            ``(aux_rewards, breakdown)`` from the underlying scorer for
            downstream logging.
        """
        aux_rewards, reward_breakdown = self._score_auxiliary_rewards(trajectories)
        if not trajectories or not reward_breakdown:
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
        """Wrap environment with tool support if tools are configured.

        Args:
            env: Underlying environment instance.

        Returns:
            ``env`` unchanged when no tools are registered, otherwise a
            :class:`ToolEnvWrapper` carrying the trainer's tool dispatch
            configuration.
        """
        if not self._tools:
            return env
        return ToolEnvWrapper(
            env=env,
            tools=[tool.clone_for_episode() for tool in self._tools],
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
            Tuple ``(batch_dict, metrics_dict)``. ``batch_dict`` carries
            ``prompt_ids``, ``prompt_mask``, ``completion_ids``,
            ``completion_mask``, ``ref_per_token_logps``, ``advantages``,
            and ``num_items_in_batch`` ready for the GRPO loss.
            ``metrics_dict`` reports reward statistics, completion
            length, rollout/advantage/ref-logps timings, and any
            auxiliary reward breakdowns.
        """
        with capture_time() as total_time_fn:
            with capture_time() as rollout_time_fn:
                step_counter = self._rollout_step if is_train else getattr(self, "_eval_rollout_step", 0)
                base_seed = step_counter * self.arguments.num_env_groups
                try:
                    trajectories = self._run_grouped_episodes_coordinated(
                        state=state,
                        base_seed=base_seed,
                    )
                    aux_rewards, aux_reward_breakdown = self._apply_auxiliary_rewards_to_trajectories(trajectories)
                    if is_train:
                        self._rollout_step += 1
                    else:
                        self._eval_rollout_step = getattr(self, "_eval_rollout_step", 0) + 1
                finally:
                    release_runtime = getattr(self, "release_generation_runtime", None)
                    if callable(release_runtime):
                        release_runtime(state=state)
                    elif state is not None and hasattr(getattr(state, "model", None), "pause_esurge"):
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

            prompt_ids = self._to_global_replicated_array(collated["prompt_ids"])
            prompt_mask = self._to_global_replicated_array(collated["prompt_mask"])
            completion_ids = self._to_global_replicated_array(collated["completion_ids"])
            completion_attention_mask = self._to_global_replicated_array(collated["completion_attention_mask"])
            completion_mask = self._to_global_replicated_array(collated["completion_mask"])
            env_rewards_np = np.asarray(collated["rewards"], dtype=np.float32)
            env_rewards = jnp.asarray(env_rewards_np)
            step_rewards_list = collated["step_rewards_list"]
            completion_turn_ids = collated.get("completion_turn_ids")
            rollout_logps_np = collated.get("sampling_per_token_logps")
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
                    advantages, _ = compute_advantages_episode(env_rewards, group_size, self.scale_rewards)
                elif self.arguments.advantage_estimator == "gigpo":
                    advantages, _ = compute_advantages_gigpo(
                        episode_rewards=env_rewards,
                        step_rewards_list=step_rewards_list,
                        group_size=group_size,
                        episode_weight=self.arguments.episode_reward_weight,
                        step_weight=self.arguments.step_reward_weight,
                        gamma=self.arguments.step_reward_gamma,
                        scale_rewards=self.scale_rewards,
                        response_masks=collated["completion_mask"],
                        turn_ids=completion_turn_ids,
                    )
                elif self.arguments.advantage_estimator == "step_reinforce":
                    step_adv = compute_advantages_step(
                        step_rewards_list,
                        group_size,
                        self.arguments.step_reward_gamma,
                        response_masks=collated["completion_mask"],
                        turn_ids=completion_turn_ids,
                        scale_rewards=self.scale_rewards,
                    )
                    advantages = jnp.asarray(step_adv, dtype=jnp.float32)
                elif self.arguments.advantage_estimator == "agentic_reinforce":
                    advantages = jnp.asarray(
                        compute_advantages_agentic(
                            step_rewards_list,
                            collated["completion_mask"],
                            group_size,
                            self.arguments.step_reward_gamma,
                            turn_ids=completion_turn_ids,
                            scale_rewards=self.scale_rewards,
                        ),
                        dtype=jnp.float32,
                    )
                elif self.arguments.advantage_estimator == "reinforce":
                    advantages = normalize_rewards_batch(env_rewards)
                else:
                    raise ValueError(f"Unknown advantage_estimator: {self.arguments.advantage_estimator}")
                advantages = self._to_global_replicated_array(advantages, dtype=jnp.float32)
            advantage_time = advantage_time_fn()

            with capture_time() as ref_logps_time_fn:
                sequences = jnp.concatenate(
                    [
                        prompt_ids,
                        completion_ids,
                    ],
                    axis=1,
                )
                seq_mask = jnp.concatenate(
                    [
                        prompt_mask,
                        completion_attention_mask,
                    ],
                    axis=1,
                )

                if self.ref_state is None:
                    # beta == 0: KL term disabled -> skip the reference forward
                    # entirely. The placeholder is never read by the loss.
                    ref_per_token_logps = self._to_global_replicated_array(
                        np.zeros((completion_ids.shape[0], completion_ids.shape[1]), dtype=np.float32),
                        dtype=jnp.float32,
                    )
                else:
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
                                    prompt_length=int(prompt_ids.shape[-1]),
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
                            prompt_length=int(prompt_ids.shape[-1]),
                        )
                    ref_per_token_logps = self._to_global_replicated_array(ref_per_token_logps, dtype=jnp.float32)
            ref_logps_time = ref_logps_time_fn()

            rollout_logps = (
                self._to_global_replicated_array(rollout_logps_np, dtype=jnp.float32)
                if rollout_logps_np is not None
                else None
            )
            old_per_token_logps = None
            importance_sampling_ratio = None
            needs_esurge_correction = (
                is_train
                and bool(getattr(self.arguments, "esurge_importance_sampling_correction", False))
                and rollout_logps is not None
                and callable(getattr(self, "compute_state_logps", None))
            )
            if needs_esurge_correction:
                policy_model_kwargs = normalize_generation_model_kwargs(
                    None,
                    model_callable=getattr(state.model, "forward", state.model),
                )
                old_per_token_logps = self.compute_state_logps(
                    state,
                    sequences,
                    seq_mask,
                    policy_model_kwargs,
                    prompt_length=int(prompt_ids.shape[-1]),
                    logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                )
                old_per_token_logps = self._to_global_replicated_array(old_per_token_logps, dtype=jnp.float32)
                importance_sampling_ratio = self._esurge_importance_sampling_ratio(
                    old_per_token_logps,
                    rollout_logps,
                    completion_mask,
                )
                importance_sampling_ratio = self._to_global_replicated_array(
                    importance_sampling_ratio,
                    dtype=jnp.float32,
                )

            if is_train:
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

        total_time = total_time_fn()

        comp_mask_np = np.asarray(collated["completion_mask"], dtype=np.int32)
        comp_lengths_np = np.sum(comp_mask_np > 0, axis=-1).astype(np.int64)
        completion_length = jnp.sum(completion_mask, -1)
        avg_steps = float(np.mean([t.num_steps for t in trajectories])) if trajectories else 0.0
        if env_rewards_np.size and group_size > 0 and env_rewards_np.size % group_size == 0:
            per_group_std = np.nanstd(env_rewards_np.reshape(-1, group_size), axis=-1)
            frac_reward_zero_std = float(np.mean(np.isclose(per_group_std, 0.0).astype(np.float32)))
        else:
            frac_reward_zero_std = 0.0

        finite_rewards_np = env_rewards_np[np.isfinite(env_rewards_np)]
        pass_rate = float(np.mean(finite_rewards_np > 0.0)) if finite_rewards_np.size else 0.0
        max_comp_len = int(getattr(self.arguments, "max_completion_length", 0) or 0)
        traj_truncated: list[bool] = []
        for traj in trajectories:
            if traj.response_mask.size == 0:
                first_resp_idx = len(traj.input_ids)
            else:
                first_resp_idx = int(np.argmax(traj.response_mask))
                if traj.response_mask[first_resp_idx] == 0:
                    first_resp_idx = len(traj.input_ids)
            raw_comp_len = max(0, len(traj.input_ids) - first_resp_idx)
            is_trunc = bool(traj.info.get("truncated", False) or (max_comp_len > 0 and raw_comp_len > max_comp_len))
            traj_truncated.append(is_trunc)
        truncation_rate = float(np.mean(traj_truncated)) if traj_truncated else 0.0

        metrics_dict: dict[str, float | int | str] = {
            "reward_mean": float(np.nanmean(env_rewards_np)) if env_rewards_np.size else 0.0,
            "reward_std": float(np.nanstd(env_rewards_np)) if env_rewards_np.size else 0.0,
            "completion_length": _host_mean_float(completion_length),
            "rollout_time": rollout_time,
            "advantage_time": advantage_time,
            "ref_logps_time": ref_logps_time,
            "preprocessing_time": total_time,
            "avg_episode_steps": avg_steps,
            "frac_reward_zero_std": frac_reward_zero_std,
            "rl/pass_rate": pass_rate,
            "rl/truncation_rate": truncation_rate,
            "rl/zero_var_group_frac": frac_reward_zero_std,
        }
        if aux_reward_breakdown:
            metrics_dict["aux_reward_mean"] = float(np.nanmean(aux_rewards))
            for name, values in aux_reward_breakdown.items():
                metrics_dict[f"aux_reward/{name}"] = float(np.nanmean(values))

        icepop_active_ratios: np.ndarray | None = None
        if importance_sampling_ratio is not None:
            is_ratio_np = _to_host_numpy(importance_sampling_ratio)
            icepop_active_ratios, ratio_mean, ratio_max, masked_frac = _compute_icepop_active_ratios(
                is_ratio_np,
                comp_mask_np,
            )
            metrics_dict["rl/icepop_ratio_mean"] = ratio_mean
            metrics_dict["rl/icepop_ratio_max"] = ratio_max
            metrics_dict["rl/icepop_masked_frac"] = masked_frac

        if is_train:
            self._log_trajectories_to_wandb(trajectories, state)

        self._log_trajectories_to_eyvan(
            trajectories=trajectories,
            state=state,
            is_train=is_train,
            env_rewards_np=env_rewards_np,
            advantages=advantages,
            comp_mask_np=comp_mask_np,
            comp_lengths_np=comp_lengths_np,
            traj_truncated=traj_truncated,
            aux_reward_breakdown=aux_reward_breakdown,
            policy_logps=old_per_token_logps if old_per_token_logps is not None else rollout_logps,
            ref_logps=ref_per_token_logps if self.ref_state is not None else None,
            icepop_active_ratios=icepop_active_ratios,
        )

        model_batch: dict[str, jax.Array] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "num_items_in_batch": jnp.sum(completion_mask),
        }
        if rollout_logps is not None:
            model_batch["sampling_per_token_logps"] = rollout_logps
        if old_per_token_logps is not None:
            model_batch["old_per_token_logps"] = old_per_token_logps
        if importance_sampling_ratio is not None:
            model_batch["importance_sampling_ratio"] = importance_sampling_ratio

        return model_batch, metrics_dict

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

        Args:
            trajectories: Rollout trajectories produced by the rollout
                manager.
            state: Current trainer state, used to obtain the global step
                for the wandb log entry.
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
                    _host_scalar_float(jnp.sum(traj.response_mask)),
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

    def _log_trajectories_to_eyvan(
        self,
        *,
        trajectories: list,
        state: EasyDeLState,
        is_train: bool,
        env_rewards_np: np.ndarray,
        advantages: tp.Any,
        comp_mask_np: np.ndarray,
        comp_lengths_np: np.ndarray,
        traj_truncated: list[bool],
        aux_reward_breakdown: dict[str, np.ndarray] | None = None,
        policy_logps: tp.Any = None,
        ref_logps: tp.Any = None,
        icepop_active_ratios: np.ndarray | None = None,
    ) -> None:
        """Emit native Eyvan episodes, rollouts, grouped completions, and histograms for AgenticMoshPit."""
        if not trajectories:
            return
        total_turns_before = int(getattr(self, "_eyvan_total_turns", 0))
        total_action_before = int(getattr(self, "_eyvan_total_action_tokens", 0))
        args = getattr(self, "arguments", None)
        if args is None or not bool(getattr(args, "eyvan_tracking_enabled", False)):
            if is_train:
                self._eyvan_total_turns = total_turns_before + sum(max(0, int(t.num_steps)) for t in trajectories)
                self._eyvan_total_action_tokens = total_action_before + int(np.sum(comp_lengths_np))
            return
        if hasattr(args, "__dict__") and "_eyvan_runtime_context" not in args.__dict__:
            bind_ctx = getattr(self, "_bind_eyvan_context", None)
            if callable(bind_ctx):
                bind_ctx()

        try:
            fallback_step = (
                int(getattr(self, "_rollout_step", 0)) if is_train else int(getattr(self, "_eval_rollout_step", 0))
            )
            cur_step = _extract_nonneg_step(state, fallback=max(0, fallback_step - 1))
            raw_pv = getattr(self, "_async_rollout_policy_version", None)
            if raw_pv is None and state is not None:
                raw_pv = getattr(state, "policy_step", None)
            policy_version = max(0, int(raw_pv)) if raw_pv is not None else cur_step
            kind = "train" if is_train else "eval"
            default_env_name = getattr(self, "_eyvan_env_name", None) or _resolve_eyvan_env_name(
                getattr(self, "env_factory", None),
                fallback="agentic_moshpit",
            )

            # Extract advantages locally (handles both 1D episode-level and 2D token-level advantages).
            adv_host = _to_host_numpy_local_only(advantages)
            if adv_host is None:
                ep_advantages = np.zeros((len(trajectories),), dtype=np.float32)
                adv_hist_values = ep_advantages
            elif adv_host.ndim == 2 and adv_host.shape == comp_mask_np.shape:
                ep_adv_list: list[float] = []
                for idx in range(len(trajectories)):
                    row_mask = comp_mask_np[idx] > 0
                    if np.any(row_mask):
                        vals = adv_host[idx][row_mask]
                        finite = vals[np.isfinite(vals)]
                        ep_adv_list.append(float(np.mean(finite)) if finite.size else 0.0)
                    else:
                        ep_adv_list.append(0.0)
                ep_advantages = np.asarray(ep_adv_list, dtype=np.float32)
                active_adv = adv_host[comp_mask_np > 0]
                adv_hist_values = active_adv if active_adv.size > 0 else ep_advantages
            else:
                ep_advantages = adv_host.reshape(-1)[: len(trajectories)].astype(np.float32, copy=False)
                adv_hist_values = ep_advantages

            # 1. Record every trajectory as an Eyvan episode.
            cum_turns = total_turns_before
            cum_action_tokens = total_action_before
            for idx, traj in enumerate(trajectories):
                ep_steps = max(0, int(traj.num_steps))
                ep_act_tokens = max(0, int(comp_lengths_np[idx])) if idx < comp_lengths_np.size else 0
                if is_train:
                    cum_turns += ep_steps
                    cum_action_tokens += ep_act_tokens
                ep_reward = float(traj.episode_reward)
                ep_success = bool(math.isfinite(ep_reward) and ep_reward > 0.0)
                finite_step_r = [float(r) for r in traj.step_rewards if math.isfinite(float(r))]
                mean_turn_reward = float(sum(finite_step_r) / len(finite_step_r)) if finite_step_r else 0.0
                traj_env = traj.info.get("env") if isinstance(traj.info, dict) else None
                env_name = (
                    str(traj_env).strip()[:128] if isinstance(traj_env, str) and traj_env.strip() else default_env_name
                )
                extra: dict[str, float] = {
                    "action_tokens": float(ep_act_tokens),
                    "truncated": 1.0 if traj_truncated[idx] else 0.0,
                    "mean_turn_reward": mean_turn_reward,
                }
                if isinstance(traj.info, dict):
                    for extra_k in ("env_reward", "aux_reward"):
                        if extra_k in traj.info:
                            try:
                                val = float(traj.info[extra_k])
                                if math.isfinite(val):
                                    extra[extra_k] = val
                            except (TypeError, ValueError):
                                pass
                _eyvan_log_episode(
                    args,
                    ret=ep_reward,
                    length=ep_steps,
                    success=ep_success,
                    kind=kind,
                    env=env_name,
                    policy_version=policy_version,
                    learner_version=cur_step,
                    step=cur_step,
                    axes={"env_turns": float(cum_turns), "action_tokens": float(cum_action_tokens)},
                    extra=extra,
                )
            if is_train:
                self._eyvan_total_turns = cum_turns
                self._eyvan_total_action_tokens = cum_action_tokens
            batch_axes = {"env_turns": float(cum_turns), "action_tokens": float(cum_action_tokens)}

            # 2. Sample up to 8 trajectories as Eyvan rollouts (best, worst, then remaining; skip 0-step).
            eligible_indices = [
                idx
                for idx, traj in enumerate(trajectories)
                if any(t.is_response for t in traj.turns) or len(traj.step_rewards) > 0
            ]
            rollout_indices = _select_rollout_sample_indices(
                [float(t.episode_reward) for t in trajectories],
                eligible_indices,
                cap=_EYVAN_MAX_ROLLOUTS_PER_STEP,
            )
            for idx in rollout_indices:
                traj = trajectories[idx]
                resp_turns = [t for t in traj.turns if t.is_response]
                n_steps = min(max(len(resp_turns), len(traj.step_rewards)), _EYVAN_MAX_ROLLOUT_STEPS)
                if n_steps <= 0:
                    continue
                rewards_track: list[float | None] = []
                act_tokens_track: list[float | None] = []
                terminated_track: list[float | None] = []
                truncated_track: list[float | None] = []
                mean_lp_track: list[float | None] = []
                has_any_lp = False
                raw_actions: list[str] = []
                for step_i in range(n_steps):
                    if step_i < len(traj.step_rewards):
                        r_val = float(traj.step_rewards[step_i])
                    elif step_i < len(resp_turns):
                        r_val = float(resp_turns[step_i].reward)
                    else:
                        r_val = 0.0
                    rewards_track.append(r_val)
                    if step_i < len(resp_turns):
                        rt = resp_turns[step_i]
                        tok_count = float(len(rt.token_ids)) if rt.token_ids is not None else float(len(rt.content))
                        act_tokens_track.append(tok_count)
                        label_src = rt.visible_content if rt.visible_content is not None else rt.content
                        raw_actions.append(label_src)
                        if rt.sampling_logps is not None and len(rt.sampling_logps) > 0:
                            has_any_lp = True
                            mean_lp_track.append(float(np.mean(np.asarray(rt.sampling_logps, dtype=np.float64))))
                        else:
                            mean_lp_track.append(None)
                    else:
                        act_tokens_track.append(0.0)
                        raw_actions.append(f"turn_{step_i}")
                        mean_lp_track.append(None)
                    is_last = step_i == (n_steps - 1)
                    truncated_track.append(1.0 if (is_last and traj_truncated[idx]) else 0.0)
                    terminated_track.append(1.0 if (is_last and not traj_truncated[idx]) else 0.0)
                tracks: dict[str, list[float | None]] = {
                    "reward": rewards_track,
                    "action_tokens": act_tokens_track,
                    "terminated": terminated_track,
                    "truncated": truncated_track,
                }
                if has_any_lp:
                    tracks["mean_logprob"] = mean_lp_track
                action_ids, action_names = _encode_action_labels(raw_actions)
                traj_env = traj.info.get("env") if isinstance(traj.info, dict) else None
                env_name = (
                    str(traj_env).strip()[:128] if isinstance(traj_env, str) and traj_env.strip() else default_env_name
                )
                _eyvan_log_rollout(
                    args,
                    tracks=tracks,
                    actions=action_ids,
                    action_names=action_names,
                    episode_id=f"s{cur_step}-e{idx}",
                    env=env_name,
                    kind=kind,
                    policy_version=policy_version,
                    step=cur_step,
                    axes=batch_axes,
                )

            # 3. Compute optional per-trajectory KL locally (never invoking cross-host collectives).
            per_ep_kl: list[float | None] | None = None
            if ref_logps is not None and policy_logps is not None:
                ref_np = _to_host_numpy_local_only(ref_logps)
                pol_np = _to_host_numpy_local_only(policy_logps)
                if (
                    ref_np is not None
                    and pol_np is not None
                    and ref_np.shape == comp_mask_np.shape
                    and pol_np.shape == comp_mask_np.shape
                ):
                    kl_estimator = str(getattr(args, "kl_estimator", "k3") or "k3")
                    per_ep_kl = _compute_per_episode_kl_numpy(pol_np, ref_np, comp_mask_np, kl_estimator=kl_estimator)

            # 4. Log up to 8 grouped multi-turn completions.
            eff_group_size = max(1, int(getattr(args, "group_size", 1) or 1))
            weight_by_name: dict[str, float] = {}
            if aux_reward_breakdown:
                names_seq = getattr(self, "reward_func_names", None) or []
                weights_host = _to_host_numpy_local_only(getattr(self, "reward_weights", None))
                for w_idx, r_name in enumerate(names_seq):
                    if weights_host is not None and w_idx < weights_host.size:
                        weight_by_name[str(r_name)] = float(weights_host.reshape(-1)[w_idx])
                    else:
                        weight_by_name[str(r_name)] = 1.0

            for g_idx, start in enumerate(range(0, len(trajectories), eff_group_size)):
                if g_idx >= _EYVAN_MAX_COMPLETION_GROUPS_PER_STEP:
                    break
                grp_indices = list(range(start, min(start + eff_group_size, len(trajectories))))[
                    :_EYVAN_MAX_GROUP_RESPONSES
                ]
                if not grp_indices:
                    continue
                first_traj = trajectories[grp_indices[0]]
                first_resp_pos = next(
                    (ti for ti, t in enumerate(first_traj.turns) if t.is_response), len(first_traj.turns)
                )
                init_turns = first_traj.turns[:first_resp_pos]
                if init_turns:
                    prompt_text = "\n".join(f"[{t.role}] {t.content}" for t in init_turns)
                elif isinstance(first_traj.info, dict) and first_traj.info.get("question"):
                    prompt_text = str(first_traj.info["question"])
                else:
                    prompt_text = ""

                responses: list[str] = []
                for idx in grp_indices:
                    traj = trajectories[idx]
                    resp_start = next((ti for ti, t in enumerate(traj.turns) if t.is_response), 0)
                    parts: list[str] = []
                    for turn in traj.turns[resp_start:]:
                        content = turn.visible_content if turn.visible_content is not None else turn.content
                        if not turn.is_response and len(content) > _EYVAN_MAX_OBS_PREVIEW_CHARS:
                            content = content[: _EYVAN_MAX_OBS_PREVIEW_CHARS - 3] + "..."
                        parts.append(f"[{turn.role}] {content}")
                    responses.append(
                        _clip_eyvan_text(
                            "\n".join(parts) if parts else "[empty trajectory]", limit=_EYVAN_MAX_TEXT_CHARS
                        )
                    )

                if aux_reward_breakdown:
                    grp_rewards: tp.Any = {
                        "env": [
                            float(trajectories[i].info.get("env_reward", trajectories[i].episode_reward))
                            if isinstance(trajectories[i].info, dict)
                            else float(trajectories[i].episode_reward)
                            for i in grp_indices
                        ],
                    }
                    for r_name, r_vals in aux_reward_breakdown.items():
                        w = weight_by_name.get(str(r_name), 1.0)
                        skey = _sanitize_eyvan_key(f"aux/{r_name}")
                        grp_rewards[skey] = [float(np.nan_to_num(r_vals[i])) * w for i in grp_indices]
                else:
                    grp_rewards = [float(trajectories[i].episode_reward) for i in grp_indices]

                grp_adv = [float(ep_advantages[i]) for i in grp_indices]
                grp_tokens = [max(0, int(comp_lengths_np[i])) for i in grp_indices]
                grp_trunc = [bool(traj_truncated[i]) for i in grp_indices]
                grp_kl = (
                    [per_ep_kl[i] for i in grp_indices]
                    if per_ep_kl is not None and any(per_ep_kl[i] is not None for i in grp_indices)
                    else None
                )
                group_prefix = f"s{cur_step}" if is_train else f"eval-s{cur_step}"
                _eyvan_log_completions(
                    args,
                    prompt=prompt_text,
                    responses=responses,
                    rewards=grp_rewards,
                    advantages=grp_adv,
                    group=f"{group_prefix}-g{g_idx}",
                    tokens=grp_tokens,
                    kl=grp_kl,
                    truncated=grp_trunc,
                    policy_version=policy_version,
                    step=cur_step,
                )

            # 5. Log RL distributions.
            _eyvan_log_histogram(args, "rl/reward_dist", env_rewards_np, step=cur_step, axes=batch_axes)
            _eyvan_log_histogram(args, "rl/advantage_dist", adv_hist_values, step=cur_step, axes=batch_axes)
            _eyvan_log_histogram(
                args,
                "rl/episode_turns_dist",
                np.asarray([float(t.num_steps) for t in trajectories], dtype=np.float32),
                step=cur_step,
                axes=batch_axes,
            )
            _eyvan_log_histogram(
                args,
                "rl/completion_length_dist",
                comp_lengths_np.astype(np.float32),
                step=cur_step,
                axes=batch_axes,
            )
            if icepop_active_ratios is not None and icepop_active_ratios.size > 0:
                _eyvan_log_histogram(
                    args,
                    "rl/icepop_ratio_dist",
                    icepop_active_ratios,
                    step=cur_step,
                    axes=batch_axes,
                )
        except Exception as exc:
            logger.warning("AgenticMoshPit Eyvan telemetry logging failed (%s): %s", type(exc).__name__, exc)

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create a data collator for agentic training.

        Since agentic training generates data through rollouts rather
        than from a dataset, the collator is minimal — it just passes
        through seed/index information if present.

        Args:
            max_sequence_length: Unused, kept for API compatibility with
                the base trainer interface.
            truncation_mode: Unused, kept for API compatibility.

        Returns:
            A minimal :class:`GRPODataCollatorTFDS` callable suitable for
            the dummy dataset used by agentic training.
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
        """Create a Grain data collator for agentic training.

        Args:
            max_sequence_length: Unused, kept for API compatibility.
            truncation_mode: Unused, kept for API compatibility.

        Returns:
            A minimal :class:`GRPODataCollatorGrain` callable suitable for
            Grain-based agentic training.
        """
        from ..utils import GRPODataCollatorGrain

        return GRPODataCollatorGrain(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )
