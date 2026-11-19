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
"""Composition wrapper trainer enabling multi-turn Agentic RL on any EasyDeL trainer."""

from __future__ import annotations

import collections.abc
import concurrent.futures
import dataclasses
import inspect
import threading
import typing as tp

import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from easydel.utils import Registry

from .agent_flow_config import AgentFlowConfig

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class AgentFlowTurn:
    """Single assistant-environment interaction turn inside a multi-turn episode."""

    turn_index: int
    prompt_text: str
    action_text: str
    action_token_ids: list[int]
    observation_text: str
    observation_token_ids: list[int]
    reward: float
    terminated: bool
    truncated: bool
    logprobs: list[float] | None = None


@dataclasses.dataclass(frozen=True)
class AgentFlowEpisode:
    """Completed multi-turn trajectory with exact token-level action masking."""

    prompt_index: int
    group_id: int
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    action_mask: list[int]
    step_rewards: list[float]
    episode_reward: float
    turns: tuple[AgentFlowTurn, ...]
    logprobs: list[float] | None = None
    env_error: bool = False


TurnGenerateFn = tp.Callable[[tp.Any, list[int], str, int], tuple[list[int], str, list[float] | None]]
ContextManagerFn = tp.Callable[[list[AgentFlowTurn], str, AgentFlowConfig], str]
EpisodeRunnerFn = tp.Callable[
    [
        tp.Any,
        tp.Callable[[list[int], str, int], tuple[list[int], str, list[float] | None]],
        dict[str, tp.Any],
        AgentFlowConfig,
    ],
    AgentFlowEpisode,
]


_TEMPLATE_PROBE = "\u2063agent-flow-template-probe\u2063"


def agent_flow_supplied_reward(**_kwargs: tp.Any) -> list[float]:
    """Placeholder ``reward_funcs`` entry for GRPO-family trainers built inside :class:`AgentFlowTrainer`.

    GRPO-family trainers require at least one reward function at construction.
    Under ``AgentFlowTrainer`` the wrapper replaces the inner trainer's
    ``_preprocess_batch_input`` and supplies episode rewards and advantages
    itself, so the inner reward path never runs. This function only satisfies
    the constructor contract and raises if it is ever invoked.

    Raises:
        RuntimeError: Always; reaching it means rewards were requested from
            the inner trainer instead of the AgentFlow episode loop.
    """
    raise RuntimeError(
        "agent_flow_supplied_reward was called: AgentFlowTrainer supplies episode rewards itself, so the inner "
        "trainer's reward path must not run. Pass real `reward_funcs` if the inner trainer scores completions."
    )


def _is_message_list(value: tp.Any) -> bool:
    """Whether ``value`` is a non-empty conversational message list (``{"role", "content"}`` mappings)."""
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(m, collections.abc.Mapping) and "role" in m and "content" in m for m in value)
    )


def _default_context_manager(
    initial_prompt: str,
    history: list[AgentFlowTurn],
    config: AgentFlowConfig,
) -> str:
    """Format multi-turn context when non-append windowing is requested."""
    if not history:
        return initial_prompt
    if config.context_mode == "sliding_window":
        window = config.sliding_window_turns or max(1, len(history))
        kept = history[-window:]
    elif config.context_mode == "reset_with_summary":
        kept = history[-1:]
    else:
        kept = history

    parts = [initial_prompt]
    for turn in kept:
        parts.append(turn.action_text)
        if turn.observation_text:
            parts.append(turn.observation_text)
    return "".join(parts)


def _aggregate_episode_reward(step_rewards: list[float], *, reward_mode: str, gamma: float) -> float:
    """Aggregate per-turn step rewards into a scalar episode return."""
    if not step_rewards:
        return 0.0
    if reward_mode == "terminal":
        return float(step_rewards[-1])
    if reward_mode == "discounted_sum":
        total = 0.0
        discount = 1.0
        for r in step_rewards:
            total += discount * float(r)
            discount *= float(gamma)
        return float(total)
    return float(sum(step_rewards))


@dataclasses.dataclass(frozen=True)
class PreferencePairSelection:
    """Per-group outcome of turning grouped episodes into one DPO preference pair.

    Attributes:
        pairs: One entry per group: ``(chosen_index, rejected_index)`` into the flat episode list,
            or ``None`` when the group yields no pair.
        skipped_ties: Groups whose valid episodes all share the same reward.
        skipped_insufficient: Groups with fewer than two valid (finite-reward, non-``env_error``) episodes.
    """

    pairs: tuple[tuple[int, int] | None, ...]
    skipped_ties: int
    skipped_insufficient: int

    @property
    def num_pairs(self) -> int:
        """Number of groups that produced a preference pair."""
        return sum(1 for pair in self.pairs if pair is not None)


def select_group_preference_pairs(
    rewards: tp.Sequence[float],
    valid: tp.Sequence[bool | float],
    group_size: int,
) -> PreferencePairSelection:
    """Pick the highest- and lowest-reward valid episode of every group as a chosen/rejected pair.

    Episodes are laid out group-major (``group_size`` consecutive episodes per prompt). Within a
    group only valid episodes with a finite reward are candidates. The chosen episode is the first
    one reaching the maximum reward and the rejected episode the first one reaching the minimum.
    A group is skipped when it has fewer than two candidates, or when every candidate has the same
    reward (no preference signal).

    Args:
        rewards: Episode rewards, length ``num_groups * group_size``.
        valid: Truthy for episodes that may enter a pair (e.g. not ``env_error``), same length.
        group_size: Episodes per group; must be positive.

    Returns:
        A :class:`PreferencePairSelection` with one entry per group.

    Raises:
        ValueError: If ``group_size`` is not positive or the lengths are inconsistent.
    """
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"`group_size` must be positive; got {group_size}.")
    rewards_list = [float(r) for r in rewards]
    valid_list = [bool(v) for v in valid]
    if len(rewards_list) != len(valid_list):
        raise ValueError(f"`rewards` ({len(rewards_list)}) and `valid` ({len(valid_list)}) lengths differ.")
    if len(rewards_list) % group_size != 0:
        raise ValueError(f"{len(rewards_list)} episodes do not split into groups of {group_size}.")

    pairs: list[tuple[int, int] | None] = []
    skipped_ties = 0
    skipped_insufficient = 0
    for start in range(0, len(rewards_list), group_size):
        candidates = [
            idx for idx in range(start, start + group_size) if valid_list[idx] and np.isfinite(rewards_list[idx])
        ]
        if len(candidates) < 2:
            pairs.append(None)
            skipped_insufficient += 1
            continue
        chosen = max(candidates, key=lambda idx: (rewards_list[idx], -idx))
        rejected = min(candidates, key=lambda idx: (rewards_list[idx], idx))
        if rewards_list[chosen] == rewards_list[rejected]:
            pairs.append(None)
            skipped_ties += 1
            continue
        pairs.append((chosen, rejected))
    return PreferencePairSelection(
        pairs=tuple(pairs),
        skipped_ties=skipped_ties,
        skipped_insufficient=skipped_insufficient,
    )


def _pad_right_2d(
    rows: list[list[int]] | list[list[float]],
    *,
    width: int,
    pad_value: int | float,
    dtype: tp.Any,
) -> np.ndarray:
    """Truncate and right-pad sequences to fixed ``width``."""
    out = np.full((len(rows), max(1, width)), pad_value, dtype=dtype)
    for idx, seq in enumerate(rows):
        clipped = seq[:width]
        if clipped:
            out[idx, : len(clipped)] = clipped
    return out


def _pad_left_2d(
    rows: list[list[int]],
    *,
    width: int,
    pad_value: int,
    dtype: tp.Any = np.int32,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncate (from the left) and left-pad prompt sequences and masks to fixed ``width``."""
    w = max(1, width)
    ids = np.full((len(rows), w), pad_value, dtype=dtype)
    mask = np.zeros((len(rows), w), dtype=dtype)
    for idx, seq in enumerate(rows):
        clipped = seq[-w:] if len(seq) > w else seq
        if clipped:
            start = w - len(clipped)
            ids[idx, start:] = clipped
            mask[idx, start:] = 1
    return ids, mask


@Registry.register("trainer", "agent_flow")
class AgentFlowTrainer:
    """Wrap any EasyDeL trainer to run multi-turn agentic RL without rewriting algorithms.

    In default ``context_mode="append"``, ``AgentFlowTrainer`` maintains a single
    token-in-token-out (TITO) token sequence per episode: each turn generates
    directly from the running token list, appends the exact generated action
    token IDs (``action_mask=1``) and wrapped observation token IDs
    (``action_mask=0``), and trains on that exact same token sequence so
    ``old_per_token_logps`` remain aligned. Note: ``"sliding_window"`` and
    ``"reset_with_summary"`` truncate older context during generation while
    still emitting the full trajectory sequence.
    """

    def __init__(
        self,
        arguments: AgentFlowConfig | None = None,
        *,
        trainer: tp.Any = None,
        environment: tp.Any = None,
        environment_factory: tp.Any = None,
        generate_turn_fn: TurnGenerateFn | None = None,
        episode_runner: EpisodeRunnerFn | None = None,
        context_manager: ContextManagerFn | None = None,
        model: tp.Any = None,
        model_state: tp.Any = None,
        train_dataset: tp.Any = None,
        eval_dataset: tp.Any = None,
        processing_class: tp.Any = None,
        reward_funcs: tp.Any = None,
        **trainer_kwargs: tp.Any,
    ) -> None:
        self.arguments = arguments or AgentFlowConfig()
        self._generate_turn_fn = generate_turn_fn
        self._episode_runner = episode_runner
        self._context_manager = context_manager
        self._rollout_lock = threading.Lock()

        if trainer is not None:
            self.trainer = trainer
            inferred = self._infer_trainer_type(trainer)
            if inferred is not None:
                self.arguments.base_trainer_type = inferred
                if self.arguments.base_config is None and hasattr(trainer, "arguments"):
                    self.arguments.base_config = trainer.arguments
        else:
            self.trainer = self._build_inner_trainer(
                self.arguments,
                model=model,
                model_state=model_state,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=processing_class,
                reward_funcs=reward_funcs,
                **trainer_kwargs,
            )

        if hasattr(self.trainer, "arguments") and hasattr(self.trainer.arguments, "async_mode"):
            if bool(self.arguments.async_mode):
                self.trainer.arguments.async_mode = True
            else:
                self.arguments.async_mode = bool(self.trainer.arguments.async_mode)

        self._env_spec = environment_factory or environment or self.arguments.environment
        self._env_factory = self._resolve_env_factory(self._env_spec)
        self._env_accepted_keys = self._resolve_accepted_env_keys(self._env_spec)
        self._logged_episode_error = False
        self._orig_preprocess_batch_input = self.trainer._preprocess_batch_input
        self.trainer._preprocess_batch_input = self._wrapped_preprocess_batch_input

    @staticmethod
    def _resolve_accepted_env_keys(spec: tp.Any) -> frozenset[str] | None:
        """Return accepted constructor parameter names if the target env has no ``**kwargs``."""
        target_cls: type | None = None
        if isinstance(spec, str):
            target_cls = Registry.get("environment", spec)
        elif isinstance(spec, collections.abc.Mapping):
            name = spec.get("name") or spec.get("type")
            if isinstance(name, str) and name:
                target_cls = Registry.get("environment", name)
        elif isinstance(spec, type):
            target_cls = spec
        if target_cls is None:
            return None
        try:
            sig = inspect.signature(target_cls.__init__)
        except (TypeError, ValueError):
            return None
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return None
        return frozenset(name for name in sig.parameters if name != "self")

    @staticmethod
    def _infer_trainer_type(trainer: tp.Any) -> str | None:
        """Infer the registered trainer type name from an existing trainer instance."""
        trainer_cls = type(trainer)
        for name in Registry.list_implementations("trainer"):
            if Registry.get("trainer", name) is trainer_cls:
                return name
        return None

    def _build_inner_trainer(
        self,
        cfg: AgentFlowConfig,
        *,
        model: tp.Any,
        model_state: tp.Any,
        train_dataset: tp.Any,
        eval_dataset: tp.Any,
        processing_class: tp.Any,
        reward_funcs: tp.Any,
        **trainer_kwargs: tp.Any,
    ) -> tp.Any:
        trainer_cls = Registry.get_or_raise("trainer", cfg.base_trainer_type)
        base_cfg = cfg.base_config
        if base_cfg is None:
            cfg_cls = Registry.get_or_raise("trainer-arguments", cfg.base_trainer_type)
            sig = inspect.signature(cfg_cls)
            candidate_kwargs: dict[str, tp.Any] = {
                "learning_rate": cfg.learning_rate,
                "max_training_steps": cfg.max_training_steps,
                "save_directory": cfg.save_directory,
                "async_mode": cfg.async_mode,
                "max_prompt_length": cfg.max_prompt_length,
                "max_completion_length": cfg.max_completion_length,
                "total_batch_size": cfg.total_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "num_generations": cfg.num_generations,
            }
            init_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters and v is not None}
            base_cfg = cfg_cls(**init_kwargs)
        elif isinstance(base_cfg, collections.abc.Mapping):
            cfg_cls = Registry.get_or_raise("trainer-arguments", cfg.base_trainer_type)
            base_cfg = cfg_cls(**dict(base_cfg))

        ctor_kwargs: dict[str, tp.Any] = {
            "arguments": base_cfg,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": processing_class,
            **trainer_kwargs,
        }
        if model is not None:
            ctor_kwargs["model"] = model
        if model_state is not None:
            ctor_kwargs["model_state"] = model_state
        if reward_funcs is None:
            from easydel.trainers.group_relative_policy_optimization import GRPOTrainer
            from easydel.trainers.online_dpo_trainer import OnlineDPOTrainer

            if isinstance(trainer_cls, type) and issubclass(trainer_cls, (GRPOTrainer, OnlineDPOTrainer)):
                # Environment-only setup: GRPO-family and OnlineDPO constructors require a reward, but the
                # wrapper supplies episode rewards (and pairs), so pass a placeholder that is never called.
                reward_funcs = agent_flow_supplied_reward
        if reward_funcs is not None:
            ctor_kwargs["reward_funcs"] = reward_funcs
        return trainer_cls(**ctor_kwargs)

    def _resolve_env_factory(self, spec: tp.Any) -> tp.Callable[..., tp.Any]:
        if spec is None:
            raise ValueError(
                "AgentFlowTrainer requires `environment`, `environment_factory`, or `arguments.environment`."
            )
        if callable(spec) and not isinstance(spec, str):
            return spec
        from easydel.environments import KiriRuntime, SandboxPool, make_environment_factory

        endpoints = self.arguments.environment_endpoints
        if endpoints:
            runtime: tp.Any = SandboxPool(endpoints, backend=self.arguments.environment_backend)
        else:
            runtime = KiriRuntime(backend=self.arguments.environment_backend)
        if isinstance(spec, str):
            return make_environment_factory(spec, runtime=runtime)
        if isinstance(spec, collections.abc.Mapping):
            merged = dict(spec)
            merged.setdefault("runtime", runtime)
            return make_environment_factory(merged)
        return lambda **_kw: spec

    def _generation_prompt_suffix(self) -> str | None:
        """Return the text the chat template emits after a user message's content (with generation prompt).

        Used to recognise prompts that were already templated upstream (e.g. decoded from a tokenized
        GRPO batch), so the template is not applied twice. ``None`` when no template is available.
        """
        if "_cached_generation_suffix" in self.__dict__:
            return self.__dict__["_cached_generation_suffix"]
        suffix: str | None = None
        tok = getattr(self.trainer, "processing_class", None)
        apply_fn = getattr(tok, "apply_chat_template", None)
        if callable(apply_fn):
            try:
                rendered = str(
                    apply_fn([{"role": "user", "content": _TEMPLATE_PROBE}], tokenize=False, add_generation_prompt=True)
                )
                pos = rendered.find(_TEMPLATE_PROBE)
                if pos >= 0:
                    tail = rendered[pos + len(_TEMPLATE_PROBE) :].rstrip()
                    suffix = tail or None
            except Exception:
                suffix = None
        self.__dict__["_cached_generation_suffix"] = suffix
        return suffix

    def _is_chat_templated(self, prompt: str) -> bool:
        """Whether ``prompt`` already ends with the chat template's generation-prompt markup."""
        suffix = self._generation_prompt_suffix()
        return bool(suffix) and prompt.rstrip().endswith(suffix)

    def _format_initial_prompt(self, prompt: str) -> str:
        if self.arguments.skip_apply_chat_template:
            return prompt
        if self._is_chat_templated(prompt):
            # Prompt recovered from a tokenized batch is already templated; templating again would nest it.
            return prompt
        tok = getattr(self.trainer, "processing_class", None)
        apply_fn = getattr(tok, "apply_chat_template", None)
        if callable(apply_fn):
            try:
                formatted = apply_fn(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return str(formatted)
            except Exception:
                return prompt
        return prompt

    def _format_initial_messages(self, messages: list[dict[str, str]]) -> str:
        """Render a conversational prompt with the chat template (generation prompt appended).

        Falls back to newline-joined message contents when templating is skipped or unavailable.
        """
        plain = "\n".join(m["content"] for m in messages)
        if self.arguments.skip_apply_chat_template:
            return plain
        tok = getattr(self.trainer, "processing_class", None)
        apply_fn = getattr(tok, "apply_chat_template", None)
        if callable(apply_fn):
            try:
                return str(apply_fn(list(messages), tokenize=False, add_generation_prompt=True))
            except Exception:
                return plain
        return plain

    def _render_observation_delta(
        self,
        raw_obs: str,
        *,
        history: list[dict[str, str]],
        ends_with_eos: bool,
    ) -> str | None:
        """Render an observation as the chat-template delta appended to the running conversation.

        Computes ``render(history + [obs as user], add_generation_prompt=True)`` minus ``render(history)``
        as a string-prefix difference, so template-level preambles (default system blocks, BOS) are never
        re-emitted mid-trajectory. ``history`` ends with the assistant turn that was just generated; the
        template's closing markup for that turn (everything rendered after the action text) is prepended
        to the delta, minus the EOS token when the generated action already ended with it.

        Args:
            raw_obs: Observation text from the environment.
            history: Conversation so far as ``{"role", "content"}`` messages, ending with the assistant turn.
            ends_with_eos: Whether the generated action token ids ended with the tokenizer EOS id.

        Returns:
            The text to append after the action tokens, or ``None`` when the template cannot be rendered
            or the prefix check fails (callers then fall back to standalone rendering).
        """
        tok = getattr(self.trainer, "processing_class", None)
        apply_fn = getattr(tok, "apply_chat_template", None)
        if not callable(apply_fn) or not history or history[-1].get("role") != "assistant":
            return None
        try:
            base = str(apply_fn(list(history), tokenize=False, add_generation_prompt=False))
            full = str(
                apply_fn(
                    [*history, {"role": "user", "content": raw_obs}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            return None
        if not full.startswith(base):
            return None
        delta = full[len(base) :]
        eos = getattr(tok, "eos_token", None) or ""
        # Locate the assistant turn's closing markup with a probe so short action texts that also occur
        # inside template markup cannot be mistaken for the action.
        closing: str | None = None
        try:
            probed = str(
                apply_fn(
                    [*history[:-1], {"role": "assistant", "content": _TEMPLATE_PROBE}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            pos = probed.find(_TEMPLATE_PROBE)
            if pos >= 0:
                closing = probed[pos + len(_TEMPLATE_PROBE) :]
        except Exception:
            closing = None
        if closing is not None:
            if ends_with_eos and eos and closing.startswith(eos):
                closing = closing[len(eos) :]
        else:
            # The template rewrote the action (e.g. stripped reasoning); only close the turn with EOS.
            closing = "" if ends_with_eos else eos
        return f"{closing}{delta}"

    def _format_observation_turn(
        self,
        raw_obs: str,
        *,
        ends_with_eos: bool,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if self.arguments.skip_apply_chat_template:
            return f"\n[Observation]: {raw_obs}\n"
        if history is not None:
            delta = self._render_observation_delta(raw_obs, history=history, ends_with_eos=ends_with_eos)
            if delta is not None:
                return delta
        tok = getattr(self.trainer, "processing_class", None)
        apply_fn = getattr(tok, "apply_chat_template", None)
        if callable(apply_fn):
            try:
                rendered = str(
                    apply_fn(
                        [{"role": "user", "content": raw_obs}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                bos = getattr(tok, "bos_token", None)
                if isinstance(bos, str) and bos and rendered.startswith(bos):
                    rendered = rendered[len(bos) :]
                eos = getattr(tok, "eos_token", None) or ""
                prefix = "\n" if ends_with_eos else f"{eos}\n"
                return f"{prefix}{rendered}"
            except Exception:
                pass
        return f"\n[Observation]: {raw_obs}\n"

    def _encode_text(self, text: str) -> list[int]:
        tok = getattr(self.trainer, "processing_class", None)
        if tok is None:
            return [ord(ch) for ch in text]
        encoded = tok(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, collections.abc.Mapping) else encoded
        if ids and isinstance(ids[0], list):
            return [int(x) for x in ids[0]]
        return [int(x) for x in ids]

    def _decode_ids(self, ids: list[int]) -> str:
        tok = getattr(self.trainer, "processing_class", None)
        if tok is None:
            return "".join(chr(max(32, min(126, int(i)))) for i in ids)
        return str(tok.decode(ids, skip_special_tokens=True))

    def _default_generate_turn(
        self,
        state: tp.Any,
        prompt_ids: list[int],
        prompt_text: str,
        turn_index: int,
    ) -> tuple[list[int], str, list[float] | None]:
        if self._generate_turn_fn is not None:
            return self._generate_turn_fn(state, prompt_ids, prompt_text, turn_index)

        rollout_fn = getattr(self.trainer, "rollout", None)
        if callable(rollout_fn):
            arr = jnp.asarray([prompt_ids], dtype=jnp.int32)
            mask = jnp.ones_like(arr, dtype=jnp.int32)
            # Serialize rollout engine calls across worker threads while env I/O runs concurrently.
            overrides: dict[str, tp.Any] = {"num_return_sequences": 1}
            if self.arguments.per_turn_max_tokens is not None:
                overrides["max_new_tokens"] = int(self.arguments.per_turn_max_tokens)
            with self._rollout_lock:
                results = rollout_fn(
                    input_ids=arr,
                    attention_mask=mask,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                    release_runtime_after_generation=False,
                    config_overrides=overrides,
                )
            c_row = np.asarray(results.completion_ids[0], dtype=np.int32)
            if getattr(results, "completion_mask", None) is not None:
                m_row = np.asarray(results.completion_mask[0], dtype=np.int32)
            else:
                pad_id = int(getattr(getattr(self.trainer, "processing_class", None), "pad_token_id", 0) or 0)
                m_row = (c_row != pad_id).astype(np.int32)
            kept = [int(t) for t, m in zip(c_row.tolist(), m_row.tolist(), strict=True) if m]
            lp_list: list[float] | None = None
            raw_lps = getattr(results, "sampling_per_token_logps", None)
            if raw_lps is None:
                raw_lps = getattr(results, "logprobs", None)
            if raw_lps is not None:
                lp_row = np.asarray(raw_lps[0], dtype=np.float32)
                lp_list = [float(x) for x, m in zip(lp_row.tolist(), m_row.tolist(), strict=True) if m]
            return kept, self._decode_ids(kept), lp_list

        raise RuntimeError(
            "AgentFlowTrainer requires either a `generate_turn_fn` callback or an inner trainer with `rollout(...)`."
        )

    def _instantiate_env(self, task: dict[str, tp.Any]) -> tp.Any:
        if not isinstance(task, dict):
            return self._env_factory()
        if self._env_accepted_keys is not None:
            filtered = {k: v for k, v in task.items() if k in self._env_accepted_keys}
            return self._env_factory(**filtered)
        return self._env_factory(**task)

    def _run_episode(
        self,
        state: tp.Any,
        task: dict[str, tp.Any],
        prompt_index: int,
        group_id: int,
    ) -> AgentFlowEpisode:
        # Let constructor errors (TypeError/ValueError) raise loudly rather than hiding bugs.
        env = self._instantiate_env(task)
        try:
            if self._episode_runner is not None:
                return self._episode_runner(
                    env,
                    lambda p_ids, p_txt, t_idx: self._default_generate_turn(state, p_ids, p_txt, t_idx),
                    task,
                    self.arguments,
                )

            reset_res = env.reset()
            prompt_source = reset_res.observation or task.get("prompt", "")
            if _is_message_list(prompt_source):
                # Conversational prompt: render the messages once and mirror them in the chat history.
                prompt_messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt_source]
                initial_prompt = self._format_initial_messages(prompt_messages)
            else:
                raw_prompt = str(prompt_source)
                prompt_messages = [{"role": "user", "content": raw_prompt}]
                initial_prompt = self._format_initial_prompt(raw_prompt)
            initial_prompt_ids = self._encode_text(initial_prompt)

            if reset_res.info.get("env_error"):
                return AgentFlowEpisode(
                    prompt_index=prompt_index,
                    group_id=group_id,
                    prompt_token_ids=initial_prompt_ids,
                    completion_token_ids=[],
                    action_mask=[],
                    step_rewards=[0.0],
                    episode_reward=0.0,
                    turns=(),
                    logprobs=[],
                    env_error=True,
                )

            turns: list[AgentFlowTurn] = []
            running_ids: list[int] = list(initial_prompt_ids)
            completion_ids: list[int] = []
            action_mask: list[int] = []
            all_logprobs: list[float] = []
            has_logprobs = True
            step_rewards: list[float] = []
            env_failed = False
            # Message-level mirror of the trajectory, used to render observations as chat-template deltas.
            chat_history: list[dict[str, str]] = list(prompt_messages)

            for turn_idx in range(self.arguments.max_turns):
                if self._context_manager is not None:
                    ctx_text = self._context_manager(turns, initial_prompt, self.arguments)
                    ctx_ids = self._encode_text(ctx_text)
                elif self.arguments.context_mode == "append":
                    # Single TITO token sequence: generate directly from running_ids without re-tokenizing.
                    ctx_ids = list(running_ids)
                    ctx_text = self._decode_ids(ctx_ids)
                else:
                    ctx_text = _default_context_manager(initial_prompt, turns, self.arguments)
                    ctx_ids = self._encode_text(ctx_text)

                act_ids, act_text, act_lps = self._default_generate_turn(state, ctx_ids, ctx_text, turn_idx)

                try:
                    step_res = env.step(act_text)
                except Exception as exc:
                    show_tb = not self._logged_episode_error
                    self._logged_episode_error = True
                    logger.warning("AgentFlow env.step raised %s: %s", type(exc).__name__, exc, exc_info=show_tb)
                    env_failed = True
                    step_rewards.append(0.0)
                    completion_ids.extend(act_ids)
                    action_mask.extend([0] * len(act_ids))
                    all_logprobs.extend([0.0] * len(act_ids))
                    break

                if step_res.info.get("env_error"):
                    env_failed = True
                raw_obs = str(step_res.observation or "")
                is_last_turn = (turn_idx + 1) >= self.arguments.max_turns
                raw_obs = str(step_res.observation or "").strip()
                obs_cap = self._resolve_obs_token_cap()
                if obs_cap is not None and raw_obs and not (step_res.terminated or step_res.truncated or is_last_turn):
                    # Reserve ~24 tokens for chat wrapper markers so closing assistant header isn't truncated.
                    raw_ids = self._encode_text(raw_obs)
                    wrapper_reserve = 0 if self.arguments.skip_apply_chat_template else 24
                    raw_limit = max(0, obs_cap - wrapper_reserve)
                    if len(raw_ids) > raw_limit:
                        raw_obs = self._decode_ids(raw_ids[:raw_limit])

                eos_id = getattr(getattr(self.trainer, "processing_class", None), "eos_token_id", None)
                ends_with_eos = bool(act_ids and eos_id is not None and int(act_ids[-1]) == int(eos_id))
                chat_history.append({"role": "assistant", "content": act_text})
                obs_text = (
                    ""
                    if (step_res.terminated or step_res.truncated or is_last_turn or not raw_obs)
                    else self._format_observation_turn(raw_obs, ends_with_eos=ends_with_eos, history=chat_history)
                )
                if obs_text:
                    chat_history.append({"role": "user", "content": raw_obs})
                obs_ids = self._encode_text(obs_text) if obs_text else []
                if obs_cap is not None and len(obs_ids) > obs_cap and self.arguments.skip_apply_chat_template:
                    obs_ids = obs_ids[:obs_cap]
                    obs_text = self._decode_ids(obs_ids)
                step_r = float(step_res.reward)
                step_rewards.append(step_r)

                running_ids.extend(act_ids)
                completion_ids.extend(act_ids)
                action_mask.extend([1] * len(act_ids))
                if act_lps is not None and len(act_lps) == len(act_ids):
                    all_logprobs.extend(act_lps)
                else:
                    has_logprobs = False
                    all_logprobs.extend([0.0] * len(act_ids))

                if obs_ids:
                    running_ids.extend(obs_ids)
                    completion_ids.extend(obs_ids)
                    action_mask.extend([0] * len(obs_ids))
                    all_logprobs.extend([0.0] * len(obs_ids))

                turns.append(
                    AgentFlowTurn(
                        turn_index=turn_idx,
                        prompt_text=ctx_text,
                        action_text=act_text,
                        action_token_ids=act_ids,
                        observation_text=obs_text,
                        observation_token_ids=obs_ids,
                        reward=step_r,
                        terminated=bool(step_res.terminated),
                        truncated=bool(step_res.truncated),
                        logprobs=act_lps,
                    )
                )
                if step_res.terminated or step_res.truncated or env_failed:
                    break

            if env_failed:
                action_mask = [0] * len(action_mask)
                ep_reward = 0.0
            else:
                ep_reward = _aggregate_episode_reward(
                    step_rewards,
                    reward_mode=self.arguments.reward_mode,
                    gamma=float(self.arguments.gamma),
                )
            return AgentFlowEpisode(
                prompt_index=prompt_index,
                group_id=group_id,
                prompt_token_ids=initial_prompt_ids,
                completion_token_ids=completion_ids,
                action_mask=action_mask,
                step_rewards=step_rewards,
                episode_reward=ep_reward,
                turns=tuple(turns),
                logprobs=all_logprobs if has_logprobs else None,
                env_error=env_failed,
            )
        except Exception as exc:
            show_tb = not self._logged_episode_error
            self._logged_episode_error = True
            logger.warning("AgentFlow episode failed (%s): %s", type(exc).__name__, exc, exc_info=show_tb)
            raise
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()

    def _decode_prompt_rows(self, ids: tp.Any, mask: tp.Any) -> list[str]:
        """Decode tokenized prompt rows back to text, keeping template special tokens.

        Padding is removed through ``mask`` (when given) instead of ``skip_special_tokens``, so chat-template
        markers survive decoding and :meth:`_format_initial_prompt` can recognise an already-templated prompt.
        """
        arr = np.asarray(ids)
        mask_arr = None if mask is None else np.asarray(mask)
        tok = getattr(self.trainer, "processing_class", None)
        rows: list[str] = []
        for r, row in enumerate(arr):
            kept = [int(t) for c, t in enumerate(row.tolist()) if mask_arr is None or int(mask_arr[r, c])]
            if tok is None:
                rows.append(self._decode_ids(kept))
                continue
            try:
                rows.append(str(tok.decode(kept, skip_special_tokens=False)))
            except TypeError:
                rows.append(str(tok.decode(kept)))
        return rows

    def _extract_tasks(self, batch: collections.abc.Mapping[str, tp.Any]) -> list[dict[str, tp.Any]]:
        if "tasks" in batch and isinstance(batch["tasks"], list):
            return [dict(t) if isinstance(t, collections.abc.Mapping) else {"prompt": str(t)} for t in batch["tasks"]]

        prompts = batch.get("prompts")
        raw_prompts = batch.get("prompt")
        if prompts is None and isinstance(raw_prompts, (list, tuple)) and raw_prompts:
            # Raw prompt text / conversational messages kept through preprocessing take priority over
            # decoding token ids; anything else (e.g. arrays) falls through to the tokenized columns.
            if all(isinstance(p, str) for p in raw_prompts) or all(_is_message_list(p) for p in raw_prompts):
                prompts = list(raw_prompts)
        if prompts is None and "prompt_ids" in batch:
            prompts = self._decode_prompt_rows(batch["prompt_ids"], batch.get("prompt_mask"))
        if prompts is None and "input_ids" in batch:
            prompts = self._decode_prompt_rows(batch["input_ids"], batch.get("attention_mask"))
        if prompts is None:
            prompts = [""]

        n = len(prompts)
        tasks: list[dict[str, tp.Any]] = []
        for idx in range(n):
            task: dict[str, tp.Any] = {"prompt": prompts[idx]}
            for k, v in batch.items():
                if k in {"prompts", "prompt", "prompt_ids", "prompt_mask", "input_ids", "attention_mask"}:
                    continue
                if isinstance(v, (list, tuple)) and len(v) == n:
                    task[k] = v[idx]
            tasks.append(task)
        return tasks

    def _resolve_group_size(self, is_train: bool) -> int:
        if self._is_online_dpo():
            # Preference pairs need at least two episodes per group, in training and evaluation alike.
            if self.arguments.group_size is not None:
                return max(2, int(self.arguments.group_size))
            inner_args = getattr(self.trainer, "arguments", None)
            val = getattr(inner_args, "generation_num_return_sequences", None)
            return max(2, int(val)) if val is not None else 2
        if not is_train:
            return 1
        if self.arguments.group_size is not None:
            return int(self.arguments.group_size)
        inner_args = getattr(self.trainer, "arguments", None)
        for attr in ("num_generations", "num_return_sequences"):
            val = getattr(inner_args, attr, None)
            if val is not None and int(val) > 0:
                return int(val)
        return 1

    def _resolve_widths(self) -> tuple[int, int]:
        inner_args = getattr(self.trainer, "arguments", None)
        p_len = int(getattr(inner_args, "max_prompt_length", None) or self.arguments.max_prompt_length)
        c_len = int(getattr(inner_args, "max_completion_length", None) or self.arguments.max_completion_length)
        return max(1, p_len), max(1, c_len)

    def _resolve_obs_token_cap(self) -> int | None:
        if self.arguments.max_obs_tokens_per_turn is not None:
            return max(0, int(self.arguments.max_obs_tokens_per_turn))
        if self.arguments.per_turn_max_tokens is not None and self.arguments.max_turns > 1:
            _, comp_width = self._resolve_widths()
            remaining = comp_width - int(self.arguments.max_turns) * int(self.arguments.per_turn_max_tokens)
            return max(0, remaining // max(1, int(self.arguments.max_turns) - 1))
        return None

    def _compute_advantages(self, rewards: np.ndarray, valid_mask: np.ndarray, group_size: int) -> np.ndarray:
        """Compute per-episode advantages with statistics taken over valid (non-``env_error``) rows only.

        Failed episodes never enter a mean or standard deviation and always receive a zero advantage.
        With ``group_size > 1`` each group is centred (and, unless ``scale_rewards == "none"``, scaled)
        by the mean/std of its valid rows; a group with fewer than two valid rows gets zero advantage.
        ``scale_rewards == "batch"`` uses the valid rows of the whole batch instead (zero if fewer than
        two). With ``group_size <= 1`` (PPO/VAPO, single rollouts) raw rewards are kept unless batch
        scaling is requested.

        Args:
            rewards: Episode rewards, shape ``[num_episodes]``.
            valid_mask: ``1.0`` for valid episodes and ``0.0`` for failed ones, same shape.
            group_size: Episodes per prompt group.

        Returns:
            Float32 advantages of shape ``[num_episodes]``.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=np.float64) > 0.0
        out = np.zeros(rewards.shape, dtype=np.float64)
        scale_rewards = str(getattr(getattr(self.trainer, "arguments", None), "scale_rewards", "group")).lower()

        def _normalize(idx: np.ndarray, *, scale: bool) -> None:
            sel = idx[valid[idx]]
            if sel.size < 2:
                return
            vals = rewards[sel]
            centred = vals - vals.mean()
            out[sel] = centred / (vals.std() + 1e-4) if scale else centred

        all_idx = np.arange(rewards.shape[0])
        # With group_size == 1 (e.g. PPO/VAPO or single-rollout), group-centering would zero all rows.
        if group_size <= 1:
            if scale_rewards == "batch":
                _normalize(all_idx, scale=True)
                return out.astype(np.float32)
            return np.where(valid, rewards, 0.0).astype(np.float32)

        if scale_rewards == "batch":
            _normalize(all_idx, scale=True)
            return out.astype(np.float32)

        num_groups = max(1, rewards.shape[0] // max(1, group_size))
        for g in range(num_groups):
            _normalize(all_idx[g * group_size : (g + 1) * group_size], scale=scale_rewards != "none")
        return out.astype(np.float32)

    def _is_online_dpo(self) -> bool:
        from easydel.trainers.online_dpo_trainer import OnlineDPOTrainer

        return isinstance(self.trainer, OnlineDPOTrainer)

    def _build_preference_pair_batch(
        self,
        episodes: list[AgentFlowEpisode],
        group_size: int,
        *,
        prompt_ids_np: np.ndarray,
        prompt_mask_np: np.ndarray,
        comp_ids_np: np.ndarray,
        action_mask_np: np.ndarray,
        comp_attn_np: np.ndarray,
    ) -> tuple[dict[str, tp.Any], dict[str, float]]:
        """Turn grouped episodes into one DPO chosen/rejected pair per group.

        Each group contributes its highest-reward valid episode as ``chosen`` and its lowest-reward
        valid episode as ``rejected`` (see :func:`select_group_preference_pairs`). Completions hold
        the whole trajectory (actions and observations) under the attention mask, while
        ``chosen_loss_mask`` / ``rejected_loss_mask`` equal the action masks, so only model-generated
        tokens enter the DPO log-prob sums.

        The batch always has one row per group so the compiled step keeps a static shape: a group
        without a usable pair (ties, fewer than two valid episodes, or differing prompts) keeps a row
        whose loss masks are all zero. Such a row contributes no gradient; it adds the constant
        zero-log-ratio loss to the reported mean.

        Returns:
            ``(batch, metrics)`` in the DPO key layout (``prompt_*``, ``chosen_*``, ``rejected_*``,
            with completion-only ``*_input_ids``) and ``agent_flow/dpo_*`` pair counters.
        """
        rewards = [float(ep.episode_reward) for ep in episodes]
        valid = [not ep.env_error for ep in episodes]
        selection = select_group_preference_pairs(rewards, valid, group_size)

        num_groups = len(selection.pairs)
        chosen_rows: list[int] = []
        rejected_rows: list[int] = []
        use_pair: list[bool] = []
        skipped_prompt_mismatch = 0
        for g, pair in enumerate(selection.pairs):
            if pair is not None:
                c, r = pair
                if np.array_equal(prompt_ids_np[c], prompt_ids_np[r]) and np.array_equal(
                    prompt_mask_np[c], prompt_mask_np[r]
                ):
                    chosen_rows.append(c)
                    rejected_rows.append(r)
                    use_pair.append(True)
                    continue
                skipped_prompt_mismatch += 1
            filler = g * group_size
            chosen_rows.append(filler)
            rejected_rows.append(filler)
            use_pair.append(False)

        c_idx = np.asarray(chosen_rows, dtype=np.int64)
        r_idx = np.asarray(rejected_rows, dtype=np.int64)
        keep = np.asarray(use_pair, dtype=np.int32)[:, None]

        batch = {
            "prompt_input_ids": jnp.asarray(prompt_ids_np[c_idx]),
            "prompt_attention_mask": jnp.asarray(prompt_mask_np[c_idx]),
            "chosen_input_ids": jnp.asarray(comp_ids_np[c_idx]),
            "chosen_attention_mask": jnp.asarray(comp_attn_np[c_idx]),
            "chosen_loss_mask": jnp.asarray(action_mask_np[c_idx] * keep),
            "rejected_input_ids": jnp.asarray(comp_ids_np[r_idx]),
            "rejected_attention_mask": jnp.asarray(comp_attn_np[r_idx]),
            "rejected_loss_mask": jnp.asarray(action_mask_np[r_idx] * keep),
        }
        num_pairs = int(sum(use_pair))
        rewards_np = np.asarray(rewards, dtype=np.float64)
        used = np.asarray(use_pair, dtype=bool)
        metrics = {
            "agent_flow/dpo_pairs": float(num_pairs),
            "agent_flow/dpo_groups": float(num_groups),
            "agent_flow/dpo_skipped_ties": float(selection.skipped_ties),
            "agent_flow/dpo_skipped_insufficient_valid": float(selection.skipped_insufficient),
            "agent_flow/dpo_skipped_prompt_mismatch": float(skipped_prompt_mismatch),
            "agent_flow/dpo_reward_margin": (
                float(np.mean(rewards_np[c_idx[used]] - rewards_np[r_idx[used]])) if num_pairs else 0.0
            ),
        }
        return batch, metrics

    def _is_ppo_family(self) -> bool:
        from easydel.trainers.proximal_policy_optimization_trainer import PPOTrainer

        return isinstance(self.trainer, PPOTrainer)

    def _is_distillation_or_preference_family(self) -> bool:
        from easydel.trainers.generalized_knowledge_distillation_trainer import GKDTrainer
        from easydel.trainers.on_policy_distillation_trainer import OnPolicyDistillationTrainer
        from easydel.trainers.sdft_trainer import SDFTTrainer
        from easydel.trainers.self_distillation_policy_optimization import SDPOTrainer
        from easydel.trainers.seq_kd_trainer import SeqKDTrainer
        from easydel.trainers.ssd_trainer import SSDTrainer

        return isinstance(
            self.trainer,
            (
                GKDTrainer,
                OnPolicyDistillationTrainer,
                SDFTTrainer,
                SDPOTrainer,
                SeqKDTrainer,
                SSDTrainer,
            ),
        )

    def _adapt_episodes_to_trainer(
        self,
        episodes: list[AgentFlowEpisode],
        group_size: int,
        *,
        state: tp.Any = None,
        is_train: bool = True,
    ) -> tuple[dict[str, tp.Any], dict[str, float]]:
        pad_id = int(getattr(getattr(self.trainer, "processing_class", None), "pad_token_id", 0) or 0)
        prompt_width, comp_width = self._resolve_widths()

        prompt_rows = [ep.prompt_token_ids for ep in episodes]
        comp_rows = [ep.completion_token_ids for ep in episodes]
        act_rows = [([0] * len(ep.action_mask) if ep.env_error else ep.action_mask) for ep in episodes]
        comp_attn_rows = [
            ([0] * len(ep.completion_token_ids) if ep.env_error else [1] * len(ep.completion_token_ids))
            for ep in episodes
        ]

        prompt_ids_np, prompt_mask_np = _pad_left_2d(prompt_rows, width=prompt_width, pad_value=pad_id)
        comp_ids_np = _pad_right_2d(comp_rows, width=comp_width, pad_value=pad_id, dtype=np.int32)
        comp_mask_np = _pad_right_2d(act_rows, width=comp_width, pad_value=0, dtype=np.int32)
        comp_attn_np = _pad_right_2d(comp_attn_rows, width=comp_width, pad_value=0, dtype=np.int32)

        prompt_ids = jnp.asarray(prompt_ids_np)
        prompt_mask = jnp.asarray(prompt_mask_np)
        completion_ids = jnp.asarray(comp_ids_np)
        completion_mask = jnp.asarray(comp_mask_np)
        completion_attention_mask = jnp.asarray(comp_attn_np)

        rewards_np = np.asarray([ep.episode_reward for ep in episodes], dtype=np.float32)
        valid_np = np.asarray([0.0 if ep.env_error else 1.0 for ep in episodes], dtype=np.float32)
        advantages_np = self._compute_advantages(rewards_np, valid_np, group_size)

        clipped_count = sum(
            1 for ep in episodes if len(ep.prompt_token_ids) > prompt_width or len(ep.completion_token_ids) > comp_width
        )
        metrics = {
            "agent_flow/mean_reward": float(np.mean(rewards_np)) if len(rewards_np) else 0.0,
            "agent_flow/mean_turns": float(np.mean([len(ep.turns) for ep in episodes])) if episodes else 0.0,
            "agent_flow/action_tokens": float(np.sum(comp_mask_np)),
            "agent_flow/env_errors": float(sum(1 for ep in episodes if ep.env_error)),
            "agent_flow/clipped_episodes": float(clipped_count),
        }

        has_lps = all(ep.logprobs is not None for ep in episodes)
        lp_arr = (
            jnp.asarray(
                _pad_right_2d(
                    [ep.logprobs or [] for ep in episodes],
                    width=comp_width,
                    pad_value=0.0,
                    dtype=np.float32,
                )
            )
            if has_lps
            else None
        )

        if self._is_online_dpo():
            pair_batch, pair_metrics = self._build_preference_pair_batch(
                episodes,
                group_size,
                prompt_ids_np=prompt_ids_np,
                prompt_mask_np=prompt_mask_np,
                comp_ids_np=comp_ids_np,
                action_mask_np=comp_mask_np,
                comp_attn_np=comp_attn_np,
            )
            metrics.update(pair_metrics)
            return pair_batch, metrics

        if self._is_ppo_family():
            input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            attention_mask = jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
            adv_2d = jnp.broadcast_to(jnp.asarray(advantages_np)[:, None], completion_ids.shape)
            ret_2d = jnp.broadcast_to(jnp.asarray(rewards_np)[:, None], completion_ids.shape)
            pos_thresh = float(getattr(getattr(self.trainer, "arguments", None), "positive_reward_threshold", 0.0))
            positive_mask = jnp.asarray((rewards_np > pos_thresh) & (valid_np > 0.0), dtype=jnp.float32)
            old_values = jnp.zeros_like(ret_2d)
            if lp_arr is not None:
                old_logps = lp_arr
            else:
                # No sampling logprobs: score the trajectory under the current policy (as PPOTrainer does)
                # so the clipped ratio starts at 1 instead of exp(new_logps - 0).
                score_fn = getattr(self.trainer, "_score_rollout_policy_and_values", None)
                if not callable(score_fn) or state is None:
                    raise RuntimeError(
                        "AgentFlowTrainer with a PPO-family trainer needs per-token sampling logprobs from "
                        "`generate_turn_fn`/rollout, or a trainer that can score rollouts "
                        "(`_score_rollout_policy_and_values`) and a policy state."
                    )
                old_logps, scored_values = score_fn(state, input_ids, attention_mask)
                old_logps = jnp.asarray(old_logps)[:, -completion_ids.shape[1] :]
                if scored_values is not None:
                    old_values = jnp.asarray(scored_values)[:, -completion_ids.shape[1] :]
            out: dict[str, tp.Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": completion_mask,
                "rewards": jnp.asarray(rewards_np),
                "returns": ret_2d,
                "advantages": adv_2d,
                "old_values": old_values,
                "old_logps": old_logps,
                "positive_mask": positive_mask,
            }
            if lp_arr is not None:
                out["ref_logps"] = lp_arr
            return out, metrics

        if self._is_distillation_or_preference_family():
            input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            attention_mask = jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
            full_completion_mask = jnp.concatenate([jnp.zeros_like(prompt_mask), completion_mask], axis=1)
            out = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": full_completion_mask,
                "action_mask": full_completion_mask,
                "rewards": jnp.asarray(rewards_np),
            }
            return out, metrics

        # GRPO / GMPO / GDPO / REINFORCE++ / GSPO / GFPO / RLOO / DPPO / PAPO family.
        out = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "completion_attention_mask": completion_attention_mask,
            "advantages": jnp.asarray(advantages_np),
            "environment_rewards": jnp.asarray(rewards_np),
            "num_items_in_batch": jnp.sum(completion_mask),
        }
        if lp_arr is not None:
            out["sampling_per_token_logps"] = lp_arr
            out["old_per_token_logps"] = lp_arr

        inner_args = getattr(self.trainer, "arguments", None)
        needs_esurge_is = (
            bool(getattr(inner_args, "esurge_importance_sampling_correction", False)) and lp_arr is not None
        )
        if is_train and needs_esurge_is:
            compute_state_logps = getattr(self.trainer, "compute_state_logps", None)
            if callable(compute_state_logps) and state is not None:
                full_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
                full_mask = jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
                policy_old_logps = compute_state_logps(
                    state,
                    full_ids,
                    full_mask,
                    None,
                    prompt_length=prompt_width,
                    logprob_vocab_chunk_size=getattr(inner_args, "logprob_vocab_chunk_size", None),
                )
                out["old_per_token_logps"] = policy_old_logps
            is_fn = getattr(self.trainer, "_esurge_importance_sampling_ratio", None)
            if callable(is_fn):
                out["importance_sampling_ratio"] = is_fn(
                    out["old_per_token_logps"],
                    lp_arr,
                    completion_mask,
                )

        beta = float(getattr(getattr(self.trainer, "arguments", None), "beta", 0.0) or 0.0)
        if beta != 0.0:
            compute_ref = getattr(self.trainer, "compute_refmodel_logps", None)
            ref_state = getattr(self.trainer, "ref_state", None)
            if callable(compute_ref) and ref_state is not None:
                full_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
                full_mask = jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
                ref_logps = compute_ref(
                    ref_state.graphstate,
                    ref_state.graphother,
                    full_ids,
                    full_mask,
                    None,
                )
                if ref_logps.shape[1] == full_ids.shape[1]:
                    ref_logps = ref_logps[:, prompt_ids.shape[1] :]
                out["ref_per_token_logps"] = ref_logps
            elif lp_arr is not None:
                out["ref_per_token_logps"] = lp_arr
            else:
                out["ref_per_token_logps"] = jnp.zeros_like(completion_ids, dtype=jnp.float32)

        return out, metrics

    def _wrapped_preprocess_batch_input(
        self,
        state: tp.Any,
        batch: collections.abc.Mapping[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        tasks = self._extract_tasks(batch)
        group_size = self._resolve_group_size(is_train)

        jobs: list[tuple[int, int, dict[str, tp.Any]]] = []
        for p_idx, task in enumerate(tasks):
            for g_idx in range(group_size):
                jobs.append((p_idx, p_idx * group_size + g_idx, task))

        episodes: list[AgentFlowEpisode] = [None] * len(jobs)  # type: ignore[list-item]
        workers = min(self.arguments.max_workers, max(1, len(jobs)))
        if workers == 1:
            for idx, (p_idx, grp_id, task) in enumerate(jobs):
                episodes[idx] = self._run_episode(state, task, p_idx, grp_id)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(self._run_episode, state, task, p_idx, grp_id): idx
                    for idx, (p_idx, grp_id, task) in enumerate(jobs)
                }
                for fut in concurrent.futures.as_completed(futs):
                    episodes[futs[fut]] = fut.result()

        if episodes and all(ep.env_error for ep in episodes):
            raise RuntimeError(
                f"All {len(episodes)} AgentFlow episodes failed with env_error=True; aborting step to avoid "
                "training on all-zero masks."
            )

        return self._adapt_episodes_to_trainer(episodes, group_size, state=state, is_train=is_train)

    def train(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Delegate training to the wrapped trainer."""
        return self.trainer.train(*args, **kwargs)

    def evaluate(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Delegate evaluation to the wrapped trainer."""
        return self.trainer.evaluate(*args, **kwargs)

    def save_pretrained(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Delegate checkpoint/model export to the wrapped trainer."""
        return self.trainer.save_pretrained(*args, **kwargs)

    def __getattr__(self, name: str) -> tp.Any:
        if "trainer" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["trainer"], name)


__all__ = [
    "AgentFlowEpisode",
    "AgentFlowTrainer",
    "AgentFlowTurn",
    "PreferencePairSelection",
    "agent_flow_supplied_reward",
    "select_group_preference_pairs",
]
