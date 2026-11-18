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

    def _format_initial_prompt(self, prompt: str) -> str:
        if self.arguments.skip_apply_chat_template:
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

    def _format_observation_turn(self, raw_obs: str, *, ends_with_eos: bool) -> str:
        if self.arguments.skip_apply_chat_template:
            return f"\n[Observation]: {raw_obs}\n"
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
            raw_prompt = str(reset_res.observation or task.get("prompt", ""))
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
                obs_text = (
                    ""
                    if (step_res.terminated or step_res.truncated or is_last_turn or not raw_obs)
                    else self._format_observation_turn(raw_obs, ends_with_eos=ends_with_eos)
                )
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

    def _extract_tasks(self, batch: collections.abc.Mapping[str, tp.Any]) -> list[dict[str, tp.Any]]:
        if "tasks" in batch and isinstance(batch["tasks"], list):
            return [dict(t) if isinstance(t, collections.abc.Mapping) else {"prompt": str(t)} for t in batch["tasks"]]

        prompts = batch.get("prompts")
        if prompts is None and "prompt_ids" in batch:
            p_arr = np.asarray(batch["prompt_ids"])
            prompts = [self._decode_ids(row.tolist()) for row in p_arr]
        if prompts is None and "input_ids" in batch:
            i_arr = np.asarray(batch["input_ids"])
            prompts = [self._decode_ids(row.tolist()) for row in i_arr]
        if prompts is None:
            prompts = [""]

        n = len(prompts)
        tasks: list[dict[str, tp.Any]] = []
        for idx in range(n):
            task: dict[str, tp.Any] = {"prompt": prompts[idx]}
            for k, v in batch.items():
                if k in {"prompts", "prompt_ids", "prompt_mask", "input_ids", "attention_mask"}:
                    continue
                if isinstance(v, (list, tuple)) and len(v) == n:
                    task[k] = v[idx]
            tasks.append(task)
        return tasks

    def _resolve_group_size(self, is_train: bool) -> int:
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
        masked = rewards * valid_mask
        scale_rewards = str(getattr(getattr(self.trainer, "arguments", None), "scale_rewards", "group")).lower()
        # With group_size == 1 (e.g. PPO/VAPO or single-rollout), group-centering would zero all rows.
        if group_size <= 1:
            if len(masked) > 1 and scale_rewards == "batch":
                mean = float(np.mean(masked))
                std = float(np.std(masked))
                return ((masked - mean) / (std + 1e-4)).astype(np.float32) * valid_mask
            return masked.astype(np.float32)

        if scale_rewards == "batch":
            mean = float(np.mean(masked))
            std = float(np.std(masked))
            return ((masked - mean) / (std + 1e-4)).astype(np.float32) * valid_mask

        num_groups = max(1, len(masked) // max(1, group_size))
        grouped = masked.reshape(num_groups, group_size)
        g_mean = np.mean(grouped, axis=1, keepdims=True)
        if scale_rewards == "none":
            return (grouped - g_mean).reshape(-1).astype(np.float32) * valid_mask
        g_std = np.std(grouped, axis=1, keepdims=True)
        return ((grouped - g_mean) / (g_std + 1e-4)).reshape(-1).astype(np.float32) * valid_mask

    def _is_ppo_family(self) -> bool:
        from easydel.trainers.proximal_policy_optimization_trainer import PPOTrainer

        return isinstance(self.trainer, PPOTrainer)

    def _is_distillation_or_preference_family(self) -> bool:
        from easydel.trainers.generalized_knowledge_distillation_trainer import GKDTrainer
        from easydel.trainers.on_policy_distillation_trainer import OnPolicyDistillationTrainer
        from easydel.trainers.online_dpo_trainer import OnlineDPOTrainer
        from easydel.trainers.sdft_trainer import SDFTTrainer
        from easydel.trainers.self_distillation_policy_optimization import SDPOTrainer
        from easydel.trainers.seq_kd_trainer import SeqKDTrainer
        from easydel.trainers.ssd_trainer import SSDTrainer

        return isinstance(
            self.trainer,
            (
                GKDTrainer,
                OnPolicyDistillationTrainer,
                OnlineDPOTrainer,
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

        if self._is_ppo_family():
            input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            attention_mask = jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
            adv_2d = jnp.broadcast_to(jnp.asarray(advantages_np)[:, None], completion_ids.shape)
            ret_2d = jnp.broadcast_to(jnp.asarray(rewards_np)[:, None], completion_ids.shape)
            pos_thresh = float(getattr(getattr(self.trainer, "arguments", None), "positive_reward_threshold", 0.0))
            positive_mask = jnp.asarray((rewards_np > pos_thresh) & (valid_np > 0.0), dtype=jnp.float32)
            out: dict[str, tp.Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": completion_mask,
                "rewards": jnp.asarray(rewards_np),
                "returns": ret_2d,
                "advantages": adv_2d,
                "old_values": jnp.zeros_like(ret_2d),
                "old_logps": lp_arr if lp_arr is not None else jnp.zeros_like(ret_2d),
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
        needs_esurge_is = bool(getattr(inner_args, "esurge_importance_sampling_correction", False)) and lp_arr is not None
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
]
