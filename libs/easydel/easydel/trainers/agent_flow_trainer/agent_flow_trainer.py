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
import functools
import inspect
import json
import math
import queue
import re
import threading
import time
import typing as tp

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from easydel.utils import Registry

from ..training_configurations import _is_eyvan_real_numeric_dtype, _resolve_eyvan_run_name, _sanitize_eyvan_key
from .agent_flow_config import (
    AgentFlowConfig,
    _normalize_base_trainer_type,
    _reconstruct_base_config,
    _unsupported_base_trainer_classes,
)

logger = get_logger(__name__)

_EYVAN_CONFIG_FIELDS: tuple[str, ...] = (
    "use_eyvan_track",
    "eyvan_track_uri",
    "eyvan_track_project",
    "eyvan_track_name",
    "eyvan_track_group",
    "eyvan_track_tags",
    "eyvan_track_flush_interval_s",
    "eyvan_report_progress",
)
_EYVAN_MAX_ROLLOUTS_PER_STEP = 8
_EYVAN_MAX_COMPLETION_GROUPS_PER_STEP = 8
_EYVAN_MAX_GROUP_RESPONSES = 64
_EYVAN_MAX_ROLLOUT_STEPS = 10_000
_EYVAN_MAX_ACTION_NAMES = 256
_EYVAN_MAX_ACTION_NAME_LEN = 64
_EYVAN_MAX_TEXT_CHARS = 32_000
_EYVAN_MAX_OBS_PREVIEW_CHARS = 400
_EYVAN_MAX_EXTRA_FIELDS = 32
_EYVAN_MAX_ENV_LEN = 128
_EYVAN_MAX_LABEL_LEN = 128


def _is_locally_readable_jax_array(arr: jax.Array) -> bool:
    """Return True if ``arr`` can be read via local ``jax.device_get`` with zero cross-host collectives.

    On multi-host JAX meshes, global replicated arrays have ``is_fully_addressable=False`` and
    ``is_fully_replicated=True``; reading their local shard via ``jax.device_get`` requires no
    collective as long as ``sharding.has_addressable_devices`` is true.
    """
    if getattr(arr, "is_fully_addressable", True):
        return True
    if getattr(arr, "is_fully_replicated", False):
        sharding = getattr(arr, "sharding", None)
        return bool(getattr(sharding, "has_addressable_devices", True))
    return False


def _to_host_numpy(arr: tp.Any) -> np.ndarray:
    """Bring a host or (potentially non-fully-addressable sharded) JAX array to host NumPy safely."""
    if isinstance(arr, jax.Array):
        if not _is_locally_readable_jax_array(arr):
            from jax.experimental import multihost_utils

            return np.asarray(multihost_utils.process_allgather(arr, tiled=True))
        return np.asarray(jax.device_get(arr))
    return np.asarray(arr)


def _to_host_numpy_local_only(arr: tp.Any) -> np.ndarray | None:
    """Convert an array to host NumPy without invoking any cross-host collective.

    Safe to call inside rank-0-only blocks (never calls ``process_allgather``).
    Accepts fully-addressable arrays as well as multi-host fully-replicated arrays
    with local addressable devices; returns ``None`` if ``arr`` is genuinely sharded
    across hosts or has a non-real numeric (e.g. complex) dtype.
    """
    if arr is None:
        return None
    if isinstance(arr, jax.Array):
        if not _is_locally_readable_jax_array(arr):
            return None
        host = np.asarray(jax.device_get(arr))
    else:
        host = np.asarray(arr)
    if not _is_eyvan_real_numeric_dtype(host.dtype):
        return None
    if host.dtype == jnp.bfloat16:
        host = host.astype(np.float32)
    return host


def _extract_nonneg_step(state: tp.Any, fallback: int = 0) -> int:
    """Safely extract a non-negative Python int step from ``state`` without cross-host collectives."""
    raw = getattr(state, "step", None) if state is not None else None
    if raw is None:
        return max(0, int(fallback))
    try:
        if isinstance(raw, jax.Array):
            if not _is_locally_readable_jax_array(raw):
                return max(0, int(fallback))
            raw = jax.device_get(raw)
        if isinstance(raw, np.ndarray):
            raw = raw.reshape(()).item()
        elif isinstance(raw, np.generic):
            raw = raw.item()
        return max(0, int(raw))
    except Exception:
        return max(0, int(fallback))


def _sanitize_eyvan_axes(axes: tp.Mapping[str, tp.Any] | None) -> dict[str, float] | None:
    """Return a mapping of sanitized Eyvan axis keys to finite Python floats."""
    if not axes:
        return None
    cleaned: dict[str, float] = {}
    for key, raw in axes.items():
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            cleaned[_sanitize_eyvan_key(key)] = val
    return cleaned or None


def _sanitize_eyvan_extra(extra: tp.Mapping[str, tp.Any] | None) -> dict[str, float] | None:
    """Clamp and sanitize episode ``extra`` fields to at most 32 valid numeric keys."""
    if not extra:
        return None
    cleaned: dict[str, float] = {}
    for key, raw in extra.items():
        if len(cleaned) >= _EYVAN_MAX_EXTRA_FIELDS:
            break
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        cleaned[_sanitize_eyvan_key(key)] = val
    return cleaned or None


def _clip_eyvan_text(text: tp.Any, limit: int = _EYVAN_MAX_TEXT_CHARS) -> str:
    """Return a UTF-8-safe string bounded to ``limit`` characters."""
    s = "" if text is None else str(text)
    s = s.encode("utf-8", errors="replace").decode("utf-8")
    if len(s) > limit:
        return s[: max(0, limit - 3)] + "..." if limit >= 3 else s[:limit]
    return s


def _resolve_eyvan_env_name(spec: tp.Any, fallback: str = "agent_flow") -> str:
    """Resolve a clean 1..128 character Eyvan environment name from ``spec``."""
    candidate: str | None = None
    explicit = getattr(spec, "environment_name", None)
    if isinstance(explicit, str) and explicit.strip():
        candidate = explicit.strip()
    elif isinstance(spec, str):
        candidate = spec.strip()
    elif isinstance(spec, collections.abc.Mapping):
        raw = spec.get("name") or spec.get("type")
        if isinstance(raw, str):
            candidate = raw.strip()
    elif isinstance(spec, type):
        candidate = spec.__name__
    elif isinstance(spec, functools.partial):
        func = spec.func
        name = getattr(func, "__name__", None)
        if isinstance(name, str) and name and name != "<lambda>" and not name.startswith("_"):
            candidate = name
    elif callable(spec):
        name = getattr(spec, "__name__", None)
        if isinstance(name, str) and name and name != "<lambda>" and not name.startswith("_"):
            candidate = name
    elif spec is not None:
        cls_name = type(spec).__name__
        if cls_name and cls_name not in {"function", "method", "builtin_function_or_method"}:
            candidate = cls_name
    resolved = (candidate or fallback).strip()
    return (resolved or fallback)[:_EYVAN_MAX_ENV_LEN]


def _eyvan_log_episode(
    args: tp.Any,
    ret: float,
    length: int | None = None,
    *,
    success: bool | None = None,
    kind: str = "train",
    env: str | None = None,
    policy_version: int | None = None,
    learner_version: int | None = None,
    step: int | None = None,
    axes: tp.Mapping[str, tp.Any] | None = None,
    extra: tp.Mapping[str, tp.Any] | None = None,
) -> bool:
    """Emit one Eyvan episode via ``TrainingArguments.log_eyvan_episode`` or direct ``Run.episode``."""
    if args is None or not bool(getattr(args, "eyvan_tracking_enabled", False)):
        return False
    clean_axes = _sanitize_eyvan_axes(axes)
    clean_extra = _sanitize_eyvan_extra(extra)
    clean_env = str(env).strip()[:_EYVAN_MAX_ENV_LEN] if env else None
    clean_env = clean_env or None
    clean_len = None if length is None else max(0, int(length))
    clean_success = None if success is None else bool(success)
    clean_step = None if step is None else max(0, int(step))
    clean_pv = None if policy_version is None else max(0, int(policy_version))
    clean_lv = None if learner_version is None else max(0, int(learner_version))
    clean_kind = kind if kind in ("train", "eval") else "train"
    helper = getattr(args, "log_eyvan_episode", None)
    if callable(helper):
        try:
            return bool(
                helper(
                    float(ret),
                    clean_len,
                    success=clean_success,
                    kind=clean_kind,
                    env=clean_env,
                    policy_version=clean_pv,
                    learner_version=clean_lv,
                    step=clean_step,
                    axes=clean_axes,
                    extra=clean_extra,
                )
            )
        except Exception as exc:
            logger.warning("AgentFlow Eyvan episode logging failed (%s): %s", type(exc).__name__, exc)
            return False
    get_run = getattr(args, "get_eyvan_track", None)
    if not callable(get_run):
        return False
    try:
        run = get_run()
        if run is None or getattr(run, "finished", False):
            return False
        run.episode(
            float(ret),
            clean_len,
            success=clean_success,
            kind=clean_kind,
            env=clean_env,
            policy_version=clean_pv,
            learner_version=clean_lv,
            step=clean_step,
            axes=clean_axes,
            extra=clean_extra,
        )
        return True
    except Exception as exc:
        logger.warning("AgentFlow Eyvan episode logging failed (%s): %s", type(exc).__name__, exc)
        return False


def _eyvan_log_rollout(
    args: tp.Any,
    *,
    tracks: tp.Mapping[str, tp.Sequence[float | None]] | None = None,
    actions: tp.Sequence[int | None] | None = None,
    action_names: tp.Sequence[str] | None = None,
    episode_id: str | None = None,
    env: str | None = None,
    kind: str = "train",
    policy_version: int | None = None,
    step: int | None = None,
    axes: tp.Mapping[str, tp.Any] | None = None,
) -> bool:
    """Emit one Eyvan rollout via ``TrainingArguments.log_eyvan_rollout`` or direct ``Run.rollout``."""
    if args is None or not bool(getattr(args, "eyvan_tracking_enabled", False)):
        return False
    if not tracks and not actions:
        return False
    clean_axes = _sanitize_eyvan_axes(axes)
    clean_env = str(env).strip()[:_EYVAN_MAX_ENV_LEN] if env else None
    clean_env = clean_env or None
    clean_ep_id = str(episode_id).strip()[:_EYVAN_MAX_LABEL_LEN] if episode_id else None
    clean_ep_id = clean_ep_id or None
    clean_step = None if step is None else max(0, int(step))
    clean_pv = None if policy_version is None else max(0, int(policy_version))
    clean_kind = kind if kind in ("train", "eval") else "train"
    clean_tracks = {_sanitize_eyvan_key(k): list(v) for k, v in tracks.items()} if tracks is not None else None
    clean_actions = [None if a is None else int(a) for a in actions] if actions is not None else None
    clean_names = (
        [(str(n).strip()[:_EYVAN_MAX_ACTION_NAME_LEN] or f"action_{i}") for i, n in enumerate(action_names)][
            :_EYVAN_MAX_ACTION_NAMES
        ]
        if action_names is not None
        else None
    )
    helper = getattr(args, "log_eyvan_rollout", None)
    if callable(helper):
        try:
            return bool(
                helper(
                    tracks=clean_tracks,
                    actions=clean_actions,
                    action_names=clean_names,
                    episode_id=clean_ep_id,
                    env=clean_env,
                    kind=clean_kind,
                    policy_version=clean_pv,
                    step=clean_step,
                    axes=clean_axes,
                )
            )
        except Exception as exc:
            logger.warning("AgentFlow Eyvan rollout logging failed (%s): %s", type(exc).__name__, exc)
            return False
    get_run = getattr(args, "get_eyvan_track", None)
    if not callable(get_run):
        return False
    try:
        run = get_run()
        if run is None or getattr(run, "finished", False):
            return False
        return bool(
            run.rollout(
                tracks=clean_tracks,
                actions=clean_actions,
                action_names=clean_names,
                episode_id=clean_ep_id,
                env=clean_env,
                kind=clean_kind,
                policy_version=clean_pv,
                step=clean_step,
                axes=clean_axes,
            )
        )
    except Exception as exc:
        logger.warning("AgentFlow Eyvan rollout logging failed (%s): %s", type(exc).__name__, exc)
        return False


def _eyvan_log_completions(
    args: tp.Any,
    prompt: str,
    responses: tp.Sequence[str],
    *,
    rewards: tp.Any = None,
    advantages: tp.Any = None,
    group: str | int | None = None,
    tokens: tp.Sequence[int] | None = None,
    kl: tp.Sequence[float | None] | None = None,
    truncated: tp.Sequence[bool] | None = None,
    policy_version: int | None = None,
    step: int | None = None,
) -> bool:
    """Emit one Eyvan completion group via ``TrainingArguments.log_eyvan_completions`` or direct ``Run.completions``."""
    if args is None or not bool(getattr(args, "eyvan_tracking_enabled", False)):
        return False
    if not responses:
        return False
    n = min(len(responses), _EYVAN_MAX_GROUP_RESPONSES)
    clean_prompt = _clip_eyvan_text(prompt)
    clean_responses = [_clip_eyvan_text(r) for r in responses[:n]]
    clean_rewards: tp.Any = None
    if isinstance(rewards, collections.abc.Mapping):
        clean_rewards = {_sanitize_eyvan_key(k): [float(x) for x in list(v)[:n]] for k, v in rewards.items()}
    elif rewards is not None:
        clean_rewards = [float(x) for x in list(rewards)[:n]]
    clean_adv = [float(x) for x in list(advantages)[:n]] if advantages is not None else None
    clean_tokens = [max(0, int(x)) for x in list(tokens)[:n]] if tokens is not None else None
    clean_kl = (
        [None if (x is None or not math.isfinite(float(x))) else float(x) for x in list(kl)[:n]]
        if kl is not None
        else None
    )
    clean_trunc = [bool(x) for x in list(truncated)[:n]] if truncated is not None else None
    clean_group = str(group).strip()[:_EYVAN_MAX_LABEL_LEN] if group is not None else None
    clean_group = clean_group or None
    clean_step = None if step is None else max(0, int(step))
    clean_pv = None if policy_version is None else max(0, int(policy_version))
    helper = getattr(args, "log_eyvan_completions", None)
    if callable(helper):
        try:
            return bool(
                helper(
                    clean_prompt,
                    clean_responses,
                    rewards=clean_rewards,
                    advantages=clean_adv,
                    group=clean_group,
                    tokens=clean_tokens,
                    kl=clean_kl,
                    truncated=clean_trunc,
                    policy_version=clean_pv,
                    step=clean_step,
                )
            )
        except Exception as exc:
            logger.warning("AgentFlow Eyvan completions logging failed (%s): %s", type(exc).__name__, exc)
            return False
    get_run = getattr(args, "get_eyvan_track", None)
    if not callable(get_run):
        return False
    try:
        run = get_run()
        if run is None or getattr(run, "finished", False):
            return False
        run.completions(
            clean_prompt,
            clean_responses,
            rewards=clean_rewards,
            advantages=clean_adv,
            group=clean_group,
            tokens=clean_tokens,
            kl=clean_kl,
            truncated=clean_trunc,
            policy_version=clean_pv,
            step=clean_step,
        )
        return True
    except Exception as exc:
        logger.warning("AgentFlow Eyvan completions logging failed (%s): %s", type(exc).__name__, exc)
        return False


def _eyvan_log_histogram(
    args: tp.Any,
    key: str,
    values: tp.Any,
    *,
    step: int | None = None,
    bins: int = 64,
    axes: tp.Mapping[str, tp.Any] | None = None,
) -> bool:
    """Emit one Eyvan histogram via ``TrainingArguments.log_eyvan_histogram`` or direct ``Run.histogram``."""
    if args is None or not bool(getattr(args, "eyvan_tracking_enabled", False)):
        return False
    arr = _to_host_numpy_local_only(values)
    if arr is None or arr.size == 0:
        return False
    flat = arr.reshape(-1)
    clean_key = _sanitize_eyvan_key(key)
    clean_step = None if step is None else max(0, int(step))
    clean_axes = _sanitize_eyvan_axes(axes)
    helper = getattr(args, "log_eyvan_histogram", None)
    if callable(helper):
        try:
            return bool(helper(clean_key, flat, step=clean_step, bins=bins, axes=clean_axes))
        except Exception as exc:
            logger.warning("AgentFlow Eyvan histogram %r logging failed (%s): %s", clean_key, type(exc).__name__, exc)
            return False
    get_run = getattr(args, "get_eyvan_track", None)
    if not callable(get_run):
        return False
    try:
        run = get_run()
        if run is None or getattr(run, "finished", False):
            return False
        run.histogram(clean_key, values=flat, step=clean_step, bins=bins, axes=clean_axes)
        return True
    except Exception as exc:
        logger.warning("AgentFlow Eyvan histogram %r logging failed (%s): %s", clean_key, type(exc).__name__, exc)
        return False


def _compute_icepop_active_ratios(
    is_ratio_np: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Compute IcePop / IS ratio statistics over active action tokens (``active_mask > 0``)."""
    mask_bool = np.asarray(active_mask) > 0
    arr = np.asarray(is_ratio_np)
    if arr.dtype == jnp.bfloat16:
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1 and mask_bool.ndim == 2 and arr.shape[0] == mask_bool.shape[0]:
        arr = arr[:, None]
    broadcasted = np.broadcast_to(arr, mask_bool.shape)
    active = broadcasted[mask_bool]
    if active.size == 0:
        return active, 0.0, 0.0, 0.0
    finite = active[np.isfinite(active)]
    ratio_mean = float(np.mean(finite)) if finite.size else 0.0
    ratio_max = float(np.max(finite)) if finite.size else 0.0
    masked_frac = float(np.mean(active == 0.0))
    return active, ratio_mean, ratio_max, masked_frac


def _compute_per_episode_kl_numpy(
    policy_logps: np.ndarray,
    ref_logps: np.ndarray,
    active_mask: np.ndarray,
    kl_estimator: str = "k3",
) -> list[float | None]:
    """Compute per-episode mean KL over active action tokens in float64 NumPy."""
    mask_bool = np.asarray(active_mask) > 0
    diff = np.asarray(ref_logps, dtype=np.float64) - np.asarray(policy_logps, dtype=np.float64)
    est = str(kl_estimator or "k3").lower()
    if est == "k1":
        per_tok = -diff
    elif est == "k2":
        per_tok = 0.5 * np.square(diff)
    elif est == "l2":
        per_tok = np.square(diff)
    else:
        clamped = np.clip(diff, -80.0, 80.0)
        per_tok = np.exp(clamped) - clamped - 1.0
    out: list[float | None] = []
    for row_idx in range(mask_bool.shape[0]):
        row_mask = mask_bool[row_idx]
        if not np.any(row_mask):
            out.append(None)
            continue
        vals = per_tok[row_idx][row_mask]
        finite = vals[np.isfinite(vals)]
        out.append(float(np.mean(finite)) if finite.size else None)
    return out


def _select_rollout_sample_indices(
    rewards: tp.Sequence[float],
    eligible_indices: tp.Sequence[int],
    cap: int = _EYVAN_MAX_ROLLOUTS_PER_STEP,
) -> list[int]:
    """Select up to ``cap`` episode indices prioritizing best reward, worst reward, then remaining."""
    if not eligible_indices or cap <= 0:
        return []
    selected: list[int] = []
    finite = [idx for idx in eligible_indices if math.isfinite(float(rewards[idx]))]
    if finite:
        best_idx = max(finite, key=lambda idx: (float(rewards[idx]), -idx))
        worst_idx = min(finite, key=lambda idx: (float(rewards[idx]), idx))
        selected.append(best_idx)
        if worst_idx != best_idx and len(selected) < cap:
            selected.append(worst_idx)
    seen = set(selected)
    for idx in eligible_indices:
        if len(selected) >= cap:
            break
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    return selected


def _encode_action_labels(raw_texts: tp.Sequence[str]) -> tuple[list[int], list[str]]:
    """Map per-turn action strings into Eyvan int action IDs and bounded ``action_names``."""
    action_names: list[str] = []
    name_to_idx: dict[str, int] = {}
    action_ids: list[int] = []
    for turn_i, raw in enumerate(raw_texts):
        cleaned = (
            " ".join(str(raw or "").split())
            .encode("utf-8", errors="replace")
            .decode("utf-8")[:_EYVAN_MAX_ACTION_NAME_LEN]
            .strip()
        )
        if not cleaned:
            cleaned = f"turn_{turn_i}"
        existing = name_to_idx.get(cleaned)
        if existing is not None:
            action_ids.append(existing)
        elif len(action_names) < _EYVAN_MAX_ACTION_NAMES:
            idx = len(action_names)
            action_names.append(cleaned)
            name_to_idx[cleaned] = idx
            action_ids.append(idx)
        else:
            action_ids.append(_EYVAN_MAX_ACTION_NAMES - 1)
    return action_ids, action_names


def _reconstruct_turns_from_tito(
    *,
    prompt_ids: tp.Sequence[int],
    comp_ids: tp.Sequence[int],
    comp_amask: tp.Sequence[int],
    comp_lps: tp.Sequence[float] | None,
    final_reward: float,
    decode_ids: tp.Callable[[tp.Sequence[int]], str],
    truncated: bool = False,
) -> tuple[AgentFlowTurn, ...]:
    """Reconstruct multi-turn :class:`AgentFlowTurn` records from a TITO token/action-mask trajectory.

    In harness TITO mode, assistant generations correspond to contiguous spans where ``action_mask == 1``
    and tool/environment observations correspond to the intervening spans where ``action_mask == 0``.
    Reconstructing turns preserves exact token/mask/logprob alignment while enabling rich multi-turn
    transcripts, episode turn lengths, and per-turn rollout telemetry.
    """
    n = min(len(comp_ids), len(comp_amask))
    if n <= 0:
        return ()

    segments: list[tuple[list[int], list[float] | None, list[int]]] = []
    i = 0
    while i < n:
        if not comp_amask[i]:
            i += 1
            continue
        act_start = i
        while i < n and comp_amask[i]:
            i += 1
        act_end = i
        obs_start = i
        while i < n and not comp_amask[i]:
            i += 1
        obs_end = i
        act_ids = [int(t) for t in comp_ids[act_start:act_end]]
        obs_ids = [int(t) for t in comp_ids[obs_start:obs_end]]
        turn_lps = [float(x) for x in comp_lps[act_start:act_end]] if comp_lps is not None else None
        segments.append((act_ids, turn_lps, obs_ids))

    if not segments:
        return ()

    initial_prompt_text = decode_ids(prompt_ids) if prompt_ids else ""
    last_idx = len(segments) - 1
    turns: list[AgentFlowTurn] = []
    for turn_idx, (act_ids, turn_lps, obs_ids) in enumerate(segments):
        is_last = turn_idx == last_idx
        turns.append(
            AgentFlowTurn(
                turn_index=turn_idx,
                prompt_text=initial_prompt_text if turn_idx == 0 else "",
                action_text=decode_ids(act_ids),
                action_token_ids=act_ids,
                observation_text=decode_ids(obs_ids) if obs_ids else "",
                observation_token_ids=obs_ids,
                reward=float(final_reward) if is_last else 0.0,
                terminated=bool(is_last and not truncated),
                truncated=bool(is_last and truncated),
                logprobs=turn_lps,
            )
        )
    return tuple(turns)


def _format_agent_flow_transcript(
    ep: AgentFlowEpisode,
    *,
    decode_ids: tp.Callable[[tp.Sequence[int]], str] | None = None,
    max_obs_chars: int = _EYVAN_MAX_OBS_PREVIEW_CHARS,
    max_total_chars: int = _EYVAN_MAX_TEXT_CHARS,
) -> str:
    """Format a multi-turn :class:`AgentFlowEpisode` as a compact completion transcript."""
    if not ep.turns:
        if ep.completion_token_ids and callable(decode_ids):
            reconstructed = _reconstruct_turns_from_tito(
                prompt_ids=ep.prompt_token_ids,
                comp_ids=ep.completion_token_ids,
                comp_amask=ep.action_mask if len(ep.action_mask) == len(ep.completion_token_ids) else [1] * len(ep.completion_token_ids),
                comp_lps=ep.logprobs,
                final_reward=ep.episode_reward,
                decode_ids=decode_ids,
            )
            if reconstructed:
                parts_fb: list[str] = []
                for turn in reconstructed:
                    segment = f"[Turn {turn.turn_index} | r={float(turn.reward):.4g}]\nAction: {turn.action_text}"
                    if turn.observation_text:
                        obs = turn.observation_text
                        if len(obs) > max_obs_chars:
                            obs = obs[: max_obs_chars - 3] + "..."
                        segment += f"\nObservation: {obs}"
                    parts_fb.append(segment)
                return _clip_eyvan_text("\n\n".join(parts_fb), limit=max_total_chars)
            decoded = decode_ids(ep.completion_token_ids)
            if decoded:
                return _clip_eyvan_text(decoded, limit=max_total_chars)
        return "[env_error: no turns]" if ep.env_error else "[empty episode]"
    parts: list[str] = []
    for turn in ep.turns:
        segment = f"[Turn {turn.turn_index} | r={float(turn.reward):.4g}]\nAction: {turn.action_text}"
        if turn.observation_text:
            obs = turn.observation_text
            if len(obs) > max_obs_chars:
                obs = obs[: max_obs_chars - 3] + "..."
            segment += f"\nObservation: {obs}"
        parts.append(segment)
    return _clip_eyvan_text("\n\n".join(parts), limit=max_total_chars)


def _serialize_episodes(episodes: list[AgentFlowEpisode]) -> bytes:
    """Serialize episodes to UTF-8 JSON bytes (preserving NaN/Inf) for multi-host broadcast."""
    payload = [
        {
            "prompt_index": ep.prompt_index,
            "group_id": ep.group_id,
            "prompt_token_ids": ep.prompt_token_ids,
            "completion_token_ids": ep.completion_token_ids,
            "action_mask": ep.action_mask,
            "step_rewards": ep.step_rewards,
            "episode_reward": ep.episode_reward,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "prompt_text": t.prompt_text,
                    "action_text": t.action_text,
                    "action_token_ids": t.action_token_ids,
                    "observation_text": t.observation_text,
                    "observation_token_ids": t.observation_token_ids,
                    "reward": t.reward,
                    "terminated": t.terminated,
                    "truncated": t.truncated,
                    "logprobs": t.logprobs,
                }
                for t in ep.turns
            ],
            "logprobs": ep.logprobs,
            "env_error": ep.env_error,
            "prompt_clipped": ep.prompt_clipped,
        }
        for ep in episodes
    ]
    return json.dumps(payload, allow_nan=True).encode("utf-8")


def _deserialize_episodes(raw: bytes) -> list[AgentFlowEpisode]:
    """Deserialize UTF-8 JSON bytes produced by :func:`_serialize_episodes`."""
    data = json.loads(raw.decode("utf-8"))
    episodes: list[AgentFlowEpisode] = []
    for item in data:
        turns = tuple(
            AgentFlowTurn(
                turn_index=int(t["turn_index"]),
                prompt_text=str(t["prompt_text"]),
                action_text=str(t["action_text"]),
                action_token_ids=[int(x) for x in t["action_token_ids"]],
                observation_text=str(t["observation_text"]),
                observation_token_ids=[int(x) for x in t["observation_token_ids"]],
                reward=float(t["reward"]),
                terminated=bool(t["terminated"]),
                truncated=bool(t["truncated"]),
                logprobs=None if t["logprobs"] is None else [float(x) for x in t["logprobs"]],
            )
            for t in item["turns"]
        )
        episodes.append(
            AgentFlowEpisode(
                prompt_index=int(item["prompt_index"]),
                group_id=int(item["group_id"]),
                prompt_token_ids=[int(x) for x in item["prompt_token_ids"]],
                completion_token_ids=[int(x) for x in item["completion_token_ids"]],
                action_mask=[int(x) for x in item["action_mask"]],
                step_rewards=[float(x) for x in item["step_rewards"]],
                episode_reward=float(item["episode_reward"]),
                turns=turns,
                logprobs=None if item["logprobs"] is None else [float(x) for x in item["logprobs"]],
                env_error=bool(item["env_error"]),
                prompt_clipped=bool(item.get("prompt_clipped", False)),
            )
        )
    return episodes


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
    prompt_clipped: bool = False


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
        self._rollout_lock = threading.RLock()
        self._rollout_call_count = 0
        self._owned_env_runtime: tp.Any = None
        self._eyvan_total_turns: int = 0
        self._eyvan_total_action_tokens: int = 0

        self._env_spec = environment_factory or environment or self.arguments.environment
        resolved_env_fallback = _resolve_eyvan_env_name(self.arguments.environment, fallback="agent_flow")
        self._eyvan_env_name = _resolve_eyvan_env_name(self._env_spec, fallback=resolved_env_fallback)
        self._eyvan_tag_env_name = _resolve_eyvan_env_name(self._env_spec, fallback="") or (
            resolved_env_fallback if resolved_env_fallback != "agent_flow" else None
        )

        if trainer is not None:
            if isinstance(trainer, _unsupported_base_trainer_classes()):
                raise ValueError(
                    f"{type(trainer).__name__} is not supported as an inner trainer for AgentFlowTrainer "
                    "(its custom training step or constructor is incompatible with AgentFlow batch adaptation)."
                )
            self.trainer = trainer
            inferred = self._infer_trainer_type(trainer)
            if inferred is not None:
                self.arguments.base_trainer_type = _normalize_base_trainer_type(inferred)
                if self.arguments.base_config is None and hasattr(trainer, "arguments"):
                    self.arguments.base_config = trainer.arguments
            inner_args = getattr(trainer, "arguments", None)
            inner_model = getattr(trainer, "_model", None) or getattr(
                getattr(trainer, "model_state", None), "model", None
            )
            if inner_args is not None:
                self._sync_eyvan_config_and_context(
                    self.arguments,
                    inner_args,
                    model=inner_model or model or model_state,
                    env_name=self._eyvan_tag_env_name,
                )
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
                if not bool(getattr(self.trainer, "supports_async_rollout", False)):
                    raise ValueError(
                        f"{type(self.trainer).__name__} does not support `async_mode=True`. "
                        "`async_mode=True` is only supported on rollout-capable trainers."
                    )
                self.trainer.arguments.async_mode = True
                if hasattr(self.trainer.arguments, "esurge_async_scheduling"):
                    self.trainer.arguments.esurge_async_scheduling = True
                if hasattr(self.trainer.arguments, "esurge_overlap_execution"):
                    self.trainer.arguments.esurge_overlap_execution = True
            else:
                self.arguments.async_mode = bool(self.trainer.arguments.async_mode)

        self._env_factory = self._resolve_env_factory(self._env_spec)
        try:
            self._env_accepted_keys = self._resolve_accepted_env_keys(self._env_spec)
            self._logged_episode_error = False
            self._orig_preprocess_batch_input = self.trainer._preprocess_batch_input
            self.trainer._preprocess_batch_input = self._wrapped_preprocess_batch_input
            orig_teardown = getattr(self.trainer, "_teardown_resources", None)
            if callable(orig_teardown):

                def _wrapped_teardown(*args: tp.Any, **kwargs: tp.Any) -> None:
                    try:
                        orig_teardown(*args, **kwargs)
                    finally:
                        self._close_owned_env_runtime()

                self.trainer._teardown_resources = _wrapped_teardown
        except BaseException:
            self._close_owned_env_runtime()
            raise

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

    @staticmethod
    def _ensure_online_dpo_lengths(
        cfg_cls: type,
        init_kwargs: dict[str, tp.Any],
        *,
        explicit_keys: collections.abc.Collection[str] = (),
    ) -> dict[str, tp.Any]:
        """Prevent ``OnlineDPOConfig`` defaults (64/512) from silently clobbering completion/total lengths."""
        params = inspect.signature(cfg_cls).parameters
        updated = dict(init_kwargs)
        if "max_new_tokens" in params and "max_new_tokens" not in updated and "max_new_tokens" not in explicit_keys:
            comp_len = updated.get("max_completion_length")
            if comp_len is not None:
                updated["max_new_tokens"] = int(comp_len)
        if "max_length" in params and "max_length" not in updated and "max_length" not in explicit_keys:
            default_ml = params["max_length"].default
            if isinstance(default_ml, int):
                eff_prompt = updated.get("max_prompt_length")
                if eff_prompt is None and "max_prompt_length" in params:
                    default_p = params["max_prompt_length"].default
                    if isinstance(default_p, int):
                        eff_prompt = default_p
                eff_comp = updated.get("max_new_tokens")
                if eff_comp is None:
                    eff_comp = updated.get("max_completion_length")
                if eff_comp is None and "max_new_tokens" in params:
                    default_c = params["max_new_tokens"].default
                    if isinstance(default_c, int):
                        eff_comp = default_c
                if eff_prompt is not None and eff_comp is not None:
                    needed = int(eff_prompt) + int(eff_comp)
                    if needed > default_ml:
                        updated["max_length"] = needed
        return updated

    @staticmethod
    def _sync_eyvan_config_and_context(
        outer_cfg: tp.Any,
        inner_cfg: tp.Any,
        *,
        model: tp.Any = None,
        env_name: str | None = None,
    ) -> None:
        """Bind AgentFlow Eyvan context onto both configs and synchronize any auto-generated run name."""
        for target in (outer_cfg, inner_cfg):
            bind_fn = getattr(target, "_bind_eyvan_runtime_context", None)
            if callable(bind_fn):
                bind_fn(
                    model=model,
                    env_name=env_name,
                    extra_trainer_tags=("agentflow",),
                )

        inner_explicit = getattr(inner_cfg, "eyvan_track_name", None) or getattr(inner_cfg, "wandb_name", None)
        outer_explicit = getattr(outer_cfg, "eyvan_track_name", None) or getattr(outer_cfg, "wandb_name", None)
        if isinstance(inner_explicit, str) and inner_explicit.strip():
            return
        if isinstance(outer_explicit, str) and outer_explicit.strip():
            if hasattr(inner_cfg, "__dict__"):
                inner_cfg.__dict__["_eyvan_auto_run_name"] = outer_explicit.strip()
            return

        cached = (inner_cfg.__dict__.get("_eyvan_auto_run_name") if hasattr(inner_cfg, "__dict__") else None) or (
            outer_cfg.__dict__.get("_eyvan_auto_run_name") if hasattr(outer_cfg, "__dict__") else None
        )
        if not (isinstance(cached, str) and cached.strip()):
            resolved_model_name = getattr(inner_cfg, "model_name", None) or getattr(outer_cfg, "model_name", None)
            cached = _resolve_eyvan_run_name(
                outer_cfg,
                model_name=resolved_model_name,
                model=model,
                trainer_slug="agentflow",
            )
        if hasattr(outer_cfg, "__dict__"):
            outer_cfg.__dict__["_eyvan_auto_run_name"] = cached
        if hasattr(inner_cfg, "__dict__"):
            inner_cfg.__dict__["_eyvan_auto_run_name"] = cached

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
        base_type = _normalize_base_trainer_type(cfg.base_trainer_type)
        trainer_cls = Registry.get_or_raise("trainer", base_type)
        if isinstance(trainer_cls, type) and issubclass(trainer_cls, _unsupported_base_trainer_classes()):
            raise ValueError(
                f"`base_trainer_type` {base_type!r} ({trainer_cls.__name__}) is not supported by AgentFlowTrainer."
            )
        base_cfg = cfg.base_config
        if base_cfg is None:
            cfg_cls = Registry.get_or_raise("trainer-arguments", base_type)
            sig = inspect.signature(cfg_cls)
            candidate_kwargs: dict[str, tp.Any] = {
                "model_name": cfg.model_name,
                "learning_rate": cfg.learning_rate,
                "max_training_steps": cfg.max_training_steps,
                "save_directory": cfg.save_directory,
                "async_mode": cfg.async_mode,
                "max_prompt_length": cfg.max_prompt_length,
                "max_completion_length": cfg.max_completion_length,
                "total_batch_size": cfg.total_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "num_generations": cfg.num_generations,
                "gamma": cfg.gamma,
            }
            for eyvan_key in _EYVAN_CONFIG_FIELDS:
                val = getattr(cfg, eyvan_key, None)
                if val is not None:
                    candidate_kwargs[eyvan_key] = list(val) if isinstance(val, list) else val
            init_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters and v is not None}
            init_kwargs = self._ensure_online_dpo_lengths(cfg_cls, init_kwargs)
            base_cfg = cfg_cls(**init_kwargs)
        elif isinstance(base_cfg, collections.abc.Mapping):
            cfg_cls = Registry.get_or_raise("trainer-arguments", base_type)
            cfg_params = inspect.signature(cfg_cls).parameters
            raw_dict = dict(base_cfg)
            if "model_name" not in base_cfg and "model_name" in cfg_params and cfg.model_name is not None:
                raw_dict["model_name"] = cfg.model_name
            if bool(cfg.async_mode) and raw_dict.get("async_mode") is None:
                if "async_mode" in cfg_params:
                    raw_dict["async_mode"] = True
            for eyvan_key in _EYVAN_CONFIG_FIELDS:
                if eyvan_key not in base_cfg and eyvan_key in cfg_params:
                    val = getattr(cfg, eyvan_key, None)
                    if val is not None:
                        raw_dict[eyvan_key] = list(val) if isinstance(val, list) else val
            raw_dict = self._ensure_online_dpo_lengths(
                cfg_cls,
                raw_dict,
                explicit_keys=base_cfg.keys(),
            )
            base_cfg = _reconstruct_base_config(cfg_cls, raw_dict)

        self._sync_eyvan_config_and_context(
            cfg,
            base_cfg,
            model=model or model_state,
            env_name=getattr(self, "_eyvan_tag_env_name", None),
        )

        ctor_params: collections.abc.Mapping[str, inspect.Parameter] = {}
        accepts_var_kw = False
        try:
            ctor_params = inspect.signature(trainer_cls.__init__).parameters
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ctor_params.values())
        except (TypeError, ValueError):
            accepts_var_kw = True

        ctor_kwargs: dict[str, tp.Any] = {
            "arguments": base_cfg,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": processing_class,
            **trainer_kwargs,
        }
        uses_student_model = "student_model" in ctor_params and "model" not in ctor_params
        if model is not None:
            if uses_student_model:
                ctor_kwargs.setdefault("student_model", model)
            else:
                ctor_kwargs["model"] = model
        if model_state is not None:
            if "model_state" in ctor_params or accepts_var_kw:
                ctor_kwargs["model_state"] = model_state
            elif uses_student_model:
                ctor_kwargs.setdefault("student_model", model_state)
            else:
                ctor_kwargs.setdefault("model", model_state)
        if reward_funcs is None:
            from easydel.trainers.group_relative_policy_optimization import GRPOTrainer
            from easydel.trainers.online_dpo_trainer import OnlineDPOTrainer
            from easydel.trainers.proximal_policy_optimization_trainer import PPOTrainer

            accepts_reward_funcs = "reward_funcs" in ctor_params or accepts_var_kw
            if (
                accepts_reward_funcs
                and isinstance(trainer_cls, type)
                and issubclass(trainer_cls, (GRPOTrainer, PPOTrainer, OnlineDPOTrainer))
            ):
                # Environment-only setup: GRPO-family, PPO/VAPO, and OnlineDPO constructors require a reward,
                # but the wrapper supplies episode rewards (and pairs), so pass a placeholder that is never called.
                reward_funcs = agent_flow_supplied_reward
        if reward_funcs is not None:
            ctor_kwargs["reward_funcs"] = reward_funcs
        return trainer_cls(**ctor_kwargs)

    def _close_owned_env_runtime(self) -> None:
        """Close and clear any environment runtime (``SandboxPool``/``KiriRuntime``) owned by this wrapper."""
        runtime = getattr(self, "_owned_env_runtime", None)
        self._owned_env_runtime = None
        if runtime is None:
            return
        close_fn = getattr(runtime, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.warning("AgentFlow owned env runtime close() failed (%s): %s", type(exc).__name__, exc)

    def _create_owned_env_runtime(self) -> tp.Any:
        """Instantiate and record an environment runtime according to ``self.arguments``."""
        from easydel.environments import KiriRuntime, SandboxPool

        endpoints = self.arguments.environment_endpoints
        if endpoints:
            runtime: tp.Any = SandboxPool(endpoints, backend=self.arguments.environment_backend)
        else:
            runtime = KiriRuntime(backend=self.arguments.environment_backend)
        self._owned_env_runtime = runtime
        return runtime

    def _resolve_env_factory(self, spec: tp.Any) -> tp.Callable[..., tp.Any]:
        if spec is None:
            raise ValueError(
                "AgentFlowTrainer requires `environment`, `environment_factory`, or `arguments.environment`."
            )
        if callable(spec) and not isinstance(spec, str):
            return spec
        if isinstance(spec, str):
            from easydel.environments import make_environment_factory

            runtime = self._create_owned_env_runtime()
            try:
                return make_environment_factory(spec, runtime=runtime)
            except BaseException:
                self._close_owned_env_runtime()
                raise
        if isinstance(spec, collections.abc.Mapping):
            from easydel.environments import make_environment_factory

            merged = dict(spec)
            created_runtime = False
            if merged.get("runtime") is None:
                merged["runtime"] = self._create_owned_env_runtime()
                created_runtime = True
            try:
                return make_environment_factory(merged)
            except BaseException:
                if created_runtime:
                    self._close_owned_env_runtime()
                raise
        return lambda **_kw: spec

    def _gather_array(self, arr: tp.Any) -> jax.Array:
        """Route an array through ``self.trainer._all_gather`` when a mesh or custom gather override is present."""
        all_gather = getattr(self.trainer, "_all_gather", None)
        if callable(all_gather):
            is_custom_override = "_all_gather" in getattr(self.trainer, "__dict__", {})
            has_mesh = getattr(getattr(self.trainer, "model", None), "mesh", None) is not None
            if is_custom_override or has_mesh:
                return all_gather(arr)
        return jnp.asarray(arr)

    def _gather_batch(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Gather every array/NumPy leaf of ``batch`` onto the inner trainer's replicated sharding."""
        return {
            key: self._gather_array(value) if isinstance(value, (jax.Array, np.ndarray)) else value
            for key, value in batch.items()
        }

    def _resolve_eos_token_ids(self) -> list[int]:
        """Collect all EOS token IDs safely from the inner trainer and its tokenizer/processor."""
        eos_ids: list[int] = []

        def _extend(raw: tp.Any) -> None:
            if raw is None:
                return
            if isinstance(raw, (list, tuple, set)):
                for item in raw:
                    if item is not None:
                        eos_ids.append(int(item))
            elif isinstance(raw, np.ndarray):
                for item in raw.reshape(-1).tolist():
                    if item is not None:
                        eos_ids.append(int(item))
            else:
                eos_ids.append(int(raw))

        try:
            _extend(getattr(self.trainer, "_eos_token_id", None))
        except Exception:
            pass
        if not eos_ids:
            proc = getattr(self.trainer, "processing_class", None)
            if proc is not None:
                tok = getattr(proc, "tokenizer", None)
                for obj in (tok, proc):
                    if obj is not None:
                        _extend(getattr(obj, "eos_token_id", None))
                        if eos_ids:
                            break
        return list(dict.fromkeys(eos_ids))

    def _resolve_pad_token_id(self) -> int:
        """Resolve an integer pad token ID safely from the inner trainer and its tokenizer/processor."""
        try:
            pad_id = getattr(self.trainer, "_pad_token_id", None)
            if pad_id is not None:
                return int(pad_id)
        except Exception:
            pass
        proc = getattr(self.trainer, "processing_class", None)
        if proc is not None:
            tok = getattr(proc, "tokenizer", None)
            for obj in (tok, proc):
                if obj is not None:
                    val = getattr(obj, "pad_token_id", None)
                    if val is not None:
                        return int(val)
        eos_ids = self._resolve_eos_token_ids()
        if eos_ids:
            return eos_ids[0]
        return 0

    def _resolve_special_token_str(self, attr: str) -> str:
        """Resolve a special token string (e.g. ``eos_token`` or ``bos_token``) through processor/tokenizer."""
        proc = getattr(self.trainer, "processing_class", None)
        if proc is None:
            return ""
        tok = getattr(proc, "tokenizer", None)
        for obj in (proc, tok):
            if obj is not None:
                val = getattr(obj, attr, None)
                if isinstance(val, str) and val:
                    return val
        return ""

    def _resolve_chat_template_kwargs(self) -> dict[str, tp.Any]:
        """Resolve optional ``chat_template_kwargs`` from inner or outer trainer arguments."""
        inner_args = getattr(self.trainer, "arguments", None)
        raw = getattr(inner_args, "chat_template_kwargs", None)
        if not raw:
            raw = getattr(self.arguments, "chat_template_kwargs", None)
        return dict(raw) if isinstance(raw, collections.abc.Mapping) else {}

    def _apply_chat_template(
        self,
        apply_fn: tp.Callable[..., tp.Any],
        messages: list[dict[str, tp.Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        tools: list[dict[str, tp.Any]] | None = None,
    ) -> tp.Any:
        kwargs: dict[str, tp.Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools is not None:
            kwargs["tools"] = tools
        ct_kwargs = self._resolve_chat_template_kwargs()
        if ct_kwargs:
            kwargs.update(ct_kwargs)
        try:
            return apply_fn(messages, **kwargs)
        except TypeError:
            # Custom / stub tokenizers may not accept extra chat_template_kwargs or tools.
            return apply_fn(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)

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
                    self._apply_chat_template(
                        apply_fn,
                        [{"role": "user", "content": _TEMPLATE_PROBE}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
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
                formatted = self._apply_chat_template(
                    apply_fn,
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
                return str(
                    self._apply_chat_template(apply_fn, list(messages), tokenize=False, add_generation_prompt=True)
                )
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
            base = str(self._apply_chat_template(apply_fn, list(history), tokenize=False, add_generation_prompt=False))
            full = str(
                self._apply_chat_template(
                    apply_fn,
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
        eos = self._resolve_special_token_str("eos_token")
        # Locate the assistant turn's closing markup with a probe so short action texts that also occur
        # inside template markup cannot be mistaken for the action.
        closing: str | None = None
        try:
            probed = str(
                self._apply_chat_template(
                    apply_fn,
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
                    self._apply_chat_template(
                        apply_fn,
                        [{"role": "user", "content": raw_obs}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                bos = self._resolve_special_token_str("bos_token")
                if bos and rendered.startswith(bos):
                    rendered = rendered[len(bos) :]
                eos = self._resolve_special_token_str("eos_token")
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

    def _resolve_turn_max_new_tokens(self, prompt_len: int) -> int | None:
        """Compute per-turn ``max_new_tokens`` clamped to the trainer's static generation length bound."""
        requested = self.arguments.per_turn_max_tokens
        inner_args = getattr(self.trainer, "arguments", None)
        if requested is None:
            requested = getattr(inner_args, "max_completion_length", None) or self.arguments.max_completion_length
        if requested is None:
            return None
        req_int = max(1, int(requested))
        if inner_args is not None:
            try:
                from easydel.trainers._shared import esurge_generation

                return esurge_generation.clamp_new_tokens_to_bound(
                    inner_args,
                    prompt_len=int(prompt_len),
                    max_new_tokens=req_int,
                )
            except Exception:
                pass
        p_w, c_w = self._resolve_widths()
        bound = p_w + c_w
        if bound > 0 and prompt_len + req_int > bound:
            return max(1, bound - int(prompt_len))
        return req_int

    def _default_generate_turns_batch(
        self,
        state: tp.Any,
        prompts_ids: list[list[int]],
        prompts_text: list[str],
        turn_index: int,
    ) -> list[tuple[list[int], str, list[float] | None]]:
        """Generate one turn for a batch of active episodes in a single ``rollout()`` call."""
        if not prompts_ids:
            return []
        if self._generate_turn_fn is not None:
            return [
                self._generate_turn_fn(state, p_ids, p_txt, turn_index)
                for p_ids, p_txt in zip(prompts_ids, prompts_text, strict=True)
            ]

        rollout_fn = getattr(self.trainer, "rollout", None)
        if not callable(rollout_fn):
            raise RuntimeError(
                "AgentFlowTrainer requires either a `generate_turn_fn` callback or an inner trainer with `rollout(...)`."
            )

        pad_id = self._resolve_pad_token_id()
        p_w, c_w = self._resolve_widths()
        total_bound = max(1, p_w + c_w)
        # Keep every prompt row strictly shorter than the static generation bound so at least 1 token can be generated.
        clipped_rows = [
            row[-(total_bound - 1) :] if len(row) >= total_bound else (row if row else [pad_id]) for row in prompts_ids
        ]
        max_len = max(len(r) for r in clipped_rows)
        batch_size = len(clipped_rows)
        arr_np = np.full((batch_size, max_len), pad_id, dtype=np.int32)
        mask_np = np.zeros((batch_size, max_len), dtype=np.int32)
        for i, row in enumerate(clipped_rows):
            n = len(row)
            arr_np[i, :n] = row
            mask_np[i, :n] = 1

        overrides: dict[str, tp.Any] = {"num_return_sequences": 1}
        if self.arguments.per_turn_max_tokens is not None or max_len > p_w:
            clamped = self._resolve_turn_max_new_tokens(max_len)
            if clamped is not None:
                overrides["max_new_tokens"] = int(clamped)

        arr = jnp.asarray(arr_np, dtype=jnp.int32)
        mask = jnp.asarray(mask_np, dtype=jnp.int32)
        with self._rollout_lock:
            self._rollout_call_count += 1
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

        c_all = _to_host_numpy(results.completion_ids).astype(np.int32, copy=False)
        if getattr(results, "completion_mask", None) is not None:
            m_all = _to_host_numpy(results.completion_mask).astype(np.int32, copy=False)
        else:
            m_all = (c_all != pad_id).astype(np.int32)
        raw_lps = getattr(results, "sampling_per_token_logps", None)
        if raw_lps is None:
            raw_lps = getattr(results, "logprobs", None)
        lp_all = _to_host_numpy(raw_lps).astype(np.float32, copy=False) if raw_lps is not None else None

        outputs: list[tuple[list[int], str, list[float] | None]] = []
        for i in range(batch_size):
            c_row = c_all[i]
            m_row = m_all[i]
            kept = [int(t) for t, m in zip(c_row.tolist(), m_row.tolist(), strict=True) if m]
            lp_list: list[float] | None = None
            if lp_all is not None:
                lp_row = lp_all[i]
                lp_list = [float(x) for x, m in zip(lp_row.tolist(), m_row.tolist(), strict=True) if m]
            outputs.append((kept, self._decode_ids(kept), lp_list))
        return outputs

    def _default_generate_turn(
        self,
        state: tp.Any,
        prompt_ids: list[int],
        prompt_text: str,
        turn_index: int,
    ) -> tuple[list[int], str, list[float] | None]:
        if self._generate_turn_fn is not None:
            return self._generate_turn_fn(state, prompt_ids, prompt_text, turn_index)
        return self._default_generate_turns_batch(state, [prompt_ids], [prompt_text], turn_index)[0]

    def _instantiate_env(self, task: dict[str, tp.Any]) -> tp.Any:
        if not isinstance(task, dict):
            return self._env_factory()
        if self._env_accepted_keys is not None:
            filtered = {k: v for k, v in task.items() if k in self._env_accepted_keys}
            return self._env_factory(**filtered)
        return self._env_factory(**task)

    @staticmethod
    def _safe_close_env(env: tp.Any) -> None:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as close_exc:
                logger.warning("AgentFlow env.close() failed (%s): %s", type(close_exc).__name__, close_exc)

    def _prepare_initial_prompt(
        self, reset_res: tp.Any, task: dict[str, tp.Any]
    ) -> tuple[str, list[dict[str, str]], list[int], bool]:
        """Format and clip initial prompt tokens to ``prompt_width`` before turn 0 for TITO alignment."""
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
        prompt_width, _ = self._resolve_widths()
        prompt_clipped = len(initial_prompt_ids) > prompt_width
        if prompt_clipped:
            initial_prompt_ids = initial_prompt_ids[-prompt_width:]
        return initial_prompt, prompt_messages, initial_prompt_ids, prompt_clipped

    def _run_episode(
        self,
        state: tp.Any,
        task: dict[str, tp.Any],
        prompt_index: int,
        group_id: int,
        *,
        generate_turn_cb: tp.Callable[[list[int], str, int], tuple[list[int], str, list[float] | None]] | None = None,
    ) -> AgentFlowEpisode:
        # Let constructor errors (TypeError/ValueError) raise loudly rather than hiding bugs.
        env = self._instantiate_env(task)
        gen_cb = (
            generate_turn_cb
            if generate_turn_cb is not None
            else (lambda p_ids, p_txt, t_idx: self._default_generate_turn(state, p_ids, p_txt, t_idx))
        )
        try:
            if self._episode_runner is not None:
                return self._episode_runner(
                    env,
                    gen_cb,
                    task,
                    self.arguments,
                )

            reset_res = env.reset()
            initial_prompt, prompt_messages, initial_prompt_ids, prompt_clipped = self._prepare_initial_prompt(
                reset_res, task
            )

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
                    prompt_clipped=prompt_clipped,
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
            eos_id_set = set(self._resolve_eos_token_ids())

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

                act_ids, act_text, act_lps = gen_cb(ctx_ids, ctx_text, turn_idx)

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
                is_last_turn = (turn_idx + 1) >= self.arguments.max_turns
                is_terminal_turn = bool(step_res.terminated or step_res.truncated or is_last_turn or env_failed)
                raw_obs = str(step_res.observation or "").strip()
                obs_cap = self._resolve_obs_token_cap()
                if obs_cap is not None and raw_obs and not is_terminal_turn:
                    # Reserve ~24 tokens for chat wrapper markers so closing assistant header isn't truncated.
                    raw_ids = self._encode_text(raw_obs)
                    wrapper_reserve = 0 if self.arguments.skip_apply_chat_template else 24
                    raw_limit = max(0, obs_cap - wrapper_reserve)
                    if len(raw_ids) > raw_limit:
                        raw_obs = self._decode_ids(raw_ids[:raw_limit])

                ends_with_eos = bool(act_ids and eos_id_set and int(act_ids[-1]) in eos_id_set)
                chat_history.append({"role": "assistant", "content": act_text})
                obs_text = (
                    ""
                    if is_terminal_turn
                    else self._format_observation_turn(raw_obs, ends_with_eos=ends_with_eos, history=chat_history)
                )
                if not is_terminal_turn:
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
                prompt_clipped=prompt_clipped,
            )
        except Exception as exc:
            show_tb = not self._logged_episode_error
            self._logged_episode_error = True
            logger.warning("AgentFlow episode failed (%s): %s", type(exc).__name__, exc, exc_info=show_tb)
            raise
        finally:
            self._safe_close_env(env)

    def _decode_prompt_rows(self, ids: tp.Any, mask: tp.Any) -> list[str]:
        """Decode tokenized prompt rows back to text, keeping template special tokens only where needed.

        Padding is removed through ``mask`` (when given). Each row is first decoded with
        ``skip_special_tokens=False`` so chat-template markers survive and :meth:`_format_initial_prompt` can
        recognise an already-templated prompt. A row that is not templated and will be templated here is
        decoded again with ``skip_special_tokens=True``, so tokenizer specials such as BOS are not re-encoded
        inside the user turn. With ``skip_apply_chat_template`` the special-token text is kept.
        """
        arr = _to_host_numpy(ids)
        mask_arr = None if mask is None else _to_host_numpy(mask)
        tok = getattr(self.trainer, "processing_class", None)
        rows: list[str] = []
        for r, row in enumerate(arr):
            kept = [int(t) for c, t in enumerate(row.tolist()) if mask_arr is None or int(mask_arr[r, c])]
            if tok is None:
                rows.append(self._decode_ids(kept))
                continue
            try:
                raw = str(tok.decode(kept, skip_special_tokens=False))
            except TypeError:
                raw = str(tok.decode(kept))
            if self.arguments.skip_apply_chat_template or self._is_chat_templated(raw):
                rows.append(raw)
            else:
                rows.append(self._decode_ids(kept))
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
        two). ``scale_rewards == "batch_advantage"`` (the REINFORCE++ default) mirrors the inner GRPO-family
        trainer: each group is centred by the mean of its valid rows, then the centred valid rows of the
        whole batch are standardised by their own mean and std (zero if fewer than two); with
        ``group_size <= 1`` every centred row is zero. With ``group_size <= 1`` (PPO/VAPO, single
        rollouts) and ``"group"``/``"none"``, raw rewards are kept.

        Args:
            rewards: Episode rewards, shape ``[num_episodes]``.
            valid_mask: ``1.0`` for valid episodes and ``0.0`` for failed ones, same shape.
            group_size: Episodes per prompt group.

        Returns:
            Float32 advantages of shape ``[num_episodes]``.

        Raises:
            ValueError: If the inner trainer's ``scale_rewards`` is not ``"group"``, ``"batch"``,
                ``"batch_advantage"`` or ``"none"``.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        valid = (np.asarray(valid_mask, dtype=np.float64) > 0.0) & np.isfinite(rewards)
        out = np.zeros(rewards.shape, dtype=np.float64)
        scale_rewards = getattr(getattr(self.trainer, "arguments", None), "scale_rewards", "group")
        if isinstance(scale_rewards, bool):
            scale_rewards = "group" if scale_rewards else "none"
        scale_rewards = str(scale_rewards).lower()
        if scale_rewards not in {"group", "batch", "batch_advantage", "none"}:
            raise ValueError(
                f"Invalid value for scale_rewards: {scale_rewards}. "
                "Must be 'batch', 'batch_advantage', 'group', or 'none'."
            )

        def _normalize(idx: np.ndarray, *, scale: bool) -> None:
            sel = idx[valid[idx]]
            if sel.size < 2:
                return
            vals = rewards[sel]
            centred = vals - vals.mean()
            out[sel] = centred / (vals.std() + 1e-4) if scale else centred

        all_idx = np.arange(rewards.shape[0])
        if scale_rewards == "batch_advantage":
            size = max(1, int(group_size))
            centred = np.zeros(rewards.shape, dtype=np.float64)
            in_stats = np.zeros(rewards.shape, dtype=bool)
            for g in range(max(1, rewards.shape[0] // size)):
                idx = all_idx[g * size : (g + 1) * size]
                sel = idx[valid[idx]]
                if sel.size:
                    centred[sel] = rewards[sel] - rewards[sel].mean()
                    in_stats[sel] = True
            vals = centred[in_stats]
            if vals.size >= 2:
                out[in_stats] = (vals - vals.mean()) / (vals.std() + 1e-4)
            return out.astype(np.float32)

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

        batch = self._gather_batch(
            {
                "prompt_input_ids": prompt_ids_np[c_idx],
                "prompt_attention_mask": prompt_mask_np[c_idx],
                "chosen_input_ids": comp_ids_np[c_idx],
                "chosen_attention_mask": comp_attn_np[c_idx],
                "chosen_loss_mask": action_mask_np[c_idx] * keep,
                "rejected_input_ids": comp_ids_np[r_idx],
                "rejected_attention_mask": comp_attn_np[r_idx],
                "rejected_loss_mask": action_mask_np[r_idx] * keep,
            }
        )
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

    def _is_ssd(self) -> bool:
        from easydel.trainers.ssd_trainer import SSDTrainer

        return isinstance(self.trainer, SSDTrainer)

    def _is_seq_kd(self) -> bool:
        from easydel.trainers.seq_kd_trainer import SeqKDTrainer

        return isinstance(self.trainer, SeqKDTrainer)

    def _is_on_policy_distillation(self) -> bool:
        from easydel.trainers.on_policy_distillation_trainer import OnPolicyDistillationTrainer

        return isinstance(self.trainer, OnPolicyDistillationTrainer)

    def _is_gkd(self) -> bool:
        from easydel.trainers.generalized_knowledge_distillation_trainer import GKDTrainer

        return isinstance(self.trainer, GKDTrainer)

    def _is_distillation_or_preference_family(self) -> bool:
        return self._is_gkd() or self._is_on_policy_distillation() or self._is_seq_kd() or self._is_ssd()

    def _adapt_episodes_to_trainer(
        self,
        episodes: list[AgentFlowEpisode],
        group_size: int,
        *,
        state: tp.Any = None,
        is_train: bool = True,
    ) -> tuple[dict[str, tp.Any], dict[str, float]]:
        pad_id = self._resolve_pad_token_id()
        prompt_width, comp_width = self._resolve_widths()

        prompt_rows = [ep.prompt_token_ids for ep in episodes]
        comp_rows = [ep.completion_token_ids for ep in episodes]
        invalid_rows = [bool(ep.env_error or not np.isfinite(ep.episode_reward)) for ep in episodes]
        act_rows = [
            ([0] * len(ep.action_mask) if bad else ep.action_mask)
            for ep, bad in zip(episodes, invalid_rows, strict=True)
        ]
        comp_attn_rows = [
            ([0] * len(ep.completion_token_ids) if bad else [1] * len(ep.completion_token_ids))
            for ep, bad in zip(episodes, invalid_rows, strict=True)
        ]

        prompt_ids_np, prompt_mask_np = _pad_left_2d(prompt_rows, width=prompt_width, pad_value=pad_id)
        comp_ids_np = _pad_right_2d(comp_rows, width=comp_width, pad_value=pad_id, dtype=np.int32)
        comp_mask_np = _pad_right_2d(act_rows, width=comp_width, pad_value=0, dtype=np.int32)
        comp_attn_np = _pad_right_2d(comp_attn_rows, width=comp_width, pad_value=0, dtype=np.int32)

        prompt_ids = self._gather_array(prompt_ids_np)
        prompt_mask = self._gather_array(prompt_mask_np)
        completion_ids = self._gather_array(comp_ids_np)
        completion_mask = self._gather_array(comp_mask_np)
        completion_attention_mask = self._gather_array(comp_attn_np)

        rewards_np = np.asarray([ep.episode_reward for ep in episodes], dtype=np.float32)
        finite_rewards = np.isfinite(rewards_np)
        valid_np = np.asarray(
            [0.0 if (ep.env_error or not np.isfinite(ep.episode_reward)) else 1.0 for ep in episodes],
            dtype=np.float32,
        )
        rewards_safe_np = np.where(valid_np > 0.0, rewards_np, 0.0).astype(np.float32)
        advantages_np = self._compute_advantages(rewards_np, valid_np, group_size)

        clipped_count = sum(
            1
            for ep in episodes
            if ep.prompt_clipped or len(ep.prompt_token_ids) > prompt_width or len(ep.completion_token_ids) > comp_width
        )
        valid_bool = valid_np > 0.0
        ep_action_tokens = [max(0, int(sum(ep.action_mask))) for ep in episodes]
        ep_obs_tokens = [max(0, int(sum(len(t.observation_token_ids) for t in ep.turns))) for ep in episodes]
        ep_truncated = [
            bool(any(t.truncated for t in ep.turns) or len(ep.completion_token_ids) > comp_width) for ep in episodes
        ]
        pass_rate = float(np.mean(rewards_np[valid_bool] > 0.0)) if np.any(valid_bool) else 0.0
        truncation_rate = float(np.mean(ep_truncated)) if episodes else 0.0
        eff_group_size = max(1, int(group_size))
        zero_var_flags: list[bool] = []
        for g_start in range(0, len(episodes), eff_group_size):
            g_end = min(g_start + eff_group_size, len(episodes))
            g_vals = rewards_np[g_start:g_end][valid_bool[g_start:g_end]]
            zero_var_flags.append(g_vals.size < 2 or bool(np.isclose(float(np.std(g_vals)), 0.0, atol=1e-8)))
        zero_var_group_frac = float(np.mean(zero_var_flags)) if zero_var_flags else 0.0

        finite_vals = rewards_np[finite_rewards]
        metrics: dict[str, float] = {
            "agent_flow/mean_reward": float(np.mean(finite_vals)) if finite_vals.size else 0.0,
            "agent_flow/mean_turns": float(np.mean([len(ep.turns) for ep in episodes])) if episodes else 0.0,
            "agent_flow/action_tokens": float(np.sum(comp_mask_np)),
            "agent_flow/mean_action_tokens": float(np.mean(ep_action_tokens)) if episodes else 0.0,
            "agent_flow/mean_obs_tokens": float(np.mean(ep_obs_tokens)) if episodes else 0.0,
            "agent_flow/env_errors": float(sum(1 for ep in episodes if ep.env_error)),
            "agent_flow/clipped_episodes": float(clipped_count),
            "agent_flow/pass_rate": pass_rate,
            "rl/pass_rate": pass_rate,
            "agent_flow/zero_var_group_frac": zero_var_group_frac,
            "rl/zero_var_group_frac": zero_var_group_frac,
            "agent_flow/truncation_rate": truncation_rate,
            "rl/truncation_rate": truncation_rate,
        }

        has_lps = all(ep.logprobs is not None for ep in episodes)
        lp_np = (
            _pad_right_2d(
                [ep.logprobs or [] for ep in episodes],
                width=comp_width,
                pad_value=0.0,
                dtype=np.float32,
            )
            if has_lps
            else None
        )
        lp_arr = self._gather_array(lp_np) if lp_np is not None else None

        final_batch: dict[str, tp.Any]
        ref_logps_for_kl: tp.Any = None
        policy_logps_for_kl: tp.Any = lp_np

        if self._is_online_dpo():
            final_batch, pair_metrics = self._build_preference_pair_batch(
                episodes,
                group_size,
                prompt_ids_np=prompt_ids_np,
                prompt_mask_np=prompt_mask_np,
                comp_ids_np=comp_ids_np,
                action_mask_np=comp_mask_np,
                comp_attn_np=comp_attn_np,
            )
            metrics.update(pair_metrics)
        elif self._is_ppo_family():
            input_ids = self._gather_array(jnp.concatenate([prompt_ids, completion_ids], axis=1))
            attention_mask = self._gather_array(jnp.concatenate([prompt_mask, completion_attention_mask], axis=1))
            inner_args = getattr(self.trainer, "arguments", None)
            adv_2d = jnp.broadcast_to(jnp.asarray(advantages_np)[:, None], completion_ids.shape)
            ret_2d = jnp.broadcast_to(jnp.asarray(rewards_safe_np)[:, None], completion_ids.shape)
            pos_thresh = float(getattr(inner_args, "positive_reward_threshold", 0.0))
            positive_mask = jnp.asarray((rewards_safe_np > pos_thresh) & (valid_np > 0.0), dtype=jnp.float32)
            old_values = jnp.zeros_like(ret_2d)

            score_fn = getattr(self.trainer, "_score_rollout_policy_and_values", None)
            has_custom_scorer = "_score_rollout_policy_and_values" in getattr(self.trainer, "__dict__", {})
            has_compiled_scorer = callable(getattr(self.trainer, "compute_rollout_logps_values", None))
            can_score_rollout = callable(score_fn) and state is not None and (has_custom_scorer or has_compiled_scorer)

            if can_score_rollout:
                scored_logps, scored_values = score_fn(state, input_ids, attention_mask)
                scored_logps = jnp.asarray(scored_logps)[:, -comp_width:]
                old_logps = lp_arr if lp_arr is not None else scored_logps
                if scored_values is not None:
                    old_values = jnp.asarray(scored_values)[:, -comp_width:]
            elif lp_arr is not None:
                old_logps = lp_arr
            else:
                raise RuntimeError(
                    "AgentFlowTrainer with a PPO-family trainer needs per-token sampling logprobs from "
                    "`generate_turn_fn`/rollout, or a trainer that can score rollouts "
                    "(`_score_rollout_policy_and_values`) and a policy state."
                )

            compute_adv_ret = getattr(self.trainer, "_compute_advantages_and_returns", None)
            has_gae_args = (
                callable(compute_adv_ret)
                and inner_args is not None
                and hasattr(inner_args, "lam")
                and hasattr(inner_args, "gamma")
            )
            if has_gae_args:
                mask_bool = completion_mask.astype(jnp.bool_)
                old_logps_masked = jnp.where(mask_bool, old_logps.astype(jnp.float32), 0.0)
                old_values_masked = jnp.where(mask_bool, old_values.astype(jnp.float32), 0.0)
                kl_coef = float(getattr(inner_args, "kl_coef", 0.0) or 0.0)
                ref_state = getattr(self.trainer, "ref_state", None)
                compute_ref = getattr(self.trainer, "compute_refmodel_logps", None)
                if kl_coef != 0.0 and ref_state is not None and callable(compute_ref):
                    from easydel.trainers._shared.policy_losses import _MAX_SAFE_LOG_RATIO, _MIN_SAFE_LOG_RATIO

                    ref_logps = compute_ref(
                        ref_state.graphstate,
                        ref_state.graphother,
                        input_ids,
                        attention_mask,
                    )
                    ref_logps = jnp.asarray(ref_logps)[:, -comp_width:]
                    ref_logps = jnp.where(mask_bool, ref_logps.astype(jnp.float32), 0.0)
                    ref_logps_for_kl = self._gather_array(ref_logps)
                    logr = jnp.where(mask_bool, ref_logps - old_logps_masked, 0.0)
                    kl_estimator = str(getattr(inner_args, "kl_estimator", "k1") or "k1").lower()
                    if kl_estimator == "k1":
                        kl = -logr
                    else:
                        safe_logr = jnp.clip(logr, _MIN_SAFE_LOG_RATIO, _MAX_SAFE_LOG_RATIO)
                        kl = jnp.where(mask_bool, jnp.exp(safe_logr) - 1.0 - logr, 0.0)
                    non_score_reward = -kl_coef * kl
                else:
                    non_score_reward = jnp.zeros_like(old_logps_masked)

                # Pack active action tokens contiguously before running GAE so observation gaps
                # (completion_mask == 0 between turns) do not sever return bootstrapping across turns.
                batch_sz = int(comp_mask_np.shape[0])
                active_pos = [np.flatnonzero(comp_mask_np[b] > 0) for b in range(batch_sz)]
                max_active = max((int(pos.size) for pos in active_pos), default=0)
                if max_active > 0:
                    gather_idx_np = np.zeros((batch_sz, max_active), dtype=np.int32)
                    packed_mask_np = np.zeros((batch_sz, max_active), dtype=np.float32)
                    packed_score_np = np.zeros((batch_sz, max_active), dtype=np.float32)
                    for b, pos in enumerate(active_pos):
                        if pos.size > 0:
                            gather_idx_np[b, : pos.size] = pos.astype(np.int32)
                            packed_mask_np[b, : pos.size] = 1.0
                            packed_score_np[b, pos.size - 1] = float(rewards_safe_np[b])
                    row_idx = jnp.arange(batch_sz)[:, None]
                    gather_idx = jnp.asarray(gather_idx_np)
                    packed_mask = jnp.asarray(packed_mask_np)
                    packed_non_score = non_score_reward[row_idx, gather_idx] * packed_mask
                    packed_values = old_values_masked[row_idx, gather_idx] * packed_mask
                    packed_rewards = (packed_non_score + jnp.asarray(packed_score_np)) * packed_mask
                    whiten_fn = getattr(self.trainer, "_masked_whiten", None)
                    if bool(getattr(inner_args, "whiten_rewards", False)) and callable(whiten_fn):
                        packed_rewards = whiten_fn(packed_rewards, packed_mask, shift_mean=False) * packed_mask
                    packed_adv, packed_ret, _ = compute_adv_ret(packed_rewards, packed_values, packed_mask)
                    if bool(getattr(inner_args, "whiten_advantages", False)) and callable(whiten_fn):
                        packed_adv = whiten_fn(packed_adv, packed_mask, shift_mean=True) * packed_mask
                    flat_rows = np.concatenate(
                        [np.full(pos.size, b, dtype=np.int32) for b, pos in enumerate(active_pos) if pos.size > 0]
                    )
                    flat_cols = np.concatenate([pos.astype(np.int32) for pos in active_pos if pos.size > 0])
                    flat_pcols = np.concatenate(
                        [np.arange(pos.size, dtype=np.int32) for pos in active_pos if pos.size > 0]
                    )
                    adv_2d = (
                        jnp.zeros((batch_sz, comp_width), dtype=jnp.float32)
                        .at[flat_rows, flat_cols]
                        .set(packed_adv[flat_rows, flat_pcols])
                    )
                    ret_2d = (
                        jnp.zeros((batch_sz, comp_width), dtype=jnp.float32)
                        .at[flat_rows, flat_cols]
                        .set(packed_ret[flat_rows, flat_pcols])
                    )
                else:
                    adv_2d = jnp.zeros((batch_sz, comp_width), dtype=jnp.float32)
                    ret_2d = jnp.zeros((batch_sz, comp_width), dtype=jnp.float32)

            comp_mask_f32 = completion_mask.astype(jnp.float32)
            out: dict[str, tp.Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": completion_mask,
                "rewards": jnp.asarray(rewards_safe_np),
                "returns": ret_2d,
                "advantages": adv_2d,
                "old_values": old_values,
                "old_logps": old_logps,
                "positive_mask": positive_mask,
                "num_items_in_batch": jnp.sum(comp_mask_f32),
                "num_positive_tokens": jnp.sum(comp_mask_f32 * positive_mask[:, None]),
            }
            final_batch = self._gather_batch(out)
            policy_logps_for_kl = final_batch.get("old_logps")
        elif self._is_ssd():
            out = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "completion_attention_mask": completion_attention_mask,
            }
            final_batch = self._gather_batch(out)
        elif self._is_seq_kd():
            input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            attention_mask = jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
            full_completion_mask = jnp.concatenate([jnp.zeros_like(prompt_mask), completion_mask], axis=1)
            labels = jnp.where(full_completion_mask > 0, input_ids, jnp.asarray(-100, dtype=input_ids.dtype))
            out = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            final_batch = self._gather_batch(out)
        elif self._is_on_policy_distillation():
            input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            attention_mask = jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
            full_completion_mask = jnp.concatenate([jnp.zeros_like(prompt_mask), completion_mask], axis=1)
            out = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": full_completion_mask,
            }
            final_batch = self._gather_batch(out)
        elif self._is_gkd():
            input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            attention_mask = jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
            full_completion_mask = jnp.concatenate([jnp.zeros_like(prompt_mask), completion_mask], axis=1)
            labels = jnp.where(full_completion_mask > 0, input_ids, jnp.asarray(-100, dtype=input_ids.dtype))
            out = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": full_completion_mask,
                "action_mask": full_completion_mask,
                "rewards": jnp.asarray(rewards_safe_np),
            }
            final_batch = self._gather_batch(out)
        else:
            # GRPO / GMPO / GDPO / REINFORCE++ / GSPO / GFPO / RLOO / DPPO / PAPO / MiniLLM family.
            out = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "completion_attention_mask": completion_attention_mask,
                "advantages": jnp.asarray(advantages_np),
                "environment_rewards": jnp.asarray(rewards_safe_np),
                "difficulty_weights": jnp.asarray(valid_np, dtype=jnp.float32),
                "num_items_in_batch": jnp.sum(completion_mask),
            }
            if lp_arr is not None:
                out["sampling_per_token_logps"] = lp_arr
                out["old_per_token_logps"] = lp_arr

            inner_args = getattr(self.trainer, "arguments", None)
            needs_esurge_is = (
                bool(getattr(inner_args, "esurge_importance_sampling_correction", False)) and lp_arr is not None
            )
            force_old_logps = bool(getattr(self.trainer, "_async_grpo_force_old_logps", False))
            needs_state_old_logps = is_train and (needs_esurge_is or (force_old_logps and lp_arr is None))
            full_ids_cached: jax.Array | None = None
            full_mask_cached: jax.Array | None = None
            empty_model_kwargs_cached: dict[str, tp.Any] | None = None

            def _get_full_inputs() -> tuple[jax.Array, jax.Array, dict[str, tp.Any]]:
                nonlocal full_ids_cached, full_mask_cached, empty_model_kwargs_cached
                if full_ids_cached is None:
                    from easydel.trainers.group_relative_policy_optimization.grpo_trainer import (
                        normalize_generation_model_kwargs,
                    )

                    full_ids_cached = self._gather_array(jnp.concatenate([prompt_ids, completion_ids], axis=1))
                    full_mask_cached = self._gather_array(
                        jnp.concatenate([prompt_mask, completion_attention_mask], axis=1)
                    )
                    empty_model_kwargs_cached = normalize_generation_model_kwargs(None)
                return full_ids_cached, full_mask_cached, empty_model_kwargs_cached  # type: ignore[return-value]

            if needs_state_old_logps:
                compute_state_logps = getattr(self.trainer, "compute_state_logps", None)
                if callable(compute_state_logps) and state is not None:
                    full_ids, full_mask, empty_model_kwargs = _get_full_inputs()
                    policy_old_logps = compute_state_logps(
                        state,
                        full_ids,
                        full_mask,
                        empty_model_kwargs,
                        prompt_length=prompt_width,
                        logprob_vocab_chunk_size=getattr(inner_args, "logprob_vocab_chunk_size", None),
                    )
                    out["old_per_token_logps"] = self._gather_array(policy_old_logps)
                if needs_esurge_is:
                    is_fn = getattr(self.trainer, "_esurge_importance_sampling_ratio", None)
                    if callable(is_fn):
                        out["importance_sampling_ratio"] = is_fn(
                            out["old_per_token_logps"],
                            lp_arr,
                            completion_mask,
                        )

            beta = float(getattr(inner_args, "beta", 0.0) or 0.0)
            if beta != 0.0:
                compute_ref = getattr(self.trainer, "compute_refmodel_logps", None)
                ref_state = getattr(self.trainer, "ref_state", None)
                if callable(compute_ref) and ref_state is not None:
                    full_ids, full_mask, empty_model_kwargs = _get_full_inputs()
                    call_ref = getattr(self.trainer, "_call_compute_refmodel_logps", None)
                    prompt_len = int(prompt_ids.shape[1])
                    if callable(call_ref):
                        try:
                            ref_logps = call_ref(
                                ref_state.graphstate,
                                ref_state.graphother,
                                full_ids,
                                full_mask,
                                empty_model_kwargs,
                                prompt_length=prompt_len,
                            )
                        except ValueError as exc:
                            if "pjit does not support kwargs" not in str(exc):
                                raise
                            ref_logps = compute_ref(
                                ref_state.graphstate,
                                ref_state.graphother,
                                full_ids,
                                full_mask,
                                empty_model_kwargs,
                            )
                    else:
                        ref_logps = compute_ref(
                            ref_state.graphstate,
                            ref_state.graphother,
                            full_ids,
                            full_mask,
                            empty_model_kwargs,
                        )
                    if ref_logps.shape[1] == full_ids.shape[1]:
                        ref_logps = ref_logps[:, prompt_len:]
                    out["ref_per_token_logps"] = self._gather_array(ref_logps)
                elif lp_arr is not None:
                    out["ref_per_token_logps"] = lp_arr
                else:
                    out["ref_per_token_logps"] = jnp.zeros_like(completion_ids, dtype=jnp.float32)

            final_batch = self._gather_batch(out)
            ref_logps_for_kl = final_batch.get("ref_per_token_logps")
            policy_logps_for_kl = final_batch.get("old_per_token_logps")
            if policy_logps_for_kl is None:
                policy_logps_for_kl = lp_np

        icepop_active_ratios: np.ndarray | None = None
        is_ratio_val = final_batch.get("importance_sampling_ratio")
        if is_ratio_val is not None:
            # Compute on all hosts unconditionally (final_batch is already gathered) so metrics match across ranks.
            is_ratio_np = _to_host_numpy(is_ratio_val)
            icepop_active_ratios, ratio_mean, ratio_max, masked_frac = _compute_icepop_active_ratios(
                is_ratio_np,
                comp_mask_np,
            )
            metrics["agent_flow/icepop_ratio_mean"] = ratio_mean
            metrics["agent_flow/icepop_ratio_max"] = ratio_max
            metrics["agent_flow/icepop_masked_frac"] = masked_frac
            metrics["rl/icepop_ratio_mean"] = ratio_mean
            metrics["rl/icepop_ratio_max"] = ratio_max
            metrics["rl/icepop_masked_frac"] = masked_frac

        self._log_eyvan_agent_flow_telemetry(
            episodes=episodes,
            rewards_np=rewards_np,
            advantages_np=advantages_np,
            comp_mask_np=comp_mask_np,
            ep_action_tokens=ep_action_tokens,
            ep_obs_tokens=ep_obs_tokens,
            ep_truncated=ep_truncated,
            state=state,
            is_train=is_train,
            policy_logps=policy_logps_for_kl,
            ref_logps=ref_logps_for_kl,
            icepop_active_ratios=icepop_active_ratios,
        )

        return final_batch, metrics

    def _log_eyvan_agent_flow_telemetry(
        self,
        *,
        episodes: list[AgentFlowEpisode],
        rewards_np: np.ndarray,
        advantages_np: np.ndarray,
        comp_mask_np: np.ndarray,
        ep_action_tokens: list[int],
        ep_obs_tokens: list[int],
        ep_truncated: list[bool],
        state: tp.Any,
        is_train: bool,
        policy_logps: tp.Any = None,
        ref_logps: tp.Any = None,
        icepop_active_ratios: np.ndarray | None = None,
    ) -> None:
        """Emit native Eyvan episodes, rollouts, grouped completions, and distributions for AgentFlow."""
        if not episodes:
            return
        total_turns_before = int(getattr(self, "_eyvan_total_turns", 0))
        total_action_before = int(getattr(self, "_eyvan_total_action_tokens", 0))
        inner_args = getattr(self.trainer, "arguments", None)
        if inner_args is None or not bool(getattr(inner_args, "eyvan_tracking_enabled", False)):
            if is_train:
                self._eyvan_total_turns = total_turns_before + sum(len(ep.turns) for ep in episodes)
                self._eyvan_total_action_tokens = total_action_before + sum(ep_action_tokens)
            return

        try:
            cur_step = _extract_nonneg_step(state, fallback=0)
            raw_pv = getattr(self.trainer, "_async_rollout_policy_version", None)
            if raw_pv is None and state is not None:
                raw_pv = getattr(state, "policy_step", None)
            policy_version = max(0, int(raw_pv)) if raw_pv is not None else cur_step
            kind = "train" if is_train else "eval"
            env_name = getattr(self, "_eyvan_env_name", None) or _resolve_eyvan_env_name(
                getattr(self, "_env_spec", None),
                fallback="agent_flow",
            )

            # 1. Record every episode with monotonically advancing cumulative axes.
            cum_turns = total_turns_before
            cum_action_tokens = total_action_before
            for idx, ep in enumerate(episodes):
                ep_turns_n = len(ep.turns)
                ep_act_n = ep_action_tokens[idx]
                ep_obs_n = ep_obs_tokens[idx]
                if is_train:
                    cum_turns += ep_turns_n
                    cum_action_tokens += ep_act_n
                ep_reward = float(ep.episode_reward)
                ep_success = bool((not ep.env_error) and math.isfinite(ep_reward) and ep_reward > 0.0)
                finite_step_rewards = [float(r) for r in ep.step_rewards if math.isfinite(float(r))]
                mean_turn_reward = (
                    float(sum(finite_step_rewards) / len(finite_step_rewards)) if finite_step_rewards else 0.0
                )
                ep_axes = {"env_turns": float(cum_turns), "action_tokens": float(cum_action_tokens)}
                _eyvan_log_episode(
                    inner_args,
                    ret=ep_reward,
                    length=ep_turns_n,
                    success=ep_success,
                    kind=kind,
                    env=env_name,
                    policy_version=policy_version,
                    learner_version=cur_step,
                    step=cur_step,
                    axes=ep_axes,
                    extra={
                        "action_tokens": float(ep_act_n),
                        "obs_tokens": float(ep_obs_n),
                        "truncated": 1.0 if ep_truncated[idx] else 0.0,
                        "env_error": 1.0 if ep.env_error else 0.0,
                        "prompt_clipped": 1.0 if ep.prompt_clipped else 0.0,
                        "mean_turn_reward": mean_turn_reward,
                    },
                )
            if is_train:
                self._eyvan_total_turns = cum_turns
                self._eyvan_total_action_tokens = cum_action_tokens
            batch_axes = {"env_turns": float(cum_turns), "action_tokens": float(cum_action_tokens)}

            # 2. Sample up to 8 multi-turn rollouts (best, worst, then remaining; skip 0-turn episodes).
            non_empty_indices = [idx for idx, ep in enumerate(episodes) if len(ep.turns) > 0]
            rollout_indices = _select_rollout_sample_indices(
                [float(ep.episode_reward) for ep in episodes],
                non_empty_indices,
                cap=_EYVAN_MAX_ROLLOUTS_PER_STEP,
            )
            for idx in rollout_indices:
                ep = episodes[idx]
                turns = ep.turns[:_EYVAN_MAX_ROLLOUT_STEPS]
                if not turns:
                    continue
                tracks: dict[str, list[float | None]] = {
                    "reward": [float(t.reward) for t in turns],
                    "action_tokens": [float(len(t.action_token_ids)) for t in turns],
                    "obs_tokens": [float(len(t.observation_token_ids)) for t in turns],
                    "terminated": [1.0 if t.terminated else 0.0 for t in turns],
                    "truncated": [1.0 if t.truncated else 0.0 for t in turns],
                }
                if any(t.logprobs for t in turns):
                    tracks["mean_logprob"] = [
                        float(np.mean(np.asarray(t.logprobs, dtype=np.float64))) if t.logprobs else None for t in turns
                    ]
                action_ids, action_names = _encode_action_labels([t.action_text for t in turns])
                _eyvan_log_rollout(
                    inner_args,
                    tracks=tracks,
                    actions=action_ids,
                    action_names=action_names,
                    episode_id=f"s{cur_step}-p{ep.prompt_index}-g{ep.group_id}",
                    env=env_name,
                    kind=kind,
                    policy_version=policy_version,
                    step=cur_step,
                    axes=batch_axes,
                )

            # 3. Compute optional per-episode KL locally (never invoking cross-host collectives).
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
                    kl_estimator = str(getattr(inner_args, "kl_estimator", "k3") or "k3")
                    per_ep_kl = _compute_per_episode_kl_numpy(pol_np, ref_np, comp_mask_np, kl_estimator=kl_estimator)

            # 4. Log up to 8 grouped multi-turn completions.
            grouped: dict[int, list[int]] = {}
            for idx, ep in enumerate(episodes):
                grouped.setdefault(int(ep.prompt_index), []).append(idx)
            for p_idx, grp_indices in list(grouped.items())[:_EYVAN_MAX_COMPLETION_GROUPS_PER_STEP]:
                grp_indices = grp_indices[:_EYVAN_MAX_GROUP_RESPONSES]
                if not grp_indices:
                    continue
                first_ep = episodes[grp_indices[0]]
                if first_ep.prompt_token_ids:
                    prompt_text = self._decode_ids(first_ep.prompt_token_ids)
                elif first_ep.turns:
                    prompt_text = first_ep.turns[0].prompt_text
                else:
                    prompt_text = ""
                responses = [
                    _format_agent_flow_transcript(episodes[i], decode_ids=self._decode_ids) for i in grp_indices
                ]
                grp_rewards = [float(episodes[i].episode_reward) for i in grp_indices]
                grp_adv = [float(advantages_np[i]) for i in grp_indices]
                grp_tokens = [ep_action_tokens[i] for i in grp_indices]
                grp_trunc = [bool(ep_truncated[i]) for i in grp_indices]
                grp_kl = (
                    [per_ep_kl[i] for i in grp_indices]
                    if per_ep_kl is not None and any(per_ep_kl[i] is not None for i in grp_indices)
                    else None
                )
                group_prefix = f"s{cur_step}" if is_train else f"eval-s{cur_step}"
                _eyvan_log_completions(
                    inner_args,
                    prompt=prompt_text,
                    responses=responses,
                    rewards=grp_rewards,
                    advantages=grp_adv,
                    group=f"{group_prefix}-p{p_idx}",
                    tokens=grp_tokens,
                    kl=grp_kl,
                    truncated=grp_trunc,
                    policy_version=policy_version,
                    step=cur_step,
                )

            # 5. Log RL distributions.
            _eyvan_log_histogram(inner_args, "rl/reward_dist", rewards_np, step=cur_step, axes=batch_axes)
            _eyvan_log_histogram(inner_args, "rl/advantage_dist", advantages_np, step=cur_step, axes=batch_axes)
            _eyvan_log_histogram(
                inner_args,
                "rl/episode_turns_dist",
                np.asarray([float(len(ep.turns)) for ep in episodes], dtype=np.float32),
                step=cur_step,
                axes=batch_axes,
            )
            _eyvan_log_histogram(
                inner_args,
                "rl/action_tokens_dist",
                np.asarray(ep_action_tokens, dtype=np.float32),
                step=cur_step,
                axes=batch_axes,
            )
            if icepop_active_ratios is not None and icepop_active_ratios.size > 0:
                _eyvan_log_histogram(
                    inner_args,
                    "rl/icepop_ratio_dist",
                    icepop_active_ratios,
                    step=cur_step,
                    axes=batch_axes,
                )
        except Exception as exc:
            logger.warning("AgentFlow Eyvan telemetry logging failed (%s): %s", type(exc).__name__, exc)

    def _decode_ids_raw(self, ids: list[int]) -> str:
        """Decode token IDs keeping structural special tokens (e.g. ``<think>``, ``<tool_call>``) intact."""
        tok = getattr(self.trainer, "processing_class", None)
        if tok is None:
            return "".join(chr(max(32, min(126, int(i)))) for i in ids)
        try:
            text = str(tok.decode(ids, skip_special_tokens=False))
        except TypeError:
            text = str(tok.decode(ids))
        eos = self._resolve_special_token_str("eos_token")
        if eos and text.endswith(eos):
            text = text[: -len(eos)]
        return text

    def _parse_harness_turn_output(
        self,
        raw_text: str,
        token_ids: list[int],
        logprobs: list[float] | None,
    ) -> tp.Any:
        """Convert raw model output into a :class:`GatewayTurnOutput` with parsed thinking and tool calls."""
        from easydel.environments.harnesses import GatewayTurnOutput

        reasoning_text: str | None = None
        visible = raw_text
        # Handle both explicit <think>...</think> and prompt-prefilled ... </think> (Qwen3.5/3.8 templates).
        m_full = re.search(r"<think>(.*?)</think>", visible, flags=re.DOTALL)
        if m_full is not None:
            reasoning_text = m_full.group(1).strip()
            visible = (visible[: m_full.start()] + visible[m_full.end() :]).strip()
        elif "</think>" in visible:
            head, tail = visible.split("</think>", 1)
            reasoning_text = head.removeprefix("<think>").strip()
            visible = tail.strip()

        tool_calls: list[dict[str, tp.Any]] = []

        def _parse_tc_block(block: str) -> dict[str, tp.Any] | None:
            body = block.strip()
            # 1) Qwen3.5/3.8 XML format: <function=name><parameter=k>v</parameter></function>
            fn_m = re.search(r"<function=([^>\s]+)>(.*?)</function>", body, flags=re.DOTALL)
            if fn_m is not None:
                fn_name = fn_m.group(1).strip()
                fn_body = fn_m.group(2)
                args_dict: dict[str, tp.Any] = {}
                for pm in re.finditer(r"<parameter=([^>\s]+)>(.*?)</parameter>", fn_body, flags=re.DOTALL):
                    pkey = pm.group(1).strip()
                    pval = pm.group(2).strip()
                    try:
                        args_dict[pkey] = json.loads(pval)
                    except Exception:
                        args_dict[pkey] = pval
                return {
                    "type": "function",
                    "function": {"name": fn_name, "arguments": json.dumps(args_dict, ensure_ascii=False)},
                }
            # 2) Hermes / OpenAI JSON format: {"name": ..., "arguments": ...}
            try:
                obj = json.loads(body)
                if isinstance(obj, dict) and "name" in obj:
                    raw_args = obj.get("arguments", {})
                    args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                    return {"type": "function", "function": {"name": str(obj["name"]), "arguments": args_str}}
            except Exception:
                pass
            return None

        spans_to_remove: list[tuple[int, int]] = []
        for tc_m in re.finditer(r"<tool_call>(.*?)</tool_call>", visible, flags=re.DOTALL):
            parsed = _parse_tc_block(tc_m.group(1))
            if parsed is not None:
                tool_calls.append(parsed)
                spans_to_remove.append((tc_m.start(), tc_m.end()))
        if spans_to_remove:
            chunks: list[str] = []
            cur = 0
            for s, e in spans_to_remove:
                chunks.append(visible[cur:s])
                cur = e
            chunks.append(visible[cur:])
            visible = "".join(chunks).strip()

        return GatewayTurnOutput(
            token_ids=token_ids,
            text=visible,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls or None,
            logprobs=logprobs,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    def _encode_harness_messages(
        self,
        messages: list[dict[str, tp.Any]],
        tools: list[dict[str, tp.Any]] | None,
    ) -> list[int]:
        """Tokenize harness messages using the model's chat template without duplicating system preambles on deltas."""
        tok = getattr(self.trainer, "processing_class", None)
        apply_fn = getattr(tok, "apply_chat_template", None)
        if not callable(apply_fn):
            text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
            return self._encode_text(text)

        # Initial prompt turn (tools passed, or messages starts with system/user conversation root).
        if tools is not None or (messages and messages[0].get("role") == "system"):
            rendered = str(
                self._apply_chat_template(
                    apply_fn,
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                )
            )
            return self._encode_text(rendered)

        # Continuation delta (e.g. tool / user response messages): diff against a sentinel prefix so
        # chat templates with default system prompts never re-inject a system block mid-trajectory.
        prefix_msgs: list[dict[str, tp.Any]] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": _TEMPLATE_PROBE},
            {"role": "assistant", "content": _TEMPLATE_PROBE},
        ]
        try:
            base = str(self._apply_chat_template(apply_fn, prefix_msgs, tokenize=False, add_generation_prompt=False))
            full = str(
                self._apply_chat_template(
                    apply_fn,
                    [*prefix_msgs, *messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            if full.startswith(base):
                return self._encode_text(full[len(base) :])
        except Exception:
            pass
        rendered = str(self._apply_chat_template(apply_fn, list(messages), tokenize=False, add_generation_prompt=True))
        return self._encode_text(rendered)

    def _collect_harness_episodes_host0(
        self,
        envs: list[tp.Any],
        jobs: list[tuple[int, int, dict[str, tp.Any]]],
        workers: int,
        gen_batch_cb: tp.Callable[
            [list[list[int]], list[str], int],
            list[tuple[list[int], str, list[float] | None]],
        ],
    ) -> list[AgentFlowEpisode]:
        """Run concurrent ``HarnessEnvironment`` episodes through a micro-batching TITO HTTP gateway.

        All active CLI harness processes run concurrently inside ``ThreadPoolExecutor``. Whenever
        harnesses POST LLM turn requests to the intercepting :class:`HarnessTITOGateway`, their
        context token lists are micro-batched on Host 0 and dispatched in a single batched SPMD
        ``gen_batch_cb`` call across all TPU/GPU hosts.
        """
        from easydel.environments.harnesses import HarnessTITOGateway
        from easydel.environments.runtime import ContainerSandboxSession

        num_jobs = len(jobs)
        pending_q: queue.Queue[tuple[list[int], concurrent.futures.Future[tp.Any]] | None] = queue.Queue()
        active_lock = threading.Lock()
        remaining_episodes = num_jobs
        turn_counter = 0

        def _gw_generate(
            context_tokens: list[int],
            _messages: list[dict[str, tp.Any]],
            _tools: list[dict[str, tp.Any]] | None,
        ) -> tp.Any:
            fut: concurrent.futures.Future[tp.Any] = concurrent.futures.Future()
            pending_q.put((list(context_tokens), fut))
            return fut.result()

        uses_container = any(
            getattr(env, "_backend", "local") in {"kiri", "docker"}
            or isinstance(getattr(env, "_session", None), ContainerSandboxSession)
            for env in envs
        )
        bind_host = "0.0.0.0" if uses_container else "127.0.0.1"
        adv_host: str | None = None
        if uses_container:
            import socket as _socket

            try:
                adv_host = _socket.gethostbyname(_socket.gethostname())
            except Exception:
                adv_host = "host.docker.internal"

        gateway = HarnessTITOGateway(
            generate_fn=_gw_generate,
            encode_messages_fn=self._encode_harness_messages,
            model_name=str(getattr(self.arguments, "model_name", None) or "easydel-rl-policy"),
            host=bind_host,
            advertise_host=adv_host,
        ).start()

        for env in envs:
            if getattr(env, "_gateway", None) is None:
                env._gateway = gateway

        def _run_one(idx: int) -> AgentFlowEpisode:
            nonlocal remaining_episodes
            p_idx, grp_id, task = jobs[idx]
            env = envs[idx]
            try:
                r0 = env.reset()
                if r0.info.get("env_error"):
                    return AgentFlowEpisode(
                        prompt_index=p_idx,
                        group_id=grp_id,
                        prompt_token_ids=self._encode_text(str(task.get("prompt", ""))),
                        completion_token_ids=[],
                        action_mask=[],
                        step_rewards=[0.0],
                        episode_reward=0.0,
                        turns=(),
                        logprobs=[],
                        env_error=True,
                    )
                step_res = env.step("")
                trajs = list(getattr(env, "last_trajectories", ()))
                env_err = bool(step_res.info.get("env_error")) or not trajs
                if env_err:
                    return AgentFlowEpisode(
                        prompt_index=p_idx,
                        group_id=grp_id,
                        prompt_token_ids=self._encode_text(str(task.get("prompt", ""))),
                        completion_token_ids=[],
                        action_mask=[],
                        step_rewards=[0.0],
                        episode_reward=0.0,
                        turns=(),
                        logprobs=[],
                        env_error=True,
                    )
                # Merge / pick the root TITO trajectory segment captured by HarnessTITOGateway.
                traj = trajs[-1]
                ids_all = [int(t) for t in np.asarray(traj.input_ids).tolist()]
                amask_all = [int(bool(m)) for m in np.asarray(traj.action_mask).tolist()]
                lps_all = (
                    [float(x) for x in np.asarray(traj.behavior_logps).tolist()]
                    if traj.behavior_logps is not None
                    else None
                )
                first_act = next((i for i, m in enumerate(amask_all) if m), len(ids_all))
                prompt_w, _ = self._resolve_widths()
                prompt_ids = ids_all[:first_act]
                prompt_clipped = len(prompt_ids) > prompt_w
                if prompt_clipped:
                    prompt_ids = prompt_ids[-prompt_w:]
                comp_ids = ids_all[first_act:]
                comp_amask = amask_all[first_act:]
                comp_lps = lps_all[first_act:] if lps_all is not None else None
                turns = _reconstruct_turns_from_tito(
                    prompt_ids=prompt_ids,
                    comp_ids=comp_ids,
                    comp_amask=comp_amask,
                    comp_lps=comp_lps,
                    final_reward=float(step_res.reward),
                    decode_ids=self._decode_ids,
                    truncated=bool(getattr(step_res, "truncated", False)),
                )
                return AgentFlowEpisode(
                    prompt_index=p_idx,
                    group_id=grp_id,
                    prompt_token_ids=prompt_ids,
                    completion_token_ids=comp_ids,
                    action_mask=comp_amask,
                    step_rewards=[float(t.reward) for t in turns] if turns else [float(step_res.reward)],
                    episode_reward=float(step_res.reward),
                    turns=turns,
                    logprobs=comp_lps,
                    env_error=False,
                    prompt_clipped=prompt_clipped,
                )
            except Exception as exc:
                logger.warning("Harness episode %d failed (%s): %s", idx, type(exc).__name__, exc)
                return AgentFlowEpisode(
                    prompt_index=p_idx,
                    group_id=grp_id,
                    prompt_token_ids=self._encode_text(str(task.get("prompt", ""))),
                    completion_token_ids=[],
                    action_mask=[],
                    step_rewards=[0.0],
                    episode_reward=0.0,
                    turns=(),
                    logprobs=[],
                    env_error=True,
                )
            finally:
                with active_lock:
                    remaining_episodes -= 1
                    if remaining_episodes <= 0:
                        pending_q.put(None)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                ep_futs = [pool.submit(_run_one, idx) for idx in range(num_jobs)]

                # Micro-batching SPMD dispatch loop on Host 0's main thread.
                while True:
                    with active_lock:
                        rem = remaining_episodes
                    if rem <= 0 and pending_q.empty():
                        break
                    try:
                        first = pending_q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if first is None:
                        continue

                    batch_items = [first]
                    # Coalesce concurrent harness HTTP turns arriving within the window so fast tool calls
                    # (pytest/sqlite/workspace setup) join the same batched eSurge call; breaks immediately once
                    # all active episodes have posted.
                    coalesce_s = 2.0 if turn_counter == 0 else 1.0
                    deadline = time.monotonic() + coalesce_s
                    while True:
                        with active_lock:
                            target = remaining_episodes
                        if len(batch_items) >= max(1, target):
                            break
                        left = deadline - time.monotonic()
                        if left <= 0:
                            break
                        try:
                            nxt = pending_q.get(timeout=min(0.05, left))
                            if nxt is not None:
                                batch_items.append(nxt)
                        except queue.Empty:
                            continue

                    ctx_ids_batch = [item[0] for item in batch_items]
                    ctx_txt_batch = [self._decode_ids(ids) for ids in ctx_ids_batch]
                    try:
                        raw_outs = gen_batch_cb(ctx_ids_batch, ctx_txt_batch, turn_counter)
                        turn_counter += 1
                    except BaseException as exc:
                        for _, fut in batch_items:
                            if not fut.done():
                                fut.set_exception(exc)
                        raise
                    for (_, fut), (tok_ids, _decoded_txt, lps) in zip(batch_items, raw_outs, strict=True):
                        if not fut.done():
                            raw_txt = self._decode_ids_raw(tok_ids)
                            fut.set_result(self._parse_harness_turn_output(raw_txt, tok_ids, lps))

                return [fut.result() for fut in ep_futs]
        finally:
            gateway.close()

    def _collect_episodes_multihost(
        self,
        state: tp.Any,
        jobs: list[tuple[int, int, dict[str, tp.Any]]],
        workers: int,
    ) -> list[AgentFlowEpisode]:
        """Coordinate multi-host episode collection so all hosts enter SPMD ``rollout()`` in deterministic lockstep.

        Host 0 owns all environment instances (running ``env.reset()`` / ``env.step()`` concurrently when
        ``workers > 1``) and broadcasts batched turn prompts to worker hosts via ``broadcast_one_to_all`` so every
        host invokes :meth:`_default_generate_turns_batch` in identical SPMD lockstep. Completed episodes are
        serialized and broadcast back to all hosts so every host builds identical training batches.
        """
        from jax.experimental import multihost_utils

        def _bcast(arr: np.ndarray) -> np.ndarray:
            return np.asarray(multihost_utils.broadcast_one_to_all(arr))

        if jax.process_index() != 0:
            while True:
                ctrl = _bcast(np.zeros((4,), dtype=np.int32))
                op, dim_a, dim_b, turn_idx = int(ctrl[0]), int(ctrl[1]), int(ctrl[2]), int(ctrl[3])
                if op == 1:
                    ids_list = _bcast(np.zeros((dim_b,), dtype=np.int32)).tolist() if dim_b > 0 else []
                    self._default_generate_turn(state, ids_list, self._decode_ids(ids_list), turn_idx)
                    multihost_utils.sync_global_devices(f"agent_flow_post_turn_{turn_idx}")
                elif op == 2:
                    ids_2d = _bcast(np.zeros((dim_a, dim_b), dtype=np.int32))
                    lens_1d = _bcast(np.zeros((dim_a,), dtype=np.int32))
                    batch_ids = [ids_2d[i, : int(lens_1d[i])].tolist() for i in range(dim_a)]
                    batch_txts = [self._decode_ids(r) for r in batch_ids]
                    self._default_generate_turns_batch(state, batch_ids, batch_txts, turn_idx)
                    multihost_utils.sync_global_devices(f"agent_flow_post_turn_batch_{turn_idx}")
                elif op == 0:
                    payload_len = int(_bcast(np.zeros((1,), dtype=np.int32))[0])
                    payload = _bcast(np.zeros((payload_len,), dtype=np.uint8)).tobytes() if payload_len > 0 else b"[]"
                    return _deserialize_episodes(payload)
                else:
                    err_len = int(_bcast(np.zeros((1,), dtype=np.int32))[0])
                    err_msg = (
                        _bcast(np.zeros((err_len,), dtype=np.uint8)).tobytes().decode("utf-8", errors="replace")
                        if err_len > 0
                        else "host 0 episode collection failed"
                    )
                    raise RuntimeError(f"AgentFlow multi-host rollout aborted on host 0: {err_msg}")

        workers_in_turn = False

        def _synced_generate_turn(
            p_ids: list[int],
            p_txt: str,
            t_idx: int,
        ) -> tuple[list[int], str, list[float] | None]:
            nonlocal workers_in_turn
            ids_np = np.asarray(p_ids, dtype=np.int32).reshape(-1)
            with self._rollout_lock:
                _bcast(np.asarray([1, 1, int(ids_np.size), int(t_idx)], dtype=np.int32))
                if ids_np.size > 0:
                    _bcast(ids_np)
                workers_in_turn = True
                result = self._default_generate_turn(state, ids_np.tolist(), p_txt, int(t_idx))
                multihost_utils.sync_global_devices(f"agent_flow_post_turn_{int(t_idx)}")
                workers_in_turn = False
                return result

        def _synced_generate_turns_batch(
            batch_ids: list[list[int]],
            batch_txts: list[str],
            t_idx: int,
        ) -> list[tuple[list[int], str, list[float] | None]]:
            nonlocal workers_in_turn
            if not batch_ids:
                return []
            pad_id = self._resolve_pad_token_id()
            n_rows = len(batch_ids)
            max_len = max(max(len(r), 1) for r in batch_ids)
            ids_2d = np.full((n_rows, max_len), pad_id, dtype=np.int32)
            lens_1d = np.zeros((n_rows,), dtype=np.int32)
            for i, row in enumerate(batch_ids):
                lens_1d[i] = len(row)
                if row:
                    ids_2d[i, : len(row)] = row
            with self._rollout_lock:
                _bcast(np.asarray([2, n_rows, max_len, int(t_idx)], dtype=np.int32))
                _bcast(ids_2d)
                _bcast(lens_1d)
                workers_in_turn = True
                results = self._default_generate_turns_batch(state, batch_ids, batch_txts, int(t_idx))
                multihost_utils.sync_global_devices(f"agent_flow_post_turn_batch_{int(t_idx)}")
                workers_in_turn = False
                return results

        use_batched = self._generate_turn_fn is None and callable(getattr(self.trainer, "rollout", None))
        try:
            if self._episode_runner is not None:
                episodes = [
                    self._run_episode(state, task, p_idx, grp_id, generate_turn_cb=_synced_generate_turn)
                    for p_idx, grp_id, task in jobs
                ]
            else:
                episodes = self._collect_episodes_lockstep_host0(
                    jobs,
                    max(1, workers),
                    _synced_generate_turn,
                    gen_batch_cb=_synced_generate_turns_batch if use_batched else None,
                )
        except BaseException as exc:
            if not workers_in_turn:
                err_bytes = f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace")
                _bcast(np.asarray([-1, 0, 0, 0], dtype=np.int32))
                _bcast(np.asarray([len(err_bytes)], dtype=np.int32))
                if err_bytes:
                    _bcast(np.frombuffer(err_bytes, dtype=np.uint8).copy())
            raise

        payload_bytes = _serialize_episodes(episodes)
        _bcast(np.asarray([0, 0, 0, 0], dtype=np.int32))
        _bcast(np.asarray([len(payload_bytes)], dtype=np.int32))
        if payload_bytes:
            _bcast(np.frombuffer(payload_bytes, dtype=np.uint8).copy())
        return episodes

    def _collect_episodes_lockstep_host0(
        self,
        jobs: list[tuple[int, int, dict[str, tp.Any]]],
        workers: int,
        gen_cb: tp.Callable[[list[int], str, int], tuple[list[int], str, list[float] | None]],
        *,
        gen_batch_cb: tp.Callable[
            [list[list[int]], list[str], int],
            list[tuple[list[int], str, list[float] | None]],
        ]
        | None = None,
    ) -> list[AgentFlowEpisode]:
        """Run host-0 env I/O concurrently across ``workers`` while batching model generation per turn."""
        from easydel.environments.harnesses import HarnessEnvironment

        num_jobs = len(jobs)
        envs: list[tp.Any] = [None] * num_jobs
        eos_id_set = set(self._resolve_eos_token_ids())
        try:
            # Instantiate synchronously so constructor errors propagate immediately, then reset concurrently.
            for idx, (_p_idx, _grp_id, task) in enumerate(jobs):
                envs[idx] = self._instantiate_env(task)

            if envs and all(isinstance(env, HarnessEnvironment) for env in envs):
                batch_fn = gen_batch_cb or (
                    lambda ids_b, txts_b, t_idx: [
                        gen_cb(ids_i, txt_i, t_idx) for ids_i, txt_i in zip(ids_b, txts_b, strict=True)
                    ]
                )
                return self._collect_harness_episodes_host0(envs, jobs, workers, batch_fn)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                reset_futs = {pool.submit(envs[idx].reset): idx for idx in range(num_jobs)}
                reset_results: list[tp.Any] = [None] * num_jobs
                for fut in concurrent.futures.as_completed(reset_futs):
                    reset_results[reset_futs[fut]] = fut.result()

                states: list[dict[str, tp.Any]] = []
                active_indices: list[int] = []
                episodes: list[AgentFlowEpisode | None] = [None] * num_jobs
                for idx, (p_idx, grp_id, task) in enumerate(jobs):
                    reset_res = reset_results[idx]
                    initial_prompt, prompt_messages, initial_prompt_ids, prompt_clipped = self._prepare_initial_prompt(
                        reset_res, task
                    )
                    if reset_res.info.get("env_error"):
                        episodes[idx] = AgentFlowEpisode(
                            prompt_index=p_idx,
                            group_id=grp_id,
                            prompt_token_ids=initial_prompt_ids,
                            completion_token_ids=[],
                            action_mask=[],
                            step_rewards=[0.0],
                            episode_reward=0.0,
                            turns=(),
                            logprobs=[],
                            env_error=True,
                            prompt_clipped=prompt_clipped,
                        )
                        states.append({})
                        continue
                    states.append(
                        {
                            "prompt_index": p_idx,
                            "group_id": grp_id,
                            "initial_prompt": initial_prompt,
                            "initial_prompt_ids": initial_prompt_ids,
                            "prompt_clipped": prompt_clipped,
                            "turns": [],
                            "running_ids": list(initial_prompt_ids),
                            "completion_ids": [],
                            "action_mask": [],
                            "all_logprobs": [],
                            "has_logprobs": True,
                            "step_rewards": [],
                            "env_failed": False,
                            "chat_history": list(prompt_messages),
                        }
                    )
                    active_indices.append(idx)

                for turn_idx in range(self.arguments.max_turns):
                    if not active_indices:
                        break
                    turn_actions: dict[int, tuple[list[int], str, list[float] | None, str]] = {}
                    batch_ctx_ids: list[list[int]] = []
                    batch_ctx_txts: list[str] = []
                    for idx in active_indices:
                        st = states[idx]
                        if self._context_manager is not None:
                            ctx_text = self._context_manager(st["turns"], st["initial_prompt"], self.arguments)
                            ctx_ids = self._encode_text(ctx_text)
                        elif self.arguments.context_mode == "append":
                            ctx_ids = list(st["running_ids"])
                            ctx_text = self._decode_ids(ctx_ids)
                        else:
                            ctx_text = _default_context_manager(st["initial_prompt"], st["turns"], self.arguments)
                            ctx_ids = self._encode_text(ctx_text)
                        batch_ctx_ids.append(ctx_ids)
                        batch_ctx_txts.append(ctx_text)

                    if gen_batch_cb is not None:
                        batch_outs = gen_batch_cb(batch_ctx_ids, batch_ctx_txts, turn_idx)
                        for idx, ctx_text, (act_ids, act_text, act_lps) in zip(
                            active_indices, batch_ctx_txts, batch_outs, strict=True
                        ):
                            turn_actions[idx] = (act_ids, act_text, act_lps, ctx_text)
                    else:
                        for idx, ctx_ids, ctx_text in zip(active_indices, batch_ctx_ids, batch_ctx_txts, strict=True):
                            act_ids, act_text, act_lps = gen_cb(ctx_ids, ctx_text, turn_idx)
                            turn_actions[idx] = (act_ids, act_text, act_lps, ctx_text)

                    def _step_one(ep_idx: int, action_text: str) -> tuple[tp.Any | None, Exception | None]:
                        try:
                            return envs[ep_idx].step(action_text), None
                        except Exception as exc:
                            return None, exc

                    step_futs = {pool.submit(_step_one, idx, turn_actions[idx][1]): idx for idx in active_indices}
                    step_outcomes: dict[int, tuple[tp.Any | None, Exception | None]] = {}
                    for fut in concurrent.futures.as_completed(step_futs):
                        step_outcomes[step_futs[fut]] = fut.result()

                    next_active: list[int] = []
                    obs_cap = self._resolve_obs_token_cap()
                    is_last_turn = (turn_idx + 1) >= self.arguments.max_turns
                    for idx in active_indices:
                        st = states[idx]
                        act_ids, act_text, act_lps, ctx_text = turn_actions[idx]
                        step_res, step_exc = step_outcomes[idx]
                        if step_exc is not None:
                            show_tb = not self._logged_episode_error
                            self._logged_episode_error = True
                            logger.warning(
                                "AgentFlow env.step raised %s: %s",
                                type(step_exc).__name__,
                                step_exc,
                                exc_info=show_tb,
                            )
                            st["env_failed"] = True
                            st["step_rewards"].append(0.0)
                            st["completion_ids"].extend(act_ids)
                            st["action_mask"].extend([0] * len(act_ids))
                            st["all_logprobs"].extend([0.0] * len(act_ids))
                            continue

                        if step_res.info.get("env_error"):
                            st["env_failed"] = True
                        is_terminal_turn = bool(
                            step_res.terminated or step_res.truncated or is_last_turn or st["env_failed"]
                        )
                        raw_obs = str(step_res.observation or "").strip()
                        if obs_cap is not None and raw_obs and not is_terminal_turn:
                            raw_ids = self._encode_text(raw_obs)
                            wrapper_reserve = 0 if self.arguments.skip_apply_chat_template else 24
                            raw_limit = max(0, obs_cap - wrapper_reserve)
                            if len(raw_ids) > raw_limit:
                                raw_obs = self._decode_ids(raw_ids[:raw_limit])

                        ends_with_eos = bool(act_ids and eos_id_set and int(act_ids[-1]) in eos_id_set)
                        st["chat_history"].append({"role": "assistant", "content": act_text})
                        obs_text = (
                            ""
                            if is_terminal_turn
                            else self._format_observation_turn(
                                raw_obs, ends_with_eos=ends_with_eos, history=st["chat_history"]
                            )
                        )
                        if not is_terminal_turn:
                            st["chat_history"].append({"role": "user", "content": raw_obs})
                        obs_ids = self._encode_text(obs_text) if obs_text else []
                        if obs_cap is not None and len(obs_ids) > obs_cap and self.arguments.skip_apply_chat_template:
                            obs_ids = obs_ids[:obs_cap]
                            obs_text = self._decode_ids(obs_ids)
                        step_r = float(step_res.reward)
                        st["step_rewards"].append(step_r)

                        st["running_ids"].extend(act_ids)
                        st["completion_ids"].extend(act_ids)
                        st["action_mask"].extend([1] * len(act_ids))
                        if act_lps is not None and len(act_lps) == len(act_ids):
                            st["all_logprobs"].extend(act_lps)
                        else:
                            st["has_logprobs"] = False
                            st["all_logprobs"].extend([0.0] * len(act_ids))

                        if obs_ids:
                            st["running_ids"].extend(obs_ids)
                            st["completion_ids"].extend(obs_ids)
                            st["action_mask"].extend([0] * len(obs_ids))
                            st["all_logprobs"].extend([0.0] * len(obs_ids))

                        st["turns"].append(
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
                        if not (step_res.terminated or step_res.truncated or st["env_failed"]):
                            next_active.append(idx)
                    active_indices = next_active

                for idx in range(num_jobs):
                    if episodes[idx] is not None:
                        continue
                    st = states[idx]
                    if st["env_failed"]:
                        action_mask_list = [0] * len(st["action_mask"])
                        ep_reward = 0.0
                    else:
                        action_mask_list = st["action_mask"]
                        ep_reward = _aggregate_episode_reward(
                            st["step_rewards"],
                            reward_mode=self.arguments.reward_mode,
                            gamma=float(self.arguments.gamma),
                        )
                    episodes[idx] = AgentFlowEpisode(
                        prompt_index=st["prompt_index"],
                        group_id=st["group_id"],
                        prompt_token_ids=st["initial_prompt_ids"],
                        completion_token_ids=st["completion_ids"],
                        action_mask=action_mask_list,
                        step_rewards=st["step_rewards"],
                        episode_reward=ep_reward,
                        turns=tuple(st["turns"]),
                        logprobs=st["all_logprobs"] if st["has_logprobs"] else None,
                        env_error=st["env_failed"],
                        prompt_clipped=st["prompt_clipped"],
                    )
                return episodes  # type: ignore[return-value]
        finally:
            for env in envs:
                if env is not None:
                    self._safe_close_env(env)

    def _wrapped_preprocess_batch_input(
        self,
        state: tp.Any,
        batch: collections.abc.Mapping[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        if is_train:
            check_eyvan_reclaim = getattr(self.trainer, "_should_save_eyvan_reclaim_checkpoint", None)
            if callable(check_eyvan_reclaim):
                raw_step = getattr(state, "step", 0)
                try:
                    cur_step = int(jax.device_get(raw_step))
                except Exception:
                    cur_step = int(raw_step) if isinstance(raw_step, (int, np.integer)) else 0
                eyvan_notice = check_eyvan_reclaim(cur_step)
                if eyvan_notice is not None:
                    handle_reclaim = getattr(self.trainer, "_handle_eyvan_reclaim_preemption", None)
                    if callable(handle_reclaim):
                        raise handle_reclaim(state=state, step=cur_step, notice=eyvan_notice)
        tasks = self._extract_tasks(batch)
        group_size = self._resolve_group_size(is_train)

        jobs: list[tuple[int, int, dict[str, tp.Any]]] = []
        for p_idx, task in enumerate(tasks):
            for g_idx in range(group_size):
                jobs.append((p_idx, p_idx * group_size + g_idx, task))

        episodes: list[AgentFlowEpisode] = [None] * len(jobs)  # type: ignore[list-item]
        workers = min(self.arguments.max_workers, max(1, len(jobs)))
        rollout_calls_before = self._rollout_call_count
        try:
            if jax.process_count() > 1:
                episodes = self._collect_episodes_multihost(state, jobs, workers)
            elif (
                self._episode_runner is None
                and self._generate_turn_fn is None
                and callable(getattr(self.trainer, "rollout", None))
            ):
                episodes = self._collect_episodes_lockstep_host0(
                    jobs,
                    max(1, workers),
                    lambda p_ids, p_txt, t_idx: self._default_generate_turn(state, p_ids, p_txt, t_idx),
                    gen_batch_cb=lambda b_ids, b_txts, t_idx: self._default_generate_turns_batch(
                        state, b_ids, b_txts, t_idx
                    ),
                )
            elif workers == 1:
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
        finally:
            used_rollout = self._rollout_call_count > rollout_calls_before
            in_async_mode = bool(getattr(self.arguments, "async_mode", False)) or bool(
                getattr(getattr(self.trainer, "arguments", None), "async_mode", False)
            )
            if used_rollout and not in_async_mode:
                release_fn = getattr(self.trainer, "release_generation_runtime", None)
                if callable(release_fn):
                    release_fn(state=state)

        if episodes and all(ep.env_error for ep in episodes):
            raise RuntimeError(
                f"All {len(episodes)} AgentFlow episodes failed with env_error=True; aborting step to avoid "
                "training on all-zero masks."
            )

        return self._adapt_episodes_to_trainer(episodes, group_size, state=state, is_train=is_train)

    @staticmethod
    def _uninstall_eyvan_preemption_handler() -> None:
        """Uninstall Eyvan's graceful preemption signal handler and clear live signal notices without raising."""
        try:
            from easydel.utils.eyvan_helpers import uninstall_preemption_handler

            uninstall_preemption_handler()
        except Exception:
            pass

    def train(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Delegate training to the wrapped trainer after installing Eyvan's preemption handler."""
        try:
            from easydel.utils.eyvan_helpers import install_preemption_handler

            install_preemption_handler()
        except Exception:
            pass
        try:
            return self.trainer.train(*args, **kwargs)
        finally:
            self._uninstall_eyvan_preemption_handler()

    def eval(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Delegate evaluation to the wrapped trainer."""
        return self.trainer.eval(*args, **kwargs)

    def evaluate(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Alias for :meth:`eval`, falling back to a wrapped ``evaluate`` method if present."""
        eval_fn = getattr(self.trainer, "eval", None)
        if callable(eval_fn):
            return eval_fn(*args, **kwargs)
        return self.trainer.evaluate(*args, **kwargs)

    def close(self) -> None:
        """Close any owned environment runtime and tear down wrapped trainer resources."""
        try:
            close_fn = getattr(self.trainer, "close", None)
            if callable(close_fn):
                close_fn()
            else:
                teardown_fn = getattr(self.trainer, "_teardown_resources", None)
                if callable(teardown_fn):
                    teardown_fn()
        finally:
            try:
                self._close_owned_env_runtime()
            finally:
                self._uninstall_eyvan_preemption_handler()

    def finish(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Delegate finalization to the wrapped trainer and close any owned environment runtime."""
        try:
            finish_fn = getattr(self.trainer, "finish", None)
            if callable(finish_fn):
                return finish_fn(*args, **kwargs)
            return None
        finally:
            try:
                self._close_owned_env_runtime()
            finally:
                self._uninstall_eyvan_preemption_handler()

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
