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

"""Asynchronous multi-turn RL environment trajectory collection and filtering.

Bridges :class:`AgenticEnvironment` / Kiri ``SandboxPool`` instances to EasyDeL's
asynchronous online RL stack (:class:`BoundedAsyncRolloutCoordinator` and
:class:`OnlineRLTrajectory`), implementing:

* **Completion-order collection** so slow multi-turn sandboxes never block short
  trajectories;
* **Token-in-Token-out (TITO)** span tracking where model-generated action
  tokens carry ``action_mask=1`` and ``behavior_logps`` while environment
  observations carry ``attention_mask=1, action_mask=0``;
* **Multi-version staleness tracking** ``(w_0, ..., w_k)`` across multi-turn
  weight syncs, dropping trajectories whose oldest policy version ``w_0`` lags
  by more than ``max_staleness``;
* **Environment crash isolation** (``info["env_error"]``), dropping infra-failed
  rollouts and padding incomplete GRPO groups when at least half of the group
  succeeded.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import math
import threading
import typing as tp

import numpy as np

from easydel.trainers._online_rl.async_rollout import (
    AsyncRolloutCancelled,
    AsyncRolloutError,
    AsyncRolloutWorkerError,
    BoundedAsyncRolloutCoordinator,
    CompletedRollout,
    PolicySnapshot,
    RolloutTicket,
)
from easydel.trainers._online_rl.trajectory import (
    OnlineRLTrajectory,
    OnlineRLTrajectoryCollator,
    TrajectorySegment,
    TrajectorySpan,
)
from easydel.trainers._shared.environment import AgenticEnvironment

from ._base import make_environment_factory


@dataclasses.dataclass(frozen=True)
class GeneratedTurn:
    """Single model turn returned by a token-in-token-out generation callback.

    Attributes:
        token_ids: Generated token IDs for this assistant turn (non-empty).
        text: Decoded assistant action text passed to ``env.step(text)``.
        logprobs: Optional per-token behavior log-probabilities aligned 1:1
            with ``token_ids``.
        policy_version: Policy weight version used to sample this turn. When
            ``None``, defaults to the snapshot version active at turn start.
    """

    token_ids: list[int]
    text: str
    logprobs: list[float] | None = None
    policy_version: int | None = None


TurnGenerateFn = tp.Callable[[tp.Any, list[int], int], GeneratedTurn]
EncodeFn = tp.Callable[[str], list[int]]


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    """Completed multi-turn episode with version provenance and TITO trajectory."""

    prompt_index: int
    group_id: int
    trajectory: OnlineRLTrajectory | None
    total_reward: float
    turns: int
    oldest_policy_version: int
    latest_policy_version: int
    terminated: bool
    truncated: bool
    env_error: bool
    env_error_message: str | None = None

    def staleness(self, current_policy_version: int) -> int:
        """Policy lag measured from the oldest turn weight version ``w_0``."""
        return max(0, int(current_policy_version) - self.oldest_policy_version)


@dataclasses.dataclass(frozen=True)
class GroupRolloutBatch:
    """Filtered and group-balanced episodes ready for group-relative RL.

    ``episodes[i].trajectory is trajectories[i]`` for every ``i``. With ``drop_env_errors=False`` the
    infra-failed episodes are reported in ``env_error_episodes`` instead: they never count toward a group's
    ``min_valid`` threshold, are never used for padding and never appear in ``episodes``.
    """

    episodes: list[EpisodeRecord]
    trajectories: list[OnlineRLTrajectory]
    dropped_stale: int
    dropped_env_error: int
    dropped_groups: int
    padded_samples: int
    env_error_episodes: list[EpisodeRecord] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class _EpisodeTask:
    prompt_index: int
    group_id: int
    seed: int | None
    env_kwargs: dict[str, tp.Any]


def _format_observation_turn(obs: str) -> str:
    return f"\n<observation>\n{obs}\n</observation>\n"


class AsyncEnvironmentRolloutCollector:
    """Completion-order asynchronous collector for multi-turn RL environments.

    Workers step environments concurrently in background threads. Before each
    turn of an ongoing episode, the worker reads the latest published
    :class:`PolicySnapshot` so multi-turn rollouts automatically pick up weight
    updates while tracking ``(oldest_policy_version, latest_policy_version)``
    for staleness filtering.
    """

    def __init__(
        self,
        environment_factory: str | collections.abc.Mapping[str, tp.Any] | tp.Callable[..., AgenticEnvironment],
        *,
        generate_turn_fn: TurnGenerateFn,
        encode_fn: EncodeFn,
        observation_formatter: tp.Callable[[str], str] = _format_observation_turn,
        max_workers: int = 8,
        max_outstanding: int = 32,
        max_turns: int = 8,
        max_staleness: int | None = 2,
        drop_env_errors: bool = True,
    ):
        if max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {max_turns}")
        if max_staleness is not None and max_staleness < 0:
            raise ValueError(f"max_staleness must be non-negative or None, got {max_staleness}")

        self._env_factory = make_environment_factory(environment_factory)
        self._generate_turn = generate_turn_fn
        self._encode = encode_fn
        self._format_obs = observation_formatter
        self._max_turns = int(max_turns)
        self._max_staleness = max_staleness
        self._drop_env_errors = bool(drop_env_errors)
        self._max_outstanding = int(max_outstanding)
        self._next_trajectory_id = 0
        self._traj_id_lock = threading.Lock()
        self._aside_lock = threading.Lock()
        self._aside: collections.deque[CompletedRollout[EpisodeRecord] | AsyncRolloutError] = collections.deque()

        self._snapshot_lock = threading.Lock()
        self._latest_snapshot: PolicySnapshot[tp.Any] | None = None
        self._coordinator: BoundedAsyncRolloutCoordinator[tp.Any, _EpisodeTask, EpisodeRecord] = (
            BoundedAsyncRolloutCoordinator(
                self._run_episode,
                max_workers=max_workers,
                max_outstanding=max_outstanding,
                thread_name_prefix="edenv-async",
            )
        )

    @property
    def coordinator(self) -> BoundedAsyncRolloutCoordinator[tp.Any, _EpisodeTask, EpisodeRecord]:
        return self._coordinator

    def _free_capacity(self) -> int:
        """Submissions that fit right now without blocking ``submit`` (may be ``0``; all callers counted)."""
        return max(0, self._max_outstanding - self._coordinator.outstanding_count)

    def _set_aside(self, outcome: CompletedRollout[EpisodeRecord] | AsyncRolloutError) -> None:
        """Keep a completion of a submission that the current ``collect_groups`` call did not make.

        Foreign completions are never counted toward the group batch; :meth:`collect_episodes` consumes them
        first, in the order they were set aside.
        """
        with self._aside_lock:
            self._aside.append(outcome)

    def _pop_aside(self) -> CompletedRollout[EpisodeRecord] | AsyncRolloutError | None:
        with self._aside_lock:
            return self._aside.popleft() if self._aside else None

    @property
    def set_aside_count(self) -> int:
        """Foreign completions drained by :meth:`collect_groups` and waiting for :meth:`collect_episodes`."""
        with self._aside_lock:
            return len(self._aside)

    def _classify(self, ep: EpisodeRecord, current_policy_version: int) -> str:
        """Return ``"keep"``, ``"env_error"`` (dropped) or ``"stale"`` for a completed episode.

        With ``drop_env_errors=False`` env-error episodes are kept as zero-reward records whose
        ``env_error`` flag is set (their ``trajectory`` is ``None``); consumers must check the flag.
        """
        if ep.env_error:
            if self._drop_env_errors:
                return "env_error"
        elif ep.trajectory is None:
            return "env_error"
        if self._max_staleness is not None and ep.staleness(current_policy_version) > self._max_staleness:
            return "stale"
        return "keep"

    def publish_policy(
        self,
        state: tp.Any,
        *,
        version: int,
        metadata: collections.abc.Mapping[str, tp.Any] | None = None,
    ) -> PolicySnapshot[tp.Any]:
        """Publish a new policy weight version visible to in-flight multi-turn episodes."""
        snap = PolicySnapshot(version=int(version), state=state, metadata=metadata or {})
        with self._snapshot_lock:
            self._latest_snapshot = snap
        return snap

    def _current_snapshot(self, fallback: PolicySnapshot[tp.Any]) -> PolicySnapshot[tp.Any]:
        with self._snapshot_lock:
            return self._latest_snapshot if self._latest_snapshot is not None else fallback

    def submit_episode(
        self,
        *,
        prompt_index: int = 0,
        group_id: int = 0,
        seed: int | None = None,
        env_kwargs: collections.abc.Mapping[str, tp.Any] | None = None,
        snapshot: PolicySnapshot[tp.Any] | None = None,
        timeout: float | None = None,
    ) -> RolloutTicket:
        """Enqueue one multi-turn environment episode for background execution and return its ticket."""
        with self._snapshot_lock:
            active_snap = snapshot or self._latest_snapshot
        if active_snap is None:
            raise RuntimeError("Call publish_policy() or pass `snapshot` before submitting episodes.")
        task = _EpisodeTask(
            prompt_index=int(prompt_index),
            group_id=int(group_id),
            seed=seed,
            env_kwargs=dict(env_kwargs or {}),
        )
        return self._coordinator.submit(task, snapshot=active_snap, timeout=timeout)

    def _run_episode(self, initial_snapshot: PolicySnapshot[tp.Any], task: _EpisodeTask) -> EpisodeRecord:
        env = self._env_factory(**task.env_kwargs)
        try:
            reset_res = env.reset(seed=task.seed)
            if reset_res.info.get("env_error"):
                return EpisodeRecord(
                    prompt_index=task.prompt_index,
                    group_id=task.group_id,
                    trajectory=None,
                    total_reward=0.0,
                    turns=0,
                    oldest_policy_version=initial_snapshot.version,
                    latest_policy_version=initial_snapshot.version,
                    terminated=True,
                    truncated=True,
                    env_error=True,
                    env_error_message=str(reset_res.info.get("env_error_message", "reset env_error")),
                )

            prompt_tokens = list(self._encode(reset_res.observation))
            if not prompt_tokens:
                prompt_tokens = [0]

            all_tokens: list[int] = list(prompt_tokens)
            action_mask: list[bool] = [False] * len(prompt_tokens)
            rewards: list[float] = [0.0] * len(prompt_tokens)
            behavior_logps: list[float] = [0.0] * len(prompt_tokens)
            has_all_logps = True
            spans: list[TrajectorySpan] = [TrajectorySpan(start=0, stop=len(prompt_tokens), kind="prompt", segment_id=0)]

            turn_versions: list[int] = []
            total_reward = 0.0
            terminated = False
            truncated = False
            max_env_steps = env.max_steps
            turn_limit = min(self._max_turns, max_env_steps) if max_env_steps is not None else self._max_turns

            for turn_idx in range(turn_limit):
                snap = self._current_snapshot(initial_snapshot)
                gen = self._generate_turn(snap.state, list(all_tokens), turn_idx)
                act_ids = [int(t) for t in gen.token_ids]
                if not act_ids:
                    truncated = True
                    break

                ver = int(gen.policy_version) if gen.policy_version is not None else snap.version
                turn_versions.append(ver)

                act_start = len(all_tokens)
                all_tokens.extend(act_ids)
                action_mask.extend([True] * len(act_ids))
                rewards.extend([0.0] * len(act_ids))
                if gen.logprobs is not None and len(gen.logprobs) == len(act_ids):
                    behavior_logps.extend(float(lp) for lp in gen.logprobs)
                else:
                    has_all_logps = False
                    behavior_logps.extend([0.0] * len(act_ids))
                spans.append(TrajectorySpan(start=act_start, stop=len(all_tokens), kind="action", segment_id=0))

                step_res = env.step(gen.text)
                if step_res.info.get("env_error"):
                    return EpisodeRecord(
                        prompt_index=task.prompt_index,
                        group_id=task.group_id,
                        trajectory=None,
                        total_reward=0.0,
                        turns=turn_idx + 1,
                        oldest_policy_version=min(turn_versions),
                        latest_policy_version=max(turn_versions),
                        terminated=True,
                        truncated=True,
                        env_error=True,
                        env_error_message=str(step_res.info.get("env_error_message", "step env_error")),
                    )

                step_reward = float(step_res.reward)
                rewards[-1] += step_reward
                total_reward += step_reward
                terminated = bool(step_res.terminated)
                truncated = bool(step_res.truncated)
                if terminated or truncated:
                    break

                if turn_idx + 1 < turn_limit and step_res.observation:
                    obs_ids = [int(t) for t in self._encode(self._format_obs(step_res.observation))]
                    if obs_ids:
                        obs_start = len(all_tokens)
                        all_tokens.extend(obs_ids)
                        action_mask.extend([False] * len(obs_ids))
                        rewards.extend([0.0] * len(obs_ids))
                        behavior_logps.extend([0.0] * len(obs_ids))
                        spans.append(
                            TrajectorySpan(start=obs_start, stop=len(all_tokens), kind="observation", segment_id=0)
                        )

            oldest_ver = min(turn_versions) if turn_versions else initial_snapshot.version
            latest_ver = max(turn_versions) if turn_versions else initial_snapshot.version

            if not any(action_mask):
                return EpisodeRecord(
                    prompt_index=task.prompt_index,
                    group_id=task.group_id,
                    trajectory=None,
                    total_reward=total_reward,
                    turns=len(turn_versions),
                    oldest_policy_version=oldest_ver,
                    latest_policy_version=latest_ver,
                    terminated=terminated,
                    truncated=True,
                    env_error=False,
                )

            with self._traj_id_lock:
                traj_id = self._next_trajectory_id
                self._next_trajectory_id += 1

            traj = OnlineRLTrajectory(
                input_ids=np.asarray(all_tokens, dtype=np.int32),
                attention_mask=np.ones(len(all_tokens), dtype=np.bool_),
                action_mask=np.asarray(action_mask, dtype=np.bool_),
                rewards=np.asarray(rewards, dtype=np.float32),
                behavior_logps=np.asarray(behavior_logps, dtype=np.float32) if has_all_logps else None,
                spans=tuple(spans),
                segments=(TrajectorySegment(segment_id=0, start=0, stop=len(all_tokens), is_terminal=True),),
                trajectory_id=traj_id,
                policy_version=oldest_ver,
                metadata={
                    "prompt_index": task.prompt_index,
                    "group_id": task.group_id,
                    "oldest_policy_version": oldest_ver,
                    "latest_policy_version": latest_ver,
                    "turns": len(turn_versions),
                    "terminated": terminated,
                    "truncated": truncated,
                },
            )
            return EpisodeRecord(
                prompt_index=task.prompt_index,
                group_id=task.group_id,
                trajectory=traj,
                total_reward=total_reward,
                turns=len(turn_versions),
                oldest_policy_version=oldest_ver,
                latest_policy_version=latest_ver,
                terminated=terminated,
                truncated=truncated,
                env_error=False,
            )
        finally:
            env.close()

    def collect_episodes(
        self,
        count: int,
        *,
        current_policy_version: int,
        timeout_per_episode: float | None = None,
    ) -> tuple[list[EpisodeRecord], dict[str, int]]:
        """Drain ``count`` valid episodes in completion order, filtering stale/crashed runs."""
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        valid: list[EpisodeRecord] = []
        stats = {"dropped_stale": 0, "dropped_env_error": 0, "worker_errors": 0}

        while len(valid) < count and (self.set_aside_count > 0 or self._coordinator.outstanding_count > 0):
            # Completions set aside by ``collect_groups`` (submissions it did not make) are consumed first.
            aside = self._pop_aside()
            try:
                if isinstance(aside, AsyncRolloutError):
                    raise aside
                completed = aside if aside is not None else self._coordinator.next_completed(timeout=timeout_per_episode)
            except AsyncRolloutCancelled:
                continue
            except AsyncRolloutWorkerError:
                stats["worker_errors"] += 1
                continue

            ep = completed.value
            verdict = self._classify(ep, current_policy_version)
            if verdict == "env_error":
                stats["dropped_env_error"] += 1
                continue
            if verdict == "stale":
                stats["dropped_stale"] += 1
                continue
            valid.append(ep)

        return valid, stats

    def collect_collated_batch(
        self,
        count: int,
        *,
        current_policy_version: int,
        collator: OnlineRLTrajectoryCollator,
        timeout_per_episode: float | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
        """Collect ``count`` valid TITO trajectories and collate into learner arrays."""
        episodes, stats = self.collect_episodes(
            count,
            current_policy_version=current_policy_version,
            timeout_per_episode=timeout_per_episode,
        )
        if not episodes:
            raise RuntimeError(f"No valid episodes survived filtering; stats={stats}")
        trajectories = [ep.trajectory for ep in episodes if ep.trajectory is not None]
        collated = collator(trajectories)
        mean_lag = float(np.mean([ep.staleness(current_policy_version) for ep in episodes]))
        mean_turns = float(np.mean([ep.turns for ep in episodes]))
        mean_reward = float(np.mean([ep.total_reward for ep in episodes]))
        metrics: dict[str, float | int] = {
            "async_env/collected": len(episodes),
            "async_env/dropped_stale": stats["dropped_stale"],
            "async_env/dropped_env_error": stats["dropped_env_error"],
            "async_env/worker_errors": stats["worker_errors"],
            "async_env/mean_staleness": mean_lag,
            "async_env/mean_turns": mean_turns,
            "async_env/mean_reward": mean_reward,
        }
        return collated, metrics

    def collect_groups(
        self,
        *,
        num_groups: int,
        group_size: int,
        current_policy_version: int,
        env_kwargs_per_group: collections.abc.Sequence[collections.abc.Mapping[str, tp.Any]] | None = None,
        base_seed: int = 0,
        timeout_per_episode: float | None = None,
    ) -> GroupRolloutBatch:
        """Submit and collect ``num_groups * group_size`` episodes with GLM-5 group recovery.

        If environment crashes or staleness drops leave a group incomplete, groups
        with ``>= ceil(group_size / 2)`` valid rollouts are padded by repeating
        valid members; sparser groups are dropped so group-relative advantages
        are never computed on degenerate singletons.
        """
        if num_groups <= 0 or group_size <= 0:
            raise ValueError("num_groups and group_size must be positive.")
        if env_kwargs_per_group is not None and len(env_kwargs_per_group) != num_groups:
            raise ValueError("env_kwargs_per_group length must match num_groups.")

        pending = [
            (g, base_seed + g * group_size + k, env_kwargs_per_group[g] if env_kwargs_per_group is not None else None)
            for g in range(num_groups)
            for k in range(group_size)
        ]
        by_group: dict[int, list[EpisodeRecord]] = {g: [] for g in range(num_groups)}
        env_error_episodes: list[EpisodeRecord] = []
        dropped_stale = 0
        dropped_env_error = 0
        ours: set[int] = set()
        next_idx = 0

        # Interleave submission and draining: the coordinator only frees capacity in ``next_completed``, so
        # submitting everything up front deadlocks when ``num_groups * group_size > max_outstanding``. Only this
        # call's own submission ids count as in flight; capacity is recomputed from the live coordinator count
        # each iteration (other callers may hold credits), and at zero capacity the loop drains instead of
        # blocking in ``submit``. Completions of foreign submissions are set aside for ``collect_episodes``.
        while next_idx < len(pending) or ours:
            while next_idx < len(pending) and self._free_capacity() > 0:
                g, seed, kw = pending[next_idx]
                ticket = self.submit_episode(prompt_index=g, group_id=g, seed=seed, env_kwargs=kw)
                ours.add(ticket.submission_id)
                next_idx += 1
            if self._coordinator.outstanding_count == 0:
                break
            try:
                completed = self._coordinator.next_completed(timeout=timeout_per_episode)
            except (AsyncRolloutCancelled, AsyncRolloutWorkerError) as exc:
                if exc.ticket.submission_id in ours:
                    ours.discard(exc.ticket.submission_id)
                    dropped_env_error += 1
                else:
                    self._set_aside(exc)
                continue
            if completed.ticket.submission_id not in ours:
                self._set_aside(completed)
                continue
            ours.discard(completed.ticket.submission_id)
            ep = completed.value
            verdict = self._classify(ep, current_policy_version)
            if verdict == "env_error":
                dropped_env_error += 1
            elif verdict == "stale":
                dropped_stale += 1
            elif ep.env_error:
                # Kept (drop_env_errors=False) but reported separately: never a valid member or padding source.
                env_error_episodes.append(ep)
            else:
                by_group.setdefault(ep.group_id, []).append(ep)

        min_valid = max(1, math.ceil(group_size / 2)) if group_size > 1 else 1
        final_episodes: list[EpisodeRecord] = []
        dropped_groups = 0
        padded_samples = 0

        for g in range(num_groups):
            members = [ep for ep in by_group.get(g, []) if ep.trajectory is not None]
            if len(members) < min_valid:
                dropped_groups += 1
                continue
            if len(members) < group_size:
                need = group_size - len(members)
                padded = [members[i % len(members)] for i in range(need)]
                members = [*members, *padded]
                padded_samples += need
            final_episodes.extend(members[:group_size])

        # Every final member carries a trajectory, so this keeps ``episodes[i]`` aligned with ``trajectories[i]``.
        trajectories = [ep.trajectory for ep in final_episodes if ep.trajectory is not None]
        return GroupRolloutBatch(
            episodes=final_episodes,
            trajectories=trajectories,
            dropped_stale=dropped_stale,
            dropped_env_error=dropped_env_error,
            dropped_groups=dropped_groups,
            padded_samples=padded_samples,
            env_error_episodes=env_error_episodes,
        )

    def close(self) -> None:
        """Cancel pending work and shut down worker threads."""
        self._coordinator.close(cancel_pending=True, wait=True)


__all__ = (
    "AsyncEnvironmentRolloutCollector",
    "EpisodeRecord",
    "GeneratedTurn",
    "GroupRolloutBatch",
)
