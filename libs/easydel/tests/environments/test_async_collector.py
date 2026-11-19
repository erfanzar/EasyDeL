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

"""Group collection behaviour of :class:`AsyncEnvironmentRolloutCollector`."""

from __future__ import annotations

import threading
import time

import pytest
from easydel.environments import (
    AsyncEnvironmentRolloutCollector,
    GeneratedTurn,
    SandboxEnvironment,
    SyntheticToolEnvironment,
)

pytestmark = pytest.mark.cpu_ok


class _Env(SyntheticToolEnvironment):
    def __init__(self, crash: bool = False, delay: float = 0.0):
        self._crash = crash
        self._delay = delay
        super().__init__(prompt="p", verifier=lambda _state, ans: 1.0 if "yes" in ans else 0.0)

    def step(self, action: str):
        if self._delay:
            time.sleep(self._delay)
        if self._crash:
            return SandboxEnvironment._infra_failure_step("simulated sandbox loss")
        return super().step(action)


def _gen(_state, _ctx, _turn) -> GeneratedTurn:
    return GeneratedTurn(token_ids=[5, 6], text="yes", logprobs=[-0.1, -0.2])


def _collector(
    factory, *, max_outstanding: int, drop_env_errors: bool = True, max_workers: int = 2
) -> AsyncEnvironmentRolloutCollector:
    collector = AsyncEnvironmentRolloutCollector(
        factory,
        generate_turn_fn=_gen,
        encode_fn=lambda text: [1] * max(1, min(4, len(text))),
        max_workers=max_workers,
        max_outstanding=max_outstanding,
        max_staleness=None,
        drop_env_errors=drop_env_errors,
    )
    collector.publish_policy("pi", version=1)
    return collector


def _run_with_deadline(fn, seconds: float = 30.0):
    box: dict[str, object] = {}

    def _target():
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "collect_groups deadlocked"
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]


def test_collect_groups_more_episodes_than_outstanding_capacity():
    collector = _collector(lambda: _Env(), max_outstanding=3)
    try:
        batch = _run_with_deadline(
            lambda: collector.collect_groups(
                num_groups=4, group_size=3, current_policy_version=1, timeout_per_episode=10.0
            )
        )
        assert len(batch.episodes) == 12
        assert sorted(ep.group_id for ep in batch.episodes) == sorted([0, 1, 2] * 3 + [3] * 3)
        assert all(ep.total_reward == 1.0 for ep in batch.episodes)
        assert batch.dropped_env_error == 0 and batch.dropped_groups == 0
    finally:
        collector.close()


def test_collect_groups_8x8_with_max_outstanding_4_does_not_deadlock():
    collector = _collector(lambda: _Env(), max_outstanding=4)
    try:
        batch = _run_with_deadline(
            lambda: collector.collect_groups(
                num_groups=8, group_size=8, current_policy_version=1, timeout_per_episode=10.0
            )
        )
        assert len(batch.episodes) == 64
        assert sorted(ep.group_id for ep in batch.episodes) == sorted(g for g in range(8) for _ in range(8))
        assert batch.dropped_groups == 0 and batch.padded_samples == 0
    finally:
        collector.close()


def test_collect_groups_keeps_env_errors_when_not_dropping():
    collector = _collector(lambda crash=False: _Env(crash=crash), max_outstanding=2, drop_env_errors=False)
    try:
        batch = _run_with_deadline(
            lambda: collector.collect_groups(
                num_groups=2,
                group_size=2,
                current_policy_version=1,
                env_kwargs_per_group=[{"crash": True}, {"crash": False}],
                timeout_per_episode=10.0,
            )
        )
        errored = batch.env_error_episodes
        assert len(errored) == 2 and {ep.group_id for ep in errored} == {0}
        assert all(ep.env_error and ep.total_reward == 0.0 and ep.trajectory is None for ep in errored)
        # Env errors never count toward min_valid: group 0 has no valid member and is dropped, not padded.
        assert batch.dropped_groups == 1 and batch.padded_samples == 0
        assert not any(ep.env_error for ep in batch.episodes)
        assert [ep.group_id for ep in batch.episodes] == [1, 1]
        assert all(ep.total_reward == 1.0 for ep in batch.episodes)
        assert batch.dropped_env_error == 0
        assert len(batch.trajectories) == len(batch.episodes)
        assert all(ep.trajectory is traj for ep, traj in zip(batch.episodes, batch.trajectories, strict=True))
    finally:
        collector.close()


def test_collect_groups_env_errors_kept_are_not_padding_sources():
    calls = {"n": 0}
    lock = threading.Lock()

    def _factory():
        with lock:
            calls["n"] += 1
            idx = calls["n"]
        return _Env(crash=idx % 4 == 0)

    collector = _collector(_factory, max_outstanding=2, drop_env_errors=False)
    try:
        batch = _run_with_deadline(
            lambda: collector.collect_groups(
                num_groups=1, group_size=4, current_policy_version=1, timeout_per_episode=10.0
            )
        )
        assert len(batch.env_error_episodes) == 1
        assert len(batch.episodes) == 4 and batch.padded_samples == 1
        assert not any(ep.env_error for ep in batch.episodes)
        assert all(ep.trajectory is traj for ep, traj in zip(batch.episodes, batch.trajectories, strict=True))
    finally:
        collector.close()


def test_collect_groups_with_foreign_submissions_filling_capacity():
    collector = _collector(lambda: _Env(), max_outstanding=3)
    try:
        for k in range(3):
            collector.submit_episode(prompt_index=100 + k, group_id=100 + k)
        batch = _run_with_deadline(
            lambda: collector.collect_groups(
                num_groups=2, group_size=2, current_policy_version=1, timeout_per_episode=10.0
            )
        )
        assert len(batch.episodes) == 4
        assert sorted(ep.group_id for ep in batch.episodes) == [0, 0, 1, 1]
        assert batch.dropped_groups == 0 and batch.padded_samples == 0
        # Foreign completions are never counted as ours; they are handed back by collect_episodes.
        foreign, stats = collector.collect_episodes(3, current_policy_version=1, timeout_per_episode=10.0)
        assert sorted(ep.group_id for ep in foreign) == [100, 101, 102]
        assert stats["dropped_env_error"] == 0
        assert collector.coordinator.outstanding_count == 0
    finally:
        collector.close()


def test_collect_groups_drops_env_errors_by_default():
    collector = _collector(lambda crash=False: _Env(crash=crash), max_outstanding=2)
    try:
        batch = _run_with_deadline(
            lambda: collector.collect_groups(
                num_groups=2,
                group_size=2,
                current_policy_version=1,
                env_kwargs_per_group=[{"crash": True}, {"crash": False}],
                timeout_per_episode=10.0,
            )
        )
        assert batch.dropped_env_error == 2
        assert batch.dropped_groups == 1
        assert {ep.group_id for ep in batch.episodes} == {1}
    finally:
        collector.close()


def test_collect_episodes_env_errors_kept_never_satisfy_count():
    collector = _collector(
        lambda crash=False, delay=0.0: _Env(crash=crash, delay=delay), max_outstanding=4, drop_env_errors=False
    )
    try:
        collector.submit_episode(prompt_index=0, group_id=0, env_kwargs={"crash": True})
        collector.submit_episode(prompt_index=1, group_id=1, env_kwargs={"crash": True})
        # The valid episode finishes last, so the two kept env errors arrive first and must not satisfy count=1.
        collector.submit_episode(prompt_index=2, group_id=2, env_kwargs={"delay": 0.3})
        episodes, stats = collector.collect_episodes(1, current_policy_version=1, timeout_per_episode=10.0)
        assert [ep.group_id for ep in episodes if ep.trajectory is not None] == [2]
        assert stats["kept_env_error"] == 2 == sum(ep.env_error for ep in episodes)
        assert collector.coordinator.outstanding_count == 0
    finally:
        collector.close()


def test_collect_collated_batch_counts_only_real_trajectories():
    from easydel.trainers._online_rl.trajectory import OnlineRLTrajectoryCollator

    collector = _collector(
        lambda crash=False, delay=0.0: _Env(crash=crash, delay=delay), max_outstanding=4, drop_env_errors=False
    )
    collator = OnlineRLTrajectoryCollator(max_length=64, pad_token_id=0)
    try:
        for k, crash in enumerate([True, False, True, False]):
            kwargs = {"crash": True} if crash else {"delay": 0.3}
            collector.submit_episode(prompt_index=k, group_id=k, env_kwargs=kwargs)
        batch, metrics = collector.collect_collated_batch(
            2, current_policy_version=1, collator=collator, timeout_per_episode=10.0
        )
        assert batch["input_ids"].shape[0] == 2
        assert metrics["async_env/collected"] == 2
        assert metrics["async_env/kept_env_error"] == 2
        assert metrics["async_env/mean_reward"] == 1.0

        collector.submit_episode(prompt_index=9, group_id=9, env_kwargs={"crash": True})
        with pytest.raises(RuntimeError, match="No valid episodes"):
            collector.collect_collated_batch(1, current_policy_version=1, collator=collator, timeout_per_episode=10.0)
    finally:
        collector.close()


def test_concurrent_collect_groups_and_collect_episodes_do_not_steal():
    collector = _collector(lambda delay=0.0: _Env(delay=delay), max_outstanding=8, max_workers=4)
    try:
        collector.submit_episode(prompt_index=100, group_id=100, env_kwargs={"delay": 0.05})
        collector.submit_episode(prompt_index=101, group_id=101, env_kwargs={"delay": 0.05})
        box: dict[str, object] = {}

        def _groups():
            box["groups"] = collector.collect_groups(
                num_groups=1,
                group_size=4,
                current_policy_version=1,
                env_kwargs_per_group=[{"delay": 0.05}],
                timeout_per_episode=10.0,
            )

        t = threading.Thread(target=_groups, daemon=True)
        t.start()
        time.sleep(0.01)
        foreign, _ = collector.collect_episodes(2, current_policy_version=1, timeout_per_episode=10.0)
        t.join(30.0)
        assert not t.is_alive()
        batch = box["groups"]
        assert sorted(ep.group_id for ep in batch.episodes) == [0, 0, 0, 0]
        assert batch.padded_samples == 0 and batch.dropped_groups == 0
        assert sorted(ep.group_id for ep in foreign) == [100, 101]
    finally:
        collector.close()


def test_collect_groups_timeout_orphans_its_submissions():
    collector = _collector(lambda delay=0.0: _Env(delay=delay), max_outstanding=6, max_workers=2)
    try:
        with pytest.raises(TimeoutError):
            collector.collect_groups(
                num_groups=1,
                group_size=4,
                current_policy_version=1,
                env_kwargs_per_group=[{"delay": 0.3}],
                timeout_per_episode=0.01,
            )
        collector.submit_episode(prompt_index=7, group_id=7, env_kwargs={"delay": 0.3})
        episodes, stats = collector.collect_episodes(1, current_policy_version=1, timeout_per_episode=10.0)
        assert [ep.group_id for ep in episodes] == [7]
        assert stats["dropped_orphaned"] == 2  # the two already-running members; the queued two were cancelled
        later = collector.collect_groups(num_groups=1, group_size=2, current_policy_version=1, timeout_per_episode=10.0)
        assert [ep.group_id for ep in later.episodes] == [0, 0] and later.padded_samples == 0
        assert collector.coordinator.outstanding_count == 0
    finally:
        collector.close()
