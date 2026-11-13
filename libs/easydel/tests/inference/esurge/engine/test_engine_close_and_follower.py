"""Public shutdown and follower-rank API of the eSurge engine: ``close``, ``wait_for_shutdown``, ``is_coordination_leader``."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.esurge_engine import eSurge

pytestmark = pytest.mark.cpu_ok


class _Recorder:
    """Records which resources were released, in order."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def hook(self, name: str, *, fail: bool = False):
        def _call(*_args, **_kwargs):
            self.calls.append(name)
            if fail:
                raise RuntimeError(f"{name} failed")

        return _call


class _ShellEngine(eSurge):
    """An engine without a model: only the attributes the lifecycle API reads are set."""

    def __init__(self, *, leader: bool | None, recorder: _Recorder, fail: str | None = None) -> None:
        self._recorder = recorder
        self._scheduler_running = False
        self._worker_replay_thread = None
        self._scheduler_exception = None
        if leader is None:
            self._step_coordinator = None
        else:
            self._step_coordinator = SimpleNamespace(
                is_leader=leader, shutdown=recorder.hook("coordinator", fail=fail == "coordinator")
            )
        self._worker_manager = SimpleNamespace(shutdown=recorder.hook("workers", fail=fail == "workers"))
        self.runner = SimpleNamespace(shutdown=recorder.hook("runner", fail=fail == "runner"))

    def terminate(self) -> None:
        self._recorder.calls.append("terminate")
        self._scheduler_running = False
        self._worker_replay_thread = None


def test_roles_follow_the_step_coordinator():
    recorder = _Recorder()
    assert _ShellEngine(leader=None, recorder=recorder).is_coordination_leader
    assert _ShellEngine(leader=True, recorder=recorder).is_coordination_leader
    assert not _ShellEngine(leader=False, recorder=recorder).is_coordination_leader


def test_close_releases_workers_coordinator_and_runner_once():
    recorder = _Recorder()
    engine = _ShellEngine(leader=True, recorder=recorder)
    engine._scheduler_running = True
    engine.close()
    assert recorder.calls == ["terminate", "workers", "coordinator", "runner"]
    engine.close()
    assert recorder.calls == ["terminate", "workers", "coordinator", "runner"], "close() must be idempotent"


def test_close_stops_a_worker_rank_replay_loop():
    recorder = _Recorder()
    engine = _ShellEngine(leader=False, recorder=recorder)
    engine._worker_replay_thread = threading.Thread(target=lambda: None)
    engine.close()
    assert recorder.calls[0] == "terminate"


def test_close_keeps_releasing_after_a_failing_step():
    recorder = _Recorder()
    _ShellEngine(leader=True, recorder=recorder, fail="workers").close()
    assert recorder.calls == ["workers", "coordinator", "runner"]


def test_wait_for_shutdown_returns_when_replay_ends():
    engine = _ShellEngine(leader=False, recorder=_Recorder())
    release = threading.Event()
    engine._worker_replay_thread = threading.Thread(target=release.wait, daemon=True)
    engine._worker_replay_thread.start()
    assert engine.wait_for_shutdown(timeout=0.05) is False
    release.set()
    assert engine.wait_for_shutdown(timeout=5.0) is True


def test_wait_for_shutdown_surfaces_a_replay_failure():
    engine = _ShellEngine(leader=False, recorder=_Recorder())
    engine._worker_replay_thread = threading.Thread(target=lambda: None)
    engine._worker_replay_thread.start()
    engine._scheduler_exception = ValueError("desync")
    with pytest.raises(RuntimeError, match="replay loop failed") as raised:
        engine.wait_for_shutdown(timeout=5.0)
    assert isinstance(raised.value.__cause__, ValueError)


def test_wait_for_shutdown_rejects_the_leader_and_an_unstarted_worker():
    with pytest.raises(RuntimeError, match="worker ranks"):
        _ShellEngine(leader=True, recorder=_Recorder()).wait_for_shutdown(timeout=0.0)
    with pytest.raises(RuntimeError, match="not running"):
        _ShellEngine(leader=False, recorder=_Recorder()).wait_for_shutdown(timeout=0.0)
