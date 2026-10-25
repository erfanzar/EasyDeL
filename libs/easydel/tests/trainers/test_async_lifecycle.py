"""Scalar-only worker cancellation, deadline, and deferred-ownership regressions."""

import threading
import time
from types import SimpleNamespace

import pytest
from easydel.trainers.async_grpo_trainer import async_grpo_trainer as async_grpo

pytestmark = pytest.mark.cpu_ok


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.mark.parametrize("exit_mode", ["heartbeat", "early_stop", "preemption"])
def test_epoch_exit_cancels_queue_without_joining_running_reader(monkeypatch, exit_mode):
    monkeypatch.setattr(async_grpo.jax, "device_get", lambda value: value)
    gate = threading.Event()
    running = threading.Event()
    released = threading.Event()
    started = []
    owner = SimpleNamespace(state=SimpleNamespace(step=0), policy_step=0, nbytes=0, release=released.set)

    def preprocess(*, batch, produced_at_step, **kwargs):
        index = batch["index"]
        started.append(index)
        if exit_mode == "heartbeat" or index == 1:
            running.set()
            assert gate.wait(5)
        return async_grpo._AsyncRolloutResult(batch, {}, produced_at_step, 0.0)

    def execute(**kwargs):
        assert running.wait(2)
        error_type = async_grpo.EasyDeLPreemptionSignal if exit_mode == "preemption" else async_grpo.EasyDeLBreakRequest
        return kwargs["state"], SimpleNamespace(), error_type("stop")

    trainer = SimpleNamespace(
        data_collator=None,
        max_training_steps=5,
        arguments=SimpleNamespace(
            max_inflight_tasks=4,
            queue_maxsize=4,
            max_staleness=100,
            weight_sync_steps=100,
            request_timeout=0.08,
            heartbeat_stale_after_s=0.04,
        ),
        _async_grpo_train_compiled_once=True,
        _get_next_batch=lambda iterator, dataset: (next(iterator), iterator),
        _sync_rollout_snapshot=lambda snapshot, *args, **kwargs: (snapshot or owner, 0.0),
        _preprocess_async_rollout=preprocess,
        _execute_preprocessed_train_step=execute,
        on_step_start=lambda **kwargs: kwargs["state"],
        train_tracker=SimpleNamespace(trace_compilation=lambda: _Context()),
    )
    steps = SimpleNamespace(start_step=lambda: None)

    def epoch():
        return async_grpo.AsyncGRPOTrainer._train_epoch(
            trainer,
            SimpleNamespace(step=0, esurge_cache_scope_key="test"),
            None,
            iter({"index": i} for i in range(5)),
            None,
            steps,
            None,
            0,
            epoch_start_step=0,
            epoch_end_step=5,
        )

    try:
        before = time.monotonic()
        if exit_mode == "heartbeat":
            with pytest.raises(TimeoutError):
                epoch()
        else:
            _, error, _ = epoch()
            assert isinstance(error, async_grpo.EasyDeLBreakRequest | async_grpo.EasyDeLPreemptionSignal)
        assert time.monotonic() - before < 1.5
        assert running.is_set()
        assert not released.is_set()
        with pytest.raises(RuntimeError, match="still retiring"):
            epoch()
    finally:
        gate.set()
    assert released.wait(2)
    assert started == ([0] if exit_mode == "heartbeat" else [0, 1])
    assert all(future.done() for future in trainer._async_grpo_retiring_futures)
