"""Scalar-only AsyncGRPO loop regressions (no model or JAX array execution)."""

from types import SimpleNamespace

import pytest
from easydel.trainers.async_grpo_trainer import async_grpo_trainer as async_grpo

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "sync_steps,max_staleness,queue_size,expected_ahead",
    [
        (1, 1, 0, 1),
        (1, 1, 2, 1),
        (100, 1, 0, 1),
        (100, 4, 0, 4),
        (100, 4, 2, 2),
    ],
)
def test_async_loop_honors_queue_bound_preserves_prompts_and_survives_hooks(
    monkeypatch, sync_steps, max_staleness, queue_size, expected_ahead
):
    monkeypatch.setattr(async_grpo.jax, "device_get", lambda value: value)
    monkeypatch.setattr(async_grpo.jax, "process_count", lambda: 1)
    consumed = []
    fetched = []
    ahead = []
    snapshots = []
    consumed_versions = []
    consumed_infos = []
    preprocess_calls = []

    def get_batch(iterator, dataset):
        batch = next(iterator)
        fetched.append(batch["index"])
        return batch, iterator

    def take(state, *, policy_step, cache_scope_key, **kwargs):
        assert cache_scope_key == state.esurge_cache_scope_key
        owner = SimpleNamespace(
            state=SimpleNamespace(step=policy_step), policy_step=policy_step, nbytes=0, released=False
        )
        owner.release = lambda: setattr(owner, "released", True)
        snapshots.append(owner)
        return owner

    def sync(*args, **kwargs):
        return async_grpo.AsyncGRPOTrainer._sync_rollout_snapshot(trainer, *args, **kwargs)

    def preprocess(*, state, batch, produced_at_step):
        preprocess_calls.append((batch["index"], produced_at_step))
        return async_grpo._AsyncRolloutResult(batch, {"version": produced_at_step}, produced_at_step, 0.0)

    def execute(*, state, batch, informations):
        consumed.append(batch["index"])
        consumed_versions.append(informations["version"])
        consumed_infos.append(dict(informations))
        ahead.append(len(fetched) - len(consumed))
        return (
            SimpleNamespace(step=state.step + 1, esurge_cache_scope_key="test"),
            SimpleNamespace(loss=0.0, accuracy=None),
            None,
        )

    def broken_hook(**kwargs):
        raise ValueError("hook unavailable")

    trainer = SimpleNamespace(
        data_collator=None,
        max_training_steps=5,
        arguments=SimpleNamespace(
            max_inflight_tasks=4,
            queue_maxsize=queue_size,
            max_staleness=max_staleness,
            weight_sync_steps=sync_steps,
            request_timeout=1.0,
            heartbeat_stale_after_s=1.0,
            learning_rate=0.1,
            max_length=4,
            merge_lora_before_save=False,
        ),
        _async_grpo_train_compiled_once=True,
        _get_next_batch=get_batch,
        _sync_rollout_snapshot=sync,
        _take_rollout_snapshot=take,
        _preprocess_async_rollout=preprocess,
        _execute_preprocessed_train_step=execute,
        on_step_start=lambda **kwargs: kwargs["state"],
        on_step_end=lambda **kwargs: (kwargs["state"], kwargs["metrics"]),
        train_tracker=SimpleNamespace(trace_compilation=lambda: _Context()),
        _maybe_start_profiler=lambda *args: None,
        apply_training_hooks=lambda **kwargs: kwargs["metrics"],
        scheduler=None,
        _backward_flops_per_token=0,
        _extra_backward_flops_per_token=0,
        training_batch_size=1,
        log_metrics=lambda **kwargs: None,
        _should_save_tpu_preemption_checkpoint=lambda *args: False,
        log_weight_distribution=lambda **kwargs: None,
        log_watchers=lambda **kwargs: None,
        maybe_generate=broken_hook,
        maybe_benchmark=broken_hook,
        _save_checkpoint_for_step=lambda **kwargs: None,
        _should_run_evaluation=lambda *args: False,
        _profiler_should_block_until_ready=lambda: False,
    )
    steps = SimpleNamespace(start_step=lambda: None, calculate=lambda **kwargs: {})
    tracker = SimpleNamespace(update=lambda **kwargs: (0.0, None))
    state, error, _ = async_grpo.AsyncGRPOTrainer._train_epoch(
        trainer,
        SimpleNamespace(step=0, esurge_cache_scope_key="test"),
        None,
        iter({"index": index} for index in range(5)),
        tracker,
        steps,
        None,
        0,
        epoch_start_step=0,
        epoch_end_step=5,
    )
    assert error is None
    assert state.step == 5
    assert consumed == [0, 1, 2, 3, 4]
    assert fetched == consumed
    assert len(preprocess_calls) == 5, f"Expected zero discarded rollouts, got {preprocess_calls}"
    assert max(ahead) == expected_ahead
    assert all(snapshot.released for snapshot in snapshots)
    # Deadline admission + sync-aware lookahead mean no rollout is ever generated and then discarded.
    assert all(info["async_rollout/stale_dropped"] == 0 for info in consumed_infos)
    assert all(step - version <= trainer.arguments.max_staleness for step, version in enumerate(consumed_versions))
    if sync_steps == 1:
        assert consumed_versions == [0, 0, 1, 2, 3]
        assert [snapshot.policy_step for snapshot in snapshots] == [0, 1, 2, 3]
    elif max_staleness == 1:
        # max_staleness=1 with a stale-prone sync interval: a rollout anchored at v is only admitted when it
        # can be consumed by v + 1, so lookahead is capped at one and the rest is deferred, not wasted.
        assert consumed_versions == [0, 0, 2, 2, 4]
        assert consumed_infos[-1]["async_rollout/admission_deferred"] >= 1


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_stuck_first_rollout_raises_timeout_error(monkeypatch):
    import threading

    monkeypatch.setattr(async_grpo.jax, "device_get", lambda value: value)
    unblock = threading.Event()

    def slow_preprocess(*, state, batch, produced_at_step):
        unblock.wait(timeout=5.0)
        return async_grpo._AsyncRolloutResult(batch, {}, produced_at_step, 0.0)

    def take(state, *, policy_step, **kwargs):
        return SimpleNamespace(
            state=SimpleNamespace(step=policy_step),
            policy_step=policy_step,
            nbytes=0,
            release=lambda: None,
        )

    trainer = SimpleNamespace(
        data_collator=None,
        max_training_steps=2,
        arguments=SimpleNamespace(
            async_mode=True,
            max_inflight_tasks=1,
            queue_maxsize=1,
            max_staleness=1,
            weight_sync_steps=1,
            request_timeout=0.05,
            first_rollout_timeout=0.05,
            heartbeat_stale_after_s=0.05,
            learning_rate=0.1,
            max_length=4,
            merge_lora_before_save=False,
        ),
        _async_grpo_train_compiled_once=False,
        _get_next_batch=lambda it, ds: (next(it), it),
        _sync_rollout_snapshot=lambda *a, **kw: async_grpo.AsyncGRPOTrainer._sync_rollout_snapshot(trainer, *a, **kw),
        _take_rollout_snapshot=take,
        _preprocess_async_rollout=slow_preprocess,
        on_step_start=lambda **kwargs: kwargs["state"],
    )
    try:
        with pytest.raises(TimeoutError, match="First async rollout timed out"):
            async_grpo.AsyncGRPOTrainer._train_epoch(
                trainer,
                SimpleNamespace(step=0, esurge_cache_scope_key="test"),
                None,
                iter([{"index": 0}]),
                SimpleNamespace(update=lambda **kw: (0.0, None)),
                SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: {}),
                None,
                0,
                epoch_start_step=0,
                epoch_end_step=2,
            )
    finally:
        unblock.set()
