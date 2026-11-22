"""PPO replays every rollout batch for ``num_ppo_epochs`` updates in both sync and async modes."""

from types import SimpleNamespace

import pytest
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer import PPOTrainer

pytestmark = pytest.mark.cpu_ok

_MODULE = "easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer"


def _stub_trainer(num_ppo_epochs, async_mode, compiled_calls, preprocess_calls):
    trainer = object.__new__(PPOTrainer)
    trainer.arguments = SimpleNamespace(num_ppo_epochs=num_ppo_epochs, num_mini_batches=1, async_mode=async_mode)
    trainer.pruning_module = None
    trainer._train_shared_fn_extra_args = ()
    trainer._train_shared_fn_static_args = ()
    trainer._runtime_trace = lambda *args, **kwargs: None
    trainer._runtime_batch_summary = lambda batch: batch
    trainer._is_memory_oom_exception = lambda exc: False

    def preprocess(*, state, batch, is_train):
        preprocess_calls.append(state.step)
        return {"input_ids": "rolled-out"}, {"rollout_score": 1.0}

    def compiled_step(state, batch):
        compiled_calls.append((state.step, batch["input_ids"]))
        return SimpleNamespace(step=state.step + 1), LossMetrics(
            loss=float(state.step), accuracy=1.0, other_metrics={"last_epoch_step": state.step}
        )

    trainer._preprocess_batch_input = preprocess
    trainer.sharded_training_step_function = compiled_step
    return trainer


@pytest.fixture(autouse=True)
def _host_only_jax(monkeypatch):
    monkeypatch.setattr(f"{_MODULE}.jax.block_until_ready", lambda value: value)
    monkeypatch.setattr(f"{_MODULE}.jax.device_get", lambda value: value)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("num_ppo_epochs", [2, 3])
def test_every_mode_runs_num_ppo_epochs_compiled_updates(async_mode, num_ppo_epochs):
    compiled_calls, preprocess_calls = [], []
    trainer = _stub_trainer(num_ppo_epochs, async_mode, compiled_calls, preprocess_calls)
    state = SimpleNamespace(step=0)

    if async_mode:
        final_state, metrics, error = trainer._execute_preprocessed_train_step(
            state=state,
            batch={"input_ids": "rolled-out"},
            informations={"rollout_score": 1.0},
        )
        assert preprocess_calls == []
    else:
        final_state, metrics, error = trainer._execute_train_step(state=state, batch={"prompt": "hi"})
        assert preprocess_calls == [0]

    minibatches = trainer.arguments.num_mini_batches
    assert error is None
    assert len(compiled_calls) == num_ppo_epochs * minibatches
    assert compiled_calls == [(step, "rolled-out") for step in range(num_ppo_epochs)]
    assert final_state.step == num_ppo_epochs
    assert metrics.other_metrics == {"rollout_score": 1.0, "last_epoch_step": num_ppo_epochs - 1}


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _run_async_ppo_epoch(*, num_ppo_epochs, max_staleness, weight_sync_steps, rollouts):
    """Drive the production async loop; each consumed rollout advances ``state.step`` by ``num_ppo_epochs``."""
    from easydel.trainers.async_rollout_mixin import AsyncRolloutMixin

    max_steps = rollouts * num_ppo_epochs
    fetched, consumed, snapshots = [], [], []

    def get_batch(iterator, dataset):
        batch = next(iterator)
        fetched.append(batch["index"])
        return batch, iterator

    def take(state, *, policy_step, **kwargs):
        owner = SimpleNamespace(state=SimpleNamespace(step=policy_step), policy_step=policy_step, nbytes=0)
        owner.release = lambda: None
        snapshots.append(policy_step)
        return owner

    def preprocess(*, state, batch, produced_at_step):
        from easydel.trainers.async_rollout_mixin import AsyncRolloutResult

        # Record the snapshot version the production loop anchored this rollout at.
        return AsyncRolloutResult(
            batch, {"async_rollout/rollout_produced_at_step": produced_at_step}, produced_at_step, 0.0
        )

    def execute(*, state, batch, informations):
        consumed.append((state.step, batch["index"], dict(informations)))
        return SimpleNamespace(step=state.step + num_ppo_epochs, esurge_cache_scope_key="test"), LossMetrics(), None

    trainer = SimpleNamespace(
        data_collator=None,
        max_training_steps=max_steps,
        arguments=SimpleNamespace(
            num_ppo_epochs=num_ppo_epochs,
            max_inflight_tasks=4,
            queue_maxsize=4,
            max_staleness=max_staleness,
            weight_sync_steps=weight_sync_steps,
            request_timeout=2.0,
            heartbeat_stale_after_s=2.0,
            learning_rate=0.1,
            max_length=4,
            merge_lora_before_save=False,
        ),
        _async_grpo_train_compiled_once=True,
        _get_next_batch=get_batch,
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
        maybe_generate=lambda **kwargs: None,
        maybe_benchmark=lambda **kwargs: None,
        _save_checkpoint_for_step=lambda **kwargs: None,
        _should_run_evaluation=lambda *args: False,
        _profiler_should_block_until_ready=lambda: False,
    )
    trainer._optimizer_steps_per_rollout = lambda: PPOTrainer._optimizer_steps_per_rollout(trainer)
    trainer._sync_rollout_snapshot = lambda *a, **kw: AsyncRolloutMixin._sync_rollout_snapshot(trainer, *a, **kw)
    state, error, _ = AsyncRolloutMixin._train_epoch(
        trainer,
        SimpleNamespace(step=0, esurge_cache_scope_key="test"),
        None,
        iter({"index": index} for index in range(rollouts)),
        SimpleNamespace(update=lambda **kwargs: (0.0, None)),
        SimpleNamespace(start_step=lambda: None, calculate=lambda **kwargs: {}),
        None,
        0,
        epoch_start_step=0,
        epoch_end_step=max_steps,
    )
    assert error is None
    assert state.step == max_steps
    return fetched, consumed, snapshots


def test_ppo_reports_num_ppo_epochs_optimizer_steps_per_rollout():
    trainer = object.__new__(PPOTrainer)
    trainer.arguments = SimpleNamespace(num_ppo_epochs=4)
    assert trainer._optimizer_steps_per_rollout() == 4
    trainer.arguments = SimpleNamespace(num_ppo_epochs=1)
    assert trainer._optimizer_steps_per_rollout() == 1


def test_async_ppo_epochs_count_staleness_admission_and_sync_in_rollouts():
    fetched, consumed, snapshots = _run_async_ppo_epoch(
        num_ppo_epochs=4, max_staleness=1, weight_sync_steps=2, rollouts=5
    )
    consumed_steps = [step for step, _, _ in consumed]
    consumed_batches = [index for _, index, _ in consumed]
    anchors = [info["async_rollout/rollout_produced_at_step"] for _, _, info in consumed]
    staleness = [info["async_rollout/rollout_staleness"] for _, _, info in consumed]

    assert consumed_steps == [0, 4, 8, 12, 16]
    assert consumed_batches == fetched == [0, 1, 2, 3, 4]
    # A rollout produced one rollout (4 optimizer steps) ago is admitted and kept; it is never dropped.
    assert anchors == [0, 0, 8, 8, 16]
    assert staleness == [0, 1, 0, 1, 0]
    assert all(info["async_rollout/stale_dropped"] == 0 for _, _, info in consumed)
    # A rollout that would be two rollouts old at consumption is deferred instead of generated.
    assert consumed[-1][2]["async_rollout/admission_deferred"] >= 1
    # weight_sync_steps=2 refreshes every two rollouts (8 optimizer steps), not every two steps.
    assert snapshots == [0, 8, 16]


def test_async_ppo_epochs_refresh_weights_after_weight_sync_steps_rollouts():
    # Staleness never binds here, so every refresh comes from weight_sync_steps counted in rollouts.
    _, consumed, snapshots = _run_async_ppo_epoch(num_ppo_epochs=4, max_staleness=100, weight_sync_steps=2, rollouts=5)
    assert [info["async_rollout/rollout_produced_at_step"] for _, _, info in consumed] == [0, 0, 8, 8, 16]
    assert snapshots == [0, 8, 16]


def test_async_ppo_single_epoch_keeps_step_unit_bookkeeping():
    _, consumed, snapshots = _run_async_ppo_epoch(num_ppo_epochs=1, max_staleness=1, weight_sync_steps=2, rollouts=5)
    assert [info["async_rollout/rollout_produced_at_step"] for _, _, info in consumed] == [0, 0, 2, 2, 4]
    assert [info["async_rollout/rollout_staleness"] for _, _, info in consumed] == [0, 1, 0, 1, 0]
    assert snapshots == [0, 2, 4]
