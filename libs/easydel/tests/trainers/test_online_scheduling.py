"""Provider orchestration tests with scalar-only fake policy state (no arrays)."""

import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from easydel.trainers._online_rl import trainer as online

pytestmark = pytest.mark.cpu_ok


def _trainer(monkeypatch, provider, *, capacity=2, queue_size=0, timeout=1.0, sync_steps=3):
    owners = []

    class Owner:
        def __init__(self, state, policy_step, **kwargs):
            self.state = SimpleNamespace(step=policy_step)
            self.policy_step = policy_step
            self.released = False
            self.release_event = threading.Event()
            owners.append(self)

        def release(self):
            self.released = True
            self.release_event.set()

    monkeypatch.setattr(online.OwnedPolicySnapshot, "from_training_state", Owner)
    monkeypatch.setattr(online.jax, "device_get", lambda value: value)
    trainer = object.__new__(online.OnlineActorCriticTrainer)
    trainer.arguments = SimpleNamespace(
        max_inflight_rollouts=capacity,
        rollout_queue_size=queue_size,
        rollout_timeout=timeout,
        weight_sync_steps=sync_steps,
        max_policy_staleness=None,
    )
    trainer.rollout_provider = provider
    trainer._test_owners = owners
    trainer._normalize_provider_batch = MethodType(lambda self, batch, **kwargs: batch, trainer)
    return trainer


def test_collect_rollouts_bounds_read_ahead_and_reuses_behavior_policy(monkeypatch):
    requested = []

    def batches():
        for index in range(4):
            requested.append(index)
            yield {"index": index}

    def provider(*, state, batch, **kwargs):
        return {"index": batch["index"], "policy": state.step}

    trainer = _trainer(monkeypatch, provider, capacity=4, queue_size=2)
    state = SimpleNamespace(step=1, esurge_cache_scope_key="test")
    iterator = trainer.collect_rollouts(state, batches())
    first, first_metrics = next(iterator)
    # One consumed result plus at most two queued/running rollouts.
    assert len(requested) == 3
    results = [first] + [batch for batch, _ in iterator]
    assert sorted(batch["index"] for batch in results) == [0, 1, 2, 3]
    assert first_metrics["rollout/policy_version"] == 1
    state.step = 2
    [(result, _)] = trainer.collect_rollouts(state, [{"index": 5}])
    assert result["policy"] == 1
    state.step = 4
    [(result, _)] = trainer.collect_rollouts(state, [{"index": 6}])
    assert result["policy"] == 4


def test_collect_rollouts_returns_completion_order(monkeypatch):
    gate = threading.Event()

    def provider(*, batch, **kwargs):
        if batch["index"] == 0:
            assert gate.wait(2)
        return batch

    trainer = _trainer(monkeypatch, provider)
    iterator = trainer.collect_rollouts(
        SimpleNamespace(step=0, esurge_cache_scope_key="test"), [{"index": 0}, {"index": 1}]
    )
    try:
        assert next(iterator)[0]["index"] == 1
    finally:
        gate.set()
    assert next(iterator)[0]["index"] == 0
    assert list(iterator) == []


def test_rollout_timeout_does_not_release_a_worker_policy(monkeypatch):
    gate = threading.Event()

    def provider(**kwargs):
        gate.wait(2)
        return {"done": True}

    trainer = _trainer(monkeypatch, provider, timeout=0.02)
    iterator = trainer.collect_rollouts(SimpleNamespace(step=0, esurge_cache_scope_key="test"), [{}])
    try:
        with pytest.raises(TimeoutError):
            next(iterator)
        assert trainer._online_rollout_snapshot is None
        owner = trainer._test_owners[0]
        assert not owner.released
        with pytest.raises(RuntimeError, match="still retiring"):
            next(trainer.collect_rollouts(SimpleNamespace(step=1), [{}]))
    finally:
        gate.set()
        iterator.close()
    assert owner.release_event.wait(2)
    assert owner.released


@pytest.mark.parametrize("sync_steps,max_lag", [(1, None), (100, 0)])
def test_live_collection_refreshes_at_sync_boundary_and_uses_current_step(monkeypatch, sync_steps, max_lag):
    versions = []
    learner_steps = []

    def provider(*, state, batch, **kwargs):
        return {**batch, "policy": state.step}

    trainer = _trainer(monkeypatch, provider, sync_steps=sync_steps)
    trainer.arguments.max_policy_staleness = max_lag
    state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    trainer._normalize_provider_batch = MethodType(
        lambda self, batch, **kwargs: learner_steps.append(kwargs["learner_step"]) or batch, trainer
    )
    iterator = trainer.collect_rollouts(state, ({"index": i} for i in range(4)), learner_state=lambda: state)
    for batch, info in iterator:
        versions.append(info["rollout/policy_version"])
        assert batch["policy"] == state.step
        state.step += 1
    assert versions == [0, 1, 2, 3]
    assert learner_steps == [0, 1, 2, 3]
    assert all(owner.released for owner in trainer._test_owners[:-1])


def test_actual_epoch_feeds_multibatch_collection_into_learner(monkeypatch):
    fetched = []
    consumed = []
    overlap = threading.Event()
    learner_started = threading.Event()

    def provider(*, state, batch, **kwargs):
        if batch["index"] == 1:
            assert learner_started.wait(2)
            overlap.set()
        return {**batch, "policy": state.step}

    trainer = _trainer(monkeypatch, provider, capacity=2, sync_steps=100)
    trainer.arguments.dataloader_prefetch = True
    trainer.arguments.critic_updates_per_actor_update = 1
    trainer.arguments.num_ppo_epochs = 1
    trainer.data_collator = lambda batch: {**batch, "collated": True}
    trainer.max_training_steps = 4
    trainer._has_buckets = False
    trainer._apply_pose_to_batch = lambda batch, step: batch
    trainer._apply_process_encoder_to_batch = lambda batch, step: batch
    trainer._refresh_policy_advantages = lambda batch: batch
    trainer.critic_state = SimpleNamespace(step=0)
    trainer._critic_shared_fn_static_args = ()
    trainer._train_shared_fn_extra_args = ()
    trainer._train_shared_fn_static_args = ()
    trainer.sharded_critic_step_function = lambda state, batch: (state, online.LossMetrics())
    monkeypatch.setattr(online.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(online.jnp, "asarray", lambda value: value)

    def update(state, batch):
        learner_started.set()
        assert overlap.wait(2)
        assert batch["collated"]
        consumed.append(batch["index"])
        return SimpleNamespace(step=state.step + 1, esurge_cache_scope_key="test"), online.LossMetrics()

    trainer.sharded_training_step_function = update

    def get_batch(iterator, dataset):
        batch = next(iterator)
        fetched.append(batch["index"])
        return batch, iterator

    trainer._get_next_batch = get_batch

    def epoch(self, state, dataset, iterator, *args, **kwargs):
        for _ in range(4):
            batch = next(iterator)
            assert isinstance(batch, dict)
            state, _, error = self._execute_train_step(state, batch)
            assert error is None
        return state, None, iterator

    monkeypatch.setattr(online.PPOTrainer, "_train_epoch", epoch)
    state, error, _ = trainer._train_epoch(
        SimpleNamespace(step=0, esurge_cache_scope_key="test"),
        None,
        iter({"index": i} for i in range(4)),
        None,
        None,
        None,
        0,
        epoch_start_step=0,
        epoch_end_step=4,
    )
    assert state.step == 4
    assert error is None
    assert sorted(consumed) == [0, 1, 2, 3]
    assert fetched == [0, 1, 2, 3]
    assert trainer.arguments.dataloader_prefetch is True
    assert all(owner.release_event.wait(2) for owner in trainer._test_owners)


def test_default_collector_never_reads_live_critic_during_donation(monkeypatch):
    """Lookahead may score its frozen actor while the learner donates the critic."""
    learner_thread = threading.get_ident()
    donating = threading.Event()
    scored_during_update = threading.Event()
    critic_versions = []
    trainer = _trainer(monkeypatch, None, capacity=2, sync_steps=100)
    trainer.arguments.max_prompt_length = 2
    state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    trainer.critic_state = SimpleNamespace(graphstate=10, graphother=None)
    logps = np.array([[0.0, -0.2, -0.4]], dtype=np.float32)
    # NumPy-only host fakes: this regression executes no JAX array operations.
    monkeypatch.setattr(online.jnp, "zeros_like", np.zeros_like)
    trainer.compute_actor_logps_full = lambda *args: logps.copy()

    def critic(graphstate, graphother, ids, mask):
        assert threading.get_ident() == learner_thread
        assert not donating.is_set()
        critic_versions.append(graphstate)
        return np.full_like(logps, graphstate)

    trainer.compute_critic_values = critic

    def inherited_preprocess(self, state, batch, is_train):
        if batch["index"] == 1:
            assert donating.wait(2)
        policy_logps, values = self._score_rollout_policy_and_values(
            SimpleNamespace(graphstate="frozen actor", graphother=None), None, None
        )
        assert np.array_equal(policy_logps, logps[:, 1:])
        if batch["index"] == 1:
            scored_during_update.set()
        return {**batch, "transport_values": values}, {}

    monkeypatch.setattr(online.PPOTrainer, "_preprocess_batch_input", inherited_preprocess)

    def adapt(batch):
        # Real target construction is a learner-side operation, not a worker
        # read of the live critic. Inspect its observable version here.
        values = trainer.compute_critic_values(
            trainer.critic_state.graphstate, trainer.critic_state.graphother, None, None
        )
        assert np.array_equal(batch["transport_values"], np.zeros((1, 2)))
        return {**batch, "critic_version": int(values[0, 0])}

    trainer._adapt_ppo_rollout_batch = adapt
    iterator = trainer.collect_rollouts(state, [{"index": 0}, {"index": 1}])
    try:
        first, _ = next(iterator)
        assert first["critic_version"] == 10
        # Simulate the donated state being unavailable until the update has
        # completed, while the inherited collector scores the next rollout.
        trainer.critic_state = None
        donating.set()
        assert scored_during_update.wait(2)
        trainer.critic_state = SimpleNamespace(graphstate=11, graphother=None)
        donating.clear()
        second, _ = next(iterator)
        assert second["critic_version"] == 11
        assert list(iterator) == []
    finally:
        donating.set()
        iterator.close()
    donating.clear()
    assert critic_versions == [10, 11]
    # Deferral is worker-local: synchronous scoring still returns real values.
    _, values = trainer._score_rollout_policy_and_values(
        SimpleNamespace(graphstate="actor", graphother=None), None, None
    )
    assert np.array_equal(values, np.full((1, 2), 11))


def test_weight_sync_steps_one_preserves_rollouts_with_zero_discard(monkeypatch):
    calls = []

    def provider(*, state, batch, **kwargs):
        calls.append((batch["index"], state.step))
        return {**batch, "policy": state.step}

    trainer = _trainer(monkeypatch, provider, capacity=8, sync_steps=1)
    state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    versions = []
    for batch, info in trainer.collect_rollouts(
        state, ({"index": i} for i in range(4)), learner_state=lambda: state
    ):
        versions.append(info["rollout/policy_version"])
        assert batch["policy"] == state.step
        state.step += 1

    assert versions == [0, 1, 2, 3]
    # Zero discarded lookahead rollouts: 4 batches -> exactly 4 provider invocations.
    assert len(calls) == 4
    assert calls == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_weight_sync_steps_two_overlaps_without_discarding_at_boundary(monkeypatch):
    calls = []
    batch0_started = threading.Event()
    batch1_started = threading.Event()

    def provider(*, state, batch, **kwargs):
        calls.append((batch["index"], state.step))
        if batch["index"] == 0:
            batch0_started.set()
            assert batch1_started.wait(2)
        elif batch["index"] == 1:
            batch1_started.set()
            assert batch0_started.wait(2)
        return {**batch, "policy": state.step}

    trainer = _trainer(monkeypatch, provider, capacity=8, sync_steps=2)
    state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    versions = []
    for _batch, info in trainer.collect_rollouts(
        state, ({"index": i} for i in range(4)), learner_state=lambda: state
    ):
        versions.append(info["rollout/policy_version"])
        state.step += 1

    assert versions == [0, 0, 2, 2]
    assert len(calls) == 4


def test_refresh_cancels_queued_pending_rollouts_before_draining(monkeypatch):
    started = []
    running_gate = threading.Event()
    batch1_started = threading.Event()

    def provider(*, state, batch, **kwargs):
        started.append((batch["index"], state.step))
        if batch["index"] == 1 and state.step == 0:
            batch1_started.set()
            assert running_gate.wait(2)
        return {**batch, "policy": state.step}

    trainer = _trainer(monkeypatch, provider, capacity=4, sync_steps=4)
    # Single worker so batches 2 and 3 sit queued while batch 1 is running.
    trainer.arguments.max_inflight_rollouts = 4
    original_coord = online.BoundedAsyncRolloutCoordinator

    def one_worker_coord(worker, *, max_workers, max_outstanding, **kw):
        return original_coord(worker, max_workers=1, max_outstanding=max_outstanding, **kw)

    monkeypatch.setattr(online, "BoundedAsyncRolloutCoordinator", one_worker_coord)
    state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    iterator = trainer.collect_rollouts(state, ({"index": i} for i in range(4)), learner_state=lambda: state)
    try:
        first, _ = next(iterator)
        assert first["index"] == 0
        assert batch1_started.wait(2)
        # Jump step past sync_interval unexpectedly so refresh triggers while 1 is running and 2,3 are queued.
        state.step = 10
        running_gate.set()
        rest = [batch for batch, _ in iterator]
    finally:
        running_gate.set()
        iterator.close()

    assert sorted(b["index"] for b in rest) == [1, 2, 3]
    assert all(b["policy"] == 10 for b in rest)
    # Queued batches 2 and 3 must have been cancelled before ever running on policy step 0.
    assert (2, 0) not in started
    assert (3, 0) not in started


def test_asynchronous_rollouts_false_disables_prefetch_and_runs_synchronously(monkeypatch):
    learner_thread = threading.get_ident()
    requested = []
    provider_threads = []

    def batches():
        for i in range(3):
            requested.append(i)
            yield {"index": i}

    def provider(*, state, batch, **kwargs):
        provider_threads.append(threading.get_ident())
        return {**batch, "policy": state.step}

    trainer = _trainer(monkeypatch, provider, capacity=4, sync_steps=100)
    trainer.arguments.asynchronous_rollouts = False
    state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    iterator = trainer.collect_rollouts(state, batches(), learner_state=lambda: state)
    first, first_info = next(iterator)
    assert first["index"] == 0
    assert first_info["rollout/queue_time"] == 0.0
    # Strict demand-driven collection: no lookahead prefetching.
    assert requested == [0]
    second, _ = next(iterator)
    assert second["index"] == 1
    assert requested == [0, 1]
    third, _ = next(iterator)
    assert third["index"] == 2
    assert list(iterator) == []
    assert provider_threads == [learner_thread, learner_thread, learner_thread]


def test_multihost_disables_concurrent_background_rollouts(monkeypatch):
    learner_thread = threading.get_ident()
    requested = []
    provider_threads = []

    def batches():
        for i in range(3):
            requested.append(i)
            yield {"index": i}

    def provider(*, state, batch, **kwargs):
        provider_threads.append(threading.get_ident())
        return {**batch, "policy": state.step}

    monkeypatch.setattr(online.jax, "process_count", lambda: 2)
    trainer = _trainer(monkeypatch, provider, capacity=4, sync_steps=100)
    trainer.arguments.asynchronous_rollouts = True
    state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    iterator = trainer.collect_rollouts(state, batches(), learner_state=lambda: state)
    first, _ = next(iterator)
    assert first["index"] == 0
    assert requested == [0]
    rest = [b for b, _ in iterator]
    assert [b["index"] for b in rest] == [1, 2]
    assert provider_threads == [learner_thread, learner_thread, learner_thread]


def test_multi_epoch_with_num_ppo_epochs_gt_one_drains_cleanly_without_retiring_crash(monkeypatch):
    fetched = []
    provider_calls = []

    def provider(*, state, batch, **kwargs):
        provider_calls.append(batch["index"])
        return {**batch, "policy": state.step}

    trainer = _trainer(monkeypatch, provider, capacity=4, sync_steps=100)
    trainer.arguments.dataloader_prefetch = False
    trainer.arguments.critic_updates_per_actor_update = 1
    trainer.arguments.num_ppo_epochs = 2
    trainer.data_collator = None
    trainer.max_training_steps = 7
    trainer._has_buckets = False
    trainer._apply_pose_to_batch = lambda batch, step: batch
    trainer._apply_process_encoder_to_batch = lambda batch, step: batch
    trainer._refresh_policy_advantages = lambda batch: batch
    trainer.critic_state = SimpleNamespace(step=0)
    trainer._critic_shared_fn_static_args = ()
    trainer._train_shared_fn_extra_args = ()
    trainer._train_shared_fn_static_args = ()
    trainer.sharded_critic_step_function = lambda state, batch: (state, online.LossMetrics())
    monkeypatch.setattr(online.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(online.jnp, "asarray", lambda value: value)

    def update(state, batch):
        return SimpleNamespace(step=state.step + 1, esurge_cache_scope_key="test"), online.LossMetrics()

    trainer.sharded_training_step_function = update

    def get_batch(iterator, dataset):
        batch = next(iterator)
        fetched.append(batch["index"])
        return batch, iterator

    trainer._get_next_batch = get_batch

    def fake_super_epoch(self, state, dataset, iterator, *args, epoch_start_step=0, epoch_end_step=4, **kwargs):
        while state.step < self.max_training_steps and state.step < epoch_end_step:
            batch = next(iterator)
            state, _, error = self._execute_train_step(state, batch)
            assert error is None
        return state, None, iterator

    monkeypatch.setattr(online.PPOTrainer, "_train_epoch", fake_super_epoch)

    data_iter = iter({"index": i} for i in range(10))
    state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    # Epoch 0: steps [0, 4) with num_ppo_epochs=2 -> consumes 2 batches (steps 0->2->4).
    state, err0, data_iter = trainer._train_epoch(
        state, None, data_iter, None, None, None, 0, epoch_start_step=0, epoch_end_step=4
    )
    assert err0 is None
    assert state.step == 4
    assert getattr(trainer, "_online_rollout_idle", True) is True

    # Epoch 1: steps [4, 8) capped by max_training_steps=7 -> 2 batches (4->6->7, no overshoot!).
    state, err1, _ = trainer._train_epoch(
        state, None, data_iter, None, None, None, 1, epoch_start_step=4, epoch_end_step=8
    )
    assert err1 is None
    assert state.step == 7
    assert fetched == [0, 1, 2, 3]
    assert sorted(provider_calls) == [0, 1, 2, 3]

    # Multi-epoch PPO step jump (2 -> 4) must still trigger evaluation_steps=3 and save_steps=3.
    trainer.arguments.evaluation_steps = 3
    trainer.arguments.save_steps = 3
    trainer._last_train_step_prev = 2
    assert trainer._should_run_evaluation(4) is True
    assert trainer._should_run_evaluation(2) is False
    saved_forces = []
    monkeypatch.setattr(
        online.PPOTrainer.__bases__[0],
        "_save_checkpoint_for_step",
        lambda self, state, *, step, force=False, merge_lora_before_save=False: saved_forces.append((step, force)),
    )
    trainer._save_checkpoint_for_step(state, step=4, force=False)
    assert saved_forces == [(4, True)]


def test_mid_epoch_eval_is_decoupled_from_training_idle_and_pauses_active_workers(monkeypatch):
    eval_calls = []
    worker_started = threading.Event()
    worker_release = threading.Event()
    worker_done = threading.Event()

    def provider(*, state, batch, is_train, **kwargs):
        if is_train:
            if batch["index"] == 1:
                worker_started.set()
                assert worker_release.wait(2)
                worker_done.set()
            return {**batch, "train": True}
        eval_calls.append((batch["index"], state.step))
        return {**batch, "eval": True}

    trainer = _trainer(monkeypatch, provider, capacity=2, sync_steps=100)
    # 1) Eval collect_rollouts must work even while _online_rollout_idle=False and
    #    must NOT allocate a second OwnedPolicySnapshot or mutate _online_rollout_idle.
    trainer._online_rollout_idle = False
    sentinel_snapshot = object()
    trainer._online_rollout_snapshot = sentinel_snapshot
    eval_state = SimpleNamespace(step=5)
    owners_before = len(trainer._test_owners)
    eval_results = list(trainer.collect_rollouts(eval_state, [{"index": 10}, {"index": 11}], is_train=False))
    assert len(eval_results) == 2
    assert len(trainer._test_owners) == owners_before, "eval must not allocate an extra policy snapshot"
    assert trainer._online_rollout_idle is False
    assert trainer._online_rollout_snapshot is sentinel_snapshot
    assert eval_calls == [(10, 5), (11, 5)]

    # 2) Mid-epoch eval() must wait for currently running training worker to finish before running eval.
    trainer._online_rollout_idle = True
    trainer._online_rollout_snapshot = None
    train_state = SimpleNamespace(step=0, esurge_cache_scope_key="test")
    rollouts = trainer.collect_rollouts(train_state, [{"index": 0}, {"index": 1}, {"index": 2}], is_train=True)
    eval_saw_worker_done = []

    def fake_super_eval(self, model_state):
        eval_saw_worker_done.append(worker_done.is_set())
        yield {"eval/loss": 0.0}

    monkeypatch.setattr(online.PPOTrainer, "eval", fake_super_eval)
    try:
        first, _ = next(rollouts)
        assert first["index"] == 0
        assert worker_started.wait(2)
        # Release the in-flight worker slightly after eval() starts waiting.
        timer = threading.Timer(0.05, worker_release.set)
        timer.start()
        try:
            outputs = list(trainer.eval(train_state))
        finally:
            timer.cancel()
        assert outputs == [{"eval/loss": 0.0}]
        assert eval_saw_worker_done == [True]
    finally:
        worker_release.set()
        rollouts.close()


def test_randomized_yarn_preserves_prepared_online_batch(monkeypatch):
    trainer = object.__new__(online.OnlineActorCriticTrainer)
    monkeypatch.setattr(
        online.PPOTrainer,
        "_apply_randomized_yarn_to_batch",
        lambda self, batch, step: {**batch, "position_ids": [0, 1]},
    )
    prepared = online._PreparedOnlineBatch({"input_ids": [1, 2]}, {"rollout/policy_version": 3})
    result = trainer._apply_randomized_yarn_to_batch(prepared, step=0)
    assert isinstance(result, online._PreparedOnlineBatch)
    assert result.information == {"rollout/policy_version": 3}
    assert result["position_ids"] == [0, 1]
