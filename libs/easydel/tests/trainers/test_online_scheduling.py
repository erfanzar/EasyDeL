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
