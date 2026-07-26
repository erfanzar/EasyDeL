# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ownership and lifetime tests for rollout policy snapshots.

Covers the invariants the asynchronous RL trainers depend on: a snapshot owns
independent buffers (the training step keeps donating the live state), it carries
parameters only (no optimizer slots), releasing frees exactly the snapshot's
buffers, and a refresh frees the outgoing snapshot *before* allocating its
replacement so only one extra copy of the policy is ever resident.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import jax
import optax
import pytest
import spectrax as spx
from easydel.trainers._shared import (
    OwnedPolicySnapshot,
    copy_frozen_policy,
    drop_optimizer_state,
    polyak_update_reference_graphstate,
)
from easydel.trainers.async_grpo_trainer.async_grpo_trainer import AsyncGRPOTrainer
from jax import numpy as jnp


def _tiny_training_state():
    """Build a small llama state with initialized Adam slots."""
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=16,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = LlamaForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    state = model.to_state(trainable_selector="parameters")
    return state.init_tx(optax.adam(1e-3))


def _array_leaves(tree):
    """Return only the JAX array leaves of a pytree."""
    return [leaf for leaf in jax.tree_util.tree_leaves(tree) if isinstance(leaf, jax.Array)]


def test_snapshot_owns_independent_params_and_drops_optimizer_state():
    state = _tiny_training_state()
    assert state.opt_state is not None, "fixture must carry optimizer slots to make the drop observable"

    snapshot = OwnedPolicySnapshot.from_training_state(
        state,
        policy_step=3,
        cache_scope_key="rollout-scope",
    )

    assert snapshot.policy_step == 3
    assert snapshot.state.tx is None
    assert snapshot.state.opt_state is None
    assert snapshot.state.esurge_cache_scope_key == "rollout-scope"
    assert state.esurge_cache_scope_key != "rollout-scope"

    live_leaves = _array_leaves(state.graphstate)
    snapshot_leaves = _array_leaves(snapshot.state.graphstate)
    assert len(snapshot_leaves) == len(live_leaves)
    for live, copied in zip(live_leaves, snapshot_leaves, strict=True):
        assert copied is not live, "snapshot must not alias the donated training buffers"
        assert copied.sharding == live.sharding
        assert copied.dtype == live.dtype
        assert jnp.array_equal(copied, live)

    # Params-only: the snapshot cannot be as large as params + two Adam moments.
    params_bytes = sum(leaf.nbytes for leaf in live_leaves)
    assert snapshot.nbytes < state.size
    assert snapshot.nbytes >= params_bytes


def test_release_frees_only_the_snapshot_buffers():
    state = _tiny_training_state()
    snapshot = OwnedPolicySnapshot.from_training_state(state, policy_step=0, cache_scope_key="scope")
    snapshot_leaves = _array_leaves(snapshot.state)

    released = snapshot.release()

    assert released == snapshot.nbytes
    assert snapshot.released
    assert all(leaf.is_deleted() for leaf in snapshot_leaves)
    # The live state is untouched and still usable.
    assert not any(leaf.is_deleted() for leaf in _array_leaves(state))
    assert jnp.isfinite(sum(jnp.sum(leaf) for leaf in _array_leaves(state.graphstate)))

    assert snapshot.release() == 0, "release must be idempotent"
    with pytest.raises(RuntimeError, match="was released"):
        _ = snapshot.state


def test_sync_releases_outgoing_snapshot_before_allocating_replacement():
    state = _tiny_training_state()
    trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
    trainer.arguments = SimpleNamespace(weight_sync_steps=4)

    first, sync_seconds = trainer._sync_rollout_snapshot(
        None,
        state,
        0,
        force=False,
        can_release=True,
        cache_scope_key="scope",
    )
    assert sync_seconds >= 0.0

    # Inside the sync interval the same snapshot is reused, at zero cost.
    reused, reuse_seconds = trainer._sync_rollout_snapshot(
        first,
        state,
        2,
        force=False,
        can_release=True,
        cache_scope_key="scope",
    )
    assert reused is first
    assert reuse_seconds == 0.0
    assert not first.released

    # At the interval boundary the outgoing snapshot must already be released by
    # the time the replacement is allocated, so peak residency stays at one copy.
    observed_released_at_alloc: list[bool] = []
    original_take = trainer._take_rollout_snapshot

    def probe(*args, **kwargs):
        observed_released_at_alloc.append(first.released)
        return original_take(*args, **kwargs)

    trainer._take_rollout_snapshot = probe
    second, refresh_seconds = trainer._sync_rollout_snapshot(
        first,
        state,
        4,
        force=False,
        can_release=True,
        cache_scope_key="scope",
    )

    assert observed_released_at_alloc == [True]
    assert refresh_seconds > 0.0
    assert second is not first
    assert second.policy_step == 4
    assert not second.released
    assert not any(leaf.is_deleted() for leaf in _array_leaves(second.state.graphstate))

    # force refreshes inside the interval; can_release=False keeps the outgoing
    # buffers alive for a rollout that is still reading them.
    third, _ = trainer._sync_rollout_snapshot(
        second,
        state,
        5,
        force=True,
        can_release=False,
        cache_scope_key="scope",
    )
    assert third is not second
    assert not second.released
    assert not any(leaf.is_deleted() for leaf in _array_leaves(second.state.graphstate))

    second.release()
    third.release()


def _device_bytes_in_use() -> int:
    """Total accelerator bytes in use across devices."""
    gc.collect()
    return sum(int(device.memory_stats()["bytes_in_use"]) for device in jax.devices())


def _measurable_training_state():
    """Build a state whose parameter set is far larger than allocator granularity.

    The tiny fixture used elsewhere is only kilobytes, which device memory
    accounting cannot distinguish from noise; this one is ~100 MB of parameters.
    """
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=8192,
        max_position_embeddings=512,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        attn_mechanism="vanilla",
    )
    model = LlamaForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    state = model.to_state(trainable_selector="parameters")
    return jax.block_until_ready(state.init_tx(optax.adam(1e-3)))


@pytest.mark.skipif(
    jax.default_backend() == "cpu",
    reason="HBM accounting requires an accelerator; CPU has no bytes_in_use to observe",
)
def test_sync_peak_stays_at_one_snapshot_in_device_memory():
    """Assert on real device memory that a refresh never holds two snapshots.

    The ordering test above proves the release happens first; this proves the
    release actually returns HBM, which is the property that keeps a large
    policy's peak at one extra parameter set instead of two.
    """
    state = _measurable_training_state()
    params_bytes = sum(leaf.nbytes for leaf in _array_leaves(state.graphstate))
    trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
    trainer.arguments = SimpleNamespace(weight_sync_steps=1)

    baseline = _device_bytes_in_use()
    first, _ = trainer._sync_rollout_snapshot(
        None, state, 0, force=False, can_release=True, cache_scope_key="scope"
    )
    one_snapshot_resident = _device_bytes_in_use()
    assert one_snapshot_resident - baseline >= params_bytes * 0.5, "snapshot should be visible in device memory"

    measured_at_allocation: list[int] = []
    original_take = trainer._take_rollout_snapshot

    def probe(*args, **kwargs):
        measured_at_allocation.append(_device_bytes_in_use())
        return original_take(*args, **kwargs)

    trainer._take_rollout_snapshot = probe
    second, _ = trainer._sync_rollout_snapshot(
        first, state, 1, force=False, can_release=True, cache_scope_key="scope"
    )
    try:
        # At allocation time the outgoing snapshot's buffers are already returned,
        # so usage is back near the no-snapshot baseline rather than carrying two.
        assert measured_at_allocation[0] < one_snapshot_resident - params_bytes * 0.5
        assert _device_bytes_in_use() - baseline < params_bytes * 1.5
    finally:
        second.release()


def _build_ppo_trainer_without_base_init(monkeypatch, *, kl_coef: float):
    """Run ``PPOTrainer.__init__`` with the base trainer initializer stubbed out."""
    import easydel as ed
    from easydel.trainers.trainer.trainer import Trainer

    monkeypatch.setattr(Trainer, "__init__", lambda self, **kwargs: None)
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=16,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    from easydel.trainers.proximal_policy_optimization_trainer.modeling_value_head import CausalLMWithValueHead

    actor = LlamaForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    policy_state = CausalLMWithValueHead(actor, rngs=spx.Rngs(1)).to_state(trainable_selector="parameters")
    trainer = ed.PPOTrainer(
        arguments=ed.PPOConfig(max_training_steps=1, kl_coef=kl_coef),
        model=policy_state,
        reward_funcs=lambda **_: [0.0],
        processing_class=SimpleNamespace(pad_token_id=0, eos_token_id=1),
    )
    return trainer, policy_state


@pytest.mark.parametrize(("kl_coef", "expects_reference"), [(0.05, True), (0.0, False)])
def test_ppo_allocates_reference_model_only_when_kl_is_active(monkeypatch, kl_coef, expects_reference):
    trainer, policy_state = _build_ppo_trainer_without_base_init(monkeypatch, kl_coef=kl_coef)

    if expects_reference:
        assert trainer.ref_state is not None
        policy_leaves = _array_leaves(policy_state.graphstate)
        reference_leaves = _array_leaves(trainer.ref_state.graphstate)
        for policy_leaf, reference_leaf in zip(policy_leaves, reference_leaves, strict=True):
            assert reference_leaf is not policy_leaf
            assert jnp.array_equal(reference_leaf, policy_leaf)
    else:
        assert trainer.ref_state is None, "kl_coef == 0 must not allocate a reference parameter set"


def test_ppo_reference_copy_never_carries_optimizer_slots(monkeypatch):
    import easydel as ed
    from easydel.trainers.trainer.trainer import Trainer

    monkeypatch.setattr(Trainer, "__init__", lambda self, **kwargs: None)
    state = _tiny_training_state()
    assert state.opt_state is not None

    trainer = ed.PPOTrainer(
        arguments=ed.PPOConfig(max_training_steps=1, kl_coef=0.05),
        model=state,
        reward_funcs=lambda **_: [0.0],
        processing_class=SimpleNamespace(pad_token_id=0, eos_token_id=1),
    )

    # A policy state that already owns Adam slots must not have them duplicated
    # into the frozen reference, which never steps.
    assert trainer.ref_state.opt_state is None
    assert trainer.ref_state.tx is None
    assert trainer.ref_state.size < state.size


def test_drop_optimizer_state_shares_params_and_clears_slots():
    state = _tiny_training_state()
    assert state.opt_state is not None

    frozen = drop_optimizer_state(state)

    assert frozen.tx is None
    assert frozen.opt_state is None
    for live, shared in zip(_array_leaves(state.graphstate), _array_leaves(frozen.graphstate), strict=True):
        assert shared is live, "dropping the optimizer must not copy parameters"
    # Idempotent, and non-state inputs pass through untouched.
    assert drop_optimizer_state(frozen) is frozen
    sentinel = object()
    assert drop_optimizer_state(sentinel) is sentinel


def test_copy_frozen_policy_copies_params_only():
    state = _tiny_training_state()

    frozen = copy_frozen_policy(state, cache_scope_key="frozen-scope")

    assert frozen.tx is None
    assert frozen.opt_state is None
    assert frozen.esurge_cache_scope_key == "frozen-scope"
    for live, copied in zip(_array_leaves(state.graphstate), _array_leaves(frozen.graphstate), strict=True):
        assert copied is not live
        assert copied.sharding == live.sharding
        assert jnp.array_equal(copied, live)


def test_polyak_reference_update_allocates_fresh_arrays():
    state = _tiny_training_state()
    reference = jax.tree_util.tree_map(lambda leaf: leaf * 0.0, state.graphstate)

    mixed = polyak_update_reference_graphstate(reference, state.graphstate, alpha=0.25)
    for policy_leaf, reference_leaf, mixed_leaf in zip(
        _array_leaves(state.graphstate),
        _array_leaves(reference),
        _array_leaves(mixed),
        strict=True,
    ):
        assert mixed_leaf is not policy_leaf, "reference must not alias the donated policy buffers"
        expected = 0.25 * policy_leaf + 0.75 * reference_leaf
        assert jnp.allclose(mixed_leaf, expected, atol=1e-6)

    replaced = polyak_update_reference_graphstate(reference, state.graphstate, alpha=1.0)
    for policy_leaf, replaced_leaf in zip(_array_leaves(state.graphstate), _array_leaves(replaced), strict=True):
        assert replaced_leaf is not policy_leaf
        assert jnp.array_equal(replaced_leaf, policy_leaf)
