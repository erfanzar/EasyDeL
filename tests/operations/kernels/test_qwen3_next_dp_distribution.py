from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from easydel.caching import HybridCache, RecurrentCacheView
from easydel.inference.esurge.runners.execution_manager import ExecutionManager


def test_qwen3_next_local_distribution_keeps_single_token_fast_lane():
    scheduled_tokens = jnp.array([1, 1, 2, 0], dtype=jnp.int32)
    slot_ids = jnp.arange(scheduled_tokens.shape[0], dtype=jnp.int32)
    active_slots = (slot_ids < 3) & (scheduled_tokens > 0)
    single_slot_mask = active_slots & (scheduled_tokens == 1)
    distribution = jnp.stack([jnp.sum(single_slot_mask), jnp.sum(active_slots), jnp.sum(active_slots)])

    np.testing.assert_array_equal(
        np.asarray(distribution),
        np.array([2, 3, 3], dtype=np.int32),
    )


def test_clear_recurrent_slots_zeros_only_freed_physical_slot():
    conv_state = jnp.arange(4 * 2 * 3, dtype=jnp.float32).reshape(4, 2, 3) + 1
    recurrent_state = jnp.arange(4 * 1 * 2 * 3, dtype=jnp.float32).reshape(4, 1, 2, 3) + 1
    view = RecurrentCacheView(
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        positions=None,
    )

    fake = SimpleNamespace(
        kv_pages=HybridCache(views=[view]),
        max_num_reqs=4,
        speculative_recurrent_state_tokens=0,
    )
    ExecutionManager.clear_recurrent_slots(fake, [2])
    updated = fake.kv_pages.views[0]

    np.testing.assert_array_equal(np.asarray(updated.conv_state[2]), np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(updated.recurrent_state[2]), np.zeros((1, 2, 3), dtype=np.float32))
    untouched = jnp.array([0, 1, 3], dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray(updated.conv_state[untouched]), np.asarray(conv_state[untouched]))
    np.testing.assert_array_equal(
        np.asarray(updated.recurrent_state[untouched]),
        np.asarray(recurrent_state[untouched]),
    )
