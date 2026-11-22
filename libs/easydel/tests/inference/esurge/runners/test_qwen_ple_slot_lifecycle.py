"""PLE continuation state must follow recurrent slot clear/reorder operations."""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.caching import HybridCache
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpOperationsLinearView


def make_manager(candidates=0):
    slots = 3 * (1 + candidates)
    row = jnp.arange(1, slots + 1)
    view = Qwen4ExpOperationsLinearView(
        metadata=None,
        layer_index=0,
        conv_state=jnp.broadcast_to(row[:, None, None], (slots, 2, 3)).astype(jnp.float32),
        recurrent_state=jnp.broadcast_to(row[:, None, None, None], (slots, 2, 2, 2)).astype(jnp.float32),
        positions=row,
        ple_conv_state=jnp.broadcast_to(row[:, None, None], (slots, 2, 3)).astype(jnp.float32),
        ple_token_context=jnp.broadcast_to(row[:, None], (slots, 2)),
        ple_segment_context=jnp.broadcast_to(row[:, None], (slots, 2)),
    )
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.max_num_reqs = 3
    manager.speculative_recurrent_state_tokens = candidates
    manager.kv_pages = HybridCache(views=[view])
    return manager


@pytest.mark.parametrize("operation", ["clear", "permute"])
@pytest.mark.parametrize("candidates", [0, 1])
@pytest.mark.parametrize("nested", [False, True])
def test_ple_sidecars_follow_slot_lifecycle(operation, candidates, nested):
    manager = make_manager(candidates)
    original = manager.kv_pages.views[0]
    if nested:
        from easydel.caching.hybrid.cache import ParallelHybridCacheView

        manager.kv_pages = HybridCache(views=[ParallelHybridCacheView(transformer=None, recurrent=original)])
    if operation == "clear":
        manager.clear_recurrent_slots([1])
        expected = np.arange(1, 3 * (1 + candidates) + 1)
        expected[1::3] = 0
    else:
        manager.permute_recurrent_slots(np.array([2, -1, 0], np.int32))
        expected = np.array([3, 0, 1] + ([6, 0, 4] if candidates else []))
    actual = manager.kv_pages.views[0]
    if nested:
        np.testing.assert_array_equal(actual.positions, expected)
        actual = actual.recurrent
    assert isinstance(actual, Qwen4ExpOperationsLinearView)
    for name in (
        "conv_state",
        "recurrent_state",
        "positions",
        "ple_conv_state",
        "ple_token_context",
        "ple_segment_context",
    ):
        array = getattr(actual, name)
        wanted = np.where(expected == 0, -1, expected) if name == "ple_segment_context" else expected
        np.testing.assert_array_equal(np.asarray(array).reshape(len(expected), -1)[:, 0], wanted)
        assert array.sharding == getattr(original, name).sharding
