"""Ring dispatch records histograms in global order even for traced rotations."""

import jax
import numpy as np
import pytest
from easydel.layers.moe import record_expert_load
from easydel.layers.moe._communication_utils import permute
from jax import numpy as jnp

NUM_EXPERTS = 8


@pytest.mark.parametrize("roll", [0, 3, 4])
def test_traced_roll_is_undone_in_recorded_histogram(roll):
    """Exercise the dispatch API with a dynamic shift and inspect its exported records."""
    selected = np.array([1, 5, 5, 6, 7], dtype=np.int32)
    logits = jnp.asarray(np.eye(NUM_EXPERTS, dtype=np.float32)[selected])
    inputs = jnp.ones((1, len(selected), 2), dtype=jnp.float32)

    @jax.jit
    def run(shift):
        return permute(
            inputs,
            logits,
            num_experts_per_tok=1,
            num_experts=NUM_EXPERTS,
            dtype=jnp.float32,
            roll_to_expert_id=shift,
            layer_idx=2,
        )[3]

    with record_expert_load(ep_size=2) as recorder:
        local_counts = run(jnp.asarray(roll, dtype=jnp.int32))
        jax.effects_barrier()

    expected = np.bincount(selected, minlength=NUM_EXPERTS)
    np.testing.assert_array_equal(np.asarray(local_counts), np.roll(expected, -roll))
    np.testing.assert_array_equal(recorder.records[0].counts, expected)
    np.testing.assert_array_equal(recorder.to_arrays()["logical_count"][0, 0], expected)


def test_python_int_roll_keeps_global_expert_order():
    selected = np.array([1, 5, 5, 6, 7], dtype=np.int32)
    with record_expert_load(ep_size=2) as recorder:
        permute(
            jnp.ones((1, len(selected), 2), dtype=jnp.float32),
            jnp.asarray(np.eye(NUM_EXPERTS, dtype=np.float32)[selected]),
            num_experts_per_tok=1,
            num_experts=NUM_EXPERTS,
            dtype=jnp.float32,
            roll_to_expert_id=3,
        )
        jax.effects_barrier()
    np.testing.assert_array_equal(recorder.records[0].counts, np.bincount(selected, minlength=NUM_EXPERTS))
