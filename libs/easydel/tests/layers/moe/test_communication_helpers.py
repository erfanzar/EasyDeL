"""Documented stand-alone communication helpers remain usable without in-tree callers."""

import jax
import numpy as np
from easydel.layers.moe._communication_utils import argsort, bincount, repeat_take_sorted, take1d, tp_global_topk
from jax import numpy as jnp


def test_sort_and_gather_helpers_preserve_rows():
    rows = jnp.asarray([[10, 20], [30, 40], [50, 60]], dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray(take1d(rows, jnp.asarray([2, 0, 1]))), [[50, 60], [10, 20], [30, 40]])
    np.testing.assert_array_equal(np.asarray(argsort(jnp.asarray([[3, 1, 2], [6, 4, 5]]))), [[1, 2, 0], [1, 2, 0]])


def test_repeat_and_histogram_helpers():
    rows = jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(repeat_take_sorted(rows, jnp.asarray([2, 0, 3, 1]), k=2)),
        [[3, 4], [1, 2], [3, 4], [1, 2]],
    )
    np.testing.assert_array_equal(np.asarray(bincount(jnp.asarray([0, 2, 1, 0, 2, 2]), length=4)), [2, 1, 3, 0])


def test_histogram_ignores_negative_entries_instead_of_counting_them_in_bin_zero():
    actual = bincount(jnp.asarray([-1, 0, 2, -7, 2]), length=4)
    np.testing.assert_array_equal(np.asarray(actual), [1, 0, 2, 0])


def test_global_topk_can_select_more_experts_than_each_local_shard_contains():
    logits = jnp.asarray([[7.0, 9.0], [1.0, 8.0]])
    values, indices = jax.vmap(lambda x: tp_global_topk(x, k=3, tp_axis="tp"), axis_name="tp")(logits)
    np.testing.assert_array_equal(np.asarray(values), [[9.0, 8.0, 7.0], [9.0, 8.0, 7.0]])
    np.testing.assert_array_equal(np.asarray(indices), [[1, 3, 0], [1, 3, 0]])
