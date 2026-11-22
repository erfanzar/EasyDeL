"""Non-numerical compatibility checks for documented linear/MoE helpers."""

import pytest
from easydel.layers.linears._linear import promote_dtype
from easydel.layers.moe._communication_utils import canon_dim, get_moe_partition_spec
from jax import numpy as jnp
from jax.sharding import PartitionSpec


@pytest.mark.cpu_ok
@pytest.mark.parametrize(("ndim", "dim", "expected"), [(3, -1, 2), (4, -2, 2), (3, 0, 0), (3, 2, 2)])
def test_canonical_dimension_contract(ndim, dim, expected):
    assert canon_dim(ndim, dim) == expected


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("direction", "expert_tensor", "expected"),
    [
        ("column", False, PartitionSpec("ep", None, "tp")),
        ("row", False, PartitionSpec("ep", "tp", None)),
        ("column", True, PartitionSpec("tp", None, None)),
        ("row", True, PartitionSpec("tp", None, None)),
    ],
)
def test_moe_weight_spec_is_rank_three(direction, expert_tensor, expected):
    assert get_moe_partition_spec(None, direction, expert_tensor) == expected


@pytest.mark.cpu_ok
def test_unspecified_dtype_preserves_input_values():
    values = (object(), None)
    assert promote_dtype(values, dtype=None) is values


@pytest.mark.cpu_ok
def test_dtype_promotion_preserves_absent_values_without_creating_arrays():
    assert promote_dtype((None, None), dtype=jnp.float32) == (None, None)
