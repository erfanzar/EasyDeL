"""Public MoE placement properties retain their full weight-rank contract."""

import pytest
import spectrax as spx
from easydel.layers.linears import ColumnParallelMoELinear, RowParallelMoELinear
from jax.sharding import PartitionSpec
from spectrax.common_types import EP, TP


@pytest.mark.parametrize(
    ("linear_class", "expected_axes"),
    [
        (ColumnParallelMoELinear, ["ep", None, "tp"]),
        (RowParallelMoELinear, ["ep", "tp", None]),
    ],
)
def test_alt_sharding_axis_lists_weight_partition_entries(linear_class, expected_axes):
    layer = linear_class(4, 8, 16, rngs=spx.Rngs(0))
    assert layer.alt_sharding == PartitionSpec(*expected_axes)
    assert layer.alt_sharding_axis == expected_axes
    assert layer.expert_axis == EP


@pytest.mark.parametrize("linear_class", [ColumnParallelMoELinear, RowParallelMoELinear])
def test_expert_tensor_mode_reports_a_three_dimensional_weight_spec(linear_class):
    layer = linear_class(4, 8, 16, use_expert_tensor_mode=True, rngs=spx.Rngs(0))
    assert layer.weight.value.ndim == 3
    assert layer.alt_sharding == PartitionSpec("tp", None, None)
    assert layer.alt_sharding_axis == ["tp", None, None]
    assert layer.expert_axis == TP
