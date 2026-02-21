from eformer import common_types
from eformer.escale import PartitionAxis

from easydel.axis import (
    ATTN_DP,
    register_attention_data_parallel_axis,
    reset_attention_data_parallel_axis,
)


def test_attn_dp_defaults_to_partition_axis_data_parallel_axis():
    reset_attention_data_parallel_axis()
    paxis = PartitionAxis(data_parallel_axis="ep")
    resolved = paxis.resolve_axis([ATTN_DP], mode=common_types.MODE_PREFILL)
    assert resolved == ["ep"]


def test_attn_dp_can_be_overridden_globally():
    reset_attention_data_parallel_axis()
    try:
        register_attention_data_parallel_axis("ep")
        paxis = PartitionAxis(data_parallel_axis="dp")
        resolved = paxis.resolve_axis([ATTN_DP], mode=common_types.MODE_PREFILL)
        assert resolved == ["ep"]
    finally:
        reset_attention_data_parallel_axis()
