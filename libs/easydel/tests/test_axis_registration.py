"""Pure-Python public attention-axis contracts, with isolated registrations."""

import pytest
from easydel.axis import ATTN_DP, register_attention_data_parallel_axis, resolve_attention_data_parallel_axis
from spectrax import PartitionAxis
from spectrax.common_types import MODE_DECODE, MODE_PREFILL, MODE_TRAIN

pytestmark = pytest.mark.cpu_ok


@pytest.fixture(autouse=True)
def restore_attention_axis_registration():
    original = PartitionAxis.get_registered_axes().get(ATTN_DP)
    try:
        yield
    finally:
        if original is None:
            PartitionAxis.unregister(ATTN_DP)
        else:
            PartitionAxis.register(ATTN_DP, **original, override=True)


@pytest.mark.parametrize("mode", [MODE_TRAIN, MODE_PREFILL, MODE_DECODE])
def test_explicitly_replicated_attention_axis_resolves_to_none(mode):
    register_attention_data_parallel_axis(None)
    assert resolve_attention_data_parallel_axis(PartitionAxis(), mode=mode) is None


@pytest.mark.parametrize("mode", [MODE_TRAIN, MODE_PREFILL, MODE_DECODE])
def test_generation_override_can_replicate_attention_axis(mode):
    register_attention_data_parallel_axis("attention_dp", generation_axis_rule=None)
    expected = None if mode == MODE_DECODE else "attention_dp"
    assert resolve_attention_data_parallel_axis(PartitionAxis(), mode=mode) == expected
