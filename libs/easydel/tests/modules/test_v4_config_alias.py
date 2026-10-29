"""The historical V4 rotary accessor remains a working public compatibility API."""

import pytest
from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("label", ["main", "compress"])
def test_v4_rope_accessor_alias_preserves_parameters(label):
    config = DeepseekV4Config()
    parameters = config.v4_rope_params(label)
    assert parameters == config.rope_parameters_for_label(label)
    assert {"rope_type", "rope_theta", "partial_rotary_factor"} <= parameters.keys()


def test_v4_rope_accessor_alias_preserves_unknown_label_error():
    with pytest.raises(KeyError):
        DeepseekV4Config().v4_rope_params("unknown")
