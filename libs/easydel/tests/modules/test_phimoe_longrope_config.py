"""PhiMoe LongRoPE configuration validation (config only, no array computation)."""

import pytest
from easydel.modules.phimoe.phimoe_configuration import PhiMoeConfig


def longrope(factor_length):
    return {
        "type": "longrope",
        "short_factor": [1.0] * factor_length,
        "long_factor": [2.0] * factor_length,
        "short_mscale": 1.1,
        "long_mscale": 1.2,
        "original_max_position_embeddings": 16,
    }


@pytest.mark.cpu_ok
def test_longrope_uses_explicit_head_dimension():
    config = PhiMoeConfig(hidden_size=16, num_attention_heads=2, head_dim=4, rope_scaling=longrope(2))
    assert config.head_dim == 4
    assert config.rope_scaling["short_factor"] == [1.0, 1.0]
    assert config.rope_scaling["long_factor"] == [2.0, 2.0]
    assert config.rope_scaling["short_mscale"] == 1.1
    assert config.original_max_position_embeddings == 16


@pytest.mark.cpu_ok
@pytest.mark.parametrize("field", ["short_factor", "long_factor"])
def test_longrope_rejects_legacy_width_for_custom_head_dimension(field):
    scaling = longrope(2)
    scaling[field] = [1.0] * 4
    with pytest.raises(ValueError, match=rf"{field} field must have length 2, got 4"):
        PhiMoeConfig(hidden_size=16, num_attention_heads=2, head_dim=4, rope_scaling=scaling)
