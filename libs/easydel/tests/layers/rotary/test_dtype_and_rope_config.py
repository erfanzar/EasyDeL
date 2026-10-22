"""Non-computation contracts for shared dtype and RoPE configuration helpers."""

from copy import deepcopy

import pytest
from easydel.layers.rotary import (
    rope_scaling_from_rope_parameters,
    vision_grid_pos_embed_interpolate,
    vision_grid_rotary_embeddings,
)

pytestmark = pytest.mark.cpu_ok


def test_dtype_promotion_compatibility_paths_share_none_preserving_implementation():
    from easydel.layers.embeddings._embeddings import _promote_dtype
    from easydel.layers.linears._linear import promote_dtype as linear_promote
    from easydel.layers.linears._linear_moe import promote_dtype as moe_promote
    from easydel.layers.norms import promote_dtype

    values = [object(), None]
    for promoter in (promote_dtype, linear_promote, moe_promote, _promote_dtype):
        assert promoter(values, inexact=True) is values
        assert promoter((None,), dtype="float32", inexact=False) == (None,)


def test_explicit_rope_scaling_wins_and_does_not_mutate_inputs():
    parameters = {"rope_type": "linear", "factor": 8.0}
    scaling = {"type": "yarn", "factor": 4.0, "custom_field": "preserved"}
    before = deepcopy((parameters, scaling))
    assert rope_scaling_from_rope_parameters(parameters, scaling) == {
        "type": "yarn",
        "rope_type": "yarn",
        "factor": 4.0,
        "custom_field": "preserved",
    }
    assert (parameters, scaling) == before
    scaling["rope_type"] = "default"
    assert rope_scaling_from_rope_parameters(parameters, scaling)["rope_type"] == "default"
    assert rope_scaling_from_rope_parameters(parameters, {}) == {}


def test_rope_parameter_conversion_keeps_all_scaling_and_multimodal_fields():
    fields = {
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "short_factor": [1.0, 2.0],
        "long_factor": [8.0, 9.0],
        "beta_fast": 32,
        "beta_slow": 1,
        "extrapolation_factor": 0.5,
        "attn_factor": 1.25,
        "mscale": 1.0,
        "mscale_all_dim": 1.5,
        "mrope_section": [8, 12, 12],
        "mrope_interleaved": True,
    }
    parameters = {"rope_type": "yarn", "rope_theta": 10000, **fields}
    before = deepcopy(parameters)
    assert rope_scaling_from_rope_parameters(parameters, None) == {"rope_type": "yarn", **fields}
    assert parameters == before
    assert rope_scaling_from_rope_parameters({"type": "linear", "factor": 2}, None) == {
        "rope_type": "linear",
        "factor": 2,
    }
    assert rope_scaling_from_rope_parameters({"type": "linear", "rope_type": "default"}, None) == {
        "rope_type": "default"
    }


def test_absent_rope_inputs_keep_text_and_vision_defaults_independent():
    assert rope_scaling_from_rope_parameters(None, None) is None
    preset = {"rope_type": "default", "mrope_section": [8, 12, 12]}
    first = rope_scaling_from_rope_parameters(None, None, preset)
    second = rope_scaling_from_rope_parameters(None, None, preset)
    assert first == second == preset
    first["mrope_section"][0] = 99
    assert second == preset == {"rope_type": "default", "mrope_section": [8, 12, 12]}
    assert rope_scaling_from_rope_parameters({}, None, preset) == {"rope_type": "default"}


@pytest.mark.parametrize("grids,merge", [([], 2), ([(1, 2, 2)], 0), ([(0, 2, 2)], 2), ([(1, 3, 2)], 2)])
def test_vision_helpers_reject_invalid_static_grids_before_array_computation(grids, merge):
    with pytest.raises(ValueError):
        vision_grid_pos_embed_interpolate(None, grids, 2, merge)
    with pytest.raises(ValueError):
        vision_grid_rotary_embeddings(grids, 8, 4, merge)


@pytest.mark.parametrize("extent,head_dim,theta", [(1, 4, 10000), (8, 3, 10000), (8, 4, 0)])
def test_rotary_helper_rejects_invalid_static_frequency_parameters(extent, head_dim, theta):
    with pytest.raises(ValueError):
        vision_grid_rotary_embeddings([(1, 2, 2)], extent, head_dim, 2, theta=theta)
