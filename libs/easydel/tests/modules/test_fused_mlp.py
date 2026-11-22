"""Fused-projection parameter and numerical compatibility (accelerator-only)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLMLP
from easydel.modules.qwen2_vl.qwen2_vl_configuration import Qwen2VLTextConfig
from easydel.modules.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
from easydel.modules.qwen3_moe.qwen3_moe_configuration import Qwen3MoeConfig
from jax import numpy as jnp


@pytest.mark.parametrize("vision", [False, True])
def test_dense_mlp_keeps_fused_weight_layout_and_matches_numpy_swiglu(vision):
    config_cls = Qwen2VLTextConfig if vision else Qwen3MoeConfig
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        use_fused_mlp=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    cls = Qwen2VLMLP if vision else Qwen3MoeMLP
    kwargs = {"layer_idx": 0} if vision else {}
    mlp = cls(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0), **kwargs)
    x = np.arange(32, dtype=np.float32).reshape(1, 2, 16) / 100
    weight = np.asarray(mlp.gate_up_proj.weight.value)
    assert weight.shape == (16, 64)
    assert weight.dtype == np.float32
    assert mlp.gate_up_proj.bias is None
    gate, up = np.split(x @ weight, 2, axis=-1)
    expected = ((gate / (1 + np.exp(-gate))) * up) @ np.asarray(mlp.down_proj.weight.value)
    np.testing.assert_allclose(mlp(jnp.asarray(x)), expected, rtol=2e-5, atol=1e-6)
