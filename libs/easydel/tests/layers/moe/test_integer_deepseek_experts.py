"""DeepSeek split gate/up/down integer mode integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.quantization import QuantizationConfig
from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config
from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock


def project(x, layer, e, bits):
    if bits != 16:
        bound = (1 << (bits - 1)) - 1
        scale = np.max(abs(x), axis=-1, keepdims=True) / bound
        x = np.clip(np.round(x / np.where(scale == 0, 1, scale)), -bound, bound) * scale
    return (x @ np.asarray(layer.quant_kernel.value).astype(np.float32)[e]) * np.asarray(layer.quant_scales.value)[e]


@pytest.mark.parametrize("mode", ["w4a16", "w8a16", "w4a4", "w8a8"])
@pytest.mark.parametrize("routing", ["moe", "hash_moe"])
def test_deepseek_split_experts_match_independent_reference(mode, routing):
    cfg = DeepseekV4Config(
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        mlp_layer_types=[routing],
        routed_scaling_factor=1.7,
        swiglu_limit=1.0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    block = DeepseekV4SparseMoeBlock(
        cfg, 0, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(11)
    )
    if routing == "hash_moe":
        # Populate the checkpoint-owned table with distinct valid expert IDs.
        block.gate.tid2eid.value = jnp.arange(128, dtype=jnp.int32).reshape(64, 2) % 4
    qcfg = QuantizationConfig.for_matmul(mode)
    for name in ["gate_proj", "up_proj", "down_proj"]:
        layer = getattr(block.experts, name)
        layer.weight.value = layer.weight.value * 5
        setattr(block.experts, name, layer.to_quantized(qcfg))
    x = jnp.asarray(np.random.default_rng(22).normal(size=(1, 8, 32)) * 3, jnp.float32)
    ids = jnp.arange(8, dtype=jnp.int32)[None]
    with cfg.mesh:
        scores = np.asarray(block.gate(x.reshape(8, 32)))
        if routing == "hash_moe":
            chosen = np.asarray(block.gate.tid2eid.value)[np.asarray(ids).ravel()]
        else:
            chosen = np.argsort(-(scores + np.asarray(block.gate.e_score_correction_bias.value)), axis=-1)[:, :2]
        weights = np.take_along_axis(scores, chosen, axis=1)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-20) * cfg.routed_scaling_factor
        want = np.zeros((8, 32), np.float32)
        for r in range(8):
            for slot in range(2):
                e = chosen[r, slot]
                a = np.asarray(x)[0, r : r + 1]
                gate = project(a, block.experts.gate_proj, e, qcfg.activation_bits)
                up = project(a, block.experts.up_proj, e, qcfg.activation_bits)
                gate = np.minimum(gate, cfg.swiglu_limit)
                up = np.clip(up, -cfg.swiglu_limit, cfg.swiglu_limit)
                h = gate / (1 + np.exp(-gate)) * up
                want[r] += project(h, block.experts.down_proj, e, qcfg.activation_bits)[0] * weights[r, slot]
        want = want.reshape(1, 8, 32) + np.asarray(block.shared_experts(x))
        got, _ = jax.jit(lambda a: block(a, input_ids=ids))(x)
    np.testing.assert_allclose(got, want, rtol=4e-5, atol=4e-5)
