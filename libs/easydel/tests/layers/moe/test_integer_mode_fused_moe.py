"""Model-level precision contract for fused routed experts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.quantization import QuantizationConfig
from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpSparseMoeBlock
from easydel.modules.qwen4_exp.qwen4_exp_configuration import Qwen4ExpTextConfig


def _project(x, layer, expert, bits):
    if bits != 16:
        bound = 7 if bits == 4 else 127
        scale = np.max(np.abs(x), axis=-1, keepdims=True) / bound
        x = np.clip(np.round(x / np.where(scale == 0, 1, scale)), -bound, bound) * scale
    codes = np.asarray(layer.quant_kernel.value).astype(np.float32)[expert]
    scales = np.asarray(layer.quant_scales.value)[expert]
    return (x @ codes) * scales


@pytest.mark.parametrize(
    "gate_mode,down_mode",
    [("w4a16", "w4a16"), ("w8a16", "w8a16"), ("w4a4", "w4a4"), ("w8a8", "w8a8"), ("w4a16", "w8a8")],
)
def test_fused_qwen_experts_honor_per_projection_precision(gate_mode, down_mode):
    config = Qwen4ExpTextConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    config.add_basic_configurations(sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    block = Qwen4ExpSparseMoeBlock(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(17)
    )
    gp = block.experts.gate_up_proj
    dp = block.experts.down_proj
    gp.weight.value = gp.weight.value * 5
    dp.weight.value = dp.weight.value * 5
    gc = QuantizationConfig.for_matmul(gate_mode)
    dc = QuantizationConfig.for_matmul(down_mode)
    block.experts.gate_up_proj = gp.to_quantized(gc)
    block.experts.down_proj = dp.to_quantized(dc)
    x = jnp.asarray(np.random.default_rng(12).normal(size=(1, 8, 32)) * 3, jnp.float32)
    with spx.use_mesh(config.mesh):
        logits = np.asarray(block.gate(x.reshape(8, 32)))
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        selected = np.argsort(-probs, axis=-1)[:, :2]
        weights = np.take_along_axis(probs, selected, axis=-1)
        weights /= weights.sum(axis=-1, keepdims=True)
        want = np.zeros((8, 32), np.float32)
        for row in range(8):
            for slot in range(2):
                e = selected[row, slot]
                gu = _project(np.asarray(x)[0, row : row + 1], block.experts.gate_up_proj, e, gc.activation_bits)
                gate, up = np.split(gu, 2, axis=-1)
                hidden = (gate / (1 + np.exp(-gate))) * up
                want[row] += _project(hidden, block.experts.down_proj, e, dc.activation_bits)[0] * weights[row, slot]
        shared = np.asarray(block.shared_expert(x))
        sg = np.asarray(block.shared_expert_gate(x))
        want = want.reshape(1, 8, 32) + shared / (1 + np.exp(-sg))
        got, _ = jax.jit(lambda a: block(a))(x)
    np.testing.assert_allclose(got, want, rtol=3e-5, atol=3e-5)
