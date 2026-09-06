"""TP4/ExpertTensor contracts for explicitly converted integer MoE modes.

CPU: ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=4.
TPU: ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu on four devices.
CPU ExpertTensor uses ring collectives because ragged-all-to-all has no CPU
lowering. TPU uses the default ragged-all-to-all communication path.
Forward is jitted; input AD differentiates that already-compiled fused call.

Standard TP packs [gate_shard0, up_shard0, ...] and calibrates the down
activation independently on each contraction shard. ExpertTensor distributes
experts instead, requiring [all_gate, all_up] packing and full-row calibration.
Thus A4/A8 need separate references, NOT cross-layout output equality. A16
compares exactly the same represented weights in both layouts. Input AD uses
the documented represented-weight STE, with routing differentiated normally
away from top-k ties. Shared experts are deliberately excluded from the oracle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import decode_mode_specs
from easydel.layers.quantization import QuantizationConfig
from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpSparseMoeBlock
from easydel.modules.qwen4_exp.qwen4_exp_configuration import Qwen4ExpTextConfig

TP = 4


def _canonical(value):
    """Unpack standard TP's interleaved gate/up channels, including scales."""
    packed = value.reshape(*value.shape[:-1], TP, 2, value.shape[-1] // (2 * TP))
    return jnp.swapaxes(packed, -3, -2).reshape(value.shape)


def _quantize_ste(x, bits):
    if bits == 16:
        return x
    bound = (1 << (bits - 1)) - 1
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / bound
    q = jnp.clip(jnp.round(x / jnp.where(scale == 0, 1, scale)), -bound, bound)
    represented = q * scale
    return x + jax.lax.stop_gradient(represented - x)


def _build(mode):
    config = Qwen4ExpTextConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=16,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    config.add_basic_configurations(sharding_axis_dims=(1, 1, 1, 1, TP, 1))
    block = Qwen4ExpSparseMoeBlock(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(17),
    )
    quant = QuantizationConfig.for_matmul(mode)
    # Nontrivial routed output prevents shared experts or tiny initialization
    # from masking a missing integer projection.
    for name in ("gate_up_proj", "down_proj"):
        projection = getattr(block.experts, name)
        projection.weight.value = projection.weight.value * 5
        setattr(block.experts, name, projection.to_quantized(quant))
    gp, dp = block.experts.gate_up_proj, block.experts.down_proj
    assert gp.quant_kernel.value.dtype == jnp.dtype(jnp.int4 if mode.startswith("w4") else jnp.int8)
    assert gp.kernel_view().activation_bits == quant.activation_bits
    assert dp.kernel_view().activation_bits == quant.activation_bits
    canonical = (
        _canonical(gp.quant_kernel.value).astype(jnp.float32),
        _canonical(gp.quant_scales.value),
        dp.quant_kernel.value.astype(jnp.float32),
        dp.quant_scales.value,
    )
    return block, canonical, quant.activation_bits


def _reference(block, kernels, bits, contraction_shards):
    gc, gs, dc, ds = kernels

    def forward(hidden):
        x = hidden.reshape(-1, hidden.shape[-1])
        logits = block.gate(x).astype(jnp.float32)
        probs = jax.nn.softmax(logits, axis=-1)
        weights, selected = jax.lax.top_k(probs, 2)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        result = jnp.zeros_like(x)
        for expert in range(gc.shape[0]):
            gu = jnp.matmul(_quantize_ste(x, bits), gc[expert], precision=jax.lax.Precision.HIGHEST) * gs[expert]
            gate, up = jnp.split(gu, 2, axis=-1)
            intermediate = jax.nn.silu(gate) * up
            # Standard TP uses LOCAL absmax for each down-projection K shard;
            # ExpertTensor has the entire K dimension on each owning device.
            local_hidden = jnp.split(intermediate, contraction_shards, axis=-1)
            local_codes = jnp.split(dc[expert], contraction_shards, axis=0)
            down = sum(
                jnp.matmul(_quantize_ste(h, bits), c, precision=jax.lax.Precision.HIGHEST) * ds[expert]
                for h, c in zip(local_hidden, local_codes, strict=True)
            )
            routing = jnp.sum(jnp.where(selected == expert, weights, 0), axis=-1)
            result = result + down * routing[:, None]
        return result.reshape(hidden.shape)

    return forward


def _routed(block, hidden):
    return block.moe_call(
        hidden_state=hidden,
        gate_layer=block.gate,
        expert_layer=block.experts,
        gate_up_kernel=block.experts.gate_up_proj.kernel_view(),
        wd_kernel=block.experts.down_proj.kernel_view(),
        act_fn=block.experts.act_fn,
    )[0]


@pytest.mark.parametrize("mode", ["w4a16", "w8a16", "w4a4", "w8a8"])
@pytest.mark.parametrize("check", ["forward", "jvp", "vjp"])
def test_integer_modes_tp4_and_expert_tensor_forward_and_input_ad(mode, check):
    if jax.default_backend() not in ("cpu", "tpu") or jax.device_count() != TP:
        pytest.skip("requires four CPU or TPU devices; see module run instructions")
    block, kernels, bits = _build(mode)
    mesh = block.config.mesh
    assert mesh.shape["tp"] == TP
    rng = np.random.default_rng(23)
    hidden = jnp.asarray(rng.normal(size=(1, 8, 32)) * 3, jnp.float32)
    tangent = jnp.asarray(rng.normal(size=hidden.shape), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=hidden.shape), jnp.float32)
    outputs = []
    with spx.use_mesh(mesh), decode_mode_specs(True):
        for expert_tensor in (False, True):
            if expert_tensor:
                # Repack the ACTUAL converted codes AND scales. Down weights
                # retain their logical layout; no re-quantization is involved.
                gp = block.experts.gate_up_proj
                for param in (gp.quant_kernel, gp.quant_scales):
                    param.value = _canonical(param.value)
                block.config.use_expert_tensor_mode = True
                # XLA:CPU has no ragged-all-to-all lowering. The existing ring
                # collective path exercises the same expert kernels/layout.
                block.config.use_ring_of_experts = jax.default_backend() == "cpu"
                np.testing.assert_array_equal(gp.quant_kernel.value.astype(jnp.float32), kernels[0])
                np.testing.assert_array_equal(gp.quant_scales.value, kernels[1])
            reference = _reference(block, kernels, bits, 1 if expert_tensor else TP)
            actual = jax.jit(lambda x: _routed(block, x))
            expected = jax.jit(reference)
            got, want = actual(hidden), expected(hidden)
            label = f"{mode} {'ExpertTensor' if expert_tensor else 'TP4'}"
            assert np.isfinite(np.asarray(got)).all(), label
            assert float(jnp.linalg.norm(want)) > 0.1, label
            np.testing.assert_allclose(got, want, rtol=4e-5, atol=4e-5, err_msg=label)
            if check == "jvp":
                _, got_jvp = jax.jvp(actual, (hidden,), (tangent,))
                _, want_jvp = jax.jvp(expected, (hidden,), (tangent,))
                np.testing.assert_allclose(got_jvp, want_jvp, rtol=8e-5, atol=8e-5, err_msg=label + " input JVP")
            elif check == "vjp":
                _, got_pullback = jax.vjp(actual, hidden)
                _, want_pullback = jax.vjp(expected, hidden)
                np.testing.assert_allclose(
                    got_pullback(cotangent)[0],
                    want_pullback(cotangent)[0],
                    rtol=8e-5,
                    atol=8e-5,
                    err_msg=label + " input VJP",
                )
            outputs.append(got)
        if bits == 16:
            np.testing.assert_allclose(outputs[0], outputs[1], rtol=4e-5, atol=4e-5)
        else:
            # Ensure this fixture actually distinguishes per-shard calibration
            # from whole-row calibration; equality is NOT the quantized contract.
            assert float(jnp.linalg.norm(outputs[0] - outputs[1])) > 1e-3
