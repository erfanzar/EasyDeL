"""Regression: int8 ``(codes, scales)`` expert kernels must route to w8a8.

``ParallelMoELinearQuantized.kernel_view`` returns ``(codes, scales)`` and the
fused-MoE dispatch is documented to feed that to the w8a8 grouped matmul
(runtime per-row activation quantization + exact int32 ragged-dot accumulation).
A regression routed the tuple through the block-float ``grouped_matmulv3``
``rhs_scale`` path instead: that contract *bakes* the scale into the rhs, which
truncates to zero on int8 codes — every expert output collapses (and the
half-open gate/up split overflows on TPU), producing NaN logits on hardware.
The XLA/CPU dequant-to-float path masked it on fake-device meshes, so the
assertion here also pins the routing through a non-zero finite-output check.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _build_block(*, sharding_axis_dims):
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpSparseMoeBlock
    from easydel.modules.qwen4_exp.qwen4_exp_configuration import Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[3],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=2000,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=16,
        num_experts=8,
        num_experts_per_tok=2,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        eos_token_id=0,
    )
    config.add_basic_configurations(sharding_axis_dims=sharding_axis_dims)
    block = Qwen4ExpSparseMoeBlock(
        config=config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=None,
        rngs=spx.Rngs(0),
    )
    return block, config.mesh


def _switch_to_expert_tensor_layout(block, mesh):
    """Preserve logical gate/up weights while changing their physical packing."""
    ways = int(mesh.shape["tp"])
    projection = block.experts.gate_up_proj
    params = (
        (projection.quant_kernel, projection.quant_scales)
        if getattr(projection, "quant_kernel", None) is not None
        else (projection.weight,)
    )
    for param in params:
        value = param.value
        packed = value.reshape(*value.shape[:-1], ways, 2, value.shape[-1] // (2 * ways))
        param.value = jnp.swapaxes(packed, -3, -2).reshape(value.shape)
    block.config.use_expert_tensor_mode = True


@pytest.mark.parametrize("sharding_axis_dims", [(1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 4, 1)])
def test_quantized_expert_block_parity(sharding_axis_dims):
    from easydel.layers.quantization import QuantizationConfig, QuantizationType
    from easydel.layers.quantization._quants import EasyQuantizer

    block, mesh = _build_block(sharding_axis_dims=sharding_axis_dims)
    quantizer = EasyQuantizer(
        quantization_config=QuantizationConfig(
            dtype=QuantizationType.CHANNELWISE,
            bits=8,
            pattern=r".*experts\.(gate_up_proj|down_proj)",
        )
    )
    quant_block = quantizer.apply_quantization(block)

    # the twin must carry int8 codes, not a second dense copy
    codes = quant_block.experts.gate_up_proj.quant_kernel.value
    assert jnp.dtype(codes.dtype).itemsize == 1

    rng = np.random.default_rng(7)
    hidden = jnp.asarray(rng.standard_normal((2, 9, 64)), jnp.bfloat16)

    with spx.use_mesh(mesh):
        dense_out, _ = block(hidden_states=hidden)
        quant_out, _ = quant_block(hidden_states=hidden)

    dense_np = np.asarray(dense_out, dtype=np.float32)
    quant_np = np.asarray(quant_out, dtype=np.float32)
    assert np.all(np.isfinite(quant_np)), "quantized experts produced non-finite output"

    # int8 channelwise quantization noise on both kernels: a few percent of
    # the dense magnitude. The broken rhs_scale routing collapses the experts
    # to zero (scale truncated via astype(int8)), which this bound rejects.
    scale = max(float(np.max(np.abs(dense_np))), 1e-6)
    rel = float(np.max(np.abs(quant_np - dense_np))) / scale
    assert rel < 0.35, f"quantized-vs-dense expert output drifted too far: {rel:.3f}"


def test_expert_tensor_mode_matches_standard_tp4_block():
    block, mesh = _build_block(sharding_axis_dims=(1, 1, 1, 1, 4, 1))
    rng = np.random.default_rng(19)
    hidden = jnp.asarray(rng.standard_normal((2, 8, 64)), jnp.bfloat16)

    from easydel.infra.sharding import decode_mode_specs

    with spx.use_mesh(mesh):
        with decode_mode_specs(True):
            standard_out, _ = block(hidden_states=hidden)
        _switch_to_expert_tensor_layout(block, mesh)
        with decode_mode_specs(True):
            expert_tensor_out, _ = block(hidden_states=hidden)

    np.testing.assert_allclose(
        np.asarray(expert_tensor_out, dtype=np.float32),
        np.asarray(standard_out, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_quantized_expert_tensor_mode_matches_standard_tp4_block():
    """Serving's channelwise-int8 experts must preserve TP4 ExpertTensor semantics."""
    from easydel.layers.quantization import QuantizationConfig, QuantizationType
    from easydel.layers.quantization._quants import EasyQuantizer

    block, mesh = _build_block(sharding_axis_dims=(1, 1, 1, 1, 4, 1))
    quantizer = EasyQuantizer(
        quantization_config=QuantizationConfig(
            dtype=QuantizationType.CHANNELWISE,
            bits=8,
            pattern=r".*experts\.(gate_up_proj|down_proj)",
        )
    )
    block = quantizer.apply_quantization(block)
    rng = np.random.default_rng(23)
    # eSurge packs eight requests on the token axis, not the batch axis.
    hidden = jnp.asarray(rng.standard_normal((1, 8, 64)), jnp.bfloat16)
    from easydel.infra.sharding import decode_mode_specs

    with spx.use_mesh(mesh), decode_mode_specs(True):
        standard_out, _ = block(hidden_states=hidden)
        _switch_to_expert_tensor_layout(block, mesh)
        expert_tensor_out, _ = block(hidden_states=hidden)
    expected = np.asarray(standard_out, dtype=np.float32)
    actual = np.asarray(expert_tensor_out, dtype=np.float32)
    assert np.all(np.isfinite(actual))
    magnitude = float(np.linalg.norm(expected))
    assert magnitude > 0, "parity oracle must not be identically zero"
    relative_error = float(np.linalg.norm(actual - expected)) / magnitude
    # An absolute 0.03 tolerance can hide a completely missing tiny-initialized
    # expert output. Bound the error relative to the actual reference instead.
    assert relative_error < 0.05, f"ExpertTensor relative L2 drift: {relative_error:.6f}"
