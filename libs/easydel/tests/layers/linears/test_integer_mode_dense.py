"""Explicit integer precision: real CPU linears and fused MLP, not dispatch mocks."""

import types

import easydel as ed
import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.linears import ColumnParallelLinearQuantized, RowParallelLinearQuantized
from easydel.layers.mlp import _quantized_fused_call
from easydel.layers.quantization import QuantizationConfig
from jax import numpy as jnp


def _model(mode, policy="explicit"):
    cfg = ed.LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    qcfg = QuantizationConfig.for_matmul(mode)
    qcfg.activation_policy = policy
    if policy == "auto" and qcfg.activation_bits == 16:
        qcfg.activation_bits = None  # Legacy weight-only default.
    with cfg.mesh:
        gu = ColumnParallelLinearQuantized(16, 64, use_bias=False, config=qcfg, dtype=jnp.float32, rngs=spx.Rngs(10))
        dn = RowParallelLinearQuantized(32, 16, use_bias=False, config=qcfg, dtype=jnp.float32, rngs=spx.Rngs(11))
    return cfg, types.SimpleNamespace(gate_up_proj=gu, down_proj=dn, act_fn=jax.nn.silu)


def _reference(x, linear, bits):
    """Independent numpy per-token symmetric quantization, including zero rows."""
    x = np.asarray(x, np.float32)
    if bits is not None:
        bound = 2 ** (bits - 1) - 1
        scale = np.max(np.abs(x), axis=-1, keepdims=True) / bound
        codes = np.clip(np.rint(x / np.where(scale == 0, 1, scale)), -bound, bound)
        x = codes * scale
    w = np.asarray(linear.quant_kernel.value, np.float32)
    s = np.asarray(linear.quant_scales.value, np.float32).reshape(1, -1)
    return (x @ w) * s


@pytest.mark.parametrize("mode", ["w4a16", "w8a16", "w4a4", "w8a8"])
@pytest.mark.parametrize("tokens", [8, 256])
@pytest.mark.parametrize("policy", ["explicit", "auto"])
def test_integer_dense_and_fused_reference(mode, tokens, policy):
    cfg, mlp = _model(mode, policy)
    x = np.random.default_rng(77).normal(size=(tokens, 16)).astype(np.float32) * 1.73
    x[0] = 0
    requested = mlp.gate_up_proj.config.activation_bits
    bits = (None if requested == 16 else requested) if policy == "explicit" else (requested if tokens >= 256 else None)
    gu = _reference(x, mlp.gate_up_proj, bits)
    g, u = np.split(gu, 2, axis=-1)
    hidden = (g / (1 + np.exp(-g))) * u
    want = _reference(hidden, mlp.down_proj, bits)
    # These inputs distinguish the requested activation regime even for A8.
    alternate = _reference(x, mlp.gate_up_proj, 8 if bits is None else None)
    assert np.max(np.abs(gu - alternate)) > 1e-4
    with cfg.mesh:
        for fn in (lambda z: mlp.gate_up_proj(z), jax.jit(lambda z: mlp.gate_up_proj(z))):
            np.testing.assert_allclose(fn(jnp.asarray(x)), gu, rtol=3e-5, atol=3e-6)
        for fn in (lambda z: _quantized_fused_call(mlp, cfg, z), jax.jit(lambda z: _quantized_fused_call(mlp, cfg, z))):
            got = fn(jnp.asarray(x))
            assert got is not None
            np.testing.assert_allclose(got, want, rtol=8e-5, atol=3e-6)


@pytest.mark.parametrize("down_mode,down_policy", [("w4a4", "explicit"), ("w4a16", "auto")])
def test_mismatched_activation_contract_declines_fusion(down_mode, down_policy):
    cfg, mlp = _model("w4a16")
    mlp.down_proj.config = QuantizationConfig.for_matmul(down_mode)
    mlp.down_proj.config.activation_policy = down_policy
    with cfg.mesh:
        assert _quantized_fused_call(mlp, cfg, jnp.ones((8, 16))) is None


def test_explicit_a16_never_supplies_packed_weights(monkeypatch):
    from easydel.layers import mlp as mlp_mod

    cfg, mlp = _model("w4a16")
    for linear in (mlp.gate_up_proj, mlp.down_proj):
        linear.quant_kernel_packed = types.SimpleNamespace(
            value=jnp.zeros((linear.in_features // 2, linear.out_features_sum), jnp.uint8)
        )
    monkeypatch.setattr(mlp_mod, "W4A4_FUSED_PAIR_MIN_PACKED_BYTES", 0)
    seen = {}

    def capture(x, **kwargs):
        seen.update(kwargs)
        return x

    monkeypatch.setattr(mlp_mod, "ejkernel_fused_mlp", capture)
    with cfg.mesh:
        _quantized_fused_call(mlp, cfg, jnp.ones((8, 16)))
    assert "packed_weights" not in seen
    assert seen["quantize_activations"] is False


@pytest.mark.parametrize("mode", ["w4a16", "w8a16", "w4a4", "w8a8"])
def test_explicit_kernel_contract_at_decode(monkeypatch, mode):
    """Trace real implementations to verify their explicit decode contracts."""
    import ejkernel.modules as kernels
    from easydel.layers import mlp as mlp_mod

    cfg, mlp = _model(mode)
    seen = []

    def record(fn):
        def wrapped(*args, **kwargs):
            seen.append(kwargs)
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(kernels, "channelwise_quantized_matmul", record(kernels.channelwise_quantized_matmul))
    monkeypatch.setattr(mlp_mod, "ejkernel_fused_mlp", record(mlp_mod.ejkernel_fused_mlp))
    with cfg.mesh:
        jax.make_jaxpr(lambda x: mlp.gate_up_proj(x))(jnp.ones((8, 16)))
        jax.make_jaxpr(lambda x: _quantized_fused_call(mlp, cfg, x))(jnp.ones((8, 16)))
    assert len(seen) == 2
    bits = mlp.gate_up_proj.config.activation_bits
    for kwargs in seen:
        assert kwargs["quantize_activations"] is (bits != 16)
        assert kwargs["activation_bits"] == (8 if bits == 16 else bits)
        assert kwargs["prefill_threshold"] == 0


def test_auto_a16_retains_legacy_backend_validation():
    cfg, mlp = _model("w4a16", "auto")
    mlp.gate_up_proj.config.activation_bits = 16
    with cfg.mesh, pytest.raises(ValueError, match="activation_bits must be 4 or 8"):
        mlp.gate_up_proj(jnp.ones((8, 16)))
