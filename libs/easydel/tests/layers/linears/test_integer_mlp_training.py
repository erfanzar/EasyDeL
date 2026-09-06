"""Training dispatch must preserve activation autodiff in integer-mode MLPs."""

import types

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.linears import ColumnParallelLinearQuantized, RowParallelLinearQuantized
from easydel.layers.mlp import gated_mlp_forward
from easydel.layers.quantization import QuantizationConfig


@pytest.mark.parametrize("mode", ["w4a16", "w8a16", "w4a4", "w8a8"])
def test_integer_mlp_training_jvp_vjp_matches_composition(mode):
    cfg = ed.LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        use_fused_mlp=True,
    )
    qcfg = QuantizationConfig.for_matmul(mode)
    with cfg.mesh:
        gu = ColumnParallelLinearQuantized(16, 64, use_bias=False, config=qcfg, dtype=jnp.float32, rngs=spx.Rngs(10))
        dn = RowParallelLinearQuantized(32, 16, use_bias=False, config=qcfg, dtype=jnp.float32, rngs=spx.Rngs(11))
    mlp = types.SimpleNamespace(config=cfg, gate_up_proj=gu, down_proj=dn, act_fn=jax.nn.silu)
    x = jax.random.normal(jax.random.key(13), (1, 8, 16))
    dx = jax.random.normal(jax.random.key(14), x.shape)

    def reference(a):
        gate, up = jnp.split(gu(a), 2, axis=-1)
        return dn(jax.nn.silu(gate) * up)

    def actual(a):
        return gated_mlp_forward(mlp, a)

    with cfg.mesh:
        got, dgot = jax.jit(lambda a, t: jax.jvp(actual, (a,), (t,)))(x, dx)
        want, dwant = jax.jit(lambda a, t: jax.jvp(reference, (a,), (t,)))(x, dx)
        vg = jax.jit(jax.grad(lambda a: jnp.sum(actual(a) ** 2)))(x)
        vw = jax.jit(jax.grad(lambda a: jnp.sum(reference(a) ** 2)))(x)
    for left, right in [(got, want), (dgot, dwant), (vg, vw)]:
        np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-6)
        assert np.isfinite(left).all()
    assert float(jnp.linalg.norm(vg)) > 0
