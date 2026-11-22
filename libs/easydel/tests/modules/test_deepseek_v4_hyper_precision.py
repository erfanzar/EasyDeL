"""Hyper-connection projections must honor the model's matmul precision."""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from easydel.modules.deepseek_v4 import DeepseekV4Config
from easydel.modules.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)


@pytest.mark.parametrize("module_type", [DeepseekV4HyperConnection, DeepseekV4HyperHead])
def test_hyper_projection_honors_explicit_precision(module_type):
    config = DeepseekV4Config(hidden_size=16, hc_mult=2, hc_sinkhorn_iters=2)
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    with config.mesh:
        module = module_type(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        with jax.default_matmul_precision("default"):
            graph = jax.make_jaxpr(module)(jnp.ones((1, 3, 2, 16), jnp.float32))
    dots = [eq for eq in graph.jaxpr.eqns if eq.primitive.name == "dot_general"]
    assert dots, "test must inspect a projection contraction"
    assert all(eq.params["precision"] == (jax.lax.Precision.HIGHEST,) * 2 for eq in dots)
