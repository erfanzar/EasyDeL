# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ParallelLinear.distributed_matmul seam and its factory.

Covers the factory's engine/decline rules, forward+gradient parity of the
explicit-collective engine against the GSPMD einsum path, and the
model-level wiring hook.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.linears._distributed_matmul import (
    VALID_COLLECTIVE_MATMUL_IMPLS,
    make_distributed_matmul,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec

_TP = 4
if len(jax.devices()) < _TP:
    pytest.skip("distributed matmul seam tests need >= 4 devices", allow_module_level=True)


def _mesh() -> Mesh:
    return Mesh(np.array(jax.devices()[:_TP]).reshape(_TP), axis_names=("tp",))


def _sharded_operands(mesh: Mesh, m: int = 16, k: int = 64, n: int = 32):
    kx, kw = jax.random.split(jax.random.PRNGKey(0))
    x = jax.device_put(
        jax.random.normal(kx, (m, k), jnp.float32),
        NamedSharding(mesh, PartitionSpec(None, "tp")),
    )
    w = jax.device_put(
        jax.random.normal(kw, (k, n), jnp.float32),
        NamedSharding(mesh, PartitionSpec("tp", None)),
    )
    return x, w


def test_factory_none_and_column_return_no_engine():
    assert make_distributed_matmul("row", "none") is None
    assert make_distributed_matmul("column", "xla") is None
    assert make_distributed_matmul(None, "xla") is None


def test_factory_auto_builds_row_engine():
    assert callable(make_distributed_matmul("row", "auto"))
    assert make_distributed_matmul("column", "auto") is None


def test_factory_rejects_unknown_impl():
    with pytest.raises(ValueError, match="collective_matmul_impl"):
        make_distributed_matmul("row", "nccl")
    assert "none" in VALID_COLLECTIVE_MATMUL_IMPLS
    assert "auto" in VALID_COLLECTIVE_MATMUL_IMPLS


def test_row_engine_matches_einsum_forward():
    mesh = _mesh()
    x, w = _sharded_operands(mesh)
    engine = make_distributed_matmul("row", "xla")
    assert engine is not None

    got = engine(x, w)
    assert got is not None
    expected = jnp.einsum("...i,io->...o", x, w)
    assert got.shape == expected.shape
    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


def _tp_only_config():
    import easydel as ed

    return ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        sharding_axis_dims=(1, 1, 1, 1, _TP, 1),
    )


def test_row_engine_matches_einsum_grad():
    # Under grad tracing the operands carry no NamedSharding: the engine
    # resolves the DECLARED weight layout through the config's sharding
    # resolver (canonical row-parallel on a tp-only mesh -> engages).
    config = _tp_only_config()
    mesh = config.mesh
    x, w = _sharded_operands(mesh)
    engine = make_distributed_matmul("row", "xla", config=config)

    def loss_engine(x, w):
        y = engine(x, w)
        assert y is not None, "canonical declared layout must engage under tracing inside the mesh context"
        return jnp.sum(y**2)

    def loss_ref(x, w):
        return jnp.sum(jnp.einsum("...i,io->...o", x, w) ** 2)

    with mesh:
        gx, gw = jax.grad(loss_engine, argnums=(0, 1))(x, w)
    gx_ref, gw_ref = jax.grad(loss_ref, argnums=(0, 1))(x, w)
    assert jnp.allclose(gx, gx_ref, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(gw, gw_ref, rtol=1e-4, atol=1e-4)


def test_traced_engine_declines_without_config():
    """No config -> no declared layout to trust under tracing -> einsum path."""
    mesh = _mesh()
    x, w = _sharded_operands(mesh)
    engine = make_distributed_matmul("row", "xla")

    declined = []

    def f(x, w):
        y = engine(x, w)
        declined.append(y is None)
        return y if y is not None else jnp.einsum("...i,io->...o", x, w)

    with mesh:
        jaxpr = jax.make_jaxpr(f)(x, w)
    assert declined == [True]
    assert "shard_map" not in str(jaxpr)


def test_traced_engine_declines_on_fsdp_sharded_weight_layout():
    """Declared RowWise layout with active fsdp must decline to the einsum path.

    On an fsdp>1 mesh the RowWise weight layout is (tp, (fsdp, sp)); the
    seam's canonical shard_map in_specs would force-gather the fsdp-sharded
    output dim on every forward, so the engine must decline under tracing.
    """
    import easydel as ed

    if len(jax.devices()) < 8:
        pytest.skip("fsdp x tp layout test needs 8 devices")

    config = ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        sharding_axis_dims=(1, 1, 2, 1, 4, 1),
    )
    mesh = config.mesh
    engine = make_distributed_matmul("row", "auto", config=config)

    x = jnp.ones((8, 16, 64), jnp.float32)
    w = jnp.ones((64, 32), jnp.float32)

    declined = []

    def f(x, w):
        y = engine(x, w)
        declined.append(y is None)
        return y if y is not None else jnp.einsum("...i,io->...o", x, w)

    with mesh:
        jaxpr = jax.make_jaxpr(f)(x, w)
    assert declined == [True]
    assert "shard_map" not in str(jaxpr)


def test_traced_engine_engages_on_canonical_tp_layout():
    """Canonical tp-only declared layout must still engage under tracing."""
    config = _tp_only_config()
    mesh = config.mesh
    x, w = _sharded_operands(mesh)
    engine = make_distributed_matmul("row", "xla", config=config)

    def f(x, w):
        y = engine(x, w)
        assert y is not None
        return y

    with mesh:
        jaxpr = jax.make_jaxpr(f)(x, w)
    assert "shard_map" in str(jaxpr)

    with mesh:
        got = jax.jit(f)(x, w)
    expected = jnp.einsum("...i,io->...o", x, w)
    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_row_engine_declines_on_indivisible_shapes():
    mesh = _mesh()
    engine = make_distributed_matmul("row", "xla")
    x = jax.device_put(
        jnp.ones((8, 30), jnp.float32),
        NamedSharding(mesh, PartitionSpec(None, None)),
    )
    w = jax.device_put(
        jnp.ones((30, 16), jnp.float32),
        NamedSharding(mesh, PartitionSpec(None, None)),
    )
    assert engine(x, w) is None


def test_row_engine_declines_without_mesh():
    engine = make_distributed_matmul("row", "xla")
    x = np.ones((8, 64), np.float32)
    w = np.ones((64, 16), np.float32)
    assert engine(jnp.asarray(x), jnp.asarray(w)) is None


def test_layer_forward_uses_engine_and_matches_unwired():
    import spectrax as spx
    from easydel.layers.linears import RowParallelLinear

    mesh = _mesh()
    layer = RowParallelLinear(64, 32, use_bias=True, rngs=spx.Rngs(0), param_dtype=jnp.float32)
    w = jax.device_put(
        jax.random.normal(jax.random.PRNGKey(1), (64, 32), jnp.float32),
        NamedSharding(mesh, PartitionSpec("tp", None)),
    )
    layer.weight.value = w
    x = jax.device_put(
        jax.random.normal(jax.random.PRNGKey(2), (4, 16, 64), jnp.float32),
        NamedSharding(mesh, PartitionSpec(None, None, "tp")),
    )

    baseline = layer(x)
    layer.distributed_matmul = make_distributed_matmul("row", "xla")
    wired = layer(x)
    assert wired.shape == baseline.shape
    assert jnp.allclose(wired, baseline, rtol=1e-5, atol=1e-5)

    layer.distributed_matmul = None
    assert jnp.allclose(layer(x), baseline, rtol=1e-6, atol=1e-6)


def test_wire_hook_installs_and_clears_engines():
    import easydel as ed

    config = ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        sharding_axis_dims=(1, 1, 1, 1, _TP, 1),
    )
    assert config.collective_matmul_impl == "auto"
    import spectrax as spx

    model = ed.LlamaForCausalLM(config=config, rngs=spx.Rngs(0), param_dtype=jnp.float32, dtype=jnp.float32)
    model.wire_distributed_matmul()

    from easydel.layers.linears import ColumnParallelLinear, RowParallelLinear

    row_engines, column_engines = [], []
    for _, module in spx.iter_modules(model):
        if isinstance(module, RowParallelLinear):
            row_engines.append(module.distributed_matmul)
        elif isinstance(module, ColumnParallelLinear):
            column_engines.append(module.distributed_matmul)
    assert row_engines and all(callable(e) for e in row_engines)
    assert column_engines and all(e is None for e in column_engines)

    config.collective_matmul_impl = "none"
    model.wire_distributed_matmul()
    for _, module in spx.iter_modules(model):
        if isinstance(module, RowParallelLinear):
            assert module.distributed_matmul is None


def test_model_logits_parity_wired_vs_unwired():
    import easydel as ed
    import spectrax as spx

    config = ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        sharding_axis_dims=(1, 1, 1, 1, _TP, 1),
    )
    model = ed.LlamaForCausalLM(config=config, rngs=spx.Rngs(0), param_dtype=jnp.float32, dtype=jnp.float32)
    input_ids = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 0, 128)

    baseline = model(input_ids=input_ids).logits

    model.wire_distributed_matmul()
    wired = model(input_ids=input_ids).logits

    assert wired.shape == baseline.shape
    assert jnp.allclose(wired, baseline, rtol=1e-4, atol=1e-4)


def test_engine_keeps_fsdp_batch_sharded_and_matches_einsum():
    import easydel as ed

    if len(jax.devices()) < 8:
        pytest.skip("fsdp x tp layout test needs 8 devices")

    config = ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        sharding_axis_dims=(1, 1, 2, 1, 4, 1),
    )
    mesh = config.mesh
    engine = make_distributed_matmul("row", "auto", config=config)

    x = jax.device_put(
        jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64), jnp.float32),
        NamedSharding(mesh, PartitionSpec(("dp", "fsdp"), "sp", "tp")),
    )
    w = jax.device_put(
        jax.random.normal(jax.random.PRNGKey(1), (64, 32), jnp.float32),
        NamedSharding(mesh, PartitionSpec("tp", None)),
    )
    with mesh:
        got = engine(x, w)
    assert got is not None
    expected = jnp.einsum("...i,io->...o", x, w)
    assert got.shape == expected.shape
    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)
