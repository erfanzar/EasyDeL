import jax
import jax.numpy as jnp
import numpy as np
from eformer.common_types import ColumnWise, Replicated, RowWise
from flax import nnx as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from easydel.layers.components.linears._linear_quantized import (
    ColumnParallelLinearQuantized,
    RowParallelLinearQuantized,
)
from easydel.layers.components.quants._configs import QuantizationConfig, QuantizationType


def test_row_parallel_affine_quantized_craft_sharding():
    layer = RowParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.INT8, group_size=64),
        use_bias=True,
        rngs=nn.Rngs(0),
    )

    specs = layer.craft_sharding()
    assert specs["quant_kernel"] is RowWise
    assert specs["quant_scales"] is RowWise
    assert specs["quant_biases"] is RowWise
    assert specs["bias"] is Replicated


def test_column_parallel_nf4_quantized_craft_sharding():
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=True,
        rngs=nn.Rngs(0),
    )

    specs = layer.craft_sharding()
    assert specs["quant_kernel"] is ColumnWise
    assert specs["quant_scales"] is ColumnWise
    assert "quant_biases" not in specs
    assert specs["bias"] is Replicated


def test_quantized_linear_uses_shard_map_for_sharded_params(monkeypatch):
    import easydel.layers.components.linears._linear_quantized as linear_quantized_mod

    calls = {"count": 0}
    original_shard_map = linear_quantized_mod.shard_map

    def _spy_shard_map(*args, **kwargs):
        calls["count"] += 1
        return original_shard_map(*args, **kwargs)

    monkeypatch.setattr(linear_quantized_mod, "shard_map", _spy_shard_map)

    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        rngs=nn.Rngs(0),
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    layer.quant_kernel.value = jax.device_put(
        layer.quant_kernel.value,
        NamedSharding(mesh, PartitionSpec(None, "tp")),
    )
    layer.quant_scales.value = jax.device_put(
        layer.quant_scales.value,
        NamedSharding(mesh, PartitionSpec(None, "tp")),
    )

    x = jax.device_put(
        jnp.ones((2, 128), dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec(None, None)),
    )
    y = layer(x)

    assert y.shape == (2, 256)
    assert calls["count"] >= 1


def test_quantized_linear_falls_back_when_k_axis_uses_non_tp_sharding(monkeypatch):
    import easydel.layers.components.linears._linear_quantized as linear_quantized_mod

    calls = {"count": 0}
    original_shard_map = linear_quantized_mod.shard_map

    def _spy_shard_map(*args, **kwargs):
        calls["count"] += 1
        return original_shard_map(*args, **kwargs)

    monkeypatch.setattr(linear_quantized_mod, "shard_map", _spy_shard_map)

    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        rngs=nn.Rngs(0),
    )

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tp"))
    layer.quant_kernel.value = jax.device_put(
        layer.quant_kernel.value,
        NamedSharding(mesh, PartitionSpec("fsdp", "tp")),
    )
    layer.quant_scales.value = jax.device_put(
        layer.quant_scales.value,
        NamedSharding(mesh, PartitionSpec("fsdp", "tp")),
    )

    x = jax.device_put(
        jnp.ones((2, 128), dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec(None, None)),
    )
    y = layer(x)

    assert y.shape == (2, 256)
    assert calls["count"] == 0


def test_quantized_linear_falls_back_when_aux_params_not_co_sharded(monkeypatch):
    import easydel.layers.components.linears._linear_quantized as linear_quantized_mod

    calls = {"count": 0}
    original_shard_map = linear_quantized_mod.shard_map

    def _spy_shard_map(*args, **kwargs):
        calls["count"] += 1
        return original_shard_map(*args, **kwargs)

    monkeypatch.setattr(linear_quantized_mod, "shard_map", _spy_shard_map)

    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        rngs=nn.Rngs(0),
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    layer.quant_kernel.value = jax.device_put(
        layer.quant_kernel.value,
        NamedSharding(mesh, PartitionSpec(None, "tp")),
    )
    # Simulate older checkpoints with replicated scales next to sharded kernels.
    layer.quant_scales.value = jax.device_put(
        layer.quant_scales.value,
        NamedSharding(mesh, PartitionSpec(None, None)),
    )

    x = jax.device_put(
        jnp.ones((2, 128), dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec(None, None)),
    )
    y = layer(x)

    assert y.shape == (2, 256)
    assert calls["count"] == 0


def test_row_quantized_linear_falls_back_when_k_axis_uses_non_tp_sharding(monkeypatch):
    import easydel.layers.components.linears._linear_quantized as linear_quantized_mod

    calls = {"count": 0}
    original_shard_map = linear_quantized_mod.shard_map

    def _spy_shard_map(*args, **kwargs):
        calls["count"] += 1
        return original_shard_map(*args, **kwargs)

    monkeypatch.setattr(linear_quantized_mod, "shard_map", _spy_shard_map)

    layer = RowParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        rngs=nn.Rngs(0),
    )

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tp"))
    layer.quant_kernel.value = jax.device_put(
        layer.quant_kernel.value,
        NamedSharding(mesh, PartitionSpec(("fsdp", "tp"), None)),
    )
    layer.quant_scales.value = jax.device_put(
        layer.quant_scales.value,
        NamedSharding(mesh, PartitionSpec(("fsdp", "tp"), None)),
    )

    x = jax.device_put(
        jnp.ones((2, 128), dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec(None, None)),
    )
    y = layer(x)

    assert y.shape == (2, 256)
    assert calls["count"] == 0


def test_row_quantized_linear_falls_back_when_tp_shards_batch_axis(monkeypatch):
    import easydel.layers.components.linears._linear_quantized as linear_quantized_mod

    calls = {"count": 0}
    original_shard_map = linear_quantized_mod.shard_map

    def _spy_shard_map(*args, **kwargs):
        calls["count"] += 1
        return original_shard_map(*args, **kwargs)

    monkeypatch.setattr(linear_quantized_mod, "shard_map", _spy_shard_map)
    monkeypatch.setattr(linear_quantized_mod, "_extract_tp_axis_name", lambda *_args, **_kwargs: "tp")

    layer = RowParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        rngs=nn.Rngs(0),
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    layer.quant_kernel.value = jax.device_put(
        layer.quant_kernel.value,
        NamedSharding(mesh, PartitionSpec("tp", None)),
    )
    layer.quant_scales.value = jax.device_put(
        layer.quant_scales.value,
        NamedSharding(mesh, PartitionSpec("tp", None)),
    )

    x = jax.device_put(
        jnp.ones((2, 128), dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec("tp", None)),
    )
    y = layer(x)

    assert y.shape == (2, 256)
    assert calls["count"] == 0
