import jax
import jax.numpy as jnp
import numpy as np
from eformer.common_types import ColumnWise, Replicated, RowWise
from flax import nnx as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from easydel.layers.linears._linear import ColumnParallelLinear
from easydel.layers.linears._linear_quantized import (
    ColumnParallelLinearQuantized,
    RowParallelLinearQuantized,
)
from easydel.layers.quantization._configs import QuantizationConfig, QuantizationType
from easydel.layers.quantization._quants import EasyQuantizer


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
    import easydel.layers.linears._linear_quantized as linear_quantized_mod

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
    import easydel.layers.linears._linear_quantized as linear_quantized_mod

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
    import easydel.layers.linears._linear_quantized as linear_quantized_mod

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
    import easydel.layers.linears._linear_quantized as linear_quantized_mod

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
    import easydel.layers.linears._linear_quantized as linear_quantized_mod

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


def test_quantized_linear_tpu_forces_shard_map_without_named_sharding(monkeypatch):
    from types import SimpleNamespace

    import easydel.layers.linears._linear_quantized as linear_quantized_mod

    shard_map_calls = {"count": 0}
    chosen_platforms: list[str | None] = []
    chosen_tpu_paths: list[str | None] = []

    fake_mesh = SimpleNamespace(axis_names=("tp",), shape={"tp": 4})

    def _fake_shard_map(func=None, *, mesh=None, in_specs=None, out_specs=None, check_vma=False):
        if func is None:

            def _decorator(real_func):
                return _fake_shard_map(
                    real_func,
                    mesh=mesh,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    check_vma=check_vma,
                )

            return _decorator
        shard_map_calls["count"] += 1
        return func

    def _spy_qmm(
        x,
        w,
        scales,
        zeros,
        *,
        transpose=False,
        group_size=None,
        bits=None,
        mode="affine",
        platform=None,
        tpu_path=None,
        **kwargs,
    ):
        del w, scales, zeros, transpose, group_size, bits, mode, kwargs
        chosen_platforms.append(platform)
        chosen_tpu_paths.append(tpu_path)
        return jnp.zeros((x.shape[0], 256), dtype=x.dtype)

    monkeypatch.setattr(linear_quantized_mod, "shard_map", _fake_shard_map)
    monkeypatch.setattr(linear_quantized_mod, "ej_quantized_matmul", _spy_qmm)
    monkeypatch.setattr(linear_quantized_mod, "_pick_mesh_from_arrays", lambda *_args, **_kwargs: fake_mesh)
    monkeypatch.setattr(linear_quantized_mod, "_spec_for_mesh", lambda *_args, **_kwargs: PartitionSpec())
    monkeypatch.setattr(linear_quantized_mod.jax, "default_backend", lambda: "tpu")

    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        qmm_fuse=True,
        rngs=nn.Rngs(0),
    )

    mode, group_size, bits, _ = layer._resolve_ejkernel_params()
    y = layer._distributed_quantized_matmul(
        jnp.ones((2, 128), dtype=jnp.float32),
        layer.quant_kernel.value,
        layer.quant_scales.value,
        layer.quant_biases.value,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )

    assert y.shape == (2, 256)
    assert shard_map_calls["count"] >= 1
    assert chosen_platforms == ["pallas"]
    assert chosen_tpu_paths == ["predecode"]


def test_quantized_linear_tpu_default_qmm_kwargs_use_xla_unfused():
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.AFFINE, bits=4, group_size=64),
        use_bias=False,
        rngs=nn.Rngs(0),
    )

    kwargs = layer._qmm_runtime_kwargs("tpu", quant_mode="affine")
    assert kwargs["fuse"] is False
    assert kwargs["platform"] == "xla"
    assert kwargs["use_best_config"] is True
    assert "tpu_path" not in kwargs


def test_quantized_linear_tpu_auto_low_m_uses_xla_unfused():
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.AFFINE, bits=4, group_size=64),
        use_bias=False,
        qmm_platform="auto",
        qmm_tpu_auto_xla_max_m=1024,
        rngs=nn.Rngs(0),
    )

    kwargs = layer._qmm_runtime_kwargs("tpu", m_tokens=4, quant_mode="affine")
    assert kwargs["fuse"] is False
    assert kwargs["platform"] == "xla"
    assert kwargs["use_best_config"] is True
    assert "tpu_path" not in kwargs


def test_quantized_linear_tpu_auto_low_m_non_affine_uses_fused_path():
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        qmm_platform="auto",
        qmm_tpu_auto_xla_max_m=1024,
        rngs=nn.Rngs(0),
    )

    kwargs = layer._qmm_runtime_kwargs("tpu", m_tokens=4, quant_mode="nf4")
    assert kwargs["fuse"] is True
    assert kwargs["platform"] == "auto"
    assert kwargs["use_best_config"] is True
    assert kwargs["tpu_path"] == "predecode"


def test_quantized_linear_tpu_auto_high_m_uses_fused_path():
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        qmm_platform="auto",
        qmm_tpu_auto_xla_max_m=1024,
        rngs=nn.Rngs(0),
    )

    kwargs = layer._qmm_runtime_kwargs("tpu", m_tokens=2048, quant_mode="nf4")
    assert kwargs["fuse"] is True
    assert kwargs["platform"] == "auto"
    assert kwargs["use_best_config"] is True
    assert kwargs["allow_dense_fallback"] is True
    assert kwargs["strict_fuse"] is False
    assert kwargs["tpu_path"] == "predecode"


def test_quantized_linear_tpu_custom_policy_table_is_used():
    custom_policy = {
        "tpu": {
            "non_affine": {
                "small": {"fuse": False, "platform": "xla"},
                "default": {"fuse": False, "platform": "xla"},
            },
            "default": {"default": {"fuse": False, "platform": "xla"}},
        }
    }
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        qmm_platform="auto",
        qmm_policy_table=custom_policy,
        rngs=nn.Rngs(0),
    )

    kwargs = layer._qmm_runtime_kwargs("tpu", m_tokens=4, quant_mode="nf4")
    assert kwargs["fuse"] is False
    assert kwargs["platform"] == "xla"
    assert kwargs["use_best_config"] is True
    assert "tpu_path" not in kwargs


def test_quantized_linear_tpu_fused_qmm_kwargs_use_predecode():
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        qmm_fuse=True,
        rngs=nn.Rngs(0),
    )

    kwargs = layer._qmm_runtime_kwargs("tpu", quant_mode="nf4")
    assert kwargs["fuse"] is True
    assert kwargs["platform"] == "pallas"
    assert kwargs["use_best_config"] is True
    assert kwargs["allow_dense_fallback"] is True
    assert kwargs["strict_fuse"] is False
    assert kwargs["tpu_path"] == "predecode"


def test_quantized_linear_tpu_path_requires_fused_qmm():
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        qmm_fuse=False,
        qmm_tpu_path="predecode",
        rngs=nn.Rngs(0),
    )

    try:
        layer._qmm_runtime_kwargs("tpu")
    except ValueError as exc:
        assert "qmm_tpu_path requires qmm_fuse=True." in str(exc)
    else:
        raise AssertionError("Expected ValueError when qmm_tpu_path is set with qmm_fuse=False.")


def test_quantized_linear_qmm_use_best_config_can_be_disabled():
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=False,
        qmm_use_best_config=False,
        rngs=nn.Rngs(0),
    )

    kwargs = layer._qmm_runtime_kwargs("tpu", quant_mode="nf4")
    assert kwargs["use_best_config"] is False


class _DummyModelConfig:
    def __init__(self, *, use_best: bool, platform: str | None, tpu_path: str | None):
        self.use_qmm_best_config = use_best
        self.qmm_platform_override = platform
        self.qmm_tpu_path_override = tpu_path
        self.quantization_config = None


class _DummyLinearModel(nn.Module):
    def __init__(self, config: _DummyModelConfig):
        self.config = config
        self.linear = ColumnParallelLinear(
            in_features=16,
            out_features=16,
            use_bias=False,
            rngs=nn.Rngs(0),
        )


def test_quantizer_applies_model_qmm_defaults_to_quantized_linears():
    model = _DummyLinearModel(_DummyModelConfig(use_best=False, platform="xla", tpu_path="hybrid"))
    quantizer = EasyQuantizer(QuantizationConfig(dtype=QuantizationType.NF4, group_size=16))

    quantizer.apply_quantization(model)

    assert isinstance(model.linear, ColumnParallelLinearQuantized)
    assert model.linear.qmm_use_best_config is False
    assert model.linear.qmm_platform == "xla"
    assert model.linear.qmm_tpu_path == "hybrid"


def test_quantizer_explicit_qmm_overrides_take_precedence_over_model_defaults():
    model = _DummyLinearModel(_DummyModelConfig(use_best=True, platform="pallas", tpu_path="predecode"))
    quantizer = EasyQuantizer(QuantizationConfig(dtype=QuantizationType.NF4, group_size=16))

    quantizer.apply_quantization(
        model,
        qmm_use_best_config=False,
        qmm_platform="xla",
        qmm_tpu_path="packed",
        qmm_strict_fuse=True,
        qmm_allow_dense_fallback=False,
    )

    assert isinstance(model.linear, ColumnParallelLinearQuantized)
    assert model.linear.qmm_use_best_config is False
    assert model.linear.qmm_platform == "xla"
    assert model.linear.qmm_tpu_path == "packed"
    assert model.linear.qmm_strict_fuse is True
    assert model.linear.qmm_allow_dense_fallback is False


def test_quantized_linear_tpu_forced_layout_keeps_vector_bias_specs_rank_1():
    from types import SimpleNamespace

    fake_mesh = SimpleNamespace(axis_names=("tp",), shape={"tp": 4})
    layer = ColumnParallelLinearQuantized(
        in_features=128,
        out_features=256,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=64),
        use_bias=True,
        rngs=nn.Rngs(0),
    )

    resolved = layer._resolve_shard_specs(
        fake_mesh,
        jnp.ones((2, 128), dtype=jnp.float32),
        layer.quant_kernel.value,
        layer.quant_scales.value,
        jnp.ones((256,), dtype=jnp.float32),
        True,
    )

    assert resolved is not None
    _input_spec, _kernel_spec, _scale_spec, bias_spec, _output_spec, _tp_axis_name = resolved
    assert bias_spec == PartitionSpec("tp")


class _TypeErrorOnOverridesLinear(nn.Module):
    def to_quantized(self, config: QuantizationConfig, **kwargs):
        del config
        if kwargs:
            raise TypeError("internal to_quantized failure")
        return self


class _DummyTypeErrorModel(nn.Module):
    def __init__(self):
        self.config = _DummyModelConfig(use_best=True, platform="xla", tpu_path="predecode")
        self.linear = _TypeErrorOnOverridesLinear()


def test_quantizer_propagates_internal_typeerror_from_to_quantized():
    model = _DummyTypeErrorModel()
    quantizer = EasyQuantizer(QuantizationConfig(dtype=QuantizationType.NF4, group_size=16))

    try:
        quantizer.apply_quantization(model, quantization_pattern=r".*linear.*")
    except TypeError as exc:
        assert "internal to_quantized failure" in str(exc)
    else:
        raise AssertionError("Expected internal TypeError from to_quantized to propagate.")
