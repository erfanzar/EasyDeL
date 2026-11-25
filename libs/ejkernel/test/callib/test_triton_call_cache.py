import importlib
import types
from dataclasses import dataclass, field
from pathlib import Path

import ejkernel.callib as callib
import ejkernel.callib._triton_call as triton_call_mod
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core


def test_triton_call_disk_cache(monkeypatch, tmp_path):
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        pytest.skip("Triton cache test requires a GPU backend")

    monkeypatch.setenv("EJKERNEL_TRITON_CACHE_COMPILES", "1")
    monkeypatch.setenv("EJKERNEL_TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("EJKERNEL_TRITON_CACHE_MAX_ITEMS", "8")
    monkeypatch.setenv("EJKERNEL_TRITON_CACHE_VERBOSE", "0")

    import ejkernel.callib._triton_call as tc

    tc = importlib.reload(tc)
    if not tc.CAN_USE_TRITON:
        pytest.skip("Triton not installed")

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, n, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    calls = {"count": 0}
    orig_compile = tc.compile_ttir_inplace

    def wrapped_compile(*args, **kwargs):
        calls["count"] += 1
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(tc, "compile_ttir_inplace", wrapped_compile)

    n = 1024
    x = jnp.arange(n, dtype=jnp.float16)
    y = jnp.ones_like(x)

    out = tc.triton_call(
        x,
        y,
        n,
        kernel=add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=lambda meta: (triton.cdiv(n, meta["BLOCK"]),),
        BLOCK=128,
    )
    out = jax.block_until_ready(out)
    assert jnp.allclose(out, x + y)

    first_count = calls["count"]
    assert first_count >= 1

    tc._COMPILED_KERNEL_CACHE.clear()

    out2 = tc.triton_call(
        x,
        y,
        n,
        kernel=add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=lambda meta: (triton.cdiv(n, meta["BLOCK"]),),
        BLOCK=128,
    )
    out2 = jax.block_until_ready(out2)
    assert jnp.allclose(out2, x + y)
    assert calls["count"] == first_count


@pytest.mark.cpu_ok
def test_get_triton_type_bool_before_int_and_numpy_scalars():
    assert triton_call_mod.get_triton_type(True) == "B"
    assert triton_call_mod.get_triton_type(False) == "B"
    assert triton_call_mod.get_triton_type(np.bool_(True)) == "B"
    assert triton_call_mod.get_triton_type(np.bool_(False)) == "B"

    assert triton_call_mod.get_triton_type(0) == "i32"
    assert triton_call_mod.get_triton_type(1) == "i32"
    assert triton_call_mod.get_triton_type(np.int32(7)) == "i32"
    assert triton_call_mod.get_triton_type(np.uint32(2**31)) == "u32"
    assert triton_call_mod.get_triton_type(np.int64(2**40)) == "i64"
    assert triton_call_mod.get_triton_type(np.uint64(2**63)) == "u64"

    assert triton_call_mod.get_triton_type(1.0) == "fp32"
    assert triton_call_mod.get_triton_type(np.float32(1.0)) == "fp32"
    assert triton_call_mod.get_triton_type(np.float64(0.5)) == "fp32"
    assert triton_call_mod.get_triton_type(None) == "constexpr"
    assert triton_call_mod.get_triton_type(core.ShapedArray((4,), jnp.bfloat16)) == "*bf16"


@pytest.mark.cpu_ok
def test_constexpr_one_and_none_never_strip_bool_true_or_float_one():
    dummy_arr = object()
    args = (
        dummy_arr,
        True,
        1.0,
        1,
        None,
        np.bool_(True),
        np.float32(1.0),
        np.float64(1.0),
        np.int32(1),
        np.int64(42),
    )
    arg_names = [
        "x_ptr",
        "flag",
        "softmax_scale",
        "sentinel_one",
        "opt_bias",
        "np_flag",
        "np_scale32",
        "np_scale64",
        "np_one",
        "stride",
    ]

    array_args, scalar_args = triton_call_mod._partition_triton_args(args)
    assert array_args == [dummy_arr]
    assert scalar_args == [
        (1, "B", True),
        (2, "fp32", 1.0),
        (3, "i32", 1),
        (4, "constexpr", None),
        (5, "B", True),
        (6, "fp32", 1.0),
        (7, "fp32", 1.0),
        (8, "i32", 1),
        (9, "i32", 42),
    ]

    constants = triton_call_mod._extract_scalar_constexprs(arg_names, scalar_args)
    assert constants == {"sentinel_one": 1, "opt_bias": None, "np_one": 1}
    assert "flag" not in constants
    assert "softmax_scale" not in constants
    assert "np_flag" not in constants
    assert "np_scale32" not in constants
    assert "np_scale64" not in constants

    constexpr_indices = triton_call_mod._constexpr_scalar_indices(scalar_args)
    assert constexpr_indices == {3, 4, 8}
    assert 1 not in constexpr_indices
    assert 2 not in constexpr_indices
    assert 5 not in constexpr_indices
    assert 6 not in constexpr_indices
    assert 7 not in constexpr_indices


@pytest.mark.cpu_ok
def test_unwrap_triton_kernel_restores_autotuner_and_handles_triton3_prune(monkeypatch):
    @dataclass
    class FakeConfig:
        kwargs: dict = field(default_factory=dict)
        num_warps: int = 4
        num_stages: int = 2
        num_ctas: int = 1
        pre_hook: object = None

    class FakeAutotuner:
        def __init__(self, fn, configs, early_config_prune=None):
            self.fn = fn
            self.configs = configs
            self.early_config_prune = early_config_prune

        def prune_configs(self, kwargs):
            if self.early_config_prune is not None:
                return self.early_config_prune(self.configs, self.nargs, **kwargs)
            return list(self.configs)

    class FakeHeuristics:
        def __init__(self, fn, values):
            self.fn = fn
            self.values = values

    monkeypatch.setattr(
        triton_call_mod,
        "autotuner",
        types.SimpleNamespace(Autotuner=FakeAutotuner, Heuristics=FakeHeuristics),
    )

    base_fn = object()
    cfg64 = FakeConfig(kwargs={"BLOCK_M": 64})
    cfg128 = FakeConfig(kwargs={"BLOCK_M": 128})

    seen_triton3_kwargs: list[dict] = []

    def triton3_prune(configs, named_args, **kwargs):
        seen_triton3_kwargs.append({"named_args": dict(named_args), "kwargs": dict(kwargs)})
        return configs

    tuner3 = FakeAutotuner(base_fn, [cfg64, cfg128], early_config_prune=triton3_prune)
    assert not hasattr(tuner3, "nargs")

    fn1, res1 = triton_call_mod._unwrap_triton_kernel(
        tuner3,
        [],
        named_args={"x": "a1"},
        metaparams={"BLOCK_M": 64},
    )
    assert fn1 is base_fn
    assert res1 == [cfg64]
    assert tuner3.early_config_prune is triton3_prune
    assert not hasattr(tuner3, "nargs")

    # Second lowering with different metaparams must not bleed BLOCK_M=64 from the first call.
    fn2, res2 = triton_call_mod._unwrap_triton_kernel(
        tuner3,
        [],
        named_args={"x": "a2"},
        metaparams={"BLOCK_M": 128},
    )
    assert fn2 is base_fn
    assert res2 == [cfg128]
    assert tuner3.early_config_prune is triton3_prune
    assert not hasattr(tuner3, "nargs")
    assert seen_triton3_kwargs == [
        {"named_args": {"x": "a1"}, "kwargs": {"BLOCK_M": 64}},
        {"named_args": {"x": "a2"}, "kwargs": {"BLOCK_M": 128}},
    ]

    # Triton 2.x 2-argument early_config_prune signature (no **kwargs) still works.
    seen_triton2_calls = 0

    def triton2_prune(configs, named_args):
        nonlocal seen_triton2_calls
        del named_args
        seen_triton2_calls += 1
        return configs

    tuner2 = FakeAutotuner(base_fn, [cfg64, cfg128], early_config_prune=triton2_prune)
    tuner2.nargs = {"sentinel": True}
    _, res_t2 = triton_call_mod._unwrap_triton_kernel(
        tuner2,
        [],
        named_args={"x": "a3"},
        metaparams={"BLOCK_M": 64},
    )
    assert res_t2 == [cfg64]
    assert seen_triton2_calls == 1
    assert tuner2.early_config_prune is triton2_prune
    assert tuner2.nargs == {"sentinel": True}

    # Even when pre_hook triggers NotImplementedError, original attributes are restored.
    bad_tuner = FakeAutotuner(base_fn, [FakeConfig(pre_hook=lambda _: None)], early_config_prune=triton3_prune)
    with pytest.raises(NotImplementedError, match="pre_hook"):
        triton_call_mod._unwrap_triton_kernel(bad_tuner, [], named_args={"x": "bad"}, metaparams={})
    assert bad_tuner.early_config_prune is triton3_prune
    assert not hasattr(bad_tuner, "nargs")


@pytest.mark.cpu_ok
def test_assert_single_device_args_and_infer_device_index(monkeypatch):
    class FakeDevice:
        def __init__(self, idx: int, platform: str = "gpu"):
            self.id = idx
            self.local_hardware_id = idx
            self.platform = platform

    gpu0, gpu1 = FakeDevice(0), FakeDevice(1)
    monkeypatch.setattr(jax, "devices", lambda backend=None: [gpu0, gpu1])
    monkeypatch.setattr(jax, "local_devices", lambda backend=None: [gpu0, gpu1])

    class FakeArray:
        def __init__(self, devs: set[FakeDevice]):
            self._devs = devs

    class FakeTracer(core.Tracer):
        def __init__(self, devs: set[FakeDevice]):
            self._devs = devs

    monkeypatch.setattr(triton_call_mod, "_array_device_set", lambda arg: getattr(arg, "_devs", None))

    arr_gpu0 = FakeArray({gpu0})
    arr_gpu1_a = FakeArray({gpu1})
    arr_gpu1_b = FakeArray({gpu1})
    arr_sharded = FakeArray({gpu0, gpu1})
    tracer_sharded = FakeTracer({gpu0, gpu1})

    # All inputs on GPU 1 with device=None succeeds and infers device index 1.
    triton_call_mod._assert_single_device_args([arr_gpu1_a, arr_gpu1_b], None, allow_sharded_tracers=False)
    assert triton_call_mod._infer_device_index([arr_gpu1_a, arr_gpu1_b], None) == 1
    assert triton_call_mod._infer_device_index([arr_gpu1_a], 0) == 0
    assert triton_call_mod._infer_device_index([], None) == 0

    # Matching explicit device index succeeds; mismatched explicit device raises AssertionError.
    triton_call_mod._assert_single_device_args([arr_gpu1_a], 1, allow_sharded_tracers=False)
    with pytest.raises(AssertionError, match="different device than the requested"):
        triton_call_mod._assert_single_device_args([arr_gpu1_a], 0, allow_sharded_tracers=False)

    # Inputs split across GPU 0 and GPU 1 raise AssertionError.
    with pytest.raises(AssertionError, match="same device"):
        triton_call_mod._assert_single_device_args([arr_gpu0, arr_gpu1_a], None, allow_sharded_tracers=False)

    # Sharded concrete array raises AssertionError outside shard_map.
    with pytest.raises(AssertionError, match="sharded across 2 devices"):
        triton_call_mod._assert_single_device_args([arr_sharded], None, allow_sharded_tracers=False)

    # Sharded tracer is rejected outside shard_map, allowed inside shard_map.
    with pytest.raises(AssertionError, match="sharded across 2 devices"):
        triton_call_mod._assert_single_device_args([tracer_sharded], None, allow_sharded_tracers=False)
    triton_call_mod._assert_single_device_args([tracer_sharded], None, allow_sharded_tracers=True)


@pytest.mark.cpu_ok
def test_rocm_hsaco_materialization_and_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(triton_call_mod, "TRITON_CACHE_DIR", tmp_path / "disk_cache")
    monkeypatch.setattr(triton_call_mod, "TRITON_CACHE_ENABLED", True)

    payload = b"\x7fELF_FAKE_HSACO_PAYLOAD"
    res = triton_call_mod.CompilationResult(
        binary=payload,
        name="fake_rocm_kernel",
        shared_mem_bytes=2048,
        cluster_dims=(0, 0, 0),
        ttgir=None,
        llir=None,
    )
    triton_call_mod._save_triton_kernel_cache(
        "rocm_key_1",
        res,
        platform="rocm",
        compute_capability=942,
        ttir=None,
    )
    loaded = triton_call_mod._load_triton_kernel_cache(
        "rocm_key_1",
        platform="rocm",
        compute_capability=942,
    )
    assert loaded is not None
    assert loaded.binary == payload
    assert isinstance(loaded.binary, bytes)

    path1 = triton_call_mod._materialize_rocm_hsaco_path("rocm_key_1", loaded.binary)
    path2 = triton_call_mod._materialize_rocm_hsaco_path("rocm_key_1", loaded.binary)
    assert path1 == path2
    assert Path(path1).read_bytes() == payload

    # Non-cached path deduplicates inside a single temp directory rather than leaking mkstemp files.
    monkeypatch.setattr(triton_call_mod, "TRITON_CACHE_ENABLED", False)
    monkeypatch.setattr(triton_call_mod, "_ROCM_HSACO_TEMP_DIR", tmp_path / "rocm_tmp")
    (tmp_path / "rocm_tmp").mkdir()
    tmp_path1 = triton_call_mod._materialize_rocm_hsaco_path("rocm_key_2", payload)
    tmp_path2 = triton_call_mod._materialize_rocm_hsaco_path("rocm_key_2", payload)
    assert tmp_path1 == tmp_path2
    assert list((tmp_path / "rocm_tmp").iterdir()) == [Path(tmp_path1)]


@pytest.mark.cpu_ok
def test_error_messages_reference_ejkernel_and_valid_extras():
    with pytest.raises(NotImplementedError, match=r"ejkernel\.callib\.triton_call"):
        triton_call_mod.triton_kernel_call_raise_on_jvp()
    with pytest.raises(NotImplementedError, match=r"ejkernel\.callib\.triton_call"):
        triton_call_mod.triton_kernel_call_raise_on_vmap()

    with pytest.raises(ValueError) as exc_triton:
        callib._raise_triton_unavailable(ImportError("no triton"))
    assert "ejkernel[gpu]" not in str(exc_triton.value)
    assert "ejkernel[cuda]" in str(exc_triton.value)
    assert "ejkernel[triton]" in str(exc_triton.value)

    with pytest.raises(ValueError) as exc_cute:
        callib._raise_cute_unavailable(ImportError("no cute"))
    assert "ejkernel[gpu]" not in str(exc_cute.value)
    assert "ejkernel[cuda]" in str(exc_cute.value)
