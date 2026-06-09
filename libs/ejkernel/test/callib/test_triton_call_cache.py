import importlib

import jax
import jax.numpy as jnp
import pytest


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
