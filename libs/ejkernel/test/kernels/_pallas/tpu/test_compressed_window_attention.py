# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Pallas-TPU parity tests for compressed-window full-sequence attention.

Compares the tiled-flash Pallas TPU kernel (``make_async_copy`` KV streaming +
online softmax) against the XLA reference over DeepSeek-V4 training / prefill
shapes: real head_dim 512, multi-tile KV axes (sliding-window token KVs plus
HCA/CSA compressed entries), sinks on and off, and gradient parity (the
training contract). Skipped when no TPU is present.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_NEG = float(np.finfo(np.float32).min)


def _has_tpu():
    """Return True when a TPU backend is available."""
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require a TPU backend")


def _inputs(batch, heads, q_len, head_dim, kv_len, dtype, use_sinks, seed=0):
    """Build random prefill inputs with ~30% of KV positions masked out."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), dtype)
    kv = jnp.asarray(rng.standard_normal((batch, kv_len, head_dim)), dtype)
    bias = np.asarray(rng.standard_normal((batch, q_len, kv_len)) * 0.5, np.float32)
    bias[rng.random((batch, q_len, kv_len)) < 0.3] = _NEG
    bias = jnp.asarray(bias)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32) if use_sinks else None
    return q, kv, bias, sinks


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "head_dim", "kv_len"),
    [
        (2, 8, 128, 512, 128),  # sliding-only prefill (single kv tile)
        (2, 8, 256, 512, 260),  # window + few HCA entries, multi q-block + multi kv-tile
        (1, 16, 512, 512, 640),  # CSA-like: window + many entries, several kv tiles
        (4, 4, 64, 512, 132),  # per-device TP shard (heads/4), padded kv tile
    ],
)
@pytest.mark.parametrize("use_sinks", [True, False])
def test_pallas_matches_xla_fwd(batch, heads, q_len, head_dim, kv_len, use_sinks):
    """Pallas forward matches the XLA reference on fp32 prefill shapes."""
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu as pallas_impl,
    )
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention as xla_impl

    q, kv, bias, sinks = _inputs(batch, heads, q_len, head_dim, kv_len, jnp.float32, use_sinks)
    scale = head_dim**-0.5
    out_xla = np.asarray(xla_impl(q, kv, bias, sinks, softmax_scale=scale))
    out_tpu = np.asarray(pallas_impl(q, kv, bias, sinks, softmax_scale=scale))
    assert out_tpu.shape == out_xla.shape
    assert np.isfinite(out_tpu).all()
    assert np.abs(out_tpu - out_xla).max() < 2e-4, np.abs(out_tpu - out_xla).max()


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "head_dim", "kv_len"),
    [(2, 8, 256, 512, 260), (1, 16, 512, 512, 640)],
)
def test_pallas_gradient_matches_xla(batch, heads, q_len, head_dim, kv_len):
    """Pallas ``jax.grad`` (custom-JVP transpose) matches XLA-reference grad (fp32 tight)."""
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu as pallas_impl,
    )
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention as xla_impl

    q, kv, bias, sinks = _inputs(batch, heads, q_len, head_dim, kv_len, jnp.float32, use_sinks=True, seed=1)
    scale = head_dim**-0.5
    rng = np.random.default_rng(2)
    cot = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), jnp.float32)

    def _xla(qq, kk, ss):
        return jnp.sum(xla_impl(qq, kk, bias, ss, softmax_scale=scale) * cot)

    def _pal(qq, kk, ss):
        return jnp.sum(pallas_impl(qq, kk, bias, ss, softmax_scale=scale) * cot)

    gx = jax.grad(_xla, argnums=(0, 1, 2))(q, kv, sinks)
    gp = jax.grad(_pal, argnums=(0, 1, 2))(q, kv, sinks)
    for a, b in zip(gx, gp, strict=True):
        assert jnp.isfinite(b).all()
        assert float(jnp.max(jnp.abs(a - b))) < 2e-4, float(jnp.max(jnp.abs(a - b)))


def _sliding_inputs(batch, heads, s, head_dim, window, n_entries, dtype, use_sinks, seed=0):
    """Sliding-window token region (length ``s``) + ``n_entries`` compressed entries."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((batch, heads, s, head_dim)), dtype)
    tok = rng.standard_normal((batch, s, head_dim))
    ent = rng.standard_normal((batch, n_entries, head_dim)) if n_entries else np.zeros((batch, 0, head_dim))
    kv = jnp.asarray(np.concatenate([tok, ent], axis=1), dtype)
    qi = np.arange(s)[:, None]
    ki = np.arange(s)[None, :]
    vis = (ki <= qi) & (ki > qi - window)
    tb = np.where(vis, 0.0, _NEG).astype(np.float32)[None].repeat(batch, 0)
    if n_entries:
        eb = (rng.standard_normal((batch, s, n_entries)) * 0.3).astype(np.float32)
        eb[rng.random((batch, s, n_entries)) < 0.4] = _NEG
        bias = jnp.asarray(np.concatenate([tb, eb], axis=2))
    else:
        bias = jnp.asarray(tb)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32) if use_sinks else None
    return q, kv, bias, sinks


@pytest.mark.parametrize(
    ("batch", "heads", "s", "head_dim", "window", "n_entries"),
    [(1, 8, 1024, 512, 128, 0), (2, 8, 512, 512, 128, 16), (1, 16, 2048, 512, 128, 32)],
)
@pytest.mark.parametrize("use_sinks", [True, False])
def test_pallas_band_skip_matches_xla_fwd(batch, heads, s, head_dim, window, n_entries, use_sinks):
    """Band-skipped forward equals the full dense XLA reference (skipped tiles are masked)."""
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu as pallas_impl,
    )
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention as xla_impl

    q, kv, bias, sinks = _sliding_inputs(batch, heads, s, head_dim, window, n_entries, jnp.float32, use_sinks)
    scale = head_dim**-0.5
    out_xla = np.asarray(xla_impl(q, kv, bias, sinks, softmax_scale=scale))
    out_band = np.asarray(pallas_impl(q, kv, bias, sinks, softmax_scale=scale, window=window, token_kv_len=s))
    assert np.isfinite(out_band).all()
    assert np.abs(out_band - out_xla).max() < 2e-4, np.abs(out_band - out_xla).max()


@pytest.mark.parametrize(("s", "n_entries"), [(512, 16), (1024, 0)])
def test_pallas_band_skip_gradient_matches_xla(s, n_entries):
    """Band-skipped forward + windowed backward matches the dense XLA gradient (fp32 tight)."""
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu as pallas_impl,
    )
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention as xla_impl

    batch, heads, head_dim, window = 2, 8, 512, 128
    q, kv, bias, sinks = _sliding_inputs(batch, heads, s, head_dim, window, n_entries, jnp.float32, True, seed=1)
    scale = head_dim**-0.5
    rng = np.random.default_rng(2)
    cot = jnp.asarray(rng.standard_normal((batch, heads, s, head_dim)), jnp.float32)

    def _xla(qq, kk, bb, ss):
        return jnp.sum(xla_impl(qq, kk, bb, ss, softmax_scale=scale) * cot)

    def _band(qq, kk, bb, ss):
        return jnp.sum(pallas_impl(qq, kk, bb, ss, softmax_scale=scale, window=window, token_kv_len=s) * cot)

    gx = jax.grad(_xla, argnums=(0, 1, 2, 3))(q, kv, bias, sinks)
    gb = jax.grad(_band, argnums=(0, 1, 2, 3))(q, kv, bias, sinks)
    for a, b in zip(gx, gb, strict=True):
        assert jnp.isfinite(b).all()
        assert float(jnp.max(jnp.abs(a - b))) < 2e-4, float(jnp.max(jnp.abs(a - b)))


def test_pallas_matches_xla_bf16():
    """bf16 Pallas output matches the bf16 XLA reference within the MXU floor."""
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu as pallas_impl,
    )
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention as xla_impl

    q, kv, bias, sinks = _inputs(2, 8, 256, 512, 260, jnp.bfloat16, use_sinks=True)
    scale = 512**-0.5
    out_xla = np.asarray(xla_impl(q, kv, bias, sinks, softmax_scale=scale).astype(jnp.float32))
    out_tpu = np.asarray(pallas_impl(q, kv, bias, sinks, softmax_scale=scale).astype(jnp.float32))
    assert np.isfinite(out_tpu).all()
    assert np.abs(out_tpu - out_xla).max() < 5e-2, np.abs(out_tpu - out_xla).max()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_pallas_forward_mode_jvp_matches_xla():
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu as pallas_impl,
    )
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention as xla_impl

    q, kv, bias, sinks = _inputs(1, 8, 128, 128, 140, jnp.float32, use_sinks=True, seed=41)
    tangents = tuple(jnp.ones_like(x) * 0.01 for x in (q, kv, bias, sinks))
    scale = 128**-0.5
    _, p_tan = jax.jvp(lambda a, b, c, d: pallas_impl(a, b, c, d, softmax_scale=scale), (q, kv, bias, sinks), tangents)
    _, x_tan = jax.jvp(lambda a, b, c, d: xla_impl(a, b, c, d, softmax_scale=scale), (q, kv, bias, sinks), tangents)
    assert jnp.isfinite(p_tan).all()
    assert float(jnp.max(jnp.abs(p_tan - x_tan))) < 2e-4


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_tiled_training_ad_with_partial_query_tile(dtype, use_sinks):
    """Exercise both AD directions, storage dtypes, and optional sink gradients."""
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu as pallas_impl,
    )
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention as xla_impl

    q, kv, bias, _ = _inputs(1, 4, 131, 128, 140, dtype, use_sinks=False, seed=17)
    sinks = jnp.asarray([0.1, -0.2, 0.3, -0.4], jnp.float32)
    args = (q, kv, bias, sinks)
    rng = np.random.default_rng(18)
    tangents = tuple(jnp.asarray(rng.normal(size=x.shape) * 0.01, x.dtype) for x in args)

    def pallas(a, b, c, d):
        return pallas_impl(a, b, c, d if use_sinks else None, softmax_scale=128**-0.5)

    def reference(a, b, c, d):
        return xla_impl(a, b, c, d if use_sinks else None, softmax_scale=128**-0.5)

    _, actual_tangent = jax.jvp(pallas, args, tangents)
    _, expected_tangent = jax.jvp(reference, args, tangents)
    tolerance = 0.03 if dtype == jnp.bfloat16 else 2e-4
    np.testing.assert_allclose(actual_tangent, expected_tangent, atol=tolerance, rtol=tolerance)
    cotangent = jnp.asarray(rng.normal(size=q.shape), dtype)
    actual_grads = jax.vjp(pallas, *args)[1](cotangent)
    expected_grads = jax.vjp(reference, *args)[1](cotangent)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    if not use_sinks:
        np.testing.assert_array_equal(actual_grads[-1], jnp.zeros_like(sinks))


@pytest.mark.parametrize("layout", ["padded", "band-skipped"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("use_sinks", [False, True], ids=["no-sink", "sink"])
def test_pallas_fully_masked_rows_compiled_output_and_jvp(layout, dtype, use_sinks):
    """Finite MASK_MIN rows retain dense semantics through Mosaic lowering and AD."""
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu as pallas_impl,
    )
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention as xla_impl

    # The implementation chooses interpret mode from the default backend. Do not
    # silently turn this hardware regression into another CPU interpret test.
    assert jax.default_backend() == "tpu"
    if layout == "padded":
        q, kv, bias, _ = _inputs(1, 4, 131, 128, 140, dtype, False, seed=71)
        options = {}
    else:
        # Five token tiles exceed the three visited by each default q-block;
        # both the token region and the compressed-entry region need padding.
        q, kv, bias, _ = _sliding_inputs(1, 4, 515, 128, 32, 17, dtype, False, seed=71)
        options = {"window": 32, "token_kv_len": 515}
    # A nonzero, position-dependent value mean exposes both padding dilution
    # and averaging only the visited tiles instead of *all real* KVs.
    kv = kv + jnp.linspace(0.5, 2.0, kv.shape[1], dtype=dtype)[None, :, None]
    masked_rows = jnp.asarray([0, 127, 128, q.shape[2] - 1])
    bias = bias.at[:, masked_rows, :].set(_NEG)
    sinks = jnp.asarray([0.1, -0.2, 0.3, -0.4], jnp.float32)
    args = (q, kv, bias, sinks)
    rng = np.random.default_rng(72)
    tangents = tuple(jnp.asarray(rng.normal(size=x.shape) * 0.01, x.dtype) for x in args)

    def pallas(a, b, c, d):
        return pallas_impl(a, b, c, d if use_sinks else None, softmax_scale=128**-0.5, **options)

    def reference(a, b, c, d):
        return xla_impl(a, b, c, d if use_sinks else None, softmax_scale=128**-0.5)

    def pallas_jvp(primals, directions):
        return jax.jvp(pallas, primals, directions)

    # Keep the primal in the compiled result: tangent-only checks can eliminate
    # the Pallas call and miss a broken uniform_ref Mosaic lowering entirely.
    compiled = jax.jit(pallas_jvp).lower(args, tangents).compile()
    actual_output, actual_tangent = compiled(args, tangents)
    expected_output, expected_tangent = jax.jvp(reference, args, tangents)
    for actual, expected in ((actual_output, expected_output), (actual_tangent, expected_tangent)):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert np.isfinite(np.asarray(actual.astype(jnp.float32))).all()
    output_error = np.abs(
        np.asarray(actual_output.astype(jnp.float32)) - np.asarray(expected_output.astype(jnp.float32))
    )
    assert output_error.max() < (5e-2 if dtype == jnp.bfloat16 else 2e-4), output_error.max()
    tolerance = 0.03 if dtype == jnp.bfloat16 else 2e-4
    np.testing.assert_allclose(
        actual_tangent.astype(jnp.float32), expected_tangent.astype(jnp.float32), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("kernel", ["compressed_window_attention", "compressed_window_decode"])
def test_compiled_xla_bf16_jvp_matches_eager(kernel):
    """Mixed bf16 dots and fp32 masked softmax must survive compiled AD."""
    import importlib

    module = importlib.import_module(f"ejkernel.kernels._xla.{kernel}._xla_impl_fwd")
    operation = getattr(module, f"{kernel}_xla")
    q, kv, bias, sinks = _inputs(1, 4, 131, 128, 140, jnp.bfloat16, use_sinks=True, seed=17)
    args = (q, kv, bias, sinks)
    tangents = tuple(jnp.ones_like(x) * 0.01 for x in args)

    def derivative(*values):
        return jax.jvp(operation, values, tangents)[1]

    expected = derivative(*args)
    actual = jax.jit(derivative)(*args)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0.01)
