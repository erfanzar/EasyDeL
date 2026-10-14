# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Shared normalisation primitives: ``rms_norm``, ``l2_norm``, and RMSNorm flags.

Ten families wrote their own norm because the shared :class:`RMSNorm` hardcoded
``weight * output`` — no ``(1 + weight)``, no scale-free mode — and the useful
free functions in ``_norms.py`` were never exported.

A caution encoded by these tests: the copies do **not** all agree. Two distinct
float orderings are in the wild, and they differ by ~1.6e-2 in bfloat16:

- cast the fp32 scale back to ``x.dtype`` and multiply there (HF's ordering,
  what ``deepseek_v4`` uses, and what :func:`rms_norm` implements);
- promote ``x`` to fp32, multiply, then cast the product (``muse_glimmer``,
  ``llama4``).

They agree exactly in float32 and diverge in bfloat16, so a family may only be
migrated onto the shared helper once its ordering has been checked. These tests
pin the ordering so it cannot drift silently.
"""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.norms import RMSNorm, l2_norm, rms_norm
from jax import numpy as jnp

DTYPES = [jnp.float32, jnp.bfloat16]


def _x(dtype, shape=(8, 64), seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_uses_the_cast_down_ordering(dtype):
    """``rms_norm`` must reduce in fp32 then cast the scale before multiplying."""
    x = _x(dtype)
    eps = 1e-6
    expected = x * jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), -1, keepdims=True) + eps).astype(x.dtype)
    assert jnp.array_equal(rms_norm(x, eps), expected)


def test_the_two_orderings_really_do_diverge_in_bfloat16():
    """Guards the reason families cannot be migrated unchecked.

    If this ever starts passing as 'equal', the migration constraint documented
    above has gone away and the remaining copies can be collapsed.
    """
    x = _x(jnp.bfloat16)
    eps = 1e-6
    cast_down = rms_norm(x, eps)
    x32 = x.astype(jnp.float32)
    promote = (x32 * jax.lax.rsqrt(jnp.square(x32).mean(-1, keepdims=True) + eps)).astype(x.dtype)

    assert not jnp.array_equal(cast_down, promote)
    assert jnp.allclose(cast_down.astype(jnp.float32), promote.astype(jnp.float32), atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_l2_norm_uses_a_sum_not_a_mean(dtype):
    """``l2_norm`` is the sum-based variant; conflating the two is a real bug."""
    x = _x(dtype)
    eps = 1e-6
    expected = x * jax.lax.rsqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), -1, keepdims=True) + eps).astype(x.dtype)
    assert jnp.array_equal(l2_norm(x, eps), expected)
    assert not jnp.array_equal(l2_norm(x, eps), rms_norm(x, eps))


def test_with_scale_false_allocates_no_parameter():
    """A scale-free norm must not add a phantom leaf to the checkpoint."""
    norm = RMSNorm(dim=16, dtype=jnp.float32, param_dtype=jnp.float32, with_scale=False, rngs=spx.Rngs(0))
    assert norm.weight is None


def test_with_scale_false_matches_the_free_function():
    """The module form and the functional form must agree."""
    norm = RMSNorm(dim=64, eps=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, with_scale=False, rngs=spx.Rngs(0))
    x = _x(jnp.float32)
    assert jnp.allclose(norm(x), rms_norm(x, 1e-6), atol=1e-6)


def test_scale_offset_applies_one_plus_weight():
    """Gemma-family convention: the stored weight is consumed as ``1 + w``."""
    plain = RMSNorm(dim=64, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    offset = RMSNorm(dim=64, dtype=jnp.float32, param_dtype=jnp.float32, scale_offset=1.0, rngs=spx.Rngs(0))
    offset.weight = plain.weight

    x = _x(jnp.float32)
    base = plain._norm(x.astype(jnp.float32))
    expected = (1.0 + plain.weight.astype(jnp.float32)) * base

    assert jnp.allclose(offset(x), expected, atol=1e-6)
    assert not jnp.allclose(offset(x), plain(x), atol=1e-6)


def test_scale_offset_zero_is_the_default_behaviour():
    """The new flag must not perturb existing users."""
    a = RMSNorm(dim=32, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    b = RMSNorm(dim=32, dtype=jnp.float32, param_dtype=jnp.float32, scale_offset=0.0, rngs=spx.Rngs(0))
    b.weight = a.weight
    x = _x(jnp.float32, shape=(4, 32))
    assert jnp.array_equal(a(x), b(x))


def _fp32_scale_reference(x, weight, eps):
    """Independent float64 NumPy ``(1 + w) * x / sqrt(mean(x^2) + eps)``."""
    x64 = np.asarray(x, np.float64)
    normed = x64 / np.sqrt(np.mean(np.square(x64), -1, keepdims=True) + eps)
    return (1.0 + np.asarray(weight, np.float64)) * normed


@pytest.mark.parametrize("dtype", DTYPES)
def test_fp32_scale_matches_float64_reference(dtype):
    """``fp32_scale``: statistics and ``(1 + w)`` in fp32, one cast to the input dtype (HF Gemma / Qwen3-Next)."""
    eps = 1e-6
    norm = RMSNorm(dim=64, eps=eps, dtype=dtype, param_dtype=dtype, scale_offset=1.0, fp32_scale=True, rngs=spx.Rngs(0))
    weight = jnp.asarray(np.random.default_rng(1).standard_normal(64) * 0.1, dtype)
    norm.weight.value = weight
    x = _x(dtype)
    out = norm(x)
    assert out.dtype == dtype
    ref = _fp32_scale_reference(x.astype(jnp.float32), weight.astype(jnp.float32), eps)
    tol = 1e-5 if dtype == jnp.float32 else 1e-2
    np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=tol, atol=tol)


def test_fp32_scale_differs_from_default_ordering_in_bfloat16():
    """The flag exists because multiplying before the cast is a different bf16 result than after it."""
    weight = jnp.asarray(np.random.default_rng(2).standard_normal(64) * 0.5, jnp.bfloat16)
    kw = dict(dim=64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, scale_offset=1.0, rngs=spx.Rngs(0))
    pre, post = RMSNorm(fp32_scale=True, **kw), RMSNorm(**kw)
    pre.weight.value = weight
    post.weight.value = weight
    x = _x(jnp.bfloat16, shape=(64, 64))
    assert not jnp.array_equal(pre(x), post(x))


def test_fp32_scale_low_precision_output_becomes_bfloat16():
    """A low-precision float ``output_dtype`` is returned as bfloat16 (the Gemma-3 rule)."""
    norm = RMSNorm(
        dim=32,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_offset=1.0,
        fp32_scale=True,
        output_dtype=jnp.float8_e4m3fn,
        rngs=spx.Rngs(0),
    )
    assert norm(_x(jnp.float32, shape=(4, 32))).dtype == jnp.bfloat16


def test_explicit_kernel_init_is_what_reinit_reads():
    """Re-initialisation calls ``module.kernel_init``; an explicit zeros init must not fall back to the class ones."""
    norm = RMSNorm(dim=16, kernel_init=jax.nn.initializers.zeros, scale_offset=1.0, rngs=spx.Rngs(0))
    assert jnp.array_equal(norm.weight.value, jnp.zeros((16,), norm.param_dtype))
    reinit = norm.kernel_init(key=jax.random.PRNGKey(0), shape=(16,), dtype=jnp.float32)
    assert jnp.array_equal(reinit, jnp.zeros((16,), jnp.float32))
    default = RMSNorm(dim=16, rngs=spx.Rngs(0))
    assert jnp.array_equal(default.kernel_init(jax.random.PRNGKey(0), (16,), jnp.float32), jnp.ones((16,)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
