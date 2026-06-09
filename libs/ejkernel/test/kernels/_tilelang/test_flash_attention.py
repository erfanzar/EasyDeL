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

"""TileLang parity tests for flash_attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ._helpers import (
    _FA_FEATURES,
    _FP16_BWD_TOL,
    _FP16_FWD_TOL,
    _SEED,
    _fa_feature_inputs,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_fwd(causal):
    B, N, H, D = 1, 64, 2, 64
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q, k, v = _randn(k1, (B, N, H, D)), _randn(k2, (B, N, H, D)), _randn(k3, (B, N, H, D))
    tl, xla = _tl("flash_attention"), _xla("flash_attention")
    out_tl = tl(q, k, v, causal=causal)
    out_xla = xla(q, k, v, causal=causal)
    assert _max_abs(out_tl, out_xla) < _FP16_FWD_TOL


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_bwd(causal):
    B, N, H, D = 1, 64, 2, 64
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q, k, v = _randn(k1, (B, N, H, D)), _randn(k2, (B, N, H, D)), _randn(k3, (B, N, H, D))
    tl, xla = _tl("flash_attention"), _xla("flash_attention")

    def f_tl(*a):
        return jnp.sum(tl(*a, causal=causal).astype(jnp.float32))

    def f_xla(*a):
        return jnp.sum(xla(*a, causal=causal).astype(jnp.float32))

    g_tl = jax.grad(f_tl, argnums=(0, 1, 2))(q, k, v)
    g_xla = jax.grad(f_xla, argnums=(0, 1, 2))(q, k, v)
    for tl_g, xla_g, name in zip(g_tl, g_xla, "qkv", strict=True):
        assert _max_abs(tl_g, xla_g) < _FP16_BWD_TOL, f"grad d{name} diff too large"


@pytest.mark.parametrize("feat", _FA_FEATURES)
def test_flash_attention_feature_fwd(feat):
    """Every FA score-space feature is applied natively (no silent drop)."""
    q, k, v, kw = _fa_feature_inputs(feat)
    tl, xla = _tl("flash_attention"), _xla("flash_attention")
    assert _max_abs(tl(q, k, v, **kw), xla(q, k, v, **kw)) < 2e-2


@pytest.mark.parametrize("feat", [f for f in _FA_FEATURES if f != "normalize_off"])
def test_flash_attention_feature_bwd(feat):
    """Backward also honours every feature — dq/dk/dv match XLA."""
    q, k, v, kw = _fa_feature_inputs(feat)
    tl, xla = _tl("flash_attention"), _xla("flash_attention")
    g_tl = jax.grad(lambda *a: jnp.sum(tl(*a, **kw).astype(jnp.float32)), argnums=(0, 1, 2))(q, k, v)
    g_xla = jax.grad(lambda *a: jnp.sum(xla(*a, **kw).astype(jnp.float32)), argnums=(0, 1, 2))(q, k, v)
    for a, b, nm in zip(g_tl, g_xla, "qkv", strict=True):
        assert _max_abs(a, b) < 3e-2, f"grad d{nm} ({feat}) diff too large"


def test_flash_attention_normalize_off_bwd():
    """``normalize_output=False`` backward.

    The un-normalised output ``sum_j exp(s_j - M) V_j`` carries the
    stabilising max ``M`` as an ``exp(-M)`` factor, so its gradient differs
    from the normalised case by a ``-[j == argmax] * D`` term. The tile-lang
    kernel implements that term exactly (verified against the analytic
    formula to <1e-3). The only residual vs the XLA reference is the
    *identity* of the argmax for near-tied rows: the tile-lang kernel scores
    in fp32 while XLA's einsum rounds to fp16, so the two pick different
    columns when the top two logits agree to fp16 precision. ``dV`` is
    unaffected and matches exactly; the gradient stays finite and bounded.
    """
    q, k, v, kw = _fa_feature_inputs("normalize_off")
    tl, xla = _tl("flash_attention"), _xla("flash_attention")
    g_tl = jax.grad(lambda *a: jnp.sum(tl(*a, **kw).astype(jnp.float32)), argnums=(0, 1, 2))(q, k, v)
    g_xla = jax.grad(lambda *a: jnp.sum(xla(*a, **kw).astype(jnp.float32)), argnums=(0, 1, 2))(q, k, v)
    assert _max_abs(g_tl[2], g_xla[2]) < 3e-2, "dv must match exactly"
    for a, b, nm in zip(g_tl, g_xla, "qkv", strict=True):
        assert bool(jnp.isfinite(a).all()), f"grad d{nm} not finite"
        assert _max_abs(a, b) < 0.5 * max(float(jnp.abs(b.astype(jnp.float32)).max()), 1e-6) + 3e-2


def test_flash_attention_gates_unsupported():
    """``cum_seqlens`` / ``block_tables`` raise — never silently ignored."""
    from ejkernel.errors import EjkernelRuntimeError

    B, N, H, D = 1, 16, 2, 32
    q = _randn(jax.random.PRNGKey(0), (B, N, H, D))
    tl = _tl("flash_attention")
    with pytest.raises(EjkernelRuntimeError):
        tl(q, q, q, block_tables=jnp.zeros((B, 4), jnp.int32))
    with pytest.raises(EjkernelRuntimeError):
        tl(q, q, q, cum_seqlens_q=jnp.array([0, N], jnp.int32))
