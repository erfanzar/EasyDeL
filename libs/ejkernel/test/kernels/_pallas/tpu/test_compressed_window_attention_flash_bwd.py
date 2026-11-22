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

"""TPU tests for the flash backward, causal tile skip and VMEM-sized tiles of compressed-window attention.

Every gradient is compared against ``jax.grad`` of the dense XLA reference (an independent implementation), at
DeepSeek-V4-like shapes (16 heads x 512, sliding window 128 over the own tokens plus compressed entries) and at a
DSA-like causal ``window = 0`` layout. Skipped when no TPU is present.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp


def _has_tpu():
    """Return True when a TPU backend is available."""
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require a TPU backend")

_NEG = float(np.finfo(np.float32).min)


def _pallas():
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        compressed_window_attention_tpu,
    )

    return compressed_window_attention_tpu


def _xla():
    from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention

    return compressed_window_attention


def _max_err(a, b):
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def _v4_inputs(s, n_entries, window, heads, dim, dtype, seed=0):
    """Implicit-mask layout: own tokens (window + causal + validity in-kernel) then compressed entries."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((1, heads, s, dim)), dtype)
    kv = jnp.asarray(rng.standard_normal((1, s + n_entries, dim)), dtype)
    entry = (rng.standard_normal((1, s, n_entries)) * 0.3).astype(np.float32)
    entry[rng.random(entry.shape) < 0.5] = _NEG
    valid = rng.random((1, s)) > 0.1
    valid[:, 0] = True
    qi, ki = np.arange(s)[:, None], np.arange(s)[None, :]
    tok = np.where(((ki <= qi) & (ki > qi - window))[None] & valid[:, None, :], 0.0, _NEG).astype(np.float32)
    dense = np.concatenate([tok, entry], axis=2)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    return q, kv, jnp.asarray(entry), jnp.asarray(dense), jnp.asarray(valid), sinks


def _causal_inputs(s, heads, dim, dtype, seed=0, masked_rows=()):
    """DSA-like layout: ``window = 0``, a causal sparse bias over the whole KV axis."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((1, heads, s, dim)), dtype)
    kv = jnp.asarray(rng.standard_normal((1, s, dim)), dtype)
    qi, ki = np.arange(s)[:, None], np.arange(s)[None, :]
    keep = (rng.random((s, s)) < 0.4) | (qi == ki)
    bias = np.where(keep & (ki <= qi), rng.standard_normal((s, s)) * 0.3, _NEG).astype(np.float32)[None]
    for r in masked_rows:
        bias[:, r, :] = _NEG
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    return q, kv, jnp.asarray(bias), sinks


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("use_sinks", [True, False], ids=["sink", "no-sink"])
def test_flash_backward_matches_xla_v4_layout(dtype, use_sinks):
    """Implicit mask + band skip + entries: value and all cotangents (q, kv, entry bias, sink) vs the dense XLA grad."""
    s, n_entries, window = 2048, 512, 128
    q, kv, entry, dense, valid, sinks = _v4_inputs(s, n_entries, window, 16, 512, dtype, seed=3)
    sinks = sinks if use_sinks else None
    scale = 512**-0.5
    cot = jnp.asarray(np.random.default_rng(4).standard_normal(q.shape), dtype)
    pal, xla = _pallas(), _xla()
    kw = dict(softmax_scale=scale, window=window, token_kv_len=s, token_valid=valid)

    def lp(a, b, c, d):
        return jnp.sum(pal(a, b, c, d, **kw).astype(jnp.float32) * cot.astype(jnp.float32))

    def lx(a, b, c, d):
        full = jnp.concatenate([dense[:, :, :s], c], axis=2)
        return jnp.sum(xla(a, b, full, d, softmax_scale=scale).astype(jnp.float32) * cot.astype(jnp.float32))

    argnums = (0, 1, 2, 3) if use_sinks else (0, 1, 2)
    out_p = pal(q, kv, entry, sinks, **kw)
    out_x = xla(q, kv, dense, sinks, softmax_scale=scale)
    tol = 3e-2 if dtype == jnp.bfloat16 else 2e-4
    assert out_p.shape == out_x.shape and out_p.dtype == out_x.dtype
    assert _max_err(out_p, out_x) < tol
    gp = jax.jit(jax.grad(lp, argnums=argnums))(q, kv, entry, sinks)
    gx = jax.jit(jax.grad(lx, argnums=argnums))(q, kv, entry, sinks)
    for a, b in zip(gp, gx, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.isfinite(np.asarray(a.astype(jnp.float32))).all()
        scale_ref = max(1.0, float(jnp.max(jnp.abs(b.astype(jnp.float32)))))
        assert _max_err(a, b) / scale_ref < tol, (_max_err(a, b), scale_ref)


@pytest.mark.parametrize("use_sinks", [True, False], ids=["sink", "no-sink"])
def test_flash_backward_dense_bias_with_fully_masked_rows(use_sinks):
    """Dense padded bias with fully masked rows: the closed-form masked-row cotangents match the dense reference."""
    rng = np.random.default_rng(11)
    b, h, s, d, kv_len = 2, 8, 260, 512, 300
    q = jnp.asarray(rng.standard_normal((b, h, s, d)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((b, kv_len, d)) + 0.5, jnp.float32)
    bias = (rng.standard_normal((b, s, kv_len)) * 0.5).astype(np.float32)
    bias[rng.random(bias.shape) < 0.3] = _NEG
    bias[:, [0, 128, 259], :] = _NEG
    bias = jnp.asarray(bias)
    sinks = jnp.asarray(rng.standard_normal((h,)), jnp.float32) if use_sinks else None
    scale = d**-0.5
    cot = jnp.asarray(rng.standard_normal(q.shape), jnp.float32)
    pal, xla = _pallas(), _xla()
    argnums = (0, 1, 2, 3) if use_sinks else (0, 1, 2)
    gp = jax.grad(lambda *a: jnp.sum(pal(*a, softmax_scale=scale) * cot), argnums=argnums)(q, kv, bias, sinks)
    gx = jax.grad(lambda *a: jnp.sum(xla(*a, softmax_scale=scale) * cot), argnums=argnums)(q, kv, bias, sinks)
    for a, b_ in zip(gp, gx, strict=True):
        assert _max_err(a, b_) < 2e-4, _max_err(a, b_)


def test_flash_and_reference_backwards_agree():
    """``backward="flash"`` and ``backward="reference"`` give the same gradients (two independent rules)."""
    q, kv, entry, _, valid, sinks = _v4_inputs(1024, 256, 128, 8, 512, jnp.float32, seed=5)
    cot = jnp.asarray(np.random.default_rng(6).standard_normal(q.shape), jnp.float32)
    pal = _pallas()
    kw = dict(window=128, token_kv_len=1024, token_valid=valid)

    def loss(rule):
        return lambda a, b, c, d: jnp.sum(pal(a, b, c, d, backward=rule, **kw) * cot)

    gf = jax.grad(loss("flash"), argnums=(0, 1, 2, 3))(q, kv, entry, sinks)
    gr = jax.grad(loss("reference"), argnums=(0, 1, 2, 3))(q, kv, entry, sinks)
    for a, b in zip(gf, gr, strict=True):
        assert _max_err(a, b) < 2e-4, _max_err(a, b)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("use_sinks", [True, False], ids=["sink", "no-sink"])
def test_causal_skip_matches_full_visit_and_xla(dtype, use_sinks):
    """``causal=True`` (tile skip) equals ``causal=False`` (full visit) and the XLA reference, value and gradients."""
    q, kv, bias, sinks = _causal_inputs(1100, 16, 512, dtype, seed=7, masked_rows=(5,))
    sinks = sinks if use_sinks else None
    cot = jnp.asarray(np.random.default_rng(8).standard_normal(q.shape), dtype)
    pal, xla = _pallas(), _xla()
    argnums = (0, 1, 2, 3) if use_sinks else (0, 1, 2)

    def loss(fn, **kw):
        return lambda *a: jnp.sum(fn(*a, softmax_scale=512**-0.5, **kw).astype(jnp.float32) * cot.astype(jnp.float32))

    skip = jax.jit(jax.grad(loss(pal, causal=True), argnums=argnums))(q, kv, bias, sinks)
    full = jax.jit(jax.grad(loss(pal, causal=False), argnums=argnums))(q, kv, bias, sinks)
    ref = jax.jit(jax.grad(loss(xla), argnums=argnums))(q, kv, bias, sinks)
    out_skip = pal(q, kv, bias, sinks, softmax_scale=512**-0.5, causal=True)
    out_full = pal(q, kv, bias, sinks, softmax_scale=512**-0.5, causal=False)
    tol = 3e-2 if dtype == jnp.bfloat16 else 2e-4
    assert _max_err(out_skip, out_full) <= (1e-2 if dtype == jnp.bfloat16 else 1e-6)
    for a, b, c in zip(skip, full, ref, strict=True):
        scale_ref = max(1.0, float(jnp.max(jnp.abs(c.astype(jnp.float32)))))
        assert _max_err(a, b) / scale_ref < tol
        assert _max_err(a, c) / scale_ref < tol


def test_causal_skip_has_teeth():
    """With a non-causal bias the causal promise is broken and the skipped output must differ (the skip is real)."""
    rng = np.random.default_rng(9)
    q = jnp.asarray(rng.standard_normal((1, 8, 1024, 128)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((1, 1024, 128)), jnp.float32)
    bias = jnp.zeros((1, 1024, 1024), jnp.float32)
    pal = _pallas()
    a = pal(q, kv, bias, None, causal=True)
    b = pal(q, kv, bias, None, causal=False)
    assert _max_err(a, b) > 1e-3


def test_window_zero_honours_arbitrary_bias():
    """Draft rows after a context with their own (non-band) bias are exact under ``window = 0``."""
    rng = np.random.default_rng(10)
    ctx, drafts, h, d = 2560, 256, 8, 128
    s = ctx + drafts
    q = jnp.asarray(rng.standard_normal((1, h, s, d)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((1, s, d)), jnp.float32)
    qi, ki = np.arange(s)[:, None], np.arange(s)[None, :]
    allowed = (ki <= qi) & (ki > qi - 128)
    anchor = ctx - 1 - (np.arange(drafts) % 64)
    allowed[ctx:] = False
    allowed[ctx:, :ctx] = (ki[0, :ctx][None] <= anchor[:, None]) & (ki[0, :ctx][None] > anchor[:, None] - 128)
    allowed[ctx:, ctx:] = True
    bias = jnp.asarray(np.where(allowed, 0.0, _NEG).astype(np.float32)[None])
    sinks = jnp.asarray(rng.standard_normal((h,)), jnp.float32)
    cot = jnp.asarray(rng.standard_normal(q.shape), jnp.float32)
    pal, xla = _pallas(), _xla()
    gp = jax.grad(lambda *a: jnp.sum(pal(*a, window=0) * cot), argnums=(0, 1, 2, 3))(q, kv, bias, sinks)
    gx = jax.grad(lambda *a: jnp.sum(xla(*a) * cot), argnums=(0, 1, 2, 3))(q, kv, bias, sinks)
    assert _max_err(pal(q, kv, bias, sinks, window=0), xla(q, kv, bias, sinks)) < 2e-4
    for a, b in zip(gp, gx, strict=True):
        assert _max_err(a, b) < 2e-4


def test_vmem_tiles_follow_capacity():
    """64 MiB parts get (128, 512) tiles at 16 x 512 bf16; the 16 MiB override reproduces the legacy (16, 128)."""
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._layout import (
        default_blocks,
        tpu_vmem_capacity_bytes,
        vmem_safe_blocks,
    )

    big, small = 64 * 2**20, 16 * 2**20
    bq, bkv = default_blocks(8192, 10240, None, big)
    assert vmem_safe_blocks(bq, bkv, 16, 512, 2, big) == (128, 512, 63 * 2**20)
    bq, bkv = default_blocks(8192, 10240, None, small)
    assert vmem_safe_blocks(bq, bkv, 16, 512, 2, small) == (16, 128, None)
    assert default_blocks(8, 12, None, big) == (8, 128)  # lane-aligned KV tile for a short prefill
    assert tpu_vmem_capacity_bytes() >= small
    # Both tilings produce the same values on hardware.
    q, kv, entry, _, valid, sinks = _v4_inputs(512, 64, 128, 16, 512, jnp.bfloat16, seed=12)
    pal = _pallas()
    kw = dict(window=128, token_kv_len=512, token_valid=valid)
    a = pal(q, kv, entry, sinks, vmem_capacity_bytes=small, **kw)
    b = pal(q, kv, entry, sinks, **kw)
    assert _max_err(a, b) < 2e-2


def test_short_prefill_compiles():
    """An 8-token prefill with 4 entries compiles with the 128-lane KV tile and matches XLA (fwd + grad)."""
    q, kv, entry, dense, valid, sinks = _v4_inputs(8, 4, 128, 4, 128, jnp.float32, seed=13)
    pal, xla = _pallas(), _xla()
    kw = dict(window=128, token_kv_len=8, token_valid=valid)
    assert _max_err(pal(q, kv, entry, sinks, **kw), xla(q, kv, dense, sinks)) < 2e-4
    g = jax.grad(lambda a: jnp.sum(pal(a, kv, entry, sinks, **kw)))(q)
    gx = jax.grad(lambda a: jnp.sum(xla(a, kv, dense, sinks)))(q)
    assert _max_err(g, gx) < 2e-4


def test_invalid_backward_rejected():
    """An unknown backward rule is rejected rather than silently mapped."""
    q, kv, bias, sinks = _causal_inputs(128, 4, 128, jnp.float32)
    with pytest.raises(ValueError, match="backward"):
        _pallas()(q, kv, bias, sinks, backward="dense")
