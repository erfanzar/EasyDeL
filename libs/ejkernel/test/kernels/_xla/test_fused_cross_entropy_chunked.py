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

"""XLA chunked / fused-linear (FLCE) cross-entropy: parity vs the dense path.

These paths (vocab/token/block chunking and the LM-head-chunked FLCE) were
migrated into ejkernel from EasyDeL so the chunked CE math lives in one place.
The tests check loss / accuracy / weight-sum and — crucially — gradients
(wrt logits, and wrt hidden + lm_head_weight for FLCE) against a plain dense
reference, plus the public ``fused_cross_entropy`` dispatch.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.fused_cross_entropy._xla_impl_chunked import (
    blockwise_cross_entropy,
    chunked_token_cross_entropy,
    chunked_vocab_cross_entropy,
)
from ejkernel.kernels._xla.fused_cross_entropy._xla_impl_linear import fused_linear_cross_entropy
from ejkernel.modules.operations import fused_cross_entropy


def _data(N=128, V=2048, seed=0, frac_ignore=0.15):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    logits = jax.random.normal(k[0], (N, V), jnp.float32) * 2.0
    targets = jax.random.randint(k[1], (N,), 0, V)
    targets = jnp.where(jax.random.uniform(k[2], (N,)) < frac_ignore, -100, targets)
    weights = (jax.random.uniform(k[3], (N,)) > 0.1).astype(jnp.float32)
    return logits, targets, weights


def _dense_ref(logits, targets, weights, ls, zl, reduction, ignore_index=-100):
    lg = logits.astype(jnp.float32)
    V = lg.shape[-1]
    valid = (targets != ignore_index).astype(jnp.float32)
    w = valid * weights.astype(jnp.float32)
    safe = jnp.where(targets != ignore_index, targets, 0)
    lse = jax.scipy.special.logsumexp(lg, axis=-1)
    ty = jnp.take_along_axis(lg, safe[:, None], axis=-1)[:, 0]
    if ls > 0:
        conf, low = 1.0 - ls, ls / (V - 1)
        nc = -(conf * np.log(conf) + (V - 1) * low * np.log(low + 1e-20))
        nll = lse - ((conf - low) * ty + low * jnp.sum(lg, -1)) - nc
    else:
        nll = lse - ty
    zt = zl * lse * lse if zl > 0 else jnp.zeros_like(lse)
    per = (nll + zt) * w
    wsum = jnp.sum(w)
    tot = jnp.sum(per)
    if reduction == "mean":
        tot = tot / jnp.maximum(wsum, 1e-8)
    acc = jnp.sum((jnp.argmax(lg, -1) == targets).astype(jnp.float32) * w) / jnp.maximum(wsum, 1e-8)
    return tot, jnp.sum(zt * w), wsum, acc


@pytest.mark.parametrize("ls,zl", [(0.0, 0.0), (0.1, 0.0), (0.0, 1e-3), (0.1, 1e-3)])
def test_chunked_vocab_matches_dense(ls, zl):
    logits, targets, weights = _data()
    loss, _zloss, wsum, acc = chunked_vocab_cross_entropy(
        logits, targets, weights, label_smoothing=ls, z_loss=zl, reduction="mean", chunk_size=512
    )
    rl, _rz, rw, ra = _dense_ref(logits, targets, weights, ls, zl, "mean")
    assert jnp.allclose(loss, rl, atol=2e-4)
    assert jnp.allclose(wsum, rw, atol=1e-4)
    assert jnp.allclose(acc, ra, atol=1e-5)


@pytest.mark.parametrize("ls,zl", [(0.0, 0.0), (0.1, 1e-3)])
def test_chunked_token_matches_dense(ls, zl):
    logits, targets, weights = _data()
    loss, _zloss, _wsum, acc = chunked_token_cross_entropy(
        logits, targets, weights, label_smoothing=ls, z_loss=zl, reduction="sum", token_chunk_size=32
    )
    rl, _rz, _rw, ra = _dense_ref(logits, targets, weights, ls, zl, "sum")
    assert jnp.allclose(loss, rl, atol=2e-3)
    assert jnp.allclose(acc, ra, atol=1e-5)


def test_blockwise_matches_dense_and_is_checkpointed():
    logits, targets, weights = _data()
    loss, _, _, acc = blockwise_cross_entropy(logits, targets, weights, reduction="sum", block_size=512)
    rl, _, _, ra = _dense_ref(logits, targets, weights, 0.0, 0.0, "sum")
    # online-softmax accumulation: equal up to fp32 rounding
    assert jnp.allclose(loss, rl, rtol=1e-3, atol=3e-3)
    assert jnp.allclose(acc, ra, atol=1e-5)
    grad_jaxpr = str(
        jax.make_jaxpr(
            jax.grad(lambda x: blockwise_cross_entropy(x, targets, weights, reduction="sum", block_size=512)[0])
        )(logits)
    )
    assert "remat2[" in grad_jaxpr or "checkpoint[" in grad_jaxpr


def test_chunked_vocab_gradient_matches_dense():
    logits, targets, weights = _data()
    g = jax.grad(lambda x: chunked_vocab_cross_entropy(x, targets, weights, reduction="mean", chunk_size=512)[0])(logits)
    gr = jax.grad(lambda x: _dense_ref(x, targets, weights, 0.0, 0.0, "mean")[0])(logits)
    assert jnp.allclose(g, gr, atol=1e-5)


@pytest.mark.parametrize("ls,zl,bias", [(0.0, 0.0, False), (0.1, 1e-3, True)])
def test_flce_matches_dense_loss_and_grads(ls, zl, bias):
    B, T, H, V = 4, 48, 256, 2048
    k = jax.random.split(jax.random.PRNGKey(3), 4)
    hidden = jax.random.normal(k[0], (B, T, H), jnp.float32) * 0.5
    W = jax.random.normal(k[1], (H, V), jnp.float32) * 0.05
    b = jax.random.normal(k[2], (V,), jnp.float32) * 0.1 if bias else None
    targets = jax.random.randint(k[3], (B, T), 0, V)
    weights = (jax.random.uniform(jax.random.PRNGKey(5), (B, T)) > 0.1).astype(jnp.float32)

    def flce(h, w_):
        return fused_linear_cross_entropy(
            h,
            targets,
            weights,
            lm_head_weight=w_,
            lm_head_bias=b,
            label_smoothing=ls,
            z_loss=zl,
            reduction="mean",
            token_chunk_size=16,
        )[0]

    def ref(h, w_):
        lg = jnp.matmul(h, w_)
        if bias:
            lg = lg + b
        return _dense_ref(lg.reshape(-1, V), targets.reshape(-1), weights.reshape(-1), ls, zl, "mean")[0]

    assert jnp.allclose(flce(hidden, W), ref(hidden, W), atol=1e-5)
    gh, gw = jax.grad(flce, argnums=(0, 1))(hidden, W)
    rh, rw = jax.grad(ref, argnums=(0, 1))(hidden, W)
    assert jnp.allclose(gh, rh, atol=1e-5)
    assert jnp.allclose(gw, rw, atol=1e-5)


def test_flce_lm_head_fn_matches_weight_path():
    B, T, H, V = 2, 32, 128, 1024
    k = jax.random.split(jax.random.PRNGKey(11), 3)
    hidden = jax.random.normal(k[0], (B, T, H), jnp.float32) * 0.5
    W = jax.random.normal(k[1], (H, V), jnp.float32) * 0.05
    targets = jax.random.randint(k[2], (B, T), 0, V)
    out_w = fused_linear_cross_entropy(hidden, targets, lm_head_weight=W, reduction="mean", token_chunk_size=8)[0]
    out_fn = fused_linear_cross_entropy(
        hidden, targets, lm_head_fn=lambda x: jnp.matmul(x, W), reduction="mean", token_chunk_size=8
    )[0]
    assert jnp.allclose(out_w, out_fn, atol=1e-6)


def test_public_dispatch_chunked_and_flce():
    logits, targets, weights = _data()
    out = fused_cross_entropy(logits, targets, weights, chunk_size=512, chunk_strategy="vocab", reduction="mean")
    ref = chunked_vocab_cross_entropy(logits, targets, weights, reduction="mean", chunk_size=512)
    assert jnp.allclose(out.loss, ref[0], atol=1e-6)
    assert out.accuracy is not None

    B, T, H, V = 2, 16, 64, 512
    k = jax.random.split(jax.random.PRNGKey(1), 3)
    hidden = jax.random.normal(k[0], (B, T, H), jnp.float32) * 0.5
    W = jax.random.normal(k[1], (H, V), jnp.float32) * 0.05
    tgt = jax.random.randint(k[2], (B, T), 0, V)
    out_f = fused_cross_entropy(hidden=hidden, targets=tgt, lm_head_weight=W, chunk_size=8, reduction="mean")
    assert out_f.loss.shape == ()
    assert jnp.isfinite(out_f.loss)


def test_public_flce_threads_non_xla_platform_to_inner_ce(monkeypatch):
    fce_mod = importlib.import_module("ejkernel.modules.operations.fused_cross_entropy")
    B, T, H, V = 2, 12, 32, 128
    k = jax.random.split(jax.random.PRNGKey(17), 3)
    hidden = jax.random.normal(k[0], (B, T, H), jnp.float32) * 0.5
    W = jax.random.normal(k[1], (H, V), jnp.float32) * 0.05
    targets = jax.random.randint(k[2], (B, T), 0, V)
    calls = []
    real_get = fce_mod.kernel_registry.get

    def fake_impl(
        logits,
        chunk_targets,
        chunk_weights,
        *,
        ignore_index=-100,
        label_smoothing=0.0,
        z_loss=0.0,
        soft_targets=None,
        reduction="mean",
        vocab_parallel_axis=None,
        block_v=0,
        block_m=0,
    ):
        del label_smoothing, z_loss, soft_targets, vocab_parallel_axis, block_v, block_m
        assert reduction == "none"
        lg = logits.astype(jnp.float32)
        valid = chunk_targets != ignore_index
        safe = jnp.where(valid, chunk_targets, 0).astype(jnp.int32)
        w = chunk_weights.astype(jnp.float32) * valid.astype(jnp.float32)
        lse = jax.scipy.special.logsumexp(lg, axis=-1)
        target_logit = jnp.take_along_axis(lg, safe[..., None], axis=-1)[..., 0]
        per_row = (lse - target_logit) * w
        correct = (jnp.argmax(lg, axis=-1) == chunk_targets).astype(jnp.float32)
        return per_row, correct

    def fake_get(algorithm, platform=None, backend=None):
        if algorithm == "fused_cross_entropy" and str(platform) == "pallas":
            calls.append((platform, backend))
            return fake_impl
        return real_get(algorithm, platform=platform, backend=backend)

    monkeypatch.setattr(fce_mod.kernel_registry, "get", fake_get)

    def pallas_flce(h):
        return fce_mod.fused_cross_entropy(
            hidden=h,
            targets=targets,
            lm_head_weight=W,
            chunk_size=4,
            reduction="mean",
            platform="pallas",
        ).loss

    def xla_flce(h):
        return fce_mod.fused_cross_entropy(
            hidden=h,
            targets=targets,
            lm_head_weight=W,
            chunk_size=4,
            reduction="mean",
            platform="xla",
        ).loss

    assert jnp.allclose(pallas_flce(hidden), xla_flce(hidden), atol=1e-6)
    assert jnp.allclose(jax.grad(pallas_flce)(hidden), jax.grad(xla_flce)(hidden), atol=1e-6)
    assert calls


def _has_remat(jaxpr_str: str) -> bool:
    return "remat2[" in jaxpr_str or "checkpoint[" in jaxpr_str


def test_flce_checkpoint_flag_toggles_remat_without_changing_numerics():
    B, T, H, V = 3, 40, 192, 1536
    k = jax.random.split(jax.random.PRNGKey(2), 3)
    hidden = jax.random.normal(k[0], (B, T, H), jnp.float32) * 0.5
    W = jax.random.normal(k[1], (H, V), jnp.float32) * 0.05
    targets = jax.random.randint(k[2], (B, T), 0, V)

    def loss(h, w_, ckpt):
        return fused_linear_cross_entropy(
            h, targets, lm_head_weight=w_, reduction="mean", token_chunk_size=8, checkpoint=ckpt
        )[0]

    jp_on = str(jax.make_jaxpr(jax.grad(lambda h: loss(h, W, True)))(hidden))
    jp_off = str(jax.make_jaxpr(jax.grad(lambda h: loss(h, W, False)))(hidden))
    assert _has_remat(jp_on), "checkpoint=True should rematerialize the chunk body"
    assert not _has_remat(jp_off), "checkpoint=False should NOT rematerialize"

    # Checkpointing is a memory/compute tradeoff — values must be identical.
    l_on, l_off = loss(hidden, W, True), loss(hidden, W, False)
    assert jnp.allclose(l_on, l_off, atol=1e-6)
    g_on = jax.grad(lambda h: loss(h, W, True))(hidden)
    g_off = jax.grad(lambda h: loss(h, W, False))(hidden)
    assert jnp.allclose(g_on, g_off, atol=1e-6)


def test_blockwise_checkpoint_flag_toggles_remat_without_changing_numerics():
    logits, targets, weights = _data()

    def loss(x, ckpt):
        return blockwise_cross_entropy(x, targets, weights, reduction="sum", block_size=512, checkpoint=ckpt)[0]

    jp_on = str(jax.make_jaxpr(jax.grad(lambda x: loss(x, True)))(logits))
    jp_off = str(jax.make_jaxpr(jax.grad(lambda x: loss(x, False)))(logits))
    assert _has_remat(jp_on)
    assert not _has_remat(jp_off)
    assert jnp.allclose(loss(logits, True), loss(logits, False), atol=1e-6)
    g_on = jax.grad(lambda x: loss(x, True))(logits)
    g_off = jax.grad(lambda x: loss(x, False))(logits)
    assert jnp.allclose(g_on, g_off, atol=1e-6)


def test_public_op_checkpoint_flag_threads_to_flce_and_block():
    # FLCE via public op
    B, T, H, V = 2, 24, 96, 1024
    k = jax.random.split(jax.random.PRNGKey(4), 3)
    hidden = jax.random.normal(k[0], (B, T, H), jnp.float32) * 0.5
    W = jax.random.normal(k[1], (H, V), jnp.float32) * 0.05
    tgt = jax.random.randint(k[2], (B, T), 0, V)
    jp_off = str(
        jax.make_jaxpr(
            jax.grad(
                lambda h: (
                    fused_cross_entropy(
                        hidden=h, targets=tgt, lm_head_weight=W, chunk_size=8, reduction="mean", checkpoint=False
                    ).loss
                )
            )
        )(hidden)
    )
    jp_on = str(
        jax.make_jaxpr(
            jax.grad(
                lambda h: (
                    fused_cross_entropy(
                        hidden=h, targets=tgt, lm_head_weight=W, chunk_size=8, reduction="mean", checkpoint=True
                    ).loss
                )
            )
        )(hidden)
    )
    assert _has_remat(jp_on) and not _has_remat(jp_off)

    # block via public op
    logits, targets, weights = _data()
    jp_off_b = str(
        jax.make_jaxpr(
            jax.grad(
                lambda x: (
                    fused_cross_entropy(
                        x, targets, weights, chunk_size=512, chunk_strategy="block", reduction="sum", checkpoint=False
                    ).loss
                )
            )
        )(logits)
    )
    assert not _has_remat(jp_off_b)


def test_flce_rejects_bad_args():
    hidden = jnp.zeros((2, 4, 8))
    targets = jnp.zeros((2, 4), jnp.int32)
    W = jnp.zeros((8, 16))
    with pytest.raises(ValueError):
        fused_linear_cross_entropy(hidden, targets, lm_head_weight=W, lm_head_fn=lambda x: x)  # both
    with pytest.raises(ValueError):
        fused_linear_cross_entropy(hidden, targets)  # neither
    with pytest.raises(ValueError):
        fused_linear_cross_entropy(hidden, targets, lm_head_weight=W, reduction="none")  # unsupported


# --------------------------------------------------------------------------------------------------
# Precision / masking regressions. References are float64 NumPy CE over the exact (bf16-rounded)
# logits the kernel sees, so they are independent of every ejkernel code path.
# --------------------------------------------------------------------------------------------------


def _f64_ce_reference(logits, targets, weights=None, ignore_index=-100):
    """Per-row float64 CE and effective weights (``valid * weights``) over the given logits."""
    lg = np.asarray(jnp.asarray(logits, jnp.float32), dtype=np.float64).reshape(-1, logits.shape[-1])
    t = np.asarray(targets).reshape(-1)
    valid = (t != ignore_index).astype(np.float64)
    w = valid if weights is None else valid * np.asarray(weights, np.float64).reshape(-1)
    m = lg.max(-1, keepdims=True)
    lse = np.log(np.exp(lg - m).sum(-1)) + m[:, 0]
    ty = np.take_along_axis(lg, np.where(t != ignore_index, t, 0)[:, None], axis=-1)[:, 0]
    return (lse - ty) * w, w


def _large_t_bf16_logits(n=65536, v=16, seed=31):
    """[n, v] bf16 logits with every 7th target ignored (an odd valid count far above 2**15)."""
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    logits = (jax.random.normal(k[0], (n, v), jnp.float32) * 2.0).astype(jnp.bfloat16)
    targets = jax.random.randint(k[1], (n,), 0, v).at[::7].set(-100)
    return logits, targets


def test_flce_bf16_large_t_accumulates_in_fp32():
    """bf16 FLCE over 65536 tokens: weight_sum is exact and the loss is not biased.

    With bf16 fori_loop carries the running weight_sum saturates at 32768 and the loss sum
    stops growing the same way (~8% off at T=65536).
    """
    B, T, H, V = 1, 65536, 8, 16
    k = jax.random.split(jax.random.PRNGKey(29), 3)
    hidden = (jax.random.normal(k[0], (B, T, H), jnp.float32) * 0.5).astype(jnp.bfloat16)
    W = (jax.random.normal(k[1], (H, V), jnp.float32) * 0.5).astype(jnp.bfloat16)
    targets = jax.random.randint(k[2], (B, T), 0, V).at[:, ::7].set(-100)

    loss_sum, _z, wsum, acc = fused_linear_cross_entropy(
        hidden, targets, lm_head_weight=W, reduction="sum", token_chunk_size=256
    )
    loss_mean = fused_linear_cross_entropy(hidden, targets, lm_head_weight=W, reduction="mean", token_chunk_size=256)[0]

    per_row, w = _f64_ce_reference(jnp.matmul(hidden, W), targets)
    assert loss_sum.dtype == jnp.float32 and wsum.dtype == jnp.float32 and acc.dtype == jnp.float32
    assert float(wsum) == float(w.sum())
    np.testing.assert_allclose(float(loss_sum), per_row.sum(), rtol=1e-3)
    np.testing.assert_allclose(float(loss_mean), per_row.sum() / w.sum(), rtol=1e-3)


@pytest.mark.parametrize("strategy,chunk", [("token", 256), ("vocab", 8), ("block", 8)])
def test_chunked_logits_bf16_large_n_accumulates_in_fp32(strategy, chunk):
    """bf16 compute dtype on the chunked-logits paths: fp32 weight_sum / loss accumulators."""
    logits, targets = _large_t_bf16_logits()
    out = fused_cross_entropy(
        logits,
        targets,
        chunk_size=chunk,
        chunk_strategy=strategy,
        reduction="sum",
        compute_dtype=jnp.bfloat16,
    )
    per_row, w = _f64_ce_reference(logits, targets)
    assert out.weight_sum.dtype == jnp.float32
    assert float(out.weight_sum) == float(w.sum())
    np.testing.assert_allclose(float(out.loss), per_row.sum(), rtol=1e-4)


def test_dense_xla_ce_bf16_logits_use_fp32_row_statistics():
    """Large-magnitude bf16 logits: the row max / log-sum-exp must not be rounded to bf16.

    Around |logit| ~ 100 bf16 spacing is 0.5, so a bf16 lse is off by up to 0.25 nats per row.
    """
    from ejkernel.kernels._xla.fused_cross_entropy import fused_cross_entropy as xla_fused_ce

    N, V = 64, 4096
    k = jax.random.split(jax.random.PRNGKey(41), 2)
    logits = (100.0 + jax.random.normal(k[0], (N, V), jnp.float32) * 2.0).astype(jnp.bfloat16)
    targets = jax.random.randint(k[1], (N,), 0, V)

    per_row, _ = xla_fused_ce(logits, targets, reduction="none")
    ref, _ = _f64_ce_reference(logits, targets)
    assert per_row.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(per_row), ref, atol=1e-3)

    # Custom-VJP forward (under grad) and analytic backward share the fp32 statistics.
    loss, grad = jax.value_and_grad(lambda x: xla_fused_ce(x, targets, reduction="sum")[0])(logits)
    np.testing.assert_allclose(float(loss), ref.sum(), rtol=1e-5)
    lg = np.asarray(jnp.asarray(logits, jnp.float32), np.float64)
    probs = np.exp(lg - lg.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    probs[np.arange(N), np.asarray(targets)] -= 1.0
    assert grad.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(grad, np.float32), probs, atol=4e-3)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_attention_mask_without_weights_keeps_ignore_index_excluded(reduction):
    """``attention_mask`` must be combined with ``targets != ignore_index``, not replace it.

    ``_data`` puts ``ignore_index`` on ~15% of rows where the mask is 1; those rows must not
    enter the loss (with a clipped-to-0 target) nor the mean normalizer.
    """
    logits, targets, _ = _data(N=96, V=512, seed=7)
    mask = jnp.ones_like(targets, dtype=jnp.int32).at[-10:].set(0)

    out = fused_cross_entropy(logits, targets, attention_mask=mask, reduction=reduction, platform="xla")
    per_row, w = _f64_ce_reference(logits, targets, mask)
    expected = per_row.sum() if reduction == "sum" else per_row.sum() / w.sum()
    np.testing.assert_allclose(float(out.loss), expected, rtol=1e-5)
    assert float(out.weight_sum) == float(w.sum())
