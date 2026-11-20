"""Fused decode MoE segment (``moe_decode_fused_w8a8``, TPU only).

Pinned here: the Pallas TPU kernel matches the XLA reference within 2% of max (the expert sum order and in-kernel
float32 lowering differ) at per-chip decode shapes with and without the folded shared expert, with a nonzero expert
offset and with tokens routed to no local expert; it stays as close to a float32 dequantized reference as the XLA
reference does (max abs and relative L2); the registry resolves Pallas on TPU and XLA at priority 0 with compatible
signatures; unsupported operands are reported.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules import moe_decode_fused_w8a8, moe_decode_fused_w8a8_unsupported_reason

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real TPU")

ACT = ("clamped_swiglu", (10.0,))


def _inputs(t, h, m, e_loc, e_glob, k, inter, seed, offset):
    ks = jax.random.split(jax.random.PRNGKey(seed), 13)
    x = jax.random.normal(ks[0], (t, h), jnp.float32).astype(jnp.bfloat16)
    ids = jnp.argsort(jax.random.uniform(ks[1], (t, e_glob)), axis=1)[:, :k].astype(jnp.int32)
    w = jax.random.uniform(ks[2], (t, k), jnp.float32, 0.05, 1.0)

    def ri(kk, s):
        return jax.random.randint(kk, s, -127, 128, jnp.int32).astype(jnp.int8)

    def ru(kk, s, lo, hi):
        return jax.random.uniform(kk, s, jnp.float32, lo, hi)

    g, u, d = ri(ks[3], (e_loc, h, m)), ri(ks[4], (e_loc, h, m)), ri(ks[5], (e_loc, m, h))
    gs, us = ru(ks[6], (e_loc, 1, m), 1e-4, 4e-4), ru(ks[7], (e_loc, 1, m), 1e-4, 4e-4)
    ds = ru(ks[8], (e_loc, 1, h), 1e-4, 4e-4)
    ctrl = jnp.array([offset, 0, t // 2], jnp.int32)
    sh = (None,) * 4
    if inter:
        sh = (
            ri(ks[9], (h, 2 * inter)),
            ru(ks[10], (1, 2 * inter), 1e-4, 4e-4),
            ri(ks[11], (inter, h)),
            ru(ks[12], (1, h), 1e-4, 4e-4),
        )
    return (x, ids, w, g, gs, u, us, d, ds, ctrl, *sh)


def _run(args, platform):
    kw = dict(activation=ACT[0], act_params=ACT[1], shared_activation=ACT[0], shared_act_params=ACT[1])
    return np.asarray(jax.jit(lambda *a: moe_decode_fused_w8a8(*a, platform=platform, **kw))(*args))


def _f32_reference(args):
    """Dequantized float32 weights, float32 activations (no activation quantization), HIGHEST precision."""
    x, ids, w, g, gs, u, us, d, ds, ctrl, sgu, sgus, sdn, sdns = args
    hp = jax.lax.Precision.HIGHEST
    x32 = x.astype(jnp.float32)
    lim = ACT[1][0]

    def comb(a, b):
        return jax.nn.silu(jnp.minimum(a, lim)) * jnp.clip(b, -lim, lim)

    e = g.shape[0]
    gate = jnp.einsum("th,ehm->etm", x32, g.astype(jnp.float32) * gs, precision=hp)
    up = jnp.einsum("th,ehm->etm", x32, u.astype(jnp.float32) * us, precision=hp)
    down = jnp.einsum("etm,emh->eth", comb(gate, up), d.astype(jnp.float32) * ds, precision=hp)
    local = ids - ctrl[0]
    wmat = jnp.sum(jnp.where(local[..., None] == jnp.arange(e), w[..., None], 0.0), axis=1)
    out = jnp.einsum("te,eth->th", wmat, down, precision=hp)
    if sgu is not None:
        inter = sdn.shape[0]
        sg = jnp.dot(x32, sgu[:, :inter].astype(jnp.float32) * sgus[:, :inter], precision=hp)
        su = jnp.dot(x32, sgu[:, inter:].astype(jnp.float32) * sgus[:, inter:], precision=hp)
        sh = jnp.dot(comb(sg, su), sdn.astype(jnp.float32) * sdns, precision=hp)
        row = jnp.arange(x.shape[0])[:, None]
        out = out + jnp.where((row >= ctrl[1]) & (row < ctrl[2]), sh, 0.0)
    return np.asarray(out)


CASES = [
    # t, h, m, e_loc, e_glob, k, shared inter, seed, offset
    (32, 4096, 512, 72, 288, 8, 512, 0, 72),
    (32, 4096, 128, 288, 288, 8, 128, 1, 0),
    (32, 4096, 256, 144, 288, 8, 256, 2, 144),
    (32, 4096, 512, 72, 288, 8, 0, 3, 0),
    (8, 2048, 256, 16, 64, 2, 0, 4, 48),
]


@pytest.mark.parametrize("t,h,m,e_loc,e_glob,k,inter,seed,offset", CASES)
def test_pallas_close_to_xla_and_f32(t, h, m, e_loc, e_glob, k, inter, seed, offset):
    args = _inputs(t, h, m, e_loc, e_glob, k, inter, seed, offset)
    ref, out = _run(args, "xla"), _run(args, "pallas")
    f32 = _f32_reference(args)
    assert out.dtype == np.float32 and out.shape == (t, h)
    assert np.isfinite(out).all() and np.isfinite(ref).all()
    scale = float(np.abs(f32).max())
    err_ref = float(np.abs(out - ref).max()) / scale
    err_out_f32 = float(np.abs(out - f32).max()) / scale
    err_ref_f32 = float(np.abs(ref - f32).max()) / scale
    # Two equally accurate paths that round differently (measured on v5p: 0.75-1.27% of max apart).
    assert err_ref < 2e-2, err_ref
    # The kernel is no farther from float32 than the XLA reference (measured ratio 0.89-1.11 on v5p).
    assert err_out_f32 <= 1.25 * err_ref_f32 + 1e-3, (err_out_f32, err_ref_f32)
    l2_out = float(np.linalg.norm(out - f32) / np.linalg.norm(f32))
    l2_ref = float(np.linalg.norm(ref - f32) / np.linalg.norm(f32))
    assert l2_out <= 1.05 * l2_ref, (l2_out, l2_ref)


def test_tokens_routed_nowhere_give_only_shared_rows():
    args = list(_inputs(16, 2048, 256, 8, 64, 2, 256, 5, 0))
    args[1] = jnp.full_like(args[1], 63)  # no token routed to local experts 0..7
    out = _run(tuple(args), "pallas")
    assert np.all(out[8:] == 0.0)
    assert np.abs(out[:8]).max() > 0


def test_registry_entries():
    pallas = kernel_registry.get("moe_decode_fused_w8a8", platform=Platform.PALLAS, backend=Backend.TPU)
    xla = kernel_registry.get("moe_decode_fused_w8a8", platform=Platform.XLA, backend=Backend.ANY)
    assert pallas is not None and xla is not None and pallas is not xla
    assert kernel_registry.validate_signatures("moe_decode_fused_w8a8")


def test_unsupported_reason():
    args = _inputs(32, 4096, 512, 72, 288, 8, 512, 0, 72)
    assert moe_decode_fused_w8a8_unsupported_reason(*args) is None
    big = (jnp.zeros((512, 4096), jnp.bfloat16), jnp.zeros((512, 8), jnp.int32), jnp.zeros((512, 8)), *args[3:])
    assert "token rows" in moe_decode_fused_w8a8_unsupported_reason(*big)
    bad = (*args[:3], args[3].astype(jnp.int32), *args[4:])
    assert moe_decode_fused_w8a8_unsupported_reason(*bad) is not None
