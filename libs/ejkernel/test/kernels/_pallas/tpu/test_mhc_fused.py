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

"""Fused mHC collapse / expand (Pallas TPU, compiled by Mosaic) against the XLA reference.

Covers fp32 parity (forward and every gradient) in the fp32 reorder band, bf16 streams in the rounding band, each
op standalone and the combined site under a ``dp x tp`` ``shard_map`` (partial hidden contractions, parameter
cotangents reduced by the transpose), the public ``Kernel`` dispatch (XLA by default), a long-row Mosaic compile at
the ``[., 4, 1024]`` production width, the VMEM request, and input validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.mhc_fused import _pallas_impl as impl
from ejkernel.kernels._pallas.tpu.mhc_fused._pallas_impl import mhc_collapse_pallas, mhc_expand_pallas
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.mhc_fused._xla_impl_fwd import mhc_collapse as xla_collapse
from ejkernel.kernels._xla.mhc_fused._xla_impl_fwd import mhc_expand as xla_expand
from ejkernel.modules import MHCCollapse, MHCExpand, MHCFusedConfig, mhc_collapse, mhc_expand
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernels need a TPU")

HC = 4
MIX = (2 + HC) * HC
KW = dict(sinkhorn_iters=6, eps=1e-6, rms_eps=1e-6)


def _rel(a, b):
    """``(max |a - b| / max |a|, rms(a - b) / rms(a))`` in float64."""
    a = np.asarray(jnp.asarray(a, jnp.float32), np.float64)
    b = np.asarray(jnp.asarray(b, jnp.float32), np.float64)
    max_rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
    rms_rel = np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a**2)), 1e-30)
    return max_rel, rms_rel


def _inputs(dtype, fn_dtype=jnp.float32, batch=1, seq=384, width=256, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((batch, seq, HC, width)), dtype)
    fn = jnp.asarray(rng.standard_normal((MIX, HC, width)) * 0.02, fn_dtype)
    base = jnp.asarray(rng.standard_normal((MIX,)) * 0.1, jnp.float32)
    scale = jnp.asarray([1.0, 1.1, 0.9], jnp.float32)
    out = jnp.asarray(rng.standard_normal((batch, seq, width)), dtype)
    post = jnp.asarray(2.0 / (1.0 + np.exp(-rng.standard_normal((batch, seq, HC)))), jnp.float32)
    logits = rng.standard_normal((batch, seq, HC, HC))
    comb = jnp.asarray(np.exp(logits) / np.exp(logits).sum(-1, keepdims=True), jnp.float32)
    cts = dict(
        post=jnp.asarray(rng.standard_normal((batch, seq, HC)), jnp.float32),
        comb=jnp.asarray(rng.standard_normal((batch, seq, HC, HC)), jnp.float32),
        col=jnp.asarray(rng.standard_normal((batch, seq, width)), jnp.float32),
        new=jnp.asarray(rng.standard_normal((batch, seq, HC, width)), jnp.float32),
    )
    return x, fn, base, scale, out, post, comb, cts


def _collapse_loss(collapse, cts):
    def loss(x, fn, base, scale):
        post, comb, col = collapse(x, fn, base, scale)
        return jnp.sum(post * cts["post"]) + jnp.sum(comb * cts["comb"]) + jnp.sum(col.astype(jnp.float32) * cts["col"])

    return loss


def _expand_loss(expand, cts):
    def loss(x, out, post, comb):
        return jnp.sum(expand(x, out, post, comb).astype(jnp.float32) * cts["new"])

    return loss


def _ref_collapse(*a):
    return xla_collapse(*a, **KW)


def _pallas_collapse(tile_rows=256, **extra):
    return lambda *a: mhc_collapse_pallas(*a, tile_rows=tile_rows, **KW, **extra)


def _pallas_expand(tile_rows=256):
    return lambda *a: mhc_expand_pallas(*a, tile_rows=tile_rows)


@pytest.mark.parametrize("tile_rows", [128, 256])
def test_collapse_fp32_forward_and_gradients_match_reference(tile_rows):
    """384 rows exercise the padded last tile at both tile sizes."""
    x, fn, base, scale, _, _, _, cts = _inputs(jnp.float32)
    ref = jax.jit(_ref_collapse)(x, fn, base, scale)
    got = jax.jit(_pallas_collapse(tile_rows))(x, fn, base, scale)
    for name, a, b in zip(("post", "comb", "collapsed"), ref, got, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert _rel(a, b)[0] < 2e-5, f"{name}: {_rel(a, b)[0]:.3e}"
    argnums = (0, 1, 2, 3)
    g_ref = jax.jit(jax.grad(_collapse_loss(_ref_collapse, cts), argnums=argnums))(x, fn, base, scale)
    g_got = jax.jit(jax.grad(_collapse_loss(_pallas_collapse(tile_rows), cts), argnums=argnums))(x, fn, base, scale)
    for name, a, b in zip(("dx", "dfn", "dbase", "dscale"), g_ref, g_got, strict=True):
        assert a.dtype == b.dtype, name
        assert _rel(a, b)[0] < 2e-4, f"grad {name}: {_rel(a, b)[0]:.3e}"


@pytest.mark.parametrize("tile_rows", [128, 256])
def test_expand_fp32_forward_and_gradients_match_reference(tile_rows):
    x, _, _, _, out, post, comb, cts = _inputs(jnp.float32)
    ref = jax.jit(xla_expand)(x, out, post, comb)
    got = jax.jit(_pallas_expand(tile_rows))(x, out, post, comb)
    assert got.shape == x.shape and got.dtype == x.dtype
    assert _rel(ref, got)[0] < 1e-5, _rel(ref, got)
    argnums = (0, 1, 2, 3)
    g_ref = jax.jit(jax.grad(_expand_loss(xla_expand, cts), argnums=argnums))(x, out, post, comb)
    g_got = jax.jit(jax.grad(_expand_loss(_pallas_expand(tile_rows), cts), argnums=argnums))(x, out, post, comb)
    for name, a, b in zip(("dx", "dout", "dpost", "dcomb"), g_ref, g_got, strict=True):
        assert a.dtype == b.dtype, name
        assert _rel(a, b)[0] < 1e-5, f"grad {name}: {_rel(a, b)[0]:.3e}"


@pytest.mark.parametrize("fn_dtype", [jnp.float32, jnp.bfloat16], ids=["fn_fp32", "fn_bf16"])
def test_bf16_streams_stay_in_the_rounding_band(fn_dtype):
    """bf16 streams: the kernels round once where XLA rounds per fusion, so parity is the bf16 band (1 ulp =
    3.9e-3 relative), not bit-level."""
    x, fn, base, scale, out, post, comb, cts = _inputs(jnp.bfloat16, fn_dtype)
    ref = jax.jit(_ref_collapse)(x, fn, base, scale)
    got = jax.jit(_pallas_collapse())(x, fn, base, scale)
    ref_new = jax.jit(xla_expand)(x, out, post, comb)
    got_new = jax.jit(_pallas_expand())(x, out, post, comb)
    for name, a, b in zip(("post", "comb", "collapsed", "new"), (*ref, ref_new), (*got, got_new), strict=True):
        max_rel, rms_rel = _rel(a, b)
        assert a.dtype == b.dtype, name
        assert max_rel < 1.6e-2 and rms_rel < 5e-3, f"{name}: max-rel {max_rel:.3e} rms-rel {rms_rel:.3e}"
    argnums = (0, 1, 2, 3)
    pairs = [
        (
            ("dx", "dfn", "dbase", "dscale"),
            jax.jit(jax.grad(_collapse_loss(_ref_collapse, cts), argnums=argnums))(x, fn, base, scale),
            jax.jit(jax.grad(_collapse_loss(_pallas_collapse(), cts), argnums=argnums))(x, fn, base, scale),
        ),
        (
            ("dx_e", "dout", "dpost", "dcomb"),
            jax.jit(jax.grad(_expand_loss(xla_expand, cts), argnums=argnums))(x, out, post, comb),
            jax.jit(jax.grad(_expand_loss(_pallas_expand(), cts), argnums=argnums))(x, out, post, comb),
        ),
    ]
    for names, g_ref, g_got in pairs:
        for name, a, b in zip(names, g_ref, g_got, strict=True):
            max_rel, rms_rel = _rel(a, b)
            assert a.dtype == b.dtype, name
            assert np.isfinite(np.asarray(b, np.float32)).all(), name
            assert max_rel < 3e-2 and rms_rel < 2e-2, f"grad {name}: max-rel {max_rel:.3e} rms-rel {rms_rel:.3e}"


def test_fn_pieces_one_is_the_deliberate_bf16_rounding_of_an_fp32_predictor():
    x, fn, base, scale, *_ = _inputs(jnp.float32)
    ref = jax.jit(_ref_collapse)(x, fn, base, scale)
    exact = jax.jit(_pallas_collapse(fn_pieces=3))(x, fn, base, scale)
    coarse = jax.jit(_pallas_collapse(fn_pieces=1))(x, fn, base, scale)
    assert _rel(ref[0], exact[0])[0] < 2e-5
    assert 2e-5 < _rel(ref[0], coarse[0])[0] < 5e-2


def _mesh():
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices for the dp x tp mesh")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))


X_SPEC, FN_SPEC, OUT_SPEC = P("dp", None, None, "tp"), P(None, None, "tp"), P("dp", None, "tp")
TOKEN_SPEC, MATRIX_SPEC = P("dp", None, None), P("dp", None, None, None)


def _sharded_collapse(mesh, body):
    def collapse(x, fn, base, scale):
        return jax.shard_map(
            lambda *a: body(*a, axis_name="tp"),
            mesh=mesh,
            in_specs=(X_SPEC, FN_SPEC, P(), P()),
            out_specs=(TOKEN_SPEC, MATRIX_SPEC, OUT_SPEC),
            check_vma=False,
        )(x, fn, base, scale)

    return collapse


def _sharded_expand(mesh, body):
    def expand(x, out, post, comb):
        return jax.shard_map(
            body, mesh=mesh, in_specs=(X_SPEC, OUT_SPEC, TOKEN_SPEC, MATRIX_SPEC), out_specs=X_SPEC, check_vma=False
        )(x, out, post, comb)

    return expand


@pytest.mark.parametrize("impl_name", ["xla", "pallas"])
def test_shard_map_dp2_tp2_each_op_and_the_site_match_unsharded_reference(impl_name):
    """The hidden is split over tp: stats / logit cotangents are summed inside the ops, ``fn``'s cotangent is its
    own shard, ``base`` / ``scale`` / ``post`` / ``comb`` cotangents are per-device partials that the
    ``check_vma=False`` transpose sums. Each op is checked standalone (so a wrong convention in one op cannot be
    cancelled by the other) and in the combined site. The XLA reference under the same ``shard_map`` validates the
    convention itself."""
    mesh = _mesh()
    x, fn, base, scale, out, post, comb, cts = _inputs(jnp.float32, batch=2, seq=256, width=256)
    if impl_name == "xla":
        c_body = lambda *a, axis_name: xla_collapse(*a, axis_name=axis_name, **KW)  # noqa: E731
        e_body = xla_expand
    else:
        c_body = lambda *a, axis_name: mhc_collapse_pallas(*a, axis_name=axis_name, tile_rows=128, **KW)  # noqa: E731
        e_body = lambda *a: mhc_expand_pallas(*a, tile_rows=128)  # noqa: E731
    collapse, expand = _sharded_collapse(mesh, c_body), _sharded_expand(mesh, e_body)
    xs = jax.device_put(x, NamedSharding(mesh, X_SPEC))
    argnums = (0, 1, 2, 3)

    fwd_ref = jax.jit(_ref_collapse)(x, fn, base, scale)
    fwd_sm = jax.jit(collapse)(xs, fn, base, scale)
    for name, a, b in zip(("post", "comb", "collapsed"), fwd_ref, fwd_sm, strict=True):
        assert _rel(a, b)[0] < 2e-5, f"{name}: {_rel(a, b)[0]:.3e}"
    g_ref = jax.jit(jax.grad(_collapse_loss(_ref_collapse, cts), argnums=argnums))(x, fn, base, scale)
    g_sm = jax.jit(jax.grad(_collapse_loss(collapse, cts), argnums=argnums))(xs, fn, base, scale)
    for name, a, b in zip(("dx", "dfn", "dbase", "dscale"), g_ref, g_sm, strict=True):
        assert _rel(a, b)[0] < 2e-4, f"collapse grad {name}: {_rel(a, b)[0]:.3e}"

    assert _rel(jax.jit(xla_expand)(x, out, post, comb), jax.jit(expand)(xs, out, post, comb))[0] < 1e-5
    g_ref = jax.jit(jax.grad(_expand_loss(xla_expand, cts), argnums=argnums))(x, out, post, comb)
    g_sm = jax.jit(jax.grad(_expand_loss(expand, cts), argnums=argnums))(xs, out, post, comb)
    for name, a, b in zip(("dx", "dout", "dpost", "dcomb"), g_ref, g_sm, strict=True):
        assert _rel(a, b)[0] < 1e-5, f"expand grad {name}: {_rel(a, b)[0]:.3e}"

    def site(collapse_fn, expand_fn):
        def loss(x, fn, base, scale):
            post, comb, col = collapse_fn(x, fn, base, scale)
            return jnp.sum(expand_fn(x, jnp.tanh(col), post, comb) * cts["new"]) + jnp.sum(post * cts["post"])

        return loss

    g_ref = jax.jit(jax.grad(site(_ref_collapse, xla_expand), argnums=argnums))(x, fn, base, scale)
    g_sm = jax.jit(jax.grad(site(collapse, expand), argnums=argnums))(xs, fn, base, scale)
    for name, a, b in zip(("dx", "dfn", "dbase", "dscale"), g_ref, g_sm, strict=True):
        assert _rel(a, b)[0] < 2e-4, f"site grad {name}: {_rel(a, b)[0]:.3e}"


def test_public_ops_default_to_xla_and_honour_platform_and_cfg():
    x, fn, base, scale, out, post, comb, _ = _inputs(jnp.float32)
    assert MHCCollapse().heuristic_cfg(None).platform == "xla"
    assert MHCExpand().heuristic_cfg(None).platform == "xla"
    default = jax.jit(lambda *a: mhc_collapse(*a, **KW))(x, fn, base, scale)
    for a, b in zip(default, jax.jit(_ref_collapse)(x, fn, base, scale), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(
        np.asarray(jax.jit(mhc_expand)(x, out, post, comb)), np.asarray(jax.jit(xla_expand)(x, out, post, comb))
    )
    cfg = MHCFusedConfig(platform="pallas", tile_rows=128)
    via_cfg = jax.jit(lambda *args: mhc_collapse(*args, cfg=cfg, **KW))(x, fn, base, scale)
    direct = jax.jit(_pallas_collapse(128))(x, fn, base, scale)
    for a, b in zip(via_cfg, direct, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(
        np.asarray(jax.jit(lambda *a: mhc_expand(*a, platform="pallas"))(x, out, post, comb)),
        np.asarray(jax.jit(_pallas_expand(256))(x, out, post, comb)),
    )


def test_executor_shard_map_path_matches_reference_with_gradients():
    mesh = _mesh()
    assert "create_shard_map_wrapper" in MHCCollapse.__dict__ and "create_shard_map_wrapper" in MHCExpand.__dict__
    x, fn, base, scale, out, post, comb, cts = _inputs(jnp.float32, batch=2, seq=256, width=256)
    xs = jax.device_put(x, NamedSharding(mesh, X_SPEC))
    cfg = MHCFusedConfig(platform="pallas", tile_rows=128)

    def collapse(x, fn, base, scale):
        return mhc_collapse(
            x,
            fn,
            base,
            scale,
            cfg=cfg,
            mesh=mesh,
            in_specs=(X_SPEC, FN_SPEC, P(), P()),
            out_specs=(TOKEN_SPEC, MATRIX_SPEC, OUT_SPEC),
            **KW,
        )

    def expand(x, out, post, comb):
        specs = (X_SPEC, OUT_SPEC, TOKEN_SPEC, MATRIX_SPEC)
        return mhc_expand(x, out, post, comb, cfg=cfg, mesh=mesh, in_specs=specs, out_specs=X_SPEC)

    argnums = (0, 1, 2, 3)
    g_ref = jax.jit(jax.grad(_collapse_loss(_ref_collapse, cts), argnums=argnums))(x, fn, base, scale)
    g_got = jax.jit(jax.grad(_collapse_loss(collapse, cts), argnums=argnums))(xs, fn, base, scale)
    for name, a, b in zip(("dx", "dfn", "dbase", "dscale"), g_ref, g_got, strict=True):
        assert _rel(a, b)[0] < 2e-4, f"collapse grad {name}: {_rel(a, b)[0]:.3e}"
    g_ref = jax.jit(jax.grad(_expand_loss(xla_expand, cts), argnums=argnums))(x, out, post, comb)
    g_got = jax.jit(jax.grad(_expand_loss(expand, cts), argnums=argnums))(xs, out, post, comb)
    for name, a, b in zip(("dx", "dout", "dpost", "dcomb"), g_ref, g_got, strict=True):
        assert _rel(a, b)[0] < 1e-5, f"expand grad {name}: {_rel(a, b)[0]:.3e}"
    with pytest.raises(ValueError, match="check_vma=False"):
        MHCExpand().create_shard_map_wrapper(
            x, out, post, comb, cfg=cfg, mesh=mesh, in_specs=(X_SPEC,) * 4, out_specs=X_SPEC, check_vma=True
        )


def test_mosaic_compiles_and_runs_long_rows_at_production_width():
    """``[1, 4096, 4, 1024]`` bf16 streams with an fp32 predictor at tile 256: every forward and backward kernel
    compiles on the chip, runs, and stays in the bf16 band of the reference."""
    x, fn, base, scale, out, post, comb, cts = _inputs(jnp.bfloat16, jnp.float32, seq=4096, width=1024)
    argnums = (0, 1, 2, 3)
    ref_c = jax.jit(jax.value_and_grad(_collapse_loss(_ref_collapse, cts), argnums=argnums))(x, fn, base, scale)
    got_c = jax.jit(jax.value_and_grad(_collapse_loss(_pallas_collapse(256), cts), argnums=argnums))(x, fn, base, scale)
    ref_e = jax.jit(jax.value_and_grad(_expand_loss(xla_expand, cts), argnums=argnums))(x, out, post, comb)
    got_e = jax.jit(jax.value_and_grad(_expand_loss(_pallas_expand(256), cts), argnums=argnums))(x, out, post, comb)
    for (lr, gr), (lg, gg) in ((ref_c, got_c), (ref_e, got_e)):
        assert abs(float(lr) - float(lg)) <= 2e-2 * abs(float(lr)) + 1.0
        for a, b in zip(gr, gg, strict=True):
            assert np.isfinite(np.asarray(b, np.float32)).all()
            max_rel, rms_rel = _rel(a, b)
            assert max_rel < 3e-2 and rms_rel < 2e-2, f"max-rel {max_rel:.3e} rms-rel {rms_rel:.3e}"


def test_vmem_request_on_this_part():
    capacity = impl._vmem_capacity()
    need_256 = impl.vmem_estimate_bytes(256, HC, 1024, 2, fn_pieces=3)
    assert 20 * 2**20 < need_256 < 40 * 2**20, need_256 / 2**20
    if capacity <= 16 * 2**20:
        assert impl.vmem_limit_bytes(256, HC, 1024, 2) is None
        return
    cap = capacity - 2**20
    assert impl.vmem_limit_bytes(256, HC, 1024, 2) == min(cap, 2 * need_256)
    need_128 = impl.vmem_estimate_bytes(128, HC, 1024, 2)
    assert impl.vmem_limit_bytes(128, HC, 1024, 2) == min(cap, max(16 * 2**20, 2 * need_128))
    if impl.vmem_estimate_bytes(1024, HC, 1024, 2) > cap:
        with pytest.raises(ValueError, match="tile_rows=1024"):
            impl.vmem_limit_bytes(1024, HC, 1024, 2)


def test_bf16_operands_are_one_piece_and_fp32_splits_exactly():
    assert impl.n_pieces(jnp.bfloat16, 3) == 1 and impl.n_pieces(jnp.float32, 3) == 3
    v = jnp.asarray(np.random.default_rng(0).standard_normal((8, 128)), jnp.float32)
    back = sum(p.astype(jnp.float32) for p in impl._split_pieces(v, 3))
    assert float(jnp.abs(back - v).max()) < 1e-6


def test_invalid_shapes_and_arguments_are_refused():
    x, fn, base, scale, out, post, comb, _ = _inputs(jnp.float32, seq=128, width=256)
    with pytest.raises(ValueError, match="tile_rows"):
        mhc_collapse_pallas(x, fn, base, scale, tile_rows=100, **KW)
    with pytest.raises(ValueError, match="multiple of 128"):
        mhc_expand_pallas(x[..., :200], out[..., :200], post, comb)
    with pytest.raises(ValueError, match="fn must be"):
        mhc_collapse_pallas(x, fn[:, :, :128], base, scale, **KW)
    with pytest.raises(ValueError, match="sinkhorn_iters"):
        mhc_collapse_pallas(x, fn, base, scale, sinkhorn_iters=0)
    with pytest.raises(ValueError, match="do not match"):
        mhc_expand_pallas(x, out, post[:, :, :2], comb)
    with pytest.raises(ValueError, match="hc="):
        mhc_collapse_pallas(jnp.zeros((1, 128, 12, 128)), jnp.zeros((168, 12, 128)), jnp.zeros((168,)), scale, **KW)


def test_registry_entries_and_signatures():
    for op in ("mhc_collapse", "mhc_expand"):
        assert kernel_registry.get(op, platform=Platform.PALLAS, backend=Backend.TPU) is not None
        assert kernel_registry.get(op, platform=Platform.XLA, backend=Backend.ANY) is not None
        assert kernel_registry.validate_signatures(op)
