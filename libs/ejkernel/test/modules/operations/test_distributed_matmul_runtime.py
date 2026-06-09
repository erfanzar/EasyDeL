from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, shard_map
from jax.sharding import Mesh, PartitionSpec

from ejkernel.modules.operations import all_gather_matmul, reduce_scatter_matmul
from ejkernel.modules.operations.configs import AllGatherMatmulConfig, ReduceScatterMatmulConfig


def _has_multi_tpu_on_host() -> bool:
    try:
        return len(jax.local_devices(backend="tpu")) >= 2
    except Exception:
        return False


def _all_tpu_devices() -> list[jax.Device]:
    return list(jax.local_devices(backend="tpu"))


def _candidate_tp_sizes() -> tuple[int, ...]:
    try:
        n = len(_all_tpu_devices())
    except Exception:
        return ()
    if n < 2:
        return ()
    sizes = [s for s in (2, 4, 8, 16) if s <= n]
    if n not in sizes:
        sizes.append(n)
    return tuple(sorted(set(sizes)))


_TP_SIZES = _candidate_tp_sizes()
_TP_SIZES_SMALL = _TP_SIZES[:2] if len(_TP_SIZES) > 2 else _TP_SIZES
_TP_SIZES_ONE = _TP_SIZES[:1]
_AG_SHAPE_BLOCK_CASES = (
    (64, 128, 64, 128, 128),
    (96, 256, 96, 96, 64),
)
_RS_SHAPE_BLOCK_CASES = (
    (64, 64, 128, 128, 128, 128),
    (128, 128, 192, 64, 96, 64),
)
_AG_SHAPE_BLOCK_CASES_ONE = _AG_SHAPE_BLOCK_CASES[:1]
_RS_SHAPE_BLOCK_CASES_ONE = _RS_SHAPE_BLOCK_CASES[:1]

pytestmark = pytest.mark.skipif(
    not _has_multi_tpu_on_host(),
    reason="Distributed mesh runtime tests require >=2 TPU devices on this host.",
)


def _mesh_for_tp(tp: int) -> Mesh:
    devices = np.array(_all_tpu_devices()[:tp])
    return Mesh(devices, axis_names=("tp",))


@pytest.mark.parametrize("tp", _TP_SIZES)
@pytest.mark.parametrize("rhs_transpose", [False, True])
@pytest.mark.parametrize(("m_local", "k", "n_local", "block_n", "block_k"), _AG_SHAPE_BLOCK_CASES)
def test_all_gather_matmul_module_mesh_sizes_xla_matches_collective_reference(
    tp: int,
    rhs_transpose: bool,
    m_local: int,
    k: int,
    n_local: int,
    block_n: int,
    block_k: int,
):
    mesh = _mesh_for_tp(tp)
    m_total = m_local * tp
    n_total = n_local * tp

    x_ids = jnp.repeat(jnp.arange(tp, dtype=jnp.float32), m_local).reshape(m_total, 1)
    x = jnp.broadcast_to(x_ids + 1.0, (m_total, k)).astype(jnp.bfloat16)

    key = jax.random.PRNGKey(1000 + tp + int(rhs_transpose))
    if rhs_transpose:
        y = jax.random.normal(key, (n_total, k), dtype=jnp.bfloat16)
        y_spec = PartitionSpec("tp", None)
    else:
        y = jax.random.normal(key, (k, n_total), dtype=jnp.bfloat16)
        y_spec = PartitionSpec(None, "tp")

    in_specs = (PartitionSpec("tp", None), y_spec)
    out_specs = PartitionSpec(None, "tp")

    out = all_gather_matmul(
        x,
        y,
        "tp",
        rhs_transpose=rhs_transpose,
        platform="xla",
        cfg=AllGatherMatmulConfig(
            block_n=block_n,
            block_k=block_k,
            platform="xla",
        ),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    def _reference(x_shard, y_shard):
        x_full = lax.all_gather(x_shard, axis_name="tp", axis=0, tiled=True)
        if rhs_transpose:
            return jnp.dot(x_full, y_shard.T)
        return jnp.dot(x_full, y_shard)

    ref = shard_map(
        _reference,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    expected = ref(x, y)

    out = jax.block_until_ready(out)
    expected = jax.block_until_ready(expected)
    assert out.shape == (m_total, n_total)
    assert jnp.allclose(out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tp", _TP_SIZES)
@pytest.mark.parametrize(("m_local", "k_shard", "n", "block_m", "block_n", "block_k"), _RS_SHAPE_BLOCK_CASES)
def test_reduce_scatter_matmul_module_mesh_sizes_xla_matches_collective_reference(
    tp: int,
    m_local: int,
    k_shard: int,
    n: int,
    block_m: int,
    block_n: int,
    block_k: int,
):
    mesh = _mesh_for_tp(tp)
    m_total = m_local * tp
    k_total = k_shard * tp

    x_ids = jnp.repeat(jnp.arange(tp, dtype=jnp.float32), m_local).reshape(m_total, 1)
    x = jnp.broadcast_to(x_ids + 1.0, (m_total, k_total)).astype(jnp.float32)
    key = jax.random.PRNGKey(2000 + tp)
    y = jax.random.normal(key, (n, k_total), dtype=jnp.float32)

    in_specs = (PartitionSpec(None, "tp"), PartitionSpec(None, "tp"))
    out_specs = PartitionSpec("tp", None)

    out = reduce_scatter_matmul(
        x,
        y,
        "tp",
        platform="xla",
        cfg=ReduceScatterMatmulConfig(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            platform="xla",
        ),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    def _reference(x_shard, y_shard):
        partial_out = jnp.dot(x_shard, y_shard.T)
        return lax.psum_scatter(partial_out, axis_name="tp", scatter_dimension=0, tiled=True)

    ref = shard_map(
        _reference,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    expected = ref(x, y)

    out = jax.block_until_ready(out)
    expected = jax.block_until_ready(expected)
    assert out.shape == (m_total, n)
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp", _TP_SIZES_ONE)
@pytest.mark.parametrize(("m_local", "k_shard", "n", "block_m", "block_n", "block_k"), _RS_SHAPE_BLOCK_CASES_ONE)
def test_reduce_scatter_matmul_module_mesh_sizes_tp_size_none_matches_explicit(
    tp: int,
    m_local: int,
    k_shard: int,
    n: int,
    block_m: int,
    block_n: int,
    block_k: int,
):
    mesh = _mesh_for_tp(tp)
    m_total = m_local * tp
    k_total = k_shard * tp

    key = jax.random.PRNGKey(3000 + tp)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m_total, k_total), dtype=jnp.float32)
    y = jax.random.normal(ky, (n, k_total), dtype=jnp.float32)

    in_specs = (PartitionSpec(None, "tp"), PartitionSpec(None, "tp"))
    out_specs = PartitionSpec("tp", None)
    cfg = ReduceScatterMatmulConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        platform="xla",
    )

    out_none = reduce_scatter_matmul(
        x,
        y,
        "tp",
        tp_size=None,
        platform="xla",
        cfg=cfg,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )
    out_explicit = reduce_scatter_matmul(
        x,
        y,
        "tp",
        tp_size=tp,
        platform="xla",
        cfg=cfg,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    out_none = jax.block_until_ready(out_none)
    out_explicit = jax.block_until_ready(out_explicit)
    assert out_none.shape == out_explicit.shape
    assert jnp.allclose(out_none, out_explicit, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp", _TP_SIZES_ONE)
@pytest.mark.parametrize("rhs_transpose", [False, True])
@pytest.mark.parametrize(("m_local", "k", "n_local", "block_n", "block_k"), _AG_SHAPE_BLOCK_CASES_ONE)
def test_all_gather_matmul_module_mesh_sizes_tp_size_none_matches_explicit(
    tp: int,
    rhs_transpose: bool,
    m_local: int,
    k: int,
    n_local: int,
    block_n: int,
    block_k: int,
):
    mesh = _mesh_for_tp(tp)
    m_total = m_local * tp
    n_total = n_local * tp

    key = jax.random.PRNGKey(3500 + tp + int(rhs_transpose))
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m_total, k), dtype=jnp.float32)
    if rhs_transpose:
        y = jax.random.normal(ky, (n_total, k), dtype=jnp.float32)
        y_spec = PartitionSpec("tp", None)
    else:
        y = jax.random.normal(ky, (k, n_total), dtype=jnp.float32)
        y_spec = PartitionSpec(None, "tp")

    in_specs = (PartitionSpec("tp", None), y_spec)
    out_specs = PartitionSpec(None, "tp")
    cfg = AllGatherMatmulConfig(
        block_n=block_n,
        block_k=block_k,
        platform="xla",
    )

    out_none = all_gather_matmul(
        x,
        y,
        "tp",
        rhs_transpose=rhs_transpose,
        tp_size=None,
        platform="xla",
        cfg=cfg,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )
    out_explicit = all_gather_matmul(
        x,
        y,
        "tp",
        rhs_transpose=rhs_transpose,
        tp_size=tp,
        platform="xla",
        cfg=cfg,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    out_none = jax.block_until_ready(out_none)
    out_explicit = jax.block_until_ready(out_explicit)
    assert out_none.shape == out_explicit.shape
    assert jnp.allclose(out_none, out_explicit, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp", _TP_SIZES_ONE)
def test_all_gather_matmul_module_mesh_runtime_auto_matches_xla_and_backward(tp: int):
    mesh = _mesh_for_tp(tp)

    m_local = 128
    m_total = m_local * tp
    k = 128
    n_local = 128
    n_total = n_local * tp

    key = jax.random.PRNGKey(0 + tp)
    kx, ky = jax.random.split(key)
    x_bf16 = jax.random.normal(kx, (m_total, k), dtype=jnp.bfloat16)
    y_bf16 = jax.random.normal(ky, (k, n_total), dtype=jnp.bfloat16)

    in_specs = (PartitionSpec("tp", None), PartitionSpec(None, "tp"))
    out_specs = PartitionSpec(None, "tp")

    out_auto = all_gather_matmul(
        x_bf16,
        y_bf16,
        "tp",
        cfg=AllGatherMatmulConfig(platform="auto"),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )
    out_xla = all_gather_matmul(
        x_bf16,
        y_bf16,
        "tp",
        platform="xla",
        cfg=AllGatherMatmulConfig(platform="xla"),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    out_auto = jax.block_until_ready(out_auto)
    out_xla = jax.block_until_ready(out_xla)
    assert out_auto.shape == out_xla.shape
    assert jnp.allclose(out_auto, out_xla, rtol=1e-2, atol=1e-2)

    x_f32 = x_bf16.astype(jnp.float32)
    y_f32 = y_bf16.astype(jnp.float32)

    def loss_fn(x, y):
        out = all_gather_matmul(
            x,
            y,
            "tp",
            cfg=AllGatherMatmulConfig(platform="auto"),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )
        return jnp.mean(out.astype(jnp.float32) ** 2)

    dx, dy = jax.grad(loss_fn, argnums=(0, 1))(x_f32, y_f32)
    dx, dy = jax.block_until_ready(dx), jax.block_until_ready(dy)
    assert dx.shape == x_f32.shape
    assert dy.shape == y_f32.shape
    assert jnp.isfinite(dx).all()
    assert jnp.isfinite(dy).all()


@pytest.mark.parametrize("tp", _TP_SIZES_ONE)
def test_reduce_scatter_matmul_module_mesh_runtime_auto_matches_xla_and_backward(tp: int):
    mesh = _mesh_for_tp(tp)

    m_local = 128
    m_total = m_local * tp
    k_shard = 128
    k_total = k_shard * tp
    n = 256

    key = jax.random.PRNGKey(1 + tp)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (m_total, k_total), dtype=jnp.float32)
    y = jax.random.normal(ky, (n, k_total), dtype=jnp.float32)

    in_specs = (PartitionSpec(None, "tp"), PartitionSpec(None, "tp"))
    out_specs = PartitionSpec("tp", None)

    out_auto = reduce_scatter_matmul(
        x,
        y,
        "tp",
        cfg=ReduceScatterMatmulConfig(platform="auto"),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )
    out_xla = reduce_scatter_matmul(
        x,
        y,
        "tp",
        platform="xla",
        cfg=ReduceScatterMatmulConfig(platform="xla"),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    out_auto = jax.block_until_ready(out_auto)
    out_xla = jax.block_until_ready(out_xla)
    assert out_auto.shape == out_xla.shape
    assert jnp.allclose(out_auto, out_xla, rtol=1e-3, atol=1e-3)

    def loss_fn(x, y):
        out = reduce_scatter_matmul(
            x,
            y,
            "tp",
            cfg=ReduceScatterMatmulConfig(platform="auto"),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )
        return jnp.mean(out.astype(jnp.float32) ** 2)

    dx, dy = jax.grad(loss_fn, argnums=(0, 1))(x, y)
    dx, dy = jax.block_until_ready(dx), jax.block_until_ready(dy)
    assert dx.shape == x.shape
    assert dy.shape == y.shape
    assert jnp.isfinite(dx).all()
    assert jnp.isfinite(dy).all()
