from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules.operations import grouped_matmul
from ejkernel.modules.operations.configs import GroupedMatmulConfig

from ._utils import assert_allclose


def _grouped_matmul_ref(
    lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, transpose_rhs: bool, existing_out: jax.Array | None
):
    sizes = [int(x) for x in list(group_sizes)]
    offset = 0
    chunks = []
    for g, sz in enumerate(sizes):
        a = lhs[offset : offset + sz]
        b = rhs[g].T if transpose_rhs else rhs[g]
        chunks.append(a @ b)
        offset += sz
    out = (
        jnp.concatenate(chunks, axis=0)
        if chunks
        else jnp.zeros((0, rhs.shape[2] if not transpose_rhs else rhs.shape[1]))
    )
    if existing_out is not None:
        out = out + existing_out
    return out


def _grouped_matmul_v3_ref(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    transpose_rhs: bool,
    existing_out: jax.Array | None,
):
    rhs_prepped = rhs.transpose(0, 2, 1) if transpose_rhs else rhs
    if rhs_scale is not None:
        num_blocks = int(rhs_scale.shape[1])
        block_size = rhs_prepped.shape[1] // num_blocks
        scale = jnp.repeat(rhs_scale[:, :, 0, :], block_size, axis=1)
        rhs_prepped = rhs_prepped * scale.astype(rhs_prepped.dtype)

    group_ids = jnp.repeat(
        jnp.arange(group_sizes.shape[0], dtype=group_sizes.dtype),
        group_sizes,
        total_repeat_length=lhs.shape[0],
    )
    out = jax.vmap(lambda row, mat: jnp.matmul(row, mat, preferred_element_type=jnp.float32))(
        lhs, rhs_prepped[group_ids]
    )
    if rhs_bias is not None:
        out = out + rhs_bias[:, 0, :][group_ids].astype(out.dtype)
    if existing_out is not None:
        out = out + existing_out
    return out


def test_grouped_matmul_matches_reference_basic_and_transpose_rhs():
    M, K, N = 32, 16, 8
    group_sizes = jnp.array([16, 16], dtype=jnp.int32)
    lhs = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.PRNGKey(1), (2, K, N), dtype=jnp.float32)

    out = grouped_matmul(lhs, rhs, group_sizes, platform="xla")
    ref = _grouped_matmul_ref(lhs, rhs, group_sizes, transpose_rhs=False, existing_out=None)
    assert out.shape == (M, N)
    assert_allclose(out, ref, atol=0.2)

    rhs_t = rhs.transpose(0, 2, 1)
    out_t = grouped_matmul(lhs, rhs_t, group_sizes, transpose_rhs=True, platform="xla")
    ref_t = _grouped_matmul_ref(lhs, rhs_t, group_sizes, transpose_rhs=True, existing_out=None)
    assert_allclose(out_t, ref_t, atol=0.2)


def test_grouped_matmul_variable_sizes_existing_out_and_v2():
    M, K, N = 40, 16, 8
    group_sizes = jnp.array([10, 6, 24], dtype=jnp.int32)
    lhs = jax.random.normal(jax.random.PRNGKey(2), (M, K), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.PRNGKey(3), (3, K, N), dtype=jnp.float32)
    existing = jax.random.normal(jax.random.PRNGKey(4), (M, N), dtype=jnp.float32)

    out = grouped_matmul(lhs, rhs, group_sizes, None, existing, do_padding=False, platform="xla")
    ref = _grouped_matmul_ref(lhs, rhs, group_sizes, transpose_rhs=False, existing_out=existing)
    assert_allclose(out, ref, atol=0.25)

    out_v2 = grouped_matmul(lhs, rhs, group_sizes, None, existing, do_padding=False, use_v2=True, platform="xla")
    assert_allclose(out_v2, ref, atol=0.25)

    out_v3 = grouped_matmul(lhs, rhs, group_sizes, None, existing, do_padding=False, use_v3=True, platform="xla")
    assert_allclose(out_v3, ref, atol=0.25)


def test_grouped_matmul_v3_supports_rhs_scale_and_bias():
    M, K, N = 20, 8, 6
    group_sizes = jnp.array([8, 12], dtype=jnp.int32)
    lhs = jax.random.normal(jax.random.PRNGKey(7), (M, K), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.PRNGKey(8), (2, K, N), dtype=jnp.float32)
    rhs_scale = jax.random.normal(jax.random.PRNGKey(9), (2, 2, 1, N), dtype=jnp.float32)
    rhs_bias = jax.random.normal(jax.random.PRNGKey(10), (2, 1, N), dtype=jnp.float32)
    existing = jax.random.normal(jax.random.PRNGKey(11), (M, N), dtype=jnp.float32)

    out = grouped_matmul(
        lhs,
        rhs,
        group_sizes,
        None,
        existing,
        do_padding=False,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        use_v3=True,
        platform="xla",
    )
    ref = _grouped_matmul_v3_ref(
        lhs,
        rhs,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        transpose_rhs=False,
        existing_out=existing,
    )
    assert_allclose(out, ref, atol=1e-5)


def test_grouped_matmul_interpret_mode_runs():
    M, K, N = 16, 8, 4
    group_sizes = jnp.array([8, 8], dtype=jnp.int32)
    lhs = jax.random.normal(jax.random.PRNGKey(5), (M, K), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.PRNGKey(6), (2, K, N), dtype=jnp.float32)

    out = grouped_matmul(lhs, rhs, group_sizes, interpret=True, platform="xla")
    assert out.shape == (M, N)


def test_grouped_matmul_rejects_conflicting_version_flags():
    lhs = jnp.ones((4, 4), dtype=jnp.float32)
    rhs = jnp.ones((1, 4, 4), dtype=jnp.float32)
    group_sizes = jnp.array([4], dtype=jnp.int32)

    try:
        grouped_matmul(lhs, rhs, group_sizes, use_v2=True, use_v3=True, platform="xla")
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected ValueError when both use_v2 and use_v3 are enabled.")


def test_grouped_matmul_rejects_rhs_scale_without_v3():
    lhs = jnp.ones((4, 4), dtype=jnp.float32)
    rhs = jnp.ones((1, 4, 4), dtype=jnp.float32)
    group_sizes = jnp.array([4], dtype=jnp.int32)
    rhs_scale = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)

    try:
        grouped_matmul(lhs, rhs, group_sizes, rhs_scale=rhs_scale, platform="xla")
    except ValueError as exc:
        assert "grouped_matmulv3" in str(exc)
    else:
        raise AssertionError("Expected ValueError when rhs_scale is used without grouped_matmulv3.")


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_grouped_matmul_xla_bypass_preserves_ragged_rows(use_v2, transpose_rhs):
    """The public bypass path still gets backend safety padding, not tile hints."""
    m, k, n = 24, 8, 4
    lhs = ((jnp.arange(m * k).reshape(m, k) % 5) - 2).astype(jnp.float32) / 8
    rhs = ((jnp.arange(4 * k * n).reshape(4, k, n) % 7) - 3).astype(jnp.float32) / 8
    if transpose_rhs:
        rhs = rhs.swapaxes(1, 2)
    existing = ((jnp.arange(m * n).reshape(m, n) % 3) - 1).astype(jnp.float32) / 16
    cfg = GroupedMatmulConfig(bypass_xla_tiling=True)

    @jax.jit
    def run(lhs, rhs, sizes, existing):
        return grouped_matmul(
            lhs,
            rhs,
            sizes,
            None,
            existing,
            preferred_element_type=jnp.bfloat16,
            precision=jax.lax.Precision.DEFAULT,
            transpose_rhs=transpose_rhs,
            platform="xla",
            cfg=cfg,
            use_v2=use_v2,
        )

    for sizes in ((3, 0, 21, 0), (0, 19, 0, 5)):
        out = run(lhs, rhs, jnp.asarray(sizes, dtype=jnp.int32), existing)
        start = 0
        chunks = []
        for expert, rows in enumerate(sizes):
            if rows:
                weight = rhs[expert].T if transpose_rhs else rhs[expert]
                chunks.append(
                    jnp.matmul(
                        lhs[start : start + rows],
                        weight,
                        precision=jax.lax.Precision.DEFAULT,
                        preferred_element_type=jnp.bfloat16,
                    )
                )
            start += rows
        expected = jnp.concatenate(chunks, axis=0) + existing.astype(jnp.bfloat16)
        assert out.shape == (m, n)
        assert out.dtype == jnp.bfloat16
        assert np.isfinite(np.asarray(out, dtype=np.float32)).all()
        np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


def test_grouped_matmul_xla_bypass_tp4_global_rows_match_numpy():
    """Global-jit TP4 gate-up preserves all 120 rows and output-column placement.

    Keep the global RHS width at 1024: slicing it to 256 before the call would
    only test an unsharded local-shaped operation, not SPMD partitioning. This
    exercises the public untiled XLA path with float32 operands and bf16 output.
    """
    devices = jax.local_devices()
    if len(devices) < 4:
        pytest.skip("requires at least four local devices for TP4 grouped matmul")

    mesh = jax.sharding.Mesh(
        np.asarray(devices[:4]).reshape(1, 1, 1, 1, 4, 1),
        ("pp", "dp", "fsdp", "ep", "tp", "sp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 6,
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rhs_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, None, "tp"))
    out_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tp"))

    m, k, n, groups = 120, 64, 1024, 512
    rng = np.random.default_rng(79)
    # Products sum to integer multiples of 1/64 in [-1, 1], exactly representable
    # in bf16. This isolates row/group placement from DEFAULT-precision drift.
    lhs_host = rng.integers(-1, 2, (m, k), dtype=np.int8).astype(np.float32) / np.float32(8)
    rhs_host = rng.integers(-1, 2, (groups, k, n), dtype=np.int8).astype(np.float32) / np.float32(8)
    sizes_host = np.zeros(groups, dtype=np.int32)
    sizes_host[np.arange(0, 100, 10)] = 12

    expected = np.zeros((m, n), dtype=np.float32)
    start = 0
    for expert, size in enumerate(sizes_host):
        end = start + int(size)
        if size:
            expected[start:end] = lhs_host[start:end] @ rhs_host[expert]
        start = end
    assert start == m

    cfg = GroupedMatmulConfig(bypass_xla_tiling=True)

    def run(lhs, rhs, sizes):
        return grouped_matmul(
            lhs,
            rhs,
            sizes,
            preferred_element_type=jnp.bfloat16,
            precision=jax.lax.Precision.DEFAULT,
            platform="xla",
            cfg=cfg,
        )

    compiled = jax.jit(
        run,
        in_shardings=(replicated, rhs_sharding, replicated),
        out_shardings=out_sharding,
    )
    with mesh:
        out = compiled(
            jax.device_put(lhs_host, replicated),
            jax.device_put(rhs_host, rhs_sharding),
            jax.device_put(sizes_host, replicated),
        )
        out.block_until_ready()

    assert out.shape == (m, n)
    assert out.dtype == jnp.bfloat16
    assert out.sharding.is_equivalent_to(out_sharding, ndim=2)
    assert len(out.addressable_shards) == 4
    assert all(shard.data.shape == (m, n // 4) for shard in out.addressable_shards)
    actual = np.asarray(out, dtype=np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("sizes", [(4, 8), (4, 6, 10, 4)])
@pytest.mark.parametrize("output_dtype", [jnp.bfloat16, jnp.float32])
def test_grouped_matmul_eager_tp4_narrow_columns_match_numpy(sizes, output_dtype):
    """Eager global-array dispatch must populate every narrow TP output shard.

    An enclosing test jit hides a native TPU eager pad/dot/slice failure on
    JAX 0.11.2: finite operands yield corrupt output values. Keep this
    call eager, including the partial 32-row tile and 32-column local output.
    Dyadic operands make the independent reference exact at DEFAULT precision.
    """
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires four devices for narrow TP4 output shards")
    mesh = jax.sharding.Mesh(
        np.asarray(devices[:4]).reshape(1, 1, 1, 1, 4, 1),
        ("pp", "dp", "fsdp", "ep", "tp", "sp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 6,
    )
    lhs_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tp"))
    rhs_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("ep", None, "tp"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    output_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tp"))
    rng = np.random.default_rng(132)
    lhs = rng.integers(-1, 2, size=(sum(sizes), 64)).astype(np.float32) / 8
    rhs = rng.integers(-1, 2, size=(len(sizes), 64, 128)).astype(np.float32) / 8
    expected = np.concatenate([lhs[sum(sizes[:i]) : sum(sizes[: i + 1])] @ rhs[i] for i in range(len(sizes))], axis=0)
    with mesh:
        out = grouped_matmul(
            jax.device_put(lhs, lhs_sharding),
            jax.device_put(rhs, rhs_sharding),
            jax.device_put(np.asarray(sizes, dtype=np.int32), replicated),
            preferred_element_type=output_dtype,
            precision=jax.lax.Precision.DEFAULT,
            platform="xla",
            cfg=GroupedMatmulConfig(bypass_xla_tiling=True),
        )
        actual = np.asarray(out, dtype=np.float32)
    assert out.shape == expected.shape
    assert out.dtype == output_dtype
    assert out.sharding.is_equivalent_to(output_sharding, ndim=2)
    assert all(shard.data.shape == (sum(sizes), 32) for shard in out.addressable_shards)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
