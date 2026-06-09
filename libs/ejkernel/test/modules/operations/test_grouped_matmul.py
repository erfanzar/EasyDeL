from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import grouped_matmul

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
