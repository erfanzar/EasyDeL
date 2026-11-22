"""TPU regression: int4 weights with a non-MXU-aligned contraction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul
from ejkernel.modules.operations import GroupedMatmulConfig


@pytest.mark.parametrize("k", [128, 160, 192, 256, 288, 384])
def test_int4_tail_matches_dequantized_reference(k):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU Mosaic lowering")
    rng = np.random.default_rng(22)
    # Existing v3 dispatch dynamically quantizes lhs to INT8. Use unit-scale
    # exactly representable rows to isolate the contraction-tail regression.
    a_np = rng.integers(-64, 65, size=(16, k)).astype(np.float32)
    a_np[:, 0] = 127
    w_np = rng.integers(-7, 8, size=(4, k, 128), dtype=np.int8)
    a = jnp.asarray(a_np, jnp.bfloat16)
    w = jnp.asarray(w_np).astype(jnp.int4)
    scales = jnp.ones((4, 1, 1, 128), jnp.bfloat16) * 0.125
    sizes = jnp.array([5, 0, 7, 4], jnp.int32)
    cfg = GroupedMatmulConfig(block_m=8, block_k=0, block_n=0)
    f = jax.jit(
        lambda x, y, s, g: grouped_matmul(
            x, y, g, rhs_scale=s, preferred_element_type=jnp.bfloat16, use_v3=True, platform="pallas", cfg=cfg
        )
    )
    got = np.asarray(f(a, w, scales, sizes)).astype(np.float32)
    refs = []
    start = 0
    for g, count in enumerate([5, 0, 7, 4]):
        refs.append(np.asarray(a).astype(np.float32)[start : start + count] @ (w_np[g].astype(np.float32) * 0.125))
        start += count
    want = np.asarray(jnp.asarray(np.concatenate(refs), jnp.bfloat16)).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=0.02, atol=0.04)
    assert got.shape == (16, 128)
