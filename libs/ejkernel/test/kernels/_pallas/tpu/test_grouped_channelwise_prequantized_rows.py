"""A8 rows quantized before a dispatch gather (``lhs_scale``) match the bf16-row grouped matmul bit for bit on TPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise, grouped_matmul_channelwise_quantize_rows

E, K, N, TOKENS, TOP_K = 8, 512, 1024, 16, 4


def _inputs(local_rows):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    h32 = jax.random.normal(k1, (TOKENS, K), jnp.float32) * 3
    codes = jax.random.randint(k2, (E, K, N), -127, 128, jnp.int32).astype(jnp.int8)
    scales = jax.random.uniform(k3, (E, 1, N), jnp.float32, 1e-3, 2e-3)
    sizes = jnp.asarray(np.random.RandomState(0).multinomial(local_rows, np.ones(E) / E), jnp.int32)
    src = (jax.random.permutation(k4, TOKENS * TOP_K) // TOP_K).astype(jnp.int32)
    return h32, codes, scales, sizes, src


@pytest.mark.parametrize("out_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("local_rows", [TOKENS * TOP_K, 40, 0])
@pytest.mark.parametrize("platform", ["xla", "pallas"])
def test_prequantized_rows_are_bitwise(platform, local_rows, out_dtype):
    if jax.default_backend() != "tpu":
        pytest.skip("real TPU required")
    h32, codes, scales, sizes, src = _inputs(local_rows)

    @jax.jit
    def gathered_rows(h, w, s, g, idx):
        # The f32 -> bf16 cast sits right before the row quantization: the excess-precision hazard of the new path.
        x = (h * 1.5).astype(jnp.bfloat16)
        return grouped_matmul_channelwise(
            x[idx], w, s, g, activation_bits=8, platform=platform, preferred_element_type=out_dtype
        )

    @jax.jit
    def prequantized_rows(h, w, s, g, idx):
        x = (h * 1.5).astype(jnp.bfloat16)
        q, scale = grouped_matmul_channelwise_quantize_rows(x, 8)
        return grouped_matmul_channelwise(
            q[idx], w, s, g, activation_bits=8, platform=platform, preferred_element_type=out_dtype, lhs_scale=scale[idx]
        )

    want = np.asarray(gathered_rows(h32, codes, scales, sizes, src).astype(jnp.float32))
    got = np.asarray(prequantized_rows(h32, codes, scales, sizes, src).astype(jnp.float32))
    np.testing.assert_array_equal(got, want)
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got[local_rows:], 0)
    if local_rows:
        assert float(np.abs(got[:local_rows]).max()) > 0
