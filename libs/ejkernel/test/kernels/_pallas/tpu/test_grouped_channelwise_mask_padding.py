"""``grouped_matmul_channelwise(mask_padding=False)``: routed rows bitwise equal to the default call (TPU only)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU entry")


@pytest.mark.parametrize("platform", ["xla", "pallas"])
@pytest.mark.parametrize("out_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("rows,routed", [(256, 200), (20480, 16000), (81920, 65000)])
def test_mask_padding_false_routed_rows_bitwise(platform, out_dtype, rows, routed):
    e, k, n = 8, 512, 1024
    key = jax.random.PRNGKey(rows)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = (jax.random.normal(k1, (rows, k), jnp.float32) * 3).astype(jnp.bfloat16)
    codes = jax.random.randint(k2, (e, k, n), -127, 128, jnp.int32).astype(jnp.int8)
    scales = jax.random.uniform(k3, (e, 1, n), jnp.float32, 1e-3, 1e-2)
    cuts = jnp.sort(jax.random.randint(k4, (e - 1,), 0, routed + 1))
    sizes = jnp.diff(jnp.concatenate([jnp.zeros(1, jnp.int32), cuts, jnp.full(1, routed, jnp.int32)])).astype(jnp.int32)

    def call(mask_padding):
        return jax.jit(
            lambda a, b, c, d: grouped_matmul_channelwise(
                a,
                b,
                c,
                d,
                activation_bits=8,
                preferred_element_type=out_dtype,
                platform=platform,
                mask_padding=mask_padding,
            )
        )(x, codes, scales, sizes)

    ref = np.asarray(call(True).astype(jnp.float32))
    out = np.asarray(call(False).astype(jnp.float32))
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out[:routed], ref[:routed])
    assert not np.any(ref[routed:])
