"""Integer dots of the Pallas v3 grouped matmul ignore an ambient default matmul precision (TPU only).

Integer MXU operands accumulate exactly in int32, and Mosaic rejects a contract precision on them. Before the kernel
pinned its integer dots to ``Precision.DEFAULT``, an ambient ``jax.default_matmul_precision("highest")`` (easydel's
MoE parity tests set one) lowered them with ``contract_precision<fp32>`` and the grouped matmul failed to compile
("Bad lhs type"). Pinned here: the CHANNELWISE entry's integer paths (A8 whole-slab tiles, A8 generic tiles, A4 with
int4 codes) and the v3 module op's in-kernel INT8 activation quantization compile under "highest" and give the bits
they give without it (for the CHANNELWISE entry also the XLA entry's bits).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul, grouped_matmul_channelwise
from ejkernel.modules.operations import GroupedMatmulConfig

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real TPU")


def _channelwise(platform, bits):
    return jax.jit(
        lambda a, b, c, d: grouped_matmul_channelwise(
            a, b, c, d, activation_bits=bits, platform=platform, preferred_element_type=jnp.float32
        )
    )


@pytest.mark.parametrize(
    "bits,e,m,k,n,local",
    [(8, 8, 128, 256, 128, 100), (8, 4, 1056, 256, 256, 900), (4, 3, 80, 640, 256, 77)],
    ids=["a8_slab", "a8_generic_tiles", "a4_int4"],
)
def test_channelwise_integer_dots_ignore_ambient_highest(bits, e, m, k, n, local):
    rng = np.random.default_rng(m + k)
    sizes = np.bincount(rng.integers(0, e, size=local), minlength=e).astype(np.int32)
    x = rng.normal(size=(m, k)).astype(np.float32)
    x[local:] = np.nan
    bound = (1 << (bits - 1)) - 1
    codes = rng.integers(-bound, bound + 1, size=(e, k, n), dtype=np.int8)
    args = (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(codes).astype(jnp.int4 if bits == 4 else jnp.int8),
        jnp.asarray(rng.uniform(0.001, 0.01, size=(e, 1, n)), jnp.float32),
        jnp.asarray(sizes),
    )
    plain = np.asarray(_channelwise("pallas", bits)(*args))
    with jax.default_matmul_precision("highest"):
        highest = np.asarray(_channelwise("pallas", bits)(*args))
    expected = np.asarray(_channelwise("xla", bits)(*args))
    assert np.isfinite(plain).all()
    np.testing.assert_array_equal(highest, plain)
    np.testing.assert_array_equal(plain, expected)


def test_v3_in_kernel_int8_activation_quantization_ignores_ambient_highest():
    rng = np.random.default_rng(22)
    k = 256
    a_np = rng.integers(-64, 65, size=(16, k)).astype(np.float32)
    a_np[:, 0] = 127
    w_np = rng.integers(-7, 8, size=(4, k, 128), dtype=np.int8)
    args = (
        jnp.asarray(a_np, jnp.bfloat16),
        jnp.asarray(w_np).astype(jnp.int4),
        jnp.ones((4, 1, 1, 128), jnp.bfloat16) * 0.125,
        jnp.array([5, 0, 7, 4], jnp.int32),
    )
    cfg = GroupedMatmulConfig(block_m=8, block_k=0, block_n=0)

    def run():
        f = jax.jit(
            lambda x, y, s, g: grouped_matmul(
                x, y, g, rhs_scale=s, preferred_element_type=jnp.bfloat16, use_v3=True, platform="pallas", cfg=cfg
            )
        )
        return np.asarray(f(*args)).astype(np.float32)

    plain = run()
    with jax.default_matmul_precision("highest"):
        highest = run()
    assert np.isfinite(plain).all() and np.abs(plain).max() > 0
    np.testing.assert_array_equal(highest, plain)
