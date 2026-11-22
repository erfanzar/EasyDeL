"""W8A8 shared expert on a few decode rows (``shared_expert_w8a8``, TPU only).

Pinned here: the Pallas TPU kernel is bitwise equal to the XLA reference (two one-group A8
``grouped_matmul_channelwise`` calls around the float32 combine), each compiled as one program, at GLM-5.3-Flash's
per-chip shared-expert shapes (8 rows, H 4096, I 512), at rows that are not a multiple of 8, with a zero row, with
large inputs that saturate the clamped SwiGLU and with a plain SiLU combine; the registry resolves Pallas on TPU and
XLA at priority 0; unsupported operands are reported.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules import shared_expert_w8a8, shared_expert_w8a8_unsupported_reason

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real TPU")


def _inputs(r, h, inter, scale_x, seed, zero_row):
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    rows = (jax.random.normal(k[0], (r, h), jnp.float32) * scale_x).astype(jnp.bfloat16)
    if zero_row:
        rows = rows.at[0].set(0)
    gu = jax.random.randint(k[1], (h, 2 * inter), -127, 128, jnp.int32).astype(jnp.int8)
    dn = jax.random.randint(k[2], (inter, h), -127, 128, jnp.int32).astype(jnp.int8)
    gus = jax.random.uniform(k[3], (1, 2 * inter), jnp.float32, 1e-5, 3e-4)
    dns = jax.random.uniform(k[4], (1, h), jnp.float32, 1e-5, 3e-4)
    return rows, gu, gus, dn, dns


CASES = [
    (8, 4096, 512, 1.0, 0, False, "clamped_swiglu", (10.0,)),
    (8, 4096, 512, 30.0, 1, True, "clamped_swiglu", (10.0,)),
    (5, 4096, 512, 1.0, 2, False, "clamped_swiglu", (10.0,)),
    (16, 4096, 512, 0.01, 3, True, "clamped_swiglu", (10.0,)),
    (8, 2048, 256, 1.0, 4, False, "silu", ()),
]


@pytest.mark.parametrize("r,h,inter,scale_x,seed,zero_row,act,params", CASES)
def test_pallas_bitwise_equals_xla(r, h, inter, scale_x, seed, zero_row, act, params):
    args = _inputs(r, h, inter, scale_x, seed, zero_row)

    def run(platform):
        fn = jax.jit(lambda *a: shared_expert_w8a8(*a, activation=act, act_params=params, platform=platform))
        return np.asarray(fn(*args))

    ref, out = run("xla"), run("pallas")
    assert out.dtype == np.float32 and out.shape == (r, h)
    assert np.isfinite(ref).all()
    assert np.array_equal(out.view(np.uint32), ref.view(np.uint32))


def test_registry_entries():
    pallas = kernel_registry.get("shared_expert_w8a8", platform=Platform.PALLAS, backend=Backend.TPU)
    xla = kernel_registry.get("shared_expert_w8a8", platform=Platform.XLA, backend=Backend.ANY)
    assert pallas is not None and xla is not None and pallas is not xla
    assert kernel_registry.validate_signatures("shared_expert_w8a8")


def test_unsupported_reason():
    rows, gu, gus, dn, dns = _inputs(8, 4096, 512, 1.0, 0, False)
    assert shared_expert_w8a8_unsupported_reason(rows, gu, gus, dn, dns) is None
    big = jnp.zeros((256, 4096), jnp.bfloat16)
    assert "rows" in shared_expert_w8a8_unsupported_reason(big, gu, gus, dn, dns)
    assert shared_expert_w8a8_unsupported_reason(rows, gu.astype(jnp.int32), gus, dn, dns) is not None
