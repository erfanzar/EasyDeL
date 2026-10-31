# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Cotangent-first MoE weight gradient: shape policy and parity with the XLA grouped matmul's own vjp.

The policy accepts only bf16 kernels of a family listed in ``_CT_FIRST_DW_FAMILIES``
(``easydel/layers/moe/_moe_module.py``) with 32-aligned rows inside its range: the shapes where the cotangent-first
form is faster on TPU v5p.
"""

import jax.numpy as jnp
import pytest
from easydel.layers.moe._moe_module import _ct_first_dw_shape

GATE_UP = (288, 4096, 512)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("rows", [32768, 131072, 196608, 262144])
def test_measured_gate_up_family_uses_ct_first(rows):
    assert _ct_first_dw_shape(rows, GATE_UP, jnp.bfloat16)


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("rows", "shape", "dtype"),
    [
        (16384, GATE_UP, jnp.bfloat16),  # below the family's row range (the swap is slower there)
        (262176, GATE_UP, jnp.bfloat16),  # above the family's row range
        (131072, (288, 512, 4096), jnp.bfloat16),  # down projection (k < n): the swap is slower
        (32768, (128, 2048, 768), jnp.bfloat16),  # unlisted family (the swap is slower)
        (131072, GATE_UP, jnp.float32),  # f32 keeps the caller's path
        (131080, GATE_UP, jnp.bfloat16),  # rows not 32-aligned
    ],
)
def test_unmeasured_cases_keep_jax_transpose(rows, shape, dtype):
    assert not _ct_first_dw_shape(rows, shape, dtype)


def test_ct_first_dw_matches_xla_vjp_in_chunk_scan():
    """Output, dx and the chunk-summed dW equal the plain XLA grouped matmul's; dW is emitted as ``[E, n, k]``."""
    import jax
    import numpy as np
    from easydel.layers.moe._moe_module import _xla_ct_first_dw_gmm
    from ejkernel.modules import GroupedMatmulConfig, grouped_matmul

    if jax.default_backend() != "tpu":
        pytest.skip("checks the TPU ragged-dot lowering")
    chunks, rows, experts, k, n = 2, 4096, 16, 1024, 256
    rng = np.random.default_rng(5)
    sizes = jnp.asarray(
        np.stack([rng.multinomial(rows, np.full(experts, 1.0 / experts)) for _ in range(chunks)]).astype(np.int32)
    )
    xs = jnp.asarray(rng.standard_normal((chunks, rows, k)) * 0.1, jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((experts, k, n)) * 0.02, jnp.bfloat16)
    cts = jnp.asarray(rng.standard_normal((chunks, rows, n)), jnp.bfloat16)
    xla_kws = {
        "preferred_element_type": jnp.bfloat16,
        "cfg": GroupedMatmulConfig(platform="xla", bypass_xla_tiling=True),
    }

    def make(proj):
        def run(xs_, w_, sizes_, cts_):
            def f(a, b):
                def body(_, inp):
                    x_c, g_c = inp
                    return None, proj(x_c, b, g_c)

                return jax.lax.scan(jax.checkpoint(body), None, (a, sizes_))[1]

            out, vjp = jax.vjp(f, xs_, w_)
            return out, *vjp(cts_)

        return jax.jit(run)

    swap = make(lambda a, b, g: _xla_ct_first_dw_gmm(a, b, g, xla_kws=xla_kws))
    plain = make(lambda a, b, g: grouped_matmul(a, b, g, **xla_kws))
    got = swap(xs, w, sizes, cts)
    want = plain(xs, w, sizes, cts)
    for name, a, b in zip(("out", "dx", "dW"), got, want, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
        assert np.linalg.norm(a32 - b32) <= 1e-3 * np.linalg.norm(b32), name

    def ragged_dot_shapes(fn):
        shapes = []

        def walk(jaxpr):
            for eqn in jaxpr.eqns:
                if eqn.primitive.name == "ragged_dot_general":
                    shapes.append(tuple(eqn.outvars[0].aval.shape))
                for param in eqn.params.values():
                    for sub in param if isinstance(param, (tuple, list)) else (param,):
                        sub = getattr(sub, "jaxpr", sub)
                        if hasattr(sub, "eqns"):
                            walk(sub)

        walk(jax.make_jaxpr(fn)(xs, w, sizes, cts).jaxpr)
        return shapes

    swap_shapes, plain_shapes = ragged_dot_shapes(swap), ragged_dot_shapes(plain)
    assert (experts, k, n) in plain_shapes, "reference must exhibit JAX's [E, k, n] weight-gradient ragged dot"
    assert (experts, n, k) in swap_shapes, "the weight gradient must be emitted cotangent-first ([E, n, k])"
    assert (experts, k, n) not in swap_shapes, "no [E, k, n] weight-gradient ragged dot may remain"
