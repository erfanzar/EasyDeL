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

"""``_reduce_scatter``: the deferred MoE dispatch's reduce-scatter as two ppermute rings in opposite directions.

On four TPU devices these tests pin:

* for both scatter dimensions and several dtypes, including the dispatch's TP reduction (a float32 partial whose
  running sums travel in bfloat16, into a float32 result), the result matches a float64 NumPy sum of every shard's
  partial and ``jax.lax.psum_scatter`` of the partial in the wire dtype up to the summation order and roundings; the
  program holds collective-permutes in the wire dtype, not a reduce-scatter, and never converts the whole partial;
* a one-element axis tuple runs the same rings, bit for bit;
* shapes whose other dimension cannot be halved fall back to ``psum_scatter``, with identical bits.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

D = 4
_PERMUTE = re.compile(r'"stablehlo\.collective_permute"\([^)]*\)\s*<\{.*?\}>\s*:\s*\(tensor<((?:\d+x)+)(\w+)>\)')
_SHORT = {"bfloat16": "bf16", "float32": "f32"}


def _mesh() -> Mesh:
    if jax.device_count() < D:
        pytest.skip("needs 4 devices")
    return Mesh(np.asarray(jax.devices()[:D]), ("x",))


def _partials(rows: int, cols: int, dtype, seed: int = 0):
    """Shard ``i``'s partial is rows ``[i * rows, (i + 1) * rows)`` of the result; the reference is their float64 sum."""
    host = np.random.default_rng(seed).standard_normal((D * rows, cols), dtype=np.float32)
    values = jnp.asarray(host).astype(dtype)
    reference = np.asarray(values.astype(jnp.float32), np.float64).reshape(D, rows, cols).sum(axis=0)
    return values, reference


def _run(fn, x, scatter_dimension: int, mesh: Mesh):
    out_spec = P(None, "x") if scatter_dimension == 1 else P("x", None)
    mapped = jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=P("x", None), out_specs=out_spec, check_vma=False))
    return np.asarray(jax.device_get(mapped(x)), np.float64), mapped.lower(x).as_text()


def _rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


@pytest.mark.parametrize(
    ("dtype", "out_dtype", "wire_dtype", "scatter_dimension", "rows", "cols", "tol"),
    [
        (jnp.bfloat16, jnp.float32, None, 1, 32, 256, 2**-7),
        (jnp.float32, jnp.float32, jnp.bfloat16, 1, 32, 256, 2**-7),
        (jnp.float32, jnp.float32, None, 0, 32, 128, 1e-6),
        (jnp.bfloat16, jnp.bfloat16, None, 0, 64, 64, 2**-7),
    ],
    ids=["tp_bf16_into_f32", "tp_f32_sent_as_bf16", "rows_f32", "rows_bf16"],
)
def test_rings_match_float64_sum_and_psum_scatter(dtype, out_dtype, wire_dtype, scatter_dimension, rows, cols, tol):
    from easydel.layers.moe._moe_module import _reduce_scatter

    mesh = _mesh()
    x, reference = _partials(rows, cols, dtype)
    wire = jnp.dtype(wire_dtype or dtype)

    def ring(a):
        return _reduce_scatter(a, "x", D, scatter_dimension, out_dtype, wire_dtype=wire_dtype)

    def xla(a):
        return jax.lax.psum_scatter(a.astype(wire), "x", scatter_dimension=scatter_dimension, tiled=True).astype(
            out_dtype
        )

    out_ring, text_ring = _run(ring, x, scatter_dimension, mesh)
    out_xla, _ = _run(xla, x, scatter_dimension, mesh)
    assert "stablehlo.collective_permute" in text_ring, "the rings did not run"
    assert "stablehlo.reduce_scatter" not in text_ring, "the rings fell back to psum_scatter"
    sent = {dt for _, dt in _PERMUTE.findall(text_ring)}
    assert sent == {_SHORT[wire.name]}, f"the rings sent {sent}, expected {wire.name}"
    if wire != jnp.dtype(dtype):
        assert f"tensor<{rows}x{cols}x{_SHORT[wire.name]}>" not in text_ring, "the whole partial was converted"
    assert np.isfinite(out_ring).all()
    assert _rel(out_ring, reference) < tol, _rel(out_ring, reference)
    assert _rel(out_xla, reference) < tol, _rel(out_xla, reference)
    assert _rel(out_ring, out_xla) < 2 * tol, _rel(out_ring, out_xla)


def test_one_element_axis_tuple_runs_the_same_rings():
    from easydel.layers.moe._moe_module import _reduce_scatter

    mesh = _mesh()
    x, _ = _partials(32, 256, jnp.bfloat16, seed=1)
    out_name, _ = _run(lambda a: _reduce_scatter(a, "x", D, 1, jnp.float32), x, 1, mesh)
    out_tuple, text_tuple = _run(lambda a: _reduce_scatter(a, ("x",), D, 1, jnp.float32), x, 1, mesh)
    assert "stablehlo.collective_permute" in text_tuple
    np.testing.assert_array_equal(out_tuple, out_name)


@pytest.mark.parametrize(
    ("scatter_dimension", "rows", "cols"), [(1, 31, 256), (0, 32, 63)], ids=["odd_rows", "odd_columns"]
)
def test_shapes_the_rings_cannot_halve_fall_back_to_psum_scatter(scatter_dimension, rows, cols):
    from easydel.layers.moe._moe_module import _reduce_scatter

    mesh = _mesh()
    x, reference = _partials(rows, cols, jnp.float32, seed=2)
    out_ring, text_ring = _run(lambda a: _reduce_scatter(a, "x", D, scatter_dimension), x, scatter_dimension, mesh)
    out_xla, _ = _run(
        lambda a: jax.lax.psum_scatter(a, "x", scatter_dimension=scatter_dimension, tiled=True),
        x,
        scatter_dimension,
        mesh,
    )
    assert "stablehlo.reduce_scatter" in text_ring and "stablehlo.collective_permute" not in text_ring
    np.testing.assert_array_equal(out_ring, out_xla)
    assert _rel(out_ring, reference) < 1e-6, _rel(out_ring, reference)
