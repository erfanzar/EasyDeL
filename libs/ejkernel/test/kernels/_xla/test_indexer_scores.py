# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""XLA ``indexer_scores`` vs an independent float64 NumPy oracle (value, all three cotangents, precision, specs)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.modules import IndexerScores, indexer_scores
from ejkernel.modules.operations.indexer_scores import validate_shard_specs
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

XLA = kernel_registry.get("indexer_scores", platform=Platform.XLA, backend=Backend.ANY)


def make_inputs(batch, heads, seq, entries, dim, dtype=jnp.float32, seed=0):
    """``(query [B,H,S,d], index_k [B,T,d], weights [B,H,S,1] fp32, cotangent [B,S,T] fp32)``."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((batch, heads, seq, dim)), dtype)
    k = jnp.asarray(rng.standard_normal((batch, entries, dim)), dtype)
    w = jnp.asarray(rng.standard_normal((batch, heads, seq, 1)), jnp.float32)
    cot = jnp.asarray(rng.standard_normal((batch, seq, entries)), jnp.float32)
    return q, k, w, cot


def _f64(x):
    return np.asarray(jnp.asarray(x, jnp.float32), np.float64)


def oracle(q, k, w, cot=None):
    """Float64 NumPy scores and (with ``cot``) the analytic cotangents ``(dq, dk, dw)``."""
    q, k, w = _f64(q), _f64(k), _f64(w)
    a = np.einsum("bhsd,btd->bhst", q, k)
    out = np.sum(np.maximum(a, 0.0) * w, axis=1)
    if cot is None:
        return out
    g = _f64(cot)[:, None] * w * (a > 0)  # d a = g * w * relu'(a), [B, H, S, T]
    dq = np.einsum("bhst,btd->bhsd", g, k)
    dk = np.einsum("bhst,bhsd->btd", g, q)
    dw = np.sum(_f64(cot)[:, None] * np.maximum(a, 0.0), axis=-1, keepdims=True)
    return out, (dq, dk, dw)


def rel(a, b) -> float:
    """Relative L2 error of ``a`` against reference ``b``."""
    a, b = _f64(a), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def grads(fn, q, k, w, cot, **kw):
    return jax.grad(lambda a, b, c: jnp.sum(fn(a, b, c, **kw) * cot), argnums=(0, 1, 2))(q, k, w)


def test_registry_has_xla_fallback_and_pallas_tpu():
    assert kernel_registry.validate_signatures("indexer_scores")
    specs = {(s.platform, s.backend): s.priority for s in kernel_registry.list_implementations("indexer_scores")}
    assert specs[(Platform.XLA, Backend.ANY)] == 0
    assert (Platform.PALLAS, Backend.TPU) in specs
    assert "create_shard_map_wrapper" in IndexerScores.__dict__


@pytest.mark.parametrize(
    ("batch", "heads", "seq", "entries", "dim"),
    [(2, 4, 64, 256, 32), (1, 8, 300, 700, 64), (1, 2, 20, 130, 128), (1, 3, 1, 5, 16)],
)
def test_xla_matches_float64_oracle(batch, heads, seq, entries, dim):
    q, k, w, cot = make_inputs(batch, heads, seq, entries, dim, seed=batch + seq)
    out = jax.jit(lambda a, b, c: XLA(a, b, c, precision="highest"))(q, k, w)
    want, want_g = oracle(q, k, w, cot)
    assert out.shape == (batch, seq, entries) and out.dtype == jnp.float32
    assert rel(out, want) < 1e-5
    got_g = jax.jit(lambda a, b, c: grads(XLA, a, b, c, cot, precision="highest"))(q, k, w)
    for name, got, ref, arg in zip(("query", "index_k", "weights"), got_g, want_g, (q, k, w), strict=True):
        assert got.shape == arg.shape and got.dtype == arg.dtype, name
        assert rel(got, ref) < 1e-5, (name, rel(got, ref))


def test_relu_gate_and_head_weights():
    q = jnp.asarray(np.eye(4, dtype=np.float32)[None, :, None, :])  # head h queries axis h
    k = jnp.asarray(np.array([[1.0, -1.0, 2.0, 0.0], [0.0, 0.0, 0.0, -3.0]], np.float32)[None])
    w = jnp.asarray(np.array([1.0, 10.0, 100.0, 1000.0], np.float32)[None, :, None, None])
    out = XLA(q, k, w, precision="highest")
    np.testing.assert_allclose(np.asarray(out)[0, 0], [201.0, 0.0], atol=1e-5)


def test_precision_is_honoured_and_validated():
    q, k, w, _ = make_inputs(1, 4, 64, 128, 64, dtype=jnp.bfloat16, seed=3)
    truth = oracle(q, k, w)  # fp64 on the bf16-valued operands
    hi = XLA(q, k, w, precision=jax.lax.Precision.HIGHEST)
    assert rel(hi, truth) < 1e-5
    assert bool(jnp.array_equal(hi, XLA(q, k, w, precision="highest")))
    assert rel(XLA(q, k, w), truth) < 2e-2  # default: one bf16 MXU pass, fp32 accumulation
    with pytest.raises(ValueError):
        indexer_scores(q, k, w, precision="exact", platform="xla")
    with pytest.raises(ValueError):
        indexer_scores(q, k, w, precision=("highest", "default"), platform="xla")


def test_modules_op_xla_matches_backend():
    q, k, w, _ = make_inputs(2, 4, 48, 96, 32, seed=5)
    np.testing.assert_array_equal(
        np.asarray(indexer_scores(q, k, w, platform="xla", precision="highest")),
        np.asarray(XLA(q, k, w, precision="highest")),
    )


def test_shard_spec_contract():
    ok = [
        ((P("dp", None, None, None), P("dp", None, None), P("dp", None, None, None)), P("dp", None, None)),
        ((P(None, None, None, None), P(None, None, None), P(None, None, None, None)), P(None, None, None)),
        ((P("dp", None, "sp", None), P("dp", None, None), P("dp", None, "sp", None)), P("dp", "sp", None)),
        ((P("dp", None, None, None), P("dp", "tp", None), P("dp", None, None, None)), P("dp", None, "tp")),
    ]
    for ins, out in ok:
        validate_shard_specs(ins, out)
    bad = [
        ((P("dp", "tp", None, None), P("dp", None, None), P("dp", "tp", None, None)), P("dp", None, None)),
        ((P("dp", None, None, "tp"), P("dp", None, "tp"), P("dp", None, None, None)), P("dp", None, None)),
        ((P(None, None, None, None), P(None, None, None), P(None, None, None, None)), P("dp", None, None)),
        ((P(None, None, "dp", None), P(None, None, None), P(None, None, "dp", None)), P("dp", None, None)),
        (
            (P(("dp", "tp"), None, None, None), P(("dp", "tp"), None, None), P(("dp", "tp"), None, None, None)),
            P(("tp", "dp"), None, None),
        ),
        ((P("dp", None, None, None), P("dp", None, None)), P("dp", None, None)),
    ]
    for ins, out in bad:
        with pytest.raises(ValueError, match="indexer_scores"):
            validate_shard_specs(ins, out)


def test_xla_shard_map_path_matches_unsharded():
    if jax.device_count() < 4:
        pytest.skip("needs four devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "sp"))
    q, k, w, cot = make_inputs(2, 4, 64, 128, 32, seed=7)
    ins = (P("dp", None, "sp", None), P("dp", None, None), P("dp", None, "sp", None))
    out = P("dp", "sp", None)

    def sharded(a, b, c):
        return indexer_scores(a, b, c, platform="xla", precision="highest", mesh=mesh, in_specs=ins, out_specs=out)

    want, want_g = oracle(q, k, w, cot)
    assert rel(jax.jit(sharded)(q, k, w), want) < 1e-5
    for got, ref in zip(jax.jit(lambda a, b, c: grads(sharded, a, b, c, cot))(q, k, w), want_g, strict=True):
        assert rel(got, ref) < 1e-5
    with pytest.raises(ValueError):
        indexer_scores(q, k, w, platform="xla", mesh=mesh)
