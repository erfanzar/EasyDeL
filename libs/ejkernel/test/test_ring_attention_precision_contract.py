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

"""Public ring precision must cover the entire forward and Q/K/V VJP."""

import inspect

import jax
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ring_attention import ring_attention as pallas_ring
from ejkernel.kernels._xla.ring_attention import ring_attention as xla_ring
from ejkernel.modules.operations import RingAttention, RingAttentionConfig, ring_attention
from ejkernel.ops import BwdParams, FwdParams
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def _config(platform="xla"):
    return RingAttentionConfig(
        fwd_params=FwdParams(q_blocksize=2, kv_blocksize=2),
        bwd_params=BwdParams(q_blocksize=2, kv_blocksize=2),
        platform=platform,
        backend="tpu" if platform == "pallas" else "any",
    )


def _forward(api, q, k, v, bias, sinks=None):
    kwargs = dict(
        causal=True,
        softmax_scale=0.5,
        logits_soft_cap=1.3,
        precision=jax.lax.Precision.HIGHEST,
    )
    if api == "module":
        return ring_attention(q, k, v, sinks, bias, cfg=_config(), **kwargs)
    if api == "run":
        return RingAttention().run(q, k, v, bias=bias, softmax_aux=sinks, cfg=_config(), **kwargs)
    if api == "shard_map":
        # Deliberately request Pallas: explicit precision must be honored even
        # when the selected fused implementation has no precision control.
        return ring_attention(
            q,
            k,
            v,
            sinks,
            bias,
            cfg=_config("pallas"),
            mesh=Mesh(np.asarray(jax.devices()[:1]), ("sp",)),
            in_specs=(P(), P(), P(), None if sinks is None else P(), P()),
            out_specs=P(),
            **kwargs,
        )
    kernel = pallas_ring if api == "pallas" else xla_ring
    return kernel(q, k, v, bias=bias, softmax_aux=sinks, chunk_size=2, **kwargs)


@pytest.mark.cpu_ok
def test_public_backend_signatures_accept_broadcast_bias_and_precision():
    """Type annotations must not reject valid distributed/broadcast bias layouts."""
    xla = inspect.signature(xla_ring)
    pallas = inspect.signature(pallas_ring)
    assert xla == pallas
    assert "bias_batch bias_heads bias_seq_len_q bias_seq_len_k" in str(xla.parameters["bias"].annotation)
    assert xla.parameters["precision"].default == jax.lax.Precision.DEFAULT
    assert inspect.signature(ring_attention).parameters["precision"].default == jax.lax.Precision.DEFAULT
    assert inspect.signature(RingAttention.run).parameters["precision"].default == jax.lax.Precision.DEFAULT


@pytest.mark.cpu_ok
@pytest.mark.parametrize("bias_shape", [(1, 1, 4, 6), (1, 1, 1, 6), (1, 1, 4, 1), (1, 1, 1, 1)])
def test_default_pallas_fallback_accepts_broadcast_bias_abstract(bias_shape):
    """Default Pallas-to-XLA dispatch retains the broadcast bias contract."""
    q = jax.ShapeDtypeStruct((2, 4, 4, 8), jnp.float32)
    kv = jax.ShapeDtypeStruct((2, 6, 2, 8), jnp.float32)
    bias = jax.ShapeDtypeStruct(bias_shape, jnp.float32)
    output = jax.eval_shape(lambda q, k, v, b: pallas_ring(q, k, v, bias=b, chunk_size=2), q, kv, kv, bias)
    assert output.shape == q.shape
    assert output.dtype == q.dtype


@pytest.mark.cpu_ok
@pytest.mark.parametrize("api", ["xla", "pallas", "module", "run", "shard_map"])
def test_requested_precision_forward_and_vjp_abstract_contract(api):
    """Trace public output/VJP contracts and contraction precision, without execution."""
    q = jax.ShapeDtypeStruct((1, 4, 2, 8), jnp.float32)
    kv = jax.ShapeDtypeStruct((1, 6, 1, 8), jnp.float32)
    bias = jax.ShapeDtypeStruct((1, 1, 4, 6), jnp.float32)

    def forward(q, k, v, bias):
        return _forward(api, q, k, v, bias)

    output = jax.eval_shape(forward, q, kv, kv, bias)
    assert output.shape == q.shape
    assert output.dtype == q.dtype
    gradient = jax.grad(lambda q, k, v, b: forward(q, k, v, b).sum(), argnums=(0, 1, 2))
    shapes = jax.eval_shape(gradient, q, kv, kv, bias)
    assert tuple(x.shape for x in shapes) == (q.shape, kv.shape, kv.shape)
    assert all(x.dtype == jnp.float32 for x in shapes)
    program = jax.make_jaxpr(gradient)(q, kv, kv, bias)

    def contraction_precisions(node):
        if hasattr(node, "eqns"):
            for eqn in node.eqns:
                if eqn.primitive.name == "dot_general":
                    yield eqn.params["precision"]
                yield from contraction_precisions(eqn.params)
        elif hasattr(node, "jaxpr"):
            yield from contraction_precisions(node.jaxpr)
        elif isinstance(node, dict):
            for value in node.values():
                yield from contraction_precisions(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                yield from contraction_precisions(value)

    precisions = list(contraction_precisions(program))
    assert precisions
    assert all(p == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST) for p in precisions)


@pytest.mark.parametrize("api", ["xla", "pallas", "module", "run", "shard_map"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_public_highest_precision_matches_dense_output_and_qkv_gradients(api, compiled, with_sinks):
    """Non-bf16-representable weights exercise PV and every backward product."""
    rng = np.random.default_rng(816)
    q = jnp.asarray(rng.normal(size=(1, 4, 2, 8)).astype(np.float32) * 0.4)
    k = jnp.asarray(rng.normal(size=(1, 6, 1, 8)).astype(np.float32) * 0.4)
    v = jnp.asarray(rng.normal(size=(1, 6, 1, 8)).astype(np.float32))
    bias = jnp.asarray(rng.normal(size=(1, 1, 4, 6)).astype(np.float32) * 0.2)
    # Query zero has no visible keys, even after causal masking. Retain it to
    # exercise the finite zero-output/zero-gradient contract independently.
    bias = bias.at[:, :, 0, :].set(-jnp.inf)
    sinks = jnp.asarray([0.1, -0.3], dtype=jnp.float32) if with_sinks else None
    cotangent = jnp.asarray(rng.normal(size=q.shape).astype(np.float32))

    def actual(q, k, v):
        return _forward(api, q, k, v, bias, sinks)

    def dense(q, k, v):
        k = jnp.repeat(k, 2, axis=2)
        v = jnp.repeat(v, 2, axis=2)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision=jax.lax.Precision.HIGHEST) * 0.5
        logits = 1.3 * jnp.tanh(logits / 1.3) + bias
        visible = (jnp.arange(4)[:, None] >= jnp.arange(6)[None, :])[None, None] & jnp.isfinite(bias)
        masked = jnp.where(visible, logits, -jnp.inf)
        if sinks is not None:
            sink_logits = jnp.broadcast_to(sinks, (*masked.shape[:-1], sinks.size))
            probabilities = jax.nn.softmax(jnp.concatenate((masked, sink_logits), axis=-1), axis=-1)[..., :6]
        else:
            has_keys = jnp.any(visible, axis=-1, keepdims=True)
            safe = jnp.where(has_keys, masked, 0.0)
            probabilities = jnp.where(visible, jax.nn.softmax(safe, axis=-1), 0.0)
        return jnp.einsum("bhqk,bkhd->bqhd", probabilities, v, precision=jax.lax.Precision.HIGHEST)

    actual_fn = jax.jit(actual) if compiled else actual
    actual_output = actual_fn(q, k, v)
    np.testing.assert_allclose(actual_output, dense(q, k, v), atol=2e-5, rtol=2e-5)
    assert actual_output.shape == q.shape
    assert actual_output.dtype == q.dtype
    actual_grad = jax.grad(lambda q, k, v: jnp.sum(actual_fn(q, k, v) * cotangent), argnums=(0, 1, 2))
    expected_grad = jax.grad(lambda q, k, v: jnp.sum(dense(q, k, v) * cotangent), argnums=(0, 1, 2))
    for observed, expected in zip(actual_grad(q, k, v), expected_grad(q, k, v), strict=True):
        assert np.isfinite(observed).all()
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(actual_output[:, 0], 0.0)


@pytest.mark.parametrize("bias_rows", [1, 16])
def test_distributed_highest_precision_global_bias_and_qkv_gradients(bias_rows):
    """Four-way public ring parity includes replicated global bias and local positions."""
    devices = jax.devices()
    assert len(devices) >= 4, "This regression requires four accelerator devices"
    mesh = Mesh(np.asarray(devices[:4]), ("sp",))
    seq = P(None, "sp", None, None)
    meta = P(None, "sp")
    rng = np.random.default_rng(108)
    q = jnp.asarray(rng.normal(size=(1, 16, 2, 8)).astype(np.float32) * 0.4)
    k = jnp.asarray(rng.normal(size=(1, 24, 1, 8)).astype(np.float32) * 0.4)
    v = jnp.asarray(rng.normal(size=(1, 24, 1, 8)).astype(np.float32))
    bias = jnp.asarray(rng.normal(size=(1, 1, bias_rows, 24)).astype(np.float32) * 0.2)
    qp = jnp.arange(16, dtype=jnp.int32)[None] % 8 + 2
    kp = jnp.arange(24, dtype=jnp.int32)[None] % 12
    cotangent = jnp.asarray(rng.normal(size=q.shape).astype(np.float32))

    def local(q, k, v, bias, qp, kp):
        return xla_ring(
            q,
            k,
            v,
            q_position_ids=qp,
            kv_position_ids=kp,
            bias=bias,
            axis_name="sp",
            causal=True,
            chunk_size=2,
            softmax_scale=0.5,
            precision=jax.lax.Precision.HIGHEST,
        )

    mapped = jax.shard_map(local, mesh=mesh, in_specs=(seq, seq, seq, P(), meta, meta), out_specs=seq, check_vma=False)

    def actual(q, k, v):
        return mapped(q, k, v, bias, qp, kp)

    def dense(q, k, v):
        logits = (
            jnp.einsum("bqhd,bkhd->bhqk", q, jnp.repeat(k, 2, axis=2), precision=jax.lax.Precision.HIGHEST) * 0.5 + bias
        )
        visible = qp[:, None, :, None] >= kp[:, None, None, :]
        probabilities = jax.nn.softmax(jnp.where(visible, logits, -jnp.inf), axis=-1)
        return jnp.einsum(
            "bhqk,bkhd->bqhd", probabilities, jnp.repeat(v, 2, axis=2), precision=jax.lax.Precision.HIGHEST
        )

    observed = actual(q, k, v)
    np.testing.assert_allclose(observed, dense(q, k, v), atol=2e-5, rtol=2e-5)
    assert observed.sharding.spec == seq
    actual_grad = jax.grad(lambda q, k, v: jnp.sum(actual(q, k, v) * cotangent), argnums=(0, 1, 2))
    expected_grad = jax.grad(lambda q, k, v: jnp.sum(dense(q, k, v) * cotangent), argnums=(0, 1, 2))
    for observed, expected in zip(actual_grad(q, k, v), expected_grad(q, k, v), strict=True):
        assert np.isfinite(observed).all()
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=2e-5)
