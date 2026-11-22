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

"""Global bias rows/columns must not follow shard-local mask metadata."""

import jax
import numpy as np
import pytest
from ejkernel.kernels._xla.ring_attention import ring_attention as ring_kernel
from ejkernel.modules.operations import RingAttentionConfig, ring_attention
from ejkernel.ops import BwdParams, FwdParams
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


@pytest.mark.cpu_ok
@pytest.mark.parametrize("bias_layout", ["query_sharded", "replicated", "broadcast_query", "broadcast_key"])
def test_global_bias_forward_backward_abstract_shapes(bias_layout):
    """Trace a four-member ring and its VJP without executing array computation."""
    mesh = jax.sharding.AbstractMesh((4,), ("sp",))
    seq_spec = P(None, "sp", None, None)
    meta_spec = P(None, "sp")
    seq_sharding = jax.sharding.NamedSharding(mesh, seq_spec)
    meta_sharding = jax.sharding.NamedSharding(mesh, meta_spec)
    q_shape = jax.ShapeDtypeStruct((1, 16, 2, 4), jnp.float32, sharding=seq_sharding)
    kv_shape = jax.ShapeDtypeStruct((1, 24, 2, 4), jnp.float32, sharding=seq_sharding)
    bias_spec = P(None, None, "sp", None) if bias_layout == "query_sharded" else P()
    bias_rows = 1 if bias_layout == "broadcast_query" else 16
    bias_cols = 1 if bias_layout == "broadcast_key" else 24
    bias_shape = jax.ShapeDtypeStruct(
        (1, 2, bias_rows, bias_cols), jnp.float32, sharding=jax.sharding.NamedSharding(mesh, bias_spec)
    )
    q_meta = jax.ShapeDtypeStruct((1, 16), jnp.int32, sharding=meta_sharding)
    kv_meta = jax.ShapeDtypeStruct((1, 24), jnp.int32, sharding=meta_sharding)

    def local(q, k, v, bias, qs, ks, qp, kp):
        return ring_kernel(q, k, v, qs, ks, qp, kp, bias=bias, axis_name="sp", causal=True, chunk_size=2)

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(seq_spec, seq_spec, seq_spec, bias_spec, meta_spec, meta_spec, meta_spec, meta_spec),
        out_specs=seq_spec,
        check_vma=False,
    )
    args = (q_shape, kv_shape, kv_shape, bias_shape, q_meta, kv_meta, q_meta, kv_meta)
    output = jax.eval_shape(mapped, *args)
    gradients = jax.eval_shape(jax.grad(lambda *xs: mapped(*xs).sum(), argnums=(0, 1, 2)), *args)
    assert output.shape == q_shape.shape
    assert output.dtype == jnp.float32
    assert tuple(x.shape for x in gradients) == (q_shape.shape, kv_shape.shape, kv_shape.shape)
    assert all(x.dtype == jnp.float32 for x in gradients)


def _inputs():
    """Small fixed-point operands isolate bias indexing from input generation."""
    q = (np.arange(16 * 2 * 4).reshape(1, 16, 2, 4) % 7 - 3).astype(np.float32) / 8
    k = (np.arange(24 * 2 * 4).reshape(1, 24, 2, 4) % 5 - 2).astype(np.float32) / 8
    v = (np.arange(24 * 2 * 4).reshape(1, 24, 2, 4) % 13 - 6).astype(np.float32) / 8
    rows = np.arange(16)[None, None, :, None]
    cols = np.arange(24)[None, None, None, :]
    heads = np.arange(2)[None, :, None, None]
    # Nonseparable row/column effects detect both first-shard reuse and
    # clamping a global offset against an already query-sharded bias.
    bias = ((rows * cols + 3 * cols + 5 * heads) % 23 - 11).astype(np.float32) / 8
    qs = np.array([[0] * 6 + [1] * 10], dtype=np.int32)
    ks = np.array([[0] * 9 + [1] * 15], dtype=np.int32)
    qp = np.array([list(range(6)) + list(range(10))], dtype=np.int32)
    kp = np.array([list(range(9)) + list(range(15))], dtype=np.int32)
    return tuple(jnp.asarray(x) for x in (q, k, v, bias, qs, ks, qp, kp))


@pytest.mark.parametrize("bias_layout", ["query_sharded", "replicated", "broadcast_query", "broadcast_key"])
@pytest.mark.parametrize("metadata", ["none", "segments", "positions", "segments_without_positions"])
def test_distributed_bias_output_and_qkv_gradients(bias_layout, metadata):
    """Compare the public module (and segment-only kernel) with independent dense math."""
    devices = jax.devices()
    assert len(devices) >= 4, "This distributed regression requires a four-device accelerator mesh"
    mesh = Mesh(np.asarray(devices[:4]), ("sp",))
    q, k, v, bias, qs, ks, qp, kp = _inputs()
    if bias_layout == "broadcast_query":
        bias = bias[:, :, :1, :]
    elif bias_layout == "broadcast_key":
        bias = bias[:, :, :, :1]
    seq_spec = P(None, "sp", None, None)
    bias_spec = P(None, None, "sp", None) if bias_layout == "query_sharded" else P()
    causal = metadata != "none"
    window = (7, 0) if metadata == "positions" else None
    if metadata == "positions":
        qp = qp + 4
    mask = None
    if metadata in {"segments", "positions"}:
        mask = MaskInfo.from_segments(
            qs,
            ks,
            q_positions=qp if metadata == "positions" else None,
            kv_positions=kp if metadata == "positions" else None,
            batch_axis_name=None,
            qheads_axis_name=None,
            kvheads_axis_name=None,
            sequence_axis_name="sp",
        )
    cfg = RingAttentionConfig(
        fwd_params=FwdParams(q_blocksize=2, kv_blocksize=2),
        bwd_params=BwdParams(q_blocksize=2, kv_blocksize=2),
        platform="xla",
        backend="any",
    )

    def distributed(q, k, v):
        if metadata == "segments_without_positions":
            # The low-level API also accepts local segments without positions;
            # causal offsets remain global, metadata offsets must remain local.
            def local(q, k, v, bias, qs, ks):
                return ring_kernel(
                    q,
                    k,
                    v,
                    qs,
                    ks,
                    bias=bias,
                    causal=True,
                    axis_name="sp",
                    chunk_size=2,
                    precision=jax.lax.Precision.HIGHEST,
                )

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(seq_spec, seq_spec, seq_spec, bias_spec, P(None, "sp"), P(None, "sp")),
                out_specs=seq_spec,
                check_vma=False,
            )(q, k, v, bias, qs, ks)
        return ring_attention(
            q,
            k,
            v,
            None,
            bias,
            mask_info=mask,
            axis_name="sp",
            causal=causal,
            sliding_window=window,
            precision=jax.lax.Precision.HIGHEST,
            cfg=cfg,
            platform="xla",
            mesh=mesh,
            in_specs=(seq_spec, seq_spec, seq_spec, None, bias_spec),
            out_specs=seq_spec,
        )

    def dense(q, k, v):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision=jax.lax.Precision.HIGHEST) * 0.5 + bias
        allowed = jnp.ones((1, 16, 24), dtype=jnp.bool_)
        if metadata != "none":
            allowed = qs[:, :, None] == ks[:, None, :]
            if metadata == "segments_without_positions":
                qpos = jnp.arange(16)[None, :]
                kpos = jnp.arange(24)[None, :]
            else:
                qpos, kpos = qp, kp
            allowed = allowed & (qpos[:, :, None] >= kpos[:, None, :])
            if window is not None:
                allowed = allowed & (qpos[:, :, None] - kpos[:, None, :] <= window[0])
        has_keys = jnp.any(allowed[:, None], axis=-1, keepdims=True)
        masked_logits = jnp.where(allowed[:, None], logits, -jnp.inf)
        safe_logits = jnp.where(has_keys, masked_logits, 0.0)
        probabilities = jnp.where(has_keys, jax.nn.softmax(safe_logits, axis=-1), 0.0)
        return jnp.einsum("bhqk,bkhd->bqhd", probabilities, v, precision=jax.lax.Precision.HIGHEST)

    cotangent = jnp.asarray((np.arange(q.size).reshape(q.shape) % 11 - 5).astype(np.float32) / 16)

    def loss(fn, q, k, v):
        return jnp.sum(fn(q, k, v) * cotangent)

    with mesh, jax.default_matmul_precision("highest"):
        actual = distributed(q, k, v)
        expected = dense(q, k, v)
        actual_grads = jax.grad(lambda q, k, v: loss(distributed, q, k, v), argnums=(0, 1, 2))(q, k, v)
        expected_grads = jax.grad(lambda q, k, v: loss(dense, q, k, v), argnums=(0, 1, 2))(q, k, v)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual.sharding.spec == seq_spec
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)
