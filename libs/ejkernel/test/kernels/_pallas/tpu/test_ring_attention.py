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


"""Tests for Pallas TPU ring attention implementation."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from ejkernel.kernels import pallas
from ejkernel.kernels._xla.attention import attention as vanilla_attention
from ejkernel.ops import FwdParams
from ejkernel.utils import numeric_gen
from jax import shard_map
from jax.sharding import PartitionSpec


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")

FWD_128 = FwdParams(q_blocksize=128, kv_blocksize=128)


class TestRingAttentionPallasFwd:
    """Test forward pass of Pallas TPU ring attention."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert out.dtype == jnp.float32
        assert not jnp.any(jnp.isnan(out))
        assert not jnp.any(jnp.isinf(out))

    def test_softmax_aux_1d(self):
        """Test with 1D softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_softmax_aux_2d(self):
        """2D per-head sinks route to the XLA ring kernel and match the reference attention."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        # Distinct per-head values, so treating them as shared sinks would change the output.
        softmax_aux = jnp.linspace(-3.0, 3.0, num_heads * num_sinks, dtype=jnp.float32).reshape(num_heads, num_sinks)

        def dense_f32(sinks_per_head):
            # Independent f32 reference: raw sink logits join the softmax denominator per head.
            qf, kf, vf = (x.astype(jnp.float32).transpose(0, 2, 1, 3) for x in (q, k, v))
            logits = jnp.einsum("bhtd,bhkd->bhtk", qf * head_dim**-0.5, kf)
            logits = jnp.where(jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool)), logits, -1e30)
            aux = jnp.broadcast_to(sinks_per_head[None, :, None, :], (batch, num_heads, seq_len, num_sinks))
            weights = jax.nn.softmax(jnp.concatenate([logits, aux], axis=-1), axis=-1)[..., :seq_len]
            return jnp.einsum("bhtk,bhkd->bhtd", weights, vf).transpose(0, 2, 1, 3)

        with jax.default_matmul_precision("highest"):
            out = pallas.tpu.ring_attention(q, k, v, softmax_aux=softmax_aux, causal=True, fwd_params=FWD_128)
            ref = dense_f32(softmax_aux)
            shared = dense_f32(jnp.broadcast_to(softmax_aux[0], softmax_aux.shape))

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        err = float(jnp.abs(out.astype(jnp.float32) - ref).max())
        gap = float(jnp.abs(shared - ref).max())
        print("2D-sink ring vs f32 ref max err:", err, "per-head vs shared gap:", gap)
        # The XLA ring kernel runs matmuls at an explicit Precision.DEFAULT (one bf16 pass on TPU),
        # so use the same 1e-2 tolerance as this file's other ring parity tests.
        assert err < 1e-2, err
        assert gap > 5 * err, (gap, err)  # per-head sinks must matter for this check

    def test_logits_soft_cap(self):
        """Test with logit soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            logits_soft_cap=30.0,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_symmetric(self):
        """Test with symmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_asymmetric(self):
        """Test with asymmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=(32, 96),
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_small(self):
        """Test with very small sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=0,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_large(self):
        """Test with sliding window larger than sequence."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=512,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_with_softmax_aux(self):
        """Test sliding window combined with softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            softmax_aux=softmax_aux,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_with_causal(self):
        """Test sliding window with causal masking."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            causal=True,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_causal_mask(self):
        """Test with causal masking."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            causal=True,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_combined_features(self):
        """Test with all features combined."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            logits_soft_cap=30.0,
            sliding_window=(32, 96),
            causal=True,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))


class TestRingAttentionPallasBwd:
    """Test backward pass (gradients) of Pallas TPU ring attention."""

    def test_gradient_with_softmax_aux(self):
        """Test gradient with softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 2
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                softmax_aux=softmax_aux,
                fwd_params=FWD_128,
            )
            return jnp.mean(out**2)

        _loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))

    def test_gradient_with_logits_soft_cap(self):
        """Test gradient with logit soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                logits_soft_cap=30.0,
                fwd_params=FWD_128,
            )
            return jnp.mean(out**2)

        _loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))

    def test_gradient_with_sliding_window_symmetric(self):
        """Test gradient with symmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                sliding_window=64,
                fwd_params=FWD_128,
            )
            return jnp.mean(out**2)

        _loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))

    def test_gradient_with_sliding_window_asymmetric(self):
        """Test gradient with asymmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                sliding_window=(32, 96),
                fwd_params=FWD_128,
            )
            return jnp.mean(out**2)

        _loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))

    def test_gradient_combined_features(self):
        """Test gradient with combined features."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 2
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                softmax_aux=softmax_aux,
                logits_soft_cap=30.0,
                sliding_window=64,
                fwd_params=FWD_128,
            )
            return jnp.mean(out**2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))
        assert loss > 0


class TestRingAttentionNumericalCorrectness:
    """Test numerical correctness of ring attention against vanilla XLA attention."""

    def test_numerical_correctness_vs_vanilla_basic(self):
        """Test basic numerical correctness against vanilla attention."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out_ring = pallas.tpu.ring_attention(
            q,
            k,
            v,
            fwd_params=FWD_128,
        )

        out_vanilla, _ = vanilla_attention(q, k, v)

        assert out_ring.shape == out_vanilla.shape
        assert jnp.allclose(out_ring, out_vanilla, rtol=1e-2, atol=1e-2), (
            f"Ring attention output differs from vanilla attention. Max diff: {jnp.max(jnp.abs(out_ring - out_vanilla))}"
        )

    def test_numerical_correctness_vs_vanilla_causal(self):
        """Test numerical correctness with causal masking."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out_ring = pallas.tpu.ring_attention(
            q,
            k,
            v,
            causal=True,
            fwd_params=FWD_128,
        )

        out_vanilla, _ = vanilla_attention(q, k, v, causal=True, dtype=jnp.float32)

        assert out_ring.shape == out_vanilla.shape
        assert jnp.allclose(out_ring, out_vanilla, rtol=1e-2, atol=1e-2), (
            f"Ring attention with causal differs from vanilla. Max diff: {jnp.max(jnp.abs(out_ring - out_vanilla))}"
        )

    def test_numerical_correctness_vs_vanilla_sliding_window(self):
        """Test numerical correctness with sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        window_size = 64

        out_ring = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=window_size,
            fwd_params=FWD_128,
        )

        out_vanilla, _ = vanilla_attention(q, k, v, sliding_window=window_size)

        assert out_ring.shape == out_vanilla.shape
        assert jnp.allclose(out_ring, out_vanilla, rtol=1e-2, atol=1e-2), (
            f"Ring attention with sliding window differs from vanilla. "
            f"Max diff: {jnp.max(jnp.abs(out_ring - out_vanilla))}"
        )

    def test_numerical_correctness_vs_vanilla_softmax_aux_1d(self):
        """Test numerical correctness with 1D softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        out_ring = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            fwd_params=FWD_128,
        )

        out_vanilla, _ = vanilla_attention(q, k, v, softmax_aux=softmax_aux)

        assert out_ring.shape == out_vanilla.shape
        assert jnp.allclose(out_ring, out_vanilla, rtol=1e-2, atol=1e-2), (
            f"Ring attention with softmax_aux differs from vanilla. Max diff: {jnp.max(jnp.abs(out_ring - out_vanilla))}"
        )

    def test_numerical_correctness_vs_vanilla_logits_soft_cap(self):
        """Test numerical correctness with logit soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        logits_soft_cap = 30.0

        out_ring = pallas.tpu.ring_attention(
            q,
            k,
            v,
            logits_soft_cap=logits_soft_cap,
            fwd_params=FWD_128,
        )

        out_vanilla, _ = vanilla_attention(q, k, v, logits_soft_cap=logits_soft_cap)

        assert out_ring.shape == out_vanilla.shape
        assert jnp.allclose(out_ring, out_vanilla, rtol=0, atol=0.125), (
            f"Ring attention with logits_soft_cap differs from vanilla. "
            f"Max diff: {jnp.max(jnp.abs(out_ring - out_vanilla))}"
        )

    def test_numerical_correctness_vs_vanilla_combined_features(self):
        """Test numerical correctness with multiple features combined."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0
        window_size = 64
        logits_soft_cap = 30.0

        out_ring = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            logits_soft_cap=logits_soft_cap,
            sliding_window=window_size,
            causal=False,
            fwd_params=FWD_128,
        )

        out_vanilla, _ = vanilla_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            logits_soft_cap=logits_soft_cap,
            sliding_window=window_size,
            causal=False,
        )

        assert out_ring.shape == out_vanilla.shape
        assert jnp.allclose(out_ring, out_vanilla, rtol=1e-2, atol=1e-2), (
            f"Ring attention with combined features differs from vanilla. "
            f"Max diff: {jnp.max(jnp.abs(out_ring - out_vanilla))}"
        )


class TestRingAttentionPallasDistributed:
    """Test distributed execution of Pallas TPU ring attention."""

    def test_distributed_shard_map(self):
        """Test distributed execution with shard_map - this is how Pallas should be used on TPU."""

        mesh = jax.make_mesh(
            (jax.process_count(), 1, 1, 1, jax.local_device_count()),
            ("dp", "fsdp", "ep", "tp", "sp"),
            (jax.sharding.AxisType.Auto,) * 5,
        )

        dp_size = jax.process_count()
        sp_size = jax.local_device_count()

        batch = max(1, 2 * dp_size)
        seq_len = 128 * sp_size
        num_heads, head_dim = 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = numeric_gen(num_sinks, dtype="f4") * -1.0

        out = shard_map(
            partial(pallas.tpu.ring_attention, axis_name="sp", fwd_params=FWD_128),
            in_specs=(
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                None,
                None,
                None,
                None,
                None,
            ),
            out_specs=PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            mesh=mesh,
            check_vma=False,
        )(q, k, v, None, None, None, None, softmax_aux)

        out_vanilla, _ = vanilla_attention(q, k, v, softmax_aux=softmax_aux, dtype=jnp.float32)

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))
        assert jnp.allclose(out, out_vanilla, rtol=1e-2, atol=1e-2), (
            f"Distributed ring attention output differs from vanilla. Max diff: {jnp.max(jnp.abs(out - out_vanilla))}"
        )

    def test_distributed_segment_ids_are_per_batch_row(self):
        """Each batch row must use its own packing boundaries, not row 0's (r12 S03-1)."""
        mesh = jax.sharding.Mesh(jax.devices()[:1], ("sp",))
        batch, seq_len, num_heads, head_dim = 2, 256, 2, 64
        kq, kk, kv = jax.random.split(jax.random.PRNGKey(3), 3)
        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), jnp.float32)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), jnp.float32)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), jnp.float32)
        # Row 0: one document. Row 1: two documents split at 128.
        seg = jnp.stack(
            [
                jnp.zeros((seq_len,), jnp.int32),
                jnp.concatenate([jnp.zeros((128,), jnp.int32), jnp.ones((128,), jnp.int32)]),
            ]
        )

        def dense(segments):
            logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision="highest") * head_dim**-0.5
            same = segments[:, None, :, None] == segments[:, None, None, :]
            logits = jnp.where(same, logits, -jnp.inf)
            return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), v, precision="highest")

        seq_spec = PartitionSpec(None, "sp", None, None)
        with jax.default_matmul_precision("highest"):
            out = shard_map(
                partial(pallas.tpu.ring_attention, axis_name="sp", fwd_params=FWD_128),
                in_specs=(seq_spec, seq_spec, seq_spec, PartitionSpec(None, "sp"), PartitionSpec(None, "sp")),
                out_specs=seq_spec,
                mesh=mesh,
                check_vma=False,
            )(q, k, v, seg, seg)

        expected = dense(seg)
        row0_everywhere = dense(jnp.broadcast_to(seg[:1], seg.shape))
        # Negative control: row 1's split boundary changes its output materially.
        assert float(jnp.max(jnp.abs(expected[1] - row0_everywhere[1]))) > 0.1
        assert not jnp.any(jnp.isnan(out))
        err = float(jnp.max(jnp.abs(out - expected)))
        assert err < 2e-2, err

    @pytest.mark.skipif(len(jax.devices()) < 2, reason="needs >= 2 devices for a rotating ring")
    def test_distributed_kv_segment_ids_rotate_with_kv(self):
        """Remote keys must be compared against their own segment ids, not the local shard's (r13 S03-2).

        Two ring devices, 256 tokens each. Device 0 holds documents [0]*128 + [1]*128 and device 1
        holds [1]*128 + [0]*128, so reusing the stationary local KV labels for rotated keys admits the
        wrong documents. Forward and dq are compared with a dense per-token segment-masked reference.
        """
        mesh = jax.sharding.Mesh(jax.devices()[:2], ("sp",))
        batch, seq_len, num_heads, head_dim = 1, 512, 2, 128
        kq, kk, kv, kw = jax.random.split(jax.random.PRNGKey(17), 4)
        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), jnp.float32)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), jnp.float32)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), jnp.float32)
        w = jax.random.normal(kw, (batch, seq_len, num_heads, head_dim), jnp.float32)
        zeros, ones = jnp.zeros((128,), jnp.int32), jnp.ones((128,), jnp.int32)
        seg = jnp.concatenate([zeros, ones, ones, zeros])[None, :]

        def dense(q, kv_labels):
            logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision="highest") * head_dim**-0.5
            same = seg[:, None, :, None] == kv_labels[:, None, None, :]
            logits = jnp.where(same, logits, -jnp.inf)
            return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), v, precision="highest")

        seq_spec = PartitionSpec(None, "sp", None, None)
        seg_spec = PartitionSpec(None, "sp")
        ring = shard_map(
            partial(pallas.tpu.ring_attention, axis_name="sp", fwd_params=FWD_128),
            in_specs=(seq_spec, seq_spec, seq_spec, seg_spec, seg_spec),
            out_specs=seq_spec,
            mesh=mesh,
            check_vma=False,
        )
        with jax.default_matmul_precision("highest"):
            out = ring(q, k, v, seg, seg)
            dq = jax.grad(lambda x: jnp.sum(ring(x, k, v, seg, seg) * w))(q)

        expected = dense(q, seg)
        # Negative control: labelling every key with device 0's local ids (the stationary-label bug as
        # seen from device 0) changes the reference materially.
        stationary = jnp.concatenate([seg[:, :256], seg[:, :256]], axis=1)
        assert float(jnp.max(jnp.abs(expected - dense(q, stationary)))) > 0.1
        assert not jnp.any(jnp.isnan(out))
        assert float(jnp.max(jnp.abs(out - expected))) < 2e-2
        ref_dq = jax.grad(lambda x: jnp.sum(dense(x, seg) * w))(q)
        assert jnp.all(jnp.isfinite(dq))
        assert float(jnp.max(jnp.abs(dq - ref_dq))) < 3e-2 * float(jnp.max(jnp.abs(ref_dq))) + 1e-2


def _assert_grad_close(got, ref, rel=3e-2, atol=1e-2):
    assert jnp.all(jnp.isfinite(got))
    err = float(jnp.max(jnp.abs(got - ref)))
    bound = rel * float(jnp.max(jnp.abs(ref))) + atol
    assert err < bound, (err, bound)


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs >= 2 devices for a rotating ring")
class TestRingAttentionPallasRingMaskGradients:
    """Backward masks and sink gradients across a rotating 2-device ring (r15 S03-1/2/3).

    Each device holds 256 tokens, so every ring step after the first attends to a remote KV shard.
    The backward must apply the same global-position mask as the forward to those remote blocks, and
    the sink gradient must be counted once, not once per ring step.
    """

    SEQ, HEADS, DIM = 512, 2, 128

    def _inputs(self, seed):
        keys = jax.random.split(jax.random.PRNGKey(seed), 4)
        shape = (1, self.SEQ, self.HEADS, self.DIM)
        return [jax.random.normal(key, shape, jnp.float32) for key in keys]

    def _dense(self, q, k, v, sinks=None, chunk_size=None):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision="highest") * self.DIM**-0.5
        pos = jnp.arange(self.SEQ)
        allowed = pos[None, :] <= pos[:, None]
        if chunk_size is not None:
            allowed = allowed & ((pos[None, :] // chunk_size) == (pos[:, None] // chunk_size))
        logits = jnp.where(allowed, logits, -jnp.inf)
        if sinks is not None:
            aux = jnp.broadcast_to(sinks[None, :, None, None], (1, self.HEADS, self.SEQ, 1))
            weights = jax.nn.softmax(jnp.concatenate([logits, aux], -1), -1)[..., : self.SEQ]
        else:
            weights = jax.nn.softmax(logits, -1)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v, precision="highest")

    def _ring(self, with_sinks=False, **kwargs):
        mesh = jax.sharding.Mesh(jax.devices()[:2], ("sp",))
        seq_spec = PartitionSpec(None, "sp", None, None)
        call = partial(pallas.tpu.ring_attention, axis_name="sp", causal=True, fwd_params=FWD_128, **kwargs)
        if with_sinks:
            return shard_map(
                lambda q, k, v, aux: call(q, k, v, softmax_aux=aux),
                in_specs=(seq_spec, seq_spec, seq_spec, PartitionSpec()),
                out_specs=seq_spec,
                mesh=mesh,
                check_vma=False,
            )
        return shard_map(
            lambda q, k, v: call(q, k, v),
            in_specs=(seq_spec, seq_spec, seq_spec),
            out_specs=seq_spec,
            mesh=mesh,
            check_vma=False,
        )

    @pytest.mark.parametrize("chunk_size", [None, 128])
    def test_causal_grads_match_dense(self, chunk_size):
        q, k, v, w = self._inputs(21)
        ring = self._ring(chunk_size=chunk_size)
        with jax.default_matmul_precision("highest"):
            out = ring(q, k, v)
            grads = jax.grad(lambda q, k, v: jnp.sum(ring(q, k, v) * w), argnums=(0, 1, 2))(q, k, v)
        expected = self._dense(q, k, v, chunk_size=chunk_size)
        ref_grads = jax.grad(
            lambda q, k, v: jnp.sum(self._dense(q, k, v, chunk_size=chunk_size) * w), argnums=(0, 1, 2)
        )(q, k, v)
        if chunk_size is not None:
            # Negative control: dropping the chunk term changes the reference materially.
            assert float(jnp.max(jnp.abs(expected - self._dense(q, k, v)))) > 0.1
        assert float(jnp.max(jnp.abs(out - expected))) < 2e-2
        for got, ref in zip(grads, ref_grads, strict=True):
            _assert_grad_close(got, ref)

    def test_sink_gradient_is_counted_once(self):
        q, k, v, w = self._inputs(23)
        # One sink per query head. The kernel reads 1-D sinks as a shared set (it only sees local head
        # counts inside shard_map), so per-head sinks are passed as 2-D ``(Hq, 1)``.
        sinks = jnp.array([0.5, -1.0], jnp.float32)
        ring2d = self._ring(with_sinks=True)

        def ring(q, k, v, s):
            return ring2d(q, k, v, s[:, None])

        with jax.default_matmul_precision("highest"):
            out = ring(q, k, v, sinks)
            dq, dsinks = jax.grad(lambda q, s: jnp.sum(ring(q, k, v, s) * w), argnums=(0, 1))(q, sinks)
        expected = self._dense(q, k, v, sinks)
        ref_dq, ref_dsinks = jax.grad(lambda q, s: jnp.sum(self._dense(q, k, v, s) * w), argnums=(0, 1))(q, sinks)
        # Negative control: a doubled sink gradient is far outside the tolerance.
        assert float(jnp.max(jnp.abs(ref_dsinks))) > 0.5
        assert float(jnp.max(jnp.abs(out - expected))) < 2e-2
        _assert_grad_close(dq, ref_dq)
        _assert_grad_close(dsinks, ref_dsinks, rel=2e-2, atol=1e-2)

    def test_chunk_size_must_divide_local_kv_len(self):
        q, k, v, _ = self._inputs(25)
        with pytest.raises(NotImplementedError, match="multiple of chunk_size"):
            self._ring(chunk_size=96)(q, k, v)


class TestRingAttentionPallasDisabledSinks:
    def test_shared_1d_disabled_sinks_have_zero_not_nan_gradient(self):
        """All--inf shared 1-D ring sinks must give a zero, finite gradient (r14 S03-2)."""
        mesh = jax.sharding.Mesh(jax.devices()[:1], ("sp",))
        q = jnp.zeros((1, 256, 2, 128), jnp.float32)
        k = jnp.zeros((1, 256, 2, 128), jnp.float32)
        v = jnp.ones((1, 256, 2, 128), jnp.float32)
        seq_spec = PartitionSpec(None, "sp", None, None)
        ring = shard_map(
            lambda q, k, v, aux: pallas.tpu.ring_attention(q, k, v, softmax_aux=aux, axis_name="sp", fwd_params=FWD_128),
            in_specs=(seq_spec, seq_spec, seq_spec, PartitionSpec()),
            out_specs=seq_spec,
            mesh=mesh,
            check_vma=False,
        )
        disabled = jnp.full((2,), -jnp.inf, jnp.float32)
        out = ring(q, k, v, disabled)
        assert jnp.allclose(out, 1.0, atol=1e-5)
        g = jax.grad(lambda a: jnp.sum(ring(q, k, v, a)))(disabled)
        assert jnp.all(jnp.isfinite(g)) and float(jnp.max(jnp.abs(g))) == 0.0, g


class TestRingAttentionPallasEdgeCases:
    """Test edge cases for Pallas TPU ring attention."""

    def test_small_sequence(self):
        """Test with small sequence length."""
        batch, seq_len, num_heads, head_dim = 1, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_different_chunk_sizes(self):
        """Test with different query and key chunk sizes."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            fwd_params=FwdParams(q_blocksize=128, kv_blocksize=256),
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_large_head_dim(self):
        """Test with large head dimension."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            fwd_params=FWD_128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
