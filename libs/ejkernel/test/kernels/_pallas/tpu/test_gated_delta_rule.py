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

"""TPU Pallas tests for the gated_delta_rule kernel.

This module verifies the Pallas TPU implementation of the gated delta rule (a linear-attention
recurrence used by Qwen3-Next/3.5 style models) against the reference XLA implementation. Tests
cover output/state shapes and finiteness, numerical agreement between the chunked Pallas path and
the recurrent XLA path (forward and backward), single-step decode continuation, dtype handling
(float32 and bfloat16), the ``use_chunked=False`` fallback that defers to the XLA recurrent kernel,
realistic model-sized dimensions, and segmented (padded-tail) behavior.

All tests are skipped unless a TPU backend is available (see ``pytestmark``). The two kernels under
comparison are imported as ``gated_delta_rule_pallas`` (Pallas TPU) and ``gated_delta_rule_xla``
(XLA reference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.gated_delta_rule import gated_delta_rule as gated_delta_rule_pallas
from ejkernel.kernels._xla.gated_delta_rule import gated_delta_rule as gated_delta_rule_xla
from ejkernel.kernels._xla.gated_delta_rule._xla_impl_fwd import _l2norm_with_inv


def _has_tpu() -> bool:
    """Report whether at least one TPU device is visible to JAX.

    Used to build the module-level skip marker so the Pallas TPU tests only run on a TPU backend.

    Returns:
        bool: ``True`` if ``jax.devices("tpu")`` returns one or more devices; ``False`` if no TPU is
        present or the device query raises (any exception is swallowed and treated as "no TPU").
    """
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def _make_inputs(batch=1, seq_len=8, heads=2, qk_dim=8, v_dim=8, dtype=jnp.float32, seed=0):
    """Generate a random set of gated-delta-rule inputs for a test case.

    Builds query/key/value tensors plus the per-step gate (``beta``) and log-decay (``decay``) signals
    expected by the kernels. ``beta`` is squashed into ``(0, 1)`` via a sigmoid and ``decay`` is scaled
    to small negative values (``* -0.01``) to emulate realistic, mildly decaying gates.

    Args:
        batch (int): Number of sequences in the batch. Defaults to 1.
        seq_len (int): Sequence length (number of time steps). Defaults to 8.
        heads (int): Number of attention heads. Defaults to 2.
        qk_dim (int): Per-head dimensionality of the query and key vectors. Defaults to 8.
        v_dim (int): Per-head dimensionality of the value vectors. Defaults to 8.
        dtype: JAX dtype for all generated arrays (e.g. ``jnp.float32`` or ``jnp.bfloat16``).
            Defaults to ``jnp.float32``.
        seed (int): Seed for the ``jax.random.PRNGKey`` driving all draws. Defaults to 0.

    Returns:
        tuple: A 5-tuple ``(q, k, v, beta, decay)`` where:
            - ``q`` has shape ``(batch, seq_len, heads, qk_dim)``.
            - ``k`` has shape ``(batch, seq_len, heads, qk_dim)``.
            - ``v`` has shape ``(batch, seq_len, heads, v_dim)``.
            - ``beta`` has shape ``(batch, seq_len, heads)`` with values in ``(0, 1)``.
            - ``decay`` has shape ``(batch, seq_len, heads)`` with small negative log-decay values.
        All arrays use the requested ``dtype``.
    """
    key = jax.random.PRNGKey(seed)
    kq, kk, kv, kb, kd = jax.random.split(key, 5)
    q = jax.random.normal(kq, (batch, seq_len, heads, qk_dim), dtype=dtype)
    k = jax.random.normal(kk, (batch, seq_len, heads, qk_dim), dtype=dtype)
    v = jax.random.normal(kv, (batch, seq_len, heads, v_dim), dtype=dtype)
    beta = jax.nn.sigmoid(jax.random.normal(kb, (batch, seq_len, heads), dtype=dtype))
    decay = jax.random.normal(kd, (batch, seq_len, heads), dtype=dtype) * -0.01
    return q, k, v, beta, decay


def test_pallas_interface_returns_shapes_and_finite():
    """Check the chunked Pallas kernel returns correctly shaped, finite output and state.

    Runs the Pallas kernel with ``chunk_size=8`` and ``use_chunked=True`` and asserts the output matches
    the value tensor shape, the returned recurrent state has shape ``(batch, heads, qk_dim, v_dim)``, and
    both output and state contain only finite values.
    """
    q, k, v, beta, decay = _make_inputs(dtype=jnp.float32)
    out, state = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    assert out.shape == v.shape
    assert state.shape == (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.isfinite(state))


def test_pallas_chunked_matches_xla_recurrent():
    """Verify the chunked Pallas forward matches the recurrent XLA forward.

    Runs both kernels over a 16-step float32 sequence (Pallas with ``use_chunked=True``, XLA with
    ``use_chunked=False``) and asserts the outputs and final states agree to an absolute tolerance of
    ``2e-2``.
    """
    q, k, v, beta, decay = _make_inputs(seq_len=16, dtype=jnp.float32, seed=1)
    out_pallas, state_pallas = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    out_xla, state_xla = gated_delta_rule_xla(q, k, v, beta, decay, chunk_size=8, use_chunked=False)

    assert jnp.allclose(out_pallas, out_xla, atol=2e-2, rtol=0)
    assert jnp.allclose(state_pallas, state_xla, atol=2e-2, rtol=0)


def test_pallas_no_decay():
    """Check the chunked Pallas kernel accepts ``decay=None`` and returns finite results.

    Passes ``None`` for the decay signal (no gating decay) with bfloat16 inputs and asserts the output and
    state have the expected shapes and contain only finite values.
    """
    q, k, v, beta, _ = _make_inputs(dtype=jnp.bfloat16, seed=2)
    out, state = gated_delta_rule_pallas(q, k, v, beta, None, chunk_size=8, use_chunked=True)
    assert out.shape == v.shape
    assert state.shape == (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.isfinite(state))


def test_pallas_single_step_continuation():
    """Verify stateful continuation: prefix + single-step decode equals a full forward.

    Processes a 9-step sequence three ways and checks consistency: (1) a full chunked forward over all 9
    steps, (2) a chunked forward over the first 8 steps to obtain a carry state, and (3) a single-step
    forward over step 9 seeded with that carry via ``initial_state``. Asserts the single-step output has
    shape ``(batch, 1, heads, v_dim)``, the returned state matches the carried state's shape, and the
    single-step output matches the corresponding slice of the full forward (atol ``2e-2``).
    """
    q, k, v, beta, decay = _make_inputs(seq_len=9, dtype=jnp.float32, seed=3)
    out_full, state_full = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    _, state = gated_delta_rule_pallas(q[:, :8], k[:, :8], v[:, :8], beta[:, :8], decay[:, :8], chunk_size=8)
    out_last, state_last = gated_delta_rule_pallas(
        q[:, 8:],
        k[:, 8:],
        v[:, 8:],
        beta[:, 8:],
        decay[:, 8:],
        initial_state=state,
    )
    assert out_last.shape == (q.shape[0], 1, q.shape[2], v.shape[3])
    assert state_last.shape == state.shape
    assert jnp.allclose(out_last, out_full[:, 8:], atol=2e-2, rtol=0)
    assert jnp.allclose(state_last, state_full, atol=2e-2, rtol=0)


@pytest.mark.parametrize(("dtype", "seed"), ((jnp.float32, 13), (jnp.bfloat16, 33)))
def test_pallas_single_step_matches_xla(dtype, seed):
    """Verify a single-step Pallas decode matches the recurrent XLA decode, preserving dtype.

    Parameterized over float32 and bfloat16. Runs both kernels for a length-1 sequence seeded with a
    random ``initial_state`` and asserts the Pallas output and state keep the input dtype and match the XLA
    results within a dtype-dependent tolerance (``1e-5`` for float32, ``0.02`` for bfloat16).

    Args:
        dtype: JAX dtype under test (``jnp.float32`` or ``jnp.bfloat16``), supplied by parametrization.
        seed (int): PRNG seed for input and initial-state generation, supplied by parametrization.
    """
    q, k, v, beta, decay = _make_inputs(seq_len=1, dtype=dtype, seed=seed)
    init = jax.random.normal(
        jax.random.PRNGKey(seed + 1),
        (q.shape[0], q.shape[2], q.shape[3], v.shape[3]),
        dtype=dtype,
    )
    out_pallas, state_pallas = gated_delta_rule_pallas(q, k, v, beta, decay, initial_state=init)
    out_xla, state_xla = gated_delta_rule_xla(q, k, v, beta, decay, initial_state=init, use_chunked=False)
    assert out_pallas.dtype == dtype
    assert state_pallas.dtype == dtype
    atol = 0.02 if dtype == jnp.bfloat16 else 1e-5
    assert jnp.allclose(out_pallas, out_xla, atol=atol, rtol=0)
    assert jnp.allclose(state_pallas, state_xla, atol=atol, rtol=0)


def test_pallas_bfloat16_inputs():
    """Check the chunked Pallas kernel runs on bfloat16 inputs and returns finite results.

    Asserts the output and state dtypes are one of ``bfloat16``/``float32`` (the kernel may up-cast its
    accumulation) and that both contain only finite values.
    """
    q, k, v, beta, decay = _make_inputs(dtype=jnp.bfloat16, seed=4)
    out, state = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    assert out.dtype in (jnp.bfloat16, jnp.float32)
    assert state.dtype in (jnp.bfloat16, jnp.float32)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.isfinite(state))


def test_pallas_bfloat16_input_dtype_phase1_matches_xla():
    """Verify the lower-HBM forward-only phase-1 dtype option stays numerically close to XLA."""
    q, k, v, beta, decay = _make_inputs(
        batch=1,
        seq_len=256,
        heads=4,
        qk_dim=128,
        v_dim=128,
        dtype=jnp.bfloat16,
        seed=41,
    )
    out_p, state_p = gated_delta_rule_pallas(
        q,
        k,
        v,
        beta,
        decay,
        chunk_size=256,
        use_chunked=True,
        use_input_dtype_phase1_outputs=True,
    )
    out_x, state_x = gated_delta_rule_xla(q, k, v, beta, decay, chunk_size=256, use_chunked=False)

    out_diff = jnp.max(jnp.abs(out_p.astype(jnp.float32) - out_x.astype(jnp.float32)))
    state_diff = jnp.max(jnp.abs(state_p.astype(jnp.float32) - state_x.astype(jnp.float32)))
    assert jnp.all(jnp.isfinite(out_p))
    assert jnp.all(jnp.isfinite(state_p))
    assert jnp.allclose(out_p, out_x, atol=5e-2, rtol=0), f"Output max diff: {out_diff}"
    assert jnp.allclose(state_p, state_x, atol=5e-2, rtol=0), f"State max diff: {state_diff}"


def test_pallas_forward_only_in_kernel_qk_norm_matches_external_norm():
    """Forward-only Pallas Q/K norm fusion should match the old external-normalize path."""
    q, k, v, beta, decay = _make_inputs(
        batch=1,
        seq_len=256,
        heads=4,
        qk_dim=128,
        v_dim=128,
        dtype=jnp.bfloat16,
        seed=42,
    )
    out_fused, state_fused = gated_delta_rule_pallas(
        q,
        k,
        v,
        beta,
        decay,
        chunk_size=256,
        use_chunked=True,
        use_qk_l2norm=True,
        use_input_dtype_phase1_outputs=True,
        use_input_dtype_state=True,
    )
    q_norm, _ = _l2norm_with_inv(q, axis=-1, eps=1e-6)
    k_norm, _ = _l2norm_with_inv(k, axis=-1, eps=1e-6)
    out_external, state_external = gated_delta_rule_pallas(
        q_norm,
        k_norm,
        v,
        beta,
        decay,
        chunk_size=256,
        use_chunked=True,
        use_qk_l2norm=False,
        use_input_dtype_phase1_outputs=True,
        use_input_dtype_state=True,
    )

    assert jnp.allclose(out_fused, out_external, atol=5e-2, rtol=0)
    assert jnp.allclose(state_fused, state_external, atol=5e-2, rtol=0)


def test_pallas_grouped_prefill_matches_repeated_heads():
    """Grouped Pallas prefill should match the old materialized Q/K repeat path."""
    batch, seq_len, q_heads, expand_ratio, qk_dim, v_dim = 1, 64, 2, 2, 64, 64
    v_heads = q_heads * expand_ratio
    q, k, _, _, _ = _make_inputs(
        batch=batch,
        seq_len=seq_len,
        heads=q_heads,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=jnp.bfloat16,
        seed=43,
    )
    _, _, v, beta, decay = _make_inputs(
        batch=batch,
        seq_len=seq_len,
        heads=v_heads,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=jnp.bfloat16,
        seed=44,
    )

    out_grouped, state_grouped = gated_delta_rule_pallas(
        q,
        k,
        v,
        beta,
        decay,
        chunk_size=64,
        use_chunked=True,
        use_input_dtype_phase1_outputs=True,
        use_input_dtype_state=True,
    )
    out_repeated, state_repeated = gated_delta_rule_pallas(
        jnp.repeat(q, expand_ratio, axis=2),
        jnp.repeat(k, expand_ratio, axis=2),
        v,
        beta,
        decay,
        chunk_size=64,
        use_chunked=True,
        use_input_dtype_phase1_outputs=True,
        use_input_dtype_state=True,
    )

    assert out_grouped.shape == v.shape
    assert state_grouped.shape == (batch, v_heads, qk_dim, v_dim)
    assert jnp.allclose(out_grouped.astype(jnp.float32), out_repeated.astype(jnp.float32), atol=5e-2, rtol=0)
    assert jnp.allclose(state_grouped.astype(jnp.float32), state_repeated.astype(jnp.float32), atol=5e-2, rtol=0)


def test_pallas_grouped_bfloat16_input_dtype_flags_backward_compiles():
    """Grouped bf16 low-HBM dtype flags must keep the custom-VJP primal dtype contract."""
    batch, seq_len, q_heads, expand_ratio, qk_dim, v_dim = 1, 64, 2, 2, 64, 64
    v_heads = q_heads * expand_ratio
    q, k, _, _, _ = _make_inputs(
        batch=batch,
        seq_len=seq_len,
        heads=q_heads,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=jnp.bfloat16,
        seed=45,
    )
    _, _, v, beta, decay = _make_inputs(
        batch=batch,
        seq_len=seq_len,
        heads=v_heads,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=jnp.bfloat16,
        seed=46,
    )

    def loss_fn(q_in, k_in, v_in, beta_in, decay_in):
        out, state = gated_delta_rule_pallas(
            q_in,
            k_in,
            v_in,
            beta_in,
            decay_in,
            chunk_size=64,
            use_chunked=True,
            use_input_dtype_phase1_outputs=True,
            use_input_dtype_state=True,
        )
        assert out.dtype == jnp.bfloat16
        assert state.dtype == jnp.bfloat16
        return jnp.sum(out.astype(jnp.float32)) + 0.1 * jnp.sum(state.astype(jnp.float32))

    value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)

    assert jnp.isfinite(value)
    for grad, reference in zip(grads, (q, k, v, beta, decay), strict=True):
        assert grad.shape == reference.shape
        assert jnp.all(jnp.isfinite(grad))


def test_pallas_bfloat16_input_dtype_flags_backward_compiles():
    """Non-grouped bf16 low-HBM dtype flags must keep the custom-VJP primal dtype contract."""
    q, k, v, beta, decay = _make_inputs(
        batch=1,
        seq_len=64,
        heads=4,
        qk_dim=64,
        v_dim=64,
        dtype=jnp.bfloat16,
        seed=47,
    )

    def loss_fn(q_in, k_in, v_in, beta_in, decay_in):
        out, state = gated_delta_rule_pallas(
            q_in,
            k_in,
            v_in,
            beta_in,
            decay_in,
            chunk_size=64,
            use_chunked=True,
            use_input_dtype_phase1_outputs=True,
            use_input_dtype_state=True,
        )
        assert out.dtype == jnp.bfloat16
        assert state.dtype == jnp.bfloat16
        return jnp.sum(out.astype(jnp.float32)) + 0.1 * jnp.sum(state.astype(jnp.float32))

    value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)

    assert jnp.isfinite(value)
    for grad, reference in zip(grads, (q, k, v, beta, decay), strict=True):
        assert grad.shape == reference.shape
        assert jnp.all(jnp.isfinite(grad))


def test_pallas_backward_matches_xla_recurrent():
    """Verify Pallas gradients match the recurrent XLA gradients for the chunked forward.

    Defines a scalar loss ``sum(out) + 0.1 * sum(state)`` for each kernel, takes gradients w.r.t. all five
    inputs ``(q, k, v, beta, decay)``, and asserts each Pallas gradient has the right shape, is finite, and
    matches the corresponding XLA gradient to atol ``2e-2``.
    """
    q, k, v, beta, decay = _make_inputs(dtype=jnp.float32, seed=5)

    def loss_pallas(q_in, k_in, v_in, beta_in, decay_in):
        """Scalar loss over the chunked Pallas forward, used to drive ``jax.grad``.

        Args:
            q_in: Query tensor, shape ``(batch, seq_len, heads, qk_dim)``.
            k_in: Key tensor, shape ``(batch, seq_len, heads, qk_dim)``.
            v_in: Value tensor, shape ``(batch, seq_len, heads, v_dim)``.
            beta_in: Gate tensor, shape ``(batch, seq_len, heads)``.
            decay_in: Log-decay tensor, shape ``(batch, seq_len, heads)``.

        Returns:
            jax.Array: Scalar loss ``sum(out) + 0.1 * sum(state)`` from the Pallas chunked kernel.
        """
        out, state = gated_delta_rule_pallas(q_in, k_in, v_in, beta_in, decay_in, chunk_size=8, use_chunked=True)
        return jnp.sum(out) + 0.1 * jnp.sum(state)

    def loss_xla(q_in, k_in, v_in, beta_in, decay_in):
        """Scalar loss over the recurrent XLA forward, the reference for the gradient comparison.

        Args:
            q_in: Query tensor, shape ``(batch, seq_len, heads, qk_dim)``.
            k_in: Key tensor, shape ``(batch, seq_len, heads, qk_dim)``.
            v_in: Value tensor, shape ``(batch, seq_len, heads, v_dim)``.
            beta_in: Gate tensor, shape ``(batch, seq_len, heads)``.
            decay_in: Log-decay tensor, shape ``(batch, seq_len, heads)``.

        Returns:
            jax.Array: Scalar loss ``sum(out) + 0.1 * sum(state)`` from the XLA recurrent kernel.
        """
        out, state = gated_delta_rule_xla(q_in, k_in, v_in, beta_in, decay_in, chunk_size=8, use_chunked=False)
        return jnp.sum(out) + 0.1 * jnp.sum(state)

    grads_pallas = jax.grad(loss_pallas, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    grads_xla = jax.grad(loss_xla, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)

    for grad_pallas, grad_xla, reference in zip(grads_pallas, grads_xla, (q, k, v, beta, decay), strict=True):
        assert grad_pallas.shape == reference.shape
        assert jnp.all(jnp.isfinite(grad_pallas))
        assert jnp.allclose(grad_pallas, grad_xla, atol=2e-2, rtol=0)


def test_pallas_single_step_backward_matches_xla():
    """Verify single-step Pallas gradients (including w.r.t. the initial state) match XLA.

    Uses a length-1 sequence seeded with a small random ``initial_state`` and takes gradients of the loss
    ``sum(out) + 0.1 * sum(state)`` w.r.t. all six inputs ``(q, k, v, beta, decay, state)`` for both
    kernels. Asserts each Pallas gradient is finite and matches the XLA gradient to atol ``1e-5``.
    """
    q, k, v, beta, decay = _make_inputs(seq_len=1, dtype=jnp.float32, seed=50)
    init = (
        jax.random.normal(
            jax.random.PRNGKey(51),
            (q.shape[0], q.shape[2], q.shape[3], v.shape[3]),
            dtype=jnp.float32,
        )
        * 0.01
    )

    def loss_pallas(q_in, k_in, v_in, beta_in, decay_in, state_in):
        """Scalar loss over the single-step Pallas forward seeded with ``state_in``.

        Args:
            q_in: Query tensor, shape ``(batch, 1, heads, qk_dim)``.
            k_in: Key tensor, shape ``(batch, 1, heads, qk_dim)``.
            v_in: Value tensor, shape ``(batch, 1, heads, v_dim)``.
            beta_in: Gate tensor, shape ``(batch, 1, heads)``.
            decay_in: Log-decay tensor, shape ``(batch, 1, heads)``.
            state_in: Initial recurrent state, shape ``(batch, heads, qk_dim, v_dim)``.

        Returns:
            jax.Array: Scalar loss ``sum(out) + 0.1 * sum(state)`` from the Pallas kernel.
        """
        out, state = gated_delta_rule_pallas(q_in, k_in, v_in, beta_in, decay_in, initial_state=state_in)
        return jnp.sum(out) + 0.1 * jnp.sum(state)

    def loss_xla(q_in, k_in, v_in, beta_in, decay_in, state_in):
        """Scalar loss over the single-step recurrent XLA forward seeded with ``state_in``.

        Args:
            q_in: Query tensor, shape ``(batch, 1, heads, qk_dim)``.
            k_in: Key tensor, shape ``(batch, 1, heads, qk_dim)``.
            v_in: Value tensor, shape ``(batch, 1, heads, v_dim)``.
            beta_in: Gate tensor, shape ``(batch, 1, heads)``.
            decay_in: Log-decay tensor, shape ``(batch, 1, heads)``.
            state_in: Initial recurrent state, shape ``(batch, heads, qk_dim, v_dim)``.

        Returns:
            jax.Array: Scalar loss ``sum(out) + 0.1 * sum(state)`` from the XLA recurrent kernel.
        """
        out, state = gated_delta_rule_xla(
            q_in,
            k_in,
            v_in,
            beta_in,
            decay_in,
            initial_state=state_in,
            use_chunked=False,
        )
        return jnp.sum(out) + 0.1 * jnp.sum(state)

    grads_pallas = jax.grad(loss_pallas, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, beta, decay, init)
    grads_xla = jax.grad(loss_xla, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, beta, decay, init)
    for grad_pallas, grad_xla in zip(grads_pallas, grads_xla, strict=True):
        assert jnp.all(jnp.isfinite(grad_pallas))
        assert jnp.allclose(grad_pallas, grad_xla, atol=1e-5, rtol=0)


def test_pallas_use_chunked_false_falls_back_to_xla_recurrent():
    """Verify ``use_chunked=False`` makes the Pallas entry point defer exactly to the XLA recurrent kernel.

    Runs both kernels with ``use_chunked=False`` and asserts bit-exact equality (``jnp.array_equal``) of
    both output and state, confirming the Pallas wrapper dispatches to the XLA path unchanged.
    """
    q, k, v, beta, decay = _make_inputs(dtype=jnp.float32, seed=6)
    out_pallas, state_pallas = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=False)
    out_xla, state_xla = gated_delta_rule_xla(q, k, v, beta, decay, chunk_size=8, use_chunked=False)
    assert jnp.array_equal(out_pallas, out_xla)
    assert jnp.array_equal(state_pallas, state_xla)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_pallas_realistic_dims_fwd_bwd(dtype):
    """Test the chunked Pallas forward and backward at realistic model dimensions.

    Uses ``qk_dim=v_dim=128``, ``chunk_size=64`` over a 512-step sequence (many chunks), parameterized over
    float32 and bfloat16. Asserts the forward output has the value shape and is finite (output and state),
    then takes gradients w.r.t. all five inputs and asserts every gradient is finite.

    Args:
        dtype: JAX dtype under test (``jnp.float32`` or ``jnp.bfloat16``), supplied by parametrization.
    """
    batch, seq_len, heads, qk_dim, v_dim = 1, 512, 2, 128, 128
    q, k, v, beta, decay = _make_inputs(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=dtype,
        seed=7,
    )

    out, state = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=64, use_chunked=True)
    assert out.shape == v.shape
    assert jnp.all(jnp.isfinite(out)), "Forward output contains NaN/Inf"
    assert jnp.all(jnp.isfinite(state)), "Forward state contains NaN/Inf"

    def loss_fn(q_in, k_in, v_in, beta_in, decay_in):
        """Scalar loss over the realistic-dimension chunked Pallas forward, used for ``jax.grad``.

        Args:
            q_in: Query tensor, shape ``(batch, seq_len, heads, qk_dim)``.
            k_in: Key tensor, shape ``(batch, seq_len, heads, qk_dim)``.
            v_in: Value tensor, shape ``(batch, seq_len, heads, v_dim)``.
            beta_in: Gate tensor, shape ``(batch, seq_len, heads)``.
            decay_in: Log-decay tensor, shape ``(batch, seq_len, heads)``.

        Returns:
            jax.Array: Scalar loss ``sum(out) + 0.1 * sum(state)`` from the Pallas chunked kernel.
        """
        o, s = gated_delta_rule_pallas(q_in, k_in, v_in, beta_in, decay_in, chunk_size=64, use_chunked=True)
        return jnp.sum(o) + 0.1 * jnp.sum(s)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    for i, g in enumerate(grads):
        assert jnp.all(jnp.isfinite(g)), f"Gradient {i} contains NaN/Inf"


def test_pallas_realistic_dims_matches_xla():
    """Verify the chunked Pallas forward matches the recurrent XLA forward at realistic dimensions.

    Uses float32 inputs with ``qk_dim=v_dim=128`` over a 256-step sequence and ``chunk_size=64``. Asserts
    the Pallas output and final state match the XLA reference to atol ``2e-2`` (assertion messages report
    the observed max absolute difference).
    """
    q, k, v, beta, decay = _make_inputs(
        batch=1,
        seq_len=256,
        heads=2,
        qk_dim=128,
        v_dim=128,
        dtype=jnp.float32,
        seed=8,
    )
    out_p, st_p = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=64, use_chunked=True)
    out_x, st_x = gated_delta_rule_xla(q, k, v, beta, decay, chunk_size=64, use_chunked=False)
    assert jnp.allclose(out_p, out_x, atol=2e-2, rtol=0), f"Output max diff: {jnp.max(jnp.abs(out_p - out_x))}"
    assert jnp.allclose(st_p, st_x, atol=2e-2, rtol=0), f"State max diff: {jnp.max(jnp.abs(st_p - st_x))}"


def test_segmented_gdr_ignores_high_t_padding_tail():
    """Verify a long padding tail (negative segment ids) emits no output and does not corrupt the state.

    Builds a 16,384-step sequence with only the first ``valid_len=128`` steps marked as a real segment
    (id 0) and the rest marked as padding (id ``-1``) via ``seg_ids``. Runs the Pallas kernel over the full
    padded sequence and the XLA kernel over just the valid prefix, then asserts: the Pallas outputs over the
    valid prefix match the prefix reference (atol ``1e-5``); the Pallas outputs over the padding tail are
    exactly zero; and the final recurrent states agree (atol ``1e-5``), confirming the padding tail is
    fully ignored.
    """
    batch, seq_len, valid_len, heads, qk_dim, v_dim = 1, 16_384, 128, 1, 8, 8
    q, k, v, beta, decay = _make_inputs(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=jnp.float32,
        seed=70,
    )
    seg_ids = jnp.concatenate(
        [
            jnp.zeros((batch, valid_len), dtype=jnp.int32),
            -jnp.ones((batch, seq_len - valid_len), dtype=jnp.int32),
        ],
        axis=1,
    )
    prefix_seg_ids = jnp.zeros((batch, valid_len), dtype=jnp.int32)

    out_full, state_full = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=256, seg_ids=seg_ids)
    out_prefix, state_prefix = gated_delta_rule_xla(
        q[:, :valid_len],
        k[:, :valid_len],
        v[:, :valid_len],
        beta[:, :valid_len],
        decay[:, :valid_len],
        chunk_size=256,
        seg_ids=prefix_seg_ids,
    )

    assert jnp.allclose(out_full[:, :valid_len], out_prefix, atol=1e-5, rtol=0)
    assert jnp.max(jnp.abs(out_full[:, valid_len:])) == 0
    assert jnp.allclose(state_full, state_prefix, atol=1e-5, rtol=0)
