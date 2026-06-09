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


"""Tests for XLA SSM1 (Mamba1-style) state space implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import state_space_v1


class TestStateSpaceV1:
    """Test suite for SSM1 (Mamba1-style) kernel."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        batch, seq_len, intermediate_size, ssm_state_size = 2, 10, 256, 16

        hidden_states = jnp.ones((batch, seq_len, intermediate_size))
        A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.1
        B = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        D = jnp.ones((intermediate_size,))
        dt = jnp.ones((batch, seq_len, intermediate_size)) * 0.1

        output, ssm_state, conv_state = state_space_v1(hidden_states, A, B, C, D, dt)

        assert output.shape == (batch, seq_len, intermediate_size)
        assert ssm_state.shape == (batch, intermediate_size, ssm_state_size)
        assert conv_state is None

    def test_gradient_shapes(self):
        """Test gradient shapes match input shapes."""
        batch, seq_len, intermediate_size, ssm_state_size = 1, 5, 64, 8

        hidden_states = jnp.ones((batch, seq_len, intermediate_size)) * 0.5
        A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.1
        B = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        D = jnp.ones((intermediate_size,))
        dt = jnp.ones((batch, seq_len, intermediate_size)) * 0.1

        def loss_fn(hidden_states, A, B, C, D, dt):
            o, _, _ = state_space_v1(hidden_states, A, B, C, D, dt)
            return jnp.sum(o)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(hidden_states, A, B, C, D, dt)

        assert grads[0].shape == hidden_states.shape
        assert grads[1].shape == A.shape
        assert grads[2].shape == B.shape
        assert grads[3].shape == C.shape
        assert grads[4].shape == D.shape
        assert grads[5].shape == dt.shape

    def test_with_initial_state(self):
        """Test with non-zero initial state."""
        batch, seq_len, intermediate_size, ssm_state_size = 2, 5, 64, 8

        hidden_states = jnp.ones((batch, seq_len, intermediate_size))
        A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.1
        B = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        D = jnp.ones((intermediate_size,))
        dt = jnp.ones((batch, seq_len, intermediate_size)) * 0.1

        initial_state = jnp.ones((batch, intermediate_size, ssm_state_size)) * 0.5

        output, ssm_state, _conv_state = state_space_v1(hidden_states, A, B, C, D, dt, initial_state=initial_state)

        assert output.shape == (batch, seq_len, intermediate_size)
        assert ssm_state.shape == (batch, intermediate_size, ssm_state_size)
        assert not jnp.allclose(ssm_state, initial_state)

    def test_with_gating(self):
        """Test with gating applied."""
        batch, seq_len, intermediate_size, ssm_state_size = 2, 8, 64, 8

        hidden_states = jnp.ones((batch, seq_len, intermediate_size))
        A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.1
        B = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        D = jnp.ones((intermediate_size,))
        dt = jnp.ones((batch, seq_len, intermediate_size)) * 0.1
        gate = jnp.ones((batch, seq_len, intermediate_size)) * 2.0

        output_no_gate, _, _ = state_space_v1(hidden_states, A, B, C, D, dt)
        output_with_gate, _, _ = state_space_v1(hidden_states, A, B, C, D, dt, gate=gate, act_fn=jax.nn.silu)

        assert output_with_gate.shape == output_no_gate.shape
        assert not jnp.allclose(output_with_gate, output_no_gate)

    def test_single_step_inference(self):
        """Test single step inference mode."""
        batch, intermediate_size, ssm_state_size = 2, 64, 8

        hidden_states = jnp.ones((batch, 1, intermediate_size))
        A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.1
        B = jnp.ones((batch, 1, ssm_state_size)) * 0.1
        C = jnp.ones((batch, 1, ssm_state_size)) * 0.1
        D = jnp.ones((intermediate_size,))
        dt = jnp.ones((batch, 1, intermediate_size)) * 0.1
        initial_state = jnp.ones((batch, intermediate_size, ssm_state_size)) * 0.5

        output, ssm_state, _ = state_space_v1(hidden_states, A, B, C, D, dt, initial_state=initial_state)

        assert output.shape == (batch, 1, intermediate_size)
        assert ssm_state.shape == (batch, intermediate_size, ssm_state_size)

    def test_conv_state_passthrough(self):
        """Test that conv_state is passed through unchanged."""
        batch, seq_len, intermediate_size, ssm_state_size, d_conv = 2, 8, 64, 8, 4

        hidden_states = jnp.ones((batch, seq_len, intermediate_size))
        A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.1
        B = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
        D = jnp.ones((intermediate_size,))
        dt = jnp.ones((batch, seq_len, intermediate_size)) * 0.1
        conv_state_in = jnp.ones((batch, intermediate_size, d_conv)) * 0.3

        _, _, conv_state_out = state_space_v1(hidden_states, A, B, C, D, dt, conv_state=conv_state_in)

        assert jnp.allclose(conv_state_out, conv_state_in)

    def test_state_propagation(self):
        """Test that state propagates correctly across timesteps."""
        batch, intermediate_size, ssm_state_size = 1, 32, 8

        hidden_states_1 = jnp.ones((batch, 5, intermediate_size))
        hidden_states_2 = jnp.ones((batch, 5, intermediate_size))
        A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.1
        B = jnp.ones((batch, 5, ssm_state_size)) * 0.1
        C = jnp.ones((batch, 5, ssm_state_size)) * 0.1
        D = jnp.ones((intermediate_size,))
        dt = jnp.ones((batch, 5, intermediate_size)) * 0.1

        # Process sequence in one go
        _output_full, state_full, _ = state_space_v1(
            jnp.concatenate([hidden_states_1, hidden_states_2], axis=1),
            A,
            jnp.concatenate([B, B], axis=1),
            jnp.concatenate([C, C], axis=1),
            D,
            jnp.concatenate([dt, dt], axis=1),
        )

        # Process in two parts
        _, state_1, _ = state_space_v1(hidden_states_1, A, B, C, D, dt)
        _output_2, state_2, _ = state_space_v1(hidden_states_2, A, B, C, D, dt, initial_state=state_1)

        # Final states should match
        assert jnp.allclose(state_full, state_2, atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with small values."""
        batch, seq_len, intermediate_size, ssm_state_size = 1, 10, 32, 8

        hidden_states = jnp.ones((batch, seq_len, intermediate_size)) * 1e-4
        A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.01
        B = jnp.ones((batch, seq_len, ssm_state_size)) * 1e-4
        C = jnp.ones((batch, seq_len, ssm_state_size)) * 1e-4
        D = jnp.ones((intermediate_size,)) * 1e-4
        dt = jnp.ones((batch, seq_len, intermediate_size)) * 0.01

        output, ssm_state, _ = state_space_v1(hidden_states, A, B, C, D, dt)

        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(ssm_state))

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        seq_len, intermediate_size, ssm_state_size = 8, 64, 8

        for batch in [1, 2, 4, 8]:
            hidden_states = jnp.ones((batch, seq_len, intermediate_size))
            A = -jnp.ones((intermediate_size, ssm_state_size)) * 0.1
            B = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
            C = jnp.ones((batch, seq_len, ssm_state_size)) * 0.1
            D = jnp.ones((intermediate_size,))
            dt = jnp.ones((batch, seq_len, intermediate_size)) * 0.1

            output, ssm_state, _ = state_space_v1(hidden_states, A, B, C, D, dt)

            assert output.shape == (batch, seq_len, intermediate_size)
            assert ssm_state.shape == (batch, intermediate_size, ssm_state_size)

    def test_bf16_state_with_fp32_A_preserves_state_dtype(self):
        """Carry/state dtype should stay stable even when A is float32."""
        batch, seq_len, intermediate_size, ssm_state_size = 2, 4, 32, 16

        hidden_states = jnp.ones((batch, seq_len, intermediate_size), dtype=jnp.bfloat16)
        A = (-jnp.ones((intermediate_size, ssm_state_size)) * 0.1).astype(jnp.float32)
        B = (jnp.ones((batch, seq_len, ssm_state_size)) * 0.1).astype(jnp.bfloat16)
        C = (jnp.ones((batch, seq_len, ssm_state_size)) * 0.1).astype(jnp.bfloat16)
        D = jnp.ones((intermediate_size,), dtype=jnp.bfloat16)
        dt = (jnp.ones((batch, seq_len, intermediate_size)) * 0.1).astype(jnp.bfloat16)
        initial_state = jnp.zeros((batch, intermediate_size, ssm_state_size), dtype=jnp.bfloat16)

        output, ssm_state, _ = state_space_v1(
            hidden_states,
            A,
            B,
            C,
            D,
            dt,
            initial_state=initial_state,
        )

        assert output.dtype == jnp.bfloat16
        assert ssm_state.dtype == jnp.bfloat16

    def test_bf16_state_with_fp32_A_backward_scan_carry_dtype(self):
        """Backward scan should keep carry dtype stable for mixed-precision inputs."""
        batch, seq_len, intermediate_size, ssm_state_size = 1, 3, 16, 8

        hidden_states = jnp.ones((batch, seq_len, intermediate_size), dtype=jnp.bfloat16)
        A = (-jnp.ones((intermediate_size, ssm_state_size)) * 0.1).astype(jnp.float32)
        B = (jnp.ones((batch, seq_len, ssm_state_size)) * 0.1).astype(jnp.bfloat16)
        C = (jnp.ones((batch, seq_len, ssm_state_size)) * 0.1).astype(jnp.bfloat16)
        D = jnp.ones((intermediate_size,), dtype=jnp.bfloat16)
        dt = (jnp.ones((batch, seq_len, intermediate_size)) * 0.1).astype(jnp.bfloat16)
        initial_state = jnp.zeros((batch, intermediate_size, ssm_state_size), dtype=jnp.bfloat16)

        def loss_fn(h):
            out, _state, _ = state_space_v1(h, A, B, C, D, dt, initial_state=initial_state)
            return jnp.sum(out.astype(jnp.float32))

        grad_h = jax.grad(loss_fn)(hidden_states)
        assert grad_h.shape == hidden_states.shape
        assert grad_h.dtype == jnp.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
