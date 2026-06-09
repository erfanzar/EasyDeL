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


"""Tests for XLA SSM2 (Mamba2-style) state space implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import state_space_v2


class TestStateSpaceV2:
    """Test suite for SSM2 (Mamba2-style) kernel."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 2, 10, 8, 64, 1, 16
        intermediate_size = num_heads * head_dim

        x = jnp.ones((batch, seq_len, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.1

        output, ssm_state, conv_state = state_space_v2(x, A, B, C, D, dt, n_groups=n_groups)

        assert output.shape == (batch, seq_len, intermediate_size)
        assert ssm_state.shape == (batch, num_heads, head_dim, ssm_state_size)
        assert conv_state is None

    def test_gradient_shapes(self):
        """Test gradient shapes match input shapes."""
        batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 1, 5, 4, 16, 1, 8

        x = jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.5
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.1

        def loss_fn(x, A, B, C, D, dt):
            o, _, _ = state_space_v2(x, A, B, C, D, dt, n_groups=n_groups)
            return jnp.sum(o)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(x, A, B, C, D, dt)

        assert grads[0].shape == x.shape
        assert grads[1].shape == A.shape
        assert grads[2].shape == B.shape
        assert grads[3].shape == C.shape
        assert grads[4].shape == D.shape
        assert grads[5].shape == dt.shape

    def test_with_initial_state(self):
        """Test with non-zero initial state."""
        batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 2, 5, 4, 16, 1, 8

        x = jnp.ones((batch, seq_len, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.1

        initial_state = jnp.ones((batch, num_heads, head_dim, ssm_state_size)) * 0.5

        output, ssm_state, _ = state_space_v2(x, A, B, C, D, dt, initial_state=initial_state, n_groups=n_groups)

        assert output.shape == (batch, seq_len, num_heads * head_dim)
        assert ssm_state.shape == (batch, num_heads, head_dim, ssm_state_size)
        assert not jnp.allclose(ssm_state, initial_state)

    def test_with_gating(self):
        """Test with gating applied."""
        batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 2, 8, 4, 16, 1, 8
        intermediate_size = num_heads * head_dim

        x = jnp.ones((batch, seq_len, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.1
        gate = jnp.ones((batch, seq_len, intermediate_size)) * 2.0

        output_no_gate, _, _ = state_space_v2(x, A, B, C, D, dt, n_groups=n_groups)
        output_with_gate, _, _ = state_space_v2(x, A, B, C, D, dt, gate=gate, n_groups=n_groups, act_fn=jax.nn.silu)

        assert output_with_gate.shape == output_no_gate.shape
        assert not jnp.allclose(output_with_gate, output_no_gate)

    def test_with_gated_rmsnorm(self):
        """Test with gated RMSNorm."""
        batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 2, 8, 4, 16, 1, 8
        intermediate_size = num_heads * head_dim

        x = jnp.ones((batch, seq_len, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.1
        gate = jnp.ones((batch, seq_len, intermediate_size)) * 2.0

        output_simple_gate, _, _ = state_space_v2(
            x, A, B, C, D, dt, gate=gate, n_groups=n_groups, use_gated_rmsnorm=False, act_fn=jax.nn.silu
        )
        output_rmsnorm_gate, _, _ = state_space_v2(
            x, A, B, C, D, dt, gate=gate, n_groups=n_groups, use_gated_rmsnorm=True, act_fn=jax.nn.silu
        )

        assert output_rmsnorm_gate.shape == output_simple_gate.shape
        # Gated RMSNorm normalizes the output before gating, so results differ
        assert not jnp.allclose(output_rmsnorm_gate, output_simple_gate)

    def test_n_groups(self):
        """Test with different n_groups values."""
        batch, seq_len, num_heads, head_dim, ssm_state_size = 2, 8, 8, 16, 8

        x = jnp.ones((batch, seq_len, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.1

        for n_groups in [1, 2, 4, 8]:
            B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
            C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1

            output, ssm_state, _ = state_space_v2(x, A, B, C, D, dt, n_groups=n_groups)

            assert output.shape == (batch, seq_len, num_heads * head_dim)
            assert ssm_state.shape == (batch, num_heads, head_dim, ssm_state_size)

    def test_single_step_inference(self):
        """Test single step inference mode."""
        batch, num_heads, head_dim, n_groups, ssm_state_size = 2, 4, 16, 1, 8

        x = jnp.ones((batch, 1, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, 1, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, 1, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, 1, num_heads)) * 0.1
        initial_state = jnp.ones((batch, num_heads, head_dim, ssm_state_size)) * 0.5

        output, ssm_state, _ = state_space_v2(x, A, B, C, D, dt, initial_state=initial_state, n_groups=n_groups)

        assert output.shape == (batch, 1, num_heads * head_dim)
        assert ssm_state.shape == (batch, num_heads, head_dim, ssm_state_size)

    def test_conv_state_passthrough(self):
        """Test that conv_state is passed through unchanged."""
        batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size, d_conv = 2, 8, 4, 16, 1, 8, 4
        conv_dim = num_heads * head_dim

        x = jnp.ones((batch, seq_len, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.1
        conv_state_in = jnp.ones((batch, conv_dim, d_conv)) * 0.3

        _, _, conv_state_out = state_space_v2(x, A, B, C, D, dt, conv_state=conv_state_in, n_groups=n_groups)

        assert jnp.allclose(conv_state_out, conv_state_in)

    def test_state_propagation(self):
        """Test that state propagates correctly across timesteps."""
        batch, num_heads, head_dim, n_groups, ssm_state_size = 1, 4, 16, 1, 8

        x_1 = jnp.ones((batch, 5, num_heads, head_dim))
        x_2 = jnp.ones((batch, 5, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, 5, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, 5, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, 5, num_heads)) * 0.1

        # Process sequence in one go
        _, state_full, _ = state_space_v2(
            jnp.concatenate([x_1, x_2], axis=1),
            A,
            jnp.concatenate([B, B], axis=1),
            jnp.concatenate([C, C], axis=1),
            D,
            jnp.concatenate([dt, dt], axis=1),
            n_groups=n_groups,
        )

        # Process in two parts
        _, state_1, _ = state_space_v2(x_1, A, B, C, D, dt, n_groups=n_groups)
        _, state_2, _ = state_space_v2(x_2, A, B, C, D, dt, initial_state=state_1, n_groups=n_groups)

        # Final states should match
        assert jnp.allclose(state_full, state_2, atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with small values."""
        batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 1, 10, 4, 16, 1, 8

        x = jnp.ones((batch, seq_len, num_heads, head_dim)) * 1e-4
        A = -jnp.ones((num_heads,)) * 0.01
        B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 1e-4
        C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 1e-4
        D = jnp.ones((num_heads,)) * 1e-4
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.01

        output, ssm_state, _ = state_space_v2(x, A, B, C, D, dt, n_groups=n_groups)

        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(ssm_state))

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        seq_len, num_heads, head_dim, n_groups, ssm_state_size = 8, 4, 16, 1, 8

        for batch in [1, 2, 4, 8]:
            x = jnp.ones((batch, seq_len, num_heads, head_dim))
            A = -jnp.ones((num_heads,)) * 0.1
            B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
            C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
            D = jnp.ones((num_heads,))
            dt = jnp.ones((batch, seq_len, num_heads)) * 0.1

            output, ssm_state, _ = state_space_v2(x, A, B, C, D, dt, n_groups=n_groups)

            assert output.shape == (batch, seq_len, num_heads * head_dim)
            assert ssm_state.shape == (batch, num_heads, head_dim, ssm_state_size)

    def test_custom_activation_function(self):
        """Test with custom activation function."""
        batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 2, 8, 4, 16, 1, 8
        intermediate_size = num_heads * head_dim

        x = jnp.ones((batch, seq_len, num_heads, head_dim))
        A = -jnp.ones((num_heads,)) * 0.1
        B = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        C = jnp.ones((batch, seq_len, n_groups, ssm_state_size)) * 0.1
        D = jnp.ones((num_heads,))
        dt = jnp.ones((batch, seq_len, num_heads)) * 0.1
        gate = jnp.ones((batch, seq_len, intermediate_size)) * 2.0

        output_silu, _, _ = state_space_v2(x, A, B, C, D, dt, gate=gate, n_groups=n_groups, act_fn=jax.nn.silu)
        output_gelu, _, _ = state_space_v2(x, A, B, C, D, dt, gate=gate, n_groups=n_groups, act_fn=jax.nn.gelu)
        output_relu, _, _ = state_space_v2(x, A, B, C, D, dt, gate=gate, n_groups=n_groups, act_fn=jax.nn.relu)

        # Different activations should produce different results
        assert not jnp.allclose(output_silu, output_gelu)
        assert not jnp.allclose(output_silu, output_relu)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
