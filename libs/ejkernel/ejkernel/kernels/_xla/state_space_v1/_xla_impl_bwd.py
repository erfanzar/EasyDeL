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


"""Backward pass implementation for SSM1 (Mamba1-style) selective state space.

Gradients for the SSM1 recurrence:
    Forward:
        dA = exp(A * dt)
        dBx = dt * B * x
        h_t = dA * h_{t-1} + dBx_t
        y_t = sum(h_t * C_t) + D * x_t

    Backward (computed in reverse order):
        dL/dh_t = dL/dy_t * C_t + dA_{t+1} * dL/dh_{t+1}
        dL/dx_t = dL/dy_t * D + sum(dL/dh_t * dt * B)
        dL/dA = sum_t(dL/dh_t * h_{t-1} * A * dt)
        dL/dB_t = sum_d(dL/dh_t * dt * x)
        dL/dC_t = sum_d(dL/dy_t * h_t)
        dL/dD = sum_t(dL/dy_t * x_t)
        dL/ddt_t = sum(dL/dh_t * (A * dA * h_{t-1} + B * x))
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _ssm1_bwd(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    all_hidden_states: Float[Array, "batch seq_len intermediate_size ssm_state_size"],
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"],
    do: Float[Array, "batch seq_len intermediate_size"],
    dfinal_state: Float[Array, "batch intermediate_size ssm_state_size"],
) -> tuple:
    """Backward pass for SSM1 recurrence.

    Args:
        hidden_states: Input tensor [batch, seq_len, intermediate_size]
        A: A matrix [intermediate_size, ssm_state_size]
        B: B parameter [batch, seq_len, ssm_state_size]
        C: C parameter [batch, seq_len, ssm_state_size]
        D: Skip connection [intermediate_size]
        dt: Time step [batch, seq_len, intermediate_size]
        all_hidden_states: Hidden states from forward [batch, seq_len, d, n]
        initial_state: Initial hidden state [batch, intermediate_size, ssm_state_size]
        do: Gradient of output [batch, seq_len, intermediate_size]
        dfinal_state: Gradient of final hidden state [batch, d, n]

    Returns:
        Tuple of (d_hidden_states, dA, dB, dC, dD, ddt, d_initial_state)
    """
    _batch_size, seq_len, _intermediate_size = hidden_states.shape

    def process_batch(
        x_b,
        A_val,
        B_b,
        C_b,
        D_val,
        dt_b,
        h_all_b,
        h0_b,
        do_b,
        dh_final,
    ):
        """Process backward pass for a single batch element.

        Runs the reverse-time scan to compute gradients for all SSM1
        parameters. Uses lax.scan in reverse order and accumulates
        per-step contributions for dA and dD.

        Args:
            x_b: Input for this batch [seq_len, intermediate_size].
            A_val: A matrix (shared across batch) [intermediate_size, ssm_state_size].
            B_b: B parameter [seq_len, ssm_state_size].
            C_b: C parameter [seq_len, ssm_state_size].
            D_val: Skip connection (shared) [intermediate_size].
            dt_b: Time step [seq_len, intermediate_size].
            h_all_b: All hidden states from forward [seq_len, d, n].
            h0_b: Initial hidden state [intermediate_size, ssm_state_size].
            do_b: Output gradient [seq_len, intermediate_size].
            dh_final: Gradient of final hidden state [d, n].

        Returns:
            Tuple of (dx_b, dA_b, dB_b, dC_b, dD_b, ddt_b, d_initial_state).
        """

        h_prev = jnp.concatenate([h0_b[None, :, :], h_all_b[:-1]], axis=0)

        def backward_step(carry, inputs):
            """Compute one reverse-time backward step for SSM1.

            Computes gradients dh, dx, dB, dC, ddt, and dA contribution
            for a single timestep, then propagates dh to the previous step.

            Args:
                carry: Gradient w.r.t. hidden state from future steps [d, n].
                inputs: Tuple of (t_idx, x_t, B_t, C_t, dt_t, h_t, h_prev_t, do_t)
                    for the current (reversed) timestep.

            Returns:
                Tuple of (dh_prev, (dx_t, dB_t, dC_t, ddt_t, dA_contrib)).
            """
            dh_next = carry
            carry_dtype = dh_next.dtype
            _t_idx, x_t, B_t, C_t, dt_t, h_t, h_prev_t, do_t = inputs

            dA_t = jnp.exp(A_val * dt_t[:, None])

            dh_from_output = do_t[:, None] * C_t[None, :]

            dh_t = dh_next + dh_from_output

            dx_t = do_t * D_val + jnp.sum(dh_t * dt_t[:, None] * B_t[None, :], axis=-1)

            dB_t = jnp.sum(dh_t * dt_t[:, None] * x_t[:, None], axis=0)

            dC_t = jnp.sum(do_t[:, None] * h_t, axis=0)

            ddt_t = jnp.sum(dh_t * (A_val * dA_t * h_prev_t + B_t[None, :] * x_t[:, None]), axis=-1)

            dA_contrib = dh_t * dt_t[:, None] * dA_t * h_prev_t

            dh_prev = (dh_t * dA_t).astype(carry_dtype)

            outputs = (dx_t, dB_t, dC_t, ddt_t, dA_contrib)
            return dh_prev, outputs

        scan_inputs = (
            jnp.arange(seq_len)[::-1],
            x_b[::-1],
            B_b[::-1],
            C_b[::-1],
            dt_b[::-1],
            h_all_b[::-1],
            h_prev[::-1],
            do_b[::-1],
        )

        d_initial_state, outputs = jax.lax.scan(backward_step, dh_final, scan_inputs)

        dx_b, dB_b, dC_b, ddt_b, dA_contribs = outputs

        dx_b = dx_b[::-1]
        dB_b = dB_b[::-1]
        dC_b = dC_b[::-1]
        ddt_b = ddt_b[::-1]
        dA_contribs = dA_contribs[::-1]

        dA_b = jnp.sum(dA_contribs, axis=0)

        dD_b = jnp.sum(do_b * x_b, axis=0)

        return dx_b, dA_b, dB_b, dC_b, dD_b, ddt_b, d_initial_state

    dx, dA_batch, dB, dC, dD_batch, ddt, d_initial_state = jax.vmap(
        process_batch,
        in_axes=(0, None, 0, 0, None, 0, 0, 0, 0, 0),
    )(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        all_hidden_states,
        initial_state,
        do,
        dfinal_state,
    )

    dA = jnp.sum(dA_batch, axis=0)
    dD = jnp.sum(dD_batch, axis=0)

    return dx, dA, dB, dC, dD, ddt, d_initial_state
