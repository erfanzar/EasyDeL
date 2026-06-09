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


"""Backward pass implementation for SSM2 (Mamba2-style) selective state space.

Gradients for the SSM2 recurrence:
    Forward:
        dA = exp(A * dt)
        dBx = dt * B * x (outer product)
        h_t = dA * h_{t-1} + dBx
        y_t = einsum(h_t, C_t) + x_t * D

    Backward (computed in reverse order):
        dL/dh_t = einsum(dL/dy_t, C_t) + dA_{t+1} * dL/dh_{t+1}
        dL/dx_t = dL/dy_t * D + einsum(dL/dh_t, dt * B)
        dL/dA = sum_t(dL/dh_t * h_{t-1} * dt * dA)
        dL/dB_t = einsum(dL/dh_t, x_t * dt)
        dL/dC_t = einsum(dL/dy_t, h_t)
        dL/dD = sum_t(dL/dy_t * x_t)
        dL/ddt_t = sum(dL/dh_t * (A * dA * h_{t-1} + B * x))
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _ssm2_bwd(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch seq_len num_heads ssm_state_size"],
    C: Float[Array, "batch seq_len num_heads ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch seq_len num_heads"],
    all_hidden_states: Float[Array, "batch seq_len num_heads head_dim ssm_state_size"],
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"],
    do: Float[Array, "batch seq_len num_heads head_dim"],
    dfinal_state: Float[Array, "batch num_heads head_dim ssm_state_size"],
) -> tuple:
    """Backward pass for SSM2 recurrence.

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim]
        A: A vector [num_heads]
        B: B parameter [batch, seq_len, num_heads, ssm_state_size]
        C: C parameter [batch, seq_len, num_heads, ssm_state_size]
        D: Skip connection [num_heads]
        dt: Time step [batch, seq_len, num_heads]
        all_hidden_states: Hidden states from forward [batch, seq_len, h, d, n]
        initial_state: Initial hidden state [batch, num_heads, head_dim, ssm_state_size]
        do: Gradient of output [batch, seq_len, num_heads, head_dim]
        dfinal_state: Gradient of final hidden state [batch, h, d, n]

    Returns:
        Tuple of (dx, dA, dB, dC, dD, ddt, d_initial_state)
    """
    _batch_size, seq_len, _num_heads, _head_dim = x.shape

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

        Runs the reverse-time scan to compute gradients for all SSM2
        parameters. Uses lax.scan in reverse order and accumulates
        per-step contributions for dA and dD.

        Args:
            x_b: Input for this batch [seq_len, num_heads, head_dim].
            A_val: A vector (shared across batch) [num_heads].
            B_b: B parameter [seq_len, num_heads, ssm_state_size].
            C_b: C parameter [seq_len, num_heads, ssm_state_size].
            D_val: Skip connection (shared) [num_heads].
            dt_b: Time step [seq_len, num_heads].
            h_all_b: All hidden states from forward [seq_len, h, d, n].
            h0_b: Initial hidden state [num_heads, head_dim, ssm_state_size].
            do_b: Output gradient [seq_len, num_heads, head_dim].
            dh_final: Gradient of final hidden state [h, d, n].

        Returns:
            Tuple of (dx_b, dA_b, dB_b, dC_b, dD_b, ddt_b, d_initial_state).
        """

        h_prev = jnp.concatenate([h0_b[None, :, :, :], h_all_b[:-1]], axis=0)

        def backward_step(carry, inputs):
            """Compute one reverse-time backward step for SSM2.

            Computes gradients dh, dx, dB, dC, ddt, and dA contribution
            for a single timestep using the SSM2 recurrence equations,
            then propagates dh to the previous step.

            Args:
                carry: Gradient w.r.t. hidden state from future steps [h, d, n].
                inputs: Tuple of (t_idx, x_t, B_t, C_t, dt_t, h_t, h_prev_t, do_t)
                    for the current (reversed) timestep.

            Returns:
                Tuple of (dh_prev, (dx_t, dB_t, dC_t, ddt_t, dA_contrib)).
            """
            dh_next = carry
            _t_idx, x_t, B_t, C_t, dt_t, h_t, h_prev_t, do_t = inputs

            dA_t = jnp.exp(dt_t[:, None, None] * A_val[:, None, None])

            dh_from_output = do_t[:, :, None] * C_t[:, None, :]

            dh_t = dh_next + dh_from_output

            dx_t = do_t * D_val[:, None] + jnp.einsum("hdn,h,hn->hd", dh_t, dt_t, B_t)

            dB_t = jnp.einsum("hdn,h,hd->hn", dh_t, dt_t, x_t)

            dC_t = jnp.einsum("hd,hdn->hn", do_t, h_t)

            ddt_t = jnp.sum(
                dh_t * (A_val[:, None, None] * dA_t * h_prev_t + B_t[:, None, :] * x_t[:, :, None]),
                axis=(-2, -1),
            )

            dA_contrib = jnp.sum(dh_t * dt_t[:, None, None] * dA_t * h_prev_t, axis=(-2, -1))

            dh_prev = dh_t * dA_t

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

        dD_b = jnp.sum(jnp.sum(do_b * x_b, axis=0), axis=-1)

        return dx_b, dA_b, dB_b, dC_b, dD_b, ddt_b, d_initial_state

    dx, dA_batch, dB, dC, dD_batch, ddt, d_initial_state = jax.vmap(
        process_batch,
        in_axes=(0, None, 0, 0, None, 0, 0, 0, 0, 0),
    )(
        x,
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
