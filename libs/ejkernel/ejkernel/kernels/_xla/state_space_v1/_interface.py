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


"""XLA implementation of SSM1 (Mamba1-style) selective state space.

This module provides a pure JAX/XLA implementation of the SSM1 algorithm
used in Mamba and FalconMamba models.

Key characteristics of SSM1:
- 2D A matrix: [intermediate_size, ssm_state_size]
- SSM state shape: [batch, intermediate_size, ssm_state_size]
- Separate dt_proj projection for time step

The algorithm:
    Discretization:
        dA = exp(A * dt)
        dB = dt * B

    Recurrence:
        h_t = dA * h_{t-1} + dB * x_t
        y_t = h_t @ C_t + D * x_t
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _ssm1_bwd
from ._xla_impl_fwd import _ssm1_fwd, _ssm1_single_step_fwd


@partial(jax.custom_vjp, nondiff_argnums=(7,))
def _ssm1_core(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
    use_single_step: bool = False,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch intermediate_size ssm_state_size"],
]:
    """Core SSM1 computation entry-point with a custom VJP registered.

    Dispatches to either the full-sequence scan (``_ssm1_fwd``) or the
    optimized single-step path (``_ssm1_single_step_fwd``) depending on
    ``use_single_step``.  Gradients flow through ``_ssm1_fwd_rule`` and
    ``_ssm1_bwd_rule``.

    Args:
        hidden_states: Input tensor [batch, seq_len, intermediate_size].
        A: A matrix in real form, typically negative for stability
            [intermediate_size, ssm_state_size].
        B: B projection [batch, seq_len, ssm_state_size].
        C: C projection [batch, seq_len, ssm_state_size].
        D: Skip-connection weight [intermediate_size].
        dt: Time step after softplus [batch, seq_len, intermediate_size].
        initial_state: Optional initial SSM hidden state
            [batch, intermediate_size, ssm_state_size].  If None, zeros are used.
        use_single_step: Static flag (nondiff_argnums).  If True and seq_len=1
            and initial_state is not None, uses the single-step inference path.

    Returns:
        Tuple of:
            - output: SSM output [batch, seq_len, intermediate_size].
            - final_state: Final hidden state [batch, intermediate_size, ssm_state_size].
    """
    _batch_size, seq_len, _intermediate_size = hidden_states.shape
    _ssm_state_size = B.shape[-1]

    if use_single_step and seq_len == 1 and initial_state is not None:
        y, final_state = _ssm1_single_step_fwd(
            hidden_state=hidden_states[:, 0, :],
            A=A,
            B=B[:, 0, :],
            C=C[:, 0, :],
            D=D,
            dt=dt[:, 0, :],
            ssm_state=initial_state,
        )
        return y[:, None, :], final_state
    else:
        output, _, final_state = _ssm1_fwd(
            hidden_states=hidden_states,
            A=A,
            B=B,
            C=C,
            D=D,
            dt=dt,
            initial_state=initial_state,
        )
        return output, final_state


def _ssm1_fwd_rule(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None,
    use_single_step: bool,
) -> tuple[
    tuple[Float[Array, "batch seq_len intermediate_size"], Float[Array, "batch intermediate_size ssm_state_size"]],
    tuple,
]:
    """Forward rule for SSM1 custom VJP, returning residuals for backward.

    Runs the SSM1 forward computation and saves all intermediate values
    needed by the backward pass (hidden states, inputs, discretized tensors).

    Args:
        hidden_states: Input tensor [batch, seq_len, intermediate_size].
        A: A matrix (real form, typically negative) [intermediate_size, ssm_state_size].
        B: B parameter [batch, seq_len, ssm_state_size].
        C: C parameter [batch, seq_len, ssm_state_size].
        D: Skip connection [intermediate_size].
        dt: Time step (after softplus) [batch, seq_len, intermediate_size].
        initial_state: Initial hidden state [batch, intermediate_size, ssm_state_size],
            or None to use zeros.
        use_single_step: If True and seq_len=1, use optimized single-step path.

    Returns:
        Tuple of:
            - primals: Tuple of (output, final_state).
            - residuals: Tuple of tensors saved for the backward pass.
    """
    batch_size, seq_len, intermediate_size = hidden_states.shape
    ssm_state_size = B.shape[-1]

    if initial_state is None:
        initial_state = jnp.zeros(
            (batch_size, intermediate_size, ssm_state_size),
            dtype=hidden_states.dtype,
        )

    if use_single_step and seq_len == 1:
        y, final_state = _ssm1_single_step_fwd(
            hidden_state=hidden_states[:, 0, :],
            A=A,
            B=B[:, 0, :],
            C=C[:, 0, :],
            D=D,
            dt=dt[:, 0, :],
            ssm_state=initial_state,
        )
        all_hidden_states = final_state[:, None, :, :]
        output = y[:, None, :]
    else:
        output, all_hidden_states, final_state = _ssm1_fwd(
            hidden_states=hidden_states,
            A=A,
            B=B,
            C=C,
            D=D,
            dt=dt,
            initial_state=initial_state,
        )

    residuals = (hidden_states, A, B, C, D, dt, all_hidden_states, initial_state)
    return (output, final_state), residuals


def _ssm1_bwd_rule(
    use_single_step: bool,
    residuals: tuple,
    grads: tuple,
) -> tuple:
    """Backward rule for SSM1 custom VJP.

    Computes gradients for all SSM1 parameters using saved residuals
    from the forward pass. Delegates to the specialized backward
    implementation in ``_ssm1_bwd``.

    Args:
        use_single_step: Non-differentiable flag from forward (True if
            single-step inference was used).
        residuals: Tuple of tensors saved by ``_ssm1_fwd_rule``, containing
            (hidden_states, A, B, C, D, dt, all_hidden_states, initial_state).
        grads: Tuple of output gradients (do, dfinal_state).

    Returns:
        Tuple of gradients (dx, dA, dB, dC, dD, ddt, d_initial_state)
        matching the primal inputs of ``_ssm1_core``.
    """
    hidden_states, A, B, C, D, dt, all_hidden_states, initial_state = residuals
    do, dfinal_state = grads

    dx, dA, dB, dC, dD, ddt, d_initial_state = _ssm1_bwd(
        hidden_states=hidden_states,
        A=A,
        B=B,
        C=C,
        D=D,
        dt=dt,
        all_hidden_states=all_hidden_states,
        initial_state=initial_state,
        do=do,
        dfinal_state=dfinal_state,
    )

    return (dx, dA, dB, dC, dD, ddt, d_initial_state)


_ssm1_core.defvjp(_ssm1_fwd_rule, _ssm1_bwd_rule)


@kernel_registry.register("state_space_v1", Platform.XLA, Backend.ANY)
@kernel_registry.register("ssm1", Platform.XLA, Backend.ANY)
@kernel_registry.register("mamba1", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def state_space_v1(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
    conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None,
    act_fn: Callable[[Array], Array] | None = None,
    *,
    block_d: int = 64,
    block_e: int = 128,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch intermediate_size ssm_state_size"],
    Float[Array, "batch intermediate_size d_conv"] | None,
]:
    """SSM1 (Mamba1-style) selective state space using XLA backend.

    Implements the original Mamba architecture with O(N) complexity.
    Processes tokens sequentially, maintaining a hidden state that
    accumulates information through discretized state transitions.

    The core algorithm:
        Discretization:
            dA = exp(A * dt)
            dB = dt * B

        Recurrence:
            h_t = dA * h_{t-1} + dB * x_t
            y_t = sum(h_t * C_t) + D * x_t

        Output gating (if gate provided):
            y = y * act_fn(gate)

    Args:
        hidden_states: Input tensor after convolution and activation
            Shape: [batch, seq_len, intermediate_size]
        A: A matrix in real form (typically negative for stability)
            Shape: [intermediate_size, ssm_state_size]
        B: B parameter from input projection
            Shape: [batch, seq_len, ssm_state_size]
        C: C parameter from input projection
            Shape: [batch, seq_len, ssm_state_size]
        D: Skip connection parameter
            Shape: [intermediate_size]
        dt: Time step after softplus activation
            Shape: [batch, seq_len, intermediate_size]
        gate: Optional gating tensor for output modulation
            Shape: [batch, seq_len, intermediate_size]
        initial_state: Optional initial SSM state for continuation
            Shape: [batch, intermediate_size, ssm_state_size]
        conv_state: Optional convolution state for caching (passed through)
            Shape: [batch, intermediate_size, d_conv]
        act_fn: Optional activation function for gating (e.g., jax.nn.silu).
            If gate is provided but act_fn is None, defaults to jax.nn.silu.

    Returns:
        Tuple of:
            - output: SSM output [batch, seq_len, intermediate_size]
            - ssm_state: Final hidden state [batch, intermediate_size, ssm_state_size]
            - conv_state: Passed through conv_state (for caching)

    Examples:
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>>
        >>> # Basic usage
        >>> batch, seq_len, d, n = 2, 64, 512, 16
        >>> hidden_states = random.normal(random.PRNGKey(0), (batch, seq_len, d))
        >>> A = -random.uniform(random.PRNGKey(1), (d, n))  # negative for stability
        >>> B = random.normal(random.PRNGKey(2), (batch, seq_len, n))
        >>> C = random.normal(random.PRNGKey(3), (batch, seq_len, n))
        >>> D = random.normal(random.PRNGKey(4), (d,))
        >>> dt = jax.nn.softplus(random.normal(random.PRNGKey(5), (batch, seq_len, d)))
        >>>
        >>> output, ssm_state, conv_state = state_space_v1(hidden_states, A, B, C, D, dt)
        >>> output.shape
        (2, 64, 512)
        >>> ssm_state.shape
        (2, 512, 16)
        >>>
        >>> # With gating
        >>> gate = random.normal(random.PRNGKey(6), (batch, seq_len, d))
        >>> output, ssm_state, _ = state_space_v1(
        ...     hidden_states, A, B, C, D, dt,
        ...     gate=gate, act_fn=jax.nn.silu,
        ... )
    """
    _, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype

    use_single_step = seq_len == 1 and initial_state is not None

    output, ssm_state = _ssm1_core(
        hidden_states=hidden_states,
        A=A,
        B=B,
        C=C,
        D=D,
        dt=dt,
        initial_state=initial_state,
        use_single_step=use_single_step,
    )

    if gate is not None:
        if act_fn is None:
            act_fn = jax.nn.silu
        output = output * act_fn(gate)
    del block_e, block_d
    return output.astype(dtype), ssm_state.astype(dtype), conv_state
