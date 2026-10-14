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


"""XLA implementation of SSM2 (Mamba2-style) selective state space.

This module provides a pure JAX/XLA implementation of the SSM2 algorithm
used in Mamba2 and FalconH1 models.

Key characteristics of SSM2:
- 1D A vector: [num_heads] (per-head scalar)
- SSM state shape: [batch, num_heads, head_dim, ssm_state_size]
- B, C with n_groups grouping

The algorithm:
    Discretization:
        dA = exp(A * dt)  where A is per-head
        dB = dt * B

    Recurrence (per head):
        dBx = dt * B * x (outer product form)
        h_t = dA * h_{t-1} + dBx
        y_t = einsum(h_t, C_t) + x_t * D
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _ssm2_bwd
from ._xla_impl_fwd import _ssm2_fwd, _ssm2_single_step_fwd


@partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def _ssm2_core(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch seq_len num_heads ssm_state_size"],
    C: Float[Array, "batch seq_len num_heads ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch seq_len num_heads"],
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
    n_groups: int = 1,
    use_single_step: bool = False,
) -> tuple[
    Float[Array, "batch seq_len num_heads head_dim"],
    Float[Array, "batch num_heads head_dim ssm_state_size"],
]:
    """Core SSM2 computation entry-point with a custom VJP registered.

    Dispatches to either the full-sequence scan (``_ssm2_fwd``) or the
    single-step path (``_ssm2_single_step_fwd``) depending on
    ``use_single_step``.  ``n_groups`` and ``use_single_step`` are static
    (nondiff_argnums) so they do not receive gradients.

    Note: ``B`` and ``C`` passed to this function are *already expanded* from
    [batch, seq_len, n_groups, ssm_state_size] to
    [batch, seq_len, num_heads, ssm_state_size] by the caller
    (``state_space_v2``).

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim].
        A: Per-head A scalar (real form, typically negative) [num_heads].
        B: Expanded B projection [batch, seq_len, num_heads, ssm_state_size].
        C: Expanded C projection [batch, seq_len, num_heads, ssm_state_size].
        D: Per-head skip-connection weight [num_heads].
        dt: Time step after softplus [batch, seq_len, num_heads].
        initial_state: Optional initial SSM hidden state
            [batch, num_heads, head_dim, ssm_state_size].  If None, zeros are used.
        n_groups: Static number of groups for B, C (nondiff).  Not used
            inside this function; retained for signature symmetry.
        use_single_step: Static flag (nondiff).  If True and seq_len=1 and
            initial_state is not None, uses the single-step path.

    Returns:
        Tuple of:
            - output: SSM output [batch, seq_len, num_heads, head_dim].
            - final_state: Final hidden state
              [batch, num_heads, head_dim, ssm_state_size].
    """
    _batch_size, seq_len, _num_heads, _head_dim = x.shape

    if use_single_step and seq_len == 1 and initial_state is not None:
        y, final_state = _ssm2_single_step_fwd(
            x=x[:, 0, :, :],
            A=A,
            B=B[:, 0, :, :],
            C=C[:, 0, :, :],
            D=D,
            dt=dt[:, 0, :],
            ssm_state=initial_state,
        )
        return y[:, None, :, :], final_state
    else:
        output, _, final_state = _ssm2_fwd(
            x=x,
            A=A,
            B=B,
            C=C,
            D=D,
            dt=dt,
            initial_state=initial_state,
        )
        return output, final_state


def _ssm2_fwd_rule(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch seq_len num_heads ssm_state_size"],
    C: Float[Array, "batch seq_len num_heads ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch seq_len num_heads"],
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None,
    n_groups: int,
    use_single_step: bool,
) -> tuple[
    tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch num_heads head_dim ssm_state_size"]],
    tuple,
]:
    """Compute forward pass and save residuals for backward differentiation.

    Runs the SSM2 forward computation (full sequence or single step),
    then packages the inputs and intermediate hidden states as residuals
    for the backward pass.

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim].
        A: A vector (real form, typically negative) [num_heads].
        B: B parameter [batch, seq_len, num_heads, ssm_state_size].
        C: C parameter [batch, seq_len, num_heads, ssm_state_size].
        D: Skip connection [num_heads].
        dt: Time step (after softplus) [batch, seq_len, num_heads].
        initial_state: Initial hidden state or None.
        n_groups: Number of groups for B, C (nondiff).
        use_single_step: Whether to use single-step optimization (nondiff).

    Returns:
        Tuple of ((output, final_state), residuals) for VJP.
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    ssm_state_size = B.shape[-1]

    if initial_state is None:
        initial_state = jnp.zeros(
            (batch_size, num_heads, head_dim, ssm_state_size),
            dtype=jnp.float32,
        )

    if use_single_step and seq_len == 1:
        y, final_state = _ssm2_single_step_fwd(
            x=x[:, 0, :, :],
            A=A,
            B=B[:, 0, :, :],
            C=C[:, 0, :, :],
            D=D,
            dt=dt[:, 0, :],
            ssm_state=initial_state,
        )
        all_hidden_states = final_state[:, None, :, :, :]
        output = y[:, None, :, :]
    else:
        output, all_hidden_states, final_state = _ssm2_fwd(
            x=x,
            A=A,
            B=B,
            C=C,
            D=D,
            dt=dt,
            initial_state=initial_state,
        )

    residuals = (x, A, B, C, D, dt, all_hidden_states, initial_state)
    return (output, final_state), residuals


def _ssm2_bwd_rule(
    n_groups: int,
    use_single_step: bool,
    residuals: tuple,
    grads: tuple,
) -> tuple:
    """Compute backward pass using stored residuals from the forward rule.

    Unpacks the residuals saved during the forward pass and delegates
    to the SSM2 backward implementation for gradient computation.

    Args:
        n_groups: Number of groups for B, C (nondiff).
        use_single_step: Whether single-step mode was used (nondiff).
        residuals: Saved tensors from forward: (x, A, B, C, D, dt,
            all_hidden_states, initial_state).
        grads: Gradient of the loss w.r.t. (output, final_state).

    Returns:
        Tuple of gradients (dx, dA, dB, dC, dD, ddt, d_initial_state).
    """
    x, A, B, C, D, dt, all_hidden_states, initial_state = residuals
    do, dfinal_state = grads

    dx, dA, dB, dC, dD, ddt, d_initial_state = _ssm2_bwd(
        x=x,
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


_ssm2_core.defvjp(_ssm2_fwd_rule, _ssm2_bwd_rule)


@kernel_registry.register("state_space_v2", Platform.XLA, Backend.ANY)
@kernel_registry.register("ssm2", Platform.XLA, Backend.ANY)
@kernel_registry.register("mamba2", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def state_space_v2(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch seq_len n_groups ssm_state_size"],
    C: Float[Array, "batch seq_len n_groups ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch seq_len num_heads"],
    gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
    conv_state: Float[Array, "batch conv_dim d_conv"] | None = None,
    n_groups: int = 1,
    act_fn: Callable[[Array], Array] | None = None,
    use_gated_rmsnorm: bool = False,
    rmsnorm_eps: float = 1e-5,
    precision: lax.Precision | None = None,
    *,
    block_e: int = 128,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch num_heads head_dim ssm_state_size"],
    Float[Array, "batch conv_dim d_conv"] | None,
]:
    """SSM2 (Mamba2-style) selective state space using XLA backend.

    Implements the Mamba2 architecture with O(N) complexity.
    Processes tokens sequentially with per-head scalar decay.

    The core algorithm:
        Discretization:
            dA = exp(A * dt)  where A is per-head scalar
            dB = dt * B

        Recurrence (per head):
            dBx = dt * B * x (outer product form)
            h_t = dA * h_{t-1} + dBx
            y_t = einsum(h_t, C_t) + x_t * D

        Output gating (if gate provided):
            If use_gated_rmsnorm:
                y = rmsnorm(y) * act_fn(gate)
            Else:
                y = y * act_fn(gate)

    Args:
        x: Input tensor
            Shape: [batch, seq_len, num_heads, head_dim]
        A: A vector in real form (typically negative for stability)
            Shape: [num_heads]
        B: B parameter (with n_groups grouping)
            Shape: [batch, seq_len, n_groups, ssm_state_size]
        C: C parameter (with n_groups grouping)
            Shape: [batch, seq_len, n_groups, ssm_state_size]
        D: Skip connection parameter
            Shape: [num_heads]
        dt: Time step after softplus activation
            Shape: [batch, seq_len, num_heads]
        gate: Optional gating tensor for output modulation
            Shape: [batch, seq_len, intermediate_size]
        initial_state: Optional initial SSM state for continuation
            Shape: [batch, num_heads, head_dim, ssm_state_size]
        conv_state: Optional convolution state for caching (passed through)
            Shape: [batch, conv_dim, d_conv]
        n_groups: Number of groups for B, C (B/C are repeated to num_heads)
        act_fn: Optional activation function for gating (e.g., jax.nn.silu).
            If gate is provided but act_fn is None, defaults to jax.nn.silu.
        use_gated_rmsnorm: If True, apply RMSNorm before gating
        rmsnorm_eps: Epsilon for RMSNorm stability
        precision: JAX precision for matmul operations

    Returns:
        Tuple of:
            - output: SSM output [batch, seq_len, intermediate_size]
            - ssm_state: Final hidden state [batch, num_heads, head_dim, ssm_state_size]
            - conv_state: Passed through conv_state (for caching)

    Examples:
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>>
        >>> # Basic usage
        >>> batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 2, 64, 8, 64, 1, 16
        >>> x = random.normal(random.PRNGKey(0), (batch, seq_len, num_heads, head_dim))
        >>> A = -random.uniform(random.PRNGKey(1), (num_heads,))  # negative for stability
        >>> B = random.normal(random.PRNGKey(2), (batch, seq_len, n_groups, ssm_state_size))
        >>> C = random.normal(random.PRNGKey(3), (batch, seq_len, n_groups, ssm_state_size))
        >>> D = random.normal(random.PRNGKey(4), (num_heads,))
        >>> dt = jax.nn.softplus(random.normal(random.PRNGKey(5), (batch, seq_len, num_heads)))
        >>>
        >>> output, ssm_state, conv_state = state_space_v2(x, A, B, C, D, dt, n_groups=n_groups)
        >>> output.shape
        (2, 64, 512)  # num_heads * head_dim
        >>> ssm_state.shape
        (2, 8, 64, 16)
        >>>
        >>> # With gated RMSNorm
        >>> gate = random.normal(random.PRNGKey(6), (batch, seq_len, num_heads * head_dim))
        >>> output, ssm_state, _ = state_space_v2(
        ...     x, A, B, C, D, dt,
        ...     gate=gate, n_groups=n_groups,
        ...     use_gated_rmsnorm=True, act_fn=jax.nn.silu,
        ... )
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    dtype = x.dtype
    intermediate_size = num_heads * head_dim

    group_rep = num_heads // n_groups
    B_expanded = jnp.repeat(B, repeats=group_rep, axis=2)
    C_expanded = jnp.repeat(C, repeats=group_rep, axis=2)

    use_single_step = seq_len == 1 and initial_state is not None

    output, ssm_state = _ssm2_core(
        x=x,
        A=A,
        B=B_expanded,
        C=C_expanded,
        D=D,
        dt=dt,
        initial_state=initial_state,
        n_groups=n_groups,
        use_single_step=use_single_step,
    )

    output = output.reshape(batch_size, seq_len, intermediate_size)

    if gate is not None:
        if act_fn is None:
            act_fn = jax.nn.silu

        if use_gated_rmsnorm:
            output_f32 = output.astype(jnp.float32)
            variance = jnp.mean(jnp.square(output_f32), axis=-1, keepdims=True)
            output_norm = output_f32 * lax.rsqrt(variance + rmsnorm_eps)
            output = (output_norm * act_fn(gate.astype(jnp.float32))).astype(dtype)
        else:
            output = output * act_fn(gate)

    del block_e
    return output.astype(dtype), ssm_state.astype(dtype), conv_state
