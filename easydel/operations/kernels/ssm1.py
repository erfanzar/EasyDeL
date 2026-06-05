# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""
SSM1 (Mamba1-style) Selective State Space operation for EasyDeL.

This module provides the SSM1 operation, implementing the original Mamba
selective state space model architecture used by Mamba and FalconMamba.

This implementation delegates to ejKernel's optimized state_space_v1 kernel
for the core SSM computation, providing automatic platform selection and
optimization.

Key characteristics of SSM1:
- 2D A matrix: [intermediate_size, ssm_state_size]
- SSM state shape: [batch, intermediate_size, ssm_state_size]
- Separate dt_proj projection for time step
- Separate x_proj for B, C, dt parameters
- Output gating: y * activation(gate)

The algorithm:
    Discretization:
        dA = exp(A * dt)
        dB = dt * B

    Recurrence:
        h_t = dA * h_{t-1} + dB * x_t
        y_t = h_t @ C_t + D * x_t

References:
    - Mamba: https://arxiv.org/abs/2312.00752
    - FalconMamba: https://huggingface.co/tiiuae/falcon-mamba-7b
"""

import jax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from ejkernel.modules import state_space_v1  # pyright: ignore[reportMissingTypeStubs]
from jaxtyping import Array, Float

from easydel.caching import RecurrentCacheView
from easydel.infra.sequence_packing import normalize_packed_segment_ids

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)


def _single_step_ssm1_fwd(
    hidden_states: Float[Array, "batch intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch ssm_state_size"],
    C: Float[Array, "batch ssm_state_size"],
    D: Float[Array, "intermediate_size"],  # noqa:F821
    dt: Float[Array, "batch intermediate_size"],
    ssm_state: Float[Array, "batch intermediate_size ssm_state_size"],
) -> tuple[Float[Array, "batch intermediate_size"], Float[Array, "batch intermediate_size ssm_state_size"]]:
    """Single-step SSM1 (Mamba1) forward pass.

    Computes one discrete step of the selective state space model.
    Bypasses ejKernel's type checking for use in JIT-compiled packed
    decode paths (eSurge).

    Args:
        hidden_states: Input [batch, intermediate_size]
        A: Real-form A matrix [intermediate_size, ssm_state_size] (negative)
        B: B parameter [batch, ssm_state_size]
        C: C parameter [batch, ssm_state_size]
        D: Skip connection [intermediate_size]
        dt: Time step (after softplus) [batch, intermediate_size]
        ssm_state: Current SSM state [batch, intermediate_size, ssm_state_size]

    Returns:
        (y, ssm_state_new): Output and updated state.
    """
    # Discretize: dA = exp(dt * A)
    dA = jnp.exp(dt[:, :, None] * A[None, :, :])  # [B, I, N]

    # dB * x: (dt * B) * x
    dBx = dt[:, :, None] * B[:, None, :] * hidden_states[:, :, None]  # [B, I, N]

    # State update
    ssm_state_new = dA * ssm_state + dBx

    # Output: y = sum(state * C, axis=-1) + D * x
    y = jnp.sum(ssm_state_new * C[:, None, :], axis=-1) + D[None, :] * hidden_states

    return y, ssm_state_new


def _segmented_ssm1_fwd(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A_real: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],  # noqa:F821
    discrete_time_step: Float[Array, "batch seq_len intermediate_size"],
    segment_ids: Array,
    *,
    gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
    ssm_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
    act_fn=None,
) -> tuple[Float[Array, "batch seq_len intermediate_size"], Float[Array, "batch intermediate_size ssm_state_size"]]:
    """Reference SSM1 scan that resets recurrent state at packed boundaries."""
    batch_size, seq_len, intermediate_size = hidden_states.shape
    state_size = A_real.shape[-1]
    segment_ids = normalize_packed_segment_ids(segment_ids, seq_len, pad_from_last=False)
    if ssm_state is None:
        ssm_state = jnp.zeros((batch_size, intermediate_size, state_size), dtype=jnp.float32)
    else:
        ssm_state = ssm_state.astype(jnp.float32)
    previous_segment = jnp.full((batch_size,), -1, dtype=segment_ids.dtype)

    gate_seq = None if gate is None else gate.swapaxes(0, 1)

    def _step(carry, step_inputs):
        state, prev_segment = carry
        x_t, b_t, c_t, dt_t, segment_t = step_inputs[:5]
        gate_t = None if gate_seq is None else step_inputs[5]
        valid = segment_t >= 0
        new_segment = valid & (segment_t != prev_segment)
        state = jnp.where(new_segment[:, None, None], jnp.zeros_like(state), state)
        y_t, next_state = _single_step_ssm1_fwd(
            hidden_states=x_t,
            A=A_real,
            B=b_t,
            C=c_t,
            D=D,
            dt=dt_t,
            ssm_state=state,
        )
        if gate_t is not None and act_fn is not None:
            y_t = y_t * act_fn(gate_t)
        y_t = jnp.where(valid[:, None], y_t, jnp.zeros_like(y_t))
        next_state = jnp.where(valid[:, None, None], next_state, jnp.zeros_like(next_state))
        next_segment = jnp.where(valid, segment_t, -1)
        return (next_state, next_segment), y_t

    scan_inputs = (
        hidden_states.swapaxes(0, 1),
        B.swapaxes(0, 1),
        C.swapaxes(0, 1),
        discrete_time_step.swapaxes(0, 1),
        segment_ids.swapaxes(0, 1),
    )
    if gate_seq is not None:
        scan_inputs = (*scan_inputs, gate_seq)
    (final_state, _), outputs = jax.lax.scan(_step, (ssm_state, previous_segment), scan_inputs)
    return outputs.swapaxes(0, 1), final_state


@auto_pytree
class SSM1Output(AttentionOutput):
    """Output container for SSM1 operation.

    Attributes:
        attention_outputs: Output tensor [batch, seq_len, intermediate_size]
        attention_weights: Always None for SSM (no attention weights)
        conv_state: Updated convolution state [batch, intermediate_size, d_conv]
        ssm_state: Updated SSM state [batch, intermediate_size, ssm_state_size]
    """

    conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None
    ssm_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None


@OperationRegistry.register
class SSM1Op(OperationImpl):
    """SSM1 (Mamba1-style) selective state space operation.

    Implements the original Mamba architecture with:
    - 2D A matrix [intermediate_size, ssm_state_size]
    - Separate dt_proj and x_proj
    - SSM state shape [batch, intermediate_size, ssm_state_size]

    This operation is used by Mamba and FalconMamba models.

    Registered under the names "ssm1", "mamba1", "mamba".

    Example:
        >>> from easydel.operations import OperationMetadata, OperationRegistry
        >>> metadata = OperationMetadata(runtime_dtype=jnp.float16)
        >>> ssm_op = OperationRegistry.create("ssm1", metadata)
        >>> output = ssm_op(
        ...     hidden_states=x,
        ...     A=A_log,
        ...     B=B,
        ...     C=C,
        ...     D=D,
        ...     discrete_time_step=dt,
        ...     gate=gate,
        ... )
    """

    @classmethod
    def get_impl_name(cls) -> tuple[str, ...]:
        """Returns the registered names of this operation."""
        return ("ssm1", "mamba1", "mamba")

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for SSM1Op.

        SSM1 requires:
        - Basic metadata plus state management fields
        - Recurrent cache type for SSM state persistence
        """
        return (
            RequirementsBuilder("ssm1")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.POSITIONS
                | MetadataField.HAS_INITIAL_STATE
                | MetadataField.STATE_INDICES
            )
            .optional_metadata(MetadataField.LOGITS_INDICES)
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .use_cache_view(RecurrentCacheView)
            .build()
        )

    @jax.named_scope("easydel-ssm1-ejkernel")
    def forward_native(
        self,
        hidden_states: Float[Array, "batch seq_len intermediate_size"],
        A: Float[Array, "intermediate_size ssm_state_size"],
        B: Float[Array, "batch seq_len ssm_state_size"],
        C: Float[Array, "batch seq_len ssm_state_size"],
        D: Float[Array, "intermediate_size"],  # noqa:F821
        discrete_time_step: Float[Array, "batch seq_len intermediate_size"],
        gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
        conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None,
        ssm_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
        segment_ids: Array | None = None,
        activation: str = "silu",
        **kwargs,
    ) -> SSM1Output:
        """Forward pass for SSM1 operation using ejKernel.

        Delegates to ejkernel.modules.operations.state_space_v1 for the core
        SSM computation, which provides optimized implementations with automatic
        platform selection.

        Args:
            hidden_states: Input after conv and activation [batch, seq_len, d]
            A: A matrix (log form, will be exp(-exp(A))) [d, n]
            B: B parameter [batch, seq_len, n]
            C: C parameter [batch, seq_len, n]
            D: Skip connection parameter [d]
            discrete_time_step: Time step after softplus [batch, seq_len, d]
            gate: Optional gating tensor [batch, seq_len, d]
            conv_state: Optional conv state for caching
            ssm_state: Optional SSM state for caching
            activation: Activation function name for gating

        Returns:
            SSM1Output with outputs and updated states
        """
        from easydel.infra.utils import ACT2FN

        dtype = hidden_states.dtype

        # Convert A from log form to real form (negative for stability)
        A_real = -jnp.exp(A.astype(jnp.float32))

        # Get activation function
        act_fn = ACT2FN.get(activation, jax.nn.silu) if gate is not None else None

        if segment_ids is not None and hidden_states.shape[1] > 1:
            y, new_ssm_state = _segmented_ssm1_fwd(
                hidden_states=hidden_states,
                A_real=A_real,
                B=B,
                C=C,
                D=D,
                discrete_time_step=discrete_time_step,
                segment_ids=segment_ids,
                gate=gate,
                ssm_state=ssm_state,
                act_fn=act_fn,
            )
            return SSM1Output(
                attention_outputs=y.astype(dtype),
                attention_weights=None,
                conv_state=conv_state,
                ssm_state=new_ssm_state.astype(dtype),
            )

        # Call ejKernel's state_space_v1
        # It handles both training (full sequence) and inference (single step) modes
        y, new_ssm_state, new_conv_state = state_space_v1(
            hidden_states,
            A_real,
            B,
            C,
            D,
            discrete_time_step,
            gate=gate,
            initial_state=ssm_state,
            conv_state=conv_state,
            act_fn=act_fn,
        )

        return SSM1Output(
            attention_outputs=y.astype(dtype),
            attention_weights=None,
            conv_state=new_conv_state,
            ssm_state=new_ssm_state.astype(dtype),
        )

    def forward_tpu(self, *args, **kwargs) -> SSM1Output:
        """TPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> SSM1Output:
        """GPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> SSM1Output:
        """CPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len intermediate_size"],
        A: Float[Array, "intermediate_size ssm_state_size"],
        B: Float[Array, "batch seq_len ssm_state_size"],
        C: Float[Array, "batch seq_len ssm_state_size"],
        D: Float[Array, "intermediate_size"],  # noqa:F821
        discrete_time_step: Float[Array, "batch seq_len intermediate_size"],
        gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
        conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None,
        ssm_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
        segment_ids: Array | None = None,
        activation: str = "silu",
        **kwargs,
    ) -> SSM1Output:
        """Execute the SSM1 operation.

        Dispatches to appropriate backend via parent __call__.

        Args:
            hidden_states: Input tensor [batch, seq_len, intermediate_size]
            A: A matrix in log form [intermediate_size, ssm_state_size]
            B: B parameter [batch, seq_len, ssm_state_size]
            C: C parameter [batch, seq_len, ssm_state_size]
            D: Skip connection [intermediate_size]
            discrete_time_step: Time step [batch, seq_len, intermediate_size]
            gate: Optional gate tensor
            conv_state: Optional conv state
            ssm_state: Optional SSM state
            activation: Activation function name

        Returns:
            SSM1Output with outputs and states
        """
        return super().__call__(
            hidden_states=hidden_states,
            A=A,
            B=B,
            C=C,
            D=D,
            discrete_time_step=discrete_time_step,
            gate=gate,
            conv_state=conv_state,
            ssm_state=ssm_state,
            segment_ids=segment_ids,
            activation=activation,
            **kwargs,
        )


if __name__ == "__main__":
    from jax import random as jr

    from easydel.infra import EasyDeLBaseConfig

    print("Testing SSM1Op...")

    batch, seq_len, d, n = 2, 64, 512, 16

    key = jr.PRNGKey(0)
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)

    hidden_states = jr.normal(k1, (batch, seq_len, d), dtype=jnp.float32) * 0.1
    A = jr.normal(k2, (d, n), dtype=jnp.float32)
    B = jr.normal(k3, (batch, seq_len, n), dtype=jnp.float32) * 0.1
    C = jr.normal(k4, (batch, seq_len, n), dtype=jnp.float32) * 0.1
    D = jr.normal(k5, (d,), dtype=jnp.float32)
    dt = jax.nn.softplus(jr.normal(k6, (batch, seq_len, d), dtype=jnp.float32))
    gate = jr.normal(jr.PRNGKey(7), (batch, seq_len, d), dtype=jnp.float32)

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )

    ssm_op = SSM1Op(metadata)

    print("Testing training mode...")
    output = ssm_op(
        hidden_states=hidden_states,
        A=A,
        B=B,
        C=C,
        D=D,
        discrete_time_step=dt,
        gate=gate,
    )
    print(f"  Output shape: {output.attention_outputs.shape}")
    print(f"  SSM state shape: {output.ssm_state.shape}")

    print("\nTesting inference mode...")
    output_infer = ssm_op(
        hidden_states=hidden_states[:, :1, :],
        A=A,
        B=B[:, :1, :],
        C=C[:, :1, :],
        D=D,
        discrete_time_step=dt[:, :1, :],
        gate=gate[:, :1, :],
        ssm_state=output.ssm_state,
    )
    print(f"  Output shape: {output_infer.attention_outputs.shape}")
    print(f"  SSM state shape: {output_infer.ssm_state.shape}")

    print("\nAll SSM1 tests passed!")
