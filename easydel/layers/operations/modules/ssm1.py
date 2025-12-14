# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from jax import lax
from jaxtyping import Array, Float

from easydel.layers.caching import RecurrentCacheView

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)


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


def _ssm1_recurrent_scan(
    hidden_states: Float[Array, "batch intermediate_size seq_len"],
    discrete_A: Float[Array, "batch intermediate_size seq_len ssm_state_size"],
    discrete_B: Float[Array, "batch intermediate_size seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],  # noqa:F821
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[
    Float[Array, "batch intermediate_size seq_len"],
    Float[Array, "batch intermediate_size ssm_state_size"],
]:
    """Recurrent scan for SSM1.

    Implements the selective state space recurrence:
        h_t = dA_t * h_{t-1} + dB_t * x_t
        y_t = h_t @ C_t + D * x_t

    Args:
        hidden_states: Input after convolution [batch, d, seq_len]
        discrete_A: Discretized A [batch, d, seq_len, n]
        discrete_B: Discretized B [batch, d, seq_len, n]
        C: Output projection [batch, seq_len, n]
        D: Skip connection [d]
        initial_state: Initial SSM state
        dtype: Computation dtype

    Returns:
        Tuple of (outputs, final_state)
    """
    batch_size, intermediate_size, _seq_len = hidden_states.shape
    ssm_state_size = discrete_B.shape[-1]

    if initial_state is None:
        initial_state = jnp.zeros(
            (batch_size, intermediate_size, ssm_state_size),
            dtype=dtype,
        )

    # Compute deltaB_u = discrete_B * hidden_states
    # [batch, d, seq_len, n] * [batch, d, seq_len, 1] -> [batch, d, seq_len, n]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].astype(jnp.float32)

    def _step(ssm_state, step_inputs):
        dA_t, dBu_t, C_t = step_inputs
        # State update: h_t = dA_t * h_{t-1} + dB_t * x_t
        # Compute in float32 for numerical stability, then cast back to state dtype
        new_state = (dA_t * ssm_state.astype(jnp.float32) + dBu_t).astype(ssm_state.dtype)
        # Output: y_t = h_t @ C_t
        y_t = lax.batch_matmul(
            new_state.astype(dtype),
            jnp.expand_dims(C_t.astype(dtype), -1),
        )[:, :, 0]
        return new_state, y_t

    # Transpose for scan: [seq_len, batch, d, n]
    scan_inputs = (
        jnp.transpose(discrete_A, (2, 0, 1, 3)),
        jnp.transpose(deltaB_u, (2, 0, 1, 3)),
        jnp.transpose(C, (1, 0, 2)),
    )

    final_state, y = lax.scan(_step, initial_state, scan_inputs)

    # Transpose back: [batch, d, seq_len]
    y = jnp.transpose(y, (1, 2, 0))

    # Add skip connection
    y = y + hidden_states * D[None, :, None]

    return y, final_state


def _ssm1_single_step(
    hidden_state: Float[Array, "batch intermediate_size"],
    B: Float[Array, "batch ssm_state_size"],
    C: Float[Array, "batch ssm_state_size"],
    discrete_time_step: Float[Array, "batch intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    D: Float[Array, "intermediate_size"],  # noqa:F821
    ssm_state: Float[Array, "batch intermediate_size ssm_state_size"],
    dtype: jnp.dtype = jnp.float32,
) -> tuple[
    Float[Array, "batch intermediate_size"],
    Float[Array, "batch intermediate_size ssm_state_size"],
]:
    """Single step SSM1 update for inference.

    Args:
        hidden_state: Current input [batch, d]
        B: B parameter [batch, n]
        C: C parameter [batch, n]
        discrete_time_step: dt [batch, d]
        A: A matrix [d, n]
        D: Skip connection [d]
        ssm_state: Previous state [batch, d, n]
        dtype: Computation dtype

    Returns:
        Tuple of (output, new_state)
    """
    # Discretization
    # dA = exp(A * dt)
    discrete_A = jnp.exp(A[None, :, :] * discrete_time_step[:, :, None])  # [batch, d, n]

    # dB = dt * B
    discrete_B = discrete_time_step[:, :, None] * B[:, None, :].astype(jnp.float32)  # [batch, d, n]

    # State update - compute in float32 for stability, cast back to input dtype
    deltaB_u = discrete_B * hidden_state[:, :, None].astype(jnp.float32)
    new_state = (discrete_A * ssm_state.astype(jnp.float32) + deltaB_u).astype(ssm_state.dtype)

    # Output
    y = lax.batch_matmul(
        new_state.astype(dtype),
        jnp.expand_dims(C.astype(dtype), -1),
    )[:, :, 0]

    # Skip connection
    y = y + hidden_state * D[None, :]

    return y, new_state


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
        >>> from easydel.layers.operations import OperationMetadata, OperationRegistry
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

    def get_impl_metadata(self) -> OperationMetadata:
        """Returns the metadata associated with this operation instance."""
        return self.metadata

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

    @jax.named_scope("easydel-ssm1-native")
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
        activation: str = "silu",
        **kwargs,
    ) -> SSM1Output:
        """Forward pass for SSM1 operation.

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

        _batch_size, seq_len, _intermediate_size = hidden_states.shape
        _ssm_state_size = B.shape[-1]
        dtype = hidden_states.dtype
        runtime_dtype = self.metadata.runtime_dtype

        # Convert A from log form
        A_real = -jnp.exp(A.astype(jnp.float32))

        # Transpose for [batch, d, seq_len] format
        hidden_states_t = jnp.transpose(hidden_states, (0, 2, 1))
        discrete_time_step_t = jnp.transpose(discrete_time_step, (0, 2, 1))

        is_inference = seq_len == 1 and ssm_state is not None

        if is_inference:
            # Single step inference
            y, new_ssm_state = _ssm1_single_step(
                hidden_state=hidden_states_t[:, :, 0],
                B=B[:, 0, :],
                C=C[:, 0, :],
                discrete_time_step=discrete_time_step_t[:, :, 0],
                A=A_real,
                D=D,
                ssm_state=ssm_state,
                dtype=runtime_dtype,
            )
            y = y[:, :, None]  # [batch, d, 1]
        else:
            # Full sequence processing
            # Discretization
            # dA = exp(A * dt) where A is [d, n] and dt is [batch, d, seq_len]
            discrete_A = jnp.exp(
                A_real[None, :, None, :] * discrete_time_step_t[:, :, :, None]
            )  # [batch, d, seq_len, n]

            # dB = dt * B
            discrete_B = discrete_time_step_t[:, :, :, None] * B[:, None, :, :].astype(
                jnp.float32
            )  # [batch, d, seq_len, n]

            y, new_ssm_state = _ssm1_recurrent_scan(
                hidden_states=hidden_states_t.astype(jnp.float32),
                discrete_A=discrete_A,
                discrete_B=discrete_B,
                C=C,
                D=D,
                initial_state=ssm_state,
                dtype=runtime_dtype,
            )

        # Transpose back to [batch, seq_len, d]
        y = jnp.transpose(y, (0, 2, 1))

        # Apply gating
        if gate is not None:
            act_fn = ACT2FN.get(activation, jax.nn.silu)
            y = y * act_fn(gate)

        return SSM1Output(
            attention_outputs=y.astype(dtype),
            attention_weights=None,
            conv_state=conv_state,
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
