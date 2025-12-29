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
from ejkernel.modules import state_space_v1
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
