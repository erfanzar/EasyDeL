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
SSM2 (Mamba2-style) Selective State Space operation for EasyDeL.

This module provides the SSM2 operation, implementing the Mamba2
selective state space model architecture used by Mamba2 and FalconH1.

This implementation delegates to ejKernel's optimized state_space_v2 kernel
for the core SSM computation, providing automatic platform selection and
optimization.

Key characteristics of SSM2:
- 1D A matrix: [num_heads] (per-head scalar)
- SSM state shape: [batch, num_heads, head_dim, ssm_state_size]
- No separate dt_proj - dt comes from input projection
- B, C extracted from convolution output with n_groups
- Output gating: via gated RMSNorm or y * silu(gate)

The algorithm:
    Discretization:
        dA = exp(A * dt)  where A is per-head
        dB = dt * B

    Recurrence (per head):
        h_t = dA * h_{t-1} + dB * (x_t outer k_t)
        y_t = einsum(h_t, C_t) + x_t * D

References:
    - Mamba2: https://arxiv.org/abs/2405.21060
    - FalconH1: https://huggingface.co/tiiuae/Falcon-H1-1B-Base
"""

import jax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from ejkernel.modules import state_space_v2
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
class SSM2Output(AttentionOutput):
    """Output container for SSM2 operation.

    Attributes:
        attention_outputs: Output tensor [batch, seq_len, num_heads, head_dim]
        attention_weights: Always None for SSM (no attention weights)
        conv_state: Updated convolution state [batch, conv_dim, d_conv]
        ssm_state: Updated SSM state [batch, num_heads, head_dim, ssm_state_size]
    """

    conv_state: Float[Array, "batch conv_dim d_conv"] | None = None
    ssm_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None


@OperationRegistry.register
class SSM2Op(OperationImpl):
    """SSM2 (Mamba2-style) selective state space operation.

    Implements the Mamba2 architecture with:
    - 1D A vector [num_heads] (per-head scalar)
    - n_groups for B, C grouping
    - SSM state shape [batch, num_heads, head_dim, ssm_state_size]

    This operation is used by Mamba2 and FalconH1 models.

    Registered under the names "ssm2", "mamba2".

    Example:
        >>> from easydel.layers.operations import OperationMetadata, OperationRegistry
        >>> metadata = OperationMetadata(runtime_dtype=jnp.float16)
        >>> ssm_op = OperationRegistry.create("ssm2", metadata)
        >>> output = ssm_op(
        ...     x=x,
        ...     A=A_log,
        ...     B=B,
        ...     C=C,
        ...     D=D,
        ...     dt=dt,
        ...     gate=gate,
        ... )
    """

    @classmethod
    def get_impl_name(cls) -> tuple[str, ...]:
        """Returns the registered names of this operation."""
        return ("ssm2", "mamba2")

    def get_impl_metadata(self) -> OperationMetadata:
        """Returns the metadata associated with this operation instance."""
        return self.metadata

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for SSM2Op.

        SSM2 requires:
        - Basic metadata plus state management fields
        - Recurrent or Hybrid cache type for SSM state persistence
        """
        return (
            RequirementsBuilder("ssm2")
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

    @jax.named_scope("easydel-ssm2-ejkernel")
    def forward_native(
        self,
        x: Float[Array, "batch seq_len num_heads head_dim"],
        A: Float[Array, "num_heads"],
        B: Float[Array, "batch seq_len n_groups ssm_state_size"],
        C: Float[Array, "batch seq_len n_groups ssm_state_size"],
        D: Float[Array, "num_heads"],
        dt: Float[Array, "batch seq_len num_heads"],
        gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
        conv_state: Float[Array, "batch conv_dim d_conv"] | None = None,
        ssm_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
        n_groups: int = 1,
        use_gated_rmsnorm: bool = False,
        rmsnorm_eps: float = 1e-5,
        precision: lax.Precision | None = None,
        **kwargs,
    ) -> SSM2Output:
        """Forward pass for SSM2 operation using ejKernel.

        Delegates to ejkernel.modules.operations.state_space_v2 for the core
        SSM computation, which provides optimized implementations with automatic
        platform selection.

        Args:
            x: Input tensor [batch, seq_len, num_heads, head_dim]
            A: A vector in log form [num_heads]
            B: B parameter [batch, seq_len, n_groups, ssm_state_size]
            C: C parameter [batch, seq_len, n_groups, ssm_state_size]
            D: Skip connection [num_heads]
            dt: Time step after softplus [batch, seq_len, num_heads]
            gate: Optional gate tensor [batch, seq_len, intermediate_size]
            conv_state: Optional conv state for caching
            ssm_state: Optional SSM state for caching
            n_groups: Number of groups for B, C
            use_gated_rmsnorm: Whether to use gated RMSNorm for output
            rmsnorm_eps: Epsilon for RMSNorm
            precision: JAX precision for matmul

        Returns:
            SSM2Output with outputs and updated states
        """
        dtype = x.dtype

        # Convert A from log form to real form (negative for stability)
        A_real = -jnp.exp(A.astype(jnp.float32))

        # Call ejKernel's state_space_v2
        # It handles both training (full sequence) and inference (single step) modes
        y, new_ssm_state, new_conv_state = state_space_v2(
            x,
            A_real,
            B,
            C,
            D,
            dt,
            gate=gate,
            initial_state=ssm_state,
            conv_state=conv_state,
            n_groups=n_groups,
            act_fn=jax.nn.silu if gate is not None else None,
            use_gated_rmsnorm=use_gated_rmsnorm,
            rmsnorm_eps=rmsnorm_eps,
            precision=precision,
        )

        return SSM2Output(
            attention_outputs=y.astype(dtype),
            attention_weights=None,
            conv_state=new_conv_state,
            ssm_state=new_ssm_state.astype(dtype),
        )

    def forward_tpu(self, *args, **kwargs) -> SSM2Output:
        """TPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> SSM2Output:
        """GPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> SSM2Output:
        """CPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        x: Float[Array, "batch seq_len num_heads head_dim"],
        A: Float[Array, "num_heads"],
        B: Float[Array, "batch seq_len n_groups ssm_state_size"],
        C: Float[Array, "batch seq_len n_groups ssm_state_size"],
        D: Float[Array, "num_heads"],
        dt: Float[Array, "batch seq_len num_heads"],
        gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
        conv_state: Float[Array, "batch conv_dim d_conv"] | None = None,
        ssm_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
        n_groups: int = 1,
        use_gated_rmsnorm: bool = False,
        rmsnorm_eps: float = 1e-5,
        precision: lax.Precision | None = None,
        **kwargs,
    ) -> SSM2Output:
        """Execute the SSM2 operation.

        Dispatches to appropriate backend via parent __call__.

        Args:
            x: Input tensor [batch, seq_len, num_heads, head_dim]
            A: A vector in log form [num_heads]
            B: B parameter [batch, seq_len, n_groups, ssm_state_size]
            C: C parameter [batch, seq_len, n_groups, ssm_state_size]
            D: Skip connection [num_heads]
            dt: Time step [batch, seq_len, num_heads]
            gate: Optional gate tensor
            conv_state: Optional conv state
            ssm_state: Optional SSM state
            n_groups: Number of groups for B, C
            use_gated_rmsnorm: Whether to use gated RMSNorm
            rmsnorm_eps: Epsilon for RMSNorm
            precision: JAX precision

        Returns:
            SSM2Output with outputs and states
        """
        return super().__call__(
            x=x,
            A=A,
            B=B,
            C=C,
            D=D,
            dt=dt,
            gate=gate,
            conv_state=conv_state,
            ssm_state=ssm_state,
            n_groups=n_groups,
            use_gated_rmsnorm=use_gated_rmsnorm,
            rmsnorm_eps=rmsnorm_eps,
            precision=precision,
            **kwargs,
        )


if __name__ == "__main__":
    from jax import random as jr

    from easydel.infra import EasyDeLBaseConfig

    print("Testing SSM2Op...")

    batch, seq_len, num_heads, head_dim, n_groups, ssm_state_size = 2, 64, 8, 64, 1, 16

    key = jr.PRNGKey(0)
    k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)

    x = jr.normal(k1, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.1
    A = jr.normal(k2, (num_heads,), dtype=jnp.float32)
    B = jr.normal(k3, (batch, seq_len, n_groups, ssm_state_size), dtype=jnp.float32) * 0.1
    C = jr.normal(k4, (batch, seq_len, n_groups, ssm_state_size), dtype=jnp.float32) * 0.1
    D = jr.normal(k5, (num_heads,), dtype=jnp.float32)
    dt = jax.nn.softplus(jr.normal(k6, (batch, seq_len, num_heads), dtype=jnp.float32))
    gate = jr.normal(k7, (batch, seq_len, num_heads * head_dim), dtype=jnp.float32)

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )

    ssm_op = SSM2Op(metadata)

    print("Testing training mode...")
    output = ssm_op(
        x=x,
        A=A,
        B=B,
        C=C,
        D=D,
        dt=dt,
        gate=gate,
        n_groups=n_groups,
    )
    print(f"  Output shape: {output.attention_outputs.shape}")
    print(f"  SSM state shape: {output.ssm_state.shape}")

    print("\nTesting inference mode...")
    output_infer = ssm_op(
        x=x[:, :1, :, :],
        A=A,
        B=B[:, :1, :, :],
        C=C[:, :1, :, :],
        D=D,
        dt=dt[:, :1, :],
        gate=gate[:, :1, :],
        ssm_state=output.ssm_state,
        n_groups=n_groups,
    )
    print(f"  Output shape: {output_infer.attention_outputs.shape}")
    print(f"  SSM state shape: {output_infer.ssm_state.shape}")

    print("\nTesting with gated RMSNorm...")
    output_norm = ssm_op(
        x=x,
        A=A,
        B=B,
        C=C,
        D=D,
        dt=dt,
        gate=gate,
        n_groups=n_groups,
        use_gated_rmsnorm=True,
    )
    print(f"  Output shape: {output_norm.attention_outputs.shape}")

    print("\nAll SSM2 tests passed!")
