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


def _ssm2_recurrent_scan(
    x: Float[Array, "batch num_heads seq_len head_dim"],
    B: Float[Array, "batch num_heads seq_len ssm_state_size"],
    C: Float[Array, "batch num_heads seq_len ssm_state_size"],
    dt: Float[Array, "batch seq_len num_heads"],
    A: Float[Array, "num_heads"],
    D: Float[Array, "num_heads"],
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
    precision: lax.Precision | None = None,
) -> tuple[
    Float[Array, "batch num_heads seq_len head_dim"],
    Float[Array, "batch num_heads head_dim ssm_state_size"],
]:
    """Recurrent scan for SSM2.

    Implements the Mamba2-style selective state space recurrence:
        dA = exp(dt * A)
        dBx = dt * B * x
        h_t = dA * h_{t-1} + dBx
        y_t = einsum(h_t, C_t) + x_t * D

    Args:
        x: Input [batch, num_heads, seq_len, head_dim]
        B: B parameter [batch, num_heads, seq_len, ssm_state_size]
        C: C parameter [batch, num_heads, seq_len, ssm_state_size]
        dt: Time step [batch, seq_len, num_heads]
        A: A vector (negative, per-head) [num_heads]
        D: Skip connection [num_heads]
        initial_state: Initial SSM state
        precision: JAX precision for einsum

    Returns:
        Tuple of (outputs, final_state)
    """
    batch_size, num_heads, _seq_len, head_dim = x.shape
    ssm_state_size = B.shape[-1]

    if initial_state is None:
        initial_state = jnp.zeros(
            (batch_size, num_heads, head_dim, ssm_state_size),
            dtype=jnp.float32,
        )
    else:
        initial_state = initial_state.astype(jnp.float32)

    x = x.astype(jnp.float32)
    B = B.astype(jnp.float32)
    C = C.astype(jnp.float32)
    dt = dt.astype(jnp.float32)

    def _step(ssm_state, step_inputs):
        x_t, B_t, C_t, dt_t = step_inputs
        # x_t: [batch, num_heads, head_dim]
        # B_t: [batch, num_heads, ssm_state_size]
        # C_t: [batch, num_heads, ssm_state_size]
        # dt_t: [batch, num_heads]

        # dA = exp(dt * A) where A is [num_heads]
        dt_b = dt_t[:, :, None, None]  # [batch, num_heads, 1, 1]
        dA = jnp.exp(dt_b * A[None, :, None, None])

        # dBx = dt * B * x (outer product form)
        # [batch, num_heads, head_dim, ssm_state_size]
        dBx = (dt_t[:, :, None, None] * B_t[:, :, None, :]) * x_t[:, :, :, None]

        # State update
        new_state = ssm_state * dA + dBx

        # Output: einsum("bhdn,bhn->bhd", new_state, C_t)
        y_t = jnp.einsum("bhdn,bhn->bhd", new_state, C_t, precision=precision)

        # Skip connection
        y_t = y_t + x_t * D[None, :, None]

        return new_state, y_t

    # Transpose for scan: [seq_len, batch, ...]
    scan_inputs = (
        jnp.swapaxes(x, 1, 2).swapaxes(0, 1),  # [seq_len, batch, num_heads, head_dim]
        jnp.swapaxes(B, 1, 2).swapaxes(0, 1),  # [seq_len, batch, num_heads, ssm_state_size]
        jnp.swapaxes(C, 1, 2).swapaxes(0, 1),  # [seq_len, batch, num_heads, ssm_state_size]
        jnp.swapaxes(dt, 0, 1),  # [seq_len, batch, num_heads]
    )

    final_state, y = lax.scan(_step, initial_state, scan_inputs)

    # Transpose back: [batch, num_heads, seq_len, head_dim]
    y = jnp.swapaxes(jnp.swapaxes(y, 0, 1), 1, 2)

    return y, final_state


def _ssm2_single_step(
    x: Float[Array, "batch num_heads head_dim"],
    B: Float[Array, "batch num_heads ssm_state_size"],
    C: Float[Array, "batch num_heads ssm_state_size"],
    dt: Float[Array, "batch num_heads"],
    A: Float[Array, "num_heads"],
    D: Float[Array, "num_heads"],
    ssm_state: Float[Array, "batch num_heads head_dim ssm_state_size"],
    precision: lax.Precision | None = None,
) -> tuple[
    Float[Array, "batch num_heads head_dim"],
    Float[Array, "batch num_heads head_dim ssm_state_size"],
]:
    """Single step SSM2 update for inference.

    Args:
        x: Input [batch, num_heads, head_dim]
        B: B parameter [batch, num_heads, ssm_state_size]
        C: C parameter [batch, num_heads, ssm_state_size]
        dt: Time step [batch, num_heads]
        A: A vector [num_heads]
        D: Skip connection [num_heads]
        ssm_state: Previous state [batch, num_heads, head_dim, ssm_state_size]
        precision: JAX precision

    Returns:
        Tuple of (output, new_state)
    """
    x = x.astype(jnp.float32)
    B = B.astype(jnp.float32)
    C = C.astype(jnp.float32)
    dt = dt.astype(jnp.float32)
    ssm_state = ssm_state.astype(jnp.float32)

    # dA = exp(dt * A)
    dt_b = dt[:, :, None, None]  # [batch, num_heads, 1, 1]
    dA = jnp.exp(dt_b * A[None, :, None, None])

    # dBx = dt * B * x
    dBx = (dt[:, :, None, None] * B[:, :, None, :]) * x[:, :, :, None]

    # State update
    new_state = ssm_state * dA + dBx

    # Output
    y = jnp.einsum("bhdn,bhn->bhd", new_state, C, precision=precision)
    y = y + x * D[None, :, None]

    return y, new_state


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

    @jax.named_scope("easydel-ssm2-native")
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
        """Forward pass for SSM2 operation.

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
        batch_size, seq_len, num_heads, head_dim = x.shape
        _ssm_state_size = B.shape[-1]
        dtype = x.dtype

        # Convert A from log form
        A_real = -jnp.exp(A.astype(jnp.float32))

        # Expand B, C from n_groups to num_heads
        group_rep = num_heads // n_groups
        B = jnp.repeat(B, repeats=group_rep, axis=2)  # [batch, seq_len, num_heads, n]
        C = jnp.repeat(C, repeats=group_rep, axis=2)  # [batch, seq_len, num_heads, n]

        # Transpose x to [batch, num_heads, seq_len, head_dim]
        x_t = jnp.transpose(x, (0, 2, 1, 3))
        B_t = jnp.transpose(B, (0, 2, 1, 3))  # [batch, num_heads, seq_len, n]
        C_t = jnp.transpose(C, (0, 2, 1, 3))  # [batch, num_heads, seq_len, n]

        is_inference = seq_len == 1 and ssm_state is not None

        if is_inference:
            # Single step inference
            y, new_ssm_state = _ssm2_single_step(
                x=x_t[:, :, 0, :],  # [batch, num_heads, head_dim]
                B=B_t[:, :, 0, :],  # [batch, num_heads, n]
                C=C_t[:, :, 0, :],  # [batch, num_heads, n]
                dt=dt[:, 0, :],  # [batch, num_heads]
                A=A_real,
                D=D,
                ssm_state=ssm_state,
                precision=precision,
            )
            y = y[:, :, None, :]  # [batch, num_heads, 1, head_dim]
        else:
            # Full sequence processing
            y, new_ssm_state = _ssm2_recurrent_scan(
                x=x_t,
                B=B_t,
                C=C_t,
                dt=dt,
                A=A_real,
                D=D,
                initial_state=ssm_state,
                precision=precision,
            )

        # Transpose back to [batch, seq_len, num_heads, head_dim]
        y = jnp.transpose(y, (0, 2, 1, 3))

        # Reshape to [batch, seq_len, intermediate_size]
        intermediate_size = num_heads * head_dim
        y = y.reshape(batch_size, seq_len, intermediate_size)

        # Apply gating
        if gate is not None:
            if use_gated_rmsnorm:
                # Gated RMSNorm
                y_f32 = y.astype(jnp.float32)
                variance = jnp.mean(jnp.square(y_f32), axis=-1, keepdims=True)
                y_norm = y_f32 * lax.rsqrt(variance + rmsnorm_eps)
                y = (y_norm * jax.nn.silu(gate.astype(jnp.float32))).astype(dtype)
            else:
                # Simple gating
                y = y * jax.nn.silu(gate.astype(jnp.float32))

        return SSM2Output(
            attention_outputs=y.astype(dtype),
            attention_weights=None,
            conv_state=conv_state,
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
