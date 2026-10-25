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
from eformer.pytree import auto_pytree
from ejkernel.modules import state_space_v1  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
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

    ssm_state_new = dA * ssm_state + dBx

    # Output: y = sum(state * C, axis=-1) + D * x
    y = jnp.sum(ssm_state_new * C[:, None, :], axis=-1) + D[None, :] * hidden_states

    return y, ssm_state_new


def _reference_ssm1_fwd(
    hidden_states: Array,
    A_real: Array,
    B: Array,
    C: Array,
    D: Array,
    discrete_time_step: Array,
    *,
    ssm_state: Array | None = None,
    valid: Array | None = None,
    reset: Array | None = None,
    gate: Array | None = None,
    act_fn=None,
    use_associative_scan: bool = False,
) -> tuple[Array, Array]:
    """Run float32 affine SSM transitions with either reference scan algorithm.

    Args:
        hidden_states: Inputs [batch, length, intermediate_size].
        A_real: Real transition coefficients [intermediate_size, state_size].
        B: Input coefficients [batch, length, state_size].
        C: Output coefficients [batch, length, state_size].
        D: Skip coefficients [intermediate_size].
        discrete_time_step: Positive steps [batch, length, intermediate_size].
        ssm_state: Optional initial state [batch, intermediate_size, state_size].
        valid: Optional [batch, length] mask; invalid cells are identity transitions.
        reset: Optional [batch, length] mask resetting state before a valid token.
        gate: Optional output gate [batch, length, intermediate_size].
        act_fn: Activation applied to the output gate, when provided.
        use_associative_scan: Compose affine transitions in parallel instead of
            applying the recurrence sequentially.

    Returns:
        Float32 outputs and final recurrent state; an empty sequence keeps its initial state.
    """
    batch, length, intermediate = hidden_states.shape
    state0 = (
        jnp.zeros((batch, intermediate, A_real.shape[-1]), dtype=jnp.float32)
        if ssm_state is None
        else ssm_state.astype(jnp.float32)
    )
    if length == 0:
        return jnp.zeros(hidden_states.shape, dtype=jnp.float32), state0

    x = hidden_states.astype(jnp.float32)
    b = B.astype(jnp.float32)
    c = C.astype(jnp.float32)
    dt = discrete_time_step.astype(jnp.float32)
    if valid is not None:
        # Padding may contain arbitrary values (including NaNs). Mask operands
        # before discretizing, so padding cannot contaminate a prefix or its gradients.
        x = jnp.where(valid[..., None], x, 0.0)
        b = jnp.where(valid[..., None], b, 0.0)
        c = jnp.where(valid[..., None], c, 0.0)
        dt = jnp.where(valid[..., None], dt, 0.0)
    transition = jnp.exp(dt[..., None] * A_real.astype(jnp.float32)[None, None, :, :])
    update = dt[..., None] * b[:, :, None, :] * x[..., None]
    if reset is not None:
        transition = jnp.where(reset[..., None, None], 0.0, transition)
    if valid is not None:
        transition = jnp.where(valid[..., None, None], transition, 1.0)
        update = jnp.where(valid[..., None, None], update, 0.0)

    if use_associative_scan:

        def compose(left, right):
            a_left, b_left = left
            a_right, b_right = right
            return a_right * a_left, a_right * b_left + b_right

        prefix_a, prefix_b = jax.lax.associative_scan(compose, (transition, update), axis=1)
        states = prefix_a * state0[:, None, :, :] + prefix_b
        final_state = states[:, -1]
        outputs = jnp.sum(states * c[:, :, None, :], axis=-1) + D.astype(jnp.float32)[None, None, :] * x
    else:

        def step(state, values):
            a_t, b_t, c_t, x_t = values
            next_state = a_t * state + b_t
            y_t = jnp.sum(next_state * c_t[:, None, :], axis=-1) + D.astype(jnp.float32)[None, :] * x_t
            return next_state, y_t

        final_state, outputs = jax.lax.scan(step, state0, tuple(v.swapaxes(0, 1) for v in (transition, update, c, x)))
        outputs = outputs.swapaxes(0, 1)

    if gate is not None and act_fn is not None:
        gate = gate.astype(jnp.float32)
        if valid is not None:
            gate = jnp.where(valid[..., None], gate, 0.0)
        outputs = outputs * act_fn(gate)
    if valid is not None:
        outputs = jnp.where(valid[..., None], outputs, 0.0)
    return outputs, final_state


def packed_ssm1_fwd(
    hidden_states: Float[Array, "slots width intermediate_size"],
    A_real: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "slots width ssm_state_size"],
    C: Float[Array, "slots width ssm_state_size"],
    D: Float[Array, "intermediate_size"],  # noqa:F821
    discrete_time_step: Float[Array, "slots width intermediate_size"],
    valid: Array,
    ssm_state: Float[Array, "slots intermediate_size ssm_state_size"],
    *,
    use_associative_scan: bool = False,
) -> tuple[Float[Array, "slots width intermediate_size"], Float[Array, "slots intermediate_size ssm_state_size"]]:
    """Run the SSM1 recurrence across slot-major packed grids.

    Invalid (padding) cells are identity transitions: they emit zero and keep
    the incoming state, allowing each slot to have an independent ragged length.

    Args:
        hidden_states: Inputs shaped [slots, width, intermediate_size].
        A_real: Negative real transition coefficients [intermediate_size, state_size].
        B: Input coefficients [slots, width, state_size].
        C: Output coefficients [slots, width, state_size].
        D: Skip coefficients [intermediate_size].
        discrete_time_step: Positive steps [slots, width, intermediate_size].
        valid: Boolean validity mask [slots, width].
        ssm_state: Initial recurrent state [slots, intermediate_size, state_size].
        use_associative_scan: Compose affine transitions in parallel when True;
            otherwise use the sequential recurrence.

    Returns:
        Outputs in the input layout and final float32 recurrent states.
    """
    return _reference_ssm1_fwd(
        hidden_states,
        A_real,
        B,
        C,
        D,
        discrete_time_step,
        valid=valid,
        ssm_state=ssm_state,
        use_associative_scan=use_associative_scan,
    )


# Compatibility for existing model callers; new callers use the public helper.
_packed_ssm1_fwd = packed_ssm1_fwd


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
    use_associative_scan: bool = False,
) -> tuple[Float[Array, "batch seq_len intermediate_size"], Float[Array, "batch intermediate_size ssm_state_size"]]:
    """Run either reference scan, resetting state between packed documents.

    The first valid token continues a supplied initial state, including after
    leading padding. Padding preserves state but separates subsequent document
    runs; a valid token after interior padding therefore starts a new run.
    """
    seq_len = hidden_states.shape[1]
    segment_ids = normalize_packed_segment_ids(segment_ids, seq_len, pad_from_last=False)
    valid = segment_ids >= 0
    previous = jnp.concatenate((jnp.full_like(segment_ids[:, :1], -1), segment_ids[:, :-1]), axis=1)
    # The initial state belongs to the first valid document. Further boundaries
    # (including an intervening pad) reset it, not the first token of a chunk.
    seen_before = jnp.cumsum(valid.astype(jnp.int32), axis=1) - valid.astype(jnp.int32) > 0
    reset = valid & seen_before & (segment_ids != previous)
    return _reference_ssm1_fwd(
        hidden_states,
        A_real,
        B,
        C,
        D,
        discrete_time_step,
        valid=valid,
        reset=reset,
        gate=gate,
        ssm_state=ssm_state,
        act_fn=act_fn,
        use_associative_scan=use_associative_scan,
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
        use_associative_scan: bool | None = None,
        use_mambapy: bool = False,
        **kwargs,
    ) -> SSM1Output:
        """Forward pass for SSM1 with automatic or explicit reference scan selection.

        Both associative and sequential non-mambapy calls use the registered
        ejkernel state_space_v1 path. An explicit use_mambapy request selects
        the JAX reference independently of the scan algorithm; it does not require
        the PyTorch mambapy package. Packed boundaries use a reset-aware scan.

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
            segment_ids: Optional document ids; negative ids mark identity padding.
                The first valid document continues ssm_state; subsequent runs reset.
            activation: Activation function name for gating
            use_associative_scan: True selects parallel affine composition, False
                selects the sequential recurrence, and None preserves automatic
                ejkernel selection (sequential for segmented inputs).
            use_mambapy: Select the JAX Mamba Python-reference path instead of
                ejkernel. Defaults to associative scan unless explicitly overridden.
            **kwargs: Additional backend-compatible arguments.

        Returns:
            SSM1Output with outputs and updated states
        """
        from easydel.infra.utils import ACT2FN

        dtype = hidden_states.dtype

        # Convert A from log form to real form (negative for stability)
        A_real = -jnp.exp(A.astype(jnp.float32))

        act_fn = ACT2FN.get(activation, jax.nn.silu) if gate is not None else None

        associative = bool(use_associative_scan) if use_associative_scan is not None else bool(use_mambapy)
        if segment_ids is not None:
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
                use_associative_scan=associative,
            )
            return SSM1Output(
                attention_outputs=y.astype(dtype),
                attention_weights=None,
                conv_state=conv_state,
                ssm_state=new_ssm_state.astype(dtype),
            )

        if use_mambapy or hidden_states.shape[1] == 0:
            y, new_ssm_state = _reference_ssm1_fwd(
                hidden_states,
                A_real,
                B,
                C,
                D,
                discrete_time_step,
                ssm_state=ssm_state,
                gate=gate,
                act_fn=act_fn,
                use_associative_scan=associative,
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
            use_associative_scan=use_associative_scan,
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
        use_associative_scan: bool | None = None,
        use_mambapy: bool = False,
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
            segment_ids: Optional document ids, with negative ids marking padding.
            activation: Activation function name
            use_associative_scan: True for parallel affine composition, False for
                sequential recurrence, None for automatic kernel selection.
            use_mambapy: Use the JAX Mamba reference instead of the optimized kernel.
            **kwargs: Additional backend-compatible arguments.

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
            use_associative_scan=use_associative_scan,
            use_mambapy=use_mambapy,
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
