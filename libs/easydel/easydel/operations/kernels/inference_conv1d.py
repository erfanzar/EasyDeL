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
"""Ragged causal depthwise 1-D convolution adapter for packed batches.

This module exposes EasyDeL's operation-registry adapter for the short causal
depthwise convolution that precedes the gated-delta-rule recurrence in models
such as Qwen3 Next, Qwen3.5, and other linear-attention / state-space hybrids.

The implementation lives in eJKernel. EasyDeL keeps this file as the public
operation layer: it declares requirements, participates in
``OperationRegistry``, and delegates execution to
``ejkernel.modules.operations.ragged_causal_conv1d``. The operation targets
packed / continuous-batching inference where single-token decode requests and
multi-token prefill requests share one contiguous token buffer.

Exports:
    ``ragged_causal_conv1d``:
        Public eJKernel functional entry point re-exported for compatibility.
    ``ragged_causal_conv1d_head_sharded``:
        eJKernel helper for feature/head-axis sharded packed conv execution.
    ``RaggedCausalConv1D``:
        Registered EasyDeL ``OperationImpl`` wrapper resolved by the operation
        registry under ``"ragged_causal_conv1d"``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ejkernel.modules.operations import (
    ragged_causal_conv1d,
    ragged_causal_conv1d_head_sharded,
    ragged_causal_conv1d_op,
)

from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)


@OperationRegistry.register
class RaggedCausalConv1D(OperationImpl):
    """Ragged causal depthwise conv1d operation for packed inference batches.

    The adapter consumes the packed token stream, rolling convolution state
    pool, depthwise kernel, and ragged request metadata produced by EasyDeL's
    inference runtime. It forwards those values to eJKernel, which owns the
    backend implementation and optional sharded execution paths.

    This class is intentionally thin: EasyDeL handles metadata and operation
    dispatch, while eJKernel handles the algorithm and backend selection.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the operation-registry key for ragged causal conv1d."""
        return "ragged_causal_conv1d"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Describe metadata and cache requirements for packed conv1d.

        The operation needs ragged sequence metadata and state-slot mappings to
        interpret the packed token buffer. It supports recurrent and hybrid
        cache layouts because the convolution state is maintained alongside
        recurrent linear-attention state.

        Args:
            mode: Execution mode. Requirements are currently identical across
                modes and the parameter is kept for the ``OperationImpl``
                protocol.

        Returns:
            Operation requirements for ``SEQ_LENS``, ``POSITIONS`` and
            ``STATE_INDICES`` metadata with recurrent/hybrid cache support.
        """
        return (
            RequirementsBuilder("ragged_causal_conv1d")
            .require_metadata(MetadataField.SEQ_LENS | MetadataField.POSITIONS | MetadataField.STATE_INDICES)
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .build()
        )

    @jax.named_scope("easydel-ragged-causal-conv1d-native")
    def forward_native(
        self,
        x: jnp.ndarray,
        conv_state: jnp.ndarray,
        kernel: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        *,
        d_conv: int,
        apply_silu: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run ragged causal conv1d through eJKernel.

        Args:
            x: Packed input stream, shape ``[num_tokens, conv_dim]``.
            conv_state: Per-slot rolling state pool, shape
                ``[num_slots, conv_dim, d_conv]``. Position
                ``d_conv - 1`` stores the most recent historical token.
            kernel: Depthwise convolution kernel, shape
                ``[conv_dim, d_conv]``. The last tap is the current-token tap.
            query_start_loc: CSR-style cumulative token offsets per request,
                shape ``[num_slots + 1]``.
            state_indices: Mapping from request index to state-pool slot,
                shape ``[num_slots]``.
            distribution: Runtime distribution vector. For this operation,
                ``distribution[2]`` is the number of valid request rows.
            d_conv: Convolution window size. Must match the trailing dimension
                of ``conv_state`` and ``kernel``.
            apply_silu: Whether eJKernel should apply SiLU to the convolution
                output, matching Qwen3 Next / GatedDeltaNet conventions.

        Returns:
            ``(output, updated_conv_state)`` where ``output`` has shape
            ``[num_tokens, conv_dim]`` and ``updated_conv_state`` has the same
            shape and dtype as ``conv_state``.
        """
        return ragged_causal_conv1d(
            x,
            conv_state,
            kernel,
            query_start_loc,
            state_indices,
            distribution,
            d_conv=d_conv,
            apply_silu=apply_silu,
        )

    def forward_tpu(self, *args, **kwargs):
        """TPU forward pass. Delegates to the public eJKernel operation."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs):
        """GPU forward pass. Delegates to the public eJKernel operation."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        """CPU forward pass for host-side reference and structure checks."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        """CUDA forward pass. Delegates to the public eJKernel operation."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs):
        """ROCm forward pass. Delegates to the public eJKernel operation."""
        return self.forward_native(*args, **kwargs)


__all__ = (
    "RaggedCausalConv1D",
    "ragged_causal_conv1d",
    "ragged_causal_conv1d_head_sharded",
    "ragged_causal_conv1d_op",
)
