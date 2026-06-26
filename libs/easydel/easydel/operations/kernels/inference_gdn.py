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
"""EasyDeL operation adapter for eJKernel ragged GDN v2.

The core recurrent kernels and scheduling logic live in eJKernel. This module
is the EasyDeL ``OperationImpl`` layer: it declares runtime requirements,
derives dtype/mesh/head-sharding metadata, and delegates the actual packed GDN
computation to ``ejkernel.modules.operations.ragged_gated_delta_rule_v2``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ejkernel.modules.operations.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule,
    ragged_gated_delta_rule_decode_only,
    ragged_gated_delta_rule_mixed_prefill,
    ragged_gated_delta_rule_v2_op,
    set_gdn_kernel_tile_policy,
)
from ejkernel.modules.operations.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule_v2 as _ejkernel_ragged_gated_delta_rule_v2,
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
class RaggedGatedDeltaRule(OperationImpl):
    """Packed-inference GDN operation used by Qwen3 Next/eSurge paths.

    Inputs are already packed by the caller into a mixed Q/K/V token stream and
    companion gate/state metadata. The adapter keeps EasyDeL-specific concerns
    here and lets eJKernel choose the XLA or accelerator backend.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the operation-registry key for ragged GDN v2.

        Returns:
            The registry name ``"ragged_gated_delta_rule_v2"`` under which this
            operation is resolved by ``OperationRegistry``.
        """
        return "ragged_gated_delta_rule_v2"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Describe metadata and cache requirements for packed GDN v2.

        Packed GDN needs ragged sequence metadata, position information, the
        per-request initial-state flag, and request-to-slot mapping to drive the
        recurrent state pool. It supports recurrent and hybrid cache layouts.
        ``LOGITS_INDICES`` is requested as optional metadata.

        Args:
            mode: Execution mode. Requirements are currently identical across
                modes; the parameter is kept for the ``OperationImpl`` protocol.

        Returns:
            Operation requirements declaring required ``SEQ_LENS``,
            ``POSITIONS``, ``HAS_INITIAL_STATE`` and ``STATE_INDICES`` metadata,
            optional ``LOGITS_INDICES`` metadata, and recurrent/hybrid cache
            support.
        """
        return (
            RequirementsBuilder("ragged_gated_delta_rule_v2")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.POSITIONS
                | MetadataField.HAS_INITIAL_STATE
                | MetadataField.STATE_INDICES
            )
            .optional_metadata(MetadataField.LOGITS_INDICES)
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .build()
        )

    @jax.named_scope("easydel-ragged-gated-delta-rule-native")
    def forward_native(
        self,
        mixed_qkv: jnp.ndarray,
        b: jnp.ndarray,
        a: jnp.ndarray,
        recurrent_state: jnp.ndarray,
        A_log: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        has_initial_state: jnp.ndarray | None = None,
        *,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        chunk_size: int = 64,
        use_qk_norm_in_gdn: bool = True,
        pre_sharded_mixed_qkv: bool = False,
        flat_tp_shard: bool = False,
        apply_silu_in_gdr: bool = False,
        use_recurrent_scan_prefill: bool = False,
        mask_initial_state: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run ragged GDN v2 through the public eJKernel operation.

        Args:
            mixed_qkv: Packed Q/K/V token buffer,
                ``[tokens, 2 * n_kq * d_k + n_v * d_v]``.
            b: Per-token beta gate input.
            a: Per-token decay gate input.
            recurrent_state: Recurrent state pool.
            A_log: Per-value-head log-``A`` parameter.
            dt_bias: Per-value-head ``dt`` bias.
            query_start_loc: CSR-style cumulative token offsets.
            state_indices: Request-to-state-slot mapping.
            distribution: Runtime distribution vector for decode/prefill.
            has_initial_state: Optional per-request state-valid mask.
            n_kq: Number of query/key heads.
            n_v: Number of value heads.
            d_k: Query/key head dimension.
            d_v: Value head dimension.
            chunk_size: Prefill chunk size.
            use_qk_norm_in_gdn: Whether to normalize Q/K in the recurrence.
            pre_sharded_mixed_qkv: Whether mixed QKV is already flat-sharded.
            flat_tp_shard: Prefer flat mixed-QKV tensor-parallel sharding.
            apply_silu_in_gdr: Whether to apply SiLU inside GDR.
            use_recurrent_scan_prefill: Enable recurrent-scan prefill when the
                selected backend supports it.
            mask_initial_state: Whether to use ``has_initial_state`` to zero
                stale recurrent slots for fresh prefill requests.

        Returns:
            ``(updated_recurrent_state, output)``.
        """
        mode = self.get_mode(query=jnp.expand_dims(mixed_qkv, 0), BTHD=False)
        shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")
        head_axis = shardings_bthd.query[2] if shardings_bthd.query is not None else None
        cfg = self.metadata.get_operation_config("ragged_gated_delta_rule_v2")

        return _ejkernel_ragged_gated_delta_rule_v2(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            has_initial_state=has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            pre_sharded_mixed_qkv=pre_sharded_mixed_qkv,
            flat_tp_shard=flat_tp_shard,
            apply_silu_in_gdr=apply_silu_in_gdr,
            use_recurrent_scan_prefill=use_recurrent_scan_prefill,
            mask_initial_state=mask_initial_state,
            runtime_dtype=self.metadata.runtime_dtype,
            mesh=self.metadata.mesh,
            head_axis=head_axis,
            cfg=cfg,
        )

    def forward_tpu(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """TPU forward pass. Delegates to :meth:`forward_native`.

        Args:
            *args: Positional arguments forwarded to :meth:`forward_native`.
            **kwargs: Keyword arguments forwarded to :meth:`forward_native`.

        Returns:
            ``(updated_recurrent_state, output)`` from :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """GPU forward pass. Delegates to :meth:`forward_native`.

        Args:
            *args: Positional arguments forwarded to :meth:`forward_native`.
            **kwargs: Keyword arguments forwarded to :meth:`forward_native`.

        Returns:
            ``(updated_recurrent_state, output)`` from :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """CPU forward pass. Delegates to :meth:`forward_native`.

        Args:
            *args: Positional arguments forwarded to :meth:`forward_native`.
            **kwargs: Keyword arguments forwarded to :meth:`forward_native`.

        Returns:
            ``(updated_recurrent_state, output)`` from :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """CUDA forward pass. Delegates to :meth:`forward_native`.

        Args:
            *args: Positional arguments forwarded to :meth:`forward_native`.
            **kwargs: Keyword arguments forwarded to :meth:`forward_native`.

        Returns:
            ``(updated_recurrent_state, output)`` from :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """ROCm forward pass. Delegates to :meth:`forward_native`.

        Args:
            *args: Positional arguments forwarded to :meth:`forward_native`.
            **kwargs: Keyword arguments forwarded to :meth:`forward_native`.

        Returns:
            ``(updated_recurrent_state, output)`` from :meth:`forward_native`.
        """
        return self.forward_native(*args, **kwargs)


__all__ = (
    "RaggedGatedDeltaRule",
    "ragged_gated_delta_rule",
    "ragged_gated_delta_rule_decode_only",
    "ragged_gated_delta_rule_mixed_prefill",
    "ragged_gated_delta_rule_v2_op",
    "set_gdn_kernel_tile_policy",
)
