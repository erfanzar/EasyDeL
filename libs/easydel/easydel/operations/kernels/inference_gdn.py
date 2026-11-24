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

import typing as tp

import jax
import jax.numpy as jnp
from ejkernel.modules.operations.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule,
    ragged_gated_delta_rule_decode_only,
    ragged_gated_delta_rule_mixed_prefill,
    ragged_gated_delta_rule_v2_op,
    ragged_gdn_conv_decode_head_sharded,
    set_gdn_kernel_tile_policy,
)
from ejkernel.modules.operations.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule_v2 as _ejkernel_ragged_gated_delta_rule_v2,
)
from ejkernel.modules.operations.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule_v2_with_window_states as _ejkernel_ragged_gdn_v2_with_window_states,
)

from easydel.utils.helpers import check_bool_flag

from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)

_UNSET: tp.Final[object] = object()


def _normalize_ejkernel_platform(platform: object | None, *, keep_auto: bool = False) -> str | None:
    """Normalize an EasyDeL/eJKernel platform enum or string for eJKernel dispatch."""
    if platform is None:
        return None
    raw = str(getattr(platform, "value", platform)).strip().lower()
    if not raw:
        return None
    if raw == "auto":
        return "auto" if keep_auto else None
    if raw == "jax":
        return "xla"
    return raw


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
        if check_bool_flag("FORCE_NATIVE_RUNTIME", False):
            platform = "xla"
        elif _normalize_ejkernel_platform(getattr(cfg, "platform", None)) is not None:
            platform = None
        else:
            platform = _normalize_ejkernel_platform(getattr(self.metadata, "platform", None))

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
            platform=platform,
            cfg=cfg,
        )

    def forward_with_window_states(
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
        window_positions: jnp.ndarray,
        window_valid: jnp.ndarray,
        has_initial_state: jnp.ndarray | None = None,
        *,
        num_base_slots: int,
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
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Ragged GDN v2 plus the speculative-window state scan in ONE region.

        Same contract as :meth:`forward_native` plus the per-prefix window
        states (``gdn_spec_window_states``) computed inside the SAME
        ``shard_map`` region, eliminating the separate window op's per-layer
        region-boundary materialization on speculative verify steps.

        Args:
            mixed_qkv: Packed Q/K/V token buffer.
            b: Per-token beta gate input.
            a: Per-token decay gate input.
            recurrent_state: Recurrent state pool.
            A_log: Per-value-head log-``A`` parameter.
            dt_bias: Per-value-head ``dt`` bias.
            query_start_loc: CSR-style cumulative token offsets.
            state_indices: Request-to-state-slot mapping.
            distribution: Runtime distribution vector.
            window_positions: ``[num_base_slots, k + 1]`` packed-token
                positions of each slot's speculative window steps.
            window_valid: ``[num_base_slots, k + 1]`` per-step validity mask.
            has_initial_state: Optional per-request state-valid mask.
            num_base_slots: Number of live state rows.
            n_kq: Number of query/key heads.
            n_v: Number of value heads.
            d_k: Query/key head dim.
            d_v: Value head dim.
            chunk_size: Prefill chunk size.
            use_qk_norm_in_gdn: Whether to normalize Q/K in the recurrence.
            pre_sharded_mixed_qkv: Whether mixed QKV is already flat-sharded.
            flat_tp_shard: Prefer flat mixed-QKV tensor-parallel sharding.
            apply_silu_in_gdr: Whether to apply SiLU inside GDR (and to the
                window scan's rows).
            use_recurrent_scan_prefill: Enable recurrent-scan prefill.
            mask_initial_state: Whether to mask stale recurrent slots.

        Returns:
            ``(updated_recurrent_state, output, window_states)``.
        """
        mode = self.get_mode(query=jnp.expand_dims(mixed_qkv, 0), BTHD=False)
        shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")
        head_axis = shardings_bthd.query[2] if shardings_bthd.query is not None else None
        cfg = self.metadata.get_operation_config("ragged_gated_delta_rule_v2")
        if check_bool_flag("FORCE_NATIVE_RUNTIME", False):
            platform = "xla"
        elif _normalize_ejkernel_platform(getattr(cfg, "platform", None)) is not None:
            platform = None
        else:
            platform = _normalize_ejkernel_platform(getattr(self.metadata, "platform", None))

        return _ejkernel_ragged_gdn_v2_with_window_states(
            mixed_qkv,
            b,
            a,
            recurrent_state,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            window_positions,
            window_valid,
            has_initial_state,
            num_base_slots=num_base_slots,
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
            platform=platform,
            cfg=cfg,
        )

    @jax.named_scope("easydel-ragged-gdn-conv-decode")
    def forward_conv_decode(
        self,
        conv_input: jnp.ndarray,
        b_raw: jnp.ndarray,
        a_raw: jnp.ndarray,
        conv_weight: jnp.ndarray,
        conv_pool: jnp.ndarray,
        recurrent_pool: jnp.ndarray,
        A_log: jnp.ndarray,
        dt_bias: jnp.ndarray,
        token_slots: jnp.ndarray,
        token_init: jnp.ndarray,
        *,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        use_qk_l2norm: bool = True,
        apply_silu: bool = True,
        conv_output_dtype: jax.typing.DTypeLike | None = None,
        out_dtype: jax.typing.DTypeLike = jnp.float32,
        pre_sharded: bool = False,
        mesh: object = _UNSET,
        head_axis: object = _UNSET,
        allow_axis_fallback: bool | None = None,
        platform: str | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run fused causal Conv1D + SiLU + L2Norm + GDN single-token decode.

        Routes through the ``OperationRegistry``-registered ``RaggedGatedDeltaRule``
        adapter so ``FORCE_NATIVE_RUNTIME=1``, ``operation_configs["ragged_gated_delta_rule_v2"]``,
        and ``self.metadata.platform`` control backend selection while normal TPU
        execution uses the in-place Pallas decode kernel.

        Args:
            conv_input: Pre-conv flat tokens ``[T, conv_dim]``.
            b_raw: Pre-sigmoid beta gate inputs ``[T, n_v]``.
            a_raw: Pre-softplus alpha gate inputs ``[T, n_v]``.
            conv_weight: Depthwise conv kernel ``[conv_dim, d_conv]``.
            conv_pool: Rolling conv-state pool ``[S, conv_dim, d_conv]``.
            recurrent_pool: GDN state pool ``[S, n_v, d_k, d_v]``.
            A_log: Per-value-head log-decay ``[n_v]``.
            dt_bias: Per-value-head dt bias ``[n_v]``.
            token_slots: Pool slot per token ``[T]`` (out-of-range = idle).
            token_init: Bool per token ``[T]`` (``False`` = fresh zero state).
            n_kq: Number of key/query heads.
            n_v: Number of value heads.
            d_k: Key head dimension.
            d_v: Value head dimension.
            use_qk_l2norm: Whether to L2-normalize Q and K.
            apply_silu: Whether to apply SiLU after Conv1D.
            conv_output_dtype: Optional dtype for post-conv activations.
            out_dtype: Output dtype. Defaults to ``jnp.float32``.
            pre_sharded: Whether ``conv_input`` and ``conv_weight`` already
                carry per-shard interleaved ``[Q_r | K_r | V_r]`` ordering.
            mesh: Optional mesh override (defaults to ``self.metadata.mesh``;
                explicit ``None`` disables mesh sharding).
            head_axis: Optional head-sharding axis override (defaults to the
                resolved ``bthd`` head axis; explicit ``None`` disables sharding
                unless ``allow_axis_fallback=True``).
            allow_axis_fallback: Whether to fall back to another non-pipeline
                mesh axis (e.g. ``fsdp``) when ``head_axis`` cannot shard the
                heads. Defaults to ``False`` when ``head_axis=None`` is passed
                explicitly, and ``True`` otherwise.
            platform: Optional explicit eJKernel platform override.

        Returns:
            ``(token_outputs [T, n_v, d_v], updated_conv_pool, updated_recurrent_pool)``.
        """
        resolved_mesh = self.metadata.mesh if mesh is _UNSET else mesh
        if head_axis is _UNSET:
            mode = self.get_mode(query=jnp.expand_dims(conv_input, 0), BTHD=False)
            shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")
            resolved_head_axis = shardings_bthd.query[2] if shardings_bthd.query is not None else None
        else:
            resolved_head_axis = head_axis
        resolved_allow_axis_fallback = (
            (head_axis is not None) if allow_axis_fallback is None else bool(allow_axis_fallback)
        )
        cfg = self.metadata.get_operation_config("ragged_gated_delta_rule_v2")
        if platform is not None:
            resolved_platform = _normalize_ejkernel_platform(platform, keep_auto=True)
        elif check_bool_flag("FORCE_NATIVE_RUNTIME", False):
            resolved_platform = "xla"
        else:
            resolved_platform = _normalize_ejkernel_platform(getattr(cfg, "platform", None))
            if resolved_platform is None:
                resolved_platform = _normalize_ejkernel_platform(getattr(self.metadata, "platform", None))
        return ragged_gdn_conv_decode_head_sharded(
            conv_input,
            b_raw,
            a_raw,
            conv_weight,
            conv_pool,
            recurrent_pool,
            A_log,
            dt_bias,
            token_slots,
            token_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            use_qk_l2norm=use_qk_l2norm,
            apply_silu=apply_silu,
            conv_output_dtype=conv_output_dtype,
            out_dtype=out_dtype,
            pre_sharded=pre_sharded,
            mesh=resolved_mesh,
            head_axis=resolved_head_axis,
            allow_axis_fallback=resolved_allow_axis_fallback,
            platform=tp.cast(tp.Any, resolved_platform),
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
    "ragged_gdn_conv_decode_head_sharded",
    "set_gdn_kernel_tile_policy",
)
