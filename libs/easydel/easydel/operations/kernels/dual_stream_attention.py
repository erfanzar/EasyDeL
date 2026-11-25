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

"""EasyDeL operation adapter for Dual-Stream Hybrid Attention.

Thin :class:`~easydel.operations.OperationImpl` over eJKernel's
:func:`ejkernel.modules.dual_stream_attention` public operation — used by
GLM-5.4-style Loop Transformers (Pass 2+ attending jointly over Pass-1 global
KV and Pass-2 local sliding-window KV under one unified softmax normalizer).

Responsibilities owned by this adapter:
    - Layout translation: EasyDeL BTHD ``[B, T, H, D]`` <-> eJKernel BHTD ``[B, H, T, D]``.
    - ``MaskInfo`` extraction: segment IDs for fast cacheless packed training,
      or explicit boolean masks for cached decode / custom masks.
    - Multi-chip ``shard_map`` sharding over ``(batch, head)`` axes when
      running the TPU Pallas SplashAttention backend (leaving XLA / SP>1 to GSPMD).
    - Honoring ``FORCE_NATIVE_RUNTIME=1`` and ``OperationMetadata`` configs.
"""

from __future__ import annotations

import typing as tp

import jax
from ejkernel.modules import DualStreamAttentionConfig
from ejkernel.modules import dual_stream_attention as _ejkernel_dual_stream_attention
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float, Int
from spectrax import common_types

from easydel.caching import TransformerCacheView
from easydel.infra.sharding import mesh_axis_size, resolve_stage_mesh
from easydel.utils.helpers import check_bool_flag

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)
from ._compressed_window_common import configured_platform, resolve_platform


def _build_explicit_stream_masks(
    *,
    bsz: int,
    q_len: int,
    kv_g_len: int,
    kv_l_len: int,
    global_mask_info: MaskInfo | None,
    local_mask_info: MaskInfo | None,
    cache_end_pos: Int[Array, "batch"] | None,  # noqa: F821
    sliding_window: int,
    overlap: bool,
) -> tuple[Array, Array]:
    """Build explicit ``[B, 1|H, Q, K]`` boolean masks for cached decode or dense-mask inputs."""
    if cache_end_pos is not None:
        g_q_pos = getattr(global_mask_info, "q_positions", None) if global_mask_info is not None else None
        g_k_pos = getattr(global_mask_info, "kv_positions", None) if global_mask_info is not None else None
        if g_q_pos is None:
            g_q_pos = (cache_end_pos - q_len)[:, None] + jnp.arange(q_len, dtype=jnp.int32)[None, :]
        if g_k_pos is None:
            g_k_pos = jnp.broadcast_to(jnp.arange(kv_g_len, dtype=jnp.int32)[None, :], (bsz, kv_g_len))
    else:
        g_q_pos = jnp.broadcast_to(jnp.arange(q_len, dtype=jnp.int32)[None, :], (bsz, q_len))
        g_k_pos = jnp.broadcast_to(jnp.arange(kv_g_len, dtype=jnp.int32)[None, :], (bsz, kv_g_len))
    rel_g = g_q_pos[:, :, None] - g_k_pos[:, None, :]

    if global_mask_info is not None:
        g_mask = global_mask_info.get_or_compute_attention_mask(dtype=jnp.bool_)
        g_causal_baked = bool(getattr(global_mask_info, "causal_mask_baked_in", False))
    else:
        g_mask = jnp.ones((bsz, 1, q_len, kv_g_len), dtype=jnp.bool_)
        g_causal_baked = False
    if not g_causal_baked:
        g_mask = g_mask & (rel_g >= 0)[:, None, :, :]
    if not overlap:
        in_local = (rel_g >= 0) & (rel_g < int(sliding_window))
        g_mask = g_mask & (~in_local)[:, None, :, :]

    if local_mask_info is not None:
        l_mask = local_mask_info.get_or_compute_attention_mask(dtype=jnp.bool_)
        l_sw_baked = bool(getattr(local_mask_info, "sliding_window_baked_in", False))
    else:
        l_mask = jnp.ones((bsz, 1, q_len, kv_l_len), dtype=jnp.bool_)
        l_sw_baked = False
    if not l_sw_baked:
        l_q_pos = (
            getattr(local_mask_info, "q_positions", None)
            if (cache_end_pos is not None and local_mask_info is not None)
            else None
        )
        l_k_pos = (
            getattr(local_mask_info, "kv_positions", None)
            if (cache_end_pos is not None and local_mask_info is not None)
            else None
        )
        if l_q_pos is not None and l_k_pos is not None:
            rel_l = l_q_pos[:, :, None] - l_k_pos[:, None, :]
        else:
            rel_l = (
                jnp.arange(q_len, dtype=jnp.int32)[None, :, None] - jnp.arange(kv_l_len, dtype=jnp.int32)[None, None, :]
            )
        l_win = (rel_l >= 0) & (rel_l < int(sliding_window))
        l_mask = l_mask & l_win[:, None, :, :]

    return g_mask, l_mask


@OperationRegistry.register
class DualStreamAttn(OperationImpl):
    """Dual-Stream Hybrid Attention (global KV + local sliding-window KV).

    Registered under ``"dual_stream_attention"``. Accepts EasyDeL BTHD tensors
    and delegates to :func:`ejkernel.modules.dual_stream_attention` (Pallas TPU
    SplashAttention with exact custom VJP, or blocked online-softmax XLA fallback).
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the operation registry key."""
        return "dual_stream_attention"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Declare operation metadata and cache requirements."""
        del mode
        return (
            RequirementsBuilder("dual_stream_attention")
            .require_metadata(MetadataField.basic())
            .support_cache(CacheType.TRANSFORMER | CacheType.HYBRID)
            .use_cache_view(TransformerCacheView)
            .build()
        )

    def _resolve_shard_map_kwargs(
        self,
        *,
        q_bhtd: Array,
        bsz: int,
        q_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_seg_operands: int,
        resolved_platform: str | None,
        pallas_eligible: bool,
    ) -> dict[str, tp.Any]:
        """Build ``shard_map`` kwargs for TPU Pallas execution, or ``{}`` to defer to GSPMD."""
        if resolved_platform == "xla" or jax.default_backend() != "tpu" or not pallas_eligible:
            return {}
        mesh = self.metadata.mesh
        stage_mesh = resolve_stage_mesh(mesh, arr=q_bhtd) if mesh is not None else None
        jax_mesh = getattr(stage_mesh, "jax_mesh", stage_mesh)
        if jax_mesh is None or getattr(jax_mesh, "size", 1) <= 1:
            return {}

        resolver = self.metadata.runtime_sharding_resolver.with_mesh(stage_mesh)
        q_bthd_spec = resolver.resolve(
            dynamic_axes=common_types.AttnQSharding,
            shape=(bsz, q_len, num_q_heads, head_dim),
        )
        kv_bthd_spec = resolver.resolve(
            dynamic_axes=common_types.AttnKVSharding,
            shape=(bsz, q_len, num_kv_heads, head_dim),
        )
        b_ax = q_bthd_spec[0] if len(q_bthd_spec) == 4 else None
        sp_ax = q_bthd_spec[1] if len(q_bthd_spec) == 4 else None
        h_q_ax = q_bthd_spec[2] if len(q_bthd_spec) == 4 else None
        h_kv_ax = kv_bthd_spec[2] if len(kv_bthd_spec) == 4 else None

        sp_size = int(mesh_axis_size(stage_mesh, sp_ax))
        b_size = int(mesh_axis_size(stage_mesh, b_ax))
        h_size = int(mesh_axis_size(stage_mesh, h_q_ax))
        can_shard_heads = (
            h_q_ax == h_kv_ax
            and h_size >= 1
            and num_q_heads % h_size == 0
            and num_kv_heads % h_size == 0
        )
        use_b_ax = b_ax if (b_size > 1 and bsz % b_size == 0) else None
        use_h_ax = h_q_ax if can_shard_heads else None
        if sp_size != 1 or (use_b_ax is None and use_h_ax is None):
            return {}

        q_spec = Ps(use_b_ax, use_h_ax, None, None)
        kv_spec = Ps(use_b_ax, use_h_ax, None, None)
        seg_spec = Ps(use_b_ax, None)
        in_specs = (q_spec, kv_spec, kv_spec, kv_spec, kv_spec) + (seg_spec,) * num_seg_operands
        return dict(mesh=jax_mesh, in_specs=in_specs, out_specs=q_spec)

    @jax.named_scope("easydel-dual-stream-attention")
    def forward_native(
        self,
        query: Float[Array, "batch q_len num_q_heads head_dim"],
        k_global: Float[Array, "batch kv_g_len num_kv_heads head_dim"],
        v_global: Float[Array, "batch kv_g_len num_kv_heads head_dim"],
        k_local: Float[Array, "batch kv_l_len num_kv_heads head_dim"],
        v_local: Float[Array, "batch kv_l_len num_kv_heads head_dim"],
        global_mask_info: MaskInfo | None = None,
        local_mask_info: MaskInfo | None = None,
        cache_end_pos: Int[Array, "batch"] | None = None,  # noqa: F821
        softmax_scale: float | None = None,
        sliding_window: int = 128,
        overlap: bool = True,
        platform: str | None = None,
        **ignores,
    ) -> AttentionOutput:
        """Execute dual-stream attention over EasyDeL BTHD tensors.

        Args:
            query: Queries ``[batch, q_len, num_q_heads, head_dim]``.
            k_global: Global-stream keys ``[batch, kv_g_len, num_kv_heads, head_dim]``.
            v_global: Global-stream values ``[batch, kv_g_len, num_kv_heads, head_dim]``.
            k_local: Local-stream keys ``[batch, kv_l_len, num_kv_heads, head_dim]``.
            v_local: Local-stream values ``[batch, kv_l_len, num_kv_heads, head_dim]``.
            global_mask_info: Optional ``MaskInfo`` for the global KV stream.
            local_mask_info: Optional ``MaskInfo`` for the local KV stream.
            cache_end_pos: Optional ``[batch]`` cache write positions for cached decode.
            softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
            sliding_window: Width ``W`` of the local sliding window.
            overlap: If ``True``, global stream attends causally ``k <= q``; if
                ``False``, global attends disjointly ``k <= q - W``.
            platform: Optional per-call platform override (``"pallas"`` / ``"xla"``).

        Returns:
            :class:`AttentionOutput` with ``attention_outputs`` of shape
            ``[batch, q_len, num_q_heads, head_dim]`` in ``query.dtype``.
        """
        del ignores
        bsz, q_len, num_q_heads, head_dim = query.shape
        kv_g_len = k_global.shape[1]
        kv_l_len = k_local.shape[1]
        num_kv_heads = k_global.shape[2]
        out_dtype = query.dtype

        resolved_platform = resolve_platform(platform, configured_platform(self.metadata, self.get_impl_name()))
        if check_bool_flag("FORCE_NATIVE_RUNTIME", False):
            resolved_platform = "xla"

        op_cfg = self.metadata.get_operation_config("dual_stream_attention")
        cfg = op_cfg if isinstance(op_cfg, DualStreamAttentionConfig) else None

        # Transpose BTHD -> BHTD for ejkernel.
        q_bhtd = query.transpose(0, 2, 1, 3)
        kg_bhtd = k_global.transpose(0, 2, 1, 3)
        vg_bhtd = v_global.transpose(0, 2, 1, 3)
        kl_bhtd = k_local.transpose(0, 2, 1, 3)
        vl_bhtd = v_local.transpose(0, 2, 1, 3)

        # Fast segment-ID path for cacheless training (avoids materializing O(S^2) masks).
        can_use_seg_path = (
            cache_end_pos is None
            and kv_g_len == q_len
            and kv_l_len == q_len
            and (global_mask_info is None or getattr(global_mask_info, "_attention_mask", None) is None)
            and (local_mask_info is None or getattr(local_mask_info, "_attention_mask", None) is None)
        )

        g_q_seg = g_kv_seg = l_q_seg = l_kv_seg = None
        global_mask = local_mask = None
        if can_use_seg_path:
            if global_mask_info is not None:
                gq, gkv = global_mask_info.get_or_compute_segment_ids()
            else:
                gq = gkv = None
            if local_mask_info is not None:
                lq, lkv = local_mask_info.get_or_compute_segment_ids()
            else:
                lq, lkv = gq, gkv
            segs_ok = all(s is None or s.ndim == 2 for s in (gq, gkv, lq, lkv))
            if segs_ok:
                g_q_seg, g_kv_seg, l_q_seg, l_kv_seg = gq, gkv, lq, lkv
            else:
                can_use_seg_path = False

        if not can_use_seg_path:
            global_mask, local_mask = _build_explicit_stream_masks(
                bsz=bsz,
                q_len=q_len,
                kv_g_len=kv_g_len,
                kv_l_len=kv_l_len,
                global_mask_info=global_mask_info,
                local_mask_info=local_mask_info,
                cache_end_pos=cache_end_pos,
                sliding_window=int(sliding_window),
                overlap=bool(overlap),
            )

        pallas_eligible = (
            can_use_seg_path
            and q_len >= 128
            and q_len % 128 == 0
            and head_dim % 128 == 0
            and num_q_heads % num_kv_heads == 0
        )
        num_seg_operands = sum(s is not None for s in (g_q_seg, g_kv_seg, l_q_seg, l_kv_seg))
        sharding_kwargs = self._resolve_shard_map_kwargs(
            q_bhtd=q_bhtd,
            bsz=bsz,
            q_len=q_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_seg_operands=num_seg_operands,
            resolved_platform=resolved_platform,
            pallas_eligible=pallas_eligible,
        )

        out_bhtd = _ejkernel_dual_stream_attention(
            q_bhtd,
            kg_bhtd,
            vg_bhtd,
            kl_bhtd,
            vl_bhtd,
            g_q_seg,
            g_kv_seg,
            l_q_seg,
            l_kv_seg,
            global_mask,
            local_mask,
            softmax_scale=softmax_scale,
            sliding_window=int(sliding_window),
            overlap=bool(overlap),
            platform=resolved_platform,
            cfg=cfg,
            **sharding_kwargs,
        )
        return AttentionOutput(
            attention_weights=None,
            attention_outputs=out_bhtd.transpose(0, 2, 1, 3).astype(out_dtype),
        )

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward pass (Pallas SplashAttention); delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass; delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass (XLA fallback); delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA forward pass; delegates to :meth:`forward_gpu`."""
        return self.forward_gpu(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm forward pass; delegates to :meth:`forward_gpu`."""
        return self.forward_gpu(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch q_len num_q_heads head_dim"],
        k_global: Float[Array, "batch kv_g_len num_kv_heads head_dim"],
        v_global: Float[Array, "batch kv_g_len num_kv_heads head_dim"],
        k_local: Float[Array, "batch kv_l_len num_kv_heads head_dim"],
        v_local: Float[Array, "batch kv_l_len num_kv_heads head_dim"],
        global_mask_info: MaskInfo | None = None,
        local_mask_info: MaskInfo | None = None,
        cache_end_pos: Int[Array, "batch"] | None = None,  # noqa: F821
        softmax_scale: float | None = None,
        sliding_window: int = 128,
        overlap: bool = True,
        platform: str | None = None,
        **ignores,
    ) -> AttentionOutput:
        """Dispatch to the backend-specific dual-stream attention forward."""
        return super().__call__(
            query=query,
            k_global=k_global,
            v_global=v_global,
            k_local=k_local,
            v_local=v_local,
            global_mask_info=global_mask_info,
            local_mask_info=local_mask_info,
            cache_end_pos=cache_end_pos,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            overlap=overlap,
            platform=platform,
            **ignores,
        )
