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

"""EasyDeL adapter for the pool-bitmap block-sparse DSA attention (opt-in).

Thin ``OperationImpl`` over the ejkernel ``dsa_sparse_attention`` public operation: absorbed-MLA queries
``[B, H, S, D]`` attend over a shared ``K == V`` latent ``[B, S, D]`` restricted to the indexer's selection, given
as a per-query pool mask ``[B, S, P]`` (pool ``p`` = keys ``p*kpool .. p*kpool + kpool - 1``, anchored at key 0) plus
up to 8 tail keys per query. On TPU the Pallas kernel skips every (q-block, kv-block) pair the selection does not
touch and never builds an ``O(S^2)`` tensor; elsewhere the dense XLA reference runs. ``jax.grad`` flows to the
query, the latent and the pooled straight-through term ``ste_pool`` (the indexer's gradient).

This path is opt-in: nothing calls it by default, and the existing ``sparse_selected_training_attention`` helper in
:mod:`.dsa_sparse_attention` is unchanged. Tiles and per-pair cost variants come from the
``dsa_pool_sparse_attention`` entry of ``OperationMetadata.operation_configs`` (a
``DsaSparseAttentionConfig`` or a dict of its fields), or from the explicit ``cfg`` argument.
"""

from __future__ import annotations

import jax
from ejkernel.modules import DsaSparseAttentionConfig
from ejkernel.modules import dsa_sparse_attention as _ejkernel_dsa_sparse_attention
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Bool, Float, Int
from spectrax import common_types

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder
from ._compressed_window_common import configured_platform, head_parallel_shard_map, resolve_platform

_CFG_FIELDS = tuple(DsaSparseAttentionConfig.__dataclass_fields__)


def _resolve_cfg(metadata, impl_name: str, cfg) -> DsaSparseAttentionConfig | None:
    """The explicit ``cfg``, else the metadata's operation config for ``impl_name``, else ``None`` (defaults)."""
    if cfg is None:
        cfg = metadata.get_operation_config(impl_name)
    if cfg is None or isinstance(cfg, DsaSparseAttentionConfig):
        return cfg
    if isinstance(cfg, dict):
        return DsaSparseAttentionConfig(**{k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return DsaSparseAttentionConfig(**{k: getattr(cfg, k) for k in _CFG_FIELDS if hasattr(cfg, k)})


@OperationRegistry.register
class DsaPoolSparseAttn(OperationImpl):
    """Pool-bitmap block-sparse DSA attention (Pallas TPU kernel, XLA reference elsewhere).

    Cache-agnostic training / prefill primitive: the caller passes the full-sequence latent and the selection.
    """

    @classmethod
    def get_impl_name(cls) -> str:
        """Return the operation registry key."""
        return "dsa_pool_sparse_attention"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.PREFILL) -> OperationRequirements:
        """Basic metadata, no engine-managed cache view (the caller materialises latent and selection).

        Args:
            mode: Execution mode (prefill / training).

        Returns:
            OperationRequirements: The declared requirements.
        """
        del mode
        return (
            RequirementsBuilder("dsa_pool_sparse_attention")
            .require_metadata(MetadataField.NONE)
            .no_cache_required()
            .build()
        )

    @jax.named_scope("easydel-dsa_pool_sparse_attention")
    def forward_native(
        self,
        query: Float[Array, "batch num_heads seq head_dim"],
        latent: Float[Array, "batch seq head_dim"],
        pool_mask: Bool[Array, "batch seq num_pools"],
        tails: Int[Array, "batch seq num_tails"] | None = None,
        ste_pool: Float[Array, "batch seq num_pools"] | None = None,
        *,
        kpool: int,
        softmax_scale: float | None = None,
        mode: str = "sparse",
        platform: str | None = None,
        cfg: DsaSparseAttentionConfig | dict | None = None,
        **ignores,
    ) -> AttentionOutput:
        """Run the block-sparse attention through the ejkernel operation.

        Casts query / latent to the runtime dtype (``ste_pool`` stays fp32). When the op resolves to the TPU Mosaic
        kernel on a multi-device mesh it runs under a batch / head-parallel ``shard_map`` (the latent and the
        selection replicated over heads, the sequence axis unsharded); the XLA reference is left to GSPMD.

        Args:
            query: ``[B, H, S, D]`` queries.
            latent: ``[B, S, D]`` shared ``K == V`` latent.
            pool_mask: ``[B, S, P]`` bool selected pools.
            tails: ``[B, S, T]`` int32 tail keys (``-1`` = none, ``T <= 8``) or ``None``.
            ste_pool: ``[B, S, P]`` pooled straight-through term (value 0) or ``None``.
            kpool: Keys per pool.
            softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
            mode: ``"sparse"``, ``"dense"`` or ``"budget:B"`` (a numerics change).
            platform: Per-call platform override (``"pallas"`` / ``"xla"`` / ``None`` = the operation config's).
            cfg: Optional ``DsaSparseAttentionConfig`` (or dict) overriding the metadata's operation config.
            **ignores: Ignored keyword arguments (interface parity).

        Returns:
            AttentionOutput: ``attention_outputs`` ``[B, H, S, D]``; no weights.
        """
        del ignores
        runtime_dtype = self.metadata.runtime_dtype
        resolved = resolve_platform(platform, configured_platform(self.metadata, self.get_impl_name()))
        sharding = head_parallel_shard_map(
            self.metadata,
            mode=common_types.MODE_TRAIN,
            platform=resolved,
            batch=int(query.shape[0]),
            num_heads=int(query.shape[1]),
        )
        if sharding:
            q_spec, lat_spec = sharding["in_specs"][0], sharding["in_specs"][1]
            rows = Ps(lat_spec[0], None, None)
            extra = tuple(rows for value in (tails, ste_pool) if value is not None)
            sharding = dict(mesh=sharding["mesh"], in_specs=(q_spec, lat_spec, rows, *extra), out_specs=q_spec)
        out = _ejkernel_dsa_sparse_attention(
            query.astype(runtime_dtype),
            latent.astype(runtime_dtype),
            pool_mask if pool_mask.dtype == jnp.bool_ else pool_mask.astype(jnp.int32),
            None if tails is None else tails.astype(jnp.int32),
            None if ste_pool is None else ste_pool.astype(jnp.float32),
            kpool=int(kpool),
            softmax_scale=softmax_scale,
            mode=mode,
            platform=resolved,
            cfg=_resolve_cfg(self.metadata, self.get_impl_name(), cfg),
            **sharding,
        )
        return AttentionOutput(attention_weights=None, attention_outputs=out)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward pass (Pallas kernel); delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass (XLA reference); delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass (XLA reference); delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA forward pass; delegates to :meth:`forward_gpu`."""
        return self.forward_gpu(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm forward pass; delegates to :meth:`forward_gpu`."""
        return self.forward_gpu(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch num_heads seq head_dim"],
        latent: Float[Array, "batch seq head_dim"],
        pool_mask: Bool[Array, "batch seq num_pools"],
        tails: Int[Array, "batch seq num_tails"] | None = None,
        ste_pool: Float[Array, "batch seq num_pools"] | None = None,
        **kwargs,
    ) -> AttentionOutput:
        """Dispatch to the backend-specific forward (see :meth:`forward_native` for the arguments).

        Returns:
            AttentionOutput: The attention result.
        """
        return super().__call__(query, latent, pool_mask, tails, ste_pool, **kwargs)


__all__ = ("DsaPoolSparseAttn",)
