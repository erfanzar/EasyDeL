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

"""EasyDeL adapter for the per-query gather DSA attention (opt-in).

Thin ``OperationImpl`` over the ejkernel ``dsa_gather_attention`` public operation: absorbed-MLA queries
``[B, N, H, D]`` (query-major) attend over a shared ``K == V`` latent ``[B, S, D]``, each query to exactly its
selected pools (``indices [B, N, K]``, pool ``p`` = keys ``p*kpool .. p*kpool + kpool - 1``) plus its own incomplete
pool. On TPU the Pallas kernel DMA-gathers only those pools, so its cost does not grow with how spread the selection is.

Layout on a multi-device TPU mesh: the Mosaic call runs under ``shard_map`` with the query sequence split over the axes
that would otherwise split heads (and over the query-length axes), every head local, and the latent replicated over
those axes. Splitting the sequence instead of the heads keeps the matmuls ``H``-wide (more MXU rows filled) and divides
the per-chip gather traffic by the split. :func:`gather_attention_layout` exposes the same layout so a model can move
the split onto a cheaper tensor before the call (see ``DeepSeekSparseAttention._forward_dsa_pool_gather``).

Tiles come from the explicit ``cfg`` argument, else from the ``dsa_gather_attention`` entry of
``OperationMetadata.operation_configs`` (a ``DsaGatherAttentionConfig`` or a dict of its fields).
"""

from __future__ import annotations

import dataclasses
import math

import jax
from ejkernel.modules import DsaGatherAttentionConfig
from ejkernel.modules import dsa_gather_attention as _ejkernel_dsa_gather_attention
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float, Int
from spectrax import common_types

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder
from ._compressed_window_common import configured_platform, resolve_platform

_CFG_FIELDS = tuple(DsaGatherAttentionConfig.__dataclass_fields__)


def _resolve_cfg(metadata, impl_name: str, cfg) -> DsaGatherAttentionConfig | None:
    """The explicit ``cfg``, else the metadata's operation config for ``impl_name``, else ``None`` (defaults)."""
    if cfg is None:
        cfg = metadata.get_operation_config(impl_name)
    if cfg is None or isinstance(cfg, DsaGatherAttentionConfig):
        return cfg
    if isinstance(cfg, dict):
        return DsaGatherAttentionConfig(**{k: v for k, v in cfg.items() if k in _CFG_FIELDS})
    return DsaGatherAttentionConfig(**{k: getattr(cfg, k) for k in _CFG_FIELDS if hasattr(cfg, k)})


def _names(entry) -> tuple[str, ...]:
    """Mesh-axis names of one PartitionSpec entry, in order."""
    if entry is None:
        return ()
    if isinstance(entry, tuple | list):
        return tuple(str(name) for name in entry)
    return (str(entry),)


def _entry(names: tuple[str, ...]):
    """PartitionSpec entry for ``names`` (``None`` when empty)."""
    if not names:
        return None
    return names[0] if len(names) == 1 else tuple(names)


@dataclasses.dataclass(frozen=True)
class GatherAttentionLayout:
    """Placement of the gather DSA call on a mesh.

    Attributes:
        mesh: The JAX mesh.
        rows: Axes splitting the batch.
        seq: Axes splitting the query sequence (empty: the sequence is not split and ``heads`` split the heads).
        heads: Axes the model splits attention heads over (the head-parallel layout outside the call).
    """

    mesh: jax.sharding.Mesh
    rows: tuple[str, ...]
    seq: tuple[str, ...]
    heads: tuple[str, ...]

    @property
    def query_spec(self) -> Ps:
        """Spec of a ``[B, N, H, D]`` tensor inside the call."""
        if self.seq:
            return Ps(_entry(self.rows), _entry(self.seq), None, None)
        return Ps(_entry(self.rows), None, _entry(self.heads), None)

    @property
    def head_parallel_spec(self) -> Ps:
        """Spec of a ``[B, N, H, D]`` tensor in the model's usual head-parallel layout."""
        return Ps(_entry(self.rows), None, _entry(self.heads), None)

    @property
    def query_row_entries(self) -> tuple:
        """``(batch_entry, query_entry)`` PartitionSpec entries of the call's query rows.

        A caller computing per-query work before the call (e.g. the indexer's scoring and top-k) can split its rows the
        same way instead of repeating that work on every device of the sequence axes.
        """
        return _entry(self.rows), _entry(self.seq)

    def constrain(self, x: Array, spec: Ps) -> Array:
        """``with_sharding_constraint`` of ``x`` to ``spec`` on this mesh."""
        return jax.lax.with_sharding_constraint(x, NamedSharding(self.mesh, spec))


def gather_attention_layout(
    metadata,
    *,
    platform: str | None,
    batch: int,
    seq: int,
    num_heads: int,
) -> GatherAttentionLayout | None:
    """The ``shard_map`` layout of the gather DSA call, or ``None`` when the call is not wrapped.

    The call is wrapped only when it resolves to the TPU Mosaic kernel (TPU backend, platform not forced to
    ``"xla"``) on a mesh of more than one device. The query sequence is split over the resolved ``QUERY_LENGTH`` and
    ``HEAD`` axes (minus the batch axes) when it divides evenly; otherwise heads are split as usual.

    Args:
        metadata: The adapter's :class:`~easydel.operations.OperationMetadata`.
        platform: Resolved platform override (``None`` = auto).
        batch: Global batch size.
        seq: Global query count.
        num_heads: Global head count.

    Returns:
        The layout, or ``None``.
    """
    if platform == "xla" or jax.default_backend() != "tpu":
        return None
    mesh = metadata.mesh
    jax_mesh = getattr(mesh, "jax_mesh", mesh)
    if jax_mesh is None or getattr(jax_mesh, "size", 1) <= 1:
        return None
    resolver = metadata.runtime_sharding_resolver.with_mesh(mesh)
    mode = common_types.MODE_TRAIN
    rows = _names(resolver.resolve(axes=[common_types.BATCH], mode=mode, shape=(batch,))[0])
    heads = _names(resolver.resolve(axes=[common_types.HEAD], mode=mode, shape=(num_heads,))[0])
    qlen = _names(resolver.resolve(axes=[common_types.QUERY_LENGTH], mode=mode, shape=(seq,))[0])
    heads = tuple(a for a in heads if a not in rows)
    sizes = dict(jax_mesh.shape)
    seq_axes = tuple(dict.fromkeys(a for a in (*qlen, *heads) if a not in rows and int(sizes.get(a, 1)) > 1))
    if not seq_axes or seq % math.prod(int(sizes[a]) for a in seq_axes):
        seq_axes = ()
    return GatherAttentionLayout(mesh=jax_mesh, rows=rows, seq=seq_axes, heads=heads)


@OperationRegistry.register
class DsaGatherAttn(OperationImpl):
    """Per-query gather DSA attention (Pallas TPU kernel, XLA reference elsewhere).

    Cache-agnostic training / prefill primitive: the caller passes the full-sequence latent and the selected pool ids.
    """

    @classmethod
    def get_impl_name(cls) -> str:
        """Return the operation registry key."""
        return "dsa_gather_attention"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.PREFILL) -> OperationRequirements:
        """No metadata fields and no engine-managed cache view (the caller materialises latent and selection).

        Args:
            mode: Execution mode (ignored).

        Returns:
            OperationRequirements: The declared requirements.
        """
        del mode
        return (
            RequirementsBuilder("dsa_gather_attention").require_metadata(MetadataField.NONE).no_cache_required().build()
        )

    def layout(self, *, batch: int, seq: int, num_heads: int, platform: str | None = None):
        """The call's mesh layout for these sizes (see :func:`gather_attention_layout`), or ``None``."""
        resolved = resolve_platform(platform, configured_platform(self.metadata, self.get_impl_name()))
        return gather_attention_layout(self.metadata, platform=resolved, batch=batch, seq=seq, num_heads=num_heads)

    @jax.named_scope("easydel-dsa_gather_attention")
    def forward_native(
        self,
        query: Float[Array, "batch seq num_heads head_dim"],
        latent: Float[Array, "batch kv_seq head_dim"],
        indices: Int[Array, "batch seq num_selected"],
        positions: Int[Array, "batch seq"] | None = None,
        *,
        kpool: int,
        softmax_scale: float | None = None,
        include_tail: bool = True,
        platform: str | None = None,
        cfg: DsaGatherAttentionConfig | dict | None = None,
        **ignores,
    ) -> AttentionOutput:
        """Run the gather attention through the ejkernel operation.

        Args:
            query: ``[B, N, H, D]`` queries (query-major).
            latent: ``[B, S, D]`` shared ``K == V`` latent.
            indices: ``[B, N, K]`` int selected pool ids (``-1`` = none).
            positions: ``[B, N]`` absolute query positions (negative = invalid query, output 0); ``None`` =
                ``0 .. N - 1``.
            kpool: Keys per pool.
            softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
            include_tail: Open each query's own incomplete pool up to the query.
            platform: Per-call platform override (``"pallas"`` / ``"xla"`` / ``None`` = the operation config's).
            cfg: Optional ``DsaGatherAttentionConfig`` (or dict) overriding the metadata's operation config.
            **ignores: Ignored keyword arguments (interface parity).

        Returns:
            AttentionOutput: ``attention_outputs`` ``[B, N, H, D]``; no weights.
        """
        del ignores
        runtime_dtype = self.metadata.runtime_dtype
        resolved = resolve_platform(platform, configured_platform(self.metadata, self.get_impl_name()))
        bsz, n, h, _ = query.shape
        layout = gather_attention_layout(self.metadata, platform=resolved, batch=bsz, seq=n, num_heads=h)
        sharding = {}
        if layout is not None:
            if positions is None:
                positions = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32)[None, :], (bsz, n))
            rows, seq = _entry(layout.rows), _entry(layout.seq)
            q_spec = layout.query_spec
            sharding = dict(
                mesh=layout.mesh,
                in_specs=(q_spec, Ps(rows, None, None), Ps(rows, seq, None), Ps(rows, seq)),
                out_specs=q_spec,
            )
        out = _ejkernel_dsa_gather_attention(
            query.astype(runtime_dtype),
            latent.astype(runtime_dtype),
            indices.astype(jnp.int32),
            None if positions is None else positions.astype(jnp.int32),
            kpool=int(kpool),
            softmax_scale=softmax_scale,
            include_tail=bool(include_tail),
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
        query: Float[Array, "batch seq num_heads head_dim"],
        latent: Float[Array, "batch kv_seq head_dim"],
        indices: Int[Array, "batch seq num_selected"],
        positions: Int[Array, "batch seq"] | None = None,
        **kwargs,
    ) -> AttentionOutput:
        """Dispatch to the backend-specific forward (see :meth:`forward_native` for the arguments).

        Returns:
            AttentionOutput: The attention result.
        """
        return super().__call__(query, latent, indices, positions, **kwargs)


__all__ = ("DsaGatherAttn", "GatherAttentionLayout", "gather_attention_layout")
