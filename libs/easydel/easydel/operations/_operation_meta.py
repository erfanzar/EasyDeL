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

"""Metadata and sharding-rule definitions for EasyDeL attention operations.

This module exposes:

* :class:`AttnShardingRules` - a ``NamedTuple`` of ``PartitionSpec`` entries
  for every tensor that flows through an attention kernel (query, key, value,
  bias, mask, output, segment ids, optional softmax auxiliary output).
* :class:`OperationMetadata` - the runtime configuration object passed to
  every concrete :class:`~easydel.operations.OperationImpl`. It centralises
  dtype, mesh, sharding policy, backend/platform selection, and per-operation
  ejkernel configuration.

These classes are consumed by the kernels under
``easydel/operations/kernels/`` and by the higher-level attention dispatch
machinery in ``easydel.layers``.
"""

from __future__ import annotations

import dataclasses
import enum
import typing as tp
from typing import NamedTuple

import jax
import jaxtyping
import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from spectrax import PartitionAxis, common_types

from easydel.infra.sharding import (
    AxisPolicy,
    MeshLike,
    RuntimeShardingResolver,
    StageMesh,
    coerce_runtime_sharding_resolver,
    resolve_stage_mesh,
)

if tp.TYPE_CHECKING:
    from ejkernel.modules.operations.configs import BaseOperationConfig  # pyright: ignore[reportMissingTypeStubs]

    from easydel.infra.base_config import EasyDeLBaseConfig
    from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
else:
    EasyDeLPlatforms = enum.Enum | str
    EasyDeLBackends = enum.Enum | str
    EasyDeLBaseConfig = object
    BaseOperationConfig = object

__all__ = ["AttnShardingRules", "OperationMetadata"]

logger = get_logger("EasyDeL-OperationOperator")
NOT_GIVEN = common_types.NOT_GIVEN
RUNTIME_MODE_TYPES = common_types.RUNTIME_MODE_TYPES
BATCH = common_types.BATCH
QUERY_LENGTH = common_types.QUERY_LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
HEAD_DIM = common_types.HEAD_DIM
KV_HEAD_DIM = common_types.KV_HEAD_DIM
BIAS_HEAD_SEQ = common_types.BIAS_HEAD_SEQ
BIAS_KV_SEQ = common_types.BIAS_KV_SEQ
EMPTY = common_types.EMPTY


class AttnShardingRules(NamedTuple):
    """Bundle of :class:`PartitionSpec` entries for every attention tensor.

    Produced by :meth:`OperationMetadata.get_shardings` and consumed by the
    attention kernels to ``shard_map`` / ``with_sharding_constraint`` the
    intermediate tensors that flow through the operation.

    Attributes:
        query3d (PartitionSpec): Sharding for the 3D query layout
            ``(batch, head, head_dim)`` used by some decode kernels.
        query (PartitionSpec): Sharding for the 4D query tensor.
        key (PartitionSpec): Sharding for the 4D key tensor.
        value (PartitionSpec): Sharding for the 4D value tensor.
        bias (PartitionSpec): Sharding for an additive attention bias.
        mask (PartitionSpec): Sharding for a boolean attention mask.
        output (PartitionSpec): Sharding for the operation's output tensor;
            mirrors ``query`` by construction.
        q_segment_ids (PartitionSpec): Sharding for the query segment ids
            used in packed-sequence training.
        kv_segment_ids (PartitionSpec): Sharding for the key/value segment
            ids used in packed-sequence training.
        softmax_aux (PartitionSpec | None): Sharding for auxiliary softmax
            outputs (e.g. log-sum-exp, row max). ``None`` when the kernel
            does not expose them.
    """

    query3d: jax.sharding.PartitionSpec
    query: jax.sharding.PartitionSpec
    key: jax.sharding.PartitionSpec
    value: jax.sharding.PartitionSpec
    bias: jax.sharding.PartitionSpec
    mask: jax.sharding.PartitionSpec
    output: jax.sharding.PartitionSpec
    q_segment_ids: jax.sharding.PartitionSpec
    kv_segment_ids: jax.sharding.PartitionSpec
    softmax_aux: jax.sharding.PartitionSpec | None


@auto_pytree
class OperationMetadata:
    """Runtime configuration shared across all attention operations.

    Carries the dtype, mesh, sharding policy, backend/platform selection,
    and per-operation ejkernel configs that every concrete
    :class:`~easydel.operations.OperationImpl` consults at dispatch time.
    Designed to be built once per model (commonly via :meth:`from_config`)
    and then handed to every attention operator instance in that model so
    they share identical runtime behaviour.

    The ``@auto_pytree`` decoration makes instances usable directly inside
    ``jax.jit`` / ``jax.vmap`` argument trees.

    Attributes:
        runtime_dtype (jax.typing.DTypeLike): Primary compute dtype for the
            attention QKV path. Defaults to ``float32`` when not pulled from
            ``base_config.attn_dtype``.
        runtime_softmax_dtype (jax.typing.DTypeLike | None): Dtype for the
            softmax normalization, typically promoted (e.g. ``float32``) for
            numerical stability.
        sequence_axis_name (str): Mesh-axis name used to shard the sequence
            dimension under pjit/shard_map; defaults to ``"sp"``.
        platform (EasyDeLPlatforms): Target hardware platform tag, used by
            kernel implementations to gate platform-specific code paths.
        backend (EasyDeLBackends | None): Concrete JAX backend the operation
            should dispatch into; consumed by
            :meth:`BaseOperation.__call__`.
        axis_policy (AxisPolicy | PartitionAxis): Semantic sharding policy
            used to resolve tensor PartitionSpecs.
        partition_axis (PartitionAxis): Per-axis partition specification,
            kept in sync with ``axis_policy``.
        runtime_sharding_resolver (RuntimeShardingResolver): Resolver that
            translates semantic axis names into concrete PartitionSpecs;
            built from ``axis_policy`` if not supplied.
        base_config (EasyDeLBaseConfig | None): Optional reference to the
            owning model config; lets ``mesh`` and other lookups follow the
            config at runtime.
        operation_configs (dict[str, BaseOperationConfig] | None): Optional
            mapping from operation name to its ejkernel
            :class:`BaseOperationConfig`. Consumed by
            :meth:`get_operation_config`.
        requires_cache (bool | None): Instance-level override for the
            operation's KV-cache requirement. ``None`` defers to the
            operation's class default; ``False`` disables caching (vision
            encoders); ``True`` forces caching.
        _stored_mesh (MeshLike | None): Fallback mesh used when
            ``base_config`` is ``None``. Resolved through the :attr:`mesh`
            property.
    """

    runtime_dtype: jax.typing.DTypeLike
    runtime_softmax_dtype: jax.typing.DTypeLike | None = None

    sequence_axis_name: str = NOT_GIVEN
    platform: EasyDeLPlatforms = NOT_GIVEN
    backend: EasyDeLBackends | None = NOT_GIVEN

    axis_policy: AxisPolicy | PartitionAxis = NOT_GIVEN
    partition_axis: PartitionAxis = NOT_GIVEN
    runtime_sharding_resolver: RuntimeShardingResolver = NOT_GIVEN

    base_config: EasyDeLBaseConfig | None = None
    operation_configs: dict[str, BaseOperationConfig] | None = None

    # Instance-level override for cache requirements.
    # None means use the operation's class-level default.
    # False disables cache (useful for encoder-only models like vision encoders).
    # True forces cache requirement.
    requires_cache: bool | None = None

    _stored_mesh: MeshLike | None = NOT_GIVEN

    def __post_init__(self) -> None:
        """Fill in defaults, infer the mesh/backend, and run safety checks.

        Walks every field that was left as ``NOT_GIVEN`` and resolves it
        from either ``base_config`` (when available) or a hard-coded
        default. Then:

        * Coerces ``partition_axis`` from a dict if needed and rebuilds
          ``axis_policy`` so the two stay in sync.
        * Picks up the JAX mesh from ``spectrax.get_incontext_mesh`` or the
          thread-local pxla resources when no ``base_config`` provides one.
        * Constructs the :class:`RuntimeShardingResolver` from
          ``axis_policy`` if the user did not supply one.
        * Calls :meth:`_safety_check` to ensure no essential attribute is
          still ``NOT_GIVEN``.
        * Resolves ``backend`` to an :class:`EasyDeLBackends` enum value.

        Raises:
            ValueError: If no mesh can be resolved (no ``base_config``, no
                in-context mesh, and no thread-local mesh).
        """

        from easydel.infra.etils import EasyDeLBackends

        # fmt:off
        self.set_attrs_carefully("runtime_dtype",  jnp.float32, "attn_dtype")
        self.set_attrs_carefully("runtime_softmax_dtype", jnp.float32, "attn_softmax_dtype")
        self.set_attrs_carefully("partition_axis", PartitionAxis())
        if isinstance(self.partition_axis, dict):
            self.partition_axis = PartitionAxis(**self.partition_axis)
        self.set_attrs_carefully("axis_policy", AxisPolicy.from_partition_axis(self.partition_axis))
        self.axis_policy = AxisPolicy.from_partition_axis(self.axis_policy)
        self.partition_axis = self.axis_policy.to_partition_axis()
        # DON'T READ FROM CONFIG
        self.set_attrs_carefully("sequence_axis_name", "sp", "sequence_axis_name", use_base_config=False)
        self.set_attrs_carefully("backend", jax.default_backend(), "backend")
        self.set_attrs_carefully("platform", NOT_GIVEN, "platform")
        self.set_attrs_carefully("_stored_mesh", NOT_GIVEN, "mesh")
        self.set_attrs_carefully("operation_configs", None, "operation_configs")
        # fmt:on
        if self._stored_mesh is NOT_GIVEN and self.base_config is None:
            mesh: MeshLike | None = spx.get_incontext_mesh(raise_error=False)
            if mesh is None:
                mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
            if mesh is None or getattr(mesh, "empty", False):
                raise ValueError(
                    "You should pass 'mesh' to `OperationMetadata` or at least create that under mesh context manager"
                )
            self._stored_mesh = mesh
        self.runtime_sharding_resolver = coerce_runtime_sharding_resolver(
            self.runtime_sharding_resolver if self.runtime_sharding_resolver is not NOT_GIVEN else self.axis_policy,
            mesh=self.mesh,
        )
        self._safety_check()
        if self.backend is None:
            current_backend: str = jax.default_backend()
            backend_enum: EasyDeLBackends = getattr(
                EasyDeLBackends, current_backend, getattr(EasyDeLBackends, current_backend.upper())
            )
            self.backend = backend_enum

    def _safety_check(self) -> None:
        """Verify that every dataclass field has been resolved away from ``NOT_GIVEN``.

        Raises:
            ValueError: If any field still holds the sentinel
                ``common_types.NOT_GIVEN``; the error message includes the
                first offending field name.
        """
        field: dataclasses.Field
        for field in dataclasses.fields(self):
            val: tp.Any = getattr(self, field.name)
            if val is NOT_GIVEN:
                raise ValueError(f"`{field.name}` shouldn't be ellipsis")

    @classmethod
    def from_config(cls, config: EasyDeLBaseConfig) -> OperationMetadata:
        """Factory method to create OperationMetadata from an EasyDeLBaseConfig.

        Pulls dtype, sharding, platform/backend, and operation-config defaults
        directly off the model config so all attention operators inside the
        same model share identical runtime behaviour.

        Args:
            config: The base configuration object (e.g. a model config). The
                following attributes are consulted: ``attn_dtype``,
                ``attn_softmax_dtype``, ``sequence_axis_name``, ``platform``,
                ``backend``, ``axis_policy``, and the optional
                ``operation_configs``.

        Returns:
            OperationMetadata: An initialized instance whose ``base_config``
            field references ``config`` for further on-demand lookups.
        """
        return cls(
            runtime_dtype=config.attn_dtype,
            runtime_softmax_dtype=config.attn_softmax_dtype,
            sequence_axis_name=config.sequence_axis_name,
            platform=config.platform,
            backend=config.backend,
            axis_policy=config.axis_policy,
            base_config=config,
            operation_configs=getattr(config, "operation_configs", None),
        )

    @property
    def mesh(self) -> StageMesh:
        """Resolved JAX mesh used by attention kernels.

        Returns:
            StageMesh: The mesh attached to ``base_config`` if it is set,
            otherwise the mesh that was supplied at construction time and
            stored in ``_stored_mesh``. Returns ``None`` when no mesh is
            available.
        """
        if self.base_config is not None:
            mesh = self.base_config.mesh
        else:
            mesh = self._stored_mesh
        if mesh is None or mesh is NOT_GIVEN:
            return None
        return resolve_stage_mesh(mesh)

    @mesh.setter
    def mesh(self, value: MeshLike | None):
        """Override the stored mesh.

        Args:
            value: The new mesh to associate with this metadata. Used when no
                ``base_config`` is available to source the mesh from.
        """
        self._stored_mesh = value

    def get_shardings(
        self,
        mode: RUNTIME_MODE_TYPES,  # type:ignore
        layout: tp.Literal["bthd", "bhtd", "thd"] = "bthd",
        qkv_mni_sharding: bool = False,
        softmax_aux: jaxtyping.Array | None = None,
    ) -> AttnShardingRules:
        """Resolve PartitionSpecs for every attention tensor.

        Uses ``self.runtime_sharding_resolver`` bound to ``self.mesh`` to
        translate semantic axis names (``BATCH``, ``QUERY_LENGTH``, ...)
        into concrete :class:`PartitionSpec` objects honouring the active
        execution mode and tensor layout.

        Args:
            mode: Runtime mode (training, decode, etc.) used to pick the
                mode-specific axis assignments inside the resolver.
            layout: Tensor layout for Q/K/V:

                * ``"bthd"`` — ``(batch, time, heads, dim)``.
                * ``"bhtd"`` — ``(batch, heads, time, dim)``.

                ``"thd"`` is reserved for future use.
            qkv_mni_sharding: If True, use the ``HEAD``/``HEAD_DIM`` axes
                for the K/V tensors instead of ``KV_HEAD``/``KV_HEAD_DIM``
                — applicable for MHA where K and V share the query head
                count rather than the smaller KV head count.
            softmax_aux: Optional auxiliary softmax tensor (e.g. LSE or
                row-max). When supplied, a sharding for it is added to the
                returned rules (2D inputs get an ``[EMPTY, KV_HEAD]``
                spec, higher rank inputs get an ``[HEAD]`` spec).

        Returns:
            AttnShardingRules: A populated NamedTuple of PartitionSpecs for
            every attention tensor (queries, keys, values, bias, mask,
            output, segment ids, optional softmax-aux).

        Raises:
            NotImplementedError: If ``layout`` is not one of the supported
                values above.
        """

        resolver = self.runtime_sharding_resolver.with_mesh(self.mesh)

        _h: common_types.DynamicShardingAxes = HEAD if qkv_mni_sharding else KV_HEAD
        _kvh: common_types.DynamicShardingAxes = HEAD_DIM if qkv_mni_sharding else KV_HEAD_DIM

        q_sharding: jax.sharding.PartitionSpec
        k_sharding: jax.sharding.PartitionSpec
        v_sharding: jax.sharding.PartitionSpec
        q_segment_ids_sharding: jax.sharding.PartitionSpec
        kv_segment_ids_sharding: jax.sharding.PartitionSpec

        if layout == "bthd":
            q_sharding = resolver.resolve(axes=[BATCH, QUERY_LENGTH, HEAD, HEAD_DIM], mode=mode)
            k_sharding = resolver.resolve(axes=[BATCH, KV_LENGTH, _h, _kvh], mode=mode)
            v_sharding = resolver.resolve(axes=[BATCH, KV_LENGTH, _h, _kvh], mode=mode)
            q_segment_ids_sharding = resolver.resolve(axes=[BATCH, QUERY_LENGTH], mode=mode)
            kv_segment_ids_sharding = resolver.resolve(axes=[BATCH, KV_LENGTH], mode=mode)
        elif layout == "bhtd":
            q_sharding = resolver.resolve(axes=[BATCH, HEAD, QUERY_LENGTH, HEAD_DIM], mode=mode)
            k_sharding = resolver.resolve(axes=[BATCH, _h, KV_LENGTH, _kvh], mode=mode)
            v_sharding = resolver.resolve(axes=[BATCH, _h, KV_LENGTH, _kvh], mode=mode)
            q_segment_ids_sharding = resolver.resolve(axes=[BATCH, QUERY_LENGTH], mode=mode)
            kv_segment_ids_sharding = resolver.resolve(axes=[BATCH, KV_LENGTH], mode=mode)
        else:
            raise NotImplementedError(f"Layout '{layout}' is not implemented")

        qk_extern: tuple[common_types.DynamicShardingAxes, common_types.DynamicShardingAxes] = (
            QUERY_LENGTH,
            BIAS_KV_SEQ,
        )

        b_sharding: jax.sharding.PartitionSpec = resolver.resolve(axes=[BATCH, BIAS_HEAD_SEQ, *qk_extern], mode=mode)
        m_sharding: jax.sharding.PartitionSpec = resolver.resolve(axes=[BATCH, None, *qk_extern], mode=mode)

        # Softmax auxiliary output sharding (e.g., LSE, max) - 2D: [batch, num_heads]
        softmax_aux_sharding: jax.sharding.PartitionSpec | None = None
        if softmax_aux is not None:
            num_dims: int = softmax_aux.ndim
            if num_dims == 2:
                softmax_aux_sharding = resolver.resolve(axes=[EMPTY, KV_HEAD], mode=mode)
            else:
                softmax_aux_sharding = resolver.resolve(axes=[HEAD], mode=mode)

        query3d_sharding: jax.sharding.PartitionSpec = resolver.resolve(axes=[BATCH, HEAD, HEAD_DIM], mode=mode)
        rules: AttnShardingRules = AttnShardingRules(
            query3d=query3d_sharding,
            query=q_sharding,
            key=k_sharding,
            value=v_sharding,
            bias=b_sharding,
            mask=m_sharding,
            output=q_sharding,
            q_segment_ids=q_segment_ids_sharding,
            kv_segment_ids=kv_segment_ids_sharding,
            softmax_aux=softmax_aux_sharding,
        )
        return rules

    def set_attrs_carefully(
        self,
        attr_name: str,
        default: tp.Any | None,
        pickup_name: str | None = None,
        use_base_config: bool = True,
    ) -> None:
        """Resolve an attribute that may still hold the ``NOT_GIVEN`` sentinel.

        If ``self.<attr_name>`` is missing or equal to
        :data:`common_types.NOT_GIVEN`, the value is filled in by reading
        ``self.base_config.<pickup_name>`` (when ``use_base_config`` is
        True and a config is attached) and finally falling back to
        ``default``. Used pervasively from :meth:`__post_init__` to wire
        defaults without overwriting user-supplied values.

        Args:
            attr_name: Name of the attribute to set on ``self``.
            default: Value used when no config-sourced value is available.
            pickup_name: Attribute name to read from ``self.base_config``;
                defaults to ``attr_name``.
            use_base_config: When ``False``, skip the config lookup entirely
                and use ``default``.
        """
        has_attr: bool = hasattr(self, attr_name)
        current_val: tp.Any = getattr(self, attr_name, NOT_GIVEN)
        if not has_attr or current_val is NOT_GIVEN:
            pn: str = attr_name if pickup_name is None else pickup_name
            should_use_default: bool = self.base_config is None or not use_base_config
            new_value: tp.Any = default if should_use_default else getattr(self.base_config, pn, default)
            setattr(self, attr_name, new_value)

    def get_operation_config(self, impl_name: str) -> "BaseOperationConfig | None":
        """Get ejkernel config for a specific operation by its registered name.

        Args:
            impl_name: The operation implementation name (must match OperationRegistry).
                Valid names:
                - "flash_attn2": Flash attention 2 implementation
                - "ring": Ring attention
                - "blocksparse": Block sparse attention
                - "ragged_page_attention_v2": Ragged page attention v2
                - "ragged_page_attention_v3": Ragged page attention v3
                - "multi_latent_ragged_page_attention_v1": Multi-latent ragged page attention v1
                - "unified_attention": Unified paged attention (serving-style)
                - "paged_flash_attention": Paged FlashAttention (CUDA, block tables)
                - "sdpa": Scaled dot product attention
                - "vanilla": Vanilla attention

        Returns:
            The operation config if set, otherwise None (which enables ejkernel autotune).

        Example:
            >>> cfg = metadata.get_operation_config("flash_attn2")
            >>> if cfg is not None:
            ...     # Use explicit config
            ...     flash_attention(..., cfg=cfg)
            >>> else:
            ...     # Use autotune
            ...     flash_attention(..., cfg=None)
        """
        if self.operation_configs is None:
            return None
        return self.operation_configs.get(impl_name)
