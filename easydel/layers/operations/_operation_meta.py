from __future__ import annotations

import dataclasses
import typing as tp
from typing import NamedTuple

import jax
import jaxtyping
from eformer import common_types
from eformer.escale import PartitionAxis, PartitionManager
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from jax import numpy as jnp

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
from easydel.layers.moe import EMPTY

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


class AttnShardingRules(NamedTuple):
    """
    Named tuple containing JAX PartitionSpecs for all attention tensors.

    Attributes:
        query: Sharding for query tensor.
        key: Sharding for key tensor.
        value: Sharding for value tensor.
        bias: Sharding for attention bias tensor.
        mask: Sharding for attention mask tensor.
        output: Sharding for attention output tensor.
        q_segment_ids: Sharding for query segment IDs (for packed sequences).
        kv_segment_ids: Sharding for key/value segment IDs (for packed sequences).
        softmax_aux: Optional sharding for 2D softmax auxiliary outputs (e.g., LSE, max).
    """

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
    """
    Holds configuration, context, and metadata for attention operations.

    This class centralizes various parameters needed by different attention
    implementations, facilitating consistent behavior and configuration. It handles
    default values and can be initialized from an `EasyDeLBaseConfig`.

    Attributes:
        runtime_dtype: The primary JAX dtype for computations (e.g., q, k, v).
        runtime_softmax_dtype: Optional JAX dtype for the softmax computation,
            allowing for higher precision if needed (e.g., float32).
        sequence_axis_name: The name used for the sequence axis in JAX parallelism
            (sharding_axis_names for pjit).
        mesh: The JAX device mesh for distributed computation. Must be provided
            or inferred from context.
        platform: The target hardware platform (e.g., TPU, GPU).
        backend: The specific JAX backend being used (e.g., TPU, CUDA, ROCM).
        partition_axis: Configuration for partitioning axes in distributed settings.
            (Likely from `eformer.escale`).
        base_config: An optional reference to the base model configuration object
            for sourcing default values.
        scan_ring_attention: Boolean flag indicating whether to use ring attention
            via `jax.lax.scan`.
        softmax_scale: The scaling factor applied before the softmax operation.
            Often `1 / sqrt(head_dim)`.
        dropout_prob: The dropout probability applied to attention weights.
        blocksize_q: Block size for the query sequence dimension in blockwise attention.
        blocksize_k: Block size for the key/value sequence dimension in blockwise attention.
        blocksize_b: Block size for the batch dimension in blockwise attention (often 1).
    """

    runtime_dtype: jax.typing.DTypeLike
    runtime_softmax_dtype: jax.typing.DTypeLike | None = None

    sequence_axis_name: str = NOT_GIVEN
    platform: EasyDeLPlatforms = NOT_GIVEN
    backend: EasyDeLBackends = NOT_GIVEN

    partition_axis: PartitionAxis = NOT_GIVEN
    partition_manager: PartitionManager = NOT_GIVEN

    base_config: EasyDeLBaseConfig | None = None

    _stored_mesh: jax.sharding.Mesh | None = NOT_GIVEN

    def __post_init__(self):
        """
        Initializes default values and performs safety checks after dataclass creation.

        Sets reasonable defaults for various parameters if they are not provided
        (or marked as Ellipsis). It attempts to source defaults from the `base_config`
        if available. It also infers the JAX mesh and backend if not explicitly given.
        Finally, it performs a safety check to ensure no essential attributes remain
        uninitialized (as Ellipsis).
        """
        # fmt:off
        self.set_attrs_carefully("runtime_dtype",  jnp.float32, "attn_dtype")
        self.set_attrs_carefully("runtime_softmax_dtype", jnp.float32, "attn_softmax_dtype")
        self.set_attrs_carefully("partition_axis", PartitionAxis())
        self.set_attrs_carefully("partition_manager", PartitionManager(self.partition_axis))
        # DON'T READ FROM CONFIG
        self.set_attrs_carefully("sequence_axis_name", "sp", "sequence_axis_name", use_base_config=False)
        self.set_attrs_carefully("backend", jax.default_backend(), "backend")
        self.set_attrs_carefully("platform", NOT_GIVEN, "platform")
        self.set_attrs_carefully("_stored_mesh", NOT_GIVEN, "mesh")
        # fmt:on
        if self._stored_mesh is NOT_GIVEN and self.base_config is None:
            mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
            assert (
                not mesh.empty
            ), "You should pass 'mesh' to `OperationMetadata` or at least create that under mesh context manager"
            self._stored_mesh = mesh
        self._safety_check()
        if self.backend is None:
            current_backend = jax.default_backend()
            self.backend = getattr(EasyDeLBackends, current_backend, getattr(EasyDeLBackends, current_backend.upper()))

    def _safety_check(self):
        """Ensures no essential attributes are left uninitialized (as NOT_GIVEN)."""
        for field in dataclasses.fields(self):
            val = getattr(self, field.name)
            if val is NOT_GIVEN:
                raise ValueError(f"`{field.name}` shouldn't be ellipsis")

    @classmethod
    def from_config(cls, config: EasyDeLBaseConfig) -> OperationMetadata:
        """
        Factory method to create OperationMetadata from an EasyDeLBaseConfig.

        Args:
            config: The base configuration object (e.g., model config).
            softmax_scale: The attention softmax scaling factor. Usually calculated
                based on head dimension.
            dropout_prob: The attention dropout probability. Defaults to 0.0.

        Returns:
            An initialized OperationMetadata instance.
        """
        return cls(
            runtime_dtype=config.attn_dtype,
            runtime_softmax_dtype=config.attn_softmax_dtype,
            sequence_axis_name=config.sequence_axis_name,
            platform=config.platform,
            backend=config.backend,
            partition_axis=config.partition_axis,
            base_config=config,
        )

    @property
    def mesh(self) -> jax.sharding.Mesh | None:
        """Get current mesh from base_config if available, otherwise return stored mesh."""
        if self.base_config is not None:
            return self.base_config.mesh
        return self._stored_mesh

    @mesh.setter
    def mesh(self, value: jax.sharding.Mesh | None):
        """Set mesh value for cases where base_config is not available."""
        self._stored_mesh = value

    def get_shardings(
        self,
        mode: RUNTIME_MODE_TYPES,  # type:ignore
        layout: tp.Literal["bthd", "bhtd", "thd"] = "bthd",
        qkv_mni_sharding: bool = False,
        softmax_aux: jaxtyping.Array | None = None,
    ) -> AttnShardingRules:
        """
        Generates JAX PartitionSpecs for attention tensors based on runtime mode.

        Args:
            mode: Runtime mode (e.g., training, inference) for partition resolution.
            layout: Tensor layout format - "bthd" (batch, time, heads, dim) or
                "bhtd" (batch, heads, time, dim).
            qkv_mni_sharding: If True, use HEAD/HEAD_DIM for K/V instead of KV_HEAD/KV_HEAD_DIM.
                Useful for multi-head attention (MHA) vs grouped-query attention (GQA/MQA).
            softmax_aux_2d: If True, create sharding for 2D softmax auxiliary outputs
                (e.g., log-sum-exp, max values) with shape [batch, num_heads].

        Returns:
            AttnShardingRules: Named tuple containing PartitionSpecs for all attention tensors:
                - query, key, value: Main attention tensors
                - bias: Attention bias tensor
                - mask: Attention mask tensor
                - output: Attention output tensor
                - q_segment_ids: Query segment IDs (for packed sequences)
                - kv_segment_ids: Key/value segment IDs (for packed sequences)
                - softmax_aux: Optional 2D softmax auxiliary output sharding
        """

        pama = self.partition_manager

        _h = HEAD if qkv_mni_sharding else KV_HEAD
        _kvh = HEAD_DIM if qkv_mni_sharding else KV_HEAD_DIM

        if layout == "bthd":
            q_sharding = pama.resolve(axes=[BATCH, QUERY_LENGTH, HEAD, HEAD_DIM], mode=mode)
            k_sharding = pama.resolve(axes=[BATCH, KV_LENGTH, _h, _kvh], mode=mode)
            v_sharding = pama.resolve(axes=[BATCH, KV_LENGTH, _h, _kvh], mode=mode)
            q_segment_ids_sharding = pama.resolve(axes=[BATCH, QUERY_LENGTH], mode=mode)
            kv_segment_ids_sharding = pama.resolve(axes=[BATCH, KV_LENGTH], mode=mode)
        elif layout == "bhtd":
            q_sharding = pama.resolve(axes=[BATCH, HEAD, QUERY_LENGTH, HEAD_DIM], mode=mode)
            k_sharding = pama.resolve(axes=[BATCH, _h, KV_LENGTH, _kvh], mode=mode)
            v_sharding = pama.resolve(axes=[BATCH, _h, KV_LENGTH, _kvh], mode=mode)
            q_segment_ids_sharding = pama.resolve(axes=[BATCH, QUERY_LENGTH], mode=mode)
            kv_segment_ids_sharding = pama.resolve(axes=[BATCH, KV_LENGTH], mode=mode)
        else:
            raise NotImplementedError(f"Layout '{layout}' is not implemented")

        qk_extern = (QUERY_LENGTH, BIAS_KV_SEQ)

        b_sharding = pama.resolve(axes=[BATCH, BIAS_HEAD_SEQ, *qk_extern], mode=mode)
        m_sharding = pama.resolve(axes=[BATCH, None, *qk_extern], mode=mode)

        # Softmax auxiliary output sharding (e.g., LSE, max) - 2D: [batch, num_heads]
        softmax_aux_sharding = None
        if softmax_aux is not None:
            if softmax_aux.ndim == 2:
                softmax_aux_sharding = pama.resolve(axes=[EMPTY, KV_HEAD], mode=mode)
            else:
                softmax_aux_sharding = pama.resolve(axes=[HEAD], mode=mode)

        return AttnShardingRules(
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

    def set_attrs_carefully(
        self,
        attr_name: str,
        default: tp.Any | None,
        pickup_name: str | None = None,
        use_base_config: bool = True,
    ):
        """
        Internal helper to set an attribute if it's not already set (or is Ellipsis).

        Optionally retrieves the value from `self.base_config` using `pickup_name`
        (or `attr_name` if `pickup_name` is None).

        Args:
            attr_name: The name of the attribute to set on `self`.
            default: The default value to use if not found in `base_config` or
                if `use_base_config` is False.
            pickup_name: The name of the attribute to look for in `base_config`.
                Defaults to `attr_name`.
            use_base_config: Whether to attempt retrieving the value from `base_config`.
        """
        if not hasattr(self, attr_name) or getattr(self, attr_name, NOT_GIVEN) is NOT_GIVEN:
            pn = attr_name if pickup_name is None else pickup_name
            new_value = (
                default if (self.base_config is None or not use_base_config) else getattr(self.base_config, pn, default)
            )
            setattr(self, attr_name, new_value)
