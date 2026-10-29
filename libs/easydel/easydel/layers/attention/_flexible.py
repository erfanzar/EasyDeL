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

"""Flexible attention module for various attention mechanisms.

Provides a unified interface for different attention implementations,
automatically selecting the optimal mechanism based on hardware and
configuration. Supports Flash Attention, Ring Attention, Splash Attention,
and other optimized implementations.

Classes:
    AttentionMechanisms: Enum of available attention mechanisms
    FlexibleAttentionModule: Main attention module with automatic optimization

Functions:
    tpu_version_check: Check TPU version for optimization
    get_optimal_config: Determine best attention mechanism for hardware
    _get_jax_dtype_from_string: Convert string to JAX dtype

Module-level constants:
    DEFAULT_ATTENTION_MECHANISM: Default attention mechanism (``"auto"``).
    Cfg: Type variable bound to :class:`EasyDeLBaseConfig`, used by
        :class:`AttentionModule` so that subclasses can preserve their
        precise config type.

Example:
    >>> from easydel.layers.attention import FlexibleAttentionModule
    >>> attn = FlexibleAttentionModule(
    ...     config=config,
    ...     dtype=jnp.bfloat16,
    ...     attention_mechanism="flash_attn2"
    ... )
    >>> output = attn(
    ...     query, key, value,
    ...     attention_mask=mask
    ... )
"""

import typing as tp
from enum import StrEnum
from functools import cached_property, partial

import einops
import jax
import spectrax as spx
from chex import Array  # pyright: ignore[reportMissingTypeStubs]
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import NamedSharding, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.sharding import PartitionSpec
from jaxtyping import Array as JArray
from jaxtyping import Bool, Complex, Float, Int
from spectrax import apply_logical_sharding, common_types

from easydel.caching import (
    OperationsMetadata,
    ParallelHybridCacheView,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCacheView,
    TransformerMetadata,
    UnifiedAttentionCacheView,
    unwrap_metadata,
)
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.sharding import StageMesh, inference_mode_forces_decode, resolve_stage_mesh
from easydel.infra.utils import AttnMaskDetail, AttnMaskType
from easydel.operations import AttentionOutput, OperationMetadata, OperationRegistry, ScaledDotProductAttn
from easydel.operations.requirements import ExecutionMode

from ..quantization import EasyQuantizer, TurboQuantConfig

logger = get_logger(__name__)


def _layer_cache_metadata(cache_metadata: tp.Any, cache_view: tp.Any) -> tp.Any:
    """Select the complete paged runtime payload for this layer before writes or reads."""
    attention_view = cache_view.transformer if isinstance(cache_view, ParallelHybridCacheView) else cache_view
    if isinstance(attention_view, (RaggedPagesCacheView, UnifiedAttentionCacheView)):
        metadata = unwrap_metadata(cache_metadata, "ragged")
        if metadata is not None and hasattr(metadata, "for_layer"):
            return metadata.for_layer(attention_view.layer_index)
    return cache_metadata


def _attention_mesh_context(config: EasyDeLBaseConfig):
    """Resolve the active stage mesh as a context manager.

    Forwards to :func:`easydel.infra.sharding.resolve_stage_mesh` so that
    inside MPMD pipeline stages the attention call binds to the local
    stage submesh rather than the global model mesh.

    Args:
        config: Owning model config. Its ``mesh`` attribute is the only
            field consulted.

    Returns:
        Context manager activating the stage-local mesh; entering it is
        a no-op when no MPMD stage is active.
    """
    return resolve_stage_mesh(config.mesh)


def _get_jax_dtype_from_string(dtype_string: str) -> jnp.dtype | str:
    """Convert string representation to JAX dtype.

    Parses string representations of JAX dtypes and returns
    the corresponding dtype object.

    Args:
        dtype_string: String representation of JAX dtype
            (e.g., "<class 'jax.numpy.float32'>").

    Returns:
        JAX dtype object if recognized, otherwise the original string.

    Example:
        >>> dtype = _get_jax_dtype_from_string("<class 'jax.numpy.float32'>")
        >>> dtype == jnp.float32
        True
    """
    dtype_mapping: dict[str, jnp.dtype] = {
        "<class 'jax.numpy.bfloat16'>": jnp.bfloat16,
        "<class 'jax.numpy.float16'>": jnp.float16,
        "<class 'jax.numpy.float32'>": jnp.float32,
        "<class 'jax.numpy.float64'>": jnp.float64,
        "<class 'jax.numpy.int32'>": jnp.int32,
        "<class 'jax.numpy.int64'>": jnp.int64,
        "<class 'jax.numpy.bool_'>": jnp.bool_,
        "<class 'jax.numpy.complex64'>": jnp.complex64,
        "<class 'jax.numpy.complex128'>": jnp.complex128,
    }
    result: jnp.dtype | str = dtype_mapping.get(dtype_string, dtype_string)
    if isinstance(result, str):
        # Also accept bare dtype names ("bfloat16", "float16", ...) and any other ``<class 'jax.numpy.X'>`` spelling.
        bare_name = result.removeprefix("<class 'jax.numpy.").removesuffix("'>")
        try:
            return jnp.dtype(bare_name)
        except TypeError:
            return result
    return result


class AttentionMechanisms(StrEnum):
    """Available attention mechanism implementations.

    Enumeration of different attention computation strategies,
    each optimized for specific hardware or use cases.

    Attributes:
        AUTO: Automatically selects best mechanism for hardware.
        FLASH_ATTN2: FlashAttention-2 for efficient GPU computation.
        RING: RingAttention for sequence parallelism.
        VANILLA: Standard dot-product attention.
        SPLASH: SplashAttention optimized for TPUs.
        CUDNN: cuDNN implementation for NVIDIA GPUs.
        BLOCKWISE: Blockwise computation for memory efficiency.
        SDPA: Scaled Dot Product Attention (JAX native).
        CUDA_FLASH_ATTN2: CUDA-specific FlashAttention-2.
        RAGGED_PAGE_ATTENTION_V3: Paged attention for efficient inference.
        RAGGED_PAGE_ATTENTION_V2: Paged attention for efficient inference.
        MULTI_LATENT_RAGGED_PAGE_ATTENTION_V1: MLA ragged page attention for
            compressed-KV inference.
        UNIFIED_ATTENTION: serving-style unified paged attention (Triton).
        PAGED_FLASH_ATTENTION: FlashAttention with paged KV cache (CUDA).
        REGRESSIVE_DECODE: Optimized autoregressive decoding.
    """

    AUTO: str = "auto"
    FLASH_ATTN2: str = "flash_attn2"
    RING: str = "ring"
    VANILLA: str = "vanilla"
    SPLASH: str = "blocksparse"
    BLOCKSPARSE: str = "blocksparse"
    CUDNN: str = "cudnn"
    BLOCKWISE: str = "blockwise"
    SDPA: str = "sdpa"
    CUDA_FLASH_ATTN2: str = "cuda_flash_attn2"
    RAGGED_PAGE_ATTENTION_V3: str = "ragged_page_attention_v3"
    RAGGED_PAGE_ATTENTION_V2: str = "ragged_page_attention_v2"
    MULTI_LATENT_RAGGED_PAGE_ATTENTION_V1: str = "multi_latent_ragged_page_attention_v1"
    MULTI_LATENT_RAGGED_PAGE_ATTENTION_V2: str = "multi_latent_ragged_page_attention_v2"
    PAGED_ATTENTION: str = "page_attention"
    UNIFIED_ATTENTION: str = "unified_attention"
    PAGED_FLASH_ATTENTION: str = "paged_flash_attention"
    REGRESSIVE_DECODE: str = "autoregressive_decodeattn"


def tpu_version_check(version: str = "v4") -> bool:
    """Check if running on specified TPU version.

    Verifies if the current JAX device matches the specified
    TPU version for hardware-specific optimizations.

    Args:
        version: TPU version string to check (e.g., "v4", "v5").
                Defaults to "v4".

    Returns:
        True if running on specified TPU version, False otherwise.

    Example:
        >>> if tpu_version_check("v5"):
        ...     # Use TPU v5 optimizations
        ...     pass
    """
    first_device: tp.Any = jax.local_devices()[0]
    device_kind: str = getattr(first_device, "device_kind", "")
    device_kind_lower: str = device_kind.lower()
    version_matches: bool = version in device_kind_lower
    if version_matches:
        return True

    return False


def get_optimal_config() -> tuple[AttentionMechanisms, jnp.dtype]:
    """Determine optimal attention configuration for hardware.

    Analyzes the current JAX backend and hardware to recommend
    the best attention mechanism and data type for performance.

    Returns:
        Tuple of (attention_mechanism, dtype) optimized for current hardware:
        - TPU v3: (FLASH_ATTN2, float32)
        - TPU v4+: (SPLASH, bfloat16)
        - GPU: (FLASH_ATTN2, float16)
        - CPU/other: (VANILLA, bfloat16)

    Example:
        >>> mechanism, dtype = get_optimal_config()
        >>> attn = FlexibleAttentionModule(
        ...     attention_mechanism=mechanism,
        ...     dtype=dtype
        ... )
    """

    current_backend: str = jax.default_backend()
    match current_backend:
        case "tpu":
            is_tpu_v3: bool = tpu_version_check("v3")
            if is_tpu_v3:
                if jax.process_count() > 1:
                    logger.warning_once(
                        "FLASH_ATTN2 on multi-host TPU v3 may compile non-identical XLA programs "
                        "across hosts and can fail at runtime; falling back to VANILLA attention."
                    )
                    result_mechanism: AttentionMechanisms = AttentionMechanisms.VANILLA
                    result_dtype: jnp.dtype = jnp.bfloat16
                    return result_mechanism, result_dtype
                result_mechanism = AttentionMechanisms.FLASH_ATTN2
                result_dtype: jnp.dtype = jnp.float32
                return result_mechanism, result_dtype
            result_mechanism_v4: AttentionMechanisms = AttentionMechanisms.BLOCKSPARSE
            result_dtype_v4: jnp.dtype = jnp.bfloat16
            return result_mechanism_v4, result_dtype_v4
        case "gpu":
            gpu_mechanism: AttentionMechanisms = AttentionMechanisms.SDPA
            gpu_dtype: jnp.dtype = jnp.bfloat16
            return (gpu_mechanism, gpu_dtype)
        case _:
            fallback_mechanism: AttentionMechanisms = AttentionMechanisms.VANILLA
            fallback_dtype: jnp.dtype = jnp.bfloat16
            return fallback_mechanism, fallback_dtype


DEFAULT_ATTENTION_MECHANISM = "auto"


class FlexibleAttentionModule(spx.Module):
    """Backend-agnostic attention dispatcher (training + paged-cache decode).

    The role of this class is *routing*, not attention math: every actual
    attention kernel — vanilla dot-product, FlashAttention 2, SplashAttention,
    RingAttention, ragged-page paged attention, MLA paged attention,
    UnifiedAttention (Triton), etc. — is implemented by an
    :class:`Operation` registered in :class:`OperationRegistry`. This module
    holds two such operations (``impl`` for prefill / training and
    ``impl_decode`` for the optional decode-only kernel), validates that
    the supplied ``cache_view`` matches the chosen backend, and forwards
    QKV plus the full bag of optional knobs (mask, RoPE-applied bias,
    sliding-window, soft-caps, dropout, …) to the right kernel.

    QKV layout and shape conventions (used throughout the operation registry):

    * ``query_states``: ``[batch, seq_q, num_q_heads, head_dim]``
    * ``key_states``:   ``[batch, seq_k, num_kv_heads, head_dim]``
    * ``value_states``: ``[batch, seq_v, num_kv_heads, head_dim_v]``

    GQA / MQA are honoured by the kernels, not by this layer. RoPE is
    applied *before* the call (the ``frequencies`` plumbing lives one level
    up in :class:`UnifiedAttention`); ALiBi is supplied via the ``bias``
    argument. Causal vs. bidirectional and sliding-window are controlled
    by the ``causal`` and ``sliding_window`` keyword arguments respectively
    — they are passed through to the kernel rather than realised as a mask.

    Multi-host / TPU specialization: the constructor short-circuits the
    ``"auto"`` mechanism via :func:`get_optimal_config`, and the forward
    pass contains a fallback that re-routes variable-length VANILLA
    attention through :class:`ScaledDotProductAttn` when running on a
    multi-host TPU mesh — a workaround for VANILLA's inability to honour
    ``cum_seqlens_*`` under TPU's MPMD scheduling.

    Attributes:
        config (EasyDeLBaseConfig): Owning model config; consulted for
            attention dtypes, mesh, and (per-layer-overridable) mechanism.
        metadata (OperationMetadata): Frozen metadata pytree built once by
            :meth:`OperationMetadata.from_config`; backends key their
            kernel selection / autotuning off this object.
        softmax_scale (float): Scale applied to the QK^T product before
            softmax. Conventionally ``1 / sqrt(head_dim)`` but may differ
            for muP / DeepSeek MLA (``softmax_scale * mscale``).
        dropout_prob (float): Attention-weight dropout probability. Applied
            in non-deterministic passes using an explicit key or constructor
            RNGs. TRAIN mode follows the module's training state by default;
            prefill and decode remain deterministic.
        impl (Operation): Prefill / training attention backend instance,
            chosen from :class:`AttentionMechanisms`.
        impl_decode (Operation | None): Optional separate backend for the
            decode-only path (e.g. ``REGRESSIVE_DECODE``); ``None`` when
            the same kernel handles both phases.
        deterministic (bool | None): Override for ``forward(deterministic=None)``.
            ``None`` follows training state in TRAIN mode and disables dropout
            in prefill/decode; an explicit boolean overrides this policy.
        _requires_cache (bool | None): Override for the operation's
            class-level cache requirement; consulted by the
            :class:`OperationExecutor` and :attr:`requires_cache`.
    """

    def __init__(
        self,
        base_config: EasyDeLBaseConfig,
        softmax_scale: float,
        dropout_prob: float = 0.0,
        *,
        rngs: spx.Rngs | None = None,
        attn_mechanism: AttentionMechanisms | None = None,
        requires_cache: bool | None = None,
    ):
        """Resolve the attention backend(s) and bind them to ``base_config``.

        On construction the module reads ``base_config.attn_mechanism``
        (or the explicit ``attn_mechanism`` override) and builds a single
        :class:`Operation` instance from :class:`OperationRegistry`. When
        the mechanism is ``"auto"``, :func:`get_optimal_config` is consulted
        and both the chosen mechanism and the resolved ``attn_dtype`` are
        written back into ``base_config`` so downstream sharding/cache code
        observes the same choice. If ``base_config.decode_attn_mechanism``
        is set, a second backend is created for the decode phase.

        Args:
            base_config: Configuration object carrying attention settings
                (mechanism, dtype, mesh, sharding, optional decode backend).
            softmax_scale: Pre-softmax scale applied to QK^T (conventionally
                ``1 / sqrt(head_dim)``; muP / DeepSeek MLA override).
            dropout_prob: Attention dropout probability. Active during TRAIN
                mode when the module is training, or when explicitly requested
                with ``deterministic=False``.
            rngs: Optional SpecTrax RNG container supplying fresh dropout keys
                when a per-call ``dropout_rng`` is not provided.
            attn_mechanism: Optional per-instance override of the backend
                chosen from :class:`AttentionMechanisms`. ``None`` defers to
                ``base_config.attn_mechanism``.
            requires_cache: Override for the backend's cache requirement.

                - ``None``: use the operation's class-level default.
                - ``False``: disable cache (encoder-only paths such as
                  vision encoders).
                - ``True``: force cache requirement on.

        Raises:
            TypeError: If a string attention or softmax dtype is not recognized.
        """

        if attn_mechanism is None:
            attn_mechanism = base_config.attn_mechanism

        attn_dtype_is_string: bool = isinstance(base_config.attn_dtype, str)
        if attn_dtype_is_string:
            base_config.attn_dtype = _get_jax_dtype_from_string(base_config.attn_dtype)
            # Fail at the public configuration boundary, not inside a compiled kernel.
            jnp.dtype(base_config.attn_dtype)

        attn_softmax_dtype_is_string: bool = isinstance(base_config.attn_softmax_dtype, str)
        if attn_softmax_dtype_is_string:
            base_config.attn_softmax_dtype = _get_jax_dtype_from_string(base_config.attn_softmax_dtype)
            jnp.dtype(base_config.attn_softmax_dtype)

        is_auto_mechanism: bool = attn_mechanism == AttentionMechanisms.AUTO
        if is_auto_mechanism:
            impl_name: AttentionMechanisms
            runtime_dtype: jnp.dtype
            impl_name, runtime_dtype = get_optimal_config()
            logger.debug(f"Automatically select OperationImpl {impl_name} | {runtime_dtype}")
            attn_mechanism = impl_name
            base_config.attn_dtype = runtime_dtype

        metadata: OperationMetadata = OperationMetadata.from_config(config=base_config)
        self.config: EasyDeLBaseConfig = base_config
        self.metadata: OperationMetadata = metadata
        self.softmax_scale: float = softmax_scale
        self.dropout_prob: float = dropout_prob
        self._requires_cache: bool | None = requires_cache
        impl_name_final: str = attn_mechanism
        self.impl: tp.Any = OperationRegistry.create(
            impl_name=impl_name_final,
            metadata=metadata,
            requires_cache=requires_cache,
        )
        self.deterministic: bool | None = None
        # Keep the RNG on the module graph so SpectraX transforms carry its advancing counter.
        if rngs is not None:
            self.rngs = rngs
        self.impl_decode: tp.Any | None = None
        has_decode_mechanism: bool = base_config.decode_attn_mechanism is not None
        if has_decode_mechanism:
            decode_impl_name: str = base_config.decode_attn_mechanism
            self.impl_decode = OperationRegistry.create(
                impl_name=decode_impl_name,
                metadata=metadata,
                requires_cache=requires_cache,
            )

    @jax.named_scope("easydel-flexible-attention")
    def forward(
        self,
        query_states: Float[JArray, "batch seq_q heads dim"],
        key_states: Float[JArray, "batch seq_k heads dim"],
        value_states: Float[JArray, "batch seq_v heads dim"],
        mode: common_types.RUNTIME_MODE_TYPES | None,  # type:ignore
        mask_info: MaskInfo | None = None,
        bias: Float[JArray, "batch heads seq_q seq_k"] | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None,
        init_bias: tp.Callable[[], Float[JArray, "batch heads seq_q seq_k"]] | None = None,
        causal: bool = True,
        softmax_aux: Float[JArray, "..."] | None = None,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        dropout_prob: float | None = None,
        dropout_rng: tp.Any | None = None,
        deterministic: bool | None = None,
        output_attentions: bool | None = None,
        precision: lax.PrecisionLike | None = None,
        prevent_cse: bool = True,
        cum_seqlens_q: Int[JArray, "batch_plus_one"] | None = None,  # noqa
        cum_seqlens_k: Int[JArray, "batch_plus_one"] | None = None,  # noqa
        normalize_output: bool = True,
        fused_backward: bool = False,
        compute_dtype: jnp.dtype | None = None,
        optimized: bool = False,
        mask_value: float | None = None,
        vmem_limit_bytes: int | None = None,
        policy: tp.Any | None = None,
        **extra_op_kwargs: tp.Any,
    ) -> AttentionOutput:
        """Dispatch the attention computation to the resolved backend.

        Validates ``cache_view`` against the active backend, fills in
        defaults from ``self``/``self.config``, routes around a known
        multi-host-TPU bug in vanilla variable-length attention, calls the
        chosen :class:`Operation`, and casts the leaf arrays of the
        returned :class:`AttentionOutput` to ``self.impl.metadata.runtime_dtype``.
        ``cache_view`` is *not* cast — quantized cache pages (e.g.
        TurboQuant uint8) must keep their original dtype.

        Args:
            query_states: Query tensor ``[batch, seq_q, heads, dim]``.
            key_states: Key tensor ``[batch, seq_k, heads, dim]``.
            value_states: Value tensor ``[batch, seq_v, heads, dim]``.
            mode: Runtime mode (TRAIN, PREFILL, DECODE); selects between
                ``self.impl`` and ``self.impl_decode``.
            mask_info: Container with the attention mask plus per-token
                segment IDs and positions; ``None`` for full visibility.
            bias: Additive attention bias ``[batch, heads, seq_q, seq_k]``.
            sliding_window: Local-window size (int ``W`` = HF window size, i.e.
                ``(W - 1, W - 1)``; tuple for inclusive ``(left, right)``).
            cache_metadata: Companion metadata for the cache view (page
                tables, cumulative lengths). Auto-derived from ``cache_view``
                when ``None``.
            cache_view: KV cache view (transformer / ragged-pages / unified).
                Mutated in place by the backend in inference modes.
            init_bias: Optional zero-argument callable that materializes
                the additive bias on demand.
            causal: Apply causal masking. Defaults to ``True``.
            softmax_aux: Optional auxiliary tensor blended into softmax
                normaliser (e.g. attention sinks, learnable bias tokens).
            softmax_scale: Override for the pre-softmax scale. Defaults to
                ``self.softmax_scale`` when ``None``.
            logits_soft_cap: Optional soft-cap value
                (``tanh(logits / cap) * cap``).
            dropout_prob: Override for attention dropout probability. Defaults
                to ``self.dropout_prob`` when ``None``. Only takes effect when
                the pass is non-deterministic.
            dropout_rng: PRNG key for dropout. When absent, uses a fresh key
                from the constructor's ``rngs``; an active dropout pass requires
                one of these sources.
            deterministic: If ``True`` disables dropout. When ``None``, uses
                ``self.deterministic`` if set, otherwise follows ``self.training``
                in TRAIN mode and disables dropout in other modes. Active dropout
                uses the registered vanilla operation (or its paged-cache adapter),
                preserving the JAX key contract and complete cached context rather
                than passing it to seed-only or no-dropout kernels.
            output_attentions: When ``True`` instructs the backend to
                materialise softmax weights. Falls back to
                ``config.output_attentions`` when ``None``.
            precision: JAX matmul precision, forwarded to the vanilla backend
                only. There ``None`` inherits the ambient JAX precision and an
                explicit value (including ``DEFAULT``) is used as given.
            prevent_cse: Whether to prevent common-subexpression elimination
                inside the kernel.
            cum_seqlens_q: Optional cumulative sequence lengths for query
                packing (``[batch + 1]``).
            cum_seqlens_k: Optional cumulative sequence lengths for key
                packing (``[batch + 1]``).
            normalize_output: Normalize the attention output by the softmax
                denominator. Set ``False`` for log-sum-exp pathways.
            fused_backward: Use the fused backward pass when available.
            compute_dtype: Compute dtype override for the kernel.
            optimized: Use the optimized kernel variant when the backend
                exposes one.
            mask_value: Float value applied to masked positions (defaults
                to a backend-specific large negative number).
            vmem_limit_bytes: VMEM budget for paged attention on TPU.
            policy: Optional gradient-checkpoint policy forwarded to the
                kernel. Defaults to ``jax.checkpoint_policies.nothing_saveable``
                when ``None``.
            **extra_op_kwargs: Additional backend-specific keyword arguments
                forwarded verbatim to the underlying operation (used by
                MLA backends for ``queries_nope`` / ``keys_values`` / ...).

        Returns:
            :class:`AttentionOutput` carrying the attention output tensor,
            optional materialised attention weights (``None`` when
            ``output_attentions`` is False), and the updated cache view.

        Raises:
            ValueError: When ``cache_view`` is incompatible with the active
                backend (e.g. a ragged cache view paired with a dense kernel)
                or when ``mode == MODE_DECODE`` is requested without a
                ``cache_view``.
        """

        def _get_impl_names(impl: tp.Any) -> set[str]:
            """Return the registry name(s) of an attention-implementation object.

            Args:
                impl: Anything that may expose ``get_impl_name`` returning
                    a single string or a tuple of strings (typically a
                    concrete :class:`OperationImpl` instance).

            Returns:
                set[str]: Set of registered names, or the empty set when
                the object does not advertise an implementation name.
            """
            impl_name = getattr(impl, "get_impl_name", lambda: None)()
            if isinstance(impl_name, tuple):
                return {str(name) for name in impl_name}
            if impl_name is None:
                return set()
            return {str(impl_name)}

        is_decode_mode = mode == common_types.MODE_DECODE
        active_impl = self.impl_decode if is_decode_mode and self.impl_decode is not None else self.impl
        cache_metadata = _layer_cache_metadata(cache_metadata, cache_view)
        masking_details = getattr(cache_view, "masking_details", None)
        attention_view = cache_view.transformer if isinstance(cache_view, ParallelHybridCacheView) else cache_view
        is_paged = isinstance(attention_view, (RaggedPagesCacheView, UnifiedAttentionCacheView))
        if not is_paged and mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            # Concatenation may bake absolute-position visibility after callers
            # have chosen their kernel options. Dense backends must not apply a
            # second, chunk-local window to those aligned (possibly rolled) keys.
            # Paged kernels consume cache metadata instead of this dense mask.
            sliding_window = None
        if is_paged and masking_details is None:
            # Paged cache schemas do not store masks. Resolve the production config
            # using the attention component's layer index, not a synthetic cache field.
            get_mask_details = getattr(self.config, "get_mask_details", None)
            layer_masks = get_mask_details() if get_mask_details is not None else None
            if layer_masks is not None:
                masking_details = layer_masks.get(attention_view.layer_index)
        # Dense cache concatenation already bakes its layer mask into MaskInfo. Paged
        # kernels do not consume that mask, so they require explicit window/chunk options.
        if masking_details is not None and isinstance(attention_view, (RaggedPagesCacheView, UnifiedAttentionCacheView)):
            if masking_details.mask_type == AttnMaskType.SLIDING and sliding_window is None:
                sliding_window = masking_details.size
            elif masking_details.mask_type == AttnMaskType.CHUNK:
                configured_chunk_size = (
                    masking_details.chunks if masking_details.chunks is not None else masking_details.size
                )
                if configured_chunk_size is None or configured_chunk_size <= 0:
                    raise ValueError("Chunked local attention requires a positive configured chunk size.")
                chunk_size = extra_op_kwargs.get("chunk_size")
                if chunk_size is None:
                    chunk_size = configured_chunk_size
                if chunk_size != configured_chunk_size:
                    raise ValueError("chunk_size must match the layer's cache masking details.")
                extra_op_kwargs["chunk_size"] = chunk_size
        deterministic_computed = self.deterministic if deterministic is None else deterministic
        if deterministic_computed is None:
            # UnifiedAttention and custom model callers already forward the runtime mode.
            # Inference must remain deterministic even when a newly loaded module is still in train mode.
            deterministic_computed = mode != common_types.MODE_TRAIN or not self.training

        if output_attentions is None:
            output_attentions_computed = bool(getattr(self.config, "output_attentions", False))
        else:
            output_attentions_computed = output_attentions

        if softmax_scale is None:
            softmax_scale_computed = self.softmax_scale
        else:
            softmax_scale_computed = softmax_scale

        dropout_prob_final: float = 0.0
        dropout_rng_final: tp.Any | None = None
        if not deterministic_computed:
            dropout_prob_final = float(self.dropout_prob if dropout_prob is None else dropout_prob)
            if not 0.0 <= dropout_prob_final < 1.0:
                raise ValueError("dropout_prob must be in [0, 1).")
            if dropout_prob_final > 0.0:
                dropout_rng_final = dropout_rng
                if dropout_rng_final is None:
                    rngs = getattr(self, "rngs", None)
                    if rngs is not None:
                        dropout_rng_final = rngs.key("dropout")
                if dropout_rng_final is None:
                    raise ValueError(
                        "dropout_rng must be provided when deterministic=False and dropout_prob > 0, "
                        "or supply rngs at construction."
                    )

        if dropout_prob_final > 0.0:
            active_impl = OperationRegistry.create(
                impl_name="paged_attention_reference" if is_paged else AttentionMechanisms.VANILLA.value,
                metadata=self.metadata,
                requires_cache=self._requires_cache,
            )
        if extra_op_kwargs.get("chunk_size") is not None:
            if extra_op_kwargs["chunk_size"] <= 0:
                raise ValueError("chunk_size must be positive.")
            if not _get_impl_names(active_impl) & {
                AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3.value,
                "paged_attention_reference",
            }:
                raise NotImplementedError("Chunked local attention requires ragged_page_attention_v3.")
        if is_paged:
            # Validate the actual operation against the component passed to it,
            # including decode overrides and the key-aware dropout adapter.
            get_requirements = getattr(active_impl, "get_instance_requirements", None)
            if get_requirements is None:
                get_requirements = getattr(active_impl, "get_requirements", None)
            if get_requirements is not None:
                execution_mode = ExecutionMode.DECODE if is_decode_mode else ExecutionMode.PREFILL
                expected_view = get_requirements(execution_mode).cache.cache_view_class
                if expected_view is not None and not isinstance(attention_view, expected_view):
                    raise ValueError(
                        f"{type(attention_view).__name__} is incompatible with {_get_impl_names(active_impl)}; "
                        f"the operation requires {expected_view.__name__}."
                    )

        # Use provided precision or default
        if precision is None:
            precision_computed = lax.Precision.DEFAULT
        else:
            precision_computed = precision

        # Use provided policy or default
        if policy is None:
            policy_computed = jax.checkpoint_policies.nothing_saveable
        else:
            policy_computed = policy

        def _maybe_route_varlen_multihost_tpu_attention(callable_attn: tp.Any) -> tp.Any:
            """Reroute variable-length attention to SDPA on multi-host TPU.

            Vanilla attention does not implement
            ``cum_seqlens_q``/``cum_seqlens_k`` correctly on multi-host
            TPU; this helper substitutes :class:`ScaledDotProductAttn`
            when (a) the runtime is multi-host TPU, (b) variable-length
            metadata is present, and (c) ``callable_attn`` is the vanilla
            implementation. Validates that SDPA can preserve the requested
            feature set before rerouting.

            Args:
                callable_attn: The attention implementation that would
                    otherwise be used.

            Returns:
                tp.Any: ``callable_attn`` itself when no rerouting is
                needed, otherwise an :class:`ScaledDotProductAttn`
                instance constructed against ``self.metadata``.

            Raises:
                ValueError: When the request cannot be honoured by SDPA
                    (e.g. mismatched head dimensions or unsupported
                    softmax features).
            """
            # Dropout is routed to vanilla below with explicit length masks, never to SDPA
            # (which ignores dropout). This also preserves varlen semantics on multi-host meshes.
            if dropout_prob_final > 0.0:
                return callable_attn
            if jax.default_backend() != "tpu" or jax.process_count() <= 1:
                return callable_attn
            if cum_seqlens_q is None and cum_seqlens_k is None:
                return callable_attn
            impl_names = _get_impl_names(callable_attn)
            if AttentionMechanisms.VANILLA not in impl_names:
                return callable_attn
            if not (query_states.shape[-1] == key_states.shape[-1] == value_states.shape[-1]):
                raise ValueError(
                    "Cannot preserve cumulative-sequence attention on multi-host TPU with VANILLA "
                    "attention when query/key/value head dimensions differ."
                )
            unsupported_sdpa_features = ScaledDotProductAttn.get_unsupported_fallback_features(
                softmax_aux=softmax_aux,
                logits_soft_cap=logits_soft_cap,
                dropout_prob=dropout_prob_final,
                normalize_output=normalize_output,
            )
            if unsupported_sdpa_features:
                raise ValueError(
                    "Cannot route cumulative-sequence attention through SDPA on multi-host TPU "
                    f"because {', '.join(unsupported_sdpa_features)} are not supported by the SDPA fallback."
                )
            logger.warning_once(
                "Routing cumulative-sequence attention through SDPA on multi-host TPU "
                "because VANILLA attention does not support cum_seqlens_*."
            )
            return ScaledDotProductAttn(metadata=self.metadata)

        def _call_attention(callable_attn: tp.Any, input_kwargs: dict[str, tp.Any]) -> AttentionOutput:
            """Invoke ``callable_attn`` with the right keyword argument set.

            Vanilla / Splash / BlockSparse kernels accept a
            ``return_attention_weights`` flag that the other backends do not
            understand; this helper opts those backends into materialising
            weights only when ``output_attentions_computed`` is ``True``.

            Args:
                callable_attn: Concrete attention :class:`Operation` to call.
                input_kwargs: Base keyword arguments destined for the kernel.

            Returns:
                :class:`AttentionOutput` returned by ``callable_attn``.
            """
            call_kwargs = dict(input_kwargs)
            impl_names = _get_impl_names(callable_attn)
            valid_queries = None
            if dropout_prob_final > 0.0:
                # SDPA silently ignores dropout; Flash uses an integer seed and several of its
                # fallback/device paths do not implement dropout. Use the registry's key-aware
                # reference operation for the entire stochastic pass, on every device.
                callable_attn = active_impl
                impl_names = _get_impl_names(callable_attn)
                if is_paged and mask_info is None and call_kwargs["bias"] is None and init_bias is not None:
                    # The paged adapter builds request masks itself, so initialize
                    # the original lazy bias before that synthesis as on dense paths.
                    call_kwargs["bias"] = init_bias()
                    call_kwargs["init_bias"] = None
                if not is_paged and (cum_seqlens_q is not None or cum_seqlens_k is not None):
                    # Vanilla initializes a lazy bias only when both bias and MaskInfo are
                    # absent. Preserve that original decision before adding our varlen mask.
                    if mask_info is None and call_kwargs["bias"] is None and init_bias is not None:
                        call_kwargs["bias"] = init_bias()
                        call_kwargs["init_bias"] = None
                    # Vanilla consumes MaskInfo rather than cum_seqlens_*. Materialize the
                    # equivalent visibility mask before dispatch, retaining any supplied mask,
                    # positions and baked-in causal/window flags.
                    batch, q_len = query_states.shape[:2]
                    kv_len = key_states.shape[1]

                    def _sequence_layout(lengths: tp.Any, seq_len: int) -> tuple[tp.Any, tp.Any, tp.Any]:
                        positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
                        if lengths is None:
                            return jnp.ones_like(positions, dtype=jnp.bool_), jnp.zeros_like(positions), positions
                        lengths = jnp.asarray(lengths)
                        if lengths.ndim != 1:
                            raise ValueError("cum_seqlens must be one-dimensional.")
                        if lengths.shape[0] == batch:
                            valid = positions < lengths[:, None]
                            return valid, jnp.zeros_like(positions), positions
                        if lengths.shape[0] == batch + 1:
                            valid = positions < jnp.diff(lengths)[:, None]
                            return valid, jnp.zeros_like(positions), positions
                        if batch == 1 and lengths.shape[0] > 1:
                            segments = jnp.searchsorted(lengths[1:], positions, side="right")
                            starts = lengths[jnp.minimum(segments, lengths.shape[0] - 1)]
                            valid = (positions >= lengths[0]) & (positions < lengths[-1])
                            return valid, segments, positions - starts
                        raise ValueError("cum_seqlens must contain per-example lengths or cumulative boundaries.")

                    q_valid, q_segments, q_positions = _sequence_layout(cum_seqlens_q, q_len)
                    k_valid, k_segments, k_positions = _sequence_layout(cum_seqlens_k, kv_len)
                    visibility = q_valid[:, None, :, None] & k_valid[:, None, None, :]
                    visibility &= q_segments[:, None, :, None] == k_segments[:, None, None, :]
                    if mask_info is None:
                        dropout_mask = MaskInfo(
                            _attention_mask=visibility, q_positions=q_positions, kv_positions=k_positions
                        )
                    else:
                        existing_mask = mask_info.attention_mask
                        if existing_mask is not None:
                            if existing_mask.ndim == 2:
                                existing_mask = existing_mask[:, None, None, :]
                            elif existing_mask.ndim == 3:
                                existing_mask = existing_mask[:, None, :, :]
                            visibility = visibility & existing_mask.astype(jnp.bool_)
                        dropout_mask = mask_info.replace(attention_mask=visibility)
                    # Bake relative-position causal/window masks for packed windows; the reference
                    # kernel's dense top-left triangle would otherwise mix per-sequence offsets.
                    if causal and not dropout_mask.causal_mask_baked_in:
                        qp = dropout_mask.q_positions if dropout_mask.q_positions is not None else q_positions
                        kp = dropout_mask.kv_positions if dropout_mask.kv_positions is not None else k_positions
                        visibility &= kp[:, None, None, :] <= qp[:, None, :, None]
                        dropout_mask = dropout_mask.replace(attention_mask=visibility, causal_mask_baked_in=True)
                    if sliding_window is not None and not dropout_mask.sliding_window_baked_in:
                        left, right = (
                            sliding_window if isinstance(sliding_window, tuple) else (max(sliding_window - 1, 0),) * 2
                        )
                        qp = dropout_mask.q_positions if dropout_mask.q_positions is not None else q_positions
                        kp = dropout_mask.kv_positions if dropout_mask.kv_positions is not None else k_positions
                        visibility &= (kp[:, None, None, :] >= qp[:, None, :, None] - left) & (
                            kp[:, None, None, :] <= qp[:, None, :, None] + right
                        )
                        dropout_mask = dropout_mask.replace(attention_mask=visibility, sliding_window_baked_in=True)
                        call_kwargs["sliding_window"] = None
                    call_kwargs["mask_info"] = dropout_mask
                    call_kwargs["cum_seqlens_q"] = call_kwargs["cum_seqlens_k"] = None
                    valid_queries = q_valid & jnp.any(visibility, axis=(1, 3))
            weight_aware_impls = {
                AttentionMechanisms.VANILLA.value,
                AttentionMechanisms.BLOCKSPARSE.value,
                AttentionMechanisms.SPLASH.value,
            }
            if impl_names & weight_aware_impls:
                call_kwargs["return_attention_weights"] = output_attentions_computed
            if impl_names & {AttentionMechanisms.VANILLA.value, "paged_attention_reference"}:
                # None inherits ambient JAX precision; explicit values (including
                # DEFAULT) are forwarded.
                call_kwargs["precision"] = precision
            result = callable_attn(**call_kwargs)
            if valid_queries is not None:
                # Softmax over a fully masked padded row is not guaranteed to yield zero.
                result = AttentionOutput(
                    attention_outputs=jnp.where(valid_queries[:, :, None, None], result.attention_outputs, 0),
                    attention_weights=(
                        jnp.where(valid_queries[:, None, :, None], result.attention_weights, 0)
                        if result.attention_weights is not None
                        else None
                    ),
                    cache_view=result.cache_view,
                )
            return result

        with _attention_mesh_context(self.config):  # pyright: ignore[reportOptionalContextManager]
            input_dict: dict[str, tp.Any] = dict(
                query=query_states,
                key=key_states,
                value=value_states,
                mask_info=mask_info,
                bias=bias,
                sliding_window=sliding_window,
                cache_metadata=cache_metadata,
                cache_view=attention_view,
                init_bias=init_bias,
                causal=causal,
                deterministic=deterministic_computed,
                dropout_rng=dropout_rng_final,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale_computed,
                logits_soft_cap=logits_soft_cap,
                dropout_prob=dropout_prob_final,
                precision=precision_computed,
                prevent_cse=prevent_cse,
                cum_seqlens_q=cum_seqlens_q,
                cum_seqlens_k=cum_seqlens_k,
                normalize_output=normalize_output,
                fused_backward=fused_backward,
                compute_dtype=compute_dtype,
                optimized=optimized,
                mask_value=mask_value,
                vmem_limit_bytes=vmem_limit_bytes,
                policy=policy_computed,
                **extra_op_kwargs,
            )
            is_decode_mode: bool = mode == common_types.MODE_DECODE
            output: AttentionOutput
            if is_decode_mode:
                if cache_view is None:
                    raise ValueError("Decode mode requires a cache_view, but None was provided.")
                has_decode_impl: bool = self.impl_decode is not None
                callable_attn: tp.Any = self.impl_decode if has_decode_impl else self.impl
                callable_attn = _maybe_route_varlen_multihost_tpu_attention(callable_attn)
                output = _call_attention(callable_attn, input_dict)
            else:
                callable_attn = _maybe_route_varlen_multihost_tpu_attention(self.impl)
                output = _call_attention(callable_attn, input_dict)

        # An explicit per-call ``compute_dtype`` (e.g. an fp32 vision tower inside a bf16-attention model) also sets
        # the output dtype; otherwise the result would be rounded back to the config's ``attn_dtype``.
        target_dtype: jnp.dtype = compute_dtype if compute_dtype is not None else self.impl.metadata.runtime_dtype

        def cast_to_dtype(x: tp.Any) -> tp.Any:
            """Cast ``x`` to the resolved attention runtime dtype.

            Used as a leaf function for ``jax.tree_util.tree_map`` so the
            cast applies to every array in the attention output pytree.

            Args:
                x: Array (or pytree leaf) produced by the attention
                    implementation.

            Returns:
                tp.Any: ``x`` cast to ``target_dtype``.
            """
            return x.astype(target_dtype)

        # Only cast attention_outputs and attention_weights — leave cache_view
        # untouched to preserve original dtypes (e.g. uint8 for TurboQuant pages).
        result = AttentionOutput(
            attention_outputs=jtu.tree_map(cast_to_dtype, output.attention_outputs),
            attention_weights=(
                jtu.tree_map(cast_to_dtype, output.attention_weights)
                if output_attentions_computed and output.attention_weights is not None
                else None
            ),
            cache_view=(
                cache_view.replace(transformer=attention_view if output.cache_view is None else output.cache_view)
                if isinstance(cache_view, ParallelHybridCacheView)
                else output.cache_view
            ),
        )
        return result

    __call__ = forward

    # Operation access properties for dynamic discovery
    @property
    def operation_executor(self):
        """Return an :class:`OperationExecutor` bundling prefill + decode backends.

        Built lazily on every access (cheap) so that ``self.impl`` /
        ``self.impl_decode`` swaps performed by tests or autotuning are
        immediately visible.

        Returns:
            :class:`OperationExecutor` wrapping the prefill (``self.impl``)
            and decode (``self.impl_decode``) operations with no mixin.
        """
        from easydel.operations.executor import OperationExecutor

        return OperationExecutor(
            prefill_impl=self.impl,
            decode_impl=self.impl_decode,
            mixin_impl=None,
        )

    @property
    def operation(self):
        """Return the prefill / training attention backend instance.

        This is what :meth:`forward` calls in every mode except
        ``MODE_DECODE``, and even in decode it is used as the fallback
        when ``impl_decode`` is ``None``.
        """
        return self.impl

    @property
    def decode_operation(self):
        """Return the decode-only attention backend, if one is configured.

        Returns:
            The :class:`Operation` instance bound to
            ``base_config.decode_attn_mechanism`` at construction time,
            or ``None`` when prefill and decode share the same backend.
        """
        return self.impl_decode

    @property
    def operation_requirements(self):
        """Return the combined metadata/cache requirements of both backends.

        Delegates to the underlying :class:`OperationExecutor` and merges
        the requirements that the prefill and (optional) decode kernels
        impose on the caller — e.g. whether a paged cache view is required,
        which metadata pytree fields must be populated, etc.

        Returns:
            :class:`OperationRequirements` aggregating prefill and decode
            requirements.
        """
        return self.operation_executor.get_combined_requirements()

    @property
    def requires_cache(self) -> bool:
        """Whether the active backend(s) consume a KV cache view.

        Combines the cache requirements of the prefill and (optional)
        decode backends with the user-supplied
        ``requires_cache`` override. Used by surrounding modules to skip
        cache allocation for encoder-only paths (vision encoders, etc.).
        """
        return self.operation_executor.requires_cache

    @property
    def has_separate_decode(self) -> bool:
        """Return ``True`` iff prefill and decode use different backends.

        Driven by ``base_config.decode_attn_mechanism`` at construction
        time; e.g. a model can run SplashAttention for prefill and
        ``REGRESSIVE_DECODE`` for token-by-token generation.
        """
        return self.operation_executor.has_separate_decode


class SoftCappedAttentionModule(FlexibleAttentionModule):
    """Attention performer with a configurable default attention-logit soft cap.

    The registered attention operation applies ``cap * tanh(logits / cap)``
    before softmax. A per-call cap overrides the constructor default; ``None``
    selects that default, matching the historical Gemma performer contract.
    """

    def __init__(self, *args, logits_soft_cap: float | None = None, **kwargs):
        """Build the performer and retain its default soft cap.

        Args:
            *args: Forwarded to :class:`FlexibleAttentionModule`.
            logits_soft_cap: Default cap; ``None`` disables soft capping.
            **kwargs: Forwarded to :class:`FlexibleAttentionModule`.
        """
        super().__init__(*args, **kwargs)
        self.logits_soft_cap = None if logits_soft_cap is None else float(logits_soft_cap)

    def forward(self, *args, logits_soft_cap: float | None = None, **kwargs):
        """Run registered attention with the configured cap or a call-time override.

        Args:
            *args: Forwarded to :meth:`FlexibleAttentionModule.forward`.
            logits_soft_cap: Per-call cap; ``None`` uses the constructor default.
            **kwargs: Forwarded to :meth:`FlexibleAttentionModule.forward`.

        Returns:
            The base performer's :class:`AttentionOutput`.
        """
        if logits_soft_cap is None:
            logits_soft_cap = self.logits_soft_cap
        return super().forward(*args, logits_soft_cap=logits_soft_cap, **kwargs)

    __call__ = forward


Cfg = tp.TypeVar("Cfg", bound=EasyDeLBaseConfig)


class AttentionModule(spx.Module, tp.Generic[Cfg]):
    """Shared sharding / mask / cache helpers for concrete attention layers.

    This abstract intermediate sits between :class:`spx.Module` and
    :class:`UnifiedAttention` (and any custom attention implementations).
    It does not own QKV projections or define a ``forward``; it instead
    bundles the small set of geometry-aware utilities every attention
    implementation needs — RoPE application, Q/K/KV sharding constraint
    application, GQA repeat, KV cache concatenation, sliding-window
    extraction, and KV-cache quantizer construction.

    Subclasses are expected to override ``__init__`` (calling
    ``super().__init__(config)``) and provide a ``forward`` method that
    composes the helpers below.

    Attributes:
        config (Cfg): Owning model config. Used for sharding axis names,
            partition manager, KV-cache quantization config, mesh resolution,
            and attention dtype.
    """

    def __init__(self, config: Cfg):
        """Bind the configuration object.

        Args:
            config: Model configuration that conforms to (or extends)
                :class:`easydel.infra.base_config.EasyDeLBaseConfig`.
                Cached on ``self.config`` for use by every helper method.
        """
        super().__init__()
        self.config = config

    def _split_heads(self, hidden_states):
        """Reshape ``[batch, seq, embed_dim]`` into ``[batch, seq, heads, head_dim]``.

        Requires ``self.num_heads`` and ``self.head_dim``. Seven encoder-style
        attention layers carried this exact body; note that
        :class:`UnifiedAttention` deliberately overrides ``_merge_heads`` with a
        different implementation, so decoder layers inheriting through it keep
        that one.

        Args:
            hidden_states: Tensor shaped ``[batch, seq, embed_dim]``.

        Returns:
            Tensor shaped ``[batch, seq, num_heads, head_dim]``.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def _softmax_aux(self) -> JArray | None:
        """Return this layer's attention-sink logits, or ``None``.

        Sinks are an optional learned per-head logit appended to the softmax
        denominator. Layers spell the attribute either ``sinks`` (GPT-OSS,
        DeepSeek-V4) or ``softmax_aux``, and it may be a raw array or a
        parameter wrapper, so the lookup has to try both names and unwrap
        ``.value``. That four-way probe was copy-pasted at a dozen call sites;
        it lives here once instead.

        Returns:
            The unwrapped sink array, or ``None`` when the layer has no sinks.
        """
        aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        return getattr(aux, "value", aux)

    def get_softmax_aux(self) -> JArray | None:
        """Return unwrapped attention-sink logits through the layer's accessor.

        Delegates to the existing ``_softmax_aux`` hook so subclasses overriding
        that hook retain their behavior. Shared attention eligibility policies
        can use this public API without inspecting parameter wrappers or family
        attributes.

        Returns:
            The layer's sink logits, or ``None`` when no sinks are configured.
        """
        return self._softmax_aux()

    @staticmethod
    def apply_complex_rotary(
        xq: Float[JArray, "... seq heads dim"],
        xk: Float[JArray, "... seq heads dim"],
        freqs_cis: Complex[JArray, "batch seq 1 dim_2"],
    ) -> tuple[Float[JArray, "... seq heads dim"], Float[JArray, "... seq heads dim"]]:
        """Apply rotary position embeddings using complex number multiplication.

        Implements rotary position embeddings (RoPE) by treating pairs of
        dimensions as complex numbers and multiplying by rotation frequencies.

        Args:
            xq: Query tensor with shape [..., seq, heads, dim].
            xk: Key tensor with shape [..., seq, heads, dim].
            freqs_cis: Complex frequency tensor with shape [batch, seq, 1, dim/2].
                Contains precomputed cos + i*sin values for rotation.

        Returns:
            Tuple of (rotated_query, rotated_key), each with the same shape
            as the input tensors, containing position-aware representations.

        Note:
            The head dimension is split into pairs and treated as complex numbers.
            The rotation is applied via complex multiplication: (a + bi) * (cos + i*sin).
        """
        xq_float32: Float[JArray, "... seq heads dim"] = xq.astype(jnp.float32)
        xk_float32: Float[JArray, "... seq heads dim"] = xk.astype(jnp.float32)
        xq_reshaped: Float[JArray, "... seq heads dim_2 2"] = xq_float32.reshape(*xq.shape[:-1], -1, 2)
        xk_reshaped: Float[JArray, "... seq heads dim_2 2"] = xk_float32.reshape(*xk.shape[:-1], -1, 2)
        xq_complex: Complex[JArray, "... seq heads dim_2"] = xq_reshaped[..., 0] + 1j * xq_reshaped[..., 1]
        xk_complex: Complex[JArray, "... seq heads dim_2"] = xk_reshaped[..., 0] + 1j * xk_reshaped[..., 1]
        xq_out_complex: Complex[JArray, "batch seq heads dim_2"] = xq_complex * freqs_cis[:, :, None, :]
        xk_out_complex: Complex[JArray, "batch seq heads dim_2"] = xk_complex * freqs_cis[:, :, None, :]
        xq_real_part: Float[JArray, "batch seq heads dim_2"] = jnp.real(xq_out_complex)
        xq_imag_part: Float[JArray, "batch seq heads dim_2"] = jnp.imag(xq_out_complex)
        xk_real_part: Float[JArray, "batch seq heads dim_2"] = jnp.real(xk_out_complex)
        xk_imag_part: Float[JArray, "batch seq heads dim_2"] = jnp.imag(xk_out_complex)
        xq_out_real = jnp.stack([xq_real_part, xq_imag_part], axis=-1)
        xk_out_real = jnp.stack([xk_real_part, xk_imag_part], axis=-1)
        xq_out_reshaped: Float[JArray, "batch seq heads dim"] = xq_out_real.reshape(*xq_out_real.shape[:-2], -1)
        xk_out_reshaped: Float[JArray, "batch seq heads dim"] = xk_out_real.reshape(*xk_out_real.shape[:-2], -1)
        xq_original_dtype: jnp.dtype = xq.dtype
        xk_original_dtype: jnp.dtype = xk.dtype
        xq_out: Float[JArray, "batch seq heads dim"] = xq_out_reshaped.astype(xq_original_dtype)
        xk_out: Float[JArray, "batch seq heads dim"] = xk_out_reshaped.astype(xk_original_dtype)
        return xq_out, xk_out

    def apply_qk_shardings(
        self,
        q: Float[JArray, "batch seq heads dim"],
        k: Float[JArray, "batch seq heads dim"],
    ) -> tuple[Float[JArray, "batch seq heads dim"], Float[JArray, "batch seq heads dim"]]:
        """Apply logical sharding constraints to query and key tensors.

        Constrains the query and key tensors to follow the partition specifications
        defined in the model configuration for distributed computation.

        Args:
            q: Query tensor with shape [batch, seq, heads, dim].
            k: Key tensor with shape [batch, seq, heads, dim].

        Returns:
            Tuple of (sharded_query, sharded_key) with sharding constraints applied
            according to the configuration's partition manager.

        Note:
            Query uses AttnQSharding and key uses AttnKVSharding from common_types.
            This ensures proper distribution of attention computations across devices.
        """
        q_sharded: Float[JArray, "batch seq heads dim"] = apply_logical_sharding(
            q,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        k_sharded: Float[JArray, "batch seq heads dim"] = apply_logical_sharding(
            k,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return q_sharded, k_sharded

    def apply_qkv_shardings(
        self,
        q: Float[JArray, "batch seq heads dim"],
        k: Float[JArray, "batch seq heads dim"],
        v: Float[JArray, "batch seq heads dim"],
    ) -> tuple[
        Float[JArray, "batch seq heads dim"], Float[JArray, "batch seq heads dim"], Float[JArray, "batch seq heads dim"]
    ]:
        """Apply logical sharding constraints to query, key, and value tensors.

        Constrains the Q, K, V tensors to follow partition specifications defined
        in the model configuration for distributed attention computation.

        Args:
            q: Query tensor with shape [batch, seq, heads, dim].
            k: Key tensor with shape [batch, seq, heads, dim].
            v: Value tensor with shape [batch, seq, heads, dim].

        Returns:
            Tuple of (sharded_query, sharded_key, sharded_value) with sharding
            constraints applied according to the configuration's partition manager.

        Note:
            Query uses AttnQSharding while key and value use AttnKVSharding.
            This allows for grouped query attention (GQA) where KV heads may
            be fewer than query heads.
        """
        q_sharded, k_sharded = self.apply_qk_shardings(q, k)
        v_sharded: Float[JArray, "batch seq heads dim"] = apply_logical_sharding(
            v,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return q_sharded, k_sharded, v_sharded

    @staticmethod
    def build_cache_pos(
        attention_mask: Bool[JArray, "batch heads seq_q seq_k"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | None = None,
    ) -> Int[JArray, "batch seq"]:
        """Compute per-token absolute positions for cache-aware attention.

        Builds the position index that RoPE / ALiBi will consume:
        a cumulative-sum of the per-token mask along the query axis, offset
        by the cache cursor in decode mode. The mask is reduced along the
        K axis first, so padded tokens contribute zero to the cumulative
        position.

        Args:
            attention_mask: Boolean mask of shape
                ``[batch, heads, q_len, k_len]``. Only the last head's row
                is used (heads are assumed to agree on masking).
            mode: SpecTrax runtime mode marker; in
                :data:`common_types.MODE_DECODE` the cache cursor is added
                to the cumulative positions.
            cache_view: Optional transformer-style cache view; consulted
                for its ``indexes`` (per-batch cursor) only in decode mode.

        Returns:
            Integer position array of shape ``[batch, q_len]`` suitable
            for RoPE/ALiBi position lookups.
        """
        end_index: int | Int[JArray, "batch 1"] = 0
        is_decode: bool = mode == common_types.MODE_DECODE
        # Support transformer-like composite cache views (e.g., ParallelHybridCacheView)
        has_indexes: bool = cache_view is not None and hasattr(cache_view, "indexes")
        should_use_cache_index: bool = has_indexes and is_decode
        if should_use_cache_index:
            cache_indexes = cache_view.indexes
            end_index = jnp.reshape(cache_indexes, (-1, 1))
        mask_any_kv: Bool[JArray, "batch heads seq_q"] = jnp.any(attention_mask, -1)
        mask_last_head: Bool[JArray, "batch seq_q"] = mask_any_kv[:, -1, :]
        inipos: Int[JArray, "batch seq_q"] = jnp.cumsum(mask_last_head, axis=-1)
        inipos_ge_one: Bool[JArray, "batch seq_q"] = inipos >= 1
        inipos_adjusted: Int[JArray, "batch seq_q"] = inipos - inipos_ge_one
        result: Int[JArray, "batch seq_q"] = inipos_adjusted + end_index
        return result

    @cached_property
    def quantizer(self) -> EasyQuantizer:
        """Lazy :class:`EasyQuantizer` for KV-cache write paths.

        Reads ``config.kv_cache_quantization_config``: configurations of
        type :class:`TurboQuantConfig` return a no-op quantizer because
        TurboQuant compresses inside the kernel, while every other
        configuration constructs a real quantizer that the cache views
        apply on writes.

        Returns:
            :class:`EasyQuantizer` instance to use for KV-cache writes.
        """
        kv_quant_cfg = self.config.kv_cache_quantization_config

        if isinstance(kv_quant_cfg, TurboQuantConfig):
            return EasyQuantizer(quantization_config=None)

        quantizer_instance: EasyQuantizer = EasyQuantizer(
            quantization_config=kv_quant_cfg,
        )
        return quantizer_instance

    @property
    def default_key_value_sharding(self) -> NamedSharding:
        """Build the default :class:`NamedSharding` for K/V tensors.

        Uses ``config.partition_axis`` to build a 4-D partition spec
        ``(batch, key_sequence, head, attention_dim)`` on the resolved
        stage mesh. Used as the fallback by :meth:`get_sharding_safely`.

        Returns:
            :class:`NamedSharding` to apply to KV tensors when no other
            sharding is available.
        """
        paxis: tp.Any = self.config.partition_axis
        mesh: StageMesh = resolve_stage_mesh(self.config.mesh)
        spec: PartitionSpec = PartitionSpec(
            paxis.batch_axis,
            paxis.key_sequence_axis,
            paxis.head_axis,
            paxis.attention_dim_axis,
        )
        sharding: NamedSharding = NamedSharding(
            mesh=mesh,
            spec=spec,
        )
        return sharding

    def get_sharding_safely(self, tensor: Float[JArray, "..."]) -> PartitionSpec:
        """Return ``tensor.sharding.spec``, falling back to KV defaults.

        Args:
            tensor: Array whose partition spec is needed. Lacking a
                ``sharding`` attribute (e.g. during eager-mode tracing)
                triggers the fallback.

        Returns:
            :class:`PartitionSpec` of ``tensor`` or, when absent, the
            spec of :attr:`default_key_value_sharding`.
        """
        sharding = getattr(tensor, "sharding", None)
        if sharding is None:
            sharding = self.default_key_value_sharding
        return sharding.spec

    def _handle_cache_concat(
        self,
        query: Float[JArray, "batch seq_q heads dim"],
        key: Float[JArray, "batch seq_k heads dim"],
        value: Float[JArray, "batch seq_v heads dim"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        mask_info: MaskInfo | None,
        cache_view: TransformerCacheView | None,
        cache_metadata: TransformerMetadata | None,
        causal: bool = True,
    ) -> tuple[
        Float[JArray, "batch seq_k heads dim"],
        Float[JArray, "batch seq_v heads dim"],
        MaskInfo | None,
        TransformerCacheView | None,
        AttnMaskDetail | None,
    ]:
        """Handle concatenation of current KV states to the cache.

        Manages KV cache operations during autoregressive generation, including
        storing new key-value pairs and retrieving cached states.

        Args:
            query: Current query tensor with shape [batch, seq_q, heads, dim].
            key: Current key tensor with shape [batch, seq_k, heads, dim].
            value: Current value tensor with shape [batch, seq_v, heads, dim].
            mode: Runtime mode (TRAIN, PREFILL, or DECODE).
            mask_info: Container for attention masks and segment information.
            cache_view: Current view into the KV cache, or None if caching disabled.
            cache_metadata: Metadata about cache state (indices, starts).
            causal: Whether cache mask alignment should constrain queries to
                past/current keys. Defaults to the existing causal behavior.

        Returns:
            Tuple containing:
                - key: Key tensor (potentially retrieved from cache in decode mode).
                - value: Value tensor (potentially retrieved from cache in decode mode).
                - mask_info: Updated mask information.
                - cache_view: Updated cache view, or None if no cache.
                - masking_details: Details about mask type and configuration.

        Note:
            If cache_view is None, the original key and value are returned unchanged.
        """
        cache_is_none: bool = cache_view is None
        if cache_is_none:
            return key, value, mask_info, None, None

        # Keep default calls compatible with cache views implementing the older
        # signature; only noncausal callers need the additional cache contract.
        causal_kwargs = {} if causal else {"causal": False}
        key, value, mask_info, cache_view, masking_details = cache_view.concatenate_to_cache(
            query=query,
            key=key,
            value=value,
            mode=mode,
            quantizer=self.quantizer,
            mask_info=mask_info,
            cache_metadata=cache_metadata,
            runtime_sharding_resolver=self.config.runtime_sharding_resolver,
            **causal_kwargs,
        )

        return key, value, mask_info, cache_view, masking_details  # pyright: ignore[reportReturnType]

    def _apply_sliding_window(
        self,
        key: Array,
        value: Array,
        mask_info: MaskInfo,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | None,
        sliding_window: int | tuple[int, int],
        query_length: int,
        masking_details: AttnMaskDetail | None,
        cache_metadata: TransformerMetadata | None,
    ) -> tuple[Array, Array, MaskInfo, TransformerMetadata | None]:
        """Apply sliding window masking and slicing to KV tensors and mask.

        Implements sliding window attention by restricting each query position
        to attend only to a local window of key-value positions.

        Args:
            key: Key tensor with shape [batch, seq_k, heads, dim].
            value: Value tensor with shape [batch, seq_v, heads, dim].
            mask_info: Container for attention mask.
            mode: Runtime mode (TRAIN, PREFILL, or DECODE).
            cache_view: View into KV cache for position tracking.
            sliding_window: Window size as int (HF convention: the query plus
                ``W - 1`` neighbours each side) or inclusive tuple (left, right).
            query_length: Length of query sequence.
            masking_details: Details about mask type from cache.
            cache_metadata: Metadata for cache position tracking.

        Returns:
            Tuple containing:
                - key: Sliced key tensor within sliding window.
                - value: Sliced value tensor within sliding window.
                - mask_info: Updated mask with window constraints.
                - cache_metadata: Updated metadata for sliced window.

        Raises:
            ValueError: If sliding_window contains negative values.

        Note:
            Explicit query/key positions preserve the cache-owned KV layout and
            intersect the mask in that coordinate system without slicing. Without
            explicit positions, decode slices retain the union of every active
            query's window and prefill keeps the full key range.
        """
        if isinstance(sliding_window, int):
            if sliding_window < 0:
                raise ValueError(
                    f"Invalid sliding_window: expected a non-negative integer, but got {sliding_window}. "
                    f"Window size must be >= 0."
                )
            # An int is a window *size* (HF convention): the query plus its ``W - 1`` neighbours.
            left_window = right_window = max(sliding_window - 1, 0)
        else:
            left_window, right_window = sliding_window
            if left_window < 0 or right_window < 0:
                raise ValueError(
                    f"Invalid sliding_window: expected non-negative values, but got ({left_window}, {right_window}). "
                    f"Both left and right window sizes must be >= 0."
                )

        attn = mask_info.attention_mask.astype(jnp.bool_)  # (B, H, Q, K)
        # Cache-owned layouts can contain a rolled tail or a transient union of
        # that tail and the current block. Their columns are not absolute token
        # indexes. Intersect in the supplied coordinate system and retain the
        # layout so positions, segments, auxiliary states and metadata stay aligned.
        q_positions = getattr(mask_info, "q_positions", None)
        kv_positions = getattr(mask_info, "kv_positions", None)
        if q_positions is not None and kv_positions is not None:
            relative = kv_positions[:, None, None, :] - q_positions[:, None, :, None]
            attn = attn & (relative >= -left_window) & (relative <= right_window)
            return key, value, mask_info.replace(attention_mask=attn, sliding_window_baked_in=True), cache_metadata

        B, _H, Q, K = attn.shape
        if query_length != Q:
            query_length = Q

        if mode == common_types.MODE_DECODE and cache_view is not None:
            indexes = cache_view.indexes.reshape(-1)  # (B,)
        elif mode == common_types.MODE_PREFILL:
            indexes = jnp.full((B,), Q - 1, dtype=jnp.int32)  # last query row
        else:
            indexes = jnp.zeros((B,), dtype=jnp.int32)

        offsets = jnp.zeros((B,), dtype=jnp.int32)
        # A decode call can verify several tokens at once. Keep the union of
        # their local windows, not just the final query's window.
        width = min(left_window + right_window + (Q if mode == common_types.MODE_DECODE else 1), K)

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None), out_axes=(0, 0, 0))
        def _select_slices(ikey, ival, imsk, offset, index, mode_):
            """Per-batch sliding-window selection of K, V and the attention mask.

            Vmapped over the batch axis: builds the window of valid
            ``(query_row, key_col)`` pairs around the current decode/prefill
            position, masks the attention mask accordingly, and slices
            the KV tensors along the K axis when the window is shorter
            than the full cache.

            Args:
                ikey: Per-batch key tensor.
                ival: Per-batch value tensor.
                imsk: Per-batch attention mask of shape ``(H, Q, K)``.
                offset: Query-row base offset for this batch element.
                index: Current cache length (post-update, decode mode) or
                    last valid query row (prefill mode).
                mode_: Runtime mode (``MODE_DECODE`` / ``MODE_PREFILL`` /
                    other), used as a static branch selector.

            Returns:
                tuple: ``(sliced_key, sliced_value, masked_mask)`` for
                this batch element.
            """
            base_row = offset + jax.lax.broadcasted_iota(jnp.int32, (Q, 1), 0)  # (Q,1)
            if mode_ == common_types.MODE_DECODE:
                # `index` is the post-update cache length, so the active query rows
                # correspond to the trailing `[index - Q, ..., index - 1]` range.
                row = (index - Q) + base_row
            else:
                row = base_row

            col = jax.lax.broadcasted_iota(jnp.int32, (1, K), 1)  # (1,K)
            win = (col >= (row - left_window)) & (col <= (row + right_window))  # (Q,K)

            imsk = imsk & win[None, :, :]  # (H=1, Q, K)

            if mode_ == common_types.MODE_DECODE:
                # The active query rows are already local to the current decode step
                # (typically Q=1), so only the KV axis needs window slicing.
                first_row = index - Q
                start_k = jnp.clip(first_row - left_window, 0, jnp.maximum(K - width, 0))
                imsk = jax.lax.dynamic_slice_in_dim(imsk, start_k, width, axis=2)  # (H,1,width)
                ikey = jax.lax.dynamic_slice_in_dim(ikey, start_k, width, axis=0)  # (width, ...)
                ival = jax.lax.dynamic_slice_in_dim(ival, start_k, width, axis=0)  # (width, ...)
                return ikey, ival, imsk

            # Multi-query prefill needs each row's own local keys, including
            # early rows. A trailing-cache slice would discard them (and can
            # consist entirely of unwritten cells in an oversized cache).
            return ikey, ival, imsk

        key, value, attn = _select_slices(key, value, attn, offsets, indexes, mode)

        mask_info = mask_info.replace(attention_mask=attn, sliding_window_baked_in=True)

        if cache_metadata is not None and mode == common_types.MODE_DECODE:
            passed = cache_metadata.indexes - cache_metadata.starts
            cache_metadata = TransformerMetadata(
                starts=jax.lax.max(0, width - passed),
                indexes=jnp.full((attn.shape[0],), attn.shape[-1]),
            )

        return key, value, mask_info, cache_metadata

    @jax.named_scope("easydel-spx-attention-concatenate")
    def concatenate(
        self,
        *,
        query: Array,
        key: Array,
        value: Array,
        mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | common_types.EMPTY_VAL = common_types.NOT_GIVEN,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        causal: bool = True,
    ) -> tuple[
        Array,
        Array,
        MaskInfo | None,
        tp.Callable[[], Array],
        TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None,
        TransformerMetadata | RaggedPagesMetadata | None,
    ]:
        """Stitch current Q/K/V to the cache and resolve the attention mask.

        Central preparation step for every cache-aware forward pass: handles
        cache concatenation (transformer / ragged / unified), creates
        cache metadata when the caller did not supply it, applies optional
        sliding-window slicing to KV (and the mask), and returns a lazy
        ``init_attention_bias`` closure so that kernels that consume
        :class:`MaskInfo` directly never materialize the float bias.

        Mode resolution:

        * ``mode`` defaults to :data:`common_types.NOT_GIVEN`; in that case
          the routine infers ``MODE_TRAIN`` when no cache is present,
          ``MODE_PREFILL`` for multi-token query, and ``MODE_DECODE`` for
          single-token query.

        Cache path selection:

        * :class:`RaggedPagesCacheView` and :class:`UnifiedAttentionCacheView`
          (and ragged :class:`ParallelHybridCacheView`) take the ragged path:
          the kernel handles the cache internally and ``init_attention_bias``
          returns zeros.
        * Otherwise the transformer-style path runs
          :meth:`_handle_cache_concat` to write new K/V into the cache view
          and update the mask, then optionally slides the window via
          :meth:`_apply_sliding_window`.

        Args:
            query: Current query states ``[batch, q_len, heads, dim]``.
            key: Current key states ``[batch, kv_len, heads, dim]``.
            value: Current value states ``[batch, kv_len, heads, dim]``.
            mask_info: Optional ejkernel :class:`MaskInfo` carrying the attention
                mask plus segment/position metadata. ``None`` means all keys
                are valid before applying causal, cache-bound and local masks.
            mode: Runtime mode. Pass :data:`common_types.NOT_GIVEN` to let
                this method infer it from ``cache_view`` and ``query``
                shape.
            cache_view: Optional KV cache view. ``None`` disables caching
                (training path).
            cache_metadata: Optional companion metadata; auto-derived from
                ``cache_view.starts`` / ``cache_view.indexes`` when ``None``.
            sliding_window: Optional window size or inclusive (left, right)
                bounds. Cache-configured windows already baked into the mask
                are preserved; a distinct explicit window intersects their
                absolute-position visibility without changing the KV layout.
                Without explicit positions, decode may slice KV to the union
                of the active queries' windows.
            causal: Whether transformer cache mask alignment is causal. Set
                ``False`` for bidirectional attention over the appended block.
                Defaults to ``True`` for backward compatibility.

        Returns:
            Six-tuple ``(key_states, value_states, mask_info,
            init_attention_bias, cache_view, cache_metadata)`` where the
            arrays reflect the post-concat / post-slide state and
            ``init_attention_bias`` is a zero-argument callable that
            materializes the additive bias on demand.

        Raises:
            AssertionError: If the query batch dimension does not match
                an existing transformer-style cache view's batch.
            ValueError: If the supplied ``sliding_window`` is negative.
        """

        query_length: int = query.shape[1]
        initial_key_length: int = key.shape[1]

        mode_is_empty: bool = isinstance(mode, common_types.EMPTY_VAL)
        mode_computed: common_types.RUNTIME_MODE_TYPES  # type:ignore
        if mode_is_empty:
            cache_view_is_none: bool = cache_view is None
            if cache_view_is_none:
                mode_computed = common_types.MODE_TRAIN
            else:
                query_is_single_token: bool = query_length != 1
                mode_computed = common_types.MODE_PREFILL if query_is_single_token else common_types.MODE_DECODE
        else:
            mode_computed = mode

        is_ragged_cache: bool = isinstance(cache_view, (RaggedPagesCacheView, UnifiedAttentionCacheView))
        if not is_ragged_cache and isinstance(cache_view, ParallelHybridCacheView):
            is_ragged_cache = cache_view.is_ragged
        if is_ragged_cache:
            cache_metadata = _layer_cache_metadata(cache_metadata, cache_view)
            cache_view = cache_view.concatenate_to_cache(key=key, value=value, cache_metadata=cache_metadata)

            batch_size: int = query.shape[0]
            dtype_for_bias: jnp.dtype = self.dtype

            def init_attention_bias() -> Float[JArray, "batch 1 query_length initial_key_length"]:
                """Lazy zero bias for the ragged-cache path.

                The ragged page / unified caches do not need an explicit
                attention bias; this stub is kept so callers that expect a
                zero-initialized bias closure receive the correct shape and
                dtype on demand.

                Returns:
                    jax.Array: A zero-filled bias of shape
                    ``(batch, 1, query_length, initial_key_length)`` and
                    dtype ``self.dtype``.
                """
                bias: Float[JArray, "batch 1 query_length initial_key_length"] = jnp.zeros(
                    (batch_size, 1, query_length, initial_key_length), dtype=dtype_for_bias
                )
                return bias

            return (
                key,
                value,
                mask_info,
                init_attention_bias,
                cache_view,
                cache_metadata,
            )  # pyright: ignore[reportReturnType]

        if cache_view is not None and cache_view.key is not None:
            query_batch: int = query.shape[0]
            cache_batch: int = cache_view.key.shape[0]  # pyright: ignore[reportOptionalSubscript]
            batches_match: bool = query_batch == cache_batch
            assert batches_match, "Batch size mismatch between query and cache."

        # Dense kernels can consume an absent mask directly. Cache alignment
        # and explicit local-window slicing need a concrete visibility mask,
        # however, even when the caller did not supply padding/segment data.
        if mask_info is None and (cache_view is not None or sliding_window is not None):
            mask_info = MaskInfo(
                _attention_mask=jnp.ones((query.shape[0], 1, query_length, initial_key_length), dtype=jnp.bool_)
            )

        key, value, mask_info, cache_view, _masking_details = self._handle_cache_concat(
            query=query,
            key=key,
            value=value,
            mode=mode_computed,
            mask_info=mask_info,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            causal=causal,
        )

        metadata_is_none: bool = cache_metadata is None
        cache_view_exists: bool = cache_view is not None
        should_create_metadata: bool = metadata_is_none and cache_view_exists

        if should_create_metadata:
            starts = cache_view.starts
            indexes = cache_view.indexes
            cache_metadata = TransformerMetadata(starts=starts, indexes=indexes)
        else:
            cache_metadata = cache_metadata

        sliding_window_provided: bool = sliding_window is not None
        has_cache_sliding: bool = (
            cache_view is not None
            and cache_view.masking_details is not None
            and cache_view.masking_details.mask_type == AttnMaskType.SLIDING
        )
        should_apply_sliding: bool = sliding_window_provided or has_cache_sliding

        if should_apply_sliding:
            # Support transformer-like composite cache views (e.g., ParallelHybridCacheView)
            masking_details_final: AttnMaskDetail | None = getattr(cache_view, "masking_details", None)
            has_masking_details: bool = masking_details_final is not None
            is_sliding_type: bool = has_masking_details and masking_details_final.mask_type == AttnMaskType.SLIDING
            sliding_window_computed: int | tuple[int, int]
            if is_sliding_type:
                sliding_window_computed = masking_details_final.size if sliding_window is None else sliding_window
            else:
                sliding_window_computed = sliding_window

            # Transformer sliding caches have already baked their configured
            # window using absolute coordinates. Reapplying it against local KV
            # columns loses valid history after overflow. Only that known-equal
            # window can be omitted: a caller-baked mask without a configured
            # sliding cache does not identify which window it contains.
            distinct_window = sliding_window_provided and (
                not is_sliding_type or sliding_window != masking_details_final.size
            )
            window_baked_in = mask_info is not None and mask_info.sliding_window_baked_in
            if not window_baked_in or distinct_window:
                key, value, mask_info, cache_metadata = self._apply_sliding_window(
                    key=key,
                    value=value,
                    mask_info=mask_info,
                    mode=mode_computed,
                    cache_view=cache_view,
                    sliding_window=sliding_window_computed,
                    query_length=query_length,
                    masking_details=masking_details_final,
                    cache_metadata=cache_metadata,
                )

        dtype_self: jnp.dtype = self.dtype

        def init_attention_bias() -> Array:
            """Materialize the additive attention bias from ``mask_info``.

            Lazy so the bias is only built when downstream code asks for
            it (e.g. for kernels that accept an explicit float bias);
            kernels that consume :class:`MaskInfo` directly skip this
            entirely.

            Returns:
                jax.Array: Bias broadcastable over ``(batch, heads,
                query, key)``, in ``self.dtype``.
            """
            if mask_info is None:
                return jnp.zeros((query.shape[0], 1, query_length, key.shape[1]), dtype=dtype_self)
            bias: Array = mask_info.create_bias(dtype_self)
            return bias

        return (
            key,
            value,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        )  # pyright: ignore[reportReturnType]

    def shard_attention_prod(
        self, attn_output: Float[JArray, "batch seq heads dim"], pre_projection: bool = False
    ) -> Float[JArray, "batch seq heads dim"]:
        """Constrain the attention output to the model's hidden-state sharding.

        Applied before and after the output projection so the residual
        stream stays in the canonical :data:`common_types.HiddenStateSharding`
        layout regardless of internal kernel sharding choices.

        Args:
            attn_output: Attention output tensor, typically of shape
                ``[batch, seq, num_heads * head_dim]`` (post-merge) or
                ``[batch, seq, hidden]`` (post-output-projection).
            pre_projection: Whether this is the constraint BEFORE the output
                projection.  That tensor is the projection's TP-contracted
                operand, not the residual stream — inside a decode-replicated
                scope forcing the (replicated) hidden layout onto it inserts a
                wasted head all-gather right before o_proj re-slices, so the
                constraint is skipped and the kernel's head sharding flows
                into the projection directly.

        Returns:
            ``attn_output`` re-constrained to the configured hidden-state
            sharding spec (or unchanged for the pre-projection tensor in a
            decode-replicated scope).
        """
        if pre_projection and inference_mode_forces_decode():
            return attn_output
        return tp.cast(
            JArray,
            apply_logical_sharding(
                x=attn_output,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            ),
        )

    def _merge_heads(self, hidden_states: Float[JArray, "batch seq heads dim"]) -> Float[JArray, "batch seq hidden"]:
        """Collapse the head and head-dim axes into a single hidden axis.

        Reshape ``[batch, seq, num_heads, head_dim]`` -> ``[batch, seq,
        num_heads * head_dim]``. The leading two axes are preserved
        verbatim so the operation is shape-polymorphic.

        Args:
            hidden_states: Attention output with separate head/dim axes.

        Returns:
            Hidden states with the head axes merged.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], -1))

    @staticmethod
    def repeat_key_value(
        key: Float[JArray, "batch seq kv_heads dim"], value: Float[JArray, "batch seq kv_heads dim"], num_reps: int
    ) -> tuple[Float[JArray, "batch seq heads dim"], Float[JArray, "batch seq heads dim"]]:
        """Broadcast K/V along the head axis for Grouped-Query Attention.

        Each KV head is repeated ``num_reps`` times so that ``num_kv_heads
        * num_reps == num_attention_heads`` and the kernel can compute
        attention as if it were vanilla MHA. Used by kernels that do not
        implement GQA broadcasting internally.

        Args:
            key: KV-head key tensor ``[batch, seq, num_kv_heads, dim]``.
            value: KV-head value tensor ``[batch, seq, num_kv_heads, dim]``.
            num_reps: Repeat factor; equal to
                ``num_attention_heads // num_kv_heads``.

        Returns:
            Tuple ``(key_expanded, value_expanded)`` each of shape
            ``[batch, seq, num_kv_heads * num_reps, dim]``.
        """
        with jax.named_scope("easydel-attention-repeat-kvheads"):
            key = einops.repeat(key, "b s h d -> b s (h r) d", r=num_reps)
            value = einops.repeat(value, "b s h d -> b s (h r) d", r=num_reps)
        return key, value


def block_diagonal_bias(cu_seqlens, seq_length: int, dtype, *, add_head_axis: bool = False):
    """Additive block-diagonal attention bias from cumulative sequence lengths.

    Vision towers pack several images into one flat token sequence and rely on
    an additive mask to stop attention crossing image boundaries. Given
    ``cu_seqlens`` (the usual exclusive-prefix offsets, ``[0, n0, n0+n1, ...]``),
    positions inside the same segment get ``0.0`` and everything else gets
    ``finfo(dtype).min``.

    Four families spelled this identically under three different names
    (``create_attention_mask``, ``_create_block_diagonal_bias``,
    ``_block_diagonal_bias``).

    Args:
        cu_seqlens: Cumulative sequence lengths, shape ``(num_segments + 1,)``.
        seq_length: Total packed sequence length.
        dtype: Output dtype; also selects the mask-out sentinel.
        add_head_axis: Return a four-dimensional, head-broadcast bias when True.

    Returns:
        Additive bias of shape ``(1, seq_length, seq_length)``, or
        ``(1, 1, seq_length, seq_length)`` when ``add_head_axis=True``.
    """
    positions = jnp.arange(seq_length)
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    in_segment = (positions[:, None] >= starts[None, :]) & (positions[:, None] < ends[None, :])
    valid_position = jnp.any(in_segment, axis=-1)
    segment_ids = jnp.argmax(in_segment.astype(jnp.int32), axis=-1)
    same_segment = (segment_ids[:, None] == segment_ids[None, :]) & valid_position[:, None] & valid_position[None, :]
    attention_mask = jnp.where(same_segment, 0.0, jnp.finfo(dtype).min).astype(dtype)
    return attention_mask[None, None, :, :] if add_head_axis else attention_mask[None, :, :]
