# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

Constants:
    DEFAULT_ATTENTION_MECHANISM: Default attention mechanism ("auto")

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
from enum import Enum
from functools import cached_property, partial

import einops
import flax.nnx as nn
import jax
from chex import Array
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo
from jax import NamedSharding, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.sharding import PartitionSpec
from jaxtyping import Array as JArray
from jaxtyping import Bool, Complex, Float, Int

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.utils import AttnMaskDetail, AttnMaskType

from .caching import RaggedPagesCacheView, RaggedPagesMetadata, TransformerCacheView, TransformerMetadata
from .operations import AttentionOutput, OperationMetadata, OperationRegistry
from .quantization.quantizers import EasyQuantizer

logger = get_logger(__name__)


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
    dtype_mapping = {
        "<class 'jax.numpy.float32'>": jnp.float32,
        "<class 'jax.numpy.float64'>": jnp.float64,
        "<class 'jax.numpy.int32'>": jnp.int32,
        "<class 'jax.numpy.int64'>": jnp.int64,
        "<class 'jax.numpy.bool_'>": jnp.bool_,
        "<class 'jax.numpy.complex64'>": jnp.complex64,
        "<class 'jax.numpy.complex128'>": jnp.complex128,
    }
    return dtype_mapping.get(dtype_string, dtype_string)


class AttentionMechanisms(str, Enum):
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
        RAGGED_PAGE_ATTENTION: Paged attention for efficient inference.
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
    RAGGED_PAGE_ATTENTION: str = "ragged_page_attention"
    PAGED_ATTENTION: str = "page_attention"
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
    if version in getattr(jax.local_devices()[0], "device_kind", "").lower():
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

    match jax.default_backend():
        case "tpu":
            if tpu_version_check("v3"):
                return AttentionMechanisms.FLASH_ATTN2, jnp.float32
            return AttentionMechanisms.BLOCKSPARSE, jnp.bfloat16
        case "gpu":
            return (AttentionMechanisms.FLASH_ATTN2, jnp.bfloat16)
        case _:
            return AttentionMechanisms.VANILLA, jnp.bfloat16


DEFAULT_ATTENTION_MECHANISM = "auto"


class FlexibleAttentionModule(nn.Module):
    """Unified interface for various attention mechanisms.

    Central hub for managing different attention implementations,
    automatically selecting and executing the optimal mechanism
    based on hardware, configuration, and runtime requirements.

    Supports optimized implementations like FlashAttention, SplashAttention,
    RingAttention, and standard dot-product attention. Provides automatic
    hardware detection and optimization selection.

    Attributes:
        config: Model configuration with attention parameters.
        dtype: Data type for computations.
        param_dtype: Data type for parameters.
        precision: Precision setting for operations.
        attention_mechanism: Selected attention mechanism.
        mesh: JAX mesh for distributed computation.
        implementation: Concrete attention implementation.

    Key Features:

    * **Attention Mechanism Selection:** Supports a wide range of attention mechanisms,
      allowing users to choose the most suitable option based on performance and hardware constraints.
    * **Sharding and Partitioning:** Integrates with JAX's sharding capabilities, enabling efficient
      distribution of computations and data across multiple devices.
    * **Block-wise Computation:** Implements block-wise attention computations for optimized memory
      usage and speed, particularly beneficial for large models.
    * **Performance Optimization:** Includes support for highly optimized implementations like
      FlashAttention, SplashAttention, and RingAttention for TPU and GPU acceleration.
    * **Flexibility and Customization:** Offers fine-grained control over attention parameters,
      sharding specifications, and block sizes, providing flexibility for different use cases.
    * **Testing and Evaluation:** Includes a `run_attention_benchmarks` method to systematically evaluate
      different attention mechanisms and help users identify the best-performing option.


    The AttentionModule class is a crucial component within EasyDeL, responsible for managing and optimizing attention
    computations. It provides a user-friendly way to select and execute different attention mechanisms,
    leveraging JAX's sharding capabilities and offering performance enhancements through specialized implementations
    like FlashAttention and SplashAttention. Its ability to handle block-wise computations and customization options
    makes it adaptable to a variety of model architectures and hardware configurations.
    Attributes:
      impl (AttentionBackend): The chosen attention implementation backend instance.
      deterministic (bool): Flag indicating whether dropout should be applied (False) or not (True).
                            Currently hardcoded to True.
      metadata (OperationMetadata): Metadata derived from the configuration, used by the backend.
    """

    def __init__(
        self,
        base_config: EasyDeLBaseConfig,
        softmax_scale: float,
        dropout_prob: float = 0.0,
        *,
        rngs: nn.Rngs | None = None,
    ):
        """
        Initializes the AttentionModule.

        Args:
            base_config (EasyDeLBaseConfig): Configuration object containing attention settings
                                             (mechanism, dtype, sharding, etc.).
            softmax_scale (float): The scaling factor to apply before the softmax function.
            dropout_prob (float, optional): The dropout probability for attention weights.
                                             Defaults to 0.0.
        """
        if rngs is None:
            rngs = nn.Rngs(42)
        if isinstance(base_config.attn_dtype, str):
            base_config.attn_dtype = _get_jax_dtype_from_string(base_config.attn_dtype)
        if isinstance(base_config.attn_softmax_dtype, str):
            base_config.attn_softmax_dtype = _get_jax_dtype_from_string(base_config.attn_softmax_dtype)
        if base_config.attn_mechanism == AttentionMechanisms.AUTO:
            impl_name, runtime_dtype = get_optimal_config()
            logger.debug(f"Automatically select OperationImpl {impl_name} | {runtime_dtype}")
            base_config.attn_mechanism = impl_name
            base_config.attn_dtype = runtime_dtype

        metadata = OperationMetadata.from_config(config=base_config)
        self.config = base_config
        self.metadata = metadata
        self.rngs = rngs
        self.softmax_scale = softmax_scale
        self.dropout_prob = dropout_prob
        self.impl = OperationRegistry.create(impl_name=base_config.attn_mechanism, metadata=metadata)
        self.deterministic = True
        self.impl_decode = None
        if base_config.decode_attn_mechanism is not None:
            self.impl_decode = OperationRegistry.create(impl_name=base_config.decode_attn_mechanism, metadata=metadata)

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
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        init_bias: tp.Callable[[], Float[JArray, "batch heads seq_q seq_k"]] | None = None,
        causal: bool = True,
        softmax_aux: Float[JArray, "..."] | None = None,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        dropout_prob: float | None = None,
        dropout_rng: tp.Any | None = None,
        deterministic: bool | None = None,
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
    ) -> AttentionOutput:
        """
        Performs the attention computation using the selected backend implementation.

        Args:
            query_states (Array): Query tensor [batch, seq_q, heads, dim].
            key_states (Array): Key tensor [batch, seq_k, heads, dim].
            value_states (Array): Value tensor [batch, seq_v, heads, dim].
            mode (common_types.RUNTIME_MODE_TYPES): Runtime mode (TRAIN, PREFILL, DECODE).
            mask_info (MaskInfo, optional): Container for attention masks and segment IDs. Defaults to None.
                If provided, contains:
                - attention_mask: Boolean mask [batch, 1, seq_q, seq_k]
                - q_segment_ids: Query segment IDs [batch, seq_q]
                - kv_segment_ids: Key/Value segment IDs [batch, seq_k]
                - q_positions: Query position indices [batch, seq_q]
                - kv_positions: Key/Value position indices [batch, seq_k]
            bias (Array, optional): Optional attention bias [batch, heads, seq_q, seq_k]. Defaults to None.
            sliding_window (int | tuple[int, int], optional): Sliding window size for attention. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Cache metadata. Defaults to None.
            cache_view (TransformerCacheView | RaggedPagesCacheView, optional): View into KV cache. Defaults to None.
            init_bias (Callable, optional): Function to initialize bias tensor. Defaults to None.
            causal (bool, optional): Apply causal masking. Defaults to True.
            softmax_aux (Array, optional): Auxiliary tensor for softmax (e.g., sink tokens). Defaults to None.
            softmax_scale (float, optional): Scaling factor for attention logits. Defaults to None.
            logits_soft_cap (float, optional): Soft capping value for attention logits. Defaults to None.
            dropout_prob (float, optional): Dropout probability for attention weights. Defaults to None.
            dropout_rng (PRNGKey, optional): PRNG key for dropout. Defaults to None.
            deterministic (bool, optional): If True, disables dropout. Defaults to None (uses self.deterministic).
            precision (lax.PrecisionLike, optional): JAX precision setting. Defaults to None.
            prevent_cse (bool, optional): Prevent common subexpression elimination. Defaults to True.
            cum_seqlens_q (Array, optional): Cumulative sequence lengths for queries. Defaults to None.
            cum_seqlens_k (Array, optional): Cumulative sequence lengths for keys. Defaults to None.
            normalize_output (bool, optional): Normalize attention output. Defaults to True.
            fused_backward (bool, optional): Use fused backward pass. Defaults to False.
            compute_dtype (jnp.dtype, optional): Computation dtype. Defaults to None.
            optimized (bool, optional): Use optimized kernel variant. Defaults to False.
            mask_value (float, optional): Value for masked positions. Defaults to None.
            vmem_limit_bytes (int, optional): VMEM limit in bytes for paged attention. Defaults to None.
            policy (Any, optional): Checkpoint policy for gradients. Defaults to None.

        Returns:
            AttentionOutput: An object containing the attention output tensor and potentially
                             attention weights (depending on the backend).
        """
        if isinstance(cache_view, RaggedPagesCacheView):
            assert self.config.attn_mechanism == AttentionMechanisms.RAGGED_PAGE_ATTENTION
        # NOTE: Attention Dropout is disabled for now.
        # try:
        #     rngs = self.rngs()
        # except flax.errors.TraceContextError:
        #     rngs = None

        # Use provided dropout_rng or rngs
        # if dropout_rng is None:
        dropout_rng = None

        # Use provided deterministic or self.deterministic
        if deterministic is None:
            deterministic = self.deterministic

        # Use provided softmax_scale or self.softmax_scale
        if softmax_scale is None:
            softmax_scale = self.softmax_scale

        # Use provided dropout_prob or self.dropout_prob
        if dropout_prob is None:
            dropout_prob = self.dropout_prob

        # Use provided precision or default
        if precision is None:
            precision = lax.Precision.DEFAULT

        # Use provided policy or default
        if policy is None:
            policy = jax.checkpoint_policies.nothing_saveable

        with self.config.mesh:
            input_dict = dict(
                query=query_states,
                key=key_states,
                value=value_states,
                mask_info=mask_info,
                bias=bias,
                sliding_window=sliding_window,
                cache_metadata=cache_metadata,
                cache_view=cache_view,
                init_bias=init_bias,
                causal=causal,
                deterministic=deterministic,
                dropout_rng=dropout_rng,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale,
                logits_soft_cap=logits_soft_cap,
                dropout_prob=dropout_prob,
                precision=precision,
                prevent_cse=prevent_cse,
                cum_seqlens_q=cum_seqlens_q,
                cum_seqlens_k=cum_seqlens_k,
                normalize_output=normalize_output,
                fused_backward=fused_backward,
                compute_dtype=compute_dtype,
                optimized=optimized,
                mask_value=mask_value,
                vmem_limit_bytes=vmem_limit_bytes,
                policy=policy,
            )
            if mode == common_types.MODE_DECODE:
                assert cache_view is not None
                callable_attn = self.impl if self.impl_decode is None else self.impl_decode
                output = callable_attn(**input_dict)
            else:
                output = self.impl(**input_dict)

        return jtu.tree_map(lambda x: x.astype(self.impl.metadata.runtime_dtype), output)

    __call__ = forward


SC = tp.TypeVar("SC")
"""Type variable for configuration objects."""


class AttentionModule(nn.Module):
    """
    Base class for Flax attention modules in EasyDeL, providing common utilities.

    This class offers helper functions and attributes commonly needed by attention
    implementations within Flax, such as handling KV caching, sharding, mask manipulation,
    and head manipulation. Concrete attention implementations often inherit from this class.

    Attributes:
        config (SC | EasyDeLBaseConfig): Configuration object for the attention module.
        cached_key (nn.Cache[Array] | None): Flax Cache for storing past key states (wont be used).
        cached_value (nn.Cache[Array] | None): Flax Cache for storing past value states (wont be used).
        cache_index (nn.Cache[Array] | None): Flax Cache for tracking the current index in the cache (wont be used).
    """

    def __init__(self, config: SC):  # type:ignore
        """
        Initializes the AttentionModule.

        Args:
            config (SC): The configuration object for this attention module.
                         It should conform to or include attributes from EasyDeLBaseConfig.
        """
        super().__init__()
        self.config: SC | EasyDeLBaseConfig = config

        self.cached_key: nn.Cache[Array] | None = None
        self.cached_value: nn.Cache[Array] | None = None
        self.cache_index: nn.Cache[Array] | None = None

    @staticmethod
    def apply_complex_rotary(
        xq: Float[JArray, "... seq heads dim"],
        xk: Float[JArray, "... seq heads dim"],
        freqs_cis: Complex[JArray, "batch seq 1 dim_2"],
    ) -> tuple[Float[JArray, "... seq heads dim"], Float[JArray, "... seq heads dim"]]:
        xq_reshaped = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
        xk_reshaped = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
        xq_complex = xq_reshaped[..., 0] + 1j * xq_reshaped[..., 1]
        xk_complex = xk_reshaped[..., 0] + 1j * xk_reshaped[..., 1]
        xq_out_complex = xq_complex * freqs_cis[:, :, None, :]
        xk_out_complex = xk_complex * freqs_cis[:, :, None, :]
        xq_out_real = jnp.stack(
            [jnp.real(xq_out_complex), jnp.imag(xq_out_complex)],
            axis=-1,
        )
        xk_out_real = jnp.stack(
            [jnp.real(xk_out_complex), jnp.imag(xk_out_complex)],
            axis=-1,
        )
        xq_out = xq_out_real.reshape(*xq_out_real.shape[:-2], -1)
        xk_out = xk_out_real.reshape(*xk_out_real.shape[:-2], -1)
        xq_out = xq_out.astype(xq.dtype)
        xk_out = xk_out.astype(xk.dtype)
        return xq_out, xk_out

    def apply_qk_shardings(
        self,
        q: Float[JArray, "batch seq heads dim"],
        k: Float[JArray, "batch seq heads dim"],
    ) -> tuple[Float[JArray, "batch seq heads dim"], Float[JArray, "batch seq heads dim"]]:
        q = apply_logical_sharding(
            q,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.partition_manager,
        )
        k = apply_logical_sharding(
            k,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.partition_manager,
        )
        return q, k

    def apply_qkv_shardings(
        self,
        q: Float[JArray, "batch seq heads dim"],
        k: Float[JArray, "batch seq heads dim"],
        v: Float[JArray, "batch seq heads dim"],
    ) -> tuple[
        Float[JArray, "batch seq heads dim"], Float[JArray, "batch seq heads dim"], Float[JArray, "batch seq heads dim"]
    ]:
        q = apply_logical_sharding(
            q,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.partition_manager,
        )
        k = apply_logical_sharding(
            k,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.partition_manager,
        )
        v = apply_logical_sharding(
            v,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.partition_manager,
        )
        return q, k, v

    @staticmethod
    def build_cache_pos(
        attention_mask: Bool[JArray, "batch heads seq_q seq_k"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView = None,
    ) -> Int[JArray, "batch seq"]:
        """
        Calculates the position indices within the sequence for cache-aware operations.

        Args:
            attention_mask (jax.Array): The attention mask (typically [batch, heads, q_len, k_len]).
            mode (common_types.RUNTIME_MODE_TYPES): The runtime mode.
            cache_view (TransformerCacheView, optional): The current KV cache view. Defaults to None.

        Returns:
            jax.Array: An array representing the position of each token in the sequence,
                       adjusted by the cache index if provided. Shape usually [batch, q_len].
        """
        end_index = 0
        if isinstance(cache_view, TransformerCacheView) and mode == common_types.MODE_DECODE:
            end_index = jnp.reshape(cache_view.indexs, (-1, 1))
        inipos = jnp.cumsum(jnp.any(attention_mask, -1)[:, -1, :], axis=-1)
        return (inipos - (inipos >= 1)) + end_index

    @cached_property
    def quantizer(self):
        """
        Provides an EasyQuantizer instance based on the module's configuration.

        Used for quantizing KV cache entries if enabled in the config.

        Returns:
            EasyQuantizer: The quantizer instance.
        """
        return EasyQuantizer(
            quantization_method=self.config.kv_cache_quantization_method,
            block_size=self.config.kv_cache_quantization_blocksize,
        )

    @property
    def default_key_value_sharding(self):
        """
        Defines the default JAX sharding for key and value tensors.

        Uses the partition specifications defined in the configuration's `partition_axis`.

        Returns:
            NamedSharding: The default sharding configuration for K/V tensors.
        """
        paxis = self.config.partition_axis
        return NamedSharding(
            mesh=self.config.mesh,
            spec=PartitionSpec(
                paxis.batch_axis,
                paxis.key_sequence_axis,
                paxis.head_axis,
                paxis.attention_dim_axis,
            ),
        )

    def get_sharding_safely(self, tensor: Float[JArray, "..."]) -> PartitionSpec:
        """
        Retrieves the PartitionSpec of a tensor, falling back to the default KV sharding.

        Args:
            tensor (jax.Array): The tensor whose sharding spec is needed.

        Returns:
            PartitionSpec: The sharding specification of the tensor.
        """
        return getattr(tensor, "sharding", self.default_key_value_sharding).spec

    @staticmethod
    def _transpose_sequence_head(*args):
        """
        Transposes the sequence and head dimensions of input tensors.

        Typically used to change tensors from [Batch, Seq, Heads, Dim] to
        [Batch, Heads, Seq, Dim] or vice-versa.

        Args:
            *args: A variable number of arrays, each expected to have at least 4 dimensions.

        Returns:
            map: A map object yielding the transposed arrays.
        """
        return map(lambda x: jnp.transpose(x, (0, 2, 1, 3)), args)

    def _handle_cache_concat(
        self,
        query: Float[JArray, "batch seq_q heads dim"],
        key: Float[JArray, "batch seq_k heads dim"],
        value: Float[JArray, "batch seq_v heads dim"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        mask_info: MaskInfo,
        cache_view: TransformerCacheView | None,
        cache_metadata: TransformerMetadata | None,
    ) -> tuple[
        Float[JArray, "batch seq_k heads dim"],
        Float[JArray, "batch seq_v heads dim"],
        MaskInfo,
        TransformerCacheView | None,
        AttnMaskDetail | None,
    ]:
        """Handles concatenation of current KV states to the cache."""
        if cache_view is None:
            return key, value, mask_info, None, None
        key, value, mask_info, cache_view, masking_details = cache_view.concatenate_to_cache(
            query=query,
            key=key,
            value=value,
            mode=mode,
            quantizer=self.quantizer,
            mask_info=mask_info,
            cache_metadata=cache_metadata,
            partition_manager=self.config.partition_manager,
        )

        return key, value, mask_info, cache_view, masking_details

    def _apply_sliding_window(
        self,
        key: Array,
        value: Array,
        mask_info: MaskInfo,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | None,
        sliding_window: int,
        query_length: int,
        masking_details: AttnMaskDetail | None,
        cache_metadata: TransformerMetadata | None,
    ) -> tuple[Array, Array, Array]:
        """Applies sliding window masking and slicing to KV and mask."""

        # offsets = jnp.zeros((key.shape[0],), "i4")
        krange = key.shape[1] if masking_details is None else mask_info.kv_len
        # if mode == common_types.MODE_DECODE and cache_view is not None:
        #     indexs = cache_view.indexs
        #     # offsets = indexs - 1
        # elif mode == common_types.MODE_PREFILL and masking_details is not None:
        #     indexs = jnp.array([query_length], "i4").repeat(key.shape[0], axis=0).reshape(-1)
        # else:
        #     indexs = jnp.zeros((key.shape[0],), "i4") + 1

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0, 0))
        def _select_slices(ikey, ival, imsk, offset, index):
            row_ids_sliding = offset + jax.lax.broadcasted_iota(jnp.int32, (query_length, 1), 0)
            col_ids_sliding = jax.lax.broadcasted_iota(jnp.int32, (1, krange), 1)
            rhm = col_ids_sliding <= row_ids_sliding
            lhm = col_ids_sliding > (row_ids_sliding - sliding_window)

            imsk = (lhm & rhm) & imsk.astype(jnp.bool_)
            if mode == common_types.MODE_DECODE and masking_details is not None:
                start_index = jax.lax.max(0, index - sliding_window)
                if imsk.shape[-1] > sliding_window:
                    imsk = jax.lax.dynamic_slice_in_dim(imsk, start_index, sliding_window, 2)
                if ikey.shape[0] > sliding_window:
                    ikey = jax.lax.dynamic_slice_in_dim(ikey, start_index, sliding_window, 0)
                    ival = jax.lax.dynamic_slice_in_dim(ival, start_index, sliding_window, 0)
            elif mode == common_types.MODE_PREFILL:
                imsk = jax.lax.dynamic_slice_in_dim(imsk, max(0, query_length - sliding_window), sliding_window, 2)

            return ikey, ival, imsk

        # key, value, attention_mask = _select_slices(key, value, attention_mask, offsets, indexs)

        # if cache_metadata is not None and mode == common_types.MODE_DECODE:
        #     passed = cache_metadata.indexs - cache_metadata.starts
        #     cache_metadata = TransformerMetadata(
        #         starts=jax.lax.max(0, sliding_window - passed),
        #         indexs=jnp.full((attention_mask.shape[0],), attention_mask.shape[-1]),
        #     )

        return key, value, mask_info, cache_metadata

    @jax.named_scope("easydel-flax-attention-concatenate")
    def concatenate(
        self,
        *,
        query: Array,
        key: Array,
        value: Array,
        mask_info: MaskInfo,
        mode: common_types.RUNTIME_MODE_TYPES | common_types.EMPTY_VAL = common_types.NOT_GIVEN,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        sliding_window: int | None = None,
    ) -> tuple[
        Array,
        Array,
        Array,
        tp.Callable[[], Array],
        TransformerCacheView | RaggedPagesCacheView | None,
        TransformerMetadata | RaggedPagesMetadata | None,
    ]:
        """
        Prepares inputs for attention calculation, handling KV caching and mask merging.

        This function combines the current query, key, and value with cached states (if applicable),
        merges various masks (attention, causal, FCM, sliding window), and returns the final
        key, value, attention mask, and a function to initialize the attention bias.

        Args:
            query (Array): Current query states [Batch, q_len, Heads, Dim].
            key (Array): Current key states [Batch, kv_len, Heads, Dim].
            value (Array): Current value states [Batch, kv_len, Heads, Dim].
            attention_mask (Array): Base attention mask (e.g., padding mask) [Batch, kv_len] or compatible.
            mode (common_types.RUNTIME_MODE_TYPES): The runtime mode (TRAIN, PREFILL, DECODE). Required.
            cache_view (tp.Optional[TransformerCacheView | RaggedPagesCacheView], optional): View into the KV cache.
                If None, caching is disabled. Defaults to None.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata], optional): Cache metadata.
                Defaults to None.
            causal_mask (tp.Optional[Array], optional): Causal mask [1, 1, q_len, kv_len]. Defaults to None.
                Defaults to None.
            fcm_mask (tp.Optional[Array], optional): Fused-Context-Mask [Batch, 1, q_len, kv_len].
                Defaults to None.
            sliding_window (tp.Optional[int], optional): Size of the sliding attention window. If None, not applied.
                Defaults to None.

        Returns:
            tp.Tuple[Array, Array, Array, tp.Callable[[], Array], tp.Optional[tp.Union[TransformerCacheView, RaggedPagesCacheView]]]:
                - key_states (Array): Final key states (potentially from cache).
                - value_states (Array): Final value states (potentially from cache).
                - attention_mask (Array): The final combined attention mask [Batch, Heads, q_len, kv_len].
                - init_attention_bias (Callable): Function to create the attention bias tensor.
                - updated_cache_view: The updated cache view (or None if no cache).
                - updated_cache_metadata: The updated cache metadata (or None if no metadata).

        Raises:
            ValueError: If shapes are mismatched.
        """
        # TODO:MOVEIT
        # assert mask_info.shape[-1] >= key.shape[1], "Attention mask length must match KV sequence length."

        query_length, initial_key_length = query.shape[1], key.shape[1]

        if isinstance(mode, common_types.EMPTY_VAL):
            if cache_view is None:
                mode = common_types.MODE_TRAIN
            else:
                mode = common_types.MODE_PREFILL if query_length != 1 else common_types.MODE_DECODE

        if isinstance(cache_view, RaggedPagesCacheView):
            cache_view = cache_view.concatenate_to_cache(key=key, value=value, cache_metadata=cache_metadata)

            def init_attention_bias():
                return jnp.zeros((query.shape[0], 1, query_length, initial_key_length), dtype=self.dtype)

            return key, value, mask_info, init_attention_bias, cache_view, cache_metadata

        if cache_view is not None:
            assert query.shape[0] == cache_view.key.shape[0], "Batch size mismatch between query and cache."
        key, value, mask_info, cache_view, masking_details = self._handle_cache_concat(
            query=query,
            key=key,
            value=value,
            mode=mode,
            mask_info=mask_info,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        if cache_metadata is None and cache_view is not None:
            cache_metadata = TransformerMetadata(starts=cache_view.starts, indexs=cache_view.indexs)
        if sliding_window is not None or (
            cache_view is not None
            and cache_view.masking_details is not None
            and cache_view.masking_details.mask_type == AttnMaskType.SLIDING
        ):
            masking_details = cache_view.masking_details if isinstance(cache_view, TransformerCacheView) else None
            if masking_details and masking_details.mask_type == AttnMaskType.SLIDING:
                sliding_window = sliding_window or masking_details.size
            key, value, mask_info, cache_metadata = self._apply_sliding_window(
                key=key,
                value=value,
                mask_info=mask_info,
                mode=mode,
                cache_view=cache_view,
                sliding_window=sliding_window,
                query_length=query_length,
                masking_details=masking_details,
                cache_metadata=cache_metadata,
            )

        def init_attention_bias():
            return mask_info.create_bias(self.dtype)

        return key, value, mask_info, init_attention_bias, cache_view, cache_metadata

    def shard_attention_prod(
        self, attn_output: Float[JArray, "batch seq heads dim"]
    ) -> Float[JArray, "batch seq heads dim"]:
        """
        Applies sharding constraints to the attention output tensor.

        This is typically done before projecting the attention output back to the
        hidden dimension size.

        Args:
            attn_output (jax.Array): The output from the attention mechanism, usually
                                     with shape [Batch, SeqLen, NumHeads * DimPerHead].

        Returns:
            jax.Array: The input tensor with applied sharding constraints based on the config.
        """
        return apply_logical_sharding(
            x=attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

    def _merge_heads(self, hidden_states: Float[JArray, "batch seq heads dim"]) -> Float[JArray, "batch seq hidden"]:
        """
        Merges the attention heads into a single hidden state tensor.

        Reshapes [Batch, SeqLen, NumHeads, DimPerHead] -> [Batch, SeqLen, NumHeads * DimPerHead].

        Args:
            hidden_states (jax.Array): The hidden states with separate head dimensions.

        Returns:
            jax.Array: The hidden states with merged head dimensions.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], -1))

    @staticmethod
    def repeat_key_value(
        key: Float[JArray, "batch seq kv_heads dim"], value: Float[JArray, "batch seq kv_heads dim"], num_reps: int
    ) -> tuple[Float[JArray, "batch seq heads dim"], Float[JArray, "batch seq heads dim"]]:
        """
        Repeats key and value tensors for Grouped Query Attention (GQA).

        Expands the head dimension by repeating `num_reps` times.
        Uses einops for concise repetition.

        Args:
            key (Array): Key tensor [Batch, Seq, NumKVHeads, Dim].
            value (Array): Value tensor [Batch, Seq, NumKVHeads, Dim].
            num_reps (int): The number of times to repeat each KV head (num_attention_heads / num_kv_heads).

        Returns:
            tp.Tuple[Array, Array]: Repeated key and value tensors, each with shape
                                    [Batch, Seq, NumKVHeads * num_reps, Dim].
        """
        with jax.named_scope("easydel-flax-attention-repeat-kvheads"):
            key = einops.repeat(key, "b s h d -> b s (h r) d", r=num_reps)
            value = einops.repeat(value, "b s h d -> b s (h r) d", r=num_reps)
        return key, value
