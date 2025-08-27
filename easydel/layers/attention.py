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
import warnings
from enum import Enum
from functools import cached_property, partial

import einops
import flax
import flax.errors
import flax.nnx as nn
import jax
from chex import Array
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
from jax import NamedSharding, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.sharding import PartitionSpec
from jaxtyping import Array as JArray
from jaxtyping import Bool, Complex, Float, Int

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.utils import AttnMaskDetail, AttnMaskType

from .attention_operator import AttentionMetadata, AttentionOutput, AttentionRegistry
from .caching import PagesCacheView, PagesMetadata, TransformerCacheView, TransformerMetadata
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

    AUTO = "auto"
    FLASH_ATTN2 = "flash_attn2"
    RING = "ring"
    VANILLA = "vanilla"
    SPLASH = "splash"
    CUDNN = "cudnn"
    BLOCKWISE = "blockwise"
    SDPA = "sdpa"
    CUDA_FLASH_ATTN2 = "cuda_flash_attn2"
    RAGGED_PAGE_ATTENTION = "ragged_page_attention"
    PAGED_ATTENTION = "ragged_page_attention"
    REGRESSIVE_DECODE = "autoregressive_decodeattn"


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
            return AttentionMechanisms.SPLASH, jnp.bfloat16
        case "gpu":
            return (AttentionMechanisms.FLASH_ATTN2, jnp.float16)
            # float16 is better for flash attention
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
      metadata (AttentionMetadata): Metadata derived from the configuration, used by the backend.
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
            logger.debug(f"Automatically select AttentionImpl {impl_name} | {runtime_dtype}")
            base_config.attn_mechanism = impl_name
            base_config.attn_dtype = runtime_dtype

        metadata = AttentionMetadata.from_config(
            config=base_config,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
        )
        self.config = base_config
        self.metadata = metadata
        self.rngs = rngs
        self.impl = AttentionRegistry.create(impl_name=base_config.attn_mechanism, metadata=metadata)
        self.deterministic = True
        self.impl_decode = None
        if base_config.decode_attn_mechanism is not None:
            self.impl_decode = AttentionRegistry.create(
                impl_name=base_config.decode_attn_mechanism,
                metadata=metadata,
            )

    @jax.named_scope("easydel-flexible-attention")
    def forward(
        self,
        query_states: Float[JArray, "batch seq_q heads dim"],
        key_states: Float[JArray, "batch seq_k heads dim"],
        value_states: Float[JArray, "batch seq_v heads dim"],
        mode: common_types.RUNTIME_MODE_TYPES | None,  # type:ignore
        bias: Float[JArray, "batch heads seq_q seq_k"] | None = None,
        sliding_window: int | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        init_bias: tp.Callable[[], Float[JArray, "batch heads seq_q seq_k"]] | None = None,
        attention_mask: Bool[JArray, "batch seq_q seq_k"] | None = None,
        segment_ids: Int[JArray, "batch seq"] | None = None,
        causal: bool = True,
        softmax_aux: Float[JArray, "..."] | None = None,
    ) -> AttentionOutput:
        """
        Performs the attention computation using the selected backend implementation.

        Args:
            query_states (Array): Query tensor.
            key_states (Array): Key tensor.
            value_states (Array): Value tensor.
            bias (tp.Optional[Array], optional): Optional attention bias. Defaults to None.
            init_bias (tp.Optional[tp.Callable[[], Array]], optional): Optional function to initialize bias.
                                                                       Defaults to None.
            attention_mask (tp.Optional[Array], optional): Mask to prevent attention to certain positions.
                                                            Defaults to None.
            segment_ids (tp.Optional[Array], optional): Segment IDs for segment-based attention (RingAttention).
                                                        Defaults to None.
            causal (bool, optional): If True, applies a causal mask. Defaults to True.
            dropout_rng (tp.Optional[random.PRNGKey], optional): PRNG key for dropout. Defaults to None.

        Returns:
            AttentionOutput: An object containing the attention output tensor and potentially
                             attention weights (depending on the backend).
        """
        if isinstance(cache_view, PagesCacheView):
            assert self.config.attn_mechanism == AttentionMechanisms.RAGGED_PAGE_ATTENTION
        try:
            rngs = self.rngs()
        except flax.errors.TraceContextError:
            rngs = None
        with self.config.mesh:
            input_dict = dict(
                q=query_states,
                k=key_states,
                v=value_states,
                bias=bias,
                sliding_window=sliding_window,
                cache_metadata=cache_metadata,
                cache_view=cache_view,
                init_bias=init_bias,
                mask=attention_mask,
                segment_ids=segment_ids,
                causal=causal,
                deterministic=self.deterministic,
                dropout_rng=rngs,
                softmax_aux=softmax_aux,
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
        attention_mask: Bool[JArray, "batch seq_q seq_k"],
        cache_view: TransformerCacheView | None,
        cache_metadata: TransformerMetadata | None,
        causal_mask: Bool[JArray, "batch heads seq_q seq_k"] | None,
        token_type_ids: Int[JArray, "batch seq"] | None,
    ) -> tuple[
        Float[JArray, "batch seq_k heads dim"],
        Float[JArray, "batch seq_v heads dim"],
        Bool[JArray, "batch seq_q seq_k"],
        TransformerCacheView | None,
        AttnMaskDetail | None,
    ]:
        """Handles concatenation of current KV states to the cache."""
        if cache_view is None:
            return key, value, attention_mask, None, None
        key, value, attention_mask, cache_view, masking_details = cache_view.concatenate_to_cache(
            query=query,
            key=key,
            value=value,
            mode=mode,
            quantizer=self.quantizer,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            cache_metadata=cache_metadata,
            token_type_ids=token_type_ids,
            partition_manager=self.config.partition_manager,
        )

        return key, value, attention_mask, cache_view, masking_details

    def _merge_masks(
        self,
        attention_mask: Bool[JArray, "... seq_q seq_k"],
        causal: bool,
        causal_mask: Bool[JArray, "batch heads seq_q seq_k"] | None,
        token_type_ids: Int[JArray, "batch seq"] | None,
        fcm_mask: Bool[JArray, "batch heads seq_q seq_k"] | None,
        query_length: int,
        initial_key_length: int,
        cache_view: TransformerCacheView | None,
    ) -> Bool[JArray, "batch heads seq_q seq_k"]:
        """Merges attention, causal, token-type, and FCM masks."""
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        if cache_view is None:
            if causal:
                target_length = initial_key_length
                if isinstance(causal_mask, bool) or causal_mask is None:
                    causal_mask = self.config._create_causal_mask(target_length=target_length)
                causal_mask = causal_mask[:, :, :query_length, :initial_key_length]
                causal_mask = jnp.broadcast_to(causal_mask, (attention_mask.shape[0], *causal_mask.shape[1:]))
                if token_type_ids is not None and query_length != 1:
                    token_type_mask = jnp.equal(jnp.expand_dims(token_type_ids, 2), jnp.expand_dims(token_type_ids, 1))
                    token_type_mask = jnp.where(
                        jnp.broadcast_to(jnp.expand_dims(token_type_ids == 0, -1), token_type_mask.shape),
                        False,
                        token_type_mask,
                    )
                    token_type_mask = jnp.expand_dims(token_type_mask, 1)
                    sequence_length = token_type_ids.shape[1]
                    masked_portion = jnp.logical_or(token_type_mask, causal_mask[:, :, :, :sequence_length])
                    causal_mask = causal_mask.at[:, :, :, :sequence_length].set(masked_portion)

                attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
                attention_mask = nn.combine_masks(attention_mask, causal_mask, fcm_mask)
            else:
                attention_mask = jnp.repeat(attention_mask, query_length, -2)

        return attention_mask

    def _apply_sliding_window(
        self,
        key: Array,
        value: Array,
        attention_mask: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | None,
        sliding_window: int,
        query_length: int,
        masking_details: AttnMaskDetail | None,
        cache_metadata: TransformerMetadata | None,
    ) -> tuple[Array, Array, Array]:
        """Applies sliding window masking and slicing to KV and mask."""

        offsets = jnp.zeros((key.shape[0],), "i4")
        krange = key.shape[1] if masking_details is None else attention_mask.shape[-1]
        if mode == common_types.MODE_DECODE and cache_view is not None:
            indexs = cache_view.indexs
            offsets = indexs - 1
        elif mode == common_types.MODE_PREFILL and masking_details is not None:
            indexs = jnp.array([query_length], "i4").repeat(key.shape[0], axis=0).reshape(-1)
        else:
            indexs = jnp.zeros((key.shape[0],), "i4") + 1

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

        key, value, attention_mask = _select_slices(key, value, attention_mask, offsets, indexs)

        if cache_metadata is not None and mode == common_types.MODE_DECODE:
            passed = cache_metadata.indexs - cache_metadata.starts
            cache_metadata = TransformerMetadata(
                starts=jax.lax.max(0, sliding_window - passed),
                indexs=jnp.full((attention_mask.shape[0],), attention_mask.shape[-1]),
            )

        return key, value, attention_mask, cache_metadata

    @jax.named_scope("easydel-flax-attention-concatenate")
    def concatenate(
        self,
        *,
        query: Array,
        key: Array,
        value: Array,
        attention_mask: Array,
        mode: common_types.RUNTIME_MODE_TYPES | common_types.EMPTY_VAL = common_types.NOT_GIVEN,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        causal_mask: Array | None = None,
        token_type_ids: Array | None = None,
        fcm_mask: Array | None = None,
        sliding_window: int | None = None,
    ) -> tuple[
        Array,
        Array,
        Array,
        tp.Callable[[], Array],
        TransformerCacheView | PagesCacheView | None,
        TransformerMetadata | PagesMetadata | None,
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
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView], optional): View into the KV cache.
                If None, caching is disabled. Defaults to None.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata], optional): Cache metadata.
                Defaults to None.
            causal_mask (tp.Optional[Array], optional): Causal mask [1, 1, q_len, kv_len]. Defaults to None.
            token_type_ids (tp.Optional[Array], optional): Token type IDs for segment masking [Batch, q_len].
                Defaults to None.
            fcm_mask (tp.Optional[Array], optional): Fused-Context-Mask [Batch, 1, q_len, kv_len].
                Defaults to None.
            sliding_window (tp.Optional[int], optional): Size of the sliding attention window. If None, not applied.
                Defaults to None.

        Returns:
            tp.Tuple[Array, Array, Array, tp.Callable[[], Array], tp.Optional[tp.Union[TransformerCacheView, PagesCacheView]]]:
                - key_states (Array): Final key states (potentially from cache).
                - value_states (Array): Final value states (potentially from cache).
                - attention_mask (Array): The final combined attention mask [Batch, Heads, q_len, kv_len].
                - init_attention_bias (Callable): Function to create the attention bias tensor.
                - updated_cache_view: The updated cache view (or None if no cache).
                - updated_cache_metadata: The updated cache metadata (or None if no metadata).

        Raises:
            ValueError: If shapes are mismatched.
        """  # noqa

        assert attention_mask.shape[-1] >= key.shape[1], "Attention mask length must match KV sequence length."

        query_length, initial_key_length = query.shape[1], key.shape[1]
        if attention_mask.dtype != jnp.bool:
            warnings.warn("attention_mask should be a boolean array", stacklevel=1)
            attention_mask = (attention_mask == 1).astype(jnp.bool_)

        if isinstance(mode, common_types.EMPTY_VAL):
            if cache_view is None:
                mode = common_types.MODE_TRAIN
            else:
                mode = common_types.MODE_PREFILL if query_length != 1 else common_types.MODE_DECODE

        attention_mask = self._merge_masks(
            attention_mask=attention_mask,
            causal=causal_mask is not None,
            causal_mask=causal_mask,
            token_type_ids=token_type_ids,
            fcm_mask=fcm_mask,
            query_length=query_length,
            initial_key_length=initial_key_length,
            cache_view=cache_view,
        )

        if isinstance(cache_view, PagesCacheView):
            cache_view = cache_view.concatenate_to_cache(key=key, value=value, cache_metadata=cache_metadata)

            def init_attention_bias():
                return jnp.zeros((query.shape[0], 1, query_length, initial_key_length), dtype=self.dtype)

            return key, value, attention_mask, init_attention_bias, cache_view, cache_metadata

        if cache_view is not None:
            assert query.shape[0] == cache_view.key.shape[0], "Batch size mismatch between query and cache."
        key, value, attention_mask, cache_view, masking_details = self._handle_cache_concat(
            query=query,
            key=key,
            value=value,
            mode=mode,
            attention_mask=attention_mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            causal_mask=causal_mask,
            token_type_ids=token_type_ids,
        )

        if cache_metadata is None and cache_view is not None:
            cache_metadata = TransformerMetadata(
                starts=cache_view.starts,
                indexs=cache_view.indexs,
            )
        if sliding_window is not None or (
            cache_view is not None
            and cache_view.masking_details is not None
            and cache_view.masking_details.mask_type == AttnMaskType.SLIDING
        ):
            masking_details = cache_view.masking_details if isinstance(cache_view, TransformerCacheView) else None
            if masking_details and masking_details.mask_type == AttnMaskType.SLIDING:
                sliding_window = sliding_window or masking_details.size
            key, value, attention_mask, cache_metadata = self._apply_sliding_window(
                key=key,
                value=value,
                attention_mask=attention_mask,
                mode=mode,
                cache_view=cache_view,
                sliding_window=sliding_window,
                query_length=query_length,
                masking_details=masking_details,
                cache_metadata=cache_metadata,
            )

        def init_attention_bias():
            return lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

        return key, value, attention_mask, init_attention_bias, cache_view, cache_metadata

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
