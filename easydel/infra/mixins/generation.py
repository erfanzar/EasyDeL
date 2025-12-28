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

"""Generation mixin for text generation capabilities.

Provides text generation functionality through the EasyGenerationMixin class,
which can be combined with EasyDeL models to enable various generation strategies
including greedy search, sampling, beam search, and more.

Classes:
    GreedyState: State container for greedy generation
    SampleState: State container for sampling generation
    BeamSearchState: State container for beam search
    EasyGenerationMixin: Mixin class providing generation methods

Key Features:
    - Multiple generation strategies (greedy, sampling, beam search)
    - Logits processing and warping
    - Support for generation constraints
    - Integration with HuggingFace generation configs
    - Efficient JAX implementations

Example:
    >>> from easydel.infra.mixins import EasyGenerationMixin
    >>> # Model class inherits from EasyGenerationMixin
    >>> output = model.generate(
    ...     input_ids=input_ids,
    ...     max_length=100,
    ...     temperature=0.8,
    ...     top_p=0.95,
    ...     do_sample=True
    ... )
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import pprint
import typing as tp
import warnings
from functools import cached_property, partial

import jax
import numpy as np
from eformer.escale import PartitionAxis
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array
from transformers.generation.configuration_utils import GenerationConfig

from easydel.inference.logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from easydel.layers.caching import RaggedPagesCache, RaggedPagesCacheConfig

from ..base_config import EasyDeLBaseConfig
from ..modeling_outputs import BeamSearchOutput, GreedySearchOutput, SampleOutput

if tp.TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from easydel.inference.sampling_params import SamplingParams
    from easydel.layers.caching import OperationsMetadata

logger = get_logger(__name__)


@auto_pytree
class GreedyState:
    """State container for greedy search generation.

    Tracks the current state during greedy decoding, where the most
    probable token is selected at each step.

    Attributes:
        cur_len: Current length of generated sequences.
        sequences: Generated token sequences.
        running_token: Currently processed token.
        is_sent_finished: Boolean flags for finished sequences.
        model_kwargs: Additional model-specific arguments.
    """

    cur_len: Array
    sequences: Array
    running_token: Array
    is_sent_finished: Array
    model_kwargs: dict[str, Array]


@auto_pytree
class SampleState:
    """State container for sampling-based generation.

    Tracks the current state during sampling generation, where tokens
    are sampled from the probability distribution.

    Attributes:
        cur_len: Current length of generated sequences.
        sequences: Generated token sequences.
        running_token: Currently processed token.
        is_sent_finished: Boolean flags for finished sequences.
        prng_key: JAX PRNG key for random sampling.
        model_kwargs: Additional model-specific arguments.
    """

    cur_len: Array
    sequences: Array
    running_token: Array
    is_sent_finished: Array
    prng_key: Array
    model_kwargs: dict[str, Array]


@auto_pytree
class BeamSearchState:
    """
    State for beam search generation.

    Attributes:
        cur_len (Array): Current length of the generated sequence.
        running_sequences (Array): Generated sequences being tracked in the beam.
        running_scores (Array): Scores of the sequences being tracked in the beam.
        sequences (Array): Best generated sequences.
        scores (Array): Scores of the best generated sequences.
        is_sent_finished (Array): Boolean array indicating if a sequence is finished.
        model_kwargs (tp.Dict[str, Array]): Model specific keyword arguments.
    """

    cur_len: Array
    running_sequences: Array
    running_scores: Array
    sequences: Array
    scores: Array
    is_sent_finished: Array
    model_kwargs: dict[str, Array]


def _safepick(config, pickname):
    vari = getattr(config, pickname, None)
    if vari is None and hasattr(config, "text_config"):
        vari = getattr(config.text_config, pickname, None)
    return vari


class EasyGenerationMixin:
    config_class: type[EasyDeLBaseConfig]
    config: EasyDeLBaseConfig
    base_model_prefix: str
    _model_task: str | None = None
    _model_type: str | None = None

    def init_ragged_pages(
        self,
        config: RaggedPagesCacheConfig | None = None,
        page_size: int | None = None,
        hbm_utilization: float | None = None,
        max_model_length: int | None = None,
    ) -> RaggedPagesCache:
        """
        Initializes and returns the actual Paged Attention KV Cache tensors.

        This method orchestrates the creation of the `RaggedPagesCache`. It either uses
        a pre-existing `RaggedPagesCacheConfig` object passed via the `config`
        argument, or if `config` is None, it first creates the config by calling
        `self.create_ragged_page_cache_config` using the other provided arguments (page_size,
        batch_size, etc.).

        Finally, it calls `RaggedPagesCache.init_cache` to allocate the necessary
        paged tensors (`key_pages`, `value_pages` for each layer) based on the
        config, model's mesh, dtype, partition manager, and quantization settings.

        Args:
            config (tp.Optional[RaggedPagesCacheConfig]): An optional pre-configured
                config object. If provided, other arguments like page_size, batch_size etc.,
                are ignored for config creation.
            page_size (tp.Optional[int]): Number of tokens per page. Required if `config` is None.
            hbm_utilization (tp.Optional[float]): Target HBM usage. Required if `config` is None.

        Returns:
            RaggedPagesCache: An initialized RaggedPagesCache object containing the allocated
                cache tensors (views) for all layers.

        Raises:
            AssertionError: If `config` is None and any of the required arguments
                (page_size, batch_size, max_sequences, dtype, hbm_utilization) are also None.
        """
        text_config = self.config.get_text_config()
        if config is None:
            assert page_size is not None, "if your not passing config you should pass `page_size`"
            assert hbm_utilization is not None, "if your not passing config you should pass `hbm_utilization`"
            assert max_model_length is not None, "if your not passing config you should pass `max_model_length`"

            config = self.create_ragged_page_cache_config(
                hbm_utilization=hbm_utilization,
                page_size=page_size,
                max_model_length=max_model_length,
            )
        return RaggedPagesCache.init_cache(
            mesh=text_config.mesh,
            config=config,
            partition_manager=text_config.partition_manager,
            quantizer=self._quant_class(
                quantization_config=text_config.kv_cache_quantization_config,
            ),
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: int | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ):
        """
        Initializes and returns the appropriate cache type for this model.

        This method automatically detects the cache type needed based on the model's
        operations and creates the appropriate cache:
        - For pure transformer models: Returns TransformerCache
        - For hybrid/recurrent models: Returns HybridCache via init_operations_cache

        Args:
            batch_size (int): The batch size for the cache.
            max_length (int): The maximum sequence length the cache needs to support.
            starts (int | None): Optional starting positions for the cache sequences.
                If provided, influences the initial state. Defaults to None (usually 0).
            shardings (dict | None): Optional dictionary specifying sharding configurations.
            pad_token_id (int | None): The ID of the padding token. If None, it's inferred.

        Returns:
            TransformerCache or HybridCache depending on the model type.
        """
        from easydel.layers.caching import TransformerCache

        text_config = self.config.get_text_config()
        cache_type = self.get_inference_cache_type()

        if cache_type == "transformer":
            return TransformerCache.init_cache(
                mesh=text_config.mesh,
                config=self.create_transformer_cache_config(
                    batch_size=batch_size,
                    max_length=max_length,
                    pad_token_id=pad_token_id,
                ),
                partition_manager=text_config.partition_manager,
                dtype=text_config.kvdtype,
                starts=starts,
                quantizer=self._quant_class(
                    quantization_config=text_config.kv_cache_quantization_config,
                ),
                mask_type_details=text_config.get_mask_details(),
            )
        else:
            # For hybrid/recurrent models, use init_operations_cache
            return self.init_operations_cache(
                batch_size=batch_size,
                max_length=max_length,
                starts=starts,
                dtype=text_config.kvdtype,
                quantizer=self._quant_class(
                    quantization_config=text_config.kv_cache_quantization_config,
                ),
                masking_details=text_config.get_mask_details() if cache_type != "ragged" else None,
            )

    def get_inference_cache_type(self) -> str:
        """Determine the appropriate cache type for inference based on model operations.

        This method uses dynamic discovery to inspect the actual operations and
        determine the best cache type:
        - "hybrid": For models with recurrent layers (e.g., Qwen3Next with attention + linear)
        - "transformer": For pure attention models using TransformerCache
        - "ragged": For models that only support RaggedPagesCache

        The returned cache type is used by execution managers to initialize the
        appropriate cache and metadata structures.

        Returns:
            str: "hybrid", "transformer", or "ragged" based on model operations.

        Example:
            >>> cache_type = model.get_inference_cache_type()
            >>> if cache_type == "ragged":
            ...     cache = model.init_ragged_pages(...)
            ... else:
            ...     cache = model.init_operations_cache(...)
        """
        cache_info = self.get_operations_cache_info()
        return cache_info.get_recommended_cache_type()

    def create_transformer_cache_config(
        self,
        batch_size: int,
        max_length: int,
        pad_token_id: int | None = None,
    ):
        """Create TransformerCacheConfig from model configuration.

        Args:
            batch_size: Batch size for inference.
            max_length: Maximum sequence length.
            pad_token_id: Optional pad token id override.

        Returns:
            TransformerCacheConfig configured for the model.
        """
        from easydel.layers.caching import TransformerCacheConfig

        text_config = self.config.get_text_config()

        if pad_token_id is None:
            pad_token_id = getattr(text_config, "pad_token_id", 0)

        num_hidden_layers = getattr(text_config, "num_hidden_layers", 1)

        # Some configs (e.g. OpenELM) store per-layer KV head counts in `num_kv_heads`.
        # Generation cache configs currently expect a single KV head count, so we
        # use the first layer's value when a list/tuple is provided.
        num_kv_heads = getattr(text_config, "num_kv_heads", None)
        if isinstance(num_kv_heads, (list, tuple)):
            num_kv_heads = int(num_kv_heads[0]) if len(num_kv_heads) > 0 else None
        if num_kv_heads is None:
            num_kv_heads = getattr(text_config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(text_config, "num_attention_heads", None)

        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(text_config, "hidden_size", None)
            num_heads = getattr(text_config, "num_attention_heads", None)
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads

        if num_kv_heads is None:
            raise ValueError(
                "Could not infer number of KV heads for TransformerCacheConfig; "
                "expected `num_key_value_heads` or `num_attention_heads` on config."
            )
        if head_dim is None:
            raise ValueError(
                "Could not infer head_dim for TransformerCacheConfig; "
                "expected `head_dim` or (`hidden_size` and `num_attention_heads`) on config."
            )

        # MLA-style attention (e.g., Kimi Linear / DeepSeek): KV head dims may differ
        # from `head_dim` and K/V dims may differ from each other.
        qk_nope_head_dim = getattr(text_config, "qk_nope_head_dim", None)
        qk_rope_head_dim = getattr(text_config, "qk_rope_head_dim", None)
        v_head_dim = getattr(text_config, "v_head_dim", None)
        if qk_nope_head_dim is not None and qk_rope_head_dim is not None and v_head_dim is not None:
            key_dim = int(qk_nope_head_dim + qk_rope_head_dim)
            value_dim = int(v_head_dim)
            if key_dim != head_dim or value_dim != head_dim:
                # MLA forward path constructs KV with `num_attention_heads` heads (not GQA/MQA),
                # so the cache must match that head count.
                mla_heads = int(getattr(text_config, "num_attention_heads", num_kv_heads) or num_kv_heads)
                return TransformerCacheConfig.create(
                    batch_size=batch_size,
                    sequence_length=max_length,
                    num_hidden_layers=num_hidden_layers,
                    pad_token_id=pad_token_id,
                    num_heads=mla_heads,
                    head_dim=None,
                    key_dim=key_dim,
                    value_dim=value_dim,
                )

        return TransformerCacheConfig.create(
            batch_size=batch_size,
            sequence_length=max_length,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            num_heads=num_kv_heads,
            head_dim=head_dim,
        )

    def create_recurrent_cache_config(self, batch_size: int):
        """Create RecurrentCacheConfig from model configuration.

        This method automatically detects the recurrent architecture type (Qwen3-Next,
        FalconH1, Mamba2, Mamba) and creates the appropriate cache configuration.

        Args:
            batch_size (int): Batch size for inference.

        Returns:
            RecurrentCacheConfig: Cache configuration appropriate for the model's
                recurrent layer architecture.

        Raises:
            ValueError: If intermediate_size cannot be inferred for certain architectures.
        """
        from easydel.layers.caching import RecurrentCacheConfig

        text_config = self.config.get_text_config()

        partition_axis = getattr(text_config, "partition_axis", None)
        if partition_axis is None:
            partition_axis = PartitionAxis()

        num_hidden_layers = getattr(text_config, "num_hidden_layers", 1)

        # 1) Qwen3-Next / GatedDeltaRule-style linear attention
        has_linear_attn_fields = any(
            hasattr(text_config, name)
            for name in (
                "linear_num_value_heads",
                "linear_num_key_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
            )
        )
        if has_linear_attn_fields:
            conv_dim = getattr(text_config, "linear_d_inner", None)
            if conv_dim is None:
                num_k_heads = getattr(text_config, "linear_num_key_heads", None)
                head_k_dim = getattr(text_config, "linear_key_head_dim", None)
                num_v_heads = getattr(text_config, "linear_num_value_heads", None)
                head_v_dim = getattr(text_config, "linear_value_head_dim", None)
                if num_k_heads and head_k_dim and num_v_heads and head_v_dim:
                    key_dim = num_k_heads * head_k_dim
                    value_dim = num_v_heads * head_v_dim
                    conv_dim = key_dim * 2 + value_dim
                elif num_k_heads and head_k_dim:
                    conv_dim = num_k_heads * head_k_dim
                else:
                    conv_dim = getattr(text_config, "intermediate_size", 2048)

            conv_kernel_size = getattr(text_config, "linear_conv_kernel_dim", None)
            if conv_kernel_size is None:
                conv_kernel_size = getattr(text_config, "d_conv", 4)

            num_heads = getattr(text_config, "linear_num_value_heads", None) or getattr(
                text_config, "linear_num_key_heads", None
            )
            head_dim = getattr(text_config, "linear_key_head_dim", None)
            d_state = getattr(text_config, "linear_d_state", None)
            if d_state is None:
                d_state = getattr(text_config, "linear_value_head_dim", None) or 64

            recurrent_shape: tuple[int, ...]
            if num_heads is not None and head_dim is not None:
                # GDR/KDA-style recurrent state: [batch, num_heads, head_dim(key), d_state(value)]
                recurrent_shape = (int(num_heads), int(head_dim), int(d_state))
            else:
                # Fallback: Mamba-style recurrent state
                recurrent_shape = (int(conv_dim), int(d_state))

            return RecurrentCacheConfig.create(
                num_hidden_layers=num_hidden_layers,
                partition_axis=partition_axis,
                batch_size=batch_size,
                conv_dim=int(conv_dim),
                conv_kernel_size=int(conv_kernel_size),
                recurrent_state_shape=recurrent_shape,
            )

        # 2) FalconH1-style (mamba_* prefixed attributes)
        is_falcon_h1 = all(
            hasattr(text_config, name)
            for name in (
                "mamba_n_heads",
                "mamba_d_head",
                "mamba_n_groups",
                "mamba_d_state",
                "mamba_d_conv",
            )
        )
        if is_falcon_h1:
            # FalconH1 uses mamba_intermediate_size property or mamba_d_ssm
            intermediate_size = getattr(text_config, "mamba_intermediate_size", None)
            if intermediate_size is None:
                mamba_d_ssm = getattr(text_config, "mamba_d_ssm", None)
                if mamba_d_ssm is not None:
                    intermediate_size = mamba_d_ssm
                else:
                    hidden_size = getattr(text_config, "hidden_size", None)
                    mamba_expand = getattr(text_config, "mamba_expand", 2)
                    if hidden_size is None:
                        raise ValueError("Could not infer intermediate_size for FalconH1 recurrent cache config.")
                    intermediate_size = int(mamba_expand * hidden_size)

            state_size = int(text_config.mamba_d_state)
            n_groups = int(text_config.mamba_n_groups)
            conv_dim = int(intermediate_size + 2 * n_groups * state_size)
            conv_kernel_size = int(text_config.mamba_d_conv)
            recurrent_shape = (int(text_config.mamba_n_heads), int(text_config.mamba_d_head), state_size)

            return RecurrentCacheConfig.create(
                num_hidden_layers=num_hidden_layers,
                partition_axis=partition_axis,
                batch_size=batch_size,
                conv_dim=conv_dim,
                conv_kernel_size=conv_kernel_size,
                recurrent_state_shape=recurrent_shape,
            )

        # 3) Mamba2-style state space models (standard naming)
        is_mamba2 = all(
            hasattr(text_config, name)
            for name in (
                "num_heads",
                "head_dim",
                "n_groups",
                "state_size",
                "conv_kernel",
            )
        )
        if is_mamba2:
            intermediate_size = getattr(text_config, "intermediate_size", None)
            if intermediate_size is None:
                hidden_size = getattr(text_config, "hidden_size", None)
                expand = getattr(text_config, "expand", 2)
                if hidden_size is None:
                    raise ValueError("Could not infer intermediate_size for Mamba2 recurrent cache config.")
                intermediate_size = int(expand * hidden_size)

            state_size = int(text_config.state_size)
            n_groups = int(text_config.n_groups)
            conv_dim = int(intermediate_size + 2 * n_groups * state_size)
            conv_kernel_size = int(text_config.conv_kernel)
            recurrent_shape = (int(text_config.num_heads), int(text_config.head_dim), state_size)

            return RecurrentCacheConfig.create(
                num_hidden_layers=num_hidden_layers,
                partition_axis=partition_axis,
                batch_size=batch_size,
                conv_dim=conv_dim,
                conv_kernel_size=conv_kernel_size,
                recurrent_state_shape=recurrent_shape,
            )

        # 4) Mamba-style state space models
        is_mamba = all(hasattr(text_config, name) for name in ("state_size", "conv_kernel", "intermediate_size"))
        if is_mamba:
            intermediate_size = int(text_config.intermediate_size)
            state_size = int(text_config.state_size)
            conv_kernel_size = int(text_config.conv_kernel)
            recurrent_shape = (intermediate_size, state_size)

            return RecurrentCacheConfig.create(
                num_hidden_layers=num_hidden_layers,
                partition_axis=partition_axis,
                batch_size=batch_size,
                conv_dim=intermediate_size,
                conv_kernel_size=conv_kernel_size,
                recurrent_state_shape=recurrent_shape,
            )

        # 5) Generic fallback
        conv_dim = getattr(text_config, "intermediate_size", 2048)
        conv_kernel_size = getattr(text_config, "d_conv", 4)
        d_state = getattr(text_config, "state_size", 64)
        return RecurrentCacheConfig.create(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            conv_dim=int(conv_dim),
            conv_kernel_size=int(conv_kernel_size),
            recurrent_state_shape=(int(conv_dim), int(d_state)),
        )

    def create_kda_cache_config(self, batch_size: int):
        """Create KDACacheConfig from model configuration.

        Creates cache configuration for models using Key-Dependent Attention (KDA)
        mechanisms like certain linear attention variants.

        Args:
            batch_size (int): Batch size for inference.

        Returns:
            KDACacheConfig: Cache configuration for KDA-style attention layers.

        Raises:
            ValueError: If num_heads cannot be inferred from the model configuration.
        """
        from easydel.layers.caching import KDACacheConfig

        text_config = self.config.get_text_config()

        partition_axis = getattr(text_config, "partition_axis", None)
        if partition_axis is None:
            partition_axis = PartitionAxis()

        num_hidden_layers = getattr(text_config, "num_hidden_layers", 1)

        linear_config = getattr(text_config, "linear_attn_config", None) or {}

        num_heads = int(linear_config.get("num_heads", getattr(text_config, "num_attention_heads", 0) or 0))
        head_k_dim = int(linear_config.get("head_k_dim", 128))
        head_v_dim = int(linear_config.get("head_v_dim", 128))
        d_conv = int(linear_config.get("d_conv", getattr(text_config, "d_conv", 4)))

        if num_heads <= 0:
            raise ValueError(
                "Could not infer `num_heads` for KDACacheConfig; "
                "expected `linear_attn_config['num_heads']` or `num_attention_heads`."
            )

        key_dim = num_heads * head_k_dim
        value_dim = num_heads * head_v_dim
        recurrent_shape = (num_heads, head_k_dim, head_v_dim)

        return KDACacheConfig.create(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            key_dim=key_dim,
            value_dim=value_dim,
            d_conv=d_conv,
            recurrent_state_shape=recurrent_shape,
        )

    def create_lightning_cache_config(self, batch_size: int):
        """Create LightningCacheConfig from model configuration.

        Creates cache configuration for models using Lightning Attention mechanisms.

        Args:
            batch_size (int): Batch size for inference.

        Returns:
            LightningCacheConfig: Cache configuration for Lightning Attention layers.
        """

        from easydel.layers.caching import LightningCacheConfig

        text_config = self.config.get_text_config()

        partition_axis = getattr(text_config, "partition_axis", None)
        if partition_axis is None:
            partition_axis = PartitionAxis()

        # Get heads/dims from config
        num_heads = getattr(text_config, "num_attention_heads", None)
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(text_config, "hidden_size", None)
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads

        # Key/value heads for MQA/GQA
        key_heads = getattr(text_config, "num_key_value_heads", num_heads)
        value_heads = getattr(text_config, "num_key_value_heads", num_heads)

        # Key/value dims (can be different from head_dim in some architectures)
        key_dim = getattr(text_config, "key_dim", head_dim)
        value_dim = getattr(text_config, "value_dim", head_dim)

        return LightningCacheConfig.create(
            partition_axis=partition_axis,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            key_heads=key_heads,
            value_heads=value_heads,
            key_dim=key_dim,
            value_dim=value_dim,
        )

    def create_ragged_page_cache_config(
        self,
        max_length: int,
        *,
        page_size: int = 128,
        hbm_utilization: float = 0.9,
        dtype: jnp.dtype | None = None,
    ):
        """Create RaggedPagesCacheConfig from model configuration.

        Creates cache configuration for Paged Attention (vLLM-style) with ragged
        page management. The batch size is determined dynamically based on HBM
        utilization.

        Args:
            max_length (int): Maximum sequence length (max_model_length).
            page_size (int): Number of tokens per page. Defaults to 128.
            hbm_utilization (float): Target HBM memory utilization fraction (0.0-1.0).
                Defaults to 0.9.
            dtype (jnp.dtype | None): Data type for cache tensors. If None, uses
                the model's kvdtype. Defaults to None.

        Returns:
            RaggedPagesCacheConfig: Cache configuration for paged attention.
        """
        from easydel.layers.caching import RaggedPagesCacheConfig

        text_config = self.config.get_text_config()

        if dtype is None:
            dtype = getattr(text_config, "kvdtype", jnp.bfloat16)
            if isinstance(dtype, str):
                dtype = getattr(jnp, dtype, jnp.bfloat16)

        num_kv_heads = getattr(text_config, "num_key_value_heads", None)
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(text_config, "hidden_size", None)
            num_heads = getattr(text_config, "num_attention_heads", None)
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads

        num_hidden_layers = getattr(text_config, "num_hidden_layers", 1)

        return RaggedPagesCacheConfig.create(
            mesh=text_config.mesh,
            partition_manager=text_config.partition_manager,
            kvdtype=dtype,
            num_hidden_layers=num_hidden_layers,
            num_kv_heads=num_kv_heads,
            max_model_length=max_length,
            kv_head_dim_size=head_dim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
        )

    def init_operations_cache(
        self,
        batch_size: int,
        max_length: int,
        *,
        # RaggedPagesCache args (for ragged page attention)
        page_size: int = 128,
        hbm_utilization: float = 0.9,
        dtype: jnp.dtype | None = None,
        quantizer=None,
        masking_details=None,
        starts: jnp.array | None = None,
    ):
        """Initialize cache using HybridCache as the universal per-layer container.

        This method uses HybridCache to hold per-layer cache views, where each
        layer gets the appropriate view type based on its operation:
        - Attention operations -> TransformerCacheView
        - GatedDeltaRule operations -> RecurrentCacheView
        - KDA operations -> KDACacheView
        - RaggedPageAttention operations -> RaggedPagesCacheView
        - Lightning attention operations -> LightningCacheView

        Args:
            batch_size (int): Batch size for inference.
            max_length (int): Maximum sequence length.
            page_size (int): Page size for RaggedPagesCache. Defaults to 128.
            hbm_utilization (float): HBM utilization for RaggedPagesCache. Defaults to 0.9.
            dtype (jnp.dtype | None): Data type for cache tensors. Defaults to model's kvdtype.
            quantizer: Optional quantizer for KV cache compression.
            masking_details: Optional masking details for attention operations.
            starts (jnp.array | None): Optional starting positions for sequences.
                Used to initialize cache state for pre-filled prompts. Defaults to None.

        Returns:
            HybridCache: Cache with per-layer views appropriate for each operation.

        Example:
            >>> # Standard usage - automatically creates appropriate views per layer
            >>> cache = model.init_operations_cache(batch_size=1, max_length=2048)
            >>> # cache.views[0] might be RecurrentCacheView (for linear attention layer)
            >>> # cache.views[3] might be TransformerCacheView (for full attention layer)
        """
        from easydel.layers.caching import (
            HybridCache,
            KDACacheView,
            LightningCacheView,
            ParallelHybridCacheView,
            RaggedPagesCacheView,
            RecurrentCacheView,
            TransformerCacheView,
        )

        text_config = self.config.get_text_config()
        cache_view_mapping = self.get_operations_cache_view()

        # Resolve dtype
        if dtype is None:
            dtype = getattr(text_config, "kvdtype", jnp.bfloat16)
            if isinstance(dtype, str):
                dtype = getattr(jnp, dtype, jnp.bfloat16)

        # Check if any layer needs RaggedPagesCacheView and create shared config
        shared_ragged_config = None
        needs_ragged = any(view_class is RaggedPagesCacheView for view_class in cache_view_mapping.values())

        if needs_ragged:
            shared_ragged_config = self.create_ragged_page_cache_config(
                max_length=max_length,
                page_size=page_size,
                hbm_utilization=hbm_utilization,
                dtype=dtype,
            )

        with self.mesh:
            num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
            if num_hidden_layers is None:
                num_hidden_layers = (max(cache_view_mapping.keys(), default=-1) + 1) if cache_view_mapping else 0

            views = [None] * num_hidden_layers

            for idx in range(num_hidden_layers):
                view_class = cache_view_mapping.get(idx)
                if view_class is None:
                    raise ValueError(
                        f"Missing cache view class for layer {idx}. "
                        "Operation discovery did not return a cache view for every layer."
                    )

                if view_class is ParallelHybridCacheView:
                    # Parallel hybrid layer: needs BOTH KV-cache and recurrent/SSM state.
                    t_config = self.create_transformer_cache_config(batch_size=batch_size, max_length=max_length)
                    transformer_view = TransformerCacheView.init(
                        config=t_config,
                        layer_index=idx,
                        mesh=text_config.mesh,
                        dtype=dtype,
                        partition_manager=text_config.partition_manager,
                        quantizer=quantizer,
                        masking_details=masking_details,
                        starts=starts,
                    )

                    r_config = self.create_recurrent_cache_config(batch_size=batch_size)
                    recurrent_view = RecurrentCacheView.init(
                        config=r_config,
                        layer_index=idx,
                        dtype=dtype,
                    )

                    view = ParallelHybridCacheView(
                        transformer=transformer_view,
                        recurrent=recurrent_view,
                        layer_index=idx,
                    )
                elif view_class is TransformerCacheView:
                    config = self.create_transformer_cache_config(batch_size=batch_size, max_length=max_length)
                    view = view_class.init(
                        config=config,
                        layer_index=idx,
                        mesh=text_config.mesh,
                        dtype=dtype,
                        partition_manager=text_config.partition_manager,
                        quantizer=quantizer,
                        masking_details=masking_details,
                        starts=starts,
                    )
                elif view_class is RecurrentCacheView:
                    config = self.create_recurrent_cache_config(batch_size=batch_size)
                    view = view_class.init(
                        config=config,
                        layer_index=idx,
                        dtype=dtype,
                    )
                elif view_class is KDACacheView:
                    config = self.create_kda_cache_config(batch_size=batch_size)
                    view = view_class.init(
                        config=config,
                        layer_index=idx,
                        dtype=dtype,
                    )
                elif view_class is RaggedPagesCacheView:
                    view = view_class.init(
                        config=shared_ragged_config,
                        layer_index=idx,
                        mesh=text_config.mesh,
                        partition_manager=text_config.partition_manager,
                        quantizer=quantizer,
                    )
                elif view_class is LightningCacheView:
                    config = self.create_lightning_cache_config(batch_size=batch_size)
                    view = view_class.init(
                        config=config,
                        layer_index=idx,
                        dtype=dtype,
                    )
                else:
                    raise ValueError(f"Unknown cache view class: {view_class}")

                views[idx] = view

            return HybridCache(views=views)

    def create_operations_metadata(
        self,
        *,
        postpadded: bool = False,
        starts: jnp.ndarray | None = None,
        indexs: jnp.ndarray | None = None,
        pages_tables: jnp.ndarray | None = None,
        context_lens: jnp.ndarray | None = None,
        query_start_loc: jnp.ndarray | None = None,
        num_seqs: jnp.ndarray | None = None,
        slot_mapping: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        page_size: int = 128,
    ) -> "OperationsMetadata":
        """Create OperationsMetadata for use with cache operations.

        This method creates OperationsMetadata that works with either:
        - HybridCache: Uses HybridMetadata with embedded transformer fields
        - RaggedPagesCache: Uses RaggedPagesMetadata (when ragged params provided)

        For HybridCache (the default), the metadata includes fields needed by
        TransformerCacheView layers (postpadded, starts, indexs). Recurrent
        layers use their own internal state management.

        Args:
            postpadded: Whether sequences are post-padded (for transformer views).
            starts: Starting positions for sequences (for transformer views).
            indexs: Current position indices (for transformer views).
            pages_tables: Page tables mapping (for ragged pages).
            context_lens: Context lengths per sequence (for ragged pages).
            query_start_loc: Query start locations (for ragged pages).
            num_seqs: Number of sequences (for ragged pages).
            slot_mapping: Slot mapping (for ragged pages).
            position_ids: Position IDs (for ragged pages).
            page_size: Page size (for ragged pages).

        Returns:
            OperationsMetadata configured for the cache type.

        Example:
            >>> # For HybridCache (default)
            >>> metadata = model.create_operations_metadata(
            ...     starts=jnp.zeros((batch_size,), dtype=jnp.int32)
            ... )
            >>>
            >>> # For RaggedPagesCache
            >>> metadata = model.create_operations_metadata(
            ...     pages_tables=..., context_lens=...
            ... )
        """
        from easydel.layers.caching import OperationsMetadata

        # Check if ragged pages metadata is requested
        if pages_tables is not None and context_lens is not None:
            return OperationsMetadata.for_ragged(
                pages_tables=pages_tables,
                context_lens=context_lens,
                query_start_loc=query_start_loc,
                num_seqs=num_seqs,
                slot_mapping=slot_mapping,
                position_ids=position_ids,
                page_size=page_size,
            )

        # Default to hybrid metadata (works with HybridCache)
        return OperationsMetadata.for_hybrid(
            postpadded=postpadded,
            starts=starts,
            indexs=indexs,
        )

    @cached_property
    def _quant_class(self):
        """
        Cached property to access the EasyQuantizer class type.

        Used internally to easily reference the quantization class without repeated imports.

        Returns:
            type: The EasyQuantizer class.
        """
        from easydel.layers.quantization.quantizers import EasyQuantizer

        return EasyQuantizer

    @staticmethod
    def compute_prefill_length(array, padding_id) -> Array:
        """
        Calculates the number of padding tokens at the beginning of each sequence.

        This is useful for determining the actual starting position in a KV cache when
        dealing with left-padded inputs.

        Args:
            array (Array): The input token ID array, typically shape (batch_size, sequence_length).
            padding_id (int): The token ID used for padding.

        Returns:
            Array: An array of shape (batch_size,) containing the number of leading
                padding tokens for each sequence in the batch.
        """
        valid = array != padding_id
        return jnp.sum(jnp.cumsum(valid, axis=-1) == 0, axis=-1)

    @staticmethod
    def compute_prefill_length_from_mask(mask) -> Array:
        """
        Calculates the number of padding tokens at the beginning of each sequence
        from a 0/1 or boolean mask.
        """
        mask = mask.astype(jnp.bool_)
        return jnp.sum(jnp.cumsum(mask, axis=-1) == 0, axis=-1)

    def _make_mask_info(
        self,
        attention_mask: jax.Array | None,
        q_segment_ids: jax.Array | None,
        kv_segment_ids: jax.Array | None,
        q_positions: jax.Array | None,
        kv_positions: jax.Array | None,
        is_self_attn: bool = True,
    ) -> MaskInfo:
        """Creates a MaskInfo object from attention masks or segment IDs.

        This method prefers explicit segment IDs over attention masks when both
        are provided, as segment-based masking is more flexible and supports
        document-level boundaries.

        Args:
            attention_mask (jax.Array | None): Optional attention mask of shape
                (B, L) or (B, 1, Q, K).
            q_segment_ids (jax.Array | None): Optional query segment IDs for
                document-aware masking.
            kv_segment_ids (jax.Array | None): Optional key/value segment IDs.
                If None and is_self_attn=True, uses q_segment_ids.
            q_positions (jax.Array | None): Optional query position indices.
            kv_positions (jax.Array | None): Optional key/value position indices.
            is_self_attn (bool): Whether this is self-attention (vs cross-attention).
                Defaults to True.

        Returns:
            MaskInfo: A MaskInfo object configured for the attention operation.

        Raises:
            ValueError: If neither segment_ids nor attention_mask is provided.
        """
        # Prefer explicit segment IDs over masks
        if q_segment_ids is not None:
            return MaskInfo.from_segments(
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids
                if kv_segment_ids is not None
                else (q_segment_ids if is_self_attn else None),
                q_positions=q_positions,
                kv_positions=kv_positions,
            )
        if attention_mask is not None:
            # Works for (B,1,1,L), (B,1,Q,K) etc. Your MaskInfo handles 3D/4D gracefully.
            return MaskInfo.from_attention_mask(attention_mask, q_positions=q_positions, kv_positions=kv_positions)
        raise ValueError("Need either segment_ids (preferred) or attention_mask to build MaskInfo.")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        shardings: int | None = None,
        attention_mask: jax.Array | None = None,
        token_type_ids: jax.Array | None = None,
        mask_info: MaskInfo | None = None,  # NEW
    ) -> dict[str, tp.Any]:
        """
        Sets up the initial inputs required for starting autoregressive generation.

        This function initializes the Key-Value cache (`past_key_values`) using `init_cache`,
        calculates the initial `position_ids` based on the input `attention_mask` (or assumes
        a contiguous range if no mask is provided), and prepares an extended `attention_mask`
        suitable for caching. It ensures inputs are placed on the correct devices/shards.

        Args:
            input_ids (Array): The initial sequence of token IDs. Shape (batch_size, seq_length).
            max_length (int): The maximum sequence length that the KV cache should support.
            pad_token_id (int): The ID used for padding tokens. Used to calculate `starts` if not provided.
            starts (int | None): Optional pre-calculated starting positions (number of leading pads).
                If None, calculated using `compute_prefill_length`. Defaults to None.
            shardings (dict | None): Optional sharding configuration passed to `init_cache`.
                Defaults to None.
            attention_mask (Array | None): An optional mask indicating which tokens
                should be attended to. Shape (batch_size, seq_length). Defaults to None.
            token_type_ids (Array | None): Optional segment IDs for models that use them.
                Defaults to None.
            mask_info (MaskInfo | None): Optional pre-computed MaskInfo object for attention.
                If provided, used directly instead of computing from attention_mask.
                Defaults to None.

        Returns:
            dict: A dictionary containing the prepared inputs, typically including:
                - "past_key_values": The initialized KV cache.
                - "mask_info": The MaskInfo object for attention masking.
                - "position_ids": The calculated initial position IDs.
                - "token_type_ids": (Optional) Prepared token type IDs.
                This dictionary is then passed through `prepare_inputs_for_call`.
        """
        batch_size, seq_length = input_ids.shape
        if starts is None:
            if attention_mask is not None:
                starts = self.compute_prefill_length_from_mask(attention_mask.astype(jnp.bool_))
            else:
                starts = self.compute_prefill_length(input_ids, pad_token_id)

        past_key_values = self.init_operations_cache(
            batch_size=batch_size,
            max_length=max_length,
            starts=starts,
        )

        if mask_info is None:
            if attention_mask is None:
                if hasattr(self, "generation_config") and self.generation_config.pad_token_id is not None:
                    pad_id = self.generation_config.pad_token_id
                else:
                    pad_id = pad_token_id
                valid = input_ids != pad_id
                seg = jnp.where(valid, jnp.int32(0), jnp.int32(-1))
                mask_info = MaskInfo.from_segments(seg)
            else:
                mask_info = MaskInfo.from_attention_mask(attention_mask)

        mask_info = self._pad_maskinfo_to_maxlen(mask_info, max_length=max_length, make_causal=True)

        if attention_mask is not None:
            am = attention_mask.astype(jnp.bool_)
            position_ids = jnp.where(am, am.astype(jnp.int32).cumsum(axis=-1) - 1, 0)
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype=jnp.int32)[None, :], (batch_size, seq_length))

        calldict = dict(past_key_values=past_key_values, mask_info=mask_info, position_ids=position_ids)

        if token_type_ids is not None:
            calldict["token_type_ids"] = token_type_ids

        return self.prepare_inputs_for_call(**calldict)

    def _pad_maskinfo_to_maxlen(
        self,
        mask_info: MaskInfo,
        max_length: int,
        make_causal: bool,
    ) -> MaskInfo:
        # Ensure we have segment ids; prefer seg-ids to avoid stale masks
        if mask_info.q_segment_ids is None or mask_info.kv_segment_ids is None:
            mask_info = mask_info.materialize_segment_ids()

        q_ids = jnp.asarray(mask_info.q_segment_ids, jnp.int32)  # (B, Q0)
        kv_ids = jnp.asarray(mask_info.kv_segment_ids, jnp.int32)  # (B, K0)
        B, Q0 = q_ids.shape
        K0 = kv_ids.shape[-1]

        if Q0 < max_length:
            q_pad = jnp.full((B, max_length - Q0), 0)
            q_ids = jnp.concatenate([q_ids, q_pad], axis=-1)
        else:
            q_ids = q_ids[:, :max_length]
        if K0 < max_length:
            kv_pad = jnp.full((B, max_length - K0), 0)
            kv_ids = jnp.concatenate([kv_ids, kv_pad], axis=-1)
        else:
            kv_ids = kv_ids[:, :max_length]

        base = MaskInfo.from_segments(q_ids, kv_ids)
        if make_causal:
            base = base.apply_causal(offset=0)
        return base

    def update_inputs_for_generation(
        self,
        model_outputs,
        model_kwargs,
    ) -> dict[str, tp.Any]:
        """
        Updates the keyword arguments for the next generation step.

        Specifically, it takes the `past_key_values` from the `model_outputs` of the
        current step and updates the `model_kwargs` with them. It also increments the
        `position_ids` by one for the next token prediction.

        Args:
            model_outputs: The output object from the model's forward pass in the previous step
                (should contain a `past_key_values` attribute).
            model_kwargs (dict): The dictionary of keyword arguments used for the model call.
                This dictionary will be modified in-place or a new one returned.

        Returns:
            dict: The updated `model_kwargs` dictionary ready for the next generation step.
        """
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

    def _create_required_props_from_kwargs(
        self, model_kwargs: dict[str, Array]
    ) -> tp.Mapping[str, dict[str, tp.Any]] | None:
        """
        Placeholder method to extract or create properties required for specific model types
        from keyword arguments. Intended to be overridden by subclasses if needed.

        Args:
            model_kwargs (dict): Keyword arguments passed to the

        Returns:
            Optional[Mapping[str, Dict[str, Any]]]: Extracted properties or None.
                Defaults to returning None.
        """
        return None

    def _validate_signature(
        self,
        method,
        args: tuple,
        kwargs: dict[str, tp.Any],
    ) -> dict[str, tp.Any]:
        """
        Validates and filters arguments against a method's signature.

        Inspects the signature of the provided `method` and filters the combined `args`
        and `kwargs` to include only those parameters that are actually accepted by the method.
        This prevents errors caused by passing unexpected arguments. Issues warnings for
        skipped parameters.

        Args:
            method (callable): The method whose signature should be checked.
            args (tuple): Positional arguments intended for the method.
            kwargs (dict): Keyword arguments intended for the method.

        Returns:
            dict: A dictionary containing only the keyword arguments that match the
                method's signature. Positional arguments are converted to keyword arguments
                based on their position.
        """
        sig = inspect.signature(method)
        valid_params = sig.parameters

        args_as_kwargs = {}
        positional_params = [
            param
            for param in valid_params.values()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]

        for i, arg in enumerate(args):
            if i < len(positional_params):
                args_as_kwargs[positional_params[i].name] = arg

        filtered_kwargs = {}
        for name, value in {**args_as_kwargs, **kwargs}.items():
            if name in valid_params:
                param = valid_params[name]
                if param.annotation != inspect.Parameter.empty:
                    try:
                        if getattr(param.annotation, "__origin__", None) is tp.Optional and value is not None:
                            expected_type = param.annotation.__args__[0]
                            if not isinstance(value, expected_type):
                                print(
                                    f"Warning: Parameter '{name}' expected type {expected_type}, "
                                    f"got {type(value)}. Skipping parameter."
                                )
                                continue
                    except Exception:
                        pass
                filtered_kwargs[name] = value
            else:
                warnings.warn(
                    f"  Parameter '{name}' not found in child class signature. Skipping.",
                    stacklevel=1,
                )

        return filtered_kwargs

    @staticmethod
    def _run_loop_in_debug(cond_fn, body_fn, init_state) -> tp.Any:
        """
        Executes a conditional loop (`while cond_fn: state = body_fn(state)`) without JAX tracing.

        This provides a standard Python loop execution equivalent to `jax.lax.while_loop`,
        which is useful for debugging the loop's body function step-by-step, as JAX's
        traced loops can be opaque. Should not be used in production code intended for JIT compilation.

        Args:
            cond_fn (callable): A function that takes the current state and returns True
                if the loop should continue.
            body_fn (callable): A function that takes the current state and returns the next state.
            init_state (Any): The initial state for the loop.

        Returns:
            Any: The final state after the loop terminates.
        """
        state = init_state
        while cond_fn(state):
            state = body_fn(state)
        return state

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        input_ids,
        model_kwargs,
    ) -> dict[str, tp.Any]:
        """
        Prepares keyword arguments specifically for encoder-decoder model generation.

        It separates arguments intended for the encoder, runs the `self.encode` method
        with those arguments, and adds the resulting `encoder_outputs` to the `model_kwargs`.
        This pre-computes the encoder representation needed by the decoder during generation.

        Args:
            input_ids (Array): The input token IDs for the encoder.
            model_kwargs (dict): The dictionary of keyword arguments. Encoder-specific
                arguments will be used, and `encoder_outputs` will be added.

        Returns:
            dict: The updated `model_kwargs` dictionary containing `encoder_outputs`.
        """
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
        }
        model_kwargs["encoder_outputs"] = self.encode(input_ids, **encoder_kwargs)
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int | None = None,
        bos_token_id: int | None = None,
        model_kwargs: dict[str, Array] | None = None,
    ) -> Array:
        """
        Creates the initial `decoder_input_ids` tensor for encoder-decoder generation.

        It checks if `decoder_input_ids` are already provided in `model_kwargs`. If not,
        it determines the appropriate starting token ID (using `_get_decoder_start_token_id`)
        and creates a tensor of shape (batch_size, 1) containing that ID repeated for each
        sequence in the batch.

        Args:
            batch_size (int): The number of sequences in the batch.
            decoder_start_token_id (int | None): Explicitly provided start token ID.
            bos_token_id (int | None): Explicitly provided BOS token ID (used if decoder start ID is missing).
            model_kwargs (dict | None): Optional dictionary of keyword arguments. If it contains
                "decoder_input_ids", those are returned directly and removed from the dict.

        Returns:
            Array: The initial `decoder_input_ids` tensor, shape (batch_size, 1).
        """
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            if decoder_input_ids is not None:
                return decoder_input_ids
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        return jnp.array(decoder_start_token_id, dtype=jnp.int32).reshape(1, -1).repeat(batch_size, axis=0)

    def _get_decoder_start_token_id(
        self,
        decoder_start_token_id: int | None = None,
        bos_token_id: int | None = None,
    ) -> int:
        """
        Determines the appropriate start token ID for the decoder during generation.

        It prioritizes `decoder_start_token_id` if provided, then checks the model's
        `generation_config`, then the main `config` (and its `decoder` sub-config if applicable),
        falling back to `bos_token_id` from similar sources if the specific decoder start ID
        is unavailable.

        Args:
            decoder_start_token_id (int | None): Explicitly provided decoder start token ID.
            bos_token_id (int | None): Explicitly provided BOS token ID.

        Returns:
            int: The determined decoder start token ID.

        Raises:
            ValueError: If neither a `decoder_start_token_id` nor a `bos_token_id` can be found
                in any of the configurations.
        """
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        raise ValueError("`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation.")

    @staticmethod
    def _expand_to_num_beams(tensor, num_beams):
        """
        Expands/repeats a tensor to match the number of beams for beam search.

        It inserts a new dimension after the batch dimension and broadcasts/repeats
        the tensor along that dimension `num_beams` times. For example, an input shape
        (batch_size, seq_len, ...) becomes (batch_size, num_beams, seq_len, ...).

        Args:
            tensor (Array): The tensor to expand. Assumed to have batch size as the first dimension.
            num_beams (int): The number of beams to expand to.

        Returns:
            Array: The tensor expanded for beam search.
        """
        return jnp.broadcast_to(tensor[:, None], (tensor.shape[0], num_beams, *tensor.shape[1:]))

    def _adapt_logits_for_beam_search(self, logits):
        """Adapts logits for beam search, allowing model-specific customization.

        This hook can be overridden in specific model classes to implement custom
        beam search behavior. The base implementation returns logits unchanged.

        Args:
            logits: The raw logits from the model.

        Returns:
            The adapted logits for beam search.

        Note:
            Currently only FlaxMarianMTModel overrides this method.
        """
        return logits

    def _validate_model_kwargs(self, model_kwargs: dict[str, tp.Any]):
        """Validates model kwargs for generation.

        Checks that all provided kwargs are recognized by the model's generation
        methods. This helps catch typos in generation arguments.

        Args:
            model_kwargs: Dictionary of keyword arguments to validate.

        Raises:
            ValueError: If any kwargs are not recognized by the model.
        """
        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)

        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.__call__).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: jnp.ndarray | None = None,
        **model_kwargs,
    ) -> tuple[jnp.ndarray, dict[str, tp.Any]]:
        """Expands input tensors for generation with multiple return sequences.

        Repeats input_ids and all array-valued model_kwargs along the batch dimension
        to support generating multiple sequences per input (e.g., for beam search or
        multiple return sequences).

        Args:
            expand_size (int): Number of times to repeat each input. Defaults to 1.
            is_encoder_decoder (bool): Whether the model is encoder-decoder. If True,
                also expands encoder_outputs. Defaults to False.
            input_ids (jnp.ndarray | None): Input token IDs to expand. Defaults to None.
            **model_kwargs: Additional model kwargs. Array values will be repeated.

        Returns:
            tuple[jnp.ndarray, dict[str, Any]]: Tuple of (expanded_input_ids, expanded_model_kwargs).

        Raises:
            ValueError: If is_encoder_decoder=True but encoder_outputs is not in model_kwargs.
        """
        if expand_size == 1:
            return input_ids, model_kwargs

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], jax.Array):
                    dict_to_expand[key] = jnp.repeat(
                        dict_to_expand[key],
                        axis=0,
                        repeats=expand_size,
                    )
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat(repeats=expand_size, axis=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def generate(
        self,
        input_ids: Array,
        generation_config: GenerationConfig | None = None,
        prng_key: Array | None = None,
        trace: bool = True,
        logits_processor: LogitsProcessorList | None = None,
        **kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head.

        Parameters:
            input_ids (`Array` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            trace (`bool`, *optional*, defaults to `True`):
                Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
                considerably slower runtime.
            logits_processor (`LogitsProcessorList `, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            kwargs (`tp.Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the  If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`].
        """

        if generation_config is None:
            if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
                self.generation_config
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration)",
                        stacklevel=1,
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        self._validate_model_kwargs(model_kwargs.copy())

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        # set init values
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask") is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        if generation_config.decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError("`decoder_start_token_id` has to be defined for encoder-decoder generation.")

        # decoder-only models should use left-padding for generation (can't be checked with `trace=True`)
        if not self.config.is_encoder_decoder and not trace:
            if (
                generation_config.pad_token_id is not None
                and jnp.sum(input_ids[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        batch_size = input_ids.shape[0]

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    input_ids,
                    model_kwargs,
                )
            # prepare decoder_input_ids for generation
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                model_kwargs=model_kwargs,
            )

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length.  recommend setting `max_new_tokens` to control"
                " the maximum length of the generation.",
                UserWarning,
                stacklevel=1,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        else:  # by default let's always generate 10 new tokens
            if generation_config.max_length == GenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_seq_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing`max_new_tokens`."
            )

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            logits_processor=logits_processor,
        )

        if not generation_config.do_sample and generation_config.num_beams == 1:
            if generation_config.num_return_sequences > 1:
                input_ids, model_kwargs = self._expand_inputs_for_generation(
                    expand_size=generation_config.num_return_sequences,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    input_ids=input_ids,
                    **model_kwargs,
                )
            return self._greedy_search(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                logits_processor=logits_processor,
                trace=trace,
                model_kwargs=model_kwargs,
            )
        elif generation_config.do_sample and generation_config.num_beams == 1:
            if generation_config.num_return_sequences > 1:
                input_ids, model_kwargs = self._expand_inputs_for_generation(
                    expand_size=generation_config.num_return_sequences,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    input_ids=input_ids,
                    **model_kwargs,
                )
            logits_warper = self._get_logits_warper(generation_config=generation_config)
            return self._sample(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                trace=trace,
                model_kwargs=model_kwargs,
            )
        elif not generation_config.do_sample and generation_config.num_beams > 1:

            def _repeat_mask_info(mi: MaskInfo, repeats: int) -> MaskInfo:
                return jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x, repeats=repeats, axis=0) if isinstance(x, jax.Array) else x,
                    mi,
                )

            input_ids = self._expand_to_num_beams(input_ids, num_beams=generation_config.num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"]["last_hidden_state"] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"],
                    num_beams=generation_config.num_beams,
                )

            if "mask_info" in model_kwargs:
                model_kwargs["mask_info"] = _repeat_mask_info(model_kwargs["mask_info"], generation_config.num_beams)

            return self._beam_search(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                length_penalty=generation_config.length_penalty,
                early_stopping=generation_config.early_stopping,
                logits_processor=logits_processor,
                trace=trace,
                num_return_sequences=generation_config.num_return_sequences,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")

    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`]
        instances used for multinomial sampling.
        """
        warpers = LogitsProcessorList()

        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))

        return warpers

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        logits_processor: LogitsProcessorList | None,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = LogitsProcessorList()

        if (
            generation_config.min_length is not None
            and generation_config.min_length > 0
            and generation_config.eos_token_id is not None
            and generation_config.min_length > -1
        ):
            processors.append(
                MinLengthLogitsProcessor(
                    generation_config.min_length,
                    generation_config.eos_token_id,
                )
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        if generation_config.suppress_tokens is not None:
            processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None and len(generation_config.forced_decoder_ids) > 0:
                begin_index += generation_config.forced_decoder_ids[-1][0]
            processors.append(SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index))
        if getattr(generation_config, "forced_decoder_ids", None) is not None:
            forced_decoder_ids = [[input_ids_seq_length + i[0] - 1, i[1]] for i in generation_config.forced_decoder_ids]
            processors.append(ForceTokensLogitsProcessor(forced_decoder_ids))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        processors = self._merge_criteria_processor_list(processors, logits_processor)

        return processors

    def _merge_criteria_processor_list(
        self,
        default_list: LogitsProcessorList,
        custom_list: LogitsProcessorList,
    ) -> LogitsProcessorList:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def _greedy_search(
        self,
        input_ids: None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        logits_processor: LogitsProcessorList | None = None,
        trace: bool = True,
        model_kwargs: dict[str, Array] | None = None,
    ):
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        batch_size, cur_len = input_ids.shape

        pad_token_id = jnp.array(pad_token_id, jnp.int32)
        cur_len = jnp.array(cur_len)

        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        model = self.decode if self.config.is_encoder_decoder else self
        model_kwargs = self.prepare_inputs_for_generation(
            input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=None,
            **model_kwargs,
        )

        state = GreedyState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def greedy_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def greedy_search_body_fn(state):
            """state update fn."""
            call_kwargs = dict(state.model_kwargs)
            running_len = state.running_token.shape[1]
            for mask_key in ("attention_mask", "decoder_attention_mask"):
                mask = call_kwargs.get(mask_key, None)
                if mask is None:
                    continue
                if hasattr(mask, "shape") and len(mask.shape) > 0 and mask.shape[-1] != running_len:
                    call_kwargs.pop(mask_key, None)
            model_outputs = model(state.running_token, **call_kwargs)
            logits = model_outputs.logits[:, -1]

            logits = logits_processor(state.sequences, logits, state.cur_len)

            next_token = jnp.argmax(logits, axis=-1)
            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            if eos_token_id is not None:
                eos_arr = jnp.atleast_1d(jnp.array(eos_token_id, jnp.int32))
            else:
                eos_arr = None
            if eos_arr is not None:
                next_is_sent_finished = state.is_sent_finished | jnp.isin(next_token, eos_arr)
            else:
                next_is_sent_finished = state.is_sent_finished

            next_token = next_token[:, None]
            next_sequences = lax.dynamic_update_slice(
                state.sequences,
                next_token,
                (0, state.cur_len),
            )
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        if input_ids.shape[1] > 1:
            state = greedy_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
        else:
            state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

        return GreedySearchOutput(sequences=state.sequences)

    def _sample(
        self,
        input_ids: None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        prng_key: Array | None = None,
        logits_processor: LogitsProcessorList | None = None,
        logits_warper: LogitsProcessorList | None = None,
        trace: bool = True,
        model_kwargs: dict[str, Array] | None = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape
        pad_token_id = jnp.array(pad_token_id, jnp.int32)
        cur_len = jnp.array(cur_len)

        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=self.prepare_inputs_for_generation(
                input_ids,
                max_length=max_length,
                pad_token_id=pad_token_id,
                starts=None,
                **model_kwargs,
            ),
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)

            call_kwargs = dict(state.model_kwargs)
            running_len = state.running_token.shape[1]
            for mask_key in ("attention_mask", "decoder_attention_mask"):
                mask = call_kwargs.get(mask_key, None)
                if mask is None:
                    continue
                if hasattr(mask, "shape") and len(mask.shape) > 0 and mask.shape[-1] != running_len:
                    call_kwargs.pop(mask_key, None)
            model_outputs = model(state.running_token, **call_kwargs)

            logits = model_outputs.logits[:, -1]
            logits = logits_processor(state.sequences, logits, state.cur_len)
            logits = logits_warper(state.sequences, logits, state.cur_len)
            next_token = (
                jax.random.categorical(prng_key, logits, axis=-1) * ~state.is_sent_finished
                + pad_token_id * state.is_sent_finished
            )
            if eos_token_id is not None:
                eos_arr = jnp.atleast_1d(jnp.array(eos_token_id, jnp.int32))
            else:
                eos_arr = None
            if eos_arr is not None:
                next_is_sent_finished = state.is_sent_finished | jnp.isin(next_token, eos_arr)
            else:
                next_is_sent_finished = state.is_sent_finished

            next_token = next_token[:, None]
            next_sequences = lax.dynamic_update_slice(
                state.sequences,
                next_token,
                (0, state.cur_len),
            )
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(
                sample_search_cond_fn,
                sample_search_body_fn,
                state,
            )
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

        return SampleOutput(sequences=state.sequences)

    def _beam_search(
        self,
        input_ids: None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        length_penalty: float | None = None,
        early_stopping: bool | str | None = None,
        logits_processor: LogitsProcessorList | None = None,
        trace: bool = True,
        num_return_sequences: int | None = None,
        model_kwargs: dict[str, Array] | None = None,
    ):
        """
        This beam search function is heavily inspired by Flax's official example:
        https://github.com/google/flax/blob/main/examples/wmt/decode.py
        """

        def flatten_beam_dim(tensor):
            """Flattens the first two dimensions of a non-scalar array."""
            # ignore scalars (e.g. cache index)
            if tensor.ndim == 0:
                return tensor
            return tensor.reshape((tensor.shape[0] * tensor.shape[1], *tensor.shape[2:]))

        def flatten_mask_info(mi: MaskInfo, batch_size, num_beams):
            return jax.tree_util.tree_map(
                lambda t: t.reshape((batch_size * num_beams, *t.shape[2:]))
                if isinstance(t, jax.Array) and t.ndim >= 2
                else t,
                mi,
            )

        def unflatten_beam_dim(tensor, batch_size, num_beams):
            """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
            # ignore scalars (e.g. cache index)
            if tensor.ndim == 0:
                return tensor
            return tensor.reshape((batch_size, num_beams, *tensor.shape[1:]))

        def gather_beams(nested, beam_indices, batch_size, new_num_beams):
            """
            Gathers the beam slices indexed by beam_indices into new beam array.
            """
            batch_indices = jnp.reshape(
                jnp.arange(batch_size * new_num_beams) // new_num_beams,
                (batch_size, new_num_beams),
            )

            def gather_fn(tensor):
                # ignore scalars (e.g. cache index)
                if tensor.ndim == 0:
                    return tensor
                else:
                    return tensor[batch_indices, beam_indices]

            return jax.tree_util.tree_map(gather_fn, nested)

        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.generation_config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.generation_config.early_stopping
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.generation_config.num_return_sequences
        )

        batch_size, num_beams, cur_len = input_ids.shape

        pad_token_id = jnp.array(pad_token_id, jnp.int32)
        cur_len = jnp.array(cur_len)
        decoder_prompt_len = input_ids.shape[-1]

        # per batch,beam-item holding current token in loop.
        sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        running_sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        running_sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0, 0))

        # per batch,beam-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size, num_beams), dtype=jnp.bool_)

        # per batch,beam-item score, logprobs
        running_scores = jnp.tile(jnp.array([0.0] + [np.array(-1.0e7)] * (num_beams - 1)), [batch_size, 1])
        scores = jnp.ones((batch_size, num_beams)) * np.array(-1.0e7)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # flatten beam dim
        if "encoder_outputs" in model_kwargs:
            model_kwargs["encoder_outputs"]["last_hidden_state"] = flatten_beam_dim(
                model_kwargs["encoder_outputs"]["last_hidden_state"]
            )
        for kwarg in ["attention_mask", "decoder_attention_mask"]:
            if kwarg in model_kwargs:
                model_kwargs[kwarg] = flatten_beam_dim(model_kwargs[kwarg])

        model_kwargs = self.prepare_inputs_for_generation(
            flatten_beam_dim(input_ids),
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=None,
            **model_kwargs,
        )

        # initialize state
        state = BeamSearchState(
            cur_len=cur_len,
            running_sequences=running_sequences,
            running_scores=running_scores,
            sequences=sequences,
            scores=scores,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def beam_search_cond_fn(state):
            """beam search state termination condition fn."""
            not_max_length_yet = state.cur_len < max_length

            if early_stopping == "never" and length_penalty > 0.0:
                best_running_score = state.running_scores[:, :1] / ((max_length - decoder_prompt_len) ** length_penalty)
            else:
                denom = jnp.maximum(state.cur_len - decoder_prompt_len, 1)
                best_running_score = state.running_scores[:, :1] / (denom**length_penalty)

            worst_finished_score = jnp.where(
                state.is_sent_finished,
                jnp.min(state.scores, axis=1, keepdims=True),
                np.array(-1.0e7),
            )
            improvement_still_possible = jnp.any(best_running_score > worst_finished_score)

            still_open_beam = ~(jnp.all(state.is_sent_finished) & (early_stopping is True))

            return not_max_length_yet & still_open_beam & improvement_still_possible

        def beam_search_body_fn(state, input_ids_length=1):
            """beam search state update fn."""

            input_token = flatten_beam_dim(
                lax.dynamic_slice(
                    state.running_sequences,
                    (0, 0, state.cur_len - input_ids_length),
                    (batch_size, num_beams, input_ids_length),
                )
            )
            # Leave mask_info as the base; cache layer will compute step-specific mask
            model_outputs = model(input_token, **state.model_kwargs)

            logits = unflatten_beam_dim(model_outputs.logits[:, -1], batch_size, num_beams)
            cache = jax.tree_util.tree_map(
                lambda tensor: unflatten_beam_dim(tensor, batch_size, num_beams),
                model_outputs.past_key_values,
            )

            logits = self._adapt_logits_for_beam_search(logits)

            log_probs = jax.nn.log_softmax(logits)
            log_probs = logits_processor(
                flatten_beam_dim(state.running_sequences),
                flatten_beam_dim(log_probs),
                state.cur_len,
            )
            log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + jnp.expand_dims(state.running_scores, axis=2)
            vocab_size = log_probs.shape[2]
            log_probs = log_probs.reshape((batch_size, num_beams * vocab_size))

            beams_to_keep = 2 * num_beams
            topk_log_probs, topk_indices = lax.top_k(log_probs, k=beams_to_keep)
            topk_beam_indices = topk_indices // vocab_size
            topk_running_sequences = gather_beams(state.running_sequences, topk_beam_indices, batch_size, beams_to_keep)
            topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
            topk_sequences = lax.dynamic_update_slice(topk_running_sequences, topk_ids, (0, 0, state.cur_len))
            if eos_token_id is None:
                did_topk_just_finished = jnp.zeros_like(topk_sequences[:, :, state.cur_len], dtype=jnp.bool_)
            else:
                did_topk_just_finished = jnp.isin(topk_sequences[:, :, state.cur_len], eos_token_id)
            running_topk_log_probs = topk_log_probs + did_topk_just_finished * np.array(-1.0e7)

            next_topk_indices = lax.top_k(running_topk_log_probs, k=num_beams)[1]
            next_running_sequences, next_running_scores = gather_beams(
                [topk_sequences, running_topk_log_probs],
                next_topk_indices,
                batch_size,
                num_beams,
            )

            topk_log_probs = topk_log_probs / ((state.cur_len + 1 - decoder_prompt_len) ** length_penalty)
            beams_in_batch_are_full = jnp.broadcast_to(
                state.is_sent_finished.all(axis=-1, keepdims=True), did_topk_just_finished.shape
            ) & (early_stopping is True)
            add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
            topk_log_probs += add_penalty * np.array(-1.0e7)

            merged_sequences = jnp.concatenate([state.sequences, topk_sequences], axis=1)
            merged_scores = jnp.concatenate([state.scores, topk_log_probs], axis=1)
            merged_is_sent_finished = jnp.concatenate([state.is_sent_finished, did_topk_just_finished], axis=1)
            topk_merged_indices = lax.top_k(merged_scores, k=num_beams)[1]
            next_sequences, next_scores, next_is_sent_finished = gather_beams(
                [merged_sequences, merged_scores, merged_is_sent_finished],
                topk_merged_indices,
                batch_size,
                num_beams,
            )

            next_running_indices = gather_beams(topk_beam_indices, next_topk_indices, batch_size, num_beams)
            next_cache = gather_beams(cache, next_running_indices, batch_size, num_beams)
            model_outputs["past_key_values"] = jax.tree_util.tree_map(lambda x: flatten_beam_dim(x), next_cache)
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return BeamSearchState(
                cur_len=state.cur_len + 1,
                running_scores=next_running_scores,
                running_sequences=next_running_sequences,
                scores=next_scores,
                sequences=next_sequences,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        state = partial(beam_search_body_fn, input_ids_length=input_ids.shape[-1])(state)

        if not trace:
            state = self._run_loop_in_debug(beam_search_cond_fn, beam_search_body_fn, state)
        else:
            state = lax.while_loop(beam_search_cond_fn, beam_search_body_fn, state)

        none_finished = jnp.any(state.is_sent_finished, axis=1)
        sequences = jnp.where(none_finished[:, None, None], state.sequences, state.running_sequences)
        scores = jnp.where(none_finished[:, None], state.scores, state.running_scores)

        sequences = flatten_beam_dim(sequences[:, :num_return_sequences, :])
        scores = flatten_beam_dim(scores[:, :num_return_sequences])

        return BeamSearchOutput(sequences=sequences, scores=scores)

    @property
    def esurge_graphdef(self):
        """Returns a graph definition compatible with eSurge inference engine.

        eSurge requires models to use ragged page attention mechanisms (v2 or v3).
        If the current model uses a different attention mechanism, this property
        creates a new graph definition with ragged_page_attention_v3.

        Returns:
            nn.GraphDef: Graph definition with eSurge-compatible attention mechanism.

        Note:
            This creates only the graph structure, not a complete model. Use
            `esurge_compatible_model` if you need a full model instance.

        Example:
            >>> gdef = model.esurge_graphdef
            >>> # Use gdef for creating eSurge-compatible model instances
        """
        gdef = self.graphdef
        if self.config.attn_mechanism not in ["ragged_page_attention_v2", "ragged_page_attention_v3"]:
            gdef = self.new_graphdef(attn_mechanism="ragged_page_attention_v3", recursive_update=True)
        return gdef

    @property
    def esurge_compatible_model(self):
        """Returns a model instance compatible with eSurge inference engine.

        eSurge requires models to use ragged page attention mechanisms (v2 or v3).
        If the current model uses a different attention mechanism, this property
        returns a new model instance with ragged_page_attention_v3 while preserving
        all parameters and state.

        Returns:
            Self: Model instance with eSurge-compatible attention mechanism.

        Note:
            If the model already uses a compatible attention mechanism, returns self.
            Otherwise, builds a new graph definition with ragged_page_attention_v3 and
            merges the existing parameters/state, leaving the original model unchanged.

        Example:
            >>> # Get eSurge-compatible version of model
            >>> esurge_model = model.esurge_compatible_model
            >>> # Now safe to use with eSurge inference
            >>> outputs = esurge_model.esurge_generate("Hello world")
        """
        if self.config.attn_mechanism in ["ragged_page_attention_v2", "ragged_page_attention_v3"]:
            return self
        compat_graphdef = self.new_graphdef(attn_mechanism="ragged_page_attention_v3")
        return self.merge_module(compat_graphdef, self.graphstate, self.graphother)

    def pause_esurge(self, engine_id: str | None = None) -> None:
        """Pause eSurge engine(s) for this model.

        Pauses the background scheduler of eSurge engines without clearing queued state.
        This is useful for temporarily freeing resources while keeping the engine ready
        for quick resumption.

        Args:
            engine_id: Optional specific engine cache key to pause. If None, pauses all
                engines associated with this model.

        Example:
            >>> # Pause all engines for this model
            >>> model.pause_esurge()
            >>>
            >>> # Later, generate will auto-resume
            >>> outputs = model.esurge_generate("prompt")  # Auto-resumes!
        """
        model_hash = self.static_hash(["attn_mechanism"])

        if engine_id is not None:
            # Pause specific engine
            if engine_id in _ESURGE_MAP_CACHE:
                eng = _ESURGE_MAP_CACHE[engine_id]
                eng.pause()
                if not getattr(eng, "silent_mode", False):
                    logger.info(f"Paused eSurge engine: {engine_id}")
            else:
                logger.warning(f"Engine not found: {engine_id}")
        else:
            # Pause all engines for this model
            paused_count = 0
            should_log = False
            for cache_key, engine in _ESURGE_MAP_CACHE.items():
                if cache_key.startswith(f"{model_hash}-"):
                    engine.pause()
                    paused_count += 1
                    should_log = should_log or not getattr(engine, "silent_mode", False)
            if paused_count > 0:
                if should_log:
                    logger.info(f"Paused {paused_count} eSurge engine(s) for this model")
            else:
                logger.info("No eSurge engines found to pause for this model")

    def resume_esurge(self, engine_id: str | None = None) -> None:
        """Resume paused eSurge engine(s) for this model.

        Resumes the background scheduler of paused eSurge engines, making them
        ready to process generation requests again.

        Args:
            engine_id: Optional specific engine cache key to resume. If None, resumes all
                engines associated with this model.

        Example:
            >>> # Pause engines to free resources
            >>> model.pause_esurge()
            >>>
            >>> # Manually resume when needed
            >>> model.resume_esurge()
            >>> outputs = model.esurge_generate("prompt")
        """
        model_hash = self.static_hash(["attn_mechanism"])

        if engine_id is not None:
            # Resume specific engine
            if engine_id in _ESURGE_MAP_CACHE:
                eng = _ESURGE_MAP_CACHE[engine_id]
                eng.resume()
                if not getattr(eng, "silent_mode", False):
                    logger.info(f"Resumed eSurge engine: {engine_id}")
            else:
                logger.warning(f"Engine not found: {engine_id}")
        else:
            # Resume all engines for this model
            resumed_count = 0
            should_log = False
            for cache_key, engine in _ESURGE_MAP_CACHE.items():
                if cache_key.startswith(f"{model_hash}-"):
                    engine.resume()
                    resumed_count += 1
                    should_log = should_log or not getattr(engine, "silent_mode", False)
            if resumed_count > 0:
                if should_log:
                    logger.info(f"Resumed {resumed_count} eSurge engine(s) for this model")
            else:
                logger.info("No eSurge engines found to resume for this model")

    def list_esurge_engines(self) -> list[dict]:
        """List all cached eSurge engines for this model.

        Returns a list of dictionaries containing information about each cached engine,
        including its status (running/paused), number of requests, and configuration hash.

        Returns:
            List of dicts with engine information:
                - cache_key: The cache key for this engine
                - paused: Whether the engine is paused
                - running_requests: Number of currently running requests
                - pending_requests: Number of pending requests
                - max_num_seqs: Maximum concurrent sequences

        Example:
            >>> engines = model.list_esurge_engines()
            >>> for engine in engines:
            ...     print(f"Engine {engine['cache_key']}: "
            ...           f"Paused={engine['paused']}, "
            ...           f"Running={engine['running_requests']}")
        """
        model_hash = self.static_hash(["attn_mechanism"])
        engines_info = []

        for cache_key, engine in _ESURGE_MAP_CACHE.items():
            if cache_key.startswith(f"{model_hash}-"):
                info = {
                    "cache_key": cache_key,
                    "paused": getattr(engine, "_paused", False),
                    "running_requests": getattr(engine, "num_running_requests", 0),
                    "pending_requests": getattr(engine, "num_pending_requests", 0),
                    "max_num_seqs": getattr(engine, "_max_num_seqs", None),
                }
                engines_info.append(info)

        return engines_info

    def get_relevant_esurge(
        self,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        max_num_seqs: int | None = None,
    ):
        """Retrieves a relevant eSurge engine instance from the cache.

        This method searches for an existing eSurge engine in the cache that matches
        the current model. If tokenizer or max_num_seqs are None, it returns the most
        recently created engine for this model. If no engine exists and parameters are
        missing, it uses sensible defaults.

        Args:
            tokenizer: Optional tokenizer path or instance. If None, retrieves from
                the most recent cached engine for this model.
            max_num_seqs: Optional maximum number of concurrent sequences. If None,
                uses value from cached engine or defaults to 32 if no cache exists.

        Returns:
            eSurge engine instance if found in cache, None otherwise.

        Example:
            >>> # Try to get existing engine with default params
            >>> engine = model.get_relevant_esurge()
            >>> if engine:
            ...     outputs = engine.generate("Hello world")
            >>>
            >>> # Get engine with specific tokenizer
            >>> engine = model.get_relevant_esurge(tokenizer="gpt2")
        """
        model_hash = self.static_hash(["attn_mechanism"])

        # Search for any cached engine matching this model
        matching_engines = []
        for cache_key, engine in _ESURGE_MAP_CACHE.items():
            if cache_key.startswith(f"{model_hash}-"):
                matching_engines.append(engine)

        if not matching_engines:
            return None

        # If tokenizer and max_num_seqs are both provided, try exact match
        if tokenizer is not None and max_num_seqs is not None:
            # Try to find exact match based on parameters
            for engine in matching_engines:
                if (
                    hasattr(engine, "tokenizer")
                    and hasattr(engine, "_max_num_seqs")
                    and engine._max_num_seqs == max_num_seqs
                ):
                    return engine

        # Return the most recently added engine (last in cache)
        return matching_engines[-1]

    def get_esurge(
        self,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        max_model_len: int | None = None,
        min_input_pad: int | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        hbm_utilization: float | None = None,
        page_size: int | None = None,
        enable_prefix_caching: bool | None = None,
        runner_verbose: bool | None = None,
        decode_truncated_prompt: bool | None = None,
        destroy_pages_on_pause: bool | None = None,
        silent_mode: bool | None = None,
    ):
        """Gets or creates an eSurge engine with the specified parameters.

        This method intelligently retrieves an existing cached engine or creates a new one.
        For any parameter that is None, it will:
        1. Try to retrieve the value from a cached engine
        2. If no cached engine exists, use sensible defaults
        3. Only require tokenizer if no cached engine is available

        Args:
            tokenizer: Tokenizer path or instance. If None, retrieves from cached engine.
                Required only if no cached engine exists.
            max_model_len: Maximum sequence length. Defaults to model's max position embeddings.
            min_input_pad: Minimum padding for input sequences. Defaults to 16.
            max_num_seqs: Maximum number of concurrent sequences. Defaults to 32.
            max_num_batched_tokens: Maximum tokens per batch. Defaults to None.
            hbm_utilization: Fraction of HBM to use for KV cache. Defaults to 0.85.
            page_size: Size of memory pages for paged attention. Defaults to 128.
            enable_prefix_caching: Enable prefix caching. Defaults to True.
            runner_verbose: Enable verbose logging. Defaults to False.
            decode_truncated_prompt: Decode truncated prompts. Defaults to True.
            destroy_pages_on_pause: Free memory on pause. Defaults to True.

        Returns:
            eSurge engine instance, either from cache or newly created.

        Raises:
            ValueError: If tokenizer is required but not provided and no cached engine exists.

        Example:
            >>> # First call with tokenizer (creates new engine)
            >>> engine = model.get_esurge(tokenizer="meta-llama/Llama-2-7b-hf")
            >>>
            >>> # Subsequent calls without parameters (reuses cached engine)
            >>> engine = model.get_esurge()
            >>>
            >>> # Override specific parameters
            >>> engine = model.get_esurge(max_num_seqs=128)
        """
        # Check if all configurable parameters are None (user wants cached engine)
        all_none = all(
            param is None
            for param in [
                tokenizer,
                max_model_len,
                min_input_pad,
                max_num_seqs,
                max_num_batched_tokens,
                hbm_utilization,
                page_size,
                enable_prefix_caching,
                runner_verbose,
                decode_truncated_prompt,
                destroy_pages_on_pause,
                silent_mode,
            ]
        )

        # If all params are None, try to return cached engine directly
        if all_none:
            cached_engine = self.get_relevant_esurge()
            if cached_engine is not None:
                # Auto-resume if paused
                if hasattr(cached_engine, "_paused") and cached_engine._paused:
                    if not getattr(cached_engine, "silent_mode", False):
                        logger.info("Auto-resuming paused eSurge engine")
                    cached_engine.resume()
                return cached_engine
            # No cache exists, will use defaults below

        # Set default for max_model_len
        if max_model_len is None:
            max_model_len = self.config.granted_freq_max_position_embedding

        # Try to get a relevant cached engine if any parameter is None
        any_none = any(
            param is None
            for param in [
                tokenizer,
                min_input_pad,
                max_num_seqs,
                max_num_batched_tokens,
                hbm_utilization,
                page_size,
                enable_prefix_caching,
                runner_verbose,
                decode_truncated_prompt,
                destroy_pages_on_pause,
                silent_mode,
            ]
        )

        cached_engine = None
        if any_none:
            cached_engine = self.get_relevant_esurge(tokenizer=tokenizer, max_num_seqs=max_num_seqs)

        # Extract parameters from cached engine or use defaults
        if tokenizer is None:
            if cached_engine is not None:
                tokenizer = cached_engine.tokenizer
            else:
                raise ValueError(
                    "tokenizer is required when no cached eSurge engine exists. "
                    "Either provide a tokenizer or create an engine first by calling get_esurge with a tokenizer."
                )

        # Set defaults for other parameters
        if min_input_pad is None:
            min_input_pad = getattr(cached_engine, "_min_input_pad", 16) if cached_engine else 16
        if max_num_seqs is None:
            max_num_seqs = getattr(cached_engine, "_max_num_seqs", 32) if cached_engine else 32
        if max_num_batched_tokens is None:
            max_num_batched_tokens = getattr(cached_engine, "_max_num_batched_tokens", None) if cached_engine else None
        if hbm_utilization is None:
            hbm_utilization = getattr(cached_engine, "_hbm_utilization", 0.85) if cached_engine else 0.85
        if page_size is None:
            page_size = getattr(cached_engine, "_page_size", 128) if cached_engine else 128
        if enable_prefix_caching is None:
            enable_prefix_caching = getattr(cached_engine, "_enable_prefix_caching", True) if cached_engine else True
        if runner_verbose is None:
            runner_verbose = getattr(cached_engine, "_runner_verbose", False) if cached_engine else False
        if decode_truncated_prompt is None:
            decode_truncated_prompt = getattr(cached_engine, "_decode_truncated_prompt", True) if cached_engine else True
        if destroy_pages_on_pause is None:
            destroy_pages_on_pause = getattr(cached_engine, "_destroy_pages_on_pause", True) if cached_engine else True
        if silent_mode is None:
            silent_mode = getattr(cached_engine, "silent_mode", False) if cached_engine else False

        # Build the configuration dict
        model_hash = self.static_hash(["attn_mechanism"])
        extra_dict = dict(
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            min_input_pad=min_input_pad,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            runner_verbose=runner_verbose,
            decode_truncated_prompt=decode_truncated_prompt,
            destroy_pages_on_pause=destroy_pages_on_pause,
            silent_mode=silent_mode,
        )

        # Check if this exact configuration exists in cache
        extra_dict_str = pprint.pformat(extra_dict)
        bytes_in = hashlib.md5(extra_dict_str.encode("utf-8")).digest()
        extra_dict_hash = int.from_bytes(bytes_in, byteorder="big", signed=True)
        esurge_hash = f"{model_hash}-{extra_dict_hash}"

        if esurge_hash in _ESURGE_MAP_CACHE:
            esurge = _ESURGE_MAP_CACHE[esurge_hash]
            # Auto-resume if paused
            if hasattr(esurge, "_paused") and esurge._paused:
                if not getattr(esurge, "silent_mode", False):
                    logger.info("Auto-resuming paused eSurge engine")
                esurge.resume()
        else:
            # Create new engine
            from easydel.inference import eSurge

            esurge = eSurge(model=self, **extra_dict)
            _ESURGE_MAP_CACHE[esurge_hash] = esurge

        if esurge.num_running_requests == 0 and esurge.num_pending_requests == 0:
            esurge.update_model_weights(self)

        return esurge

    def _call_esurge_engine(
        self,
        engine,
        prompts: list[dict[str, str]] | list[str] | str,
        tools: list[dict] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
        use_tqdm: bool = False,
    ):
        """Internal helper to call an eSurge engine with the appropriate method.

        Determines whether to use chat or completion mode and calls the
        corresponding engine method.

        Args:
            engine: The eSurge engine instance to use.
            prompts: Input prompts (string, list of strings, or list of message dicts).
            tools: Optional tool definitions for chat mode.
            sampling_params: Generation parameters.
            request_id: Optional request ID.
            stream: Whether to stream results.
            chat_template: Optional chat template.

        Returns:
            Generation results from the engine.
        """
        # Determine if prompts is chat format (list of message dicts)
        is_chat_mode = isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], dict)

        if is_chat_mode:
            # Chat mode: use engine.chat()
            return engine.chat(
                messages=prompts,
                tools=tools,
                sampling_params=sampling_params,
                request_id=request_id,
                stream=stream,
                chat_template=chat_template,
            )
        else:
            # Completion mode: use engine.stream() or engine.generate()
            if stream:
                return engine.stream(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    request_id=request_id,
                )
            else:
                return engine.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    use_tqdm=use_tqdm,
                )

    def esurge_generate(
        self,
        prompts: list[dict[str, str]] | list[str] | str,
        tools: list[dict] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
        *,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        max_model_len: int | None = None,
        min_input_pad: int | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        hbm_utilization: float | None = None,
        page_size: int | None = None,
        enable_prefix_caching: bool | None = None,
        runner_verbose: bool | None = None,
        decode_truncated_prompt: bool | None = None,
        destroy_pages_on_pause: bool | None = None,
        silent_mode: bool | None = None,
        use_tqdm: bool = False,
    ):
        """High-level interface for text generation using eSurge engine.

        This method provides a convenient way to generate text using the eSurge inference
        engine with automatic caching and configuration management. It supports both chat
        and completion modes, with optional streaming.

        All engine configuration parameters are optional. When omitted, the method will:
        1. Try to retrieve values from a cached engine for this model
        2. Fall back to sensible defaults if no cached engine exists
        3. Only require tokenizer on the first call when no cache exists

        Args:
            prompts: Input prompts. Can be:
                - Single string for simple completion
                - List of strings for batch completion
                - List of dicts with 'role' and 'content' keys for chat mode
            tools: Optional list of tool/function definitions for function calling in chat mode.
            sampling_params: Generation parameters (temperature, top_p, max_tokens, etc.).
                Defaults to SamplingParams(max_tokens=128) if None.
            request_id: Optional unique identifier for tracking. Auto-generated if None.
            stream: If True, returns an iterator for streaming generation.
                If False, returns complete results.
            chat_template: Optional custom Jinja2 template for chat formatting.
            tokenizer: Tokenizer path or instance. Required only on first call if no cached
                engine exists. Subsequent calls can omit this to reuse the cached tokenizer.
            max_model_len: Maximum sequence length. Defaults to model's max position embeddings.
            min_input_pad: Minimum padding for input sequences. Defaults to 16.
            max_num_seqs: Maximum number of concurrent sequences. Defaults to 32.
            max_num_batched_tokens: Maximum tokens per batch. Defaults to None (auto-calculate).
            hbm_utilization: Fraction of HBM to use for KV cache. Defaults to 0.85.
            page_size: Size of memory pages for paged attention. Defaults to 128.
            enable_prefix_caching: Enable prefix caching for shared prompts. Defaults to True.
            runner_verbose: Enable verbose logging in the model runner. Defaults to False.
            decode_truncated_prompt: Decode and display truncated prompts. Defaults to True.
            destroy_pages_on_pause: Free memory pages when requests are paused. Defaults to True.

        Returns:
            - For chat mode (prompts is list[dict]):
                - If stream=True: Iterator[RequestOutput] with delta updates
                - If stream=False: RequestOutput with complete response
            - For completion mode (prompts is str or list[str]):
                - If stream=True: Iterator[RequestOutput] with delta updates
                - If stream=False: list[RequestOutput] with complete responses

        Example:
            >>> # Simple completion
            >>> outputs = model.esurge_generate("Tell me about AI")
            >>> print(outputs[0].get_text())
            >>>
            >>> # Streaming completion
            >>> for chunk in model.esurge_generate("Tell me a story", stream=True):
            ...     print(chunk.delta_text, end="", flush=True)
            >>>
            >>> # Chat mode
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "What is 2+2?"}
            ... ]
            >>> response = model.esurge_generate(messages)
            >>> print(response.get_text())
        """
        # Get or create eSurge engine with specified parameters
        esurge = self.get_esurge(
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            min_input_pad=min_input_pad,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            runner_verbose=runner_verbose,
            decode_truncated_prompt=decode_truncated_prompt,
            destroy_pages_on_pause=destroy_pages_on_pause,
            silent_mode=silent_mode,
        )

        # Call the engine with the appropriate method
        return self._call_esurge_engine(
            esurge,
            prompts=prompts,
            tools=tools,
            sampling_params=sampling_params,
            request_id=request_id,
            stream=stream,
            chat_template=chat_template,
            use_tqdm=use_tqdm,
        )


_ESURGE_MAP_CACHE = {}
