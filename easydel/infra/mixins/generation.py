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
import inspect
import typing as tp
import warnings
from functools import cached_property, partial

import chex
import jax
import numpy as np
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
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
from easydel.layers.caching import PagesCache, PagesCacheMetaData

from ..base_config import EasyDeLBaseConfig
from ..modeling_outputs import BeamSearchOutput, GreedySearchOutput, SampleOutput

if tp.TYPE_CHECKING:
    from easydel.inference import vInference, vInferenceConfig, vInferencePreCompileConfig
    from easydel.infra.utils import ProcessingClassType
    from easydel.layers.caching import TransformerCache, TransformerCacheMetaData

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

    cur_len: chex.Array
    sequences: chex.Array
    running_token: chex.Array
    is_sent_finished: chex.Array
    model_kwargs: dict[str, chex.Array]


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

    cur_len: chex.Array
    sequences: chex.Array
    running_token: chex.Array
    is_sent_finished: chex.Array
    prng_key: chex.Array
    model_kwargs: dict[str, chex.Array]


@auto_pytree
class BeamSearchState:
    """
    State for beam search generation.

    Attributes:
        cur_len (chex.Array): Current length of the generated sequence.
        running_sequences (chex.Array): Generated sequences being tracked in the beam.
        running_scores (chex.Array): Scores of the sequences being tracked in the beam.
        sequences (chex.Array): Best generated sequences.
        scores (chex.Array): Scores of the best generated sequences.
        is_sent_finished (chex.Array): Boolean array indicating if a sequence is finished.
        model_kwargs (tp.Dict[str, chex.Array]): Model specific keyword arguments.
    """

    cur_len: chex.Array
    running_sequences: chex.Array
    running_scores: chex.Array
    sequences: chex.Array
    scores: chex.Array
    is_sent_finished: chex.Array
    model_kwargs: dict[str, chex.Array]


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

    def create_paged_metadata(
        self,
        hbm_utilization: float,
        page_size: int,
        max_model_length: int,
    ) -> PagesCacheMetaData:
        """
        Creates the static configuration metadata required for initializing a Paged KV Cache.

        This method gathers necessary parameters from the model's configuration
        (like number of layers, heads, dimensions) and combines them with the provided
        arguments to instantiate and return a `PagesCacheMetaData` object.
        This metadata object defines the structure and allocation parameters for the paged cache.

        Returns:
            PagesCacheMetaData: An initialized metadata object containing the
                static configuration for the paged cache.
        """
        num_hidden_layers = _safepick(self.config, "num_hidden_layers")

        num_key_value_heads = _safepick(self.config, "num_key_value_heads")
        num_attention_heads = _safepick(self.config, "num_attention_heads")

        hidden_size = _safepick(self.config, "hidden_size")

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        head_dim = _safepick(self.config, "head_dim")
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        return PagesCacheMetaData.create(
            mesh=self.mesh,
            partition_manager=self.config.partition_manager,
            kvdtype=self.config.kvdtype,
            max_model_length=max_model_length,
            num_hidden_layers=num_hidden_layers,
            num_kv_heads=num_key_value_heads,
            kv_head_dim_size=head_dim,
            k_headdim=None,
            v_headdim=None,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
        )

    def create_cache_metadata(
        self,
        batch_size: int,
        max_length: int,
        pad_token_id: int | None = None,
    ) -> TransformerCacheMetaData:
        """
        Creates the metadata required for initializing a standard (non-paged) KV Cache.

        This method gathers parameters like layer count, head dimensions, and determines
        the appropriate padding token ID to instantiate and return a
        `TransformerCacheMetaData` object suitable for a standard sequential KV cache.

        Args:
            batch_size (int): The batch size for which the cache is being configured.
            max_length (int): The maximum sequence length the cache needs to support.
            pad_token_id (int | None): The ID of the padding token. If None, it attempts
                to find it from `self.generation_config` or `self.config`, defaulting to 0.

        Returns:
            TransformerCacheMetaData: An initialized metadata object for a standard KV cache.
        """
        if pad_token_id is None:
            if hasattr(self, "generation_config"):
                pad_token_id = self.generation_config.pad_token_id
            elif hasattr(self.config, "pad_token_id"):
                pad_token_id = self.config.pad_token_id
            else:
                pad_token_id = 0
        head_dim = getattr(self.config, "head_dim", None)
        if head_dim is None:
            head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_key_value_heads = getattr(self.config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = self.config.num_attention_heads

        from easydel.layers.caching import TransformerCacheMetaData

        return TransformerCacheMetaData.create(
            num_hidden_layers=self.config.num_hidden_layers,
            pad_token_id=pad_token_id,
            batch_size=batch_size,
            sequence_length=max_length,
            num_heads=num_key_value_heads,
            head_dim=head_dim,
        )

    def init_pages(
        self,
        metadata: PagesCacheMetaData | None = None,
        page_size: int | None = None,
        hbm_utilization: float | None = None,
        max_model_length: int | None = None,
    ) -> PagesCache:
        """
        Initializes and returns the actual Paged Attention KV Cache tensors.

        This method orchestrates the creation of the `PagesCache`. It either uses
        a pre-existing `PagesCacheMetaData` object passed via the `metadata`
        argument, or if `metadata` is None, it first creates the metadata by calling
        `self.create_paged_metadata` using the other provided arguments (page_size,
        batch_size, etc.).

        Finally, it calls `PagesCache.init_cache` to allocate the necessary
        paged tensors (`key_pages`, `value_pages` for each layer) based on the
        metadata, model's mesh, dtype, partition manager, and quantization settings.

        Args:
            metadata (tp.Optional[PagesCacheMetaData]): An optional pre-configured
                metadata object. If provided, other arguments like page_size, batch_size etc.,
                are ignored for metadata creation.
            page_size (tp.Optional[int]): Number of tokens per page. Required if `metadata` is None.
            hbm_utilization (tp.Optional[float]): Target HBM usage. Required if `metadata` is None.

        Returns:
            PagesCache: An initialized PagesCache object containing the allocated
                cache tensors (views) for all layers.

        Raises:
            AssertionError: If `metadata` is None and any of the required arguments
                (page_size, batch_size, max_sequences, dtype, hbm_utilization) are also None.
        """
        if metadata is None:
            assert page_size is not None, "if your not passing metadata you should pass `page_size`"
            assert hbm_utilization is not None, "if your not passing metadata you should pass `hbm_utilization`"
            assert max_model_length is not None, "if your not passing metadata you should pass `max_model_length`"

            metadata = self.create_paged_metadata(
                hbm_utilization=hbm_utilization,
                page_size=page_size,
                max_model_length=max_model_length,
            )
        return PagesCache.init_cache(
            mesh=self.config.mesh,
            metadata=metadata,
            partition_manager=self.config.partition_manager,
            quantizer=self._quant_class(
                quantization_method=self.config.kv_cache_quantization_method,
                block_size=self.config.kv_cache_quantization_blocksize,
                quantization_platform=self.config.platform,
            ),
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: int | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ) -> TransformerCache:
        """
        Initializes and returns a standard (non-paged) Key-Value cache.

        This method first creates the necessary metadata using `create_cache_metadata`
        and then calls `TransformerCache.init_cache` to allocate and initialize
        the cache tensors based on the model's configuration, dtype, sharding,
        quantization settings, and provided batch size and maximum length.

        Args:
            batch_size (int): The batch size for the cache.
            max_length (int): The maximum sequence length the cache needs to support.
            starts (int | None): Optional starting positions for the cache sequences.
                If provided, influences the initial state. Defaults to None (usually 0).
            shardings (dict | None): Optional dictionary specifying sharding configurations.
                (Note: This argument appears unused in the current implementation shown).
            pad_token_id (int | None): The ID of the padding token. If None, it's inferred.

        Returns:
            TransformerCache: An initialized standard TransformerCache object.
        """

        from easydel.layers.caching import TransformerCache

        return TransformerCache.init_cache(
            dtype=self.config.kvdtype,
            partition_manager=self.config.partition_manager,
            metadata=self.create_cache_metadata(
                batch_size=batch_size,
                max_length=max_length,
                pad_token_id=pad_token_id,
            ),
            quantizer=self._quant_class(
                quantization_method=self.config.kv_cache_quantization_method,
                block_size=self.config.kv_cache_quantization_blocksize,
                quantization_platform=self.config.platform,
            ),
            mesh=self.config.mesh,
            starts=starts,
            mask_type_details=self.config.get_mask_details(),
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
    def compute_prefill_length(array, padding_id) -> chex.Array:
        """
        Calculates the number of padding tokens at the beginning of each sequence.

        This is useful for determining the actual starting position in a KV cache when
        dealing with left-padded inputs.

        Args:
            array (chex.Array): The input token ID array, typically shape (batch_size, sequence_length).
            padding_id (int): The token ID used for padding.

        Returns:
            chex.Array: An array of shape (batch_size,) containing the number of leading
                padding tokens for each sequence in the batch.
        """
        return jnp.sum(jnp.cumsum(array == padding_id, axis=-1) == 0, axis=-1)

    @staticmethod
    def compute_prefill_length_from_mask(mask) -> chex.Array:
        """
        Calculates the number of padding tokens at the beginning of each sequence.

        This is useful for determining the actual starting position in a KV cache when
        dealing with left-padded inputs.

        Returns:
            chex.Array: An array of shape (batch_size,) containing the number of leading
                padding tokens for each sequence in the batch.
        """
        return jnp.sum(jnp.cumsum(mask, axis=-1) == 0, axis=-1)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        shardings: int | None = None,
        attention_mask: chex.Array | None = None,
        token_type_ids: chex.Array | None = None,
    ) -> dict[str, tp.Any]:
        """
        Sets up the initial inputs required for starting autoregressive generation.

        This function initializes the Key-Value cache (`past_key_values`) using `init_cache`,
        calculates the initial `position_ids` based on the input `attention_mask` (or assumes
        a contiguous range if no mask is provided), and prepares an extended `attention_mask`
        suitable for caching. It ensures inputs are placed on the correct devices/shards.

        Args:
            input_ids (chex.Array): The initial sequence of token IDs. Shape (batch_size, seq_length).
            max_length (int): The maximum sequence length that the KV cache should support.
            pad_token_id (int): The ID used for padding tokens. Used to calculate `starts` if not provided.
            starts (int | None): Optional pre-calculated starting positions (number of leading pads).
                If None, calculated using `compute_prefill_length`.
            shardings (dict | None): Optional sharding configuration passed to `init_cache`.
            attention_mask (tp.Optional[chex.Array]): An optional mask indicating which tokens
                should be attended to. Shape (batch_size, seq_length).
            token_type_ids (tp.Optional[chex.Array]): Optional segment IDs for models that use them.

        Returns:
            dict: A dictionary containing the prepared inputs, typically including:
                - "past_key_values": The initialized KV cache.
                - "attention_mask": The extended attention mask for generation.
                - "position_ids": The calculated initial position IDs.
                - "token_type_ids": (Optional) Prepared token type IDs.
                This dictionary is then passed through `prepare_inputs_for_call`.
        """
        batch_size, seq_length = input_ids.shape
        if starts is None:
            if attention_mask is not None:
                starts = self.compute_prefill_length_from_mask(attention_mask)
            else:
                starts = self.compute_prefill_length(input_ids, pad_token_id)
        past_key_values = self.init_cache(
            batch_size,
            max_length,
            starts,
            shardings,
            pad_token_id,
        )
        sharding = input_ids.sharding if hasattr(input_ids, "sharding") else None
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="b1")
        if attention_mask is not None:
            if attention_mask.dtype != jnp.bool:
                attention_mask = attention_mask.astype("b1")
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask,
                attention_mask,
                (0, 0),
            )
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
        if token_type_ids is not None:
            token_type_ids = lax.dynamic_update_slice(
                jnp.zeros((batch_size, max_length), dtype="i4"),
                token_type_ids,
                (0, 0),
            )
            token_type_ids = jax.device_put(token_type_ids, device=sharding)
        calldict = {
            "past_key_values": past_key_values,
            "attention_mask": jax.device_put(extended_attention_mask, device=sharding),
            "position_ids": jax.device_put(position_ids, device=sharding),
        }
        if token_type_ids is not None:
            calldict.update({"token_type_ids": token_type_ids})

        return self.prepare_inputs_for_call(**calldict)

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
        self,
        model_kwargs: dict[str, chex.Array],
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

    def _get_compile_model_kwargs(
        self,
        batch_size: int,
        input_tokens_length: int,
        input_sharding: jax.sharding.PartitionSpec,
        rngs: jax.random.PRNGKey,
        vision_included: bool = False,
        vision_batch_size: int = 1,
        vision_channels: int = 3,
        vision_height: int | None = None,
        vision_width: int | None = None,
        required_props: tp.Mapping[str, dict[str, tp.Any]] | None = None,
        **kwargs,
    ) -> dict[str, tp.Any]:
        """
        Generates a dictionary of placeholder keyword arguments needed for model compilation.

        Creates dummy tensors (like `input_ids`, `attention_mask`) with the specified shapes
        and sharding, along with necessary RNGs, to allow JAX to trace and compile the
        model's forward pass without needing real data. This is often used for AOT compilation
        or initialization checks. Specific models might override this to add more required inputs.

        Args:
            batch_size (int): The batch size for the dummy inputs.
            input_tokens_length (int): The sequence length for the dummy inputs.
            input_sharding (jax.sharding.PartitionSpec): Sharding for the dummy inputs.
            rngs (jax.random.PRNGKey): RNG keys required by the model (e.g., for dropout).
            vision_included (bool): Flag indicating if vision inputs are needed (for multimodal).
            vision_batch_size (int): Batch size for dummy vision inputs.
            vision_channels (int): Channels for dummy vision inputs.
            vision_height (tp.Optional[int]): Height for dummy vision inputs.
            vision_width (tp.Optional[int]): Width for dummy vision inputs.
            required_props (tp.Optional[tp.Mapping[str, tp.Dict[str, tp.Any]]]): Optional
                additional properties extracted by `_create_required_props_from_kwargs`.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing dummy keyword arguments suitable for model compilation.
        """
        deteshape = (batch_size, input_tokens_length)
        return dict(
            input_ids=jnp.ones(deteshape, dtype="i4", device=input_sharding),
            attention_mask=jnp.ones(deteshape, dtype="b1", device=input_sharding),
            rng=rngs,
        )

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
            input_ids (chex.Array): The input token IDs for the encoder.
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
        model_kwargs: dict[str, chex.Array] | None = None,
    ) -> chex.Array:
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
            chex.Array: The initial `decoder_input_ids` tensor, shape (batch_size, 1).
        """
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            if decoder_input_ids is not None:
                return decoder_input_ids
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        return jnp.array(decoder_start_token_id, dtype="i4").reshape(1, -1).repeat(batch_size, axis=0)

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
            tensor (chex.Array): The tensor to expand. Assumed to have batch size as the first dimension.
            num_beams (int): The number of beams to expand to.

        Returns:
            chex.Array: The tensor expanded for beam search.
        """
        return jnp.broadcast_to(tensor[:, None], (tensor.shape[0], num_beams, *tensor.shape[1:]))

    def _adapt_logits_for_beam_search(self, logits):
        """
        This function can be overwritten in the specific modeling_flax_<model-name>.py classes to allow for custom beam
        search behavior. Note that the only model that overwrites this method is [`~transformes.FlaxMarianMTModel`].
        """
        return logits

    def _validate_model_kwargs(self, model_kwargs: dict[str, tp.Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
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
        input_ids: chex.Array,
        generation_config: GenerationConfig | None = None,
        prng_key: chex.Array | None = None,
        trace: bool = True,
        logits_processor: LogitsProcessorList | None = None,
        **kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head.

        Parameters:
            input_ids (`chex.Array` of shape `(batch_size, sequence_length)`):
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
            # broadcast input_ids & encoder_outputs
            input_ids = self._expand_to_num_beams(input_ids, num_beams=generation_config.num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"]["last_hidden_state"] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"],
                    num_beams=generation_config.num_beams,
                )

            for kwarg in ["attention_mask", "decoder_attention_mask"]:
                if kwarg in model_kwargs:
                    model_kwargs[kwarg] = self._expand_to_num_beams(
                        model_kwargs[kwarg], num_beams=generation_config.num_beams
                    )

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
        model_kwargs: dict[str, chex.Array] | None = None,
    ):
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
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
            model_outputs = model(state.running_token, **state.model_kwargs)
            logits = model_outputs.logits[:, -1]

            logits = logits_processor(state.sequences, logits, state.cur_len)

            next_token = jnp.argmax(logits, axis=-1)
            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_is_sent_finished = state.is_sent_finished | jnp.isin(
                next_token,
                eos_token_id,
            )
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
        prng_key: chex.Array | None = None,
        logits_processor: LogitsProcessorList | None = None,
        logits_warper: LogitsProcessorList | None = None,
        trace: bool = True,
        model_kwargs: dict[str, chex.Array] | None = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(
            eos_token_id,
            dtype=jnp.int32 if eos_token_id is not None else None,
        )
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
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
            model_outputs = model(state.running_token, **state.model_kwargs)
            logits = model_outputs.logits[:, -1]
            logits = logits_processor(state.sequences, logits, state.cur_len)
            logits = logits_warper(logits, logits, state.cur_len)
            next_token = (
                jax.random.categorical(prng_key, logits, axis=-1) * ~state.is_sent_finished
                + pad_token_id * state.is_sent_finished
            )
            next_is_sent_finished = state.is_sent_finished | jnp.isin(
                next_token,
                eos_token_id,
            )
            next_token = next_token[:, None]
            next_sequences = lax.dynamic_update_slice(
                state.sequences,
                next_token,
                (0, state.cur_len),
            )
            next_model_kwargs = self.update_inputs_for_generation(
                model_outputs,
                state.model_kwargs,
            )

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
        model_kwargs: dict[str, chex.Array] | None = None,
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

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # record the prompt length of decoder
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

        # initialize model specific kwargs
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
                best_running_score = state.running_scores[:, :1] / (
                    (state.cur_len - decoder_prompt_len) ** length_penalty
                )
            worst_finished_score = jnp.where(
                state.is_sent_finished,
                jnp.min(state.scores, axis=1, keepdims=True),
                np.array(-1.0e7),
            )
            improvement_still_possible = jnp.any(best_running_score > worst_finished_score)

            # 3. is there still a beam that has not finished?
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

            did_topk_just_finished = topk_sequences[:, :, state.cur_len] == eos_token_id
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

    def create_vinference(
        self,
        processor: ProcessingClassType,
        generation_config: vInferenceConfig,
        compile_config: vInferencePreCompileConfig | None = None,
        input_partition_spec: PartitionSpec | None = None,
        seed: int | None = None,
    ) -> vInference:
        from easydel import SamplingParams, vInference, vInferenceConfig

        if hasattr(self, "generation_config"):
            if self.generation_config is not None:
                sampling_params = generation_config.sampling_params
                generation_config = vInferenceConfig(
                    bos_token_id=generation_config.bos_token_id or self.generation_config.bos_token_id,
                    eos_token_id=generation_config.eos_token_id or self.generation_config.eos_token_id,
                    pad_token_id=generation_config.pad_token_id or self.generation_config.pad_token_id,
                    max_new_tokens=generation_config.max_new_tokens or self.generation_config.max_new_tokens,
                    streaming_chunks=generation_config.streaming_chunks or 64,
                    sampling_params=SamplingParams(
                        max_tokens=sampling_params.max_tokens or self.generation_config.max_new_tokens,
                        temperature=sampling_params.temperature or self.generation_config.temperature,
                        top_k=sampling_params.top_k or self.generation_config.top_k,
                        top_p=sampling_params.top_p or self.generation_config.top_p,
                    ),
                )
        num_params = sum(n.size for n in jax.tree_util.tree_flatten(self.graphstate)[0])
        size_in_billions = num_params / 1e9
        size_in_billions = f"{size_in_billions:.2f}b"
        vinference = vInference(
            model=None,
            processor_class=processor,
            generation_config=generation_config,
            graphdef=self.graphdef,
            mesh=self.mesh,
            partition_axis=self.config.partition_axis,
            inference_name=str(getattr(self._model_task, "value", self._model_task))
            + "-"
            + str(self._model_type)
            + ":"
            + size_in_billions,
            input_partition_spec=input_partition_spec,
            seed=seed,
            report_metrics=False,
        )
        if compile_config is not None:
            vinference.precompile(
                config=compile_config,
                graphother=self.graphother,
                graphstate=self.graphstate,
            )
        return vinference
