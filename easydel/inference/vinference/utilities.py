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

import dataclasses
import typing as tp

import jax
import jax.random
from eformer.escale import PartitionAxis
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import numpy as jnp
from jax import random
from jax import random as jrandom
from jax.sharding import PartitionSpec

from easydel.utils.compiling_utils import get_safe_hash_int

from ..logits_process import hash_fn
from ..sampling_params import JitableSamplingParams, SamplingParams

AmapType = tp.Mapping[str, dict[str, tp.Any]]
PropType = AmapType | list[AmapType] | None

logger = get_logger("vInference-Utils")


@auto_pytree
class vInferencePreCompileConfig:
    """
    Configuration class for pre-compiling vInference functions.

    This class holds parameters that define the shape and properties of inputs
    expected by the vInference engine during pre-compilation. It allows specifying
    different configurations, potentially in lists, to compile for multiple scenarios.

    Attributes:
        batch_size: Batch size or list of batch sizes for text generation.
        prefill_length: Prefill sequence length or list of lengths.
            If None, it might be inferred or not used depending on the context.
        vision_included: Whether vision inputs are included in the model.
        vision_batch_size: Batch size for vision inputs. Only relevant if `vision_included` is True.
        vision_channels: Number of channels for vision inputs. Only relevant if `vision_included` is True.
        vision_height: Height of vision inputs. Only relevant if `vision_included` is True.
        vision_width: Width of vision inputs. Only relevant if `vision_included` is True.
        required_props: Optional dictionary or list of dictionaries specifying
            required properties for advanced configuration (e.g., specific model arguments).
    """

    batch_size: int | list[int] = 1
    prefill_length: int | list[int] | None = None
    vision_included: bool | list[bool] = False
    vision_batch_size: int | list[int] | None = None
    vision_channels: int | list[int] | None = None
    vision_height: int | list[int] | None = None
    vision_width: int | list[int] | None = None
    required_props: PropType = None

    @staticmethod
    def _get_paddings(
        min_token_size: int,
        max_token_size: int,
        padding_gap: int,
    ) -> list[int]:
        paddings = []
        if min_token_size == max_token_size:
            return [min_token_size]
        num = min_token_size
        if padding_gap == 0:
            while num <= max_token_size:
                paddings.append(num)
                num *= 2
        else:
            while num <= padding_gap:
                paddings.append(num)
                num *= 2
            num //= 2
            while num < max_token_size:
                num += padding_gap
                paddings.append(num)
        return paddings

    @classmethod
    def create_optimized_compo(
        cls,
        batch_size: int | list[int] = 1,
        max_prefill_length: int = 2048,
        min_prefill_length: int = 64,
        vision_included: bool | list[bool] = False,
        vision_batch_size: int | list[int] | None = None,
        vision_channels: int | list[int] | None = None,
        vision_height: int | list[int] | None = None,
        vision_width: int | list[int] | None = None,
        required_props: PropType = None,
    ):
        prefill_length = cls._get_paddings(min_prefill_length, max_prefill_length, 0)
        logger.info(f"Prefill Lengths {prefill_length}")
        return cls(
            batch_size=batch_size,
            prefill_length=prefill_length,
            vision_included=vision_included,
            vision_batch_size=vision_batch_size,
            vision_channels=vision_channels,
            vision_height=vision_height,
            vision_width=vision_width,
            required_props=required_props,
        )

    def _im_standalone(self) -> bool:
        """
        Checks if the configuration represents a single, standalone compilation scenario.

        This method iterates through the configuration fields. If any field's value
        is a list, it indicates multiple scenarios, and the method returns False.
        Otherwise, it represents a single scenario, and it returns True.

        Returns:
            True if the configuration defines a single scenario, False otherwise.
        """
        standalone = True
        for rfield in dataclasses.fields(self):
            attr = getattr(self, rfield.name)
            if isinstance(attr, list):
                standalone = False
        return standalone

    _is_standalone = _im_standalone

    def extract(self) -> dict:
        """
        Converts the configuration instance into a dictionary.

        This method is useful for serialization or easily accessing all configuration
        values.

        Returns:
            A dictionary representation of the `vInferencePreCompileConfig` instance.
        """
        return dataclasses.asdict(self)

    def get_default_hash(self) -> int:
        """
        Generates a unique integer hash representing the configuration.

        This hash is calculated based on the string representation of all configuration
        attributes, ensuring that identical configurations produce the same hash. This
        is crucial for caching compiled functions based on their configuration.

        Returns:
            An integer hash value representing the configuration.
        """
        hash_str = ""
        hash_str += str(self.batch_size) + "-"
        hash_str += str(self.prefill_length) + "-"
        hash_str += str(self.vision_included) + "-"
        hash_str += str(self.vision_batch_size) + "-"
        hash_str += str(self.vision_channels) + "-"
        hash_str += str(self.vision_height) + "-"
        hash_str += str(self.vision_width) + "-"
        hash_str += str(self.required_props)
        hash_out = get_safe_hash_int(hash_str)
        return hash_out

    __hash__ = get_default_hash

    def get_standalones(self) -> list["vInferencePreCompileConfig"]:
        """
        Generates a list of standalone configurations from a potentially multi-value config.

        If any attribute in the current configuration is a list (indicating multiple
        scenarios), this method expands the configuration into multiple individual
        `vInferencePreCompileConfig` instances. Each resulting instance represents
        a single, specific compilation scenario.

        If an attribute's list is shorter than the longest list among all attributes,
        its last element is repeated to ensure all generated configurations have
        values for all attributes.

        If the original configuration is already standalone (no list attributes),
        this method returns a list containing only the original instance.

        Returns:
            A list of `vInferencePreCompileConfig` instances, each representing a
            single, standalone compilation scenario.
        """
        if self._is_standalone():
            return [self]

        list_fields = {}
        max_length = 0

        for rfield in dataclasses.fields(self):
            attr = getattr(self, rfield.name)
            if isinstance(attr, list):
                list_fields[rfield.name] = attr
                max_length = max(max_length, len(attr))

        # Create standalone configs
        standalone_configs = []

        for i in range(max_length):
            config_kwargs = {}

            for rfield in dataclasses.fields(self):
                attr = getattr(self, rfield.name)
                field_name = rfield.name

                if field_name in list_fields:
                    list_attr = list_fields[field_name]
                    # Use value at index i if available, otherwise use the last value
                    if i < len(list_attr):
                        config_kwargs[field_name] = list_attr[i]
                    else:
                        config_kwargs[field_name] = list_attr[-1]
                else:
                    # For non-list fields, use the original value
                    config_kwargs[field_name] = attr

            standalone_configs.append(vInferencePreCompileConfig(**config_kwargs))

        return standalone_configs


vInferencePreCompileConfig.__hash__ = vInferencePreCompileConfig.get_default_hash


@auto_pytree
class SampleState:
    """
    Represents the state of the sampling process during token generation within the vInference engine.

    This class encapsulates all necessary information to pause and resume the
    generation loop. It tracks the progress of generation, including the tokens
    generated so far, the current position, completion status, random number
    generator state, and any model-specific state (like attention caches).

    Attributes:
        current_length: The current length of the generated sequences (number of tokens generated so far).
        sequences: The tensor holding the generated token IDs for each sequence in the batch.
            Shape: `(batch_size, max_sequence_length)`.
        running_token: The most recently generated token for each sequence. Used as input for the next step.
            Shape: `(batch_size, 1)`.
        is_sequence_finished: A boolean tensor indicating whether each sequence in the batch
            has reached an end-of-sequence (EOS) token or the maximum generation length.
            Shape: `(batch_size,)`.
        prng_key: The JAX pseudo-random number generator key used for stochastic sampling.
        model_kwargs: A dictionary containing any additional arguments required by the
            model for the next generation step (e.g., attention cache/past key-values).
            The structure depends on the specific model implementation.
        tokens_per_second: Estimated generation speed in tokens per second. Defaults to -inf.
        generated_tokens: The total count of tokens generated across all sequences in the
            current generation process up to this state. Defaults to 0.
        padded_length: The target length to which sequences are padded. This might be
            different from `max_sequence_length` in some scenarios. Defaults to 0.
        _time_spent_computing: Internal tracker for the cumulative computation time spent
            to reach this state. Defaults to 0.0.
        _compile_config: The `vInferencePreCompileConfig` instance used for pre-compiling
            the functions associated with this generation state. Defaults to None.
    """

    current_length: jax.Array
    sequences: jax.Array
    running_token: jax.Array
    is_sequence_finished: jax.Array
    prng_key: random.PRNGKey
    model_kwargs: dict[str, jax.Array]

    # vInference Ops
    tokens_per_second: float | None = float("-inf")
    generated_tokens: int | None = 0
    padded_length: int | None = 0

    _time_spent_computing: float | None = 0.0
    _compile_config: vInferencePreCompileConfig | None = None


def create_sampling_step(
    sampling_params: JitableSamplingParams,
    eos_token_id: jax.Array,
    pad_token_id: jax.Array,
) -> tp.Callable:
    """
    Creates a callable function that performs a single step of token generation (sampling).

    This factory function returns a `sampling_step` function tailored with the provided
    logit processors/warpers and token IDs. The returned function is designed to be
    used within a generation loop (e.g., `jax.lax.scan`).

    Args:
        sampling_params: A `JitableSamplingParams`.
        eos_token_id: A JAX array containing the token ID(s) representing the
            end-of-sequence. Generation stops for a sequence once an EOS token is sampled.
        pad_token_id: The JAX array representing the padding token ID. Once a sequence
            is finished (EOS sampled), subsequent steps will generate this token.

    Returns:
        A callable function `sampling_step(graphdef, graphstate, graphother, state)`
        which takes the model's NNX graph components (`graphdef`, `graphstate`, `graphother`)
        and the current `SampleState`, performs one generation step, and returns the
        updated `SampleState`.
    """

    def sampling_step(graphdef, graphstate, graphother, state: SampleState):
        """
        Performs a single sampling step using the provided model components and state.

        Args:
            graphdef: The definition part of the NNX graph model.
            graphstate: The state part of the NNX graph model.
            graphother: Other components of the NNX graph model.
            state (SampleState): The current generation state.

        Returns:
            SampleState: The updated generation state after one sampling step.
        """
        model = nn.merge(graphdef, graphstate, graphother)
        model_outputs = model(input_ids=state.running_token, **state.model_kwargs)

        logits = model_outputs.logits[:, -1]

        def _random(sequ, logits, length, key):
            logits = sampling_params.logits_processor(sequ, logits, length)
            logits = sampling_params.logits_warper(sequ, logits, length)
            return jax.random.categorical(key, logits, axis=-1)

        def _gready(sequ, logits, length, key):
            return jnp.argmax(logits, axis=-1)

        next_token = jax.lax.cond(
            sampling_params.random_sampling,
            _random,
            _gready,
            state.sequences,
            logits,
            state.current_length,
            state.prng_key,
        )

        next_token = next_token * ~state.is_sequence_finished + pad_token_id * state.is_sequence_finished

        next_sequence_finished = jnp.logical_or(
            state.generated_tokens >= (sampling_params.max_tokens * state.sequences.shape[0]),
            state.is_sequence_finished | jnp.isin(next_token, eos_token_id),
        )

        next_token = next_token[:, None]
        next_sequences = jax.lax.dynamic_update_slice(state.sequences, next_token, (0, state.current_length))
        next_model_kwargs = model.update_inputs_for_generation(model_outputs, state.model_kwargs)

        return state.replace(
            current_length=state.current_length + 1,
            sequences=next_sequences,
            running_token=next_token,
            is_sequence_finished=next_sequence_finished,
            prng_key=jax.random.split(state.prng_key, 2)[0],  # Update PRNG key
            model_kwargs=next_model_kwargs,
            generated_tokens=state.generated_tokens + state.sequences.shape[0],
        )

    return sampling_step


@auto_pytree
class vInferenceConfig:
    """
    Configuration class for the vInference engine, controlling the overall generation process.

    This class holds parameters that define how the generation loop behaves,
    including length constraints, token control, sharding strategies, and sampling settings.

    Attributes:
        max_new_tokens: The maximum number of new tokens to generate, excluding the
            initial prompt tokens. Defaults to 64.
        streaming_chunks: The number of generation steps to compile and execute together
            as a single unit. Larger chunks can improve performance on TPUs by reducing
            compilation overhead and kernel launch times, but may increase memory usage.
            Defaults to 16.
        num_return_sequences: The number of sequences to generate and return. Can be:
            - An integer: Generate this many sequences for all inputs.
            - A dictionary mapping precompile hash (from `vInferencePreCompileConfig`)
              to an integer: Generate a specific number of sequences based on the
              compilation configuration. Defaults to 1.
        pad_token_id: The token ID used for padding sequences. If None, the model's
            default pad token ID might be used, or padding might not be applied.
        bos_token_id: The token ID representing the beginning-of-sequence. May be
            used implicitly by the model or generation logic.
        eos_token_id: The token ID(s) representing the end-of-sequence. Generation
            stops for a sequence when one of these tokens is sampled. Can be a single
            integer or a list/tuple of integers.
        partition_rules: A tuple of custom sharding rules (regex pattern, PartitionSpec)
            to apply to the model's parameters and intermediate states (like attention cache).
            If None, default rules based on `partition_axis` are generated.
            Example: `((".*kernel.*", PartitionSpec("fsdp", None)), ...)`
        partition_axis: A `PartitionAxis` object defining the logical names for sharding
            axes (e.g., 'batch', 'sequence', 'head'). Required if `partition_rules` is None,
            used to generate default sharding rules.
        _loop_rows: (Internal) The calculated number of iterations needed in the
            generation loop based on `max_new_tokens` and `streaming_chunks`.
            Automatically computed in `__post_init__`.
        sampling_params: A `JitableSamplingParams` object containing parameters for the
            sampling process itself (e.g., temperature, top_k, top_p, repetition penalty).
            If None, a default `JitableSamplingParams` instance with `max_tokens` set to
            `max_new_tokens` is created in `__post_init__`.
    """

    max_new_tokens: int = 64
    streaming_chunks: int = 16

    num_return_sequences: int | dict[int, int] | None = 1

    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    partition_rules: tuple[tuple[str, tp.Any]] | None = None
    partition_axis: PartitionAxis | None = None
    _loop_rows: int | None = None

    sampling_params: JitableSamplingParams | SamplingParams | None = None

    def get_partition_rules(
        self,
        # in case that someone needs to customize this
        runtime_config: vInferencePreCompileConfig | None = None,
    ) -> tuple[tuple[str, tp.Any], ...]:
        """
        Generates or retrieves the sharding partition rules for the vInference engine.

        If `self.partition_rules` is already set (custom rules provided), it returns
        them directly.

        Otherwise, it constructs a default set of partition rules based on the axis names
        defined in `self.partition_axis`. These default rules aim to provide sensible
        sharding for common model components:
        - Input sequences (`sequences`, `running_token`) are sharded along batch and sequence axes.
        - Attention masks and position IDs are sharded similarly.
        - Past key-value states (attention cache), including common quantized formats
          (8-bit, NF4), are sharded across batch, key sequence, head, and attention dimension axes.
        - Any parameters/states not matching the specific rules are replicated by default (`.*`).

        Args:
            runtime_config: An optional `vInferencePreCompileConfig`. Currently unused
                in the default rule generation but available for potential customization
                in subclasses or future versions.

        Returns:
            A tuple of partition rules. Each rule is a tuple containing:
                - A regex pattern (string) matching parameter or state names.
                - A `jax.sharding.PartitionSpec` defining how the matched items should be sharded.

        Raises:
            AssertionError: If `self.partition_rules` is None and `self.partition_axis`
                is also None, as axis names are required to generate default rules.
        """
        if self.partition_rules is not None:
            return self.partition_rules
        assert (
            self.partition_axis is not None
        ), "partition axis is required for state sharding if partition_rules is not provided"
        paxis = self.partition_axis
        kvps = PartitionSpec(
            paxis.batch_axis,
            paxis.key_sequence_axis,
            paxis.head_axis,
            paxis.attention_dim_axis,
        )
        idps = PartitionSpec(paxis.batch_axis, paxis.sequence_axis)
        return (
            ("(sequences|running_token)", idps),
            ("model_kwargs/(attention_mask|position_ids)", idps),
            ("model_kwargs/past_key_values/views/[0-9]+/(key|value)/(scale|weight)", kvps),
            ("model_kwargs/past_key_values/views/[0-9]+/(key|value)/(packed|absmax)", kvps),
            ("model_kwargs/past_key_values/views/[0-9]+/(key|value)", kvps),
            (".*", PartitionSpec()),
        )

    def __post_init__(self):
        """
        Performs initialization tasks after the dataclass is created.

        Specifically, it:
        1. Calculates `_loop_rows`: Determines the number of iterations required for the
           generation loop based on `max_new_tokens` and `streaming_chunks`.
        2. Initializes `sampling_params`: If `sampling_params` was not provided during
           instantiation, it creates a default `SamplingParams` instance, setting its
           `max_tokens` attribute to the value of `self.max_new_tokens`.
        """

        if isinstance(self.max_new_tokens, int):
            self._loop_rows = (self.max_new_tokens + self.streaming_chunks - 1) // self.streaming_chunks
        if self.sampling_params is None:
            self.sampling_params = SamplingParams(max_tokens=self.max_new_tokens)

    __hash__ = hash_fn


class GenerateRNG:
    """An infinite generator of JAX PRNGKeys, useful for iterating over seeds."""

    def __init__(self, seed: int = 0):
        """Initializes the generator with a starting seed.

        Args:
            seed: The seed to use for the initial PRNGKey.
        """
        self.seed = seed
        self._rng = jrandom.PRNGKey(seed)

    def __next__(self) -> jrandom.PRNGKey:
        """Generates and returns the next PRNGKey in the sequence.

        Returns:
            The next PRNGKey derived from the internal state.
        """
        self._rng, key = jrandom.split(self._rng)
        return key

    @property
    def rng(self) -> jrandom.PRNGKey:
        """Provides access to the next PRNGKey without advancing the generator.

        Returns:
            The next PRNGKey in the sequence.
        """
        return next(self)
