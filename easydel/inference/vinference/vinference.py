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


from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os
import pathlib
import pickle
import random
import time
import typing as tp
import warnings
from datetime import datetime
from functools import cached_property
from uuid import uuid4

import eformer
import eformer.common_types
import eformer.escale
import eformer.escale as es
import jax
import numpy as np
from chex import PRNGKey
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from jax._src.stages import Compiled
from jax.sharding import NamedSharding, PartitionSpec
from pydantic import BaseModel, Field
from transformers import ProcessorMixin

from easydel.utils.compiling_utils import load_compiled_fn, save_compiled_fn, smart_compile
from easydel.utils.helpers import capture_time, check_bool_flag, get_logger
from easydel.utils.lazy_import import is_package_available

from ..sampling_params import JitableSamplingParams, SamplingParams
from .functions import expand_inputs_for_generation, get_compiled_funcs, interval_func, prefill_func, put_compiled_funcs
from .utilities import GenerateRNG, SampleState, vInferenceConfig, vInferencePreCompileConfig

if tp.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule
    from easydel.infra.utils import ProcessingClassType

logger = get_logger("vInference")
TIME = str(datetime.fromtimestamp(time.time())).split(" ")[0]

# Sentinels for async generator communication
_ITEM_SENTINEL = object()
_RETURN_SENTINEL = object()
_EXCEPTION_SENTINEL = object()


class PromptOutput(BaseModel):
    """Output structure for generated text.

    Contains the generated text and associated metadata.

    Attributes:
        text: Generated text string.
        generated_tokens: Number of tokens generated.
        tokens_per_second: Generation speed.
        error: Error message if generation failed.
        finish_reason: Reason generation stopped ('stop', 'length', etc.).
    """

    text: str | None = None
    generated_tokens: int | None = None
    tokens_per_second: float | None = None
    error: str | None = None
    finish_reason: str | None = Field(
        default=None,
        description="Reason generation finished (e.g., 'stop', 'length')",
    )
    model_config = dict(arbitrary_types_allowed=True)


class vInferenceMetaData(BaseModel):
    """Metadata for vInference instance.

    Stores configuration and state information.

    Attributes:
        inference_name: Name identifier for this instance.
        generation_config: Generation configuration settings.
        precompiled_configs: Cached pre-compilation configurations.
        in_compiling_process: Set of configs currently being compiled.
        input_partition_spec: JAX partitioning specification.
        uuid4: Unique identifier for this instance.
    """

    inference_name: str
    generation_config: vInferenceConfig
    precompiled_configs: dict[int, vInferencePreCompileConfig]
    in_compiling_process: set
    input_partition_spec: jax.sharding.PartitionSpec
    uuid4: str
    model_config = dict(arbitrary_types_allowed=True)


class vInference:
    """Streamlined inference engine for text generation.

    vInference provides a simple yet powerful interface for generating text
    with pre-trained language models. It handles model initialization,
    pre-compilation for optimal performance, and both streaming and
    non-streaming generation.

    Features:
        - Automatic model compilation and optimization
        - Support for various sampling strategies
        - Streaming and non-streaming generation
        - Mixed-precision inference
        - Distributed inference with JAX sharding
        - Persistent compilation caching

    Attributes:
        model: The underlying EasyDeL model
        processor_class: Tokenizer/processor for text handling
        generation_config: Configuration for text generation
        inference_name: Unique name for this instance
        graphdef: Model graph definition
        graphstate: Model state

    Example:
        >>> from easydel.inference.vinference import vInference
        >>> engine = vInference(
        ...     model=my_model,
        ...     processor_class=tokenizer,
        ...     max_new_tokens=100
        ... )
        >>> # Pre-compile for a specific batch size
        >>> engine.precompile(batch_size=1, sequence_length=512)
        >>> # Generate text
        >>> output = engine.generate(
        ...     "Write a poem about",
        ...     temperature=0.8
        ... )
        >>> print(output.text)
    """

    def __init__(
        self,
        model: EasyDeLBaseModule | None = None,
        processor_class: ProcessingClassType = None,
        graphdef: nn.GraphDef | None = None,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
        mesh: eformer.common_types.Mesh | None = None,
        generation_config: vInferenceConfig | None = None,
        seed: int | None = None,
        input_partition_spec: PartitionSpec | None = None,
        max_new_tokens: int = 512,
        inference_name: str | None = None,
        partition_axis: eformer.escale.PartitionAxis | None = None,
        report_metrics: bool = True,
        verbose: bool = True,
    ):
        """Initialize vInference engine.

        Args:
            model: Pre-trained EasyDeL model (extracts graphdef/state if provided).
            processor_class: Tokenizer/processor for text handling.
            graphdef: Model graph definition (required if model is None).
            graphstate: Model state (optional for lazy initialization).
            graphother: Additional graph state (optional).
            mesh: JAX mesh for distributed computation.
            generation_config: Generation settings (temperature, top_p, etc.).
            seed: Random seed for reproducible generation.
            input_partition_spec: Sharding specification for inputs.
            max_new_tokens: Default maximum tokens to generate.
            inference_name: Unique identifier for this instance.
            partition_axis: Axis configuration for model parallelism.
            report_metrics: Whether to report performance metrics.
            verbose: Enable verbose logging.

        Note:
            Either `model` or `graphdef` must be provided. If using graphdef
            without graphstate, lazy initialization will be used.
        """
        self.log = logger.info if verbose else logger.debug
        self._init_model_components(model, graphdef, graphstate, graphother, mesh, inference_name, partition_axis)
        self.processor_class = processor_class
        self.generation_config = self._init_generation_config(generation_config, max_new_tokens)
        self._setup_partitioning(partition_axis, input_partition_spec)
        self._setup_rng(seed)
        self._init_internal_state()
        self._setup_metrics_reporting(report_metrics)

    def _init_model_components(
        self,
        model: EasyDeLBaseModule | None,
        graphdef: nn.GraphDef | None,
        graphstate: nn.GraphState | None,
        graphother: nn.GraphState | None,
        mesh: eformer.common_types.Mesh | None,
        inference_name: str | None,
        partition_axis: eformer.escale.PartitionAxis | None,
    ) -> None:
        """Initialize model components from either a full model or individual components."""
        if model is not None:
            self.graphdef, self.graphstate, self.graphother = nn.split(model, nn.Param, ...)
            self.mesh = model.config.mesh
            self._inference_name = inference_name or self._generate_inference_name(model)
            self._partition_axis = partition_axis or model.config.partition_axis
        else:
            self.graphdef = graphdef
            self.graphstate = graphstate
            self.graphother = graphother
            self.mesh = mesh
            self._inference_name = inference_name
            self._partition_axis = partition_axis

        if self.graphdef is None:
            raise ValueError("graphdef is required. Either provide a complete model or specify graphdef explicitly.")
        if self._inference_name is None:
            raise ValueError("inference_name is required for lazy initialization.")
        if self.mesh is None:
            raise ValueError("mesh is required for distributed computation.")

        if self._partition_axis is None:
            self._partition_axis = eformer.escale.PartitionAxis()

    def _setup_partitioning(
        self,
        partition_axis: eformer.escale.PartitionAxis | None = None,
        input_partition_spec: PartitionSpec | None = None,
    ) -> None:
        """Configure partitioning specifications."""
        if self.generation_config.partition_axis is None:
            _partition_axis = partition_axis if partition_axis is not None else self._partition_axis
            self.generation_config.partition_axis = _partition_axis
        self.input_partition_spec = input_partition_spec or PartitionSpec(("dp", "fsdp"), "sp")

    def _setup_rng(self, seed: int | None) -> None:
        """Initialize random number generator with given or random seed."""
        if seed is None:
            seed = random.randint(0, int(1e6))

        self._rng_generator = GenerateRNG(seed)

    def _init_internal_state(self) -> None:
        """Initialize internal state variables."""
        self._precompile_lock = asyncio.Lock()
        self._precompiled_configs: dict[int, vInferencePreCompileConfig] = {}
        self._in_compiling_process: set = set()
        self._uuid4 = uuid4().hex
        self._init_variables()
        self._validate_token_ids()

    def _setup_metrics_reporting(self, report_metrics: bool) -> None:
        """Configure metrics reporting based on environment and available packages."""
        metrics_enabled = check_bool_flag("EASYDEL_RECORDS_METRICS")
        is_main_process = jax.process_count() == 1
        has_prometheus = is_package_available("prometheus_client")
        has_psutil = is_package_available("psutil")

        self._report_metrics = metrics_enabled and is_main_process and has_prometheus and report_metrics and has_psutil

        if not self._report_metrics:
            if has_prometheus:
                self.log("vInference metrics reporting is disabled")
            else:
                self.log("prometheus_client or psutil not found - vInference metrics will be disabled")

    def get_model(self, graphstate: nn.GraphState | None = None, graphother: nn.GraphState | None = None):
        if graphstate is None:
            graphstate = self.graphstate
        if graphother is None:
            graphother = self.graphother

        assert graphstate is not None
        assert graphother is not None
        return nn.merge(self.graphdef, graphstate, graphother)

    @property
    def model(self) -> nn.Module:
        """Get the complete model by merging graph components.

        Returns:
            Complete model instance.

        Raises:
            ValueError: If using lazy initialization without graphstate/graphother.
        """
        if self.graphstate is None or self.graphother is None:
            raise ValueError(
                "Model property cannot be accessed when using lazy initialization. "
                "Provide graphstate and graphother or use a complete model."
            )
        return nn.merge(self.graphdef, self.graphstate, self.graphother)

    @cached_property
    def metrics(self):
        if self._report_metrics:
            from .metrics import vInferenceMetrics

            if is_package_available("prometheus_client"):
                return vInferenceMetrics(self._inference_name)
            else:
                self._report_metrics = False
                self.log("`prometheus_client` not found!, metrics logging in vinference will be disabled")
        return None

    def _metrics_increase_queue(self):
        if self._report_metrics:
            self.metrics.queue_size.labels(model_name=self.metrics.model_name).inc()

    def _metrics_decrease_queue(self):
        if self._report_metrics:
            self.metrics.queue_size.labels(model_name=self.metrics.model_name).dec()

    def _inference_latency_context_manager(self, stage):
        if self._report_metrics:
            return self.metrics.inference_latency.labels(
                model_name=self.metrics.model_name,
                stage=stage,
            ).time()
        return contextlib.nullcontext()

    def _post_generation_metrics_update(self, state):
        if self._report_metrics:
            self.metrics.token_throughput.labels(
                model_name=self.metrics.model_name,
                operation="output",
            ).inc(state.generated_tokens)
            self.metrics.generation_length.labels(
                model_name=self.metrics.model_name,
            ).observe(state.generated_tokens)
            self.metrics.inference_requests.labels(
                model_name=self.metrics.model_name,
                status="success",
            ).inc()

    def _submit_during_generation_metrics_update(self):
        if self._report_metrics:
            self.metrics.inference_requests.labels(model_name=self.metrics.model_name, status="error").inc()

    def _compilation_metrics_recorder(self):
        if self._report_metrics:
            return self.metrics.compilation_time.labels(
                model_name=self.metrics.model_name,
                function_name="_compile_and_lower_funs",
            ).time()
        return contextlib.nullcontext()

    @cached_property
    def tokenizer(self):
        from transformers import PreTrainedTokenizerBase

        if isinstance(self.processor_class, PreTrainedTokenizerBase):
            return self.processor_class
        from transformers import ProcessorMixin

        if isinstance(self.processor_class, ProcessorMixin):
            return self.processor_class.tokenizer
        raise ValueError("Unknown `processor_class` to extract `tokenizer` from.")

    def _generate_inference_name(self, model) -> str:
        """
        Generate a standardized inference name combining model type, size, and timestamp.

        Format: {model_type}-{size_in_B}B-{timestamp}
        Example: llama-7.00B-20240311
        """
        model_type = self._get_model_type(model)
        model_size = self._calculate_model_size(self.graphstate)
        timestamp = datetime.now().strftime("%Y%m%d")

        return f"{model_type}-{model_size}B-{timestamp}"

    def _get_model_type(self, model) -> str:
        """Get the model type, with fallback to 'unknown' if not found."""
        return getattr(model.config, "model_type", "unknown").lower()

    def _calculate_model_size(self, graphstate) -> str:
        """
        Calculate model size in billions of parameters.
        Returns formatted string with 2 decimal places.
        """
        try:
            num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
            size_in_billions = num_params / 1e9
            return f"{size_in_billions:.2f}"
        except Exception as e:
            logger.warning(f"Failed to calculate model size: {e}")
            return "unknown"

    @property
    def inference_name(self):
        return self._inference_name

    @property
    def model_prefill_length(self) -> int:
        """
        Calculate the maximum length available for input prefill by subtracting
        the maximum new tokens from the model's maximum sequence length.

        Returns:
            int: The maximum length available for input prefill

        Raises:
            ValueError: If no maximum sequence length configuration is found
        """
        possible_length_attributes = [
            "granted_freq_max_position_embedding",
            "granted_mask_max_position_embedding",
            "max_position_embedding",
            "max_sequence_length",
        ]

        max_length = self._get_model_max_length(possible_length_attributes)

        if max_length is None:
            raise ValueError(
                "Could not determine model's maximum sequence length. "
                f"Looked for attributes: {', '.join(possible_length_attributes)}"
            )

        return max_length - self.generation_config.max_new_tokens

    def _get_model_max_length(self, attributes: list[str]) -> int | None:
        """
        Find the first available maximum length configuration from a list of possible attributes.

        Args:
            attributes: tp.List of attribute names to check in order of preference

        Returns:
            tp.Optional[int]: The maximum length if found, None otherwise
        """
        try:
            model = self.model
        except ValueError:
            return None
        for attr in attributes:
            max_length = getattr(model.config, attr, None)
            if max_length is not None:
                return max_length
        return None

    def _validate_token_ids(self):
        """
        Validates the token IDs for padding, end-of-sequence, and beginning-of-sequence.
        """
        try:
            model = self.model
        except ValueError:
            model = None
        if hasattr(model, "generation_config"):
            if self.generation_config.pad_token_id is None:
                self.generation_config.pad_token_id = self.model.generation_config.pad_token_id
            if self.generation_config.eos_token_id is None:
                self.generation_config.eos_token_id = self.model.generation_config.eos_token_id
            if self.generation_config.bos_token_id is None:
                self.generation_config.bos_token_id = self.model.generation_config.bos_token_id

        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if self.generation_config.eos_token_id is None:
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        if self.generation_config.bos_token_id is None:
            self.generation_config.bos_token_id = self.tokenizer.bos_token_id

        assert self.generation_config.pad_token_id is not None, (
            "`pad_token_id` cannot be None. "
            "(Set `tokenizer.pad_token_id = tokenizer.eos_token_id` if undefined"
            " or (`processing_class.tokenizer.pad_token_id = processing_class.tokenizer.eos_token_id`))"
        )
        assert self.generation_config.eos_token_id is not None, "`eos_token_id` cannot be None."

    @property
    def SEQUENCE_DIM_MAPPING(self):
        return {
            "input_ids": 1,
            "attention_mask": 1,
            "position_ids": 1,
            "past_key_values": 2,
            "token_type_ids": 1,
            "inputs_embeds": 1,
        }

    @SEQUENCE_DIM_MAPPING.setter
    def SEQUENCE_DIM_MAPPING(self, val):
        return val

    def _init_generation_config(
        self,
        generation_config: vInferenceConfig | None,
        max_new_tokens: int,
    ) -> vInferenceConfig:
        """
        Initializes the generation configuration.

        Args:
          generation_config: The generation configuration.
          max_new_tokens: The maximum number of new tokens to generate.

        Returns:
          vInferenceConfig: The initialized generation configuration.
        """
        if generation_config is None:
            top_k = self.model.generation_config.top_k
            top_p = self.model.generation_config.top_p
            temperature = self.model.generation_config.temperature
            max_new_tokens = self.model.generation_config.max_new_tokens or max_new_tokens
            if self.model.generation_config is not None:
                generation_config = vInferenceConfig(
                    bos_token_id=self.model.generation_config.bos_token_id,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.model.generation_config.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    sampling_params=SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    ).make_jitable(),
                )
            else:
                generation_config = vInferenceConfig(max_new_tokens=max_new_tokens)
        return generation_config

    def _init_variables(self):
        """
        Initializes the shardings for input data.
        """
        mesh = self.mesh
        fsdp = self.input_partition_spec[0]

        self.input_sharding = NamedSharding(spec=self.input_partition_spec, mesh=mesh)
        self.empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)
        self.generation_input_shape = NamedSharding(spec=PartitionSpec(fsdp, None), mesh=mesh)

    def _get_init_state(
        self,
        graphstate: nn.GraphState,
        graphother: nn.GraphState,
        vinference_compile_config: vInferencePreCompileConfig,
        model_kwargs: tp.Mapping[str | int, jax.Array],
    ) -> SampleState:
        func = get_compiled_funcs(standalone_config=vinference_compile_config, id="_init_state", safe=False)

        if func is None:
            self.log(f"registering new signature({vinference_compile_config.get_default_hash()}) for `_init_state`")
            eshape = jax.eval_shape(
                self._init_state_pure,
                graphstate=graphstate,
                graphother=graphother,
                vinference_compile_config=vinference_compile_config,
                **model_kwargs,
            )
            prules = self.generation_config.get_partition_rules(vinference_compile_config)
            out_shardings = jax.tree_util.tree_map(
                lambda spec: NamedSharding(mesh=self.mesh, spec=spec),
                es.match_partition_rules(prules, eshape),
            )
            lowered = jax.jit(self._init_state_pure, out_shardings=out_shardings).lower(
                graphstate=graphstate,
                graphother=graphother,
                vinference_compile_config=vinference_compile_config,
                **model_kwargs,
            )
            func, _ = smart_compile(lowered, tag="vinference-init-state")
            put_compiled_funcs(
                funcs=func,
                standalone_config=vinference_compile_config,
                id="_init_state",
            )
        return func(
            graphstate=graphstate,
            graphother=graphother,
            vinference_compile_config=vinference_compile_config,
            **model_kwargs,
        )

    def _init_state_pure(
        self,
        graphstate: nn.GraphState,
        graphother: nn.GraphState,
        input_ids: jax.Array = None,
        rng: PRNGKey = None,
        vinference_compile_config: vInferencePreCompileConfig | None = None,
        **model_kwargs,
    ):
        model = self.get_model(graphstate=graphstate, graphother=graphother)
        num_return_sequences = self.generation_config.num_return_sequences
        if num_return_sequences is None:
            num_return_sequences = 1
        elif isinstance(num_return_sequences, dict):
            num_return_sequences = num_return_sequences.get(input_ids.shape[1], 1)

        assert isinstance(num_return_sequences, int), "`num_return_sequences` should be int or dict mapping int to int."
        input_ids, model_kwargs = expand_inputs_for_generation(
            num_return_sequences,
            False,
            input_ids=input_ids,
            **model_kwargs,
        )
        pad_token_id = jnp.array(self.generation_config.pad_token_id, dtype=jnp.int32)
        batch_size, current_length = input_ids.shape
        max_length = current_length + self.generation_config.max_new_tokens
        current_length = jnp.array(current_length)
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))
        is_sequence_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        model_kwargs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=None,
            **model_kwargs,
        )

        return SampleState(
            current_length=current_length,
            sequences=sequences,
            running_token=input_ids,
            is_sequence_finished=is_sequence_finished,
            prng_key=rng,
            model_kwargs=model_kwargs,
            generated_tokens=0,
            _compile_config=vinference_compile_config,
        )

    def _find_optimal_config(self, batch_size: int, sequence_length: int) -> tuple[int, int]:
        """
        Finds the optimal precompiled configuration for given input dimensions.

        Args:
            batch_size: The batch size of input
            sequence_length: The sequence length of input

        Returns:
            tuple[int, int]: The optimal (batch_size, sequence_length) configuration
        """
        if not self._precompiled_configs:
            logger.warning(
                f"vInference [{self.inference_name}] doesn't contain any precompiled "
                "config please precompile instance for best performance",
                stacklevel=1,
            )
            return (batch_size, sequence_length)

        # Group configs by batch size
        batch_configs = {}
        for confs in self._precompiled_configs.values():
            if confs.batch_size not in batch_configs:
                batch_configs[confs.batch_size] = []
            batch_configs[confs.batch_size].append(confs.prefill_length)

        # Find best batch size
        available_batches = sorted(batch_configs.keys())
        best_batch = None
        for b in available_batches:
            if b >= batch_size:
                best_batch = b
                break

        if best_batch is None:
            best_batch = max(available_batches)

        # Find best sequence length
        available_lengths = sorted(batch_configs[best_batch])
        max_length = max(available_lengths)

        # If sequence length exceeds maximum, use maximum
        if sequence_length > max_length:
            best_length = max_length
            logger.warning(
                f"Input sequence length {sequence_length} exceeds maximum available length "
                f"{max_length}. Input will be truncated."
            )
        else:
            # Find smallest config that fits
            best_length = None
            for length in available_lengths:
                if length >= sequence_length:
                    best_length = length
                    break

            if best_length is None:
                best_length = max_length

        return (best_batch, best_length)

    def _create_vinference_config_from_kwargs(
        self,
        graphstate: nn.GraphState,
        graphother: nn.GraphState,
        kwargs: dict,
        batch_size: int | None = None,
        prefill_length: int | None = None,
    ) -> vInferencePreCompileConfig:
        if batch_size is None or prefill_length is None:
            _input_ids = getattr(kwargs, "input_ids", None)
            assert (
                _input_ids is not None
            ), "if `batch_size` or `prefill_length` is None `input_ids` must be present in your model kwargs."
            batch_size, prefill_length = _input_ids.shape
        vision_included = False
        vision_batch_size = None
        vision_channels = None
        vision_height = None
        vision_width = None
        if "pixel_values" in kwargs.keys():
            vision_included = True
            if kwargs["pixel_values"].ndim == 4:
                vision_batch_size, vision_channels, vision_height, vision_width = kwargs["pixel_values"].shape
            elif kwargs["pixel_values"].ndim == 3:
                vision_batch_size = 1
                vision_channels, vision_height, vision_width = kwargs["pixel_values"].shape
            elif kwargs["pixel_values"].ndim == 2:
                vision_batch_size = 1
                vision_channels = 1
                vision_height, vision_width = kwargs["pixel_values"].shape
        model = self.get_model(graphstate=graphstate, graphother=graphother)
        required_props = model._create_required_props_from_kwargs(model_kwargs=kwargs)
        vinf_config = vInferencePreCompileConfig(
            batch_size=batch_size,
            prefill_length=prefill_length,
            vision_included=vision_included,
            vision_batch_size=vision_batch_size,
            vision_channels=vision_channels,
            vision_height=vision_height,
            vision_width=vision_width,
            required_props=required_props,
        )

        return vinf_config

    def _adjust_inputs_to_config(
        self,
        model_kwargs: dict,
        target_batch: int,
        target_length: int,
    ) -> dict:
        """
        Adjusts all model inputs to match target configuration dimensions through truncation or padding.

        Args:
            model_kwargs: Dictionary containing all model inputs
            target_batch: Target batch size
            target_length: Target sequence length

        Returns:
            dict: Adjusted model inputs
        """
        adjusted_kwargs = {}

        # Get current dimensions from input_ids
        input_ids = model_kwargs["input_ids"]
        current_batch, current_length = input_ids.shape
        # Define dimension adjustments for different input types

        # Process each input tensor
        for key, tensor in model_kwargs.items():
            if tensor is None:
                adjusted_kwargs[key] = None
                continue

            if not isinstance(tensor, jax.Array | jax.numpy.ndarray | np.generic | np.ndarray):
                adjusted_kwargs[key] = tensor
                continue

            seq_dim = self.SEQUENCE_DIM_MAPPING.get(key, None)
            if seq_dim is None:
                adjusted_kwargs[key] = tensor
                continue

            tensor_shape = list(tensor.shape)

            if seq_dim < len(tensor_shape):
                if current_length > target_length:
                    slicing = [slice(None)] * len(tensor_shape)
                    slicing[seq_dim] = slice(0, target_length)
                    tensor = tensor[tuple(slicing)]
                elif current_length < target_length:
                    pad_width = [(0, 0)] * len(tensor_shape)
                    pad_width[seq_dim] = (target_length - current_length, 0)

                    if key == "input_ids":
                        pad_value = self.generation_config.pad_token_id
                    elif key in ["attention_mask", "token_type_ids"]:
                        pad_value = 0
                    elif key == "position_ids":
                        pad_value = -1
                    else:
                        pad_value = 0

                    tensor = jax.numpy.pad(tensor, pad_width, constant_values=pad_value)

            if current_batch != target_batch:
                batch_dim = 0
                if current_batch > target_batch:
                    slicing = [slice(None)] * len(tensor_shape)
                    slicing[batch_dim] = slice(0, target_batch)
                    tensor = tensor[tuple(slicing)]
                else:
                    pad_width = [(0, 0)] * len(tensor_shape)
                    pad_width[batch_dim] = (target_batch - current_batch, 0)

                    if key == "input_ids":
                        pad_value = self.generation_config.pad_token_id
                    elif key in ["attention_mask", "token_type_ids"]:
                        pad_value = 0
                    elif key == "position_ids":
                        pad_value = tensor_shape[seq_dim] - 1
                    else:
                        pad_value = 0
                    tensor = jax.numpy.pad(tensor, pad_width, constant_values=pad_value)

            adjusted_kwargs[key] = tensor

        return adjusted_kwargs

    def adjust_kwargs(
        self,
        graphstate: nn.GraphState,
        graphother: nn.GraphState,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        **model_kwargs,
    ):
        if not isinstance(input_ids, jax.Array):
            input_ids = jnp.array(input_ids, dtype=jnp.int32)

        model_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask

        batch_size, sequence_length = input_ids.shape
        target_batch, target_length = self._find_optimal_config(
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        adjusted_kwargs = self._adjust_inputs_to_config(
            model_kwargs=model_kwargs,
            target_batch=target_batch,
            target_length=target_length,
        )
        vinference_compile_config = self._create_vinference_config_from_kwargs(
            kwargs=adjusted_kwargs,
            batch_size=target_batch,
            prefill_length=target_length,
            graphstate=graphstate,
            graphother=graphother,
        )
        self.precompile(
            config=vinference_compile_config,
            graphstate=graphstate,
            graphother=graphother,
        )

        if target_batch <= 0 or target_length <= 0:
            raise ValueError(f"Invalid target dimensions: ({target_batch}, {target_length})")
        return adjusted_kwargs, vinference_compile_config

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
        sampling_params: SamplingParams | None = None,
        **model_kwargs,
    ) -> tp.Generator[SampleState | tp.Any, SampleState, SampleState]:
        """
        Generates text in streaming chunks with comprehensive input adjustment.

        Args:
            input_ids: Input token IDs as a JAX array
            attention_mask: Optional attention mask for the input
            graphstate (nn.GraphState, optional): in case that you want to update model state for generation.
            graphother (nn.GraphState, optional): in case that you want to update model ostate for generation.
            **model_kwargs: Additional model-specific keyword arguments

        Returns:
            Generator yielding SampleState objects containing generation results and metrics
        """
        self._metrics_increase_queue()
        try:
            adjusted_kwargs, vinference_compile_config = self.adjust_kwargs(
                graphstate=graphstate,
                graphother=graphother,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_kwargs,
            )

            # Prepare generation context
            with self._inference_latency_context_manager("preprocessing"):
                input_ids = adjusted_kwargs.pop("input_ids", None)
                attention_mask = adjusted_kwargs.pop("attention_mask", None)

                state = self._prepare_sample_state(
                    graphstate=graphstate,
                    graphother=graphother,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    model_kwargs=adjusted_kwargs,
                    vinference_compile_config=vinference_compile_config,
                )
                state.padded_length = vinference_compile_config.prefill_length

                generate_func, interval_func = get_compiled_funcs(
                    standalone_config=vinference_compile_config,
                    id=self._uuid4,
                )

            with self._inference_latency_context_manager("inference"):
                state = yield from self._inner_generate(
                    state,
                    generate_func,
                    interval_func,
                    graphstate=graphstate,
                    graphother=graphother,
                    compile_config=vinference_compile_config,
                    sampling_params=sampling_params,
                )

            self._post_generation_metrics_update(state)
            return state

        except Exception as e:
            raise e
            # self._handle_generation_error(e)

        finally:
            self._metrics_decrease_queue()

    def _prepare_sample_state(
        self,
        graphstate: nn.GraphState,
        graphother: nn.GraphState,
        input_ids: jax.Array,
        attention_mask: jax.Array | None,
        model_kwargs: dict,
        vinference_compile_config: vInferencePreCompileConfig,
    ) -> SampleState:
        """Prepares the initial state for text generation."""
        if attention_mask is None:
            warnings.warn("No attention mask provided. Using default mask.", UserWarning, stacklevel=2)
            attention_mask = jnp.ones_like(input_ids, dtype="b1")

        attention_mask = jnp.asarray(attention_mask, dtype="b1", device=self.input_sharding)
        input_ids = jnp.asarray(input_ids, dtype="i4", device=self.input_sharding)
        model_kwargs.update({"input_ids": input_ids, "attention_mask": attention_mask})
        if model_kwargs.get("rng") is None:
            model_kwargs["rng"] = self._rng_generator.rng
        if graphstate is None:
            graphstate = self.graphstate
        if graphother is None:
            graphother = self.graphother

        return self._get_init_state(
            graphstate=graphstate,
            graphother=graphother,
            vinference_compile_config=vinference_compile_config,
            model_kwargs=model_kwargs,
        )

    def _inner_generate(
        self,
        state: SampleState,
        prefill_func: tp.Callable[[tp.Any], SampleState] | None,
        interval_func: tp.Callable[[tp.Any], SampleState] | None,
        *,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
        compile_config: vInferencePreCompileConfig | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> tp.Generator[SampleState, tp.Any, tp.Any]:
        """Core generation loop with performance monitoring."""
        if sampling_params is not None:
            if sampling_params.max_tokens > self.generation_config.max_new_tokens:
                sampling_params.max_tokens = self.generation_config.max_new_tokens
            sampling_params = sampling_params.make_jitable()
            assert isinstance(sampling_params, JitableSamplingParams)
        # Initial generation step
        state = self.execute_prefill(
            state,
            graphstate=graphstate,
            graphother=graphother,
            compile_config=compile_config,
            sampling_params=sampling_params,
            func=prefill_func,
        )
        if not state.is_sequence_finished.all():
            # Subsequent generation steps
            for _ in range(self.generation_config._loop_rows):
                state = self.execute_decode(
                    state,
                    graphstate=graphstate,
                    graphother=graphother,
                    compile_config=compile_config,
                    sampling_params=sampling_params,
                    func=interval_func,
                )
                yield state
                if state.is_sequence_finished.all():
                    break
        return state

    def _prepare_prefill_inputs(
        self,
        state,
        *,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
        compile_config: vInferencePreCompileConfig | None = None,
        sampling_params: SamplingParams | None = None,
        func: tp.Callable[[tp.Any], SampleState] | None = None,
    ) -> tuple[tp.Any | jax.Array]:
        _ = compile_config

        if graphstate is None:
            graphstate = self.graphstate
        if graphother is None:
            graphother = self.graphother
        if sampling_params is None:
            sampling_params = self.generation_config.sampling_params.make_jitable()

        assert graphstate is not None
        assert graphother is not None

        if func is None:
            func = get_compiled_funcs(state._compile_config, self._uuid4)[0]
        if isinstance(func, Compiled):
            return (
                graphstate,
                graphother,
                state,
                sampling_params,
            )
        return (
            self.graphdef,
            graphstate,
            graphother,
            state,
            self.generation_config,
            sampling_params,
        )

    def _prepare_decode_inputs(
        self,
        state: SampleState,
        *,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
        compile_config: vInferencePreCompileConfig | None = None,
        sampling_params: SamplingParams | None = None,
        func: tp.Callable[[tp.Any], SampleState] | None = None,
    ) -> tuple[tp.Any | jax.Array]:
        _ = compile_config
        if graphstate is None:
            graphstate = self.graphstate
        if graphother is None:
            graphother = self.graphother
        if sampling_params is None:
            sampling_params = self.generation_config.sampling_params.make_jitable()

        if func is None:
            func = get_compiled_funcs(state._compile_config, self._uuid4)[1]
        if isinstance(func, Compiled):
            return (
                graphstate,
                graphother,
                state,
                sampling_params,
                self.generation_config.streaming_chunks,
            )
        return (
            self.graphdef,
            graphstate,
            graphother,
            state,
            self.sampling_params,
            self.generation_config.streaming_chunks,
        )

    def execute_prefill(
        self,
        state: SampleState,
        *,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
        compile_config: vInferencePreCompileConfig | None = None,
        sampling_params: SamplingParams | None = None,
        func: tp.Callable[[tp.Any], SampleState] | None,
    ) -> SampleState:
        """Executes a single generation step with performance monitoring."""
        inputs = self._prepare_prefill_inputs(
            state,
            graphstate=graphstate,
            graphother=graphother,
            compile_config=compile_config,
            sampling_params=sampling_params,
            func=func,
        )
        state = jax.block_until_ready(func(*inputs))
        return state

    def execute_decode(
        self,
        state: SampleState,
        *,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
        compile_config: vInferencePreCompileConfig | None = None,
        sampling_params: SamplingParams | None = None,
        func: tp.Callable[[tp.Any], SampleState] | None,
    ) -> SampleState:
        inputs = self._prepare_decode_inputs(
            state,
            graphstate=graphstate,
            graphother=graphother,
            compile_config=compile_config,
            sampling_params=sampling_params,
            func=func,
        )

        with capture_time() as time_spent:
            state = jax.block_until_ready(func(*inputs))
        state._time_spent_computing += time_spent()
        state.tokens_per_second = state.generated_tokens / state._time_spent_computing
        return state

    def _handle_generation_error(self, error: Exception):
        """Handles errors during generation with appropriate logging and cleanup."""
        self._submit_during_generation_metrics_update()

        if isinstance(error, ValueError):
            raise ValueError(f"Invalid input configuration: {error!s}") from error

        raise RuntimeError(f"Generation failed: {error!s}") from error

    def _compile_and_lower_funs(
        self,
        standalone_config: vInferencePreCompileConfig,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
    ):
        assert standalone_config._im_standalone()
        funs = get_compiled_funcs(standalone_config=standalone_config, id=self._uuid4, safe=False)
        do_compile = funs is None
        if do_compile:
            graphstate = graphstate if graphstate is not None else self.graphstate
            graphother = graphother if graphother is not None else self.graphother
            self.log("initiating state for lowering and compiling func.")
            model = self.get_model(graphstate=graphstate, graphother=graphother)
            wargs = model._get_compile_model_kwargs(
                input_tokens_length=standalone_config.prefill_length,
                input_sharding=self.input_sharding,
                rngs=self._rng_generator.rng,
                **standalone_config.extract(),
            )
            state = self._get_init_state(
                graphstate=graphstate,
                graphother=graphother,
                vinference_compile_config=standalone_config,
                model_kwargs=wargs,
            )

            self.log("smart compiling `prefill`")
            self.log("lowering `prefill`")
            prefill_lowered = jax.jit(
                prefill_func,
                donate_argnums=(3,),
                static_argnums=(0, 4),
                in_shardings=(
                    es.extract_shardings(graphstate),
                    es.extract_shardings(graphother),
                    es.extract_shardings(state),
                    es.extract_shardings(self.generation_config.sampling_params.make_jitable()),
                ),
            ).lower(
                self.graphdef,  # Static
                graphstate,
                graphother,
                state,
                self.generation_config,  # Static
                self.generation_config.sampling_params.make_jitable(),
            )
            self.log("`prefill` lowered successfully.")
            compiled_prefill_func, _ = smart_compile(prefill_lowered, tag="vinference.prefill_func")
            self.log("smart compiling `decode`")
            self.log("lowering `decode`")
            sample_state = compiled_prefill_func(
                graphstate,
                graphother,
                state,
                self.generation_config.sampling_params.make_jitable(),
            )
            sample_state_shardings = es.extract_shardings(sample_state)
            decode_lowered = jax.jit(
                interval_func,
                donate_argnums=(3,),
                static_argnums=(0, 4),
                in_shardings=(
                    es.extract_shardings(graphstate),
                    es.extract_shardings(graphother),
                    sample_state_shardings,
                    es.extract_shardings(self.generation_config.sampling_params.make_jitable()),
                    None,
                ),
                out_shardings=sample_state_shardings,
            ).lower(
                self.graphdef,
                graphstate,
                graphother,
                sample_state,
                self.generation_config,  # STATIC
                self.generation_config.sampling_params.make_jitable(),
                self.generation_config.streaming_chunks,
            )
            self.log("`decode` lowered successfully.")
            compiled_interval_func, _ = smart_compile(decode_lowered, tag="vinference.interval_func")

            del state
            self.log("saving compiled functions...")
            put_compiled_funcs(
                funcs=(compiled_prefill_func, compiled_interval_func),
                standalone_config=standalone_config,
                id=self._uuid4,
            )

    def precompile(
        self,
        config: vInferencePreCompileConfig,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
    ):
        """
        Precompiles the generation functions for a given batch size and input length.

        This function checks if the generation functions have already been compiled for
        the given configuration. If not, it compiles them asynchronously and stores them
        in a cache.

        Returns:
          bool: True if precompilation was successful, False otherwise.
        """
        if config.prefill_length is None:
            config.prefill_length = self.model_prefill_length
            self.log("`input_tokens_length` is None using `vInference.model_prefill_length`")
        for standalone_config in config.get_standalones():
            config_hash = standalone_config.get_default_hash()

            if config_hash in self._precompiled_configs.keys():
                return True
            if config_hash in self._in_compiling_process:
                self.log(
                    f"lowering and compiling with `config` {config_hash} have "
                    "already been requested adding 5 second timeout"
                )
                time.sleep(5)
                return self.precompile(
                    config=standalone_config,
                    graphstate=graphstate,
                    graphother=graphother,
                )
            else:
                with self._compilation_metrics_recorder():
                    self.log(f"lowering and compiling with `config` {config_hash}")
                    self._in_compiling_process.add(config_hash)
                    with self.mesh:
                        self._compile_and_lower_funs(
                            standalone_config=standalone_config,
                            graphstate=graphstate,
                            graphother=graphother,
                        )
                    self._precompiled_configs.update({config_hash: standalone_config})
        return True

    def save_inference(self, path: os.PathLike | str):
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True, parents=True)
        metadata = vInferenceMetaData(
            inference_name=self.inference_name,
            generation_config=self.generation_config,
            precompiled_configs=self._precompiled_configs,
            in_compiling_process=self._in_compiling_process,
            input_partition_spec=self.input_partition_spec,
            uuid4=self._uuid4,
        )
        for config_key, config in self._precompiled_configs.items():
            metafile = f"{metadata.uuid4}-{config_key}"
            compiled_prefill_func, compiled_interval_func = get_compiled_funcs(
                standalone_config=config, id=metadata.uuid4
            )
            save_compiled_fn(path=path, fn=compiled_prefill_func, prefix=f"compiled-prefill-{metafile}")
            save_compiled_fn(path=path, fn=compiled_interval_func, prefix=f"compiled-decode-{metafile}")

        metadata = pickle.dump(metadata, open(path / "config", "wb"))

    @classmethod
    def load_inference(
        cls,
        path: os.PathLike | str,
        model: EasyDeLBaseModule,
        processor_class: ProcessingClassType,
    ):
        path = pathlib.Path(path)
        assert path.exists(), "provided path to vInference doesn't exists."
        metadata = pickle.load(open(path / "config", "rb"))
        for config_key, standalone_config in metadata.precompiled_configs.items():
            metafile = f"{metadata.uuid4}-{config_key}"
            compiled_prefill_func = load_compiled_fn(path=path, prefix=f"compiled-prefill-{metafile}")
            compiled_interval_func = load_compiled_fn(path=path, prefix=f"compiled-decode-{metafile}")
            put_compiled_funcs(
                funcs=(compiled_prefill_func, compiled_interval_func),
                standalone_config=standalone_config,
                id=metadata.uuid4,
            )
        self = cls(
            model=model,
            processor_class=processor_class,
            generation_config=metadata.generation_config,
            input_partition_spec=metadata.input_partition_spec,
            inference_name=metadata.inference_name,
        )
        self._uuid4 = metadata.uuid4
        self._precompiled_configs = metadata.precompiled_configs
        return self

    @tp.overload
    def count_tokens(
        self,
        messages: list[dict[str, str]],
        oai_like: bool = False,
    ): ...

    @tp.overload
    def count_tokens(
        self,
        text: str,
        oai_like: bool = False,
    ): ...

    def count_tokens(
        self,
        conv: str | list[dict[str, str]],
        oai_like: bool = False,
    ) -> int:
        if isinstance(conv, list) and all(isinstance(item, dict) for item in conv):
            if isinstance(self.processor_class, ProcessorMixin) and oai_like:
                from easydel.trainers.prompt_utils import convert_to_openai_format

                conv = convert_to_openai_format(conv)

            tokens = self.processor_class.apply_chat_template(conv, tokenize=True)
            return len(tokens)
        else:
            tokens = self.tokenizer.encode(conv)
            return len(tokens)

    def process_prompt(
        self,
        prompt: str | list[str] | list[dict[str, str]],
        sampling_params: SamplingParams | dict,
        stream: bool = False,
    ) -> (
        PromptOutput
        | list[PromptOutput]
        | tp.Generator[str, None, SampleState]
        | list[tp.Generator[str, None, SampleState]]
    ):
        """
        Processes a prompt (string, list of strings, or OpenAI messages) and generates a response.

        Args:
                prompt: The input prompt. Can be a single string, a list of strings (processed sequentially),
                                or a list of dictionaries representing OpenAI-style chat messages
                                (processed as a single batch).
                sampling_params: Configuration for the generation process (temperature, top_p, etc.).
                                                 Can be a SamplingParams object or a dictionary.
                stream: If True, yields generated tokens incrementally. If False, returns the
                                complete generation(s) at the end.

        Returns:
                - If input is `str` or `List[Dict]` and `stream=False`: A `PromptOutput`
                    object containing the full text and metrics.
                - If input is `str` or `List[Dict]` and `stream=True`: A generator yielding string chunks.
                    The generator's return value (accessible via try/except StopIteration) is the final `SampleState`.
                - If input is `List[str]` and `stream=False`: A list of `PromptOutput` objects.
                - If input is `List[str]` and `stream=True`: A list where each element is a generator as described above
                    for the single stream case.

        Raises:
                TypeError: If the prompt format is invalid or processor does not support chat templates.
                ValueError: If tokenization or processing fails.
        """

        if isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            results = []
            for sub_prompt in prompt:
                result = self.process_prompt(sub_prompt, sampling_params, stream=stream)
                results.append(result)
            return results

        if isinstance(sampling_params, dict):
            _sampling_params = SamplingParams(**sampling_params)
        elif isinstance(sampling_params, SamplingParams):
            _sampling_params = sampling_params
        else:
            raise TypeError(f"sampling_params must be a dict or SamplingParams object, got {type(sampling_params)}")

        try:
            if isinstance(prompt, str):
                tokenized = self.tokenizer(
                    prompt,
                    return_tensors="np",
                    return_attention_mask=True,
                )

            elif isinstance(prompt, list) and all(isinstance(p, dict) for p in prompt):
                if not hasattr(self.processor_class, "apply_chat_template"):
                    raise TypeError(
                        "Processor class does not have 'apply_chat_template' method required for chat messages."
                    )
                tokenized = self.processor_class.apply_chat_template(
                    conversation=prompt,
                    return_tensors="np",
                    add_generation_prompt=True,
                    return_dict=True,
                    tokenize=True,
                )
            else:
                raise TypeError(
                    "prompt must be a string, a list of strings, or a list of dictionaries (OpenAI chat format)."
                )
        except Exception as e:
            logger.error(f"Error during tokenization: {e}", exc_info=True)
            raise ValueError(f"Failed to tokenize input prompt: {e}") from e

        model_kwargs = {key: val for key, val in tokenized.items()}
        generator = self.generate(**model_kwargs, sampling_params=_sampling_params)

        if not stream:
            last_state = None
            for state in generator:
                last_state = state

            if last_state is None:
                logger.warning("Generation finished without producing any state.")
                return ""

            sequences = last_state.sequences
            padded_length = last_state.padded_length

            decoded_list = self.tokenizer.batch_decode(
                sequences[..., padded_length:],
                skip_special_tokens=True,
            )
            finish_reason = "length" if last_state.generated_tokens >= self.generation_config.max_new_tokens else "stop"
            return PromptOutput(
                text=decoded_list[0] if decoded_list else "",
                generated_tokens=last_state.generated_tokens,
                tokens_per_second=last_state.tokens_per_second,
                finish_reason=finish_reason,
            )
        else:

            def _stream_helper(
                gen: tp.Generator[SampleState, None, None],
            ) -> tp.Generator[str, None, SampleState]:
                last_decoded_length = 0
                last_state = None
                try:
                    for response_state in gen:
                        last_state = response_state
                        if response_state.sequences.shape[0] > 1:
                            logger.warning("Streaming output expects batch size 1, but got > 1. Using first sequence.")

                        full_sequence = response_state.sequences[0]
                        padded_length = response_state.padded_length

                        current_full_decoded = self.tokenizer.decode(
                            full_sequence[padded_length:],
                            skip_special_tokens=True,
                        )
                        new_text = current_full_decoded[last_decoded_length:]
                        if new_text:
                            yield new_text
                            last_decoded_length = len(current_full_decoded)
                except Exception as e:
                    logger.error(f"Error during streaming generation: {e}", exc_info=True)
                    yield f"[ERROR: {e}]"
                return last_state

            return _stream_helper(generator)

    async def process_prompts_concurrently(
        self,
        prompts: list[str | list[dict[str, str]]],
        max_concurrent_requests: int,
        sampling_params: SamplingParams | None = None,
        stream: bool = False,
        progress_callback: tp.Callable[[int, int], None] | None = None,
    ) -> list[PromptOutput] | tp.AsyncGenerator[tuple[int, str, tp.Any], None]:
        """
        Processes a list of prompts concurrently, supporting both streaming and non-streaming modes.

        Args:
                prompts: A list of prompts (strings or OpenAI-style message lists).
                max_concurrent_requests: The maximum number of prompts to process in parallel.
                                         If <= 0, processing will be sequential (but still async).
                sampling_params: Optional sampling parameters to override the default ones
                                for this batch of requests. Passed to process_prompt.
                stream: If True, returns an async generator yielding tuples of (index, type, data).
                        If False, returns a list of PromptOutput objects.
                progress_callback: An optional function called after each prompt *finishes* processing.
                                   Receives (completed_count, total_count).

        Returns:
                - If stream=False: A list of `PromptOutput` objects containing the full text, metrics,
                                                   or error information for each prompt, in the original order.
                - If stream=True: An async generator yielding tuples `(index, type, data)` where:
                                                        - `type` is 'text' and `data` is the string chunk.
                                                        - `type` is 'error' and `data` is the error string.
                                                        - `type` is 'final' and `data` is the final `PromptOutput`
                                                            object with metrics.
                                                  The generator finishes when all prompts are processed.

        Raises:
                ValueError: If input arguments are invalid.
                RuntimeError: If an unexpected error occurs during processing.
        """
        num_prompts = len(prompts)
        if num_prompts == 0:
            if stream:

                async def _empty_generator():
                    if False:
                        yield

                return _empty_generator()
            else:
                return []

        effective_concurrency = max(1, max_concurrent_requests) if max_concurrent_requests > 0 else 1

        if not stream:
            self.log(
                f"Processing {num_prompts} prompts concurrently (non-streaming, max_workers={effective_concurrency})"
            )
            results = [None] * num_prompts
            completed_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
                loop = asyncio.get_running_loop()
                tasks = []

                def process_and_collect_full(index, p, sp):
                    full_response_text = ""
                    final_state = None
                    error_msg = None
                    try:
                        generator = self.process_prompt(prompt=p, sampling_params=sp, stream=True)
                        try:
                            while True:
                                chunk = next(generator)  # Get chunks
                                if isinstance(chunk, str) and not chunk.startswith("[ERROR:"):
                                    full_response_text += chunk
                        except StopIteration as stop:
                            final_state = stop.value  # Get final SampleState from return value
                        except Exception as gen_e:  # Catch errors during generation
                            logger.error(f"Error during generation for prompt {index}: {gen_e}")
                            error_msg = f"[Error: {type(gen_e).__name__}: {gen_e}]"
                            # Try to get last state even if error occurred mid-stream
                            # This part depends on how _stream_helper handles errors and returns state
                            # Assuming it might return the last valid state before error
                            if hasattr(generator, "__return__"):  # Check if return value accessible
                                final_state = generator.__return__

                    except Exception as e:  # Catch errors in process_prompt setup itself
                        logger.error(f"Error setting up process_prompt for index {index}: {e}")
                        error_msg = f"[Error: {type(e).__name__}: {e}]"

                    if error_msg:
                        return index, PromptOutput(error=error_msg)
                    elif final_state:
                        finish_reason = (
                            "length" if final_state.generated_tokens >= self.generation_config.max_new_tokens else "stop"
                        )
                        return index, PromptOutput(
                            text=full_response_text,
                            generated_tokens=final_state.generated_tokens,
                            tokens_per_second=final_state.tokens_per_second,
                            finish_reason=finish_reason,
                        )
                    else:
                        logger.error(f"Prompt {index} finished without error or final state.")
                        return index, PromptOutput(error="[Error: Unknown processing error]")

                for index, prompt in enumerate(prompts):
                    task = loop.run_in_executor(
                        executor,
                        process_and_collect_full,
                        index,
                        prompt,
                        sampling_params,
                    )
                    tasks.append(task)

                for future in asyncio.as_completed(tasks):
                    try:
                        index, result = await future
                        results[index] = result
                    except Exception as e:
                        # This shouldn't ideally happen if process_and_collect catches, but as a safeguard
                        logger.error(f"Unexpected error collecting result: {e}")
                        # Find which index this future corresponds to (less efficient)
                        # This part is tricky without the future_to_index mapping directly with asyncio tasks
                        # For simplicity, we rely on process_and_collect returning the index
                        pass  # Error already logged in process_and_collect

                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, num_prompts)

            return results

        else:
            # TODO: impl streaming method
            raise NotImplementedError()
