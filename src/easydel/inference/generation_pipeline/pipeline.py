import functools
import pathlib
import pickle
import warnings
from typing import Optional, Union

import fjformer
import flax.core
import jax
import jax.experimental
from fjformer import GenerateRNG
from jax import lax, random
from jax import numpy as jnp
from jax.experimental.serialize_executable import deserialize_and_load, serialize
from jax.sharding import NamedSharding, PartitionSpec

from easydel.etils.etils import get_logger
from easydel.inference.generation_pipeline import utils as inference_utils
from easydel.modules.modeling_utils import EDPretrainedModel

GenerationPipelineConfig = inference_utils.GenerationPipelineConfig
_DynamicGenerationConfig = inference_utils._DynamicGenerationConfig
logger = get_logger(__name__)


class GenerationPipeline:
    """
    This class implements the generation pipeline for text generation models using JAX/Flax.

    Args:
        model (EDPretrainedModel): The pretrained model to use for generation.
        params (Union[flax.core.FrozenDict, dict]): The model parameters.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        generation_config (Optional[GenerationPipelineConfig], optional): The configuration for the generation
            pipeline. Defaults to None.
        add_params_field (None, optional): Deprecated argument.
        seed (Optional[int], optional): The random seed for generation. Defaults to None.
        input_partition_spec (PartitionSpec, optional): The partition specification for input data.
            Defaults to PartitionSpec(("dp", "fsdp")).
        partition_rules (None, optional): Custom partition rules for sharding. Defaults to None, which uses
            model's default partition rules.
        parameters_are_quantized (bool, optional): Whether the model parameters are quantized. Defaults to False.
        force_sharding (bool): Whether to pass in and out shardings to `jax.jit` function
            (must be off in case of using `parameters_are_quantized`). Defaults to False.
    """

    def __init__(
        self,
        model: EDPretrainedModel,
        params: Union[flax.core.FrozenDict, dict],
        tokenizer: "PreTrainedTokenizer",  # noqa #type:ignore
        generation_config: Optional[GenerationPipelineConfig] = None,
        add_params_field=None,
        seed: Optional[int] = None,
        input_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp")),
        partition_rules=None,
        parameters_are_quantized: bool = False,
        force_sharding: bool = False,
    ):
        if add_params_field is not None:
            warnings.warn("`add_params_field` is deprecated and soon will be removed.")
        if generation_config is None:
            generation_config = GenerationPipelineConfig()
        params_get = params.get("params", None)
        if params_get is not None:
            warnings.warn("`params` field should be like {k:v} not {'params':{k:v}}")
            params = params_get
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.parameters_are_quantized = parameters_are_quantized
        self.force_sharding = force_sharding
        self.shard_generation_state = None
        self.compiled_sample_fn_over = None
        self.compiled_func = None
        self.compiled_model_kwargs_sharding = None
        self.over_compiled_func = None
        self._rng_gen = GenerateRNG(seed or 42)
        self.input_partition_spec = input_partition_spec
        self.mesh = self.model.config.mesh
        if partition_rules is None:
            partition_rules = self.model.config.get_partition_rules(True)
        self.model_sharding = self.model.get_named_sharding(
            partition_rules=partition_rules
        )
        self.input_sharding = NamedSharding(
            spec=input_partition_spec,
            mesh=self.model.mesh,
        )
        self.empty_sharding = NamedSharding(
            spec=PartitionSpec(),
            mesh=self.model.mesh,
        )
        self.gen_input_sharding = NamedSharding(
            spec=PartitionSpec(input_partition_spec[0], None),
            mesh=self.model.mesh,
        )
        self.compiled_sample_fn = None
        self.compiled_sample_fn_multi_sequence = None  # multi sequences
        # Ensure essential token IDs are set
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id
        if self.generation_config.eos_token_id is None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id
        if self.generation_config.bos_token_id is None:
            self.generation_config.bos_token_id = tokenizer.bos_token_id
        self.dynamic_generation_config = _DynamicGenerationConfig(
            self.generation_config
        )
        assert self.generation_config.pad_token_id is not None, (
            "`pad_token_id` cannot be None. "
            "(Set `tokenizer.pad_token_id = tokenizer.eos_token_id` if undefined)"
        )
        assert (
            self.generation_config.eos_token_id is not None
        ), "`eos_token_id` cannot be None."

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        echo: bool = False,
        **kwargs,  # Ignored arguments.
    ):
        """
        Generates text sequences based on the provided input_ids.

        Args:
            input_ids (jax.Array): The input token IDs.
            attention_mask (Optional[jax.Array], optional): The attention mask for the input_ids.
                Defaults to None.
            position_ids (Optional[jax.Array], optional): The position IDs for the input_ids.
                Defaults to None.
            echo (bool, optional): Whether to echo the input sequence in the output. Defaults to False.

        Yields:
            jax.Array: Generated token IDs.
        """
        partition_axes = self.model.config.partition_axis
        mesh = self.mesh
        eos_token_id = jnp.array(self.generation_config.eos_token_id, dtype=jnp.int32)
        pad_token_id = jnp.array(self.generation_config.pad_token_id, dtype=jnp.int32)
        batch_size, current_length = input_ids.shape
        max_length = current_length + self.generation_config.max_new_tokens
        current_length = jnp.array(current_length)
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))
        is_sequence_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        if attention_mask is None:
            warnings.warn(
                "`attention_mask` is not provided, it's recommended to "
                "pass an attention mask for better results."
            )
            attention_mask = jnp.ones_like(input_ids)

        if position_ids is None:
            position_ids = attention_mask.cumsum(axis=-1, dtype="i4") - 1
        with mesh:
            input_ids = fjformer.with_sharding_constraint(
                input_ids,
                PartitionSpec(
                    partition_axes.batch_axis, partition_axes.key_sequence_axis
                ),
            )
            attention_mask = fjformer.with_sharding_constraint(
                attention_mask,
                PartitionSpec(
                    partition_axes.batch_axis, partition_axes.key_sequence_axis
                ),
            )
            position_ids = fjformer.with_sharding_constraint(
                position_ids,
                PartitionSpec(
                    partition_axes.batch_axis, partition_axes.key_sequence_axis
                ),
            )
        assert (
            position_ids.shape == attention_mask.shape
        ), "`position_ids` and `attention_mask` must have the same shape."
        model_kwargs_sharding_spec = self.get_model_arguments_sharding(
            input_ids, max_length, attention_mask
        )

        model_kwargs = self.get_compiled_model_kwargs_sharding(
            input_ids,
            max_length,
            attention_mask,
            model_kwargs_sharding_spec,
        )()
        generation_state = inference_utils.SampleState(
            current_length=current_length,
            sequences=sequences,
            running_token=input_ids,
            is_sequence_finished=is_sequence_finished,
            prng_key=self._rng_gen.rng,
            model_kwargs=model_kwargs,
        )

        def should_continue_sampling(state):
            """state termination condition fn."""
            all_sequence_finished = jnp.all(state.is_sequence_finished)
            return ~jnp.logical_or(
                all_sequence_finished, state.current_length >= max_length
            )

        def perform_sampling_step(params, state: inference_utils.SampleState):
            """
            Performs a single sampling step for text generation.

            Args:
                params: Model parameters.
                state (inference_utils.SampleState): The current generation state.

            Returns:
                inference_utils.SampleState: The updated generation state.
            """
            model_outputs = self.model(
                input_ids=state.running_token,
                params=params,
                add_params_field=True,
                return_dict=True,
                **state.model_kwargs,
            )
            next_token = inference_utils.inference_step_compiled(
                model_outputs.logits[:, -1],
                state.sequences,
                state.prng_key,
                self.generation_config,
                current_length,
                self.generation_config.max_new_tokens,
            )

            next_token = (
                next_token * ~state.is_sequence_finished
                + pad_token_id * state.is_sequence_finished
            )

            next_sequence_finished = (
                state.is_sequence_finished | next_token == eos_token_id
            )

            next_token = next_token[:, None]
            next_sequences = lax.dynamic_update_slice(
                state.sequences, next_token, (0, state.current_length)
            )
            next_model_kwargs = self.model.update_inputs_for_generation(
                model_outputs, state.model_kwargs
            )
            return inference_utils.SampleState(
                current_length=state.current_length + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sequence_finished=next_sequence_finished,
                prng_key=random.split(state.prng_key, 2)[0],
                model_kwargs=next_model_kwargs,
            )

        with self.mesh:
            if input_ids.shape[1] > 1:
                if self.parameters_are_quantized:
                    generation_state = self.compile_multi_step_sampling(
                        perform_sampling_step=perform_sampling_step,
                        generation_state=generation_state,
                        model_kwargs_sharding_spec=model_kwargs_sharding_spec,
                    )(
                        self.params,
                        generation_state,
                    )
                else:
                    generation_state = perform_sampling_step(
                        self.params,
                        generation_state,
                    )
            while True:
                if should_continue_sampling(generation_state):
                    yield (
                        generation_state.sequences[
                            :, current_length : generation_state.current_length
                        ]
                        if echo
                        else generation_state.running_token
                    )
                    generation_state = self.get_or_compile_sampling_function(
                        perform_sampling_step=perform_sampling_step,
                        generation_state=generation_state,
                        model_kwargs_sharding_spec=model_kwargs_sharding_spec,
                    )(self.params, generation_state)
                else:
                    break
            del generation_state.model_kwargs
            del generation_state.sequences
            del generation_state.running_token
            del generation_state

    @property
    def partition_rules(self):
        """
        Defines the partition rules for model sharding.

        Returns:
            tuple: A tuple of partition rules for different model components.
        """
        partition_axes = self.model.config.partition_axis
        return (
            (
                "position_ids",
                PartitionSpec(partition_axes.batch_axis, None),
            ),
            (
                "attention_mask",
                PartitionSpec(partition_axes.batch_axis, None),
            ),
            (
                "cached_key",
                PartitionSpec(
                    partition_axes.batch_axis,
                    partition_axes.key_sequence_axis,
                    partition_axes.head_axis,
                    None,
                ),
            ),
            (
                "cached_value",
                PartitionSpec(
                    partition_axes.batch_axis,
                    partition_axes.key_sequence_axis,
                    partition_axes.head_axis,
                    None,
                ),
            ),
        )

    def get_model_arguments_sharding(self, input_ids, max_length, attention_mask):
        """
        Determines the sharding strategy for model arguments.

        Args:
            input_ids (jax.Array): Input token IDs.
            max_length (int): Maximum sequence length.
            attention_mask (jax.Array): Attention mask.

        Returns:
            jax.tree_util.tree_map: Sharding specifications for model arguments.
        """
        return jax.tree_util.tree_map(
            lambda spec: NamedSharding(spec=spec, mesh=self.mesh),
            fjformer.match_partition_rules(
                self.partition_rules,
                jax.eval_shape(
                    lambda: self.model.prepare_inputs_for_generation(
                        input_ids, max_length, attention_mask
                    )
                ),
            ),
        )

    def update_generation_config(self, **kwargs):
        """
        Updates the dynamic generation configuration.

        Args:
            **kwargs: Keyword arguments for updating the configuration.

        Raises:
            AttributeError: If an invalid configuration key is provided.
        """
        for key, value in kwargs.items():
            if hasattr(self.dynamic_generation_config, key):
                setattr(self.dynamic_generation_config, key, value)
            else:
                raise AttributeError(
                    f"DynamicGenerationConfig has no attribute '{key}'"
                )

    def compile_sampling_function(
        self,
        model_kwargs_sharding_spec,
        generation_state,
        perform_sampling_step,
    ):
        """
        Compiles the sampling function for efficient execution.

        Args:
            model_kwargs_sharding_spec: Sharding specifications for model keyword arguments.
            generation_state (inference_utils.SampleState): Initial generation state.
            perform_sampling_step: The sampling function to compile.
        """
        state_sharding = inference_utils.SampleState(
            self.empty_sharding,
            self.input_sharding,
            self.gen_input_sharding,
            self.empty_sharding,
            self.empty_sharding,
            model_kwargs_sharding_spec,
        )

        @functools.partial(
            jax.jit,
            in_shardings=(
                inference_utils.SampleState(
                    self.empty_sharding,
                    self.input_sharding,
                    self.gen_input_sharding,
                    self.empty_sharding,
                    self.empty_sharding,
                    self.empty_sharding,
                ),
            ),
            out_shardings=state_sharding,
        )
        def shard_generation_state(st):
            return st

        self.shard_generation_state = shard_generation_state
        self.compiled_sample_fn = inference_utils.compile_function(
            fjformer.core.implicit_compact(perform_sampling_step),
            (self.params, generation_state),
            {},
            mesh=self.mesh,
            in_shardings=(
                (
                    self.model_sharding,
                    state_sharding,
                )
                if self.force_sharding
                else None
            ),
            out_shardings=state_sharding if self.force_sharding else None,
        )

    def get_compiled_model_kwargs_sharding(
        self,
        input_ids,
        max_length,
        attention_mask,
        model_kwargs_sharding_spec,
    ):
        if self.compiled_model_kwargs_sharding is None:
            logger.info("compiling `model_kwargs_sharding`.")
            self.compiled_model_kwargs_sharding = inference_utils.compile_function(
                lambda: self.model.prepare_inputs_for_generation(
                    input_ids,
                    max_length,
                    attention_mask,
                ),
                out_shardings=model_kwargs_sharding_spec,
                func_input_args=(),
                func_input_kwargs=dict(),
            )
        return self.compiled_model_kwargs_sharding

    def compile_multi_step_sampling(
        self,
        model_kwargs_sharding_spec,
        generation_state,
        perform_sampling_step,
    ):
        if self.compiled_sample_fn_over is None:
            logger.info("compiling `sample_fn_over`.")

            @fjformer.core.implicit_compact
            def compile_multi_step_sampling(params_, generation_state_):
                return perform_sampling_step(params_, generation_state_)

            state_sharding = inference_utils.SampleState(
                self.empty_sharding,
                self.input_sharding,
                self.input_sharding,
                self.empty_sharding,
                self.empty_sharding,
                model_kwargs_sharding_spec,
            )
            self.compiled_sample_fn_over = inference_utils.compile_function(
                compile_multi_step_sampling,
                (self.params, generation_state),
                {},
                mesh=self.mesh,
                in_shardings=(
                    (
                        self.model_sharding,
                        state_sharding,
                    )
                    if self.force_sharding
                    else None
                ),
                out_shardings=state_sharding if self.force_sharding else None,
            )

        return self.compiled_sample_fn_over

    def get_or_compile_sampling_function(
        self,
        model_kwargs_sharding_spec,
        generation_state,
        perform_sampling_step,
    ):
        """
        Compiles and returns the compiled sampling function.

        Args:
            model_kwargs_sharding_spec: Sharding specifications for model keyword arguments.
            generation_state (inference_utils.SampleState): Initial generation state.
            perform_sampling_step: The sampling function to compile.

        Returns:
            Callable: The compiled sampling function.
        """
        if self.compiled_sample_fn is None:
            logger.info("compiling `perform_sampling_step`.")
            self.compile_sampling_function(
                model_kwargs_sharding_spec,
                generation_state,
                perform_sampling_step,
            )
        return self.compiled_sample_fn

    def save_aot_functions(self, path: str):
        """
        Save the ahead-of-time (AOT) compiled functions to disk.

        This method serializes and saves the compiled sample functions if they exist.
        It saves the serialized function and its associated tree structures in separate files.

        Args:
            path (str): The directory path where the serialized functions will be saved.
        """
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.compiled_sample_fn is not None:
            serialized_function, in_tree, out_tree = serialize(self.compiled_sample_fn)
            base_name = f"serialized_gen_step_{hash(self.generation_config)}"
            with open(path / f"{base_name}.fn", "wb") as buffer:
                buffer.write(serialized_function)
            with open(path / f"{base_name}.tree", "wb") as buffer:
                pickle.dump((in_tree, out_tree), buffer)

        if self.compiled_sample_fn_over is not None:
            serialized_function, in_tree, out_tree = serialize(
                self.compiled_sample_fn_over
            )
            base_name = f"serialized_gen_over_{hash(self.generation_config)}"
            with open(path / f"{base_name}.fn", "wb") as buffer:
                buffer.write(serialized_function)
            with open(path / f"{base_name}.tree", "wb") as buffer:
                pickle.dump((in_tree, out_tree), buffer)

        if self.compiled_model_kwargs_sharding is not None:
            serialized_function, in_tree, out_tree = serialize(
                self.compiled_model_kwargs_sharding
            )
            base_name = f"serialized_kwarg_{hash(self.generation_config)}"
            with open(path / f"{base_name}.fn", "wb") as buffer:
                buffer.write(serialized_function)
            with open(path / f"{base_name}.tree", "wb") as buffer:
                pickle.dump((in_tree, out_tree), buffer)

    def load_aot_functions(self, path: str):
        """
        Load the ahead-of-time (AOT) compiled functions from disk.

        This method deserializes and loads the compiled sample functions if they exist on disk.
        It loads the serialized function and its associated tree structures from separate files.

        Args:
            path (str): The directory path where the serialized functions are stored.
        """
        path = pathlib.Path(path)

        # Load compiled_sample_fn
        base_name = f"serialized_gen_step_{hash(self.generation_config)}"
        function_path = path / f"{base_name}.fn"
        tree_structure_path = path / f"{base_name}.tree"

        if function_path.exists() and tree_structure_path.exists():
            with open(function_path, "rb") as fn_buffer:
                serialized_function = fn_buffer.read()
            with open(tree_structure_path, "rb") as tree_buffer:
                in_tree, out_tree = pickle.load(tree_buffer)
            self.compiled_sample_fn = deserialize_and_load(
                serialized_function,
                in_tree,
                out_tree,
            )
            logger.info("`compiled_sample_fn` loaded.")
        else:
            logger.warn("Couldn't load `compiled_sample_fn`")

        # Load compiled_sample_fn_over
        base_name = f"serialized_gen_over_{hash(self.generation_config)}"
        function_path = path / f"{base_name}.fn"
        tree_structure_path = path / f"{base_name}.tree"

        if function_path.exists() and tree_structure_path.exists():
            with open(function_path, "rb") as fn_buffer:
                serialized_function = fn_buffer.read()
            with open(tree_structure_path, "rb") as tree_buffer:
                in_tree, out_tree = pickle.load(tree_buffer)
            self.compiled_sample_fn_over = deserialize_and_load(
                serialized_function,
                in_tree,
                out_tree,
            )
            logger.info("`compiled_model_kwargs_sharding` loaded.")
        else:
            logger.warn("Couldn't load `compiled_model_kwargs_sharding`")

        base_name = f"serialized_kwarg_{hash(self.generation_config)}"
        function_path = path / f"{base_name}.fn"
        tree_structure_path = path / f"{base_name}.tree"

        if function_path.exists() and tree_structure_path.exists():
            with open(function_path, "rb") as fn_buffer:
                serialized_function = fn_buffer.read()
            with open(tree_structure_path, "rb") as tree_buffer:
                in_tree, out_tree = pickle.load(tree_buffer)
            self.compiled_sample_fn_over = deserialize_and_load(
                serialized_function,
                in_tree,
                out_tree,
            )
            logger.info("`compiled_sample_fn_over` loaded.")
        else:
            logger.warn("Couldn't load `compiled_sample_fn_over`")


class ChatPipeline:
    """
    This class extends the GenerationPipeline for conversational text generation.

    Args:
        pipeline (GenerationPipeline): The underlying generation pipeline.
        max_prefill_length (int): The maximum length of the conversation history to consider.
        chat_template (str | None, optional): The chat template to use for formatting
            conversations. Defaults to None.
    """

    def __init__(
        self,
        pipeline: GenerationPipeline,
        max_prefill_length: int,
        chat_template: str | None = None,
    ):
        self.pipeline = pipeline
        self.max_prefill_length = max_prefill_length
        self.chat_template = chat_template

    def stream_generate(self, conversation):
        """
        Generates text for a conversational context in a streaming fashion.

        Args:
            conversation: The conversation history.

        Yields:
            str: Generated text chunks.
        """
        self.pipeline.tokenizer.padding_side = "left"
        inputs = self.pipeline.tokenizer.apply_chat_template(
            conversation=conversation,
            max_length=self.max_prefill_length,
            padding="max_length",
            return_tensors="np",
            return_dict=True,
            add_generation_prompt=True,
            tokenize=True,
            chat_template=self.chat_template,
        )
        captured_length = 0
        for sequence in self.pipeline.generate(echo=True, **inputs):
            decoded_sequence = self.pipeline.tokenizer.decode(sequence[0])[
                captured_length:
            ]
            yield decoded_sequence
            captured_length += len(decoded_sequence)

    def generate(self, conversation):
        """
        Generates text for a conversational context.

        Args:
            conversation: The conversation history.

        Returns:
            str: The generated text.
        """
        self.pipeline.tokenizer.padding_side = "left"
        inputs = self.pipeline.tokenizer.apply_chat_template(
            conversation=conversation,
            max_length=self.max_prefill_length,
            padding="max_length",
            return_tensors="np",
            return_dict=True,
            add_generation_prompt=True,
            tokenize=True,
            chat_template=self.chat_template,
        )
        decoded_sequence = ""
        for sequence in self.pipeline.generate(echo=True, **inputs):
            decoded_sequence = self.pipeline.tokenizer.decode(sequence[0])
        return decoded_sequence
