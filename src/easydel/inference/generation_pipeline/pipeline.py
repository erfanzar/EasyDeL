import functools
import warnings
from typing import Optional, Union

import fjformer
import flax.core
import jax
from fjformer import GenerateRNG
from jax import lax, random
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from transformers import PreTrainedTokenizer

from easydel.inference.generation_pipeline import utils as inference_utils
from easydel.modules.modeling_utils import EDPretrainedModel

GenerationPipelineConfig = inference_utils.GenerationPipelineConfig
_DynamicGenerationConfig = inference_utils._DynamicGenerationConfig


class GenerationPipeline:
    def __init__(
        self,
        model: EDPretrainedModel,
        params: Union[flax.core.FrozenDict, dict],
        tokenizer: PreTrainedTokenizer,
        generation_config: Optional[GenerationPipelineConfig] = None,
        add_params_field=None,
        seed: Optional[int] = None,
        input_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp")),
        partition_rules=None,
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
        self._shard_state = None
        self.compiled_func = None
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
            spec=input_partition_spec, mesh=self.model.mesh
        )
        self.empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=self.model.mesh)
        self.gen_input_sharding = NamedSharding(
            spec=PartitionSpec(input_partition_spec[0], None),
            mesh=self.model.mesh,
        )
        self.compiled_sample_fn = None
        self.compiled_sample_fn_ms = None  # multi sequences
        # Ensure essential token IDs are set
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id
        if self.generation_config.eos_token_id is None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id
        if self.generation_config.bos_token_id is None:
            self.generation_config.bos_token_id = tokenizer.bos_token_id
        self._dynamic_config = _DynamicGenerationConfig(self.generation_config)
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
    ):
        paxis = self.model.config.partition_axis
        mesh = self.mesh
        eos_token_id = jnp.array(self.generation_config.eos_token_id, dtype=jnp.int32)
        pad_token_id = jnp.array(self.generation_config.pad_token_id, dtype=jnp.int32)
        batch_size, cur_len = input_ids.shape
        max_length = cur_len + self.generation_config.max_new_tokens
        cur_len = jnp.array(cur_len)
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

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
                PartitionSpec(paxis.batch_axis, paxis.key_sequence_axis),
            )
            attention_mask = fjformer.with_sharding_constraint(
                attention_mask,
                PartitionSpec(paxis.batch_axis, paxis.key_sequence_axis),
            )
            position_ids = fjformer.with_sharding_constraint(
                position_ids,
                PartitionSpec(paxis.batch_axis, paxis.key_sequence_axis),
            )
        assert (
            position_ids.shape == attention_mask.shape
        ), "`position_ids` and `attention_mask` must have the same shape."
        model_kwargs_sharding = self.get_model_arguments_sharding(
            input_ids, max_length, attention_mask
        )
        model_kwargs = jax.jit(
            lambda: self.model.prepare_inputs_for_generation(
                input_ids, max_length, attention_mask
            ),
            out_shardings=model_kwargs_sharding,
        )()
        # Initial GenerationContent
        generation_state = inference_utils.SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=self._rng_gen.rng,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            all_sequence_finished = jnp.all(state.is_sent_finished)
            return ~jnp.logical_or(all_sequence_finished, state.cur_len >= max_length)

        def sample_fn(params, state: inference_utils.SampleState):
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
                cur_len,
                self.generation_config.max_new_tokens,
            )

            next_token = (
                next_token * ~state.is_sent_finished
                + pad_token_id * state.is_sent_finished
            )

            next_is_sent_finished = state.is_sent_finished | next_token == eos_token_id

            next_token = next_token[:, None]
            next_sequences = lax.dynamic_update_slice(
                state.sequences, next_token, (0, state.cur_len)
            )
            next_model_kwargs = self.model.update_inputs_for_generation(
                model_outputs, state.model_kwargs
            )
            return inference_utils.SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                prng_key=random.split(state.prng_key, 2)[0],
                model_kwargs=next_model_kwargs,
            )

        with self.mesh:
            if input_ids.shape[1] > 1:
                generation_state = sample_fn(self.params, generation_state)
            while True:
                if sample_search_cond_fn(generation_state):
                    yield (
                        generation_state.sequences[
                            :, cur_len : generation_state.cur_len
                        ]
                        if echo
                        else generation_state.running_token
                    )
                    generation_state = self.compiled_sample_state(
                        sample_fn=sample_fn,
                        generation_state=generation_state,
                        model_kwargs_sharding=model_kwargs_sharding,
                    )(self.params, generation_state)
                else:
                    break
            del generation_state.model_kwargs
            del generation_state.sequences
            del generation_state.running_token
            del generation_state

    @property
    def partition_rules(self):
        paxis = self.model.config.partition_axis
        return (
            (
                "position_ids",
                PartitionSpec(paxis.batch_axis, None),
            ),
            (
                "attention_mask",
                PartitionSpec(paxis.batch_axis, None),
            ),
            (
                "cached_key_scale",
                PartitionSpec(
                    paxis.batch_axis, paxis.key_sequence_axis, paxis.head_axis, None
                ),
            ),
            (
                "cached_value_scale",
                PartitionSpec(
                    paxis.batch_axis, paxis.key_sequence_axis, paxis.head_axis, None
                ),
            ),
            (
                "cached_key_minval",
                PartitionSpec(
                    paxis.batch_axis, paxis.key_sequence_axis, paxis.head_axis, None
                ),
            ),
            (
                "cached_value_minval",
                PartitionSpec(
                    paxis.batch_axis, paxis.key_sequence_axis, paxis.head_axis, None
                ),
            ),
            (
                "cached_key",
                PartitionSpec(
                    paxis.batch_axis, paxis.key_sequence_axis, paxis.head_axis, None
                ),
            ),
            (
                "cached_value",
                PartitionSpec(
                    paxis.batch_axis, paxis.key_sequence_axis, paxis.head_axis, None
                ),
            ),
        )

    def get_model_arguments_sharding(self, input_ids, max_length, attention_mask):
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
        for key, value in kwargs.items():
            if hasattr(self._dynamic_config, key):
                setattr(self._dynamic_config, key, value)
            else:
                raise AttributeError(
                    f"DynamicGenerationConfig has no attribute '{key}'"
                )

    def _compile_sample_fn(
        self,
        model_kwargs_sharding,
        generation_state,
        sample_fn,
    ):

        state_sharding = inference_utils.SampleState(
            self.empty_sharding,
            self.input_sharding,
            self.gen_input_sharding,
            self.empty_sharding,
            self.empty_sharding,
            model_kwargs_sharding,
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
        def _shard_state(st):
            return st

        self._shard_state = _shard_state
        self.compiled_sample_fn = inference_utils.compile_function(
            sample_fn,
            (self.params, generation_state),
            {},
            mesh=self.mesh,
            in_shardings=(self.model_sharding, state_sharding),
            out_shardings=state_sharding,
        )

    def compiled_sample_state(
        self,
        model_kwargs_sharding,
        generation_state,
        sample_fn,
    ):

        if self.compiled_sample_fn is None:
            self._compile_sample_fn(
                model_kwargs_sharding,
                generation_state,
                sample_fn,
            )
        return self.compiled_sample_fn


class ChatPipeline:

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
