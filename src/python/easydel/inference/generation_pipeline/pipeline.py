import dataclasses
import functools
import warnings
from typing import Optional, Union, Dict

import fjformer
import flax.core
import jax
import transformers
from jax.sharding import PartitionSpec
from ...modules.easydel_modelling_utils import EasyDeLFlaxPretrainedModel
from jax import numpy as jnp, random, lax, sharding
from fjformer import GenerateRNG
from transformers import PreTrainedTokenizer

RNG_GEN = GenerateRNG()


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng_key: random.PRNGKey
    model_kwargs: Dict[str, jnp.ndarray]

    def tree_flatten(self):
        return (
            self.cur_len,
            self.sequences,
            self.running_token,
            self.is_sent_finished,
            self.prng_key,
            self.model_kwargs
        ), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def apply_repetition_penalty(logits, tokens, penalty):
    """Applies repetition penalty more efficiently using JAX operations."""
    if penalty == 1.0:
        return logits

    token_counts = jax.ops.segment_sum(jnp.ones_like(tokens), tokens, num_segments=logits.shape[-1])
    for token in jnp.unique(tokens):
        token_penalty = jnp.where(token_counts[token] > 1, penalty, 1.0)
        logits = logits.at[token].mul(token_penalty)

    return logits


def apply_top_k_sampling(logits, top_k):
    """Applies top-k sampling to the logits."""
    indices_to_remove = logits < jnp.sort(logits)[-top_k]
    return jnp.where(indices_to_remove, jnp.inf, logits)


def apply_top_p_sampling(logits, top_p, prng_key):
    """Applies top-p (nucleus) sampling to the logits."""
    assert 0 <= top_p <= 1

    probs_sort, probs_idx = jax.lax.sort_key_val(logits, -jnp.ones_like(logits))
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort = jnp.where(mask, 0.0, probs_sort)
    probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
    next_token = jax.random.categorical(prng_key, probs_sort, axis=-1, shape=probs_sort.shape[:-1] + (1,))
    return jnp.take_along_axis(probs_idx, jnp.squeeze(next_token, axis=-1), axis=-1)


def compile_function(
        func,
        func_input_args,
        func_input_kwargs,
        mesh=None,
        in_shardings=None,
        out_shardings=None,
        static_argnums=None,
        donate_argnums=None,
):
    if mesh is None:
        return jax.jit(
            func,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums
        ).lower(*func_input_args, **func_input_kwargs).compile()
    with mesh:
        return jax.jit(
            func,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums
        ).lower(*func_input_args, **func_input_kwargs).compile()


class GenerationPipelineConfig:
    def __init__(self, **kwargs):
        self.max_new_tokens = kwargs.pop("max_new_tokens", 64)
        self.temperature = kwargs.pop("temperature", 0)
        self.top_p = kwargs.pop("top_p", 0.95)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)


class GenerationPipeline:
    def __init__(
            self,
            model: EasyDeLFlaxPretrainedModel,
            params: Union[flax.core.FrozenDict, dict],
            tokenizer: PreTrainedTokenizer,
            generation_config: Optional[GenerationPipelineConfig] = None,
            add_params_field=False,
            seed: Optional[int] = None,
            input_partition_spec: sharding.PartitionSpec = sharding.PartitionSpec(("dp", "fsdp")),
            partition_rules=None
    ):
        if generation_config is None:
            generation_config = GenerationPipelineConfig()

        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.add_params_field = add_params_field
        self._shard_state = None
        self.compiled_func = None
        self.over_compiled_func = None
        self._rng_gen = GenerateRNG(seed or 42)
        self.input_partition_spec = input_partition_spec
        self.mesh = self.model.config.get_mesh()
        if partition_rules is None:
            partition_rules = self.model.config.get_partition_rules(True)
        self.model_sharding = self.model.get_named_sharding(partition_rules=partition_rules)
        self.input_sharding = jax.sharding.NamedSharding(spec=input_partition_spec, mesh=self.model.mesh)
        self.empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=self.model.mesh)
        self.gen_input_sharding = jax.sharding.NamedSharding(
            spec=jax.sharding.PartitionSpec(input_partition_spec[0], None),
            mesh=self.model.mesh
        )
        self.state_sample = None
        self.state_sample_ms = None  # multi sequences
        # Ensure essential token IDs are set
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id
        if self.generation_config.eos_token_id is None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id
        if self.generation_config.bos_token_id is None:
            self.generation_config.bos_token_id = tokenizer.bos_token_id

        assert self.generation_config.pad_token_id is not None, (
            "`pad_token_id` cannot be None. "
            "(Set `tokenizer.pad_token_id = tokenizer.eos_token_id` if undefined)"
        )
        assert self.generation_config.eos_token_id is not None, (
            "`eos_token_id` cannot be None."
        )

    def generate(
            self,
            input_ids: jax.Array,
            attention_mask: Optional[jax.Array] = None,
            position_ids: Optional[jax.Array] = None,
            streamer: Optional[transformers.TextIteratorStreamer] = None
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
            position_ids = (attention_mask.cumsum(axis=-1, dtype="i4") - 1)  # Check this logic
        with mesh:
            input_ids = fjformer.with_sharding_constraint(input_ids, PartitionSpec(
                paxis.batch_axis,
                paxis.key_sequence_axis,
            ))
            attention_mask = fjformer.with_sharding_constraint(attention_mask, PartitionSpec(
                paxis.batch_axis,
                paxis.key_sequence_axis,
            ))
            position_ids = fjformer.with_sharding_constraint(position_ids, PartitionSpec(
                paxis.batch_axis,
                paxis.key_sequence_axis,
            ))
        assert position_ids.shape == attention_mask.shape, (
            "`position_ids` and `attention_mask` must have the same shape."
        )
        _model_kwargs_sharding = jax.tree_util.tree_map(
            lambda spec: jax.sharding.NamedSharding(spec=spec, mesh=self.mesh),
            fjformer.match_partition_rules(
                (
                    (
                        "position_ids", PartitionSpec(
                            paxis.batch_axis,
                            None
                        )
                    ),
                    (
                        "attention_mask", PartitionSpec(
                            paxis.batch_axis,
                            None
                        )
                    ),
                    (
                        "cached_key_scale", PartitionSpec(
                            paxis.batch_axis,
                            paxis.key_sequence_axis,
                            paxis.head_axis,
                            None
                        )
                    ),
                    (
                        "cached_value_scale", PartitionSpec(
                            paxis.batch_axis,
                            paxis.key_sequence_axis,
                            paxis.head_axis,
                            None
                        )
                    ),
                    (
                        "cached_key_minval", PartitionSpec(
                            paxis.batch_axis,
                            paxis.key_sequence_axis,
                            paxis.head_axis,
                            None
                        )
                    ),
                    (
                        "cached_value_minval", PartitionSpec(
                            paxis.batch_axis,
                            paxis.key_sequence_axis,
                            paxis.head_axis,
                            None
                        )
                    ),
                    (
                        "cached_key", PartitionSpec(
                            paxis.batch_axis,
                            paxis.key_sequence_axis,
                            paxis.head_axis,
                            None
                        )
                    ),
                    (
                        "cached_value", PartitionSpec(
                            paxis.batch_axis,
                            paxis.key_sequence_axis,
                            paxis.head_axis,
                            None
                        )
                    ),

                ),
                jax.eval_shape(lambda: self.model.prepare_inputs_for_generation(input_ids, max_length, attention_mask))
            ),
        )

        model_kwargs = jax.jit(
            lambda: self.model.prepare_inputs_for_generation(input_ids, max_length, attention_mask),
            out_shardings=_model_kwargs_sharding
        )()
        # Initial GenerationContent
        generation_state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=self._rng_gen.rng,
            model_kwargs=model_kwargs
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            all_sequence_finished = jnp.all(state.is_sent_finished)
            return ~jnp.logical_or(all_sequence_finished, state.cur_len >= max_length)

        def inference_step(logits, token, prng_key, temperature, top_p):
            if temperature > 0.:
                logits = jax.nn.softmax(logits / temperature, axis=-1)
                return apply_top_p_sampling(logits, top_p, prng_key)
            return jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1).reshape(-1)

        inference_step_compiled = jax.jit(inference_step, static_argnames=["top_p", "temperature"])

        def generation_func_body(params, state: SampleState):
            model_outputs = self.model(
                input_ids=state.running_token,
                params=params,
                add_params_field=self.add_params_field,
                return_dict=True,
                **state.model_kwargs
            )

            logits = model_outputs.logits

            next_token = inference_step_compiled(
                logits[:, -1],
                state.running_token,
                state.prng_key,
                self.generation_config.temperature,
                self.generation_config.top_p
            )

            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token[:, None]
            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.model.update_inputs_for_generation(model_outputs, state.model_kwargs)
            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                prng_key=random.split(state.prng_key, 2)[-1],
                model_kwargs=next_model_kwargs,
            )

        if input_ids.shape[1] > 1:
            # if self.state_sample_ms is None:
            #     self.state_sample_ms = compile_function(
            #         generation_func_body,
            #         (self.params, generation_state),
            #         {},
            #         mesh=self.mesh,
            #         in_shardings=(
            #             self.model_sharding,
            #             SampleState(None, self.input_sharding, self.input_sharding, None, None, None)),
            #         out_shardings=SampleState(None, self.input_sharding, self.input_sharding, None, None, None)
            #     )
            generation_state = generation_func_body(self.params, generation_state)
        if streamer is None:
            yield generation_state.running_token
        else:
            streamer.put(generation_state.running_token)
        if self.state_sample is None:
            _args_tree = jax.eval_shape(lambda: generation_state.model_kwargs)

            state_sharding = SampleState(
                self.empty_sharding,
                self.input_sharding,
                self.gen_input_sharding,
                self.empty_sharding,
                self.empty_sharding,
                _model_kwargs_sharding
            )

            @functools.partial(
                jax.jit,
                in_shardings=(SampleState(
                    self.empty_sharding,
                    self.input_sharding,
                    self.gen_input_sharding,
                    self.empty_sharding,
                    self.empty_sharding,
                    self.empty_sharding,
                ),),
                out_shardings=state_sharding
            )
            def _shard_state(st):
                return st

            self._shard_state = _shard_state
            # generation_state = _shard_state(generation_state)  # noqa
            self.state_sample = compile_function(
                generation_func_body,
                (self.params, generation_state),
                {},
                mesh=self.mesh,
                in_shardings=(
                    self.model_sharding,
                    state_sharding,
                ),
                out_shardings=state_sharding
            )
        # else:
        #     generation_state = self._shard_state(generation_state)  # noqa
        while sample_search_cond_fn(generation_state):
            generation_state = self.state_sample(self.params, generation_state)

            if streamer is None:
                yield generation_state.running_token
            else:
                streamer.put(generation_state.running_token)
        if streamer is not None:
            streamer.end()
        del generation_state.model_kwargs
        del generation_state.sequences
        del generation_state.running_token
        del generation_state
