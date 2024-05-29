import dataclasses
import warnings
from typing import Optional, Union, Dict

import flax.core
import jax

from src.python.easydel.modules.easydel_modelling_utils import EasyDeLFlaxPretrainedModel
from jax import numpy as jnp, random, lax
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


def apply_top_p_sampling(logits, top_p):
    """Applies top-p (nucleus) sampling to the logits."""
    sorted_logits = jnp.sort(logits)
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1)
    sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)

    indices_to_remove = sorted_indices_to_remove[jnp.argsort(logits)]
    return jnp.where(indices_to_remove, jnp.inf, logits)


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
        self.max_time = kwargs.pop("max_time", None)  # Not implemented yet
        self.stop_strings = kwargs.pop("stop_strings", None)  # Not implemented yet
        self.do_sample = kwargs.pop("do_sample", False)  # Add this for potential future use
        self.num_beams = kwargs.pop("num_beams", 1)  # Add for beam search (not implemented)
        self.num_beam_groups = kwargs.pop(
            "num_beam_groups", 1
        )  # Add for beam search (not implemented)
        self.penalty_alpha = kwargs.pop(
            "penalty_alpha", None
        )  # Add for contrastive search (not implemented)
        self.use_cache = kwargs.pop("use_cache", True)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)  # Add this for potential future use
        self.typical_p = kwargs.pop("typical_p", 1.0)  # Add for potential future use
        self.epsilon_cutoff = kwargs.pop(
            "epsilon_cutoff", 0.0
        )  # Add for potential future use
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)  # Add for potential future use
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
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
    ):
        if generation_config is None:
            generation_config = GenerationPipelineConfig()

        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.add_params_field = add_params_field
        self.compiled_func = None
        self.over_compiled_func = None
        self._rng_gen = GenerateRNG(seed or 42)
        self.mesh = self.model.config.jax_mesh()

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
            stream: bool = True
    ):

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
        assert position_ids.shape == attention_mask.shape, (
            "`position_ids` and `attention_mask` must have the same shape."
        )
        model_kwargs = self.model.prepare_inputs_for_generation(input_ids, max_length, attention_mask)
        # Initial GenerationContent
        generation_state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=self._rng_gen.RNG_GEN,
            model_kwargs=model_kwargs
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            all_sequence_finished = jnp.all(state.is_sent_finished)
            return ~jnp.logical_or(all_sequence_finished, state.cur_len >= max_length)

        def inference_step(logits, token, prng_key):
            if self.generation_config.do_sample:
                logits = logits / self.generation_config.temperature
                logits = apply_repetition_penalty(logits, token, self.generation_config.repetition_penalty)
                if self.generation_config.top_k > 0:
                    logits = apply_top_k_sampling(logits, self.generation_config.top_k)
                if self.generation_config.top_p < 1.0:
                    logits = apply_top_p_sampling(logits, self.generation_config.top_p)
                probs = jax.nn.softmax(logits, axis=-1)
                return jax.random.categorical(prng_key, jnp.log(probs), axis=-1).reshape(-1)

            return jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1).reshape(-1)

        def generation_func_body(state: SampleState):
            model_outputs = self.model(
                input_ids=state.running_token,
                params=self.params,
                add_params_field=self.add_params_field,
                return_dict=True,
                **state.model_kwargs
            )

            logits = model_outputs.logits

            next_token = inference_step(logits[:, -1], state.running_token, state.prng_key)

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
            if self.over_compiled_func is None:
                self.over_compiled_func = compile_function(
                    generation_func_body,
                    (generation_state,),
                    {},
                    mesh=self.mesh
                )
            generation_state = self.over_compiled_func(generation_state)
        if stream:
            if self.compiled_func is None:
                self.compiled_func = compile_function(
                    generation_func_body,
                    (generation_state,),
                    {},
                    mesh=self.mesh
                )
            yield generation_state.running_token
            while sample_search_cond_fn(generation_state):
                generation_state = self.compiled_func(generation_state)
                yield generation_state.running_token
        else:
            generation_state = lax.while_loop(sample_search_cond_fn, generation_func_body, generation_state)
            return generation_state.sequences
