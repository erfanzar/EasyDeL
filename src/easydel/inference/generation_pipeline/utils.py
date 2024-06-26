import dataclasses
from functools import partial
from typing import Dict, Union

import jax
from fjformer import GenerateRNG
from jax import numpy as jnp
from jax import random, sharding

RNG_GEN = GenerateRNG()


class GenerationPipelineConfig:
    def __init__(self, **kwargs):
        self.max_new_tokens = kwargs.pop("max_new_tokens", 64)

        self.temperature = kwargs.pop("temperature", 0)
        self.top_p = kwargs.pop("top_p", 0.95)
        self.top_k = kwargs.pop("top_k", 50)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)

        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)


class _DynamicGenerationConfig:
    def __init__(self, config):
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.repetition_penalty = config.repetition_penalty
        self.length_penalty = config.length_penalty


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
        return (
            jax.jit(
                func,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                static_argnums=static_argnums,
                donate_argnums=donate_argnums,
            )
            .lower(*func_input_args, **func_input_kwargs)
            .compile()
        )
    with mesh:
        return (
            jax.jit(
                func,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                static_argnums=static_argnums,
                donate_argnums=donate_argnums,
            )
            .lower(*func_input_args, **func_input_kwargs)
            .compile()
        )


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SampleState:
    cur_len: Union[jax.Array, sharding.NamedSharding]
    sequences: Union[jax.Array, sharding.NamedSharding]
    running_token: Union[jax.Array, sharding.NamedSharding]
    is_sent_finished: Union[jax.Array, sharding.NamedSharding]
    prng_key: Union[random.PRNGKey, sharding.NamedSharding]
    model_kwargs: Union[Dict[str, jax.Array], sharding.NamedSharding]

    def tree_flatten(self):
        return (
            self.cur_len,
            self.sequences,
            self.running_token,
            self.is_sent_finished,
            self.prng_key,
            self.model_kwargs,
        ), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def apply_repetition_penalty(logits, tokens, penalty):
    """Applies repetition penalty efficiently using JAX operations."""

    # Create a mask for the tokens that appear in the input
    vocab_size = logits.shape[-1]
    token_mask = jnp.zeros(vocab_size, dtype=jnp.bool_)
    token_mask = token_mask.at[tokens].set(True)

    # Apply the penalty
    logits = jnp.where(token_mask, logits / penalty, logits * penalty)

    return logits


def apply_length_penalty(logits, cur_len, max_len, length_penalty):
    """Applies length penalty to logits."""

    # Calculate the penalty factor
    penalty_factor = ((5 + cur_len) / 6) ** length_penalty

    # Apply the penalty
    return logits / penalty_factor


@partial(jax.jit, static_argnames=["top_k", "temperature"])
def apply_top_k_sampling(logits, temperature, top_k):
    """Applies top-k sampling to the logits."""
    logits, token_ids = jax.lax.top_k(logits, top_k)  # along the last axis
    return logits / temperature, token_ids


def apply_top_p_sampling(
    logits,
    *,
    token_ids=None,
    temperature: float = 1.0,
    top_p: float,
):
    if token_ids is None:
        token_ids = jnp.argsort(-logits, axis=-1)
        logits = jnp.take_along_axis(logits, token_ids, axis=-1)

    cutoff_index = jnp.sum(
        jnp.cumsum(jax.nn.softmax(logits, axis=-1), axis=-1) < top_p, axis=-1
    ).reshape(-1, 1)
    logits = (
        jnp.where(
            jnp.arange(logits.shape[-1])[None, :] <= cutoff_index,
            logits,
            -jnp.inf,
        )
        / temperature
    )

    return logits, token_ids


def sampling(sampling_logits, tokens_ids, key):
    sampling_index = jax.random.categorical(
        key, jax.nn.softmax(sampling_logits)
    ).reshape(-1, 1)
    selected_token = jnp.take_along_axis(tokens_ids, sampling_index, axis=-1)
    return selected_token.reshape(-1)


def temperature_branch(logits, prng_key, top_k, temperature, top_p):
    token_ids = None
    if top_k > 1:
        logits, token_ids = apply_top_k_sampling(
            logits=logits, top_k=top_k, temperature=temperature
        )
        temperature = 1
    if 0 < top_p < 1.0:
        logits, token_ids = apply_top_p_sampling(
            logits=logits, token_ids=token_ids, temperature=temperature, top_p=top_p
        )
        temperature = 1
    return sampling(logits / temperature, token_ids=token_ids, key=prng_key)


def gready_branch(logits):
    return jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1).reshape(-1)


def inference_step(
    logits,
    tokens,
    prng_key,
    config,
    cur_len,
    max_length,
):
    length_penalty = config.length_penalty
    repetition_penalty = config.repetition_penalty
    # Apply repetition penalty
    logits = jax.lax.cond(
        repetition_penalty == 1.0,
        apply_repetition_penalty,
        lambda x, *u: x,
        logits,
        tokens,
        repetition_penalty,
    )

    # Apply length penalty
    logits = jax.lax.cond(
        length_penalty == 1.0,
        apply_length_penalty,
        lambda x, *u: x,
        logits,
        cur_len,
        max_length,
        length_penalty,
    )

    if config.temperature > 0.0:
        return temperature_branch(
            logits=logits,
            prng_key=prng_key,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
        )
    return gready_branch(logits=logits)


inference_step_compiled = jax.jit(
    inference_step, static_argnames=["max_length", "config"]
)
