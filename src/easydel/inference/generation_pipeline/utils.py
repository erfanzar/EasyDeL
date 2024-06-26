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


@partial(jax.jit, static_argnames=["top_k"])
def apply_top_k_sampling(logits, top_k):
    """Applies top-k sampling to the logits."""
    batch_size, vocab_size = logits.shape
    next_scores_flat = jnp.full(batch_size * vocab_size, -float("Inf"))

    topk = min(top_k, logits.shape[-1])  # Safety check
    topk_scores, topk_indices = jax.lax.top_k(logits, topk)
    shift = jnp.broadcast_to(
        (jnp.arange(batch_size) * vocab_size)[:, None], (batch_size, topk)
    ).flatten()
    topk_scores_flat = topk_scores.flatten()
    topk_indices_flat = topk_indices.flatten() + shift

    next_scores_flat = next_scores_flat.at[topk_indices_flat].set(topk_scores_flat)
    return next_scores_flat.reshape(batch_size, vocab_size)


def apply_top_p_sampling(logits, top_p, prng_key):
    """Applies top-p (nucleus) sampling to the logits."""
    sorted_probs, sorted_indices = jax.lax.sort(logits, dimension=-1)

    # Calculate the cumulative sum of the sorted probabilities
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

    # Create a mask to filter out the probabilities that contribute to more than top_p
    mask = cumulative_probs <= top_p

    # Ensure at least one token is always included (the most probable one)
    mask = mask.at[-1].set(True)

    # Mask out the probabilities that are not in the top_p set
    filtered_probs = sorted_probs * mask
    filtered_probs /= jnp.sum(
        filtered_probs
    )  # Renormalize to make it a valid distribution

    # Sample from the filtered distribution using the sorted indices
    sampled_index = jax.random.categorical(prng_key, jnp.log(filtered_probs))

    # Map the sampled index back to the original indices
    result = sorted_indices[sampled_index]

    return result


def inference_step(
    logits,
    tokens,
    prng_key,
    config,
    cur_len,
    max_length,
):
    top_k = config.top_k
    top_p = config.top_p
    temperature = config.temperature
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

    def temperature_branch(logits, prng_key):
        logits = jax.nn.softmax(logits / temperature, axis=-1)
        if config.top_k > 0:
            logits = apply_top_k_sampling(
                logits=logits,
                top_k=top_k,
            )
        return jax.lax.cond(
            top_p < 1.0,
            lambda x, k, p: apply_top_p_sampling(x, p, k),
            lambda x, k, p: jax.random.categorical(k, x, -1),
            logits,
            prng_key,
            top_p,
        )

    def gready_branch(logits, prng_key):
        return jnp.argmax(
            jax.nn.softmax(logits, axis=-1),
            axis=-1,
        ).reshape(-1)

    if temperature > 0.0:
        return temperature_branch(
            logits=logits,
            prng_key=prng_key,
        )
    return gready_branch(
        logits=logits,
        prng_key=prng_key,
    )


inference_step_compiled = jax.jit(
    inference_step,
    static_argnames=["max_length", "config"],
)
