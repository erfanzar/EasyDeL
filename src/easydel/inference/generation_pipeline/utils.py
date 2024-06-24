import dataclasses
from typing import Dict
import jax
from jax import numpy as jnp, random
from fjformer import GenerateRNG

RNG_GEN = GenerateRNG()


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
            self.model_kwargs,
        ), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def apply_repetition_penalty(logits, tokens, penalty):
    """Applies repetition penalty efficiently using JAX operations."""
    if penalty == 1.0:
        return logits

    # Create a mask for the tokens that appear in the input
    vocab_size = logits.shape[-1]
    token_mask = jnp.zeros(vocab_size, dtype=jnp.bool_)
    token_mask = token_mask.at[tokens].set(True)

    # Apply the penalty
    logits = jnp.where(token_mask, logits / penalty, logits * penalty)

    return logits


def apply_length_penalty(logits, cur_len, max_len, length_penalty):
    """Applies length penalty to logits."""
    if length_penalty == 1.0:
        return logits

    # Calculate the penalty factor
    penalty_factor = ((5 + cur_len) / 6) ** length_penalty

    # Apply the penalty
    return logits / penalty_factor


def apply_top_k_sampling(logits, top_k):
    """Applies top-k sampling to the logits."""
    top_k = jnp.minimum(top_k, logits.shape[-1])  # Safety check
    values, _ = jax.lax.top_k(logits, top_k)
    min_value = values[..., -1, jnp.newaxis]
    return jnp.where(logits < min_value, -jnp.inf, logits)


def apply_top_p_sampling(logits, top_p, prng_key):
    """Applies top-p (nucleus) sampling to the logits."""
    assert 0 <= top_p <= 1

    probs_sort, probs_idx = jax.lax.sort_key_val(logits, -jnp.ones_like(logits))
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort = jnp.where(mask, 0.0, probs_sort)
    probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
    next_token = jax.random.categorical(
        prng_key, probs_sort, axis=-1, shape=probs_sort.shape[:-1] + (1,)
    )
    return jnp.take_along_axis(probs_idx, jnp.squeeze(next_token, axis=-1), axis=-1)


def inference_step(
    logits,
    tokens,
    prng_key,
    config,
    cur_len,
    max_length,
):
    # Apply repetition penalty
    logits = apply_repetition_penalty(logits, tokens, config.repetition_penalty)

    # Apply length penalty
    logits = apply_length_penalty(logits, cur_len, max_length, config.length_penalty)
    if config.temperature > 0.0:
        logits = jax.nn.softmax(logits / config.temperature, axis=-1)
        if config.top_k > 0:
            logits = apply_top_k_sampling(logits, config.top_k)

        if config.top_p < 1.0:
            return apply_top_p_sampling(logits, config.top_p, prng_key)
        else:
            return jax.random.categorical(prng_key, logits, axis=-1)
    return jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1).reshape(-1)


inference_step_compiled = jax.jit(
    inference_step,
    static_argnames=["max_length", "config"],
)
