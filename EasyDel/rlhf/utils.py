import jax
from jax import numpy as jnp
from jax import lax, random
from einops import rearrange
from typing import Union, Optional, List

from EasyDel import FlaxMptForCausalLM, FlaxLlamaForCausalLM, MptConfig, LlamaConfig


# Converted from Pytorch To jax from LudicRrain Guy :)

def masked_mean(prediction, attention_mask=None, axis=1, keepdims=False):
    if attention_mask is None:
        pool = jnp.mean(prediction, axis=axis, keepdims=keepdims)

    else:
        if prediction.ndim == 3:
            attention_mask = rearrange(attention_mask, 'b n -> b n 1')

        masked_seq = lax.select(
            attention_mask,
            prediction,
            0.

        )
        numer = jnp.sum(masked_seq, axis=axis, keepdims=keepdims)
        denom = jnp.sum(attention_mask, axis=axis, keepdim=keepdims)

        pool = numer / jnp.clip(denom, a_min=1e-3)
        pool = lax.select(
            denom == 0,
            0.,
            pool
        )
    return pool


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def masked_normalize(array, eps=1e-5, attention_mask=None, axis=None):
    axis = default(axis, tuple(range(array.ndim)))
    kwargs = dict(axis=axis, keepdim=True)

    mean = masked_mean(array, attention_mask=attention_mask, **kwargs)
    mean_centered = array - mean
    var = masked_mean(mean_centered ** 2, attention_mask=attention_mask, **kwargs)

    return mean_centered * lax.rsqrt(var.clip(min=eps))


# def pad_sequence_fixed(sequences, *args, **kwargs):
#     first_el = sequences[0]
#     has_no_dimension = first_el.ndim == 0
#
#     # if no dimensions, add a single dimension
#     if has_no_dimension:
#         sequences = tuple(map(lambda array: array[None], sequences))
#
#     out = pad_sequence(sequences, *args, **kwargs)
#
#     if has_no_dimension:
#         out = rearrange(out, '... 1 -> ...')
#
#     return out


def log(array, eps=1e-20):
    return lax.log(array.clip(min=eps))


def log_prob(prob, indices):
    assert prob.shape[
           :2] == indices.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
    return log(prob.gather(-1, indices[..., None])).squeeze(-1)


def shift(array, value=0, shift=1, axis=-1):
    zeros = (0, 0) * (-axis - 1)
    return jnp.pad(array, (*zeros, shift, -shift), value=value)


def masked_entropy(prob, axis=-1, attention_mask=None):
    entropies = (prob * log(prob)).sum(axis=axis)
    return masked_mean(entropies, attention_mask=attention_mask).mean()


def masked_kl_div(prob1, prob2, attention_mask=None, reduce_batch=False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(axis=-1)
    loss = masked_mean(kl_divs, attention_mask)

    if reduce_batch:
        return loss.mean()

    return loss


def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clip(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return jnp.mean(jnp.max(value_loss_1, value_loss_2))




AVAILABLE_MODELS_FOR_RLHF = Union[
    FlaxLlamaForCausalLM, FlaxMptForCausalLM
]
AVAILABLE_MODELS_CONFIG_FOR_RLHF = Union[
    LlamaConfig, MptConfig
]
