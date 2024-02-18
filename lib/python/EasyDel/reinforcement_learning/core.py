import random
from contextlib import contextmanager

import jax
import numpy as np
import jax.numpy as jnp
from transformers import top_k_top_p_filtering

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

WANDB_PADDING = -1


def u_flatten_dict(nested, sep="/"):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def convert_to_scalar(stats):
    tensorboard_stats = {}
    for k, v in stats.items():
        if (isinstance(v, jax.Array) or isinstance(v, np.ndarray)) and (
                len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1)
        ):
            v = v.item()
        tensorboard_stats[k] = v
    return tensorboard_stats


def pad_sequence(
        sequences,
        batch_first=False,
        padding_value=0,
        max_len: int | None = None
):
    max_len = max(seq.shape[-1] for seq in sequences) if max_len is None else max_len
    padding_value = jnp.array(padding_value).reshape(1)
    if batch_first:
        padded_seqs = [
            jnp.concatenate(
                [
                    seq.reshape(1, -1),
                    jnp.ones((1, max_len - seq.shape[-1])) * padding_value
                ],
                axis=1
            ) if seq.shape[-1] < max_len else seq.reshape(1, -1)
            for seq in sequences
        ]
    else:
        padded_seqs = [
            jnp.concatenate(
                [
                    jnp.ones((1, max_len - seq.shape[-1])) * padding_value,
                    seq.reshape(1, -1)
                ],
                axis=1
            ) if seq.shape[-1] < max_len else seq.reshape(1, -1)
            for seq in sequences
        ]

    return jnp.array(padded_seqs)


def pad(input_, pad_, mode='constant', value=0):
    if mode != 'constant':
        raise NotImplementedError("Only 'constant' padding mode is supported in this reimplementation.")

    pad_width = [(0, 0) if p == 0 else (p, p) for p in pad_]
    return jnp.pad(input_, pad_width, mode='constant', constant_values=value)


def stack_dicts(stats_dicts):
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [jax.flatten(d[k]) for d in stats_dicts]
        results[k] = pad_sequence(stats_list, batch_first=True, padding_value=WANDB_PADDING)
    return results


def add_suffix(input_dict, suffix):
    """Add suffix to dict keys."""
    return dict((k + suffix, v) for k, v in input_dict.items())


def pad_to_size(tensor, size, dim=1, padding=50256):
    t_size = tensor.shape[dim]
    if t_size == size:
        return tensor
    else:
        return pad(tensor, (0, size - t_size), "constant", padding)


def logprobs_from_logits(logits, labels, gather=True):
    logp = jax.nn.log_softmax(logits, dim=2)
    if not gather:
        return logp
    logpy = jnp.take_along_axis(logp, labels[:, :, None], axis=2).squeeze(-1)
    return logpy


def whiten(values, shift_mean=True):
    mean, var = jnp.mean(values), jnp.var(values)
    whitened = (values - mean) * jax.lax.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def masked_mean(values, mask, axis=None):
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * jax.lax.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    clipped = jax.max(jax.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits):
    pd = jax.nn.softmax(logits, axis=-1)
    entropy = jax.logsumexp(logits, axis=-1) - jax.sum(pd * logits, axis=-1)
    return entropy


def average_torch_dicts(list_of_dicts):
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = jnp.mean(jnp.stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict


def listify_batch(tensor):
    return [tensor[i] for i in range(tensor.shape[0])]


def build_bert_batch_from_txt(text_list, tokenizer, device):
    tensors = [tokenizer.encode(txt, return_tensors="pt").to(device) for txt in text_list]

    max_len = max([t.size()[1] for t in tensors])
    padded_tensors = []
    attention_masks = []
    for tensor in tensors:
        attention_mask = jnp.ones(tensor.size())
        padded_tensors.append(pad_to_size(tensor, max_len, padding=0))
        attention_masks.append(pad_to_size(attention_mask, max_len, padding=0))

    padded_tensors = jnp.concatenate(padded_tensors)
    attention_masks = jnp.concatenate(attention_masks)

    return padded_tensors, attention_masks


def multinomial(logits, num_samples: int, replacement: bool = False):
    """
    Implements the `torch.multinomial` function in JAX.

    Args:
        logits (jnp.array): The unnormalized log probabilities of the events.
        num_samples (int): The number of samples to draw.
        replacement (bool): Don't use this ;\

    Returns:
        jnp.array: A matrix of shape (num_samples, batch_size) containing the
            sampled indices.
    """
    logits = jax.nn.log_softmax(logits, axis=-1)
    if replacement:
        return jax.random.categorical(logits, num_samples)
    else:
        samples = []
        for _ in range(num_samples):
            sample = jax.random.categorical(logits, 1)
            samples.append(sample[0])
            logits.at[sample[0]].set(-jnp.inf)
        return jnp.array(samples)


def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    input_ids = queries
    for i in range(txt_len):
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = jax.nn.softmax(next_token_logits, axis=-1)
        next_token = multinomial(probs, num_samples=1).squeeze(1)
        input_ids = jnp.concatenate([input_ids, next_token[..., None]], axis=-1)
    return input_ids[:, -txt_len:]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class LengthSampler:

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)


class PPODecorators(object):
    optimize_device_cache = False

    @classmethod
    @contextmanager
    def empty_device_cache(cls):
        yield
