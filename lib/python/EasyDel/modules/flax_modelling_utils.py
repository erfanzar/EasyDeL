from jax.interpreters import pxla
from jax.experimental.pjit import with_sharding_constraint as wsc
import jax
from flax import linen as nn
from functools import partial
import chex

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),

}


def get_names_from_partition_spec(partition_specs):
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))

    return list(names)


def names_in_mesh(*names):
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def with_sharding_constraint(x, partition_specs):
    axis_names = get_names_from_partition_spec(partition_specs)
    if names_in_mesh(*axis_names):
        x = wsc(x, partition_specs)
    return x


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


def repeat_kv_bnsh(x: chex.Array, n_rep: int) -> chex.Array:
    bs, n_kv_heads, s, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, jax.numpy.newaxis, :, :]
    x = jax.numpy.repeat(x, n_rep, axis=2)

    return x.reshape(bs, n_kv_heads * n_rep, s, head_dim)


def repeat_kv_bsnh(x: chex.Array, n_rep: int) -> chex.Array:
    bs, s, n_kv_heads, head_dim = x.shape
    x = x.transpose(0, 2, 1, 3)
    if n_rep == 1:
        return x
    x = x[:, :, jax.numpy.newaxis, :, :]
    x = jax.numpy.repeat(x, n_rep, axis=2)

    x = x.transpose(0, 2, 1, 3)

    return x.reshape(bs, s, n_kv_heads * n_rep, head_dim)


def precompute_freq_cis(max_position_embedding, head_dim):
    inv_freq = 1.0 / (10000 ** (jax.numpy.arange(0, head_dim, 2, dtype=jax.numpy.float32) / head_dim))
    freq = jax.numpy.einsum("i , j -> i j", jax.numpy.arange(max_position_embedding), inv_freq).astype("float32")

    embed = jax.numpy.concatenate((freq, freq), axis=-1)
    return jax.numpy.sin(embed)[:, :], jax.numpy.cos(embed)[:, :]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jax.numpy.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(tensor, sin_, cos_):
    return (tensor * cos_) + (rotate_half(tensor) * sin_)
