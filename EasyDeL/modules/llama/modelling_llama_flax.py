from jax import jit, vmap, pmap
from flax import linen as nn
from jax import numpy as jnp


def compute_freq(dim: int, man_length: int, theta: int = 10000):
    freq = 1 / (theta ** (jnp.arange(0, dim, 2)))
    t = jnp.arange(man_length)
    m = jnp.einsum('i,j->ij', t, freq)
    cos = jnp.cos(m)
    sin = jnp.sin(m)
    return cos, sin


def rotate_half(tensor):
    depth = tensor.shape[-1]
    x1 = tensor[..., :depth]
    x2 = tensor[..., depth:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_embedding(q, k, c, s):
    q = (q * c) + (rotate_half(q) * s)
    k = (k * c) + (rotate_half(k) * s)
    return q, k


if __name__ == "__main__":
    co, si = compute_freq(64, 1024)
