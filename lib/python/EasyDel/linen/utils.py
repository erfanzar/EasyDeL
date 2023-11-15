import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict


def to_8bit(tree: dict):
    """
    takes a pytree(unflatten) and convert that into 8bit array pytree
    """
    tree = flatten_dict(tree)

    for key, array in tree.items():
        scale_factor = 127 / jnp.max(array)
        array = (array * scale_factor).astype(jnp.int8)
        tree[key] = array
    return unflatten_dict(tree)


def from_8bit(tree: dict, dtype: jnp.dtype = jnp.float32):
    """
    takes a pytree(unflatten) and convert that into original pytree
    """
    tree = flatten_dict(tree)

    for key, array in tree.items():
        scale_factor = jnp.max(array)
        array = (array / scale_factor).astype(dtype)
        tree[key] = array
    return unflatten_dict(tree)


def array_to_bit8(array: jax.Array):
    return (array * (127 / jnp.max(array))).astype(jnp.int8)


def array_from_8bit(array: jax.Array, dtype: jnp.dtype = jnp.float32):
    return (array / jnp.max(array)).astype(dtype)
