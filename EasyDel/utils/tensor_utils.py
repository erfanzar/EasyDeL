import jax
import torch
from jax import numpy as jnp
import numpy as np


def pt2np(array: torch.Tensor) -> np.array:
    """
        Convert Pytorch Array to Numpy Array
        """
    return array.detach().cpu().numpy()


def np2jax(array: np.array) -> jax.Array:
    """
        Convert Numpy Array to JAX Array
        """
    return jnp.asarray(array)


def pt2jax(array: torch.Tensor) -> jax.Array:
    """
    Convert Pytorch Array to JAX Array
    """
    return np2jax(pt2np(array))
