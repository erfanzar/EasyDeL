"""Bounded decode expansion of channel scales without repeat's scatter map."""

import jax.numpy as jnp


def expand_group_scales(scales, sizes, count):
    """Match repeat's dtype, empty-group handling and final-element padding.

    Only bounded decode shapes use the M-by-E comparison. Larger shapes retain
    the existing repeat implementation rather than allocating a large matrix.
    Group sizes obey the caller's nonnegative-size contract.
    """
    values = scales[:, 0, :]
    if count <= 128 and 0 < sizes.shape[0] <= 1024:
        ends = jnp.cumsum(sizes)
        ids = jnp.sum(jnp.arange(count)[:, None] >= ends[None, :], axis=1)
        return values[jnp.minimum(ids, sizes.shape[0] - 1)]
    return jnp.repeat(values, sizes, axis=0, total_repeat_length=count)
