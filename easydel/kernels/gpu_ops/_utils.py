import typing as tp

import chex
import numpy
from jax import numpy as jnp

F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


def safe_autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
) -> tp.Callable[[F], F]:
    """
    Applies `triton.autotune` safely. Falls back to the original function if autotuning fails.
    """
    try:
        from triton.runtime.autotuner import Autotuner

        def decorator(fn):
            try:
                return Autotuner(
                    fn,
                    fn.arg_names,
                    configs,
                    key,
                    reset_to_zero,
                    restore_value,
                    pre_hook=pre_hook,
                    post_hook=post_hook,
                    prune_configs_by=prune_configs_by,
                    warmup=warmup,
                    rep=rep,
                    use_cuda_graph=use_cuda_graph,
                )
            except Exception:
                return fn

        return decorator
    except (Exception, RuntimeError) as err:
        print(f"Couldn't autotune given function due to {err}")

        def decorator(fn):
            return fn

        return decorator


def dtype_index(x: jnp.array) -> int:
    if x.dtype == jnp.float16:
        return 1
    if x.dtype == jnp.bfloat16:
        return 2
    if x.dtype == jnp.float32:
        return 3
    raise ValueError(x.dtype)


def get_sharding(arr: chex.Array):
    """Gets the sharding of an array.

    Args:
            arr: Array to get sharding from.

    Returns:
            Sharding of the array.
    """
    return getattr(arr, "sharding", None)


def get_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Calculates strides for a given shape.

    Args:
            shape: Shape of the array.

    Returns:
            Tuple of strides.
    """
    if hasattr(shape, "shape"):
        shape = shape.shape
    size = numpy.prod(shape)
    strides = []
    for s in shape:
        size = int(size // s)
        strides.append(size)
    return tuple(strides)
