from . import is_jax_available

if is_jax_available():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jax.experimental.pjit import pjit


    def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
        float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

        def make_to_dtype_fn(dtype_spec):
            def to_dtype(tensor):
                if dtype_specs in float_dtypes and getattr(tensor, 'dtype', None) in float_dtypes:
                    return tensor.astype(dtype_specs)
                elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                    return tensor.astype(dtype_spec.dtype)
                return tensor

            return to_dtype

        def make_shard_fn(partition_spec, dtype_spec=None):
            jax_shard_function = pjit(
                make_to_dtype_fn(dtype_spec),
                in_shardings=None,
                out_shardings=partition_spec
            )

            def shard_fn(tensor):
                return jax_shard_function(tensor).block_until_ready()

            return shard_fn

        def make_gather_fn(partition_spec, dtype_spec=None):
            jax_gather_fn = pjit(
                make_to_dtype_fn(dtype_spec),
                in_shardings=partition_spec,
                out_shardings=None
            )

            def gather_fn(tensor):
                return jax.device_get(jax_gather_fn(tensor))

            return gather_fn

        if dtype_specs is None or dtype_specs in float_dtypes:
            shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
            gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
        else:
            shard_fns = jax.tree_util.tree_map(
                make_shard_fn, partition_specs, dtype_specs
            )
            gather_fns = jax.tree_util.tree_map(
                make_gather_fn, partition_specs, dtype_specs
            )
        return shard_fns, gather_fns
