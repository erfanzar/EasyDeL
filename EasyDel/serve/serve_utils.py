
from jax import numpy as jnp

import jax
from flax.core import freeze
from jax.experimental import mesh_utils
from fjutils import utils
from flax.traverse_util import unflatten_dict
from fjutils import easylm

dtypes = {
    'fp16': jnp.float16,
    'bf16': jnp.bfloat16,
    'fp32': jnp.float32,
    'fp64': jnp.float64,

}


def get_dtype(dtype):
    if isinstance(dtype, str):
        dtype = dtypes[dtype]
    return dtype


read_ckpt = utils.read_ckpt
create_shard_gather_fns = easylm.make_shard_and_gather_fns
match_partition_rules = easylm.match_partition_rules
with_sharding_constraint = easylm.with_sharding_constraint
get_jax_mesh = easylm.get_jax_mesh


def shard_params(params, partition_rules,
                 shard_mesh_shape=(1, -1, 1),
                 backend='gpu',
                 shard_mesh=('dp', 'fsdp', 'mp'), do_unf=True,
                 dtype='fp16'):
    dtype = get_dtype(dtype)
    params = unflatten_dict(params) if do_unf else params
    params = freeze(params)
    mxd = jax.device_count(backend)
    rsp = jnp.asarray([1, mxd, 1]).reshape(shard_mesh_shape)
    phs_mesh = mesh_utils.create_device_mesh(rsp.tolist(), )
    mesh = jax.sharding.Mesh(phs_mesh, shard_mesh)
    ps = match_partition_rules(
        partition_rules,
        params
    )
    with mesh:
        shard_fns, _ = create_shard_gather_fns(
            ps, dtype
        )
        params = jax.tree_util.tree_map(lambda fn, x: fn(x), shard_fns, params)
    return params, mesh
