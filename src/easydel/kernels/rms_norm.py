import functools

import jax.experimental.pallas as pl
import jax.random

from easydel.kernels.utils import calculate_settings
from easydel.modules.common import RMSNorm
from jax import numpy as jnp


# x * (1 / sqrt(sum(x*x)/cols + eps))
def _rms_norm_forward_kernel(
        w_ref,
        x_ref,
        o_ref,
        r_ref,
        *,
        eps,
        dim,
):
    org_dtype = x_ref[...].dtype
    X_R = x_ref[...].astype(jnp.float32)  # make sure it's always in Float32
    inv_var = jax.lax.rsqrt((jnp.sum(X_R * X_R, -1, keepdims=True) / dim) + eps)
    r_ref[...] = inv_var
    normed = (X_R * inv_var).astype(w_ref.dtype)
    o_ref[...] = (w_ref[...] * normed).astype(org_dtype)


def _rms_norm_forward_main(
        W: jax.Array,
        X: jax.Array,
        eps: float,
        interpret: bool = False
):
    B, S, DIM = X.shape
    W = W.reshape(1, -1)
    X = X.reshape(-1, DIM)
    block_size, num_wraps = calculate_settings(DIM, X.dtype)

    out_shape = [
        jax.ShapeDtypeStruct(shape=X.shape, dtype=X.dtype, sharding=X.sharding),
        jax.ShapeDtypeStruct(shape=(S, 1), dtype=jnp.float32)
    ]
    in_specs = [
        pl.BlockSpec(lambda i: (0, 0), (1, DIM,)),
        pl.BlockSpec(lambda i: (i, 0), (S * B, DIM)),
    ]
    out_specs = [
        pl.BlockSpec(lambda i: (i, 0), (S * B, DIM)),
        pl.BlockSpec(lambda i: (0, 0), (S, 1)),
    ]
    method = pl.pallas_call(
        functools.partial(
            _rms_norm_forward_kernel,
            eps=eps,
            dim=DIM,
        ),
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,  # type:ignore
        grid=(block_size,),
        interpret=interpret,
        name="rms_forward_main",
        debug=True
    )
    result, R = method(W, X)
    return result.reshape(B, S, DIM)


if __name__ == "__main__":
    inputs = jax.random.normal(jax.random.key(564), (1, 256, 64), dtype=jnp.float16)
    print(inputs.devices())
    norm = RMSNorm(
        64,
        1e-6,
        jnp.float16,
        jnp.float16
    )
    params = norm.init(jax.random.PRNGKey(0), inputs)

    out = norm.apply(params, inputs)

    norm_kernel_out = _rms_norm_forward_main(params["params"]["kernel"], inputs, 1e-6, interpret=True)
    print(jnp.allclose(out, norm_kernel_out))
