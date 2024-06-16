import functools

import jax.experimental.pallas as pl
import jax.random

from easydel.kernels.utils import calculate_settings
from easydel.modules.common import RMSNorm
from jax import numpy as jnp


# x * (1 / sqrt(sum(x*x)/cols + eps)) * w
def _rms_norm_forward_kernel(
        w_ref,
        x_ref,
        o_ref,
        r_ref,
        *,
        eps,
        dim,
        exp_tun
):
    org_dtype = x_ref[...].dtype
    X_R = x_ref[...].astype(jnp.float32)  # make sure it's always in Float32

    if exp_tun:
        row_var = jnp.sum(X_R * X_R, axis=0) / dim
        inv_var = jax.lax.rsqrt(row_var + eps)
        r_ref[...] = inv_var
        normed = X_R * inv_var
        o_ref[...] = (normed * (w_ref[...] + 1.0)).astype(org_dtype)
    else:
        inv_var = jax.lax.rsqrt((jnp.sum(X_R * X_R, -1, keepdims=True) / dim) + eps)
        r_ref[...] = inv_var
        normed = (X_R * inv_var).astype(w_ref.dtype)
        o_ref[...] = (w_ref[...] * normed).astype(org_dtype)


def _rms_norm_backward_kernel(
        g_ref,
        x_ref,
        w_ref,
        r_ref,
        dy_ref,
        *,
        eps,
        dim,
        exp_tun
):
    ...


def _rms_norm_backward_main(
        eps: float,
        interpret: bool,
        exp_tun: bool,
        res,
        g,
):
    X, R, W = res
    B, S, DIM = g.shape
    g = g.reshape(-1, DIM)
    block_size, num_wraps = calculate_settings(DIM, X.dtype)

    out_shape = jax.ShapeDtypeStruct(shape=X.shape, dtype=X.dtype, sharding=X.sharding)
    in_specs = [
        pl.BlockSpec(lambda i: (i, 0), (S * B, DIM)),
        pl.BlockSpec(lambda i: (i, 0), (S * B, DIM)),
        pl.BlockSpec(lambda i: (0, 0), (1, DIM,)),
        pl.BlockSpec(lambda i: (0, 0), (S, 1)),
    ]
    out_specs = pl.BlockSpec(lambda i: (i, 0), (S * B, DIM))
    method = pl.pallas_call(
        functools.partial(
            _rms_norm_backward_kernel,
            eps=eps,
            dim=DIM,
            exp_tun=exp_tun,
        ),
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,  # type:ignore
        grid=(block_size,),
        interpret=interpret,
        name="rms_backward_main",
        debug=True
    )

    # g_ref,
    # x_ref,
    # w_ref,
    # r_ref,
    # dy_ref,

    g_y = method(
        g,
        X,
        W,
        R,
    )

    return g_y.reshape(B, S, DIM), None


def _rms_norm_forward_main(
        X: jax.Array,
        W: jax.Array,
        eps: float,
        interpret: bool = False,
        exp_tun: bool = False
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
            exp_tun=exp_tun,
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
    return result.reshape(B, S, DIM), (X, R, W)


@functools.partial(jax.custom_vjp, nondiff_argnums=[2, 3, 4])
def rms_norm(
        X: jax.Array,
        W: jax.Array,
        eps: float,
        interpret: bool = False,
        exp_tun: bool = False
):
    return _rms_norm_forward_main(
        W=W,
        X=X,
        eps=eps,
        interpret=interpret,
        exp_tun=exp_tun
    )[0]


rms_norm.defvjp(_rms_norm_forward_main, _rms_norm_backward_main)
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

    out, grad_org = jax.value_and_grad(lambda p: jnp.mean(norm.apply(p, inputs)))(params)

    norm_kernel_out, grad = jax.value_and_grad(
        lambda w: jnp.mean(
            rms_norm(
                X=inputs,
                W=w,
                eps=1e-6,
                interpret=True
            )
        )
    )(
        params["params"]["kernel"],
    )
    print(jnp.allclose(out, norm_kernel_out))
    print(grad)
    print(grad_org["params"]["kernel"])
