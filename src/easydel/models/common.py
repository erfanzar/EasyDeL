from flax import nnx
from jax import lax
from jax import numpy as jnp


class RMSNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.eps = eps
        self.dtype = dtype
        self.kernel = nnx.Param(
            nnx.initializers.ones(
                rngs.params(),
                (dim,),
                param_dtype,
            ),
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = self.kernel.value.astype(self.dtype)
        return output * weight
