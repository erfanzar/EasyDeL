from flax import linen as nn
from jax import lax
from jax import numpy as jnp


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            "kernel",
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)

        weight = self.weight.astype(self.dtype)
        return weight * output


class LayerNormRaw(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Applies layer normalization to the input.

        Args:
            hidden_states: Input tensor.

        Returns:
            Normalized tensor.
        """
        orig_dtype = hidden_states.dtype
        hidden_states = jnp.asarray(hidden_states, jnp.float32)
        normalized_hidden_states = nn.LayerNorm(
            epsilon=self.eps, use_bias=False, use_scale=False
        )(hidden_states)
        return jnp.asarray(normalized_hidden_states, orig_dtype)
