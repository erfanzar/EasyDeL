import chex
import jax.scipy.special
from jax import grad, jit, numpy as jnp, lax


class PPOTrainer:
    def __init__(self):
        ...

    def step(self):
        ...

    def generate(self):
        ...

    def prepare_model_inputs(self, queries: chex.Array, responses: chex.Array):
        ...

    def compute_rewards(self):
        ...

    @staticmethod
    def _kl_penalty(kl_penalty: str, logprobs: chex.Array, ref_logprobs: chex.Array) -> chex.Array:
        if kl_penalty == "kl":
            return logprobs - ref_logprobs

        if kl_penalty == "abs":
            return (logprobs - ref_logprobs).abs()

        if kl_penalty == "mse":
            return 0.5 * (logprobs - ref_logprobs).square()

        if kl_penalty == "full":
            return jnp.sum(
                jax.scipy.special.kl_div(
                    ref_logprobs,
                    logprobs,
                ),
                axis=-1
            )

    def compute_advantages(self, values: chex.Array, rewards: chex.Array, mask: chex.Array):
        ...

    def loss(self):
        ...
