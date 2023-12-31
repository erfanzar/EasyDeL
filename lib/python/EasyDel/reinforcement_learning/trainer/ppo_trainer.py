import chex
import jax.scipy.special
from typing import Optional
from jax import grad, jit, numpy as jnp, lax
from ..models import AutoRLModelForCasualLMWithValueHead
from .ppo_config import PPOConfig
from transformers import PreTrainedTokenizerBase
from ...trainer.utils import get_optimizer_and_scheduler
from termcolor import colored


class PPOTrainer:
    def __init__(
            self,
            config: PPOConfig,
            model: AutoRLModelForCasualLMWithValueHead,
            model_ref: AutoRLModelForCasualLMWithValueHead | None = None,
            tokenizer: PreTrainedTokenizerBase = None,
            scheduler: str | None = "none",
            optimizer: str = "adamw",
            steps: Optional[int] = None
    ):
        if steps is None and config.steps is None:
            print(colored("Warning", color="red") + " : " + "If you don't set steps in PPOConfig or PPOTrainer"
                                                            "steps will be set as 1_000_000 by default")
            steps = 1_000_000
        self.config = config
        self.model = model
        self.model_ref = model_ref
        self.tokenizer = tokenizer
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(
            scheduler=scheduler,
            optimizer=optimizer,
            weight_decay=config.weight_decay,
            extra_optimizer_kwargs=config.extra_optimizer_kwargs,
            warmup_steps=config.warmup_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            learning_rate_end=config.learning_rate_end,
            steps=config.steps or steps
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

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
        elif kl_penalty == "abs":
            return (
                    logprobs - ref_logprobs
            ).abs()
        elif kl_penalty == "mse":
            return 0.5 * (
                    logprobs - ref_logprobs
            ).square()
        elif kl_penalty == "full":
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
