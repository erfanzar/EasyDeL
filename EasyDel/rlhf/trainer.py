import os
import pathlib
import pprint

import datasets
from fjutils import StreamingCheckpointer
from jax import numpy as jnp
from flax import linen as nn

import flax
import jax

import collections
from typing import Union, Optional, List, Dict, OrderedDict, NamedTuple, Callable, Any, Sequence, assert_type

from jax._src.maps import Mesh
from jax.experimental.mesh_utils import create_device_mesh
from transformers import PretrainedConfig
from .utils import AVAILABLE_MODELS_FOR_RLHF, AVAILABLE_MODELS_CONFIG_FOR_RLHF
from .reward import RewardModel
from .ppo import ActorCritic
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
import fjutils
import wandb


class RLHFConfig(PretrainedConfig):
    def __init__(self,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 actor_wd: float = 0.,
                 critic_wd: float = 0.,
                 actor_adam_eps: float = 1e-7,
                 critic_adam_eps: float = 1e-7,
                 critic_pooled_values=True,
                 actor_dropout: float = 0.,
                 critic_dropout: float = 0.,
                 betas=(0.9, 0.999),
                 max_norm=None,
                 eps_clip: float = 0.2,
                 value_clip: float = 0.4,
                 beta_s: float = .01,
                 pad_value: Union[float, int] = 0.,
                 minibatch_size: int = 16,
                 epochs: int = 1,
                 kl_div_loss_weight: Optional[float] = 0.100002167,
                 optimizer: str = 'adam',
                 scheduler: str = 'linear',
                 dtype: Union[str, jnp.dtype] = 'bf16',
                 param_dtype: Union[str, jnp.dtype] = 'bf16',
                 precision: Optional[Union[str, jax.lax.Precision, None]] = 'fastest',
                 sharding_array: tuple = (1, -1, 1),
                 extra_optimizer_kwargs: dict = None,
                 model_name: str = 'RLHF',
                 save_dir: str = 'easydel_ckpt',
                 backend: str = 'tpu',
                 **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.scheduler = scheduler
        self.extra_optimizer_kwargs = extra_optimizer_kwargs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_wd = actor_wd
        self.critic_wd = critic_wd
        self.optimizer = optimizer
        self.actor_adam_eps = actor_adam_eps
        self.critic_adam_eps = critic_adam_eps
        self.critic_pooled_values = critic_pooled_values
        self.actor_dropout = actor_dropout
        self.critic_dropout = critic_dropout
        self.betas = betas
        self.max_norm = max_norm
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.beta_s = beta_s
        self.epochs = epochs
        self.kl_div_loss_weight = kl_div_loss_weight
        self.minibatch_size = minibatch_size
        self.pad_value = pad_value
        self.model_name = model_name
        self.save_dir = save_dir
        self.sharding_array = jnp.ones((1, jax.devices(backend=backend))).reshape(sharding_array).shape

    def get_path(self):
        return pathlib.Path(
            self.save_dir, self.model_name
        )

    def get_meter_dict(self):

        import torch
        return {f"hyperparameters/{k}": v for k, v in self.__dict__.items() if
                isinstance(v, (int, float, str, bool, torch.Tensor))}

    def get_wandb_init(self):
        return wandb.init(
            project=f'easydel-{self.model_name}',
            config=self(),
            tags=[
                'Easy Del',
                'OST-OpenSourceTransformers',
                'Jax/Flax'
            ]
        )

    def __call__(self):
        return pprint.pformat(
            {k: str(v) for k, v in self.__dict__.items()}
        )

    def ckpt_path_exists(self):
        path = self.get_path()
        if not path.exists():
            path.mkdir(parents=True)

    def get_mesh(self):
        return Mesh(
            create_device_mesh(
                self.sharding_array
            ),
            self.get_mesh_names()
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_mesh_names():
        return 'dp', 'fsdp', 'mp'

    def get_optimizer_and_scheduler(self, steps=None):
        steps = self.max_steps or steps
        assert steps is not None, 'if you haven\'t pass max steps to init you should pass init in func'

        if self.optimizer == 'adafactor':
            if self.scheduler == 'linear':
                tx, sc = fjutils.optimizers.get_adafactor_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate_end,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'cosine':
                tx, sc = fjutils.optimizers.get_adafactor_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'none':
                tx, sc = fjutils.optimizers.get_adafactor_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'warm_up_cosine':
                tx, sc = fjutils.optimizers.get_adafactor_with_warm_up_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            else:
                raise ValueError('seems like you have choose wrong type or unavailable scheduler')
        elif self.optimizer == 'lion':
            if self.scheduler == 'linear':
                tx, sc = fjutils.optimizers.get_lion_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate_end,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'cosine':
                tx, sc = fjutils.optimizers.get_lion_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'none':
                tx, sc = fjutils.optimizers.get_lion_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'warm_up_cosine':
                tx, sc = fjutils.optimizers.get_lion_with_warm_up_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            else:
                raise ValueError('seems like you have choose wrong type or unavailable scheduler')
        elif self.optimizer == 'adamw':
            if self.scheduler == 'linear':
                tx, sc = fjutils.optimizers.get_adamw_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate_end,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'cosine':
                tx, sc = fjutils.optimizers.get_adamw_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'none':
                tx, sc = fjutils.optimizers.get_adamw_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'warm_up_cosine':
                tx, sc = fjutils.optimizers.get_adamw_with_warm_up_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            else:
                raise ValueError('seems like you have choose wrong type or unavailable scheduler')
        else:
            raise ValueError('seems like you have choose wrong type or unavailable optimizer')
        return tx, sc

    def get_streaming_checkpointer(self):
        return StreamingCheckpointer(StreamingCheckpointer.get_default_config(),
                                     os.path.join(self.save_dir, self.model_name))

    def get_board(self):
        import torch
        return torch.utils.tensorboard.SummaryWriter(
            log_dir=str(self.get_path()),
            comment=f'{self.model_name}',
            filename_suffix='easydel'
        )


class RLHFTrainer(nn.Module):
    config: RLHFConfig
    dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
    tokenizer: Callable
    model: AVAILABLE_MODELS_FOR_RLHF
    reward_model: RewardModel
    critic_model: Optional[AVAILABLE_MODELS_FOR_RLHF] = None
    actor_critic: Optional[ActorCritic] = None

    def setup(self) -> None:
        if self.actor_critic is None:
            self.actor_critic = ActorCritic(
                model=self.model,
                critic_model=self.critic_model,
                pooled_values=False,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                precision=jax.lax.Precision('fastest')

            )

    def __call__(self, *args, **kwargs):
        ...
