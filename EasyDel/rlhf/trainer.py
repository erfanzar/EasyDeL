import collections
import os
import pathlib
import pprint
import typing
from typing import Union, Optional, Callable, List

import jax
import torch
import wandb
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from fjutils import StreamingCheckpointer
from fjutils import optimizers
from jax import numpy as jnp
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from torch.utils.data.dataloader import DataLoader
from transformers import PretrainedConfig

from .ppo import ActorCritic
from .reward import RewardModel
from .utils import AVAILABLE_MODELS_FOR_RLHF, shift, log_prob, masked_entropy

Memory = collections.namedtuple('Memory', [
    'logits',
    'prompt_mask',
    'attention_mask',
    'action_prob',
    'action_log_prob',
    'reward',
    'value'
])


class ExperienceDataset(Dataset):
    def __init__(
            self,
            data: List[torch.Tensor],
            device=None
    ):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind].to(self.device), self.data))


def create_dataloader(data, batch_size, shuffle=True, device=None, **kwargs):
    ds = ExperienceDataset(data, device=device)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)


class RLHFConfig(PretrainedConfig):
    def __init__(self,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 actor_wd: float = 0.,
                 critic_wd: float = 0.,
                 actor_adam_eps: float = 1e-7,
                 gradient_accumulation_steps: int = 1,
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
        self.precision = precision if not isinstance(precision, str) else jax.lax.Precision(precision)
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_dir = save_dir
        self.sharding_array = jnp.ones((1, jax.devices(backend=backend))).reshape(sharding_array).shape

    def get_path(self):
        return pathlib.Path(
            self.save_dir, self.model_name
        )

    def get_meter_dict(self):

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

    @staticmethod
    def get_optimizer_and_scheduler(
            steps,
            optimizer: str = 'adam',
            scheduler: str = 'cosine',

            learning_rate: float = 5e-5,
            learning_rate_end: float = 6e-5,
            gradient_accumulation_steps: int = 8,
            weight_decay: float = 1e-2,
            **kwargs
    ):

        if optimizer == 'adafactor':
            if scheduler == 'linear':
                tx, sc = optimizers.get_adafactor_with_linear_scheduler(
                    learning_rate_start=learning_rate,
                    learning_rate_end=learning_rate_end,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    steps=steps,
                    **kwargs
                )
            elif scheduler == 'cosine':
                tx, sc = optimizers.get_adafactor_with_cosine_scheduler(
                    learning_rate=learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    weight_decay=weight_decay,
                    **kwargs
                )
            elif scheduler == 'none':
                tx, sc = optimizers.get_adafactor_with_linear_scheduler(
                    learning_rate_start=learning_rate,
                    learning_rate_end=learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    **kwargs
                )
            elif scheduler == 'warm_up_cosine':
                tx, sc = optimizers.get_adafactor_with_warm_up_cosine_scheduler(
                    learning_rate=learning_rate,
                    steps=steps,
                    weight_decay=weight_decay,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    **kwargs
                )
            else:
                raise ValueError('seems like you have choose wrong type or unavailable scheduler')
        elif optimizer == 'lion':
            if scheduler == 'linear':
                tx, sc = optimizers.get_lion_with_linear_scheduler(
                    learning_rate_start=learning_rate,
                    learning_rate_end=learning_rate_end,
                    steps=steps,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    **kwargs
                )
            elif scheduler == 'cosine':
                tx, sc = optimizers.get_lion_with_cosine_scheduler(
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    steps=steps,
                    **kwargs
                )
            elif scheduler == 'none':
                tx, sc = optimizers.get_lion_with_linear_scheduler(
                    learning_rate_start=learning_rate,
                    learning_rate_end=learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    **kwargs
                )
            elif scheduler == 'warm_up_cosine':
                tx, sc = optimizers.get_lion_with_warm_up_cosine_scheduler(
                    learning_rate=learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    **kwargs
                )
            else:
                raise ValueError('seems like you have choose wrong type or unavailable scheduler')
        elif optimizer == 'adamw':
            if scheduler == 'linear':
                tx, sc = optimizers.get_adamw_with_linear_scheduler(
                    learning_rate_start=learning_rate,
                    learning_rate_end=learning_rate_end,
                    steps=steps,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    **kwargs
                )
            elif scheduler == 'cosine':
                tx, sc = optimizers.get_adamw_with_cosine_scheduler(
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    steps=steps,
                    weight_decay=weight_decay,
                    **kwargs
                )
            elif scheduler == 'none':
                tx, sc = optimizers.get_adamw_with_linear_scheduler(
                    learning_rate_start=learning_rate,
                    learning_rate_end=learning_rate,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    steps=steps,
                    **kwargs
                )
            elif scheduler == 'warm_up_cosine':
                tx, sc = optimizers.get_adamw_with_warm_up_cosine_scheduler(
                    learning_rate=learning_rate,
                    steps=steps,
                    weight_decay=weight_decay,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    **kwargs
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
        return torch.utils.tensorboard.SummaryWriter(
            log_dir=str(self.get_path()),
            comment=f'{self.model_name}',
            filename_suffix='easydel'
        )


class RLHFTrainer:

    def __init__(self,
                 config: RLHFConfig,
                 dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
                 tokenizer: Callable,
                 model: AVAILABLE_MODELS_FOR_RLHF,
                 reward_model: RewardModel,
                 critic_model: Optional[AVAILABLE_MODELS_FOR_RLHF] = None,
                 actor_critic: Optional[ActorCritic] = None
                 ):
        self.model = model

        if actor_critic is None:
            actor_critic = ActorCritic(
                model=model,
                critic_model=critic_model,
                pooled_values=False,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
                precision=config.precision

            )
        self.actor_critic = actor_critic
        self.reward_model = reward_model
        self.critic_model = critic_model

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config

        self.actor_optim, self.actor_scheduler = RLHFConfig.get_optimizer_and_scheduler(
            learning_rate=config.actor_lr,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            steps=int(1e5) * 5,
            scheduler=config.scheduler,
            optimizer=config.optimizer,
            learning_rate_end=config.actor_lr - 1e6
        )

        self.critic_optim, self.critic_scheduler = RLHFConfig.get_optimizer_and_scheduler(
            learning_rate=config.critic_lr,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            steps=int(1e5) * 5,
            scheduler=config.scheduler,
            optimizer=config.optimizer,
            learning_rate_end=config.critic_lr - 1e6
        )

    def learn(
            self,
            memory=typing.Deque[Memory]
    ):
        memories_stacked = ...
        memory_dataloader = create_dataloader(
            memories_stacked,
            self.config.minibatch_size
        )

        for _ in range(
                self.config.epochs
        ):
            for (
                    input_ids,
                    pm,
                    attention_mask,
                    old_action_probs,
                    old_log_probs,
                    rewards,
                    old_values
            ) in memory_dataloader:
                action_masks = ~pm & attention_mask
                action_logits, values = self.actor_critic(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                action_logits = shift(action_logits, shift=1, axis=-2)
                action_len = old_log_probs.shape[-1]
                action_probs = jax.nn.softmax(action_logits, axis=-1)
                action_log_probs = log_prob(action_probs, input_ids)
                action_log_probs = action_log_probs[:, -action_len:]
                entropies = masked_entropy(action_probs, attention_mask=action_masks)

                kl_penalty = 0.
                ...

    def __call__(self, *args, **kwargs):
        ...
