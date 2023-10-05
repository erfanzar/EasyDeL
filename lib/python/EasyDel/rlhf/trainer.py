import collections
import functools
import os
import pathlib
import pprint
import random
import typing
from typing import Union, Optional, Callable, List, Any

import optax
from fjutils import tracker
import IPython
import einops
import fjutils
import flax.training.train_state
import jax
import torch
import tqdm.autonotebook
import wandb
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from fjutils import StreamingCheckpointer
from fjutils import optimizers
from flax import struct, core

from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax.experimental import pjit
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from torch.utils.data.dataloader import DataLoader
from transformers import PretrainedConfig

from .ppo import ActorCritic
from .reward import RewardModel
from .utils import AVAILABLE_MODELS_FOR_RLHF, shift, log_prob, masked_entropy, masked_mean, masked_normalize, \
    clipped_value_loss

Memory = collections.namedtuple('Memory', [
    'logits',
    'prompt_mask',
    'attention_mask',
    'action_prob',
    'action_log_prob',
    'reward',
    'value'
])


class TrainStateRLHF(struct.PyTreeNode):
    step: int
    apply_fn_critic: Callable = struct.field(pytree_node=False)
    apply_fn_reward: Callable = struct.field(pytree_node=False)
    apply_fn_actor: Callable = struct.field(pytree_node=False)

    actor_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    critic_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    reward_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    actor_optim: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_optim: optax.GradientTransformation = struct.field(pytree_node=False)

    actor_opt_state: optax.OptState = struct.field(pytree_node=True)
    critic_opt_state: optax.OptState = struct.field(pytree_node=True)

    def apply_gradients(self,
                        *,
                        grads_critic,
                        grad_actor,
                        **kwargs
                        ):
        updates_critic, new_state_critic = self.critic_optim.update(
            grads_critic, self.critic_opt_state, self.critic_params)
        critic_params = optax.apply_updates(self.critic_params, updates_critic)

        updates_actor, new_state_actor = self.actor_optim.update(
            grad_actor, self.actor_opt_state, self.actor_params
        )
        actor_params = optax.apply_updates(self.actor_params, updates_actor)

        return self.replace(
            critic_opt_state=new_state_critic,
            actor_opt_state=new_state_actor,

            critic_params=critic_params,
            actor_params=actor_params,

            step=self.step + 1,
            **kwargs
        )

    @classmethod
    def create(cls,
               *,
               apply_fn_critic,
               apply_fn_reward,
               apply_fn_actor,
               actor_params,
               critic_params,
               reward_params,
               actor_optim: optax.GradientTransformation,
               critic_optim: optax.GradientTransformation,
               **kwargs
               ):
        actor_opt_state = actor_optim.init(actor_params)
        critic_opt_state = critic_optim.init(critic_params)
        return cls(
            step=0,

            apply_fn_actor=apply_fn_actor,
            apply_fn_critic=apply_fn_critic,
            apply_fn_reward=apply_fn_reward,

            actor_params=actor_params,
            critic_params=critic_params,
            reward_params=reward_params,

            actor_optim=actor_optim,
            critic_optim=critic_optim,

            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            **kwargs,
        )


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
                 backend_offload: str = 'cpu',
                 track_memory: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.backend_offload = backend_offload
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
        if config.track_memory:
            tracker.initialise_tracking()
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
        self.mesh = config.get_mesh()

    def init_params(
            self,
            params_lm: core.FrozenDict = None,
            params_actor: core.FrozenDict = None,
            params_critic: core.FrozenDict = None,
            params_reward: core.FrozenDict = None
    ):
        with ((jax.default_device(jax.devices(self.config.backend_offload)[0]))):

            if params_actor is None:
                params_actor = self.actor_critic.init(
                    {
                        'params': jax.random.PRNGKey(
                            0
                        )
                    },
                    input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                    attention_mask=jnp.ones((1, 1), dtype=jnp.int32)

                )

                params_actor = flax.traverse_util.unflatten_dict(
                    params_actor
                )['params']
            else:
                params_actor = flax.traverse_util.unflatten_dict(
                    flax.core.unfreeze(params_actor)
                )['params']

            if params_reward is None:
                params_reward = self.reward_model.init(
                    {
                        'params': jax.random.PRNGKey(
                            0
                        )
                    },
                    input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                    attention_mask=jnp.ones((1, 1), dtype=jnp.int32)

                )

                params_reward = flax.traverse_util.unflatten_dict(
                    params_reward
                )['params']
            else:
                params_reward = flax.traverse_util.unflatten_dict(
                    flax.core.unfreeze(params_reward)
                )['params']

            if params_critic is None:
                params_critic = self.actor_critic.init(
                    {
                        'params': jax.random.PRNGKey(
                            0
                        )
                    },
                    input_ids=jnp.ones((1, 1), dtype=jnp.int32),
                    attention_mask=jnp.ones((1, 1), dtype=jnp.int32)

                )

                params_critic = flax.traverse_util.unflatten_dict(
                    params_critic
                )['params']
            else:
                params_critic = flax.traverse_util.unflatten_dict(
                    flax.core.unfreeze(params_actor)
                )['params']

            if params_lm is not None:
                params_lm = flax.traverse_util.unflatten_dict(flax.core.unfreeze(params_lm)['params'])
                lm_keys = [k for k in params_lm.keys()]

                print(
                    f'\033[1;31mLoadable Parameters from the Lm Params are {len(lm_keys)}'
                )
                for key, _ in params_actor.items():
                    if key in lm_keys:
                        params_actor[key] = params_lm[key]
                for key, _ in params_critic.items():
                    if key in lm_keys:
                        params_critic[key] = params_lm[key]
                for key, _ in params_reward.items():
                    if key in lm_keys:
                        params_reward[key] = params_lm[key]

            return (
                params_lm,
                params_actor,
                params_critic,
                params_reward
            )

    def configure_funcs(
            self,
            partition_rules,
            params_lm: core.FrozenDict = None,
            params_actor: core.FrozenDict = None,
            params_critic: core.FrozenDict = None,
            params_reward: core.FrozenDict = None
    ):

        params_lm, params_actor, params_critic, params_reward = self.init_params(
            params_lm=params_lm,
            params_actor=params_actor,
            params_critic=params_critic,
            params_reward=params_reward
        )

        def create_train_state(
                params_actor_,
                params_critic_,
                params_reward_
        ) -> Union[Any, TrainStateRLHF]:
            return TrainStateRLHF.create(
                actor_params=params_actor_,
                critic_params=params_critic_,
                reward_params=params_reward_,
                actor_optim=self.actor_optim,
                critic_optim=self.critic_optim,
                apply_fn_critic=self.critic_model.apply,
                apply_fn_actor=self.actor_critic.apply,
                apply_fn_reward=self.reward_model.apply
            )

        shape = jax.eval_shape(
            create_train_state(
                params_actor_=params_actor,
                params_critic_=params_critic,
                params_reward_=params_reward
            )
        )
        partition_specs = fjutils.match_partition_rules(params=shape, rules=partition_rules)
        shard_fns, _ = fjutils.make_shard_and_gather_fns(
            partition_specs=partition_specs,
            dtype_specs=self.config.dtype
        )

        sharded_create_train_state = pjit.pjit(
            create_train_state,
            in_shardings=(PartitionSpec(), PartitionSpec(), PartitionSpec()),
            out_shardings=(partition_specs,),
            backend=self.config.backend
        )

        return sharded_create_train_state(
            params_actor, params_critic, params_reward
        ), partition_specs

    def learn(
            self,
            memory: typing.Deque[Memory],
            train_state: TrainStateRLHF,
    ):
        """
        Memory Must be Deque of all prevision memories
        train_state : TrainStateRLHF Type
        """
        memory_dataloader = create_dataloader(
            memory,
            self.config.minibatch_size
        )

        def forward(
                train_state: TrainStateRLHF,
                input_ids,
                pm,
                rewards,
                old_values,
                attention_mask,
                old_action_probs,
                old_log_probs,
        ):
            def calculate_loss(params):
                global rewards
                global old_values

                action_masks = ~pm & attention_mask
                action_logits, values = train_state.apply_fn_actor(
                    params["params"],
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                action_logits = shift(action_logits, shift=1, axis=-2)
                action_len = old_log_probs.shape[-1]
                action_probs = jax.nn.softmax(action_logits, axis=-1)
                action_log_probs = log_prob(action_probs, input_ids)
                action_log_probs = action_log_probs[:, -action_len:]
                entropies = masked_entropy(action_probs, attention_mask=action_masks)
                kl_penalty = masked_mean(
                    jnp.sum((old_action_probs * (jnp.log(old_action_probs) - jnp.log(action_probs))), axis=-1),
                    attention_mask=attention_mask
                ) * self.config.kl_div_loss_weight

                rewards = rewards - kl_penalty
                normalize_kwargs = dict()

                if old_values.ndim == 2:
                    old_values, values = map(lambda t: shift(t, shift=1, axis=-2), (old_values, values))

                    old_values = old_values[:, -action_len:]
                    values = values[:, -action_len:]
                    rewards = einops.rearrange(rewards, 'b -> b 1')
                    normalize_kwargs = dict(axis=-1, attention_mask=action_masks[:, -action_len:])

                if values.ndim < rewards.ndim:
                    values = einops.rearrange(values, '... -> ... 1')

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = masked_normalize(rewards - old_values, **normalize_kwargs)

                if advantages.ndim == 1:
                    advantages = einops.rearrange(advantages, 'b -> b 1')

                surr1 = ratios * advantages
                surr2 = jnp.clip(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip)
                policy_loss = -jnp.minimum(surr1, surr2) - self.config.beta_s * entropies  # Policy Loss
                loss = jnp.mean(policy_loss)  # Loss

                value_loss = jnp.mean(
                    clipped_value_loss(values, rewards, old_values, self.config.value_clip))  # VLoss
                return loss, value_loss  # I need gradient for these two losses

            grad, (loss_, value_loss_) = jax.value_and_grad(calculate_loss)(train_state.actor_params)
            train_state = train_state.apply_gradients(
                grad_actor=grad,  # Based on Loss
                grads_critic=jax.grad(lambda p: value_loss_)(train_state.critic_params)  # Based on Value Loss
            )
            return train_state, (loss_, value_loss_)

        pbar = tqdm.autonotebook.tqdm(
            total=self.config.epochs * len(memory_dataloader)
        )
        if self.config.track_memory:
            mem_res = tracker.get_mem()
        else:
            mem_res = 'Tracking Option is OFF'
        for _ in range(
                self.config.epochs
        ):
            for (
                    input_ids_,
                    pm_,
                    attention_mask_,
                    old_action_probs_,
                    old_log_probs_,
                    rewards_,
                    old_values_
            ) in memory_dataloader:
                train_state, loss = forward(
                    train_state=train_state,
                    attention_mask=attention_mask_,
                    input_ids=input_ids_,
                    pm=pm_,
                    rewards=rewards_,
                    old_values=old_values_,
                    old_log_probs=old_log_probs_,
                    old_action_probs=old_action_probs_
                )

                pbar.set_postfix(
                    loss=loss
                )
                if self.config.track_memory:
                    IPython.display.clear_output(True)
                    pbar.display(mem_res)
                pbar.update(1)
        return train_state

    def train(
            self,
            params_actor: core.FrozenDict,
            params_critic: core.FrozenDict,
            params_reward: core.FrozenDict,
            params_lm: core.FrozenDict,
            partition_rules,
            num_episodes=50000,
            max_time_steps=500,
            update_time_steps=5000,
            max_batch_size=16,
            max_sequence_length=2048,
            eos_token=None,
            temperature=1.,
    ):
        with self.mesh:

            time = 0
            memories = collections.deque([])
            max_train_eps = self.dataset['train'].num_rows
            assert max_train_eps > max_batch_size
            train_state, partition_specs = self.configure_funcs(
                partition_rules=partition_rules,
                params_lm=params_lm,
                params_actor=params_actor,
                params_critic=params_critic,
                params_reward=params_reward
            )

            def get_rand():
                return int(random.random() * max_train_eps)

            @functools.partial(
                pjit.pjit,
                in_shardings=(
                        partition_specs,
                        PartitionSpec(),
                        PartitionSpec()
                ),
                out_shardings=(
                        PartitionSpec(),
                        PartitionSpec(),
                        PartitionSpec(),
                        PartitionSpec(),
                        PartitionSpec(),
                        PartitionSpec(),
                        PartitionSpec()
                )
            )
            def step(
                    train_state_: TrainStateRLHF,
                    input_ids_,
                    attention_mask_
            ):
                action_, sequence_, attention_mask_, prompt_mask_, action_logits_, value_ = self.actor_critic.generate(
                    params=train_state_.actor_params,
                    input_ids=einops.rearrange(
                        input_ids_, 'n -> 1 n'
                    ),
                    attention_mask_=attention_mask_,
                    max_sequence_length=max_sequence_length,
                    eos_token=eos_token,
                    temperature=temperature,
                    return_values=True
                )
                action_logits_ = shift(action_logits_, shift=1, axis=-2)
                action_probs_ = jax.nn.softmax(action_logits_, axis=-1)
                action_len_ = action_.shape[-1]
                action_log_prob_ = log_prob(action_probs_, sequence_)
                action_log_prob_ = action_log_prob_[:, -action_len_:]
                action_ = einops.rearrange(action_, '1 ... -> ...')
                sequence_ = jnp.concatenate(
                    (
                        input_ids_, action_
                    ), axis=0
                )
                prompt_length_ = len(input_ids_)
                prompt_mask_ = jnp.arange(sequence_.shape[-1]) < prompt_length_

                sequence_ = einops.rearrange(sequence_, 'n -> 1 n')
                prompt_mask_ = einops.rearrange(prompt_mask_, 'n -> 1 n')

                reward_ = train_state_.apply_fn_reward(
                    train_state_.reward_params,
                    sequence_,
                    prompt_mask=prompt_mask_,
                    attention_mask=attention_mask_,
                    sample=True
                )
                return (
                    sequence_,
                    prompt_mask_,
                    attention_mask_,
                    action_probs_,
                    action_log_prob_,
                    reward_,
                    value_
                )

            sharded_step = step(
                train_state_=train_state,
                input_ids_=jnp.ones((1, 128), dtype=jnp.int32),
                attention_mask_=jnp.ones((1, 128), dtype=jnp.int32),
            )
            rearrange_ = lambda t: einops.rearrange(t, '1 ... -> ...')
            pbar = tqdm.tqdm(iterable=range(num_episodes), desc='Episode')
            for _  in pbar:
                for time_step in range(max_time_steps):
                    time += 1
                    index = get_rand()
                    index = index - max_batch_size if index > max_batch_size else index
                    input_ids = self.dataset['train'][index:index + max_batch_size]['input_ids']
                    attention_mask = self.dataset['train'][index:index + max_batch_size]['attention_mask']

                    (
                        sequence,
                        prompt_mask,
                        attention_mask,
                        action_probs,
                        action_log_prob,
                        reward,
                        value
                    ) = sharded_step(
                        train_state,
                        input_ids,
                        attention_mask
                    )
                    pbar.set_postfix(
                        time=time, time_step=f"{time_step}/{max_time_steps}"
                    )
                    memories.append(Memory(*map(rearrange_, (
                        sequence,
                        prompt_mask,
                        attention_mask,
                        action_probs,
                        action_log_prob,
                        reward,
                        value
                    ))))

                    # learn from the stored memories

                    if time % update_time_steps == 0:
                        train_state = self.learn(
                            memories,
                            train_state=train_state
                        )
                        memories.clear()
        return train_state
