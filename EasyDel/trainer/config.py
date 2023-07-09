import os.path
import pathlib
import typing
from typing import OrderedDict, List, Union

import fjutils.optimizers
import torch.utils.tensorboard
import wandb
from fjutils import StreamingCheckpointer
from jax.experimental.mesh_utils import create_device_mesh

from jax.sharding import Mesh
from jax import numpy as jnp
import jax

AVAILABLE_OPTIMIZERS: List[str] = ['adafactor', 'lion', 'adamw']
AVAILABLE_SCHEDULERS: List[str] = ['linear', 'cosine', 'none']
AVAILABLE_GRADIENT_CHECK_POINTING: List[str] = ['everything_saveable',
                                                'nothing_saveable',
                                                'checkpoint_dots',
                                                'checkpoint_dots_with_no_batch_dims']
AVAILABLE_BACKENDS: List[str] = [
    'cpu', 'gpu', 'tpu', None
]


class TrainArguments(
    OrderedDict
):
    def __init__(
            self,
            model_id: str,
            model_name: str,
            num_train_epochs: int,
            total_batch_size: int = 32,
            max_steps: Union[int, None] = None,
            optimizer: str = 'lion',
            scheduler: str = 'linear',
            learning_rate: Union[int, float] = 5e-5,
            learning_rate_end: Union[None, float] = 5e-6,
            weight_decay: float = 0.01,
            gradient_checkpointing: str = 'nothing_saveable',
            max_length: Union[int, None] = 4096,
            sharding_array: Union[tuple, int] = (1, -1, 1),
            is_fine_tuning: bool = True,
            do_train: bool = True,
            do_eval: bool = False,
            do_test: Union[bool, None] = False,
            backend: Union[str, None] = None,
            extra_optimizer_kwargs: dict = {},
            save_steps: Union[int, None] = None,
            save_dir: str = 'easydel_ckpt',
            use_pjit_attention_force: bool = True,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            fully_fsdp=True,
            use_wandb: bool = True,
            custom_rule=None,
            extra_configs=None,
            ids_to_pop_from_dataset=[],
            remove_ckpt_after_load: bool = False,
            model_class=None,
            configs_to_init_model_class=None,
            do_last_save: bool = True,
            model_parameters=None,
            do_shard_fns: bool = True,
            **kwargs
    ):
        super().__init__()

        assert backend in AVAILABLE_BACKENDS, f'{backend} is not recognized, ' \
                                              f'available backends are {AVAILABLE_BACKENDS}'
        assert gradient_checkpointing in AVAILABLE_GRADIENT_CHECK_POINTING, f'{gradient_checkpointing} is not ' \
                                                                            f'recognized, available gradient ' \
                                                                            f'checkpointing methods are ' \
                                                                            f'{AVAILABLE_GRADIENT_CHECK_POINTING}'
        assert scheduler in AVAILABLE_SCHEDULERS, f'{scheduler} is not recognized, ' \
                                                  f'available schedulers are {AVAILABLE_SCHEDULERS}'
        assert optimizer in AVAILABLE_OPTIMIZERS, f'{optimizer} is not recognized, ' \
                                                  f'available optimizers are {AVAILABLE_OPTIMIZERS}'
        self.available_backends = len(jax.devices(backend))
        array_devices = jnp.ones((self.available_backends, 1)).reshape(sharding_array)
        self.array_devices_shape = array_devices.shape
        self.model_id = model_id
        self.num_train_epochs = num_train_epochs
        self.total_batch_size = total_batch_size
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.extra_optimizer_kwargs = extra_optimizer_kwargs
        self.learning_rate = learning_rate
        self.learning_rate_end = learning_rate_end
        self.weight_decay = weight_decay
        self.model_name = model_name
        self.gradient_checkpointing = gradient_checkpointing
        self.max_length = max_length
        self.sharding_array = sharding_array
        self.is_fine_tuning = is_fine_tuning
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.save_steps = save_steps
        self.save_dir = save_dir
        self.use_pjit_attention_force = use_pjit_attention_force
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.fully_fsdp = fully_fsdp
        self.use_wandb = use_wandb
        self.custom_rule = custom_rule
        self.extra_configs = extra_configs
        self.ids_to_pop_from_dataset = ids_to_pop_from_dataset
        self.remove_ckpt_after_load = remove_ckpt_after_load
        self.model_class = model_class
        self.configs_to_init_model_class = configs_to_init_model_class
        self.do_last_save = do_last_save
        self.model_parameters = model_parameters
        self.do_shard_fns = do_shard_fns
        torch.set_default_device('cpu')
        self.__dict__.update(**kwargs)

    def __call__(self):
        return {k: v for k, v in self.__dict__.items()}

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

    def __str__(self):
        string = f'TrainingArguments(\n'
        for k, v in self.__call__().items():
            if isinstance(v, typing.Callable):
                def string_func(it_self):
                    string_ = f'{it_self.__class__.__name__}(\n'
                    for k_, v_ in it_self.__dict__.items():
                        string_ += f'\t\t{k_} : {v_}\n'
                    string_ += '\t)'
                    return string_

                v.__str__ = string_func
                v = v.__str__(v)
            string += f'\t{k} : {v}\n'
        string += ')'
        return string

    def get_path(self):
        return pathlib.Path(
            self.save_dir, self.model_name
        )

    def ckpt_path_exists(self):
        path = self.get_path()
        if not path.exists():
            path.mkdir(parents=True)

    def get_mesh(self):
        return Mesh(
            create_device_mesh(
                self.array_devices_shape
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
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'cosine':
                tx, sc = fjutils.optimizers.get_adafactor_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'none':
                tx, sc = fjutils.optimizers.get_adafactor_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
        elif self.optimizer == 'lion':
            if self.scheduler == 'linear':
                tx, sc = fjutils.optimizers.get_lion_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate_end,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'cosine':
                tx, sc = fjutils.optimizers.get_lion_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'none':
                tx, sc = fjutils.optimizers.get_lion_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
        elif self.optimizer == 'adamw':
            if self.scheduler == 'linear':
                tx, sc = fjutils.optimizers.get_adamw_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate_end,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'cosine':
                tx, sc = fjutils.optimizers.get_adamw_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == 'none':
                tx, sc = fjutils.optimizers.get_adamw_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
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
