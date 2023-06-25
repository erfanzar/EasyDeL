import os.path
from typing import OrderedDict, List

import fjutils.optimizers
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
AVAILABLE_BACKENDS: List[str | None] = [
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
            total_batch_size: int,
            max_steps: int | None = None,
            optimizer: str = 'lion',
            scheduler: str = 'linear',
            learning_rate: int | float = 5e-5,
            learning_rate_end: None | float = 5e-6,
            weight_decay: float = 0.01,
            gradient_checkpointing: str = 'nothing_saveable',
            max_length: int | None = 4096,
            sharding_array: tuple | int = (1, -1, 1),
            is_fine_tuning: bool = True,
            do_train: bool = True,
            do_eval: bool = False,
            do_test: bool | None = False,
            backend: str | None = None,
            extra_optimizer_kwargs: dict = {},
            save_steps: int | None = None,
            save_dir: str = 'easydel_ckpt',
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

        self.array_devices = jnp.asarray([len(jax.devices(backend))]).reshape(sharding_array)
        self.array_devices_shape = self.array_devices.shape
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
        self.__dict__.update(**kwargs)

    def __call__(self):
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self):
        string = f'TrainingArguments(\n'
        for k, v in self.__call__().items():
            string += f'\t{k} : {v}\n'
        string += ')'
        return string

    def ckpt_path_exists(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def get_mesh(self):
        return Mesh(
            create_device_mesh(
                self.array_devices
            ),
            self.get_mesh_names()
        )

    @staticmethod
    def get_mesh_names():
        return 'dp', 'fsdp', 'mp'

    def get_optimizer_and_scheduler(self, steps=None):
        steps = self.max_steps or steps
        assert steps is not None
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
                    weight_decay=self.weight_decay,
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
        return StreamingCheckpointer(StreamingCheckpointer.get_default_config(), self.save_dir)
