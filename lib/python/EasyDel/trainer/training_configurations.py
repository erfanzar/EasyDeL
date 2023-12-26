import os.path
import pathlib
import re
import typing
from typing import OrderedDict, List, Union

import fjformer.optimizers

import torch.utils.tensorboard
import wandb
from fjformer import StreamingCheckpointer
from jax.experimental.mesh_utils import create_device_mesh

from jax.sharding import Mesh
from jax import numpy as jnp
import jax
from ..etils import EasyDelGradientCheckPointers, EasyDelOptimizers, EasyDelSchedulers

AVAILABLE_BACKENDS: List[str] = [
    "cpu", "gpu", "tpu", None
]


class TrainArguments(
    OrderedDict
):
    def __init__(
            self,
            model_name: str,
            num_train_epochs: int,
            model_id: str = None,
            model_class=None,
            total_batch_size: int = 32,
            max_steps: Union[int, None] = None,
            optimizer: EasyDelOptimizers | str = EasyDelOptimizers.ADAMW,
            scheduler: EasyDelSchedulers | str = EasyDelSchedulers.NONE,
            learning_rate: Union[int, float] = 5e-5,
            learning_rate_end: Union[None, float] = 5e-6,
            gradient_accumulation_steps: int = 1,
            weight_decay: float = 0.01,
            gradient_checkpointing: EasyDelGradientCheckPointers | str = EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
            max_length: Union[int, None] = 4096,
            sharding_array: Union[tuple, int] = (1, -1, 1, 1),
            is_fine_tuning: bool = True,
            do_train: bool = True,
            do_eval: bool = False,
            do_test: Union[bool, None] = False,
            backend: Union[str, None] = None,
            extra_optimizer_kwargs: dict = None,
            save_steps: Union[int, None] = None,
            save_dir: str = "easydel_checkpoint",
            use_pjit_attention_force: bool = False,
            dtype: jnp.dtype = jnp.bfloat16,
            param_dtype: jnp.dtype = jnp.bfloat16,
            fully_fsdp: bool = True,
            use_wandb: bool = True,
            custom_rule: typing.Mapping[str, jax.sharding.PartitionSpec] = None,
            extra_configs: dict = None,
            ids_to_pop_from_dataset: list = None,
            remove_ckpt_after_load: bool = False,
            configs_to_init_model_class=None,
            do_last_save: bool = True,
            model_parameters=None,
            do_shard_fns: bool = True,
            track_memory: bool = True,
            loss_remat: str = "",
            loss_chunk: int = 1024,
            is_left_padded: bool = False,
            warmup_steps: int = 500,
            init_input_shape: typing.Tuple[int, int] = (1, 1),
            step_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
                ("dp", "fsdp"), "sp"),
            training_time: typing.Optional[str] = None,
            **kwargs
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and makes sure that it has all of
        the attributes necessary for proper functioning. It also allows you to set
        default values for those attributes if they are not provided as arguments by
        the person creating an instance.

        :param self: Refer to the class instance itself
        :param model_name: str: Specify the model name
        :param num_train_epochs: int: Set the number of epochs for training
        :param model_id: str: Load a model from the save_dir
        :param model_class: Initialize the model, and the configs_to_init_model_class parameter is used to
        :param total_batch_size: int: Set the batch size of the model
        :param max_steps: Union[int,None]: Determine the maximum number of steps to train for
        :param optimizer: str: Specify which optimizer to use
        :param scheduler: str: Set the learning rate scheduler
        :param learning_rate: Union[int,float]: Set the learning rate , Set the dtype of the model parameters
        :param learning_rate_end: Union[None,float]: Set the end learning rate, Set the dtype of the model parameters
        :param gradient_accumulation_steps: int: Accumulate gradients over multiple batches
        :param weight_decay: float: Control the weight decay
        :param gradient_checkpointing: str: Control the gradient checkpointing method
        :param max_length: Union[int, None]: Set the maximum length of a sequence, Pass the model_class to the trainer class
        :param sharding_array: Union[tuple: Shard the model across multiple devices
        :param is_fine_tuning: bool: Determine whether the model is being trained from scratch or not
        :param do_train: bool: Determine whether the model should be trained or not
        :param do_eval: bool: Determine whether to run the eval loop or not
        :param do_test: Union[bool,None]: Determine whether to run the test or not, Pass the model_class to the trainer
        :param backend: Union[str, None]:: Specify the device that will be used for training, Define the default value of a parameter
        :param extra_optimizer_kwargs: dict: Pass extra arguments to the optimizer
        :param save_steps: Union[int,None]: Save the model after a number of steps,  Set the default value of do_test to none
        :param save_dir: str: Specify the directory where the model checkpoints will be saved
        :param use_pjit_attention_force: bool: Determine whether to use the jax
        :param dtype: Set the data type of the model parameters and inputs
        :param param_dtype: Specify the data type of the model parameters
        :param fully_fsdp: Control the use of fully fused sdp
        :param use_wandb: bool: Determine whether to use wandb or not
        :param custom_rule: Pass a custom rule to the optimizer,
        :param extra_configs: Pass extra configurations to the model class
        :param ids_to_pop_from_dataset: list: Pop some keys from the dataset
        :param training_time: str: maximum time for Trainer to Train the Model
        :param remove_ckpt_after_load: bool: Remove the checkpoint after loading it
        :param configs_to_init_model_class: Pass the configs to the model class
        :param do_last_save: bool: Save the model at the end of training
        :param model_parameters: Pass the model parameters to the trainer
        :param do_shard_fns: bool: Shard the model across multiple devices
        :param track_memory: bool: Track the memory usage of the model
        :param loss_remat: str: Specify how to rematerialize the loss function
        :param loss_chunk: int: Chunk the loss function
        :param is_left_padded: bool: Indicate whether the input is left padded or not
        :param warmup_steps: int: Warm up the learning rate
        :param init_input_shape: typing.Tuple[int]: Initialize the input shape of the model
        :param step_partition_spec: jax.sharding.PartitionSpec: PartitionSpec Custom to be used in training and eval or test loop
        :param **kwargs: Pass a variable number of keyword arguments to a function
        :return: Nothing

        """
        super().__init__()
        if ids_to_pop_from_dataset is None:
            ids_to_pop_from_dataset = []
        if extra_optimizer_kwargs is None:
            extra_optimizer_kwargs = {}
        assert model_class is not None or model_id is not None, "you cant pass model_class and model_id both None " \
                                                                "you should at least pass one of them to build " \
                                                                "model with"
        assert backend in AVAILABLE_BACKENDS, f"{backend} is not recognized, " \
                                              f"available backends are {AVAILABLE_BACKENDS}"
        self.available_backends = len(jax.devices(backend))
        total_batch_size *= gradient_accumulation_steps
        array_devices = jnp.ones(
            (self.available_backends, 1)).reshape(sharding_array)
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
        self.warmup_steps = warmup_steps
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.track_memory = track_memory
        self.loss_chunk = loss_chunk
        self.loss_remat = loss_remat
        self.init_input_shape = init_input_shape
        self.is_left_padded = is_left_padded
        self.step_partition_spec = step_partition_spec

        self.training_time = self._time_to_seconds(
            training_time) if training_time is not None else None
        torch.set_default_device("cpu")
        self.__dict__.update(**kwargs)

    @staticmethod
    def _time_to_seconds(time_str):
        pattern = r"(\d+)\s*(H|min|Min)"
        match = re.match(pattern, time_str)

        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()

            if unit == "h":
                return value * 3600  # Convert hours to seconds
            elif unit == "min":
                return value * 60  # Convert minutes to seconds
        else:
            raise SyntaxError(
                "Invalid input format it should be like 50Min for M and 23H for hours")

    def __call__(self):
        return {k: v for k, v in self.__dict__.items()}

    def get_meter_dict(self):
        """
        The get_meter_dict function is used to return a dictionary of the hyperparameters.
        The function iterates through all the attributes in the class and returns a dictionary with
        the key as &quot;hyperparameters/{k}&quot; and value as v for each attribute k,v in self.__dict__ if it is an instance of int, float, str, bool or torch.Tensor.

        :param self: Represent the instance of the class
        :return: A dictionary of hyperparameters

        """
        return {f"hyperparameters/{k}": v for k, v in self.__dict__.items() if
                isinstance(v, (int, float, str, bool, torch.Tensor))}

    def get_wandb_init(self):
        """
        The get_wandb_init function is a helper function that returns the wandb.init() call with
        the project name, config object, and tags set to appropriate values for this model.

        :param self: Pass the class instance to the function
        :return: A wandb

        """
        return wandb.init(
            project=f"easydel-{self.model_name}",
            config=self(),
            tags=[
                "Easy Del",
                "OST-OpenSourceTransformers",
                "Jax/Flax"
            ]
        )

    def __str__(self):
        string = f"TrainingArguments(\n"
        for k, v in self.__call__().items():
            if isinstance(v, typing.Callable):
                def string_func(it_self):
                    string_ = f"{it_self.__class__.__name__}(\n"
                    for k_, v_ in it_self.__dict__.items():
                        string_ += f"\t\t{k_} : {v_}\n"
                    string_ += "\t)"
                    return string_

                v.__str__ = string_func
                v = v.__str__(v)
            string += f"\t{k} : {v}\n"
        string += ")"
        return string

    def get_path(self):
        """
        The get_path function returns a pathlib.Path object, which is a class that
        represents file paths and provides methods for interacting with the files at
        those paths. The get_path function takes no arguments and returns an instance of
        the Path class initialized with two arguments: self.save_dir (a string) and
        self.model_name (also a string). The save directory is the directory where we'll
        store our model checkpoints, while the model name will be used to create unique
        filenames for each checkpoint.

        :param self: Represent the instance of the class
        :return: A pathlib

        """
        return pathlib.Path(
            self.save_dir, self.model_name
        )

    def ckpt_path_exists(self):
        """
        The ckpt_path_exists function checks to see if the path exists. If it does not, then it creates a new directory.

        :param self: Represent the instance of the class
        :return: A path

        """
        path = self.get_path()
        if not path.exists():
            path.mkdir(parents=True)

    def get_mesh(self):
        """
        The get_mesh function is used to create a mesh object that can be used
        to define the geometry of the device. The mesh object contains two arrays:
        a list of vertices and a list of faces. Each face is defined by three indices,
        which correspond to three vertices in the vertex array. The get_mesh function
        is called when creating an instance of DeviceGeometry, which is then passed
        into an instance of DeviceSimulation.

        :param self: Refer to the object itself
        :return: A mesh object with the device array shape and the mesh names

        """
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
        return "dp", "fsdp", "tp", "sp"

    def get_optimizer_and_scheduler(self, steps=None):
        """
        The get_optimizer_and_scheduler function is a helper function that returns the optimizer and scheduler
            based on the parameters passed to it.

        :param self: Represent the instance of the class
        :param steps: Calculate the number of steps to train
        :return: A tuple of two objects:

        """
        steps = self.max_steps or steps
        assert steps is not None, "if you haven\'t pass max steps to init you should pass init in func"

        if self.optimizer == EasyDelOptimizers.ADAFACTOR:
            if self.scheduler == EasyDelSchedulers.LINEAR:
                tx, sc = fjformer.optimizers.get_adafactor_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate_end,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.COSINE:
                tx, sc = fjformer.optimizers.get_adafactor_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.NONE:
                tx, sc = fjformer.optimizers.get_adafactor_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.WARM_UP_COSINE:
                tx, sc = fjformer.optimizers.get_adafactor_with_warm_up_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.WARM_UP_LINEAR:
                tx, sc = fjformer.optimizers.get_adafactor_with_warmup_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    steps=steps,
                    learning_rate_end=self.learning_rate_end,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    warmup_steps=self.warmup_steps,
                    **self.extra_optimizer_kwargs

                )

            else:
                raise ValueError(
                    "seems like you have choose wrong type or unavailable scheduler"
                )
        elif self.optimizer == EasyDelOptimizers.LION:
            if self.scheduler == EasyDelSchedulers.LINEAR:
                tx, sc = fjformer.optimizers.get_lion_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate_end,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.COSINE:
                tx, sc = fjformer.optimizers.get_lion_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.NONE:
                tx, sc = fjformer.optimizers.get_lion_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.WARM_UP_COSINE:
                tx, sc = fjformer.optimizers.get_lion_with_warm_up_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )

            elif self.scheduler == EasyDelSchedulers.WARM_UP_LINEAR:
                tx, sc = fjformer.optimizers.get_lion_with_with_warmup_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    steps=steps,
                    learning_rate_end=self.learning_rate_end,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    warmup_steps=self.warmup_steps,
                    **self.extra_optimizer_kwargs
                )
            else:
                raise ValueError(
                    "seems like you have choose wrong type or unavailable scheduler")
        elif self.optimizer == EasyDelOptimizers.ADAMW:
            if self.scheduler == EasyDelSchedulers.LINEAR:
                tx, sc = fjformer.optimizers.get_adamw_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate_end,
                    steps=steps,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.COSINE:
                tx, sc = fjformer.optimizers.get_adamw_with_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.NONE:
                tx, sc = fjformer.optimizers.get_adamw_with_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    learning_rate_end=self.learning_rate,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    steps=steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.WARM_UP_COSINE:
                tx, sc = fjformer.optimizers.get_adamw_with_warm_up_cosine_scheduler(
                    learning_rate=self.learning_rate,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    **self.extra_optimizer_kwargs
                )
            elif self.scheduler == EasyDelSchedulers.WARM_UP_LINEAR:
                tx, sc = fjformer.optimizers.get_adamw_with_warmup_linear_scheduler(
                    learning_rate_start=self.learning_rate,
                    steps=steps,
                    weight_decay=self.weight_decay,
                    learning_rate_end=self.learning_rate_end,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    warmup_steps=self.warmup_steps,
                    **self.extra_optimizer_kwargs
                )
            else:
                raise ValueError(
                    "seems like you have choose wrong type or unavailable scheduler"
                )
        else:
            raise ValueError(
                "seems like you have choose wrong type or unavailable optimizer"
            )
        return tx, sc

    def get_streaming_checkpointer(self):
        """
        The get_streaming_checkpointer function is used to save the model's weights.
        The streaming checkpointer saves the model's weights in a file called &quot;checkpoint&quot; and then
        saves a copy of that file with an incrementing number appended to it (e.g., checkpoint_001,
        checkpoint_002, etc.). This allows you to keep multiple versions of your trained models.

        :param self: Represent the instance of the class
        :return: A streamingcheckpointer object

        """
        return StreamingCheckpointer(StreamingCheckpointer.get_default_config(),
                                     os.path.join(self.save_dir, self.model_name))

    def get_board(self):
        """
        The get_board function is a helper function that returns a TensorBoard object.
        The TensorBoard object is used to log the training and validation loss, as well as
        the accuracy of the model during training. The get_board function takes no arguments,
        and returns an instance of torch.utils.tensorboard SummaryWriter class.

        :param self: Represent the instance of the class
        :return: A summary-writer object

        """
        return torch.utils.tensorboard.SummaryWriter(
            log_dir=str(self.get_path()),
            comment=f"{self.model_name}",
            filename_suffix="easydel"
        )
