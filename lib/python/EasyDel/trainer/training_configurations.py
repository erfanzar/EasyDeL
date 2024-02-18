import os.path
import pathlib
import re
import typing
from typing import OrderedDict, List, Union, Mapping, Optional, Tuple, Callable, Type

import termcolor
from wandb.apis.public import Run
from wandb.sdk.lib import RunDisabled

from .utils import JaxDistributedConfig
from ..etils.auto_tx import get_optimizer_and_scheduler
from jax.sharding import PartitionSpec
import torch.utils.tensorboard
import wandb
from fjformer import CheckpointManager
from jax.experimental.mesh_utils import create_device_mesh
from fjformer.xrapture import XRapTureConfig, XRapTure
from jax.sharding import Mesh
from jax import numpy as jnp
import jax
from ..etils import (
    EasyDelGradientCheckPointers,
    EasyDelOptimizers,
    EasyDelSchedulers,
    AVAILABLE_GRADIENT_CHECKPOINTS,
    AVAILABLE_SCHEDULERS,
    AVAILABLE_OPTIMIZERS
)
from ..modules.easydel_modelling_utils import EasyDelFlaxPretrainedModel

from jax.tree_util import PyTreeDef

AVAILABLE_BACKENDS: List[str] = [
    "cpu", "gpu", "tpu", None
]


class EasyDeLXRapTureConfig(XRapTureConfig):  # Don't Make user involved with FJFormer
    def __init__(
            self,
            parameters: PyTreeDef | dict,
            **kwargs
    ):
        self.parameters = parameters
        super().__init__(**kwargs)


class TrainArguments(
    OrderedDict
):
    def __init__(
            self,
            model_name: str,
            num_train_epochs: int,
            model_class: Optional[EasyDelFlaxPretrainedModel | Type[EasyDelFlaxPretrainedModel]] = None,
            model_huggingface_repo_id: Optional[str] = None,
            total_batch_size: int = 32,
            max_training_steps: Optional[int] = None,
            optimizer: AVAILABLE_OPTIMIZERS = EasyDelOptimizers.ADAMW,
            scheduler: AVAILABLE_SCHEDULERS = EasyDelSchedulers.NONE,
            learning_rate: Union[int, float] = 5e-5,
            learning_rate_end: Optional[float] = 5e-6,
            gradient_accumulation_steps: int = 1,
            weight_decay: float = 0.01,
            gradient_checkpointing: AVAILABLE_GRADIENT_CHECKPOINTS = EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
            max_sequence_length: Optional[int] = 4096,
            sharding_array: Union[tuple, int] = (1, -1, 1, 1),
            is_fine_tuning: bool = True,
            do_train: bool = True,
            do_eval: bool = False,
            do_test: Optional[bool] = False,
            backend: Optional[str] = None,
            extra_optimizer_kwargs: dict = None,
            save_steps: Optional[int] = None,
            save_dir: str = "EasyDel-Checkpoints",
            use_pjit_attention_force: bool = False,
            dtype: jnp.dtype = jnp.bfloat16,
            param_dtype: jnp.dtype = jnp.bfloat16,
            fully_sharded_data_parallel: bool = True,
            use_wandb: bool = True,
            custom_rule: Mapping[str, PartitionSpec] = None,
            extra_configs: Optional[dict] = None,
            ids_to_pop_from_dataset: Optional[list] = None,
            remove_ckpt_after_load: bool = False,
            configs_to_initialize_model_class: Optional[dict] = None,
            do_last_save: bool = True,
            model_parameters: Optional[dict] = None,
            do_shard_fns: bool = True,
            track_memory: Optional[bool] = None,
            loss_re_mat: str = "",
            loss_chunk: int = 1024,
            truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
            warmup_steps: int = 500,
            init_input_shape: Tuple[int, int] = (1, 1),
            step_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp"),
            training_time: Optional[str] = None,
            dataloader_num_workers: Optional[int] = 0,
            dataloader_pin_memory: Optional[bool] = False,
            jax_distributed_config: Optional[dict] = None,
            log_all_workers: bool = False,
            wandb_entity: Optional[str] = None,
            save_optimizer_state: bool = False,
            step_start_point: Optional[int] = None,
            verbose: bool = True,
            offload_device: jax.Device = jax.devices("cpu")[0],
            rapture_config: Optional[EasyDeLXRapTureConfig] = None,
            merge_lora_rapture_parameters: bool = True,
            state_apply_fn_kwarguments_to_model: Optional[dict] = None,
            remove_unused_columns: bool = True,
            **kwargs
    ):
        """
    The __init__ function is called when the class is instantiated.
    It sets up the attributes of an object, which are sometimes called fields or properties.
    The __init__ function can accept arguments, just like a normal function.

    :param self: Represent the instance of the class
    :param model_name: str: Specify the model name
    :param num_train_epochs: int: Set the number of epochs for training
    :param model_huggingface_repo_id: Optional[str]: Load a pretrained model from the huggingface model hub
    :param model_class: Optional[EasyDelFlaxPretrainedModel]: Pass a model class to the trainer
    :param total_batch_size: int: Set the batch size of the model
    :param max_training_steps: Optional[int]: Set the maximum number of steps to train for
    :param optimizer: AVAILABLE_OPTIMIZERS: Specify the optimizer used to train the model
    :param scheduler: AVAILABLE_SCHEDULERS: Set the learning rate scheduler
    :param learning_rate: Union[int: Set the learning rate for the optimizer
    :param float]: Set the learning rate
    :param learning_rate_end: Optional[float]: Set the learning rate at the end of training
    :param gradient_accumulation_steps: int: Accumulate gradients over multiple batches
    :param weight_decay: float: Specify the weight decay to be used by the optimizer
    :param gradient_checkpointing: AVAILABLE_GRADIENT_CHECKPOINTS: Determine how to use gradient checkpointing
    :param max_sequence_length: Optional[int]: Set the maximum length of the input sequence
    :param sharding_array: Union[tuple,int]: Specify the mesh of devices to use for training
    :param is_fine_tuning: bool: Tell the model whether or not to initialize the weights of
    :param do_train: bool: Indicate whether to train the model or not
    :param do_eval: bool: Determine whether to run evaluation on the validation set after training
    :param do_test: Optional[bool]: Determine if the model should be tested
    :param backend: Optional[str]: Specify the backend of jax
    :param extra_optimizer_kwargs: dict: Pass extra arguments to the optimizer
    :param save_steps: Optional[int]: Save the model after every n steps
    :param save_dir: str: Define the directory where the checkpoints will be saved
    :param use_pjit_attention_force: bool: Force the use of pjit for attention layers
    :param dtype: jnp.dtype: Set the dtype of the model parameters
    :param param_dtype: jnp.dtype: Specify the data type of the model parameters
    :param fully_sharded_data_parallel: bool: Determine if the model should be fully fsdp or not
    :param use_wandb: bool: Enable or disable the wandb logging
    :param custom_rule: Mapping[str, PartitionSpec]: Specify the partitioning rules of the model
    :param extra_configs: Optional[dict]: Pass extra configurations to the model class
    :param ids_to_pop_from_dataset: Optional[list]: Remove some of the ids from the dataset
    :param remove_ckpt_after_load: bool: Remove the checkpoint after loading it
    :param configs_to_initialize_model_class: Optional[dict]: Pass extra configurations to the model class
    :param do_last_save: bool: Save the model after training is complete
    :param model_parameters: Optional[dict]: Pass the model parameters to the model class
    :param do_shard_fns: bool: Shard the model functions across devices
    :param track_memory: bool: Track the memory usage of the model
    :param loss_re_mat: str: Specify the regular expression to match the loss function name
    :param loss_chunk: int: Chunk the loss to avoid memory overflow
    :param truncation_mode: typing.Literal["keep_end", "keep_start"]: Determine if the input is left padded or not and
    which side of the array should remain in case of using maximum padding.
    :param warmup_steps: int: Specify the number of steps to warm up the learning rate
    :param init_input_shape: Tuple[int, int]: Initialize the model with a shape that is not (batch_size, length)
    :param step_partition_spec: PartitionSpec: Partition the model for training
    :param training_time: Optional[str]: Set a time limit for the training process
    :param dataloader_num_workers: Optional[int]: Set the number of workers used by pytorch's
    :param dataloader_pin_memory: Optional[bool]: Pin the memory of the dataloader
    :param jax_distributed_config: Optional[dict]: Configure the jax distributed backend
    :param log_all_workers: bool: Log all workers in wandb,
    :param wandb_entity: Optional[str]: Specify the entity to use when logging to weights &amp; biases
    :param save_optimizer_state : bool: when ever to save optimizer state and other args in checkpoint
    :param step_start_point: Optional[int]: start training from given step for example instead of starting training from
    step 0 it will start from 20000 and leave the data behind
    :param verbose: bool: when ever to turn verbose mode of or on
    :param offload_device: jax.Device: device to be used to offload parameters on
    :param rapture_config: Optional[EasyDeLXRaptureConfig]: LoRA Config for models
    :param merge_lora_rapture_parameters: bool: whenever to merge lora parameters with original parameters before saving
    :param state_apply_fn_kwarguments_to_model: Optional[dict]: state_apply_fn_kwarguments_to_model is a dictionary that
    be used to apply the parameters and extra things that you want to deliver to model.
    :param remove_unused_columns: bool: when ever to remove the unused data columns from dataset
    :param **kwargs: Pass keyword, variable-length argument list
    :return: Nothing
        """
        super().__init__()

        if ids_to_pop_from_dataset is None:
            ids_to_pop_from_dataset = []
        if extra_optimizer_kwargs is None:
            extra_optimizer_kwargs = {}

        if model_class is None and model_huggingface_repo_id is None:
            print(
                termcolor.colored(
                    "Warning : ", color="red", force_color=True
                ) + termcolor.colored(
                    "You should at least pass model_class or model_huggingface_repo_id if you want to use "
                    "CasualLanguageModel Trainer But in case that you want to use "
                    "DPOTrainer you can ignore this warning", color="white",
                    force_color=True
                )
            )
        assert backend in AVAILABLE_BACKENDS, (
            f"{backend} is not recognized, "
            f"available backends are {AVAILABLE_BACKENDS}"
        )

        if track_memory is None:
            # https://github.com/erfanzar/EasyDeL/pull/100/commits/523ce7b1515d7896d456759d0bcd0bd02369bd10
            print(
                termcolor.colored("Information : ", color="red", force_color=True),
                termcolor.colored(
                    "track_memory is set to False by default inorder make make training faster. "
                    "you can turn it on with just passing `track_memory=True` in TrainArguments",
                    color="white", force_color=True
                )
            )
            track_memory = False

        available_backends = len(jax.devices(backend))

        total_batch_size *= gradient_accumulation_steps

        array_devices = jnp.ones((available_backends, 1)).reshape(sharding_array)

        JaxDistributedConfig.initialize(jax_distributed_config)

        self.available_backends = available_backends
        self.array_devices_shape = array_devices.shape
        self.model_huggingface_repo_id = model_huggingface_repo_id
        self.num_train_epochs = num_train_epochs
        self.wandb_entity = wandb_entity
        self.total_batch_size = total_batch_size
        self.max_training_steps = max_training_steps
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.extra_optimizer_kwargs = extra_optimizer_kwargs
        self.learning_rate = learning_rate
        self.learning_rate_end = learning_rate_end
        self.weight_decay = weight_decay
        self.model_name = model_name
        self.gradient_checkpointing = gradient_checkpointing
        self.max_sequence_length = max_sequence_length
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
        self.fully_sharded_data_parallel = fully_sharded_data_parallel
        self.use_wandb = use_wandb
        self.custom_rule = custom_rule
        self.extra_configs = extra_configs
        self.ids_to_pop_from_dataset = ids_to_pop_from_dataset
        self.remove_ckpt_after_load = remove_ckpt_after_load
        self.model_class = model_class
        self.configs_to_initialize_model_class = configs_to_initialize_model_class
        self.do_last_save = do_last_save
        self.model_parameters = model_parameters
        self.do_shard_fns = do_shard_fns
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.track_memory = track_memory
        self.loss_chunk = loss_chunk
        self.loss_re_mat = loss_re_mat
        self.init_input_shape = init_input_shape
        self.truncation_mode = truncation_mode
        self.step_partition_spec = step_partition_spec
        self.jax_distributed_config = jax_distributed_config
        self.log_all_workers = log_all_workers
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.save_optimizer_state = save_optimizer_state
        self.step_start_point = step_start_point if step_start_point is not None else 0
        self.verbose = verbose
        self.offload_device = offload_device
        self.optimizer_kwargs = dict(
            learning_rate=self.learning_rate,
            learning_rate_end=self.learning_rate_end,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            extra_optimizer_kwargs=self.extra_optimizer_kwargs,
            warmup_steps=self.warmup_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            weight_decay=self.weight_decay,
            steps=self.max_training_steps,
        )
        self.training_time = self._time_to_seconds(training_time) if training_time is not None else None
        torch.set_default_device("cpu")
        self.merge_lora_rapture_parameters = merge_lora_rapture_parameters
        self.rapture = None
        self.rapture_config = None
        self.remove_unused_columns = remove_unused_columns
        self.state_apply_fn_kwarguments_to_model = (
            state_apply_fn_kwarguments_to_model
        ) if state_apply_fn_kwarguments_to_model is not None else {}
        if rapture_config is not None:
            print(
                termcolor.colored("Warning : ", color="red", force_color=True),
                termcolor.colored(
                    "You are using LoRA (Low-Rank Adaptation of Large Language Models) and this feature is"
                    "still in Beta mode so it might act unexpected", color="red", force_color=True
                )
            )
            self.rapture_config = rapture_config
            self.rapture = XRapTure(config=rapture_config)

        self.__dict__.update(**kwargs)

    @staticmethod
    def _time_to_seconds(time_str):
        pattern = r"(\d+)\s*(h|min)"
        match = re.match(pattern, time_str.lower())

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
        the key as &quot;hyperparameters/{k}&quot; and value as v for each attribute k,v in self.__dict__ if it is an
         instance of int, float, str, bool or torch.Tensor.

        :param self: Represent the instance of the class
        :return: A dictionary of hyperparameters

        """
        return {
            f"hyperparameters/{k}": v for k, v in self.__dict__.items() if
            isinstance(v, (int, float, str, bool, torch.Tensor))
        }

    def get_wandb_init(self) -> Run | RunDisabled | None:
        """
        The get_wandb_init function is a helper function that returns the wandb.init() call with
        the project name, config object, and tags set to appropriate values for this model.

        :param self: Pass the class instance to the function
        :return: A wandb or None

        """
        return wandb.init(
            project=f"EasyDeL-{self.model_name}",
            config=self(),
            tags=[
                "Easy Del",
                "FJFormer",
                "OST-OpenSourceTransformers",
                "Jax/Flax"
            ],
            entity=self.wandb_entity

        ) if self.log_all_workers or (jax.process_index() == 0) else None

    def __str__(self):
        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__call__().items():
            if isinstance(v, Callable):
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

    def get_optimizer_and_scheduler(
            self,
            steps: int | None = None
    ):
        self.optimizer_kwargs["steps"] = steps or self.optimizer_kwargs["steps"]
        return get_optimizer_and_scheduler(
            **self.optimizer_kwargs
        )

    def get_streaming_checkpointer(self):
        """
        The get_streaming_checkpointer function is used to save the model's weights.
        The streaming checkpointer saves the model's weights in a file called &quot;checkpoint&quot; and then
        saves a copy of that file with an incrementing number appended to it (e.g., checkpoint_001,
        checkpoint_002, etc.). This allows you to keep multiple versions of your trained models.

        :param self: Represent the instance of the class
        :return: A CheckpointManager object

        """
        return CheckpointManager(
            os.path.join(self.save_dir, self.model_name),
            save_optimizer_state=self.save_optimizer_state,
            verbose=self.verbose
        )

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
