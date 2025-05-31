# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import glob
import json
import os
import pathlib
import typing as tp
from functools import cached_property

import jax
from eformer.escale import PartitionAxis
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizer

from easydel.infra import (
    EasyDeLBaseConfig,
    EasyDeLBaseModule,
    EasyDeLState,
)
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import TaskType
from easydel.layers.attention import AttentionMechanisms
from easydel.modules.auto.auto_configuration import get_modules_by_type
from easydel.utils.checkpoint_managers.path_utils import EasyPath
from easydel.utils.helpers import get_logger

from ...utils.checkpoint_managers.streamer import (
    DTYPE_TO_STRING_MAP,
    STRING_TO_DTYPE_MAP,
)
from ..base_trainer import BaseTrainer
from ..trainer.trainer import Trainer
from ..training_configurations import TrainingArguments

if tp.TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = tp.Any

logger = get_logger("RayTrainer")


class RayDistributedConfig(BaseModel):
    pretrained_model_name_or_path: str
    model_task: TaskType | None = None
    model_type: str | None = None
    offload_device: str | None = None
    config_scaling_variables: dict[str, int] | None = None
    config_variables: dict[str, tp.Any] | None = None

    def _saveing_preprocess(self):
        for k, v in list(self.config_variables.items()):
            if v in STRING_TO_DTYPE_MAP.values():
                self.config_variables[k] = DTYPE_TO_STRING_MAP[v]

        for k, v in list(self.config_scaling_variables.items()):
            if v in STRING_TO_DTYPE_MAP.values():
                self.config_scaling_variables[k] = DTYPE_TO_STRING_MAP[v]

    def _loading_postprocess(self):
        for k, v in list(self.config_variables.items()):
            if v in DTYPE_TO_STRING_MAP.values():
                self.config_variables[k] = STRING_TO_DTYPE_MAP[v]

        for k, v in list(self.config_scaling_variables.items()):
            if v in DTYPE_TO_STRING_MAP.values():
                self.config_scaling_variables[k] = STRING_TO_DTYPE_MAP[v]
        if "partition_axis" in self.config_variables.keys():
            paxis = self.config_variables["partition_axis"]
            if not isinstance(paxis, PartitionAxis):
                self.config_variables["partition_axis"] = PartitionAxis(**paxis)


class RayDistributedTrainer:
    """
    A Ray-based distributed trainer for EasyDeL models.

    This class facilitates distributed training of language models using Ray,
    allowing for scaling experiments and efficient utilization of resources.
    It handles model configuration, creation, state management, and the
    training process.

    Attributes:
            model_task (TaskType): The task type of the model (e.g., Causal Language Modeling).
            model_type (str): The type of the model (e.g., "llama", "mistral").
            model_class (tp.Type[EasyDeLBaseModule]): The EasyDeL module class for the model.
            state_class (tp.Type[EasyDeLState]): The EasyDeL state class for managing model state.
            trainer_module (tp.Type[BaseTrainer | Trainer]): The trainer class to be used.
            config_scaling_variables (tp.Dict[str, int]): Configuration parameters that scale
                    with the `scaling_index`.
            config_variables (tp.Dict[str, tp.Any]): Fixed configuration parameters.
            pretrained_model_name_or_path (str): Path or identifier for the pretrained model
                    or tokenizer.
            _processor_loader_class (tp.Type[PreTrainedTokenizer]): The class used to load the tokenizer.
    """

    model_task: TaskType
    model_type: str

    model_class: type[EasyDeLBaseModule]
    state_class: type[EasyDeLState]

    offload_device: str

    trainer_module: type[BaseTrainer | Trainer]

    CONFIG_SCALING_VARIABLES: tp.ClassVar = {
        "hidden_size": 256,
        "intermediate_size": 256 * 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }
    """Default scaling variables for model configuration."""

    CONFIG_VARIABLES: tp.ClassVar = {
        "dtype": jnp.bfloat16,
        "param_dtype": jnp.bfloat16,
        "precision": lax.Precision.DEFAULT,
        "seed": 654,
        "max_position_embeddings": 2**13,
        "gradient_checkpointing": EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        "initializer_range": 0.02,
        "partition_axis": PartitionAxis(),
        "attn_mechanism": AttentionMechanisms.AUTO,
        "attn_dtype": jnp.bfloat16,
        "attn_softmax_dtype": jnp.bfloat16,
        "sharding_axis_dims": (1, -1, 1, 1, 1),
    }
    """Default fixed variables for model configuration."""

    _processor_loader_class: type[PreTrainedTokenizer] = AutoTokenizer
    """The class used to load the tokenizer, defaults to AutoTokenizer."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_task: TaskType | None = None,
        model_type: str | None = None,
        model_class: type[EasyDeLBaseModule] | None = None,
        state_class: type[EasyDeLState] | None = None,
        offload_device: str | None = None,
        trainer_module: type[BaseTrainer | Trainer] | None = None,
        config_scaling_variables: dict[str, int] | None = None,
        config_variables: dict[str, tp.Any] | None = None,
    ):
        """
        Initializes the RayDistributedTrainer.

        Args:
                pretrained_model_name_or_path: Path or identifier for the pretrained
                        model or tokenizer. This is required.
                model_task: The task type of the model. If None, it's inferred from
                        `model_class` or requires `model_type` to be set.
                model_type: The type of the model. If None, it's inferred from
                        `model_class` or requires `model_task` to be set.
                model_class: The EasyDeL module class. If None, it's determined using
                        `model_type` and `model_task`.
                state_class: The EasyDeL state class. Defaults to `EasyDeLState`.
                trainer_module: The trainer class. Defaults to `Trainer`.
                config_scaling_variables: Custom scaling variables to override defaults.
                config_variables: Custom fixed variables to override defaults.

        Raises:
                AssertionError: If `model_class` is None and `model_type` or `model_task`
                        is also None.
                AssertionError: If `model_task` and `model_type` are provided but
                        `model_class` is also provided (ambiguous).
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if model_task is None or model_type is None:
            assert model_task is None and model_type is None, (
                "If one of model_task or model_type is None, both must be None."
            )
            assert model_class is not None, "model_class must be provided if model_task and model_type are not."
            model_type = model_class._model_type
            model_task = model_class._model_task
        elif model_class is not None:
            logger.warning(
                "Both model_class and model_type/model_task were provided. "
                "Using the provided model_class and inferring type/task from it."
            )
            model_type = model_class._model_type
            model_task = model_class._model_task

        if model_class is None:
            assert model_type is not None and model_task is not None, (
                "model_type and model_task must be provided if model_class is not specified."
            )
            _, model_class_retrieved = get_modules_by_type(
                model_type=model_type,
                task_type=model_task,
            )
            assert model_class_retrieved is not None, f"Could not retrieve model class for {model_type} and {model_task}"
            self.model_class = model_class_retrieved
        else:
            self.model_class = model_class

        self.config_scaling_variables = RayDistributedTrainer.CONFIG_SCALING_VARIABLES.copy()
        self.config_variables = RayDistributedTrainer.CONFIG_VARIABLES.copy()

        if config_scaling_variables is not None:
            self.config_scaling_variables.update(config_scaling_variables)
        if config_variables is not None:
            self.config_variables.update(config_variables)

        self.model_task = model_task
        self.model_type = model_type
        self.offload_device = offload_device if offload_device is not None else "cpu"
        self.state_class = state_class if state_class is not None else EasyDeLState
        self.trainer_module = trainer_module if trainer_module is not None else Trainer

    @classmethod
    def from_config(
        cls,
        path: str | os.PathLike,
        model_class: type[EasyDeLBaseModule] | None = None,
        state_class: type[EasyDeLState] | None = None,
        trainer_module: type[BaseTrainer | Trainer] | None = None,
    ):
        config = RayDistributedConfig(**json.loads(EasyPath(path).read_text()))

        config._loading_postprocess()

        return cls(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            model_task=config.model_task,
            model_type=config.model_type,
            config_scaling_variables=config.config_scaling_variables,
            config_variables=config.config_variables,
            offload_device=config.offload_device,
            trainer_module=trainer_module,
            state_class=state_class,
            model_class=model_class,
        )

    def save_config(self, path: str | os.PathLike):
        config = RayDistributedConfig(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            model_task=self.model_task,
            model_type=self.model_type,
            offload_device=self.offload_device,
            config_scaling_variables=self.config_scaling_variables,
            config_variables=self.config_variables,
        )
        config._saveing_preprocess()
        EasyPath(path).write_text(config.model_dump_json(indent=2))

    def load_processor(self) -> PreTrainedTokenizer:
        """
        Loads and returns the tokenizer/processor.

        Returns:
                The loaded PreTrainedTokenizer.
        """
        base = self._processor_loader_class
        processor = base.from_pretrained(self.pretrained_model_name_or_path)
        has_eos = hasattr(processor, "eos_token_id")
        pad_tkn = getattr(processor, "pad_token_id", None)
        if pad_tkn is None and has_eos:
            logger.warning(
                "Your tokenizer doesn't have a specific token for padding (pad_token). "
                "We'll use the end-of-sequence token (eos_token) for padding instead."
            )
            processor.pad_token_id = processor.eos_token_id
        return processor

    @cached_property
    def processor(self) -> PreTrainedTokenizer:
        """
        Provides cached access to the tokenizer/processor.

        Returns:
                The loaded PreTrainedTokenizer.
        """
        return self.load_processor()

    @staticmethod
    def extract_column_names(dataset: Dataset) -> list[str] | None:
        if hasattr(dataset, "column_names") and len(dataset.column_names) != 0:
            return dataset.column_names
        keys = None
        for _sample in dataset:
            keys = _sample.keys()
            break
        return keys

    def process_sample_data(
        self,
        sample: tp.Any,
        max_length: int,
        padding_side: str = "left",
    ) -> dict[str, jax.Array]:
        """
        Processes a single sample of data using the tokenizer.

        Args:
                sample: The input sample (e.g., text).
                max_length: The maximum sequence length for padding/truncation.
                padding_side: The side to pad on ('left' or 'right').
                        Defaults to "left".

        Returns:
                A dictionary of tokenized data (e.g., 'input_ids', 'attention_mask').
        """
        out = self.processor(
            sample,
            padding="max_length",
            max_length=max_length,
            return_tensors="jax",
            padding_side=padding_side,
            return_attention_mask=True,
            truncation=True,
        )
        out = {k: (v.reshape(-1) if hasattr(v, "shape") else v) for k, v in out.items()}
        return out

    def process_messages_data(
        self,
        messages: tp.Any,
        max_length: int,
        padding_side: str = "left",
    ) -> dict[str, jax.Array]:
        """
        Processes conversational data (messages) using the tokenizer's chat template.

        Args:
                messages: A list of messages in a conversational format compatible
                        with the tokenizer's chat template.
                max_length: The maximum sequence length for padding/truncation.
                padding_side: The side to pad on ('left' or 'right').
                        Defaults to "left".

        Returns:
                A dictionary of tokenized data.
        """
        out = self.processor.apply_chat_template(
            messages,
            padding="max_length",
            max_length=max_length,
            return_tensors="jax",
            padding_side=padding_side,
            return_dict=True,
            truncation=True,
        )

        out = {k: (v.reshape(-1) if hasattr(v, "shape") else v) for k, v in out.items()}
        return out

    def create_config(self, scaling_index: int) -> EasyDeLBaseConfig:
        """
        Creates a model configuration based on a scaling index.

        The `scaling_index` multiplies a base set of parameters (defined in
        `self.config_scaling_variables`) to allow for easy scaling experiments.

        Args:
                scaling_index: An integer factor to scale certain configuration
                        parameters (e.g., hidden_size, num_attention_heads).

        Returns:
                An EasyDeLBaseConfig instance.
        """
        current_scaling_variables = {
            k: v * scaling_index for k, v in copy.deepcopy(self.config_scaling_variables).items()
        }
        config_kwargs = {**self.config_variables, **current_scaling_variables}
        config_class = self.model_class.config_class
        if config_class is None:
            config_class, _ = get_modules_by_type(
                model_type=self.model_type,
                task_type=self.model_task,
            )
        return config_class(**config_kwargs)

    def create_model(
        self,
        config: EasyDeLBaseConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.PrecisionLike | None = None,
        seed: int = 684,
        lazy: bool = False,
    ) -> EasyDeLBaseModule:
        """
        Creates an instance of the model.

        Args:
                config: The model configuration object.
                dtype: The data type for computations (default: bfloat16).
                param_dtype: The data type for model parameters (default: bfloat16).
                precision: The JAX precision level (e.g., lax.Precision.DEFAULT).
                seed: Random seed for model initialization.
                lazy: If True, uses lazy initialization for the model. Defaults to False.

        Returns:
                An instance of EasyDeLBaseModule.
        """
        if precision is None:
            precision = lax.Precision.DEFAULT

        init_kwargs = dict(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=nn.Rngs(seed),
        )

        with jax.default_device(jax.local_devices(backend=self.offload_device)[-1]):
            if lazy:
                return self.model_class.lazy_init(**init_kwargs)
            else:
                return self.model_class(**init_kwargs)

    def convert_model_to_state(self, model: EasyDeLBaseModule) -> EasyDeLState:
        """
        Converts an initialized model module to an EasyDeLState object.

        Args:
                model: The initialized EasyDeLBaseModule instance.

        Returns:
                An EasyDeLState instance containing the model's parameters and state.
        """
        state = model.to_state(self.state_class)
        state = state.shard_state()
        return state

    def load_state(
        self,
        load_directory: str | os.PathLike,
        scaling_index,
        **kwargs,
    ) -> EasyDeLState:
        """
        Loads a model state from a specified directory.

        Args:
                load_directory: The directory from which to load the state.
                **kwargs: Additional keyword arguments to pass to the state loading method.

        Returns:
                The loaded EasyDeLState instance.
        """

        def _create():
            model = self.create_model(
                config=self.create_config(scaling_index=scaling_index),
                dtype=self.config_variables["dtype"],
                param_dtype=self.config_variables["param_dtype"],
                precision=self.config_variables["precision"],
                seed=self.config_variables["seed"],
            )
            return self.state_class.create(step=0, model=model)

        if pathlib.Path(load_directory).exists():
            checkpoint_files = glob.glob(os.path.join(load_directory, "run-*"))
            checkpoint_files.sort(key=os.path.getmtime)
            if len(checkpoint_files) != 0:
                load_directory = checkpoint_files[-1]
            else:
                load_directory = load_directory
            if pathlib.Path(load_directory).exists():
                try:
                    return self.state_class.load_state(load_directory=load_directory, **kwargs)
                except Exception:
                    logger.info("failed to load from provided checkpoint path creating new model.")
                    return _create()
            else:
                return _create()
        else:
            return _create()

    def create_model_from_config(self, scaling_index: int):
        return self.create_model(
            config=self.create_config(scaling_index=scaling_index),
            dtype=self.config_variables["dtype"],
            param_dtype=self.config_variables["param_dtype"],
            precision=self.config_variables["precision"],
            seed=self.config_variables["seed"],
        )

    def create_trainer(
        self,
        arguments: TrainingArguments,
        dataset_train: Dataset,
        dataset_eval: Dataset | None = None,
        data_collator: tp.Callable | None = None,
        state: EasyDeLState | None = None,
    ) -> BaseTrainer | Trainer:
        """
        Creates and configures a trainer instance.

        This method handles the logic for initializing the model state,
        either by converting a provided model, loading from a checkpoint,
        or using a directly provided state.

        Args:
                arguments: TrainingArguments for configuring the trainer.
                dataset_train: The training dataset.
                dataset_eval: The evaluation dataset (optional).
                data_collator: Callable to collate data batches (optional).
                checkpoint_path: Path to a checkpoint to load model state from.
                        If `model` or `state` is provided, this is ignored.
                state: An EasyDeLState object. If provided, this is used directly.

        Returns:
                An instance of the configured trainer (BaseTrainer or Trainer).

        Raises:
                AssertionError: If no valid state can be obtained (from model,
                        checkpoint, or direct input).
                FileNotFoundError: If `checkpoint_path` is provided but the
                        checkpoint is not found.
        """

        return self.trainer_module(
            arguments=arguments,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
            tokenizer=self.processor,
            data_collator=data_collator,
            model_state=state,
        )

    def train(
        self,
        scaling_index: int,
        arguments: TrainingArguments,
        dataset_train: Dataset,
        dataset_eval: Dataset | None = None,
        data_collator: tp.Callable | None = None,
        checkpoint_path: str | os.PathLike | None = None,
        model: EasyDeLBaseModule | None = None,
        state: EasyDeLState | None = None,
        load_state_kwargs: dict[str, tp.Any] | None = None,
    ):
        """
        Initializes a model (if not provided) and starts the training process.

        Args:
                scaling_index: Index for scaling model configuration. Used if a new
                        model needs to be created.
                arguments: TrainingArguments for the trainer.
                dataset_train: The training dataset.
                dataset_eval: Evaluation dataset (optional).
                data_collator: Data collator function (optional).
                checkpoint_path: Path to a checkpoint. Used if `model` and `state`
                        are None.
                model: An existing EasyDeLBaseModule instance (optional).
                state: An existing EasyDeLState instance (optional).
                load_state_kwargs: Arguments for loading state from checkpoint (optional).

        Returns:
                The result of the trainer's train() method.
        """

        if state is None and model is None and checkpoint_path is None:
            logger.info(
                f"No model, state, or checkpoint provided. Creating a new model with scaling_index={scaling_index}."
            )
            model = self.create_model_from_config(scaling_index=scaling_index)
            state = self.convert_model_to_state(model)
        elif checkpoint_path is not None:
            load_state_kwargs = self.config_variables if load_state_kwargs is None else load_state_kwargs
            state = self.load_state(checkpoint_path, scaling_index, **load_state_kwargs)
            state = state.shard_state()
        elif model is not None:
            state = self.convert_model_to_state(model)

        del model
        assert state is not None, "Couldn't load or verify state."

        return self.create_trainer(
            arguments=arguments,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
            data_collator=data_collator,
            state=state,
        ).train()

    def __repr__(self):
        cls_name = self.__class__.__name__
        items = []

        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                try:
                    repr_str = str(v).replace("\n", "\n  ")
                    if len(repr_str) > 200:
                        repr_str = f"{v.__class__.__name__}(...)"
                    items.append(f"  {k} : {repr_str}")
                except TypeError:
                    items.append(f"  {k} : <unrepresentable>")

        return f"{cls_name}(\n" + "\n".join(items) + "\n)"

    __str__ = __repr__
