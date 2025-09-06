# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import copy
import json
import os
import typing as tp
from functools import cached_property

import jax
from eformer.escale import PartitionAxis
from eformer.loggings import get_logger
from eformer.mpric import DTYPE_TO_STRING_MAP, STRING_TO_DTYPE_MAP
from eformer.paths import ePath
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizer

from easydel.infra import EasyDeLBaseConfig, EasyDeLBaseModule, EasyDeLState
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import TaskType
from easydel.layers.attention import AttentionMechanisms
from easydel.modules.auto.auto_configuration import get_modules_by_type

from ..base_trainer import BaseTrainer
from ..trainer.trainer import Trainer
from ..training_configurations import TrainingArguments

if tp.TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("RayTrainer")


class RayDistributedConfig(BaseModel):
    """
    Configuration for RayDistributedTrainer that can be persisted to JSON.

    This class handles serialization and deserialization of distributed training
    configurations, with special handling for JAX dtypes and PartitionAxis objects.

    Attributes:
        pretrained_model_name_or_path: Path or identifier for the pretrained model
        model_task: The task type for the model (e.g., CAUSAL_LM, SEQ2SEQ)
        model_type: The model architecture type (e.g., 'llama', 'gpt2')
        offload_backend: Backend device for offloading (e.g., 'cpu', 'gpu')
        config_scaling_variables: Variables to scale by scaling_index (e.g., hidden_size)
        config_variables: Fixed configuration variables (e.g., dtype, precision)

    Notes:
        - JAX dtype fields are converted to/from strings for JSON serialization
        - PartitionAxis objects are converted to/from dictionary representation
        - Use _saving_preprocess() before saving and _loading_postprocess() after loading
    """

    pretrained_model_name_or_path: str
    model_task: TaskType | None = None
    model_type: str | None = None
    offload_backend: str | None = None
    config_scaling_variables: dict[str, int] | None = None
    config_variables: dict[str, tp.Any] | None = None

    def _saving_preprocess(self):
        """Convert dtypes and PartitionAxis to JSON-serializable formats before saving."""
        if self.config_variables:
            for k, v in list(self.config_variables.items()):
                if v in STRING_TO_DTYPE_MAP.values():
                    self.config_variables[k] = DTYPE_TO_STRING_MAP[v]
            if "partition_axis" in self.config_variables and isinstance(
                self.config_variables["partition_axis"], PartitionAxis
            ):
                self.config_variables["partition_axis"] = self.config_variables["partition_axis"].__dict__

        if self.config_scaling_variables:
            for k, v in list(self.config_scaling_variables.items()):
                if v in STRING_TO_DTYPE_MAP.values():
                    self.config_scaling_variables[k] = DTYPE_TO_STRING_MAP[v]

    def _loading_postprocess(self):
        """Convert string representations back to dtypes and PartitionAxis after loading."""
        if self.config_variables:
            for k, v in list(self.config_variables.items()):
                if v in DTYPE_TO_STRING_MAP.values():
                    self.config_variables[k] = STRING_TO_DTYPE_MAP[v]
            if "partition_axis" in self.config_variables:
                pa = self.config_variables["partition_axis"]
                if not isinstance(pa, PartitionAxis):
                    self.config_variables["partition_axis"] = PartitionAxis(**pa)

        if self.config_scaling_variables:
            for k, v in list(self.config_scaling_variables.items()):
                if v in DTYPE_TO_STRING_MAP.values():
                    self.config_scaling_variables[k] = STRING_TO_DTYPE_MAP[v]


class RayDistributedTrainer:
    """
    Distributed trainer for Ray-based training with EasyDeL models.

    This class provides a lightweight wrapper for distributed training that:
    - Manages model configuration and scaling for different nodes
    - Handles model/state initialization and checkpoint loading
    - Delegates actual training to the underlying Trainer implementation

    The trainer supports:
    - Dynamic model scaling based on scaling_index
    - Automatic tokenizer/processor setup with padding configuration
    - Flexible checkpoint loading from various sources
    - Integration with Ray for distributed training orchestration

    Key Design Principles:
    - Resume logic is handled by BaseTrainer (set arguments.resume_if_possible=True)
    - State sharding is deferred to the main Trainer according to partition rules
    - Explicit checkpoint paths are used without automatic run-* resolution

    Attributes:
        model_task: The task type for the model (e.g., CAUSAL_LM)
        model_type: The model architecture type (e.g., 'llama')
        model_class: The EasyDeL model class to instantiate
        state_class: The state class for model checkpointing
        offload_backend: Backend for memory offloading
        trainer_module: The trainer class to use for actual training
        CONFIG_SCALING_VARIABLES: Variables that scale with scaling_index
        CONFIG_VARIABLES: Fixed configuration variables
    """

    # Model identity
    model_task: TaskType
    model_type: str
    model_class: type[EasyDeLBaseModule]
    state_class: type[EasyDeLState]

    offload_backend: str

    trainer_module: type[BaseTrainer | Trainer]

    CONFIG_SCALING_VARIABLES: tp.ClassVar[dict[str, int]] = {
        "hidden_size": 256,
        "intermediate_size": 256 * 4,
        "moe_intermediate_size": 256 * 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }

    CONFIG_VARIABLES: tp.ClassVar[dict[str, tp.Any]] = {
        "dtype": jnp.bfloat16,
        "param_dtype": jnp.bfloat16,
        "precision": lax.Precision.DEFAULT,
        "seed": 654,
        "max_position_embeddings": 2**13,
        "gradient_checkpointing": EasyDeLGradientCheckPointers.NONE,
        "initializer_range": 0.02,
        "partition_axis": PartitionAxis(),
        "attn_mechanism": AttentionMechanisms.AUTO,
        "attn_dtype": jnp.bfloat16,
        "attn_softmax_dtype": jnp.bfloat16,
        "sharding_axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
        "sharding_axis_dims": (1, -1, 1, 1, 1),
        "sharding_dcn_axis_dims": (1, -1, 1, 1, 1),
    }

    _processor_loader_class: type[PreTrainedTokenizer] = AutoTokenizer

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        bucket_path: str | None = None,
        model_task: TaskType | None = None,
        model_type: str | None = None,
        model_class: type[EasyDeLBaseModule] | None = None,
        state_class: type[EasyDeLState] | None = None,
        offload_backend: str | None = None,
        trainer_module: type[BaseTrainer | Trainer] | None = None,
        config_scaling_variables: dict[str, int] | None = None,
        config_variables: dict[str, tp.Any] | None = None,
    ):
        """
        Initialize the RayDistributedTrainer.

        Args:
            pretrained_model_name_or_path: Path or identifier for the pretrained model
            bucket_path: Optional path to load checkpoints from cloud storage
            model_task: Task type (inferred from model_class if not provided)
            model_type: Model architecture type (inferred from model_class if not provided)
            model_class: EasyDeL model class to use (auto-resolved if not provided)
            state_class: State class for checkpointing (defaults to EasyDeLState)
            offload_backend: Backend for memory offloading (defaults to 'cpu')
            trainer_module: Trainer class to use (defaults to Trainer)
            config_scaling_variables: Variables to scale with scaling_index
            config_variables: Fixed configuration variables

        Raises:
            AssertionError: If model class cannot be resolved or parameters are inconsistent
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if model_task is None or model_type is None:
            assert model_task is None and model_type is None, (
                "If one of model_task or model_type is None, both must be None."
            )
            assert model_class is not None, "model_class must be provided when model_task/model_type are omitted."
            model_type = model_class._model_type
            model_task = model_class._model_task
        elif model_class is not None:
            logger.warning(
                "Both model_class and model_type/model_task provided. Using model_class and inferring type/task from it."
            )
            model_type = model_class._model_type
            model_task = model_class._model_task

        if model_class is None:
            assert model_type is not None and model_task is not None, (
                "model_type and model_task must be provided if model_class is not specified."
            )
            _, resolved_class = get_modules_by_type(model_type=model_type, task_type=model_task)
            assert resolved_class is not None, f"Could not resolve model class for {model_type}/{model_task}"
            self.model_class = resolved_class
        else:
            self.model_class = model_class

        self.config_scaling_variables = copy.deepcopy(self.CONFIG_SCALING_VARIABLES)
        self.config_variables = copy.deepcopy(self.CONFIG_VARIABLES)
        if config_scaling_variables is not None:
            self.config_scaling_variables.update(config_scaling_variables)
        if config_variables is not None:
            self.config_variables.update(config_variables)

        self.bucket_path = bucket_path
        self.model_task = model_task
        self.model_type = model_type
        self.offload_backend = offload_backend if offload_backend is not None else "cpu"
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
        """
        Create a RayDistributedTrainer from a saved configuration file.

        Args:
            path: Path to the JSON configuration file
            model_class: Optional model class override
            state_class: Optional state class override
            trainer_module: Optional trainer module override

        Returns:
            RayDistributedTrainer: Initialized trainer instance
        """
        cfg = RayDistributedConfig(**json.loads(ePath(path).read_text()))
        cfg._loading_postprocess()
        return cls(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            model_task=cfg.model_task,
            model_type=cfg.model_type,
            config_scaling_variables=cfg.config_scaling_variables,
            config_variables=cfg.config_variables,
            offload_backend=cfg.offload_backend,
            trainer_module=trainer_module,
            state_class=state_class,
            model_class=model_class,
        )

    def save_config(self, path: str | os.PathLike):
        """
        Save the current configuration to a JSON file.

        Args:
            path: Path where the configuration will be saved
        """
        cfg = RayDistributedConfig(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            model_task=self.model_task,
            model_type=self.model_type,
            offload_backend=self.offload_backend,
            config_scaling_variables=self.config_scaling_variables,
            config_variables=self.config_variables,
        )
        cfg._saving_preprocess()
        ePath(path).write_text(cfg.model_dump_json(indent=2))

    def load_processor(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer/processor for the model.

        Returns:
            PreTrainedTokenizer: Loaded tokenizer with padding configuration

        Notes:
            - Automatically sets pad_token to eos_token if not defined
            - Logs a warning when falling back to eos_token for padding
        """
        tok_cls = self._processor_loader_class
        tokenizer = tok_cls.from_pretrained(self.pretrained_model_name_or_path)

        has_eos = hasattr(tokenizer, "eos_token_id")
        if getattr(tokenizer, "pad_token_id", None) is None and has_eos:
            logger.warning("Tokenizer has no pad_token. Falling back to eos_token for padding.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @cached_property
    def processor(self) -> PreTrainedTokenizer:
        """Cached property for the tokenizer/processor."""
        return self.load_processor()

    @staticmethod
    def extract_column_names(dataset: Dataset) -> list[str] | None:
        """
        Extract column names from a dataset.

        Args:
            dataset: The dataset to extract column names from

        Returns:
            list[str] | None: Column names if available, None otherwise
        """
        if hasattr(dataset, "column_names") and dataset.column_names:
            return list(dataset.column_names)
        for sample in dataset:
            return list(sample.keys())
        return None

    def process_sample_data(
        self,
        sample: tp.Any,
        max_length: int,
        padding_side: str = "left",
    ) -> dict[str, jax.Array]:
        """
        Process a text sample into model inputs.

        Args:
            sample: Raw text sample to process
            max_length: Maximum sequence length
            padding_side: Side to pad sequences ('left' or 'right')

        Returns:
            dict[str, jax.Array]: Tokenized and padded inputs with flattened shapes
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
        return {k: (v.reshape(-1) if hasattr(v, "shape") else v) for k, v in out.items()}

    def process_messages_data(
        self,
        messages: tp.Any,
        max_length: int,
        padding_side: str = "left",
    ) -> dict[str, jax.Array]:
        """
        Process chat messages using the tokenizer's chat template.

        Args:
            messages: Chat messages to process
            max_length: Maximum sequence length
            padding_side: Side to pad sequences ('left' or 'right')

        Returns:
            dict[str, jax.Array]: Tokenized and padded inputs with flattened shapes
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
        return {k: (v.reshape(-1) if hasattr(v, "shape") else v) for k, v in out.items()}

    def create_config(self, scaling_index: int) -> EasyDeLBaseConfig:
        """
        Create a model configuration with scaled dimensions.

        Args:
            scaling_index: Multiplier for scaling variables (e.g., hidden_size)

        Returns:
            EasyDeLBaseConfig: Configuration with scaled and fixed variables

        Notes:
            - Scaling variables are multiplied by scaling_index
            - Fixed variables remain unchanged
            - Useful for creating different model sizes in distributed training
        """
        not_allowed = ["precision", "dtype", "param_dtype"]
        scaled = {k: v * scaling_index for k, v in copy.deepcopy(self.config_scaling_variables).items()}
        config_kwargs = {**{k: v for k, v in self.config_variables.items() if k not in not_allowed}, **scaled}
        config_class = self.model_class.config_class
        if config_class is None:
            config_class, _ = get_modules_by_type(model_type=self.model_type, task_type=self.model_task)
        return config_class(**config_kwargs)

    def _get_offload_device(self):
        """
        Get the device for memory offloading.

        Returns:
            Device: Preferred local device or first available global device

        Notes:
            - Attempts to use local devices first for better performance
            - Falls back to global devices if local unavailable
        """
        try:
            devs = jax.local_devices(backend=self.offload_backend)
            if len(devs) > 0:
                return devs[0]
        except Exception:
            pass
        return jax.devices(self.offload_backend)[0]

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
        Create a model instance from configuration.

        Args:
            config: Model configuration
            dtype: Computation dtype
            param_dtype: Parameter dtype
            precision: JAX precision setting
            seed: Random seed for initialization
            lazy: Whether to use lazy initialization (memory efficient)

        Returns:
            EasyDeLBaseModule: Initialized model instance
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

        if lazy:
            return self.model_class.lazy_init(**init_kwargs)
        return self.model_class.sequential_init(**init_kwargs)

    def convert_model_to_state(self, model: EasyDeLBaseModule) -> EasyDeLState:
        """
        Convert a model module to a state object.

        Args:
            model: The model to convert

        Returns:
            EasyDeLState: State object for checkpointing

        Notes:
            - Does NOT perform sharding (handled by trainer)
            - Uses the configured state_class for conversion
        """
        return model.to_state(self.state_class)

    def create_model_from_config(self, scaling_index: int) -> EasyDeLBaseModule:
        """
        Create a model with configuration scaled by the given index.

        Args:
            scaling_index: Multiplier for scaling variables

        Returns:
            EasyDeLBaseModule: Initialized model with scaled configuration
        """
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
        Create a trainer instance for model training.

        Args:
            arguments: Training configuration and hyperparameters
            dataset_train: Training dataset
            dataset_eval: Optional evaluation dataset
            data_collator: Optional data collator for batching
            state: Model state to train

        Returns:
            BaseTrainer | Trainer: Configured trainer instance
        """
        return self.trainer_module(
            arguments=arguments,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
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
        model: EasyDeLBaseModule | None = None,
        state: EasyDeLState | None = None,
    ):
        """
        Execute distributed training with the configured model.

        This method handles model/state initialization from various sources:
        1. Provided state (highest priority)
        2. Provided model (converted to state)
        3. Checkpoint from bucket_path
        4. New model creation with scaling_index

        Args:
            scaling_index: Multiplier for model scaling (used if creating new model)
            arguments: Training configuration
            dataset_train: Training dataset
            dataset_eval: Optional evaluation dataset
            data_collator: Optional data collator
            model: Optional pre-initialized model
            state: Optional pre-initialized state

        Returns:
            Training results from the underlying trainer

        Notes:
            - For automatic resume from interruptions, set:
                - arguments.resume_if_possible = True
                - arguments.save_directory = "path/to/checkpoints"
            - State sharding is handled by the trainer based on partition rules
            - Checkpoint loading respects the priority order above

        Raises:
            AssertionError: If no valid model state can be obtained
        """
        if state is None and model is None:
            if self.bucket_path is not None:
                import easydel as ed

                state = self.state_class.load_state(
                    load_directory=self.bucket_path,
                    dtype=self.config_variables["dtype"],
                    param_dtype=self.config_variables["param_dtype"],
                    precision=self.config_variables["precision"],
                    auto_shard_model=True,
                    sharding_axis_names=self.config_variables["sharding_axis_names"],
                    sharding_axis_dims=self.config_variables["sharding_axis_dims"],
                    sharding_dcn_axis_dims=self.config_variables["sharding_dcn_axis_dims"],
                    config_kwargs=ed.EasyDeLBaseConfigDict(
                        freq_max_position_embeddings=self.config_variables["max_position_embeddings"],
                        mask_max_position_embeddings=self.config_variables["max_position_embeddings"],
                        kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
                        attn_mechanism=self.config_variables["attn_mechanism"],
                        attn_dtype=self.config_variables["attn_dtype"],
                        attn_softmax_dtype=self.config_variables["attn_softmax_dtype"],
                        gradient_checkpointing=self.config_variables["gradient_checkpointing"],
                        use_pallas_group_matmul=self.config_variables.get("use_pallas_group_matmul", False),
                    ),
                    partition_axis=self.config_variables["partition_axis"],
                    quantization_method=ed.EasyDeLQuantizationMethods.NONE,
                )
            else:
                logger.info(f"No model/state/checkpoint. Creating a new model (scaling_index={scaling_index}).")
                model = self.create_model_from_config(scaling_index=scaling_index)
                state = self.convert_model_to_state(model)

        elif model is not None and state is None:
            state = self.convert_model_to_state(model)

        assert state is not None, "Unable to obtain a valid model state."

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
                    s = str(v).replace("\n", "\n  ")
                    if len(s) > 200:
                        s = f"{v.__class__.__name__}(...)"
                    items.append(f"  {k} : {s}")
                except TypeError:
                    items.append(f"  {k} : <unrepresentable>")
        return f"{cls_name}(\n" + "\n".join(items) + "\n)"

    __str__ = __repr__
