# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Ray-based distributed trainer implementation for EasyDeL.

This module provides a distributed training implementation using Ray for scaling
language model training across multiple GPUs and nodes. It integrates Ray's
distributed computing capabilities with EasyDeL's training infrastructure to
enable efficient large-scale model training.

The module includes:
- RayDistributedTrainer: Main class for distributed training with Ray
- Integration with Ray Train for distributed data loading and gradient synchronization
- Support for both data and model parallelism strategies
- Automatic resource management and fault tolerance
- Checkpointing and recovery mechanisms for long-running training jobs

Key Components:
- Automatic distribution of training data across workers
- Gradient synchronization using Ray's collective communication
- Dynamic resource allocation and load balancing
- Integration with Ray Tune for hyperparameter optimization
- Support for heterogeneous hardware configurations

The trainer abstracts away the complexity of distributed training, allowing users
to scale from single GPU to multi-node clusters with minimal code changes.
"""

from __future__ import annotations

import copy
import json
import os
import typing as tp
from functools import cached_property

import jax
import spectrax as spx
from eformer.loggings import get_logger
from eformer.mpric import DTYPE_TO_STRING_MAP, STRING_TO_DTYPE_MAP
from eformer.paths import ePath
from jax import lax
from jax import numpy as jnp
from pydantic import BaseModel
from spectrax import PartitionAxis
from transformers import AutoTokenizer, PreTrainedTokenizer

from easydel.infra import EasyDeLBaseConfig, EasyDeLBaseModule, EasyDeLState
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import TaskType
from easydel.modules.auto.auto_configuration import get_modules_by_type
from easydel.utils import Registry

from ..base_trainer import BaseTrainer
from ..trainer.trainer import Trainer
from ..training_configurations import TrainingArguments

if tp.TYPE_CHECKING:
    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

logger = get_logger("RayTrainer")


@Registry.register("trainer-arguments", "ray_dist")
class RayDistributedConfig(BaseModel):
    """JSON-persistable configuration payload for :class:`RayDistributedTrainer`.

    Captures the minimum identifying state of a Ray-orchestrated trainer (model
    identity, scaling variables, fixed runtime knobs) so that a checkpoint of
    the trainer object itself can be round-tripped through disk. Two preprocess
    hooks (:meth:`_saving_preprocess` and :meth:`_loading_postprocess`) bridge
    the JAX-native runtime types (JAX dtypes, :class:`PartitionAxis`) to and
    from the JSON-safe primitives that Pydantic can serialise natively.

    Attributes:
        pretrained_model_name_or_path (str): Path or identifier for the
            pretrained model.
        model_task (TaskType | None): Task type for the model (for example
            ``CAUSAL_LM``, ``SEQ2SEQ``).
        model_type (str | None): Model architecture type (for example
            ``'llama'``, ``'gpt2'``).
        offload_backend (str | None): Backend device for offloading (for
            example ``'cpu'``, ``'gpu'``).
        config_scaling_variables (dict[str, int] | None): Variables to scale
            by ``scaling_index`` (for example ``hidden_size``).
        config_variables (dict[str, tp.Any] | None): Fixed configuration
            variables (for example ``dtype``, ``precision``).

    Note:
        JAX dtype fields are converted to/from strings for JSON serialization,
        and :class:`PartitionAxis` objects are converted to/from a dictionary
        representation. Use :meth:`_saving_preprocess` before saving and
        :meth:`_loading_postprocess` after loading.
    """

    pretrained_model_name_or_path: str
    model_task: TaskType | None = None
    model_type: str | None = None
    offload_backend: str | None = None
    config_scaling_variables: dict[str, int] | None = None
    config_variables: dict[str, tp.Any] | None = None

    def _saving_preprocess(self):
        """Replace non-JSON-friendly values in the config dicts with string surrogates.

        Pydantic dumps ``RayDistributedConfig`` to JSON, but two of the
        nested values are not naturally JSON-encodable:

        * **JAX dtypes** stored in ``config_variables`` /
          ``config_scaling_variables``. Each occurrence is replaced with
          its canonical string representation from
          ``DTYPE_TO_STRING_MAP`` (e.g. ``jnp.bfloat16`` becomes
          ``"bfloat16"``). The reverse mapping happens in
          :meth:`_loading_postprocess`.
        * **PartitionAxis** under
          ``config_variables["partition_axis"]``. The dataclass instance
          is converted to its ``__dict__`` so Pydantic can serialise the
          plain field-name -> axis mapping.

        The mutation is in-place so subsequent calls to
        ``model_dump_json`` see the JSON-safe payload. Symmetrically
        :meth:`_loading_postprocess` must be invoked after deserialising
        to restore live objects.
        """
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
        """Reverse :meth:`_saving_preprocess` after JSON deserialisation.

        After parsing the on-disk JSON Pydantic returns plain Python
        primitives (strings for dtypes, dicts for ``PartitionAxis``).
        This hook converts them back into the live runtime values that
        :class:`RayDistributedTrainer` expects:

        * dtype strings in ``config_variables`` / ``config_scaling_variables``
          are reverse-looked-up through ``STRING_TO_DTYPE_MAP``;
        * ``config_variables["partition_axis"]`` -- if it is still a
          dict -- is rebuilt into a real :class:`PartitionAxis` via
          ``PartitionAxis(**dict)``.

        The mutation is in-place. The method is idempotent: already-live
        values pass through untouched because they are not present in
        ``DTYPE_TO_STRING_MAP.values()`` and are already
        ``PartitionAxis`` instances.
        """
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
    """Lightweight Ray-aware wrapper that drives a stock EasyDeL :class:`Trainer`.

    This class is intentionally *not* a :class:`BaseTrainer` subclass; it
    composes one. Its responsibility is the small set of decisions that must
    happen before training begins -- selecting the model class from the
    registry, building / scaling a model config, materialising a state (either
    fresh or from a checkpoint at ``bucket_path``), and constructing the
    underlying :class:`Trainer` once. After that, all loop logic, resume
    handling, and sharding decisions live on the inner trainer.

    Design choices encoded here:

    * Resume logic is owned by :class:`BaseTrainer` (set
      ``arguments.resume_if_possible=True``); this class does not perform
      ``run-*`` directory resolution.
    * State sharding is deferred to the inner trainer according to its
      ``PartitionAxis`` rules; no manual ``with_sharding_constraint`` here.
    * Checkpoint paths must be explicit -- the wrapper does no automatic
      directory probing.

    Attributes:
        model_task (TaskType): Task type for the model (for example
            ``CAUSAL_LM``).
        model_type (str): Model architecture type (for example ``'llama'``).
        model_class (type[EasyDeLBaseModule]): The EasyDeL model class to
            instantiate.
        state_class (type[EasyDeLState]): State class used for
            checkpointing and to wrap the freshly built model.
        offload_backend (str): Backend identifier for memory offloading.
        trainer_module (type[BaseTrainer | Trainer]): Trainer class used to
            run the actual loop.
        CONFIG_SCALING_VARIABLES (ClassVar[dict[str, int]]): Defaults for
            the per-axis size knobs that get multiplied by
            ``scaling_index``.
        CONFIG_VARIABLES (ClassVar[dict[str, tp.Any]]): Fixed configuration
            variables that do not scale with ``scaling_index``.
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
        "attn_mechanism": "auto",
        "attn_dtype": jnp.bfloat16,
        "attn_softmax_dtype": jnp.bfloat16,
        "sharding_axis_names": ("pp", "dp", "fsdp", "ep", "tp", "sp"),
        "sharding_axis_dims": (1, 1, -1, 1, 1, 1),
        "sharding_dcn_axis_dims": (1, 1, -1, 1, 1, 1),
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
        """Initialize the Ray-distributed trainer wrapper.

        Args:
            pretrained_model_name_or_path: Path or identifier for the pretrained
                model.
            bucket_path: Optional path used to load a checkpoint from cloud
                storage (or any :class:`ePath`-resolvable location) when
                neither ``model`` nor ``state`` is supplied to :meth:`train`.
            model_task: Task type. Inferred from ``model_class`` when omitted;
                must then be ``None`` together with ``model_type``.
            model_type: Model architecture type. Inferred from ``model_class``
                when omitted; must then be ``None`` together with ``model_task``.
            model_class: EasyDeL model class to instantiate. When omitted, the
                class is resolved through :func:`get_modules_by_type` using
                ``model_type`` and ``model_task``.
            state_class: State class used for checkpointing and ``model.to_state``
                conversion. Defaults to :class:`EasyDeLState`.
            offload_backend: Backend identifier used when offloading parameters
                (passed to :func:`jax.devices` / :func:`jax.local_devices`).
                Defaults to ``'cpu'``.
            trainer_module: Inner trainer class to instantiate at
                :meth:`create_trainer`. Defaults to :class:`Trainer`.
            config_scaling_variables: Per-axis size knobs that override the
                class-level ``CONFIG_SCALING_VARIABLES`` defaults; values get
                multiplied by ``scaling_index`` at :meth:`create_config`.
            config_variables: Fixed configuration variables overriding the
                class-level ``CONFIG_VARIABLES`` defaults.

        Raises:
            ValueError: If exactly one of ``model_task``/``model_type`` is
                ``None``, or if ``model_class`` is ``None`` while
                ``model_task``/``model_type`` are also ``None``.
            RuntimeError: If :func:`get_modules_by_type` cannot locate a model
                class for ``model_type``/``model_task``.
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if model_task is None or model_type is None:
            if not (model_task is None and model_type is None):
                raise ValueError("If one of model_task or model_type is None, both must be None.")
            if model_class is None:
                raise ValueError("model_class must be provided when model_task/model_type are omitted.")
            model_type = model_class._model_type
            model_task = model_class._model_task
        elif model_class is not None:
            logger.warning(
                "Both model_class and model_type/model_task provided. Using model_class and inferring type/task from it."
            )
            model_type = model_class._model_type
            model_task = model_class._model_task

        if model_class is None:
            if model_type is None or model_task is None:
                raise ValueError("model_type and model_task must be provided if model_class is not specified.")
            _, resolved_class = get_modules_by_type(model_type=model_type, task_type=model_task)
            if resolved_class is None:
                raise RuntimeError(f"Could not resolve model class for {model_type}/{model_task}")
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
        """Construct a :class:`RayDistributedTrainer` from a saved JSON config.

        Reads the file via :class:`ePath`, parses it into
        :class:`RayDistributedConfig`, runs :meth:`_loading_postprocess` to
        restore live JAX dtypes and :class:`PartitionAxis`, then instantiates
        the trainer with the recovered fields plus any per-call overrides.

        Args:
            path: Path to the JSON configuration file.
            model_class: Optional model class override applied after the
                config is parsed.
            state_class: Optional state class override.
            trainer_module: Optional trainer-module override.

        Returns:
            RayDistributedTrainer: Initialized trainer instance.
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
        """Serialise the trainer's configuration to a JSON file.

        Builds a :class:`RayDistributedConfig` from the trainer's current
        fields, runs :meth:`_saving_preprocess` to coerce JAX dtypes and
        :class:`PartitionAxis` into JSON-safe primitives, and writes the
        indented JSON dump to ``path`` via :class:`ePath`.

        Args:
            path: Destination path where the JSON configuration will be
                written.
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
        """Load the tokenizer/processor for the wrapped model.

        Uses ``_processor_loader_class.from_pretrained`` (defaults to
        :class:`AutoTokenizer`) against
        ``self.pretrained_model_name_or_path``. When the resulting tokenizer
        has no ``pad_token_id`` but does expose ``eos_token_id``, the EOS
        token id is reused for padding and a warning is logged.

        Returns:
            PreTrainedTokenizer: Loaded tokenizer with a guaranteed padding
            configuration.
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
        """Tokenizer/processor for the model, loaded once per trainer instance.

        Backed by :func:`functools.cached_property`: the first access
        triggers :meth:`load_processor` (which downloads or loads the
        :class:`PreTrainedTokenizer` and reconciles ``pad_token_id``);
        subsequent accesses on the same trainer return the cached
        object without re-downloading. The cache lives on the instance,
        so different :class:`RayDistributedTrainer` instances do not
        share processors.
        """
        return self.load_processor()

    @staticmethod
    def extract_column_names(dataset: Dataset) -> list[str] | None:
        """Return the column names of a dataset, falling back to a sample probe.

        Prefers ``dataset.column_names`` when defined. If not, materialises the
        first sample to probe its keys. Returns ``None`` for empty or
        opaque datasets.

        Args:
            dataset: The dataset to inspect.

        Returns:
            list[str] | None: List of column names when discoverable; ``None``
            when the dataset is empty or exposes no schema.
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
        """Tokenize and pad a raw text sample into flat model inputs.

        Runs :attr:`processor` with ``padding="max_length"`` and truncation,
        then flattens any returned 2-D arrays to 1-D so a single sample fits
        into the per-step batch layout expected by the trainer.

        Args:
            sample: Raw text sample (or sequence of samples) to process.
            max_length: Maximum sequence length used for both padding and
                truncation.
            padding_side: Side to pad sequences on (``'left'`` or
                ``'right'``).

        Returns:
            dict[str, jax.Array]: Tokenizer outputs with values reshaped to
            ``(-1,)`` when the original value carries a ``shape`` attribute.
        """
        out = self.processor(
            sample,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
            padding_side=padding_side,
            return_attention_mask=True,
            truncation=True,
        )
        return {k: v.reshape(-1) if hasattr(v, "shape") else v for k, v in out.items()}

    def process_messages_data(
        self,
        messages: tp.Any,
        max_length: int,
        padding_side: str = "left",
    ) -> dict[str, jax.Array]:
        """Apply the chat template and flatten the resulting tensor inputs.

        Calls ``processor.apply_chat_template`` with ``return_dict=True`` so
        the chat-formatted text is tokenised in one pass, then flattens any
        returned 2-D arrays to 1-D so a single sample fits into the per-step
        batch layout.

        Args:
            messages: Chat messages (list of role/content dicts) to process.
            max_length: Maximum sequence length used for both padding and
                truncation.
            padding_side: Side to pad sequences on (``'left'`` or
                ``'right'``).

        Returns:
            dict[str, jax.Array]: Tokenizer outputs with values reshaped to
            ``(-1,)`` when the original value carries a ``shape`` attribute.
        """
        out = self.processor.apply_chat_template(
            messages,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
            padding_side=padding_side,
            return_dict=True,
            truncation=True,
        )
        return {k: v.reshape(-1) if hasattr(v, "shape") else v for k, v in out.items()}

    def create_config(self, scaling_index: int) -> EasyDeLBaseConfig:
        """Build a config object whose width axes scale linearly with an index.

        Multiplies each entry of ``self.config_scaling_variables`` by
        ``scaling_index`` and merges the result with the fixed entries of
        ``self.config_variables`` (excluding ``precision``, ``dtype``, and
        ``param_dtype``, which are reserved for the model-builder call). The
        chosen config class is taken from ``self.model_class.config_class`` and
        falls back to :func:`get_modules_by_type` resolution when the class
        attribute is unset.

        Args:
            scaling_index: Multiplier applied to every entry of
                ``config_scaling_variables`` (for example to fan out
                ``hidden_size``).

        Returns:
            EasyDeLBaseConfig: Configuration object populated with scaled
            width parameters and the fixed runtime entries.
        """
        not_allowed = ["precision", "dtype", "param_dtype"]
        scaled = {k: v * scaling_index for k, v in copy.deepcopy(self.config_scaling_variables).items()}
        config_kwargs = {**{k: v for k, v in self.config_variables.items() if k not in not_allowed}, **scaled}
        config_class: type | None = self.model_class.config_class
        if config_class is None:
            config_class, _ = get_modules_by_type(model_type=self.model_type, task_type=self.model_task)
        return config_class(**config_kwargs)

    def _get_offload_device(self):
        """Return the JAX device to use for parameter offloading.

        Prefers ``jax.local_devices(backend=self.offload_backend)[0]`` because
        local devices avoid extra cross-host transfers. Any failure (no local
        devices, backend unsupported) falls back to
        ``jax.devices(self.offload_backend)[0]``.

        Returns:
            jax.Device: Preferred local device, or the first available global
            device for ``self.offload_backend`` when no local device is
            present.
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
        """Instantiate the wrapped model class from a configuration object.

        Dispatches to ``self.model_class.lazy_init`` (when ``lazy=True``) or
        ``sequential_init`` otherwise. ``precision`` defaults to
        :attr:`lax.Precision.DEFAULT` when ``None``.

        Args:
            config: Model configuration consumed by the model class.
            dtype: Computation dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX precision setting. ``None`` is normalised to
                ``Precision.DEFAULT``.
            seed: Random seed used to build ``spx.Rngs``.
            lazy: When True, build the module with lazy initialisation
                (parameters are :class:`jax.ShapeDtypeStruct`).

        Returns:
            EasyDeLBaseModule: Initialised model instance.
        """
        if precision is None:
            precision = lax.Precision.DEFAULT

        init_kwargs = dict(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=spx.Rngs(seed),
        )

        if lazy:
            return self.model_class.lazy_init(**init_kwargs)
        return self.model_class.sequential_init(**init_kwargs)

    def convert_model_to_state(self, model: EasyDeLBaseModule) -> EasyDeLState:
        """Wrap a model module in an :class:`EasyDeLState` using ``state_class``.

        Sharding is deliberately *not* applied here; the inner :class:`Trainer`
        owns sharding decisions according to its partition rules.
        ``self.arguments.trainable_selector`` is forwarded so non-trainable
        leaves are routed correctly.

        Args:
            model: The model module to convert.

        Returns:
            EasyDeLState: State object built via
            ``model.to_state(self.state_class, ...)``.
        """
        return model.to_state(self.state_class, trainable_selector=self.arguments.trainable_selector)

    def create_model_from_config(self, scaling_index: int) -> EasyDeLBaseModule:
        """Convenience helper that chains :meth:`create_config` and :meth:`create_model`.

        Reads ``dtype``, ``param_dtype``, ``precision`` and ``seed`` from
        ``self.config_variables`` so the model is built with the same runtime
        settings that the saved config encodes.

        Args:
            scaling_index: Multiplier applied to the width-related entries of
                ``config_scaling_variables``.

        Returns:
            EasyDeLBaseModule: Initialised model built from the scaled config.
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
        """Instantiate the inner trainer class with the given training inputs.

        Forwards ``arguments``, datasets, collator, and ``state`` to
        ``self.trainer_module(...)`` without further preprocessing -- the inner
        trainer owns sharding, scheduling, and resume.

        Args:
            arguments: Training configuration and hyperparameters.
            dataset_train: Training dataset.
            dataset_eval: Optional evaluation dataset.
            data_collator: Optional data collator for batching.
            state: Model state to train.

        Returns:
            BaseTrainer | Trainer: Configured trainer instance ready to call
            ``.train()`` on.
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
        """Resolve a training state from available sources and launch training.

        Model/state acquisition follows this priority order:

        1. Provided ``state`` (used directly).
        2. Provided ``model`` (converted to state via
           :meth:`convert_model_to_state`).
        3. Checkpoint loaded from ``self.bucket_path`` via
           ``state_class.load_state`` when set.
        4. Freshly built model from :meth:`create_model_from_config` with the
           given ``scaling_index``.

        For automatic resume from interruption, set
        ``arguments.resume_if_possible = True`` and ``arguments.save_directory``
        on the inner trainer; this method does not perform run-directory
        probing.

        Args:
            scaling_index: Multiplier used by :meth:`create_model_from_config`
                when a new model needs to be created.
            arguments: Training configuration forwarded to the inner trainer.
            dataset_train: Training dataset.
            dataset_eval: Optional evaluation dataset.
            data_collator: Optional data collator.
            model: Optional pre-initialised model (converted to state when
                ``state`` is also ``None``).
            state: Optional pre-initialised state taking highest priority.

        Returns:
            The return value of ``self.create_trainer(...).train()``.

        Raises:
            RuntimeError: If no valid model state can be obtained from any of
                the four sources.
        """
        self.arguments = arguments
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
                    config_kwargs=ed.EasyDeLBaseConfigDict(  # pyright: ignore[reportPrivateLocalImportUsage]
                        freq_max_position_embeddings=self.config_variables["max_position_embeddings"],
                        mask_max_position_embeddings=self.config_variables["max_position_embeddings"],
                        attn_mechanism=self.config_variables["attn_mechanism"],
                        attn_dtype=self.config_variables["attn_dtype"],
                        attn_softmax_dtype=self.config_variables["attn_softmax_dtype"],
                        gradient_checkpointing=self.config_variables["gradient_checkpointing"],
                    ),
                    partition_axis=self.config_variables["partition_axis"],
                )
            else:
                logger.info(f"No model/state/checkpoint. Creating a new model (scaling_index={scaling_index}).")
                model = self.create_model_from_config(scaling_index=scaling_index)
                state = self.convert_model_to_state(model)

        elif model is not None and state is None:
            state = self.convert_model_to_state(model)

        if state is None:
            raise RuntimeError("Unable to obtain a valid model state.")

        return self.create_trainer(
            arguments=arguments,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
            data_collator=data_collator,
            state=state,
        ).train()

    def __repr__(self):
        """Return a multi-line, human-readable representation of the trainer.

        Long attribute values are truncated to keep the output readable.

        Returns:
            str: Indented description showing the public attributes and their
            (possibly truncated) string values.
        """
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
