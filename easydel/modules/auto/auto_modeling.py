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
from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NotRequired, Required, TypedDict, Unpack

import jax
import spectrax as spx
from eformer.paths import ePath
from jax import numpy as jnp
from spectrax import PartitionAxis

from easydel.infra.base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict, is_remote_url
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
from easydel.infra.factory import TaskType, registry
from easydel.infra.mixins.bridge import TENSORSTORE_INDEX_NAME
from easydel.layers import QuantizationConfig
from easydel.typings import typed_config

SAFETENSOR_INDEX_NAME = "tensorstore_index.json"
MODEL_INDEX_NAME = "model_structure.json"


def _normalize_pretrained_loading(self) -> None:
    if self.precision is None:
        self.precision = jax.lax.Precision("fastest")
    if self.partition_axis is None:
        self.partition_axis = PartitionAxis()


@typed_config(
    defaults={
        "tokenizer": None,
        "processor": None,
        "device": None,
        "dtype": jnp.float32,
        "param_dtype": jnp.float32,
        "precision": None,
        "sharding_axis_dims": (1, 1, -1, 1, 1, 1),
        "sharding_dcn_axis_dims": None,
        "sharding_axis_names": ("pp", "dp", "fsdp", "ep", "tp", "sp"),
        "partition_axis": None,
        "shard_fns": None,
        "backend": None,
        "platform": None,
        "config_kwargs": None,
        "auto_shard_model": True,
        "quantization_config": None,
        "apply_quantization": False,
        "verbose": True,
        "from_torch": None,
    },
    post_init=_normalize_pretrained_loading,
)
class PreTrainedLoading(TypedDict, total=False):
    """Unified loading config for pretrained EasyDeL models.

    Single source of truth for ``from_pretrained`` kwargs. Type checkers can use
    ``Unpack[PreTrainedLoading]`` to validate ``**kwargs`` at call sites; the
    runtime instance is a :class:`~easydel.typings.ConfigDict` that round-trips
    through ``from_dict`` / ``to_dict`` and supports both attribute and dict
    access.

    ``pretrained_model_name_or_path`` is the only required field — pass a
    model name/path or an already-loaded module. The field name matches
    :meth:`EasyDeLBaseModule.from_pretrained`'s positional argument so
    ``**config.to_dict()`` spreads cleanly into the underlying loader.
    ``tokenizer`` / ``processor`` are kept here for callers that bundle them
    with the loader config (e.g. eSurge); they are popped before forwarding to
    the underlying ``EasyDeLBaseModule`` loader.
    """

    pretrained_model_name_or_path: Required[Any]
    tokenizer: NotRequired[Any | None]
    processor: NotRequired[Any | None]
    device: NotRequired[jax.Device | None]  # type: ignore
    dtype: NotRequired[Any]
    param_dtype: NotRequired[Any]
    precision: NotRequired[jax.lax.Precision | None]
    sharding_axis_dims: NotRequired[Sequence[int]]
    sharding_dcn_axis_dims: NotRequired[Sequence[int] | None]
    sharding_axis_names: NotRequired[Sequence[str]]
    partition_axis: NotRequired[PartitionAxis | None]
    shard_fns: NotRequired[Mapping[tuple, Callable] | dict | None]
    backend: NotRequired[EasyDeLBackends | None]
    platform: NotRequired[EasyDeLPlatforms | None]
    config_kwargs: NotRequired[EasyDeLBaseConfigDict | None]
    auto_shard_model: NotRequired[bool]
    quantization_config: NotRequired[QuantizationConfig | None]
    apply_quantization: NotRequired[bool]
    verbose: NotRequired[bool]
    from_torch: NotRequired[bool | None]
    # Hugging Face Hub options forwarded to ``EasyDeLBaseModule.from_pretrained``.
    trust_remote_code: NotRequired[bool]
    cache_dir: NotRequired[str | os.PathLike | None]
    force_download: NotRequired[bool]
    local_files_only: NotRequired[bool]
    token: NotRequired[str | bool | None]
    revision: NotRequired[str]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["PreTrainedLoading"],
    ) -> "PreTrainedLoading": ...


class BaseAutoEasyModel:
    """Base class for all Auto EasyDeL model classes.

    Provides ``from_config`` and ``from_pretrained`` class methods. The latter
    accepts kwargs typed via :class:`PreTrainedLoading` —
    ``pretrained_model_name_or_path`` is required (a model name, path, or
    pre-loaded module).

    Attributes:
        model_task: The specific task the model class is designed for
            (e.g. ``TaskType.CAUSAL_LM``).
    """

    model_task: TaskType

    @classmethod
    def from_config(
        cls,
        config: EasyDeLBaseConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs | None = None,
    ) -> EasyDeLBaseModule:
        """Instantiate a model module directly from a configuration object."""
        registration = registry.get_module_registration(cls.model_task, config.model_type)
        if rngs is None:
            rngs = spx.Rngs(42)
        return registration.module(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    @classmethod
    def from_pretrained(cls, **kwargs: Unpack[PreTrainedLoading]) -> EasyDeLBaseModule:
        """Load and shard a pretrained model into an EasyDeL-compatible module.

        Args:
            **kwargs: Loading options. See :class:`PreTrainedLoading` for the
                full set of accepted fields. ``pretrained_model_name_or_path``
                is required (model name, path, or pre-loaded module).

        Returns:
            The loaded EasyDeL model module.
        """
        config = PreTrainedLoading.coerce_config(kwargs)

        if config.from_torch is None:
            try:
                config.from_torch = not cls._is_easydel(config.pretrained_model_name_or_path)
            except OSError as e:
                config.from_torch = "Error no file named easydel-model.parameters" in str(e)

        if config.from_torch:
            return cls._from_torch_pretrained(config)

        cmg = jax.default_device(config.device) if config.device is not None else contextlib.nullcontext()
        with cmg:
            return cls._from_easydel_params(config)

    # Fields on :class:`PreTrainedLoading` that must NOT flow into the
    # underlying ``EasyDeLBaseModule`` loader: ``device`` is consumed at this
    # layer (used as a JAX default-device context manager); ``from_torch`` is
    # a routing flag for picking between the two loaders; ``tokenizer`` /
    # ``processor`` are bundled here for callers like eSurge but aren't loader
    # arguments.
    _LOADER_PASSTHROUGH_DROP: tuple[str, ...] = (
        "device",
        "from_torch",
        "tokenizer",
        "processor",
    )

    @classmethod
    def _loader_kwargs(cls, config: PreTrainedLoading) -> dict[str, Any]:
        data = config.to_dict()
        for key in cls._LOADER_PASSTHROUGH_DROP:
            data.pop(key, None)
        return data

    @classmethod
    def _from_easydel_params(cls, config: PreTrainedLoading) -> EasyDeLBaseModule:
        """Load a model from EasyDeL-saved parameters using the resolved config."""

        class Base(EasyDeLBaseModule):
            _model_task = cls.model_task

        return Base.from_pretrained(**cls._loader_kwargs(config))

    @classmethod
    def _from_torch_pretrained(cls, config: PreTrainedLoading) -> EasyDeLBaseModule:
        """Load a model from PyTorch weights using the resolved config."""

        class Base(EasyDeLBaseModule):
            _model_task = cls.model_task

        return Base._from_torch_pretrained(**cls._loader_kwargs(config))

    @classmethod
    def _is_easydel(
        cls,
        pretrained_model_name_or_path,
        SPX_WEIGHTS_NAME="easydel-model.parameters",
        MULTI_PART_NAME="easydel-model.parameters.safetensors.index.json",
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
    ):
        """Check whether the given path/identifier points to an EasyDeL checkpoint."""
        from transformers.utils import cached_file as _cached_file

        proxies = None
        subfolder = ""
        commit_hash = None
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        epath = ePath(pretrained_model_name_or_path)
        if epath.is_dir():
            if (epath / subfolder / SAFETENSOR_INDEX_NAME).exists():
                return True
            if (epath / subfolder / MODEL_INDEX_NAME).exists():
                return True
            if (epath / subfolder / MULTI_PART_NAME).exists():
                return True
            if not (epath / subfolder / SPX_WEIGHTS_NAME).is_file():
                raise OSError(
                    f"Error no file named {SPX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}"
                )
            if oo := (epath / subfolder).glob("run-*"):
                if len(list(oo)) != 0:
                    return True

        elif (ePath(subfolder) / epath).is_file():
            ...
        elif is_remote_url(pretrained_model_name_or_path):
            ...
        else:
            try:
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "local_files_only": local_files_only,
                    "token": token,
                    "user_agent": {"file_type": "model", "from_auto_class": False},
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }
                resolved_archive_file = None
                for filename in [SPX_WEIGHTS_NAME, MULTI_PART_NAME, TENSORSTORE_INDEX_NAME, MODEL_INDEX_NAME]:
                    resolved_archive_file = _cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
                    if resolved_archive_file is not None:
                        break
                if resolved_archive_file is None:
                    resolved_archive_file = _cached_file(
                        pretrained_model_name_or_path,
                        MULTI_PART_NAME,
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is None:
                        return False
            except OSError:
                raise
            except Exception:
                return False
        return True


class BaseAutoEasyState:
    """Base class for Auto EasyDeL state classes.

    Provides ``from_config`` and ``from_pretrained`` class methods built on top
    of the corresponding :class:`BaseAutoEasyModel` subclass.

    Attributes:
        _base: The corresponding Auto EasyDeL model class.
    """

    _base: BaseAutoEasyModel

    @classmethod
    def from_config(
        cls,
        config: EasyDeLBaseConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs | None = None,
    ) -> EasyDeLState:
        """Create an :class:`EasyDeLState` directly from a configuration object."""
        return cls._base.from_config(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        ).to_state()

    @classmethod
    def from_pretrained(cls, **kwargs: Unpack[PreTrainedLoading]) -> EasyDeLState:
        """Load a pretrained model and wrap it in an :class:`EasyDeLState`.

        Accepts the same kwargs as :meth:`BaseAutoEasyModel.from_pretrained` —
        see :class:`PreTrainedLoading`.
        """
        model = cls._base.from_pretrained(**kwargs)
        return EasyDeLState.create(
            model=model,
            tx=None,
            init_opt_state=False,
            step=0,
        )


class AutoEasyDeLModelForCausalLM(BaseAutoEasyModel):
    """Auto loader for causal-LM models.

    Examples:

        >>> import jax
        >>> from easydel import AutoEasyDeLModelForCausalLM

        >>> # Load on a single CPU
        >>> model = AutoEasyDeLModelForCausalLM.from_pretrained(
        ...     pretrained_model_name_or_path="gpt2", device=jax.devices("cpu")[0]
        ... )

        >>> # Load sharded across 8 devices (DP + FSDP)
        >>> model = AutoEasyDeLModelForCausalLM.from_pretrained(
        ...     pretrained_model_name_or_path="gpt2",
        ...     sharding_axis_dims=(1, 1, 8, 1, 1, 1),
        ...     sharding_axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"),
        ...     device=jax.devices("cpu")[0],
        ...     from_torch=True,
        ... )
    """

    model_task: TaskType = TaskType.CAUSAL_LM


class AutoStateForCausalLM(BaseAutoEasyState):
    """Loads saved states for causal language modeling tasks."""

    _base = AutoEasyDeLModelForCausalLM


class AutoEasyDeLModelForDiffusionLM(BaseAutoEasyModel):
    """Auto loader for diffusion-based language models."""

    model_task: TaskType = TaskType.DIFFUSION_LM


class AutoStateForDiffusionLM(BaseAutoEasyState):
    """Loads saved states for diffusion-based language models."""

    _base = AutoEasyDeLModelForDiffusionLM


class AutoEasyDeLModelForZeroShotImageClassification(BaseAutoEasyModel):
    """Auto loader for zero-shot image classification models."""

    model_task: TaskType = TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION


class AutoStateForZeroShotImageClassification(BaseAutoEasyState):
    """Loads saved states for zero-shot image classification models."""

    _base = AutoEasyDeLModelForZeroShotImageClassification


class AutoEasyDeLModelForSpeechSeq2Seq(BaseAutoEasyModel):
    """Auto loader for speech sequence-to-sequence models.

    Examples:

        >>> import jax
        >>> from easydel import AutoEasyDeLModelForSpeechSeq2Seq

        >>> model = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
        ...     pretrained_model_name_or_path="openai/whisper-large-v3-turbo",
        ...     auto_shard_model=True,
        ... )

        >>> model = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
        ...     pretrained_model_name_or_path="openai/whisper-large-v3-turbo",
        ...     sharding_axis_dims=(1, 1, 8, 1, 1, 1),
        ...     sharding_axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"),
        ...     device=jax.devices("cpu")[0],
        ...     from_torch=True,
        ... )
    """

    model_task: TaskType = TaskType.SPEECH_SEQUENCE_TO_SEQUENCE


class AutoStateForSpeechSeq2Seq(BaseAutoEasyState):
    """Loads saved states for speech-to-text sequence-to-sequence models."""

    _base = AutoEasyDeLModelForSpeechSeq2Seq


class AutoEasyDeLModelForSeq2SeqLM(BaseAutoEasyModel):
    """Auto loader for text-to-text sequence-to-sequence models."""

    model_task: TaskType = TaskType.SEQUENCE_TO_SEQUENCE


class AutoStateForSeq2SeqLM(BaseAutoEasyState):
    """Loads saved states for text-to-text sequence-to-sequence models."""

    _base = AutoEasyDeLModelForSeq2SeqLM


class AutoEasyDeLModelForImageTextToText(BaseAutoEasyModel):
    """Auto loader for image-conditioned text-to-text models."""

    model_task: TaskType = TaskType.IMAGE_TEXT_TO_TEXT


class AutoStateForImageTextToText(BaseAutoEasyState):
    """Loads saved states for image-conditioned text-to-text models."""

    _base = AutoEasyDeLModelForImageTextToText


class AutoEasyDeLModelForSequenceClassification(BaseAutoEasyModel):
    """Auto loader for sequence classification models."""

    model_task: TaskType = TaskType.SEQUENCE_CLASSIFICATION


class AutoStateForImageSequenceClassification(BaseAutoEasyState):
    """Loads saved states for image-conditioned sequence classification."""

    _base = AutoEasyDeLModelForSequenceClassification


class AutoEasyDeLModelForEmbedding(BaseAutoEasyModel):
    """Auto loader for embedding models that produce dense vector representations.

    Loads models registered under ``TaskType.EMBEDDING`` and wraps them with
    pooling and optional L2 normalization for similarity-ready embeddings.
    Compatible with models like GTE-Qwen2, E5-Mistral, BGE, and others.
    """

    model_task: TaskType = TaskType.EMBEDDING


class AutoStateForEmbedding(BaseAutoEasyState):
    """Loads saved states for embedding models."""

    _base = AutoEasyDeLModelForEmbedding


class AutoEasyDeLModel(BaseAutoEasyModel):
    """Auto loader for generic text-only EasyDeL modules."""

    model_task: TaskType = TaskType.BASE_MODULE


class AutoState(BaseAutoEasyState):
    """Loads saved states for generic text-only EasyDeL modules."""

    _base = AutoEasyDeLModel


class AutoEasyDeLVisionModel(BaseAutoEasyModel):
    """Auto loader for vision-only EasyDeL modules."""

    model_task: TaskType = TaskType.BASE_VISION


class AutoStateVisionModel(BaseAutoEasyState):
    """Loads saved states for vision-only EasyDeL modules."""

    _base = AutoEasyDeLVisionModel


class AutoEasyDeLAnyToAnyModel(BaseAutoEasyModel):
    """Auto loader for generic models that map arbitrary input modalities to any output type."""

    model_task: TaskType = TaskType.ANY_TO_ANY


class AutoStateAnyToAnyModel(BaseAutoEasyModel):
    """Loads or builds states for the generic any-to-any EasyDeL modules."""

    _base = AutoEasyDeLAnyToAnyModel
