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
"""Auto model and state factories for EasyDeL.

This module defines the unified loading config :class:`PreTrainedLoading` and
the two base factories (:class:`BaseAutoEasyModel`, :class:`BaseAutoEasyState`)
that every ``AutoEasyDeLModelFor*`` / ``AutoStateFor*`` class subclasses.

Concrete factories registered in this file (each binds a specific
:class:`TaskType`):

- :class:`AutoEasyDeLModelForCausalLM` / :class:`AutoStateForCausalLM`
- :class:`AutoEasyDeLModelForDiffusionLM` / :class:`AutoStateForDiffusionLM`
- :class:`AutoEasyDeLModelForZeroShotImageClassification` /
  :class:`AutoStateForZeroShotImageClassification`
- :class:`AutoEasyDeLModelForSpeechSeq2Seq` /
  :class:`AutoStateForSpeechSeq2Seq`
- :class:`AutoEasyDeLModelForSeq2SeqLM` / :class:`AutoStateForSeq2SeqLM`
- :class:`AutoEasyDeLModelForImageTextToText` /
  :class:`AutoStateForImageTextToText`
- :class:`AutoEasyDeLModelForSequenceClassification` /
  :class:`AutoStateForImageSequenceClassification`
- :class:`AutoEasyDeLModelForEmbedding` / :class:`AutoStateForEmbedding`
- :class:`AutoEasyDeLModel` / :class:`AutoState` (generic text)
- :class:`AutoEasyDeLVisionModel` / :class:`AutoStateVisionModel`
- :class:`AutoEasyDeLAnyToAnyModel` / :class:`AutoStateAnyToAnyModel`

All ``from_pretrained`` entry points accept ``**kwargs: Unpack[PreTrainedLoading]``
— see :class:`PreTrainedLoading` for the full set of accepted fields.
"""

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
    """Fill in defaults for unset fields on a :class:`PreTrainedLoading` instance.

    Sets ``precision`` to ``jax.lax.Precision("fastest")`` and
    ``partition_axis`` to a default :class:`PartitionAxis()` when either is
    ``None``. Used as the ``post_init`` hook for the typed config.

    Args:
        self (PreTrainedLoading): The config instance being normalized.

    Returns:
        None: Modifies ``self`` in-place.
    """
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
    ) -> "PreTrainedLoading":
        """Build a :class:`PreTrainedLoading` from a mapping plus overrides.

        Stub injected by ``@typed_config``; the runtime implementation lives in
        :class:`~easydel.typings.ConfigDict`. Documented here for typing
        completeness.

        Args:
            data (Mapping[str, Any] | None, optional): Initial field values.
                Defaults to ``None``.
            **kwargs: Field overrides typed via :class:`PreTrainedLoading`.

        Returns:
            PreTrainedLoading: A fully populated config instance.
        """
        ...


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

        Routes between the EasyDeL-native loader and the PyTorch-import loader
        based on ``from_torch`` (auto-detected when not provided), then loads
        the module under an optional JAX default-device context.

        Args:
            **kwargs: Loading options forwarded to :class:`PreTrainedLoading`.
                Accepted fields:

                - ``pretrained_model_name_or_path`` (Any, required): Model name,
                  local path, or already-loaded module.
                - ``tokenizer`` (Any | None): Optional tokenizer bundled with
                  the loader; popped before forwarding to the underlying
                  loader. Defaults to ``None``.
                - ``processor`` (Any | None): Optional processor bundled with
                  the loader; popped before forwarding. Defaults to ``None``.
                - ``device`` (jax.Device | None): JAX default device used as a
                  context manager during loading. Defaults to ``None``.
                - ``dtype`` (Any): Compute dtype. Defaults to ``jnp.float32``.
                - ``param_dtype`` (Any): Parameter dtype. Defaults to
                  ``jnp.float32``.
                - ``precision`` (jax.lax.Precision | None): Matmul precision;
                  normalized to ``Precision("fastest")`` when ``None``.
                - ``sharding_axis_dims`` (Sequence[int]): Mesh shape across
                  ``(pp, dp, fsdp, ep, tp, sp)``. Defaults to
                  ``(1, 1, -1, 1, 1, 1)``.
                - ``sharding_dcn_axis_dims`` (Sequence[int] | None): Optional
                  outer DCN mesh shape. Defaults to ``None``.
                - ``sharding_axis_names`` (Sequence[str]): Mesh axis names.
                  Defaults to ``("pp", "dp", "fsdp", "ep", "tp", "sp")``.
                - ``partition_axis`` (PartitionAxis | None): Per-tensor sharding
                  policy; normalized to a default ``PartitionAxis()`` when
                  ``None``.
                - ``shard_fns`` (Mapping | dict | None): Optional pre-built
                  per-leaf shard functions. Defaults to ``None``.
                - ``backend`` (EasyDeLBackends | None): Backend for custom
                  kernels. Defaults to ``None``.
                - ``platform`` (EasyDeLPlatforms | None): Target platform.
                  Defaults to ``None``.
                - ``config_kwargs`` (EasyDeLBaseConfigDict | None): Extra
                  config overrides applied after load. Defaults to ``None``.
                - ``auto_shard_model`` (bool): Whether to shard the model
                  immediately on load. Defaults to ``True``.
                - ``quantization_config`` (QuantizationConfig | None): Optional
                  quantization spec. Defaults to ``None``.
                - ``apply_quantization`` (bool): Whether to apply quantization
                  during load. Defaults to ``False``.
                - ``verbose`` (bool): Verbose logging. Defaults to ``True``.
                - ``from_torch`` (bool | None): Force the PyTorch-import
                  loader. ``None`` triggers auto-detection. Defaults to
                  ``None``.
                - ``trust_remote_code`` (bool): Trust HF Hub remote code
                  modules during config loading.
                - ``cache_dir`` (str | os.PathLike | None): HF Hub cache dir.
                - ``force_download`` (bool): Force re-download from HF Hub.
                - ``local_files_only`` (bool): Disallow network access.
                - ``token`` (str | bool | None): HF Hub auth token.
                - ``revision`` (str): HF Hub revision/branch.

        Returns:
            EasyDeLBaseModule: The loaded EasyDeL model module.
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
        """Convert a :class:`PreTrainedLoading` to kwargs for the underlying loader.

        Drops the auto-layer-only fields listed in
        :attr:`_LOADER_PASSTHROUGH_DROP` (``device``, ``from_torch``,
        ``tokenizer``, ``processor``) so the inner ``EasyDeLBaseModule``
        loader sees only its accepted arguments.

        Args:
            config (PreTrainedLoading): The resolved auto-loader config.

        Returns:
            dict[str, Any]: Plain dict suitable for ``**kwargs`` spreading.
        """
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

        Forwards all kwargs to :meth:`BaseAutoEasyModel.from_pretrained`, then
        wraps the resulting module in an :class:`EasyDeLState` with no
        optimizer and ``step=0``.

        Args:
            **kwargs: Loading options forwarded to
                :meth:`BaseAutoEasyModel.from_pretrained`. See
                :class:`PreTrainedLoading` for the full field list.

        Returns:
            EasyDeLState: A state wrapping the freshly loaded module.
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
    """Auto state factory for causal-LM training/inference.

    Wraps :class:`AutoEasyDeLModelForCausalLM` and emits an
    :class:`EasyDeLState` (model + optimizer placeholder + step counter) ready
    to be threaded into a trainer or eSurge runner.
    """

    _base = AutoEasyDeLModelForCausalLM


class AutoEasyDeLModelForDiffusionLM(BaseAutoEasyModel):
    """Auto loader for diffusion-style discrete language models.

    Resolves the registered module for ``TaskType.DIFFUSION_LM``. Diffusion
    LMs (e.g. SEDD-style) replace the autoregressive decoder with a denoising
    head over masked / corrupted tokens; the factory honours the same
    ``PreTrainedLoading`` field set as the causal-LM variant.
    """

    model_task: TaskType = TaskType.DIFFUSION_LM


class AutoStateForDiffusionLM(BaseAutoEasyState):
    """Auto state factory for diffusion-LM training/inference.

    Same lifecycle as :class:`AutoStateForCausalLM` but parametrised by the
    diffusion-LM task registry.
    """

    _base = AutoEasyDeLModelForDiffusionLM


class AutoEasyDeLModelForZeroShotImageClassification(BaseAutoEasyModel):
    """Auto loader for zero-shot image-classification models (CLIP family).

    Loads modules registered under
    ``TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION``: dual-encoder vision-language
    models that score arbitrary class prompts against an image embedding
    (e.g. CLIP). The vision tower's pixel input contract and the text
    tokenizer match the upstream HF processor; pass it through
    ``processor=...`` if eSurge / inference layers need it.
    """

    model_task: TaskType = TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION


class AutoStateForZeroShotImageClassification(BaseAutoEasyState):
    """Auto state factory for zero-shot image-classification models.

    Wraps :class:`AutoEasyDeLModelForZeroShotImageClassification` to produce
    an :class:`EasyDeLState`.
    """

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
    """Auto state factory for speech-to-text seq2seq models.

    Wraps :class:`AutoEasyDeLModelForSpeechSeq2Seq` and produces an
    :class:`EasyDeLState`.
    """

    _base = AutoEasyDeLModelForSpeechSeq2Seq


class AutoEasyDeLModelForSeq2SeqLM(BaseAutoEasyModel):
    """Auto loader for text-to-text seq2seq models (encoder-decoder).

    Resolves the module registered under ``TaskType.SEQUENCE_TO_SEQUENCE`` —
    architectures that take text on the encoder side and generate text on the
    decoder side (T5, BART, mT5, …). Generation entry points expect the
    standard ``input_ids`` / ``decoder_input_ids`` contract.
    """

    model_task: TaskType = TaskType.SEQUENCE_TO_SEQUENCE


class AutoStateForSeq2SeqLM(BaseAutoEasyState):
    """Auto state factory for text-to-text seq2seq models.

    Wraps :class:`AutoEasyDeLModelForSeq2SeqLM`.
    """

    _base = AutoEasyDeLModelForSeq2SeqLM


class AutoEasyDeLModelForImageTextToText(BaseAutoEasyModel):
    """Auto loader for image-conditioned text generation models.

    Resolves the module registered under ``TaskType.IMAGE_TEXT_TO_TEXT``:
    multimodal models that take pixel inputs alongside a text prompt and
    emit text (LLaVA / Aya-Vision / Idefics-style). The vision tower's pixel
    contract and the language-side tokenizer follow the upstream HF
    processor; route it via ``processor=...`` so eSurge can reuse it.
    """

    model_task: TaskType = TaskType.IMAGE_TEXT_TO_TEXT


class AutoStateForImageTextToText(BaseAutoEasyState):
    """Auto state factory for image-conditioned text-generation models."""

    _base = AutoEasyDeLModelForImageTextToText


class AutoEasyDeLModelForSequenceClassification(BaseAutoEasyModel):
    """Auto loader for sequence-classification models.

    Resolves modules registered under ``TaskType.SEQUENCE_CLASSIFICATION``:
    a transformer backbone with a pooled classification head emitting
    ``num_labels`` logits per sequence. Use ``config_kwargs={"num_labels": N}``
    to override the head width at load time.
    """

    model_task: TaskType = TaskType.SEQUENCE_CLASSIFICATION


class AutoStateForImageSequenceClassification(BaseAutoEasyState):
    """Auto state factory for the sequence-classification task family.

    Despite the historical class name, this maps onto
    :class:`AutoEasyDeLModelForSequenceClassification` and is reused for
    any sequence-level classification head (text or image-conditioned).
    """

    _base = AutoEasyDeLModelForSequenceClassification


class AutoEasyDeLModelForEmbedding(BaseAutoEasyModel):
    """Auto loader for embedding models that produce dense vector representations.

    Loads models registered under ``TaskType.EMBEDDING`` and wraps them with
    pooling and optional L2 normalization for similarity-ready embeddings.
    Compatible with models like GTE-Qwen2, E5-Mistral, BGE, and others.
    """

    model_task: TaskType = TaskType.EMBEDDING


class AutoStateForEmbedding(BaseAutoEasyState):
    """Auto state factory for embedding models.

    Wraps :class:`AutoEasyDeLModelForEmbedding`.
    """

    _base = AutoEasyDeLModelForEmbedding


class AutoEasyDeLModel(BaseAutoEasyModel):
    """Auto loader for the generic text-only ``BASE_MODULE`` task.

    Returns the bare backbone (no LM head, no classification head) registered
    under ``TaskType.BASE_MODULE`` — useful for representation extraction or
    for callers that want to attach their own task head on top.
    """

    model_task: TaskType = TaskType.BASE_MODULE


class AutoState(BaseAutoEasyState):
    """Auto state factory for the generic ``BASE_MODULE`` task.

    Wraps :class:`AutoEasyDeLModel` and produces an :class:`EasyDeLState`.
    """

    _base = AutoEasyDeLModel


class AutoEasyDeLVisionModel(BaseAutoEasyModel):
    """Auto loader for vision-only backbones.

    Resolves modules registered under ``TaskType.BASE_VISION``: ViT-style
    encoders that consume ``[batch, channels, H, W]`` pixels (or
    ``[batch, num_patches, hidden]`` patch embeddings) and produce per-patch
    or pooled visual features without a language model attached.
    """

    model_task: TaskType = TaskType.BASE_VISION


class AutoStateVisionModel(BaseAutoEasyState):
    """Auto state factory for vision-only backbones.

    Wraps :class:`AutoEasyDeLVisionModel`.
    """

    _base = AutoEasyDeLVisionModel


class AutoEasyDeLAnyToAnyModel(BaseAutoEasyModel):
    """Auto loader for the generic ``ANY_TO_ANY`` task family.

    Catch-all factory for models that don't fit a fixed input/output modality
    pair (e.g. unified multimodal generators with text + image + audio
    streams). Resolves the module registered under ``TaskType.ANY_TO_ANY``.
    """

    model_task: TaskType = TaskType.ANY_TO_ANY


class AutoStateAnyToAnyModel(BaseAutoEasyModel):
    """Auto state factory for the generic ``ANY_TO_ANY`` task family.

    Despite extending :class:`BaseAutoEasyModel` historically (rather than
    :class:`BaseAutoEasyState`), this class is consumed in state-creation
    flows; ``_base`` points at :class:`AutoEasyDeLAnyToAnyModel`.
    """

    _base = AutoEasyDeLAnyToAnyModel
