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
"""Auto-discovery utilities for EasyDeL configurations and shard/gather plans.

This module exposes:

- :func:`get_modules_by_type` — resolve ``(config_cls, module_cls)`` from a
  registered ``model_type`` and :class:`TaskType`.
- :func:`infer_task_from_hf_config` — read a HuggingFace ``config.json`` (local
  or hub) and map ``architectures[0]`` to a :class:`TaskType`.
- :func:`normalize_task` — normalize loose strings/aliases to :class:`TaskType`.
- :class:`AutoEasyDeLConfig` — load the correct ``EasyDeLBaseConfig`` subclass
  for a HuggingFace identifier or local path.
- :class:`AutoShardAndGatherFunctions` — build per-leaf shard/gather closures
  from a config, a lazy-initialized module, or raw params + mesh; used to move
  HF/PyTorch tensors onto the JAX device mesh and back.
"""

import json
import typing as tp
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import jax
import spectrax as spx
from eformer.loggings import get_logger
from eformer.paths import ePath
from jax.sharding import PartitionSpec
from spectrax import PartitionAxis, make_shard_and_gather_fns

from easydel.infra.base_module import EasyDeLBaseConfig, EasyDeLBaseModule
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
from easydel.infra.factory import TaskType, registry
from easydel.infra.sharding import MeshLike, replicated_named_sharding
from easydel.utils.instrumentation import phase_timer
from easydel.utils.traversals import flatten_dict, unflatten_dict

logger = get_logger(name=__name__)

# Tagged variant of the shared phase timer; preserves the prior log prefix
# (``[AutoShardAndGatherFunctions] ...``) at every existing call site.
_shardgen_phase = partial(phase_timer, tag="AutoShardAndGatherFunctions")


def get_modules_by_type(
    model_type: str,
    task_type: TaskType,
) -> tuple[type[EasyDeLBaseConfig], type[EasyDeLBaseModule] | tp.Any]:
    """Resolve a registered ``(config, module)`` pair for ``(model_type, task)``.

    Looks the registration up in :data:`easydel.infra.factory.registry`, the
    same registry populated by :func:`register_config` and
    :func:`register_module` decorators on each model file.

    Args:
        model_type (str): HuggingFace-style ``model_type`` string (e.g.
            ``"llama"``, ``"deepseek_v3"``).
        task_type (TaskType): Task slot to fetch (``CAUSAL_LM``,
            ``SEQUENCE_CLASSIFICATION``, …).

    Returns:
        tuple: ``(config_cls, module_cls)`` — for example,
        ``(LlamaConfig, LlamaForCausalLM)``.

    Raises:
        KeyError: If no registration exists for the requested pair.
    """
    registred_module = registry.get_module_registration(
        task_type=task_type,
        model_type=model_type,
    )
    return (registred_module.config, registred_module.module)


def is_flatten(pytree: dict):
    """Detect whether ``pytree`` is in the flattened-keys form used here.

    The shard/gather code paths above represent pytrees in two equivalent
    layouts: nested dicts (``{"layers": {"0": {"q_proj": ...}}}``) and
    flattened dicts whose keys are tuples of path components
    (``{("layers", "0", "q_proj"): ...}``). The check inspects the first key
    only — both flatten / unflatten in this module guarantee homogeneous keys.

    Args:
        pytree (dict): Either a nested or flattened pytree dict.

    Returns:
        bool: ``True`` if the first key is a ``tuple`` (flattened form),
        ``False`` otherwise.
    """
    mpl = next(iter(pytree.keys()))
    return True if isinstance(mpl, tuple) else False


TASK_ALIASES: dict[str, TaskType] = {
    "causal_lm": TaskType.CAUSAL_LM,
    "lm": TaskType.CAUSAL_LM,
    "seq2seq": TaskType.SEQUENCE_TO_SEQUENCE,
    "sequence_to_sequence": TaskType.SEQUENCE_TO_SEQUENCE,
    "speech_seq2seq": TaskType.SPEECH_SEQUENCE_TO_SEQUENCE,
    "image_text_to_text": TaskType.IMAGE_TEXT_TO_TEXT,
    "zero_shot_image_classification": TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION,
    "diffusion_lm": TaskType.DIFFUSION_LM,
    "embedding": TaskType.EMBEDDING,
    "base": TaskType.BASE_MODULE,
}


def normalize_task(t: TaskType | str | None) -> TaskType | None:
    """Normalize task type specification to TaskType enum.

    Handles string aliases, case variations, and hyphen/underscore differences.

    Args:
        t: Task type specification (TaskType, string alias, or None)

    Returns:
        Normalized TaskType or None if not recognized

    Example:
        >>> normalize_task("causal-lm")
        <TaskType.CAUSAL_LM: 'causal_lm'>
        >>> normalize_task("LM")
        <TaskType.CAUSAL_LM: 'causal_lm'>
    """
    if t is None:
        return None
    if isinstance(t, TaskType):
        return t
    return TASK_ALIASES.get(str(t).strip().lower().replace("-", "_"))


def infer_task_from_hf_config(model_name_or_path: str) -> TaskType | None:
    """Infer task type from HuggingFace model config without downloading the model.

    Fetches the config.json from HuggingFace Hub and determines the task type
    based on the model architecture. Supports gated models through HF authentication.

    Args:
        model_name_or_path: HuggingFace model ID or local path

    Returns:
        Inferred TaskType, or None if unable to determine (will trigger fallback to CAUSAL_LM)

    Example:
        >>> infer_task_from_hf_config("meta-llama/Llama-2-7b")
        <TaskType.CAUSAL_LM: 'causal-language-model'>
        >>> infer_task_from_hf_config("Qwen/Qwen2-VL-7B")
        <TaskType.IMAGE_TEXT_TO_TEXT: 'image-text-to-text'>
    """
    try:
        # Try loading from local path first
        local_path = ePath(model_name_or_path)
        if local_path.is_dir():
            config_file = local_path / "config.json"
            if config_file.exists():
                config = json.loads(config_file.read_text())
            else:
                logger.warning(
                    f"No config.json found in local path: {model_name_or_path}. Task type will fallback to CAUSAL_LM."
                )
                return None
        else:
            # Try using huggingface_hub first (handles authentication for gated models)
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename="config.json",
                    repo_type="model",
                )
                config = json.loads(ePath(config_path).read_text())
            except Exception as hf_error:
                # Fallback to requests (for non-gated models)
                try:
                    import requests
                except ImportError:
                    logger.warning(
                        f"Cannot fetch config for {model_name_or_path}: "
                        f"Neither huggingface_hub nor requests library available. "
                        f"Task type will fallback to CAUSAL_LM."
                    )
                    return None

                config_url = f"https://huggingface.co/{model_name_or_path}/raw/main/config.json"
                try:
                    response = requests.get(config_url, timeout=10)
                    response.raise_for_status()
                    config = response.json()
                except requests.exceptions.RequestException as req_error:
                    # Check if it's a gated model (401 error)
                    if "401" in str(hf_error) or "gated" in str(hf_error).lower():
                        logger.warning(
                            f"Cannot access config for {model_name_or_path}: Model is gated and requires authentication. "
                            f"Run 'huggingface-cli login' to authenticate. Task type will fallback to CAUSAL_LM."
                        )
                    else:
                        logger.warning(
                            f"Failed to fetch config for {model_name_or_path}. "
                            f"Task type will fallback to CAUSAL_LM. Error: {req_error}"
                        )
                    return None

        architectures = config.get("architectures", [])
        model_type = config.get("model_type", "").lower()

        if not architectures:
            logger.warning(
                f"No architectures found in config for {model_name_or_path}. Task type will fallback to CAUSAL_LM."
            )
            return None

        arch = architectures[0]
        if "ForCausalLM" in arch:
            return TaskType.CAUSAL_LM

        elif "Omni" in arch:
            return TaskType.ANY_TO_ANY

        elif "ForConditionalGeneration" in arch:
            if any(x in model_type for x in ["whisper", "speech2text"]):
                return TaskType.SPEECH_SEQUENCE_TO_SEQUENCE
            else:
                return TaskType.IMAGE_TEXT_TO_TEXT

        elif "ForSequenceClassification" in arch:
            return TaskType.SEQUENCE_CLASSIFICATION

        elif "ForAudioClassification" in arch:
            return TaskType.AUDIO_CLASSIFICATION

        elif "ForImageClassification" in arch:
            return TaskType.IMAGE_CLASSIFICATION

        elif any(x in arch for x in ["ForSpeechSeq2Seq", "Whisper"]):
            return TaskType.SPEECH_SEQUENCE_TO_SEQUENCE

        elif "ForZeroShotImageClassification" in arch:
            return TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION

        elif "ForEmbedding" in arch or "ForSentenceEmbedding" in arch:
            return TaskType.EMBEDDING

        if "vision" in model_type or "clip" in model_type:
            return TaskType.BASE_VISION
        elif "diffusion" in model_type:
            return TaskType.DIFFUSION_LM

        logger.warning(
            f"Could not map architecture '{arch}' to a TaskType for {model_name_or_path}. "
            f"Task type will fallback to CAUSAL_LM."
        )
        return None

    except Exception as e:
        logger.warning(
            f"Unexpected error inferring task for {model_name_or_path}: {e}. Task type will fallback to CAUSAL_LM."
        )
        return None


class AutoEasyDeLConfig:
    """Factory class for automatically loading EasyDeL model configurations.

    Provides ``from_pretrained`` to load a model config from a HuggingFace Hub
    identifier or local checkpoint path, automatically resolving the correct
    config class based on the model type and task. Supports auto-binding of
    task types from model architectures, sharding configuration, and
    conversion from PyTorch configs.

    Example::

        config = AutoEasyDeLConfig.from_pretrained(
            "meta-llama/Llama-2-7b",
            sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        )
    """

    @staticmethod
    def bind_model_task(model_task: TaskType, architectures: list[str] | str):
        """Resolve ``TaskType.AUTO_BIND`` to the concrete task of an EasyDeL class.

        When ``model_task`` is :attr:`TaskType.AUTO_BIND`, looks up the named
        architecture in the top-level ``easydel`` namespace and returns its
        ``_model_task``. Otherwise returns ``model_task`` unchanged.

        Args:
            model_task (TaskType): Requested task. ``AUTO_BIND`` triggers lookup;
                any other value is returned as-is.
            architectures (list[str] | str): HF architecture name(s) from the
                model config. A list must contain exactly one entry.

        Returns:
            TaskType: The resolved task type.

        Raises:
            AssertionError: If ``architectures`` is a list of length != 1, or
                if the architecture name cannot be resolved on ``easydel``.
        """
        if model_task == TaskType.AUTO_BIND:
            if isinstance(architectures, list):
                assert len(architectures) == 1, "AutoBind is not supported for multi architecture loading!"
                architectures = architectures[0]
            import easydel as ed

            module_class: EasyDeLBaseModule = getattr(ed, architectures, None)
            assert module_class is not None, f"we couldn't find {architectures} in easydel collections!"
            model_task = module_class._model_task
        return model_task

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sharding_axis_dims: Sequence[int] = (1, 1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: Sequence[int] | None = None,
        sharding_axis_names: Sequence[str] = ("pp", "dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        model_task: TaskType = TaskType.AUTO_BIND,
        from_torch: bool = False,
        **kwargs,
    ) -> EasyDeLBaseConfig:
        """Load and finalize an :class:`EasyDeLBaseConfig` for a HF identifier or local path.

        Pipeline:

        1. Read the upstream HF ``config.json`` (via ``EasyDeLBaseConfig`` or
           ``transformers.AutoConfig``, depending on ``from_torch``).
        2. Cross-check against the ``trust_remote_code`` HF config in case the
           dynamic ``model_type`` differs from the static one.
        3. Resolve ``model_task`` (``AUTO_BIND`` infers it from
           ``architectures[0]``; otherwise the value is honoured as-is).
        4. Look up the registered EasyDeL config class via
           :func:`get_modules_by_type` and reload through it so
           model-specific defaults run.
        5. Apply ``add_basic_configurations`` to attach the mesh / sharding
           layout, then splat ``**kwargs`` as attribute overrides.

        Args:
            pretrained_model_name_or_path (str): HF Hub repo id or local
                directory containing a ``config.json``.
            sharding_axis_dims (Sequence[int], optional): Mesh shape across
                ``(pp, dp, fsdp, ep, tp, sp)``. Defaults to
                ``(1, 1, -1, 1, 1, 1)``.
            sharding_dcn_axis_dims (Sequence[int] | None, optional): Optional
                outer DCN mesh shape. Defaults to ``None``.
            sharding_axis_names (Sequence[str], optional): Axis names paired
                with ``sharding_axis_dims``.
            partition_axis (PartitionAxis | None, optional): Per-tensor
                sharding policy; default :class:`PartitionAxis` is used when
                ``None``.
            backend (EasyDeLBackends | None, optional): Custom-kernel backend
                hint forwarded to the config.
            platform (EasyDeLPlatforms | None, optional): Target platform
                hint forwarded to the config.
            model_task (TaskType, optional): Task slot to register against.
                ``AUTO_BIND`` triggers architecture-based inference. Defaults
                to ``TaskType.AUTO_BIND``.
            from_torch (bool, optional): Bypass the EasyDeL config registry and
                read with ``transformers.AutoConfig`` directly. Defaults to
                ``False``.
            **kwargs: Additional attributes copied onto the resulting config
                via ``setattr`` (overrides any value set above).

        Returns:
            EasyDeLBaseConfig: The fully resolved, mesh-aware model config.
        """
        if partition_axis is None:
            partition_axis = PartitionAxis()
        from transformers import AutoConfig

        cls_main = AutoConfig if from_torch else EasyDeLBaseConfig
        config = cls_main.from_pretrained(pretrained_model_name_or_path)
        model_type: str = config.model_type

        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            ovo_model_type: str = config.model_type

            if model_type != ovo_model_type:
                model_type = ovo_model_type
        except Exception:
            ...

        if model_task == TaskType.AUTO_BIND:
            model_task = infer_task_from_hf_config(pretrained_model_name_or_path)
        config_class = get_modules_by_type(
            model_type,
            cls.bind_model_task(model_task, config.architectures),
        )[0]
        config = config_class.from_pretrained(pretrained_model_name_or_path)
        if hasattr(config, "attach_custom_arguments"):
            config.attach_custom_arguments()

        config.add_basic_configurations(
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            platform=platform,
        )
        for k, v in kwargs.items():
            setattr(config, k, v)

        return config


def _shard_gather_fns_from_named_shardings(
    named_shardings: tp.Any,
    mesh: MeshLike,
) -> tuple[tp.Any, tp.Any]:
    """Build per-leaf shard/gather closures from a tree of ``NamedShardings``.

    MPMD-safe analogue of :func:`spectrax.make_shard_and_gather_fns`: the
    spectrax helper rebinds every leaf to the *full* mesh inside its
    ``_named`` wrapper, which collapses per-stage submeshes.  This helper
    trusts the per-leaf NamedShardings as-is, so PP placements (where each
    leaf may live on a different submesh) survive.
    """
    replicated = replicated_named_sharding(mesh)

    def _resolve_target(ns):
        # NamedSharding -> trust it (PP-stage submesh preserved).
        # None / SingleDeviceSharding / anything else -> replicate to the mesh.
        return ns if isinstance(ns, jax.sharding.NamedSharding) else replicated

    def _make_shard(ns):
        target = _resolve_target(ns)

        def _shard(x, _ns=target):
            return jax.device_put(x, _ns) if hasattr(x, "shape") else x

        return _shard

    def _make_gather(ns):
        # Always replicate to the full mesh on gather, regardless of input sharding.
        del ns

        def _gather(x, _r=replicated):
            return jax.device_put(x, _r) if hasattr(x, "shape") else x

        return _gather

    is_leaf = lambda x: isinstance(x, jax.sharding.Sharding) or x is None  # noqa: E731
    shard_fns = jax.tree_util.tree_map(_make_shard, named_shardings, is_leaf=is_leaf)
    gather_fns = jax.tree_util.tree_map(_make_gather, named_shardings, is_leaf=is_leaf)
    return shard_fns, gather_fns


class AutoShardAndGatherFunctions:
    """Factory of per-leaf shard / gather closures for an EasyDeL model.

    Used during checkpoint loading to move HF / PyTorch tensors onto the JAX
    device mesh (``shard_fns``) and during checkpoint saving / extraction to
    pull them back to the host (``gather_fns``). The closures are organised in
    a pytree mirroring the model's parameter pytree, optionally flattened to a
    ``{(path,): fn}`` dict via :func:`is_flatten`.

    Three entry points cover the common loading flows:

    * :meth:`from_config` — given an :class:`EasyDeLBaseConfig`, performs a
      full ``module.lazy_init`` to derive shapes/shardings (slow on large
      models but the most common path).
    * :meth:`from_model` — cheap path when a ``lazy_init`` already happened;
      reads the live ``NamedSharding`` from each Variable, preserving
      pipeline-stage submeshes (MPMD-safe).
    * :meth:`from_params` — derive from raw params + a mesh, replicating
      every leaf with a default ``PartitionSpec()``.

    See also :func:`_shard_gather_fns_from_named_shardings` for the MPMD-aware
    builder used internally.
    """

    @classmethod
    def from_config(
        cls,
        config: EasyDeLBaseConfig,
        flatten: bool = True,
        model_task: TaskType = TaskType.CAUSAL_LM,
        depth_target: list[str] | None = None,
    ):
        """
        Generates shard and gather functions based on a provided `EasyDeLBaseConfig` object.

        Args:
            config: An `EasyDeLBaseConfig` object containing the model configuration.
            flatten: Whether to flatten the shard and gather functions. Defaults to True.
                        model_task (TaskType): Task type of model load and find.
            depth_target: Pad the sharding to depth, for example make {params:tensor} with
                depth_target = ["row"] to {row:{params:tensor}}. Defaults to None.

        Returns:
            A tuple containing the shard and gather functions.

        Note:
            This performs a full ``module.lazy_init`` to derive parameter
            shapes. For large models that lazy_init dominates the cost
            (often minutes for 27B+). If you already have a lazy-initialized
            model instance, prefer :meth:`from_model` to avoid the duplicate
            traversal.
        """
        _, module = get_modules_by_type(config.model_type, model_task)
        with _shardgen_phase("from_config: module.lazy_init"):
            model = module.lazy_init(config=config, rngs=spx.Rngs(0))
        return cls.from_model(model, flatten=flatten)

    @classmethod
    def from_model(
        cls,
        model: EasyDeLBaseModule,
        flatten: bool = True,
    ):
        """Derive shard/gather functions from an already lazy-initialized model.

        This is the cheap path used by ``from_pretrained`` after it has
        already paid the lazy_init cost — it avoids running ``lazy_init`` a
        second time just to compute parameter shapes.

        Uses the MPMD-aware path: ``spx.extract_sharding_structure`` reads each
        Variable's *live* ``NamedSharding`` (with per-stage submeshes
        preserved) and we build per-leaf ``device_put`` closures from
        that.
        """
        with _shardgen_phase("from_model: extract_sharding_structure"):
            _gdef, gstate = spx.export(model)
            named_shardings = spx.extract_sharding_structure(gstate.raw(), mesh=model.mesh)
        with _shardgen_phase("from_model: build per-leaf shard/gather fns"):
            shard_fns, gather_fns = _shard_gather_fns_from_named_shardings(named_shardings, model.mesh)

        if flatten and not is_flatten(shard_fns):
            gather_fns = flatten_dict(gather_fns)
            shard_fns = flatten_dict(shard_fns)
        elif not flatten and is_flatten(shard_fns):
            gather_fns = unflatten_dict(gather_fns)
            shard_fns = unflatten_dict(shard_fns)

        return shard_fns, gather_fns

    @staticmethod
    def from_params(params, mesh):
        """
        Generates shard and gather functions directly from model parameters and a mesh.

        Args:
            params: The model parameters (pytree) to generate functions for.
            mesh: The JAX device mesh to use for sharding.

        Returns:
            A tuple containing the shard and gather functions.
        """
        partition_specs = jax.tree_util.tree_map(lambda x: PartitionSpec() if hasattr(x, "shape") else None, params)
        return make_shard_and_gather_fns(
            partition_specs=partition_specs,
            mesh=mesh,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sharding_axis_dims: Sequence[int] = (1, 1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: Sequence[int] | None = None,
        sharding_axis_names: Sequence[str] = ("pp", "dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        flatten: bool = True,
        config_kwargs: Mapping[str, tp.Any] | None = None,
        model_task: TaskType = TaskType.CAUSAL_LM,
        from_torch: bool = False,
        trust_remote_code: bool = False,
    ) -> tuple[Mapping[str, Callable], Mapping[str, Callable]]:
        """
        Generates shard and gather functions based on a pretrained model name or path.

        Args:
            pretrained_model_name_or_path: The name or path of the pretrained model.
            sharding_axis_dims: The dimensions of the sharding axes. Defaults to (1, 1, -1, 1, 1, 1).
            sharding_axis_names: The names of the sharding axes. Defaults to ("pp", "dp", "fsdp",  "ep", "tp", "sp").
            partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
            backend: The backend to use for custom kernels. Defaults to None.
            flatten: Whether to flatten the shard and gather functions. Defaults to True.
            config_kwargs: Additional keyword arguments to pass to the `AutoEasyDeLConfig` constructor. Defaults to None.
                        model_task (TaskType): Task type of model load and find.
                        from_torch: should config be loaded from torch models or not.
            trust_remote_code (bool): whenever to trust remote code loaded from HF.
        Returns:
            A tuple containing the shard and gather functions.
        """
        if partition_axis is None:
            partition_axis = PartitionAxis()
        config = AutoEasyDeLConfig.from_pretrained(
            pretrained_model_name_or_path,
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            platform=platform,
            from_torch=from_torch,
            trust_remote_code=trust_remote_code,
            model_task=model_task,
        )
        if config_kwargs is not None:
            for k, v in config_kwargs.items():
                setattr(config, k, v)
        return cls.from_config(  # pyright: ignore[reportReturnType]
            config=config,
            flatten=flatten,
            model_task=model_task,
        )
