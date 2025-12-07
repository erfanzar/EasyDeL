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
import json
import typing as tp

import flax
from eformer.escale import PartitionAxis, make_shard_and_gather_fns, match_partition_rules
from eformer.loggings import get_logger
from eformer.paths import ePath
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig, EasyDeLBaseModule
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
from easydel.infra.factory import TaskType, registry
from easydel.utils.traversals import flatten_dict, unflatten_dict

logger = get_logger(name=__name__)


def get_modules_by_type(
    model_type: str,
    task_type: TaskType,
) -> tuple[type[EasyDeLBaseConfig], type[EasyDeLBaseModule] | tp.Any]:
    """
    The get_modules_by_type function is a helper function that returns the following:
        1. The config class for the model type specified (e.g., LlamaConfig, FalconConfig)
        2. The EasyDeL Model class for the model type specified (e.g., FlaxLlamaForCausalLM, FalconForCausalLM)
    """
    registred_module = registry.get_module_registration(
        task_type=task_type,
        model_type=model_type,
    )
    return (registred_module.config, registred_module.module)


def is_flatten(pytree: dict):
    """The is_flatten function checks if the pytree is flattened.
        If it is, then the first key in the dictionary will be a tuple of (mpl, mpl_id).
        Otherwise, it will be an integer representing mpl_id.

    Args:
        pytree: dict: Pass the pytree to the function

    Returns:
        True if the pytree is a flattened tree, and false otherwise
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
    """Factory helpers to load EasyDeL configs from identifiers or checkpoints."""

    @staticmethod
    def bind_model_task(model_task: TaskType, architectures: list[str] | str):
        if model_task == TaskType.AUTO_BIND:
            if isinstance(architectures, list):
                assert len(architectures) == 1, "AutoBind is not supported for multi architecture loading!"
                architectures = architectures[0]
            import easydel as ed

            module_class: ed.EasyDeLBaseModule = getattr(ed, architectures, None)
            assert module_class is not None, f"we couldn't find {architectures} in easydel collections!"
            model_task = module_class._model_task
        return model_task

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        model_task: TaskType = TaskType.AUTO_BIND,
        from_torch: bool = False,
        **kwargs,
    ) -> EasyDeLBaseConfig:
        """The from_pretrained function is a helper function that allows you to instantiate a model from the pretrained
        model repository. It takes as input the name of the model (e.g., 'bert-base-uncased') and returns an instance of
        the class corresponding to your model, with all weights loaded from disk.

        Args:
            cls: Create an instance of the class that called this function.
            pretrained_model_name_or_path: str: Identify the model in the huggingface model hub.
            sharding_axis_dims: tp.Sequence[int]: Specify the dimension of each axis in the sharded
                model_tasking arrays in easydel.
            backend: tp.Optional[EasyDeLBackends] : backend to use for model.
                        model_task (TaskType): Task type of model load and find.
            from_torch: should config be loaded from torch models or not.
            **kwargs: Pass additional arguments to the model and config classes.
        generation process

        Returns:
            A Model Config
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


class AutoShardAndGatherFunctions:
    """
    A class to automatically generate shard and gather functions for a given model configuration.

    This class provides two methods to generate shard and gather functions:

    - `from_config`: Generates functions based on a provided `EasyDeLBaseConfig` object.
    - `from_pretrained`: Generates functions based on a pretrained model name or path.

    Attributes:
        None

    Methods:
        from_config: Generates shard and gather functions based on a provided `EasyDeLBaseConfig` object.
        from_pretrained: Generates functions based on a pretrained model name or path.
    """

    @classmethod
    def from_config(
        cls,
        config: EasyDeLBaseConfig,
        partition_rules: tuple[tuple[str, PartitionSpec]] | None = None,
        flatten: bool = True,
        model_task: TaskType = TaskType.CAUSAL_LM,
        depth_target: list[str] | None = None,
    ):
        """
        Generates shard and gather functions based on a provided `EasyDeLBaseConfig` object.

        Args:
            config: An `EasyDeLBaseConfig` object containing the model configuration.
            partition_rules: A tuple of tuples containing partition rule names and `PartitionSpec` objects.
              If None, uses the default partition rules from the `config`.
            flatten: Whether to flatten the shard and gather functions. Defaults to True.
                        model_task (TaskType): Task type of model load and find.
            depth_target: Pad the sharding to depth, for example make {params:tensor} with
                depth_target = ["row"] to {row:{params:tensor}}. Defaults to None.

        Returns:
            A tuple containing the shard and gather functions.
        """
        if partition_rules is None:
            partition_rules = config.get_partition_rules(True)
        _, module = get_modules_by_type(config.model_type, model_task)
        model = module.lazy_init(config=config, rngs=flax.nnx.Rngs(0))

        partition_specs = match_partition_rules(partition_rules, model.graphtree_shape)

        shard_fns, gather_fns = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=config.mesh)
        if flatten and not is_flatten(shard_fns):
            gather_fns = flatten_dict(gather_fns)
            shard_fns = flatten_dict(shard_fns)
        elif not flatten and is_flatten(shard_fns):
            gather_fns = unflatten_dict(gather_fns)
            shard_fns = unflatten_dict(shard_fns)

        return shard_fns, gather_fns

    @staticmethod
    def from_params(params, partition_rules, mesh):
        """
        Generates shard and gather functions directly from model parameters, partition rules, and a mesh.

        Args:
            params: The model parameters (pytree) to generate functions for.
            partition_rules: A tuple of tuples defining the partitioning strategy.
            mesh: The JAX device mesh to use for sharding.

        Returns:
            A tuple containing the shard and gather functions.
        """
        partition_specs = match_partition_rules(partition_rules, params)
        return make_shard_and_gather_fns(
            partition_specs=partition_specs,
            mesh=mesh,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        partition_rules: tuple[tuple[str, PartitionSpec]] | None = None,
        flatten: bool = True,
        config_kwargs: tp.Mapping[str, tp.Any] | None = None,
        model_task: TaskType = TaskType.CAUSAL_LM,
        from_torch: bool = False,
        trust_remote_code: bool = False,
    ) -> tuple[tp.Mapping[str, tp.Callable], tp.Mapping[str, tp.Callable]]:
        """
        Generates shard and gather functions based on a pretrained model name or path.

        Args:
            pretrained_model_name_or_path: The name or path of the pretrained model.
            sharding_axis_dims: The dimensions of the sharding axes. Defaults to (1, -1, 1, 1, 1).
            sharding_axis_names: The names of the sharding axes. Defaults to ("dp", "fsdp",  "ep", "tp", "sp").
            partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
            backend: The backend to use for custom kernels. Defaults to None.
            partition_rules: A tuple of tuples containing partition rule names and `PartitionSpec` objects.
                If None, uses the default partition rules from the `config`.
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
        return cls.from_config(
            config=config,
            partition_rules=partition_rules,
            flatten=flatten,
            model_task=model_task,
        )
