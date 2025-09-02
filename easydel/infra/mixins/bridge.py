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

import gc
import json
import os
import typing as tp
import warnings
from copy import deepcopy

import huggingface_hub
import huggingface_hub.errors
import jax
import jax.extend
import jax.tree_util
from eformer.escale import PartitionAxis
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from eformer.serialization import AsyncCheckpointManager
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers.utils.generic import working_or_temp_dir
from transformers.utils.hub import PushToHubMixin

from easydel.utils.readme_generator import ModelInfo, ReadmeGenerator
from easydel.utils.traversals import flatten_dict, is_flatten, merge_model_and_tree, string_key_to_int, unflatten_dict

from ..base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict
from ..etils import EasyDeLBackends, EasyDeLPlatforms, EasyDeLQuantizationMethods

if tp.TYPE_CHECKING:
    from ..base_module import EasyDeLBaseModule
logger = get_logger(__name__)

FLAX_WEIGHTS_NAME = "easydel-model.parameters"
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
ED_SAFE_WEIGHTS_INDEX_NAME = "easydel-model.parameters.safetensors.index.json"
TENSORSTORE_INDEX_NAME = "tensorstore_index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
PROCESSOR_NAME = "processor_config.json"
CHAT_TEMPLATE_NAME = "chat_template.json"
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

CANDIDATE_FILENAMES = [
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    ED_SAFE_WEIGHTS_INDEX_NAME,
    FLAX_WEIGHTS_NAME,
    TENSORSTORE_INDEX_NAME,
]


class EasyBridgeMixin(PushToHubMixin):
    """
    Mixin class for adding bridging functionalities like saving, loading, and pushing models to Hugging Face Hub.
    """

    config: EasyDeLBaseConfig
    hf_torch_auto_loader: tp.Any | None = None
    config_class: type[EasyDeLBaseConfig] | None = None
    base_model_prefix: str | None = None
    _model_task: str | None = None
    _model_type: str | None = None

    def _model_card(self, name: str, repo_id: str) -> str:
        """Generates a model card (README.md) for the given model.

        Args:
            name (str): The name of the model.
            repo_id (str): The repository ID on Hugging Face Hub.

        Returns:
            str: The generated README.md content.
        """
        from easydel import __version__

        # Retrieve attention mechanism from config, default to "vanilla" if not found
        attn_mechanism = getattr(self.config, "attn_mechanism", "vanilla")
        if not isinstance(attn_mechanism, str):  # Handle cases where it might be an Enum
            try:
                attn_mechanism = attn_mechanism.value
            except AttributeError:
                attn_mechanism = str(attn_mechanism).split(".")[-1].lower()  # Fallback

        model_info = ModelInfo(
            name=name,
            type=self.__class__.__name__,
            repo_id=repo_id,
            model_type=self._model_type,
            model_task=self._model_task or "CausalLM",  # Default to CausalLM if not set
            attn_mechanism=attn_mechanism,
            version=__version__,
        )
        return ReadmeGenerator().generate_readme(model_info)

    def _save_model_files(
        self,
        save_directory: ePathLike,
        gather_fns: dict[tp.Callable] | None = None,
        float_dtype=None,
    ):
        """Saves the model's configuration, weights, and potentially the generation config to the specified directory.

        Args:
          save_directory (ePath): The directory where the model files will be saved.
          gather_fns (dict[Callable], optional): Custom gather functions for checkpoint saving.
          float_dtype (dtype, optional): Data type for saving weights. Defaults to None.
        """
        save_directory.mkdir(parents=True, exist_ok=True)

        config_to_save = deepcopy(self.config)
        config_to_save.__dict__.pop("attn_dtype", None)
        config_to_save.__dict__.pop("attn_softmax_dtype", None)
        config_to_save.architectures = [self.__class__.__name__]
        config_to_save.save_pretrained(str(save_directory))

        if self.can_generate() and hasattr(self, "generation_config"):
            if self.generation_config is not None:
                self.generation_config.save_pretrained(str(save_directory))

        state = nn.split(self, nn.Param, ...)[1]  # NOTE: This one here ignores LoRA Params...
        if gather_fns is None:
            gather_fns = self._gather_fns
        output_model_file = AsyncCheckpointManager().save(
            tree=state.to_pure_dict(),
            path=str(save_directory),
            mesh=self.mesh,
            float_dtype=float_dtype,
            prefix="model",
        )

        logger.info(f"Model weights saved in {output_model_file}")

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        push_to_hub: bool = False,
        token: str | bool | None = None,
        gather_fns: dict[tp.Callable] | None = None,
        float_dtype: jnp.dtype | None = None,
        **kwargs,
    ):
        """Saves the model, its configuration, and optionally pushes it to the Hugging Face Hub.

        Args:
            save_directory (str or PathLike): The directory where to save the model.
            push_to_hub (bool, optional): If True, pushes the model to the Hugging Face Hub.
            token (str or bool, optional): The Hugging Face Hub token.
            gather_fns (dict[Callable], optional): Custom gather functions for checkpoint saving.
            float_dtype (dtype, optional): Data type for saving weights.
            **kwargs: Additional keyword arguments for Hugging Face Hub.
        """

        easy_directory = ePath(save_directory)
        if easy_directory.is_file():
            logger.error(f"Provided path ({easy_directory}) should be a directory, not a file")
            return

        repo_id = kwargs.pop("repo_id", easy_directory.name)
        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(easy_directory)

        self._save_model_files(
            save_directory=easy_directory,
            gather_fns=gather_fns,
            float_dtype=float_dtype,
        )
        readme_path = easy_directory / "README.md"
        readme_path.write_text(self._model_card(repo_id, repo_id))

        if push_to_hub and jax.process_index() == 0:
            self._upload_modified_files(
                str(easy_directory),
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: bool | None = None,
        commit_message: str | None = None,
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
        gather_fns: dict[tp.Callable] | None = None,
        float_dtype: jnp.dtype | None = None,
        verbose: bool = True,
        mismatch_allowed: bool = True,
        revision: str | None = None,
        commit_description: str | None = None,
    ) -> str:
        """Pushes the model to the Hugging Face Hub.

        Args:
            repo_id (str): The repository ID on Hugging Face Hub.
            params (any): Model parameters.
            use_temp_dir (bool, optional): If True, uses a temporary directory. Defaults to None
            commit_message (str, optional): The commit message for the push.
            private (bool, optional): If True, creates a private repository.
            token (str or bool, optional): The Hugging Face Hub token.
            create_pr (bool, optional): If True, creates a pull request.
            gather_fns (dict[Callable], optional): Custom gather functions for checkpoint saving.
            float_dtype (dtype, optional): Data type for saving weights.
            verbose (bool, optional): Whether to print verbose messages. Defaults to True.
            mismatch_allowed (bool, optional): If True, allows mismatch in parameters while loading. Defaults to True.
            revision (str, optional): The revision to push to.
            commit_description (str, optional): The commit description for the push.

        Returns:
            str: The URL of the created repository.
        """
        working_dir = ePath(repo_id.split("/")[-1])

        repo_id = self._create_repo(
            repo_id,
            private=private,
            token=token,
            repo_url=None,
            organization=None,
        )

        if use_temp_dir is None:
            use_temp_dir = not working_dir.is_dir()

        with working_or_temp_dir(working_dir=str(working_dir), use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)
            self.save_pretrained(
                save_directory=work_dir,
                push_to_hub=False,
                token=token,
                gather_fns=gather_fns,
                float_dtype=float_dtype,
                verbose=verbose,
                mismatch_allowed=mismatch_allowed,
                repo_id=repo_id,
            )

            return self._upload_modified_files(
                str(work_dir),
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )

    @classmethod
    def can_generate(cls) -> bool:
        """Checks if the model can generate sequences with `.generate()`.

        Returns:
            bool: True if the model can generate, False otherwise.
        """
        return True

    @classmethod
    def _load_model_weights(
        cls,
        resolved_archive_file: str | None,
        model: EasyDeLBaseModule,
        param_dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
        shard_fns: dict[tp.Callable] | None,
        quantization_method: EasyDeLQuantizationMethods | None,
        quantization_platform: EasyDeLQuantizationMethods | None,
        quantization_block_size: int,
        quantization_pattern: str | None,
        quantize_tensors: bool,
        vebose: bool,
    ) -> EasyDeLBaseModule:
        """Loads model weights from a checkpoint file.

        Args:
            resolved_archive_file: The path to the checkpoint file.
            model: an easydel model.
            mismatch_allowed: If True, allows mismatch in parameters while loading.
            verbose: Whether to print verbose messages.
            shard_fns: Custom shard functions for loading checkpoint.

        Returns:
            an easydel, with loaded parameter.
        """
        if quantize_tensors:
            from easydel.layers.quantization.quantizers import EasyQuantizer

            quantizer = EasyQuantizer(
                quantization_method=quantization_method,
                quantization_platform=quantization_platform,
                quantization_pattern=quantization_pattern,
                block_size=quantization_block_size,
            )
            if quantize_tensors:

                def callback(x, p):
                    if shard_fns is not None:
                        key_get = p
                        if isinstance(p, str):
                            key_get = tuple(p.split("."))
                        callable_fn = shard_fns.get(key_get)
                        if callable_fn is not None:
                            x = callable_fn(x)
                    return quantizer(x, p)

        extraargs = {}
        if resolved_archive_file:
            if str(resolved_archive_file).endswith(TENSORSTORE_INDEX_NAME):
                resolved_archive_file = str(resolved_archive_file)[: -len(TENSORSTORE_INDEX_NAME)]
            else:
                extraargs["callback"] = callback
            state, _ = AsyncCheckpointManager().load(
                path=ePath(resolved_archive_file),
                mesh=mesh,
                dtype=param_dtype,
                partition_rules=model.config.get_partition_rules(),
                prefix="model",
                **extraargs,
            )
            params = state.get("params", None)
            if params is not None:
                state = params
            state = flatten_dict(state)
            state = string_key_to_int(state)

            required_params = set(flatten_dict(model.graphtree_params_shape))
            unexpected_keys = set(state.keys()) - required_params
            if any([k[-1].startswith("quant_") for k in state.keys()]):
                model = model.quantize(
                    method=quantization_method,
                    block_size=quantization_block_size,
                    verbose=vebose,
                )
            for unexpected_key in unexpected_keys:
                del state[unexpected_key]

            return merge_model_and_tree(
                model=model,
                tree=unflatten_dict(state),
            )

        else:
            return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis = PartitionAxis(),
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = jax.lax.Precision("fastest"),
        config_kwargs: dict[str, tp.Any] | None = None,
        partition_rules: tuple[tuple[str, PartitionSpec]] | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = "jax",
        shard_fns: dict[tp.Callable] | None = None,
        auto_shard_model: bool = True,
        verbose: bool = True,
        mismatch_allowed: bool = True,
        *model_args,
        config: EasyDeLBaseConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        vebose: bool = True,
        quantization_platform: EasyDeLPlatforms | None = None,
        quantization_method: EasyDeLQuantizationMethods | None = None,
        quantization_block_size: int = 128,
        quantization_pattern: str | None = None,
        quantize_tensors: bool = True,
        **kwargs,
    ):
        """Loads an EasyDeL model from a pretrained model or path.

        Args:
            pretrained_model_name_or_path (str, optional): The name or path of the pretrained model.
            sharding_axis_dims (Sequence[int], optional): The dimensions of sharding axes.
            sharding_axis_names (Sequence[str], optional): The names of sharding axes.
            partition_axis (PartitionAxis, optional): The partition axis configuration.
            dtype (dtype, optional): The data type of the model.
            param_dtype (dtype, optional): The data type of the parameters.
            precision (PrecisionLike, optional): The computation precision.
            config_kwargs (dict[str, Any], optional): Additional configuration parameters.
            partition_rules (tuple, optional): Custom partitioning rules for sharding.
            backend (EasyDeLBackends, optional): The backend to use.
            platform (EasyDeLPlatforms, optional): The platform to use.
            shard_fns (dict[Callable], optional): Custom shard functions for loading checkpoint.
            auto_shard_model (bool, optional): Whether to automatically shard the model.
            verbose (bool, optional): Whether to print verbose messages. Defaults to True.
            mismatch_allowed (bool, optional): If True, allows mismatch in parameters while loading. Defaults to True.
            *model_args: Additional arguments for the model.
            config (str, optional): configuration for the model.
            cache_dir (str, optional): The cache directory for the pretrained model.
            force_download (bool, optional): Whether to force download the model.
            local_files_only (bool, optional): Whether to use only local files.
            token (str, optional): The Hugging Face Hub token.
            revision (str, optional): The revision of the model to load.
            **kwargs: Additional keyword arguments.

        Returns:
            The loaded EasyDeL model.
        """

        from huggingface_hub import HfApi
        from transformers import GenerationConfig
        from transformers.utils import download_url as _download_url
        from transformers.utils import is_offline_mode as _is_offline_mode
        from transformers.utils import is_remote_url as _is_remote_url

        from easydel.modules.auto.auto_configuration import (
            AutoEasyDeLConfig,
            AutoShardAndGatherFunctions,
            get_modules_by_type,
        )

        api = HfApi(token=token)

        proxies = kwargs.pop("proxies", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        # Not relevant for Flax Models
        _ = kwargs.pop("adapter_kwargs", None)

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored."
            )

        if _is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        config_path = config if config is not None else pretrained_model_name_or_path

        config = AutoEasyDeLConfig.from_pretrained(
            config_path,
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            from_torch=False,
            backend=backend,
            platform=platform,
            model_task=cls._model_task,
        )
        config_kwargs = {} if config_kwargs is None else config_kwargs
        config.add_basic_configurations(
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            platform=platform,
            **config_kwargs,
        )

        if commit_hash is None:
            commit_hash = getattr(config, "_commit_hash", None)
        if auto_shard_model and shard_fns is None:
            shard_fns, _ = AutoShardAndGatherFunctions.from_config(
                config=config,
                flatten=False,
                partition_rules=partition_rules,
                model_task=cls._model_task,
            )
            fns = {"params": shard_fns}
            fns.update(shard_fns)
            shard_fns = fns

        elif auto_shard_model and shard_fns is not None:
            logger.warning("`auto_shard_model` will be ignored since `shard_fns` is provided.")

        resolved_archive_file = None
        if pretrained_model_name_or_path:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)

            is_local = ePath(pretrained_model_name_or_path).is_dir()

            def _pick_first_existing(dir_path: ePath) -> tuple[ePath | None, str | None]:  # type:ignore
                for cand in CANDIDATE_FILENAMES:
                    p = dir_path / cand
                    if p.is_file():
                        return p, cand
                return None, None

            if is_local:
                # Local directory: look for index -> safetensors -> msgpack
                root = ePath(pretrained_model_name_or_path) / subfolder
                archive_file, picked_name = _pick_first_existing(root)
                if not archive_file:
                    raise FileNotFoundError(
                        f"No model weights found in '{root}'. Tried: {', '.join(CANDIDATE_FILENAMES)}"
                    )
                is_local = True
            else:
                # Sometimes the path is nested under subfolder
                alt_root = ePath(subfolder) / pretrained_model_name_or_path
                if alt_root.is_dir():
                    archive_file, picked_name = _pick_first_existing(alt_root)
                    if archive_file:
                        is_local = True

                if not is_local:
                    if _is_remote_url(pretrained_model_name_or_path):
                        filename = pretrained_model_name_or_path
                        resolved_archive_file = _download_url(pretrained_model_name_or_path)
                    else:
                        filename = None
                        for cand in CANDIDATE_FILENAMES:
                            try:
                                resolved_archive_file = api.hf_hub_download(
                                    repo_id=pretrained_model_name_or_path,
                                    filename=cand,
                                    subfolder=subfolder,
                                    revision=revision,
                                    cache_dir=cache_dir,
                                    force_download=force_download,
                                    proxies=proxies,
                                    token=token,
                                    local_files_only=local_files_only,
                                )
                                filename = cand
                                break
                            except (FileNotFoundError, huggingface_hub.errors.EntryNotFoundError):
                                continue

                        if resolved_archive_file is None:
                            raise OSError(
                                f"Can't load the model for '{pretrained_model_name_or_path}'. "
                                f"Tried to download one of: {', '.join(CANDIDATE_FILENAMES)}. "
                                f"If you're loading from https://huggingface.co/models, make sure the repo exists and "
                                f"contains one of the expected files, or provide a local directory."
                            )

                        if filename == SAFE_WEIGHTS_INDEX_NAME:
                            try:
                                with open(resolved_archive_file, "r", encoding="utf-8") as f:
                                    index_data = json.load(f)
                                shard_names = sorted(set(index_data.get("weight_map", {}).values()))
                                for shard_name in shard_names:
                                    api.hf_hub_download(
                                        repo_id=pretrained_model_name_or_path,
                                        filename=shard_name,
                                        subfolder=subfolder,
                                        revision=revision,
                                        cache_dir=cache_dir,
                                        force_download=force_download,
                                        proxies=proxies,
                                        token=token,
                                        local_files_only=local_files_only,
                                    )
                            except Exception as e:
                                raise RuntimeError(f"Downloaded sharded index but failed to fetch shards: {e}") from e

            if is_local:
                logger.debug(f"loading weights file {archive_file}")
                resolved_archive_file = str(archive_file)
                filename = os.path.basename(str(archive_file))
            else:
                logger.debug(f"loading weights file {filename} from cache at {resolved_archive_file}")
        cls = get_modules_by_type(config.model_type, cls._model_task)[1]
        model = cls.lazy_init(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=nn.Rngs(0),
        )

        model = cls._load_model_weights(
            resolved_archive_file,
            model,
            param_dtype,
            model.mesh,
            shard_fns,
            quantization_method,
            quantization_platform,
            quantization_block_size,
            quantization_pattern,
            quantize_tensors,
            vebose,
        )

        if not quantize_tensors:  # already quantized
            model = model.quantize(
                method=quantization_method,
                block_size=quantization_block_size,
                quantize_tensors=quantize_tensors,
                verbose=vebose,
            )
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except OSError:
                logger.info("Generation config file not found, using a generation config created from the model config.")

        return model

    @classmethod
    def _from_torch_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: jax.Device | None = None,  # type:ignore
        dtype: jax.numpy.dtype = jax.numpy.float32,
        param_dtype: jax.numpy.dtype = jax.numpy.float32,
        precision: jax.lax.Precision | None = None,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        shard_attention_computation: bool = True,
        shard_fns: tp.Mapping[tuple, tp.Callable] | dict | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        config_kwargs: EasyDeLBaseConfigDict | None = None,
        auto_shard_model: bool = True,
        partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
        quantization_platform: EasyDeLPlatforms | None = None,
        quantization_method: EasyDeLQuantizationMethods | None = None,
        quantization_block_size: int = 128,
        quantization_pattern: str | None = None,
        quantize_tensors: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        from transformers import AutoConfig

        from easydel.layers.quantization.quantizers import EasyQuantizer
        from easydel.modules.auto.auto_configuration import AutoShardAndGatherFunctions, get_modules_by_type

        try:
            import torch

            def _clear():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except ModuleNotFoundError as er:
            raise ModuleNotFoundError(
                "in order to load model from torch you should install torch first run `pip install torch`"
            ) from er

        logger.debug(f"Downloading model config from {pretrained_model_name_or_path}")
        trust_remote_code = kwargs.get("trust_remote_code", False)
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        model_type: str = config.model_type

        config_class, module = get_modules_by_type(model_type, task_type=cls._model_task)

        logger.debug(f"Downloading hf_model weights from {pretrained_model_name_or_path}")
        if "torch_dtype" not in kwargs.keys():
            kwargs["torch_dtype"] = torch.float16

        hf_model = cls.get_torch_loader().from_pretrained(pretrained_model_name_or_path, **kwargs)
        generation_config = getattr(hf_model, "generation_config", None)
        config_class = config_class.from_pretrained(pretrained_model_name_or_path)
        state_dict = hf_model.state_dict()

        # Clear and collect memory after deleting the hf_model
        del hf_model
        _clear()

        logger.debug("adding hf_model basic EasyDeL configurations.")
        if hasattr(config_class, "attach_custom_arguments"):
            config_class.attach_custom_arguments()
        config_kwargs = {} if config_kwargs is None else config_kwargs
        config_class.add_basic_configurations(
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            platform=platform,
            shard_attention_computation=shard_attention_computation,
            **config_kwargs,
        )

        logger.debug("creating easydel model")
        model = module.lazy_init(
            config=config_class,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=nn.Rngs(0),
        )
        model.generation_config = generation_config

        _clear()

        if shard_fns is not None:
            if auto_shard_model:
                warnings.warn(
                    "`auto_shard_model` will be ignored since you are passing custom sharding functions",
                    stacklevel=1,
                )
            logger.debug("sharding model parameters based on the given `shard_fns`.")
            if not is_flatten(shard_fns):
                shard_fns = flatten_dict(shard_fns)
        elif auto_shard_model:
            shard_fns, _ = AutoShardAndGatherFunctions.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                partition_rules=partition_rules,
                sharding_axis_dims=sharding_axis_dims,
                sharding_dcn_axis_dims=sharding_dcn_axis_dims,
                sharding_axis_names=sharding_axis_names,
                partition_axis=partition_axis,
                shard_attention_computation=shard_attention_computation,
                backend=backend,
                platform=platform,
                config_kwargs=config_kwargs,
                trust_remote_code=trust_remote_code,
                model_task=cls._model_task,
            )
        logger.debug("converting huggingface-model to easydel-model.")
        params_pattern_selection = None
        uses_tie_word_embedding = getattr(config, "tie_word_embeddings", False)

        quantizer = EasyQuantizer(
            quantization_method=quantization_method,
            block_size=quantization_block_size,
            quantization_platform=quantization_platform,
            quantization_pattern=quantization_pattern,
        )
        callback = None
        passed_shard_fns = shard_fns
        if quantize_tensors:
            passed_shard_fns = None

            def callback(x, p):
                if shard_fns is not None:
                    key_get = p
                    if isinstance(p, str):
                        key_get = tuple(p.split("."))
                    callable_fn = shard_fns.get(key_get)
                    if callable_fn is not None:
                        x = callable_fn(x)
                return quantizer(x, p)

        params = model.pure_transform_fn(
            state_dict,
            config=config,
            device=device,
            shard_fns=passed_shard_fns,
            params_pattern_selection=params_pattern_selection,
            remove_state_dict=True,
            uses_tie_word_embedding=uses_tie_word_embedding,
            callback=callback,
        )
        del state_dict
        _clear()
        if is_flatten(params):
            logger.info("converted parameters are flatten making them unflatten.")
            params = unflatten_dict(params)

        logger.debug("merging model and parameters pytree.")
        model = merge_model_and_tree(model=model, tree=params)
        logger.debug("model and parameters pytree merged.")
        if (
            quantization_method is not None
            and quantization_method != EasyDeLQuantizationMethods.NONE
            and not quantize_tensors
        ):
            logger.debug("quantizing model.")
            model = model.quantize(
                method=quantization_method,
                block_size=quantization_block_size,
                quantization_pattern=quantization_pattern,
                verbose=verbose,
            )
        logger.debug("returning model.")
        return model

    @classmethod
    def get_torch_loader(cls):
        from ..factory import TaskType

        auto_loader = getattr(cls, "hf_torch_auto_loader", None)
        if auto_loader is not None:
            return auto_loader
        if cls._model_task == TaskType.CAUSAL_LM:
            from transformers import AutoModelForCausalLM as module
        elif cls._model_task == TaskType.AUDIO_CLASSIFICATION:
            from transformers import AutoModelForAudioClassification as module
        elif cls._model_task == TaskType.SEQUENCE_TO_SEQUENCE:
            from transformers import AutoModelForSeq2SeqLM as module
        elif cls._model_task == TaskType.SPEECH_SEQUENCE_TO_SEQUENCE:
            from transformers import AutoModelForSpeechSeq2Seq as module
        elif cls._model_task == TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION:
            from transformers import AutoModelForZeroShotImageClassification as module
        elif cls._model_task == TaskType.IMAGE_TEXT_TO_TEXT:
            from transformers import AutoModelForImageTextToText as module
        elif cls._model_task == TaskType.SEQUENCE_CLASSIFICATION:
            from transformers import AutoModelForSequenceClassification as module
        elif cls._model_task == TaskType.BASE_MODULE:
            from transformers import AutoModel as module
        elif cls._model_task == TaskType.BASE_VISION:
            # hf dont see anything diff between base and vision modules
            from transformers import AutoModel as module
        else:
            raise ValueError("couldn't find requested hf autoloader, you can set `hf_torch_auto_loader` to your class")
        return module
