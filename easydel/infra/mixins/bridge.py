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

"""Bridge mixin for EasyDeL-HuggingFace interoperability.

This module provides the EasyBridgeMixin class that enables seamless integration
between EasyDeL models and the HuggingFace ecosystem. It handles model serialization,
loading from various formats, conversion between frameworks, and integration with
the HuggingFace Hub.

The bridge supports:
- Loading models from HuggingFace Hub or local paths
- Converting between PyTorch and JAX/Flax formats
- Saving models in EasyDeL or HuggingFace formats
- Pushing models to HuggingFace Hub
- Automatic weight format detection and loading
- Quantization during loading
- Distributed loading with sharding

Classes:
    EasyBridgeMixin: Main mixin class providing bridge functionality

Constants:
    FLAX_WEIGHTS_NAME: Standard name for Flax model weights
    SAFE_WEIGHTS_NAME: Standard name for SafeTensors weights
    CANDIDATE_FILENAMES: List of possible weight file names to search

Example:
    >>> from easydel.infra.mixins import EasyBridgeMixin
    >>>
    >>> class MyModel(EasyDeLBaseModule, EasyBridgeMixin):
    ...     pass
    >>>
    >>> # Load from HuggingFace
    >>> model = MyModel.from_pretrained("gpt2")
    >>>
    >>> # Save locally
    >>> model.save_pretrained("./my_model")
    >>>
    >>> # Push to Hub
    >>> model.push_to_hub("username/my-model")
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import tempfile
import typing as tp
import warnings
from copy import deepcopy
from dataclasses import dataclass

import huggingface_hub
import huggingface_hub.errors
import jax
from eformer.escale import PartitionAxis
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from eformer.serialization import Checkpointer
from eformer.serialization.checkpointer import find_latest_checkpoint
from flax import nnx as nn
from huggingface_hub import CommitOperationAdd, create_branch, create_commit
from huggingface_hub.utils import HfHubHTTPError
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers.utils.generic import working_or_temp_dir
from transformers.utils.hub import PushToHubMixin

from easydel.layers.quantization import EasyDeLQuantizationConfig
from easydel.utils.readme_generator import ModelInfo, ReadmeGenerator
from easydel.utils.traversals import flatten_dict, is_flatten, merge_model_and_tree, string_key_to_int, unflatten_dict

from ..base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict
from ..etils import EasyDeLBackends, EasyDeLPlatforms

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


@dataclass
class TorchLoadOptions:
    """Options for torch_load_mode in _from_torch_pretrained."""

    mode: str
    streaming_cache: str
    streaming_tmp_dir: str | None
    hub_kwargs: dict[str, tp.Any]


@dataclass
class StreamingCheckpointInfo:
    """Metadata for streaming PyTorch checkpoint conversion."""

    ckpt_weight_format: tp.Literal["safetensors", "bin"]
    ckpt_key_to_filename: dict[str, str]
    ckpt_filename_to_path: dict[str, str]
    ed_config: tp.Any
    generation_config: tp.Any
    pretrained_model_name_or_path: str
    resolve_shard: tp.Callable[[str, str | None], str]


def _parse_torch_load_options(kwargs: dict[str, tp.Any]) -> TorchLoadOptions:
    """Parse torch_load_mode related kwargs and return a TorchLoadOptions struct."""
    torch_load_mode = str(kwargs.pop("torch_load_mode", "full")).lower()
    if torch_load_mode not in {"full", "streaming"}:
        warnings.warn(
            f"Unknown torch_load_mode={torch_load_mode!r}; falling back to 'streaming'. Expected: 'full'|'streaming'.",
            stacklevel=2,
        )
        torch_load_mode = "streaming"

    torch_streaming_cache = str(kwargs.pop("torch_streaming_cache", "hf_cache")).lower()
    torch_streaming_tmp_dir = kwargs.pop("torch_streaming_tmp_dir", None)
    if torch_load_mode == "streaming" and torch_streaming_cache not in {"hf_cache", "temp"}:
        warnings.warn(
            f"Unknown torch_streaming_cache={torch_streaming_cache!r}; falling back to 'hf_cache'. "
            "Expected: 'hf_cache'|'temp'.",
            stacklevel=2,
        )
        torch_streaming_cache = "hf_cache"

    hub_kwargs = {
        k: kwargs[k] for k in ("cache_dir", "revision", "token", "local_files_only", "force_download") if k in kwargs
    }
    if "subfolder" in kwargs:
        hub_kwargs["subfolder"] = kwargs["subfolder"]
    if "proxies" in kwargs:
        hub_kwargs["proxies"] = kwargs["proxies"]

    return TorchLoadOptions(
        mode=torch_load_mode,
        streaming_cache=torch_streaming_cache,
        streaming_tmp_dir=torch_streaming_tmp_dir,
        hub_kwargs=hub_kwargs,
    )


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
        *,
        step: int | None = None,
    ):
        """Saves the model's configuration, weights, and potentially the generation config to the specified directory.

        Args:
            save_directory (ePathLike): The directory where the model files will be saved.
            gather_fns (dict[Callable], optional): Custom gather functions for checkpoint saving.
                Defaults to None, which uses the model's default gather functions.
            float_dtype (dtype, optional): Data type for saving weights. Defaults to None.
            step (int, optional): The training step number for versioned checkpoints.
                Defaults to None.
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
        output_model_file = Checkpointer(
            base_path=str(save_directory),
            save_interval=None,
            step_policies=[],
        ).save_pytree(
            tree=state.to_pure_dict(),
            prefix="model",
            mesh=self.mesh,
            dtype=float_dtype,
            # Don't pass step here - save_directory is already the checkpoint directory
            # Passing step would create duplicate run-{step}/run-{step} structure
        )

        logger.info(f"Model weights saved in {output_model_file}")

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        push_to_hub: bool = False,
        token: str | bool | None = None,
        gather_fns: dict[tp.Callable] | None = None,
        float_dtype: jnp.dtype | None = None,
        step: int | None = None,
        **kwargs,
    ):
        """Saves the model, its configuration, and optionally pushes it to the Hugging Face Hub.

        Args:
            save_directory (str or PathLike): The directory where to save the model.
            push_to_hub (bool, optional): If True, pushes the model to the Hugging Face Hub.
                Defaults to False.
            token (str or bool, optional): The Hugging Face Hub token for authentication.
                Defaults to None.
            gather_fns (dict[Callable], optional): Custom gather functions for checkpoint saving.
                Defaults to None.
            float_dtype (jnp.dtype, optional): Data type for saving weights. Defaults to None.
            step (int, optional): The training step number for versioned checkpoints.
                Defaults to None.
            **kwargs: Additional keyword arguments for Hugging Face Hub (e.g., repo_id,
                commit_message).
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
            step=step,
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

    def _upload_modified_files(
        self,
        working_dir: str | os.PathLike,
        repo_id: str,
        files_timestamps: dict[str, float],
        commit_message: str | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
        revision: str | None = None,
        commit_description: str | None = None,
    ):
        """
        Uploads all modified files under `working_dir` to `repo_id`, at arbitrary depth, based on `files_timestamps`.

        files_timestamps should ideally map each file's relative POSIX path from `working_dir` (e.g. "a/b/c.txt")
        to the last-uploaded mtime (float). For backward compatibility, if a full relative path key is missing,
        this function falls back to a top-level key (first path segment) if present.
        """
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"

        operations = []
        modified_paths = []

        for root, dirs, files in os.walk(working_dir):
            if ".git" in dirs:
                dirs.remove(".git")

            for name in files:
                src = os.path.join(root, name)
                rel = os.path.relpath(src, working_dir)
                rel_posix = rel.replace(os.sep, "/")

                mtime = os.path.getmtime(src)
                last = files_timestamps.get(rel_posix)

                if last is None:
                    top_level = rel_posix.split("/", 1)[0]
                    top_last = files_timestamps.get(top_level)
                    if top_last is not None and mtime <= top_last:
                        continue
                else:
                    if mtime <= last:
                        continue

                operations.append(CommitOperationAdd(path_or_fileobj=src, path_in_repo=rel_posix))
                modified_paths.append(rel_posix)

        if not operations:
            logger.info("No modified files found in %s; nothing to upload.", working_dir)
            return None

        if revision is not None and not revision.startswith("refs/pr"):
            try:
                create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
            except HfHubHTTPError as e:
                if e.response.status_code == 403 and create_pr:
                    pass
                else:
                    raise

        logger.info(f"Uploading the files to {repo_id}")
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            create_pr=create_pr,
            revision=revision,
        )

    @classmethod
    def _load_model_weights(
        cls,
        resolved_archive_file: str | None,
        model: EasyDeLBaseModule,
        param_dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
        shard_fns: dict[tp.Callable] | None,
        quantization_config: EasyDeLQuantizationConfig | None,
        quantize_tensors: bool,
        vebose: bool,
    ) -> EasyDeLBaseModule:
        """Loads model weights from a checkpoint file.

        Args:
            resolved_archive_file (str | None): The path to the checkpoint file.
                Can be None if no weights file is provided.
            model (EasyDeLBaseModule): The EasyDeL model instance to load weights into.
            param_dtype (jnp.dtype): The data type for model parameters.
            mesh (jax.sharding.Mesh): The JAX mesh for distributed sharding.
            shard_fns (dict[Callable] | None): Custom shard functions for loading checkpoint.
                Defaults to None.
            quantization_config (EasyDeLQuantizationConfig | None): Quantization configuration
                for loading. Pass None to disable quantization.
            quantize_tensors (bool): Whether to quantize tensors during loading.
            vebose (bool): Whether to print verbose messages during loading.

        Returns:
            EasyDeLBaseModule: The model with loaded parameters.
        """
        callback = None
        if quantize_tensors and quantization_config is not None:
            from easydel.layers.quantization.quantizers import EasyQuantizer

            quantizer = EasyQuantizer(quantization_config=quantization_config)
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

        if callback is None:

            def callback(x, p):
                if shard_fns is not None:
                    key_get = p
                    if isinstance(p, str):
                        key_get = tuple(p.split("."))
                    callable_fn = shard_fns.get(key_get)
                    if callable_fn is not None:
                        x = callable_fn(x)
                return x

        extraargs = {}
        if resolved_archive_file:
            if str(resolved_archive_file).endswith(TENSORSTORE_INDEX_NAME):
                resolved_archive_file = str(resolved_archive_file)[: -len(TENSORSTORE_INDEX_NAME)]
            else:
                extraargs["callback"] = callback
            state, _ = Checkpointer(
                base_path=str(resolved_archive_file),
                save_interval=None,
                step_policies=[],
            ).load_pytree(
                mesh=mesh,
                dtype=param_dtype,
                partition_rules=model.config.get_partition_rules(),
                prefix="model",
                discover_latest=True,
                discover_raise=False,
                load_treedef=False,
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
                model = model.quantize(quantization_config=quantization_config, verbose=vebose)
            for unexpected_key in unexpected_keys:
                del state[unexpected_key]

            state = jax.tree_util.tree_map(lambda x: x.astype(param_dtype) if hasattr(x, "astype") else x, state)

            return merge_model_and_tree(model=model, tree=unflatten_dict(state))

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
        quantization_config: EasyDeLQuantizationConfig | None = None,
        quantize_tensors: bool = True,
        **kwargs,
    ):
        """Loads an EasyDeL model from a pretrained model or path.

        Args:
            pretrained_model_name_or_path (str | PathLike | None): The name or path of the
                pretrained model. Can be a HuggingFace Hub model ID or a local directory path.
            sharding_axis_dims (Sequence[int]): The dimensions for sharding axes.
                Defaults to (1, -1, 1, 1, 1).
            sharding_dcn_axis_dims (Sequence[int] | None): The dimensions for DCN (Data Center
                Network) sharding axes for multi-host setups. Defaults to None.
            sharding_axis_names (Sequence[str]): The names of sharding axes.
                Defaults to ("dp", "fsdp", "ep", "tp", "sp").
            partition_axis (PartitionAxis): The partition axis configuration for model sharding.
                Defaults to PartitionAxis().
            dtype (jnp.dtype): The data type for model computations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): The data type for model parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): The computation precision.
                Defaults to jax.lax.Precision("fastest").
            config_kwargs (dict[str, Any] | None): Additional configuration parameters to override.
                Defaults to None.
            partition_rules (tuple[tuple[str, PartitionSpec]] | None): Custom partitioning rules
                for sharding. Defaults to None.
            backend (EasyDeLBackends | None): The backend to use. Defaults to None.
            platform (EasyDeLPlatforms | None): The platform to use. Defaults to "jax".
            shard_fns (dict[Callable] | None): Custom shard functions for loading checkpoint.
                Defaults to None.
            auto_shard_model (bool): Whether to automatically shard the model. Defaults to True.
            verbose (bool): Whether to print verbose messages. Defaults to True.
            mismatch_allowed (bool): If True, allows mismatch in parameters while loading.
                Defaults to True.
            *model_args: Additional positional arguments passed to the model.
            config (EasyDeLBaseConfig | str | PathLike | None): Model configuration or path
                to configuration file. Defaults to None.
            cache_dir (str | PathLike | None): Directory to cache downloaded models.
                Defaults to None.
            force_download (bool): Whether to force re-download of model files. Defaults to False.
            local_files_only (bool): Whether to only use local files without downloading.
                Defaults to False.
            token (str | bool | None): HuggingFace Hub authentication token. Defaults to None.
            revision (str): The specific model version to use (branch, tag, or commit).
                Defaults to "main".
            vebose (bool): Legacy parameter for verbose output. Defaults to True.
            quantization_config (EasyDeLQuantizationConfig | None): Quantization configuration
                for loading. Pass None to disable. Defaults to None.
            quantize_tensors (bool): Whether to quantize tensors during loading. Defaults to True.
            **kwargs: Additional keyword arguments (e.g., proxies, trust_remote_code, subfolder).

        Returns:
            EasyDeLBaseModule: The loaded EasyDeL model with weights.
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
                raise_errors_on_him = True
                root = ePath(pretrained_model_name_or_path) / subfolder
                archive_file, _ = _pick_first_existing(root)
                if not archive_file:
                    latest = find_latest_checkpoint(str(root))
                    if latest is not None:
                        archive_file, _ = _pick_first_existing(ePath(latest))
                        if archive_file:
                            raise_errors_on_him = False
                    if raise_errors_on_him:
                        raise FileNotFoundError(
                            f"No model weights found in '{root}'. Tried: {', '.join(CANDIDATE_FILENAMES)} "
                            "we also tried to find latest checkpoint but there was nothing ;/"
                        )
                is_local = True
            else:
                # Sometimes the path is nested under subfolder
                alt_root = ePath(subfolder) / pretrained_model_name_or_path
                if alt_root.is_dir():
                    archive_file, _ = _pick_first_existing(alt_root)
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

            if not local_files_only and not is_local:
                api.snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    token=token,
                    local_files_only=local_files_only,
                )

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
            quantization_config,
            quantize_tensors,
            vebose,
        )

        if not quantize_tensors and quantization_config is not None:
            model = model.quantize(
                quantization_config=quantization_config,
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
        if auto_shard_model:
            # double check to make sure weights are correct or just a simple non-op.
            model = model.shard_model(partition_rules=partition_rules)
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
        shard_fns: tp.Mapping[tuple, tp.Callable] | dict | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        config_kwargs: EasyDeLBaseConfigDict | None = None,
        auto_shard_model: bool = True,
        partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
        quantization_config: EasyDeLQuantizationConfig | None = None,
        quantize_tensors: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """Loads an EasyDeL model from a PyTorch pretrained checkpoint.

        This method converts PyTorch model weights to JAX format and creates an EasyDeL model.
        Supports both full loading (loads entire model into memory) and streaming loading
        (processes one shard at a time to reduce memory usage).

        Args:
            pretrained_model_name_or_path (str): The HuggingFace Hub model ID or local path
                to the PyTorch checkpoint.
            device (jax.Device | None): The JAX device for tensor placement. Defaults to None.
            dtype (jax.numpy.dtype): The data type for model computations.
                Defaults to jax.numpy.float32.
            param_dtype (jax.numpy.dtype): The data type for model parameters.
                Defaults to jax.numpy.float32.
            precision (jax.lax.Precision | None): The computation precision. Defaults to None.
            sharding_axis_dims (Sequence[int]): The dimensions for sharding axes.
                Defaults to (1, -1, 1, 1, 1).
            sharding_dcn_axis_dims (Sequence[int] | None): The dimensions for DCN sharding axes
                for multi-host setups. Defaults to None.
            sharding_axis_names (Sequence[str]): The names of sharding axes.
                Defaults to ("dp", "fsdp", "ep", "tp", "sp").
            partition_axis (PartitionAxis | None): The partition axis configuration.
                Defaults to None.
            shard_fns (Mapping[tuple, Callable] | dict | None): Custom shard functions for
                sharding tensors. Defaults to None.
            backend (EasyDeLBackends | None): The backend to use. Defaults to None.
            platform (EasyDeLPlatforms | None): The platform to use. Defaults to None.
            config_kwargs (EasyDeLBaseConfigDict | None): Additional configuration parameters.
                Defaults to None.
            auto_shard_model (bool): Whether to automatically shard the model. Defaults to True.
            partition_rules (tuple[tuple[str, PartitionSpec], ...] | None): Custom partitioning
                rules for sharding. Defaults to None.
            quantization_config (EasyDeLQuantizationConfig | None): Quantization configuration.
                Pass None to disable. Defaults to None.
            quantize_tensors (bool): Whether to quantize tensors during loading. Defaults to True.
            verbose (bool): Whether to print verbose messages. Defaults to True.
            **kwargs: Additional keyword arguments including:
                - torch_load_mode (str): "full" or "streaming". Defaults to "streaming".
                - torch_streaming_cache (str): "hf_cache" or "temp". Defaults to "hf_cache".
                - torch_streaming_tmp_dir (str): Temporary directory for streaming.
                - trust_remote_code (bool): Whether to trust remote code. Defaults to False.
                - cache_dir, revision, token, local_files_only, force_download, subfolder, proxies.

        Returns:
            EasyDeLBaseModule: The loaded EasyDeL model with converted weights.

        Raises:
            ModuleNotFoundError: If PyTorch is not installed.
        """
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

        load_options = _parse_torch_load_options(kwargs)
        trust_remote_code = kwargs.get("trust_remote_code", False)

        logger.debug(f"Downloading model config from {pretrained_model_name_or_path}")
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **load_options.hub_kwargs,
        )
        model_type: str = hf_config.model_type
        config_class, module = get_modules_by_type(model_type, task_type=cls._model_task)

        generation_config = None
        state_dict = None
        ckpt_info: StreamingCheckpointInfo | None = None

        if load_options.mode == "full":
            ed_config, generation_config, state_dict = cls._load_full_torch_checkpoint(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config_class=config_class,
                hub_kwargs=load_options.hub_kwargs,
                torch_loader=cls.get_torch_loader(),
                clear_fn=_clear,
                kwargs=kwargs,
            )
        else:
            ckpt_info = cls._resolve_streaming_checkpoint(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config_class=config_class,
                hub_kwargs=load_options.hub_kwargs,
                kwargs=kwargs,
            )
            ed_config = ckpt_info.ed_config
            generation_config = ckpt_info.generation_config

        logger.debug("adding hf_model basic EasyDeL configurations.")
        if hasattr(ed_config, "attach_custom_arguments"):
            ed_config.attach_custom_arguments()
        config_kwargs = {} if config_kwargs is None else config_kwargs
        ed_config.add_basic_configurations(
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            platform=platform,
            **config_kwargs,
        )

        logger.debug("creating easydel model")
        model = module.lazy_init(
            config=ed_config,
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
                backend=backend,
                platform=platform,
                config_kwargs=config_kwargs,
                trust_remote_code=trust_remote_code,
                model_task=cls._model_task,
            )

        logger.debug("converting huggingface-model to easydel-model.")
        uses_tie_word_embedding = getattr(hf_config, "tie_word_embeddings", False)

        quantizer = EasyQuantizer(quantization_config=quantization_config)
        callback = None
        passed_shard_fns = shard_fns
        if quantize_tensors and quantization_config is not None:
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

        if load_options.mode == "full":
            params = model.pure_transform_fn(
                state_dict,
                config=hf_config,
                device=device,
                shard_fns=passed_shard_fns,
                params_pattern_selection=None,
                remove_state_dict=True,
                uses_tie_word_embedding=uses_tie_word_embedding,
                callback=callback,
            )
            del state_dict
            _clear()
        else:
            if ckpt_info is None:
                raise RuntimeError("torch_load_mode='streaming' was selected but no checkpoint info was resolved.")

            parameters_flat = cls._convert_streaming_checkpoint_to_params(
                model=model,
                ckpt_info=ckpt_info,
                hf_config=hf_config,
                load_options=load_options,
                shard_fns=passed_shard_fns,
                callback=callback,
                device=device,
                clear_fn=_clear,
            )
            params = unflatten_dict(parameters_flat)

        if is_flatten(params):
            logger.info("converted parameters are flatten making them unflatten.")
            params = unflatten_dict(params)

        logger.debug("merging model and parameters pytree.")
        model = merge_model_and_tree(model=model, tree=params)
        logger.debug("model and parameters pytree merged.")
        if quantization_config is not None and not quantize_tensors:
            logger.debug("quantizing model.")
            model = model.quantize(
                quantization_config=quantization_config,
                verbose=verbose,
            )
        logger.debug("returning model.")
        return model

    @classmethod
    def _resolve_streaming_checkpoint(
        cls,
        pretrained_model_name_or_path: str,
        config_class: type,
        hub_kwargs: dict[str, tp.Any],
        kwargs: dict[str, tp.Any],
    ) -> StreamingCheckpointInfo:
        """Resolve checkpoint files for streaming mode, returning checkpoint info.

        This method discovers available checkpoint files (SafeTensors or PyTorch .bin format)
        and builds the metadata needed for streaming conversion without loading all weights
        into memory.

        Args:
            pretrained_model_name_or_path (str): The HuggingFace Hub model ID or local path.
            config_class (type): The EasyDeL config class to use for loading configuration.
            hub_kwargs (dict[str, Any]): Keyword arguments for HuggingFace Hub operations
                (cache_dir, revision, token, etc.).
            kwargs (dict[str, Any]): Additional keyword arguments including revision, token,
                local_files_only, force_download, proxies, subfolder, and cache_dir.

        Returns:
            StreamingCheckpointInfo: Metadata containing checkpoint format, key-to-file mappings,
                resolved file paths, EasyDeL config, generation config, and a resolve_shard
                callable for on-demand shard downloading.

        Raises:
            FileNotFoundError: If no valid PyTorch checkpoint files can be found.
            ValueError: If the shard index file is invalid (missing weight_map).
        """
        from transformers import GenerationConfig

        logger.debug(f"Resolving checkpoint files (streaming) for {pretrained_model_name_or_path}")

        ckpt_weight_format: tp.Literal["safetensors", "bin"] | None = None
        ckpt_key_to_filename: dict[str, str] = {}
        ckpt_filename_to_path: dict[str, str] = {}

        revision = kwargs.get("revision", None)
        token = kwargs.get("token", None)
        local_files_only = bool(kwargs.get("local_files_only", False))
        force_download = bool(kwargs.get("force_download", False))
        proxies = kwargs.get("proxies", None)
        subfolder = str(kwargs.get("subfolder", "") or "")
        cache_dir = kwargs.get("cache_dir", None)

        def _strip_or_keep_subfolder(filename: str) -> tuple[str, str]:
            if not subfolder:
                return filename, ""
            normalized = subfolder.strip("/").replace("\\", "/")
            if filename.startswith(normalized + "/"):
                return filename, ""
            return filename, normalized

        def _hf_download(filename: str, *, cache_dir_override: str | None = None) -> str:
            filename, effective_subfolder = _strip_or_keep_subfolder(filename)
            return huggingface_hub.hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=filename,
                subfolder=effective_subfolder,
                revision=revision,
                cache_dir=cache_dir_override if cache_dir_override is not None else cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )

        def _find_local(filename: str) -> str | None:
            if not os.path.isdir(pretrained_model_name_or_path):
                return None
            filename, effective_subfolder = _strip_or_keep_subfolder(filename)
            root = pretrained_model_name_or_path
            if effective_subfolder:
                root = os.path.join(root, effective_subfolder)
            candidate = os.path.join(root, filename)
            return candidate if os.path.isfile(candidate) else None

        def _resolve_shard(fname: str, cache_dir_override: str | None = None) -> str:
            """Unified shard resolution: local first, then HF download."""
            local_path = _find_local(fname)
            if local_path is not None:
                return local_path
            return _hf_download(fname, cache_dir_override=cache_dir_override)

        def _load_index(path: str) -> dict[str, tp.Any]:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        resolved_index_file: str | None = None
        resolved_single_file: str | None = None

        if os.path.isdir(pretrained_model_name_or_path):
            resolved_index_file = _find_local(SAFE_WEIGHTS_INDEX_NAME)
            if resolved_index_file is not None:
                ckpt_weight_format = "safetensors"
            else:
                resolved_index_file = _find_local(WEIGHTS_INDEX_NAME)
                if resolved_index_file is not None:
                    ckpt_weight_format = "bin"

            if resolved_index_file is None:
                resolved_single_file = _find_local(SAFE_WEIGHTS_NAME)
                if resolved_single_file is not None:
                    ckpt_weight_format = "safetensors"
                else:
                    resolved_single_file = _find_local(WEIGHTS_NAME)
                    if resolved_single_file is not None:
                        ckpt_weight_format = "bin"
        else:
            try:
                resolved_index_file = _hf_download(SAFE_WEIGHTS_INDEX_NAME)
                ckpt_weight_format = "safetensors"
            except (FileNotFoundError, huggingface_hub.errors.EntryNotFoundError):
                resolved_index_file = None

            if resolved_index_file is None:
                try:
                    resolved_index_file = _hf_download(WEIGHTS_INDEX_NAME)
                    ckpt_weight_format = "bin"
                except (FileNotFoundError, huggingface_hub.errors.EntryNotFoundError):
                    resolved_index_file = None

            if resolved_index_file is None:
                try:
                    resolved_single_file = _hf_download(SAFE_WEIGHTS_NAME)
                    ckpt_weight_format = "safetensors"
                except (FileNotFoundError, huggingface_hub.errors.EntryNotFoundError):
                    resolved_single_file = None

            if resolved_single_file is None:
                try:
                    resolved_single_file = _hf_download(WEIGHTS_NAME)
                    ckpt_weight_format = "bin"
                except (FileNotFoundError, huggingface_hub.errors.EntryNotFoundError):
                    resolved_single_file = None

        if ckpt_weight_format is None or (resolved_index_file is None and resolved_single_file is None):
            raise FileNotFoundError(
                "Couldn't locate a PyTorch checkpoint to convert. Expected one of: "
                f"{SAFE_WEIGHTS_INDEX_NAME}, {SAFE_WEIGHTS_NAME}, {WEIGHTS_INDEX_NAME}, {WEIGHTS_NAME}"
            )

        if resolved_index_file is not None:
            index_data = _load_index(resolved_index_file)
            weight_map = index_data.get("weight_map") or {}
            if not weight_map:
                raise ValueError(f"Invalid shard index file (missing weight_map): {resolved_index_file}")
            ckpt_key_to_filename = dict(weight_map)
            for fname in sorted(set(weight_map.values())):
                if os.path.isdir(pretrained_model_name_or_path):
                    resolved = _find_local(fname)
                    if resolved is None:
                        raise FileNotFoundError(f"Missing shard file {fname!r} under {pretrained_model_name_or_path!r}")
                    ckpt_filename_to_path[fname] = resolved
        else:
            assert resolved_single_file is not None
            if ckpt_weight_format == "safetensors":
                ckpt_filename_to_path[SAFE_WEIGHTS_NAME] = resolved_single_file
            else:
                ckpt_filename_to_path[WEIGHTS_NAME] = resolved_single_file

        ed_config = config_class.from_pretrained(pretrained_model_name_or_path, **hub_kwargs)
        try:
            generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, **hub_kwargs)
        except Exception:
            generation_config = None

        return StreamingCheckpointInfo(
            ckpt_weight_format=ckpt_weight_format,
            ckpt_key_to_filename=ckpt_key_to_filename,
            ckpt_filename_to_path=ckpt_filename_to_path,
            ed_config=ed_config,
            generation_config=generation_config,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            resolve_shard=_resolve_shard,
        )

    @classmethod
    def _convert_streaming_checkpoint_to_params(
        cls,
        model,
        ckpt_info: StreamingCheckpointInfo,
        hf_config: tp.Any,
        load_options: TorchLoadOptions,
        shard_fns: dict[tuple, tp.Callable] | None,
        callback: tp.Callable | None,
        device: tp.Any | None,
        clear_fn: tp.Callable[[], None],
    ) -> dict[tuple, tp.Any]:
        """Convert streaming PyTorch checkpoint to flattened JAX params dict.

        This function handles the streaming conversion of PyTorch checkpoints to JAX format,
        including MoE expert grouping and multi-file handling.

        Args:
            model (EasyDeLBaseModule): The EasyDeL model instance used for transformation
                configuration and parameter shape information.
            ckpt_info (StreamingCheckpointInfo): Checkpoint metadata with paths and
                resolve_shard callable.
            hf_config (Any): The HuggingFace config for the model.
            load_options (TorchLoadOptions): Options with streaming cache settings.
            shard_fns (dict[tuple, Callable] | None): Optional sharding functions to apply
                to tensors.
            callback (Callable | None): Optional callback for post-processing tensors
                (e.g., quantization).
            device (Any | None): Optional JAX device for tensor placement.
            clear_fn (Callable[[], None]): Function to call for memory cleanup between shards.

        Returns:
            dict[tuple, Any]: Flattened dict of (tuple_key -> jax_array) ready for unflattening.
        """
        import torch
        from safetensors.torch import safe_open

        from easydel.utils.parameters_transformation import StateDictConverter

        ckpt_weight_format = ckpt_info.ckpt_weight_format
        ckpt_key_to_filename = ckpt_info.ckpt_key_to_filename
        ckpt_filename_to_path = ckpt_info.ckpt_filename_to_path
        pretrained_model_name_or_path = ckpt_info.pretrained_model_name_or_path
        resolve_shard = ckpt_info.resolve_shard
        torch_streaming_cache = load_options.streaming_cache
        torch_streaming_tmp_dir = load_options.streaming_tmp_dir

        transformer = model.pure_transform_fn
        embedding_layer_names = transformer.keywords.get("embedding_layer_names")
        layernorm_names = transformer.keywords.get("layernorm_names")
        moe_block_names = transformer.keywords.get("moe_block_names")
        moe_names = transformer.keywords.get("moe_names")
        moe_block_path = transformer.keywords.get("moe_block_path")
        moe_path = transformer.keywords.get("moe_path")
        reform_param = transformer.keywords.get("reform_param")

        moe_names_set = set(moe_names or [])
        excepted_expert_name = moe_path[0].split(".")[-2] if moe_path else "experts"
        expert_prefix = f".{excepted_expert_name}."

        consolidated_moe_keys: set[str] = set()
        moe_groups: dict[str, dict[int, str]] = {}
        moe_expert_keys: set[str] = set()

        def _register_moe_key(k: str):
            if not (moe_block_path and moe_names_set and moe_path):
                return
            if expert_prefix not in k:
                return
            for block_path in moe_block_path:
                block_expert_prefix = block_path + expert_prefix
                if not k.startswith(block_expert_prefix):
                    continue
                remainder = k[len(block_expert_prefix) :]
                dot_idx = remainder.find(".")
                if dot_idx <= 0:
                    continue
                expert_part = remainder[:dot_idx]
                if not expert_part.isdigit():
                    continue
                expert_idx = int(expert_part)
                moe_name_part = remainder[dot_idx + 1 :]
                moe_name = moe_name_part[:-7] if moe_name_part.endswith(".weight") else moe_name_part
                if moe_name not in moe_names_set:
                    continue
                target_path = f"{block_path}.{excepted_expert_name}.{moe_name}"
                moe_groups.setdefault(target_path, {})[expert_idx] = k
                moe_expert_keys.add(k)
                return

        if not ckpt_key_to_filename:
            filename = next(iter(ckpt_filename_to_path.keys()))
            resolved_path = ckpt_filename_to_path[filename]
            if ckpt_weight_format == "safetensors":
                with safe_open(resolved_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        ckpt_key_to_filename[k] = filename
                        _register_moe_key(k)
            else:
                shard = None
                try:
                    shard = torch.load(resolved_path, map_location="cpu", weights_only=True)
                except TypeError:
                    shard = torch.load(resolved_path, map_location="cpu")
                for k in shard.keys():
                    ckpt_key_to_filename[k] = filename
                    _register_moe_key(k)
                del shard
                clear_fn()
        else:
            for k in ckpt_key_to_filename:
                _register_moe_key(k)

        for target_path in moe_groups:
            consolidated_moe_keys.add(f"{target_path}.weight")

        uses_tie_word_embedding = getattr(hf_config, "tie_word_embeddings", False)
        converter_config = {
            "embedding_layer_names": set(embedding_layer_names or []),
            "layernorm_names": set(layernorm_names or []),
            "moe_block_names": set(moe_block_names or []),
            "moe_names": set(moe_names or []),
            "lm_head_name": None,
            "uses_tie_word_embedding": uses_tie_word_embedding,
            "dtype": model.param_dtype,
            "consolidated_moe_keys": consolidated_moe_keys,
            "reform_param": reform_param,
        }

        parameters_flat: dict[tuple, tp.Any] = {}

        def _maybe_shard_and_callback(key_tuple, arr):
            if shard_fns and key_tuple in shard_fns:
                arr = shard_fns[key_tuple](arr)
            if callback is not None:
                arr = callback(arr, key_tuple)
            return arr

        def _process_tensor(key: str, tensor):
            results = StateDictConverter.process_tensor(key, tensor, converter_config)
            if results is None:
                return
            for key_tuple, jax_array in results:
                parameters_flat[key_tuple] = _maybe_shard_and_callback(key_tuple, jax_array)

        @contextlib.contextmanager
        def _with_resolved_shard(fname: str):
            if fname in ckpt_filename_to_path and os.path.isfile(ckpt_filename_to_path[fname]):
                yield ckpt_filename_to_path[fname]
                return

            if os.path.isdir(pretrained_model_name_or_path):
                resolved = resolve_shard(fname, None)
                ckpt_filename_to_path[fname] = resolved
                yield resolved
                return

            if torch_streaming_cache == "hf_cache":
                resolved = resolve_shard(fname, None)
                ckpt_filename_to_path[fname] = resolved
                yield resolved
                return

            if torch_streaming_tmp_dir is not None:
                if os.path.isfile(torch_streaming_tmp_dir):
                    raise NotADirectoryError(f"torch_streaming_tmp_dir points to a file: {torch_streaming_tmp_dir!r}")
                os.makedirs(torch_streaming_tmp_dir, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=torch_streaming_tmp_dir) as tmpdir:
                yield resolve_shard(fname, tmpdir)

        def _run_streaming_conversion():
            file_to_non_moe_keys: dict[str, list[str]] = {}
            for key, fname in ckpt_key_to_filename.items():
                if key in moe_expert_keys:
                    continue
                file_to_non_moe_keys.setdefault(fname, []).append(key)

            file_to_moe_groups: dict[str, list[tuple[str, dict[int, str]]]] = {}
            multi_file_moe_groups: dict[str, dict[int, str]] = {}
            for target_path, expert_key_by_idx in moe_groups.items():
                files = {ckpt_key_to_filename[k] for k in expert_key_by_idx.values() if k in ckpt_key_to_filename}
                if len(files) == 1:
                    fname = next(iter(files))
                    file_to_moe_groups.setdefault(fname, []).append((target_path, expert_key_by_idx))
                else:
                    multi_file_moe_groups[target_path] = expert_key_by_idx

            all_files = sorted(set(ckpt_key_to_filename.values()) | set(ckpt_filename_to_path.keys()))

            for fname in all_files:
                non_moe_keys = file_to_non_moe_keys.get(fname, [])
                moe_groups_for_file = file_to_moe_groups.get(fname, [])
                if not non_moe_keys and not moe_groups_for_file:
                    continue

                with _with_resolved_shard(fname) as resolved_path:
                    if ckpt_weight_format == "safetensors":
                        with safe_open(resolved_path, framework="pt", device="cpu") as f:
                            for k in sorted(non_moe_keys):
                                _process_tensor(k, f.get_tensor(k))

                            for target_path, expert_key_by_idx in moe_groups_for_file:
                                expert_indices = sorted(expert_key_by_idx.keys())
                                if not expert_indices:
                                    continue
                                stacked_key = f"{target_path}.weight"
                                first_key = expert_key_by_idx[expert_indices[0]]
                                first_tensor = f.get_tensor(first_key)
                                stacked_shape = (len(expert_indices), *first_tensor.shape)
                                stacked_tensor = torch.empty(
                                    stacked_shape,
                                    dtype=first_tensor.dtype,
                                    device=first_tensor.device,
                                )
                                for pos, expert_idx in enumerate(expert_indices):
                                    stacked_tensor[pos] = f.get_tensor(expert_key_by_idx[expert_idx])
                                _process_tensor(stacked_key, stacked_tensor)
                                del stacked_tensor
                                clear_fn()
                    else:
                        shard = None
                        try:
                            shard = torch.load(resolved_path, map_location="cpu", weights_only=True)
                        except TypeError:
                            shard = torch.load(resolved_path, map_location="cpu")

                        for k in sorted(non_moe_keys):
                            _process_tensor(k, shard[k])

                        for target_path, expert_key_by_idx in moe_groups_for_file:
                            expert_indices = sorted(expert_key_by_idx.keys())
                            if not expert_indices:
                                continue
                            stacked_key = f"{target_path}.weight"
                            first_key = expert_key_by_idx[expert_indices[0]]
                            first_tensor = shard[first_key]
                            stacked_shape = (len(expert_indices), *first_tensor.shape)
                            stacked_tensor = torch.empty(
                                stacked_shape,
                                dtype=first_tensor.dtype,
                                device=first_tensor.device,
                            )
                            for pos, expert_idx in enumerate(expert_indices):
                                stacked_tensor[pos] = shard[expert_key_by_idx[expert_idx]]
                            _process_tensor(stacked_key, stacked_tensor)
                            del stacked_tensor
                            clear_fn()

                        del shard
                        clear_fn()

            if multi_file_moe_groups:
                if torch_streaming_cache == "temp":
                    raise RuntimeError(
                        "Encountered MoE expert groups spanning multiple shard files while "
                        "torch_streaming_cache='temp'. Use torch_streaming_cache='hf_cache' "
                        "(and a local cache_dir) to avoid re-downloading shards."
                    )

                for target_path, expert_key_by_idx in sorted(multi_file_moe_groups.items()):
                    expert_indices = sorted(expert_key_by_idx.keys())
                    if not expert_indices:
                        continue

                    stacked_key = f"{target_path}.weight"
                    first_key = expert_key_by_idx[expert_indices[0]]
                    first_fname = ckpt_key_to_filename[first_key]
                    with _with_resolved_shard(first_fname) as first_path:
                        if ckpt_weight_format == "safetensors":
                            with safe_open(first_path, framework="pt", device="cpu") as f:
                                first_tensor = f.get_tensor(first_key)
                        else:
                            shard = None
                            try:
                                shard = torch.load(first_path, map_location="cpu", weights_only=True)
                            except TypeError:
                                shard = torch.load(first_path, map_location="cpu")
                            first_tensor = shard[first_key]
                            del shard
                            clear_fn()

                    stacked_shape = (len(expert_indices), *first_tensor.shape)
                    stacked_tensor = torch.empty(
                        stacked_shape,
                        dtype=first_tensor.dtype,
                        device=first_tensor.device,
                    )

                    pos_by_expert = {expert_idx: pos for pos, expert_idx in enumerate(expert_indices)}
                    per_file: dict[str, list[tuple[int, str]]] = {}
                    for expert_idx, key in expert_key_by_idx.items():
                        fname = ckpt_key_to_filename[key]
                        per_file.setdefault(fname, []).append((pos_by_expert[expert_idx], key))

                    for fname, items in per_file.items():
                        with _with_resolved_shard(fname) as resolved_path:
                            if ckpt_weight_format == "safetensors":
                                with safe_open(resolved_path, framework="pt", device="cpu") as f:
                                    for pos, key in items:
                                        stacked_tensor[pos] = f.get_tensor(key)
                            else:
                                shard = None
                                try:
                                    shard = torch.load(resolved_path, map_location="cpu", weights_only=True)
                                except TypeError:
                                    shard = torch.load(resolved_path, map_location="cpu")
                                for pos, key in items:
                                    stacked_tensor[pos] = shard[key]
                                del shard
                                clear_fn()

                    _process_tensor(stacked_key, stacked_tensor)
                    del stacked_tensor
                    clear_fn()

        if device is not None:
            with jax.default_device(device):
                _run_streaming_conversion()
        else:
            _run_streaming_conversion()

        return parameters_flat

    @classmethod
    def _load_full_torch_checkpoint(
        cls,
        pretrained_model_name_or_path: str,
        config_class: type,
        hub_kwargs: dict[str, tp.Any],
        torch_loader: tp.Any,
        clear_fn: tp.Callable[[], None],
        kwargs: dict[str, tp.Any],
    ) -> tuple[tp.Any, tp.Any, dict[str, tp.Any]]:
        """Load full PyTorch checkpoint into memory (for torch_load_mode='full').

        Args:
            pretrained_model_name_or_path (str): The HuggingFace Hub model ID or local path.
            config_class (type): The EasyDeL config class to use for loading configuration.
            hub_kwargs (dict[str, Any]): Keyword arguments for HuggingFace Hub operations
                (cache_dir, revision, token, etc.).
            torch_loader (Any): The HuggingFace AutoModel class for loading the PyTorch model.
            clear_fn (Callable[[], None]): Function to call for memory cleanup after loading.
            kwargs (dict[str, Any]): Additional keyword arguments for model loading
                (e.g., torch_dtype).

        Returns:
            tuple[Any, Any, dict[str, Any]]: A tuple of (ed_config, generation_config, state_dict)
                where ed_config is the EasyDeL configuration, generation_config is the
                generation configuration (if available), and state_dict contains the model weights.
        """
        import torch

        logger.debug(f"Downloading hf_model weights from {pretrained_model_name_or_path}")
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16
        torch_dtype = kwargs.pop("torch_dtype")
        hf_model = torch_loader.from_pretrained(
            pretrained_model_name_or_path,
            dtype=torch_dtype,
            **kwargs,
        )
        generation_config = getattr(hf_model, "generation_config", None)
        ed_config = config_class.from_pretrained(pretrained_model_name_or_path, **hub_kwargs)
        state_dict = hf_model.state_dict()

        del hf_model
        clear_fn()
        return ed_config, generation_config, state_dict

    @classmethod
    def huggingface_to_easydel_sequential(
        cls,
        pretrained_model_name_or_path: str,
        save_directory: str | os.PathLike,
        *,
        output_repo_id: str | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        config_kwargs: EasyDeLBaseConfigDict | None = None,
        partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
        trust_remote_code: bool = False,
        torch_streaming_cache: str = "temp",
        torch_streaming_tmp_dir: str | None = None,
        tensorstore_chunk_bytes: int = 2_147_483_648,
        verbose: bool = True,
        **kwargs,
    ) -> str:
        """Convert a HuggingFace PyTorch checkpoint to an EasyDeL checkpoint sequentially.

        This avoids materializing the full `state_dict`/params tree in memory by:
        - downloading one shard at a time (optionally to a temp dir),
        - converting each tensor to the EasyDeL naming/layout,
        - writing each tensor immediately into a TensorStore (zarr) checkpoint.

        MoE expert weights are written slice-by-slice to avoid allocating stacked expert tensors.

        Args:
            pretrained_model_name_or_path (str): The HuggingFace Hub model ID or local path
                to the PyTorch checkpoint.
            save_directory (str | PathLike): The directory where the converted checkpoint
                will be saved.
            output_repo_id (str | None): If provided, it is used in the generated README.md
                so the model card points to the final Hub repo. Defaults to None.
            dtype (jnp.dtype): The data type for model computations. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): The data type for model parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.Precision | None): The computation precision. Defaults to None.
            sharding_axis_dims (Sequence[int]): The dimensions for sharding axes.
                Defaults to (1, -1, 1, 1, 1).
            sharding_dcn_axis_dims (Sequence[int] | None): The dimensions for DCN sharding axes
                for multi-host setups. Defaults to None.
            sharding_axis_names (Sequence[str]): The names of sharding axes.
                Defaults to ("dp", "fsdp", "ep", "tp", "sp").
            partition_axis (PartitionAxis | None): The partition axis configuration.
                Defaults to None.
            backend (EasyDeLBackends | None): The backend to use. Defaults to None.
            platform (EasyDeLPlatforms | None): The platform to use. Defaults to None.
            config_kwargs (EasyDeLBaseConfigDict | None): Additional configuration parameters
                to override. Defaults to None.
            partition_rules (tuple[tuple[str, PartitionSpec], ...] | None): Custom partitioning
                rules for sharding. Defaults to None.
            trust_remote_code (bool): Whether to trust remote code from HuggingFace Hub.
                Defaults to False.
            torch_streaming_cache (str): Where to cache downloaded shards. Options are
                "hf_cache" (uses HuggingFace cache) or "temp" (uses temporary directory).
                Defaults to "temp".
            torch_streaming_tmp_dir (str | None): Custom temporary directory for streaming.
                Only used when torch_streaming_cache="temp". Defaults to None.
            tensorstore_chunk_bytes (int): The chunk size in bytes for TensorStore arrays.
                Defaults to 2_147_483_648 (2GB).
            verbose (bool): Whether to print verbose messages. Defaults to True.
            **kwargs: Additional keyword arguments (e.g., cache_dir, revision, token,
                local_files_only, force_download, subfolder, proxies).

        Returns:
            str: The path to the saved checkpoint directory.
        """
        from datetime import datetime

        import tensorstore as ts
        from eformer.escale import match_partition_rules
        from huggingface_hub.errors import EntryNotFoundError
        from jax.experimental.array_serialization import serialization as jax_ser
        from jax.experimental.array_serialization import tensorstore_impl as ts_impl
        from transformers import AutoConfig, GenerationConfig

        from easydel.modules.auto.auto_configuration import get_modules_by_type
        from easydel.utils.parameters_transformation import StateDictConverter, TensorConverter

        if jax.process_count() > 1 and jax.process_index() != 0:
            logger.info("Skipping sequential conversion on non-zero process index.")
            return str(save_directory)

        torch_streaming_cache = str(torch_streaming_cache).lower().strip()
        if torch_streaming_cache not in {"hf_cache", "temp"}:
            warnings.warn(
                f"Unknown torch_streaming_cache={torch_streaming_cache!r}; falling back to 'temp'. "
                "Expected: 'hf_cache'|'temp'.",
                stacklevel=1,
            )
            torch_streaming_cache = "temp"

        cpu_device = None
        try:
            cpu_device = jax.devices("cpu")[0]
        except Exception:
            cpu_device = None
        import contextlib

        convert_ctx = jax.default_device(cpu_device) if cpu_device is not None else contextlib.nullcontext()

        save_root = ePath(save_directory)
        save_root.mkdir(parents=True, exist_ok=True)

        hub_kwargs = {
            k: kwargs[k] for k in ("cache_dir", "revision", "token", "local_files_only", "force_download") if k in kwargs
        }
        if "subfolder" in kwargs:
            hub_kwargs["subfolder"] = kwargs["subfolder"]
        if "proxies" in kwargs:
            hub_kwargs["proxies"] = kwargs["proxies"]

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **hub_kwargs,
        )
        model_type: str = hf_config.model_type
        ed_config_cls, module = get_modules_by_type(model_type, task_type=cls._model_task)

        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
            )
        except Exception:
            generation_config = None

        ed_config: EasyDeLBaseConfig = ed_config_cls.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **hub_kwargs,
        )
        if hasattr(ed_config, "attach_custom_arguments"):
            ed_config.attach_custom_arguments()
        config_kwargs = {} if config_kwargs is None else config_kwargs
        if partition_axis is None:
            partition_axis = PartitionAxis()
        add_config_overrides: dict[str, tp.Any] = {}
        if isinstance(getattr(ed_config, "gradient_checkpointing", None), bool):
            from easydel.infra.etils import EasyDeLGradientCheckPointers

            add_config_overrides["gradient_checkpointing"] = (
                EasyDeLGradientCheckPointers.CHECKPOINT_DOTS
                if bool(ed_config.gradient_checkpointing)
                else EasyDeLGradientCheckPointers.NONE
            )
        ed_config.add_basic_configurations(
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            platform=platform,
            **add_config_overrides,
            **config_kwargs,
        )

        model = module.lazy_init(
            config=ed_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=nn.Rngs(0),
        )

        required_params = set(flatten_dict(model.graphtree_params_shape))
        if partition_rules is None:
            try:
                partition_rules = ed_config.get_partition_rules(True)
            except Exception:
                partition_rules = None
        spec_map: dict[tuple, PartitionSpec] = {}
        if partition_rules is not None:
            try:
                specs_tree = match_partition_rules(partition_rules, model.graphtree_params_shape)
                spec_map = flatten_dict(specs_tree)
            except Exception:
                spec_map = {}

        transformer = model.pure_transform_fn
        embedding_layer_names = transformer.keywords.get("embedding_layer_names")
        layernorm_names = transformer.keywords.get("layernorm_names")
        moe_block_names = transformer.keywords.get("moe_block_names")
        moe_names = transformer.keywords.get("moe_names")
        moe_block_path = transformer.keywords.get("moe_block_path")
        moe_path = transformer.keywords.get("moe_path")
        reform_param = transformer.keywords.get("reform_param")

        uses_tie_word_embedding = getattr(hf_config, "tie_word_embeddings", False)

        moe_names_set = set(moe_names or [])
        excepted_expert_name = moe_path[0].split(".")[-2] if moe_path else "experts"
        expert_prefix = f".{excepted_expert_name}."

        consolidated_moe_keys: set[str] = set()
        moe_groups: dict[str, dict[int, str]] = {}
        expert_key_to_group: dict[str, tuple[str, int]] = {}

        def _register_moe_key(k: str):
            if not (moe_block_path and moe_names_set and moe_path):
                return
            if expert_prefix not in k:
                return
            for block_path in moe_block_path:
                block_expert_prefix = block_path + expert_prefix
                if not k.startswith(block_expert_prefix):
                    continue
                remainder = k[len(block_expert_prefix) :]
                dot_idx = remainder.find(".")
                if dot_idx <= 0:
                    continue
                expert_part = remainder[:dot_idx]
                if not expert_part.isdigit():
                    continue
                expert_idx = int(expert_part)
                moe_name_part = remainder[dot_idx + 1 :]
                moe_name = moe_name_part[:-7] if moe_name_part.endswith(".weight") else moe_name_part
                if moe_name not in moe_names_set:
                    continue
                target_path = f"{block_path}.{excepted_expert_name}.{moe_name}"
                moe_groups.setdefault(target_path, {})[expert_idx] = k
                expert_key_to_group[k] = (target_path, expert_idx)
                return

        ckpt_weight_format: tp.Literal["safetensors", "bin"] | None = None
        ckpt_key_to_filename: dict[str, str] = {}

        revision = kwargs.get("revision", None)
        token = kwargs.get("token", None)
        local_files_only = bool(kwargs.get("local_files_only", False))
        force_download = bool(kwargs.get("force_download", False))
        proxies = kwargs.get("proxies", None)
        subfolder = str(kwargs.get("subfolder", "") or "")
        cache_dir = kwargs.get("cache_dir", None)

        def _strip_or_keep_subfolder(filename: str) -> tuple[str, str]:
            if not subfolder:
                return filename, ""
            normalized = subfolder.strip("/").replace("\\", "/")
            if filename.startswith(normalized + "/"):
                return filename, ""
            return filename, normalized

        def _hf_download(filename: str, *, cache_dir_override: str | None = None) -> str:
            filename, effective_subfolder = _strip_or_keep_subfolder(filename)
            return huggingface_hub.hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=filename,
                subfolder=effective_subfolder,
                revision=revision,
                cache_dir=cache_dir_override if cache_dir_override is not None else cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )

        def _find_local(filename: str) -> str | None:
            if not os.path.isdir(pretrained_model_name_or_path):
                return None
            filename, effective_subfolder = _strip_or_keep_subfolder(filename)
            root = pretrained_model_name_or_path
            if effective_subfolder:
                root = os.path.join(root, effective_subfolder)
            candidate = os.path.join(root, filename)
            return candidate if os.path.isfile(candidate) else None

        def _load_index(path: str) -> dict[str, tp.Any]:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        resolved_index_file: str | None = None
        resolved_single_file: str | None = None

        if os.path.isdir(pretrained_model_name_or_path):
            resolved_index_file = _find_local(SAFE_WEIGHTS_INDEX_NAME)
            if resolved_index_file is not None:
                ckpt_weight_format = "safetensors"
            else:
                resolved_index_file = _find_local(WEIGHTS_INDEX_NAME)
                if resolved_index_file is not None:
                    ckpt_weight_format = "bin"

            if resolved_index_file is None:
                resolved_single_file = _find_local(SAFE_WEIGHTS_NAME)
                if resolved_single_file is not None:
                    ckpt_weight_format = "safetensors"
                else:
                    resolved_single_file = _find_local(WEIGHTS_NAME)
                    if resolved_single_file is not None:
                        ckpt_weight_format = "bin"
        else:
            try:
                resolved_index_file = _hf_download(SAFE_WEIGHTS_INDEX_NAME)
                ckpt_weight_format = "safetensors"
            except (FileNotFoundError, EntryNotFoundError):
                resolved_index_file = None

            if resolved_index_file is None:
                try:
                    resolved_index_file = _hf_download(WEIGHTS_INDEX_NAME)
                    ckpt_weight_format = "bin"
                except (FileNotFoundError, EntryNotFoundError):
                    resolved_index_file = None

            if resolved_index_file is None:
                try:
                    resolved_single_file = _hf_download(SAFE_WEIGHTS_NAME)
                    ckpt_weight_format = "safetensors"
                except (FileNotFoundError, EntryNotFoundError):
                    resolved_single_file = None

            if resolved_single_file is None:
                try:
                    resolved_single_file = _hf_download(WEIGHTS_NAME)
                    ckpt_weight_format = "bin"
                except (FileNotFoundError, EntryNotFoundError):
                    resolved_single_file = None

        if ckpt_weight_format is None or (resolved_index_file is None and resolved_single_file is None):
            raise FileNotFoundError(
                "Couldn't locate a PyTorch checkpoint to convert. Expected one of: "
                f"{SAFE_WEIGHTS_INDEX_NAME}, {SAFE_WEIGHTS_NAME}, {WEIGHTS_INDEX_NAME}, {WEIGHTS_NAME}"
            )

        ckpt_filename_to_path: dict[str, str] = {}
        if resolved_index_file is not None:
            index_data = _load_index(resolved_index_file)
            weight_map = index_data.get("weight_map") or {}
            if not weight_map:
                raise ValueError(f"Invalid shard index file (missing weight_map): {resolved_index_file}")
            ckpt_key_to_filename = dict(weight_map)
            for k in ckpt_key_to_filename:
                _register_moe_key(k)
            for target_path in moe_groups:
                consolidated_moe_keys.add(f"{target_path}.weight")
        else:
            assert resolved_single_file is not None
            filename = SAFE_WEIGHTS_NAME if ckpt_weight_format == "safetensors" else WEIGHTS_NAME
            ckpt_filename_to_path[filename] = resolved_single_file

        converter_config = {
            "embedding_layer_names": set(embedding_layer_names or []),
            "layernorm_names": set(layernorm_names or []),
            "moe_block_names": set(moe_block_names or []),
            "moe_names": set(moe_names or []),
            "lm_head_name": None,
            "uses_tie_word_embedding": uses_tie_word_embedding,
            "dtype": model.param_dtype,
            "consolidated_moe_keys": consolidated_moe_keys,
            "reform_param": reform_param,
        }

        def _chunk_shape_for(
            key_tuple: tuple, global_shape: tuple[int, ...], dtype_: jnp.dtype, *, moe: bool
        ) -> list[int]:
            spec = spec_map.get(key_tuple)
            local_shape = global_shape
            if spec is not None:
                try:
                    sharding = jax.sharding.NamedSharding(ed_config.mesh, spec)
                    local_shape = tuple(int(x) for x in sharding.shard_shape(global_shape))
                except Exception:
                    local_shape = global_shape
            chunk = ts_impl._compute_chunk_shape(
                tuple(int(max(1, d)) for d in local_shape), dtype_, tensorstore_chunk_bytes
            )
            if moe and chunk:
                chunk = [1, *list(chunk[1:])]
            return [int(max(1, d)) for d in chunk]

        def _tensorstore_path_for_params(key_tuple: tuple) -> tuple[str, str]:
            rel = os.path.join("model", *[str(p) for p in key_tuple])
            abs_path = os.path.join(str(save_root), rel)
            return abs_path, rel

        array_index: list[dict[str, tp.Any]] = []

        def _write_tensor(key_tuple: tuple, value) -> None:
            if key_tuple not in required_params:
                return
            abs_path, rel_path = _tensorstore_path_for_params(key_tuple)
            spec = jax_ser.get_tensorstore_spec(abs_path)
            chunks = _chunk_shape_for(key_tuple, tuple(int(i) for i in value.shape), value.dtype, moe=False)
            spec["metadata"] = ts_impl._get_tensorstore_metadata_cached(
                tuple(int(i) for i in value.shape),
                value.dtype,
                tuple(chunks),
                driver="zarr",
            )
            spec["dtype"] = jnp.dtype(value.dtype).name
            t = ts.open(ts.Spec(spec), create=True, open=True).result()
            t.write(value).result()
            array_index.append(
                {
                    "path": rel_path,
                    "shape": [int(i) for i in value.shape],
                    "dtype": str(value.dtype),
                }
            )

        def _write_checkpoint_index() -> None:
            index_path = save_root / TENSORSTORE_INDEX_NAME
            index_data = {
                "format": "tensorstore",
                "version": "easydel",
                "prefixes": {"model": array_index},
            }
            index_path.write_text(json.dumps(index_data, indent=2))
            meta_path = save_root / "checkpoint_metadata.json"
            meta_path.write_text(
                json.dumps(
                    {"timestamp": datetime.now().isoformat(), "custom_metadata": {"step": 0}},
                    indent=2,
                )
            )

        def _save_configs() -> None:
            config_to_save = deepcopy(ed_config)
            config_to_save.__dict__.pop("attn_dtype", None)
            config_to_save.__dict__.pop("attn_softmax_dtype", None)
            config_to_save.architectures = [module.__name__]
            config_to_save.save_pretrained(str(save_root))
            if generation_config is not None:
                generation_config.save_pretrained(str(save_root))

        _save_configs()

        import tempfile

        @contextlib.contextmanager
        def _with_resolved_shard(fname: str):
            if fname in ckpt_filename_to_path and os.path.isfile(ckpt_filename_to_path[fname]):
                yield ckpt_filename_to_path[fname]
                return

            if os.path.isdir(pretrained_model_name_or_path):
                resolved = _find_local(fname)
                if resolved is None:
                    raise FileNotFoundError(f"Missing shard file {fname!r} under {pretrained_model_name_or_path!r}")
                ckpt_filename_to_path[fname] = resolved
                yield resolved
                return

            if torch_streaming_cache == "hf_cache":
                resolved = _hf_download(fname)
                ckpt_filename_to_path[fname] = resolved
                yield resolved
                return

            if torch_streaming_tmp_dir is not None:
                if os.path.isfile(torch_streaming_tmp_dir):
                    raise NotADirectoryError(f"torch_streaming_tmp_dir points to a file: {torch_streaming_tmp_dir!r}")
                os.makedirs(torch_streaming_tmp_dir, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=torch_streaming_tmp_dir) as tmpdir:
                yield _hf_download(fname, cache_dir_override=tmpdir)

        def _iter_keys_single_file(filename: str, path: str):
            import torch
            from safetensors.torch import safe_open  # type:ignore

            if ckpt_weight_format == "safetensors":
                with safe_open(path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        yield k
            else:
                shard = None
                try:
                    shard = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    shard = torch.load(path, map_location="cpu")
                for k in shard.keys():
                    yield k
                del shard

        if not ckpt_key_to_filename:
            filename = next(iter(ckpt_filename_to_path.keys()))
            resolved_path = ckpt_filename_to_path[filename]
            for k in _iter_keys_single_file(filename, resolved_path):
                ckpt_key_to_filename[k] = filename
                _register_moe_key(k)
            for target_path in moe_groups:
                consolidated_moe_keys.add(f"{target_path}.weight")
            converter_config["consolidated_moe_keys"] = consolidated_moe_keys

        file_to_keys: dict[str, list[str]] = {}
        for k, fname in ckpt_key_to_filename.items():
            file_to_keys.setdefault(fname, []).append(k)

        moe_expected: dict[str, int] = {tp: len(expert_key_by_idx) for tp, expert_key_by_idx in moe_groups.items()}
        moe_remaining: dict[str, int] = dict(moe_expected)
        moe_handles: dict[str, dict[str, tp.Any]] = {}

        def _ensure_moe_group_ready(target_path: str, *, sample_expert_tensor) -> dict[str, tp.Any] | None:
            if target_path in moe_handles:
                return moe_handles[target_path]

            stacked_key = f"{target_path}.weight"
            out_key = stacked_key[:-7] + ".kernel" if stacked_key.endswith(".weight") else stacked_key + ".kernel"
            key_tuple = tuple(int(n) if n.isdigit() else n for n in out_key.split("."))
            if key_tuple not in required_params:
                return None

            abs_path, rel_path = _tensorstore_path_for_params(key_tuple)

            out_features, in_features = tuple(int(i) for i in sample_expert_tensor.shape)
            num_experts = int(moe_expected[target_path])
            global_shape = (num_experts, in_features, out_features)
            chunks = _chunk_shape_for(key_tuple, global_shape, model.param_dtype, moe=True)

            ts_spec = jax_ser.get_tensorstore_spec(abs_path)
            ts_spec["metadata"] = ts_impl._get_tensorstore_metadata_cached(
                global_shape,
                model.param_dtype,
                tuple(chunks),
                driver="zarr",
            )
            ts_spec["dtype"] = jnp.dtype(model.param_dtype).name
            ts_arr = ts.open(ts.Spec(ts_spec), create=True, open=True).result()

            handle = {
                "key_tuple": key_tuple,
                "abs_path": abs_path,
                "rel_path": rel_path,
                "num_experts": num_experts,
                "in_features": in_features,
                "out_features": out_features,
                "pos_by_expert": {ei: pos for pos, ei in enumerate(sorted(moe_groups[target_path].keys()))},
                "ts": ts_arr,
            }
            moe_handles[target_path] = handle
            array_index.append(
                {
                    "path": rel_path,
                    "shape": [num_experts, in_features, out_features],
                    "dtype": str(jnp.dtype(model.param_dtype)),
                }
            )
            return handle

        def _finalize_moe_group(target_path: str) -> None:
            if target_path in moe_handles:
                moe_handles.pop(target_path, None)

        def _process_and_write(key: str, tensor) -> None:
            with convert_ctx:
                results = StateDictConverter.process_tensor(key, tensor, converter_config)
            if results is None:
                return
            for key_tuple, jax_array in results:
                _write_tensor(key_tuple, jax_array)

        import torch
        from safetensors.torch import safe_open  # type:ignore

        if verbose:
            logger.info(f"Sequential conversion started: {pretrained_model_name_or_path} -> {save_root}")

        for fname in sorted(file_to_keys.keys()):
            keys = sorted(file_to_keys[fname])
            if not keys:
                continue

            if verbose:
                logger.info(f"Converting shard: {fname} ({len(keys)} tensors)")

            with _with_resolved_shard(fname) as resolved_path:
                if ckpt_weight_format == "safetensors":
                    with safe_open(resolved_path, framework="pt", device="cpu") as f:
                        for k in keys:
                            if k in expert_key_to_group:
                                target_path, expert_idx = expert_key_to_group[k]
                                expert_tensor = f.get_tensor(k)
                                handle = _ensure_moe_group_ready(target_path, sample_expert_tensor=expert_tensor)
                                if handle is not None:
                                    pos = handle["pos_by_expert"][expert_idx]
                                    with convert_ctx:
                                        expert_slice = TensorConverter.convert_pytorch_to_jnp(
                                            expert_tensor.permute(1, 0),
                                            model.param_dtype,
                                        )
                                    handle["ts"][pos].write(expert_slice).result()
                                moe_remaining[target_path] -= 1
                                if moe_remaining[target_path] <= 0:
                                    _finalize_moe_group(target_path)
                                del expert_tensor
                                continue

                            _process_and_write(k, f.get_tensor(k))
                else:
                    shard = None
                    try:
                        shard = torch.load(resolved_path, map_location="cpu", weights_only=True)
                    except TypeError:
                        shard = torch.load(resolved_path, map_location="cpu")

                    for k in keys:
                        if k in expert_key_to_group:
                            target_path, expert_idx = expert_key_to_group[k]
                            expert_tensor = shard[k]
                            handle = _ensure_moe_group_ready(target_path, sample_expert_tensor=expert_tensor)
                            if handle is not None:
                                pos = handle["pos_by_expert"][expert_idx]
                                with convert_ctx:
                                    expert_slice = TensorConverter.convert_pytorch_to_jnp(
                                        expert_tensor.permute(1, 0),
                                        model.param_dtype,
                                    )
                                handle["ts"][pos].write(expert_slice).result()
                            moe_remaining[target_path] -= 1
                            if moe_remaining[target_path] <= 0:
                                _finalize_moe_group(target_path)
                            del expert_tensor
                            continue
                        _process_and_write(k, shard[k])
                    del shard

            gc.collect()

        incomplete = {k: v for k, v in moe_remaining.items() if v > 0}
        if incomplete:
            raise RuntimeError(
                "Some MoE expert groups were incomplete after processing all shards: "
                + ", ".join([f"{k} missing={v}" for k, v in sorted(incomplete.items())][:10])
            )

        _write_checkpoint_index()
        try:
            attn_mechanism = getattr(ed_config, "attn_mechanism", "auto")
            if hasattr(attn_mechanism, "value"):
                attn_mechanism = attn_mechanism.value

            model_display_name = getattr(hf_config, "_name_or_path", None) or pretrained_model_name_or_path
            readme_repo_id = output_repo_id or str(save_root)
            model_info = ModelInfo(
                name=str(model_display_name),
                type=module.__name__,
                repo_id=readme_repo_id,
                description=f"EasyDeL checkpoint converted from {pretrained_model_name_or_path}.",
                model_type=model_type,
                model_task=cls._model_task or "CausalLM",
                attn_mechanism=str(attn_mechanism),
            )
            ReadmeGenerator().generate_readme(model_info, output_path=str(save_root / "README.md"))
        except Exception:
            logger.exception("Failed to generate README for converted checkpoint at %s", save_root)
        if verbose:
            logger.info(f"Sequential conversion finished. Wrote {len(array_index)} tensors to {save_root}")

        return str(save_root)

    @classmethod
    def get_torch_loader(cls):
        """Gets the appropriate HuggingFace AutoModel class for loading PyTorch weights.

        This method returns the correct transformers AutoModel class based on the model's
        task type (e.g., CausalLM, Seq2Seq, AudioClassification). If the class has a
        custom `hf_torch_auto_loader` attribute set, that will be used instead.

        Returns:
            type: The HuggingFace AutoModel class appropriate for this model's task type.

        Raises:
            ValueError: If no matching AutoModel class can be found for the model's task type.
                In this case, set `hf_torch_auto_loader` on your class manually.
        """
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
        elif cls._model_task == TaskType.ANY_TO_ANY:
            from transformers import AutoModelForTextToWaveform as module
        else:
            raise ValueError("couldn't find requested hf autoloader, you can set `hf_torch_auto_loader` to your class")
        return module
