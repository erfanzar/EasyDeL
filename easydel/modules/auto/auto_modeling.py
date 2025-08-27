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
import contextlib
import os
import typing as tp

import flax
import flax.nnx
import jax
from eformer.escale import PartitionAxis
from eformer.paths import ePath
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_config import EasyDeLBaseConfig, EasyDeLBaseConfigDict
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms, EasyDeLQuantizationMethods
from easydel.infra.factory import TaskType, registry

SAFETENSOR_INDEX_NAME = "tensorstore_index.json"


class BaseAutoEasyModel:
    """
    Base class for all Auto EasyDeL model classes. Provides common class methods
    for loading models from configurations or pretrained checkpoints.

    Attributes:
            model_task (TaskType): The specific task the model class is designed for (e.g., CAUSAL_LM).
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
        rngs: flax.nnx.Rngs | None = None,
    ) -> EasyDeLBaseModule:
        """Instantiates a model module directly from a configuration object.

        Args:
                config (EasyDeLBaseConfig): The configuration object for the model.
                dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
                param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
                precision (Optional[jax.lax.Precision]): JAX precision level. Defaults to None.
                rngs (Optional[flax.nnx.Rngs]): Random number generators. Defaults to Rngs(42).

        Returns:
                EasyDeLBaseModule: An instance of the specific EasyDeL model module.
        """
        registration = registry.get_module_registration(cls.model_task, config.model_type)
        if rngs is None:
            rngs = flax.nnx.Rngs(42)
        return registration.module(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: jax.Device | None = None,  # type: ignore
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
        from_torch: bool | None = None,
        **kwargs,
    ) -> EasyDeLBaseModule:
        """
        Loads and shards a pretrained model from the Hugging Face Hub and converts it into an EasyDeL compatible model.

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model in the Hugging Face Hub.
            device (jax.Device, optional): Device to load the model on. Defaults to the first CPU.
            dtype (jnp.dtype, optional): Data type of the model. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type of the model parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision, optional): Precision for computations.
                Defaults to jax.lax.Precision("fastest").
            sharding_axis_dims (tp.Sequence[int], optional): Dimensions of each sharding axis.
                Defaults to (1, -1, 1, 1, 1).
            sharding_axis_names (tp.Sequence[str], optional): Names of the sharding axes.
                Defaults to ("dp", "fsdp",  "ep", "tp", "sp").
            partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
            shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
            shard_fns (tp.Optional[tp.Mapping[tuple, tp.Callable] | dict], optional): Sharding functions to use for the
                model. If None, auto-sharding is used if auto_shard_model is True. Defaults to None.
            platform (tp.Optional[EasyDeLPlatforms], optional): platform to use for the model. Defaults to None.
                        backend (tp.Optional[EasyDeLBackends], optional): backend to use for the model. Defaults to None.
            config_kwargs (tp.Optional[tp.Mapping[str, tp.Any] | EasyDeLBaseConfigDict], optional): Configuration
                keyword arguments to pass to the model config. Defaults to None.
            auto_shard_model (bool, optional): Whether to automatically shard the model parameters. Defaults to False.
            partition_rules (tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec]]], optional): Custom partition rules
                for parameter sharding. If not None, shard_fns should also be provided. Defaults to None.
            quantization_method (EasyDeLQuantizationMethods, optional): quantization_method to be used to
                quantize model weights. Defaults to None.
            quantization_block_size (int): block size to be used for quantizing arrays (only for NF4).
            bit_targeted_params (tp.Optional[tp.List[str]], optional): tp.List of parameter names to convert
                to 8-bit precision. If  None and 8bit is True, all kernels and embeddings are converted to 8-bit.
                Defaults to None.
            from_torch (bool): whenever to load the model from transformers-pytorch.
            **kwargs: Additional keyword arguments to pass to the model and config classes.

        Returns:
            tp.Tuple[EasyDeLBaseModule, dict]: A tuple containing the EasyDeL model and the loaded and sharded
                model parameters.
        """
        if precision is None:
            precision = jax.lax.Precision("fastest")
        if partition_axis is None:
            partition_axis = PartitionAxis()
        if from_torch is None:
            try:
                from_torch = not cls._is_easydel(pretrained_model_name_or_path)
            except OSError as e:
                from_torch = False
                if "Error no file named easydel-model.parameters" in str(e):
                    from_torch = True
        if from_torch:
            return cls._from_torch_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                device=device,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                sharding_axis_dims=sharding_axis_dims,
                sharding_dcn_axis_dims=sharding_dcn_axis_dims,
                sharding_axis_names=sharding_axis_names,
                partition_axis=partition_axis,
                shard_attention_computation=shard_attention_computation,
                shard_fns=shard_fns,
                backend=backend,
                platform=platform,
                config_kwargs=config_kwargs,
                auto_shard_model=auto_shard_model,
                partition_rules=partition_rules,
                quantization_platform=quantization_platform,
                quantization_method=quantization_method,
                quantization_block_size=quantization_block_size,
                quantization_pattern=quantization_pattern,
                quantize_tensors=quantize_tensors,
                verbose=verbose,
                **kwargs,
            )
        cmg = jax.default_device(device) if device is not None else contextlib.nullcontext()
        with cmg:
            return cls._from_easydel_params(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                device=device,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                sharding_axis_dims=sharding_axis_dims,
                sharding_dcn_axis_dims=sharding_dcn_axis_dims,
                sharding_axis_names=sharding_axis_names,
                partition_axis=partition_axis,
                shard_attention_computation=shard_attention_computation,
                shard_fns=shard_fns,
                backend=backend,
                platform=platform,
                config_kwargs=config_kwargs,
                auto_shard_model=auto_shard_model,
                partition_rules=partition_rules,
                quantization_platform=quantization_platform,
                quantization_method=quantization_method,
                quantization_block_size=quantization_block_size,
                quantization_pattern=quantization_pattern,
                quantize_tensors=quantize_tensors,
                verbose=verbose,
                **kwargs,
            )

    @classmethod
    def _from_easydel_params(
        cls,
        pretrained_model_name_or_path: str,
        device: jax.Device | None = None,  # type: ignore
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
        """Loads a model from EasyDeL saved parameters.

        This is a helper method called by `from_pretrained` when the source
        is identified as an EasyDeL checkpoint.

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model.
            device (jax.Device, optional): Device to load the model on. Defaults to None.
            dtype (jnp.dtype, optional): Data type of the model. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type of the model parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision, optional): Precision for computations. Defaults to None.
            sharding_axis_dims (tp.Sequence[int], optional): Dimensions of each sharding axis.
                Defaults to (1, -1, 1, 1, 1).
            sharding_dcn_axis_dims (tp.Optional[tp.Sequence[int]], optional): Dimensions for DCN sharding.
                Defaults to None.
            sharding_axis_names (tp.Sequence[str], optional): Names of the sharding axes.
                Defaults to ("dp", "fsdp",  "ep", "tp", "sp").
            partition_axis (PartitionAxis, optional): Partitioning configuration. Defaults to None.
            shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
            shard_fns (tp.Optional[tp.Mapping[tuple, tp.Callable] | dict], optional): Custom sharding functions.
                Defaults to None.
            backend (tp.Optional[EasyDeLBackends], optional): Backend to use. Defaults to None.
            platform (tp.Optional[EasyDeLPlatforms], optional): Platform to use. Defaults to None.
            config_kwargs (tp.Optional[EasyDeLBaseConfigDict], optional): Configuration overrides. Defaults to None.
            auto_shard_model (bool, optional): Whether to automatically shard. Defaults to False.
            partition_rules (tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec], ...]], optional): Custom partition rules.
                Defaults to None.
            quantization_platform (tp.Optional[EasyDeLPlatforms], optional): Platform for quantization. Defaults to None.
            quantization_method (tp.Optional[EasyDeLQuantizationMethods], optional): Quantization method.
                Defaults to None.
            quantization_block_size (int): Block size for quantization. Defaults to 128.
            quantization_pattern (tp.Optional[str]): Pattern for quantization target modules. Defaults to None.
            quantize_tensors (bool): Whether to quantize tensors. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
            **kwargs: Additional keyword arguments passed to the underlying `EasyDeLBaseModule.from_pretrained`.

        Returns:
            EasyDeLBaseModule: The loaded and potentially sharded EasyDeL model module.
        """

        class Base(EasyDeLBaseModule):
            _model_task = cls.model_task

        return Base.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            shard_attention_computation=shard_attention_computation,
            shard_fns=shard_fns,
            backend=backend,
            platform=platform,
            config_kwargs=config_kwargs,
            auto_shard_model=auto_shard_model,
            partition_rules=partition_rules,
            quantization_platform=quantization_platform,
            quantization_method=quantization_method,
            quantization_block_size=quantization_block_size,
            quantization_pattern=quantization_pattern,
            quantize_tensors=quantize_tensors,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def _from_torch_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: jax.Device | None = None,  # type: ignore
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
        """Loads a model from PyTorch pretrained weights.

        This is a helper method called by `from_pretrained` when the source
        is identified as a PyTorch checkpoint (or requires conversion).

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model.
            device (jax.Device, optional): Device to load the model on. Defaults to None.
            dtype (jnp.dtype, optional): Data type of the model. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type of the model parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision, optional): Precision for computations. Defaults to None.
            sharding_axis_dims (tp.Sequence[int], optional): Dimensions of each sharding axis.
                Defaults to (1, -1, 1, 1, 1).
            sharding_dcn_axis_dims (tp.Optional[tp.Sequence[int]], optional): Dimensions for DCN sharding.
                Defaults to None.
            sharding_axis_names (tp.Sequence[str], optional): Names of the sharding axes.
                Defaults to ("dp", "fsdp",  "ep", "tp", "sp").
            partition_axis (PartitionAxis, optional): Partitioning configuration. Defaults to None.
            shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
            shard_fns (tp.Optional[tp.Mapping[tuple, tp.Callable] | dict], optional): Custom sharding functions.
                Defaults to None.
            backend (tp.Optional[EasyDeLBackends], optional): Backend to use. Defaults to None.
            platform (tp.Optional[EasyDeLPlatforms], optional): Platform to use. Defaults to None.
            config_kwargs (tp.Optional[EasyDeLBaseConfigDict], optional): Configuration overrides. Defaults to None.
            auto_shard_model (bool, optional): Whether to automatically shard. Defaults to False.
            partition_rules (tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec], ...]], optional): Custom partition rules.
                Defaults to None.
            quantization_platform (tp.Optional[EasyDeLPlatforms], optional): Platform for quantization. Defaults to None.
            quantization_method (tp.Optional[EasyDeLQuantizationMethods], optional): Quantization method.
                Defaults to None.
            quantization_block_size (int): Block size for quantization. Defaults to 128.
            quantization_pattern (tp.Optional[str]): Pattern for quantization target modules. Defaults to None.
            quantize_tensors (bool): Whether to quantize tensors. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
            **kwargs: Additional keyword arguments passed to the underlying `EasyDeLBaseModule._from_torch_pretrained`.

        Returns:
            EasyDeLBaseModule: The loaded, converted, and potentially sharded EasyDeL model module.
        """

        class Base(EasyDeLBaseModule):
            _model_task = cls.model_task

        return Base._from_torch_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            shard_attention_computation=shard_attention_computation,
            shard_fns=shard_fns,
            backend=backend,
            platform=platform,
            config_kwargs=config_kwargs,
            auto_shard_model=auto_shard_model,
            partition_rules=partition_rules,
            quantization_platform=quantization_platform,
            quantization_method=quantization_method,
            quantization_block_size=quantization_block_size,
            quantization_pattern=quantization_pattern,
            quantize_tensors=quantize_tensors,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def _is_easydel(
        cls,
        pretrained_model_name_or_path,
        FLAX_WEIGHTS_NAME="easydel-model.parameters",
        MULTI_PART_NAME="easydel-model.parameters.safetensors.index.json",
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
    ):
        """Checks if the given path or identifier points to an EasyDeL model checkpoint.

        Args:
            pretrained_model_name_or_path: Identifier or path to check.
            FLAX_WEIGHTS_NAME (str): The standard filename for EasyDeL weights.
            cache_dir (Optional[Union[str, os.PathLike]]): Cache directory.
            force_download (bool): Force download even if cached.
            local_files_only (bool): Only check local files.
            token (Optional[Union[str, bool]]): Hugging Face Hub token.
            revision (str): Git revision identifier.

        Returns:
            bool: True if it's an EasyDeL checkpoint, False otherwise.
        """
        from transformers.utils import cached_file as _cached_file
        from transformers.utils import is_remote_url as _is_remote_url

        proxies = None
        subfolder = ""
        commit_hash = None
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        epath = ePath(pretrained_model_name_or_path)

        if epath.is_dir():
            if (epath / subfolder / SAFETENSOR_INDEX_NAME).exists():
                return True
            if (epath / subfolder / MULTI_PART_NAME).exists():
                return True
            if not (epath / subfolder / FLAX_WEIGHTS_NAME).is_file():
                raise OSError(
                    f"Error no file named {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}"
                )
        elif (ePath(subfolder) / epath).is_file():
            ...
        elif _is_remote_url(pretrained_model_name_or_path):
            ...
        else:
            filename = FLAX_WEIGHTS_NAME
            try:
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "local_files_only": local_files_only,
                    "token": token,
                    "user_agent": {
                        "file_type": "model",
                        "framework": "flax",
                        "from_auto_class": False,
                    },
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }
                resolved_archive_file = _cached_file(
                    pretrained_model_name_or_path,
                    filename,
                    **cached_file_kwargs,
                )

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
    """
    Base class for Auto EasyDeL state classes. Provides common class methods
    for creating model states from configurations or pretrained checkpoints.

    Attributes:
            _base (BaseAutoEasyModel): The corresponding Auto EasyDeL model class.
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
        rngs: flax.nnx.Rngs | None = None,
    ) -> EasyDeLState:
        """Creates an EasyDeLState directly from a configuration object.

        Args:
                config (EasyDeLBaseConfig): The configuration object for the model.
                dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
                param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
                precision (Optional[jax.lax.Precision]): JAX precision level. Defaults to None.
                rngs (Optional[flax.nnx.Rngs]): Random number generators. Defaults to Rngs(42).

        Returns:
                EasyDeLState: An initialized EasyDeLState for the model.
        """
        return cls._base.from_config(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        ).to_state()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: jax.Device | None = None,  # type: ignore
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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
        quantization_method: EasyDeLQuantizationMethods | None = None,
        quantization_block_size: int = 128,
        from_torch: bool | None = None,
        **kwargs,
    ) -> EasyDeLState:
        """
        Loads and shards a pretrained model from the Hugging Face Hub and converts it into an EasyDeL compatible state.

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model in the Hugging Face Hub.
            device (jax.Device, optional): Device to load the model on. Defaults to the first CPU.
            dtype (jnp.dtype, optional): Data type of the model. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type of the model parameters. Defaults to jnp.float32.
            precision (jax.lax.Precision, optional): Precision for computations.
                Defaults to jax.lax.Precision("fastest").
            sharding_axis_dims (tp.Sequence[int], optional): Dimensions of each sharding axis.
                Defaults to (1, -1, 1, 1, 1).
            sharding_axis_names (tp.Sequence[str], optional): Names of the sharding axes.
                Defaults to ("dp", "fsdp",  "ep", "tp", "sp").
            partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
            shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
            shard_fns (tp.Optional[tp.Mapping[tuple, tp.Callable] | dict], optional): Sharding functions to use for
                the model. If None, auto-sharding is used if auto_shard_model is True. Defaults to None.
            backend (tp.Optional[str], optional): Backend to use for the model. Defaults to None.
            config_kwargs (tp.Optional[tp.Mapping[str, tp.Any]], optional): Configuration keyword
                arguments to pass to the model config. Defaults to None.
            auto_shard_model (bool, optional): Whether to automatically shard the model parameters. Defaults to False.
            partition_rules (tp.Optional[tp.Tuple[tp.Tuple[str, PartitionSpec]]], optional): Custom partition
                rules for parameter sharding. If not None, shard_fns should also be provided. Defaults to None.
            quantization_method (EasyDeLQuantizationMethods, optional): quantization_method to be used
                to quantize model weights. Defaults to None.
            bit_targeted_params (tp.Optional[tp.List[str]], optional): tp.List of parameter names to
                convert to 8-bit precision. If  None and 8bit is True, all kernels and embeddings are
                converted to 8-bit. Defaults to None.
            verbose_params (bool): whenever to log number of parameters in converting state.
            safe (bool): whenever to use safetensors to load engine or parameters (requires engine or parameters
                to be saved with safe=True while saving them)
            from_torch (bool): whenever to load the model from transformers-pytorch.
            **kwargs: Additional keyword arguments to pass to the model and config classes.

        Returns:
            EasyDeLState: containing the EasyDeL state and the loaded and sharded model parameters.
        """
        model = cls._base.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            param_dtype=param_dtype,
            dtype=dtype,
            shard_fns=shard_fns,
            auto_shard_model=auto_shard_model,
            precision=precision,
            backend=backend,
            platform=platform,
            partition_axis=partition_axis,
            quantization_method=quantization_method,
            quantization_block_size=quantization_block_size,
            partition_rules=partition_rules,
            sharding_axis_names=sharding_axis_names,
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            config_kwargs=config_kwargs,
            device=device,
            shard_attention_computation=shard_attention_computation,
            from_torch=from_torch,
            **kwargs,
        )
        return EasyDeLState.create(
            model=model,
            tx=None,
            init_opt_state=False,
            step=0,
        )


class AutoEasyDeLModelForCausalLM(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.

    Attributes:
        None

    Examples:

        >>> import jax
        >>> from easydel import AutoEasyDeLModelForCausalLM

        >>> # Load a GPT-2 model on a single CPU
        >>> model = AutoEasyDeLModelForCausalLM.from_pretrained(
        >>>   "gpt2", device=jax.devices("cpu")[0]
        >>> )

        >>> # Load a GPT-2 model sharded across 8 GPUs with data parallelism (DP) and
        >>> # fully sharded data parallelism (FSDP)
        >>> model = AutoEasyDeLModelForCausalLM.from_pretrained(
        ...  "gpt2",
        ...  sharding_axis_dims=(1, 8, 1, 1, 1),
        ...  sharding_axis_names=("dp", "fsdp",  "ep", "tp", "sp"),
        ...  device=jax.devices("cpu")[0],  # offload to CPU [OPTIONAL]
        ...  from_torch=True,
        >>> )
        ```
    """

    model_task: TaskType = TaskType.CAUSAL_LM  # Static


class AutoStateForCausalLM(BaseAutoEasyState):
    _base = AutoEasyDeLModelForCausalLM


class AutoEasyDeLModelForDiffusionLM(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.

    """

    model_task: TaskType = TaskType.DIFFUSION_LM  # Static


class AutoStateForDiffusionLM(BaseAutoEasyState):
    _base = AutoEasyDeLModelForDiffusionLM


class AutoEasyDeLModelForZeroShotImageClassification(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.

    """

    model_task: TaskType = TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION  # Static


class AutoStateForZeroShotImageClassification(BaseAutoEasyState):
    _base = AutoEasyDeLModelForZeroShotImageClassification


class AutoEasyDeLModelForSpeechSeq2Seq(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.

    Attributes:
        None

    Examples:

        >>> import jax
        >>> from easydel import AutoEasyDeLModelForSpeechSeq2Seq

        >>> # Load a openai/whisper-large-v3-turbo sharded
        >>> model = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
        ...  "openai/whisper-large-v3-turbo",
        ...  auto_shard_model=True,
        >>> )

        >>> # Load a openai/whisper-large-v3-turbo model sharded across 8 GPUs with data parallelism (DP) and
        >>> # fully sharded data parallelism (FSDP)
        >>> model = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
        ...  "openai/whisper-large-v3-turbo",
        ...  sharding_axis_dims=(1, 8, 1, 1, 1),
        ...  sharding_axis_names=("dp", "fsdp",  "ep", "tp", "sp"),
        ...  device=jax.devices("cpu")[0],  # offload to CPU [OPTIONAL]
        ...  from_torch=True,
        >>> )
        ```
    """

    model_task: TaskType = TaskType.SPEECH_SEQUENCE_TO_SEQUENCE  # Static


class AutoStateForSpeechSeq2Seq(BaseAutoEasyState):
    _base = AutoEasyDeLModelForSpeechSeq2Seq


class AutoEasyDeLModelForSeq2SeqLM(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.
    """

    model_task: TaskType = TaskType.SEQUENCE_TO_SEQUENCE  # Static


class AutoStateForSeq2SeqLM(BaseAutoEasyState):
    _base = AutoEasyDeLModelForSeq2SeqLM


class AutoEasyDeLModelForImageTextToText(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.
    """

    model_task: TaskType = TaskType.IMAGE_TEXT_TO_TEXT  # Static


class AutoStateForImageTextToText(BaseAutoEasyState):
    _base = AutoEasyDeLModelForImageTextToText


class AutoEasyDeLModelForSequenceClassification(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.
    """

    model_task: TaskType = TaskType.SEQUENCE_CLASSIFICATION  # Static


class AutoStateForImageSequenceClassification(BaseAutoEasyState):
    _base = AutoEasyDeLModelForSequenceClassification


class AutoEasyDeLModel(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.
    """

    model_task: TaskType = TaskType.BASE_MODULE  # Static


class AutoState(BaseAutoEasyState):
    _base = AutoEasyDeLModel


class AutoEasyDeLVisionModel(BaseAutoEasyModel):
    """
    This class provides a convenient way to load and shard pretrained  models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed
    training and inference with JAX.

    This class inherits from the `EasyDeLBaseModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.
    """

    model_task: TaskType = TaskType.BASE_VISION  # Static


class AutoStateVisionModel(BaseAutoEasyState):
    _base = AutoEasyDeLVisionModel
