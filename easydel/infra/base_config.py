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

"""Base configuration classes for EasyDeL models.

This module provides the foundational configuration system for all EasyDeL models,
extending HuggingFace's PretrainedConfig with EasyDeL-specific features like
attention mechanisms, quantization, gradient checkpointing, and hardware optimization.

Classes:
    EasyDeLBaseConfig: Main configuration class with all EasyDeL features
    EasyDeLBaseConfigDict: Simplified dictionary-based configuration

Key Features:
    - Multiple attention mechanism support (flash, ring, etc.)
    - Quantization configuration
    - Gradient checkpointing policies
    - Hardware abstraction and optimization
    - RoPE (Rotary Position Embedding) configuration
    - Custom kernel support

Example:
    >>> from easydel.infra import EasyDeLBaseConfig
    >>> config = EasyDeLBaseConfig(
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     attention_mechanism="flash",
    ...     gradient_checkpointing_policy="",
    ...     use_hardware_abstraction=True
    ... )
"""

from __future__ import annotations

import json
import os
import re
import typing as tp
import warnings

import chex
import jax
import jax.extend
import jax.tree_util
from eformer import common_types
from eformer.common_types import NOT_GIVEN
from eformer.escale import PartitionAxis, PartitionManager
from eformer.paths import ePath, ePathLike
from eformer.pytree import auto_pytree
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.utils import CONFIG_NAME, cached_file, download_url, is_remote_url
from transformers.utils.generic import is_timm_config_dict

from easydel.utils.compiling_utils import hash_fn
from easydel.utils.helpers import check_bool_flag, get_logger

from .etils import (
    AVAILABLE_ATTENTION_MECHANISMS,
    DEFAULT_ATTENTION_MECHANISM,
    EasyDeLBackends,
    EasyDeLGradientCheckPointers,
    EasyDeLPlatforms,
    EasyDeLQuantizationMethods,
)

if tp.TYPE_CHECKING:
    from easydel.layers.rotary_embedding import RopeConfig

    from .utils import AttnMaskDetail, ModuleCaches
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
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
PROCESSOR_NAME = "processor_config.json"
CHAT_TEMPLATE_NAME = "chat_template.json"
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

DEFAULT_PALLAS_M_BLOCK_SIZE = 128
DEFAULT_PALLAS_K_BLOCK_SIZE = 128
DEFAULT_PALLAS_N_BLOCK_SIZE = 128
DEFAULT_HARDWARE_ABSTRACTION = False

ED_DEFAULT_HARDWARE_ABSTRACTION = check_bool_flag("ED_DEFAULT_HARDWARE_ABSTRACTION", default=False)
EKERNEL_OPS = check_bool_flag("EKERNEL_OPS", default=False)

if ED_DEFAULT_HARDWARE_ABSTRACTION:
    DEFAULT_HARDWARE_ABSTRACTION = True


if DEFAULT_HARDWARE_ABSTRACTION:
    logger.info("HARDWARE_ABSTRACTION is ON by default")

if EKERNEL_OPS:
    logger.info("`EKERNEL_OPS` is ON and some operations will automatically be replaced by EasyDeL.")
    from easydel.kernels.matmul import replace_dot_general_with_matmul

    replace_dot_general_with_matmul()


def extract_commit_hash(resolved_file: str | None, commit_hash: str | None) -> str | None:
    """Extract the commit hash from a resolved cache filename.

    Parses the resolved file path to extract the git commit hash
    if not already provided.

    Args:
        resolved_file: Path to the resolved cache file.
        commit_hash: Existing commit hash if available.

    Returns:
        The commit hash string or None if not found.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(ePath(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def set_attrs_smartly(self, attr_name: str, default: tp.Any, new_attr: tp.Any):
    """Set attributes intelligently with default values.

    Sets an attribute if it doesn't exist, and updates it if a new
    non-NOT_GIVEN value is provided.

    Args:
        attr_name: Name of the attribute to set.
        default: Default value if attribute doesn't exist.
        new_attr: New value to set (if not NOT_GIVEN).
    """
    if not hasattr(self, attr_name):
        setattr(self, attr_name, default)
    if new_attr is not NOT_GIVEN:
        setattr(self, attr_name, new_attr)


@auto_pytree
class EasyMethod:
    """Constants for EasyDeL operation modes.

    Defines the different modes in which EasyDeL models can operate.

    Attributes:
        TRAIN: Training mode for model optimization.
        SERVE: Serving mode for inference.
        EVAL: Evaluation mode (alias for serve).
        CONVERT: Conversion mode for model format changes.
    """

    TRAIN: str = "train"
    SERVE: str = "serve"
    EVAL: str = "serve"
    CONVERT: str = "convert"


warnings.filterwarnings(
    "ignore",
    message="Passing `gradient_checkpointing` to a config initialization is deprecated",  # EasyDeL will handle this
)


warnings.filterwarnings("ignore", message="You are using a model of type")
warnings.filterwarnings("ignore", message="Some donated buffers were not usable:")


class EasyDeLBaseConfigDict(tp.TypedDict, total=False):
    """TypedDict for EasyDeL configuration parameters.

    Provides type hints for all configuration options that can be
    passed to EasyDeLBaseConfig. All fields are optional (total=False).

    This is useful for type checking when creating configurations
    from dictionaries or JSON.
    """

    sharding_axis_dims: tp.Sequence[int]
    sharding_dcn_axis_dims: tp.Sequence[int] | None
    sharding_axis_names: tp.Sequence[str]
    attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS  # type:ignore
    decode_attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS  # type:ignore
    blocksize_k: int
    blocksize_q: int
    blocksize_b: int
    moe_tiling_size_batch: int
    moe_tiling_size_seqlen: int
    moe_tiling_size_dim: int
    partition_axis: PartitionAxis
    shard_attention_computation: bool
    use_sharded_kv_caching: bool
    use_sharding_constraint: bool
    use_pallas_group_matmul: bool
    backend: EasyDeLBackends | None
    platform: EasyDeLPlatforms | None
    easy_method: tp.Literal["train", "serve", "convert"]
    bits: int | None
    scan_ring_attention: bool
    scan_attention_layers: bool
    use_scan_mlp: bool
    scan_mlp_chunk_size: int
    sequence_axis_name: str
    gradient_checkpointing: EasyDeLGradientCheckPointers
    kv_cache_quantization_method: EasyDeLQuantizationMethods
    kv_cache_quantization_blocksize: int
    kv_cache_sharding_sequence_axis_name: str | tuple[str, ...]
    flash_attention_backward_pass_impl: tp.Literal["triton", "xla"]
    attn_dtype: jnp.dtype
    kvdtype: jnp.dtype
    attn_softmax_dtype: jnp.dtype
    fcm_max_ratio: float
    fcm_min_ratio: float
    hardware_abstraction: bool
    pallas_m_block_size: int
    pallas_k_block_size: int
    pallas_n_block_size: int
    mask_max_position_embeddings: int
    freq_max_position_embeddings: int
    precompute_masks: bool


class EasyDeLBaseConfig(PretrainedConfig):
    """Base configuration class for all EasyDeL models.

    Extends HuggingFace's PretrainedConfig with EasyDeL-specific features
    for distributed training, custom attention mechanisms, quantization,
    and hardware optimization.
    Args:
        sharding_axis_dims (tp.Sequence[int]): Dimensions of the axes. Default is (1, -1, 1, 1, 1).
        sharding_axis_names (tp.Sequence[str]): Names of the axes. Default is ("dp", "fsdp",  "ep", "tp", "sp").
        attn_mechanism (AVAILABLE_ATTENTION_MECHANISMS): Attention mechanism to use.
            Default is DEFAULT_ATTENTION_MECHANISM.
        decode_attn_mechanism (AVAILABLE_ATTENTION_MECHANISMS): Attention mechanism to use for decode phase.
            Default is None.
        blocksize_k (int): Block size for key. Default is 128.
        blocksize_q (int): Block size for query. Default is 128.
        blocksize_b (int): Block size for batch. Default is 1.
        moe_tiling_size_batch (int): Block Size for batch in MoE tiling. Default is 4.
        moe_tiling_size_seqlen (int): Block Size for sequence length in MoE tiling. Default is 128.
        moe_tiling_size_dim (int): Block Size for dimension in MoE tiling. Default is 128.
        partition_axis (PartitionAxis): Partition axis configuration. Default is PartitionAxis().
        shard_attention_computation (bool): Whether to shard attention computation. Default is True.
        use_sharded_kv_caching (bool): Whether to use sharded key-value caching. Default is False.
        use_sharding_constraint (bool): Whether to use sharding constraint. Default is False.
        use_pallas_group_matmul (bool): Whether to use pallas group matmul. Default is True.
        backend (tp.Optional[EasyDeLBackends]): Backend to use. Default is None.
        platform (tp.Optional[EasyDeLPlatforms]): Platform to use. Default is None.
        easy_method (tp.Literal["train", "serve", "convert"]): Method to use. Default is EasyMethod.TRAIN.
        bits (tp.Optional[int]): Number of bits for quantization. Default is None.
        scan_ring_attention (bool): Whether to scan ring attention. Default is True.
        scan_attention_layers (bool): Whether to scan attention layers. Default is False.
        use_scan_mlp (bool): Whether to use scan MLP. Default is False.
        scan_mlp_chunk_size (int): Chunk size for scan MLP. Default is 1024.
        sequence_axis_name (str): Name of the attention axis. Default is "sp".
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing method.
            Default is EasyDeLGradientCheckPointers.NONE.
        kv_cache_quantization_method (EasyDeLQuantizationMethods): Key-value cache quantization method.
            Default is EasyDeLQuantizationMethods.NONE.
        kv_cache_quantization_blocksize (int): Block size for key-value cache quantization. Default is 64.
        quantization_method (EasyDeLQuantizationMethods): Quantization method.
            Default is EasyDeLQuantizationMethods.NONE.
        quantization_pattern (str): Pattern for quantization. Default is ".*".
        quantization_blocksize (int): Block size for quantization. Default is 64.
        kv_cache_sharding_sequence_axis_name (tp.Union[str, tp.Tuple[str, ...]]): Name of the key-value cache
            sharding sequence axis. Default is "sp".
        flash_attention_backward_pass_impl (tp.Literal["triton", "xla"]): Implementation for flash attention
            backward pass. Default is "triton".
        attn_dtype (jnp.dtype): Data type for attention. Default is float32.
        kvdtype (jnp.dtype): Data type for attention kv cache. Default is bfloat16.
        attn_softmax_dtype (jnp.dtype): Data type for softmax ops in attention. Default is jnp.float32.
        fcm_max_ratio (float): Maximum ratio for FCM. Default is 0.0.
        fcm_min_ratio (float): Minimum ratio for FCM. Default is 0.0.
        hardware_abstraction (bool): Whether to use hardware abstraction. Default is DEFAULT_HARDWARE_ABSTRACTION.
        pallas_m_block_size (int): Block size for Pallas M. Default is DEFAULT_PALLAS_M_BLOCK_SIZE.
        pallas_k_block_size (int): Block size for Pallas K. Default is DEFAULT_PALLAS_K_BLOCK_SIZE.
        pallas_n_block_size (int): Block size for Pallas N. Default is DEFAULT_PALLAS_N_BLOCK_SIZE.
        **kwargs: Additional keyword arguments.
    Raises:
      Warning: If `kv_cache_quantization_method` is not NONE and `use_sharded_kv_caching` is True.
    """

    _show_private_attrs: bool = False
    _hidden_mesh: common_types.Mesh | None = None

    def __init__(
        self,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = DEFAULT_ATTENTION_MECHANISM,
        decode_attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = None,
        blocksize_k: int = 128,
        blocksize_q: int = 128,
        blocksize_b: int = 1,
        moe_tiling_size_batch: int = 4,
        moe_tiling_size_seqlen: int = 128,
        moe_tiling_size_dim: int = 128,
        partition_axis: PartitionAxis = PartitionAxis(),
        shard_attention_computation: bool = True,
        use_sharded_kv_caching: bool = False,
        use_sharding_constraint: bool = False,
        use_pallas_group_matmul: bool = True,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        easy_method: tp.Literal["train", "serve", "convert"] = EasyMethod.TRAIN,
        bits: int | None = None,
        scan_ring_attention: bool = True,
        scan_attention_layers: bool = False,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        sequence_axis_name: str = "sp",
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        precompute_masks: bool = True,
        kv_cache_quantization_method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.NONE,
        kv_cache_quantization_blocksize: int = 64,
        quantization_method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.NONE,
        quantization_pattern: str = ".*",
        quantization_blocksize: int = 64,
        kv_cache_sharding_sequence_axis_name: str | tuple[str, ...] = "sp",
        flash_attention_backward_pass_impl: tp.Literal["triton", "xla"] = "triton",
        attn_dtype: jnp.dtype = jnp.bfloat16,
        kvdtype: jnp.dtype | None = None,
        attn_softmax_dtype: jnp.dtype = jnp.float32,
        fcm_max_ratio: float = 0.0,
        fcm_min_ratio: float = 0.0,
        hardware_abstraction: bool = DEFAULT_HARDWARE_ABSTRACTION,
        pallas_m_block_size: int = DEFAULT_PALLAS_M_BLOCK_SIZE,
        pallas_k_block_size: int = DEFAULT_PALLAS_K_BLOCK_SIZE,
        pallas_n_block_size: int = DEFAULT_PALLAS_N_BLOCK_SIZE,
        **kwargs,
    ):
        self.sharding_axis_dims = getattr(self, "sharding_axis_dims", sharding_axis_dims)
        self.sharding_dcn_axis_dims = getattr(self, "sharding_dcn_axis_dims", sharding_dcn_axis_dims)
        self.sharding_axis_names = getattr(self, "sharding_axis_names", sharding_axis_names)
        self.backend = getattr(
            self,
            "backend",
            backend if backend is not None else jax.default_backend(),
        )
        self.platform = getattr(
            self,
            "platform",
            platform if platform is not None else ("triton" if jax.default_backend() == "gpu" else "jax"),
        )

        self.easy_method = getattr(self, "easy_method", easy_method)
        self.attn_mechanism = getattr(self, "attn_mechanism", attn_mechanism)
        self.decode_attn_mechanism = getattr(self, "decode_attn_mechanism", decode_attn_mechanism)
        self.blocksize_b = getattr(self, "blocksize_b", blocksize_b)
        self.blocksize_k = getattr(self, "blocksize_k", blocksize_k)
        self.blocksize_q = getattr(self, "blocksize_q", blocksize_q)
        self.moe_tiling_size_batch = getattr(self, "moe_tiling_size_batch", moe_tiling_size_batch)
        self.moe_tiling_size_seqlen = getattr(self, "moe_tiling_size_seqlen", moe_tiling_size_seqlen)
        self.moe_tiling_size_dim = getattr(self, "moe_tiling_size_dim", moe_tiling_size_dim)
        self.partition_axis = getattr(self, "partition_axis", partition_axis)
        self.shard_attention_computation = getattr(self, "shard_attention_computation", shard_attention_computation)
        self.bits = getattr(self, "bits", bits)
        self.scan_attention_layers = getattr(self, "scan_attention_layers", scan_attention_layers)
        self.scan_ring_attention = getattr(self, "scan_ring_attention", scan_ring_attention)
        self.use_sharded_kv_caching = getattr(self, "use_sharded_kv_caching", use_sharded_kv_caching)
        self.use_scan_mlp = getattr(self, "use_scan_mlp", use_scan_mlp)
        self.scan_mlp_chunk_size = getattr(self, "scan_mlp_chunk_size", scan_mlp_chunk_size)
        self.use_sharding_constraint = getattr(self, "use_sharding_constraint", use_sharding_constraint)
        self.use_pallas_group_matmul = getattr(self, "use_pallas_group_matmul", use_pallas_group_matmul)
        self.sequence_axis_name = getattr(self, "sequence_axis_name", sequence_axis_name)
        self.kv_cache_sharding_sequence_axis_name = getattr(
            self, "kv_cache_sharding_sequence_axis_name", kv_cache_sharding_sequence_axis_name
        )
        self.gradient_checkpointing = getattr(self, "gradient_checkpointing", gradient_checkpointing)
        self.precompute_masks = getattr(self, "precompute_masks", precompute_masks)

        self.kv_cache_quantization_method = getattr(self, "kv_cache_quantization_method", kv_cache_quantization_method)
        self.kv_cache_quantization_blocksize = getattr(
            self, "kv_cache_quantization_blocksize", kv_cache_quantization_blocksize
        )
        self.quantization_method = getattr(self, "quantization_method", quantization_method)
        self.quantization_blocksize = getattr(self, "quantization_blocksize", quantization_blocksize)
        self.quantization_pattern = getattr(self, "quantization_pattern", quantization_pattern)
        self.flash_attention_backward_pass_impl = getattr(
            self, "flash_attention_backward_pass_impl", flash_attention_backward_pass_impl
        )
        self.attn_dtype = getattr(self, "attn_dtype", attn_dtype)
        self.kvdtype = getattr(self, "kvdtype", kvdtype if kvdtype is not None else self.attn_dtype)
        self.attn_softmax_dtype = getattr(self, "attn_softmax_dtype", attn_softmax_dtype)
        self.fcm_max_ratio = getattr(self, "fcm_max_ratio", fcm_max_ratio)
        self.fcm_min_ratio = getattr(self, "fcm_min_ratio", fcm_min_ratio)
        self.hardware_abstraction = getattr(self, "hardware_abstraction", hardware_abstraction)
        self.pallas_m_block_size = getattr(self, "pallas_m_block_size", pallas_m_block_size)
        self.pallas_k_block_size = getattr(self, "pallas_k_block_size", pallas_k_block_size)
        self.pallas_n_block_size = getattr(self, "pallas_n_block_size", pallas_n_block_size)

        self.pretraining_tp = 1  # it's for pytorch models.
        if self.kv_cache_quantization_method != EasyDeLQuantizationMethods.NONE and self.use_sharded_kv_caching:
            use_sharded_kv_caching = self.use_sharded_kv_caching
            warnings.warn(
                f"`{self.kv_cache_quantization_method=}` and `{use_sharded_kv_caching=}`"
                " can't be used together at the moment.",
                stacklevel=1,
            )
        super().__init__(**kwargs)

    @staticmethod
    def create_mesh(
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        process_is_granule: bool = False,
        should_sort_granules_by_key: bool = True,
        allow_split_physical_axes: bool = True,
        backend: str | None = None,
    ):
        """
        The create_mesh function creates a mesh object that can be used to shard arrays.

        Returns:
            A mesh object
        """
        from eformer.escale import create_mesh

        if backend == "":
            backend = None

        mesh = create_mesh(
            axis_dims=sharding_axis_dims,
            axis_names=sharding_axis_names,
            dcn_mesh_dims=sharding_dcn_axis_dims,
            should_sort_granules_by_key=should_sort_granules_by_key,
            allow_split_physical_axes=allow_split_physical_axes,
            backend=backend,
        )
        return mesh

    @property
    def mesh(self):
        """
        The mesh property is a helper property that creates a Mesh object from the
        sharding_axis_dims and sharding_axis_names attributes of an object, which are assumed to be lists of integers
        and strings, respectively, The platform attribute is also used if it exists.

        Args:
            self: Refer to the object itself

        Returns:
            A jaxMesh
        """
        if self._hidden_mesh is not None:
            return self._hidden_mesh

        sharding_axis_dims = (
            [v for k, v in self.sharding_axis_dims.items()]
            if isinstance(self.sharding_axis_dims, dict)
            else self.sharding_axis_dims
        )
        sharding_axis_names = (
            [v for k, v in self.sharding_axis_names.items()]
            if isinstance(self.sharding_axis_names, dict)
            else self.sharding_axis_names
        )
        sharding_dcn_axis_dims = (
            [v for k, v in self.sharding_dcn_axis_dims.items()]
            if isinstance(self.sharding_dcn_axis_dims, dict)
            else self.sharding_dcn_axis_dims
        )
        mesh = self.create_mesh(
            sharding_axis_dims=tuple(sharding_axis_dims) if sharding_axis_dims is not None else sharding_axis_dims,
            sharding_axis_names=tuple(sharding_axis_names) if sharding_axis_names is not None else sharding_axis_names,
            sharding_dcn_axis_dims=tuple(sharding_dcn_axis_dims)
            if sharding_dcn_axis_dims is not None
            else sharding_dcn_axis_dims,
            should_sort_granules_by_key=(
                (self.should_sort_granules_by_key if self.should_sort_granules_by_key is not None else True)
                if hasattr(self, "should_sort_granules_by_key")
                else True
            ),
            allow_split_physical_axes=(
                (self.allow_split_physical_axes if self.allow_split_physical_axes is not None else True)
                if hasattr(self, "allow_split_physical_axes")
                else True
            ),
            backend=((self.backend if self.backend is not None else "") if hasattr(self, "backend") else ""),
        )

        return mesh

    def set_model_mesh(self, mesh: common_types.Mesh):
        self._hidden_mesh = mesh

    def jax_mesh(self):
        warnings.warn("`jax_mesh` is deprecated use `get_mesh` or `mesh`", stacklevel=1)
        return self.get_mesh()

    def get_partition_rules(self, *args, **kwargs):
        """
        Get the partition rules for the model.
        Returns:
            `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
        """
        raise NotImplementedError("`get_partition_rules` is not implemented.")

    def get_axis_dims(self) -> tp.Sequence[int]:
        """The get_axis_dims function returns a sequence of integers representing the dimensions of each axis.

        Args:
            self: Represent the instance of the class

        Returns:
            The dimensions of the axes
        """
        return self.sharding_axis_dims

    def get_axis_names(self) -> tp.Sequence[str]:
        """The get_axis_names function returns a list of the names of the axes.

        Args:
            self: Represent the instance of the class

        Returns:
            A list of the names of all axes
        """
        return self.sharding_axis_names

    def get_backend(self) -> str:
        """The get_backend function returns the backend that is currently being used.
        If no backend has been set, it will return the default JAX backend.

        Args:
            self: Bind the method to an object

        Returns:
            The backend platform
        """
        return self.backend if not self.backend == "" else jax.extend.backend.get_backend().platform

    def read_basics_from_config(self, config: EasyDeLBaseConfig):
        base_reads = [
            "sharding_axis_dims",
            "sharding_dcn_axis_dims",
            "sharding_axis_names",
            "attn_mechanism",
            "decode_attn_mechanism",
            "blocksize_k",
            "blocksize_q",
            "blocksize_b",
            "moe_tiling_size_batch",
            "moe_tiling_size_seqlen",
            "moe_tiling_size_dim",
            "partition_axis",
            "shard_attention_computation",
            "use_sharded_kv_caching",
            "backend",
            "platform",
            "easy_method",
            "bits",
            "scan_ring_attention",
            "scan_attention_layers",
            "use_sharding_constraint",
            "use_pallas_group_matmul",
            "use_scan_mlp",
            "scan_mlp_chunk_size",
            "sequence_axis_name",
            "gradient_checkpointing",
            "precompute_masks",
            "kv_cache_quantization_method",
            "kv_cache_quantization_blocksize",
            "quantization_method",
            "quantization_blocksize",
            "quantization_pattern",
            "kv_cache_sharding_sequence_axis_name",
            "flash_attention_backward_pass_impl",
            "attn_dtype",
            "kvdtype",
            "attn_softmax_dtype",
            "hardware_abstraction",
            "pallas_m_block_size",
            "pallas_k_block_size",
            "pallas_n_block_size",
        ]
        for key in base_reads:
            if hasattr(config, key):
                setattr(self, key, getattr(config, key))

    def add_basic_configurations(
        self,
        sharding_axis_dims: tp.Sequence[int] = NOT_GIVEN,
        sharding_dcn_axis_dims: tp.Sequence[int] | None = NOT_GIVEN,
        sharding_axis_names: tp.Sequence[str] = NOT_GIVEN,
        attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = NOT_GIVEN,
        decode_attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = NOT_GIVEN,
        blocksize_k: int = NOT_GIVEN,
        blocksize_q: int = NOT_GIVEN,
        blocksize_b: int = NOT_GIVEN,
        moe_tiling_size_batch: int = NOT_GIVEN,
        moe_tiling_size_seqlen: int = NOT_GIVEN,
        moe_tiling_size_dim: int = NOT_GIVEN,
        partition_axis: PartitionAxis = NOT_GIVEN,
        shard_attention_computation: bool = NOT_GIVEN,
        use_sharded_kv_caching: bool = NOT_GIVEN,
        backend: EasyDeLBackends | None = NOT_GIVEN,
        platform: EasyDeLPlatforms | None = NOT_GIVEN,
        easy_method: tp.Literal["train", "serve", "convert"] = NOT_GIVEN,
        bits: int | None = NOT_GIVEN,
        scan_ring_attention: bool = NOT_GIVEN,
        scan_attention_layers: bool = NOT_GIVEN,
        use_sharding_constraint: bool = NOT_GIVEN,
        use_pallas_group_matmul: bool = NOT_GIVEN,
        use_scan_mlp: bool = NOT_GIVEN,
        scan_mlp_chunk_size: int = NOT_GIVEN,
        sequence_axis_name: str = NOT_GIVEN,
        gradient_checkpointing: EasyDeLGradientCheckPointers = NOT_GIVEN,
        precompute_masks: bool = NOT_GIVEN,
        kv_cache_quantization_method: EasyDeLQuantizationMethods = NOT_GIVEN,
        kv_cache_quantization_blocksize: int = NOT_GIVEN,
        quantization_method: EasyDeLQuantizationMethods = NOT_GIVEN,
        quantization_blocksize: int = NOT_GIVEN,
        quantization_pattern: str = NOT_GIVEN,
        kv_cache_sharding_sequence_axis_name: str | tuple[str, ...] = NOT_GIVEN,
        flash_attention_backward_pass_impl: tp.Literal["triton", "xla"] = NOT_GIVEN,
        attn_dtype: jnp.dtype = NOT_GIVEN,
        kvdtype: jnp.dtype | None = NOT_GIVEN,
        attn_softmax_dtype: jnp.dtype = NOT_GIVEN,
        hardware_abstraction: bool = NOT_GIVEN,
        pallas_m_block_size: int = NOT_GIVEN,
        pallas_k_block_size: int = NOT_GIVEN,
        pallas_n_block_size: int = NOT_GIVEN,
        **kwargs,
    ):
        """
        It initializes all the attributes of an object, and it's called when you create a new instance of that class.

        Args:
            sharding_axis_dims (tp.Sequence[int], optional): Specify the number of dimensions for each axis.
                Defaults to (1, -1, 1, 1, 1).
            sharding_axis_names (tp.Sequence[str], optional): Set the names of the axes.
                Defaults to ("dp", "fsdp",  "ep", "tp", "sp").
            attn_mechanism (AVAILABLE_ATTENTION_MECHANISMS, optional): attention mechanism to use.
                Defaults to DEFAULT_ATTENTION_MECHANISM.
            decode_attn_mechanism (AVAILABLE_ATTENTION_MECHANISMS): Attention mechanism to use for decode phase.
                Default is None.
            blocksize_k (int, optional): block size of key_states. Defaults to 128.
            blocksize_q (int, optional): block size of query_states. Defaults to 128.
            blocksize_b (int, optional): block size of bias. Defaults to 1.
            moe_tiling_size_batch (int, optional): Block Size for batch in MoE tiling. Default is 4.
            moe_tiling_size_seqlen (int, optional): Block Size for sequence length in MoE tiling. Default is 128.
            moe_tiling_size_dim (int, optional): Block Size for dimension in MoE tiling. Default is 128.
            partition_axis (PartitionAxis, optional): PartitionAxis is new module used for partitioning arrays
                in easydel. Defaults to PartitionAxis().
            shard_attention_computation (bool, optional): whenever to use shard_map for attention. Defaults to True.
            use_sharded_kv_caching (bool, optional): whenever to use shard_map and sharding for key and value.
                Defaults to True.
            backend (tp.Optional[EasyDeLBackends], optional): Specify the backend to use. Defaults to None.
            platform (tp.Optional[EasyDeLPlatforms], optional): Specify the platform to used to use. Defaults to None.
            easy_method (tp.Literal["train", "serve", "convert"], optional): easydel Quantization Method to be applied
                for. Defaults to EasyMethod.TRAIN.
            bits (tp.Optional[int], optional): Model bits for quantization. Defaults to None.
            scan_ring_attention (bool, optional): Whether to use can for ring attention. Defaults to True.
            scan_attention_layers (bool, optional): Whether to use can for attention layers. Defaults to False.
            use_sharding_constraint (bool, optional): whether to use sharding constraint for the arrays.
                Defaults to False.
            use_pallas_group_matmul (bool): Whether to use pallas group matmul. Default is True.
            use_scan_mlp (bool, optional): Determine whether to use scan_mlp or not. Defaults to False.
            scan_mlp_chunk_size (int, optional): Size of chunks in scan MLP. Defaults to 1024.
            sequence_axis_name (str, optional): Name of the attention axis name. Defaults to "sp".
            gradient_checkpointing (EasyDeLQuantizationMethods, optional): Gradient Checkpointing method for
                created or loaded module (applied on mlp and attn layers most of the times).
            kv_cache_quantization_method (EasyDeLQuantizationMethods, optional): key and value quantization
                type. Defaults to EasyDeLQuantizationMethods.NONE.
            kv_cache_quantization_blocksize (int, optional): size of kv cache quantization. Defaults to 64.
            quantization_method (EasyDeLQuantizationMethods, optional): linear modules quantization type.
                Defaults to EasyDeLQuantizationMethods.NONE.
            quantization_blocksize (int, optional): size of linear quantization. Defaults to 64.
            quantization_pattern (str): re pattern to be used for quantizing layers.
            kv_cache_sharding_sequence_axis_name (tp.Union[str, tp.Tuple[str, ...]], optional): axis name to target
                for sharding sequences. Defaults to "sp".
            flash_attention_backward_pass_impl (tp.Literal["triton", "xla"], optional): Specify the backward
                pass kernel for flash attention. Defaults to "triton".
            attn_dtype (jnp.dtype, optional): Data type for attention computations. Defaults to float32.
            kvdtype (jnp.dtype, optional): Data type for attention kv cache. Default is bfloat16.
            attn_softmax_dtype (jnp.dtype, optional): Data type for softmax in attention op computations.
                Defaults to jnp.float32.
            fcm_max_ratio (float, optional): Maximum ratio for flash cross attention. Defaults to 0.0.
            fcm_min_ratio (float, optional): Minimum ratio for flash cross attention. Defaults to 0.0.
            hardware_abstraction (bool, optional): whenever to switch to custom pallas kernels instead of JAX. Defaults
                to DEFAULT_HARDWARE_ABSTRACTION.
            pallas_m_block_size (int, optional): block size m dim in matmul for pallas kernel `A(mk)@B(kn)=B(mn)`.
                Defaults to DEFAULT_PALLAS_M_BLOCK_SIZE.
            pallas_k_block_size (int, optional): block size k dim in matmul for pallas kernel `A(mk)@B(kn)=B(mn)`.
                Defaults to DEFAULT_PALLAS_K_BLOCK_SIZE.
            pallas_n_block_size (int, optional): block size n dim in matmul for pallas kernel `A(mk)@B(kn)=B(mn)`.
                Defaults to DEFAULT_PALLAS_N_BLOCK_SIZE.

        """

        set_attrs_smartly(self, "sharding_axis_dims", (1, -1, 1, 1, 1), sharding_axis_dims)
        set_attrs_smartly(self, "sharding_dcn_axis_dims", None, sharding_dcn_axis_dims)
        set_attrs_smartly(self, "sharding_axis_names", ("dp", "fsdp", "ep", "tp", "sp"), sharding_axis_names)
        set_attrs_smartly(self, "blocksize_q", 512, blocksize_q)
        set_attrs_smartly(self, "blocksize_k", 512, blocksize_k)
        set_attrs_smartly(self, "blocksize_b", 1, blocksize_b)
        set_attrs_smartly(self, "moe_tiling_size_batch", 4, moe_tiling_size_batch)
        set_attrs_smartly(self, "moe_tiling_size_seqlen", 128, moe_tiling_size_seqlen)
        set_attrs_smartly(self, "moe_tiling_size_dim", 128, moe_tiling_size_dim)
        set_attrs_smartly(self, "partition_axis", PartitionAxis(), partition_axis)
        set_attrs_smartly(self, "use_sharding_constraint", False, use_sharding_constraint)
        set_attrs_smartly(self, "use_pallas_group_matmul", True, use_pallas_group_matmul)

        set_attrs_smartly(self, "backend", None, backend)
        set_attrs_smartly(self, "platform", "jax", platform)
        set_attrs_smartly(self, "shard_attention_computation", True, shard_attention_computation)
        set_attrs_smartly(self, "use_sharded_kv_caching", False, use_sharded_kv_caching)
        set_attrs_smartly(self, "attn_mechanism", "vanilla", attn_mechanism)
        set_attrs_smartly(self, "decode_attn_mechanism", None, decode_attn_mechanism)

        set_attrs_smartly(self, "easy_method", EasyMethod.TRAIN, easy_method)
        set_attrs_smartly(self, "bits", None, bits)
        set_attrs_smartly(self, "scan_attention_layers", True, scan_attention_layers)
        set_attrs_smartly(self, "scan_ring_attention", True, scan_ring_attention)
        set_attrs_smartly(self, "use_scan_mlp", False, use_scan_mlp)
        set_attrs_smartly(self, "scan_mlp_chunk_size", 1024, scan_mlp_chunk_size)
        set_attrs_smartly(self, "sequence_axis_name", "sp", sequence_axis_name)
        set_attrs_smartly(self, "kv_cache_quantization_blocksize", 128, kv_cache_quantization_blocksize)
        set_attrs_smartly(self, "kv_cache_sharding_sequence_axis_name", "sp", kv_cache_sharding_sequence_axis_name)
        set_attrs_smartly(self, "gradient_checkpointing", EasyDeLGradientCheckPointers.NONE, gradient_checkpointing)
        set_attrs_smartly(self, "precompute_masks", True, precompute_masks)
        set_attrs_smartly(
            self, "kv_cache_quantization_method", EasyDeLQuantizationMethods.NONE, kv_cache_quantization_method
        )
        set_attrs_smartly(self, "quantization_method", EasyDeLQuantizationMethods.NONE, quantization_method)
        set_attrs_smartly(self, "quantization_blocksize", EasyDeLQuantizationMethods.NONE, quantization_blocksize)
        set_attrs_smartly(self, "quantization_pattern", ".*", quantization_pattern)
        set_attrs_smartly(self, "flash_attention_backward_pass_impl", "triton", flash_attention_backward_pass_impl)
        set_attrs_smartly(self, "attn_dtype", jnp.float32, attn_dtype)
        set_attrs_smartly(self, "kvdtype", jnp.bfloat16, kvdtype if kvdtype is not None else self.attn_dtype)
        set_attrs_smartly(self, "attn_softmax_dtype", jnp.float32, attn_softmax_dtype)
        set_attrs_smartly(self, "hardware_abstraction", DEFAULT_HARDWARE_ABSTRACTION, hardware_abstraction)
        set_attrs_smartly(self, "pallas_m_block_size", DEFAULT_PALLAS_M_BLOCK_SIZE, pallas_m_block_size)
        set_attrs_smartly(self, "pallas_k_block_size", DEFAULT_PALLAS_K_BLOCK_SIZE, pallas_k_block_size)
        set_attrs_smartly(self, "pallas_n_block_size", DEFAULT_PALLAS_N_BLOCK_SIZE, pallas_n_block_size)

        for key_, value_ in kwargs.items():
            setattr(self, key_, value_)
        if getattr(self, "sub_configs", None) is not None:
            for name, _ in getattr(self, "sub_configs", {}).items():
                getattr(self, name).read_basics_from_config(self)
                for key_, value_ in kwargs.items():
                    setattr(getattr(self, name), key_, value_)

    def __repr__(self):
        """The __repr__ function is used to generate a string representation of an object.
        This function should return a string that can be parsed by the Python interpreter
        to recreate the object. The __repr__ function is called when you use print() on an
        object, or when you type its name in the REPL.

        Args:
            self: Refer to the instance of the class

        Returns:
            A string representation of the object
        """

        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                try:
                    repr_src = f"  {k} : " + v.__str__().replace("\n", "\n  ") + "\n"
                    string += repr_src if len(repr_src) < 500 else f"  {k} : " + f"{v.__class__.__name__}(...)" + "\n"
                except TypeError:
                    pass
        return string + ")"

    def to_dict(self) -> dict[str, tp.Any]:
        sd = self.__dict__
        forbidden_types = ["_ScalarMeta"]
        extracted_values = {k: sd.pop(k) for k in list(sd.keys()) if sd.get(k).__class__.__name__ in forbidden_types}
        result = super().to_dict()
        for k, v in extracted_values.items():
            sd[k] = v
        return result

    def attach_custom_arguments(self, **kwargs):
        for k, v in kwargs.items():
            set_attrs_smartly(self, k, v, v)

    def __str__(self):
        """The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        Args:
            self: Refer to the instance of the class

        Returns:
            The object's string representation
        """
        return self.__repr__()

    @classmethod  # From HF.
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`PretrainedConfig`] (or a derived class) from a pretrained model configuration.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~PretrainedConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `tp.Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`Dict[str, tp.Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from this pretrained model.

        Examples:


        >>> # We can't instantiate directly the base class *PretrainedConfig* so let's show the examples on a
        >>> # derived class: BertConfig
        >>> config = BertConfig.from_pretrained(
        ...   "google-bert/bert-base-uncased"
        >>> )  # Download configuration from huggingface.co and cache.
        >>> config = BertConfig.from_pretrained(
        ...   "./test/saved_model/"
        >>> )  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
        >>> config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
        >>> config = BertConfig.from_pretrained(
        ...  "google-bert/bert-base-uncased", output_attentions=True, foo=False
        >>> )
        >>> assert config.output_attentions == True
        >>> config, unused_kwargs = BertConfig.from_pretrained(
        ...  "google-bert/bert-base-uncased",
        ...  output_attentions=True,
        ...  foo=False,
        ...  return_unused_kwargs=True,
        >>> )
        >>> assert config.output_attentions == True
        >>> assert unused_kwargs == {"foo": False}

        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        cls._set_token_in_kwargs(kwargs, token)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(config_dict, **kwargs)

    def save_pretrained(self, save_directory: str | os.PathLike | ePathLike, push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)
        easy_directory = ePath(save_directory)
        if easy_directory.is_file():
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        non_default_generation_parameters = self._get_non_default_generation_parameters()
        if len(non_default_generation_parameters) > 0:
            warnings.warn(
                "Some non-default generation parameters are set in the model config. These should go into either a) "
                "`model.generation_config` (as opposed to `model.config`); OR b) a GenerationConfig file "
                "(https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model)."
                "This warning will become an exception in the future."
                f"\nNon-default generation parameters: {non_default_generation_parameters!s}",
                UserWarning,
                stacklevel=1,
            )

        easy_directory.mkdir(parents=True, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = easy_directory / CONFIG_NAME

        self.to_json_file(output_config_file, use_diff=True)
        logger.debug(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    def to_json_file(self, json_file_path: str | os.PathLike | ePathLike, use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        ePath(json_file_path).write_text(self.to_json_string(use_diff=use_diff))

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike | ePathLike):
        return json.loads(ePath(json_file).read_text(encoding="utf-8"))

    @classmethod
    def _get_config_dict(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        gguf_file = kwargs.get("gguf_file", None)

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored."
            )

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        is_local = ePath(pretrained_model_name_or_path).is_dir()
        if (ePath(subfolder) / pretrained_model_name_or_path).is_file():
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            configuration_file = pretrained_model_name_or_path if gguf_file is None else gguf_file
            resolved_config_file = download_url(pretrained_model_name_or_path)
        else:
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME) if gguf_file is None else gguf_file
            google_cloud_file = ePath(pretrained_model_name_or_path) / configuration_file
            if not google_cloud_file.exists():
                try:
                    resolved_config_file = cached_file(
                        pretrained_model_name_or_path,
                        configuration_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        token=token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _commit_hash=commit_hash,
                    )
                    if resolved_config_file is None:
                        return None, kwargs
                except OSError:
                    raise
                except Exception as e:
                    raise OSError(
                        f"Can't load the configuration of '{pretrained_model_name_or_path}'. "
                        "If you were trying to load it from 'https://huggingface.co/models', "
                        "make sure you don't have a local directory with the same name. Otherwise, "
                        f"make sure '{pretrained_model_name_or_path}' is the correct path to a "
                        f"directory containing a {configuration_file} file"
                    ) from e
            else:
                resolved_config_file = google_cloud_file

            commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
        try:
            if gguf_file:
                config_dict = load_gguf_checkpoint(resolved_config_file, return_tensors=False)["config"]
            else:
                config_dict = cls._dict_from_json_file(resolved_config_file)

            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise OSError(f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file.") from e

        if is_local:
            logger.debug(f"loading configuration file {resolved_config_file}")
        else:
            logger.debug(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        if "model_type" not in config_dict and is_timm_config_dict(config_dict):
            config_dict["model_type"] = "timm_wrapper"

        return config_dict, kwargs

    @property
    def granted_freq_max_position_embedding(self) -> int:
        return getattr(self, "freq_max_position_embeddings", self.max_position_embeddings)

    @property
    def granted_mask_max_position_embedding(self) -> int:
        return getattr(self, "mask_max_position_embeddings", self.max_position_embeddings)

    def _get_rope_config(self) -> RopeConfig:
        """Get RoPE configuration from the instance attributes."""
        from easydel.layers.rotary_embedding import RopeConfig

        if not hasattr(self, "rope_scaling") or self.rope_scaling is None:
            config = RopeConfig()
        else:
            config = RopeConfig.from_dict(self.rope_scaling)

            if config.original_max_position_embeddings is None:
                config.original_max_position_embeddings = getattr(self, "original_max_position_embeddings", None)

        return config

    def get_basic_rope(
        self,
        dtype: chex.Array,
        head_size: int,
        rotary_dim: int | None = None,
        is_neox_style: bool = True,
        base: float | None = None,
    ):
        """
        Get basic rotary position embeddings.

        Args:
            dtype: Data type for the embeddings
            head_size: Size of attention heads
            rotary_dim: Dimension for rotary embeddings (defaults to head_size)
            is_neox_style: Whether to use NeoX style embeddings
            base: Base value for frequency computation (defaults to self.rope_theta)

        Returns:
            Rotary position embeddings func
        """
        from easydel.layers.rotary_embedding import get_rope

        rotary_dim = rotary_dim or head_size
        rope_config = self._get_rope_config()
        return get_rope(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=self.granted_freq_max_position_embedding,
            base=base or self.rope_theta,
            dtype=dtype,
            is_neox_style=is_neox_style,
            rope_scaling=rope_config.to_dict(),
        )

    def get_basic_inv_frequencies(
        self,
        head_size: int | None = None,
        rotary_dim: int | None = None,
        base: float | None = None,
        partial_rotary_factor: float = 1.0,
    ) -> ModuleCaches:
        """
        Get basic inv frequencies for rotary embeddings.

        Args:
            head_size: Size of attention heads (defaults to self.head_dim)
            rotary_dim: Dimension for rotary embeddings (defaults to head_size)
            base: Base value for frequency computation (defaults to self.rope_theta)

        Returns:
            ModuleCaches instance containing computed frequencies
        """
        from easydel.layers.rotary_embedding import get_inv_frequencies

        from .utils import ModuleCaches

        head_size = head_size or self.head_dim
        rotary_dim = rotary_dim or head_size
        rope_config = self._get_rope_config()

        frequencies = get_inv_frequencies(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=self.granted_freq_max_position_embedding,
            base=base or self.rope_theta,
            rope_scaling=rope_config.to_dict(),
            partial_rotary_factor=partial_rotary_factor,
        )

        return ModuleCaches(frequencies)

    def get_basic_frequencies(
        self,
        head_size: int | None = None,
        rotary_dim: int | None = None,
        base: float | None = None,
    ) -> ModuleCaches:
        """
        Get basic frequencies for rotary embeddings.

        Args:
            head_size: Size of attention heads (defaults to self.head_dim)
            rotary_dim: Dimension for rotary embeddings (defaults to head_size)
            base: Base value for frequency computation (defaults to self.rope_theta)

        Returns:
            ModuleCaches instance containing computed frequencies
        """
        from easydel.layers.rotary_embedding import get_frequencies

        from .utils import ModuleCaches

        head_size = head_size or self.head_dim
        rotary_dim = rotary_dim or head_size
        rope_config = self._get_rope_config()

        frequencies = get_frequencies(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=self.granted_freq_max_position_embedding,
            base=base or self.rope_theta,
            rope_scaling=rope_config.to_dict(),
        )

        return ModuleCaches(jax.device_put(frequencies, Ns(self.mesh, Ps())))

    @staticmethod
    def _create_causal_mask(target_length):
        causal_mask_bool = jnp.zeros((target_length, target_length), dtype=jnp.bool_)

        if target_length != 1:
            row_indices = jnp.arange(target_length)[:, None]
            col_indices = jnp.arange(target_length)[None, :]
            lower_triangular = row_indices >= col_indices
            causal_mask_bool = jnp.logical_or(causal_mask_bool, lower_triangular)
        else:
            causal_mask_bool = causal_mask_bool.at[:, 0].set(True)

        row_indices = jnp.arange(target_length)[:, None]
        col_indices = jnp.arange(target_length)[None, :]
        cache_mask = col_indices <= row_indices
        causal_mask_bool = jnp.logical_and(causal_mask_bool, cache_mask)
        causal_mask_bool = causal_mask_bool[None, None, :, :].astype("b1")
        return causal_mask_bool

    def get_mask_details(self) -> dict[int, AttnMaskDetail] | None:
        if hasattr(self, "text_config"):
            return self.get_text_config().get_mask_details()
        return None

    def get_basic_causal_mask(self, *args, **kwargs):
        from .utils import ModuleCaches

        if self.precompute_masks is False:
            return False

        target_length = self.granted_mask_max_position_embedding

        return ModuleCaches(jax.device_put(self._create_causal_mask(target_length), Ns(self.mesh, Ps())))

    def get_fcm_mask(self, batch_size, seq_length, deterministic: bool):
        if not deterministic and self.fcm_max_ratio > 0:
            # Apply forgetful causal mask

            fcm_ratio = jax.random.uniform(
                self.make_rng("fcm"),
                shape=(batch_size, 1, 1, 1),
                minval=self.fcm_min_ratio,
                maxval=self.fcm_max_ratio,
            )
            fcm_mask = (
                jax.random.uniform(self.make_rng("fcm"), shape=(batch_size, 1, seq_length, seq_length)) > fcm_ratio
            )
            fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
            fcm_mask = fcm_mask.astype("bool")
        else:
            fcm_mask = None
        return fcm_mask

    @property
    def partition_manager(self) -> PartitionManager:
        if self.partition_axis is None:
            self.partition_axis = PartitionAxis()
        return PartitionManager(self.partition_axis)

    __hash__ = hash_fn


EasyDeLBaseConfigDict.__doc__ = EasyDeLBaseConfig.__init__.__doc__
EasyDeLBaseConfigDict.__annotations__ = EasyDeLBaseConfig.__annotations__
