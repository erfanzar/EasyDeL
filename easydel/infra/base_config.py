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

import copy
import json
import os
import re
import typing as tp
import warnings
from typing import Any, NotRequired

import jax
import jax.extend
import jax.tree_util
import transformers
from eformer import common_types
from eformer.common_types import DP, EP, FSDP, MODE_TRAIN, NOT_GIVEN, SP, TP
from eformer.escale import PartitionAxis, PartitionManager
from eformer.paths import ePath, ePathLike
from eformer.pytree import auto_pytree
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array

# .venv/lib/python3.13/site-packages/transformers/configuration_utils.py
from transformers.configuration_utils import PretrainedConfig, recursive_diff_dict
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.utils import CONFIG_NAME, cached_file, download_url, is_remote_url
from transformers.utils.generic import is_timm_config_dict

from easydel.layers.quantization import EasyDeLQuantizationConfig
from easydel.utils.compiling_utils import hash_fn
from easydel.utils.helpers import check_bool_flag, get_logger

from .etils import (
    AVAILABLE_ATTENTION_MECHANISMS,
    AVAILABLE_GRADIENT_CHECKPOINT_TARGETS,
    AVAILABLE_GRADIENT_CHECKPOINTS,
    AVAILABLE_MOE_METHODS,
    DEFAULT_ATTENTION_MECHANISM,
    EasyDeLBackends,
    EasyDeLGradientCheckPointers,
    EasyDeLPlatforms,
)

if tp.TYPE_CHECKING:
    from ejkernel.modules.operations.configs import BaseOperationConfig

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
DEFAULT_MOE_METHOD = "fused_moe"
EXPERT_TP_MODE = False
FSDP_IS_EP_BOUND = True
SP_IS_EP_BOUND = True
RING_EXPERTS = False
ED_DEFAULT_HARDWARE_ABSTRACTION = check_bool_flag("ED_DEFAULT_HARDWARE_ABSTRACTION", default=False)
EKERNEL_OPS = check_bool_flag("EKERNEL_OPS", default=False)


if ED_DEFAULT_HARDWARE_ABSTRACTION:
    DEFAULT_HARDWARE_ABSTRACTION = True


if DEFAULT_HARDWARE_ABSTRACTION:
    logger.info("HARDWARE_ABSTRACTION is ON by default")


def _mesh_shape_ep(mesh, pm, fsdp_is_ep_bound, sp_is_ep_bound):
    """Derive flattened mesh shape and axis names for expert-parallel layouts.

    Args:
        mesh: JAX device mesh to extract dimensions from.
        pm: PartitionManager instance for axis name resolution.
        fsdp_is_ep_bound: Whether to fold FSDP axis into expert-parallel axis.
        sp_is_ep_bound: Whether to fold sequence-parallel axis into expert-parallel axis.

    Returns:
        Tuple of ((dp_size, ep_size, tp_size), (dp_name, ep_name, tp_name)) containing
        the flattened mesh dimensions and corresponding axis names.
    """
    # Resolve Names
    dpname, fsdpname, epname, tpname, spname = (
        _resolve_eformer_axis(DP, pm),
        _resolve_eformer_axis(FSDP, pm),
        _resolve_eformer_axis(EP, pm),
        _resolve_eformer_axis(TP, pm),
        _resolve_eformer_axis(SP, pm),
    )

    # Resolve sizes
    odpsize, ofsdpsize, oepsize, otpsize, ospsize = (
        mesh.shape.get(dpname, 1),
        mesh.shape.get(fsdpname, 1),
        mesh.shape.get(epname, 1),
        mesh.shape.get(tpname, 1),
        mesh.shape.get(spname, 1),
    )

    epsize = oepsize
    if fsdp_is_ep_bound:
        epsize *= ofsdpsize
    else:
        odpsize *= ofsdpsize

    if sp_is_ep_bound:
        epsize *= ospsize
    else:
        odpsize *= ospsize
    return (odpsize, epsize, otpsize), (dpname, epname, tpname)


def _resolve_eformer_axis(axis: str | list[str], manager: PartitionManager):
    """Resolve logical axis name(s) to the concrete mesh axis names for training.

    Axis labels such as ``"tp"`` or ``"ep"`` are symbolic and need to be translated
    into the actual mesh axis names chosen by the `PartitionManager`. This helper
    keeps the caller agnostic to how axes are laid out on the physical device mesh.

    Args:
        axis: Single axis name or a list/tuple of axis names to resolve (for example
            ``"tp"``, ``"ep"``, ``"dp"``, ``"fsdp"``, ``"sp"``).
        manager: Partition manager that owns the axis resolution rules.

    Returns:
        Resolved axis name(s). A string is returned for a single input, otherwise a
        list preserving the provided order.

    Example:
        >>> _resolve_eformer_axis("tp", partition_manager)
        >>> _resolve_eformer_axis(["tp", "ep"], partition_manager)
    """
    was_list = isinstance(axis, list | tuple)
    if not was_list:
        axis = [axis]
    out = manager.paxis.resolve_axis(axes=axis, mode=MODE_TRAIN)
    if not was_list:
        return out[0]
    return out


def extract_commit_hash(resolved_file: str | None, commit_hash: str | None) -> str | None:
    """Extracts the git commit hash from a HuggingFace cache file path.

    When models are downloaded from HuggingFace Hub, they're cached locally with paths
    containing the commit hash in the format: `.../snapshots/<commit_hash>/...`.
    This function extracts that hash for tracking model versions.

    Args:
        resolved_file: Path to the resolved cache file. If None or if commit_hash
            is already provided, returns the existing commit_hash immediately.
        commit_hash: Existing commit hash if already known. If provided, this
            function returns it without parsing the file path.

    Returns:
        The extracted commit hash string (40-character hex), or None if:
        - No file path is provided
        - The path doesn't contain a snapshots directory
        - The extracted string doesn't match git commit hash format

    Example:
        >>> path = "/cache/snapshots/abc123def456.../model.safetensors"
        >>> commit_hash = extract_commit_hash(path, None)
        >>> # commit_hash = "abc123def456..." if valid
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
    """Sets attributes intelligently with default values and optional updates.

    This helper function provides smart attribute management:
    1. If the attribute doesn't exist, sets it to the default value
    2. If new_attr is provided (not NOT_GIVEN sentinel), updates the attribute

    This pattern allows configuration classes to have default values while
    supporting explicit overrides through constructor parameters.

    Args:
        self: The object to set the attribute on.
        attr_name: Name of the attribute to set/update.
        default: Default value to use if the attribute doesn't exist yet.
        new_attr: New value to set if provided. If equal to NOT_GIVEN sentinel,
            the existing value (or default) is preserved.

    Example:
        >>> config = SomeConfig()
        >>> set_attrs_smartly(config, "hidden_size", 768, 1024)
        >>> # config.hidden_size = 1024 (updated)
        >>>
        >>> set_attrs_smartly(config, "num_layers", 12, NOT_GIVEN)
        >>> # config.num_layers = 12 (default, since NOT_GIVEN)
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

    sharding_axis_dims: NotRequired[tp.Sequence[int]]
    sharding_dcn_axis_dims: NotRequired[tp.Sequence[int] | None]
    sharding_axis_names: NotRequired[tp.Sequence[str]]
    attn_mechanism: NotRequired[AVAILABLE_ATTENTION_MECHANISMS]
    decode_attn_mechanism: NotRequired[AVAILABLE_ATTENTION_MECHANISMS]
    blocksize_k: NotRequired[int]
    blocksize_q: NotRequired[int]
    blocksize_b: NotRequired[int]
    moe_tiling_size_batch: NotRequired[int]
    moe_tiling_size_seqlen: NotRequired[int]
    moe_tiling_size_dim: NotRequired[int]
    partition_axis: NotRequired[PartitionAxis]
    use_sharded_kv_caching: NotRequired[bool]
    use_sharding_constraint: NotRequired[bool]
    backend: NotRequired[EasyDeLBackends | str | None]
    platform: NotRequired[EasyDeLPlatforms | str | None]
    easy_method: NotRequired[tp.Literal["train", "serve", "convert"]]
    bits: NotRequired[int | None]
    scan_ring_attention: NotRequired[bool]
    scan_attention_layers: NotRequired[bool]
    use_scan_mlp: NotRequired[bool]
    scan_mlp_chunk_size: NotRequired[int]
    sequence_axis_name: NotRequired[str]
    gradient_checkpointing: NotRequired[EasyDeLGradientCheckPointers | str | AVAILABLE_GRADIENT_CHECKPOINTS]
    gradient_checkpointing_targets: NotRequired[list[AVAILABLE_GRADIENT_CHECKPOINT_TARGETS] | None]
    kv_cache_quantization_config: NotRequired[EasyDeLQuantizationConfig | None]
    kv_cache_sharding_sequence_axis_name: NotRequired[str | tuple[str, ...]]
    flash_attention_backward_pass_impl: NotRequired[tp.Literal["triton", "xla"]]
    attn_dtype: NotRequired[jnp.dtype]
    kvdtype: NotRequired[jnp.dtype]
    attn_softmax_dtype: NotRequired[jnp.dtype]
    fcm_max_ratio: NotRequired[float]
    fcm_min_ratio: NotRequired[float]
    hardware_abstraction: NotRequired[bool]
    pallas_m_block_size: NotRequired[int]
    pallas_k_block_size: NotRequired[int]
    pallas_n_block_size: NotRequired[int]
    use_expert_tensor_mode: NotRequired[bool]
    moe_method: NotRequired[AVAILABLE_MOE_METHODS]
    moe_force_xla_gmm: NotRequired[bool]
    use_ring_of_experts: NotRequired[bool]
    fsdp_is_ep_bound: NotRequired[bool]
    sp_is_ep_bound: NotRequired[bool]
    quantization_config: NotRequired[EasyDeLQuantizationConfig | None]
    operation_configs: NotRequired[dict[str, BaseOperationConfig] | None]
    mask_max_position_embeddings: NotRequired[int]
    freq_max_position_embeddings: NotRequired[int]
    precompute_masks: NotRequired[bool]


class EasyDeLBaseConfig(PretrainedConfig):
    """Base configuration shared across EasyDeL models.

    Extends `transformers.PretrainedConfig` with distributed sharding metadata,
    attention kernel selection, quantization knobs, RoPE helpers, and hardware
    abstraction flags used for both training and serving.

    Args:
        sharding_axis_dims: Parallelism sizes for ``(dp, fsdp, ep, tp, sp)``.
            ``-1`` consumes all remaining devices. Defaults to ``(1, -1, 1, 1, 1)``.
        sharding_dcn_axis_dims: Optional mesh sizes for DCN slices when running
            multi-host or multi-slice setups.
        sharding_axis_names: Logical mesh axis names, defaults to
            ``("dp", "fsdp", "ep", "tp", "sp")``.
        attn_mechanism: Attention implementation to use during training/forward passes.
        decode_attn_mechanism: Attention implementation to use during decoding
            (falls back to ``attn_mechanism`` if left as ``None``).
        blocksize_k: Key block size for attention kernels. Defaults to ``128``.
        blocksize_q: Query block size for attention kernels. Defaults to ``128``.
        blocksize_b: Batch/block size used by some attention backends. Defaults to ``1``.
        moe_tiling_size_batch: Batch tiling used by MoE kernels. Defaults to ``4``.
        moe_tiling_size_seqlen: Sequence length tiling for MoE kernels. Defaults to ``128``.
        moe_tiling_size_dim: Hidden dimension tiling for MoE kernels. Defaults to ``128``.
        partition_axis: `PartitionAxis` describing how logical axes map to the mesh.
        use_sharded_kv_caching: Whether to shard KV cache placement instead of replicating.
        use_sharding_constraint: Insert explicit sharding constraints during model build.
        backend: Explicit JAX backend (falls back to ``jax.default_backend()``).
        platform: Platform hint for kernel selection (defaults to ``"triton"`` on GPU,
            otherwise ``"jax"``).
        easy_method: Workflow context (``"train"``, ``"serve"``, or ``"convert"``).
        bits: Optional quantization bit width for weights.
        scan_ring_attention: Use scanning for ring attention implementations.
        scan_attention_layers: Apply scan to attention blocks to save memory.
        use_scan_mlp: Apply scan to MLP blocks.
        scan_mlp_chunk_size: Chunk size when scanning MLPs. Defaults to ``1024``.
        sequence_axis_name: Name of the sequence/attention axis. Defaults to ``"sp"``.
        gradient_checkpointing: Gradient checkpointing policy enum/string.
        gradient_checkpointing_targets: Optional list of target names to include or
            exclude when using selective checkpointing policies.
        precompute_masks: Whether to precompute and cache causal masks on the mesh.
        kv_cache_quantization_config: Quantization config for KV cache tensors. Pass ``None`` to disable.
        quantization_config: Quantization config for linear layers. Pass ``None`` to disable.
        kv_cache_sharding_sequence_axis_name: Axis (or axes) used when sharding the KV cache.
        flash_attention_backward_pass_impl: Backward kernel for flash attention
            (``"triton"`` or ``"xla"``). Defaults to ``"triton"``.
        attn_dtype: Attention activation dtype. Defaults to ``jnp.bfloat16``.
        kvdtype: KV cache dtype. Defaults to ``attn_dtype`` when ``None``.
        attn_softmax_dtype: Softmax computation dtype. Defaults to ``jnp.float32``.
        fcm_max_ratio: Maximum ratio used when sampling forgetful causal masks.
        fcm_min_ratio: Minimum ratio used when sampling forgetful causal masks.
        hardware_abstraction: Enable EasyDeL hardware abstraction and custom kernels.
        pallas_m_block_size: Matmul M dimension block size for Pallas kernels.
        pallas_k_block_size: Matmul K dimension block size for Pallas kernels.
        pallas_n_block_size: Matmul N dimension block size for Pallas kernels.
        moe_method: Mixture-of-experts implementation to use.
        moe_force_xla_gmm: Force XLA GMM kernels for MoE even when fused kernels exist.
        use_ring_of_experts: Whether to dispatch experts with a ring topology.
        use_expert_tensor_mode: Treat experts as an additional tensor-parallel axis.
        fsdp_is_ep_bound: Fold the FSDP axis into the expert axis when building expert meshes.
        sp_is_ep_bound: Fold the sequence-parallel axis into the expert axis when building expert meshes.
        **kwargs: Forwarded to `PretrainedConfig`.

    Raises:
        UserWarning: If KV-cache quantization is requested together with sharded KV caching.
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
        use_sharded_kv_caching: bool = False,
        use_sharding_constraint: bool = False,
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
        gradient_checkpointing_targets: list[AVAILABLE_GRADIENT_CHECKPOINT_TARGETS] | None = None,
        precompute_masks: bool = True,
        kv_cache_quantization_config: EasyDeLQuantizationConfig | None = None,
        quantization_config: EasyDeLQuantizationConfig | None = None,
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
        moe_method: AVAILABLE_MOE_METHODS = DEFAULT_MOE_METHOD,
        moe_force_xla_gmm: bool = False,
        use_expert_tensor_mode: bool = EXPERT_TP_MODE,
        use_ring_of_experts: bool = RING_EXPERTS,
        fsdp_is_ep_bound: bool = FSDP_IS_EP_BOUND,
        sp_is_ep_bound: bool = SP_IS_EP_BOUND,
        operation_configs: dict[str, BaseOperationConfig] | None = None,
        **kwargs,
    ):
        """Initialize base EasyDeL config fields and honor user overrides."""
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
        if isinstance(partition_axis, dict):
            partition_axis = PartitionAxis(**partition_axis)
        self.partition_axis = getattr(self, "partition_axis", partition_axis)
        self.bits = getattr(self, "bits", bits)
        self.scan_attention_layers = getattr(self, "scan_attention_layers", scan_attention_layers)
        self.scan_ring_attention = getattr(self, "scan_ring_attention", scan_ring_attention)
        self.use_sharded_kv_caching = getattr(self, "use_sharded_kv_caching", use_sharded_kv_caching)
        self.use_scan_mlp = getattr(self, "use_scan_mlp", use_scan_mlp)
        self.scan_mlp_chunk_size = getattr(self, "scan_mlp_chunk_size", scan_mlp_chunk_size)
        self.use_sharding_constraint = getattr(self, "use_sharding_constraint", use_sharding_constraint)
        self.sequence_axis_name = getattr(self, "sequence_axis_name", sequence_axis_name)
        self.kv_cache_sharding_sequence_axis_name = getattr(
            self, "kv_cache_sharding_sequence_axis_name", kv_cache_sharding_sequence_axis_name
        )
        self.gradient_checkpointing = getattr(self, "gradient_checkpointing", gradient_checkpointing)
        self.gradient_checkpointing_targets = getattr(
            self, "gradient_checkpointing_targets", gradient_checkpointing_targets
        )
        self.precompute_masks = getattr(self, "precompute_masks", precompute_masks)

        self.kv_cache_quantization_config = getattr(self, "kv_cache_quantization_config", kv_cache_quantization_config)
        self.quantization_config = getattr(self, "quantization_config", quantization_config)
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
        self.moe_method = getattr(self, "moe_method", moe_method)
        self.moe_force_xla_gmm = getattr(self, "moe_force_xla_gmm", moe_force_xla_gmm)
        self.use_ring_of_experts = getattr(self, "use_ring_of_experts", use_ring_of_experts)
        self.use_expert_tensor_mode = getattr(self, "use_expert_tensor_mode", use_expert_tensor_mode)
        self.fsdp_is_ep_bound = getattr(self, "fsdp_is_ep_bound", fsdp_is_ep_bound)
        self.sp_is_ep_bound = getattr(self, "sp_is_ep_bound", sp_is_ep_bound)
        self.operation_configs = getattr(self, "operation_configs", operation_configs)
        self.pretraining_tp = 1  # it's for pytorch models.
        if self.kv_cache_quantization_config is not None and self.use_sharded_kv_caching:
            use_sharded_kv_caching = self.use_sharded_kv_caching
            warnings.warn(
                f"`kv_cache_quantization_config={self.kv_cache_quantization_config}` and `{use_sharded_kv_caching=}`"
                " can't be used together at the moment.",
                stacklevel=1,
            )

        self._external_rope_config_kwargs = getattr(self, "_external_rope_config_kwargs", {})
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
        """Creates a JAX device mesh for distributed model execution.

        This function constructs a multi-dimensional mesh of devices that defines how
        model parameters and computations are distributed across hardware. The mesh axes
        correspond to different parallelism strategies:

        - dp (data parallel): Replicate model across data batches
        - fsdp (fully sharded data parallel): Shard parameters and optimizer states
        - ep (expert parallel): Distribute experts in MoE models
        - tp (tensor parallel): Partition individual weight matrices
        - sp (sequence parallel): Split sequence dimension across devices

        Args:
            sharding_axis_dims: Size of each parallelism dimension. Use -1 to automatically
                fill remaining devices. Default: (1, -1, 1, 1, 1) means all devices go
                to FSDP axis.
            sharding_axis_names: Names for each mesh axis. Must match length of
                sharding_axis_dims. Default: ("dp", "fsdp", "ep", "tp", "sp").
            sharding_dcn_axis_dims: Dimensions for data center network (DCN) sharding.
                Used for multi-host/multi-slice setups. Default: None.
            process_is_granule: Deprecated parameter, not used.
            should_sort_granules_by_key: Whether to sort device granules by key for
                deterministic ordering. Default: True.
            allow_split_physical_axes: Whether to allow splitting physical device axes
                when mapping to logical mesh axes. Default: True.
            backend: Backend platform to create mesh for ('gpu', 'tpu', etc.).
                If None or empty string, uses default backend.

        Returns:
            A JAX Mesh object configured for distributed execution with the specified
            parallelism dimensions.

        Example:
            >>> # Create mesh with DP=4, TP=2 on 8 GPUs
            >>> mesh = EasyDeLBaseConfig.create_mesh(
            ...     sharding_axis_dims=(4, 1, 1, 2, 1),
            ...     sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
            ... )
            >>> # mesh.shape = {'dp': 4, 'fsdp': 1, 'ep': 1, 'tp': 2, 'sp': 1}
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
            use_jax=not check_bool_flag("ED_CREATE_MESH", default=False),
        )
        return mesh

    @property
    def mesh(self):
        """Gets or creates the JAX device mesh for this configuration.

        This property lazily constructs a device mesh from the configuration's sharding
        parameters. Once created, the mesh is cached for reuse. The mesh can be explicitly
        set using `set_model_mesh()` to override the auto-generated one.

        The mesh is constructed from:
        - sharding_axis_dims: Device counts per axis
        - sharding_axis_names: Logical names for each axis
        - sharding_dcn_axis_dims: Multi-host configuration (if applicable)
        - Various granule sorting and axis splitting options

        Returns:
            JAX Mesh object defining the device topology for distributed execution.
            The mesh axes correspond to parallelism strategies (dp, fsdp, ep, tp, sp).

        Note:
            If a custom mesh was set via `set_model_mesh()`, that mesh is returned
            instead of creating a new one.

        Example:
            >>> config = EasyDeLBaseConfig(
            ...     sharding_axis_dims=(2, 1, 1, 4, 1),
            ...     sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
            ... )
            >>> mesh = config.mesh
            >>> # mesh.shape = {'dp': 2, 'fsdp': 1, 'ep': 1, 'tp': 4, 'sp': 1}
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
        self.set_model_mesh(mesh)
        return self._hidden_mesh

    @property
    def expert_mesh(self) -> jax.sharding.Mesh:
        """Get the mesh configuration for expert parallelism.

        Creates a mesh with expert-parallel axes folded according to the
        `fsdp_is_ep_bound` and `sp_is_ep_bound` configuration flags. This mesh
        is used for MoE (Mixture of Experts) models to distribute experts
        across devices with explicit axis types.

        Returns:
            jax.sharding.Mesh: A mesh with explicit axis types configured for
                expert parallelism with (dp, ep, tp) axis ordering.
        """
        (odpsize, epsize, otpsize), (dpname, epname, tpname) = _mesh_shape_ep(
            self.mesh,
            self.partition_manager,
            self.fsdp_is_ep_bound,
            self.sp_is_ep_bound,
        )
        return jax.sharding.Mesh(
            self.mesh.devices.flatten().reshape(odpsize, epsize, otpsize),
            axis_names=(dpname, epname, tpname),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
        )

    @property
    def expert_abstract_mesh(self) -> jax.sharding.AbstractMesh:
        """Get the abstract mesh descriptor for expert parallelism.

        Returns an abstract mesh that matches the `expert_mesh` axis sizes
        and names. Abstract meshes are lightweight representations used for
        sharding specification without device assignment.

        Returns:
            jax.sharding.AbstractMesh: An abstract mesh descriptor with the same
                axis configuration as `expert_mesh`.
        """
        (odpsize, epsize, otpsize), (dpname, epname, tpname) = _mesh_shape_ep(
            self.mesh,
            self.partition_manager,
            self.fsdp_is_ep_bound,
            self.sp_is_ep_bound,
        )
        return self.expert_mesh.abstract_mesh.update(
            axis_sizes=(odpsize, epsize, otpsize),
            axis_names=(dpname, epname, tpname),
        )

    @property
    def auto_expert_mesh(self) -> jax.sharding.Mesh:
        """Get the mesh for expert parallelism with automatic axis types.

        Similar to `expert_mesh`, but uses `jax.sharding.AxisType.Auto` for
        all axes, allowing JAX to automatically determine the optimal sharding
        strategy based on the computation graph.

        Returns:
            jax.sharding.Mesh: A mesh with auto axis types configured for
                expert parallelism with (dp, ep, tp) axis ordering.
        """
        (odpsize, epsize, otpsize), (dpname, epname, tpname) = _mesh_shape_ep(
            self.mesh,
            self.partition_manager,
            self.fsdp_is_ep_bound,
            self.sp_is_ep_bound,
        )
        return jax.sharding.Mesh(
            self.mesh.devices.flatten().reshape(odpsize, epsize, otpsize),
            axis_names=(dpname, epname, tpname),
            axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
        )

    def set_model_mesh(self, mesh: common_types.Mesh):
        """Sets a custom mesh for the model, overriding the auto-generated one.

        Args:
            mesh: JAX device mesh to use for this model.
        """
        self._hidden_mesh = mesh

        sub_configs = getattr(self, "sub_configs", None)
        if not isinstance(sub_configs, dict):
            return

        for attr_name in sub_configs.keys():
            sub_cfg = getattr(self, attr_name, None)
            if sub_cfg is None:
                continue
            try:
                if hasattr(sub_cfg, "set_model_mesh"):
                    sub_cfg.set_model_mesh(mesh)
                else:
                    sub_cfg._hidden_mesh = mesh
            except Exception:
                try:
                    sub_cfg._hidden_mesh = mesh
                except Exception:
                    pass

    def jax_mesh(self):
        """Deprecated method for getting the JAX mesh.

        Deprecated:
            Use `mesh` property or `get_mesh()` method instead.

        Returns:
            JAX device mesh.
        """
        warnings.warn("`jax_mesh` is deprecated use `get_mesh` or `mesh`", stacklevel=1)
        return self.get_mesh()

    def get_partition_rules(self, *args, **kwargs):
        """Gets the parameter sharding partition rules for the model.

        Partition rules define how model parameters should be sharded across the device mesh.
        Each rule maps a parameter name pattern (regex) to a PartitionSpec that specifies
        which mesh axes the parameter dimensions should be distributed across.

        This method must be implemented by model-specific configuration classes.

        Args:
            *args: Positional arguments (model-specific).
            **kwargs: Keyword arguments (model-specific).

        Returns:
            Tuple of (pattern, PartitionSpec) pairs defining how to shard parameters.
            For example: (("model/embed.*", PartitionSpec("tp", None)),
                         ("model/layers/\\d+/attn/.*", PartitionSpec(None, "tp")))

        Raises:
            NotImplementedError: This base class does not provide default partition rules.
                Subclasses must implement this method.

        Example:
            >>> class MyModelConfig(EasyDeLBaseConfig):
            ...     def get_partition_rules(self):
            ...         return (
            ...             ("embed.*", PartitionSpec("tp", None)),
            ...             ("attn.*", PartitionSpec(None, "tp", None)),
            ...             ("mlp.*", PartitionSpec(None, "tp")),
            ...         )
        """
        raise NotImplementedError("`get_partition_rules` is not implemented.")

    def get_axis_dims(self) -> tp.Sequence[int]:
        """Returns the device mesh axis dimensions for parallelism.

        Returns:
            Sequence of integers specifying the size of each parallelism axis.
            Typically (dp_size, fsdp_size, ep_size, tp_size, sp_size).
            Value of -1 means "use all remaining devices for this axis".

        Example:
            >>> config.sharding_axis_dims = (2, 4, 1, 1, 1)
            >>> dims = config.get_axis_dims()
            >>> # dims = (2, 4, 1, 1, 1) - 2 data parallel, 4 FSDP, rest replicated
        """
        return self.sharding_axis_dims

    def get_axis_names(self) -> tp.Sequence[str]:
        """Returns the logical names for each device mesh axis.

        Returns:
            Sequence of strings naming each parallelism axis.
            Typically ("dp", "fsdp", "ep", "tp", "sp") for data parallel,
            fully sharded data parallel, expert parallel, tensor parallel,
            and sequence parallel respectively.

        Example:
            >>> names = config.get_axis_names()
            >>> # names = ('dp', 'fsdp', 'ep', 'tp', 'sp')
        """
        return self.sharding_axis_names

    def get_backend(self) -> str:
        """Returns the JAX backend platform being used.

        Retrieves the configured backend (e.g., 'gpu', 'tpu', 'cpu'), or falls back
        to the default JAX backend if not explicitly set.

        Returns:
            Backend platform string. Common values: 'gpu', 'tpu', 'cpu'.

        Example:
            >>> config = EasyDeLBaseConfig(backend='gpu')
            >>> config.get_backend()
            'gpu'
            >>>
            >>> # With no backend set, returns JAX default
            >>> config = EasyDeLBaseConfig(backend='')
            >>> config.get_backend()  # Might return 'gpu', 'tpu', etc.
        """
        return self.backend if not self.backend == "" else jax.extend.backend.get_backend().platform

    def read_basics_from_config(self, config: EasyDeLBaseConfig):
        """Reads and applies basic configuration attributes from another config instance.

        Copies EasyDeL-specific attributes like sharding, attention mechanism,
        quantization settings, etc. from the provided config.

        Args:
            config: Source configuration to read attributes from.
        """
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
            "use_sharded_kv_caching",
            "backend",
            "platform",
            "easy_method",
            "bits",
            "scan_ring_attention",
            "scan_attention_layers",
            "use_sharding_constraint",
            "use_scan_mlp",
            "scan_mlp_chunk_size",
            "sequence_axis_name",
            "gradient_checkpointing",
            "gradient_checkpointing_targets",
            "precompute_masks",
            "kv_cache_quantization_config",
            "quantization_config",
            "kv_cache_sharding_sequence_axis_name",
            "flash_attention_backward_pass_impl",
            "attn_dtype",
            "kvdtype",
            "attn_softmax_dtype",
            "hardware_abstraction",
            "pallas_m_block_size",
            "pallas_k_block_size",
            "pallas_n_block_size",
            "moe_method",
            "moe_force_xla_gmm",
            "use_ring_of_experts",
            "use_expert_tensor_mode",
            "fsdp_is_ep_bound",
            "sp_is_ep_bound",
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
        use_sharded_kv_caching: bool = NOT_GIVEN,
        backend: EasyDeLBackends | None = NOT_GIVEN,
        platform: EasyDeLPlatforms | None = NOT_GIVEN,
        easy_method: tp.Literal["train", "serve", "convert"] = NOT_GIVEN,
        bits: int | None = NOT_GIVEN,
        scan_ring_attention: bool = NOT_GIVEN,
        scan_attention_layers: bool = NOT_GIVEN,
        use_sharding_constraint: bool = NOT_GIVEN,
        use_scan_mlp: bool = NOT_GIVEN,
        scan_mlp_chunk_size: int = NOT_GIVEN,
        sequence_axis_name: str = NOT_GIVEN,
        gradient_checkpointing: EasyDeLGradientCheckPointers = NOT_GIVEN,
        gradient_checkpointing_targets: list[AVAILABLE_GRADIENT_CHECKPOINT_TARGETS] | None = NOT_GIVEN,
        precompute_masks: bool = NOT_GIVEN,
        kv_cache_quantization_config: EasyDeLQuantizationConfig | None = NOT_GIVEN,
        quantization_config: EasyDeLQuantizationConfig | None = NOT_GIVEN,
        kv_cache_sharding_sequence_axis_name: str | tuple[str, ...] = NOT_GIVEN,
        flash_attention_backward_pass_impl: tp.Literal["triton", "xla"] = NOT_GIVEN,
        attn_dtype: jnp.dtype = NOT_GIVEN,
        kvdtype: jnp.dtype | None = NOT_GIVEN,
        attn_softmax_dtype: jnp.dtype = NOT_GIVEN,
        hardware_abstraction: bool = NOT_GIVEN,
        pallas_m_block_size: int = NOT_GIVEN,
        pallas_k_block_size: int = NOT_GIVEN,
        pallas_n_block_size: int = NOT_GIVEN,
        moe_method: AVAILABLE_MOE_METHODS = NOT_GIVEN,
        moe_force_xla_gmm: bool = NOT_GIVEN,
        use_ring_of_experts: bool = NOT_GIVEN,
        use_expert_tensor_mode: bool = NOT_GIVEN,
        fsdp_is_ep_bound: bool = NOT_GIVEN,
        sp_is_ep_bound: bool = NOT_GIVEN,
        **kwargs,
    ):
        """
        Populate baseline EasyDeL attributes on an existing config instance.

        Each argument mirrors the constructor but is optional: passing `NOT_GIVEN`
        leaves any existing attribute untouched, while a provided value overwrites
        the current setting. If an attribute is missing entirely, a sensible default
        is applied via `set_attrs_smartly`. This helper is used by derived configs
        (and their `sub_configs`) to keep sharding/attention/quantization knobs in
        sync without re-implementing initialization logic.

        Args:
            sharding_axis_dims: Fallback mesh sizes for ``(dp, fsdp, ep, tp, sp)``,
                defaulting to ``(1, -1, 1, 1, 1)``.
            sharding_dcn_axis_dims: Optional DCN mesh sizes (default ``None``).
            sharding_axis_names: Mesh axis labels, default ``("dp", "fsdp", "ep", "tp", "sp")``.
            attn_mechanism: Attention mechanism to use (default ``"vanilla"``).
            decode_attn_mechanism: Optional decode-time attention mechanism.
            blocksize_k: Attention key block size, default ``512`` when unset.
            blocksize_q: Attention query block size, default ``512`` when unset.
            blocksize_b: Batch/block size used by attention kernels (default ``1``).
            moe_tiling_size_batch: Batch tiling for MoE kernels (default ``4``).
            moe_tiling_size_seqlen: Sequence tiling for MoE kernels (default ``128``).
            moe_tiling_size_dim: Hidden-dim tiling for MoE kernels (default ``128``).
            partition_axis: PartitionAxis describing logical mesh layout (default ``PartitionAxis()``).
            use_sharded_kv_caching: Whether to shard KV caches (default ``False``).
            backend: Backend string, default ``None`` (falls back to JAX default).
            platform: Platform hint, default ``"jax"``.
            easy_method: EasyDeL execution mode, default ``EasyMethod.TRAIN``.
            bits: Optional quantization bit width, default ``None``.
            scan_ring_attention: Enable scan for ring attention (default ``True``).
            scan_attention_layers: Enable scan for attention blocks (default ``True``).
            use_sharding_constraint: Insert sharding constraints (default ``False``).
            use_scan_mlp: Enable scan for MLPs (default ``False``).
            scan_mlp_chunk_size: Chunk size for scanned MLPs (default ``1024``).
            sequence_axis_name: Label for the sequence/attention axis (default ``"sp"``).
            gradient_checkpointing: Gradient checkpointing policy (default ``EasyDeLGradientCheckPointers.NONE``).
            gradient_checkpointing_targets: Optional list of checkpoint targets to include/exclude (default ``None``).
            precompute_masks: Whether to precompute and cache masks (default ``True``).
            kv_cache_quantization_config: KV cache quantization config (default ``None`` = no quantization).
            quantization_config: Linear-layer quantization config (default ``None`` = no quantization).
            kv_cache_sharding_sequence_axis_name: Axis name(s) for KV cache sharding (default ``"sp"``).
            flash_attention_backward_pass_impl: Backward kernel for flash attention (default ``"triton"``).
            attn_dtype: Attention activation dtype (default ``jnp.float32``).
            kvdtype: KV cache dtype (defaults to `attn_dtype` when unset).
            attn_softmax_dtype: Softmax computation dtype (default ``jnp.float32``).
            hardware_abstraction: Toggle EasyDeL hardware abstraction (default ``DEFAULT_HARDWARE_ABSTRACTION``).
            pallas_m_block_size: Pallas matmul M block size (default ``DEFAULT_PALLAS_M_BLOCK_SIZE``).
            pallas_k_block_size: Pallas matmul K block size (default ``DEFAULT_PALLAS_K_BLOCK_SIZE``).
            pallas_n_block_size: Pallas matmul N block size (default ``DEFAULT_PALLAS_N_BLOCK_SIZE``).
            moe_method: MoE implementation to use (default ``DEFAULT_MOE_METHOD``).
            moe_force_xla_gmm: Force XLA GMM kernels for MoE (default ``False``).
            use_ring_of_experts: Dispatch experts with a ring topology (default ``RING_EXPERTS``).
            use_expert_tensor_mode: Treat experts as a tensor-parallel axis (default ``EXPERT_TP_MODE``).
            fsdp_is_ep_bound: Fold FSDP into the expert axis when building expert meshes.
            sp_is_ep_bound: Fold sequence-parallel into the expert axis when building expert meshes.
            **kwargs: Extra attributes to attach to this config and any defined ``sub_configs``.
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

        set_attrs_smartly(self, "backend", None, backend)
        set_attrs_smartly(self, "platform", "jax", platform)
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
        set_attrs_smartly(self, "kv_cache_sharding_sequence_axis_name", "sp", kv_cache_sharding_sequence_axis_name)
        set_attrs_smartly(self, "gradient_checkpointing", EasyDeLGradientCheckPointers.NONE, gradient_checkpointing)
        set_attrs_smartly(self, "gradient_checkpointing_targets", None, gradient_checkpointing_targets)
        set_attrs_smartly(self, "precompute_masks", True, precompute_masks)
        set_attrs_smartly(self, "kv_cache_quantization_config", None, kv_cache_quantization_config)
        set_attrs_smartly(self, "quantization_config", None, quantization_config)
        set_attrs_smartly(self, "flash_attention_backward_pass_impl", "triton", flash_attention_backward_pass_impl)
        set_attrs_smartly(self, "attn_dtype", jnp.float32, attn_dtype)
        set_attrs_smartly(self, "kvdtype", jnp.bfloat16, kvdtype if kvdtype is not None else self.attn_dtype)
        set_attrs_smartly(self, "attn_softmax_dtype", jnp.float32, attn_softmax_dtype)
        set_attrs_smartly(self, "hardware_abstraction", DEFAULT_HARDWARE_ABSTRACTION, hardware_abstraction)
        set_attrs_smartly(self, "pallas_m_block_size", DEFAULT_PALLAS_M_BLOCK_SIZE, pallas_m_block_size)
        set_attrs_smartly(self, "pallas_k_block_size", DEFAULT_PALLAS_K_BLOCK_SIZE, pallas_k_block_size)
        set_attrs_smartly(self, "pallas_n_block_size", DEFAULT_PALLAS_N_BLOCK_SIZE, pallas_n_block_size)
        set_attrs_smartly(self, "moe_method", DEFAULT_MOE_METHOD, moe_method)
        set_attrs_smartly(self, "moe_force_xla_gmm", False, moe_force_xla_gmm)
        set_attrs_smartly(self, "use_ring_of_experts", RING_EXPERTS, use_ring_of_experts)
        set_attrs_smartly(self, "use_expert_tensor_mode", EXPERT_TP_MODE, use_expert_tensor_mode)
        set_attrs_smartly(self, "fsdp_is_ep_bound", FSDP_IS_EP_BOUND, fsdp_is_ep_bound)
        set_attrs_smartly(self, "sp_is_ep_bound", SP_IS_EP_BOUND, sp_is_ep_bound)

        for key_, value_ in kwargs.items():
            setattr(self, key_, value_)
        if getattr(self, "sub_configs", None) is not None:
            for name, _ in getattr(self, "sub_configs", {}).items():
                getattr(self, name).read_basics_from_config(self)
                for key_, value_ in kwargs.items():
                    setattr(getattr(self, name), key_, value_)

    # Attributes to hide from __repr__ and __str__ output
    _hidden_repr_attrs: tp.ClassVar[set[str]] = {
        # EasyDeL-specific attributes
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
        "use_sharded_kv_caching",
        "use_sharding_constraint",
        "backend",
        "platform",
        "easy_method",
        "bits",
        "scan_ring_attention",
        "scan_attention_layers",
        "use_scan_mlp",
        "scan_mlp_chunk_size",
        "sequence_axis_name",
        "gradient_checkpointing",
        "gradient_checkpointing_targets",
        "precompute_masks",
        "kv_cache_quantization_config",
        "quantization_config",
        "kv_cache_sharding_sequence_axis_name",
        "flash_attention_backward_pass_impl",
        "attn_dtype",
        "kvdtype",
        "attn_softmax_dtype",
        "fcm_max_ratio",
        "fcm_min_ratio",
        "hardware_abstraction",
        "pallas_m_block_size",
        "pallas_k_block_size",
        "pallas_n_block_size",
        "moe_method",
        "moe_force_xla_gmm",
        "use_expert_tensor_mode",
        "use_ring_of_experts",
        "fsdp_is_ep_bound",
        "sp_is_ep_bound",
        "operation_configs",
        # HuggingFace PretrainedConfig default attributes
        "pretraining_tp",
        "return_dict",
        "output_hidden_states",
        "torchscript",
        "dtype",
        "task_specific_params",
        "problem_type",
        "tokenizer_class",
        "prefix",
        "bos_token_id",
        "pad_token_id",
        "eos_token_id",
        "sep_token_id",
        "decoder_start_token_id",
        "max_length",
        "min_length",
        "do_sample",
        "early_stopping",
        "num_beams",
        "temperature",
        "top_k",
        "top_p",
        "typical_p",
        "repetition_penalty",
        "length_penalty",
        "no_repeat_ngram_size",
        "encoder_no_repeat_ngram_size",
        "bad_words_ids",
        "num_return_sequences",
        "output_scores",
        "return_dict_in_generate",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "remove_invalid_values",
        "exponential_decay_length_penalty",
        "suppress_tokens",
        "begin_suppress_tokens",
        "num_beam_groups",
        "diversity_penalty",
        "transformers_version",
        "add_cross_attention",
        "tie_encoder_decoder",
        "architectures",
        "finetuning_task",
        "id2label",
        "label2id",
        "chunk_size_feed_forward",
        "is_encoder_decoder",
        "is_decoder",
        "cross_attention_hidden_size",
        "tie_word_embeddings",
        "output_attentions",
        "pruned_heads",
        "tf_legacy_loss",
        "use_bfloat16",
    }

    def __repr__(self):
        """Return a multi-line summary of public config fields.

        The output lists non-private attributes on separate lines and truncates
        long values to keep the representation readable. EasyDeL-specific internal
        attributes are hidden unless `_show_private_attrs` is True.
        """

        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if not self._show_private_attrs and k in self._hidden_repr_attrs:
                continue
            try:
                repr_src = f"  {k} : " + v.__str__().replace("\n", "\n  ") + "\n"
                string += repr_src if len(repr_src) < 1500 else f"  {k} : " + f"{v.__class__.__name__}(...)" + "\n"
            except TypeError:
                pass
        return string + ")"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from the configuration that correspond to the default config attributes for
        better readability, while always retaining the `config` attribute from the class. Serializes to a
        Python dictionary.

        Returns:
            dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        config_dict = self.to_dict()
        default_config_dict = PretrainedConfig().to_dict()
        class_config_dict = self.__class__().to_dict() if not self.has_no_defaults_at_init else {}

        serializable_config_dict = {}

        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ) or key in self.sub_configs:
                diff = recursive_diff_dict(value, default_config_dict, config_obj=getattr(self, key, None))
                if "model_type" in value:
                    diff["model_type"] = value["model_type"]

                serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or key == "vocab_file"
                or value != default_config_dict[key]
                or (key in default_config_dict and value != class_config_dict.get(key, value))
            ):
                serializable_config_dict[key] = value

        self._remove_keys_not_serialized(serializable_config_dict)

        serializable_config_dict.pop("_name_or_path", None)

        self.dict_dtype_to_str(serializable_config_dict)

        return serializable_config_dict

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize config to a dictionary while temporarily hiding forbidden types.

        Notes:
            EasyDeL caches the active JAX mesh on the config (``_hidden_mesh``) for runtime use.
            That object contains non-picklable JAX devices, so we must exclude it from any deep
            copies performed during serialization.
        """
        sd = self.__dict__
        forbidden_types = {"_ScalarMeta"}
        extracted_values: dict[str, tp.Any] = {}

        for key in list(sd.keys()):
            value = sd.get(key)
            if key == "_hidden_mesh" or value.__class__.__name__ in forbidden_types:
                extracted_values[key] = sd.pop(key)

        try:
            result = copy.deepcopy(self.__dict__)
            if hasattr(self.__class__, "model_type"):
                result["model_type"] = self.__class__.model_type

            result["transformers_version"] = transformers.__version__

            for key, value in result.items():
                if isinstance(value, PretrainedConfig):
                    value = value.to_dict()
                    del value["transformers_version"]

                result[key] = value

            self._remove_keys_not_serialized(result)
            self.dict_dtype_to_str(result)
            return result
        finally:
            for key, value in extracted_values.items():
                sd[key] = value

    def __deepcopy__(self, memo):
        """Deep copy the config while keeping the cached runtime mesh by reference."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            if key == "_hidden_mesh":
                setattr(result, key, value)
            else:
                setattr(result, key, copy.deepcopy(value, memo))

        return result

    def attach_custom_arguments(self, **kwargs):
        """Attaches custom arguments as attributes to the configuration.

        Args:
            **kwargs: Arbitrary key-value pairs to attach as attributes.
        """
        for k, v in kwargs.items():
            set_attrs_smartly(self, k, v, v)

    def __str__(self):
        """Alias for `__repr__` to provide a readable config summary."""
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
    ):
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
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        # We can't instantiate directly the base class *PretrainedConfig* so let's show the examples on a
        # derived class: BertConfig
        config = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased"
        )  # Download configuration from huggingface.co and cache.
        config = BertConfig.from_pretrained(
            "./test/saved_model/"
        )  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
        config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
        config = BertConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        assert config.output_attentions == True
        config, unused_kwargs = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        )
        assert config.output_attentions == True
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        cls._set_token_in_kwargs(kwargs, token)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            for v in config_dict.values():
                if isinstance(v, dict) and v.get("model_type") == cls.model_type:
                    config_dict = v

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
        """Loads a configuration dictionary from a JSON file.

        Args:
            json_file: Path to the JSON configuration file.

        Returns:
            Dictionary containing the parsed configuration.
        """
        return json.loads(ePath(json_file).read_text(encoding="utf-8"))

    @classmethod
    def _get_config_dict(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Load a configuration dictionary from local path or Hub, handling gguf."""
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
        """Return the max position embedding allowed for frequency-based caches."""
        return getattr(self, "freq_max_position_embeddings", self.max_position_embeddings)

    @property
    def granted_mask_max_position_embedding(self) -> int:
        """Return the max position embedding allowed for mask precomputation."""
        return getattr(self, "mask_max_position_embeddings", self.max_position_embeddings)

    def _get_rope_config(self) -> RopeConfig:
        """Build a `RopeConfig` from the config fields.

        If ``rope_scaling`` is provided, it is converted to a `RopeConfig` and
        missing ``original_max_position_embeddings`` values are filled from the
        base config. Otherwise a default `RopeConfig` instance is returned.
        """
        from easydel.layers.rotary_embedding import RopeConfig

        if not hasattr(self, "rope_scaling") or self.rope_scaling is None:
            config = RopeConfig()
        else:
            config = RopeConfig.from_dict(self.rope_scaling)

        if config.original_max_position_embeddings is None:
            config.original_max_position_embeddings = getattr(self, "original_max_position_embeddings", None)

        if self._external_rope_config_kwargs is not None and len(self._external_rope_config_kwargs.keys()) > 0:
            config.update(**self._external_rope_config_kwargs)

        return config

    def get_basic_rope(
        self,
        dtype: Array,
        head_size: int,
        rotary_dim: int | None = None,
        is_neox_style: bool = True,
        base: float | None = None,
    ):
        """Return a rotary position embedding function configured for this model.

        Args:
            dtype: Target dtype for the generated embeddings.
            head_size: Attention head size used to derive the rotary dimension.
            rotary_dim: Number of rotary dimensions (defaults to ``head_size``).
            is_neox_style: Whether to generate NeoX-style rotary embeddings.
            base: Optional base used for frequency computation (defaults to ``self.rope_theta``).

        Returns:
            Callable from `get_rope` ready to be applied to query/key tensors.
        """
        from easydel.layers.rotary_embedding import get_rope

        partial_rotary_factor = getattr(self, "partial_rotary_factor", 1.0)
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
            partial_rotary_factor=partial_rotary_factor,
        )

    def get_basic_inv_frequencies(
        self,
        head_size: int | None = None,
        rotary_dim: int | None = None,
        base: float | None = None,
        partial_rotary_factor: float = 1.0,
    ) -> ModuleCaches:
        """Compute inverse frequencies for rotary embeddings.

        Args:
            head_size: Attention head size (defaults to ``self.head_dim``).
            rotary_dim: Number of rotary dimensions (defaults to ``head_size``).
            base: Optional base for frequency computation (defaults to ``self.rope_theta``).
            partial_rotary_factor: Ratio of the head dimension to apply RoPE to.

        Returns:
            `ModuleCaches` wrapping the computed frequency tensor.
        """
        from easydel.layers.rotary_embedding import get_inv_frequencies

        from .utils import ModuleCaches

        partial_rotary_factor = getattr(self, "partial_rotary_factor", partial_rotary_factor)
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
        """Compute frequencies for rotary embeddings placed on the configured mesh.

        Args:
            head_size: Attention head size (defaults to ``self.head_dim``).
            rotary_dim: Number of rotary dimensions (defaults to ``head_size``).
            base: Optional base for frequency computation (defaults to ``self.rope_theta``).

        Returns:
            `ModuleCaches` containing the frequencies sharded with `NamedSharding`.
        """
        from easydel.layers.rotary_embedding import get_frequencies

        from .utils import ModuleCaches

        partial_rotary_factor = getattr(self, "partial_rotary_factor", 1.0)
        head_size = head_size or self.head_dim
        rotary_dim = rotary_dim or head_size
        rope_config = self._get_rope_config()

        frequencies = get_frequencies(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=self.granted_freq_max_position_embedding,
            base=base or self.rope_theta,
            rope_scaling=rope_config.to_dict(),
            partial_rotary_factor=partial_rotary_factor,
        ).astype(jnp.bfloat16)

        return ModuleCaches(jax.device_put(frequencies, Ns(self.mesh, Ps())))

    @staticmethod
    def _create_causal_mask(target_length):
        """Creates a causal attention mask for autoregressive models.

        Generates a lower triangular boolean mask that prevents attending
        to future tokens in the sequence.

        Args:
            target_length: The sequence length for the mask.

        Returns:
            4D boolean array with shape [1, 1, target_length, target_length].
        """
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
        """Get attention mask details for each layer.

        Retrieves layer-specific attention mask configurations, which is
        particularly useful for models with heterogeneous attention patterns
        (e.g., models using different attention types per layer like sliding
        window attention in some layers and full attention in others).

        Returns:
            dict[int, AttnMaskDetail] | None: A dictionary mapping layer indices
                to their corresponding AttnMaskDetail configurations, or None
                if the model doesn't define layer-specific mask types.
        """
        config = self.get_text_config()
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            from easydel.infra.utils import AttnMaskDetail, AttnMaskType

            mapping = {}
            for layer_idx, layer_type in enumerate(layer_types):
                mapping[layer_idx] = AttnMaskDetail(
                    mask_type=AttnMaskType.from_hf(layer_type),
                    size=getattr(config, "sliding_window", getattr(config, "sliding_windows", None)),
                    chunks=getattr(config, "attention_chunk_size", None),
                )
            return mapping
        return None

    def get_basic_causal_mask(self, *args, **kwargs):
        """Gets or creates the basic causal attention mask.

        Creates a causal mask for the maximum position embeddings and
        places it on the appropriate device with sharding.

        Returns:
            ModuleCaches containing the causal mask, or False if masks
            are not precomputed.
        """
        from .utils import ModuleCaches

        if self.precompute_masks is False:
            return False

        target_length = self.granted_mask_max_position_embedding

        return ModuleCaches(jax.device_put(self._create_causal_mask(target_length), Ns(self.mesh, Ps())))

    def get_fcm_mask(self, batch_size, seq_length, deterministic: bool):
        """Generates a Forgetful Causal Mask (FCM) for training.

        FCM randomly drops causal constraints during training to improve
        model robustness. Only applied in non-deterministic mode.

        Args:
            batch_size: Number of sequences in the batch.
            seq_length: Length of each sequence.
            deterministic: If True, returns None (no FCM applied).

        Returns:
            Boolean mask array or None if deterministic or FCM not configured.
        """
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

    @staticmethod
    def _fix_parent_kws(kw1, kw2):
        """Merge two keyword argument dictionaries, with kw1 taking precedence.

        Args:
            kw1: Primary dictionary (takes precedence).
            kw2: Secondary dictionary (provides defaults for missing keys).

        Returns:
            Merged dictionary with all keys from both inputs, kw1 values preferred.
        """
        result = copy.deepcopy(kw1)
        tkey = result.keys()
        for k, v in kw2.items():
            if k not in tkey:
                result[k] = v
        return result

    @staticmethod
    def _prefix_partition_rules(rules: tuple, prefix: str) -> tuple:
        """Add a prefix to all regex patterns in partition rules.

        Args:
            rules: Tuple of (regex_pattern, partition_spec) pairs.
            prefix: Prefix to add (e.g., "thinker/").

        Returns:
            New tuple with prefixed patterns.
        """
        prefixed = []
        for pattern, spec in rules:
            if pattern in (".*", r".*"):
                continue
            prefixed.append((f"{prefix}/{pattern}", spec))
        return tuple(prefixed)

    @property
    def partition_manager(self) -> PartitionManager:
        """Gets the partition manager for this configuration.

        The PartitionManager handles translation between logical axis names (like 'dp', 'tp')
        and their actual configurations in the device mesh. It provides utilities for
        resolving partition specifications and managing distributed execution.

        Returns:
            PartitionManager instance configured with this config's partition_axis.
            If partition_axis is None, creates a default PartitionAxis() first.

        Example:
            >>> config = EasyDeLBaseConfig()
            >>> pm = config.partition_manager
            >>> # Use partition manager to resolve sharding specs
            >>> spec = pm.resolve(axes=["dp", "tp"], mode="train", shape=(8, 1024))
        """
        if self.partition_axis is None:
            self.partition_axis = PartitionAxis()
        return PartitionManager(self.partition_axis)

    __hash__ = hash_fn


EasyDeLBaseConfig.__init__.__doc__ = EasyDeLBaseConfig.__doc__
EasyDeLBaseConfigDict.__doc__ = EasyDeLBaseConfig.__init__.__doc__
EasyDeLBaseConfigDict.__annotations__ = EasyDeLBaseConfig.__annotations__
