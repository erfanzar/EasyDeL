# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Operation-specific configuration classes.

This module defines configuration dataclasses for each attention operation,
providing type-safe, operation-specific parameters for kernel execution
and autotuning.

Configuration Hierarchy:
    BaseOperationConfig
    ├── FlashAttentionConfig (block sizes via FwdParams/BwdParams)
    ├── BlockSparseAttentionConfig (sparse patterns + block sizes)
    ├── NativeSparseAttentionConfig (block_q, block_k, block_d, block_size)
    ├── PageAttentionConfig (num_splits, pages_per_compute_block)
    ├── RingAttentionConfig (distributed attention)
    ├── FlashMLAConfig (block_q, block_k for MLA)
    ├── GroupedMatmulConfig (block_m, block_n, block_k for MoE)
    ├── QuantizedMatmulConfig (block_m, block_n, block_k for quantized matmul)
    ├── MeanPoolingConfig (block_size for pooling)
    ├── KernelDeltaAttentionConfig (linear attention)
    ├── RWKV4Config, RWKV6Config, RWKV7Config (recurrent models)
    └── StateSpaceV1Config, StateSpaceV2Config (Mamba-style SSMs)

Common Parameters:
    All configs inherit from BaseOperationConfig which provides:
        - platform: Target execution platform (triton/pallas/cuda/cute/xla/auto)
        - backend: Hardware backend (gpu/tpu/cpu/any)

Block Size Guidelines:
    - GPU (Triton): 64-256 for query/key blocks, 4-8 warps
    - TPU (Pallas): 128-256 for larger tiles to leverage matrix units
    - XLA: Block sizes are hints; XLA compiler optimizes automatically

Example:
    >>> from ejkernel.modules.operations import FlashAttention, FlashAttentionConfig
    >>> from ejkernel.ops import FwdParams
    >>>
    >>> # Custom configuration for flash attention
    >>> cfg = FlashAttentionConfig(
    ...     fwd_params=FwdParams(q_blocksize=128, kv_blocksize=128),
    ...     platform="triton",
    ... )
    >>> output = flash_attention(q, k, v, causal=True, cfg=cfg)
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from ejkernel.ops import BwdParams, FwdParams


def get_safe_hash_int(text, algorithm="md5"):
    """Generate a hash of text using specified algorithm with safety checks.

    Args:
        text: Input text to hash. Will be converted to string if not already.
        algorithm: Hash algorithm name (default: "md5"). Must be a valid hashlib algorithm.

    Returns:
        Integer representation of the hash digest.

    Raises:
        ValueError: If the hash algorithm is not supported.
        Exception: If any other error occurs during hash generation.
    """
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


def hash_fn(self) -> int:
    """Generate a hash for an object based on its dictionary values.

    This function creates a deterministic hash by concatenating string representations
    of numeric and collection-type attributes, then hashing the result.

    Args:
        self: The object instance to hash.

    Returns:
        Integer hash value derived from the object's attribute values.

    Note:
        Only includes float, int, bool, dict, list, and tuple attributes in
        the hash.
    """
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | bool | dict | list | tuple))
    return get_safe_hash_int(shu)


@dataclass
class BaseOperationConfig:
    """Base configuration for all operations.

    This is the parent class for all operation-specific configurations.
    It provides common platform and backend selection parameters that
    are inherited by all specialized configurations.

    Args:
        platform: Target execution platform for kernel dispatch.
            - "triton": Triton GPU kernels (NVIDIA/AMD GPUs)
            - "pallas": Pallas kernels (TPU/GPU)
            - "cuda": CUDA-specific implementations
            - "cute": CUTLASS CuTe DSL implementations
            - "xla": XLA compiler-based implementations (most portable)
            - "auto": Automatic platform selection based on hardware (default)
        backend: Target hardware backend specification.
            - "gpu": GPU-specific optimizations
            - "tpu": TPU-specific optimizations
            - "cpu": CPU-specific optimizations
            - "any": No backend preference (default)

    Note:
        When platform is "xla", the backend is automatically set to "any"
        since XLA handles backend selection internally.
    """

    platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] = "auto"
    backend: str = "any"

    __hash__ = hash_fn

    def to_dict(self) -> dict:
        """Serialize this config to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize a config from a plain dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize this config to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str):
        """Deserialize a config from a JSON string."""
        return cls.from_dict(json.loads(s))


@dataclass
class FlashAttentionConfig(BaseOperationConfig):
    """Configuration for Flash Attention operation.

    Args:
        fwd_params: Forward kernel parameters (uses `q_blocksize`/`kv_blocksize` for tiling).
        bwd_params: Backward kernel parameters (optional).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    fwd_params: FwdParams | None = None
    bwd_params: BwdParams | None = None

    def __post_init__(self):
        """Convert dict-typed forward/backward params to FwdParams/BwdParams."""
        if isinstance(self.fwd_params, dict):
            self.fwd_params = FwdParams(**self.fwd_params)
        if isinstance(self.bwd_params, dict):
            self.bwd_params = BwdParams(**self.bwd_params)

    __hash__ = hash_fn


@dataclass
class BlockSparseAttentionConfig(BaseOperationConfig):
    """Configuration for Block Sparse Attention operation.

    Args:
        fwd_params: Forward kernel parameters (q/kv block sizes, warps, stages).
        bwd_params: Backward kernel parameters (q/kv block sizes, warps, stages).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: "any").
    """

    fwd_params: FwdParams | None = None
    bwd_params: BwdParams | None = None

    def __post_init__(self):
        """Convert dict-typed forward/backward params to FwdParams/BwdParams."""
        if isinstance(self.fwd_params, dict):
            self.fwd_params = FwdParams(**self.fwd_params)
        if isinstance(self.bwd_params, dict):
            self.bwd_params = BwdParams(**self.bwd_params)

    __hash__ = hash_fn


@dataclass
class NativeSparseAttentionConfig(BaseOperationConfig):
    """Configuration for Native Sparse Attention operation.

    Args:
        block_q: Query block size (default: 64)
        block_k: Key block size (default: 64)
        block_d: Head dimension block size (default: 64)
        block_size: Size of attention blocks for sparsity (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 64
    block_k: int = 64
    block_d: int = 64
    block_size: int = 64
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class RecurrentAttentionConfig(BaseOperationConfig):
    """Configuration for Recurrent Attention operation.

    Args:
        block_q: Query block size (default: 64)
        block_k: Key block size (default: 64)
        block_d: Head dimension block size (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 64
    block_k: int = 64
    block_d: int = 64
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class RingAttentionConfig(BaseOperationConfig):
    """Configuration for Ring Attention operation.

    Args:
        fwd_params: Forward pass block size parameters
        bwd_params: Backward pass block size parameters
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    fwd_params: FwdParams | None = None
    bwd_params: BwdParams | None = None

    def __post_init__(self):
        """Convert dict-typed forward/backward params to FwdParams/BwdParams."""
        if isinstance(self.fwd_params, dict):
            self.fwd_params = FwdParams(**self.fwd_params)
        if isinstance(self.bwd_params, dict):
            self.bwd_params = BwdParams(**self.bwd_params)

    __hash__ = hash_fn


@dataclass
class PageAttentionConfig(BaseOperationConfig):
    """Configuration for Page Attention operation.

    Args:
        num_splits: Number of partitions for splitting contexts (default: 0 for auto)
        pages_per_compute_block: Pages per compute block (default: None)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    num_splits: int = 0
    pages_per_compute_block: int | None = None
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class UnifiedAttentionConfig(BaseOperationConfig):
    """Configuration for vLLM-style unified (paged) attention operation.

    Args:
        seq_threshold_3d: Threshold (in #seqs) for selecting the segmented 3D
            decode kernel on GPU (Triton only).
        num_par_softmax_segments: Number of parallel softmax segments used by
            the segmented 3D decode kernel (Triton only).
        block_dim: CUDA thread-block dimension for the CUDA backend.
        num_warps: Optional Triton kernel override.
        num_stages: Optional Triton kernel override.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    seq_threshold_3d: int | None = None
    num_par_softmax_segments: int | None = None
    block_dim: int = 128
    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn


@dataclass
class DecodeAttentionConfig(BaseOperationConfig):
    """Configuration for vLLM-style paged decode attention operation.

    Args:
        num_kv_splits: Number of KV splits used by the Triton kernel (default: 16).
        num_warps: Optional Triton kernel override.
        num_stages: Optional Triton kernel override.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    num_kv_splits: int = 16
    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn


@dataclass
class ChunkedPrefillPagedDecodeConfig(BaseOperationConfig):
    """Configuration for chunked prefill + paged decode attention operation.

    This op updates a block-tabled KV cache and then runs unified paged attention
    on the packed queries. On GPU, the underlying unified attention kernel may
    use a 2D or 3D grid depending on `seq_threshold_3d`.

    Args:
        seq_threshold_3d: Threshold (#seqs) for selecting 3D segmented decode kernel (Triton only).
        num_par_softmax_segments: Parallel softmax segments for 3D kernel (Triton only).
        num_warps: Optional Triton kernel override.
        num_stages: Optional Triton kernel override.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    seq_threshold_3d: int | None = None
    num_par_softmax_segments: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn


@dataclass
class AttentionConfig(BaseOperationConfig):
    """Configuration for basic Attention operation.

    Args:
        block_q: Query block size for the FlashAttention path (default: 128).
        block_k: Key block size for the FlashAttention path (default: 128).
        weights_block_q: Q-tile size for the dense ``(B, H, Sq, Sk)``
            attention-weights kernel (default: 64). The operation picks
            this from seq_len via ``heuristic_cfg`` — the kernel does not.
        weights_block_k: K-tile size for the dense attention-weights kernel
            (default: 64). Same heuristic policy as ``weights_block_q``.
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 2)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 128
    block_k: int = 128
    weights_block_q: int = 64
    weights_block_k: int = 64
    num_warps: int = 4
    num_stages: int = 2

    __hash__ = hash_fn


@dataclass
class GroupedMatmulConfig(BaseOperationConfig):
    """Configuration for Grouped Matrix Multiplication operation.

    Args:
        block_m: M dimension tile size (default: 128).
        block_n: N dimension tile size (default: 128).
        block_k: K (contraction) dimension tile size (default: 128).
        num_warps: Number of warps for Triton kernels (default: 4).
            Pass ``None`` to let the backend choose.
        num_stages: Number of pipeline stages (default: 2).
            Pass ``None`` to let the backend choose.
        bypass_xla_tiling: If True and the resolved platform is XLA, skip
            the tiling computation and pass ``tiling=None`` to the XLA impl
            (default: False).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: "any").
    """

    block_m: int = 128
    block_n: int = 128
    block_k: int = 128
    num_warps: int = 4
    num_stages: int = 2
    bypass_xla_tiling: bool = False

    __hash__ = hash_fn


@dataclass
class AllGatherMatmulConfig(BaseOperationConfig):
    """Configuration for the fused All-Gather + Matmul operation.

    Block sizes are silently clamped to the largest divisor of the actual
    matrix dimension that is still ``<=`` the requested value, so they need
    not exactly divide the problem size.

    Args:
        block_n: N-dimension block size for RHS tiling (default: 128).
        block_k: K-dimension (contraction) block size (default: 128).
        num_warps: Number of warps for Triton kernels (default: 4).
            Unused in the XLA/Pallas path.
        num_stages: Pipeline stages for Triton kernels (default: 2).
            Unused in the XLA/Pallas path.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: "any").
    """

    block_n: int = 128
    block_k: int = 128
    num_warps: int = 4
    num_stages: int = 2

    __hash__ = hash_fn


@dataclass
class ReduceScatterMatmulConfig(BaseOperationConfig):
    """Configuration for the fused Matmul + Reduce-Scatter operation.

    Block sizes are silently clamped to the largest divisor of the actual
    matrix dimension that is still ``<=`` the requested value.

    Args:
        block_m: M-dimension block size (default: 128).
        block_n: N-dimension block size (default: 128).
        block_k: K-dimension (contraction) block size (default: 128).
        num_warps: Number of warps for Triton kernels (default: 4).
            Unused in the XLA/Pallas path.
        num_stages: Pipeline stages for Triton kernels (default: 2).
            Unused in the XLA/Pallas path.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: "any").
    """

    block_m: int = 128
    block_n: int = 128
    block_k: int = 128
    num_warps: int = 4
    num_stages: int = 2

    __hash__ = hash_fn


@dataclass
class QuantizedMatmulConfig(BaseOperationConfig):
    """Configuration for Quantized Matrix Multiplication operation.

    Args:
        block_m: M dimension block size (default: 128)
        block_n: N dimension block size (default: 128)
        block_k: K dimension block size (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 3)
        use_bf16: Use BF16 for the dot input tiles (default: True)
        split_k: Optional split-K factor for Triton kernels (default: None)
        tpu_path: Optional TPU Pallas path override.
            TPU Pallas uses packed-only execution. Legacy values
            ("hybrid", "predecode") are accepted and normalized to packed.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_m: int = 128
    block_n: int = 128
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 3
    use_bf16: bool = True
    split_k: int | None = None
    tpu_path: Literal["packed"] | None = None

    __hash__ = hash_fn


@dataclass
class MeanPoolingConfig(BaseOperationConfig):
    """Configuration for Mean Pooling operation.

    Args:
        block_size: Block size for pooling (default: 64)
        block_dim: Triton hidden-dimension tile size (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_size: int = 64
    block_dim: int = 64
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class RaggedDecodeAttentionConfig(BaseOperationConfig):
    """Configuration for Ragged Decode Attention operation.

    The only tunable block-size parameter is ``fwd_params``, which passes
    ``q_blocksize`` and ``kv_blocksize`` to the underlying kernel.  Triton
    ``num_warps``/``num_stages`` are embedded in ``FwdParams``; there are no
    separate top-level fields for those.

    Args:
        fwd_params: Forward-pass tiling parameters (``FwdParams`` with
            ``q_blocksize``, ``kv_blocksize``, ``num_warps``, ``num_stages``).
            Defaults to ``None`` (kernel selects defaults).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: "any").
    """

    fwd_params: FwdParams | None = None

    def __post_init__(self):
        """Convert dict-typed forward params to FwdParams."""
        if isinstance(self.fwd_params, dict):
            self.fwd_params = FwdParams(**self.fwd_params)

    __hash__ = hash_fn


@dataclass
class RaggedPageAttentionv2Config(BaseOperationConfig):
    """Configuration for Ragged Page Attention operation.

    Args:
        num_kv_pages_per_block: Number of KV pages to process per compute block (default: None for auto)
        num_queries_per_block: Number of queries to process per compute block (default: None for auto)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    num_kv_pages_per_block: int | None = None
    num_queries_per_block: int | None = None
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class RaggedPageAttentionv2TurboQuantConfig(BaseOperationConfig):
    """Configuration for Ragged Page Attention v2 with TurboQuant compression.

    Controls the tiling strategy for the read-only TurboQuant-compressed
    paged attention kernel.  The two block-size parameters determine how
    queries and KV pages are grouped during the attention computation:

    * Larger ``num_kv_pages_per_block`` reduces loop overhead but increases
      per-iteration memory.
    * Larger ``num_queries_per_block`` amortises the cost of pre-rotating
      and pre-projecting queries across more tokens.

    When either is ``None``, the kernel selects a default based on the
    total number of pages per sequence.

    Attributes:
        num_kv_pages_per_block: Number of KV pages to process per block
            in the inner attention loop.  ``None`` for automatic selection.
        num_queries_per_block: Number of query tokens to process per block.
            ``None`` defaults to 8.
        num_warps: Kept for API parity with backend kernels (unused on XLA).
        num_stages: Kept for API parity with backend kernels (unused on XLA).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: ``"any"``).
    """

    num_kv_pages_per_block: int | None = None
    num_queries_per_block: int | None = None
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class RaggedPageAttentionv3Config(BaseOperationConfig):
    """Configuration for Ragged Page Attention operation.

    Args:
        num_kv_pages_per_block: Number of KV pages to process per compute block (default: None for auto)
        num_queries_per_block: Number of queries to process per compute block (default: None for auto)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    chunk_prefill_size: int | None = None
    num_kv_pages_per_block: int | None = None
    num_queries_per_block: int | None = None
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class RaggedPageAttentionv3TurboQuantConfig(BaseOperationConfig):
    """Configuration for Ragged Page Attention v3 with TurboQuant compression.

    Controls the tiling strategy for the fused compress-and-attend
    TurboQuant paged attention kernel.  This extends the v2 config with
    an additional ``chunk_prefill_size`` parameter for prefill workloads.

    The two block-size parameters determine how queries and KV pages are
    grouped during both the compression phase and the attention computation:

    * Larger ``num_kv_pages_per_block`` reduces loop overhead but increases
      per-iteration memory.
    * Larger ``num_queries_per_block`` amortises pre-rotation/pre-projection
      cost and batches more tokens in the KV-write phase.

    When either is ``None``, the kernel selects a default based on the
    total number of pages per sequence.

    Attributes:
        chunk_prefill_size: Optional chunk size for prefill workloads.
            Accepted for API parity with the standard RPA v3 config;
            currently unused in the XLA TurboQuant backend.
        num_kv_pages_per_block: Number of KV pages to process per block
            in the inner attention loop.  ``None`` for automatic selection.
        num_queries_per_block: Number of query tokens to process per block.
            ``None`` defaults to 8.
        num_warps: Kept for API parity with backend kernels (unused on XLA).
        num_stages: Kept for API parity with backend kernels (unused on XLA).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: ``"any"``).
    """

    chunk_prefill_size: int | None = None
    num_kv_pages_per_block: int | None = None
    num_queries_per_block: int | None = None
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class MultiLatentRaggedPageAttentionConfig(BaseOperationConfig):
    """Configuration for Multi-Latent Ragged Page Attention.

    Args:
        chunk_prefill_size: Optional chunk size for prefill sequences.
        num_kv_pages_per_block: KV pages per kernel block (None = auto).
        num_queries_per_block: Queries per kernel block (None = auto).
        vmem_limit_bytes: Optional TPU VMEM hint for Pallas backend.
        num_warps: Kept for API consistency (unused on XLA/Pallas paths).
        num_stages: Kept for API consistency (unused on XLA/Pallas paths).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: \"any\").
    """

    chunk_prefill_size: int | None = None
    num_kv_pages_per_block: int | None = None
    num_queries_per_block: int | None = None
    vmem_limit_bytes: int | None = None
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class MultiLatentRaggedPageAttentionV2Config(BaseOperationConfig):
    """Configuration for Multi-Latent Ragged Page Attention v2.

    Args:
        chunk_prefill_size: Optional chunk size for prefill sequences.
        num_kv_pages_per_block: KV pages per kernel block. May be a single
            int or a `(decode, prefill, mixed)` tuple.
        num_queries_per_block: Queries per kernel block. May be a single int
            or a `(decode, prefill, mixed)` tuple.
        vmem_limit_bytes: Optional TPU VMEM hint for Pallas backend.
        num_warps: Kept for API consistency (unused on XLA/Pallas paths).
        num_stages: Kept for API consistency (unused on XLA/Pallas paths).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: "any").
    """

    chunk_prefill_size: int | None = None
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None
    num_queries_per_block: tuple[int, int, int] | int | None = None
    vmem_limit_bytes: int | None = None
    num_warps: int = 4
    num_stages: int = 1

    def __post_init__(self):
        """Normalize list-based JSON payloads back into tuples."""

        def _normalize(
            value: tuple[int, int, int] | list[int] | int | None,
            field_name: str,
        ) -> tuple[int, int, int] | int | None:
            """Convert JSON-deserialized lists back to tuples and validate length.

            Args:
                value: Block-size value — may be a tuple, list, int, or None.
                field_name: Config field name (for error messages).

            Returns:
                Validated value as tuple, int, or None.

            Raises:
                ValueError: If a tuple/list with length != 3 is provided.
            """
            if isinstance(value, list):
                value = tuple(value)
            if isinstance(value, tuple) and len(value) != 3:
                raise ValueError(f"{field_name} must have exactly three entries.")
            return value

        self.num_kv_pages_per_block = _normalize(
            self.num_kv_pages_per_block,
            "num_kv_pages_per_block",
        )
        self.num_queries_per_block = _normalize(
            self.num_queries_per_block,
            "num_queries_per_block",
        )

    __hash__ = hash_fn


@dataclass
class GLAttentionConfig(BaseOperationConfig):
    """Configuration for Gated Linear Attention operation.

    Args:
        block_q: Query block size (default: 64)
        block_k: Key block size (default: 64)
        block_d: Head dimension block size (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 64
    block_k: int = 64
    block_d: int = 64
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class LightningAttentionConfig(BaseOperationConfig):
    """Configuration for Lightning Attention operation.

    Args:
        block_q: Query block size (default: 64)
        block_k: Key block size (default: 64)
        block_d: Head dimension block size (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 64
    block_k: int = 64
    block_d: int = 64
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class KernelDeltaAttentionConfig(BaseOperationConfig):
    """Configuration for Kernel Delta Attention (KDA) operation.

    Args:
        chunk_size: Chunk size for the XLA chunked recurrence path. TileLang
            uses a recurrent native scan and treats this as a compatibility hint.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    chunk_size: int = 64

    __hash__ = hash_fn


@dataclass
class GatedDeltaRuleConfig(BaseOperationConfig):
    """Configuration for Gated Delta Rule (GDR) operation.

    The primary tunable parameter is ``chunk_size``, which controls the
    trade-off between intra-chunk parallelism and inter-chunk state
    propagation overhead.

    Args:
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
        chunk_size: Chunk size for the chunked forward pass (default: 64).
            Larger chunks increase parallelism but also increase memory for
            intra-chunk attention matrices. Typical values: 32, 64, 128.
    """

    chunk_size: int = 64

    __hash__ = hash_fn


@dataclass
class RaggedGatedDeltaRuleConfig(BaseOperationConfig):
    """Configuration for Ragged Gated Delta Rule operation.

    Processes variable-length sequences packed into a flat token stream
    with chunked parallel intra-chunk computation.

    Args:
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
        chunk_size: Chunk size for the chunked prefill path (default: 64).
    """

    chunk_size: int = 64

    __hash__ = hash_fn


@dataclass
class RWKV4Config(BaseOperationConfig):
    """Configuration for RWKV-4 recurrence operation.

    RWKV-4 uses a linear attention mechanism with time-decay factors and
    gating for efficient language modeling without quadratic complexity.

    The core operation computes:
        wkv_t = sum_{i<t} exp(-w*(t-i)) * k_i * v_i + u * k_t * v_t

    Args:
        block_c: Channel-axis tile size for the tile-lang backend
            (default ``64``). Ignored by the XLA backend. The
            operation's ``heuristic_cfg`` picks this from channel
            count; the autotuner enumerates a small set of values.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_c: int = 64

    __hash__ = hash_fn


@dataclass
class RWKV6Config(BaseOperationConfig):
    """Configuration for RWKV-6 recurrence operation.

    RWKV-6 extends RWKV-4 with token-dependent decay factors (w parameter),
    allowing the model to learn how long to retain information per token.

    Key improvements over RWKV-4:
        - Token-dependent time decay (learned per-position)
        - Improved receptance gating mechanism

    Args:
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")

    Note:
        This operation currently uses XLA implementation without
        tunable block sizes.
    """

    pass

    __hash__ = hash_fn


@dataclass
class RWKV7Config(BaseOperationConfig):
    """Configuration for RWKV-7 recurrence operation.

    RWKV-7 introduces additional improvements to the RWKV architecture
    with enhanced state update mechanisms and gating operations.

    Args:
        block_v: Value-dimension tile size for Triton kernels.
        num_warps: Number of warps for Triton kernels.
        num_stages: Number of pipeline stages for Triton kernels.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_v: int = 64
    num_warps: int = 4
    num_stages: int = 3

    __hash__ = hash_fn


@dataclass
class RWKV7MulConfig(BaseOperationConfig):
    """Configuration for RWKV-7 multiplicative recurrence operation.

    This variant of RWKV-7 uses multiplicative state updates instead
    of additive updates, providing different expressiveness characteristics
    for sequence modeling tasks.

    Args:
        block_v: Value-dimension tile size for Triton kernels.
        num_warps: Number of warps for Triton kernels.
        num_stages: Number of pipeline stages for Triton kernels.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_v: int = 64
    num_warps: int = 4
    num_stages: int = 3

    __hash__ = hash_fn


@dataclass
class FlashMLAConfig(BaseOperationConfig):
    """Configuration for Flash Multi-head Latent Attention operation.

    Args:
        block_q: Query block size (default: 128)
        block_k: Key block size (default: 128)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 2)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 128
    block_k: int = 128
    num_warps: int = 4
    num_stages: int = 2

    __hash__ = hash_fn


@dataclass
class DeepSeekAttentionConfig(BaseOperationConfig):
    """Configuration for DeepSeek Sparse Attention (DSA) operation.

    DSA combines MLA attention with a Lightning Indexer that dynamically
    selects top-k KV tokens per query, reducing complexity from O(L^2) to O(L*k).

    Args:
        index_topk: Number of KV tokens to select per query (default: 2048).
        block_q: Query block size for the inner FlashAttention call (default: 128).
        block_k: Key block size for the inner FlashAttention call (default: 128).
        gemm_block: GEMM/indexer/topk tile size shared by the DSA-internal
            tile-lang sub-kernels (indexer, KV-recon GEMMs, elementwise
            add/cast, top-k bias). The operation's ``heuristic_cfg`` picks
            this from seq_len; the autotuner enumerates ``{64, 128}``.
            Default ``128`` is the cold-start fallback for direct callers.
        num_warps: Number of warps for Triton kernels (default: 4).
        num_stages: Number of pipeline stages (default: 2).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto).
        backend: Backend specification (default: "any").

    Reference:
        DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
    """

    index_topk: int = 2048
    block_q: int = 128
    block_k: int = 128
    gemm_block: int = 128
    num_warps: int = 4
    num_stages: int = 2

    __hash__ = hash_fn


@dataclass
class ScaledDotProductAttentionConfig(BaseOperationConfig):
    """Configuration for Scaled Dot Product Attention operation.

    Args:
        block_q: Query tile size for tile-lang FlashAttention-backed SDPA.
        block_k: Key/value tile size for tile-lang FlashAttention-backed SDPA.
        num_warps: GPU warp-count hint when the selected backend uses it.
        num_stages: Pipeline-stage hint when the selected backend uses it.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 128
    block_k: int = 128
    num_warps: int = 4
    num_stages: int = 2

    __hash__ = hash_fn


@dataclass
class PrefillPageAttentionConfig(BaseOperationConfig):
    """Configuration for Prefill Page Attention operation.

    Args:
        block_k: KV-token tile size for TileLang prefill paged attention.
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_k: int | None = None
    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class StateSpaceV1Config(BaseOperationConfig):
    """Configuration for SSM1 (Mamba1-style) Selective State Space operation.

    Args:
        block_d: ``D``-axis tile size for the tile-lang SSM-1 scan kernel
            (default ``64``). Ignored by XLA. The operation picks this from
            shape via ``heuristic_cfg`` — the kernel does not.
        block_e: Width-axis tile size for the gating helper (``silu_gate``)
            applied after the scan (default ``128``).
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_d: int = 64
    block_e: int = 128

    __hash__ = hash_fn


@dataclass
class StateSpaceV2Config(BaseOperationConfig):
    """Configuration for SSM2 (Mamba2-style) Selective State Space operation.

    Args:
        n_groups: Number of groups for B, C parameters (default: 1)
        use_gated_rmsnorm: Whether to use gated RMSNorm for output (default: False)
        rmsnorm_eps: Epsilon for RMSNorm stability (default: 1e-5)
        block_e: Width-axis tile size for the silu gate helper applied after
            the scan (default ``128``). Ignored when ``use_gated_rmsnorm``.
        platform: Target platform (triton/pallas/cuda/cute/xla/auto)
        backend: Backend specification (default: "any")
    """

    n_groups: int = 1
    use_gated_rmsnorm: bool = False
    rmsnorm_eps: float = 1e-5
    block_e: int = 128

    __hash__ = hash_fn


@dataclass
class FusedCrossEntropyConfig(BaseOperationConfig):
    """Configuration for fused cross-entropy.

    Args:
        block_v: Tile size along the **vocab** axis for the inner
            online-softmax scan (default ``0`` = auto: 256 / 512 /
            1024 / 2048 depending on ``V``). Each CTA streams its
            row(s) through ``ceildiv(V, block_v)`` chunks; larger
            ``block_v`` amortises chunk-loop overhead but uses more
            SMEM / registers per CTA.
        block_m: Rows per CTA (``M`` = flattened ``batch · seq``).
            Default ``0`` = auto: ``1`` when ``N < 1024`` else ``4``.
            ``block_m=1`` maximises SM occupancy (smallest SMEM
            footprint, preferred for wide ``V``); larger amortises
            per-CTA fixed cost when ``N`` is huge but burns more
            registers / SMEM per CTA.
        num_warps: Number of warps (kernel-only hint).
        num_stages: Pipeline depth (kernel-only hint).
        platform: Target platform (tilelang/xla/auto).
        backend: Backend specification.
    """

    block_v: int = 0
    block_m: int = 0
    num_warps: int = 4
    num_stages: int = 2

    __hash__ = hash_fn


@dataclass
class FusedKLDivergenceConfig(BaseOperationConfig):
    """Configuration for fused KL divergence (forward / reverse / JSD).

    Args:
        block_v: Tile size along the **vocab** axis for the inner
            online-softmax scan (default ``0`` = auto: 256-4096 by
            ``V``). Larger ``block_v`` amortises chunk-loop overhead;
            smaller eases register / SMEM pressure for huge ``V``.
        block_m: Rows per CTA (``M`` = flattened ``batch · seq``).
            Default ``0`` = auto: ``1`` when ``N < 1024`` else ``4``.
            ``block_m=1`` is preferred for wide vocabularies — keeps
            occupancy high; larger ``block_m`` amortises fixed cost
            on small-V/big-N shapes.
        num_warps: Number of warps (kernel-only hint).
        num_stages: Pipeline depth (kernel-only hint).
        platform: Target platform (tilelang/xla/auto).
        backend: Backend specification.
    """

    block_v: int = 0
    block_m: int = 0
    num_warps: int = 4
    num_stages: int = 2

    __hash__ = hash_fn
