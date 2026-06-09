# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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


"""
This module defines common types, constants, and named tuples used across the
eformer library, particularly for JAX and sharding configurations.
"""

import typing as tp

import jax
import numpy as np
from jax import numpy as jnp


class _Empty:
    """Sentinel class indicating an unset or missing value.

    This class serves as a sentinel type to distinguish between:
    - A value that was explicitly set to None
    - A value that was never provided

    The singleton instance `NOT_GIVEN` should be used for comparisons.

    Example:
        >>> from eformer.common_types import NOT_GIVEN, EMPTY_VAL
        >>>
        >>> def func(value=NOT_GIVEN):
        ...     if isinstance(value, EMPTY_VAL):
        ...         print("Value was not provided")
        ...     elif value is None:
        ...         print("Value was explicitly set to None")
    """

    pass


Array = jnp.ndarray
"""Type alias for JAX arrays."""
PRNGKey = jnp.ndarray
"""Type alias for JAX PRNG keys."""
DType = jnp.dtype
"""Type alias for JAX data types."""
Shape = tp.Sequence[int]
"""Type alias for array shapes."""

Mesh = jax.sharding.Mesh
"""Type alias for JAX Mesh objects."""

AxisNames = tuple[str, ...]
"""Type alias for a tuple of mesh axis names."""
AxisIdxes = tuple[int, ...]
"""Type alias for a tuple of axis indices."""
AxisType = tuple[str, ...] | str | tp.Any | None
"""
Type alias for a mesh axis specification.

Can be a single string (axis name), a tuple of strings, None (for no sharding),
or potentially other types depending on context (though typically str or tuple[str, ...]).
"""


EMPTY = "_"
BATCH = "__BATCH__"
"""Semantic axis name for the batch dimension."""
LENGTH = "__LENGTH__"
"""Semantic axis name for the sequence length dimension (general)."""
KV_LENGTH = "__KV_LENGTH__"
"""Semantic axis name for the key/value sequence length dimension."""
QUERY_LENGTH = "__QUERY_LENGTH__"
"""Semantic axis name for the query sequence length dimension."""
EMBED = "__EMBED__"
"""Semantic axis name for the embedding or hidden state dimension."""
HEAD = "__HEAD__"
KV_HEAD = "__KV_HEAD__"
"""Semantic axis name for the attention head dimension."""
MLP_INTERMEDIATE = "__MLP_INTERMEDIATE__"
"""Semantic axis name for the intermediate dimension in MLP layers."""
VOCAB = "__VOCAB__"
"""Semantic axis name for the vocabulary dimension."""
EXPERT = "__EXPERT__"
"""Semantic axis name for the expert dimension in Mixture-of-Experts models."""
EXPERT_GATE = "__EXPERT_GATE__"
"""Semantic axis name for the expert gate dimension in Mixture-of-Experts models."""
HEAD_DIM = "__HEAD_DIM__"
KV_HEAD_DIM = "__KV_HEAD_DIM__"
"""Semantic axis name for the dimension within each attention head."""
BIAS_HEAD_SEQ = "__BIAS_HEAD_SEQ__"
"""Semantic axis name for bias related to head and sequence dimensions."""
BIAS_KV_SEQ = "__BIAS_KV_SEQ__"
"""Semantic axis name for bias related to key/value sequence dimensions."""


DATA_PARALLEL = "__DATA_PARALLEL__"
FULLY_SHARDED_DATA_PARALLEL = "__FULLY_SHARDED_DATA_PARALLEL__"
TENSOR_PARALLEL = "__TENSOR_PARALLEL__"
EXPERT_PARALLEL = "__EXPERT_PARALLEL__"
SEQUENCE_PARALLEL = "__SEQUENCE_PARALLEL__"

DP = DATA_PARALLEL
FSDP = FULLY_SHARDED_DATA_PARALLEL
TP = TENSOR_PARALLEL
EP = EXPERT_PARALLEL
SP = SEQUENCE_PARALLEL


MODE_DECODE = "__autoregressive__"
"""Runtime mode for autoregressive decoding."""
MODE_PREFILL = "__prefill__"
"""Runtime mode for prefilling the cache."""
MODE_TRAIN = "__train__"
"""Runtime mode for training."""
MODE_INSERT = "__insert__"
"""Runtime mode for inserting into the cache."""


GENERATION_MODES = {
    MODE_DECODE,
    MODE_INSERT,
}
"""Set of runtime modes considered as generation modes."""


RUNTIME_MODE_TYPES = tp.Literal[
    MODE_DECODE,
    MODE_PREFILL,
    MODE_TRAIN,
    MODE_INSERT,
]
"""Type alias for the possible runtime modes."""


class DynamicShardingAxes(tp.NamedTuple):
    """
    A NamedTuple to define sharding axes and mode dynamically.

    Used to specify sharding based on the runtime mode or other dynamic factors.

    Attributes:
        axes: A sequence of semantic axis names or None.
        mode: The runtime mode (string constant) or an integer representing
              the dimension index to check for mode inference.
    """

    axes: tp.Sequence[str | None]
    mode: RUNTIME_MODE_TYPES | int  # type:ignore


class HiddenStateSharding(DynamicShardingAxes):
    """Dynamic sharding specification for hidden states."""

    axes: tp.ClassVar = [BATCH, QUERY_LENGTH, EMBED]
    mode: tp.ClassVar = 1


class AttnQSharding(DynamicShardingAxes):
    """Dynamic sharding specification for attention queries."""

    axes: tp.ClassVar = [BATCH, QUERY_LENGTH, HEAD, HEAD_DIM]
    mode: tp.ClassVar = 1


class AttnKVSharding(DynamicShardingAxes):
    """Dynamic sharding specification for attention keys/values."""

    axes: tp.ClassVar = [BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM]
    mode: tp.ClassVar = 1


class RowWise(DynamicShardingAxes):
    """Dynamic sharding specification for Row Wise sharding."""

    axes: tp.ClassVar = [TP, [FSDP, SP]]
    mode: tp.ClassVar = MODE_TRAIN


class SRowWise(DynamicShardingAxes):
    """Dynamic sharding specification for Row Wise sharding."""

    axes: tp.ClassVar = [TP]
    mode: tp.ClassVar = MODE_TRAIN


class ColumnWise(DynamicShardingAxes):
    """Dynamic sharding specification for Column Wise sharding."""

    axes: tp.ClassVar = [[FSDP, SP], TP]
    mode: tp.ClassVar = MODE_TRAIN


class SColumnWise(DynamicShardingAxes):
    """Dynamic sharding specification for Column Wise sharding."""

    axes: tp.ClassVar = [[FSDP, SP]]
    mode = MODE_TRAIN


class Replicated(DynamicShardingAxes):
    """Dynamic sharding specification for Column Wise sharding."""

    axes: tp.ClassVar = [EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertColumnWise(DynamicShardingAxes):
    """
    Dynamic sharding specification for Column Wise sharding.

    For a typical expert layer weight tensor of shape [num_experts, hidden_size, intermediate_size]:
    - Dimension 0 (num_experts): Shard across EP (expert parallel)
    - Dimension 1 (hidden_size): Shard across FSDP (parameter sharding)
    - Dimension 2 (intermediate_size): Shard across TP (tensor parallel - column-wise)

    DP is used for batch dimension in activations, SP for sequence length.
    """

    axes: tp.ClassVar = [EP, FSDP, TP]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertRowWise(DynamicShardingAxes):
    """
    Dynamic sharding specification for Row Wise sharding.

    For a typical expert layer weight tensor of shape [num_experts, intermediate_size, hidden_size]:
    - Dimension 0 (num_experts): Shard across EP (expert parallel)
    - Dimension 1 (intermediate_size): Shard across TP (tensor parallel - row-wise)
    - Dimension 2 (hidden_size): Shard across FSDP (parameter sharding)

    DP is used for batch dimension in activations, SP for sequence length.
    """

    axes: tp.ClassVar = [EP, TP, FSDP]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertColumnWiseAlt(DynamicShardingAxes):
    """
    Alternative column-wise sharding using SP for sequence-related parameters.
    Use this if your expert weights have a sequence-related dimension.
    """

    axes: tp.ClassVar = [EP, [FSDP, SP], TP]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertRowWiseAlt(DynamicShardingAxes):
    """
    Alternative row-wise sharding using SP for sequence-related parameters.
    Use this if your expert weights have a sequence-related dimension.
    """

    axes: tp.ClassVar = [EP, TP, [FSDP, SP]]
    mode: tp.ClassVar = MODE_TRAIN


class UnifiedExpertColumnWise(DynamicShardingAxes):
    """
    Unified column-wise sharding using SP for sequence-related parameters.
    Use this if your expert weights have a sequence-related dimension.
    """

    axes: tp.ClassVar = [[FSDP, SP, EP], EMPTY, TP]
    mode: tp.ClassVar = MODE_TRAIN


class UnifiedExpertRowWise(DynamicShardingAxes):
    """
    Unified row-wise sharding using SP for sequence-related parameters.
    Use this if your expert weights have a sequence-related dimension.
    """

    axes: tp.ClassVar = [[FSDP, SP, EP], TP, EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertActivations(DynamicShardingAxes):
    """
    Sharding for expert activation tensors of shape [batch, sequence, num_experts, hidden].
    - Batch dimension: DP (data parallel)
    - Sequence dimension: SP (sequence parallel)
    - Expert dimension: EP (expert parallel)
    - Hidden dimension: TP (tensor parallel) or FSDP
    """

    axes: tp.ClassVar = [DP, SP, EP, TP]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertActivationsAlt(DynamicShardingAxes):
    """
    Alternative activation sharding for shape [batch, sequence, hidden].
    When experts are already selected/routed.
    """

    axes: tp.ClassVar = [DP, SP, [TP, FSDP]]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertTensorParallel(DynamicShardingAxes):
    """Expert Tensor Parallelism (EPxTP) sharding axes."""

    axes: tp.ClassVar = [TP, EMPTY, EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
"""Default value used for masking, typically in attention mechanisms."""
NOT_GIVEN = _Empty()
EMPTY_VAL = _Empty
"""Sentinel value indicating that a parameter was not explicitly provided."""
