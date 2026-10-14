# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared semantic sharding tokens used by SpectraX and EasyDeL.

This module is the single source of truth for the *symbolic* axis names
that flow through SpectraX's sharding system (e.g. ``BATCH``,
``QUERY_LENGTH``, ``EMBED``, ``HEAD_DIM``, …) and for the
parallelism *role* tokens (``DP``/``FSDP``/``TP``/``EP``/``SP``) that
:class:`spectrax.sharding.PartitionManager` resolves to physical mesh
axes at sharding time.

Why "symbolic" tokens? Layer authors do not know whether a tensor
will eventually live on a 1-D, 2-D, or 4-D physical mesh — they only
know that *this* axis is a batch axis, *that* one is the embedding
axis, and so on. By tagging tensors with these tokens (via
:class:`DynamicShardingAxes` subclasses such as :class:`HiddenStateSharding`
or :class:`ColumnWise`), the sharding decision can be deferred until
runtime, when a :class:`~spectrax.sharding.PartitionManager` looks up
the active runtime mode (``MODE_TRAIN`` / ``MODE_DECODE`` / …) and
maps the symbolic axes to a concrete ``PartitionSpec``.

Public surface:

* **Type aliases** — ``Array`` (``jnp.ndarray``), ``PRNGKey``,
  ``DType``, ``Shape``, ``Mesh`` (``jax.sharding.Mesh``),
  ``AxisNames``, ``AxisIdxes``, ``AxisType``.
* **Tensor-axis tokens** — ``BATCH``, ``LENGTH``, ``KV_LENGTH``,
  ``QUERY_LENGTH``, ``EMBED``, ``HEAD``, ``KV_HEAD``,
  ``HEAD_DIM``, ``KV_HEAD_DIM``, ``MLP_INTERMEDIATE``, ``VOCAB``,
  ``EXPERT``, ``EXPERT_GATE``, ``BIAS_HEAD_SEQ``, ``BIAS_KV_SEQ``,
  and ``EMPTY`` (token marking a replicated dim).
* **Parallelism tokens** — ``PIPELINE_PARALLEL``/ ``PP``, ``DATA_PARALLEL``/``DP``,
  ``FULLY_SHARDED_DATA_PARALLEL``/``FSDP``, ``TENSOR_PARALLEL``/``TP``,
  ``EXPERT_PARALLEL``/``EP``, ``SEQUENCE_PARALLEL``/``SP``.
* **Runtime modes** — ``MODE_TRAIN``, ``MODE_PREFILL``,
  ``MODE_DECODE``, ``MODE_INSERT`` and the membership set
  ``GENERATION_MODES`` (modes that do KV-cache reads/writes).
* **Pre-baked sharding shapes** — :class:`HiddenStateSharding`,
  :class:`AttnQSharding`, :class:`AttnKVSharding`,
  :class:`RowWise`, :class:`SRowWise`, :class:`ColumnWise`,
  :class:`SColumnWise`, :class:`Replicated`, and the
  ``Expert*`` family for MoE layers.
* **Sentinels** — ``NOT_GIVEN`` (a single :class:`_Empty` instance,
  compared with ``is``) and ``EMPTY_VAL`` (the class itself, kept
  for ``isinstance`` checks).
* **Numeric constants** — ``DEFAULT_MASK_VALUE``, the standard fill
  for masked attention logits (``-0.7 x float32_max``, chosen to
  avoid NaN-on-fully-masked-row while staying well clear of overflow).
"""

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from jax import numpy as jnp


class _Empty:
    """Sentinel singleton type used to disambiguate "not provided" from ``None``.

    Many SpectraX APIs accept ``None`` as a meaningful value (e.g. "no
    mask", "no bias"), so a separate sentinel is needed to express
    "the caller did not pass this argument". An instance of this class
    is exported as ``NOT_GIVEN``.

    The class deliberately defines no equality, so identity comparison
    (``x is NOT_GIVEN``) is the supported check. ``repr`` yields the
    string ``"NOT_GIVEN"`` so the sentinel renders sensibly in error
    messages and tracebacks.
    """

    def __repr__(self) -> str:
        """Return the literal string ``"NOT_GIVEN"``.

        Overridden so error messages and tracebacks render the sentinel
        as ``NOT_GIVEN`` rather than ``<_Empty object at 0x...>``.

        Returns:
            The string ``"NOT_GIVEN"``.
        """
        return "NOT_GIVEN"


Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = jnp.dtype
Shape = tp.Sequence[int]
Mesh = jax.sharding.Mesh
AxisNames = tuple[str, ...]
AxisIdxes = tuple[int, ...]
AxisType = tuple[str, ...] | str | _Empty | None

EMPTY: tp.Final = "_"
BATCH: tp.Final = "__BATCH__"
LENGTH: tp.Final = "__LENGTH__"
KV_LENGTH: tp.Final = "__KV_LENGTH__"
QUERY_LENGTH: tp.Final = "__QUERY_LENGTH__"
EMBED: tp.Final = "__EMBED__"
HEAD: tp.Final = "__HEAD__"
KV_HEAD: tp.Final = "__KV_HEAD__"
MLP_INTERMEDIATE: tp.Final = "__MLP_INTERMEDIATE__"
VOCAB: tp.Final = "__VOCAB__"
EXPERT: tp.Final = "__EXPERT__"
EXPERT_GATE: tp.Final = "__EXPERT_GATE__"
HEAD_DIM: tp.Final = "__HEAD_DIM__"
KV_HEAD_DIM: tp.Final = "__KV_HEAD_DIM__"
BIAS_HEAD_SEQ: tp.Final = "__BIAS_HEAD_SEQ__"
BIAS_KV_SEQ: tp.Final = "__BIAS_KV_SEQ__"

PIPELINE_PARALLEL: tp.Final = "__PIPELINE_PARALLEL__"
DATA_PARALLEL: tp.Final = "__DATA_PARALLEL__"
FULLY_SHARDED_DATA_PARALLEL: tp.Final = "__FULLY_SHARDED_DATA_PARALLEL__"
TENSOR_PARALLEL: tp.Final = "__TENSOR_PARALLEL__"
EXPERT_PARALLEL: tp.Final = "__EXPERT_PARALLEL__"
SEQUENCE_PARALLEL: tp.Final = "__SEQUENCE_PARALLEL__"

PP: tp.Final = PIPELINE_PARALLEL
DP: tp.Final = DATA_PARALLEL
FSDP: tp.Final = FULLY_SHARDED_DATA_PARALLEL
TP: tp.Final = TENSOR_PARALLEL
EP: tp.Final = EXPERT_PARALLEL
SP: tp.Final = SEQUENCE_PARALLEL

MODE_DECODE: tp.Final = "__autoregressive__"
MODE_PREFILL: tp.Final = "__prefill__"
MODE_TRAIN: tp.Final = "__train__"
MODE_INSERT: tp.Final = "__insert__"

GENERATION_MODES = {
    MODE_DECODE,
    MODE_INSERT,
}

RUNTIME_MODE_TYPES = tp.Literal[
    "__autoregressive__",
    "__prefill__",
    "__train__",
    "__insert__",
]


class DynamicShardingAxes(tp.NamedTuple):
    """Symbolic sharding spec: a sequence of axis tokens plus a runtime mode.

    A ``DynamicShardingAxes`` instance describes how a tensor *should*
    be sharded under a given runtime mode, in terms of symbolic axis
    tokens. The :class:`~spectrax.sharding.PartitionManager` later
    resolves each token to a concrete physical mesh axis.

    Attributes:
        axes: Per-tensor-dimension specification. Each entry is either
            a symbolic axis token (e.g. :data:`BATCH`, :data:`EMBED`),
            a list of tokens (representing a *fused* mesh axis whose
            elements are tried left-to-right at resolution time), or
            :data:`EMPTY` (replicated). The length of ``axes`` must
            equal the rank of the tensor.
        mode: Either a :data:`RUNTIME_MODE_TYPES` literal (e.g.
            :data:`MODE_TRAIN`, :data:`MODE_DECODE`) or an integer
            *axis index*. When an integer is given,
            :meth:`~spectrax.sharding.PartitionManager.resolve`
            inspects ``shape[mode]`` and dispatches to
            :data:`MODE_DECODE` when that dim equals 1 (single-token
            decode) and :data:`MODE_TRAIN` otherwise.

    Subclasses (e.g. :class:`HiddenStateSharding`, :class:`ColumnWise`)
    set ``axes`` and ``mode`` as :class:`typing.ClassVar` defaults so
    they can be used in annotations like::

        x: jax.Array  # shape: HiddenStateSharding.axes
    """

    axes: tp.Sequence[str | None]
    mode: RUNTIME_MODE_TYPES | int


class HiddenStateSharding(DynamicShardingAxes):
    """Standard 3-D sharding for transformer hidden states ``[B, T, D]``.

    Maps to ``(BATCH, QUERY_LENGTH, EMBED)``. ``mode=1`` is an *axis
    index*: at resolution time
    :meth:`~spectrax.sharding.PartitionManager.resolve` reads
    ``shape[1]`` and dispatches to ``MODE_DECODE`` when that dim is
    1 (single-token decode) and ``MODE_TRAIN`` otherwise.
    """

    axes: tp.ClassVar = [BATCH, QUERY_LENGTH, EMBED]
    mode: tp.ClassVar = 1


class AttnQSharding(DynamicShardingAxes):
    """4-D sharding for attention queries ``[B, T_q, H, D_h]``.

    Maps to ``(BATCH, QUERY_LENGTH, HEAD, HEAD_DIM)``."""

    axes: tp.ClassVar = [BATCH, QUERY_LENGTH, HEAD, HEAD_DIM]
    mode: tp.ClassVar = 1


class AttnKVSharding(DynamicShardingAxes):
    """4-D sharding for attention keys/values ``[B, T_kv, H_kv, D_kv]``.

    Maps to ``(BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM)``. Separate from
    :class:`AttnQSharding` to support grouped-query attention where
    the KV head count differs from the Q head count.
    """

    axes: tp.ClassVar = [BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM]
    mode: tp.ClassVar = 1


class RowWise(DynamicShardingAxes):
    """Row-parallel weight sharding under training.

    Layout ``(TP, [FSDP, SP])`` — the input/contracting dim is
    tensor-parallel, the output dim is FSDP/SP-fused. Use for the
    *second* matmul in a ``Linear → Linear`` block (the one that
    performs an all-reduce on its output).
    """

    axes: tp.ClassVar = [TP, [FSDP, SP]]
    mode: tp.ClassVar = MODE_TRAIN


class SRowWise(DynamicShardingAxes):
    """Slim row-parallel sharding ``(TP,)`` for 1-D / bias-like params."""

    axes: tp.ClassVar = [TP]
    mode: tp.ClassVar = MODE_TRAIN


class ColumnWise(DynamicShardingAxes):
    """Column-parallel weight sharding under training.

    Layout ``([FSDP, SP], TP)`` — the input dim is FSDP/SP-fused, the
    output (column) dim is tensor-parallel. Use for the *first*
    matmul in a ``Linear → Linear`` block.
    """

    axes: tp.ClassVar = [[FSDP, SP], TP]
    mode: tp.ClassVar = MODE_TRAIN


class SColumnWise(DynamicShardingAxes):
    """Slim column-parallel sharding ``([FSDP, SP],)`` for 1-D params."""

    axes: tp.ClassVar = [[FSDP, SP]]
    mode: tp.ClassVar = MODE_TRAIN


class Replicated(DynamicShardingAxes):
    """Fully replicated sharding — every dimension is :data:`EMPTY`."""

    axes: tp.ClassVar = [EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertColumnWise(DynamicShardingAxes):
    """MoE column-parallel weight sharding ``(EP, FSDP, TP)``.

    First dim is the expert axis, second is FSDP across the expert's
    input dim, third is tensor-parallel across the expert's output dim.
    """

    axes: tp.ClassVar = [EP, FSDP, TP]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertRowWise(DynamicShardingAxes):
    """MoE row-parallel weight sharding ``(EP, TP, FSDP)`` — mirror of
    :class:`ExpertColumnWise` for the second matmul of an expert MLP."""

    axes: tp.ClassVar = [EP, TP, FSDP]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertColumnWiseAlt(DynamicShardingAxes):
    """Alt MoE column-parallel layout ``(EP, [FSDP, SP], TP)`` —
    fuses sequence parallelism into the input dim."""

    axes: tp.ClassVar = [EP, [FSDP, SP], TP]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertRowWiseAlt(DynamicShardingAxes):
    """Alt MoE row-parallel layout ``(EP, TP, [FSDP, SP])`` — mirror
    of :class:`ExpertColumnWiseAlt`."""

    axes: tp.ClassVar = [EP, TP, [FSDP, SP]]
    mode: tp.ClassVar = MODE_TRAIN


class UnifiedExpertColumnWise(DynamicShardingAxes):
    """MoE column-parallel layout that fuses ``EP`` into the data axis.

    ``([FSDP, SP, EP], EMPTY, TP)`` — the leading axis is a single
    fused mesh group containing FSDP, SP and EP, useful when expert
    parallelism shares the same mesh axis as data parallelism.
    """

    axes: tp.ClassVar = [[FSDP, SP, EP], EMPTY, TP]
    mode: tp.ClassVar = MODE_TRAIN


class UnifiedExpertRowWise(DynamicShardingAxes):
    """MoE row-parallel layout with EP fused into data axis.

    ``([FSDP, SP, EP], TP, EMPTY)`` — mirror of
    :class:`UnifiedExpertColumnWise`."""

    axes: tp.ClassVar = [[FSDP, SP, EP], TP, EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertActivations(DynamicShardingAxes):
    """Activation sharding for MoE layers ``(DP, SP, EP, TP)``.

    Used for the dispatched activation tensor whose four dimensions
    are batch (DP), sequence (SP), expert (EP), and feature (TP).
    """

    axes: tp.ClassVar = [DP, SP, EP, TP]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertActivationsAlt(DynamicShardingAxes):
    """Alt MoE activation sharding ``(DP, SP, [TP, FSDP])`` — used when
    the feature dim is fused tensor-and-data-parallel rather than
    split into separate axes."""

    axes: tp.ClassVar = [DP, SP, [TP, FSDP]]
    mode: tp.ClassVar = MODE_TRAIN


class ExpertTensorParallel(DynamicShardingAxes):
    """Pure tensor-parallel sharding for MoE bias-like tensors:
    ``(TP, EMPTY, EMPTY)``."""

    axes: tp.ClassVar = [TP, EMPTY, EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NOT_GIVEN = _Empty()
EMPTY_VAL = _Empty


__all__ = [
    "BATCH",
    "BIAS_HEAD_SEQ",
    "BIAS_KV_SEQ",
    "DATA_PARALLEL",
    "DEFAULT_MASK_VALUE",
    "DP",
    "EMBED",
    "EMPTY",
    "EMPTY_VAL",
    "EP",
    "EXPERT",
    "EXPERT_GATE",
    "EXPERT_PARALLEL",
    "FSDP",
    "FULLY_SHARDED_DATA_PARALLEL",
    "GENERATION_MODES",
    "HEAD",
    "HEAD_DIM",
    "KV_HEAD",
    "KV_HEAD_DIM",
    "KV_LENGTH",
    "LENGTH",
    "MLP_INTERMEDIATE",
    "MODE_DECODE",
    "MODE_INSERT",
    "MODE_PREFILL",
    "MODE_TRAIN",
    "NOT_GIVEN",
    "PIPELINE_PARALLEL",
    "PP",
    "QUERY_LENGTH",
    "RUNTIME_MODE_TYPES",
    "SEQUENCE_PARALLEL",
    "SP",
    "TENSOR_PARALLEL",
    "TP",
    "VOCAB",
    "Array",
    "AttnKVSharding",
    "AttnQSharding",
    "AxisIdxes",
    "AxisNames",
    "AxisType",
    "ColumnWise",
    "DType",
    "DynamicShardingAxes",
    "ExpertActivations",
    "ExpertActivationsAlt",
    "ExpertColumnWise",
    "ExpertColumnWiseAlt",
    "ExpertRowWise",
    "ExpertRowWiseAlt",
    "ExpertTensorParallel",
    "HiddenStateSharding",
    "Mesh",
    "PRNGKey",
    "Replicated",
    "RowWise",
    "SColumnWise",
    "SRowWise",
    "Shape",
    "UnifiedExpertColumnWise",
    "UnifiedExpertRowWise",
    "_Empty",
]
