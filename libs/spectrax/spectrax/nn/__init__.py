# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neural-network layers for the Spectrax framework.

Every public symbol exported here is either a :class:`~spectrax.Module`
subclass (layer / container) or a free-standing factory (e.g.
:func:`~spectrax.nn.wrap_lora`). Layers expose the standard
``forward(...)`` entry-point and follow the conventions documented on
each individual class — most notably the *channels-last* tensor layout
for convolutional and pooling layers (``(N, *spatial, C)``) and the
*sequence-second* layout for attention / RNN inputs (``(N, T, ...)``).

Containers
The :class:`~spectrax.nn.Sequential`,
:class:`~spectrax.nn.ModuleList`,
:class:`~spectrax.nn.StackedModuleList`,
:class:`~spectrax.nn.ModuleDict`, and
:class:`~spectrax.nn.ParameterList` symbols are re-exported from
:mod:`spectrax.core.containers` so callers do not need to import
    from two different paths.

Mixed precision
Layers that perform matmuls (Linear, attention projections, MLP)
consult the active :func:`~spectrax.core.policy.current_policy` in
``forward`` and downcast the parameters / activations to the
policy's ``compute_dtype`` before the dot. Storage dtype is
independent and controlled per-layer through ``dtype`` /
``param_dtype`` constructor arguments.

Sharding
Constructors that allocate parameters accept ``sharding`` /
``bias_sharding`` keyword arguments and attach logical axis names
(``("in", "out")`` for dense weights, ``(*"k", "in", "out")`` for
convolution kernels, ``("vocab", "embed")`` for embeddings,
``("features",)`` / ``("channels",)`` for normalization parameters)
so the surrounding mesh can resolve them automatically.
"""

from ..core.containers import ModuleDict, ModuleList, ParameterList, Sequential, StackedModuleList
from .activation import GELU, ReLU, Sigmoid, SiLU, Tanh
from .attention import CausalSelfAttention, MultiheadAttention
from .conv import (
    Conv,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .dense import DenseGeneral, Einsum
from .dropout import Dropout
from .embed import Embed
from .fp8 import Fp8DotGeneral, Fp8Einsum, Fp8Linear, Fp8Meta
from .identity import Identity
from .linear import Bilinear, Linear
from .lora import LoRA, LoRALinear, LoraParameter, wrap_lora
from .mlp import MLPBlock
from .norm import BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm, LayerNorm, RMSNorm
from .pipeline_sequential import PipelineSequential
from .pool import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from .recurrent import (
    RNN,
    Bidirectional,
    ConvLSTMCell,
    GRUCell,
    LSTMCell,
    OptimizedLSTMCell,
    RNNCellBase,
    SimpleRNNCell,
)

__all__ = [
    "GELU",
    "RNN",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "Bidirectional",
    "Bilinear",
    "CausalSelfAttention",
    "Conv",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvLSTMCell",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DenseGeneral",
    "Dropout",
    "Einsum",
    "Embed",
    "Fp8DotGeneral",
    "Fp8Einsum",
    "Fp8Linear",
    "Fp8Meta",
    "GRUCell",
    "GroupNorm",
    "Identity",
    "InstanceNorm",
    "LSTMCell",
    "LayerNorm",
    "Linear",
    "LoRA",
    "LoRALinear",
    "LoraParameter",
    "MLPBlock",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "ModuleDict",
    "ModuleList",
    "MultiheadAttention",
    "OptimizedLSTMCell",
    "ParameterList",
    "PipelineSequential",
    "RMSNorm",
    "RNNCellBase",
    "ReLU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SimpleRNNCell",
    "StackedModuleList",
    "Tanh",
    "wrap_lora",
]
