# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stateless tensor operations.

Every function exported here is pure — it takes JAX-compatible inputs
and returns an :class:`~spectrax.typing.Array` with no module state on
the side. The :mod:`spectrax.nn` layer modules wrap these primitives
and supply learnable parameters; calling them directly is the right
choice when:

* a transform (``jit`` / ``vmap`` / ``scan`` / ``shard_map``) is easier
  to reason about over plain functions than over module instances, or
* the caller already owns the parameters (e.g. inside a custom training
  step or a meta-learning inner loop).

Submodules:

* :mod:`spectrax.functional.activation` — elementwise activations.
* :mod:`spectrax.functional.attention` — scaled dot-product attention.
* :mod:`spectrax.functional.conv` — N-D convolution / transposed conv.
* :mod:`spectrax.functional.dropout` — inverted dropout.
* :mod:`spectrax.functional.linear` — dense matmul + bias.
* :mod:`spectrax.functional.norm` — LayerNorm and RMSNorm.
* :mod:`spectrax.functional.pool` — reduce-window pooling primitives.
* :mod:`spectrax.functional.util` — shared dtype helpers.
"""

from .activation import (
    celu,
    elu,
    gelu,
    glu,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    mish,
    prelu,
    relu,
    selu,
    sigmoid,
    silu,
    soft_sign,
    softmax,
    tanh,
)
from .attention import scaled_dot_product_attention
from .conv import conv, conv_transpose
from .dropout import dropout
from .linear import linear
from .norm import layer_norm, rms_norm
from .pool import avg_pool, max_pool, pool
from .util import promote_dtype

__all__ = [
    "avg_pool",
    "celu",
    "conv",
    "conv_transpose",
    "dropout",
    "elu",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_sigmoid",
    "log_softmax",
    "max_pool",
    "mish",
    "pool",
    "prelu",
    "promote_dtype",
    "relu",
    "rms_norm",
    "scaled_dot_product_attention",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "tanh",
]
