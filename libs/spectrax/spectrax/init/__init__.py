# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure parameter initializer factories.

Each public function here returns a callable implementing the
:class:`~spectrax.typing.Initializer` protocol —
``(key: PRNGKey, shape: Shape, dtype: DType) -> Array`` — suitable for
direct use when constructing a :class:`~spectrax.Parameter` or for
wrapping with :func:`spectrax.with_partitioning` to attach sharding
metadata.

Submodules:

* :mod:`spectrax.init.constant` — :func:`zeros`, :func:`ones`,
  :func:`constant`.
* :mod:`spectrax.init.normal` — :func:`normal`, :func:`truncated_normal`.
* :mod:`spectrax.init.uniform` — symmetric :func:`uniform`.
* :mod:`spectrax.init.xavier` — Glorot uniform/normal scaled by
  ``sqrt(./(fan_in + fan_out))``.
* :mod:`spectrax.init.kaiming` — He uniform/normal scaled by
  ``sqrt(./fan)`` with a per-nonlinearity gain.
* :mod:`spectrax.init.orthogonal` — orthogonal matrices via QR.
"""

from .constant import constant, ones, zeros
from .kaiming import kaiming_normal, kaiming_uniform
from .normal import normal, truncated_normal
from .orthogonal import orthogonal
from .uniform import uniform
from .xavier import xavier_normal, xavier_uniform

__all__ = [
    "constant",
    "kaiming_normal",
    "kaiming_uniform",
    "normal",
    "ones",
    "orthogonal",
    "truncated_normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]
