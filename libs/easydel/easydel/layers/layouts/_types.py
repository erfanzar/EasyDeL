# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type aliases and TYPE_CHECKING-only imports for the layouts package.

Centralises forward-declared types so the rest of the package never
needs to manage its own ``TYPE_CHECKING`` blocks:

* ``Array`` and ``EasyDeLBaseConfig`` are imported under
  ``TYPE_CHECKING`` for accurate static analysis; at runtime they
  collapse to :class:`typing.Any` so circular imports are avoided.
* :data:`ReformRule` describes one entry of a reform-rule mapping
  (``{"sources": ..., "fuser": ..., "inverse_fuser": ..., ...}``).
* :data:`ReformParam` is the full mapping from regex-anchored target
  tensor names to their :data:`ReformRule` payload, as consumed by the
  EasyDeL checkpoint loader / exporter.
"""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from jax import Array

    from easydel.infra.base_config import EasyDeLBaseConfig
else:
    Array = tp.Any
    EasyDeLBaseConfig = tp.Any

ReformRule = dict[str, tp.Any]
ReformParam = dict[str, ReformRule]
