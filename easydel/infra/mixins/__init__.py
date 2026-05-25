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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mixin classes layered onto :class:`EasyDeLBaseModule`.

This package decomposes the very large surface area of an EasyDeL model
module into focused mixins, each owning one cross-cutting concern. They are
all composed by :class:`easydel.infra.base_module.EasyDeLBaseModule`, so
model authors normally do not need to import them directly — subclass
``EasyDeLBaseModule`` and the capabilities below are present automatically.

Mixin roster:
    - :class:`BaseModuleProtocol`: Structural protocol that every mixin
      relies on (config, dtype, mesh accessors, parameter helpers). Acts as
      the typing contract between mixins so they can be combined without
      cyclic class-level imports.
    - :class:`EasyBridgeMixin`: HuggingFace ↔ EasyDeL parameter bridge —
      loading ``from_pretrained``, saving ``save_pretrained``, weight name
      remapping, and ``pytorch_model.bin`` / safetensors shard streaming.
    - :class:`EasyGenerationMixin`: ``generate``, ``prefill``/``decode``
      loops, sampling controllers, and the per-model logits-processor /
      stopping-criteria plumbing.
    - :class:`OperationCacheMixin`: Compiled-function cache keyed by mesh,
      input shapes and dtype, so re-issuing the same forward/decode call
      hits a warm executable.
    - :class:`EasyShardingMixin`: NamedSharding / partition-spec utilities,
      mesh-aware ``shard``/``gather`` helpers.

Re-exports:
    :class:`BaseModuleProtocol`, :class:`EasyBridgeMixin`,
    :class:`EasyGenerationMixin`, :class:`EasyShardingMixin`,
    :class:`LayerOperationInfo`, :class:`OperationCacheMixin`,
    :class:`OperationsCacheInfo`.
"""

from .bridge import EasyBridgeMixin
from .generation import EasyGenerationMixin
from .operation_cache import (
    LayerOperationInfo,
    OperationCacheMixin,
    OperationsCacheInfo,
)
from .protocol import BaseModuleProtocol
from .sharding import EasyShardingMixin

__all__ = (
    "BaseModuleProtocol",
    "EasyBridgeMixin",
    "EasyGenerationMixin",
    "EasyShardingMixin",
    "LayerOperationInfo",
    "OperationCacheMixin",
    "OperationsCacheInfo",
)
