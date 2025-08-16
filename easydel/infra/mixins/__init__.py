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

"""Mixin classes for EasyDeL modules.

Provides reusable functionality through mixin classes that can be combined
with base modules to add specific capabilities.

Classes:
    BaseModuleProtocol: Protocol defining the interface for base modules
    EasyBridgeMixin: Mixin for bridging between EasyDeL and HuggingFace
    EasyGenerationMixin: Mixin for text generation capabilities

Key Features:
    - Protocol-based interface definitions
    - Model conversion and compatibility
    - Text generation with various strategies
    - Seamless integration with base modules

Example:
    >>> from easydel.infra.mixins import EasyGenerationMixin
    >>> class MyModel(EasyDeLBaseModule, EasyGenerationMixin):
    ...     # Model implementation
    ...     pass
    >>>
    >>> # Now MyModel has generation capabilities
    >>> output = model.generate(
    ...     input_ids=input_ids,
    ...     max_length=100
    ... )
"""

from .bridge import EasyBridgeMixin
from .generation import EasyGenerationMixin
from .protocol import BaseModuleProtocol

__all__ = "BaseModuleProtocol", "EasyBridgeMixin", "EasyGenerationMixin"
