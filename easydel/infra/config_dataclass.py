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

"""Re-exports of typed config dataclass helpers.

Thin shim that surfaces ``ConfigDataclass`` and ``typed_config_dataclass``
from :mod:`easydel.typings` under the ``easydel.infra`` namespace so that
downstream modules can build typed configuration containers without importing
the deeper typing helpers directly.

Exports:
    ConfigDataclass: Generic typed config dataclass base.
    typed_config_dataclass: Decorator that turns a class into a typed config
        dataclass with automatic field validation and ``TypedDict`` mirrors.
"""

from easydel.typings import ConfigDataclass, typed_config_dataclass

__all__ = ("ConfigDataclass", "typed_config_dataclass")
