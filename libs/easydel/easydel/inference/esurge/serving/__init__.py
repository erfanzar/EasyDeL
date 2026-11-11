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

"""Multi-host OpenAI-compatible serving of one eSurge engine from a single YAML config.

Modules: :mod:`.config` (typed deployment config), :mod:`.cluster` (bind addresses and the per-launch coordination
token), :mod:`.keys` (the stable API-key file), :mod:`.app` (per-host entry point). Run with
``python -m easydel.inference.esurge.serving serve --config deployment.yaml`` on every host.
"""

from .config import CoordinationSpec, ModelSpec, QuantizationSpec, ServerSpec, ServingConfig, ServingConfigError
from .keys import ApiKeyFileError, load_or_create_api_key

__all__ = [
    "ApiKeyFileError",
    "CoordinationSpec",
    "ModelSpec",
    "QuantizationSpec",
    "ServerSpec",
    "ServingConfig",
    "ServingConfigError",
    "load_or_create_api_key",
]
