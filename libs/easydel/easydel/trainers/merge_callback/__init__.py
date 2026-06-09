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

"""Model-merge callback utilities for EasyDeL trainer callbacks.

The package exposes a lightweight native pytree merge callback used when a
training run needs a checkpoint-time or train-end merged model state.
"""

from ._fn import merge_pytrees
from .merge_callback import MergeModelCallback
from .merge_config import MergeConfig
from .merge_methods import get_merge_method, register_merge_method, registered_merge_methods

__all__ = (
    "MergeConfig",
    "MergeModelCallback",
    "get_merge_method",
    "merge_pytrees",
    "register_merge_method",
    "registered_merge_methods",
)
