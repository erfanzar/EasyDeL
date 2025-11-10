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

"""Worker processes for handling external dependencies.

This package contains internal worker implementations that are not exposed
in EasyDeL's public API. These workers are designed to handle tasks that
depend on external processes, such as tokenization and detokenization services.

Note:
    This module is for internal use only and is not part of EasyDeL's public API.
    It is only accessible to EasyDeL modules that require external worker processes
    to handle specific tasks.
"""
