# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Precision handler submodule for mixed precision operations.

This submodule provides the PrecisionHandler class, which serves as the main
interface for managing mixed precision training and inference in JAX.

The PrecisionHandler integrates precision policies with loss scaling to enable
efficient low-precision training while maintaining numerical stability.
"""

from .precision_handler import PrecisionHandler

__all__ = ("PrecisionHandler",)
