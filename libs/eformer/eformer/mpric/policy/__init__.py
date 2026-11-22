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

"""Precision policy submodule for defining mixed precision behavior.

This submodule provides the Policy dataclass that defines the dtypes used for
different aspects of mixed precision computation: parameters, compute operations,
and outputs.

Policies can be created from simple or detailed string specifications, making
it easy to configure precision settings for different hardware and use cases.
"""

from .policy import Policy

__all__ = ("Policy",)
