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

"""Loss scaling submodule for mixed precision training stability.

This submodule provides loss scaling utilities that help maintain numerical
stability during mixed precision training. Loss scaling is essential when
using low-precision dtypes (float16) to prevent gradient underflow.

Classes:
    DynamicLossScale: Automatically adjusts scale based on gradient health.
    NoOpLossScale: No-op scaler for when loss scaling is not needed.
    LossScaleConfig: Configuration for loss scaling hyperparameters.
"""

from .loss_scaler import DynamicLossScale, LossScaleConfig, NoOpLossScale

__all__ = ("DynamicLossScale", "LossScaleConfig", "NoOpLossScale")
