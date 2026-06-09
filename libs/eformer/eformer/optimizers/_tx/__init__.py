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

"""Custom gradient transformations for eFormer optimizers.

This submodule provides custom optax-compatible gradient transformations
including advanced optimizers like Mars and White Kron variants (Quad/Skew).

Modules:
    mars: Mars (Matrix-wise Adaptive Regularized Scaling) optimizer.
    white_kron: White Kron optimizer variants with different preconditioner updates.
    utils: Utility functions for creating schedulers and weight decay transformations.

Key Components:
    - scale_by_mars: Gradient scaling transformation for Mars optimizer.
    - mars: Complete Mars optimizer with learning rate scaling.
    - scale_by_quad: Gradient scaling with QUAD preconditioner update.
    - scale_by_skew: Gradient scaling with skew preconditioner update.
    - quad: Complete Quad optimizer with weight decay and learning rate.
    - skew: Complete Skew optimizer with weight decay and learning rate.
    - optax_add_scheduled_weight_decay: Weight decay with scheduled rate.
    - create_linear_scheduler: Linear learning rate schedule builder.
    - create_cosine_scheduler: Cosine learning rate schedule builder.
"""

from .mars import mars, scale_by_mars
from .utils import (
    OptaxScheduledWeightDecayState,
    create_cosine_scheduler,
    create_linear_scheduler,
    get_base_optimizer,
    optax_add_scheduled_weight_decay,
)
from .white_kron import quad, scale_by_quad, scale_by_skew, skew

__all__ = (
    "OptaxScheduledWeightDecayState",
    "create_cosine_scheduler",
    "create_linear_scheduler",
    "get_base_optimizer",
    "mars",
    "optax_add_scheduled_weight_decay",
    "quad",
    "scale_by_mars",
    "scale_by_quad",
    "scale_by_skew",
    "skew",
)
