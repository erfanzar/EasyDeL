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

"""Optimizers module for eFormer.

This module provides a comprehensive set of optimizer builders, configurations,
and factories for creating and managing optimizers in JAX-based training pipelines.

Key components:
    - OptimizerBuilder: Abstract base class for creating optimizer builders.
    - SchedulerBuilder: Abstract base class for creating scheduler builders.
    - OptimizerFactory: Factory class for creating optimizers with validated configurations.
    - SchedulerFactory: Factory class for creating learning rate schedulers.

Available Optimizers:
    - AdamW: Adam with weight decay.
    - Adafactor: Memory-efficient adaptive optimizer.
    - Lion: Evolved Sign Momentum optimizer.
    - RMSProp: Root Mean Square Propagation optimizer.
    - Muon: Momentum Orthogonalized by Newton-schulz optimizer.
    - Mars: Matrix-wise Adaptive Regularized Scaling optimizer.
    - Quad: White Kron optimizer with QUAD preconditioner update style.
    - Skew: White Kron optimizer with skew preconditioner update style.

Available Schedulers:
    - Constant: Fixed learning rate.
    - Linear: Linear learning rate decay with optional warmup.
    - Cosine: Cosine annealing learning rate with optional warmup.

Example:
    >>> from eformer.optimizers import OptimizerFactory, SchedulerConfig, AdamWConfig
    >>> scheduler_config = SchedulerConfig(
    ...     scheduler_type="cosine",
    ...     learning_rate=1e-4,
    ...     warmup_steps=1000,
    ...     steps=10000,
    ... )
    >>> optimizer_config = AdamWConfig(b1=0.9, b2=0.999)
    >>> optimizer, scheduler = OptimizerFactory.create(
    ...     "adamw",
    ...     scheduler_config=scheduler_config,
    ...     optimizer_config=optimizer_config,
    ... )
"""

from ._base import (
    OptimizerBuilder,
    SchedulerBuilder,
    register_optimizer,
    register_scheduler,
)
from ._builders import (
    AdafactorOptimizer,
    AdamWOptimizer,
    ConstantSchedulerBuilder,
    CosineSchedulerBuilder,
    LinearSchedulerBuilder,
    LionOptimizer,
    MarsOptimizer,
    MuonOptimizer,
    QuadOptimizer,
    RMSPropOptimizer,
    SkewOptimizer,
)
from ._config import (
    AdafactorConfig,
    AdamWConfig,
    KronConfig,
    LionConfig,
    MarsConfig,
    MuonConfig,
    RMSPropConfig,
    SchedulerConfig,
    ScionConfig,
    SerializationMixin,
    SoapConfig,
    WhiteKronConfig,
)
from ._factory import OptimizerFactory, SchedulerFactory
from ._stage_local import (
    StageLocalGradientTransformation,
    StageLocalOptimizerMetadata,
    make_stage_local_gradient_transformation,
)

__all__ = (
    "AdafactorConfig",
    "AdafactorOptimizer",
    "AdamWConfig",
    "AdamWOptimizer",
    "ConstantSchedulerBuilder",
    "CosineSchedulerBuilder",
    "KronConfig",
    "LinearSchedulerBuilder",
    "LionConfig",
    "LionOptimizer",
    "MarsConfig",
    "MarsOptimizer",
    "MuonConfig",
    "MuonOptimizer",
    "OptimizerBuilder",
    "OptimizerFactory",
    "QuadOptimizer",
    "RMSPropConfig",
    "RMSPropOptimizer",
    "SchedulerBuilder",
    "SchedulerConfig",
    "SchedulerFactory",
    "ScionConfig",
    "SerializationMixin",
    "SkewOptimizer",
    "SoapConfig",
    "StageLocalGradientTransformation",
    "StageLocalOptimizerMetadata",
    "WhiteKronConfig",
    "make_stage_local_gradient_transformation",
    "register_optimizer",
    "register_scheduler",
)
