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


import dataclasses
import typing as tp
from abc import ABC, abstractmethod

import optax

from ._config import SchedulerConfig


@dataclasses.dataclass
class OptimizerBuilder(ABC):
    """
    Abstract base class for optimizer builders.

    Optimizer builders encapsulate the configuration and construction logic
    for creating optax GradientTransformation objects.

    Attributes:
            config: Optimizer-specific configuration object.

    Methods:
            build: Creates the base optimizer transformation.
            validate: Optional validation hook called before building.
    """

    config: tp.Any  # Will be overridden with specific config type in subclasses

    @abstractmethod
    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """
        Build the base optimizer transformation.

        Args:
                scheduler: Learning rate schedule to use.

        Returns:
                optax.GradientTransformation: The optimizer transformation.
        """
        pass

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the MPMD/pipeline-parallel optimizer transformation.

        Registered optimizers can override this hook to expose an explicit
        stage-local update API for scheduled pipeline-parallel training while
        preserving the normal :meth:`build` path for regular Optax use.

        When overridden, this method should return a
        :class:`StageLocalGradientTransformation` (or any
        :class:`optax.GradientTransformation` whose ``update`` callable carries
        a ``_eformer_stage_local_apply`` attribute) so that the factory can
        dispatch stage-local gradient applications during PP training.

        Args:
            scheduler: Learning rate schedule paired with the optimizer.
            optimizer: The fully assembled optimizer chain (clip, base, weight
                decay, multi-step) produced by the factory.
            **tx_kwargs: Factory-level transformation options such as
                ``weight_decay``, ``weight_decay_mask``,
                ``gradient_accumulation_steps``, and ``clip_grad``.

        Returns:
            A :class:`optax.GradientTransformation` that supports both the
            standard ``update`` path and the stage-local ``apply_gradients_stage_local``
            path when appropriate.

        Raises:
            NotImplementedError: By default, indicating that the optimizer does
                not yet provide PP stage-local semantics.
        """
        del scheduler, optimizer, tx_kwargs
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement build_mpmd(...). "
            "Override OptimizerBuilder.build_mpmd to provide PP stage-local optimizer semantics."
        )

    def validate(self) -> None:  # noqa
        """
        Optional validation hook called before building the optimizer.

        Raises:
                ValueError: If the configuration is invalid.
        """
        pass


@dataclasses.dataclass
class SchedulerBuilder(ABC):
    """
    Abstract base class for scheduler builders.

    Scheduler builders encapsulate the configuration and construction logic
    for creating optax Schedule objects.

    Attributes:
            config: Scheduler configuration object.

    Methods:
            build: Creates the learning rate schedule.
    """

    config: SchedulerConfig

    @abstractmethod
    def build(self) -> optax.Schedule:
        """
        Build the learning rate schedule.

        Returns:
                optax.Schedule: The learning rate schedule.
        """
        pass


# Registry dictionaries
_OPTIMIZER_BUILDER_REGISTRY: dict[str, type[OptimizerBuilder]] = {}
_SCHEDULER_BUILDER_REGISTRY: dict[str, type[SchedulerBuilder]] = {}


def register_optimizer(name: str) -> tp.Callable[[type[OptimizerBuilder]], type[OptimizerBuilder]]:
    """
    Decorator to register an optimizer builder class.

    Args:
            name: Name to register the optimizer under.

    Returns:
            Decorator function that registers the class.

    Example:
            @register_optimizer("adamw")
            @dataclass
            class AdamWOptimizer(OptimizerBuilder):
                    config: AdamWConfig

                    def build(self, scheduler):
                            return optax.adamw(learning_rate=scheduler, ...)
    """

    def decorator(cls: type[OptimizerBuilder]) -> type[OptimizerBuilder]:
        if name in _OPTIMIZER_BUILDER_REGISTRY:
            raise ValueError(f"Optimizer '{name}' is already registered")
        _OPTIMIZER_BUILDER_REGISTRY[name] = cls
        return cls

    return decorator


def register_scheduler(name: str) -> tp.Callable[[type[SchedulerBuilder]], type[SchedulerBuilder]]:
    """
    Decorator to register a scheduler builder class.

    Args:
            name: Name to register the scheduler under.

    Returns:
            Decorator function that registers the class.

    Example:
            @register_scheduler("cosine")
            @dataclass
            class CosineSchedulerBuilder(SchedulerBuilder):
                    config: SchedulerConfig

                    def build(self):
                            return optax.cosine_decay_schedule(...)
    """

    def decorator(cls: type[SchedulerBuilder]) -> type[SchedulerBuilder]:
        if name in _SCHEDULER_BUILDER_REGISTRY:
            raise ValueError(f"Scheduler '{name}' is already registered")
        _SCHEDULER_BUILDER_REGISTRY[name] = cls
        return cls

    return decorator
