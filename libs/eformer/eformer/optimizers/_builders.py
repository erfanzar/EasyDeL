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

import optax

from ._base import OptimizerBuilder, SchedulerBuilder, register_optimizer, register_scheduler
from ._config import (
    AdafactorConfig,
    AdamWConfig,
    LionConfig,
    MarsConfig,
    MuonConfig,
    RMSPropConfig,
    SchedulerConfig,
    WhiteKronConfig,
)
from ._stage_local import (
    StageLocalOptimizerMetadata,
    _apply_adafactor_stage_local,
    _apply_adamw_stage_local,
    _apply_lion_stage_local,
    _apply_mars_stage_local,
    _apply_muon_stage_local,
    _apply_quad_stage_local,
    _apply_rmsprop_stage_local,
    _apply_skew_stage_local,
    make_stage_local_gradient_transformation,
)
from ._tx import mars, quad, skew


def _build_metadata_mpmd_optimizer(
    config: tp.Any,
    scheduler: optax.Schedule,
    *,
    optimizer: optax.GradientTransformation,
    stage_local_apply: tp.Callable[..., tuple[optax.Params, optax.OptState]],
    **tx_kwargs: tp.Any,
) -> optax.GradientTransformation:
    """Wrap an optimizer chain with stage-local metadata for PP training.

    This helper constructs a :class:`StageLocalOptimizerMetadata` dataclass
    from the builder's configuration and the factory-level hyperparameters,
    then attaches it to the optimizer chain via
    :func:`make_stage_local_gradient_transformation`. The resulting
    transformation supports both normal Optax updates and explicit
    stage-local applications inside pipeline-parallel training loops.

    Args:
        config: The optimizer-specific configuration object (e.g. ``AdamWConfig``).
        scheduler: Learning rate schedule paired with the optimizer.
        optimizer: Fully assembled optimizer chain from the factory.
        stage_local_apply: Optimizer-specific stage-local update kernel.
        **tx_kwargs: Factory-level transform options such as ``weight_decay``,
            ``weight_decay_mask``, ``gradient_accumulation_steps``, and
            ``clip_grad``.

    Returns:
        A :class:`StageLocalGradientTransformation` wrapping ``optimizer``.
    """

    known_tx_kwargs = {
        "weight_decay",
        "weight_decay_mask",
        "gradient_accumulation_steps",
        "clip_grad",
    }
    extra_kwargs = {key: value for key, value in tx_kwargs.items() if key not in known_tx_kwargs}
    metadata = StageLocalOptimizerMetadata(
        scheduler=scheduler,
        weight_decay=float(tx_kwargs.get("weight_decay", 0.0)),
        weight_decay_mask=tx_kwargs.get("weight_decay_mask"),
        gradient_accumulation_steps=int(tx_kwargs.get("gradient_accumulation_steps", 1)),
        clip_grad=tx_kwargs.get("clip_grad"),
        adamw_b1=getattr(config, "b1", None),
        adamw_b2=getattr(config, "b2", None),
        adamw_eps=getattr(config, "eps", None),
        adamw_eps_root=getattr(config, "eps_root", None),
        adamw_mu_dtype=getattr(config, "mu_dtype", None),
        optimizer_config=config,
        extra_kwargs=extra_kwargs,
    )
    return make_stage_local_gradient_transformation(optimizer, metadata=metadata, apply_fn=stage_local_apply)


@register_scheduler("constant")
@dataclasses.dataclass
class ConstantSchedulerBuilder(SchedulerBuilder):
    """Builder for constant learning rate schedule.

    This builder creates a scheduler that maintains a fixed learning rate
    throughout training.

    Attributes:
        config (SchedulerConfig): Configuration object containing the learning rate.

    Example:
        >>> from eformer.optimizers import SchedulerConfig
        >>> config = SchedulerConfig(learning_rate=1e-4)
        >>> builder = ConstantSchedulerBuilder(config=config)
        >>> scheduler = builder.build()
        >>> scheduler(0)  # Returns 1e-4
    """

    config: SchedulerConfig

    def build(self) -> optax.Schedule:
        """Build a constant learning rate schedule.

        Returns:
            optax.Schedule: A schedule function that returns the configured
                learning rate regardless of the step count.
        """
        return optax.constant_schedule(self.config.learning_rate)


@register_scheduler("linear")
@dataclasses.dataclass
class LinearSchedulerBuilder(SchedulerBuilder):
    """Builder for linear learning rate schedule with optional warmup.

    This builder creates a scheduler that linearly decays the learning rate
    from an initial value to an end value over a specified number of steps.
    Optionally, a warmup phase can be added at the beginning of training.

    Attributes:
        config (SchedulerConfig): Configuration object containing learning rate
            parameters, steps, and optional warmup settings.

    Example:
        >>> from eformer.optimizers import SchedulerConfig
        >>> config = SchedulerConfig(
        ...     scheduler_type="linear",
        ...     learning_rate=1e-4,
        ...     learning_rate_end=1e-6,
        ...     steps=10000,
        ...     warmup_steps=1000,
        ... )
        >>> builder = LinearSchedulerBuilder(config=config)
        >>> scheduler = builder.build()
    """

    config: SchedulerConfig

    def build(self) -> optax.Schedule:
        """Build a linear learning rate schedule with optional warmup.

        Creates a linear decay schedule from `learning_rate` to `learning_rate_end`.
        If warmup_steps is specified, prepends a linear warmup phase from a very
        small value (1e-8) to the initial learning rate.

        Returns:
            optax.Schedule: A schedule function that returns the learning rate
                for a given step count.

        Raises:
            ValueError: If learning_rate_end is not specified in the config.
        """
        if self.config.learning_rate_end is None:
            raise ValueError("Linear scheduler requires learning_rate_end")

        decay_steps = self.config.steps
        if self.config.warmup_steps:
            decay_steps = self.config.steps - self.config.warmup_steps

        base_scheduler = optax.linear_schedule(
            init_value=self.config.learning_rate,
            end_value=self.config.learning_rate_end,
            transition_steps=decay_steps,
        )

        if self.config.warmup_steps:
            warmup = optax.linear_schedule(
                init_value=1e-8,
                end_value=self.config.learning_rate,
                transition_steps=self.config.warmup_steps,
            )
            return optax.join_schedules(
                schedules=[warmup, base_scheduler],
                boundaries=[self.config.warmup_steps],
            )
        return base_scheduler


@register_scheduler("cosine")
@dataclasses.dataclass
class CosineSchedulerBuilder(SchedulerBuilder):
    """Builder for cosine learning rate schedule with optional warmup.

    This builder creates a scheduler that decays the learning rate following
    a cosine curve. This is a popular choice for training neural networks as
    it provides smooth decay with a "warm restart" capability.

    Attributes:
        config (SchedulerConfig): Configuration object containing learning rate
            parameters, steps, warmup settings, and cosine decay exponent.

    Example:
        >>> from eformer.optimizers import SchedulerConfig
        >>> config = SchedulerConfig(
        ...     scheduler_type="cosine",
        ...     learning_rate=1e-4,
        ...     learning_rate_end=1e-6,
        ...     steps=10000,
        ...     warmup_steps=1000,
        ...     exponent=1.0,
        ... )
        >>> builder = CosineSchedulerBuilder(config=config)
        >>> scheduler = builder.build()
    """

    config: SchedulerConfig

    def build(self) -> optax.Schedule:
        """Build a cosine learning rate schedule with optional warmup.

        Creates a cosine decay schedule that smoothly decreases the learning rate
        from the peak value to the end value. If warmup_steps is specified,
        includes a linear warmup phase from a very small value (1e-8) to the
        peak learning rate before the cosine decay begins.

        Returns:
            optax.Schedule: A schedule function that returns the learning rate
                for a given step count, following a cosine decay pattern.
        """
        if self.config.warmup_steps:
            end_value = self.config.learning_rate_end or 0.0
            return optax.warmup_cosine_decay_schedule(
                init_value=1e-8,
                peak_value=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.steps,
                end_value=end_value,
                exponent=self.config.exponent,
            )

        if self.config.learning_rate_end is not None and self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0 when learning_rate_end is set")

        cosine_alpha = (
            0.0 if self.config.learning_rate_end is None else self.config.learning_rate_end / self.config.learning_rate
        )
        return optax.cosine_decay_schedule(
            init_value=self.config.learning_rate,
            decay_steps=self.config.steps,
            alpha=cosine_alpha,
        )


@register_optimizer("adamw")
@dataclasses.dataclass
class AdamWOptimizer(OptimizerBuilder):
    """Builder for AdamW optimizer.

    AdamW is a variant of Adam that decouples weight decay from the gradient
    update, which often leads to better generalization. It is one of the most
    widely used optimizers for training transformers and other deep learning models.

    Attributes:
        config (AdamWConfig): Configuration object containing AdamW hyperparameters
            including momentum coefficients (b1, b2), epsilon values, and data type.

    Example:
        >>> from eformer.optimizers import AdamWConfig
        >>> import optax
        >>> config = AdamWConfig(b1=0.9, b2=0.999, eps=1e-8)
        >>> builder = AdamWOptimizer(config=config)
        >>> scheduler = optax.constant_schedule(1e-4)
        >>> optimizer = builder.build(scheduler)
    """

    config: AdamWConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """Build the AdamW optimizer transformation.

        Args:
            scheduler (optax.Schedule): Learning rate schedule to use for the optimizer.

        Returns:
            optax.GradientTransformation: The AdamW optimizer transformation that can
                be used with optax.apply_updates to update model parameters.
        """
        return optax.adamw(
            learning_rate=scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            eps=self.config.eps,
            eps_root=self.config.eps_root,
            mu_dtype=self.config.mu_dtype,
            weight_decay=0.0,
        )

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the AdamW stage-local optimizer for pipeline-parallel training.

        Delegates to :func:`_build_metadata_mpmd_optimizer` to attach stage-local
        metadata and a leafwise AdamW update kernel to the optimizer chain.

        Args:
            scheduler: Learning rate schedule.
            optimizer: Fully assembled optimizer chain from the factory.
            **tx_kwargs: Factory-level transformation options.

        Returns:
            A :class:`StageLocalGradientTransformation` supporting both normal
            Optax updates and stage-local PP updates.
        """

        return _build_metadata_mpmd_optimizer(
            self.config,
            scheduler,
            optimizer=optimizer,
            stage_local_apply=_apply_adamw_stage_local,
            **tx_kwargs,
        )


@register_optimizer("adafactor")
@dataclasses.dataclass
class AdafactorOptimizer(OptimizerBuilder):
    """Builder for Adafactor optimizer.

    Adafactor is a memory-efficient adaptive learning rate optimizer designed
    for training large models. It uses factored second-moment estimation to
    reduce memory usage while maintaining adaptive learning rate capabilities.

    This optimizer is particularly useful for training large language models
    where memory constraints are significant.

    Attributes:
        config (AdafactorConfig): Configuration object containing Adafactor
            hyperparameters including factorization settings, decay rates,
            and clipping thresholds.

    Example:
        >>> from eformer.optimizers import AdafactorConfig
        >>> import optax
        >>> config = AdafactorConfig(decay_rate=0.8, factored=True)
        >>> builder = AdafactorOptimizer(config=config)
        >>> scheduler = optax.constant_schedule(1e-4)
        >>> optimizer = builder.build(scheduler)
    """

    config: AdafactorConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """Build the Adafactor optimizer transformation.

        Args:
            scheduler (optax.Schedule): Learning rate schedule to use for the optimizer.

        Returns:
            optax.GradientTransformation: The Adafactor optimizer transformation
                configured with factored second-moment estimation for memory efficiency.
        """
        return optax.adafactor(
            learning_rate=scheduler,
            min_dim_size_to_factor=self.config.min_dim_size_to_factor,
            decay_rate=self.config.decay_rate,
            decay_offset=self.config.decay_offset,
            multiply_by_parameter_scale=self.config.multiply_by_parameter_scale,
            clipping_threshold=self.config.clipping_threshold,
            momentum=self.config.momentum,
            dtype_momentum=self.config.dtype_momentum,
            weight_decay_rate=self.config.weight_decay_rate,
            eps=self.config.eps,
            factored=self.config.factored,
        )

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the Adafactor stage-local optimizer for pipeline-parallel training.

        Delegates to :func:`_build_metadata_mpmd_optimizer` to attach stage-local
        metadata and a leafwise Adafactor update kernel to the optimizer chain.

        Args:
            scheduler: Learning rate schedule.
            optimizer: Fully assembled optimizer chain from the factory.
            **tx_kwargs: Factory-level transformation options.

        Returns:
            A :class:`StageLocalGradientTransformation` supporting both normal
            Optax updates and stage-local PP updates.
        """

        return _build_metadata_mpmd_optimizer(
            self.config,
            scheduler,
            optimizer=optimizer,
            stage_local_apply=_apply_adafactor_stage_local,
            **tx_kwargs,
        )


@register_optimizer("lion")
@dataclasses.dataclass
class LionOptimizer(OptimizerBuilder):
    """Builder for Lion (Evolved Sign Momentum) optimizer.

    Lion is an optimizer discovered through neural architecture search that
    uses sign-based updates with momentum. It often achieves better
    generalization than Adam with fewer hyperparameters to tune.

    Reference: https://arxiv.org/abs/2302.06675

    Attributes:
        config (LionConfig): Configuration object containing Lion hyperparameters
            including momentum coefficients (b1, b2) and data type for momentum.

    Example:
        >>> from eformer.optimizers import LionConfig
        >>> import optax
        >>> config = LionConfig(b1=0.9, b2=0.99)
        >>> builder = LionOptimizer(config=config)
        >>> scheduler = optax.constant_schedule(1e-4)
        >>> optimizer = builder.build(scheduler)
    """

    config: LionConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """Build the Lion optimizer transformation.

        Args:
            scheduler (optax.Schedule): Learning rate schedule to use for the optimizer.

        Returns:
            optax.GradientTransformation: The Lion optimizer transformation that uses
                sign-based updates with momentum for efficient parameter updates.
        """
        return optax.lion(
            learning_rate=scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            mu_dtype=self.config.mu_dtype,
        )

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the Lion stage-local optimizer for pipeline-parallel training.

        Delegates to :func:`_build_metadata_mpmd_optimizer` to attach stage-local
        metadata and a leafwise Lion update kernel to the optimizer chain.

        Args:
            scheduler: Learning rate schedule.
            optimizer: Fully assembled optimizer chain from the factory.
            **tx_kwargs: Factory-level transformation options.

        Returns:
            A :class:`StageLocalGradientTransformation` supporting both normal
            Optax updates and stage-local PP updates.
        """

        return _build_metadata_mpmd_optimizer(
            self.config,
            scheduler,
            optimizer=optimizer,
            stage_local_apply=_apply_lion_stage_local,
            **tx_kwargs,
        )


@register_optimizer("rmsprop")
@dataclasses.dataclass
class RMSPropOptimizer(OptimizerBuilder):
    """Builder for RMSProp (Root Mean Square Propagation) optimizer.

    RMSProp is an adaptive learning rate optimizer that divides the gradient
    by a running average of the magnitude of recent gradients. It is effective
    for training recurrent neural networks and other models with non-stationary
    objectives.

    Attributes:
        config (RMSPropConfig): Configuration object containing RMSProp hyperparameters
            including decay rate, epsilon, momentum, and Nesterov momentum settings.

    Example:
        >>> from eformer.optimizers import RMSPropConfig
        >>> import optax
        >>> config = RMSPropConfig(decay=0.9, eps=1e-8, momentum=0.9)
        >>> builder = RMSPropOptimizer(config=config)
        >>> scheduler = optax.constant_schedule(1e-4)
        >>> optimizer = builder.build(scheduler)
    """

    config: RMSPropConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """Build the RMSProp optimizer transformation.

        Args:
            scheduler (optax.Schedule): Learning rate schedule to use for the optimizer.

        Returns:
            optax.GradientTransformation: The RMSProp optimizer transformation that
                adapts the learning rate based on a moving average of squared gradients.
        """
        return optax.rmsprop(
            learning_rate=scheduler,
            decay=self.config.decay,
            eps=self.config.eps,
            initial_scale=self.config.initial_scale,
            centered=False,
            momentum=self.config.momentum,
            nesterov=self.config.nesterov,
        )

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the RMSProp stage-local optimizer for pipeline-parallel training.

        Delegates to :func:`_build_metadata_mpmd_optimizer` to attach stage-local
        metadata and a leafwise RMSProp update kernel to the optimizer chain.

        Args:
            scheduler: Learning rate schedule.
            optimizer: Fully assembled optimizer chain from the factory.
            **tx_kwargs: Factory-level transformation options.

        Returns:
            A :class:`StageLocalGradientTransformation` supporting both normal
            Optax updates and stage-local PP updates.
        """

        return _build_metadata_mpmd_optimizer(
            self.config,
            scheduler,
            optimizer=optimizer,
            stage_local_apply=_apply_rmsprop_stage_local,
            **tx_kwargs,
        )


@register_optimizer("muon")
@dataclasses.dataclass
class MuonOptimizer(OptimizerBuilder):
    """Builder for Muon (Momentum Orthogonalized by Newton-schulz) optimizer.

    Muon is designed specifically for 2D parameters (matrices) and uses the
    Newton-Schulz method to orthogonalize momentum. Non-2D parameters are
    processed through an Adam optimizer fallback. This makes it particularly
    effective for training models with large matrix parameters.

    The optimizer maintains orthogonality of the momentum, which can lead to
    more stable training and better convergence for certain architectures.

    Attributes:
        config (MuonConfig): Configuration object containing Muon hyperparameters
            including Newton-Schulz coefficients, number of steps, momentum
            parameters, and Adam fallback settings.

    Example:
        >>> from eformer.optimizers import MuonConfig
        >>> import optax
        >>> config = MuonConfig(ns_steps=5, beta=0.95, nesterov=True)
        >>> builder = MuonOptimizer(config=config)
        >>> scheduler = optax.constant_schedule(1e-4)
        >>> optimizer = builder.build(scheduler)
    """

    config: MuonConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """Build the Muon optimizer transformation.

        Args:
            scheduler (optax.Schedule): Learning rate schedule to use for the optimizer.

        Returns:
            optax.GradientTransformation: The Muon optimizer transformation that uses
                Newton-Schulz orthogonalization for 2D parameters and Adam for others.
        """
        return optax.contrib.muon(
            learning_rate=scheduler,
            ns_steps=self.config.ns_steps,
            ns_coeffs=self.config.ns_coeffs,
            beta=self.config.beta,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            weight_decay_mask=self.config.weight_decay_mask,
            mu_dtype=self.config.mu_dtype,
            nesterov=self.config.nesterov,
            adaptive=self.config.adaptive,
            adam_b1=self.config.adam_b1,
            adam_b2=self.config.adam_b2,
            adam_eps_root=self.config.adam_eps_root,
        )

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the Muon stage-local optimizer for pipeline-parallel training.

        Delegates to :func:`_build_metadata_mpmd_optimizer` to attach stage-local
        metadata and a leafwise Muon update kernel to the optimizer chain.

        Args:
            scheduler: Learning rate schedule.
            optimizer: Fully assembled optimizer chain from the factory.
            **tx_kwargs: Factory-level transformation options.

        Returns:
            A :class:`StageLocalGradientTransformation` supporting both normal
            Optax updates and stage-local PP updates.
        """

        return _build_metadata_mpmd_optimizer(
            self.config,
            scheduler,
            optimizer=optimizer,
            stage_local_apply=_apply_muon_stage_local,
            **tx_kwargs,
        )


@register_optimizer("quad")
@dataclasses.dataclass
class QuadOptimizer(OptimizerBuilder):
    """Builder for Quad (White Kron with QUAD update) optimizer.

    Quad is a Kronecker-factored preconditioned optimizer that uses the QUAD
    preconditioner update style. It provides efficient second-order optimization
    by approximating the inverse Fisher information matrix using Kronecker products.

    This optimizer is particularly effective for training deep neural networks,
    especially transformers, where second-order information can significantly
    improve convergence.

    Attributes:
        config (WhiteKronConfig): Configuration object containing Quad optimizer
            hyperparameters including preconditioner settings, block size,
            sharding configurations, and numerical stability parameters.

    Example:
        >>> from eformer.optimizers import WhiteKronConfig
        >>> import optax
        >>> config = WhiteKronConfig(b1=0.95, preconditioner_lr=0.7)
        >>> builder = QuadOptimizer(config=config)
        >>> scheduler = optax.constant_schedule(1e-4)
        >>> optimizer = builder.build(scheduler)
    """

    config: WhiteKronConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """Build the Quad optimizer transformation.

        Args:
            scheduler (optax.Schedule): Learning rate schedule to use for the optimizer.

        Returns:
            optax.GradientTransformation: The Quad optimizer transformation using
                QUAD-style Kronecker-factored preconditioning for efficient
                second-order optimization.
        """
        return quad(
            learning_rate=scheduler,
            lr_style=self.config.lr_style,
            b1=self.config.b1,
            weight_decay=self.config.weight_decay,
            weight_decay_mask=self.config.weight_decay_mask,
            normalize_grads=self.config.normalize_grads,
            max_size_dense=self.config.max_size_dense,
            preconditioner_lr=self.config.preconditioner_lr,
            preconditioner_init_scale=self.config.preconditioner_init_scale,
            dtype=self.config.dtype,
            scanned_layers=self.config.scanned_layers,
            block_size=self.config.block_size,
            pipeline_axis_name=self.config.pipeline_axis_name,
            pipeline_axis_size=self.config.pipeline_axis_size,
            params_partition_specs=self.config.params_partition_specs,
            noise_scale=self.config.noise_scale,
        )

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the Quad stage-local optimizer for pipeline-parallel training.

        Delegates to :func:`_build_metadata_mpmd_optimizer` to attach stage-local
        metadata and a leafwise WhiteKron (Quad) update kernel to the optimizer
        chain.

        Args:
            scheduler: Learning rate schedule.
            optimizer: Fully assembled optimizer chain from the factory.
            **tx_kwargs: Factory-level transformation options.

        Returns:
            A :class:`StageLocalGradientTransformation` supporting both normal
            Optax updates and stage-local PP updates.
        """

        return _build_metadata_mpmd_optimizer(
            self.config,
            scheduler,
            optimizer=optimizer,
            stage_local_apply=_apply_quad_stage_local,
            **tx_kwargs,
        )


@register_optimizer("skew")
@dataclasses.dataclass
class SkewOptimizer(OptimizerBuilder):
    """Builder for Skew (White Kron with skew update) optimizer.

    Skew is a Kronecker-factored preconditioned optimizer that uses the skew
    preconditioner update style. It provides efficient second-order optimization
    with a different update rule compared to the QUAD variant.

    The skew update uses a Procrustes step to maintain orthogonality of the
    preconditioner, which can lead to more stable training in certain scenarios.

    Attributes:
        config (WhiteKronConfig): Configuration object containing Skew optimizer
            hyperparameters including preconditioner settings, block size,
            sharding configurations, and numerical stability parameters.

    Example:
        >>> from eformer.optimizers import WhiteKronConfig
        >>> import optax
        >>> config = WhiteKronConfig(b1=0.95, preconditioner_lr=0.7)
        >>> builder = SkewOptimizer(config=config)
        >>> scheduler = optax.constant_schedule(1e-4)
        >>> optimizer = builder.build(scheduler)
    """

    config: WhiteKronConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """Build the Skew optimizer transformation.

        Args:
            scheduler (optax.Schedule): Learning rate schedule to use for the optimizer.

        Returns:
            optax.GradientTransformation: The Skew optimizer transformation using
                skew-style Kronecker-factored preconditioning with Procrustes
                orthogonalization for efficient second-order optimization.
        """
        return skew(
            learning_rate=scheduler,
            lr_style=self.config.lr_style,
            b1=self.config.b1,
            weight_decay=self.config.weight_decay,
            weight_decay_mask=self.config.weight_decay_mask,
            normalize_grads=self.config.normalize_grads,
            max_size_dense=self.config.max_size_dense,
            preconditioner_lr=self.config.preconditioner_lr,
            preconditioner_init_scale=self.config.preconditioner_init_scale,
            dtype=self.config.dtype,
            scanned_layers=self.config.scanned_layers,
            block_size=self.config.block_size,
            pipeline_axis_name=self.config.pipeline_axis_name,
            pipeline_axis_size=self.config.pipeline_axis_size,
            params_partition_specs=self.config.params_partition_specs,
            noise_scale=self.config.noise_scale,
        )

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the Skew stage-local optimizer for pipeline-parallel training.

        Delegates to :func:`_build_metadata_mpmd_optimizer` to attach stage-local
        metadata and a leafwise WhiteKron (Skew) update kernel to the optimizer
        chain.

        Args:
            scheduler: Learning rate schedule.
            optimizer: Fully assembled optimizer chain from the factory.
            **tx_kwargs: Factory-level transformation options.

        Returns:
            A :class:`StageLocalGradientTransformation` supporting both normal
            Optax updates and stage-local PP updates.
        """

        return _build_metadata_mpmd_optimizer(
            self.config,
            scheduler,
            optimizer=optimizer,
            stage_local_apply=_apply_skew_stage_local,
            **tx_kwargs,
        )


@register_optimizer("mars")
@dataclasses.dataclass
class MarsOptimizer(OptimizerBuilder):
    """Builder for Mars (Matrix-wise Adaptive Regularized Scaling) optimizer.

    Mars improves upon Adam by using a variance reduction technique with gradient
    momentum from the previous step. This can lead to improved convergence and
    better generalization, particularly for training large language models.

    Reference: https://arxiv.org/abs/2411.10438

    Attributes:
        config (MarsConfig): Configuration object containing Mars hyperparameters
            including beta coefficients, gamma for gradient momentum, epsilon
            for numerical stability, and gradient clipping threshold.

    Example:
        >>> from eformer.optimizers import MarsConfig
        >>> import optax
        >>> config = MarsConfig(beta1=0.95, beta2=0.99, gamma=0.025)
        >>> builder = MarsOptimizer(config=config)
        >>> scheduler = optax.constant_schedule(1e-4)
        >>> optimizer = builder.build(scheduler)
    """

    config: MarsConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        """Build the Mars optimizer transformation.

        Args:
            scheduler (optax.Schedule): Learning rate schedule to use for the optimizer.

        Returns:
            optax.GradientTransformation: The Mars optimizer transformation that uses
                variance reduction with gradient momentum for improved convergence.
        """
        return mars(
            learning_rate=scheduler,
            b1=self.config.beta1,
            b2=self.config.beta2,
            gamma=self.config.gamma,
            eps=self.config.epsilon,
            max_grad_norm=self.config.max_grad_norm,
        )

    def build_mpmd(
        self,
        scheduler: optax.Schedule,
        *,
        optimizer: optax.GradientTransformation,
        **tx_kwargs: tp.Any,
    ) -> optax.GradientTransformation:
        """Build the Mars stage-local optimizer for pipeline-parallel training.

        Delegates to :func:`_build_metadata_mpmd_optimizer` to attach stage-local
        metadata and a leafwise Mars update kernel to the optimizer chain.

        Args:
            scheduler: Learning rate schedule.
            optimizer: Fully assembled optimizer chain from the factory.
            **tx_kwargs: Factory-level transformation options.

        Returns:
            A :class:`StageLocalGradientTransformation` supporting both normal
            Optax updates and stage-local PP updates.
        """

        return _build_metadata_mpmd_optimizer(
            self.config,
            scheduler,
            optimizer=optimizer,
            stage_local_apply=_apply_mars_stage_local,
            **tx_kwargs,
        )
