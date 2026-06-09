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
import difflib
import inspect
import typing as tp
from dataclasses import fields

import jax
import optax

# Import builders to trigger registration (side-effect import)
# Import custom optimizers to trigger registration (side-effect import)
from . import _builders, _tx  # noqa

# Import base classes and registries
from ._base import _OPTIMIZER_BUILDER_REGISTRY, OptimizerBuilder
from ._config import (
    AdafactorConfig,
    AdamWConfig,
    KronConfig,
    LionConfig,
    MarsConfig,
    MuonConfig,
    RMSPropConfig,
    SchedulerConfig,
    SerializationMixin,
    SoapConfig,
    WhiteKronConfig,
)
from ._stage_local import make_unsupported_stage_local_gradient_transformation
from ._tx import optax_add_scheduled_weight_decay

TxConfigs = (
    AdafactorConfig
    | AdamWConfig
    | KronConfig
    | LionConfig
    | MarsConfig
    | MuonConfig
    | RMSPropConfig
    | SoapConfig
    | WhiteKronConfig
)


class SchedulerFactory:
    """
    Factory class for creating learning rate schedulers.

    This class provides methods to create schedulers based on a configuration object (`SchedulerConfig`).
    It supports linear and cosine schedulers with optional warmup steps.

    Methods:
        create_scheduler: Creates a scheduler based on the provided configuration.
        _create_linear: Creates a linear scheduler with optional warmup.
        _create_cosine: Creates a cosine scheduler with optional warmup.
    """

    @staticmethod
    def create_scheduler(
        config: SchedulerConfig,
        custom_scheduler: tp.Callable[[int], optax.Schedule] | None = None,
    ) -> optax.Schedule:
        """
        Create a scheduler based on the provided configuration.

        Args:
            config (SchedulerConfig): Configuration object for the scheduler.
            custom_scheduler (Optional[Callable[[int], optax.Schedule]]): Custom scheduler function. Defaults to None.

        Returns:
            optax.Schedule: The created scheduler.

        Raises:
            ValueError: If the configuration is invalid or unsupported scheduler type is provided.
        """
        if custom_scheduler is not None:
            if config.steps is None:
                raise ValueError("Custom schedulers require steps configuration")
            return custom_scheduler(config.steps)
        if config.scheduler_type is None:
            return optax.constant_schedule(config.learning_rate)
        if config.steps is None:
            raise ValueError("Steps must be specified for configured schedulers")
        if config.scheduler_type == "linear":
            return SchedulerFactory._create_linear(config)
        elif config.scheduler_type == "cosine":
            return SchedulerFactory._create_cosine(config)
        else:
            raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")

    @staticmethod
    def _create_linear(config: SchedulerConfig) -> optax.Schedule:
        """
        Create a linear scheduler with optional warmup.

        Args:
            config (SchedulerConfig): Configuration object for the scheduler.

        Returns:
            optax.Schedule: The created linear scheduler.
        """
        decay_steps = config.steps
        if config.warmup_steps:
            decay_steps = config.steps - config.warmup_steps

        base_scheduler = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=config.learning_rate_end,
            transition_steps=decay_steps,
        )

        if config.warmup_steps:
            warmup = optax.linear_schedule(
                init_value=1e-8,
                end_value=config.learning_rate,
                transition_steps=config.warmup_steps,
            )
            return optax.join_schedules(
                schedules=[warmup, base_scheduler],
                boundaries=[config.warmup_steps],
            )
        return base_scheduler

    @staticmethod
    def _create_cosine(config: SchedulerConfig) -> optax.Schedule:
        """
        Create a cosine scheduler with optional warmup.

        Args:
            config (SchedulerConfig): Configuration object for the scheduler.

        Returns:
            optax.Schedule: The created cosine scheduler.
        """
        if config.warmup_steps:
            end_value = config.learning_rate_end or 0.0
            return optax.warmup_cosine_decay_schedule(
                init_value=1e-8,
                peak_value=config.learning_rate,
                warmup_steps=config.warmup_steps,
                decay_steps=config.steps,
                end_value=end_value,
                exponent=config.exponent,
            )

        if config.learning_rate_end is not None and config.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0 when learning_rate_end is set")

        cosine_alpha = 0.0 if config.learning_rate_end is None else config.learning_rate_end / config.learning_rate
        return optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.steps,
            alpha=cosine_alpha,
        )


class OptimizerFactory:
    """
    Factory class for creating optimizers with validated configurations.

    This class provides methods to create optimizers based on a configuration object.
    All optimizers are registered using the @register_optimizer decorator pattern.

    Methods:
        create: Creates an optimizer with validated configuration.
        generate_template: Generates a configuration template for the specified optimizer.
        serialize_config: Serializes configuration to different formats.
        deserialize_config: Deserializes configuration from different formats.

    Private Methods:
        _get_config_class: Gets the configuration class for an optimizer type.
        _convert_dtypes: Converts string dtype representations to JAX dtypes.
        _validate_kwargs: Validates additional parameters for the optimizer.
        _build_optimizer_chain: Constructs the final optimizer chain.
    """

    @staticmethod
    def _get_config_class(optimizer_type: str) -> type:
        """
        Get the configuration class for an optimizer type.

        Args:
            optimizer_type: Name of the optimizer.

        Returns:
            Configuration class for the optimizer.

        Raises:
            ValueError: If optimizer type is not registered.
        """
        if optimizer_type not in _OPTIMIZER_BUILDER_REGISTRY:
            available = sorted(_OPTIMIZER_BUILDER_REGISTRY.keys())
            raise ValueError(f"Unsupported optimizer: {optimizer_type}. Available: {available}")

        builder_cls = _OPTIMIZER_BUILDER_REGISTRY[optimizer_type]
        # Get the config type from the builder's type hint
        config_field = next((f for f in fields(builder_cls) if f.name == "config"), None)
        if config_field and config_field.type:
            return config_field.type

        raise ValueError(f"Builder class for '{optimizer_type}' does not have a valid config field")

    @classmethod
    def create(
        cls,
        optimizer_type: str,
        scheduler_config: SchedulerConfig | None = None,
        optimizer_config: TxConfigs | None = None,
        *,
        weight_decay: float = 0.0,
        weight_decay_mask: tp.Any | None = None,
        gradient_accumulation_steps: int = 1,
        clip_grad: float | None = None,
        custom_scheduler: tp.Callable[[int], optax.Schedule] | None = None,
        **kwargs,
    ) -> tuple[optax.GradientTransformation, optax.Schedule]:
        """
        Create an optimizer with validated configuration.

        Args:
            optimizer_type (str): One of the registered optimizer types.
            scheduler_config (SchedulerConfig): Configured scheduler parameters.
            optimizer_config (Union[AdafactorConfig, AdamWConfig, LionConfig, MuonConfig, RMSPropConfig]):
                Optimizer-specific configuration.
            weight_decay (float): Global weight decay rate. Defaults to 0.0.
            weight_decay_mask (Optional[Any]): Mask for weight decay application. Defaults to None.
            gradient_accumulation_steps (int): Steps for gradient accumulation. Defaults to 1.
            clip_grad (Optional[float]): Global clip gradient norm value. Defaults to None.
            custom_scheduler (Optional[Callable[[int], optax.Schedule]]): Optional custom scheduler function.
                Defaults to None.
            **kwargs: Additional optimizer-specific parameters.

        Returns:
            Tuple[optax.GradientTransformation, optax.Schedule]: A tuple containing the optimizer and scheduler.

        Raises:
            ValueError: If the optimizer type is unsupported or the configuration is invalid.
            TypeError: If the configuration type is invalid.
        """

        # Get the appropriate config class
        config_cls = cls._get_config_class(optimizer_type)

        # Create default config if none provided
        if optimizer_config is None:
            optimizer_config = config_cls()
            for key in list(kwargs.keys()):
                if key in inspect.signature(optimizer_config.__class__).parameters:
                    setattr(optimizer_config, key, kwargs.pop(key))
        if scheduler_config is None:
            scheduler_config = SchedulerConfig()
        # Convert string dtypes to JAX dtypes
        cls._convert_dtypes(optimizer_config)

        # Validate config type
        if not isinstance(optimizer_config, config_cls):
            raise TypeError(
                f"Invalid config type {type(optimizer_config)} for optimizer {optimizer_type}. Expected {config_cls}"
            )

        # Validate scheduler config
        if scheduler_config.scheduler_type is None and scheduler_config.warmup_steps:
            raise ValueError("Warmup steps require specifying a scheduler type")

        # Create scheduler
        scheduler = SchedulerFactory.create_scheduler(scheduler_config, custom_scheduler)

        # Create optimizer using builder pattern
        builder_cls = _OPTIMIZER_BUILDER_REGISTRY[optimizer_type]
        builder = builder_cls(config=optimizer_config)
        builder.validate()
        base_optimizer = builder.build(scheduler)

        # Build the full optimizer chain (clip, base optimizer, weight decay, multi-step)
        return cls._build_optimizer_chain(
            builder=builder,
            base_optimizer=base_optimizer,
            optimizer_type=optimizer_type,
            scheduler=scheduler,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clip_grad=clip_grad,
        )

    @staticmethod
    def _convert_dtypes(config: tp.Any):
        """
        Automatically convert string dtype representations to JAX dtypes.

        Args:
            config (Any): Configuration object.

        Raises:
            ValueError: If an invalid dtype is specified.
        """
        for field in fields(config):
            if "dtype" in field.name and isinstance(getattr(config, field.name), str):
                dtype = getattr(jax.numpy, getattr(config, field.name), None)
                if dtype is None:
                    raise ValueError(f"Invalid dtype specified: {getattr(config, field.name)}")
                setattr(config, field.name, dtype)

    @classmethod
    def _validate_kwargs(cls, config: tp.Any, kwargs: dict[str, tp.Any]):
        """
        Validate additional parameters with helpful error messages.

        Args:
            config (Any): Configuration object.
            kwargs (Dict[str, Any]): Additional parameters.

        Raises:
            ValueError: If unexpected parameters are provided.
        """
        valid_params = inspect.signature(config.__class__).parameters
        for kwarg in kwargs:
            if kwarg not in valid_params:
                suggestions = ", ".join(difflib.get_close_matches(kwarg, valid_params.keys()))
                msg = (
                    f"Unexpected parameter '{kwarg}' for {config.__class__.__name__}. "
                    f"Valid parameters: {list(valid_params.keys())}"
                )
                if suggestions:
                    msg += f". Did you mean: {suggestions}?"
                raise ValueError(msg)

    @staticmethod
    def _build_optimizer_chain(
        builder: OptimizerBuilder,
        base_optimizer: optax.GradientTransformation,
        optimizer_type: str,
        scheduler: optax.Schedule,
        weight_decay: float = 0.0,
        weight_decay_mask: tp.Any | None = None,
        gradient_accumulation_steps: int = 1,
        clip_grad: float | None = None,
    ) -> tuple[optax.GradientTransformation, optax.Schedule]:
        """Construct the final optimizer chain with gradient clipping, weight decay, and accumulation.

        This method assembles the standard eFormer chain (clip -> base -> weight
        decay -> multi-step) and then attempts to wrap it with a pipeline-parallel
        stage-local path by calling ``builder.build_mpmd(...)``. If the builder
        does not implement :meth:`OptimizerBuilder.build_mpmd`, a fallback
        wrapper is installed that preserves the normal Optax path but raises a
        clear error when stage-local application is attempted.

        Args:
            builder: The optimizer builder instance used to construct the base
                optimizer. Its :meth:`build_mpmd` hook is invoked at the end of
                the chain.
            base_optimizer: Base optimizer transformation produced by
                ``builder.build(scheduler)``.
            optimizer_type: Registered optimizer name (e.g. ``"adamw"``).
            scheduler: Learning rate scheduler paired with the optimizer.
            weight_decay: Global weight-decay coefficient. Defaults to 0.0.
            weight_decay_mask: Mask for weight decay application. Defaults to None.
            gradient_accumulation_steps: Steps for gradient accumulation. Defaults to 1.
            clip_grad: Global gradient norm clipping value. Defaults to None.

        Returns:
            Tuple[optax.GradientTransformation, optax.Schedule]: A tuple containing
            the fully assembled optimizer chain (possibly with a stage-local
            wrapper) and the scheduler.
        """
        chain = []

        # Add gradient clipping if specified
        if clip_grad:
            chain.append(optax.clip_by_global_norm(clip_grad))

        # Add base optimizer
        chain.append(base_optimizer)

        # Add weight decay if specified
        if weight_decay != 0.0:
            chain.append(
                optax_add_scheduled_weight_decay(
                    lambda step: -scheduler(step) * weight_decay,
                    weight_decay_mask,
                )
            )

        # Chain all transformations
        tx = optax.chain(*chain)

        # Add gradient accumulation if specified
        if gradient_accumulation_steps > 1:
            tx = optax.MultiSteps(tx, gradient_accumulation_steps)

        try:
            tx = builder.build_mpmd(
                scheduler,
                optimizer=tx,
                weight_decay=weight_decay,
                weight_decay_mask=weight_decay_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad=clip_grad,
            )
        except NotImplementedError as exc:
            tx = make_unsupported_stage_local_gradient_transformation(
                tx,
                optimizer_type=optimizer_type,
                reason=str(exc),
            )

        return tx, scheduler

    @classmethod
    def generate_template(cls, optimizer_type: str) -> str:
        """
        Generate a configuration template for the specified optimizer.

        Args:
            optimizer_type (str): Name of the optimizer.

        Returns:
            str: Configuration template.

        Raises:
            ValueError: If the optimizer type is unknown.
        """
        config_cls = cls._get_config_class(optimizer_type)

        fields_list = []
        for field in dataclasses.fields(config_cls):
            field_type = tp.get_type_hints(config_cls)[field.name]
            default = f" = {field.default}" if not isinstance(field.default, dataclasses._MISSING_TYPE) else ""

            if hasattr(field_type, "__name__"):
                type_name = field_type.__name__
            else:
                type_name = str(field_type)

            fields_list.append(f"    {field.name}: {type_name}{default}")

        return f"{config_cls.__name__}(\n" + "\n".join(fields_list) + "\n)"

    @classmethod
    def serialize_config(
        cls,
        config: SerializationMixin,
        format: str = "dict",  # noqa:A002
    ) -> dict | str:
        """
        Serialize configuration to different formats.

        Args:
            config (SerializationMixin): Configuration object.
            format (str): Serialization format. Supported formats: 'dict', 'json'.

        Returns:
            Union[Dict, str]: Serialized configuration.

        Raises:
            ValueError: If the format is unsupported.
        """
        if format not in ["dict", "json"]:
            raise ValueError("Supported formats: 'dict', 'json'")

        if format == "dict":
            return config.to_dict()
        return config.to_json()

    @classmethod
    def deserialize_config(
        cls,
        optimizer_type: str,
        data: dict | str,
        format: str = "dict",  # noqa:A002
    ) -> SerializationMixin:
        """
        Deserialize configuration from different formats.

        Args:
            optimizer_type (str): Name of the optimizer.
            data (Union[Dict, str]): Serialized configuration data.
            format (str): Serialization format. Supported formats: 'dict', 'json'.

        Returns:
            SerializationMixin: Deserialized configuration object.

        Raises:
            ValueError: If the optimizer type is unknown or the format is unsupported.
            TypeError: If the input data type is invalid.
        """
        config_cls = cls._get_config_class(optimizer_type)

        if format == "json":
            if not isinstance(data, str):
                raise TypeError("Expected string input for JSON format")
            return config_cls.from_json(data)

        if format == "dict":
            if not isinstance(data, dict):
                raise TypeError("Expected dictionary input for dict format")
            return config_cls.from_dict(data)

        raise ValueError("Unsupported format")
