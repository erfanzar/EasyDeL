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

"""Configuration dataclass for Rotary Position Embeddings (RoPE).

This module provides a dataclass for storing and managing RoPE configuration
parameters. The RopeConfig class supports various RoPE scaling methods and
can be created from dictionaries (e.g., HuggingFace config dicts).

Classes:
    RopeConfig: Dataclass storing RoPE type and scaling parameters.

Example:
    >>> from easydel.layers.rotary import RopeConfig
    >>> # Create config from dictionary (e.g., from HuggingFace config)
    >>> config_dict = {
    ...     "rope_type": "yarn",
    ...     "factor": 2.0,
    ...     "original_max_position_embeddings": 2048,
    ...     "beta_fast": 32,
    ...     "beta_slow": 1,
    ... }
    >>> rope_config = RopeConfig.from_dict(config_dict)
    >>> # Convert back to dictionary for use with get_rope/get_frequencies
    >>> scaling_dict = rope_config.to_dict()
"""

from __future__ import annotations

import typing as tp

import chex


@chex.dataclass
class RopeConfig:
    """
    Configuration class for RoPE (Rotary Position Embedding) parameters.

    Stores the configuration related to RoPE type and its scaling parameters,
    making it easy to manage and pass around RoPE settings.

    Attributes:
        rope_type (str): The type of RoPE scaling to use (e.g., "default", "linear", "yarn", "llama3").
                         Defaults to "default".
        factor (tp.Optional[float]): General scaling factor used by some types (linear, dynamic, yarn, llama3).
        low_freq_factor (tp.Optional[float]): Specific factor for Llama3 scaling.
        high_freq_factor (tp.Optional[float]): Specific factor for Llama3 scaling.
        original_max_position_embeddings (tp.Optional[int]): Original context window size,
                                                            required by some scaling methods
                                                            (yarn, llama3, phi3, deepseek).
        long_factor (tp.Optional[float]): Specific factor for Phi3 LongRoPE scaling (used for lengths > original).
        short_factor (tp.Optional[float]): Specific factor for Phi3 LongRoPE scaling (used for lengths <= original).
        long_mscale (tp.Optional[float]): Potentially used by variants like Phi3. (Not used in current `get_rope`).
        short_mscale (tp.Optional[float]): Potentially used by variants like Phi3. (Not used in current `get_rope`).
        # Add other potential scaling parameters here as needed (e.g., from YaRN, Deepseek)
        extrapolation_factor (tp.Optional[float]): YaRN/Deepseek parameter.
        attn_factor (tp.Optional[float]): YaRN/Deepseek parameter.
        beta_fast (tp.Optional[int]): YaRN/Deepseek parameter.
        beta_slow (tp.Optional[int]): YaRN/Deepseek parameter.
        mscale (tp.Optional[float]): Deepseek parameter.
        mscale_all_dim (tp.Optional[float]): Deepseek parameter.
    """

    rope_type: str = "default"
    factor: float | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    original_max_position_embeddings: int | None = None
    long_factor: float | None = None
    short_factor: float | None = None
    long_mscale: float | None = None
    short_mscale: float | None = None
    extrapolation_factor: float | None = None
    attn_factor: float | None = None
    beta_fast: int | None = None
    beta_slow: int | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None
    mrope_interleaved: bool | None = None
    mrope_section: list[int] | None = None
    repetition_style: bool | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, tp.Any]) -> RopeConfig:
        """
        Create a RopeConfig instance from a dictionary.

        Handles potential alias 'type' for 'rope_type'.

        Args:
            config_dict (tp.Dict[str, tp.Any]): Dictionary containing RoPE configuration.

        Returns:
            RopeConfig: An instance populated from the dictionary.
        """

        rope_type = config_dict.get("rope_type")
        if rope_type is None:
            rope_type = config_dict.get("type", "default")

        factor = config_dict.get("factor")
        if factor is None:
            factor = config_dict.get("scaling_factor")

        return cls(
            rope_type=rope_type,
            factor=factor,
            low_freq_factor=config_dict.get("low_freq_factor"),
            high_freq_factor=config_dict.get("high_freq_factor"),
            original_max_position_embeddings=config_dict.get("original_max_position_embeddings"),
            long_factor=config_dict.get("long_factor"),
            short_factor=config_dict.get("short_factor"),
            long_mscale=config_dict.get("long_mscale"),
            short_mscale=config_dict.get("short_mscale"),
            extrapolation_factor=config_dict.get("extrapolation_factor"),
            attn_factor=config_dict.get("attn_factor"),
            beta_fast=config_dict.get("beta_fast"),
            beta_slow=config_dict.get("beta_slow"),
            mscale=config_dict.get("mscale"),
            mscale_all_dim=config_dict.get("mscale_all_dim"),
            mrope_interleaved=config_dict.get("mrope_interleaved"),
            mrope_section=config_dict.get("mrope_section"),
            repetition_style=config_dict.get("repetition_style"),
        )

    def update(
        self,
        config_dict: tp.Mapping[str, tp.Any] | RopeConfig | None = None,
        /,
        **kwargs: tp.Any,
    ) -> None:
        """Update the RopeConfig instance in-place with new values.

        Supports HuggingFace-style aliases for compatibility:
        - ``type`` -> ``rope_type``
        - ``scaling_factor`` -> ``factor``

        Also handles nested ``rope_scaling`` dictionaries commonly found in
        HuggingFace model configurations.

        Args:
            config_dict: A mapping, RopeConfig instance, or None containing
                configuration values to update. Positional-only argument.
            **kwargs: Additional keyword arguments to update. These take
                precedence over values in config_dict.

        Example:
            >>> config = RopeConfig()
            >>> config.update({"rope_type": "yarn", "factor": 2.0})
            >>> config.update(scaling_factor=4.0)  # Uses alias
            >>> config.factor
            4.0
        """
        updates: dict[str, tp.Any]
        if config_dict is None:
            updates = {}
        elif isinstance(config_dict, RopeConfig):
            updates = dict(config_dict.to_dict())
        else:
            updates = dict(config_dict)
        if kwargs:
            updates.update(kwargs)

        # Allow passing a full config dict containing a nested `rope_scaling` dict.
        rope_scaling = updates.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            updates.pop("rope_scaling", None)
            updates = {**rope_scaling, **updates}

        if "rope_type" not in updates and "type" in updates:
            updates["rope_type"] = updates.pop("type")
        else:
            updates.pop("type", None)

        if "factor" not in updates and "scaling_factor" in updates:
            updates["factor"] = updates.pop("scaling_factor")
        else:
            updates.pop("scaling_factor", None)

        valid_keys = set(self.__dataclass_fields__)
        for key, value in updates.items():
            if key in valid_keys:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, tp.Any]:
        """Convert the RopeConfig instance to a dictionary.

        Creates a dictionary containing only non-None configuration values.
        The returned dictionary uses a custom hashable subclass to support
        use as a static argument in JAX JIT compilation contexts.

        Returns:
            A hashable dictionary containing non-None configuration values.
            The dictionary can be passed directly to functions like
            `get_rope` and `get_frequencies` as the `rope_scaling` argument.

        Example:
            >>> config = RopeConfig(rope_type="yarn", factor=2.0)
            >>> scaling_dict = config.to_dict()
            >>> scaling_dict
            {'rope_type': 'yarn', 'factor': 2.0}
        """
        from easydel.utils.compiling_utils import hash_fn

        class rope_scaling(dict):
            """A dictionary subclass that is hashable."""

            __hash__ = hash_fn

        scale = rope_scaling({k: v for k, v in self.__dict__.items() if v is not None})
        return scale
