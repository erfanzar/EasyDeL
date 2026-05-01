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

import collections.abc
import typing as tp

import chex  # pyright: ignore[reportMissingTypeStubs]


@chex.dataclass
class RopeConfig:
    """Discriminated union of every RoPE scaling-method configuration EasyDeL supports.

    A single dataclass deliberately covers all rotary-embedding variants
    (default, linear, dynamic NTK, YaRN, Llama-3, Phi-3 long-RoPE, Deepseek
    YaRN, multimodal MRoPE) so that model configs can carry one field —
    ``rope_scaling`` — and downstream :func:`get_rope` / :func:`get_frequencies`
    dispatchers can branch on ``rope_type`` while pulling only the fields
    they need. Fields that don't apply to the active scaling method are
    simply left as ``None`` and dropped by :meth:`to_dict`.

    The HuggingFace ecosystem uses two different field names depending on the
    model (``type`` vs ``rope_type``, ``scaling_factor`` vs ``factor``);
    :meth:`from_dict` and :meth:`update` normalize both forms so loaders
    don't need per-architecture branches.

    Attributes:
        rope_type (str): Discriminator that selects the scaling routine in
            :mod:`easydel.layers.rotary._compute_fns`. One of
            ``"default" | "linear" | "dynamic" | "yarn" | "llama3" | "phi3" |
            "longrope" | "deepseek_yarn" | "mrope"``. Defaults to
            ``"default"`` (vanilla RoPE, no scaling).
        factor (float | None): Generic scaling factor. For ``"linear"`` it
            divides the position index; for ``"dynamic"`` it scales the base;
            for ``"yarn"`` and ``"deepseek_yarn"`` it is the target
            context-length multiplier; for ``"llama3"`` it gates the
            piecewise frequency rescale. Required by every method except
            ``"default"`` and ``"phi3"``.
        low_freq_factor (float | None): Llama-3 piecewise frequency cutoff
            (low-frequency boundary in wavelengths-per-original-context).
        high_freq_factor (float | None): Llama-3 piecewise frequency cutoff
            (high-frequency boundary).
        original_max_position_embeddings (int | None): Original training
            context length. Required by every length-extending method
            (``yarn``, ``llama3``, ``phi3`` / ``longrope``, ``deepseek_yarn``)
            so the scaler can compute the *ratio* between deployed and
            trained context.
        long_factor (float | None): Phi-3 LongRoPE per-frequency scaling
            list applied when the runtime sequence exceeds
            ``original_max_position_embeddings``.
        short_factor (float | None): Phi-3 LongRoPE per-frequency scaling
            list applied within the original context window.
        long_mscale (float | None): Phi-3 / DeepSeek post-rotation magnitude
            scale applied for long contexts (``sqrt(1 + log(s)/log(orig))``).
        short_mscale (float | None): Phi-3 / DeepSeek magnitude scale used
            for short contexts.
        extrapolation_factor (float | None): YaRN extrapolation mix factor
            (interpolates between linear-PI and NTK regimes).
        attn_factor (float | None): YaRN attention temperature multiplier
            applied after rotation; compensates for log-scaled softmax.
        beta_fast (int | None): YaRN/DeepSeek "fast" boundary in
            wavelengths-per-original-context (rotations beyond which we
            extrapolate as-is).
        beta_slow (int | None): YaRN/DeepSeek "slow" boundary
            (rotations within which we apply linear-PI scaling).
        mscale (float | None): DeepSeek magnitude scale parameter (default 1.0).
        mscale_all_dim (float | None): DeepSeek magnitude scale applied to
            every head-dim coordinate (vs. only the rotated half).
        mrope_interleaved (bool | None): Multimodal-RoPE flag: when ``True``
            the temporal/height/width axes are interleaved coordinate-wise,
            when ``False`` (the Qwen2-VL default) they are concatenated
            into contiguous halves.
        mrope_section (list[int] | None): Multimodal-RoPE per-axis section
            sizes, e.g. ``[t, h, w]`` summing to ``head_dim // 2``.
        repetition_style (bool | None): Selects between Llama-style
            ``[cos, sin]`` interleaving and GPT-NeoX-style halved layout
            for the trailing rotation; consumed by
            :func:`apply_rotary_emb`.
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
        config_dict: collections.abc.Mapping[str, tp.Any] | RopeConfig | None = None,
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
