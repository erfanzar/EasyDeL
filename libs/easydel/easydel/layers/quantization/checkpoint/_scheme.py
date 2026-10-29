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

"""Per-layer resolution of a checkpoint's quantization scheme.

:class:`CheckpointQuantScheme` answers one question — *is this module
quantized, and if so how?* — for any dotted module path. It owns only the
generic parts: locating ``quantization_config``, selecting the adapter,
matching paths against ignore rules and targeted overrides, and enforcing
that all shards of a fused projection agree. Everything format-specific is
delegated to the adapter.

Because resolution is keyed on *paths*, it composes with the existing module
traversal used by :class:`~easydel.layers.quantization.EasyQuantizer`, and no
model family needs to know a checkpoint was quantized.
"""

from __future__ import annotations

import re
import typing as tp
from dataclasses import dataclass

from eformer.loggings import get_logger

from ._adapter import CheckpointQuantAdapter, ConfigParse, IgnoreRules, get_adapter
from ._canonical import QuantSpec, SourceFormat

logger = get_logger(__name__)

#: Config keys that may hold the quantization block, in priority order.
_QUANT_CONFIG_KEYS = ("quantization_config",)

#: Sub-config keys searched when the top level has no quantization block.
#: Multimodal checkpoints frequently nest it under the text tower.
_NESTED_CONFIG_KEYS = ("text_config", "llm_config", "language_config")


@dataclass(frozen=True, slots=True)
class ResolvedScheme:
    """Everything needed to build and load one quantized module.

    Attributes:
        path: Dotted module path this resolution applies to.
        adapter: Adapter class owning the format.
        source: On-disk format for this module.
        target: Runtime target the module will execute.
    """

    path: str
    adapter: type[CheckpointQuantAdapter]
    source: SourceFormat
    target: QuantSpec

    @property
    def source_suffixes(self) -> tuple[str, ...]:
        """Checkpoint tensor suffixes this module consumes."""
        return self.adapter.source_suffixes(self.source)


class CheckpointQuantScheme:
    """A checkpoint's quantization scheme, resolvable per module path.

    Build with :meth:`from_hf_config`; query with :meth:`for_path` (single
    module) or :meth:`for_fused` (a fused projection assembled from several
    checkpoint modules).

    Attributes:
        quant_method: The checkpoint's declared ``quant_method``.
        adapter: The adapter selected for that method.
        parse: The adapter's reading of the config block.
    """

    def __init__(
        self,
        *,
        quant_method: str,
        adapter: type[CheckpointQuantAdapter],
        parse: ConfigParse,
    ) -> None:
        """Store the parsed scheme and precompile override patterns.

        Args:
            quant_method: The checkpoint's ``quant_method`` string.
            adapter: Adapter class handling this method.
            parse: The adapter's parsed view of ``quantization_config``.
        """
        self.quant_method = quant_method
        self.adapter = adapter
        self.parse = parse
        self._overrides: tuple[tuple[re.Pattern[str], SourceFormat], ...] = tuple(
            (re.compile(pattern), source) for pattern, source in parse.overrides
        )

    @classmethod
    def from_hf_config(cls, config: tp.Any) -> CheckpointQuantScheme | None:
        """Build a scheme from a HuggingFace config object or mapping.

        Args:
            config: A ``PretrainedConfig``-like object or a plain mapping
                holding ``quantization_config``, possibly nested under a text
                sub-config as multimodal checkpoints do.

        Returns:
            The resolved scheme, or ``None`` when the checkpoint declares no
            quantization.

        Raises:
            ValueError: If a quantization block exists but names no
                ``quant_method``.
            NotImplementedError: If no adapter is registered for the declared
                method.
        """
        quant_config = _find_quant_config(config)
        if not quant_config:
            return None

        quant_method = quant_config.get("quant_method")
        if not quant_method:
            raise ValueError(
                f"Checkpoint declares a quantization_config without 'quant_method': {sorted(quant_config)}."
            )

        adapter = get_adapter(str(quant_method))
        parse = adapter.parse_config(quant_config)
        logger.debug("Resolved checkpoint quantization scheme %s via %s", quant_method, adapter.__name__)
        return cls(quant_method=str(quant_method), adapter=adapter, parse=parse)

    @property
    def ignored(self) -> IgnoreRules:
        """Layers the checkpoint leaves unquantized."""
        return self.parse.ignored

    def source_for(self, path: str) -> SourceFormat | None:
        """Return the on-disk format for ``path``, if it is quantized.

        Args:
            path: Dotted module path.

        Returns:
            The matching :class:`SourceFormat`, or ``None`` when the module is
            ignored or no rule targets it.
        """
        if self.ignored.matches(path):
            return None
        for pattern, source in self._overrides:
            if pattern.search(path):
                return source
        return self.parse.default

    def for_path(self, path: str, *, expert_dim: bool = False) -> ResolvedScheme | None:
        """Resolve one module path to a full scheme.

        Args:
            path: Dotted module path, e.g.
                ``model.layers.0.self_attn.q_proj``.
            expert_dim: ``True`` when the module holds stacked MoE experts.

        Returns:
            The :class:`ResolvedScheme`, or ``None`` when the module is not
            quantized and should keep its dense layer.
        """
        source = self.source_for(path)
        if source is None:
            return None
        target = self.adapter.target_spec(source, expert_dim=expert_dim)
        return ResolvedScheme(path=path, adapter=self.adapter, source=source, target=target)

    def for_fused(
        self,
        path: str,
        shard_paths: tp.Sequence[str],
        *,
        expert_dim: bool = False,
    ) -> ResolvedScheme | None:
        """Resolve a fused projection whose shards are separate checkpoint modules.

        A fused ``qkv_proj`` is assembled from the checkpoint's ``q_proj`` /
        ``k_proj`` / ``v_proj``. Those shards must all be quantized the same
        way: mixing precisions inside one packed weight is not representable
        and would otherwise corrupt silently at load.

        Args:
            path: Dotted path of the fused module in the EasyDeL tree.
            shard_paths: Dotted paths of the checkpoint modules being fused.
            expert_dim: ``True`` when the module holds stacked MoE experts.

        Returns:
            The :class:`ResolvedScheme` for the fused module, or ``None`` when
            every shard is unquantized.

        Raises:
            ValueError: If the shards disagree — some quantized and some not,
                or quantized under different formats.
        """
        if not shard_paths:
            return self.for_path(path, expert_dim=expert_dim)

        sources = {shard: self.source_for(shard) for shard in shard_paths}
        distinct = {source for source in sources.values()}
        if len(distinct) != 1:
            detail = ", ".join(
                f"{shard}={'unquantized' if source is None else source.quant_method}"
                for shard, source in sources.items()
            )
            raise ValueError(
                f"Fused module {path!r} draws from shards with disagreeing quantization schemes ({detail}). "
                f"All shards of a fused projection must share one scheme."
            )

        source = distinct.pop()
        if source is None:
            return None
        target = self.adapter.target_spec(source, expert_dim=expert_dim)
        return ResolvedScheme(path=path, adapter=self.adapter, source=source, target=target)

    def __repr__(self) -> str:
        """Return a short debug representation naming the method and adapter."""
        return (
            f"{type(self).__name__}(quant_method={self.quant_method!r}, "
            f"adapter={self.adapter.__name__}, overrides={len(self._overrides)})"
        )


def _find_quant_config(config: tp.Any) -> tp.Mapping[str, tp.Any] | None:
    """Locate a ``quantization_config`` mapping on a config object or dict.

    Searches the top level first, then the text/language sub-configs used by
    multimodal checkpoints.

    Args:
        config: Config object or mapping to search. ``None`` yields ``None``.

    Returns:
        The quantization mapping, or ``None`` when absent or empty.
    """
    if config is None:
        return None

    for key in _QUANT_CONFIG_KEYS:
        found = _get_field(config, key)
        if found:
            return _as_mapping(found)

    for nested_key in _NESTED_CONFIG_KEYS:
        nested = _get_field(config, nested_key)
        if nested is None:
            continue
        for key in _QUANT_CONFIG_KEYS:
            found = _get_field(nested, key)
            if found:
                return _as_mapping(found)
    return None


def _get_field(container: tp.Any, key: str) -> tp.Any:
    """Read ``key`` from a mapping or an attribute-style config object.

    Args:
        container: Mapping or object to read from.
        key: Field name.

    Returns:
        The value, or ``None`` when missing.
    """
    if isinstance(container, tp.Mapping):
        return container.get(key)
    return getattr(container, key, None)


def _as_mapping(value: tp.Any) -> tp.Mapping[str, tp.Any]:
    """Coerce a quantization config value into a mapping.

    HuggingFace stores this either as a plain ``dict`` or as a config object
    exposing ``to_dict()``.

    Args:
        value: The raw quantization config value.

    Returns:
        A mapping view of the config.

    Raises:
        TypeError: If the value is neither a mapping nor ``to_dict``-able.
    """
    if isinstance(value, tp.Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"quantization_config must be a mapping or expose to_dict(), got {type(value).__name__}.")
