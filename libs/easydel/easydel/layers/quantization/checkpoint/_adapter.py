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

"""Adapter contract for externally quantized checkpoint formats.

A checkpoint format is supported by writing **one** :class:`CheckpointQuantAdapter`
subclass and registering it. The adapter owns every format-specific decision —
how to read the checkpoint's ``quantization_config``, which tensor suffixes it
consumes, which runtime target it maps onto, and how to convert (and invert)
the tensors. Everything else — path matching, rule generation, layer swapping,
sharding, kernels — is shared.

If a format needs anything beyond an adapter (a layer class, a forward-path
branch, a model edit), the seam in :mod:`._canonical` is wrong and should be
fixed there rather than worked around here.

Adapters compose *primitives*, they do not implement them. Bit-unpacking,
e8m0/e4m3 decoding, block dequantization and requantization live once in
``ejkernel.quantization``; an adapter that reimplements one is a review
failure.
"""

from __future__ import annotations

import abc
import re
import typing as tp
from dataclasses import dataclass, field

from eformer.loggings import get_logger

from easydel.utils.registery import Registry

from ._canonical import CanonicalQuantizedWeight, QuantSpec, SourceFormat

if tp.TYPE_CHECKING:
    from jax import Array

logger = get_logger(__name__)

#: Registry category holding every checkpoint-quantization adapter.
CHECKPOINT_QUANT_CATEGORY = "checkpoint-quant"


@dataclass(frozen=True, slots=True)
class IgnoreRules:
    """Layers a checkpoint declares as *not* quantized.

    Checkpoints express this two different ways and the distinction is
    load-bearing, so both are kept rather than collapsed:

    * ``ignored_layers`` (compressed-tensors, fp8) lists **exact leaf module
      names** such as ``model.layers.0.self_attn.q_proj``.
    * ``modules_to_not_convert`` (AWQ, MXFP4) uses HF's matching (transformers
      ``should_convert_module``): an entry excludes a path when it names a
      container the path sits beneath, a **suffix** of the path (Mixtral's
      AWQ checkpoints ship ``"gate"`` for every ``block_sparse_moe.gate``),
      or matches as a start-anchored regex — which is how gpt-oss's
      glob-style entries (``model.layers.*.self_attn``) work.

    Matching one with the other's semantics silently quantizes layers the
    checkpoint never quantized — and then *demands* quantized tensors the
    checkpoint never wrote — which shows up as garbage logits or a failed
    load far from the cause.

    compressed-tensors additionally allows ``re:`` prefixed regexes in its
    ``ignore`` list, which is the third form.

    Attributes:
        exact: Fully-qualified module paths that must match exactly.
        prefixes: ``modules_to_not_convert`` entries, matched with HF's
            semantics (containment / suffix / start-anchored regex).
        patterns: Regexes searched against the path.
    """

    exact: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()

    def matches(self, path: str) -> bool:
        """Whether ``path`` is excluded from quantization.

        Args:
            path: Dotted module path, e.g. ``model.layers.0.self_attn.q_proj``.

        Returns:
            ``True`` when the path is listed exactly, sits under an ignored
            container, or matches an ignore regex.
        """
        if path in self.exact:
            return True
        if any(_module_pattern_matches(entry, path) for entry in self.prefixes):
            return True
        return any(re.search(pattern, path) for pattern in self.patterns)

    def merged_with(self, other: IgnoreRules) -> IgnoreRules:
        """Union this rule set with another.

        Args:
            other: Rules to merge in.

        Returns:
            A new :class:`IgnoreRules` covering both.
        """
        return IgnoreRules(
            exact=self.exact | other.exact,
            prefixes=(*self.prefixes, *other.prefixes),
            patterns=(*self.patterns, *other.patterns),
        )


def _module_pattern_matches(entry: str, path: str) -> bool:
    """HF-compatible ``modules_to_not_convert`` matching for one entry.

    Mirrors transformers' ``should_convert_module``: an entry excludes
    ``path`` when

    1. ``path`` ends with the entry verbatim (``"gate"`` excludes
       ``model.layers.0.block_sparse_moe.gate`` — how Mixtral-AWQ protects
       its routers);
    2. the entry, read as a start-anchored regex, covers a dot-terminated
       prefix of the path (``"model.vision_tower"`` excludes everything
       beneath the tower; ``"model.layers.*.self_attn"`` — gpt-oss's
       glob-style form — excludes every attention projection); or
    3. the entry, read as a start-anchored regex, matches the path itself.

    Entries that are not valid regexes fall back to literal container
    matching (exact name or dotted-prefix).

    Args:
        entry: One ``modules_to_not_convert`` entry.
        path: Dotted module path being resolved.

    Returns:
        ``True`` when the entry excludes the path.
    """
    if path.endswith(entry):
        return True
    try:
        return bool(re.match(f"{entry}\\.", path) or re.match(entry, path))
    except re.error:
        return path == entry or path.startswith(f"{entry}.")


@dataclass(frozen=True, slots=True)
class ConfigParse:
    """An adapter's reading of a checkpoint's ``quantization_config``.

    Attributes:
        default: Format applied to every non-ignored layer, or ``None`` when
            the checkpoint only quantizes explicitly targeted layers.
        overrides: Ordered ``(pattern, format)`` pairs, consulted before
            :attr:`default`. Patterns are regexes matched against the dotted
            module path — this is how compressed-tensors' ``config_groups``
            target different schemes at different layers.
        ignored: Layers the checkpoint leaves unquantized.
    """

    default: SourceFormat | None = None
    overrides: tuple[tuple[str, SourceFormat], ...] = ()
    ignored: IgnoreRules = field(default_factory=IgnoreRules)


class CheckpointQuantAdapter(abc.ABC):
    """Base class for one on-disk quantized weight format.

    Subclasses are registered under the checkpoint's ``quant_method`` string
    and are used purely as namespaces — every hook is a ``classmethod`` and no
    instance is ever created.

    Class Attributes:
        quant_methods: The ``quant_method`` values this adapter claims.
            Registered as lookup aliases by :func:`register_adapter`.
    """

    quant_methods: tp.ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abc.abstractmethod
    def parse_config(cls, quant_config: tp.Mapping[str, tp.Any]) -> ConfigParse:
        """Read the checkpoint's ``quantization_config`` block.

        Args:
            quant_config: The raw ``quantization_config`` mapping from the
                checkpoint's ``config.json``.

        Returns:
            The parsed per-format view used to resolve individual layers.

        Raises:
            ValueError: If the config is malformed or declares a scheme this
                adapter does not implement.
        """

    @classmethod
    @abc.abstractmethod
    def target_spec(cls, source: SourceFormat, *, expert_dim: bool = False) -> QuantSpec:
        """Choose the runtime target for a given on-disk format.

        This is where requantization policy lives. A source with no matching
        ejkernel mode (e.g. ``128x128`` block-scaled fp8) must map onto one
        that exists; :meth:`to_canonical` then performs the conversion.

        Args:
            source: The on-disk format for this layer.
            expert_dim: ``True`` when the weight carries a leading expert
                axis (stacked MoE experts).

        Returns:
            The :class:`QuantSpec` the runtime will execute.
        """

    @classmethod
    @abc.abstractmethod
    def source_suffixes(cls, source: SourceFormat) -> tuple[str, ...]:
        """Checkpoint tensor suffixes consumed for one weight.

        Order is significant: :meth:`to_canonical` receives the tensors keyed
        by these suffixes, and reform rules list them as ``sources`` in this
        order.

        Args:
            source: The on-disk format for this layer.

        Returns:
            Suffixes relative to the layer prefix, e.g.
            ``("weight", "weight_scale_inv")``.
        """

    @classmethod
    @abc.abstractmethod
    def to_canonical(
        cls,
        tensors: tp.Mapping[str, Array],
        *,
        source: SourceFormat,
        target: QuantSpec,
    ) -> CanonicalQuantizedWeight:
        """Convert checkpoint tensors into the canonical representation.

        Args:
            tensors: Checkpoint tensors keyed by the suffixes returned from
                :meth:`source_suffixes`.
            source: The on-disk format.
            target: The runtime target chosen by :meth:`target_spec`.

        Returns:
            The canonical weight the layer will hold.
        """

    @classmethod
    def from_canonical(
        cls,
        weight: CanonicalQuantizedWeight,
        *,
        source: SourceFormat,
        target: QuantSpec,
    ) -> tuple[Array, ...]:
        """Invert :meth:`to_canonical` for checkpoint export.

        Optional. Formats whose target differs from their source may be
        genuinely non-invertible (requantization loses the original codes); in
        that case leaving this unimplemented is correct, and export raises
        with a clear message instead of writing a silently wrong checkpoint.

        Args:
            weight: The canonical weight to serialize back.
            source: The on-disk format to reproduce.
            target: The runtime target the weight is held in.

        Returns:
            Tensors in the order given by :meth:`source_suffixes`.

        Raises:
            NotImplementedError: If this format cannot be re-serialized.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement checkpoint export. "
            f"Exporting a model loaded from a {source.quant_method!r} checkpoint requires "
            f"either dequantizing first or implementing from_canonical()."
        )


def register_adapter(
    *quant_methods: str,
    overwrite: bool = False,
) -> tp.Callable[[type[CheckpointQuantAdapter]], type[CheckpointQuantAdapter]]:
    """Register a :class:`CheckpointQuantAdapter` under its ``quant_method`` names.

    Args:
        *quant_methods: Checkpoint ``quant_method`` strings this adapter
            handles. Defaults to the class's own ``quant_methods`` attribute
            when omitted.
        overwrite: Replace an existing registration instead of raising.

    Returns:
        A class decorator that records the adapter and returns it unchanged.

    Example:
        >>> @register_adapter("fp8")
        ... class Fp8Adapter(CheckpointQuantAdapter):
        ...     ...
    """

    def decorator(adapter_cls: type[CheckpointQuantAdapter]) -> type[CheckpointQuantAdapter]:
        """Record ``adapter_cls`` in the global registry.

        Args:
            adapter_cls: The adapter class being registered.

        Returns:
            ``adapter_cls`` unchanged.

        Raises:
            TypeError: If the class is not a :class:`CheckpointQuantAdapter`.
            ValueError: If no ``quant_method`` names are available.
        """
        if not issubclass(adapter_cls, CheckpointQuantAdapter):
            raise TypeError(f"{adapter_cls.__name__} must subclass CheckpointQuantAdapter to be registered.")
        names = tuple(quant_methods) or adapter_cls.quant_methods
        if not names:
            raise ValueError(f"{adapter_cls.__name__} must declare `quant_methods` or pass names to register_adapter().")
        adapter_cls.quant_methods = names
        Registry.register(
            CHECKPOINT_QUANT_CATEGORY,
            list(names),
            metadata={"quant_methods": tuple(names)},
            overwrite=overwrite,
        )(adapter_cls)
        logger.debug("Registered checkpoint quantization adapter %s for %s", adapter_cls.__name__, names)
        return adapter_cls

    return decorator


def get_adapter(quant_method: str) -> type[CheckpointQuantAdapter]:
    """Look up the adapter registered for a checkpoint ``quant_method``.

    Args:
        quant_method: The checkpoint's ``quant_method`` string.

    Returns:
        The registered adapter class.

    Raises:
        NotImplementedError: If no adapter claims this method, listing what is
            available so the message is actionable.
    """
    adapter = Registry.get(CHECKPOINT_QUANT_CATEGORY, quant_method)
    if adapter is None:
        available = sorted(Registry.list_implementations(CHECKPOINT_QUANT_CATEGORY))
        raise NotImplementedError(
            f"No checkpoint quantization adapter registered for quant_method={quant_method!r}. "
            f"Available: {available or '(none)'}."
        )
    return tp.cast(type[CheckpointQuantAdapter], adapter)


def available_quant_methods() -> list[str]:
    """List every ``quant_method`` with a registered adapter.

    Returns:
        Sorted list of supported ``quant_method`` strings.
    """
    return sorted(Registry.list_implementations(CHECKPOINT_QUANT_CATEGORY))
