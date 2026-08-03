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

"""Turn a resolved checkpoint scheme into ``reform_param`` rules.

Loading a quantized checkpoint is an N-to-M problem: several checkpoint
tensors (``weight`` + ``weight_scale`` + optional zero-points) become several
EasyDeL parameters (``quant_kernel`` + ``quant_scales`` + optional
``quant_biases``). EasyDeL's existing ``reform_param`` machinery already
expresses both halves — ``sources``/``fuser`` collapses N checkpoint tensors
into one value, and ``splits``/``spliter`` expands one value into M
parameters — so quantized loading needs **no new loader**: it reuses the same
path that fuses QKV projections today.

The intermediate value carried between the two halves is a
:class:`~._canonical.CanonicalQuantizedWeight`; each split simply projects one
field out of it.

Rules generated here are hosted the same way the fused-projection rules are
(see ``modeling_qwen3.py``'s ``reform_param`` property): a module exposes them
and :meth:`~easydel.infra.base_module.EasyDeLBaseModule._get_reform_param`
walks the tree, prefixing rule keys, ``sources`` and split names with the
owning module's path.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from ._canonical import ActivationQuantKind, CanonicalQuantizedWeight, QuantSpec, SourceFormat

if tp.TYPE_CHECKING:
    from jax import Array

    from ._adapter import CheckpointQuantAdapter
    from ._scheme import ResolvedScheme


def canonical_param_names(target: QuantSpec, source: SourceFormat) -> tuple[str, ...]:
    """List the EasyDeL parameter names a quantized weight expands into.

    Determined statically from the scheme — split rules must be declared
    before any tensor is seen — and kept in sync with
    :meth:`~._canonical.CanonicalQuantizedWeight.arrays`.

    Args:
        target: The runtime quantization target.
        source: The on-disk format, which decides whether a per-tensor global
            scale is carried.

    Returns:
        Parameter names in a stable order.
    """
    names = ["quant_kernel", "quant_scales"]
    if target.needs_biases:
        names.append("quant_biases")
    if source.has_global_scale:
        names.append("output_scale")
    if target.activation.kind is ActivationQuantKind.STATIC:
        names.append("activation_scale")
    return tuple(names)


def _join(prefix: str, name: str) -> str:
    """Join a dotted prefix and a name, tolerating an empty prefix.

    Args:
        prefix: Dotted prefix, possibly ``""`` for self-hosted rules.
        name: Leaf name.

    Returns:
        ``"prefix.name"``, or ``"name"`` when the prefix is empty.
    """
    return f"{prefix}.{name}" if prefix else name


def _to_jax(tensor: tp.Any) -> Array:
    """Convert a checkpoint tensor to a JAX array without changing its dtype.

    Delegates to the shared converter so the dtype-preserving transfer lives
    in exactly one place — the same helper the load path uses when it honors
    a rule's ``preserve_dtype`` flag.

    Args:
        tensor: A torch tensor, NumPy array or JAX array.

    Returns:
        The value as a JAX array with its original dtype.
    """
    from easydel.utils.parameters_transformation import TensorConverter

    return TensorConverter.to_jax_preserving_dtype(tensor)


@dataclass(frozen=True, slots=True)
class ToCanonical:
    """``fuser`` callable: checkpoint tensors -> canonical quantized weight.

    Attributes:
        adapter: Adapter owning the format.
        source: On-disk format for this weight.
        target: Runtime target to produce.
        suffixes: Source suffixes, in the order the fuser receives tensors.
    """

    adapter: type[CheckpointQuantAdapter]
    source: SourceFormat
    target: QuantSpec
    suffixes: tuple[str, ...]

    def __call__(self, *tensors: tp.Any) -> CanonicalQuantizedWeight:
        """Convert positional checkpoint tensors into the canonical form.

        Args:
            *tensors: One tensor per entry of :attr:`suffixes`, in order, as
                supplied by the reform machinery.

        Returns:
            The canonical quantized weight.

        Raises:
            ValueError: If the number of tensors does not match the declared
                source suffixes.
        """
        if len(tensors) != len(self.suffixes):
            raise ValueError(
                f"{self.adapter.__name__} expected {len(self.suffixes)} source tensors "
                f"{self.suffixes}, got {len(tensors)}."
            )
        payload = {suffix: _to_jax(tensor) for suffix, tensor in zip(self.suffixes, tensors, strict=True)}
        return self.adapter.to_canonical(payload, source=self.source, target=self.target)


@dataclass(frozen=True, slots=True)
class ExtractField:
    """``spliter`` callable: project one parameter out of a canonical weight.

    Attributes:
        name: Parameter name to extract.
    """

    name: str

    def __call__(self, weight: CanonicalQuantizedWeight) -> Array:
        """Return the named array from a canonical weight.

        Args:
            weight: The fused canonical value produced by :class:`ToCanonical`.

        Returns:
            The requested array.

        Raises:
            TypeError: If the fused value is not a canonical weight, which
                means the fusion half of the rule did not run.
            KeyError: If the adapter omitted a parameter the rule declared.
        """
        if not isinstance(weight, CanonicalQuantizedWeight):
            raise TypeError(
                f"Split {self.name!r} expected a CanonicalQuantizedWeight, got {type(weight).__name__}. "
                f"This means the rule's fuser did not run before its splits."
            )
        arrays = weight.arrays()
        if self.name not in arrays:
            raise KeyError(
                f"Adapter produced no {self.name!r}, but the reform rule declared it. "
                f"Available: {sorted(arrays)}."
            )
        return arrays[self.name]


@dataclass(frozen=True, slots=True)
class FromCanonical:
    """``inverse_fuser`` callable used when exporting back to checkpoint form.

    Attributes:
        adapter: Adapter owning the format.
        source: On-disk format to reproduce.
        target: Runtime target the weight is held in.
        sources: Fully-qualified source keys, in order.
    """

    adapter: type[CheckpointQuantAdapter]
    source: SourceFormat
    target: QuantSpec
    sources: tuple[str, ...]

    def __call__(self, weight: CanonicalQuantizedWeight) -> dict[str, Array]:
        """Recover checkpoint tensors from a canonical weight.

        Args:
            weight: The canonical weight to serialize back.

        Returns:
            Mapping of source key to tensor.

        Raises:
            NotImplementedError: If the adapter cannot invert its conversion.
        """
        tensors = self.adapter.from_canonical(weight, source=self.source, target=self.target)
        return dict(zip(self.sources, tensors, strict=True))


@dataclass(frozen=True, slots=True)
class RebuildCanonical:
    """``inverse_spliter`` placeholder for the export direction.

    The load direction is complete; export of a checkpoint-quantized model is
    not implemented yet because the M-parameters-to-one-value contract for
    ``inverse_spliter`` has no consumer to validate against. Raising here is
    deliberate: the reform schema requires the callable to exist, and a
    guessed implementation would write silently wrong checkpoints.

    Attributes:
        adapter: Adapter owning the format, named in the error message.
        quant_method: Checkpoint method, named in the error message.
    """

    adapter: type[CheckpointQuantAdapter]
    quant_method: str

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Never:
        """Always raise: exporting checkpoint-quantized weights is unsupported.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"Exporting a model loaded from a {self.quant_method!r} checkpoint is not supported yet "
            f"(adapter={self.adapter.__name__}). Dequantize the model before exporting."
        )


def checkpoint_quant_reform_param(
    scheme: ResolvedScheme,
    *,
    target_prefix: str = "",
    source_prefix: str | None = None,
    log_label: str | None = None,
) -> dict[str, dict[str, tp.Any]]:
    """Build the ``reform_param`` rule that loads one quantized weight.

    The generated rule performs both halves in one entry: it fuses the
    checkpoint's tensors into a canonical weight, then splits that weight into
    the quantized layer's parameters.

    Names are relative to whichever module hosts the rule, matching the
    convention used by the fused-projection builders in
    :mod:`easydel.layers.layouts`:

    * ``target_prefix=""`` — the quantized layer hosts the rule itself, so
      names resolve against its own path. This is the case for a plain
      (non-fused) projection.
    * ``target_prefix="gate_up_proj"`` — the parent module hosts the rule, so
      sibling names resolve correctly.

    The rule is keyed on the *target* parameter (``quant_kernel``), not on a
    checkpoint suffix. That is required, not cosmetic: fusion is skipped when
    the fused key is already present in the state dict, so keying a
    self-hosted rule on ``weight`` — which is also one of its own sources —
    would silently never fire.

    Args:
        scheme: The resolved scheme for this module.
        target_prefix: Prefix for the rule key and split names.
        source_prefix: Prefix for checkpoint source keys. Defaults to
            ``target_prefix``; pass it explicitly when the checkpoint's module
            name differs from EasyDeL's.
        log_label: Human-readable label reported in fusion counts.

    Returns:
        A ``reform_param`` mapping with a single rule, ready to merge into a
        module's ``reform_param``.

    Raises:
        ValueError: If the adapter declares no source suffixes, declares
            duplicates, or declares a source that collides with the rule key.
    """
    suffixes = scheme.source_suffixes
    if not suffixes:
        raise ValueError(f"{scheme.adapter.__name__}.source_suffixes() returned no suffixes for {scheme.path!r}.")
    if len(set(suffixes)) != len(suffixes):
        raise ValueError(f"{scheme.adapter.__name__}.source_suffixes() returned duplicates: {suffixes}.")

    src_prefix = target_prefix if source_prefix is None else source_prefix
    param_names = canonical_param_names(scheme.target, scheme.source)
    rule_key = _join(target_prefix, param_names[0])
    sources = tuple(_join(src_prefix, suffix) for suffix in suffixes)

    if rule_key in sources:
        raise ValueError(
            f"{scheme.adapter.__name__} declares a source {rule_key!r} that collides with the rule key; "
            f"the fusion would be skipped because the target already exists in the checkpoint."
        )

    return {
        rule_key: {
            "sources": sources,
            "fuser": ToCanonical(
                adapter=scheme.adapter,
                source=scheme.source,
                target=scheme.target,
                suffixes=suffixes,
            ),
            "inverse_fuser": FromCanonical(
                adapter=scheme.adapter,
                source=scheme.source,
                target=scheme.target,
                sources=sources,
            ),
            "splits": [{"name": _join(target_prefix, name), "spliter": ExtractField(name)} for name in param_names],
            "inverse_spliter": RebuildCanonical(
                adapter=scheme.adapter,
                quant_method=scheme.source.quant_method,
            ),
            "preserve_dtype": True,
            "log_label": log_label or f"{scheme.source.quant_method} quantized weights",
        }
    }
