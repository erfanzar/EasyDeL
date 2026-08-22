# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-path rules: what gets quantized, where, and how.

Quantizing a model here is a graph edit, not a call-time interception.
:func:`quantize_model` walks the live module tree with
:func:`spectrax.iter_modules`, matches each module's canonical path
against an ordered list of :class:`QuantRule`, and stamps the winning
rules onto the module as a :class:`QuantPlan`. Layers then read their own
plan at their matmul site via :func:`rule_for`.

Doing the match *statically* — once, before the first trace — is what
lets this work at all on spectrax:

* No module-path stack has to exist during tracing, and no global JAX
  function has to be monkey-patched (the approach Qwix is forced into by
  Flax, which resolves paths from the live module stack).
* The plan is stored under a leading-underscore attribute, so it lands in
  the module's opaque-attribute map and survives
  :func:`~spectrax.export` / :func:`~spectrax.bind`. It therefore rides
  the ``GraphDef`` through ``jit``, ``grad``, ``scan``, ``remat`` and the
  MPMD split without any further plumbing.
* Because it is part of the ``GraphDef``, ``Module.structure_hash()``
  changes when a model is quantized, so compile caches key on the
  quantization config automatically instead of silently reusing an
  executable built for a different numeric regime.

Rule matching mirrors Qwix's semantics so that MaxText configurations
port unchanged: rules are consulted in order, the first whose
``module_path`` fully matches and whose ``op_names`` admit the op wins,
and a rule with no ``op_names`` applies to every op.
"""

from __future__ import annotations

import dataclasses
import json
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp

from .._internal.logging import get_logger
from ..core.graph import iter_modules
from ..core.module import Module
from ._numerics import QType, qtype_bits, qtype_name

__all__ = [
    "DEFAULT_OP_NAMES",
    "PLAN_ATTRIBUTE",
    "QuantPlan",
    "QuantProvider",
    "QuantRule",
    "quantize_model",
    "resolve_qtype",
    "rule_for",
    "unquantize_model",
]

logger = get_logger("spectrax.quantization")


PLAN_ATTRIBUTE = "_quant_plan"
"""Module attribute holding the resolved :class:`QuantPlan`.

The name matters, and only one underscore is correct.
:meth:`spectrax.Module.__setattr__` routes a single-underscore name into
the module's opaque-attribute map, which is carried by the ``GraphDef``
and restored by :func:`~spectrax.bind`. A ``_spx_``-prefixed name is
instead treated as a private implementation detail and set directly on
the instance, so it would *not* be exported — a quantized model would
run quantized when called eagerly and silently fall back to full
precision inside ``spx.jit`` or ``spx.grad``.
"""

DEFAULT_OP_NAMES: tuple[str, ...] = ("dot_general", "einsum", "ragged_dot")
"""Op vocabulary rules are resolved against.

``dot_general`` covers ordinary projections (including fused QKV and
gate-up, which are a single contraction), ``ragged_dot`` covers grouped
per-expert matmuls, and ``einsum`` covers explicitly-equation-driven
contractions. Names match Qwix's so that ported rules mean the same
thing.

Qwix also defines ``conv_general_dilated``. It is deliberately absent
here rather than accepted and ignored: no layer in this library consults
it, so a rule naming it would be stamped onto modules and silently change
nothing. Add the name when a quantized convolution exists to consult it.
"""

_SCALE_BITS = 16
"""Bits per scale element for an ordinary floating-point scale (bfloat16).

Used only to reason about the *effective* width a rule achieves; the
runtime scale dtype follows the array being quantized.
"""

_POWER_OF_TWO_SCALE_BITS = 8
"""Bits per scale element when scales are constrained to powers of two.

A power-of-two scale is just an exponent, which the microscaling formats
store as E8M0 -- eight bits, no mantissa. Counting it as a full bfloat16
would overstate MXFP4's cost by a quarter of a bit per value and trip the
small-tile warning on the block size the format actually specifies.
"""

_MAX_COMFORTABLE_SCALE_OVERHEAD = 0.25
"""Scale bits per payload bit above which a tiling is flagged as wasteful.

Expressed as a fraction rather than a tile size because the same tile is
cheap or expensive depending on whether its scale is an eight-bit
exponent or a sixteen-bit float."""


def resolve_qtype(spec: QType | int | None) -> QType | None:
    """Normalize a user-supplied quantized-type spec.

    Accepts what configuration files and humans actually write: a bit
    count (``4``), a short name (``"int4"``, ``"fp8"``, ``"nf4"``), or a
    real JAX dtype. Bit counts map to signed integers, which is what
    ``w_bits``/``a_bits`` mean in a MaxText ``intmp`` config.

    Args:
        spec: Bit count, name, dtype, or ``None``.

    Returns:
        A quantized type usable by the numerics layer, or ``None`` when
        ``spec`` is ``None`` (meaning "do not quantize this operand").

    Raises:
        ValueError: If a bit count is outside 2-8, or a name is unknown.
    """
    if spec is None:
        return None
    if isinstance(spec, int) and not isinstance(spec, bool):
        if spec == 4:
            return jnp.int4
        if spec == 8:
            return jnp.int8
        if 2 <= spec <= 7:
            return f"int{spec}"
        raise ValueError(f"Unsupported bit width {spec}; integer quantization supports 2-8 bits.")
    if isinstance(spec, str):
        aliases: dict[str, QType] = {
            "int4": jnp.int4,
            "int8": jnp.int8,
            "fp8": jnp.float8_e4m3fn,
            "fp8_e4m3": jnp.float8_e4m3fn,
            "float8_e4m3fn": jnp.float8_e4m3fn,
            "fp8_e4m3b11": jnp.float8_e4m3b11fnuz,
            "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
            "fp8_e5m2": jnp.float8_e5m2,
            "float8_e5m2": jnp.float8_e5m2,
            "fp4": jnp.float4_e2m1fn,
            "fp4_e2m1": jnp.float4_e2m1fn,
            "float4_e2m1fn": jnp.float4_e2m1fn,
            "nf4": "nf4",
        }
        lowered = spec.lower()
        if lowered in aliases:
            return aliases[lowered]
        if re.fullmatch(r"int[2-7]", lowered):
            return lowered
        raise ValueError(f"Unknown quantized type {spec!r}. Known: {', '.join(sorted(aliases))}, int2-int7.")
    return spec


def _normalize_module_path_pattern(pattern: str) -> str:
    """Make a path pattern accept both spectrax and MaxText separators.

    spectrax canonical paths are dot-joined (``layers.0.mlp.gate_proj``);
    Qwix and MaxText write slash-joined ones (``.*/wi_0``). A literal
    forward slash in the pattern is rewritten to ``[./]`` so a
    configuration written for either convention matches here. Slashes
    inside a character class are left alone.

    Args:
        pattern: The user-written regular expression.

    Returns:
        The pattern with bare separators generalized.
    """
    out: list[str] = []
    escaped = False
    in_class = False
    for char in pattern:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        if char == "/" and not in_class:
            out.append("[./]")
        else:
            out.append(char)
    return "".join(out)


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuantRule:
    """One quantization rule: a module-path matcher plus a numeric regime.

    Frozen and hashable so it can be stamped onto a module as opaque
    static metadata and participate in ``structure_hash``.

    Attributes:
        module_path: Regular expression fully matched against a module's
            canonical path. A literal ``/`` also matches ``.``, so
            slash-style patterns from MaxText configs work unchanged.
        op_names: Ops this rule applies to; empty means all of them. See
            :data:`DEFAULT_OP_NAMES`.
        weight_qtype: Type the weight operand is quantized to. ``None``
            disables the rule entirely — there is no activation-only
            quantization regime.
        act_qtype: Type the activation operand is quantized to. ``None``
            gives weight-only quantization (A16W4, A16W8, ...), where the
            matmul still runs in the compute dtype.
        tile_size: Subchannel tile size on the contracted axis. ``int`` is
            a literal size; ``float`` means ``1 / tile_count``. ``None``
            gives one scale per channel of the non-contracted axes.
        weight_calibration_method: Calibration for the weight operand;
            see :mod:`spectrax.quantization._calibrate`.
        act_calibration_method: Calibration for the activation operand.
        power_of_two_scale: Constrain every scale to a power of two, as
            the microscaling formats require. With ``tile_size=32`` and
            ``weight_qtype=float4_e2m1fn`` this is MXFP4 -- the format
            DeepSeek-V4 applies to its mixture-of-experts weights.
        weight_block_size: Tile the weight's non-contracted axes too,
            giving square blocks instead of one scale per output
            channel. ``tile_size=128`` with ``weight_block_size=128``
            is DeepSeek-V3's 128x128 weight blocking. Leaving it unset
            keeps per-channel scales, which are finer and slightly more
            accurate but store one scale per output channel.
        bwd_qtype: Type the incoming cotangent is quantized to in the
            backward pass. ``None`` keeps the backward pass in full
            precision, which is the safe default.
        bwd_calibration_method: Calibration for the cotangent.
        bwd_weight_grad_tile_size: Subchannel tile size used when forming
            the weight gradient. Applied to the cotangent and the residual
            activation, not to any weight.
        bwd_stochastic_rounding: ``"uniform"`` to round the cotangent
            stochastically, which removes the bias that deterministic
            rounding introduces into accumulated gradients.
        channelwise_noise_axes: Axes that get independent stochastic
            rounding noise.
        disable_channelwise_axes: Collapse every non-contracted axis to a
            single shared scale. Rarely wanted: on a fused QKV projection
            it makes Q, K and V share one range. Kept for parity with
            Qwix's rule surface.
    """

    module_path: str = ".*"
    op_names: tuple[str, ...] = ()

    weight_qtype: QType | None = None
    act_qtype: QType | None = None
    tile_size: int | float | None = None
    weight_calibration_method: str = "absmax"
    act_calibration_method: str = "absmax"
    power_of_two_scale: bool = False
    weight_block_size: int | None = None

    bwd_qtype: QType | None = None
    bwd_calibration_method: str = "absmax"
    bwd_weight_grad_tile_size: int | float | None = None
    bwd_stochastic_rounding: str | None = None
    channelwise_noise_axes: tuple[int, ...] = (0,)

    disable_channelwise_axes: bool = False

    def __post_init__(self) -> None:
        """Normalize types and separators, then validate the combination.

        Raises:
            ValueError: If ``bwd_stochastic_rounding`` is not ``"uniform"``
                or ``None``, or if the rule's effective width shows the
                scale costing at least as much as the values it scales.
        """
        object.__setattr__(self, "module_path", _normalize_module_path_pattern(self.module_path))
        object.__setattr__(self, "op_names", tuple(self.op_names))
        object.__setattr__(self, "channelwise_noise_axes", tuple(self.channelwise_noise_axes))
        for field in ("weight_qtype", "act_qtype", "bwd_qtype"):
            object.__setattr__(self, field, resolve_qtype(getattr(self, field)))
        if self.bwd_stochastic_rounding not in (None, "uniform"):
            raise ValueError(
                f"bwd_stochastic_rounding must be 'uniform' or None, got {self.bwd_stochastic_rounding!r}."
            )
        self._validate_reciprocal("tile_size", self.tile_size)
        self._validate_reciprocal("bwd_weight_grad_tile_size", self.bwd_weight_grad_tile_size)
        self._validate_tile_size()

    @staticmethod
    def _validate_reciprocal(field: str, value: int | float | None) -> None:
        """Reject tile sizes that cannot describe a real tiling.

        An integer tile size is a literal element count and must be
        positive. A float is a *reciprocal tile count*, so it must lie in
        ``(0, 1]``; anything else cannot be inverted into a whole number
        of tiles.

        The float case is worth checking explicitly because the natural
        way to compute one is ``1 / shard_count``, and a sentinel shard
        count of ``-1`` or ``0`` silently produces ``-1.0`` or a division
        error rather than "no tiling". Caught here, that is a clear
        configuration message; caught later, it surfaces as a shape error
        inside a backward pass.

        Args:
            field: Name of the field being validated, for the message.
            value: The configured tile size.

        Raises:
            ValueError: If the value cannot describe a tiling.
        """
        if value is None:
            return
        if isinstance(value, float):
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{field}={value!r} is a float, which means a reciprocal tile count and must lie in (0, 1]. "
                    f"A value of {value!r} usually comes from computing 1/shard_count with a sentinel shard "
                    f"count; pass None to disable tiling instead."
                )
            return
        if value <= 0:
            raise ValueError(f"{field}={value!r} must be a positive number of elements, or None to disable tiling.")

    def _validate_tile_size(self) -> None:
        """Reject tile sizes so small the scale defeats the quantization.

        A subchannel scale costs ``scale_bits / tile_size`` bits per
        quantized value. At ``tile_size=4`` with bfloat16 scales that is
        4 extra bits on top of a 4-bit value — the weight occupies as much
        memory as int8 while carrying int4's precision, which is strictly
        worse than simply using int8.

        The check is on the *overhead fraction*, not on the tile size
        itself, because how small a tile may usefully be depends on how
        many bits its scale costs. A power-of-two scale is an eight-bit
        exponent rather than a sixteen-bit float, which is exactly why
        MXFP4 can specify a 32-element block and still spend only a
        quarter of a bit per value; keying the warning on raw tile size
        would flag the format's own block size as a mistake.

        Raises:
            ValueError: If the scale overhead meets or exceeds the payload
                width.
        """
        if self.weight_qtype is None or self.tile_size is None or isinstance(self.tile_size, float):
            return
        payload = qtype_bits(self.weight_qtype)
        overhead = self._scale_bits / self.tile_size
        if overhead >= payload:
            raise ValueError(
                f"tile_size={self.tile_size} with {qtype_name(self.weight_qtype)} weights costs "
                f"{overhead:.2f} scale bits per {payload}-bit value ({payload + overhead:.2f} effective bits), "
                f"so it uses at least as much memory as the next wider type while keeping the lower precision. "
                f"Use a larger tile size, a power-of-two scale, or a wider weight_qtype."
            )
        if overhead > _MAX_COMFORTABLE_SCALE_OVERHEAD * payload:
            warnings.warn(
                f"tile_size={self.tile_size} with {qtype_name(self.weight_qtype)} weights spends "
                f"{overhead:.2f} scale bits per {payload}-bit value "
                f"({self.effective_bits:.2f} effective bits, {100 * overhead / payload:.0f}% overhead). "
                f"Widen the tile or use power_of_two_scale=True to halve the scale's cost.",
                stacklevel=4,
            )

    @property
    def _scale_bits(self) -> int:
        """Bits one scale element occupies, given the scale constraint.

        Returns:
            Eight for a power-of-two (E8M0) scale, else a full bfloat16.
        """
        return _POWER_OF_TWO_SCALE_BITS if self.power_of_two_scale else _SCALE_BITS

    @property
    def effective_bits(self) -> float:
        """Bits per weight value once the subchannel scale is counted.

        Returns:
            ``payload_bits + scale_bits / tile_size`` for an integer tile
            size, or the payload width alone when there is no subchannel
            tiling (channelwise scales amortize to nothing over a long
            contracted axis).
        """
        if self.weight_qtype is None:
            return float(self._scale_bits)
        payload = float(qtype_bits(self.weight_qtype))
        if self.tile_size is None or isinstance(self.tile_size, float):
            return payload
        return payload + self._scale_bits / self.tile_size

    @property
    def is_weight_only(self) -> bool:
        """Whether only the weight operand is quantized (A16Wn)."""
        return self.weight_qtype is not None and self.act_qtype is None

    @property
    def trains_in_narrow_precision(self) -> bool:
        """Whether the matmul itself runs in the quantized type.

        This is the line between two regimes that are often both called
        "QAT" and are not the same thing.

        When only the weight is quantized the contraction still happens in
        the compute dtype: the weight is discretized and immediately
        reconstructed, so the model *experiences* quantization error
        without any narrow arithmetic taking place. That is
        quantization-aware training in the original sense -- the model
        adapts to a degradation it will suffer later, at deployment.

        When the activation is quantized too, both operands reach the
        matmul in the narrow type and the hardware really does contract
        int8 against int8. That is *quantized training*: the point is
        throughput now, not robustness later. Qwix names its provider
        "QT" for exactly this reason.

        Returns:
            ``True`` when both operands are quantized, so the contraction
            runs narrow.
        """
        return self.weight_qtype is not None and self.act_qtype is not None

    def matches(self, path: str, op_name: str) -> bool:
        """Whether this rule claims ``op_name`` inside module ``path``.

        Args:
            path: The module's canonical path (``""`` for the root).
            op_name: The op being performed, e.g. ``"dot_general"``.

        Returns:
            ``True`` when the path fully matches and the op is admitted.
        """
        if self.op_names and op_name not in self.op_names:
            return False
        return re.fullmatch(self.module_path, path) is not None


@dataclasses.dataclass(frozen=True)
class QuantPlan:
    """The rules resolved for one module, as stamped onto it.

    Attributes:
        path: The module's canonical path at stamping time. Carried for
            diagnostics — after ``export``/``bind`` the module itself no
            longer knows where it sits.
        rules: Resolved ``(op_name, rule)`` pairs. A tuple rather than a
            dict so the plan stays hashable and therefore legal as static
            graph metadata.
    """

    path: str
    rules: tuple[tuple[str, QuantRule], ...] = ()

    def rule(self, op_name: str) -> QuantRule | None:
        """Return the rule governing ``op_name``, if any.

        Args:
            op_name: The op being performed.

        Returns:
            The matching :class:`QuantRule`, or ``None`` when this module
            performs that op unquantized.
        """
        for name, rule in self.rules:
            if name == op_name:
                return rule
        return None

    def __bool__(self) -> bool:
        """Whether the plan governs any op at all."""
        return bool(self.rules)


class QuantProvider:
    """An ordered list of rules, resolved first-match-wins.

    Order is precedence: put specific paths before general ones, and any
    catch-all last. This mirrors Qwix so that rule lists lifted from a
    MaxText config behave identically.
    """

    def __init__(self, rules: Sequence[QuantRule]) -> None:
        """Store the rules in precedence order.

        Args:
            rules: Rules, most specific first.

        Raises:
            TypeError: If any element is not a :class:`QuantRule`.
        """
        for rule in rules:
            if not isinstance(rule, QuantRule):
                raise TypeError(f"QuantProvider takes QuantRule instances, got {type(rule).__name__}.")
        self._rules: tuple[QuantRule, ...] = tuple(rules)

    @property
    def rules(self) -> tuple[QuantRule, ...]:
        """The rules, in precedence order."""
        return self._rules

    def rule_for_path(self, path: str, op_name: str) -> QuantRule | None:
        """Resolve the rule governing ``op_name`` at module ``path``.

        Args:
            path: The module's canonical path.
            op_name: The op being performed.

        Returns:
            The first matching rule, or ``None`` when no rule claims it.
        """
        for rule in self._rules:
            if rule.matches(path, op_name):
                return rule
        return None

    def plan_for_path(self, path: str, op_names: Iterable[str] = DEFAULT_OP_NAMES) -> QuantPlan:
        """Resolve every op at ``path`` into a single stampable plan.

        Args:
            path: The module's canonical path.
            op_names: The op vocabulary to resolve against.

        Returns:
            A :class:`QuantPlan`, empty when no rule claims this module.
        """
        resolved: list[tuple[str, QuantRule]] = []
        for op_name in op_names:
            rule = self.rule_for_path(path, op_name)
            if rule is not None and rule.weight_qtype is not None:
                resolved.append((op_name, rule))
        return QuantPlan(path=path, rules=tuple(resolved))

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        module_path: str = ".*",
        tile_size: int | float | None = None,
        quantize_backward: bool = True,
        op_names: Sequence[str] = ("dot_general", "ragged_dot"),
    ) -> QuantProvider:
        """Build the single-rule provider for a named numeric regime.

        The presets mirror MaxText's ``quantization`` values so that a run
        configured there transfers by name. ``w*`` presets are weight-only
        (the activation stays in the compute dtype); the bare type names
        quantize activations and, by default, the backward pass too — the
        regime MaxText's ``int8``/``int4`` configure.

        Args:
            name: One of ``int4``, ``int8``, ``fp8``, ``fp8_e4m3b11``,
                ``fp8_e5m2``, ``fp4``, ``nf4``, ``w4a16``, ``w8a16``,
                ``nf4a16``.
            module_path: Path pattern the rule applies to.
            tile_size: Optional subchannel tile size on the contracted axis.
            quantize_backward: Whether to also quantize the cotangent, for
                presets that quantize activations.
            op_names: Ops the rule claims.

        Returns:
            A provider holding exactly one rule.

        Raises:
            ValueError: If ``name`` is not a known preset.
        """
        weight_only = {"w4a16": jnp.int4, "w8a16": jnp.int8, "nf4a16": "nf4"}
        symmetric = {
            "int4": jnp.int4,
            "int8": jnp.int8,
            "fp8": jnp.float8_e4m3fn,
            "fp8_e4m3": jnp.float8_e4m3fn,
            # TPU's native 8-bit float. Measured on TPU v5 at 8192^3: 420
            # TFLOP/s against 390 for e4m3fn and 321 for the fnuz variants,
            # which are evidently emulated. Same 3-bit mantissa as e4m3fn, and
            # the narrower exponent range costs nothing once a per-channel
            # scale has normalized the values (measured round-trip error
            # 0.0258 versus 0.0260). Not the default, because `fp8` tracks
            # MaxText and portability; pick this one when targeting TPU.
            "fp8_e4m3b11": jnp.float8_e4m3b11fnuz,
            "fp8_e5m2": jnp.float8_e5m2,
            "fp4": jnp.float4_e2m1fn,
            "fp4_e2m1": jnp.float4_e2m1fn,
            "mxfp4": jnp.float4_e2m1fn,
        }
        key = name.lower()
        if key in weight_only:
            rule = QuantRule(
                module_path=module_path,
                op_names=tuple(op_names),
                weight_qtype=weight_only[key],
                tile_size=tile_size,
            )
        elif key == "nf4":
            # nf4 is a weight code book: it has no activation counterpart
            # and cannot be dequantized on the output, so it is only ever
            # weight-only.
            rule = QuantRule(
                module_path=module_path,
                op_names=tuple(op_names),
                weight_qtype="nf4",
                tile_size=tile_size,
            )
        elif key == "deepseek_fp8":
            # DeepSeek-V3's pretraining recipe: all three GEMMs (forward,
            # activation-backward, weight-backward) in fp8 e4m3, with
            # fine-grained scaling -- activations on 1x128 tiles, weights on
            # 128x128 blocks -- and the master weights left wide.
            rule = QuantRule(
                module_path=module_path,
                op_names=tuple(op_names),
                weight_qtype=jnp.float8_e4m3fn,
                act_qtype=jnp.float8_e4m3fn,
                bwd_qtype=jnp.float8_e4m3fn if quantize_backward else None,
                tile_size=128 if tile_size is None else tile_size,
                weight_block_size=128,
            )
        elif key == "mxfp4":
            # Microscaling FP4 as DeepSeek-V4 applies it to mixture-of-experts
            # weights: E2M1 values with a power-of-two shared scale over 1x32
            # sub-blocks of the contracted axis. Weight-only, because that is
            # the regime the format is defined and deployed for -- the paper
            # dequantizes these weights to fp8 for the actual matmul rather
            # than contracting fp4 against fp4.
            rule = QuantRule(
                module_path=module_path,
                op_names=tuple(op_names),
                weight_qtype=jnp.float4_e2m1fn,
                tile_size=32 if tile_size is None else tile_size,
                power_of_two_scale=True,
            )
        elif key in symmetric:
            qtype = symmetric[key]
            # fp8 gradients want the wider exponent of e5m2; MaxText's
            # fp8_full rule makes the same split.
            bwd = jnp.float8_e5m2 if qtype in (jnp.float8_e4m3fn, jnp.float8_e4m3b11fnuz) else qtype
            rule = QuantRule(
                module_path=module_path,
                op_names=tuple(op_names),
                weight_qtype=qtype,
                act_qtype=qtype,
                bwd_qtype=bwd if quantize_backward else None,
                tile_size=tile_size,
            )
        else:
            known = sorted(set(weight_only) | set(symmetric) | {"nf4"})
            raise ValueError(f"Unknown quantization preset {name!r}. Known presets: {', '.join(known)}.")
        return cls([rule])

    @classmethod
    def from_intmp(
        cls,
        config: str | Mapping[str, Mapping[str, Any]],
        *,
        op_names: Sequence[str] = ("dot_general", "ragged_dot"),
    ) -> QuantProvider:
        """Build a provider from a MaxText mixed-precision (``intmp``) config.

        Reads MaxText's JSON schema verbatim so their configuration files
        work unchanged::

            {
              "__default__": {"w_bits": 8, "a_bits": 8},
              ".*/query":    {"w_bits": 4, "tile_size": 128},
              ".*/wo":       {"w_bits": 4}
            }

        Keys are module-path patterns; ``__default__`` becomes a catch-all
        placed last regardless of where it appears in the file. Values
        support ``w_bits``, ``a_bits`` (absent means weight-only),
        ``w_scale``/``a_scale`` (clipping factors folded into the absmax
        calibration method) and ``tile_size`` (``-1`` means none).

        Args:
            config: Path to a JSON file, or an already-parsed mapping.
            op_names: Ops the generated rules claim.

        Returns:
            A provider with one rule per entry, catch-all last.

        Raises:
            ValueError: If an entry carries an unknown key.
        """
        if isinstance(config, str):
            with open(config) as handle:
                parsed: Mapping[str, Mapping[str, Any]] = json.load(handle)
        else:
            parsed = config

        known_keys = {"w_bits", "a_bits", "w_scale", "a_scale", "tile_size"}
        specific: list[QuantRule] = []
        default: list[QuantRule] = []
        for pattern, spec in parsed.items():
            unknown = set(spec) - known_keys
            if unknown:
                raise ValueError(
                    f"Unknown key(s) {sorted(unknown)} in intmp entry {pattern!r}; supported: {sorted(known_keys)}."
                )
            tile_size = spec.get("tile_size")
            if tile_size == -1:
                tile_size = None
            w_scale = spec.get("w_scale")
            a_scale = spec.get("a_scale")
            rule = QuantRule(
                module_path=".*" if pattern == "__default__" else pattern,
                op_names=tuple(op_names),
                weight_qtype=resolve_qtype(spec.get("w_bits")),
                act_qtype=resolve_qtype(spec.get("a_bits")),
                tile_size=tile_size,
                weight_calibration_method="absmax" if w_scale is None else f"absmax,{w_scale}",
                act_calibration_method="absmax" if a_scale is None else f"absmax,{a_scale}",
            )
            (default if pattern == "__default__" else specific).append(rule)
        return cls([*specific, *default])

    def __repr__(self) -> str:
        """Return a one-line summary naming the rule count."""
        return f"QuantProvider({len(self._rules)} rule(s))"


def rule_for(module: Module, op_name: str) -> QuantRule | None:
    """Return the quantization rule a module should apply to ``op_name``.

    The single lookup layer code uses. Cheap enough to call on every
    forward: it is an attribute fetch and a short linear scan over the
    module's own resolved ops, with no regular expressions involved.

    Args:
        module: The module performing the op.
        op_name: The op being performed, e.g. ``"dot_general"``.

    Returns:
        The governing rule, or ``None`` when the op runs unquantized —
        which is the case for every module of a model that was never
        passed through :func:`quantize_model`.
    """
    plan: QuantPlan | None = getattr(module, PLAN_ATTRIBUTE, None)
    if plan is None:
        return None
    return plan.rule(op_name)


def quantize_model(
    model: Module,
    provider: QuantProvider,
    *,
    op_names: Sequence[str] = DEFAULT_OP_NAMES,
    strict: bool = True,
) -> Module:
    """Stamp ``provider``'s rules onto every matching module of ``model``.

    Mutates the module tree in place and returns it, matching the
    convention of :meth:`spectrax.Module.train` and
    :meth:`spectrax.Module.freeze`. Re-stamping is idempotent: a previous
    plan on a module is replaced, and modules that no longer match are
    cleared, so calling this twice with different providers yields the
    same result as calling it once with the second.

    Args:
        model: Root of the live module tree.
        provider: The rules to resolve.
        op_names: The op vocabulary to resolve against.
        strict: When ``True``, raise if no module matched at all. A
            provider that silently matches nothing produces a model that
            trains at full precision while reporting itself as quantized,
            which is the single most expensive way for this to fail.

    Returns:
        ``model``, for chaining.

    Raises:
        ValueError: If ``strict`` and no module matched any rule.
    """
    matched = 0
    for path, module in iter_modules(model):
        plan = provider.plan_for_path(path, op_names)
        if plan:
            setattr(module, PLAN_ATTRIBUTE, plan)
            matched += 1
        elif getattr(module, PLAN_ATTRIBUTE, None) is not None:
            setattr(module, PLAN_ATTRIBUTE, None)

    if matched == 0:
        message = (
            f"{provider!r} matched no module in {type(model).__name__}. "
            "Check the module_path patterns against spx.iter_modules(model) paths, which are dot-joined."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    else:
        narrow = sum(
            1
            for rule in provider.rules
            if rule.weight_qtype is not None and rule.trains_in_narrow_precision
        )
        regime = (
            "quantized training (the matmul itself runs in the narrow type)"
            if narrow
            else "quantization-aware training (operands discretized, matmul stays in the compute dtype)"
        )
        logger.info(f"quantize_model: {matched} module(s) matched by {provider!r} -- {regime}.")
    return model


def unquantize_model(model: Module) -> Module:
    """Remove every stamped plan, returning the model to full precision.

    Args:
        model: Root of the live module tree.

    Returns:
        ``model``, for chaining.
    """
    for _path, module in iter_modules(model):
        if getattr(module, PLAN_ATTRIBUTE, None) is not None:
            setattr(module, PLAN_ATTRIBUTE, None)
    return model
