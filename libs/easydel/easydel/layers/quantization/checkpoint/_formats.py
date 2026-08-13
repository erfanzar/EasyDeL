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

"""Adapters for the quantized checkpoint formats published on the Hub.

Each adapter is thin by construction: it parses its ``quantization_config``,
names the checkpoint tensors it consumes, picks a runtime target, and calls a
decoder from :mod:`._codecs`. Packing is never reimplemented — every
``to_canonical`` ends in
:meth:`~._canonical.CanonicalQuantizedWeight.from_dense`, which routes through
the same ``prepack_quantized_weights`` the self-quantization path uses.

Runtime targets are chosen, not inherited from the checkpoint, because
ejkernel's modes are a fixed set:

===========================  ==========================================
checkpoint element encoding  runtime target
===========================  ==========================================
``int4`` + zero-points       ``affine`` bits=4 at the checkpoint's group
``int8``                     ``affine`` bits=8
``float8_e4m3``              ``affine`` bits=8 (measured; see below)
``e2m1`` + E8M0 (MXFP4)      ``mxfp4`` (group 32)
``e2m1`` + E4M3 (NVFP4)      ``nvfp4`` (group 16) + per-tensor
                             ``output_scale``
===========================  ==========================================

**Decoding is exact for every format.** All observed error comes from
*repacking* the decoded weight into EasyDeL's runtime grouping, which runs
along output features while every checkpoint groups along input features. For
8-bit targets that costs little; for 4-bit it dominates. The gap is measured
and pinned by ``TestGroupingAxisLimitation`` in the format tests — on data
lying exactly on an MXFP4 grid, input-axis grouping round-trips at 0.0 error
while output-axis grouping costs 0.042. Closing it is a change to
``ParallelLinearQuantized``'s layout, not to any adapter.
"""

from __future__ import annotations

import typing as tp

from eformer.loggings import get_logger

from .._configs import QuantizationType
from ._adapter import CheckpointQuantAdapter, ConfigParse, IgnoreRules, register_adapter
from ._canonical import ActivationPolicy, CanonicalQuantizedWeight, QuantSpec, SourceFormat
from ._codecs import (
    decode_awq_int4,
    decode_compressed_int4,
    decode_gptq_int4,
    decode_mxfp4,
    decode_nvfp4,
    decode_scaled_elements,
)
from ._decode import largest_supported_group_size

logger = get_logger(__name__)


def _ignore_rules(quant_config: tp.Mapping[str, tp.Any]) -> IgnoreRules:
    """Read a config's ignore lists, honouring each key's own semantics.

    ``ignored_layers`` and compressed-tensors' ``ignore`` name exact leaf
    modules; ``modules_to_not_convert`` entries follow HF's matching
    (containment/suffix/start-anchored regex — see
    :class:`~._adapter.IgnoreRules`), covering container names, bare leaf
    names like Mixtral-AWQ's ``"gate"``, and gpt-oss's glob-style patterns.
    Entries prefixed ``re:`` are regexes.

    Args:
        quant_config: Raw ``quantization_config`` mapping.

    Returns:
        The combined :class:`IgnoreRules`.
    """
    exact: set[str] = set()
    patterns: list[str] = []
    for key in ("ignored_layers", "ignore"):
        for entry in quant_config.get(key) or ():
            if isinstance(entry, str) and entry.startswith("re:"):
                patterns.append(entry[3:])
            else:
                exact.add(str(entry))

    prefixes = tuple(str(entry) for entry in (quant_config.get("modules_to_not_convert") or ()))
    return IgnoreRules(exact=frozenset(exact), prefixes=prefixes, patterns=tuple(patterns))


def _activation_policy(
    scheme: str | None,
    *,
    dtype: tp.Literal["int8", "fp8_e4m3"] = "fp8_e4m3",
) -> ActivationPolicy:
    """Translate an ``activation_scheme`` string into a policy.

    Args:
        scheme: ``"dynamic"``, ``"static"``, or ``None``/``"none"``.
        dtype: Quantized activation dtype implied by the weight encoding.

    Returns:
        The corresponding :class:`ActivationPolicy`.
    """
    if scheme == "dynamic":
        return ActivationPolicy.dynamic(dtype)
    if scheme == "static":
        return ActivationPolicy.static(dtype)
    return ActivationPolicy.none()


def _affine_spec(
    *,
    bits: int,
    group_size: int | None,
    expert_dim: bool,
    activation: ActivationPolicy,
    contracting_hint: int | None = None,
) -> QuantSpec:
    """Build an affine target, resolving a legal group size.

    Args:
        bits: Weight bit width.
        group_size: The checkpoint's group size, or ``None``/``-1`` for
            per-channel schemes where any grouping reproduces the scale
            exactly once broadcast.
        expert_dim: Whether a leading expert axis is present.
        activation: Activation policy.
        contracting_hint: Contracting-dimension length used to choose a
            divisor when the checkpoint is per-channel.

    Returns:
        The affine :class:`QuantSpec`.
    """
    if not group_size or int(group_size) <= 0:
        resolved = largest_supported_group_size(int(contracting_hint or 128))
    else:
        resolved = int(group_size)
    return QuantSpec(
        dtype=QuantizationType.AFFINE,
        group_size=resolved,
        bits=bits,
        expert_dim=expert_dim,
        activation=activation,
    )


@register_adapter("fp8", overwrite=True)
class ScaledElementsAdapter(CheckpointQuantAdapter):
    """FP8 checkpoints: ``float8_e4m3`` elements with float scales.

    Handles all three scale granularities the format ships with — per-tensor,
    per-channel, and DeepSeek-style ``128x128`` blocks (serialized as
    ``weight_scale_inv``) — through one decoder, because they differ only in
    how coarse the scale grid is.

    Class Attributes:
        float8_target: Runtime target for fp8 weights.
        float8_group_size: Requested group size for the target; clamped to the
            weight's grouping axis at pack time.

    The target is ``AFFINE`` int8 on measurement, not on intuition. The
    plausible-sounding argument for ``MXFP8`` — that 8-bit floats should stay
    8-bit floats rather than pass through an integer grid — does not survive
    contact with grouped quantization: within a group the dynamic range is
    small, so int8's 256 levels with an exact float scale beat E4M3's 3-bit
    mantissa with a power-of-two E8M0 scale. Round-trip error on gaussian
    weights, ``[256, 128]``:

    ==========================  =========
    target                      rel. err
    ==========================  =========
    ``affine`` int8, group 64   0.003
    ``mxfp8``, group 32         0.049
    ==========================  =========

    Override the attribute to re-measure on real weights; nothing else needs
    to change.
    """

    float8_target: tp.ClassVar[QuantizationType] = QuantizationType.AFFINE
    float8_group_size: tp.ClassVar[int] = 128

    @classmethod
    def parse_config(cls, quant_config):
        """Read fp8 block size, activation scheme and ignore lists."""
        block_size = quant_config.get("weight_block_size")
        activation = _activation_policy(quant_config.get("activation_scheme"))
        return ConfigParse(
            default=SourceFormat(
                quant_method="fp8",
                weight_dtype="float8_e4m3fn",
                block_shape=tuple(int(dim) for dim in block_size) if block_size else None,
                activation=activation,
                raw=dict(quant_config),
            ),
            ignored=_ignore_rules(quant_config),
        )

    @classmethod
    def target_spec(cls, source, *, expert_dim=False):
        """Target grouped affine int8, or the configured override."""
        target = cls.float8_target
        if target is QuantizationType.AFFINE:
            return _affine_spec(
                bits=8,
                group_size=cls.float8_group_size,
                expert_dim=expert_dim,
                activation=source.activation,
            )
        return QuantSpec(
            dtype=target,
            group_size=32 if target is QuantizationType.MXFP8 else 16,
            bits=8,
            expert_dim=expert_dim,
            activation=source.activation,
        )

    @classmethod
    def source_suffixes(cls, source):
        """Name the scale tensor the checkpoint actually ships.

        Block-scaled fp8 serializes ``weight_scale_inv``; the per-tensor and
        per-channel forms use ``weight_scale``. Declaring the wrong one makes
        the fusion silently never fire, since a rule only triggers when every
        source key is present.
        """
        scale_name = "weight_scale_inv" if source.is_block_scaled else "weight_scale"
        suffixes = ["weight", scale_name]
        if source.activation.kind.value == "static":
            suffixes.append("input_scale")
        return tuple(suffixes)

    @classmethod
    def to_canonical(cls, tensors, *, source, target):
        """Dequantize the fp8 elements and repack to the runtime target."""
        scale_name = "weight_scale_inv" if source.is_block_scaled else "weight_scale"
        dense = decode_scaled_elements(tensors["weight"], tensors[scale_name], block_shape=source.block_shape)
        return CanonicalQuantizedWeight.from_dense(
            dense,
            spec=target,
            activation_scale=tensors.get("input_scale"),
        )


@register_adapter("awq", "auto_awq", overwrite=True)
class AwqAdapter(CheckpointQuantAdapter):
    """AWQ ``uint4`` checkpoints with per-group scales and zero-points."""

    @classmethod
    def parse_config(cls, quant_config):
        """Read AWQ bit width, group size and zero-point flag."""
        return ConfigParse(
            default=SourceFormat(
                quant_method="awq",
                weight_dtype="uint4",
                group_size=int(quant_config.get("group_size", 128)),
                symmetric=not bool(quant_config.get("zero_point", True)),
                has_zero_point=bool(quant_config.get("zero_point", True)),
                raw=dict(quant_config),
            ),
            ignored=_ignore_rules(quant_config),
        )

    @classmethod
    def target_spec(cls, source, *, expert_dim=False):
        """Target affine 4-bit at the checkpoint's own group size."""
        bits = int(source.raw.get("bits", 4))
        return _affine_spec(
            bits=bits,
            group_size=source.group_size,
            expert_dim=expert_dim,
            activation=source.activation,
        )

    @classmethod
    def source_suffixes(cls, source):
        """AWQ ships packed weights, scales and packed zero-points."""
        if source.has_zero_point:
            return ("qweight", "scales", "qzeros")
        return ("qweight", "scales")

    @classmethod
    def to_canonical(cls, tensors, *, source, target):
        """Unpack the interleaved AWQ lanes and repack to affine."""
        dense = decode_awq_int4(tensors["qweight"], tensors["scales"], tensors.get("qzeros"))
        return CanonicalQuantizedWeight.from_dense(dense, spec=target)


@register_adapter("gptq", overwrite=True)
class GptqAdapter(CheckpointQuantAdapter):
    """GPTQ ``uint4`` checkpoints, including act-order (``desc_act``)."""

    @classmethod
    def parse_config(cls, quant_config):
        """Read GPTQ bits, group size, symmetry and act-order flag."""
        return ConfigParse(
            default=SourceFormat(
                quant_method="gptq",
                weight_dtype="uint4",
                group_size=int(quant_config.get("group_size", 128)),
                symmetric=bool(quant_config.get("sym", True)),
                has_zero_point=True,
                raw=dict(quant_config),
            ),
            ignored=_ignore_rules(quant_config),
        )

    @classmethod
    def target_spec(cls, source, *, expert_dim=False):
        """Target affine at the checkpoint's bit width and group size."""
        return _affine_spec(
            bits=int(source.raw.get("bits", 4)),
            group_size=source.group_size,
            expert_dim=expert_dim,
            activation=source.activation,
        )

    @classmethod
    def source_suffixes(cls, source):
        """GPTQ ships packed weights, scales, packed zeros and act-order idx."""
        suffixes = ["qweight", "scales", "qzeros"]
        if source.raw.get("desc_act", False):
            suffixes.append("g_idx")
        return tuple(suffixes)

    @classmethod
    def to_canonical(cls, tensors, *, source, target):
        """Unpack GPTQ's input-axis packing and repack to affine."""
        dense = decode_gptq_int4(
            tensors["qweight"],
            tensors["scales"],
            tensors.get("qzeros"),
            g_idx=tensors.get("g_idx"),
        )
        return CanonicalQuantizedWeight.from_dense(dense, spec=target)


@register_adapter("mxfp4", "gpt_oss_mxfp4", overwrite=True)
class Mxfp4Adapter(CheckpointQuantAdapter):
    """MXFP4 checkpoints: E2M1 elements with E8M0 shared exponents.

    EasyDeL's ``mxfp4`` mode uses the same group size (32) and the same E8M0
    ``uint8`` scales as the on-disk format, so no *format* conversion happens
    and no wider requantization block is imposed — unlike implementations that
    rewrite MXFP4 to a coarser block to suit a matmul unit, which is a real
    accuracy cost taken at load time.

    The round trip is nonetheless not yet bit-exact, because the repack groups
    along output features while the checkpoint grouped along input features.
    Measured on data lying exactly on an MXFP4 grid, grouping along inputs
    round-trips at 0.0 error against 0.042 for the current axis; that is the
    whole of the loss, and it is a layout property of the quantized linear
    rather than anything this adapter controls.
    """

    @classmethod
    def parse_config(cls, quant_config):
        """Read the MXFP4 ignore lists; the format has no other knobs."""
        return ConfigParse(
            default=SourceFormat(
                quant_method="mxfp4",
                weight_dtype="float4_e2m1fn",
                group_size=32,
                raw=dict(quant_config),
            ),
            ignored=_ignore_rules(quant_config),
        )

    @classmethod
    def target_spec(cls, source, *, expert_dim=False):
        """Target native group-32 MXFP4."""
        return QuantSpec(
            dtype=QuantizationType.MXFP4,
            group_size=32,
            bits=4,
            expert_dim=expert_dim,
            activation=source.activation,
        )

    @classmethod
    def source_suffixes(cls, source):
        """MXFP4 ships packed E2M1 blocks alongside E8M0 scales."""
        return ("weight", "weight_scale")

    @classmethod
    def to_canonical(cls, tensors, *, source, target):
        """Decode E2M1/E8M0 and repack losslessly."""
        dense = decode_mxfp4(tensors["weight"], tensors["weight_scale"])
        return CanonicalQuantizedWeight.from_dense(dense, spec=target)


@register_adapter("modelopt_fp4", "nvfp4", overwrite=True)
class Nvfp4Adapter(CheckpointQuantAdapter):
    """NVFP4 checkpoints: E2M1 elements, E4M3 block scales, FP32 global scale.

    The global scale is carried as ``output_scale`` and applied after the
    matmul rather than folded into the block codes: a per-tensor scalar
    commutes with the matmul, so this is exact, whereas folding it into an
    E4M3 code would not be.
    """

    @classmethod
    def parse_config(cls, quant_config):
        """Read NVFP4 group size, global-scale presence and activation scheme."""
        return ConfigParse(
            default=SourceFormat(
                quant_method="modelopt_fp4",
                weight_dtype="float4_e2m1fn",
                group_size=int(quant_config.get("group_size", 16)),
                has_global_scale=True,
                activation=_activation_policy(quant_config.get("activation_scheme")),
                raw=dict(quant_config),
            ),
            ignored=_ignore_rules(quant_config),
        )

    @classmethod
    def target_spec(cls, source, *, expert_dim=False):
        """Target native group-16 NVFP4."""
        return QuantSpec(
            dtype=QuantizationType.NVFP4,
            group_size=16,
            bits=4,
            expert_dim=expert_dim,
            activation=source.activation,
        )

    @classmethod
    def source_suffixes(cls, source):
        """NVFP4 ships packed weights plus both scale levels."""
        suffixes = ["weight", "weight_scale", "weight_scale_2"]
        if source.activation.kind.value == "static":
            suffixes.append("input_scale")
        return tuple(suffixes)

    @classmethod
    def to_canonical(cls, tensors, *, source, target):
        """Fold in the block scale, carry the global scale separately."""
        import jax.numpy as jnp

        dense = decode_nvfp4(tensors["weight"], tensors["weight_scale"])
        global_scale = jnp.asarray(tensors["weight_scale_2"]).astype(jnp.float32).reshape(-1)[0]
        return CanonicalQuantizedWeight.from_dense(
            dense,
            spec=target,
            output_scale=global_scale,
            activation_scale=tensors.get("input_scale"),
        )


@register_adapter("compressed-tensors", "compressed_tensors", overwrite=True)
class CompressedTensorsAdapter(CheckpointQuantAdapter):
    """compressed-tensors checkpoints, whose scheme varies per config group.

    Unlike the single-scheme formats, one checkpoint may quantize different
    layers differently: ``config_groups`` maps target selectors to weight and
    activation schemes. Each group is parsed into its own
    :class:`~._canonical.SourceFormat`, and path resolution picks between them,
    so no other layer of the stack learns that this format is heterogeneous.

    Only the element encoding differs from the single-scheme formats, so the
    decoders are reused rather than reimplemented.
    """

    @classmethod
    def parse_config(cls, quant_config):
        """Expand ``config_groups`` into a default plus targeted overrides."""
        ignored = _ignore_rules(quant_config)
        default: SourceFormat | None = None
        overrides: list[tuple[str, SourceFormat]] = []

        for group_name, group in (quant_config.get("config_groups") or {}).items():
            weights = group.get("weights") or {}
            source = cls._source_from_group(weights, group.get("input_activations"), quant_config)
            targets = [str(target) for target in (group.get("targets") or ())]

            # compressed-tensors targets are either module *class* names
            # ("Linear", "QKVParallelLinear"), which select every instance and
            # so define the default scheme, or module *instance* names
            # ("q_proj", "model.layers.0.mlp"), which target specific layers.
            # The two are distinguished by case, as the ecosystem does:
            # classes are CamelCase, module names are snake_case paths.
            #
            # One group may legally list both — a scheme covering a class plus
            # named extras — so the two kinds are collected independently. An
            # early `continue` on seeing a class name drops the group's module
            # targets, leaving those layers on whatever catch-all was recorded
            # last, which for a heterogeneous checkpoint is a different bit
            # width decoded with the wrong scheme's suffixes.
            if any(_is_class_name(target) for target in targets):
                if default is not None:
                    logger.warning(
                        "compressed-tensors declares multiple catch-all config groups; %s overrides the earlier one.",
                        group_name,
                    )
                default = source
            for target in targets:
                if _is_class_name(target):
                    continue
                pattern = target[3:] if target.startswith("re:") else re_escape_path(target)
                overrides.append((pattern, source))

        return ConfigParse(default=default, overrides=tuple(overrides), ignored=ignored)

    @classmethod
    def _source_from_group(
        cls,
        weights: tp.Mapping[str, tp.Any],
        activations: tp.Mapping[str, tp.Any] | None,
        quant_config: tp.Mapping[str, tp.Any],
    ) -> SourceFormat:
        """Build a source format from one ``config_groups`` entry.

        Args:
            weights: The group's ``weights`` scheme.
            activations: The group's ``input_activations`` scheme, if any.
            quant_config: Whole config, retained for adapter access.

        Returns:
            The :class:`SourceFormat` describing this group.

        Raises:
            NotImplementedError: If the group is asymmetric. Asymmetric
                schemes ship a packed ``weight_zero_point`` tensor whose
                layout this adapter does not decode; reading the codes
                without it offsets every group by a full scale, which loads
                cleanly and produces garbage. Refusing is the only honest
                option until the packing is implemented.
        """
        num_bits = int(weights.get("num_bits", 8))
        weight_type = str(weights.get("type", "int")).lower()
        block = weights.get("block_structure")
        group_size = weights.get("group_size")
        symmetric = bool(weights.get("symmetric", True))

        if not symmetric:
            raise NotImplementedError(
                f"compressed-tensors group declares symmetric=false (type={weight_type!r}, "
                f"num_bits={num_bits}), which ships a weight_zero_point tensor. This adapter reads only "
                f"weight_packed/weight and weight_scale, so the zero-point would be silently dropped. "
                f"Load a symmetric checkpoint, or add zero-point decoding to CompressedTensorsAdapter."
            )

        if activations is None:
            activation = ActivationPolicy.none()
        else:
            act_dtype = "int8" if str(activations.get("type", "int")).lower() == "int" else "fp8_e4m3"
            kind = "dynamic" if activations.get("dynamic", True) else "static"
            activation = _activation_policy(kind, dtype=act_dtype)

        return SourceFormat(
            quant_method="compressed-tensors",
            weight_dtype=f"{weight_type}{num_bits}",
            block_shape=tuple(int(dim) for dim in block) if block else None,
            group_size=int(group_size) if group_size else None,
            symmetric=symmetric,
            has_zero_point=not symmetric,
            has_global_scale=weight_type == "float" and num_bits == 4,
            activation=activation,
            raw={"weights": dict(weights), "quantization_config": dict(quant_config)},
        )

    @classmethod
    def _encoding(cls, source: SourceFormat) -> str:
        """Classify a group's element encoding.

        Args:
            source: The group's source format.

        Returns:
            One of ``"int4"``, ``"int8"``, ``"fp8"`` or ``"nvfp4"``.

        Raises:
            NotImplementedError: For a bit width/type pair with no decoder.
        """
        weights = source.raw.get("weights", {})
        num_bits = int(weights.get("num_bits", 8))
        weight_type = str(weights.get("type", "int")).lower()
        if weight_type == "float" and num_bits == 4:
            return "nvfp4"
        if weight_type == "float" and num_bits == 8:
            return "fp8"
        if num_bits == 8:
            return "int8"
        if num_bits == 4:
            return "int4"
        raise NotImplementedError(f"compressed-tensors scheme type={weight_type!r} num_bits={num_bits} has no decoder.")

    @classmethod
    def target_spec(cls, source, *, expert_dim=False):
        """Pick the runtime target matching this group's encoding."""
        encoding = cls._encoding(source)
        if encoding == "nvfp4":
            return QuantSpec(
                dtype=QuantizationType.NVFP4,
                group_size=16,
                bits=4,
                expert_dim=expert_dim,
                activation=source.activation,
            )
        if encoding == "fp8":
            return ScaledElementsAdapter.target_spec(source, expert_dim=expert_dim)
        bits = 8 if encoding == "int8" else 4
        return _affine_spec(
            bits=bits,
            group_size=source.group_size,
            expert_dim=expert_dim,
            activation=source.activation,
        )

    @classmethod
    def source_suffixes(cls, source):
        """Name this group's tensors, which differ per encoding."""
        encoding = cls._encoding(source)
        if encoding == "nvfp4":
            return ("weight_packed", "weight_scale", "weight_global_scale")
        if encoding == "int4":
            return ("weight_packed", "weight_scale")
        return ("weight", "weight_scale")

    @classmethod
    def to_canonical(cls, tensors, *, source, target):
        """Decode with the codec matching this group's encoding."""
        import jax.numpy as jnp

        encoding = cls._encoding(source)
        if encoding == "nvfp4":
            dense = decode_nvfp4(tensors["weight_packed"], tensors["weight_scale"])
            # compressed-tensors and ModelOpt store this scalar in opposite
            # directions. llm-compressor computes
            # `global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax` and folds
            # it INTO the e4m3 block scales, so its dequantization divides:
            # `w = codes * weight_scale / weight_global_scale`. ModelOpt's
            # `weight_scale_2` is already the reciprocal and multiplies (see
            # Nvfp4Adapter). Carrying this one through unchanged would scale
            # every weight by `global_scale ** 2`.
            global_scale = jnp.asarray(tensors["weight_global_scale"]).astype(jnp.float32).reshape(-1)[0]
            return CanonicalQuantizedWeight.from_dense(dense, spec=target, output_scale=1.0 / global_scale)

        if encoding == "int4":
            # compressed-tensors packs int4 sequentially along the input axis
            # — but along the LAST axis of the HF [out, in] layout, i.e. the
            # transpose of GPTQ's [in // 8, out] packing — with codes stored
            # unsigned at a +8 offset. It needs its own decoder.
            dense = decode_compressed_int4(tensors["weight_packed"], tensors["weight_scale"])
            return CanonicalQuantizedWeight.from_dense(dense, spec=target)

        dense = decode_scaled_elements(tensors["weight"], tensors["weight_scale"], block_shape=source.block_shape)
        return CanonicalQuantizedWeight.from_dense(dense, spec=target)


def _is_class_name(target: str) -> bool:
    """Whether a ``targets`` entry names a module class rather than a module.

    compressed-tensors overloads one list for both. Class names select every
    instance in the model (a catch-all scheme); module names select specific
    layers. They are told apart by case: ``"Linear"`` versus ``"q_proj"``.

    Args:
        target: One entry from a config group's ``targets``.

    Returns:
        ``True`` when the entry looks like a class name.
    """
    return bool(target) and not target.startswith("re:") and "." not in target and target[0].isupper()


def re_escape_path(target: str) -> str:
    """Turn a literal module target into a regex matching it as a path suffix.

    Args:
        target: A literal dotted module name from a ``targets`` list.

    Returns:
        An anchored regex matching that module at any depth.
    """
    import re

    return rf"(^|\.){re.escape(target)}$"
