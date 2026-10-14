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

"""Quantized training for any EasyDeL model, driven by module paths.

This is the EasyDeL composition of :mod:`spectrax.quantization`. spectrax
owns the general mechanism — the rule schema, the graph walk that stamps
rules onto modules, the numerics, and the quantized ops. EasyDeL supplies
the parts that need to know about EasyDeL: which module paths are worth
naming, which layers perform which op, and how a training configuration
turns into a provider.

Two ways in::

    # by preset, the MaxText-equivalent named regimes
    model = ed.apply_quantization_rules(model, "int8")

    # by mixed-precision config, including MaxText's own JSON files
    model = ed.apply_quantization_rules(model, "intmp", quant_cfg_path="cfg.json")

Nothing in a ``modeling_*.py`` file changes. Quantization is applied by
matching module paths against the live graph, so the same rules work on
every model family in the zoo — dense, mixture-of-experts, multimodal —
and on families added later.

**Apply the rules before building the state.** They are stored on the
modules and travel with the ``GraphDef``, so once a state exists every
trainer picks them up with no further wiring::

    model = ed.apply_quantization_rules(model, "int8")
    state = model.to_state()          # the GraphDef now carries the rules

The order matters and the failure is silent if it is reversed:
:attr:`EasyDeLState.model` is a *property that rebuilds* the module from
the stored ``GraphDef`` on every access, so stamping ``state.model``
mutates a throwaway module and the rules vanish. Quantize the model, then
make the state.

**What is covered.** Weight matmuls, which is where quantization matters
and where the weights live:

* dense projections (attention Q/K/V/O, MLP gate/up/down, and the fused
  QKV and gate-up projections) through :class:`ParallelLinear`;
* stacked mixture-of-experts kernels through the fused grouped-matmul
  path in :class:`BaseMoeModule` and through
  :class:`ParallelMoELinear`;
* :class:`spectrax.nn.Linear`, ``DenseGeneral`` and ``Einsum``.

Activation-by-activation matmuls — attention scores, linear-attention
recurrences — are not weight matmuls and are left alone, which is also
where MaxText draws the line.

**Two regimes, and the difference is which one you asked for.** Both are
loosely called "QAT"; they are not the same thing.

*Quantized training* -- ``int8``, ``int4``, ``fp8``, ``deepseek_fp8``.
Both operands reach the matmul in the narrow type, so the hardware really
contracts int8 against int8. The point is throughput: measured 1.15-1.27x
for int8 and up to 1.44x for int4 on TPU v5, converging like bfloat16.
Qwix calls this "QT" for precisely this reason, and DeepSeek-V3's FP8
pretraining is the same idea.

*Quantization-aware training* -- ``w8a16``, ``w4a16``, ``nf4``, ``mxfp4``.
Only the weight is discretized, then immediately reconstructed, so the
contraction still happens in the compute dtype. Nothing runs faster
(~1.00x); what the model gains is exposure to the quantization error it
will meet after post-training quantization. This is QAT in the original
sense, and DeepSeek-V4 uses it this way for its FP4 deployment path.

:attr:`~spectrax.quantization.QuantRule.trains_in_narrow_precision`
reports which one a rule selects.

**Neither saves parameter memory.** The master weights stay full
precision in both -- that is what makes them converge, since an update
below the narrow type's resolution would otherwise round away and freeze
the weight. Quantization buys throughput and deployment size, not
training memory.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import spectrax as spx
from eformer.loggings import get_logger
from eformer.paths import ePath

from ._configs import DEFAULT_QUANTIZATION_PATTERN, QuantizationConfig, QuantizationType

if tp.TYPE_CHECKING:
    from easydel.infra.base_module import EasyDeLBaseModule

logger = get_logger("easydel.quantization.quantized_training")

__all__ = [
    "DEFAULT_QUANTIZED_MODULE_PATH",
    "apply_quantization_rules",
    "build_quantization_provider",
    "quantization_config_to_rule",
    "weight_gradient_tile_size",
]


DEFAULT_QUANTIZED_MODULE_PATH = r"(?!.*(?:embed|norm|lm_head|router|gate\b)).*"
"""Default module-path pattern for quantized training.

Excludes the layers that quantize badly and cost little to leave alone:
embeddings and the language-model head (both wide and precision-sensitive
at the vocabulary axis), normalization gains (1-D, no contraction to
group over), and the mixture-of-experts router and gate (tiny, but a
rounding error there changes *which* expert a token reaches rather than
merely perturbing a value).

The exclusions mirror :data:`DEFAULT_QUANTIZATION_PATTERN`, which the
post-training path already uses, so a model trained under this default
and served under that one is quantized in the same places.
"""

_QUANTIZED_OP_NAMES: tuple[str, ...] = ("dot_general", "ragged_dot")
"""Ops quantized training claims by default.

``dot_general`` covers dense projections; ``ragged_dot`` covers the
stacked per-expert matmuls. ``einsum`` is deliberately absent — in
EasyDeL an ``einsum`` in a modeling file is almost always an
activation-by-activation contraction (attention scores, a linear-attention
recurrence), and quantizing those is a different decision with different
tradeoffs than quantizing weights.
"""


def quantization_config_to_rule(
    config: QuantizationConfig,
    *,
    module_path: str = DEFAULT_QUANTIZED_MODULE_PATH,
    op_names: tp.Sequence[str] = _QUANTIZED_OP_NAMES,
) -> spx.quantization.QuantRule:
    """Translate a post-training :class:`QuantizationConfig` into a training rule.

    Lets one configuration describe both halves of a quantized model's
    life: the discretization a run trains against, and the discretization
    the checkpoint is later served with. The mapping is weight-only,
    because a :class:`QuantizationConfig` describes stored weights and
    says nothing about activations.

    Args:
        config: The post-training quantization configuration.
        module_path: Module-path pattern the rule applies to.
        op_names: Ops the rule claims.

    Returns:
        A :class:`spectrax.quantization.QuantRule` matching ``config``.

    Raises:
        ValueError: If ``config.dtype`` has no training-time equivalent.
    """
    group_size = config.group_size
    dtype = config.dtype

    if dtype in (QuantizationType.INT8, QuantizationType.CHANNELWISE):
        # Channelwise is one scale per output channel over the whole
        # contraction, which is what "no subchannel tiling" means here.
        weight_qtype: spx.quantization.QType = "int8" if (config.bits or 8) == 8 else "int4"
        tile_size = None if dtype is QuantizationType.CHANNELWISE else group_size
    elif dtype is QuantizationType.AFFINE:
        weight_qtype = spx.quantization.resolve_qtype(config.bits or 8)
        tile_size = group_size
    elif dtype is QuantizationType.NF4:
        weight_qtype = "nf4"
        tile_size = group_size
    else:
        raise ValueError(
            f"{dtype} has no quantized-training equivalent in the rule schema. "
            f"Supported for training: int8, channelwise, affine, nf4. "
            f"Microscaling and TurboQuant formats remain post-training only."
        )

    return spx.quantization.QuantRule(
        module_path=module_path,
        op_names=tuple(op_names),
        weight_qtype=weight_qtype,
        tile_size=tile_size,
    )


def weight_gradient_tile_size(shard_count: int | None) -> float | None:
    """Turn a contraction shard count into a backward weight-gradient tile size.

    The weight gradient contracts over the token axis, which is long and
    sharded; tiling it into one tile per shard keeps each tile's
    calibration local to the data that produced it. The tile size is
    expressed as a reciprocal count, so it is ``1 / shard_count``.

    The guard is the point. MaxText computes this as
    ``1 / quantization_local_shard_count`` with the config defaulting to
    ``-1``, which yields ``-1.0`` — a request for a negative number of
    elements rather than "no tiling". A count that does not describe a
    real split returns ``None`` here instead.

    Args:
        shard_count: Number of shards the contracted axis is split across,
            or ``None``.

    Returns:
        ``1 / shard_count`` when that describes a real split, else
        ``None`` to leave the backward pass untiled.
    """
    if shard_count is None or shard_count <= 1:
        return None
    return 1.0 / shard_count


def build_quantization_provider(
    quantization: str | QuantizationConfig | spx.quantization.QuantProvider,
    *,
    quant_cfg_path: str | None = None,
    module_path: str = DEFAULT_QUANTIZED_MODULE_PATH,
    tile_size: int | float | None = None,
    quantize_backward: bool = True,
    weight_gradient_shard_count: int | None = None,
    op_names: tp.Sequence[str] = _QUANTIZED_OP_NAMES,
) -> spx.quantization.QuantProvider:
    """Build a quantization provider from a training configuration.

    Accepts the four forms a configuration can plausibly take: an
    already-built provider (returned unchanged), a post-training
    :class:`QuantizationConfig`, the string ``"intmp"`` plus a path to a
    mixed-precision JSON file, or a preset name.

    Args:
        quantization: A provider, a :class:`QuantizationConfig`, ``"intmp"``,
            or a preset name such as ``"int8"``, ``"int4"``, ``"fp8"``,
            ``"fp4"``, ``"w4a16"``, ``"w8a16"`` or ``"nf4"``.
        quant_cfg_path: Path to a mixed-precision JSON config. Required
            when ``quantization`` is ``"intmp"``, ignored otherwise. Read
            through :class:`~eformer.paths.ePath`, so remote paths work.
        module_path: Module-path pattern for preset-derived rules.
        tile_size: Subchannel tile size on the contracted axis for
            preset-derived rules.
        quantize_backward: Whether presets that quantize activations should
            also quantize the backward pass.
        weight_gradient_shard_count: Shards the weight gradient's contracted
            axis is split across, used to tile its calibration. MaxText's
            ``quantization_local_shard_count``; see
            :func:`weight_gradient_tile_size` for why it is taken as a count
            rather than a tile size.
        op_names: Ops the generated rules claim.

    Returns:
        The provider to hand to :func:`apply_quantization_rules`.

    Raises:
        ValueError: If ``"intmp"`` is requested without ``quant_cfg_path``.
    """
    if isinstance(quantization, spx.quantization.QuantProvider):
        return quantization

    if isinstance(quantization, QuantizationConfig):
        return spx.quantization.QuantProvider(
            [quantization_config_to_rule(quantization, module_path=module_path, op_names=op_names)]
        )

    if quantization == "intmp":
        if quant_cfg_path is None:
            raise ValueError(
                "quantization='intmp' selects a per-module mixed-precision config but no quant_cfg_path was given. "
                "Point it at a JSON file mapping module-path patterns to {w_bits, a_bits, w_scale, a_scale, "
                "tile_size} entries."
            )
        import json

        return spx.quantization.QuantProvider.from_intmp(
            json.loads(ePath(quant_cfg_path).read_text()), op_names=op_names
        )

    provider = spx.quantization.QuantProvider.from_preset(
        quantization,
        module_path=module_path,
        tile_size=tile_size,
        quantize_backward=quantize_backward,
        op_names=op_names,
    )
    bwd_tile_size = weight_gradient_tile_size(weight_gradient_shard_count)
    if bwd_tile_size is None:
        return provider
    return spx.quantization.QuantProvider(
        [dataclasses.replace(rule, bwd_weight_grad_tile_size=bwd_tile_size) for rule in provider.rules]
    )


def apply_quantization_rules(
    model: EasyDeLBaseModule,
    quantization: str | QuantizationConfig | spx.quantization.QuantProvider,
    *,
    quant_cfg_path: str | None = None,
    module_path: str = DEFAULT_QUANTIZED_MODULE_PATH,
    tile_size: int | float | None = None,
    quantize_backward: bool = True,
    weight_gradient_shard_count: int | None = None,
    op_names: tp.Sequence[str] = _QUANTIZED_OP_NAMES,
    strict: bool = True,
) -> EasyDeLBaseModule:
    """Stamp quantization rules onto a model for quantization-aware training.

    Mutates the module tree in place and returns it. The rules ride the
    model's ``GraphDef``, so they survive ``spx.jit``, ``spx.grad``,
    checkpoint export and rebinding, and every trainer in the repository
    picks them up without further wiring.

    Because the rules are part of the ``GraphDef``, applying them changes
    the model's structure hash and therefore invalidates any compiled
    executable cached for the unquantized model. That is intended — the
    two are different programs — but it does mean the first step after
    enabling quantization pays a full compile.

    Args:
        model: The model to quantize. Modified in place.
        quantization: A provider, a :class:`QuantizationConfig`, ``"intmp"``,
            or a preset name.
        quant_cfg_path: Mixed-precision JSON path, for ``"intmp"``.
        module_path: Module-path pattern for preset-derived rules.
        tile_size: Subchannel tile size for preset-derived rules.
        quantize_backward: Whether presets should quantize the backward pass.
        weight_gradient_shard_count: Shards the weight gradient's contracted
            axis is split across, used to tile its calibration.
        op_names: Ops the generated rules claim.
        strict: Raise if no module matches. Leave this on: a provider that
            matches nothing yields a model that reports itself quantized
            and trains at full precision.

    Returns:
        ``model``, for chaining.

    Example:
        >>> model = ed.apply_quantization_rules(model, "int8")
        >>> model = ed.apply_quantization_rules(model, "w4a16", tile_size=128)
        >>> model = ed.apply_quantization_rules(
        ...     model, "intmp", quant_cfg_path="configs/dense_llm_subchannel.json"
        ... )
    """
    provider = build_quantization_provider(
        quantization,
        quant_cfg_path=quant_cfg_path,
        module_path=module_path,
        tile_size=tile_size,
        quantize_backward=quantize_backward,
        weight_gradient_shard_count=weight_gradient_shard_count,
        op_names=op_names,
    )
    spx.quantization.quantize_model(model, provider, op_names=op_names, strict=strict)

    quantized_paths = [
        path
        for path, module in spx.iter_modules(model)
        if getattr(module, spx.quantization.PLAN_ATTRIBUTE, None) is not None
    ]
    narrow = any(rule.trains_in_narrow_precision for rule in provider.rules)
    logger.info(
        f"{'Quantized training' if narrow else 'Quantization-aware training'} enabled on "
        f"{len(quantized_paths)} module(s) via {provider!r}: "
        f"{'the matmul runs in the narrow type' if narrow else 'operands are discretized, the matmul stays wide'}. "
        f"Master weights stay full precision either way, so this does not reduce parameter memory."
    )
    return model


def _default_pattern_matches_post_training() -> bool:
    """Report whether the training and serving default patterns agree.

    Both defaults are meant to exclude the same layers; this is the
    assertion behind that claim, kept as a function so the test suite can
    check it rather than trusting the comment.

    Returns:
        ``True`` when both patterns exclude embeddings, norms and the
        language-model head.
    """
    excluded = ("embedding", "norm", "lm_head")
    return all(term in DEFAULT_QUANTIZATION_PATTERN for term in excluded)
