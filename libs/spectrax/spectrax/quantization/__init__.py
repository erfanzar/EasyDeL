# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantization-aware training, driven by module-path rules.

Quantize any :class:`~spectrax.Module` model without touching the model's
own code. Write rules that match module paths, stamp them onto the graph
once, and let layers apply them::

    import spectrax as spx

    provider = spx.quantization.QuantProvider([
        spx.quantization.QuantRule(
            module_path="layers.*",
            weight_qtype="int4",
            act_qtype="int4",
            bwd_qtype="int4",
            tile_size=128,
            op_names=("dot_general",),
        ),
    ])
    model = spx.quantization.quantize_model(model, provider)

:func:`quantize_model` walks the tree with :func:`spectrax.iter_modules`,
resolves each module's path against the rules, and stores the winner on
the module. The match happens **once, before the first trace**, which is
what lets the mechanism be model-agnostic without a module-path stack at
trace time and without patching any JAX function. Because the stamp lands
in the module's opaque attributes it survives ``export``/``bind`` and
rides the ``GraphDef`` through ``jit``, ``grad``, ``scan``, ``remat`` and
the MPMD split; because it is part of the ``GraphDef`` it also changes
``structure_hash()``, so compile caches key on the numeric regime rather
than silently reusing a full-precision executable.

Layer authors use two entry points, and the choice is about what the
layer is willing to give up:

* :func:`fake_quant` keeps the layer's own matmul — a Pallas kernel, a
  fused collective matmul, a grouped matmul with tuned tiles — and only
  makes the operands carry quantization error. Works everywhere; buys
  accuracy simulation, not speed.
* :func:`qdot_general` and :func:`qeinsum` hand the contraction over, so
  the forward may run in the narrow type and the backward pass can be
  quantized too. Only applies where the op really is a contraction.

Supported types: ``int2``-``int8`` (``int4`` and ``int8`` natively, the
rest stored in the next wider integer), ``float8_e4m3fn``,
``float8_e5m2``, ``float4_e2m1fn``, and ``nf4``. Scales may be
per-tensor, per-channel, or subchannel with an explicit tile size.

The framework-neutral numerics are ported from Google's Qwix
(Apache-2.0), which is Flax-only at every entry point and would pull Flax
into spectrax's dependency tree; see ``.claude/projects/quantized-training.md``
for the full reasoning.
"""

from __future__ import annotations

from ._calibrate import (
    Calibration,
    HowToQuantize,
    calibrate,
    compute_scale_zero_point,
    dequantize,
    quantize,
    quantize_with_scale_zero_point,
    scale_shape,
)
from ._dot import (
    MIN_TILE_SIZE_FOR_OUTPUT_DEQUANT,
    accumulator_and_result_type,
    dot_general,
    how_to_quantize_for_dot,
    loop_dot_general,
)
from ._narrow_storage import (
    narrow_storage_update,
    scale_loss_for_narrow_gradients,
    stochastic_round,
    suggested_loss_scale,
)
from ._numerics import (
    NoiseFn,
    QType,
    asymmetric_bound,
    can_dequantize_on_output,
    convert_from,
    convert_to,
    is_pseudo_qtype,
    nf4_buckets,
    qtype_bits,
    qtype_name,
    should_quantize,
    storage_dtype,
    symmetric_bound,
    uniform_noise,
)
from ._ops import fake_quant, qdot_general, qeinsum
from ._qarray import (
    MaybeQArray,
    QArray,
    generic_broadcast_op,
    resolve_tile_size,
    split_axis,
    tiled_axes,
    transpose_array,
    validate_qarray,
)
from ._ragged import qragged_dot
from ._rules import (
    DEFAULT_OP_NAMES,
    PLAN_ATTRIBUTE,
    QuantPlan,
    QuantProvider,
    QuantRule,
    quantize_model,
    resolve_qtype,
    rule_for,
    unquantize_model,
)

__all__ = [
    "DEFAULT_OP_NAMES",
    "MIN_TILE_SIZE_FOR_OUTPUT_DEQUANT",
    "PLAN_ATTRIBUTE",
    "Calibration",
    "HowToQuantize",
    "MaybeQArray",
    "NoiseFn",
    "QArray",
    "QType",
    "QuantPlan",
    "QuantProvider",
    "QuantRule",
    "accumulator_and_result_type",
    "asymmetric_bound",
    "calibrate",
    "can_dequantize_on_output",
    "compute_scale_zero_point",
    "convert_from",
    "convert_to",
    "dequantize",
    "dot_general",
    "fake_quant",
    "generic_broadcast_op",
    "how_to_quantize_for_dot",
    "is_pseudo_qtype",
    "loop_dot_general",
    "narrow_storage_update",
    "nf4_buckets",
    "qdot_general",
    "qeinsum",
    "qragged_dot",
    "qtype_bits",
    "qtype_name",
    "quantize",
    "quantize_model",
    "quantize_with_scale_zero_point",
    "resolve_qtype",
    "resolve_tile_size",
    "rule_for",
    "scale_loss_for_narrow_gradients",
    "scale_shape",
    "should_quantize",
    "split_axis",
    "stochastic_round",
    "storage_dtype",
    "suggested_loss_scale",
    "symmetric_bound",
    "tiled_axes",
    "transpose_array",
    "uniform_noise",
    "unquantize_model",
    "validate_qarray",
]
