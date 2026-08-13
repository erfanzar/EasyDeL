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

"""Shared gated-MLP forward for the model zoo.

Every modeling MLP that follows the standard shape (fused ``gate_up_proj``
column projection -> activation-gated combine -> ``down_proj`` row
projection, or the two-projection ``gate_proj``/``up_proj`` variant)
delegates its forward to :func:`gated_mlp_forward`. The helper owns the
routing decision in ONE place for the whole zoo:

* **dense bf16, supported sharding** -> the ejkernel ``fused_mlp`` Pallas
  path (single-dispatch fused kernel, measured 1.01-1.02x model-level on the
  layouts where it engages; the ejkernel resolver enforces the never-slower
  policy: tp / K-sharded fsdp weights / decode row counts all fall back);
* **per-channel symmetric int8 quantized linears** -> the ejkernel
  ``fused_mlp`` channelwise integer path (fused-upcast / int-MXU dots) —
  inference only: the frozen-weight ``custom_vjp`` would zero trainable
  ``quant_scales`` gradients during training;
* **anything else** (biases, grouped/affine quantization, separate
  projections, unknown activation, sequence-parallel activations, pipeline
  stages) -> the exact composition the modeling files ran before this
  helper existed, byte-identical, including the ``checkpoint_name`` tags
  remat name-policies rely on.

Training under a *name-based* remat policy (``MLP_NOTSAVEABLE`` family,
``SAVE_ONLY/EXCEPT_THESE_NAMES``) always takes the unfused composition —
the fused ``custom_vjp`` would silently defeat those policies — and
``FORCE_NATIVE_RUNTIME`` (the repo-wide XLA-reference flag) forces the
unfused composition everywhere so the standard bisect covers this route.

The specs handed to ejkernel describe how EasyDeL's partition rules
ACTUALLY shard the arrays (rows over ``("dp","fsdp")``; column weight
``(("fsdp","sp"),"tp")``; row weight ``("tp",("fsdp","sp"))``); nothing is
re-replicated and the output keeps the input's row sharding.
"""

from __future__ import annotations

import typing as tp

import jax
from ejkernel.modules import fused_mlp as ejkernel_fused_mlp
from ejkernel.modules import resolve_mlp_combine
from ejkernel.modules.operations.fused_mlp import resolve_dense_sharded_plan
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
from spectrax import apply_logical_sharding, common_types

from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.sharding import inference_mode_forces_decode
from easydel.layers.layouts import split_fused_gate_up_projection, tensor_parallel_size
from easydel.layers.linears import W4A4_FUSED_PAIR_MIN_PACKED_BYTES, ParallelLinearQuantized
from easydel.utils.helpers import check_bool_flag
from easydel.utils.inference_mode import is_inference_mode

if tp.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig

__all__ = ["gated_mlp_forward"]

#: Remat policies that select residuals by ``checkpoint_name``. The ejkernel
#: fused route replaces the checkpoint_name-tagged composition with a
#: ``custom_vjp`` that pins its own residuals, so these policies would
#: silently lose effect on the MLP — training falls back to the unfused
#: composition when one is active (inference is unaffected).
_NAME_BASED_REMAT_POLICIES: frozenset[str] = frozenset(
    str(policy)
    for policy in (
        EasyDeLGradientCheckPointers.SAVE_ANYTHING_EXCEPT_THESE_NAMES,
        EasyDeLGradientCheckPointers.SAVE_ANY_NAMES_BUT_THESE,
        EasyDeLGradientCheckPointers.SAVE_ONLY_THESE_NAMES,
        EasyDeLGradientCheckPointers.MLP_NOTSAVEABLE,
        EasyDeLGradientCheckPointers.ATTN_NOTSAVEABLE,
        EasyDeLGradientCheckPointers.MLP_ATTN_NOTSAVEABLE,
        EasyDeLGradientCheckPointers.LMHEAD_NOTSAVEABLE,
        EasyDeLGradientCheckPointers.ATTN_LMHEAD_NOTSAVEABLE,
        EasyDeLGradientCheckPointers.MLP_LMHEAD_NOTSAVEABLE,
        EasyDeLGradientCheckPointers.SAVE_FROM_BOTH_POLICIES,
    )
)


def _uses_name_based_remat(config: EasyDeLBaseConfig) -> bool:
    """Whether the config's gradient-checkpointing policy is name-based."""
    policy = getattr(config, "gradient_checkpointing", None)
    if policy is None:
        return False
    return str(policy) in _NAME_BASED_REMAT_POLICIES


#: HF ``hidden_act`` names the ejkernel fused kernels support, mapped to the
#: shared ejkernel activation table.
_ACT_NAME_MAP: dict[str, str] = {
    "silu": "silu",
    "swish": "silu",
    "gelu": "gelu",
    "gelu_new": "gelu_tanh",
    "gelu_pytorch_tanh": "gelu_tanh",
    "relu": "relu",
    "sigmoid": "sigmoid",
}

#: Fallback weight specs matching EasyDeL's stock partition rules; used only
#: when a parameter carries no resolvable sharding metadata.
_COLUMN_W_SPEC = PartitionSpec(("fsdp", "sp"), "tp")
_ROW_W_SPEC = PartitionSpec("tp", ("fsdp", "sp"))


def _flatten_entries(*entries) -> tuple[str, ...] | None:
    """Merge PartitionSpec entries into one compound entry (or ``None``)."""
    names: list[str] = []
    for entry in entries:
        if entry is None:
            continue
        if isinstance(entry, str):
            names.append(entry)
        else:
            names.extend(a for a in entry if a is not None)
    return tuple(names) if names else None


def _resolved_specs(mlp, config, hidden_states):
    """Resolve the ACTUAL specs the model's sharding machinery assigns.

    Mirrors how the XLA path gets its shardings: the hidden-state spec comes
    from the same ``PartitionManager`` resolution ``apply_logical_sharding``
    uses (``(BATCH, QUERY_LENGTH, EMBED)``); weight specs resolve from each
    parameter's declared ``spx.Sharding`` axis names through the same manager.

    The activation mode is taken from the ambient
    :func:`~easydel.infra.sharding.decode_mode_specs` scope first and only
    then from the shape. Both matter: ``_partition_spec_for_axis_names``
    takes an ALREADY-resolved mode and so bypasses the scope-aware
    ``_resolve_mode``, and eSurge packs decode as ``[1, N]`` tokens — which
    shape-classifies as training. Deriving the mode from ``shape[1]`` alone
    therefore handed the fused kernel train-mode (tp-sharded) in_specs for a
    residual stream the rest of the trace had already replicated.

    Returns:
        ``(x2d_spec, wf_spec, wd_spec)`` where the x spec merges the batch
        and sequence entries for the ``[B, S, H] -> [B*S, H]`` flatten, or
        ``None`` when resolution is unavailable.
    """
    pm = getattr(config, "runtime_sharding_resolver", None)
    if pm is None:
        return None
    if inference_mode_forces_decode() or (hidden_states.ndim >= 2 and hidden_states.shape[1] == 1):
        mode = common_types.MODE_DECODE
    else:
        mode = common_types.MODE_TRAIN
    try:
        if hasattr(pm, "resolve_spec"):  # spectrax PartitionManager
            spec3 = pm.resolve_spec(axes=list(common_types.HiddenStateSharding.axes), mode=mode)
        else:  # EasyDeL RuntimeShardingResolver
            spec3 = pm._partition_spec_for_axis_names(list(common_types.HiddenStateSharding.axes), mode)
    except Exception:
        return None
    entries = tuple(spec3) + (None,) * (3 - len(spec3))
    rows = _flatten_entries(entries[0], entries[1])
    x2d_spec = PartitionSpec(rows, entries[2])

    def param_spec(param, fallback):
        names = getattr(getattr(param, "sharding", None), "axis_names", None)
        if not names:
            return fallback
        try:
            if hasattr(pm, "resolve_spec"):
                return pm.resolve_spec(axes=list(names), mode=common_types.MODE_TRAIN)
            return pm._partition_spec_for_axis_names(list(names), common_types.MODE_TRAIN)
        except Exception:
            return fallback

    wf_spec = param_spec(getattr(mlp.gate_up_proj, "weight", None), _COLUMN_W_SPEC)
    wd_spec = param_spec(getattr(mlp.down_proj, "weight", None), _ROW_W_SPEC)
    return (x2d_spec, wf_spec, wd_spec)


def _axis_size(mesh, name: str) -> int:
    try:
        return int(dict(getattr(mesh, "jax_mesh", mesh).shape)[name])
    except (KeyError, TypeError, AttributeError):
        return 1


def _mesh_supported(config: EasyDeLBaseConfig):
    """The mesh regimes the fused route understands today.

    ``pp > 1`` runs layers under stage meshes — fall back to the legacy
    composition there. Sequence sharding is fine: the flatten merges the
    batch and sequence entries into the rows entry of the x spec.
    """
    mesh = getattr(config, "mesh", None)
    if mesh is None:
        return None
    if _axis_size(mesh, "pp") != 1:
        return None
    return getattr(mesh, "jax_mesh", mesh)


def _declared_combine(mlp) -> tuple[str, tuple[float, ...]] | None:
    """Model-declared combine: ``mlp.fused_act_name`` / ``mlp.fused_act_params``.

    Families whose gate/up combine is not a unary activation (clamped
    SwiGLU, muP-scaled silu, SITU, ...) declare it explicitly in
    ``__init__``; the declaration bypasses the ``ACT2FN`` identity guard
    because the model is stating — not implying — what it computes. The
    name/params must resolve in ejkernel's combine table or the declaration
    is ignored (safe fallback would be wrong here, so we validate eagerly).
    """
    name = getattr(mlp, "fused_act_name", None)
    if name is None:
        return None
    params = tuple(float(v) for v in getattr(mlp, "fused_act_params", ()) or ())
    resolve_mlp_combine(name, params)
    return name, params


def _act_name(config: EasyDeLBaseConfig, mlp=None) -> str | None:
    """Activation name for the fused kernels, or ``None`` to fall back.

    Guards the wrong-activation hazard: some families derive ``act_fn`` from
    a different config field than ``hidden_act`` (gemma-v1 reads
    ``hidden_activation``, xerxes reads ``swish_run``). The fused route only
    engages when the module's own ``act_fn`` is exactly the callable
    ``ACT2FN[hidden_act]`` would give — an identity mismatch means we cannot
    prove the kernel computes the same function, so the legacy composition
    (which calls ``mlp.act_fn`` itself) runs instead.
    """
    from easydel.infra.utils import ACT2FN

    for field in ("hidden_act", "hidden_activation"):
        hidden_act = str(getattr(config, field, "") or "").lower()
        name = _ACT_NAME_MAP.get(hidden_act)
        if name is None:
            continue
        if mlp is None:
            return name
        try:
            expected = ACT2FN[hidden_act]
        except Exception:
            continue
        if getattr(mlp, "act_fn", None) is expected:
            return name
    return None


def _dense_fused_call(mlp, config, x2d: jax.Array, specs) -> jax.Array | None:
    """Dense bf16 route: ejkernel fused_mlp when the plan engages."""
    gate_up_proj = getattr(mlp, "gate_up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    if gate_up_proj is None or down_proj is None:
        return None
    if isinstance(gate_up_proj, ParallelLinearQuantized) or isinstance(down_proj, ParallelLinearQuantized):
        return None
    if getattr(gate_up_proj, "bias", None) is not None or getattr(down_proj, "bias", None) is not None:
        return None
    declared = _declared_combine(mlp)
    act, act_params = declared if declared is not None else (_act_name(config, mlp), ())
    mesh = _mesh_supported(config)
    if act is None or mesh is None:
        return None
    wf = getattr(getattr(gate_up_proj, "weight", None), "value", None)
    wd = getattr(getattr(down_proj, "weight", None), "value", None)
    if wf is None or wd is None or not jnp.issubdtype(wf.dtype, jnp.floating):
        return None
    if wf.ndim != 2 or wd.ndim != 2 or wf.shape[1] != 2 * wd.shape[0] or wd.shape[1] != wf.shape[0]:
        return None

    if specs is None:
        return None
    segments = max(1, int(tensor_parallel_size(config, arr=wf)))
    layout = "interleaved" if segments > 1 else "concat"
    plan = resolve_dense_sharded_plan(
        x2d,
        wf,
        wd,
        mesh=mesh,
        in_specs=specs,
        activation=act,
        act_params=act_params,
        layout=layout,
        segments=segments,
    )
    if plan is None:
        return None
    return ejkernel_fused_mlp(
        x2d,
        gate_up=wf,
        w_down=wd,
        gate_up_layout=layout,
        gate_up_segments=segments,
        activation=act,
        act_params=act_params,
        mesh=mesh,
        in_specs=specs,
    )


def _quantized_fused_call(mlp, config, x2d: jax.Array) -> jax.Array | None:
    """Per-channel symmetric int8 route through the fused integer path."""
    gate_up_proj = getattr(mlp, "gate_up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    if not isinstance(gate_up_proj, ParallelLinearQuantized) or not isinstance(down_proj, ParallelLinearQuantized):
        return None
    if getattr(gate_up_proj, "use_bias", False) or getattr(down_proj, "use_bias", False):
        return None
    declared = _declared_combine(mlp)
    act, act_params = declared if declared is not None else (_act_name(config, mlp), ())
    if act is None or _mesh_supported(config) is None:
        return None

    def per_channel_codes(linear):
        try:
            mode, _group_size, bits, needs_biases = linear._resolve_ejkernel_params()
        except Exception:
            return None
        if needs_biases or bits not in (4, 8):
            return None
        codes = getattr(linear.quant_kernel, "value", linear.quant_kernel)
        scales = getattr(linear.quant_scales, "value", linear.quant_scales)
        if codes is None or scales is None or codes.ndim != 2:
            return None
        if bits == 8 and codes.dtype != jnp.int8:
            return None
        if bits == 4 and codes.dtype != jnp.int4:
            return None
        # Per-output-channel: exactly one scale group along the contraction.
        if mode != "channelwise" or int(scales.size) != int(codes.shape[-1]):
            return None
        packed = getattr(getattr(linear, "quant_kernel_packed", None), "value", None)
        return codes, scales.reshape(1, codes.shape[-1]).astype(jnp.float32), bits, packed

    gu = per_channel_codes(gate_up_proj)
    dn = per_channel_codes(down_proj)
    if gu is None or dn is None:
        return None
    gu_codes, gu_scales, gu_bits, gu_packed = gu
    dn_codes, dn_scales, dn_bits, dn_packed = dn
    if gu_bits != dn_bits:
        return None
    if gu_codes.shape[1] != 2 * dn_codes.shape[0] or dn_codes.shape[1] != gu_codes.shape[0]:
        return None

    segments = max(1, int(tensor_parallel_size(config, arr=gu_codes)))
    layout = "interleaved" if segments > 1 else "concat"

    act_bits = getattr(getattr(gate_up_proj, "config", None), "activation_bits", None)
    kwargs: dict = {}
    if act_bits is not None:
        kwargs["quantize_activations"] = True
        kwargs["activation_bits"] = int(act_bits)
    packed_bytes = (gu_packed.size + dn_packed.size) if (gu_packed is not None and dn_packed is not None) else 0
    if (
        gu_bits == 4
        and gu_packed is not None
        and dn_packed is not None
        and not act_params
        # The fused W4A4 decode kernel wins where packed-weight INGEST
        # dominates its dispatch cost (measured ~2x at 27B decode shapes);
        # at small-model shapes it measured 0.79x vs bf16 (v5p, 4B) —
        # never-slower gates it to big per-shard weights. Below the gate,
        # decode runs the W4A16 fused-upcast path (measured 1.17x) and
        # prefill still uses the int4 MXU dots. The same floor gates
        # whether a quantized linear materializes its packed companion at
        # all (``should_pack_int4_companion``).
        and packed_bytes >= W4A4_FUSED_PAIR_MIN_PACKED_BYTES
    ):
        # Packed gate/up halves come from splitting the packed fused array
        # along its columns (rows are packed K-pairs, so the column split
        # matches the unpacked layout exactly).
        from ejkernel.modules import split_gate_up as _split_packed

        gate_packed, up_packed = _split_packed(gu_packed, layout=layout, segments=segments)
        kwargs["packed_weights"] = (gate_packed, up_packed, dn_packed)
    return ejkernel_fused_mlp(
        x2d,
        gate_up=gu_codes,
        w_down=dn_codes,
        gate_up_layout=layout,
        gate_up_segments=segments,
        gate_scale=gu_scales,
        down_scale=dn_scales,
        activation=act,
        act_params=act_params,
        **kwargs,
    )


def _legacy_composition(mlp, config, hidden_states: jax.Array) -> jax.Array:
    """The exact composition the modeling files ran before the helper.

    Declared-combine families (SITU, clamped SwiGLU, ...) run their combine
    here too — for them ``act_fn(gate) * up`` would be wrong math, so the
    same declared combine executes on both the fused and fallback paths.
    """
    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    if gate_proj is not None and up_proj is not None:
        gate_raw = checkpoint_name(gate_proj(hidden_states), "mlp_gate_raw")
        up = checkpoint_name(up_proj(hidden_states), "mlp_up")
    else:
        gate_up = checkpoint_name(mlp.gate_up_proj(hidden_states), "mlp_gate_up")
        gate_raw, up = split_fused_gate_up_projection(gate_up, config=config)
    declared = _declared_combine(mlp)
    if declared is not None:
        name, params = declared
        hidden = checkpoint_name(resolve_mlp_combine(name, params)(gate_raw, up), "mlp_gate")
        return checkpoint_name(mlp.down_proj(hidden), "mlp_down")
    gate = checkpoint_name(mlp.act_fn(gate_raw), "mlp_gate")
    return checkpoint_name(mlp.down_proj(gate * up), "mlp_down")


def gated_mlp_forward(
    mlp,
    hidden_states: jax.Array,
    *,
    dropout: tp.Callable[[jax.Array], jax.Array] | None = None,
    output_scale: float | None = None,
) -> jax.Array:
    """Shared forward for standard gated MLP blocks.

    Args:
        mlp: The modeling MLP module. Expected attributes: ``config``,
            ``act_fn``, ``down_proj``, and either ``gate_up_proj`` (fused)
            or ``gate_proj``/``up_proj`` (separate).
        hidden_states: Activations ``[..., hidden]``.
        dropout: Optional callable applied after the down projection
            (e.g. ``self.dropout`` for families that keep MLP dropout).
        output_scale: Optional scalar multiplier applied right after the
            down projection (Falcon-H1's muP ``down_multiplier``).

    Returns:
        ``[..., hidden]`` activations, logical-sharded, tagged
        ``"mlp_output"``.
    """
    config = mlp.config
    hidden_states = apply_logical_sharding(
        hidden_states,
        dynamic_axes=common_types.HiddenStateSharding,
        partition_manager=config.runtime_sharding_resolver,
    )

    out = None
    # ``use_fused_mlp`` is a declared EasyDeLBaseConfig field (default True);
    # setting it False forces the legacy unfused composition. The repo-wide
    # ``FORCE_NATIVE_RUNTIME`` convention (same flag the FusedMlpOp adapter
    # honors) also forces the unfused native composition so the standard
    # numerics bisection covers this route.
    fused_route_allowed = (
        config.use_fused_mlp
        and getattr(mlp, "gate_up_proj", None) is not None
        and hidden_states.ndim >= 2
        and not check_bool_flag("FORCE_NATIVE_RUNTIME", False)
    )
    inference = is_inference_mode()
    if fused_route_allowed and not inference and _uses_name_based_remat(config):
        # The fused route's custom_vjp pins two bf16 [m, intermediate]
        # residuals and drops the checkpoint_name tags, so name-based remat
        # policies (MLP_NOTSAVEABLE family, SAVE_ONLY/EXCEPT_THESE_NAMES)
        # would silently lose effect during training. Inference has no
        # backward pass and keeps the fused route.
        fused_route_allowed = False
    if fused_route_allowed:
        lead = hidden_states.shape[:-1]
        x2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        specs = _resolved_specs(mlp, config, hidden_states)
        out2d = _dense_fused_call(mlp, config, x2d, specs)
        if out2d is None and inference:
            # QUANT fused route is inference-only: its frozen-weight
            # custom_vjp gives hard-zero gradients to trainable quant_scales
            # leaves, and there is no per-leaf requires-grad signal at trace
            # time to detect whether scales are being trained (the STE
            # trainer decides the differentiation set outside this module).
            # bf16 fused stays available for training above.
            out2d = _quantized_fused_call(mlp, config, x2d)
        if out2d is not None:
            out = checkpoint_name(out2d.reshape(*lead, out2d.shape[-1]), "mlp_down")

    if out is None:
        out = _legacy_composition(mlp, config, hidden_states)

    if output_scale is not None:
        out = out * output_scale
    if dropout is not None:
        out = dropout(out)
    out = apply_logical_sharding(
        out,
        dynamic_axes=common_types.HiddenStateSharding,
        partition_manager=config.runtime_sharding_resolver,
    )
    return checkpoint_name(out, "mlp_output")
