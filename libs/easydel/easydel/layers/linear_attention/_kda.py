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

"""Kernel Delta Attention (KDA): the linear-attention layer shared by every KDA model family.

:class:`KernelDeltaAttention` is the single implementation of the KDA block. Model families subclass it for their
forget-gate form and Hugging Face checkpoint layout: ``KimiDeltaAttention`` (Kimi-Linear, unbounded softplus gate) and
``Glm5NextLinearAttention`` (GLM-5-Next, safe sigmoid gate bounded below by ``config.linear_lower_bound``).

Shapes come from ``config.linear_attn_config`` through :func:`kda_layer_spec`, the same reader the KDA cache uses, so
a layer and its cache cannot disagree. The runtime choices are
:class:`~easydel.infra.base_config.EasyDeLBaseConfig` fields that every KDA model honours: ``kda_chunk_impl`` /
``kda_chunk_precision`` (the chunked kernel, see :class:`~easydel.operations.kernels.KernelDeltaAttnOp`),
``gather_shared_projection_input`` (one tensor-parallel gather for the six input projections) and
``kda_serving_ep_split`` (eSurge serving: heads split over the head and expert-parallel axes, see
:func:`kda_serving_head_axes`).
"""

import functools
import math
import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float
from spectrax import apply_logical_sharding, common_types, nn, with_sharding_constraint

from easydel.axis import resolve_attention_data_parallel_axis
from easydel.caching import KDACacheView, KDAMetadata
from easydel.infra.modeling_outputs import AttentionLayerOutput
from easydel.infra.sequence_packing import packed_segment_ids_from_mask_info
from easydel.infra.sharding import inference_mode_forces_decode, mesh_axis_size
from easydel.infra.utils import ArrayParam
from easydel.layers.linears import (
    ColumnParallelLinear,
    RowParallelLinear,
    gather_shared_input,
    resolve_gather_shared_projection_input,
)
from easydel.layers.norms import RMSNormGated, lowfloats, promote_dtype
from easydel.operations import OperationMetadata
from easydel.operations.kernels import (
    KDAOutput,
    KernelDeltaAttnOp,
    fused_kda_gate_per_channel,
    kda_packed_rows_forward,
    kda_ragged_rows_forward,
    kda_rows_forward,
)
from easydel.operations.kernels.ragged_kda_rank_major import kda_ragged_rows_rank_major_forward

from ._conv_state import apply_conv_with_state
from ._masking import apply_mask_to_padding_states

logger = get_logger(__name__)

# FLA initialisation of ``dt_bias`` (HF ``_init_weights``): dt ~ U(1e-3, 1e-1) stored as its inverse softplus. Parameter
# initialisers are plain JAX factories, so a constant is used instead: the mean of the two bounds' inverse softplus;
# pretrained loads overwrite it.
_DT_BIAS_INIT = 0.5 * (math.log(math.expm1(1e-3)) + math.log(math.expm1(1e-1)))


class KDALayerSpec(tp.NamedTuple):
    """Shapes of a KDA layer (and of its cache), read from ``config.linear_attn_config`` by :func:`kda_layer_spec`.

    Attributes:
        num_heads: Linear-attention heads.
        head_k_dim: Query / key width per head (also the per-channel decay width).
        head_v_dim: Value width per head.
        d_conv: Width of the causal depthwise q/k/v convolutions.
        gate_low_rank_dim: Bottleneck of the low-rank forget-gate and output-gate MLPs.
        chunk_size: Chunk length of the chunked delta-rule kernels.
    """

    num_heads: int
    head_k_dim: int
    head_v_dim: int
    d_conv: int
    gate_low_rank_dim: int
    chunk_size: int

    @property
    def key_dim(self) -> int:
        """Total query / key width ``num_heads * head_k_dim``."""
        return self.num_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        """Total value width ``num_heads * head_v_dim``."""
        return self.num_heads * self.head_v_dim


def kda_layer_spec(config: tp.Any) -> KDALayerSpec:
    """Read the KDA layer shapes from ``config.linear_attn_config``.

    Accepts Moonshot's keys (``head_dim``, ``short_conv_kernel_size``) and the fla-style overrides (``head_k_dim``,
    ``head_v_dim``, ``d_conv``, ``gate_low_rank_dim``, ``chunk_size``). Without ``gate_low_rank_dim`` the gate
    bottleneck is ``head_dim`` when given, else ``head_v_dim`` (fla's ``f_proj`` / ``g_proj`` width).

    Args:
        config: Model (text) config with an optional ``linear_attn_config`` dict.

    Returns:
        The :class:`KDALayerSpec`.

    Raises:
        ValueError: If no positive head count is available (``linear_attn_config['num_heads']`` or
            ``num_attention_heads``).
    """
    linear = getattr(config, "linear_attn_config", None) or {}
    num_heads = int(linear.get("num_heads", getattr(config, "num_attention_heads", 0) or 0))
    if num_heads <= 0:
        raise ValueError(
            "Could not infer the KDA head count; expected `linear_attn_config['num_heads']` or `num_attention_heads`."
        )
    head_dim = int(linear.get("head_dim", 128))
    head_v_dim = int(linear.get("head_v_dim", head_dim))
    return KDALayerSpec(
        num_heads=num_heads,
        head_k_dim=int(linear.get("head_k_dim", head_dim)),
        head_v_dim=head_v_dim,
        d_conv=int(linear.get("d_conv", linear.get("short_conv_kernel_size", getattr(config, "d_conv", 4)))),
        gate_low_rank_dim=int(linear.get("gate_low_rank_dim", linear.get("head_dim", head_v_dim))),
        chunk_size=int(linear.get("chunk_size", 64)),
    )


def _axis_names(entry: tp.Any) -> tuple[str, ...]:
    """Mesh axis names of a partition-spec entry (a name, a tuple of names or ``None``)."""
    if entry is None:
        return ()
    if isinstance(entry, str):
        return (entry,)
    return tuple(name for name in entry if isinstance(name, str))


def _active_mesh_axes(mesh: tp.Any, entry: tp.Any) -> tuple[str, ...] | None:
    """Mesh axes of more than one device in a partition-spec entry; ``None`` for an entry that names no layout."""
    if entry is not None and not isinstance(entry, str | tuple | list):
        return None
    return tuple(name for name in _axis_names(entry) if mesh_axis_size(mesh, name) > 1)


def kda_serving_head_axes(config: tp.Any, *, serving: bool = False) -> tuple[str, ...] | None:
    """Mesh axes an eSurge KDA cache splits the heads over, else ``None`` (the resolver of ``kda_serving_ep_split``).

    KDA heads normally split over the head axis (tensor parallelism) and are replicated over the expert-parallel axis,
    so every expert-parallel group runs the same heads. The split puts them over the head axis and the expert-parallel
    axis together, head axis major: each device's heads are then a subset of the heads it holds without the split, so
    every column-parallel weight reaches the split layout by a local slice (no weight collective; parameters and
    checkpoints keep their layout).

    ``kda_serving_ep_split`` is tri-state: ``None`` (automatic) and ``True`` split every eSurge cache (``serving``)
    whose mesh allows it, ``True`` warning once when the heads do not divide; ``False`` never splits. Caches created
    outside eSurge (``serving=False``: ``init_cache``, training) never split. The mesh allows it when the
    expert-parallel axis has more than one device and is not part of the head axis, the mesh is not MPMD, eSurge's
    attention data-parallel axis has a single device (rank-major DP keeps today's layout) and the split divides the
    head count.

    Args:
        config: Model (text) config with ``kda_serving_ep_split``, ``partition_axis``, ``mesh`` and the KDA shapes.
        serving: Whether the cache is an eSurge serving cache.

    Returns:
        ``(*head_axis, expert_parallel_axis)`` as mesh axis names, or ``None``.
    """
    setting = getattr(config, "kda_serving_ep_split", None)
    if not serving or setting is False:
        return None
    paxis = getattr(config, "partition_axis", None)
    if paxis is None:
        return None
    ep_axis = paxis.expert_parallel_axis
    base = _axis_names(paxis.head_axis)
    if not base or ep_axis in base:
        return None
    mesh = getattr(config, "mesh", None)
    if mesh is None or getattr(mesh, "is_mpmd", False) or mesh_axis_size(mesh, ep_axis) <= 1:
        return None
    if mesh_axis_size(mesh, resolve_attention_data_parallel_axis(paxis)) > 1:
        # eSurge rank-major SPMD data parallelism keeps today's head layout.
        return None
    axes = (*base, ep_axis)
    num_heads = kda_layer_spec(config).num_heads
    if num_heads % mesh_axis_size(mesh, axes):
        if setting is True:
            logger.warn_once(
                f"kda_serving_ep_split: {num_heads} KDA heads do not split over {axes} "
                f"({mesh_axis_size(mesh, axes)} devices); keeping the head axis {base}."
            )
        return None
    return axes


class _ServingHeadSplit(tp.NamedTuple):
    """Head split of one eSurge KDA step (``kda_serving_ep_split``).

    Attributes:
        axes: Mesh axes of the heads (and of the q / k / v / gate features).
        lead: Partition-spec entries of the two leading activation dims (``(None, None)`` for a packed ``[1, T, ...]``
            stream, ``(attention dp axis, None)`` for a rank-major ``[dp, T / dp, ...]`` batch).
    """

    axes: tuple[str, ...]
    lead: tuple[tp.Any, ...]


def unnest_legacy_kda_forget_gate(state: dict, expected_leaves: dict) -> dict:
    """Move KDA forget-gate leaves saved under a ``forget_gate`` sub-module to the layer's own names.

    Some GLM-5-Next native checkpoints nest ``A_log``, ``dt_bias``, ``f_a_proj`` and ``f_b_proj`` under
    ``self_attn.forget_gate``. The values are unchanged; only the path loses that segment. Meant as a model's
    ``_upgrade_legacy_native_state`` hook.

    Args:
        state: Flat native state ``{(collection, *path): array}``; updated in place.
        expected_leaves: The model's leaves under the same keys.

    Returns:
        ``state``.
    """
    moved = 0
    for key in list(state):
        if key in expected_leaves or "forget_gate" not in key:
            continue
        flat = tuple(part for part in key if part != "forget_gate")
        if flat in expected_leaves and flat not in state:
            state[flat] = state.pop(key)
            moved += 1
    if moved:
        logger.info(f"Moved {moved} KDA forget-gate leaves out of the legacy `forget_gate` sub-module.")
    return state


class _KDAConv1d(nn.Conv1d):
    """Depthwise conv whose weight keeps its storage dtype (``param_dtype``) and is cast to the input at use."""

    def forward(self, x, **kwargs):
        """Convolve in the incoming projection dtype without changing the stored parameters."""
        return spx.functional.conv(
            x,
            self.weight.value.astype(x.dtype),
            self.bias.value.astype(x.dtype) if self.use_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            precision=self.precision,
        )


class KernelDeltaAttention(spx.Module):
    """Kernel Delta Attention (KDA) block, the linear-attention layer of Kimi-Linear and GLM-5-Next.

    KDA replaces softmax attention with a recurrent key/value memory updated by the **delta rule**, giving O(seq_len)
    compute and O(1) decode. Each head keeps :math:`S \\in \\mathbb{R}^{d_k \\times d_v}` and

    .. math::
        S_t = (I - \\beta_t k_t k_t^\\top) \\, \\operatorname{diag}(\\alpha_t) \\, S_{t-1}
              + \\beta_t \\, k_t v_t^\\top, \\qquad o_t = S_t^\\top q_t

    with a per-channel (one rate per head and key channel) data-dependent decay
    :math:`\\log \\alpha_t = \\mathrm{gate}(f_b(f_a(x_t)) + \\mathrm{dt\\_bias})` and a per-head write strength
    :math:`\\beta_t = \\sigma(W_b x_t)`. The gate is either the safe form
    ``lower_bound * sigmoid(exp(A_log) * z)`` (values in ``[lower_bound, 0)``) or the softplus form
    ``-exp(A_log) * softplus(z)`` (``gate_lower_bound=None``); see
    :func:`~easydel.operations.kernels.fused_kda_gate_per_channel`. Q/K/V first pass through separate causal
    depthwise convolutions of width ``d_conv``; the output goes through a low-rank sigmoid gate
    (``g_a_proj -> g_b_proj``) inside a gated RMSNorm before ``o_proj``. The recurrence runs in
    :class:`~easydel.operations.kernels.KernelDeltaAttnOp` (chunked for multi-token input, one recurrent step on
    decode); eSurge packed streams go through :func:`~easydel.operations.kernels.kda_ragged_rows_forward` (the
    row-padded :func:`~easydel.operations.kernels.kda_packed_rows_forward` when the pool holds speculative candidate
    rows).

    State carried in :class:`~easydel.caching.KDACacheView`: the three rolling conv windows and ``S``
    ``(num_heads, head_k_dim, head_v_dim)``.

    A model family subclasses this layer to fix ``gate_lower_bound`` and to declare its checkpoint ``reform_param``;
    everything else is shared.

    Attributes:
        q_proj, k_proj, v_proj: Projections to Q/K/V before the convolutions.
        q_conv1d, k_conv1d, v_conv1d: Causal depthwise convolutions (weights stored in ``param_dtype``).
        f_a_proj, f_b_proj: Low-rank forget-gate MLP (``key_dim`` outputs).
        A_log: Per-head log decay rate ``(num_heads,)``; created in float32 (Hugging Face keeps the gate in fp32).
            Checkpoint loaders store it in ``param_dtype`` unless the model pins it through ``fp32_param_patterns``;
            the gate computes in float32 either way.
        dt_bias: Per-channel gate bias ``(key_dim,)``; same storage rule as ``A_log``.
        b_proj: Per-head write-strength projection (sigmoid-activated).
        g_a_proj, g_b_proj: Low-rank output-gate MLP.
        o_norm: Sigmoid-gated RMSNorm on the head outputs.
        o_proj: Output projection back to ``hidden_size``.
        kda_op: The KDA operation.
        gate_lower_bound: Safe-gate lower bound, or ``None`` for the softplus gate.
        num_heads, head_k_dim, head_v_dim, d_conv, gate_low_rank_dim, chunk_size, key_dim, value_dim: Layer shapes
            (:class:`KDALayerSpec`).
    """

    # ``layer_idx`` is init/informational only, so identical KDA layers share one lowered scan body.
    _spx_scan_safe_static_fields: tp.ClassVar[frozenset[str]] = frozenset({"layer_idx"})

    # eSurge (runners/slot_contract.py): row r of every serving step reads and writes pool slot
    # cache_metadata.recurrent_state_indices[r], so requests keep stable slots across row moves and row windows.
    esurge_recurrent_slot_contract: tp.ClassVar[str] = "slots"

    def __init__(
        self,
        config: tp.Any,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        gate_lower_bound: float | None,
        rngs: spx.Rngs,
    ):
        """Build the KDA block.

        Args:
            config: Model config (``linear_attn_config`` shapes, ``hidden_size``, ``rms_norm_eps``,
                ``initializer_range`` and the base-config KDA settings).
            layer_idx: Index of this layer in the decoder stack.
            dtype: Computation dtype.
            param_dtype: Parameter storage dtype (``A_log`` / ``dt_bias`` are float32).
            precision: JAX matmul precision.
            gate_lower_bound: Safe-gate lower bound (negative), or ``None`` for the softplus gate. Required: the two
                gates are different models.
            rngs: Random number generator collection.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx
        self.gate_lower_bound = gate_lower_bound

        spec = kda_layer_spec(config)
        self.num_heads = spec.num_heads
        self.head_k_dim = spec.head_k_dim
        self.head_v_dim = spec.head_v_dim
        self.d_conv = spec.d_conv
        self.gate_low_rank_dim = spec.gate_low_rank_dim
        self.chunk_size = spec.chunk_size
        self.key_dim = spec.key_dim
        self.value_dim = spec.value_dim

        column_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        depthwise_conv = functools.partial(
            _KDAConv1d,
            kernel_size=self.d_conv,
            padding=((self.d_conv - 1, 0),),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            use_bias=False,
            precision=precision,
        )

        self.q_proj = column_linear(config.hidden_size, self.key_dim)
        self.k_proj = column_linear(config.hidden_size, self.key_dim)
        self.v_proj = column_linear(config.hidden_size, self.value_dim)
        self.q_conv1d = depthwise_conv(in_channels=self.key_dim, out_channels=self.key_dim, groups=self.key_dim)
        self.k_conv1d = depthwise_conv(in_channels=self.key_dim, out_channels=self.key_dim, groups=self.key_dim)
        self.v_conv1d = depthwise_conv(in_channels=self.value_dim, out_channels=self.value_dim, groups=self.value_dim)

        self.f_a_proj = column_linear(config.hidden_size, self.gate_low_rank_dim)
        self.f_b_proj = column_linear(self.gate_low_rank_dim, self.key_dim)
        # FLA initialisation (A_log = 0, dt_bias = _DT_BIAS_INIT), in float32 like HF's strict-fp32 gate.
        self.A_log = ArrayParam.bound(shape=(self.num_heads,), dtype=jnp.float32, init_method="zeros", key=rngs.param)
        self.dt_bias = ArrayParam.bound(
            shape=(self.key_dim,),
            dtype=jnp.float32,
            init_method="constant",
            init_kwargs={"value": _DT_BIAS_INIT},
            key=rngs.param,
        )
        self.b_proj = column_linear(config.hidden_size, self.num_heads)
        self.g_a_proj = column_linear(config.hidden_size, self.gate_low_rank_dim)
        self.g_b_proj = column_linear(self.gate_low_rank_dim, self.value_dim)

        self.o_proj = RowParallelLinear(
            self.value_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.o_norm = RMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            activation="sigmoid",
            rngs=rngs,
        )
        self.kda_op = KernelDeltaAttnOp(
            OperationMetadata(runtime_dtype=self.dtype, runtime_softmax_dtype=jnp.float32, base_config=self.config)
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        cache_view: KDACacheView | None = None,
        cache_metadata: KDAMetadata | None = None,
    ) -> AttentionLayerOutput:
        """Run the per-channel delta rule over a block of tokens.

        Padding tokens are zeroed before the projections, so they feed neither the conv windows nor the memory.
        Packed documents (segment ids in ``mask_info``) reset both the convolutions and the recurrence at every
        boundary. With a cache view the conv windows and memory continue from the cache (one recurrent step for a
        single token); an eSurge packed stream (``cache_metadata.query_start_loc``) advances one cache row per
        request (:meth:`_forward_packed_rows`).

        Args:
            hidden_states: Layer input ``(batch, seq_len, hidden_size)``.
            mask_info: Optional :class:`MaskInfo`; only the padding mask and packed segment ids are used (the
                recurrence is causal by construction).
            cache_view: Per-layer KDA cache (conv windows + recurrent memory), or ``None``.
            cache_metadata: Optional :class:`KDAMetadata`; its ``query_start_loc`` selects the packed-row path.

        Returns:
            :class:`AttentionLayerOutput` with the ``(batch, seq_len, hidden_size)`` output,
            ``attention_weight=None`` and the updated cache view.
        """
        if mask_info is not None:
            q_mask: Array | None = tp.cast("Array | None", mask_info.q_attention_mask)
            if q_mask is not None and q_mask.shape[1] != hidden_states.shape[1]:
                q_mask = q_mask[:, : hidden_states.shape[1]]
            hidden_states = apply_mask_to_padding_states(hidden_states, q_mask)
        segment_ids = packed_segment_ids_from_mask_info(mask_info, hidden_states.shape[1])

        batch_size, seq_len, _ = hidden_states.shape
        is_inference = seq_len == 1 and cache_view is not None

        query_start_loc = getattr(cache_metadata, "query_start_loc", None) if cache_metadata is not None else None
        serving = cache_view is not None and query_start_loc is not None
        rank_major = serving and getattr(query_start_loc, "ndim", 1) == 2
        # eSurge KDA cache views created under ``kda_serving_ep_split`` split the heads over (head axis, ep axis).
        split = self._serving_head_split(cache_view, rank_major) if serving else None
        query, key, value, beta, decay, gate_low_rank = self._input_projections(hidden_states, split=split)

        if rank_major:
            # eSurge rank-major SPMD data parallelism (data_parallel_size > 1): [dp, T / dp, D] with per-rank rows.
            return self._forward_rank_major_rows(
                hidden_states,
                query,
                key,
                value,
                beta,
                decay,
                cache_view,
                query_start_loc,
                gate_low_rank,
                cache_metadata,
                split=split,
            )
        if serving:
            if batch_size != 1:
                raise ValueError(
                    f"Packed-row KDA serving expects one flattened token stream [1, T, D], got batch_size={batch_size}."
                )
            return self._forward_packed_rows(
                hidden_states=hidden_states,
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                cache_view=cache_view,
                query_start_loc=query_start_loc,
                gate_low_rank=gate_low_rank,
                cache_metadata=cache_metadata,
                split=split,
            )

        conv_output_dtype = jnp.bfloat16 if self.dtype in lowfloats else self.dtype
        conv_states = {}
        convolved = {}
        for name, stream in (("q", query), ("k", key), ("v", value)):
            convolved[name], conv_states[name] = apply_conv_with_state(
                stream,
                getattr(self, f"{name}_conv1d"),
                getattr(cache_view, f"{name}_conv_state") if cache_view is not None else None,
                is_inference=is_inference,
                d_conv=self.d_conv,
                output_dtype=conv_output_dtype,
                reuse_partial_state=True,
                segment_ids=segment_ids,
            )

        resolver = self.config.runtime_sharding_resolver
        query = apply_logical_sharding(
            convolved["q"].reshape(batch_size, seq_len, self.num_heads, self.head_k_dim),
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=resolver,
        )
        key = apply_logical_sharding(
            convolved["k"].reshape(batch_size, seq_len, self.num_heads, self.head_k_dim),
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=resolver,
        )
        value = apply_logical_sharding(
            convolved["v"].reshape(batch_size, seq_len, self.num_heads, self.head_v_dim),
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=resolver,
        )

        output_gate = self.g_b_proj(gate_low_rank).reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)

        kda_output: KDAOutput = self.kda_op(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            q_conv_state=conv_states["q"],
            k_conv_state=conv_states["k"],
            v_conv_state=conv_states["v"],
            recurrent_state=cache_view.recurrent_state if cache_view is not None else None,
            chunk_size=self.chunk_size,
            per_channel_decay=True,
            segment_ids=segment_ids,
            # A dense cache carries memory from previous chunks: keep the first live segment connected while packed
            # training still resets at every boundary.
            continue_initial_state=cache_view is not None,
            chunk_impl=self.config.kda_chunk_impl,
            chunk_precision=self.config.kda_chunk_precision,
            # The safe gate keeps every log-decay in [lower_bound, 0), so the bound is a valid ``decay_lower_bound``
            # promise; the softplus gate passes None.
            decay_lower_bound=self.gate_lower_bound,
        )

        output = self.o_norm(kda_output.attention_outputs, output_gate)
        output = apply_logical_sharding(
            output.reshape(batch_size, seq_len, -1),
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=resolver,
        )
        output = apply_logical_sharding(
            checkpoint_name(self.o_proj(output), name="attn_output"),
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=resolver,
        )

        new_cache_view = cache_view
        if cache_view is not None:
            new_cache_view = cache_view.update_kda_states(
                new_q_conv_state=kda_output.q_conv_state,
                new_k_conv_state=kda_output.k_conv_state,
                new_v_conv_state=kda_output.v_conv_state,
                new_recurrent_state=kda_output.recurrent_state,
            )
        return AttentionLayerOutput(attention_output=output, attention_weight=None, cache_view=new_cache_view)

    def _forward_rank_major_rows(
        self,
        hidden_states: Float[Array, "dp tokens_per_rank hidden_dim"],
        query: Float[Array, "dp tokens_per_rank key_dim"],
        key: Float[Array, "dp tokens_per_rank key_dim"],
        value: Float[Array, "dp tokens_per_rank value_dim"],
        beta: Float[Array, "dp tokens_per_rank num_heads"],
        decay: Float[Array, "dp tokens_per_rank num_heads head_k_dim"],
        cache_view: KDACacheView,
        query_start_loc: Array,
        gate_low_rank: Float[Array, "dp tokens_per_rank gate_low_rank_dim"],
        cache_metadata: KDAMetadata | None,
        split: _ServingHeadSplit | None = None,
    ) -> AttentionLayerOutput:
        """Advance per-request KDA state over an eSurge rank-major SPMD DP batch ``[dp, T / dp, D]``.

        eSurge serves with ``data_parallel_size > 1`` by giving every DP rank its own token block, rows
        (``query_start_loc`` ``[dp, rows + 1]``) and block of pool slots (rank-local
        ``cache_metadata.recurrent_state_indices`` ``[dp, rows]``). The conv and the delta rule run in
        :func:`~easydel.operations.kernels.kda_ragged_rows_rank_major_forward` (the flat driver per rank under one
        ``shard_map``); this method adds the output gate, norm and projection, as :meth:`_forward_packed_rows` does for
        a single rank. Only eSurge rank-major steps reach it; training and single-rank serving never do.

        Args:
            hidden_states: Rank-major layer input ``[dp, T / dp, D]``.
            query: Q projection ``[dp, T / dp, key_dim]``.
            key: K projection ``[dp, T / dp, key_dim]``.
            value: V projection ``[dp, T / dp, value_dim]``.
            beta: Write strength ``[dp, T / dp, heads]`` (sigmoid applied).
            decay: Per-channel log-decay ``[dp, T / dp, heads, head_k_dim]``.
            cache_view: Per-layer KDA cache view (pool slot axis split into ``dp`` rank blocks).
            query_start_loc: Rank-local cumulative row offsets ``[dp, rows + 1]``.
            gate_low_rank: ``g_a_proj(hidden_states)`` from :meth:`_input_projections`.
            cache_metadata: KDA metadata (context lengths, rank-local recurrent slots).
            split: Head split of the step (``kda_serving_ep_split`` cache views), or ``None``.

        Returns:
            :class:`AttentionLayerOutput` with the rank-major output and the updated cache view.

        Raises:
            ValueError: For a speculative step (``spec_recurrent_commit`` set), which rank-major DP does not serve.
        """
        if getattr(cache_metadata, "spec_recurrent_commit", None) is not None:
            raise ValueError("Rank-major SPMD DP does not serve speculative KDA steps (spec_recurrent_commit is set).")
        out, new_cache_view = kda_ragged_rows_rank_major_forward(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            conv_kernels=tuple(getattr(self, f"{n}_conv1d").weight.value.squeeze(1).T for n in ("q", "k", "v")),
            cache_view=cache_view,
            query_start_loc=query_start_loc,
            context_lens=getattr(cache_metadata, "context_lens", None),
            metadata=self.kda_op.get_impl_metadata(),
            conv_dtype=jnp.bfloat16 if self.dtype in lowfloats else self.dtype,
            chunk_size=self.chunk_size,
            decay_lower_bound=self.gate_lower_bound,
            precision=self.config.kda_chunk_precision if self.config.kda_chunk_impl == "kda_chunk" else "highest",
            state_indices=getattr(cache_metadata, "recurrent_state_indices", None),
            flat_forward=kda_ragged_rows_forward,
            head_axis=None if split is None else split.axes,
        )
        output = self._gated_output_projection(out, gate_low_rank, hidden_states.dtype, split)
        return AttentionLayerOutput(attention_output=output, attention_weight=None, cache_view=new_cache_view)

    def _serving_head_split(self, cache_view: tp.Any, rank_major: bool) -> _ServingHeadSplit | None:
        """Head split of an eSurge step, from the cache view's ``head_shard_axes`` (``kda_serving_ep_split``).

        The pools of a KDA cache view created under ``kda_serving_ep_split`` (:func:`kda_serving_head_axes`) split
        their heads over ``metadata.head_shard_axes``; the step then splits its heads the same way, so the conv windows
        and recurrent states never move between devices. ``None`` (today's head-axis layout) for any other view, when
        the axes do not divide the heads on this mesh, or when they share an axis with a rank-major step's
        data-parallel axis.

        Args:
            cache_view: The layer's cache view.
            rank_major: Whether the step is an eSurge rank-major SPMD DP step (``[dp, T / dp, D]``).

        Returns:
            The :class:`_ServingHeadSplit`, or ``None``.
        """
        axes = getattr(getattr(cache_view, "metadata", None), "head_shard_axes", None)
        if not axes:
            return None
        axes = tuple(axes)
        mesh = self.config.mesh
        shards = mesh_axis_size(mesh, axes)
        if shards <= 1 or self.num_heads % shards:
            return None
        if not rank_major:
            return _ServingHeadSplit(axes, (None, None))
        dp_axis = resolve_attention_data_parallel_axis(self.config.runtime_sharding_resolver)
        if set(_axis_names(dp_axis)) & set(axes):
            return None
        return _ServingHeadSplit(axes, (dp_axis, None))

    def _constrain_heads(self, x: Array, split: _ServingHeadSplit, trailing: int = 0) -> Array:
        """Constrain ``x`` ``[lead0, lead1, features or heads, *trailing]`` to the split's head axes."""
        spec = PartitionSpec(*split.lead, split.axes, *((None,) * trailing))
        return with_sharding_constraint(x, spec, mesh=self.config.mesh)

    def _gated_output_projection(
        self,
        out: Array,
        gate_low_rank: Array,
        hidden_dtype: jnp.dtype,
        split: _ServingHeadSplit | None,
    ) -> Array:
        """Output gate, gated RMSNorm and ``o_proj`` of a serving step's KDA output ``[B, T, heads, head_v_dim]``.

        With a head split the gate and the normalized heads stay split over the head axes, and ``o_proj``'s partial
        sums over every device's heads are reduced straight into the residual stream's layout.

        Args:
            out: Driver output ``[B, T, heads, head_v_dim]``.
            gate_low_rank: ``g_a_proj(hidden_states)`` ``[B, T, gate_low_rank_dim]``.
            hidden_dtype: Dtype of the layer input (the driver output is cast to it before the norm).
            split: Head split of the step, or ``None``.

        Returns:
            The layer output ``[B, T, hidden_size]``.
        """
        lead0, lead1 = int(out.shape[0]), int(out.shape[1])
        if split is None:
            gate = self.g_b_proj(gate_low_rank)
        else:
            gate = self._constrain_heads(self._split_projection(self.g_b_proj, gate_low_rank, split, dim=1), split)
        output_gate = gate.reshape(lead0, lead1, self.num_heads, self.head_v_dim)
        normed = self.o_norm(out.astype(hidden_dtype), output_gate).reshape(lead0, lead1, -1)
        if split is None:
            return checkpoint_name(self.o_proj(normed), name="attn_output")
        projected = self._split_output_projection(self._constrain_heads(normed, split), split)
        return checkpoint_name(projected, name="attn_output")

    def _split_output_projection(self, normed: Array, split: _ServingHeadSplit) -> Array:
        """``o_proj`` of a head-split serving step: heads traded for tokens when the residual is feature-sharded.

        Every device holds the gated-norm features of its own heads (``split.axes``, head axis major). When the
        residual stream's hidden axis (``HiddenStateSharding``, resolved mode-aware) is split over the leading head
        axes ``H`` and exactly one head axis ``R`` remains (prefill and mixed steps of a ``kda_serving_ep_split`` cache:
        ``H`` = tensor parallelism, ``R`` = expert parallelism), one ``shard_map`` runs: an ``all_to_all`` over ``R``
        trades tokens for heads, so each device holds its ``H`` group's heads (in the order of its ``H`` shard of the
        ``o_proj`` rows) for ``1 / |R|`` of the tokens; the dot contracts that shard, a reduce-scatter over ``H`` leaves
        the residual's hidden slice and an all-gather over ``R`` restores the tokens. Per token these are the dot and
        the reduce-scatter of the unsplit layer. The path it replaces (the SPMD partitioner, or the row-parallel
        ``distributed_matmul`` engine, whose ``shard_map`` takes the input split over ``H`` only) all-gathers the heads
        over ``R`` first and then runs the dot and the reduce-scatter for every token on every device. Calculated per
        device and layer at ep4 x tp4 (4 heads of 128 per chip, hidden 4096, 32,768 bf16 tokens): 25 + 50 + 50 MB of
        collectives and a quarter of the dot, against 100 + 201 MB and the whole dot.

        eSurge's decode buckets (``decode_mode_specs``) whose residual is replicated take
        :meth:`_replicated_output_projection` instead. Other steps keep :meth:`_split_projection`: token counts that
        ``|R|`` does not divide and residual layouts outside the head axes; so do projections that are not a plain
        bias-free :class:`RowParallelLinear` (quantized, LoRA, a quantization rule) or whose declared layout does not
        resolve.

        Args:
            normed: Gated-norm output ``[lead0, lead1, value_dim]``, features split over ``split.axes``.
            split: Head split of the step.

        Returns:
            The projection ``[lead0, lead1, hidden_size]`` in the residual stream's layout.
        """
        linear = self.o_proj
        weight = getattr(getattr(linear, "weight", None), "value", None)
        mesh = self.kda_op.get_impl_metadata().mesh
        if (
            type(linear) is not RowParallelLinear
            or linear.use_bias
            or not isinstance(weight, jax.Array)
            or weight.ndim != 2
            or spx.quantization.rule_for(linear, "dot_general") is not None
            or mesh is None
        ):
            return self._split_projection(linear, normed, split, dim=0)
        tokens, hidden_size = int(normed.shape[1]), int(weight.shape[1])
        resolver = self.config.runtime_sharding_resolver
        active = functools.partial(_active_mesh_axes, mesh)
        heads = active(split.axes) or ()
        residual = tuple(
            resolver.resolve(
                dynamic_axes=common_types.HiddenStateSharding,
                shape=(int(normed.shape[0]), tokens, hidden_size),
            )
        )
        try:
            # The weight's declared layout: its rows must split over the residual's hidden axes only.
            kernel = tuple(
                resolver.resolve_layout(
                    linear.sharding_layout if linear.sharding_layout is not None else common_types.RowWise,
                    shape=tuple(int(dim) for dim in weight.shape),
                    mesh=resolver.mesh,
                )
            )
        except Exception:
            # No resolvable layout: the shard_maps below cannot name the weight's sharding.
            return self._split_projection(linear, normed, split, dim=0)
        # A resolved spec may omit trailing replicated entries: pad both to their array's rank.
        residual = residual + (None,) * (normed.ndim - len(residual))
        kernel = kernel + (None,) * (2 - len(kernel))
        out_entries = [active(entry) for entry in residual] if len(residual) == normed.ndim else [None]
        kernel_entries = [active(entry) for entry in kernel] if len(kernel) == 2 else [None]
        if (
            inference_mode_forces_decode()
            and all(entry is not None for entry in (*out_entries, *kernel_entries))
            and not any(out_entries)
            and heads
            and heads[: len(kernel_entries[0])] == kernel_entries[0]
            and not kernel_entries[1]
            and int(normed.shape[-1]) % mesh_axis_size(mesh, heads) == 0
        ):
            # eSurge decode bucket with a replicated residual: one reduction over every split axis.
            return self._replicated_output_projection(normed, weight, split, mesh, kernel_entries[0])
        out_axes = out_entries[-1] or ()
        rest = heads[len(out_axes) :]
        if (
            any(entry is None for entry in (*out_entries, *kernel_entries))
            or not out_axes
            or heads[: len(out_axes)] != out_axes
            or len(rest) != 1
            or any(out_entries[:-1])
            or kernel_entries[0] != out_axes
            or kernel_entries[1]
            or tokens % mesh_axis_size(mesh, rest[0])
            or hidden_size % mesh_axis_size(mesh, out_axes)
        ):
            return self._split_projection(linear, normed, split, dim=0)
        out_entry = out_axes[0] if len(out_axes) == 1 else out_axes
        token_axis = rest[0]
        sources = mesh_axis_size(mesh, token_axis)
        scale = float(getattr(linear, "output_scale", 1.0))

        def body(x: Array, w: Array) -> Array:
            x, w = promote_dtype((x, w), dtype=linear.dtype)
            lead0, local_tokens, features = x.shape
            # [lead0, T, own heads] -> [lead0, |R| sources, T / |R|, the source's heads] for this device's token block.
            # The sources stay an axis of their own, in R order (the order of this device's H shard of the o_proj
            # rows), so no relayout moves the received heads into the feature axis: the dot contracts (source,
            # feature) against the shard viewed as [|R|, features, hidden].
            x = x.reshape(lead0, sources, local_tokens // sources, features)
            x = jax.lax.all_to_all(x, token_axis, split_axis=1, concat_axis=1, tiled=True)
            w = w.reshape(sources, features, w.shape[-1])
            y = jnp.einsum("bstf,sfo->bto", x, w, precision=linear.precision)
            y = jax.lax.psum_scatter(y, out_entry, scatter_dimension=2, tiled=True)
            return jax.lax.all_gather(y, token_axis, axis=1, tiled=True)

        y = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(PartitionSpec(*split.lead, split.axes), PartitionSpec(out_entry, None)),
            out_specs=PartitionSpec(*split.lead, out_entry),
            check_vma=False,
        )(normed, weight)
        return y * scale if scale != 1.0 else y

    def _replicated_output_projection(
        self,
        normed: Array,
        weight: Array,
        split: _ServingHeadSplit,
        mesh: tp.Any,
        row_axes: tuple[str, ...],
    ) -> Array:
        """``o_proj`` of a head-split eSurge decode bucket (``decode_mode_specs``), whose residual is replicated.

        Every device contracts its own heads' features with its own rows of ``o_proj`` (a local slice of the rows it
        stores: the split is head axis major, so they lie inside its ``row_axes`` shard) and one ``psum`` over all the
        split axes adds the partial sums, giving the replicated residual directly. The path it replaces (the SPMD
        partitioner, or the row-parallel ``distributed_matmul`` engine) all-gathers the heads over the split axes outside
        ``row_axes`` first, contracts the whole ``row_axes`` shard on every device and then reduces over ``row_axes``:
        at ep4 x tp4 an all-gather over ep, a dot with 4x the weight reads and an all-reduce over tp, against one dot on
        a quarter of the rows and one all-reduce over (tp, ep). The partial sums stay in float32 through the reduction
        and are rounded to the layer dtype once. The path it replaces rounds each of its ``|row_axes|`` partial sums to
        the layer dtype and reduces them in that dtype; reducing more partials that way would add rounding error, so
        this path is the more exact of the two.

        Args:
            normed: Gated-norm output ``[lead0, lead1, value_dim]``, features split over ``split.axes``.
            weight: ``o_proj`` kernel ``[value_dim, hidden_size]``, rows split over ``row_axes`` only.
            split: Head split of the step.
            mesh: The mesh of the split axes.
            row_axes: Active mesh axes of the kernel rows, a leading part of the split's active axes.

        Returns:
            The projection ``[lead0, lead1, hidden_size]``, replicated over the split axes.
        """
        linear = self.o_proj
        heads = _active_mesh_axes(mesh, split.axes) or ()
        rest = heads[len(row_axes) :]
        own = int(normed.shape[-1]) // mesh_axis_size(mesh, heads)
        row_entry = (row_axes[0] if len(row_axes) == 1 else row_axes) if row_axes else None
        reduce_entry = heads[0] if len(heads) == 1 else heads
        scale = float(getattr(linear, "output_scale", 1.0))

        def body(x: Array, w: Array) -> Array:
            x, w = promote_dtype((x, w), dtype=linear.dtype)
            if rest:
                # Position of this device's heads inside its row shard: the split axes after ``row_axes``, major first.
                index = 0
                for name in rest:
                    index = index * mesh_axis_size(mesh, name) + jax.lax.axis_index(name)
                w = jax.lax.dynamic_slice_in_dim(w, index * own, own, axis=0)
            y = jnp.einsum("...i,io->...o", x, w, precision=linear.precision, preferred_element_type=jnp.float32)
            return jax.lax.psum(y, reduce_entry).astype(x.dtype)

        y = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(PartitionSpec(*split.lead, split.axes), PartitionSpec(row_entry, None)),
            out_specs=PartitionSpec(*split.lead, None),
            check_vma=False,
        )(normed, weight)
        return y * scale if scale != 1.0 else y

    def _column_output_axes(self, linear: tp.Any, mesh: tp.Any) -> tuple[str, ...] | None:
        """Active mesh axes of a plain column-parallel projection's output features, from its declared kernel layout.

        ``None`` when the projection is not a plain :class:`ColumnParallelLinear` with an array kernel, when its
        layout does not resolve, or when its kernel rows are split too (its output is then not a column split).
        """
        weight = getattr(getattr(linear, "weight", None), "value", None)
        if type(linear) is not ColumnParallelLinear or not isinstance(weight, jax.Array) or weight.ndim != 2:
            return None
        resolver = self.config.runtime_sharding_resolver
        try:
            kernel = tuple(
                resolver.resolve_layout(
                    linear.sharding_layout if linear.sharding_layout is not None else common_types.ColumnWise,
                    shape=tuple(int(dim) for dim in weight.shape),
                    mesh=resolver.mesh,
                )
            )
        except Exception:
            return None
        kernel = kernel + (None,) * (2 - len(kernel))
        if len(kernel) != 2:
            return None
        rows, cols = _active_mesh_axes(mesh, kernel[0]), _active_mesh_axes(mesh, kernel[1])
        if rows is None or cols is None or rows:
            return None
        return cols

    def _gather_low_rank_pair(self, forget: Array, gate: Array, split: _ServingHeadSplit) -> tuple[Array, Array]:
        """The two low-rank gate inputs (``f_a_proj`` and ``g_a_proj`` outputs) whole on every device, in one all-gather.

        Both are column-parallel outputs split over the same mesh axes, and ``f_b_proj`` / ``g_b_proj`` read them whole
        (their rows are replicated). Without this each is all-gathered on its own: two latency-bound collectives per
        layer on a decode step. The pair is stacked, gathered once and unstacked; the values are unchanged. Inputs
        that are already whole, or whose layouts differ or do not resolve, are returned as they are.

        Args:
            forget: ``f_a_proj(x)`` ``[lead0, lead1, gate_low_rank_dim]``.
            gate: ``g_a_proj(x)``, same shape and dtype.
            split: Head split of the step.

        Returns:
            ``(forget, gate)``, replicated over their column axes.
        """
        mesh = self.kda_op.get_impl_metadata().mesh
        if mesh is None or forget.ndim != 3 or forget.shape != gate.shape or forget.dtype != gate.dtype:
            return forget, gate
        axes = self._column_output_axes(self.f_a_proj, mesh)
        if not axes or axes != self._column_output_axes(self.g_a_proj, mesh):
            return forget, gate
        if int(forget.shape[-1]) % mesh_axis_size(mesh, axes):
            return forget, gate
        entry = axes[0] if len(axes) == 1 else axes

        def body(a: Array, b: Array) -> tuple[Array, Array]:
            pair = jax.lax.all_gather(jnp.stack((a, b)), entry, axis=3, tiled=True)
            return pair[0], pair[1]

        split_spec = PartitionSpec(*split.lead, entry)
        whole_spec = PartitionSpec(*split.lead, None)
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(split_spec, split_spec),
            out_specs=(whole_spec, whole_spec),
            check_vma=False,
        )(forget, gate)

    def _split_projection(self, linear: tp.Any, x: Array, split: _ServingHeadSplit, dim: int) -> Array:
        """``linear(x)`` with its kernel constrained to the split's head axes on ``dim``.

        ``dim=1``: a column-parallel projection's output features (q / k / v, beta, the gates); ``dim=0``: a
        row-parallel projection's input features (``o_proj``). The head axes refine the tensor-parallel axis (head axis
        major), so the constraint is a local slice of each device's tensor-parallel shard and the matmul runs on the
        device's own heads only; without it XLA may re-gather the activation to the kernel's layout. Layers whose kernel
        is not a plain array (quantized, LoRA, a fused distributed matmul) keep their own module call.

        Args:
            linear: A projection of this layer.
            x: Its input.
            split: Head split of the step.
            dim: Kernel axis carrying the per-head features.

        Returns:
            The projection output.
        """
        weight = getattr(getattr(linear, "weight", None), "value", None)
        if (
            type(linear) not in (ColumnParallelLinear, RowParallelLinear)
            or getattr(linear, "distributed_matmul", None) is not None
            or not isinstance(weight, jax.Array)
        ):
            return linear(x)
        spec = PartitionSpec(None, split.axes) if dim == 1 else PartitionSpec(split.axes, None)
        return linear(x, w=with_sharding_constraint(weight, spec, mesh=self.config.mesh))

    def _input_projections(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"], split: _ServingHeadSplit | None = None
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """The six projections of the layer input: ``q/k/v``, beta, the log-decay and the output-gate low rank.

        With ``config.gather_shared_projection_input`` the input that the six column-parallel projections read
        (``q/k/v_proj``, ``b_proj``, ``g_a_proj`` and ``f_a_proj``) is all-gathered over tensor parallelism once
        (:func:`easydel.layers.linears.gather_shared_input`), so the rematerialized backward does not re-gather it for
        every projection's weight gradient. Each projection still runs its own module call.

        With a serving head split (``split``, ``kda_serving_ep_split`` cache views) the input is gathered the same way
        and the per-head projections (``q/k/v``, beta, the forget gate and its log-decay) are constrained to the
        split's head axes: each device computes only its own heads' columns, from a local slice of the column-parallel
        weights.

        Args:
            hidden_states: Layer input ``(batch, seq_len, hidden_size)``.
            split: Head split of an eSurge serving step, or ``None``.

        Returns:
            ``(query, key, value, beta, decay, gate_low_rank)``: the three projections before the convolutions,
            ``sigmoid(b_proj(x))`` ``[B, S, heads]``, the per-channel log-decay ``[B, S, heads, head_k_dim]`` and
            ``g_a_proj(x)`` ``[B, S, gate_low_rank_dim]``.
        """
        if split is not None or resolve_gather_shared_projection_input(self.config, hidden_states):
            hidden_states = gather_shared_input(hidden_states, self.config)

        def project(linear: tp.Any, x: Array) -> Array:
            if split is None:
                return linear(x)
            return self._constrain_heads(self._split_projection(linear, x, split, dim=1), split)

        query = project(self.q_proj, hidden_states)
        key = project(self.k_proj, hidden_states)
        value = project(self.v_proj, hidden_states)
        b_logits = project(self.b_proj, hidden_states)
        gate_low_rank = self.g_a_proj(hidden_states)
        forget_low_rank = self.f_a_proj(hidden_states)
        if split is not None:
            # f_b / g_b read both low-rank inputs whole: one all-gather for the pair instead of one each.
            forget_low_rank, gate_low_rank = self._gather_low_rank_pair(forget_low_rank, gate_low_rank, split)
        decay = fused_kda_gate_per_channel(
            project(self.f_b_proj, forget_low_rank),
            self.A_log.value,
            self.dt_bias.value,
            lower_bound=self.gate_lower_bound,
        )
        if split is not None:
            decay = self._constrain_heads(decay, split, trailing=1)
        return query, key, value, jax.nn.sigmoid(b_logits), decay, gate_low_rank

    def _forward_packed_rows(
        self,
        hidden_states: Float[Array, "1 seq_len hidden_dim"],
        query: Float[Array, "1 seq_len key_dim"],
        key: Float[Array, "1 seq_len key_dim"],
        value: Float[Array, "1 seq_len value_dim"],
        beta: Float[Array, "1 seq_len num_heads"],
        decay: Float[Array, "1 seq_len num_heads head_k_dim"],
        cache_view: KDACacheView,
        query_start_loc: Array,
        gate_low_rank: Float[Array, "1 seq_len gate_low_rank_dim"],
        cache_metadata: KDAMetadata | None = None,
        split: _ServingHeadSplit | None = None,
    ) -> AttentionLayerOutput:
        """Advance per-request KDA state over an eSurge packed token stream ``[1, T, D]``.

        Without speculative candidate rows (``cache_metadata.spec_recurrent_commit`` is ``None``) the conv and the
        delta rule run on the flat stream in :func:`~easydel.operations.kernels.kda_ragged_rows_forward` (ragged
        conv1d plus the ``"ragged_kda"`` operation), so the work follows the scheduled token count; the pool may hold
        more slots than the step has rows. Row ``r`` owns pool slot ``cache_metadata.recurrent_state_indices[r]`` when
        the metadata carries it (eSurge must ship it when a row window does not start at slot 0: multi-window steps,
        sparse row packing), else slot ``r``; the row-padded driver always uses slot ``r`` and lays candidate rows out by
        the step's row count, so speculative pools need single-window, unpacked steps. eSurge allocates
        ``rows * (1 + C)`` slots for speculative candidate rows and
        sends the commit metadata exactly then; those steps use the row-padded
        :func:`~easydel.operations.kernels.kda_packed_rows_forward`, which writes the per-prefix states. The flat
        driver's chunk dots follow ``config.kda_chunk_precision`` only with ``kda_chunk_impl="kda_chunk"`` (the
        setting it documents) and are ``"highest"`` otherwise. This method adds the output gate, norm and projection.

        Args:
            hidden_states: Packed layer input ``[1, T, D]``.
            query: Packed Q projection ``[1, T, key_dim]``.
            key: Packed K projection ``[1, T, key_dim]``.
            value: Packed V projection ``[1, T, value_dim]``.
            beta: Packed write strength ``[1, T, heads]`` (sigmoid applied).
            decay: Packed per-channel log-decay ``[1, T, heads, head_k_dim]``.
            cache_view: Per-layer KDA cache view (per-request rows).
            query_start_loc: Cumulative segment offsets ``[rows + 1]``.
            gate_low_rank: ``g_a_proj(hidden_states)`` from :meth:`_input_projections`.
            cache_metadata: Optional KDA metadata (context lengths, speculative commit).
            split: Head split of the step (``kda_serving_ep_split`` cache views), or ``None``. The flat and decode
                drivers then run their conv and recurrence on the split's head axes; the row-padded speculative driver
                leaves the layout to XLA.

        Returns:
            :class:`AttentionLayerOutput` with the packed output row and the updated cache view.

        Raises:
            ValueError: If a speculative step (``spec_recurrent_commit`` set) also names explicit pool slots
                (``recurrent_state_indices``), which the row-padded driver cannot follow.
        """
        conv_kernels = tuple(getattr(self, f"{n}_conv1d").weight.value.squeeze(1).T for n in ("q", "k", "v"))
        conv_dtype = jnp.bfloat16 if self.dtype in lowfloats else self.dtype
        context_lens = getattr(cache_metadata, "context_lens", None)
        # eSurge sends the commit metadata exactly when the pool holds speculative candidate rows; the pool size alone
        # cannot tell (it keeps max_num_reqs slots, which may exceed twice the step's row cap without any candidates).
        spec_recurrent_commit = getattr(cache_metadata, "spec_recurrent_commit", None)
        # Pool slot of every row when eSurge ships them (a row window that does not start at slot 0); None: row r owns
        # slot r. The row-padded (speculative) driver below cannot use them.
        recurrent_state_indices = getattr(cache_metadata, "recurrent_state_indices", None)
        # ``kda_chunk_precision`` is documented for ``kda_chunk_impl="kda_chunk"``; other impls keep exact chunk dots.
        chunk_precision = self.config.kda_chunk_precision if self.config.kda_chunk_impl == "kda_chunk" else "highest"
        if spec_recurrent_commit is None:
            # Decode-only steps (every row <= 1 token) take the fused decode path, other steps the flat driver.
            out_packed, new_cache_view = kda_rows_forward(
                flat_forward=kda_ragged_rows_forward,
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                conv_kernels=conv_kernels,
                cache_view=cache_view,
                query_start_loc=query_start_loc,
                context_lens=context_lens,
                metadata=self.kda_op.get_impl_metadata(),
                conv_dtype=conv_dtype,
                chunk_size=self.chunk_size,
                # The safe gate keeps every log-decay in [lower_bound, 0); the softplus gate passes None.
                decay_lower_bound=self.gate_lower_bound,
                precision=chunk_precision,
                state_indices=recurrent_state_indices,
                head_axis=None if split is None else split.axes,
            )
        else:
            if recurrent_state_indices is not None:
                raise ValueError(
                    "A speculative KDA step (cache_metadata.spec_recurrent_commit is set) runs the row-padded driver, "
                    "which keeps row r in pool slot r and lays the candidate rows out by the step's row count, so it "
                    "cannot follow cache_metadata.recurrent_state_indices: run speculative steps as one unpacked row "
                    "window that starts at slot 0, without explicit slots."
                )
            out_packed, new_cache_view = kda_packed_rows_forward(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                conv_kernels=conv_kernels,
                cache_view=cache_view,
                query_start_loc=query_start_loc,
                context_lens=context_lens,
                conv_dtype=conv_dtype,
                chunk_size=self.chunk_size,
                spec_recurrent_commit=spec_recurrent_commit,
            )
        output = self._gated_output_projection(out_packed, gate_low_rank, hidden_states.dtype, split)
        return AttentionLayerOutput(attention_output=output, attention_weight=None, cache_view=new_cache_view)


__all__ = (
    "KDALayerSpec",
    "KernelDeltaAttention",
    "kda_layer_spec",
    "kda_serving_head_axes",
    "unnest_legacy_kda_forget_gate",
)
