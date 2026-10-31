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
``kda_chunk_precision`` (the chunked kernel, see :class:`~easydel.operations.kernels.KernelDeltaAttnOp`) and
``gather_shared_projection_input`` (one tensor-parallel gather for the six input projections).
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
from jaxtyping import Array, Float
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import KDACacheView, KDAMetadata
from easydel.infra.modeling_outputs import AttentionLayerOutput
from easydel.infra.sequence_packing import packed_segment_ids_from_mask_info
from easydel.infra.utils import ArrayParam
from easydel.layers.linears import ColumnParallelLinear, RowParallelLinear, gather_shared_input
from easydel.layers.norms import RMSNormGated, lowfloats
from easydel.operations import OperationMetadata
from easydel.operations.kernels import (
    KDAOutput,
    KernelDeltaAttnOp,
    fused_kda_gate_per_channel,
    kda_packed_rows_forward,
)

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
    decode); eSurge packed streams go through :func:`~easydel.operations.kernels.kda_packed_rows_forward`.

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

        query, key, value, beta, decay, gate_low_rank = self._input_projections(hidden_states)

        query_start_loc = getattr(cache_metadata, "query_start_loc", None) if cache_metadata is not None else None
        if cache_view is not None and query_start_loc is not None:
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

    def _input_projections(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """The six projections of the layer input: ``q/k/v``, beta, the log-decay and the output-gate low rank.

        With ``config.gather_shared_projection_input`` the input that the six column-parallel projections read
        (``q/k/v_proj``, ``b_proj``, ``g_a_proj`` and ``f_a_proj``) is all-gathered over tensor parallelism once
        (:func:`easydel.layers.linears.gather_shared_input`), so the rematerialized backward does not re-gather it for
        every projection's weight gradient. Each projection still runs its own module call.

        Args:
            hidden_states: Layer input ``(batch, seq_len, hidden_size)``.

        Returns:
            ``(query, key, value, beta, decay, gate_low_rank)``: the three projections before the convolutions,
            ``sigmoid(b_proj(x))`` ``[B, S, heads]``, the per-channel log-decay ``[B, S, heads, head_k_dim]`` and
            ``g_a_proj(x)`` ``[B, S, gate_low_rank_dim]``.
        """
        if self.config.gather_shared_projection_input:
            hidden_states = gather_shared_input(hidden_states, self.config)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        b_logits = self.b_proj(hidden_states)
        gate_low_rank = self.g_a_proj(hidden_states)
        decay = fused_kda_gate_per_channel(
            self.f_b_proj(self.f_a_proj(hidden_states)),
            self.A_log.value,
            self.dt_bias.value,
            lower_bound=self.gate_lower_bound,
        )
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
    ) -> AttentionLayerOutput:
        """Advance per-request KDA state over an eSurge packed token stream ``[1, T, D]``.

        The conv and the delta rule run per request row in :func:`~easydel.operations.kernels.kda_packed_rows_forward`;
        this method adds the output gate, norm and projection.

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

        Returns:
            :class:`AttentionLayerOutput` with the packed output row and the updated cache view.
        """
        seq_len = hidden_states.shape[1]
        out_packed, new_cache_view = kda_packed_rows_forward(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            conv_kernels=tuple(getattr(self, f"{n}_conv1d").weight.value.squeeze(1).T for n in ("q", "k", "v")),
            cache_view=cache_view,
            query_start_loc=query_start_loc,
            context_lens=getattr(cache_metadata, "context_lens", None),
            conv_dtype=jnp.bfloat16 if self.dtype in lowfloats else self.dtype,
            chunk_size=self.chunk_size,
            spec_recurrent_commit=getattr(cache_metadata, "spec_recurrent_commit", None),
        )
        output_gate = self.g_b_proj(gate_low_rank).reshape(1, seq_len, self.num_heads, self.head_v_dim)
        output = self.o_norm(out_packed.astype(hidden_states.dtype), output_gate)
        output = checkpoint_name(self.o_proj(output.reshape(1, seq_len, -1)), name="attn_output")
        return AttentionLayerOutput(attention_output=output, attention_weight=None, cache_view=new_cache_view)


__all__ = ("KDALayerSpec", "KernelDeltaAttention", "kda_layer_spec", "unnest_legacy_kda_forget_gate")
