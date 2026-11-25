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

"""Residual short causal convolutions (static depthwise and dynamic per-head taps).

Provides reusable float32 short causal convolution modules with variance-scaled
residual connections, packed-sequence segment isolation, and rolling cache-state
updates compatible with :class:`~easydel.caching.ConvKVCacheView`:

- :class:`ResidualShortConvolution`: static float32 depthwise causal 1-D
  convolution returning ``(conv(x) + x) / sqrt(K + 1)``.
- :class:`DynamicResidualShortConvolution`: combines static per-channel taps
  with input-conditioned per-head dynamic taps returning
  ``(conv(x) + x) / sqrt(2 * K + 1)``.
"""

from __future__ import annotations

import math
import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from spectrax import nn

from ..linears import ColumnParallelLinear

if tp.TYPE_CHECKING:
    from easydel.caching import ConvKVCacheView

__all__ = (
    "DynamicResidualShortConvolution",
    "ResidualShortConvolution",
)


def _apply_conv_mask(hidden_states_fp32: Array, conv_mask: Array | None) -> Array:
    """Zero out padded token positions prior to causal convolution."""
    if conv_mask is None:
        return hidden_states_fp32
    return hidden_states_fp32 * conv_mask[:, :, None].astype(hidden_states_fp32.dtype)


class ResidualShortConvolution(spx.Module):
    """Static float32 depthwise causal short convolution with scaled residual ``(conv + x) / sqrt(K + 1)``."""

    def __init__(
        self,
        hidden_size: int,
        conv_kernel_size: int,
        layer_idx: int,
        conv_idx: int,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the static depthwise short convolution.

        Args:
            hidden_size: Channel count (one depthwise filter per channel).
            conv_kernel_size: Causal kernel width ``K``.
            layer_idx: Owning decoder layer index.
            conv_idx: Index of this convolution's buffer inside ``cache_view.conv_states``.
            rngs: Random number generator container.
        """
        self.hidden_size = int(hidden_size)
        self.conv_kernel_size = int(conv_kernel_size)
        self.layer_idx = int(layer_idx)
        self.conv_idx = int(conv_idx)
        self._out_scale = 1.0 / math.sqrt(self.conv_kernel_size + 1)
        # Depthwise causal Conv1d stored in float32; weight layout is (K, 1, C).
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_kernel_size,
            groups=self.hidden_size,
            padding="VALID",
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len dim"],
        dynamic_input: Array | None = None,
        cache_view: ConvKVCacheView | None = None,
        conv_mask: Bool[Array, "batch seq_len"] | None = None,
        seq_idx: Int[Array, "batch seq_len"] | None = None,
    ) -> tuple[Array, ConvKVCacheView | None]:
        """Run the causal short convolution and optionally update ``cache_view``."""
        del dynamic_input
        input_dtype = hidden_states.dtype
        x = hidden_states.astype(jnp.float32)
        residual = x
        masked_x = _apply_conv_mask(x, conv_mask)

        k = self.conv_kernel_size
        seq_len = masked_x.shape[1]
        if cache_view is not None:
            context, cache_view = cache_view.update_conv_state(self.conv_idx, masked_x)
        else:
            context = jnp.pad(masked_x, ((0, 0), (k - 1, 0), (0, 0)))

        if seq_idx is not None and cache_view is None:
            # Segment-isolated causal depthwise convolution for packed sequences.
            # conv1d.weight has shape (K, 1, C) with index K-1 weighting the current token (lag=0).
            w = self.conv1d.weight.value.astype(jnp.float32)[:, 0, :]  # (K, C)
            ctx_seq = jnp.concatenate([jnp.broadcast_to(seq_idx[:, :1], (seq_idx.shape[0], k - 1)), seq_idx], axis=1)
            acc = jnp.zeros_like(masked_x)
            for lag in range(k):
                start = (k - 1) - lag
                shifted = context[:, start : start + seq_len, :]
                same_seg = (ctx_seq[:, start : start + seq_len] == seq_idx)[:, :, None]
                acc = acc + jnp.where(same_seg, shifted, 0.0) * w[k - 1 - lag]
            conv_out = acc
        else:
            conv_out = self.conv1d(context)

        out = (conv_out + residual) * self._out_scale
        return out.astype(input_dtype), cache_view


class DynamicResidualShortConvolution(spx.Module):
    """Short causal convolution with static per-channel taps plus dynamic per-head taps.

    Tap index ``0`` weights the current token (backward distance ``lag``). The
    output includes a variance-normalized residual ``(conv + x) / sqrt(2K + 1)``.
    """

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        num_heads: int,
        conv_kernel_size: int,
        layer_idx: int,
        conv_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        initializer_range: float = 0.02,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the dynamic + static short convolution.

        Args:
            hidden_size: Total feature width ``num_heads * head_dim`` being convolved.
            input_size: Width of ``dynamic_input`` conditioning the per-head taps.
            num_heads: Number of heads sharing dynamic taps.
            conv_kernel_size: Causal kernel width ``K``.
            layer_idx: Owning decoder layer index.
            conv_idx: Index of this convolution's buffer inside ``cache_view.conv_states``.
            dtype: Activation dtype for ``weight_proj``.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision for ``weight_proj``.
            initializer_range: Standard deviation for ``weight_proj`` init.
            rngs: Random number generator container.
        """
        from easydel.infra.utils import ArrayParam

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.conv_kernel_size = int(conv_kernel_size)
        self.layer_idx = int(layer_idx)
        self.conv_idx = int(conv_idx)
        self._out_scale = 1.0 / math.sqrt(2 * self.conv_kernel_size + 1)

        self.static_weight = ArrayParam.bound(
            shape=(self.conv_kernel_size, self.hidden_size),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        self.weight_proj = ColumnParallelLinear(
            input_size,
            self.num_heads * self.conv_kernel_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=precision,
            rngs=rngs,
        )

    @property
    def reform_param(self) -> dict[str, tp.Any]:
        """Protect 2-D ``static_weight`` from generic linear-weight transposition on load/export."""
        return {
            "static_weight$": {
                "splits": [{"name": "static_weight", "spliter": lambda x: x}],
                "inverse_spliter": lambda x: x,
            }
        }

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len dim"],
        dynamic_input: Float[Array, "batch seq_len input_dim"],
        cache_view: ConvKVCacheView | None = None,
        conv_mask: Bool[Array, "batch seq_len"] | None = None,
        seq_idx: Int[Array, "batch seq_len"] | None = None,
    ) -> tuple[Array, ConvKVCacheView | None]:
        """Run the dynamic + static causal short convolution and optionally update ``cache_view``."""
        input_dtype = hidden_states.dtype
        batch_size, seq_len, hidden_size = hidden_states.shape
        k = self.conv_kernel_size

        dynamic_weights = self.weight_proj(dynamic_input).astype(jnp.float32)
        dynamic_weights = dynamic_weights.reshape(batch_size, seq_len, self.num_heads, 1, k)
        static_w = self.static_weight.value.astype(jnp.float32).T.reshape(1, 1, self.num_heads, self.head_dim, k)
        weights = static_w + dynamic_weights

        x = hidden_states.astype(jnp.float32)
        residual = x
        masked_x = _apply_conv_mask(x, conv_mask)

        if cache_view is not None:
            context, cache_view = cache_view.update_conv_state(self.conv_idx, masked_x)
        else:
            context = jnp.pad(masked_x, ((0, 0), (k - 1, 0), (0, 0)))

        # windows[..., lag] holds the token `lag` steps back (lag = 0 is current token).
        windows = jnp.stack(
            [context[:, (k - 1 - lag) : (k - 1 - lag + seq_len), :] for lag in range(k)],
            axis=-1,
        ).reshape(batch_size, seq_len, self.num_heads, self.head_dim, k)

        if seq_idx is not None and cache_view is None:
            ctx_seq = jnp.concatenate([jnp.broadcast_to(seq_idx[:, :1], (batch_size, k - 1)), seq_idx], axis=1)
            win_seq = jnp.stack(
                [ctx_seq[:, (k - 1 - lag) : (k - 1 - lag + seq_len)] for lag in range(k)],
                axis=-1,
            )
            windows = jnp.where((win_seq == seq_idx[..., None])[:, :, None, None, :], windows, 0.0)

        out = jnp.sum(windows * weights, axis=-1).reshape(batch_size, seq_len, hidden_size)
        out = (out + residual) * self._out_scale
        return out.astype(input_dtype), cache_view
