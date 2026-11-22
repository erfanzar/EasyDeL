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

"""Channel-major 1D convolution with the HF Mamba parameter layout."""

from __future__ import annotations

import spectrax as spx
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array

__all__ = ("ChannelMajorConv1D",)


class ChannelMajorConv1D(spx.Module):
    """1D convolution over ``(batch, channels, seq_len)`` inputs (Mamba / FalconMamba mixer front-end).

    Parameter layout matches HuggingFace after conversion: ``weight`` is ``[kernel_size, 1, features]`` (transposed
    to the ``OIW`` conv kernel at apply time) and ``bias`` is ``[features]``. Set ``groups=features`` for the
    depthwise form the mixers use; causality comes from ``padding=kernel_size - 1`` plus slicing by the caller.

    Attributes:
        weight: Kernel ``[kernel_size, 1, features]``.
        bias: Optional per-channel bias ``[features]``.
    """

    def __init__(
        self,
        features: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        num_spatial_dims: int = 1,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the convolution.

        Args:
            features: Number of output channels.
            kernel_size: Kernel width.
            stride: Convolution stride.
            padding: Zero padding applied to both sides of the sequence axis.
            dilation: Kernel dilation.
            groups: Feature groups (``features`` for depthwise).
            use_bias: Whether to add a learnable per-channel bias.
            num_spatial_dims: Spatial dimensions (always 1).
            dtype: Compute dtype.
            param_dtype: Parameter storage dtype.
            precision: Convolution precision.
            rngs: Random number generators.
        """
        # Imported lazily: ``easydel.infra.utils`` imports ``easydel.layers``.
        from easydel.infra.utils import ArrayParam

        self.weight = ArrayParam.bound(
            shape=(kernel_size, 1, features),
            dtype=param_dtype,
            init_method="lecun_normal",
            key=rngs.parameters,
        )
        if use_bias:
            self.bias = ArrayParam.bound(
                shape=(features,),
                dtype=param_dtype,
                init_method="zeros",
                key=rngs.parameters,
            )
        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.num_spatial_dims = num_spatial_dims
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

    def forward(self, x: Array) -> Array:
        """Convolve ``x`` ``(batch, channels, seq_len)``; the result is cast back to ``x.dtype``.

        Raises:
            ValueError: If ``x`` does not have rank ``num_spatial_dims + 2``.
        """
        unbatched_rank = self.num_spatial_dims + 2
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank}, but input has shape {x.shape}.",
            )
        org_x_dtype = x.dtype
        x = lax.conv_general_dilated(
            lhs=x.astype(self.dtype),
            rhs=jnp.asarray(jnp.swapaxes(self.weight.value, 0, 2), dtype=self.dtype),
            window_strides=(self.stride,),
            padding=((self.padding, self.padding),),
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
            precision=self.precision,
        )
        if self.use_bias:
            x = x + jnp.asarray(self.bias.value.reshape(1, -1, 1), dtype=self.dtype)
        return x.astype(org_x_dtype)
