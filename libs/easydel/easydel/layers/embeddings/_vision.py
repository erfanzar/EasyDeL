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

"""Shared spatiotemporal patch embedding for vision-language encoders."""

import jax
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array, Float
from spectrax import apply_logical_sharding, common_types, nn


class VisionPatchEmbed(spx.Module):
    """Patch embedding layer for converting images/video frames into embeddings.

    Uses 3D convolution to convert image or video patches into embeddings,
    handling both spatial (height, width) and temporal dimensions for video input.
    The bias-free convolution remains under ``proj`` for checkpoint compatibility.
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        precision: jax.lax.PrecisionLike = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        partition_manager=None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the patch embedding layer.

        Args:
            patch_size (int, optional): Spatial patch size (height and width). Defaults to 14.
            temporal_patch_size (int, optional): Temporal patch size for video frames. Defaults to 2.
            in_channels (int, optional): Number of input channels (RGB=3). Defaults to 3.
            embed_dim (int, optional): Output embedding dimension. Defaults to 1152.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            partition_manager: Optional logical sharding resolver for the output embeddings.
            rngs (spx.Rngs): Random number generator state.
        """
        self.dtype = dtype
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.partition_manager = partition_manager

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            precision=precision,
        )

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Apply patch embedding to input pixels.

        Args:
            hidden_states (Array): Input pixel values with each patch flattened in
                channel/temporal/height/width order.

        Returns:
            Array: Patch embeddings of shape (num_patches, embed_dim).
        """
        hidden_states = jnp.transpose(
            hidden_states.reshape(
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ),
            (0, 2, 3, 4, 1),
        )
        hidden_states = self.proj(hidden_states.astype(self.dtype))
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.partition_manager,
        )
        return hidden_states
