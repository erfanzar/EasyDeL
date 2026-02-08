# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Neural network modules for Rotary Position Embeddings (RoPE).

This module provides Flax NNX Module classes that implement various RoPE
scaling methods for use in transformer attention layers. Each class computes
and applies rotary positional embeddings to query and key tensors.

Classes:
    RotaryEmbedding: Standard RoPE with no scaling (default).
    MultiModalRotaryEmbedding: Multi-modal RoPE for vision-language models (mRoPE).
    LinearScalingRotaryEmbedding: RoPE with linear position scaling.
    DynamicNTKScalingRotaryEmbedding: RoPE with dynamic NTK-aware scaling.
    YaRNScalingRotaryEmbedding: RoPE with YaRN context extension method.
    DeepseekScalingRotaryEmbedding: RoPE with Deepseek-YaRN variant scaling.
    Phi3LongRoPEScaledRotaryEmbedding: RoPE with Phi-3 LongRoPE scaling.
    Llama3RotaryEmbedding: RoPE with Llama-3 wavelength-based scaling.

The `rope_wraper` decorator registers each class in `AVAILABLE_ROPE_TYPES`
for dynamic lookup by rope type name.

Example:
    >>> import jax.numpy as jnp
    >>> from easydel.layers.components.rotary_embedding import RotaryEmbedding
    >>> # Create a standard RoPE module
    >>> rope = RotaryEmbedding(
    ...     head_size=64,
    ...     rotary_dim=64,
    ...     max_position_embeddings=2048,
    ...     base=10000,
    ...     is_neox_style=True,
    ...     dtype=jnp.float32,
    ... )
    >>> # Apply RoPE to query and key tensors
    >>> query = jnp.ones((1, 128, 8, 64))  # [batch, seq, heads, head_dim]
    >>> key = jnp.ones((1, 128, 8, 64))
    >>> positions = jnp.arange(128)
    >>> q_rot, k_rot = rope(positions, query, key)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx as nn

from ._compute_fns import (
    apply_basic_rope,
    apply_phi3_rope,
    compute_basic_frequencies,
    compute_basic_inv_frequencies,
    compute_deepseek_frequencies,
    compute_dynamic_frequencies,
    compute_linear_frequencies,
    compute_llama3_frequencies,
    compute_phi3_frequencies,
    compute_yarn_frequencies,
)
from ._utils import _rotate_gptj, _rotate_neox

AVAILABLE_ROPE_TYPES = {}
"""A dictionary to store registered RoPE (Rotary Position Embedding) types and their configurations."""


def rope_wraper(type):  # noqa
    """
    A decorator factory that registers a RotaryEmbedding class under a specific type name.

    This allows retrieving RoPE configurations by type name later. It also sets
    basic __str__ and __repr__ for the decorated class.

    Args:
        type (str): The name to register the RoPE class under (e.g., "linear", "yarn").

    Returns:
        Callable: A decorator function that takes a RotaryEmbedding class, registers it,
                  and returns the class.
    """

    def w(rope: RotaryEmbedding):
        """
        Decorator function that registers the RoPE class.

        Args:
            rope (RotaryEmbedding): The RotaryEmbedding class to register.

        Returns:
            RotaryEmbedding: The registered RotaryEmbedding class.
        """
        properties = {k: v for k, v in rope.__dict__.items()}
        AVAILABLE_ROPE_TYPES[type] = properties
        rope.__str__ = lambda cls: str(cls.__class__.__name__)
        rope.__repr__ = lambda cls: repr(cls.__class__.__name__)
        rope._type = type
        return rope

    return w


@rope_wraper("default")
class RotaryEmbedding(nn.Module):
    """
    Standard Rotary Positional Embedding (RoPE) module.

    Attributes:
        head_size (int): The dimension size of each attention head.
        rotary_dim (int): The dimension size of the rotary embeddings applied. Can be <= head_size.
        max_position_embeddings (int): The maximum sequence length the model can handle.
        base (int): The base value for calculating frequencies.
        is_neox_style (bool): Flag indicating whether to use Neox-style rotation.
        dtype (jnp.dtype): Data type for computations.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        """Initialize the RotaryEmbedding module.

        Args:
            head_size: The dimension size of each attention head.
            rotary_dim: The dimension size of the rotary embeddings applied.
                Can be less than or equal to head_size.
            max_position_embeddings: The maximum sequence length the model can handle.
            base: The base value for calculating inverse frequencies in the
                geometric progression.
            is_neox_style: If True, uses Neox-style rotation (split and concatenate).
                If False, uses GPT-J-style rotation (interleaved).
            dtype: Data type for the output embeddings (e.g., jnp.float32).
        """
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

    @jax.named_scope("easydel-rope-embedding")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply rotary positional embeddings to query and key tensors.

        Computes the frequency cache if not provided, then applies the rotary
        transformation to the query and key tensors based on the given positions.

        Args:
            positions: Position indices for each token in the sequence.
                Shape: [sequence_length] or broadcastable shape.
            query: Query tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            key: Key tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            offsets: Optional position offsets to add to positions (e.g., for
                KV cache continuation). Shape: broadcastable with positions.
                Defaults to None.
            frequencies: Optional pre-computed frequency cache. If None,
                frequencies are computed internally. Shape: [max_length, rotary_dim].
                Defaults to None.

        Returns:
            A tuple of (rotated_query, rotated_key) tensors with the same
            shapes as the input query and key, cast to self.dtype.
        """
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_basic_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return dynamic partition specs for this module's parameters."""
        return {}


@rope_wraper("mrope")
class MultiModalRotaryEmbedding(RotaryEmbedding):
    """Multi-dimensional RoPE (MRoPE) with interleaved THW layout for Qwen2/3-VL models.

    MRoPE (Multi-dimensional Rotary Position Embedding) extends standard RoPE to handle
    3D position information (Temporal, Height, Width) for vision-language models.

    The interleaving pattern reorganizes frequencies from chunked [TTT...HHH...WWW] to
    interleaved [T₀,H₀,W₀, T₁,H₁,W₁, ...], preserving frequency continuity for each
    spatial/temporal dimension.

    Attributes:
        mrope_section: Tuple of (T, H, W) dimensions specifying how many frequency
            components are allocated to each dimension. Default: (24, 20, 20) for
            64-dim rotary embeddings (128 head_dim / 2).
        attention_scaling: Post-processing scaling factor applied to cos/sin.
            Default 1.0 for standard mRoPE. Can be set for advanced RoPE types.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        mrope_section: tuple[int, int, int] | None = None,
        attention_scaling: float = 1.0,
        mrope_interleaved: bool = True,
        repetition_style: bool = False,
    ):
        """Initialize the MultiModalRotaryEmbedding module.

        Args:
            head_size: The dimension size of each attention head.
            rotary_dim: The dimension size of the rotary embeddings applied.
            max_position_embeddings: The maximum sequence length the model can handle.
            base: The base value for calculating inverse frequencies.
            is_neox_style: If True, uses Neox-style rotation. If False, uses GPT-J-style.
            dtype: Data type for the output embeddings.
            mrope_section: Tuple of (T, H, W) dimensions specifying frequency allocation
                for Temporal, Height, and Width dimensions. Defaults to (24, 20, 20)
                for 64-dim rotary embeddings (128 head_dim / 2).
            attention_scaling: Post-processing scaling factor applied to cos/sin values.
                Defaults to 1.0 for standard mRoPE.
            mrope_interleaved: If True, uses Qwen3-VL style interleaved pattern.
                If False, uses Qwen2-VL style chunked pattern. Defaults to True.
            repetition_style: If True, uses HuggingFace-style list repetition for
                chunked mRoPE. Defaults to False.

        Raises:
            ValueError: If rotary_dim is not positive.
            ValueError: If mrope_section is incompatible with rotary_dim.
        """
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
        section = tuple(mrope_section) if mrope_section is not None else (24, 20, 20)
        expected = self.rotary_dim // 2
        actual = sum(section)
        if expected <= 0:
            raise ValueError(f"rotary_dim must be positive for mRoPE; got rotary_dim={self.rotary_dim}.")
        if actual != expected:
            scaled_section: list[int] = []
            for size in section:
                num = size * expected
                if num % actual != 0:
                    raise ValueError(
                        "mrope_section is incompatible with rotary_dim. "
                        f"Expected sum(mrope_section)={expected}, got {actual} for rotary_dim={self.rotary_dim}."
                    )
                scaled_section.append(num // actual)
            if sum(scaled_section) != expected:
                scaled_section[-1] += expected - sum(scaled_section)
            section = tuple(scaled_section)
        self.repetition_style = repetition_style
        self.mrope_section = section
        self.attention_scaling = attention_scaling
        self.mrope_interleaved = mrope_interleaved

    def _apply_chunked_mrope(self, emb: jax.Array) -> jax.Array:
        """Apply Qwen2-VL style chunked mRoPE pattern.

        This method implements the HuggingFace apply_multimodal_rotary_pos_emb
        pattern used in Qwen2-VL models. It takes doubled embeddings (frequencies
        concatenated with themselves) and selects chunks from appropriate
        dimensions (T=0, H=1, W=2).

        The chunked pattern produces contiguous frequency blocks for each
        spatial/temporal dimension, e.g., [T:0-31, H:32-79, W:80-127].

        Args:
            emb: Doubled frequency embeddings with shape (3, batch, seq, rotary_dim),
                where axis 0 corresponds to T, H, W dimensions.

        Returns:
            Combined frequencies with shape (batch, seq, rotary_dim) after
            selecting and concatenating appropriate chunks from each dimension.
        """
        if self.repetition_style:
            # HuggingFace splits using `mrope_section * 2` (list repetition), then
            # takes chunk i from dimension (i % 3): T=0, H=1, W=2.
            split_sizes = list(self.mrope_section) * 2
            chunks = jax.lax.split(emb, split_sizes, axis=-1)
            selected = [chunk[i % 3] for i, chunk in enumerate(chunks)]
            return jnp.concatenate(selected, axis=-1)
        else:
            t_size, h_size, _w_size = self.mrope_section
            t_size_doubled = t_size * 2
            h_size_doubled = h_size * 2

            # Extract chunks and select from appropriate dimension (T=0, H=1, W=2)
            t_chunk = emb[0, ..., :t_size_doubled]
            h_chunk = emb[1, ..., t_size_doubled : t_size_doubled + h_size_doubled]
            w_chunk = emb[2, ..., t_size_doubled + h_size_doubled :]

            return jnp.concatenate([t_chunk, h_chunk, w_chunk], axis=-1)

    def _apply_interleaved_mrope(self, freqs: jax.Array) -> jax.Array:
        """Apply Qwen3-VL style interleaved mRoPE pattern.

        This method interleaves frequencies from T, H, W dimensions to create
        a pattern where frequencies alternate: [T0, H0, W0, T1, H1, W1, ...].
        This preserves frequency continuity for each spatial/temporal dimension.

        Args:
            freqs: Frequency embeddings with shape (3, batch, seq, rotary_dim//2),
                where axis 0 corresponds to T, H, W dimensions.

        Returns:
            Interleaved frequencies with shape (batch, seq, rotary_dim//2).
        """
        freqs_t = freqs[0]
        for dim_idx, offset in enumerate((1, 2), start=1):
            section_size = self.mrope_section[dim_idx] * 3
            idx = slice(offset, section_size, 3)
            freqs_t = freqs_t.at[..., idx].set(freqs[dim_idx, ..., idx])
        return freqs_t

    @jax.named_scope("easydel-mrope")
    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
        offsets: jax.Array | None = None,
        frequencies: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply multimodal rotary position embedding (mRoPE) to query/key.

        Supports two mRoPE patterns:
        - Chunked (Qwen2-VL): mrope_interleaved=False - contiguous T/H/W chunks
        - Interleaved (Qwen3-VL): mrope_interleaved=True - interleaved T/H/W pattern

        Args:
            positions: Position IDs with shape (batch, seq) or (3, batch, seq).
                If 2D, broadcasts to 3D with same positions for T, H, W.
                For vision-language tasks, should be (3, batch, seq) with separate
                T, H, W positions computed via get_rope_index.
            query: Query tensor to apply rotary embedding to.
            key: Key tensor to apply rotary embedding to.
            offsets: Optional position offsets (e.g., for KV cache).
            frequencies: Optional pre-computed frequency cache.

        Returns:
            Tuple of (rotated_query, rotated_key) with same dtype as input.
        """
        # Normalize positions to (3, batch, seq)
        if positions.ndim == 2:
            positions = jnp.broadcast_to(positions[jnp.newaxis, ...], (3, *positions.shape))
        elif positions.ndim != 3 or positions.shape[0] != 3:
            raise ValueError(f"Position IDs must have shape (batch, seq) or (3, batch, seq); got {positions.shape}.")

        if offsets is not None:
            positions = positions + offsets

        if frequencies is not None:
            freq_cache = getattr(frequencies, "value", frequencies)
            # freq_cache expected shape: [max_pos, rotary_dim] containing [cos, sin] concat
            freq_cache = jnp.asarray(freq_cache)
            freqs_full = jnp.stack(
                [
                    freq_cache[positions[0]],
                    freq_cache[positions[1]],
                    freq_cache[positions[2]],
                ],
                axis=0,
            )  # (3, b, seq, rotary_dim)
            cos_half, sin_half = jnp.split(freqs_full, 2, axis=-1)  # each (3, b, seq, dim/2)

            if self.mrope_interleaved:
                # Qwen3-VL style: apply interleaving on half-dim freqs, then double
                cos_interleaved = self._apply_interleaved_mrope(cos_half)  # (b, seq, dim/2)
                sin_interleaved = self._apply_interleaved_mrope(sin_half)  # (b, seq, dim/2)
                cos = jnp.concatenate([cos_interleaved, cos_interleaved], axis=-1)  # (b, seq, dim)
                sin = jnp.concatenate([sin_interleaved, sin_interleaved], axis=-1)  # (b, seq, dim)
            else:
                # Qwen2-VL style: double first, then apply chunked pattern
                cos_doubled = jnp.concatenate([cos_half, cos_half], axis=-1)  # (3, b, seq, dim)
                sin_doubled = jnp.concatenate([sin_half, sin_half], axis=-1)  # (3, b, seq, dim)
                cos = self._apply_chunked_mrope(cos_doubled)  # (b, seq, dim)
                sin = self._apply_chunked_mrope(sin_doubled)  # (b, seq, dim)
        else:
            inv_freq = compute_basic_inv_frequencies(self.base, self.rotary_dim)  # (rotary_dim//2,)
            inv_freq = inv_freq[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
            freqs = positions[..., jnp.newaxis].astype(jnp.float32) * inv_freq  # (3, b, seq, dim/2)

            if self.mrope_interleaved:
                # Qwen3-VL style: apply interleaving on half-dim freqs, then double
                freqs_interleaved = self._apply_interleaved_mrope(freqs)  # (b, seq, dim/2)
                emb = jnp.concatenate([freqs_interleaved, freqs_interleaved], axis=-1)  # (b, seq, dim)
            else:
                # Qwen2-VL style: double first, then apply chunked pattern
                emb = jnp.concatenate([freqs, freqs], axis=-1)  # (3, b, seq, dim)
                emb = self._apply_chunked_mrope(emb)  # (b, seq, dim)

            cos = jnp.cos(emb)
            sin = jnp.sin(emb)

        # Apply attention scaling (typically 1.0 for standard mRoPE, can be different for advanced types)
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        # HF mRoPE helpers may compute cos/sin in NeoX-style (duplicated halves) and then
        # convert to GPT-J-style (repeat_interleave) depending on the rotation style.
        #
        # EasyDeL's mRoPE builds cos/sin by duplicating half-dim values. For GPT-J style
        # rotation, we must interleave the half-dim values to match even/odd layout.
        if not self.is_neox_style:
            cos = jnp.repeat(cos[..., : cos.shape[-1] // 2], 2, axis=-1)
            sin = jnp.repeat(sin[..., : sin.shape[-1] // 2], 2, axis=-1)

        cos = cos[:, :, jnp.newaxis, :]
        sin = sin[:, :, jnp.newaxis, :]
        if self.repetition_style:
            rotary_dim = self.rotary_dim
            q_rot = query[..., :rotary_dim]
            k_rot = key[..., :rotary_dim]

            q_pass = None
            k_pass = None
            if rotary_dim < self.head_size:
                q_pass = query[..., rotary_dim:]
                k_pass = key[..., rotary_dim:]

            rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
            q_rot = (q_rot * cos) + (rotate_fn(q_rot) * sin)
            k_rot = (k_rot * cos) + (rotate_fn(k_rot) * sin)

            if q_pass is not None:
                q_rot = jnp.concatenate([q_rot, q_pass], axis=-1)
                k_rot = jnp.concatenate([k_rot, k_pass], axis=-1)

            return q_rot.astype(self.dtype), k_rot.astype(self.dtype)
        else:
            q_embed = (query * cos) + (_rotate_neox(query) * sin)
            k_embed = (key * cos) + (_rotate_neox(key) * sin)
            return q_embed.astype(self.dtype), k_embed.astype(self.dtype)


@rope_wraper("linear")
class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding extended with Linear Scaling.

    Linearly scales the position indices before calculating frequencies.

    Attributes:
        scaling_factors (tp.Union[tp.List[float], float]): The factor(s) to scale positions by.
        Inherits other attributes from RotaryEmbedding.
    """

    def __init__(
        self,
        scaling_factors: list[float] | float,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        """Initialize the LinearScalingRotaryEmbedding module.

        Args:
            scaling_factors: The factor(s) to scale positions by. Can be a single
                float or a list of floats for multiple scaling factors.
            head_size: The dimension size of each attention head.
            rotary_dim: The dimension size of the rotary embeddings applied.
            max_position_embeddings: The base maximum sequence length before scaling.
            base: The base value for calculating inverse frequencies.
            is_neox_style: If True, uses Neox-style rotation. If False, uses GPT-J-style.
            dtype: Data type for the output embeddings.
        """
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
        self.scaling_factors = scaling_factors

    @jax.named_scope("easydel-rope-linear-scaling")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply linearly scaled rotary positional embeddings.

        Computes the frequency cache with linear scaling if not provided,
        then applies the rotary transformation to query and key tensors.

        Args:
            positions: Position indices for each token in the sequence.
            query: Query tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            key: Key tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            offsets: Optional position offsets to add to positions.
                Defaults to None.
            frequencies: Optional pre-computed frequency cache. If None,
                frequencies are computed with linear scaling. Defaults to None.

        Returns:
            A tuple of (rotated_query, rotated_key) tensors.
        """
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_linear_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factors=self.scaling_factors,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("dynamic")
class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding extended with Dynamic NTK scaling.

    Dynamically adjusts the `base` parameter based on the scaling factor.

    Attributes:
        scaling_factor (float): The scaling factor applied to sequence length and base calculation.
        Inherits other attributes from RotaryEmbedding.
    """

    def __init__(
        self,
        scaling_factor: list[float] | float,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        """Initialize the DynamicNTKScalingRotaryEmbedding module.

        Args:
            scaling_factor: The scaling factor applied to sequence length and
                used to dynamically adjust the base parameter.
            head_size: The dimension size of each attention head.
            rotary_dim: The dimension size of the rotary embeddings applied.
            max_position_embeddings: The base maximum sequence length before scaling.
            base: The initial base value before dynamic adjustment.
            is_neox_style: If True, uses Neox-style rotation. If False, uses GPT-J-style.
            dtype: Data type for the output embeddings.
        """
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
        self.scaling_factor = scaling_factor

    @jax.named_scope("easydel-rope-dynamic-ntk-scaling")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply Dynamic NTK scaled rotary positional embeddings.

        Computes the frequency cache with dynamic NTK scaling if not provided,
        then applies the rotary transformation to query and key tensors.
        The base parameter is dynamically adjusted based on the scaling factor.

        Args:
            positions: Position indices for each token in the sequence.
            query: Query tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            key: Key tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            offsets: Optional position offsets to add to positions.
                Defaults to None.
            frequencies: Optional pre-computed frequency cache. If None,
                frequencies are computed with dynamic NTK scaling. Defaults to None.

        Returns:
            A tuple of (rotated_query, rotated_key) tensors.
        """
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_dynamic_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.scaling_factor,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("yarn")
class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding extended with the YaRN (Yet another RoPE extensioN method) scaling.

    Combines interpolation and extrapolation with frequency correction and magnitude scaling.

    Attributes:
        scaling_factor (tp.Union[float, int]): The primary scaling factor for context length.
        extrapolation_factor (float): Controls the strength of extrapolation correction.
        attn_factor (float): Scales the output attention values.
        beta_fast (int): YaRN parameter for high-frequency dimensions correction range.
        beta_slow (int): YaRN parameter for low-frequency dimensions correction range.
        Inherits other attributes from RotaryEmbedding. Note: `max_position_embeddings`
        in the parent init likely refers to the *original* max length for YaRN calculations.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        scaling_factor: float | int = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        """Initialize the YaRNScalingRotaryEmbedding module.

        YaRN (Yet another RoPE extensioN) combines interpolation and extrapolation
        with frequency correction and magnitude scaling for context extension.

        Args:
            head_size: The dimension size of each attention head.
            rotary_dim: The dimension size of the rotary embeddings applied.
            max_position_embeddings: The original maximum sequence length before scaling.
            base: The base value for calculating inverse frequencies.
            is_neox_style: If True, uses Neox-style rotation. If False, uses GPT-J-style.
            dtype: Data type for the output embeddings.
            scaling_factor: The primary scaling factor for context length extension.
                Defaults to 1.0.
            extrapolation_factor: Controls the strength of extrapolation correction.
                Defaults to 1.0.
            attn_factor: Scales the output attention values (applied to cos/sin).
                Defaults to 1.0.
            beta_fast: YaRN parameter for high-frequency dimensions correction range.
                Defaults to 32.
            beta_slow: YaRN parameter for low-frequency dimensions correction range.
                Defaults to 1.
        """
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

    @jax.named_scope("easydel-rope-yarn-scaling")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply YaRN scaled rotary positional embeddings.

        Computes the frequency cache with YaRN scaling if not provided,
        then applies the rotary transformation to query and key tensors.
        YaRN combines interpolation and extrapolation with mscale adjustment.

        Args:
            positions: Position indices for each token in the sequence.
            query: Query tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            key: Key tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            offsets: Optional position offsets to add to positions.
                Defaults to None.
            frequencies: Optional pre-computed frequency cache. If None,
                frequencies are computed with YaRN scaling. Defaults to None.

        Returns:
            A tuple of (rotated_query, rotated_key) tensors.
        """
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_yarn_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.scaling_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                    extrapolation_factor=self.extrapolation_factor,
                    attn_factor=self.attn_factor,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("deepseek_yarn")
class DeepseekScalingRotaryEmbedding(nn.Module):
    """
    RotaryEmbedding implementing a YaRN-like scaling method, potentially from Deepseek models.

    Uses YaRN parameters (`beta_fast`, `beta_slow`, `extrapolation_factor`) and includes
    additional m-scale parameters (`mscale`, `mscale_all_dim`). This version has a custom
    `__call__` method differing slightly from `apply_basic_rope`.

    Attributes:
        head_size (int): Dimension of each attention head.
        rotary_dim (int): Dimension subjected to rotary embedding.
        max_position_embeddings (int): Original maximum sequence length before scaling.
        base (int): Base for frequency calculation.
        is_neox_style (bool): Use Neox rotation if True, GPT-J otherwise.
        dtype (jnp.dtype): Data type for embeddings.
        scaling_factor (float): Primary scaling factor.
        extrapolation_factor (float): YaRN extrapolation factor.
        attn_factor (float): Attention scaling factor.
        beta_fast (int): YaRN parameter.
        beta_slow (int): YaRN parameter.
        mscale (float): Parameter for m-scale calculation.
        mscale_all_dim (float): Parameter for m-scale calculation.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ):
        """Initialize the DeepseekScalingRotaryEmbedding module.

        This implements a YaRN-like scaling method with additional mscale
        parameters as used in Deepseek models.

        Args:
            head_size: The dimension size of each attention head.
            rotary_dim: The dimension size of the rotary embeddings applied.
            max_position_embeddings: The original maximum sequence length before scaling.
            base: The base value for calculating inverse frequencies.
            is_neox_style: If True, uses Neox-style rotation. If False, uses GPT-J-style.
            dtype: Data type for the output embeddings.
            scaling_factor: The primary scaling factor for context length extension.
            extrapolation_factor: Controls the strength of extrapolation correction.
                Defaults to 1.
            attn_factor: Scales the output attention values. Defaults to 1.
            beta_fast: YaRN parameter for high-frequency dimensions. Defaults to 32.
            beta_slow: YaRN parameter for low-frequency dimensions. Defaults to 1.
            mscale: Parameter for mscale calculation in yarn_get_mscale. Defaults to 1.
            mscale_all_dim: Parameter for mscale calculation. Defaults to 0.
        """
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

    @jax.named_scope("easydel-rope-deepseek")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply Deepseek-YaRN scaled rotary positional embeddings.

        Computes the frequency cache with Deepseek-YaRN scaling if not provided,
        then applies the rotary transformation to query and key tensors.

        Args:
            positions: Position indices for each token in the sequence.
            query: Query tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            key: Key tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            offsets: Optional position offsets to add to positions.
                Defaults to None.
            frequencies: Optional pre-computed frequency cache. If None,
                frequencies are computed with Deepseek scaling. Defaults to None.

        Returns:
            A tuple of (rotated_query, rotated_key) tensors.
        """
        if frequencies is None:
            frequencies = compute_deepseek_frequencies(
                self.base,
                self.rotary_dim,
                self.scaling_factor,
                self.extrapolation_factor,
                self.beta_fast,
                self.beta_slow,
                self.max_position_embeddings,
                self.mscale,
                self.mscale_all_dim,
                self.attn_factor,
            )
        cos, sin = jnp.split(frequencies[positions], 2, -1)
        if offsets is not None:
            positions += offsets
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]

        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        target_sc_shape = (query.shape[0], -1, 1, self.rotary_dim)
        if self.is_neox_style:
            cos = cos.repeat(2, axis=1).reshape(target_sc_shape)
            sin = sin.repeat(2, axis=1).reshape(target_sc_shape)
        else:
            cos = cos.repeat_interleave(2, axis=1).reshape(target_sc_shape)
            sin = sin.repeat_interleave(2, axis=1).reshape(target_sc_shape)
        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = jnp.concatenate((query_rot, query_pass), axis=-1)
            key = jnp.concatenate((key_rot, key_pass), axis=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key


@rope_wraper("longrope")
class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """
    RotaryEmbedding using the Phi-3 LongRoPE scaling method.

    Applies different frequency scaling factors (`short_factor`, `long_factor`)
    depending on the target sequence length relative to the original maximum.
    Requires `rotary_dim` to be equal to `head_size`.

    Attributes:
        head_size (int): Dimension of each attention head. Must equal rotary_dim.
        rotary_dim (int): Dimension subjected to rotary embedding. Must equal head_size.
        max_position_embeddings (int): The target maximum sequence length after scaling.
        original_max_position_embeddings (int): Original maximum sequence length before scaling.
        base (int): Base for frequency calculation.
        is_neox_style (bool): Flag indicating whether Neox-style rotation is assumed (used by `apply_phi3_rope`).
        dtype (jnp.dtype): Data type for computations.
        short_factor (tp.List[float]): Scaling factors applied when target length <= original max length.
        long_factor (tp.List[float]): Scaling factors applied when target length > original max length.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        short_factor: list[float],
        long_factor: list[float],
    ):
        """Initialize the Phi3LongRoPEScaledRotaryEmbedding module.

        Phi-3 LongRoPE applies different scaling factors based on whether the
        target sequence length exceeds the original maximum length.

        Args:
            head_size: The dimension size of each attention head. Must equal rotary_dim.
            rotary_dim: The dimension size of the rotary embeddings. Must equal head_size.
            max_position_embeddings: The target maximum sequence length after scaling.
            original_max_position_embeddings: The original maximum sequence length
                before scaling, used to determine which factor to apply.
            base: The base value for calculating inverse frequencies.
            is_neox_style: If True, uses Neox-style rotation (expected by apply_phi3_rope).
            dtype: Data type for the output embeddings.
            short_factor: List of scaling factors for each frequency dimension,
                applied when max_position_embeddings <= original_max_position_embeddings.
            long_factor: List of scaling factors for each frequency dimension,
                applied when max_position_embeddings > original_max_position_embeddings.
        """
        super().__init__()

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.short_factor = short_factor
        self.long_factor = long_factor

    @jax.named_scope("easydel-rope-phi3-long")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply Phi-3 LongRoPE scaled rotary positional embeddings.

        Computes the frequency cache with Phi-3 LongRoPE scaling if not provided,
        then applies the rotary transformation to query and key tensors.

        Args:
            positions: Position indices for each token in the sequence.
            query: Query tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            key: Key tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            offsets: Optional position offsets to add to positions.
                Defaults to None.
            frequencies: Optional pre-computed frequency cache. If None,
                frequencies are computed with Phi-3 LongRoPE scaling. Defaults to None.

        Returns:
            A tuple of (rotated_query, rotated_key) tensors.
        """
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_phi3_frequencies(
                    base=self.base,
                    head_size=self.head_size,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    original_max_position_embeddings=self.original_max_position_embeddings,
                    short_factor=self.short_factor,
                    long_factor=self.long_factor,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value
            return apply_phi3_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                offsets=offsets,
                dtype=self.dtype,
            )


@rope_wraper("llama3")
class Llama3RotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding implementing the Llama-3 scaling method.

    Adjusts frequencies based on wavelength thresholds (`low_freq_factor`, `high_freq_factor`)
    and applies an overall scaling factor.

    Attributes:
        scaling_factor (float): Overall scaling factor.
        low_freq_factor (float): Factor related to low frequency wavelength threshold.
        high_freq_factor (float): Factor related to high frequency wavelength threshold.
        orig_max_position (int): Original maximum sequence length before scaling.
        Inherits other attributes from RotaryEmbedding.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ):
        """Initialize the Llama3RotaryEmbedding module.

        Llama-3 style RoPE adjusts frequencies based on wavelength thresholds
        determined by low_freq_factor and high_freq_factor.

        Args:
            head_size: The dimension size of each attention head.
            rotary_dim: The dimension size of the rotary embeddings applied.
            max_position_embeddings: The target maximum sequence length.
            base: The base value for calculating inverse frequencies.
            is_neox_style: If True, uses Neox-style rotation. If False, uses GPT-J-style.
            dtype: Data type for the output embeddings.
            scaling_factor: The overall scaling factor applied to frequencies.
            low_freq_factor: Factor used to compute the low frequency wavelength
                threshold (orig_max_position / low_freq_factor).
            high_freq_factor: Factor used to compute the high frequency wavelength
                threshold (orig_max_position / high_freq_factor).
            orig_max_position: The original maximum sequence length before scaling,
                used to compute wavelength thresholds.
        """
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position

    @jax.named_scope("easydel-rope-llama3")
    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
        frequencies: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply Llama-3 scaled rotary positional embeddings.

        Computes the frequency cache with Llama-3 wavelength-based scaling if not
        provided, then applies the rotary transformation to query and key tensors.

        Args:
            positions: Position indices for each token in the sequence.
            query: Query tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            key: Key tensor from attention computation.
                Shape: [batch_size, sequence_length, num_heads, head_dim].
            offsets: Optional position offsets to add to positions.
                Defaults to None.
            frequencies: Optional pre-computed frequency cache. If None,
                frequencies are computed with Llama-3 scaling. Defaults to None.

        Returns:
            A tuple of (rotated_query, rotated_key) tensors.
        """
        with jax.ensure_compile_time_eval():
            if frequencies is None:
                frequencies = compute_llama3_frequencies(
                    base=self.base,
                    rotary_dim=self.rotary_dim,
                    low_freq_factor=self.low_freq_factor,
                    high_freq_factor=self.high_freq_factor,
                    scaling_factor=self.scaling_factor,
                    max_position_embeddings=self.orig_max_position,
                )
            if hasattr(frequencies, "value"):
                frequencies = frequencies.value

            return apply_basic_rope(
                query=query,
                key=key,
                positions=positions,
                frequencies=frequencies,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
                offsets=offsets,
                dtype=self.dtype,
            )
