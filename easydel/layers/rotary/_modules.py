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

"""Neural network modules for Rotary Position Embeddings (RoPE).

This module provides spectrax Module classes that implement various RoPE
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

Module-level variables:
    AVAILABLE_ROPE_TYPES: Registry mapping rope-type identifiers (strings)
        to their concrete :class:`spx.Module` classes. Populated by the
        ``rope_wrapper`` decorator at import time so that
        :func:`get_rope` / :func:`get_frequencies` can dispatch by name.

The `rope_wrapper` decorator registers each class in `AVAILABLE_ROPE_TYPES`
for dynamic lookup by rope type name.

Example:
    >>> import jax.numpy as jnp
    >>> from easydel.layers.rotary import RotaryEmbedding
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

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx

from easydel.layers.norms import lowfloats

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


_T = tp.TypeVar("_T")


def _promote_rotary_operands(*operands: jax.Array) -> tuple[jax.Array, ...]:
    """Promote rotary operands to a common dtype, escalating to float32 for low-precision.

    Mixed-precision (fp8/fp4) cos/sin or query/key tensors lose accuracy if
    the rotary multiply runs in their native dtype, so this helper bumps the
    computation up to ``float32`` whenever any operand is in
    :data:`lowfloats`. Otherwise it picks the standard NumPy promoted dtype.

    Args:
        *operands: Two or more JAX arrays to be cast to a shared dtype.

    Returns:
        Tuple of arrays cast to the chosen common dtype, in the original
        argument order.
    """
    if any(operand.dtype in lowfloats for operand in operands):
        compute_dtype = jnp.float32
    else:
        compute_dtype = operands[0].dtype
        for operand in operands[1:]:
            compute_dtype = jnp.promote_types(compute_dtype, operand.dtype)
    return tuple(operand.astype(compute_dtype) for operand in operands)


def rope_wrapper(type: str) -> tp.Callable[[_T], _T]:  # noqa
    """Build a decorator that registers a rotary class under ``type`` in :data:`AVAILABLE_ROPE_TYPES`.

    The decorator captures the class's ``__dict__`` so callers can introspect
    the registered RoPE without instantiating it, sets simple ``__str__`` /
    ``__repr__`` returning the class name, and stashes the chosen ``type``
    name on ``cls._type`` for round-tripping. Used to populate the
    name-to-class table consumed by :func:`get_rope` / :func:`get_frequencies`.

    Args:
        type: Registry key (e.g. ``"linear"``, ``"yarn"``, ``"llama3"``).

    Returns:
        Decorator that takes the rotary class, registers it, and returns
        it unchanged (apart from the dunders / ``_type`` patch).
    """

    def w(rope: _T) -> _T:
        """Apply the registration side-effect to ``rope`` and return it.

        Args:
            rope: The rotary class being registered.

        Returns:
            The same class object, now reachable via
            ``AVAILABLE_ROPE_TYPES[type]``.
        """
        properties = {k: v for k, v in rope.__dict__.items()}
        AVAILABLE_ROPE_TYPES[type] = properties
        rope.__str__ = lambda cls: str(cls.__class__.__name__)
        rope.__repr__ = lambda cls: repr(cls.__class__.__name__)
        rope._type = type
        return rope

    return w


@rope_wrapper("default")
class RotaryEmbedding(spx.Module):
    """Vanilla (un-scaled) Rotary Position Embedding module.

    Base class registered under ``"default"`` in :data:`AVAILABLE_ROPE_TYPES`;
    all other RoPE variants in this module extend it. The class owns just the
    geometric parameters (``base`` / ``rotary_dim`` / ``max_position_embeddings``)
    and delegates frequency computation to :func:`compute_basic_frequencies`
    and rotation application to :func:`apply_basic_rope`. Subclasses override
    one or both halves.

    Attributes:
        head_size (int): Per-head attention dimension.
        rotary_dim (int): Number of channels rotated; remaining ``head_size -
            rotary_dim`` channels (NoPE tail) pass through untouched.
        max_position_embeddings (int): Length of the precomputed cos/sin cache.
        base (int): Geometric-progression base ``θ`` for the unscaled spectrum.
        is_neox_style (bool): ``True`` for Neox interleaving, ``False`` for the
            GPT-J pairing.
        dtype (jnp.dtype): Output dtype of the rotated query/key.
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
    def forward(
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


@rope_wrapper("mrope")
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
    def forward(
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

        # mRoPE can be partial (e.g. rotary_dim=64 on head_size=256). Align rotary
        # dimensions with Q/K tensors and keep the non-rotary tail as pass-through.
        rotary_dim = min(int(cos.shape[-1]), int(query.shape[-1]), int(key.shape[-1]))
        cos = cos[..., :rotary_dim]
        sin = sin[..., :rotary_dim]
        if self.repetition_style:
            q_rot = query[..., :rotary_dim]
            k_rot = key[..., :rotary_dim]
            q_rot, k_rot, cos, sin = _promote_rotary_operands(q_rot, k_rot, cos, sin)

            q_pass = None
            k_pass = None
            if rotary_dim < query.shape[-1]:
                q_pass = query[..., rotary_dim:]
                k_pass = key[..., rotary_dim:]

            rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
            q_rot = (q_rot * cos) + (rotate_fn(q_rot) * sin)
            k_rot = (k_rot * cos) + (rotate_fn(k_rot) * sin)

            if q_pass is not None:
                q_rot = jnp.concatenate([q_rot, q_pass.astype(q_rot.dtype)], axis=-1)
                k_rot = jnp.concatenate([k_rot, k_pass.astype(k_rot.dtype)], axis=-1)

            return q_rot.astype(self.dtype), k_rot.astype(self.dtype)
        else:
            q_rot = query[..., :rotary_dim]
            k_rot = key[..., :rotary_dim]
            q_rot, k_rot, cos, sin = _promote_rotary_operands(q_rot, k_rot, cos, sin)

            q_embed = (q_rot * cos) + (_rotate_neox(q_rot) * sin)
            k_embed = (k_rot * cos) + (_rotate_neox(k_rot) * sin)

            if rotary_dim < query.shape[-1]:
                q_embed = jnp.concatenate([q_embed, query[..., rotary_dim:].astype(q_embed.dtype)], axis=-1)
                k_embed = jnp.concatenate([k_embed, key[..., rotary_dim:].astype(k_embed.dtype)], axis=-1)

            return q_embed.astype(self.dtype), k_embed.astype(self.dtype)


@rope_wrapper("linear")
class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RoPE variant using Position Interpolation (linear position scaling).

    Implements the Chen et al. 2023 Position-Interpolation context extension:
    positions are divided by the scaling factor so a model trained at
    ``max_position_embeddings`` keeps the same rotation angles when evaluated
    at ``max_position_embeddings * scaling_factor`` tokens. Frequency
    construction is delegated to :func:`compute_linear_frequencies`.

    Attributes:
        scaling_factors (list[float] | float): Single factor (the usual
            case) or a list of factors used to build a concatenated cache
            with multiple scaling regimes.

    Plus all attributes inherited from :class:`RotaryEmbedding`.
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
    def forward(
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


@rope_wrapper("dynamic")
class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RoPE variant using Dynamic NTK-aware scaling.

    Adjusts the ``base`` (frequency progression base) rather than the
    positions, so high-frequency dimensions are perturbed less than
    low-frequency ones. See :func:`compute_dynamic_frequencies` for the
    closed-form adjustment.

    Attributes:
        scaling_factor (float | list[float]): Target context-length
            multiplier. Lists are accepted for API parity but the dynamic
            variant uses a single scalar.

    Plus all attributes inherited from :class:`RotaryEmbedding`.
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
    def forward(
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


@rope_wrapper("yarn")
class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """RoPE variant using YaRN (Yet another RoPE extensioN) scaling.

    Blends Position Interpolation (slow dims) with raw extrapolation (fast
    dims) via a smooth ramp on the rotation-count axis, and applies a
    log-derived ``mscale`` correction to the cos/sin magnitudes so attention
    score variance stays roughly constant. Defers frequency construction to
    :func:`compute_yarn_frequencies`.

    Attributes:
        scaling_factor (float | int): Target context-length multiplier.
        extrapolation_factor (float): Scalar in ``[0, 1]`` weighting the
            extrapolation branch (``1.0`` is full YaRN).
        attn_factor (float): User-tunable multiplier composed with the YaRN
            mscale and applied to cos/sin.
        beta_fast (int): Fast-band boundary in rotations-per-original-context
            (typical 32).
        beta_slow (int): Slow-band boundary (typical 1).

    Plus all attributes inherited from :class:`RotaryEmbedding`.
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
    def forward(
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


@rope_wrapper("deepseek_yarn")
class DeepseekScalingRotaryEmbedding(spx.Module):
    """RoPE variant for DeepSeek models: YaRN with a two-mscale rescaling.

    Same interpolation/extrapolation blend as YaRN, but the attention-magnitude
    correction is a *ratio* of two ``yarn_get_mscale`` evaluations
    (``mscale`` numerator, ``mscale_all_dim`` denominator), as used by the
    DeepSeek-V2/V3 series. Unlike :class:`YaRNScalingRotaryEmbedding`, this
    class implements its own ``forward`` rather than delegating to
    :func:`apply_basic_rope` because DeepSeek's rotation uses a ``repeat``
    layout for cos/sin that differs from the canonical concat layout.

    Note that this class inherits directly from :class:`spx.Module` (not
    :class:`RotaryEmbedding`) and therefore re-declares the geometric fields.

    Attributes:
        head_size (int): Per-head attention dimension.
        rotary_dim (int): Rotary feature dimension.
        max_position_embeddings (int): Original (pre-extension) context.
        base (int): Geometric-progression base ``θ``.
        is_neox_style (bool): Neox vs GPT-J rotation layout.
        dtype (jnp.dtype): Output dtype.
        scaling_factor (float): Target context-length multiplier.
        extrapolation_factor (float): YaRN extrapolation mix factor.
        attn_factor (float): User-tunable multiplier composed with mscale.
        beta_fast (int): YaRN fast-band boundary.
        beta_slow (int): YaRN slow-band boundary.
        mscale (float): Per-dim mscale exponent.
        mscale_all_dim (float): All-dim mscale exponent (denominator).
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
    def forward(
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


@rope_wrapper("longrope")
class Phi3LongRoPEScaledRotaryEmbedding(spx.Module):
    """RoPE variant implementing Phi-3 LongRoPE per-dimension scaling.

    Phi-3 ships two learned per-frequency scaling vectors: ``short_factor``
    (used when the runtime context fits in the original training window) and
    ``long_factor`` (used when extending). The selected vector multiplies the
    raw ``base**(2i/d)`` denominators before the cos/sin cache is built, and
    a ``sqrt(1 + log(scale)/log(orig_max))`` magnitude rescale is applied
    afterwards. Implementation differs from the YaRN family in that there is
    no smooth ramp — the selection is binary on context length. Requires
    ``rotary_dim == head_size``.

    Attributes:
        head_size (int): Per-head attention dimension; must equal ``rotary_dim``.
        rotary_dim (int): Rotary feature dimension.
        max_position_embeddings (int): Post-scaling target context length.
        original_max_position_embeddings (int): Original training context.
        base (int): Geometric-progression base ``θ``.
        is_neox_style (bool): Forwarded into :func:`apply_phi3_rope`; Phi-3
            assumes the Neox interleaving.
        dtype (jnp.dtype): Output dtype.
        short_factor (list[float]): Per-pair scalar applied when the runtime
            context does not exceed ``original_max_position_embeddings``.
        long_factor (list[float]): Per-pair scalar applied when the runtime
            context exceeds ``original_max_position_embeddings``.
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
    def forward(
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


@rope_wrapper("llama3")
class Llama3RotaryEmbedding(RotaryEmbedding):
    """RoPE variant implementing Llama-3's wavelength-piecewise scaling.

    Llama-3 partitions inverse-frequencies by their wavelength
    ``λ = 2π / θ_i`` relative to the original context and applies
    Position-Interpolation only to the long-wavelength dims while leaving
    short-wavelength dims alone, with a linear blend in between. See
    :func:`compute_llama3_frequencies`.

    Attributes:
        scaling_factor (float): Target context-length multiplier (8 for the
            Llama-3 8K→64K extension).
        low_freq_factor (float): Low-frequency boundary parameter (1 in
            Llama-3).
        high_freq_factor (float): High-frequency boundary parameter (4 in
            Llama-3).
        orig_max_position (int): Original training context length (8192).

    Plus all attributes inherited from :class:`RotaryEmbedding`.
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
    def forward(
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
