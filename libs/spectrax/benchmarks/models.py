# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parallel model builders for spectrax and flax.nnx.

Each builder returns a tuple ``(module, example_input)``. The two
libraries build comparable architectures using their native layer
APIs — the goal is equivalent shapes, not byte-identical HLO.
Top-level class definitions are required so spectrax's registry can
resolve them by qualified name during ``bind``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

import spectrax as spx
import spectrax.nn as spx_nn


class SpxMLP(spx.Module):
    """Flat MLP — children stored on ``self`` under ``l0..lN-1``."""

    def __init__(self, depth: int = 12, hidden: int = 1024, in_dim: int = 1024, seed: int = 0):
        """Create ``depth`` GELU-activated Linear layers.

        Args:
            depth: Number of layers.
            hidden: Hidden feature count.
            in_dim: Input feature count.
            seed: PRNG seed for parameter init.
        """
        super().__init__()
        rngs = spx.Rngs(seed)
        self.depth = depth
        self.l0 = spx_nn.Linear(in_dim, hidden, rngs=rngs)
        for i in range(1, depth):
            setattr(self, f"l{i}", spx_nn.Linear(hidden, hidden, rngs=rngs))

    def forward(self, x):
        """Run every layer with a GELU nonlinearity."""
        for i in range(self.depth):
            x = jax.nn.gelu(getattr(self, f"l{i}")(x))
        return x


def spx_mlp(depth: int = 12, hidden: int = 1024, in_dim: int = 1024, seed: int = 0):
    """Build a spectrax MLP and a matching input tensor.

    Args:
        depth: Number of layers.
        hidden: Hidden feature count.
        in_dim: Input feature count.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = SpxMLP(depth=depth, hidden=hidden, in_dim=in_dim, seed=seed)
    return mdl, jnp.ones((32, in_dim), dtype=jnp.float32)


class SpxTransformerBlock(spx.Module):
    """Single transformer block for spectrax: pre-LN attention + GELU MLP residual."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, ffn: int = 2048, seed: int = 0, dtype=None):
        """Initialize LayerNorms, attention projections, and FFN layers.

        Args:
            d_model: Hidden dimension.
            n_heads: Number of attention heads.
            ffn: FFN hidden dimension.
            seed: PRNG seed for parameter init.
            dtype: Optional dtype for parameters.
        """
        super().__init__()
        rngs = spx.Rngs(seed)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.ffn = ffn
        self.ln1 = spx_nn.LayerNorm(d_model, dtype=dtype)
        self.q = spx_nn.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.k = spx_nn.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.v = spx_nn.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.o = spx_nn.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.ln2 = spx_nn.LayerNorm(d_model, dtype=dtype)
        self.fc1 = spx_nn.Linear(d_model, ffn, rngs=rngs, dtype=dtype)
        self.fc2 = spx_nn.Linear(ffn, d_model, rngs=rngs, dtype=dtype)

    def forward(self, x):
        """Run the residual attention + MLP block."""
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.q(h).reshape(b, t, self.n_heads, self.d_head)
        k = self.k(h).reshape(b, t, self.n_heads, self.d_head)
        v = self.v(h).reshape(b, t, self.n_heads, self.d_head)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(self.d_head).astype(x.dtype)
        attn = jax.nn.softmax(scores, axis=-1)
        a = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(b, t, d)
        x = x + self.o(a)
        h2 = self.ln2(x)
        return x + self.fc2(jax.nn.gelu(self.fc1(h2)))


def spx_transformer(
    d_model: int = 512, n_heads: int = 8, ffn: int = 2048, seq_len: int = 128, batch: int = 8, seed: int = 0
):
    """Build a spectrax transformer block and a matching input tensor.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        ffn: FFN hidden dimension.
        seq_len: Sequence length of the example input.
        batch: Batch size of the example input.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = SpxTransformerBlock(d_model, n_heads, ffn, seed=seed)
    x = jnp.ones((batch, seq_len, d_model), dtype=jnp.float32)
    return mdl, x


class SpxTransformerStack(spx.Module):
    """Stacked transformer blocks for a ~1B-param benchmark model."""

    def __init__(
        self, n_layers: int = 24, d_model: int = 2048, n_heads: int = 16, ffn: int = 8192, seed: int = 0, dtype=None
    ):
        """Initialize ``n_layers`` :class:`SpxTransformerBlock` instances.

        Args:
            n_layers: Number of blocks to stack.
            d_model: Hidden dimension per block.
            n_heads: Attention head count per block.
            ffn: FFN width per block.
            seed: Base PRNG seed; incremented per block.
            dtype: Optional dtype override for parameters.
        """
        super().__init__()
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, f"blk{i}", SpxTransformerBlock(d_model, n_heads, ffn, seed=seed + i, dtype=dtype))

    def forward(self, x):
        """Run every block sequentially and return the final activation."""
        for i in range(self.n_layers):
            x = getattr(self, f"blk{i}")(x)
        return x


def spx_transformer_1b(
    n_layers: int = 24,
    d_model: int = 2048,
    n_heads: int = 16,
    ffn: int = 8192,
    seq_len: int = 512,
    batch: int = 4,
    seed: int = 0,
):
    """Build a ~1B-param spectrax transformer and a matching bf16 input tensor.

    Args:
        n_layers: Number of stacked blocks.
        d_model: Hidden dimension.
        n_heads: Attention head count.
        ffn: FFN width.
        seq_len: Sequence length of the example input.
        batch: Batch size of the example input.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = SpxTransformerStack(n_layers, d_model, n_heads, ffn, seed=seed, dtype=jnp.bfloat16)
    x = jnp.ones((batch, seq_len, d_model), dtype=jnp.bfloat16)
    return mdl, x


class SpxFp8TransformerBlock(spx.Module):
    """Transformer block where every Linear is :class:`Fp8Linear`."""

    def __init__(self, d_model: int, n_heads: int, ffn: int, seed: int = 0, dtype=None):
        """Initialize FP8 attention and FFN sublayers.

        Args:
            d_model: Hidden dimension.
            n_heads: Number of attention heads.
            ffn: FFN hidden dimension.
            seed: PRNG seed for parameter init.
            dtype: Optional dtype override for parameters.
        """
        super().__init__()
        rngs = spx.Rngs(seed)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.ffn = ffn
        self.ln1 = spx_nn.LayerNorm(d_model, dtype=dtype)
        self.q = spx_nn.Fp8Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.k = spx_nn.Fp8Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.v = spx_nn.Fp8Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.o = spx_nn.Fp8Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.ln2 = spx_nn.LayerNorm(d_model, dtype=dtype)
        self.fc1 = spx_nn.Fp8Linear(d_model, ffn, rngs=rngs, dtype=dtype)
        self.fc2 = spx_nn.Fp8Linear(ffn, d_model, rngs=rngs, dtype=dtype)

    def forward(self, x):
        """Run the residual FP8 attention + MLP block.

        Args:
            x: Input tensor of shape ``(batch, seq, d_model)``.

        Returns:
            Output tensor of shape ``(batch, seq, d_model)``.
        """
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.q(h).reshape(b, t, self.n_heads, self.d_head)
        k = self.k(h).reshape(b, t, self.n_heads, self.d_head)
        v = self.v(h).reshape(b, t, self.n_heads, self.d_head)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(self.d_head).astype(x.dtype)
        attn = jax.nn.softmax(scores, axis=-1)
        a = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(b, t, d)
        x = x + self.o(a)
        h2 = self.ln2(x)
        return x + self.fc2(jax.nn.gelu(self.fc1(h2)))


class SpxFp8TransformerStack(spx.Module):
    """Stacked FP8 transformer blocks for benchmarking."""

    def __init__(
        self, n_layers: int = 24, d_model: int = 2048, n_heads: int = 16, ffn: int = 8192, seed: int = 0, dtype=None
    ):
        """Initialize ``n_layers`` :class:`SpxFp8TransformerBlock` instances.

        Args:
            n_layers: Number of blocks to stack.
            d_model: Hidden dimension per block.
            n_heads: Attention head count per block.
            ffn: FFN width per block.
            seed: Base PRNG seed; incremented per block.
            dtype: Optional dtype override for parameters.
        """
        super().__init__()
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, f"blk{i}", SpxFp8TransformerBlock(d_model, n_heads, ffn, seed=seed + i, dtype=dtype))

    def forward(self, x):
        """Run every FP8 block sequentially and return the final activation."""
        for i in range(self.n_layers):
            x = getattr(self, f"blk{i}")(x)
        return x


def spx_fp8_transformer_1b(
    n_layers: int = 24,
    d_model: int = 2048,
    n_heads: int = 16,
    ffn: int = 8192,
    seq_len: int = 512,
    batch: int = 4,
    seed: int = 0,
):
    """Build a ~1B-param FP8 spectrax transformer and a matching bf16 input.

    Args:
        n_layers: Number of stacked blocks.
        d_model: Hidden dimension.
        n_heads: Attention head count.
        ffn: FFN width.
        seq_len: Sequence length of the example input.
        batch: Batch size of the example input.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = SpxFp8TransformerStack(n_layers, d_model, n_heads, ffn, seed=seed, dtype=jnp.bfloat16)
    x = jnp.ones((batch, seq_len, d_model), dtype=jnp.bfloat16)
    return mdl, x


class SpxConvStack(spx.Module):
    """Tiny ConvNet for spectrax benchmarking: three conv layers + classifier."""

    def __init__(self, seed: int = 0):
        """Create three Conv2d layers and a final Linear classifier.

        Args:
            seed: PRNG seed for parameter init.
        """
        super().__init__()
        rngs = spx.Rngs(seed)
        self.c1 = spx_nn.Conv2d(3, 32, kernel_size=3, padding="SAME", rngs=rngs)
        self.c2 = spx_nn.Conv2d(32, 64, kernel_size=3, padding="SAME", rngs=rngs)
        self.c3 = spx_nn.Conv2d(64, 64, kernel_size=3, padding="SAME", rngs=rngs)
        self.ln = spx_nn.Linear(64 * 8 * 8, 10, rngs=rngs)

    def forward(self, x):
        """Three ReLU conv layers then a flatten + linear classifier."""
        x = jax.nn.relu(self.c1(x))
        x = jax.nn.relu(self.c2(x))
        x = jax.nn.relu(self.c3(x))
        x = x.reshape(x.shape[0], -1)
        return self.ln(x)


def spx_conv(seed: int = 0):
    """Build a spectrax ConvNet and a matching input tensor.

    Args:
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = SpxConvStack(seed=seed)
    x = jnp.ones((8, 8, 8, 3), dtype=jnp.float32)
    return mdl, x


class SpxDropoutMLP(spx.Module):
    """Dropout-heavy MLP for RNG-split-merge benchmarking."""

    def __init__(self, depth: int = 6, hidden: int = 512, in_dim: int = 512, seed: int = 0):
        """Create an input projection, ``depth`` dropout-linear pairs, and an output head.

        Args:
            depth: Number of hidden layers.
            hidden: Hidden feature count.
            in_dim: Input feature count.
            seed: PRNG seed for parameter init.
        """
        super().__init__()
        rngs = spx.Rngs(seed)
        self.fc_in = spx_nn.Linear(in_dim, hidden, rngs=rngs)
        self.depth = depth
        for i in range(depth):
            setattr(self, f"l{i}", spx_nn.Linear(hidden, hidden, rngs=rngs))
            setattr(self, f"d{i}", spx_nn.Dropout(0.1))
        self.fc_out = spx_nn.Linear(hidden, 10, rngs=rngs)

    def forward(self, x, rngs: spx.Rngs | None = None):
        """Run the dropout MLP forward, optionally using ``rngs`` for stochastic layers.

        Args:
            x: Input tensor.
            rngs: Optional PRNG key container for dropout sampling.

        Returns:
            Output logits tensor.
        """
        x = jax.nn.gelu(self.fc_in(x))
        for i in range(self.depth):
            x = getattr(self, f"l{i}")(x)
            x = getattr(self, f"d{i}")(x, rngs=rngs)
            x = jax.nn.gelu(x)
        return self.fc_out(x)


def spx_dropout_mlp(depth: int = 6, hidden: int = 512, in_dim: int = 512, seed: int = 0):
    """Build a spectrax dropout MLP and a matching input tensor.

    Args:
        depth: Number of hidden layers.
        hidden: Hidden feature count.
        in_dim: Input feature count.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = SpxDropoutMLP(depth=depth, hidden=hidden, in_dim=in_dim, seed=seed)
    x = jnp.ones((32, in_dim), dtype=jnp.float32)
    return mdl, x


class NnxMLP(nnx.Module):
    """Flat MLP in Flax NNX for benchmarking parity with :class:`SpxMLP`."""

    def __init__(self, depth: int = 12, hidden: int = 1024, in_dim: int = 1024, seed: int = 0):
        """Create ``depth`` GELU-activated Linear layers.

        Args:
            depth: Number of layers.
            hidden: Hidden feature count.
            in_dim: Input feature count.
            seed: PRNG seed for parameter init.
        """
        rngs = nnx.Rngs(seed)
        layers = [nnx.Linear(in_dim, hidden, rngs=rngs)]
        for _ in range(depth - 1):
            layers.append(nnx.Linear(hidden, hidden, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, x):
        """Run every layer with a GELU nonlinearity."""
        for layer in self.layers:
            x = jax.nn.gelu(layer(x))
        return x


def nnx_mlp(depth: int = 12, hidden: int = 1024, in_dim: int = 1024, seed: int = 0):
    """Build a Flax NNX MLP and a matching input tensor.

    Args:
        depth: Number of layers.
        hidden: Hidden feature count.
        in_dim: Input feature count.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    return NnxMLP(depth=depth, hidden=hidden, in_dim=in_dim, seed=seed), jnp.ones((32, in_dim), dtype=jnp.float32)


class NnxTransformerBlock(nnx.Module):
    """Single transformer block in Flax NNX: pre-LN attention + GELU MLP residual."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, ffn: int = 2048, seed: int = 0, param_dtype=None):
        """Initialize LayerNorms, attention projections, and FFN layers.

        Args:
            d_model: Hidden dimension.
            n_heads: Number of attention heads.
            ffn: FFN hidden dimension.
            seed: PRNG seed for parameter init.
            param_dtype: Optional dtype for parameters.
        """
        rngs = nnx.Rngs(seed)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.ffn = ffn
        kw = {} if param_dtype is None else {"param_dtype": param_dtype}
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs, **kw)
        self.q = nnx.Linear(d_model, d_model, rngs=rngs, **kw)
        self.k = nnx.Linear(d_model, d_model, rngs=rngs, **kw)
        self.v = nnx.Linear(d_model, d_model, rngs=rngs, **kw)
        self.o = nnx.Linear(d_model, d_model, rngs=rngs, **kw)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs, **kw)
        self.fc1 = nnx.Linear(d_model, ffn, rngs=rngs, **kw)
        self.fc2 = nnx.Linear(ffn, d_model, rngs=rngs, **kw)

    def __call__(self, x):
        """Run the residual attention + MLP block."""
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.q(h).reshape(b, t, self.n_heads, self.d_head)
        k = self.k(h).reshape(b, t, self.n_heads, self.d_head)
        v = self.v(h).reshape(b, t, self.n_heads, self.d_head)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(self.d_head).astype(x.dtype)
        attn = jax.nn.softmax(scores, axis=-1)
        a = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(b, t, d)
        x = x + self.o(a)
        h2 = self.ln2(x)
        return x + self.fc2(jax.nn.gelu(self.fc1(h2)))


def nnx_transformer(
    d_model: int = 512, n_heads: int = 8, ffn: int = 2048, seq_len: int = 128, batch: int = 8, seed: int = 0
):
    """Build a Flax NNX transformer block and a matching input tensor.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        ffn: FFN hidden dimension.
        seq_len: Sequence length of the example input.
        batch: Batch size of the example input.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = NnxTransformerBlock(d_model, n_heads, ffn, seed=seed)
    x = jnp.ones((batch, seq_len, d_model), dtype=jnp.float32)
    return mdl, x


class NnxTransformerStack(nnx.Module):
    """Stacked transformer blocks in Flax NNX for ~1B-param benchmarking."""

    def __init__(
        self,
        n_layers: int = 24,
        d_model: int = 2048,
        n_heads: int = 16,
        ffn: int = 8192,
        seed: int = 0,
        param_dtype=None,
    ):
        """Initialize ``n_layers`` :class:`NnxTransformerBlock` instances.

        Args:
            n_layers: Number of blocks to stack.
            d_model: Hidden dimension per block.
            n_heads: Attention head count per block.
            ffn: FFN width per block.
            seed: Base PRNG seed; incremented per block.
            param_dtype: Optional dtype for parameters.
        """
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, f"blk{i}", NnxTransformerBlock(d_model, n_heads, ffn, seed=seed + i, param_dtype=param_dtype))

    def __call__(self, x):
        """Run every block sequentially and return the final activation."""
        for i in range(self.n_layers):
            x = getattr(self, f"blk{i}")(x)
        return x


def nnx_transformer_1b(
    n_layers: int = 24,
    d_model: int = 2048,
    n_heads: int = 16,
    ffn: int = 8192,
    seq_len: int = 512,
    batch: int = 4,
    seed: int = 0,
):
    """Build a ~1B-param Flax NNX transformer and a matching bf16 input tensor.

    Args:
        n_layers: Number of stacked blocks.
        d_model: Hidden dimension.
        n_heads: Attention head count.
        ffn: FFN width.
        seq_len: Sequence length of the example input.
        batch: Batch size of the example input.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = NnxTransformerStack(n_layers, d_model, n_heads, ffn, seed=seed, param_dtype=jnp.bfloat16)
    x = jnp.ones((batch, seq_len, d_model), dtype=jnp.bfloat16)
    return mdl, x


class NnxConvStack(nnx.Module):
    """Tiny ConvNet in Flax NNX for benchmarking parity with :class:`SpxConvStack`."""

    def __init__(self, seed: int = 0):
        """Create three Conv layers and a final Linear classifier.

        Args:
            seed: PRNG seed for parameter init.
        """
        rngs = nnx.Rngs(seed)
        self.c1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(64, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln = nnx.Linear(64 * 8 * 8, 10, rngs=rngs)

    def __call__(self, x):
        """Three ReLU conv layers then a flatten + linear classifier."""
        x = jax.nn.relu(self.c1(x))
        x = jax.nn.relu(self.c2(x))
        x = jax.nn.relu(self.c3(x))
        x = x.reshape(x.shape[0], -1)
        return self.ln(x)


def nnx_conv(seed: int = 0):
    """Build a Flax NNX ConvNet and a matching input tensor.

    Args:
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    return NnxConvStack(seed=seed), jnp.ones((8, 8, 8, 3), dtype=jnp.float32)


class NnxDropoutMLP(nnx.Module):
    """Dropout-heavy MLP in Flax NNX for RNG benchmarking parity."""

    def __init__(self, depth: int = 6, hidden: int = 512, in_dim: int = 512, seed: int = 0):
        """Create an input projection, ``depth`` dropout-linear pairs, and an output head.

        Args:
            depth: Number of hidden layers.
            hidden: Hidden feature count.
            in_dim: Input feature count.
            seed: PRNG seed for parameter init.
        """
        rngs = nnx.Rngs(seed)
        self.fc_in = nnx.Linear(in_dim, hidden, rngs=rngs)
        self._depth = depth
        for i in range(depth):
            setattr(self, f"l{i}", nnx.Linear(hidden, hidden, rngs=rngs))
            setattr(self, f"d{i}", nnx.Dropout(0.1, rngs=rngs))
        self.fc_out = nnx.Linear(hidden, 10, rngs=rngs)

    def __call__(self, x):
        """Run the dropout MLP forward."""
        x = jax.nn.gelu(self.fc_in(x))
        for i in range(self._depth):
            x = getattr(self, f"l{i}")(x)
            x = getattr(self, f"d{i}")(x)
            x = jax.nn.gelu(x)
        return self.fc_out(x)


def nnx_dropout_mlp(depth: int = 6, hidden: int = 512, in_dim: int = 512, seed: int = 0):
    """Build a Flax NNX dropout MLP and a matching input tensor.

    Args:
        depth: Number of hidden layers.
        hidden: Hidden feature count.
        in_dim: Input feature count.
        seed: PRNG seed for parameter init.

    Returns:
        ``(module, example_input)`` tuple.
    """
    mdl = NnxDropoutMLP(depth=depth, hidden=hidden, in_dim=in_dim, seed=seed)
    x = jnp.ones((32, in_dim), dtype=jnp.float32)
    return mdl, x
