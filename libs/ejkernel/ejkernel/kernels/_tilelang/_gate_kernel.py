# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Native TileLang gate and gated-RMSNorm prim_func factories.

Each factory returns a ``@T.prim_func`` AST with all shape, dtype, and tile
parameters baked in.  The JAX glue in :mod:`._gate_impl` compiles these into
``jax.ffi`` callables and caches them.

Forward kernels:
    * :func:`make_silu_gate_fwd_prim_func`      — ``out = y * silu(gate)``
    * :func:`make_rmsnorm_silu_gate_fwd_prim_func` — ``out = rmsnorm(y) * silu(gate)``
    * :func:`make_head_gate_fwd_prim_func`       — ``out[b,s,h,d] = y[b,s,h,d] * gate[b,s,h]``

Backward kernels (matching forward signatures + ``dout`` input):
    * :func:`make_silu_gate_bwd_prim_func`
    * :func:`make_rmsnorm_silu_gate_bwd_prim_func`
    * :func:`make_head_gate_bwd_prim_func`

All accumulators are float32; inputs and outputs are cast to ``dtype`` /
``gate_dtype`` at the boundaries.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Return the tile-lang dtype string for a JAX/NumPy activation dtype.

    Raises:
        TypeError: if ``dtype`` is not float16, bfloat16 or float32.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for tile-lang gate kernel: {dtype}")
    return mapping[canonical]


def make_silu_gate_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    width: int,
    block_e: int,
    dtype,
    gate_dtype,
    threads: int = 128,
):
    """Build the silu-gate forward ``@T.prim_func``.

    Computes ``out[b,t,d] = y[b,t,d] * silu(gate[b,t,d])`` where
    ``silu(x) = x * sigmoid(x)``.

    Grid: ``(ceildiv(B * S * D, block_e),)``. Each CTA processes a flat
    tile of ``block_e`` elements; 3D indices are recovered by modular
    arithmetic. Out-of-bounds elements are guarded.

    All arithmetic is done in float32; inputs and output are cast at the
    tile boundaries.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        width: feature dimension ``D``.
        block_e: flat tile size (must be a power of two; typical values 128/256).
        dtype: dtype of ``YIn`` and ``YOut``.
        gate_dtype: dtype of ``Gate``.
        threads: threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(YIn[B,S,D], Gate[B,S,D], YOut[B,S,D])``.
    """
    ts = _dtype_str(dtype)
    gate_ts = _dtype_str(gate_dtype)
    accum = "float32"
    B, S, D = batch, seq_len, width
    BE = block_e
    E = B * S * D

    @T.prim_func
    def silu_gate_fwd(
        YIn: T.Tensor((B, S, D), ts),
        Gate: T.Tensor((B, S, D), gate_ts),
        YOut: T.Tensor((B, S, D), ts),
    ):
        with T.Kernel(T.ceildiv(E, BE), threads=threads) as bx:
            vals = T.alloc_fragment((BE,), accum)
            gates = T.alloc_fragment((BE,), accum)
            sig = T.alloc_fragment((BE,), accum)
            _b_ref = T.alloc_fragment((1,), accum)
            _gate_ref = T.alloc_fragment((1,), gate_ts)
            _ts_ref = T.alloc_fragment((1,), ts)
            _b_ref[0] = B
            _gate_ref[0] = Gate[0, 0, 0]
            _ts_ref[0] = YIn[0, 0, 0]

            for i in T.Parallel(BE):
                flat = bx * BE + i
                d = flat % D
                t = (flat // D) % S
                b = flat // (S * D)
                if flat < E:
                    vals[i] = T.Cast(accum, YIn[b, t, d])
                    gates[i] = T.Cast(accum, Gate[b, t, d])
                    sig[i] = 1.0 / (1.0 + T.exp(0.0 - gates[i]))
                    YOut[b, t, d] = T.Cast(ts, vals[i] * gates[i] * sig[i])

    return silu_gate_fwd


def make_silu_gate_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    width: int,
    block_e: int,
    dtype,
    gate_dtype,
    threads: int = 128,
):
    """Build the silu-gate backward ``@T.prim_func``.

    Given ``dYOut = d(loss)/d(out)``, computes:

    * ``dYIn[b,t,d] = dYOut * silu(gate)``
    * ``dGate[b,t,d] = dYOut * y * d_silu/d_gate``
      where ``d_silu/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))``.

    Grid and tile layout are identical to the forward kernel.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        width: feature dimension ``D``.
        block_e: flat tile size.
        dtype: dtype of ``YIn``, ``dYOut`` and ``dYIn``.
        gate_dtype: dtype of ``Gate`` and ``dGate``.
        threads: threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(YIn[B,S,D], Gate[B,S,D], dYOut[B,S,D], dYIn[B,S,D], dGate[B,S,D])``.
    """
    ts = _dtype_str(dtype)
    gate_ts = _dtype_str(gate_dtype)
    accum = "float32"
    B, S, D = batch, seq_len, width
    BE = block_e
    E = B * S * D

    @T.prim_func
    def silu_gate_bwd(
        YIn: T.Tensor((B, S, D), ts),
        Gate: T.Tensor((B, S, D), gate_ts),
        dYOut: T.Tensor((B, S, D), ts),
        dYIn: T.Tensor((B, S, D), ts),
        dGate: T.Tensor((B, S, D), gate_ts),
    ):
        with T.Kernel(T.ceildiv(E, BE), threads=threads) as bx:
            vals = T.alloc_fragment((BE,), accum)
            gates = T.alloc_fragment((BE,), accum)
            grads = T.alloc_fragment((BE,), accum)
            sig = T.alloc_fragment((BE,), accum)
            silu = T.alloc_fragment((BE,), accum)
            dsilu = T.alloc_fragment((BE,), accum)
            _b_ref = T.alloc_fragment((1,), accum)
            _gate_ref = T.alloc_fragment((1,), gate_ts)
            _ts_ref = T.alloc_fragment((1,), ts)
            _b_ref[0] = B
            _gate_ref[0] = Gate[0, 0, 0]
            _ts_ref[0] = YIn[0, 0, 0]

            for i in T.Parallel(BE):
                flat = bx * BE + i
                d = flat % D
                t = (flat // D) % S
                b = flat // (S * D)
                if flat < E:
                    vals[i] = T.Cast(accum, YIn[b, t, d])
                    gates[i] = T.Cast(accum, Gate[b, t, d])
                    grads[i] = T.Cast(accum, dYOut[b, t, d])
                    sig[i] = 1.0 / (1.0 + T.exp(0.0 - gates[i]))
                    silu[i] = gates[i] * sig[i]
                    dsilu[i] = sig[i] * (1.0 + gates[i] * (1.0 - sig[i]))
                    dYIn[b, t, d] = T.Cast(ts, grads[i] * silu[i])
                    dGate[b, t, d] = T.Cast(gate_ts, grads[i] * vals[i] * dsilu[i])

    return silu_gate_bwd


def make_rmsnorm_silu_gate_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    width: int,
    eps: float,
    dtype,
    gate_dtype,
    threads: int = 128,
):
    """Build the RMSNorm-silu-gate forward ``@T.prim_func``.

    Computes per-token RMSNorm of ``y`` then multiplies by ``silu(gate)``:

        ``inv = 1 / sqrt(mean(y^2) + eps)``
        ``out[b,t,d] = y[b,t,d] * inv * silu(gate[b,t,d])``

    Grid: ``(B * S,)`` — one CTA per ``(batch, seq)`` pair.  Each CTA
    holds the full ``D``-wide row fragment in registers.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        width: feature dimension ``D``.
        eps: RMSNorm epsilon (e.g. 1e-6).
        dtype: dtype of ``YIn`` and ``YOut``.
        gate_dtype: dtype of ``Gate``.
        threads: threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature ``(YIn[B,S,D], Gate[B,S,D], YOut[B,S,D])``.
    """
    ts = _dtype_str(dtype)
    gate_ts = _dtype_str(gate_dtype)
    accum = "float32"
    B, S, D = batch, seq_len, width
    Eps = float(eps)

    @T.prim_func
    def rmsnorm_silu_gate_fwd(
        YIn: T.Tensor((B, S, D), ts),
        Gate: T.Tensor((B, S, D), gate_ts),
        YOut: T.Tensor((B, S, D), ts),
    ):
        with T.Kernel(B * S, threads=threads) as bx:
            vals = T.alloc_fragment((D,), accum)
            gates = T.alloc_fragment((D,), accum)
            sq = T.alloc_fragment((D,), accum)
            total = T.alloc_fragment((1,), accum)
            inv = T.alloc_fragment((1,), accum)
            sig = T.alloc_fragment((D,), accum)
            _b_ref = T.alloc_fragment((1,), accum)
            _gate_ref = T.alloc_fragment((1,), gate_ts)
            _ts_ref = T.alloc_fragment((1,), ts)
            _eps_ref = T.alloc_fragment((1,), accum)
            b = bx // S
            t = bx % S
            _b_ref[0] = B
            _gate_ref[0] = Gate[0, 0, 0]
            _ts_ref[0] = YIn[0, 0, 0]
            _eps_ref[0] = Eps

            for d in T.Parallel(D):
                vals[d] = T.Cast(accum, YIn[b, t, d])
                gates[d] = T.Cast(accum, Gate[b, t, d])
                sq[d] = vals[d] * vals[d]
            T.reduce_sum(sq, total, dim=0, clear=True)
            inv[0] = 1.0 / T.sqrt(total[0] / D + Eps)
            for d in T.Parallel(D):
                sig[d] = 1.0 / (1.0 + T.exp(0.0 - gates[d]))
                YOut[b, t, d] = T.Cast(ts, vals[d] * inv[0] * gates[d] * sig[d])

    return rmsnorm_silu_gate_fwd


def make_rmsnorm_silu_gate_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    width: int,
    eps: float,
    dtype,
    gate_dtype,
    threads: int = 128,
):
    """Build the RMSNorm-silu-gate backward ``@T.prim_func``.

    Uses the chain rule for RMSNorm + silu-gate.  Let
    ``v = dYOut * silu(gate)`` (scaled upstream gradient) and
    ``inv = 1/sqrt(mean(y^2) + eps)``.  Then:

        ``dYIn[d] = inv * v[d] - y[d] * (inv^3 / D) * dot(v, y)``
        ``dGate[d] = dYOut[d] * y_norm[d] * dsilu_dgate[d]``

    where ``y_norm = y * inv``.

    Grid: ``(B * S,)`` — one CTA per ``(batch, seq)`` pair.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        width: feature dimension ``D``.
        eps: RMSNorm epsilon.
        dtype: dtype of ``YIn``, ``dYOut`` and ``dYIn``.
        gate_dtype: dtype of ``Gate`` and ``dGate``.
        threads: threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(YIn[B,S,D], Gate[B,S,D], dYOut[B,S,D], dYIn[B,S,D], dGate[B,S,D])``.
    """
    ts = _dtype_str(dtype)
    gate_ts = _dtype_str(gate_dtype)
    accum = "float32"
    B, S, D = batch, seq_len, width
    Eps = float(eps)

    @T.prim_func
    def rmsnorm_silu_gate_bwd(
        YIn: T.Tensor((B, S, D), ts),
        Gate: T.Tensor((B, S, D), gate_ts),
        dYOut: T.Tensor((B, S, D), ts),
        dYIn: T.Tensor((B, S, D), ts),
        dGate: T.Tensor((B, S, D), gate_ts),
    ):
        with T.Kernel(B * S, threads=threads) as bx:
            vals = T.alloc_fragment((D,), accum)
            gates = T.alloc_fragment((D,), accum)
            grads = T.alloc_fragment((D,), accum)
            sq = T.alloc_fragment((D,), accum)
            sig = T.alloc_fragment((D,), accum)
            silu = T.alloc_fragment((D,), accum)
            dsilu = T.alloc_fragment((D,), accum)
            v = T.alloc_fragment((D,), accum)
            dot_terms = T.alloc_fragment((D,), accum)
            total = T.alloc_fragment((1,), accum)
            dot = T.alloc_fragment((1,), accum)
            inv = T.alloc_fragment((1,), accum)
            inv3_scale = T.alloc_fragment((1,), accum)
            _b_ref = T.alloc_fragment((1,), accum)
            _gate_ref = T.alloc_fragment((1,), gate_ts)
            _ts_ref = T.alloc_fragment((1,), ts)
            _eps_ref = T.alloc_fragment((1,), accum)
            b = bx // S
            t = bx % S
            _b_ref[0] = B
            _gate_ref[0] = Gate[0, 0, 0]
            _ts_ref[0] = YIn[0, 0, 0]
            _eps_ref[0] = Eps

            for d in T.Parallel(D):
                vals[d] = T.Cast(accum, YIn[b, t, d])
                gates[d] = T.Cast(accum, Gate[b, t, d])
                grads[d] = T.Cast(accum, dYOut[b, t, d])
                sq[d] = vals[d] * vals[d]
            T.reduce_sum(sq, total, dim=0, clear=True)
            inv[0] = 1.0 / T.sqrt(total[0] / D + Eps)

            for d in T.Parallel(D):
                sig[d] = 1.0 / (1.0 + T.exp(0.0 - gates[d]))
                silu[d] = gates[d] * sig[d]
                dsilu[d] = sig[d] * (1.0 + gates[d] * (1.0 - sig[d]))
                v[d] = grads[d] * silu[d]
                dot_terms[d] = v[d] * vals[d]
            T.reduce_sum(dot_terms, dot, dim=0, clear=True)
            inv3_scale[0] = inv[0] * inv[0] * inv[0] * dot[0] / D

            for d in T.Parallel(D):
                dYIn[b, t, d] = T.Cast(ts, inv[0] * v[d] - vals[d] * inv3_scale[0])
                dGate[b, t, d] = T.Cast(gate_ts, grads[d] * vals[d] * inv[0] * dsilu[d])

    return rmsnorm_silu_gate_bwd


def make_head_gate_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    block_e: int,
    dtype,
    gate_dtype,
    threads: int = 128,
):
    """Build the head-gate forward ``@T.prim_func``.

    Computes ``out[b,t,h,d] = y[b,t,h,d] * gate[b,t,h]`` — each head is
    scaled by a single per-head scalar without broadcasting the gate over
    ``head_dim`` in global memory.

    Grid: ``(ceildiv(B * S * H * D, block_e),)``. 4D indices are recovered
    by modular arithmetic from the flat index.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of heads ``H``.
        head_dim: head feature dimension ``D``.
        block_e: flat tile size.
        dtype: dtype of ``YIn`` and ``YOut``.
        gate_dtype: dtype of ``Gate``.
        threads: threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(YIn[B,S,H,D], Gate[B,S,H], YOut[B,S,H,D])``.
    """
    ts = _dtype_str(dtype)
    gate_ts = _dtype_str(gate_dtype)
    accum = "float32"
    B, S, H, D = batch, seq_len, num_heads, head_dim
    BE = block_e
    E = B * S * H * D

    @T.prim_func
    def head_gate_fwd(
        YIn: T.Tensor((B, S, H, D), ts),
        Gate: T.Tensor((B, S, H), gate_ts),
        YOut: T.Tensor((B, S, H, D), ts),
    ):
        with T.Kernel(T.ceildiv(E, BE), threads=threads) as bx:
            _b_ref = T.alloc_fragment((1,), accum)
            _gate_ref = T.alloc_fragment((1,), gate_ts)
            _ts_ref = T.alloc_fragment((1,), ts)
            _b_ref[0] = B
            _gate_ref[0] = Gate[0, 0, 0]
            _ts_ref[0] = YIn[0, 0, 0, 0]

            for i in T.Parallel(BE):
                flat = bx * BE + i
                d = flat % D
                h = (flat // D) % H
                t = (flat // (D * H)) % S
                b = flat // (S * H * D)
                if flat < E:
                    YOut[b, t, h, d] = T.Cast(
                        ts,
                        T.Cast(accum, YIn[b, t, h, d]) * T.Cast(accum, Gate[b, t, h]),
                    )

    return head_gate_fwd


def make_head_gate_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype,
    gate_dtype,
    threads: int = 128,
):
    """Build the head-gate backward ``@T.prim_func``.

    Computes:

    * ``dYIn[b,t,h,d] = dYOut[b,t,h,d] * gate[b,t,h]``
    * ``dGate[b,t,h] = sum_d(dYOut[b,t,h,d] * y[b,t,h,d])``

    Grid: ``(H, S, B)`` — one CTA per ``(head, token, batch)`` tuple.
    The ``D``-wide reduction for ``dGate`` uses ``T.reduce_sum`` inside
    the CTA, avoiding cross-CTA atomics.

    Args:
        batch: batch size ``B``.
        seq_len: sequence length ``S``.
        num_heads: number of heads ``H``.
        head_dim: head feature dimension ``D``.
        dtype: dtype of ``YIn``, ``dYOut`` and ``dYIn``.
        gate_dtype: dtype of ``Gate`` and ``dGate``.
        threads: threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(YIn[B,S,H,D], Gate[B,S,H], dYOut[B,S,H,D], dYIn[B,S,H,D], dGate[B,S,H])``.
    """
    ts = _dtype_str(dtype)
    gate_ts = _dtype_str(gate_dtype)
    accum = "float32"
    B, S, H, D = batch, seq_len, num_heads, head_dim

    @T.prim_func
    def head_gate_bwd(
        YIn: T.Tensor((B, S, H, D), ts),
        Gate: T.Tensor((B, S, H), gate_ts),
        dYOut: T.Tensor((B, S, H, D), ts),
        dYIn: T.Tensor((B, S, H, D), ts),
        dGate: T.Tensor((B, S, H), gate_ts),
    ):
        with T.Kernel(H, S, B, threads=threads) as (hx, tx, bx):
            total = T.alloc_fragment((D,), accum)
            gate_grad = T.alloc_fragment((1,), accum)
            _b_ref = T.alloc_fragment((1,), accum)
            _gate_ref = T.alloc_fragment((1,), gate_ts)
            _ts_ref = T.alloc_fragment((1,), ts)
            _b_ref[0] = B
            _gate_ref[0] = Gate[0, 0, 0]
            _ts_ref[0] = YIn[0, 0, 0, 0]

            for d in T.Parallel(D):
                total[d] = T.Cast(accum, dYOut[bx, tx, hx, d]) * T.Cast(accum, YIn[bx, tx, hx, d])
                dYIn[bx, tx, hx, d] = T.Cast(
                    ts,
                    T.Cast(accum, dYOut[bx, tx, hx, d]) * T.Cast(accum, Gate[bx, tx, hx]),
                )
            T.reduce_sum(total, gate_grad, dim=0, clear=True)
            dGate[bx, tx, hx] = T.Cast(gate_ts, gate_grad[0])

    return head_gate_bwd
