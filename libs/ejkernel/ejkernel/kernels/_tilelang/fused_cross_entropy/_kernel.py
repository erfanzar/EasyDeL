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

"""tile-lang prim_func factories for fused sparse cross-entropy.

Two execution flows are exported:

* **Single-shard fused forward** (``make_fused_ce_prim_func``): the row-wise
  log-softmax, target gather, and analytic gradient
  ``softmax - onehot`` are fused into a single sweep of the vocabulary.
  Used when the entire vocab lives on one device.
* **Vocab-parallel two-stage flow** for Megatron-style TP:
  ``make_ce_partial_stats_prim_func`` emits per-shard
  ``(local_max, local_sum_exp, local_target_logit)``; the wrapper merges
  them with ``pmax``/``psum`` over the TP mesh axis; then
  ``make_ce_dlogits_prim_func`` writes the local ``dlogits`` slab given
  the global softmax denominator.

Rows whose target equals ``ignore_index`` or whose weight is zero are
skipped: loss is set to zero, the entire dlogits row is zeroed, and the
target lookup is clamped to a safe index.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Map a NumPy/JAX dtype to the TileLang dtype string."""
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for tile-lang fused_cross_entropy: {dtype}")
    return mapping[canonical]


def make_fused_ce_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    ignore_index: int = -100,
    threads: int = 128,
):
    """Build the fused sparse-CE forward (loss + dlogits) ``@T.prim_func``.

    Grid: one CTA per row of ``logits``. Each CTA performs three sequential
    sweeps of the vocabulary in ``BLOCK_V`` chunks (max, sum-exp, write).

    Args:
        num_rows: Number of rows ``N`` (= ``batch * seq_len`` for an LM).
        vocab_size: Vocabulary dimension ``V``.
        block_v: Chunk size along the vocab axis.
        dtype: Logits dtype (float16, bfloat16, float32).
        ignore_index: Target value that disables loss/grad for a row.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(Logits[N,V] dtype, Targets[N] int32, Weights[N] float32,
            Loss[N] float32, DLogits[N,V] dtype)``.
        ``DLogits`` already absorbs the per-row ``Weights`` factor.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    neg_inf = -3.4028234663852886e38

    @T.prim_func
    def fused_ce_fwd(
        Logits: T.Tensor((N, V), ts),
        Targets: T.Tensor((N,), "int32"),
        Weights: T.Tensor((N,), accum),
        Loss: T.Tensor((N,), accum),
        DLogits: T.Tensor((N, V), ts),
    ):
        with T.Kernel(N, threads=threads) as bx:
            chunk = T.alloc_fragment((BV,), accum)
            chunk_red = T.alloc_fragment((1,), accum)
            max_buf = T.alloc_fragment((1,), accum)
            sum_buf = T.alloc_fragment((1,), accum)

            target = T.Cast("int32", Targets[bx])
            weight = Weights[bx]
            valid = T.if_then_else((target != ignore_index) & (weight != 0.0), 1.0, 0.0)
            safe_target = T.if_then_else((target >= 0) & (target < V), target, 0)
            target_logit = T.Cast(accum, Logits[bx, safe_target])

            max_buf[0] = neg_inf
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, Logits[bx, v_idx]),
                        neg_inf,
                    )
                T.reduce_max(chunk, chunk_red, dim=0, clear=True)
                max_buf[0] = T.max(max_buf[0], chunk_red[0])

            sum_buf[0] = 0.0
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.exp(T.Cast(accum, Logits[bx, v_idx]) - max_buf[0]),
                        0.0,
                    )
                T.reduce_sum(chunk, chunk_red, dim=0, clear=True)
                sum_buf[0] = sum_buf[0] + chunk_red[0]

            lse = T.log(sum_buf[0]) + max_buf[0]
            inv_sum = 1.0 / sum_buf[0]
            Loss[bx] = valid * weight * (lse - target_logit)

            for vi in T.serial(T.ceildiv(V, BV)):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    if v_idx < V:
                        prob = T.exp(T.Cast(accum, Logits[bx, v_idx]) - max_buf[0]) * inv_sum
                        onehot = T.if_then_else(v_idx == safe_target, 1.0, 0.0)
                        DLogits[bx, v_idx] = T.Cast(ts, valid * weight * (prob - onehot))

    return fused_ce_fwd


def make_ce_fwd_only_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    block_m: int = 1,
    threads: int = 128,
):
    """Build the lean cross-entropy forward (loss + lse residual) ``@T.prim_func``.

    Supports:
      * **Label smoothing** (``label_smoothing``) — uses the EasyDeL
        smoothed-target encoding: ``p[target] = 1 - α``,
        ``p[v ≠ target] = α / (V - 1)``. The loss becomes
        ``lse - (1-α-α/(V-1)) · target_logit - α/(V-1) · sum_v logits[v]
        - normalising_constant``.
      * **z-loss regularisation** (``z_loss``) — adds ``z_loss · lse²``
        to the loss (matches ``cross_entropy_with_logits`` z-loss).

    Restructured per TileLang's ``online_softmax`` example:
      * Grid: ``ceildiv(N, block_m)`` CTAs, each handling ``block_m`` rows.
      * 2D fragments ``(block_m, block_v)`` so ``T.reduce_*(..., dim=1)``
        gives per-row results in one call.
      * SMEM tile + ``T.copy`` for vectorised HBM→SMEM transfer
        (cp.async, ~2x effective bandwidth vs scalar loads).
      * Inline online softmax:
        ``lse[i] = max_x[i]*log2e + log2(exp2(lse[i] - max_x[i]*log2e) + sum_exp[i])``
        — works in log2 space for fast exp2/log2 SFU instructions.
      * ``T.Serial`` (not ``T.Pipelined``) for the chunk loop — TileLang
        pipelines vectorised ``T.copy`` automatically via cp.async without
        IR-level unrolling, so compile-time RAM stays bounded.

    Output contract: ``(Loss[N] fp32, Lse[N] fp32)``. Loss already absorbs
    label-smoothing + z-loss + per-row weight. ``Lse`` is saved for the
    backward kernel (needed both for ``softmax = exp(logits - lse)`` and
    for the z-loss gradient term).
    """
    import math

    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_scale = 1.0 / scale

    label_smoothing = float(label_smoothing)
    z_loss = float(z_loss)
    confidence = 1.0 - label_smoothing
    low_conf = (label_smoothing / (V - 1)) if V > 1 and label_smoothing > 0.0 else 0.0
    eff_target_w = confidence - low_conf
    if label_smoothing > 0.0:
        normalizing_constant = -(
            confidence * math.log(max(confidence, 1e-20)) + (V - 1) * low_conf * math.log(max(low_conf, 1e-20))
        )
    else:
        normalizing_constant = 0.0
    needs_sum_logits = label_smoothing > 0.0

    if needs_sum_logits:

        @T.prim_func
        def ce_fwd_smoothed(
            Logits: T.Tensor((N, V), ts),
            Targets: T.Tensor((N,), "int32"),
            Weights: T.Tensor((N,), accum),
            Loss: T.Tensor((N,), accum),
            Lse: T.Tensor((N,), accum),
            Correct: T.Tensor((N,), accum),
        ):
            with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
                x_smem = T.alloc_shared((BM, BV), ts)
                x_local = T.alloc_fragment((BM, BV), accum)
                exp_x = T.alloc_fragment((BM, BV), accum)
                x_safe = T.alloc_fragment((BM, BV), accum)
                arg_candidate = T.alloc_fragment((BM, BV), "int32")
                max_x = T.alloc_fragment((BM,), accum)
                sum_exp = T.alloc_fragment((BM,), accum)
                sum_logits_chunk = T.alloc_fragment((BM,), accum)
                chunk_argmax = T.alloc_fragment((BM,), "int32")
                lse_log2 = T.alloc_fragment((BM,), accum)
                sum_logits = T.alloc_fragment((BM,), accum)
                running_max = T.alloc_fragment((BM,), accum)
                running_argmax = T.alloc_fragment((BM,), "int32")
                w_local = T.alloc_fragment((BM,), accum)
                block_active = T.alloc_fragment((1,), accum)
                for i in T.Parallel(BM):
                    row = bx * BM + i
                    w_local[i] = T.if_then_else(row < N, T.abs(Weights[row]), 0.0)
                T.reduce_sum(w_local, block_active, dim=0, clear=True)

                with T.If(block_active[0] > 0.0):
                    with T.Then():
                        T.fill(lse_log2, -T.infinity(accum))
                        T.fill(sum_logits, 0.0)
                        T.fill(running_max, -T.infinity(accum))
                        T.fill(running_argmax, 0)

                        for k in T.serial(T.ceildiv(V, BV)):
                            T.copy(Logits[bx * BM, k * BV], x_smem)
                            for i, j in T.Parallel(BM, BV):
                                v_idx = k * BV + j
                                x_local[i, j] = T.if_then_else(
                                    v_idx < V,
                                    T.Cast(accum, x_smem[i, j]),
                                    -T.infinity(accum),
                                )
                            T.reduce_max(x_local, max_x, dim=1, clear=True)
                            for i, j in T.Parallel(BM, BV):
                                exp_x[i, j] = T.exp2(x_local[i, j] * scale - max_x[i] * scale)
                            T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                            for i in T.Parallel(BM):
                                lse_log2[i] = max_x[i] * scale + T.log2(
                                    T.exp2(lse_log2[i] - max_x[i] * scale) + sum_exp[i]
                                )
                            for i, j in T.Parallel(BM, BV):
                                v_idx = k * BV + j
                                arg_candidate[i, j] = T.if_then_else(
                                    (v_idx < V) & (x_local[i, j] >= max_x[i]),
                                    v_idx,
                                    V,
                                )
                            T.reduce_min(arg_candidate, chunk_argmax, dim=1, clear=True)
                            for i in T.Parallel(BM):
                                running_argmax[i] = T.if_then_else(
                                    max_x[i] > running_max[i], chunk_argmax[i], running_argmax[i]
                                )
                                running_max[i] = T.max(running_max[i], max_x[i])
                            for i, j in T.Parallel(BM, BV):
                                v_idx = k * BV + j
                                x_safe[i, j] = T.if_then_else(v_idx < V, x_local[i, j], 0.0)
                            T.reduce_sum(x_safe, sum_logits_chunk, dim=1, clear=True)
                            for i in T.Parallel(BM):
                                sum_logits[i] = sum_logits[i] + sum_logits_chunk[i]

                        for i in T.Parallel(BM):
                            row = bx * BM + i
                            if row < N:
                                target = T.Cast("int32", Targets[row])
                                weight = Weights[row]
                                valid = T.if_then_else((target != ignore_index) & (weight != 0.0), 1.0, 0.0)
                                safe_t = T.if_then_else((target >= 0) & (target < V), target, 0)
                                target_logit = T.Cast(accum, Logits[row, safe_t])
                                lse_nat = lse_log2[i] * inv_scale
                                base = (
                                    lse_nat
                                    - eff_target_w * target_logit
                                    - low_conf * sum_logits[i]
                                    - normalizing_constant
                                )
                                z_term = z_loss * lse_nat * lse_nat
                                Loss[row] = valid * weight * (base + z_term)
                                Lse[row] = lse_nat
                                Correct[row] = valid * T.if_then_else(running_argmax[i] == target, 1.0, 0.0)
                    with T.Else():
                        for i in T.Parallel(BM):
                            row = bx * BM + i
                            if row < N:
                                Loss[row] = 0.0
                                Lse[row] = 0.0
                                Correct[row] = 0.0

        return ce_fwd_smoothed

    @T.prim_func
    def ce_fwd(
        Logits: T.Tensor((N, V), ts),
        Targets: T.Tensor((N,), "int32"),
        Weights: T.Tensor((N,), accum),
        Loss: T.Tensor((N,), accum),
        Lse: T.Tensor((N,), accum),
        Correct: T.Tensor((N,), accum),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            x_smem = T.alloc_shared((BM, BV), ts)
            x_local = T.alloc_fragment((BM, BV), accum)
            exp_x = T.alloc_fragment((BM, BV), accum)
            arg_candidate = T.alloc_fragment((BM, BV), "int32")
            max_x = T.alloc_fragment((BM,), accum)
            sum_exp = T.alloc_fragment((BM,), accum)
            chunk_argmax = T.alloc_fragment((BM,), "int32")
            lse_log2 = T.alloc_fragment((BM,), accum)
            running_max = T.alloc_fragment((BM,), accum)
            running_argmax = T.alloc_fragment((BM,), "int32")
            w_local = T.alloc_fragment((BM,), accum)
            block_active = T.alloc_fragment((1,), accum)
            for i in T.Parallel(BM):
                row = bx * BM + i
                w_local[i] = T.if_then_else(row < N, T.abs(Weights[row]), 0.0)
            T.reduce_sum(w_local, block_active, dim=0, clear=True)

            with T.If(block_active[0] > 0.0):
                with T.Then():
                    T.fill(lse_log2, -T.infinity(accum))
                    T.fill(running_max, -T.infinity(accum))
                    T.fill(running_argmax, 0)

                    for k in T.serial(T.ceildiv(V, BV)):
                        T.copy(Logits[bx * BM, k * BV], x_smem)
                        for i, j in T.Parallel(BM, BV):
                            v_idx = k * BV + j
                            x_local[i, j] = T.if_then_else(
                                v_idx < V,
                                T.Cast(accum, x_smem[i, j]),
                                -T.infinity(accum),
                            )
                        T.reduce_max(x_local, max_x, dim=1, clear=True)
                        for i, j in T.Parallel(BM, BV):
                            exp_x[i, j] = T.exp2(x_local[i, j] * scale - max_x[i] * scale)
                        T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                        for i in T.Parallel(BM):
                            lse_log2[i] = max_x[i] * scale + T.log2(T.exp2(lse_log2[i] - max_x[i] * scale) + sum_exp[i])
                        for i, j in T.Parallel(BM, BV):
                            v_idx = k * BV + j
                            arg_candidate[i, j] = T.if_then_else(
                                (v_idx < V) & (x_local[i, j] >= max_x[i]),
                                v_idx,
                                V,
                            )
                        T.reduce_min(arg_candidate, chunk_argmax, dim=1, clear=True)
                        for i in T.Parallel(BM):
                            running_argmax[i] = T.if_then_else(
                                max_x[i] > running_max[i], chunk_argmax[i], running_argmax[i]
                            )
                            running_max[i] = T.max(running_max[i], max_x[i])

                    for i in T.Parallel(BM):
                        row = bx * BM + i
                        if row < N:
                            target = T.Cast("int32", Targets[row])
                            weight = Weights[row]
                            valid = T.if_then_else((target != ignore_index) & (weight != 0.0), 1.0, 0.0)
                            safe_t = T.if_then_else((target >= 0) & (target < V), target, 0)
                            target_logit = T.Cast(accum, Logits[row, safe_t])
                            lse_nat = lse_log2[i] * inv_scale
                            base = lse_nat - target_logit
                            z_term = z_loss * lse_nat * lse_nat
                            Loss[row] = valid * weight * (base + z_term)
                            Lse[row] = lse_nat
                            Correct[row] = valid * T.if_then_else(running_argmax[i] == target, 1.0, 0.0)
                with T.Else():
                    for i in T.Parallel(BM):
                        row = bx * BM + i
                        if row < N:
                            Loss[row] = 0.0
                            Lse[row] = 0.0
                            Correct[row] = 0.0

    return ce_fwd


def make_ce_bwd_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    block_m: int = 1,
    threads: int = 128,
):
    """Build the cross-entropy backward (``dlogits``) ``@T.prim_func``.

    Reads ``Logits`` + per-row ``Lse`` and writes
    ``DLogits = weight * dy * (z_mult · softmax - low_conf - eff_target_w · onehot)``
    where ``z_mult = 1 + 2·z_loss·lse`` and ``softmax[v] = exp(logits[v] - lse)``.

    For ``label_smoothing == 0`` and ``z_loss == 0`` this collapses to
    the classic ``factor · (softmax - onehot)`` writer; the extra terms
    add 2 fma's per element with no extra HBM.

    Grid: ``(ceildiv(V, BV), ceildiv(N, BM))`` — each CTA writes one
    ``(BM rows, BV cols)`` slab of ``DLogits``.  Vectorised HBM I/O via
    ``T.alloc_shared`` + ``T.copy`` (cp.async on Hopper).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634

    label_smoothing = float(label_smoothing)
    z_loss = float(z_loss)
    confidence = 1.0 - label_smoothing
    low_conf = (label_smoothing / (V - 1)) if V > 1 and label_smoothing > 0.0 else 0.0
    eff_target_w = confidence - low_conf

    @T.prim_func
    def ce_bwd(
        Logits: T.Tensor((N, V), ts),
        Lse: T.Tensor((N,), accum),
        Targets: T.Tensor((N,), "int32"),
        Weights: T.Tensor((N,), accum),
        DY: T.Tensor((N,), accum),
        DLogits: T.Tensor((N, V), ts),
    ):
        with T.Kernel(T.ceildiv(V, BV), T.ceildiv(N, BM), threads=threads) as (vx, bx):
            x_smem = T.alloc_shared((BM, BV), ts)
            y_smem = T.alloc_shared((BM, BV), ts)
            x_local = T.alloc_fragment((BM, BV), accum)
            factor = T.alloc_fragment((BM,), accum)
            z_mult = T.alloc_fragment((BM,), accum)
            lse_local = T.alloc_fragment((BM,), accum)
            safe_t = T.alloc_fragment((BM,), "int32")
            in_bounds = T.alloc_fragment((BM,), accum)

            w_local = T.alloc_fragment((BM,), accum)
            block_active = T.alloc_fragment((1,), accum)
            for i in T.Parallel(BM):
                row = bx * BM + i
                w_local[i] = T.if_then_else(row < N, T.abs(Weights[row]), 0.0)
            T.reduce_sum(w_local, block_active, dim=0, clear=True)

            with T.If(block_active[0] > 0.0):
                with T.Then():
                    for i in T.Parallel(BM):
                        row = bx * BM + i
                        row_ok = T.if_then_else(row < N, 1.0, 0.0)
                        in_bounds[i] = row_ok
                        safe_row = T.if_then_else(row < N, row, 0)
                        target = T.Cast("int32", Targets[safe_row])
                        weight = Weights[safe_row]
                        dy = DY[safe_row]
                        valid = T.if_then_else(
                            row_ok * T.if_then_else((target != ignore_index) & (weight != 0.0), 1.0, 0.0) > 0.5,
                            1.0,
                            0.0,
                        )
                        factor[i] = valid * weight * dy
                        safe_t[i] = T.if_then_else((target >= 0) & (target < V), target, 0)
                        lse_local[i] = Lse[safe_row]
                        z_mult[i] = 1.0 + 2.0 * z_loss * lse_local[i]

                    T.copy(Logits[bx * BM, vx * BV], x_smem)
                    for i, j in T.Parallel(BM, BV):
                        x_local[i, j] = T.Cast(accum, x_smem[i, j])

                    for i, j in T.Parallel(BM, BV):
                        v_idx = vx * BV + j
                        prob = T.exp2((x_local[i, j] - lse_local[i]) * scale)
                        onehot = T.if_then_else(v_idx == safe_t[i], 1.0, 0.0)
                        in_v = T.if_then_else(v_idx < V, 1.0, 0.0)
                        y_smem[i, j] = T.Cast(
                            ts,
                            factor[i] * in_v * (z_mult[i] * prob - low_conf - eff_target_w * onehot),
                        )

                    T.copy(y_smem, DLogits[bx * BM, vx * BV])
                with T.Else():
                    for i, j in T.Parallel(BM, BV):
                        y_smem[i, j] = T.Cast(ts, 0.0)
                    T.copy(y_smem, DLogits[bx * BM, vx * BV])

    return ce_bwd


def make_ce_fwd_dense_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    z_loss: float = 0.0,
    block_m: int = 1,
    threads: int = 128,
):
    """Dense-target cross-entropy forward.

    For soft targets where each ``SoftTargets[n, :]`` is a full
    probability distribution over the vocab (e.g. distillation teacher
    softmax, manual one-hot with label smoothing, mixup/cutmix). Reads
    ``Logits`` + ``SoftTargets`` together and writes
    ``Loss[n] = weight[n] · (lse - Σ_v target[n,v] · logits[n,v] + z_loss · lse²)``.

    Same chunked online-softmax structure as the sparse forward but
    one extra chunk reduction per pass (``Σ p · x``), which adds one
    full ``N·V`` read to HBM versus the sparse path. Use the sparse
    fwd + ``label_smoothing`` for cases reducible to label smoothing —
    this kernel is for genuinely soft distributions.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_scale = 1.0 / scale
    z_loss = float(z_loss)

    @T.prim_func
    def ce_fwd_dense(
        Logits: T.Tensor((N, V), ts),
        SoftTargets: T.Tensor((N, V), ts),
        Weights: T.Tensor((N,), accum),
        Loss: T.Tensor((N,), accum),
        Lse: T.Tensor((N,), accum),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            x_smem = T.alloc_shared((BM, BV), ts)
            t_smem = T.alloc_shared((BM, BV), ts)
            x_local = T.alloc_fragment((BM, BV), accum)
            exp_x = T.alloc_fragment((BM, BV), accum)
            pt_x = T.alloc_fragment((BM, BV), accum)
            max_x = T.alloc_fragment((BM,), accum)
            sum_exp = T.alloc_fragment((BM,), accum)
            sum_pt_x = T.alloc_fragment((BM,), accum)
            lse_log2 = T.alloc_fragment((BM,), accum)
            inner = T.alloc_fragment((BM,), accum)

            T.fill(lse_log2, -T.infinity(accum))
            T.fill(inner, 0.0)

            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Logits[bx * BM, k * BV], x_smem)
                T.copy(SoftTargets[bx * BM, k * BV], t_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    x_local[i, j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, x_smem[i, j]),
                        -T.infinity(accum),
                    )
                T.reduce_max(x_local, max_x, dim=1, clear=True)
                for i, j in T.Parallel(BM, BV):
                    exp_x[i, j] = T.exp2(x_local[i, j] * scale - max_x[i] * scale)
                T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                for i in T.Parallel(BM):
                    lse_log2[i] = max_x[i] * scale + T.log2(T.exp2(lse_log2[i] - max_x[i] * scale) + sum_exp[i])
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    pt_x[i, j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, t_smem[i, j]) * x_local[i, j],
                        0.0,
                    )
                T.reduce_sum(pt_x, sum_pt_x, dim=1, clear=True)
                for i in T.Parallel(BM):
                    inner[i] = inner[i] + sum_pt_x[i]

            for i in T.Parallel(BM):
                row = bx * BM + i
                if row < N:
                    weight = Weights[row]
                    valid = T.if_then_else(weight != 0.0, 1.0, 0.0)
                    lse_nat = lse_log2[i] * inv_scale
                    base = lse_nat - inner[i]
                    z_term = z_loss * lse_nat * lse_nat
                    Loss[row] = valid * weight * (base + z_term)
                    Lse[row] = lse_nat

    return ce_fwd_dense


def make_ce_bwd_dense_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    z_loss: float = 0.0,
    block_m: int = 1,
    threads: int = 128,
):
    """Dense-target cross-entropy backward.

    Writes ``DLogits = factor · (z_mult · softmax - soft_target)`` where
    ``factor = weight · dy``, ``z_mult = 1 + 2·z_loss·lse``,
    ``softmax[v] = exp(logits[v] - lse)``. Reads ``Logits`` and
    ``SoftTargets`` (2 N·V reads + 1 N·V write).

    Row-per-CTA design (1D grid, inner ``T.serial`` over V chunks) —
    the CE-style 2D grid trips TileLang's ThreadSync planner when the
    compute pass mixes ``ts`` and ``accum`` dtype SMEM tiles. Same fix
    we applied for the KL dstudent kernel.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    z_loss = float(z_loss)

    @T.prim_func
    def ce_bwd_dense(
        Logits: T.Tensor((N, V), ts),
        SoftTargets: T.Tensor((N, V), ts),
        Lse: T.Tensor((N,), accum),
        Weights: T.Tensor((N,), accum),
        DY: T.Tensor((N,), accum),
        DLogits: T.Tensor((N, V), ts),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            x_smem = T.alloc_shared((BM, BV), ts)
            t_smem = T.alloc_shared((BM, BV), ts)
            y_smem = T.alloc_shared((BM, BV), ts)
            x_local = T.alloc_fragment((BM, BV), accum)
            t_local = T.alloc_fragment((BM, BV), accum)
            factor = T.alloc_fragment((BM,), accum)
            z_mult = T.alloc_fragment((BM,), accum)
            lse_local = T.alloc_fragment((BM,), accum)

            for i in T.Parallel(BM):
                row = bx * BM + i
                row_ok = T.if_then_else(row < N, 1.0, 0.0)
                safe_row = T.if_then_else(row < N, row, 0)
                weight = Weights[safe_row]
                dy = DY[safe_row]
                valid = T.if_then_else(
                    row_ok * T.if_then_else(weight != 0.0, 1.0, 0.0) > 0.5,
                    1.0,
                    0.0,
                )
                factor[i] = valid * weight * dy
                lse_local[i] = Lse[safe_row]
                z_mult[i] = 1.0 + 2.0 * z_loss * lse_local[i]

            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Logits[bx * BM, k * BV], x_smem)
                T.copy(SoftTargets[bx * BM, k * BV], t_smem)
                for i, j in T.Parallel(BM, BV):
                    x_local[i, j] = T.Cast(accum, x_smem[i, j])
                    t_local[i, j] = T.Cast(accum, t_smem[i, j])
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    prob = T.exp2((x_local[i, j] - lse_local[i]) * scale)
                    in_v = T.if_then_else(v_idx < V, 1.0, 0.0)
                    y_smem[i, j] = T.Cast(
                        ts,
                        factor[i] * in_v * (z_mult[i] * prob - t_local[i, j]),
                    )
                T.copy(y_smem, DLogits[bx * BM, k * BV])

    return ce_bwd_dense


def make_ce_partial_stats_prim_func(
    *,
    num_rows: int,
    vocab_local: int,
    block_v: int,
    dtype,
    threads: int = 128,
):
    """Build the per-shard stats kernel used by vocab-parallel CE.

    For each row of the local ``(N, V_local)`` shard, emit:
        local_max[n]          = max_v  logits_local[n, v]
        local_sum_exp[n]      = sum_v  exp(logits_local[n, v] - local_max[n])
        local_target_logit[n] = logits_local[n, target[n] - vocab_start]
                                if vocab_start <= target[n] < vocab_start + V_local
                                else 0

    The wrapper consumes these with ``pmax`` / ``psum`` over the TP mesh
    axis to produce the global ``(max, sum_exp, target_logit)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_local
    BV = block_v
    neg_inf = -3.4028234663852886e38

    @T.prim_func
    def ce_partial_stats(
        Logits: T.Tensor((N, V), ts),
        Targets: T.Tensor((N,), "int32"),
        VocabStart: T.Tensor((1,), "int32"),
        LocalMax: T.Tensor((N,), accum),
        LocalSumExp: T.Tensor((N,), accum),
        LocalTargetLogit: T.Tensor((N,), accum),
    ):
        with T.Kernel(N, threads=threads) as bx:
            chunk = T.alloc_fragment((BV,), accum)
            chunk_red = T.alloc_fragment((1,), accum)
            max_buf = T.alloc_fragment((1,), accum)
            sum_buf = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            v_start = T.Cast("int32", VocabStart[0])
            target = T.Cast("int32", Targets[bx])
            local_idx_raw = target - v_start
            is_local = (local_idx_raw >= 0) & (local_idx_raw < V)
            safe_local_idx = T.if_then_else(is_local, local_idx_raw, 0)

            max_buf[0] = neg_inf
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, Logits[bx, v_idx]),
                        neg_inf,
                    )
                T.reduce_max(chunk, chunk_red, dim=0, clear=True)
                max_buf[0] = T.max(max_buf[0], chunk_red[0])

            sum_buf[0] = 0.0
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.exp(T.Cast(accum, Logits[bx, v_idx]) - max_buf[0]),
                        0.0,
                    )
                T.reduce_sum(chunk, chunk_red, dim=0, clear=True)
                sum_buf[0] = sum_buf[0] + chunk_red[0]

            LocalMax[bx] = max_buf[0]
            LocalSumExp[bx] = sum_buf[0]
            LocalTargetLogit[bx] = T.if_then_else(
                is_local,
                T.Cast(accum, Logits[bx, safe_local_idx]),
                0.0,
            )

    return ce_partial_stats


def make_ce_dlogits_prim_func(
    *,
    num_rows: int,
    vocab_local: int,
    block_v: int,
    dtype,
    threads: int = 128,
):
    """Build the per-shard dlogits writer used by vocab-parallel CE.

    Given the **global** ``max`` and ``sum_exp`` per row (produced by the
    pmax/psum reductions over the TP axis), write the local gradient slab:

        dlogits_local[n, v] = weight[n] * (
            exp(logits_local[n, v] - global_max[n]) / global_sum_exp[n]
            - 1{target[n] - vocab_start == v}
        )

    Rows whose ``weight`` is zero produce a zero gradient row.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_local
    BV = block_v

    @T.prim_func
    def ce_dlogits(
        Logits: T.Tensor((N, V), ts),
        GlobalMax: T.Tensor((N,), accum),
        GlobalSumExp: T.Tensor((N,), accum),
        Targets: T.Tensor((N,), "int32"),
        VocabStart: T.Tensor((1,), "int32"),
        Weights: T.Tensor((N,), accum),
        DLogits: T.Tensor((N, V), ts),
    ):
        with T.Kernel(N, threads=threads) as bx:
            weight = Weights[bx]
            valid = T.if_then_else(weight != 0.0, 1.0, 0.0)

            v_start = T.Cast("int32", VocabStart[0])
            target = T.Cast("int32", Targets[bx])
            local_idx_raw = target - v_start
            is_local = (local_idx_raw >= 0) & (local_idx_raw < V)
            safe_local_idx = T.if_then_else(is_local, local_idx_raw, -1)

            g_max = GlobalMax[bx]
            g_se = GlobalSumExp[bx]
            inv_g_se = 1.0 / g_se

            for vi in T.serial(T.ceildiv(V, BV)):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    if v_idx < V:
                        prob = T.exp(T.Cast(accum, Logits[bx, v_idx]) - g_max) * inv_g_se
                        onehot = T.if_then_else(
                            is_local & (v_idx == safe_local_idx),
                            1.0,
                            0.0,
                        )
                        DLogits[bx, v_idx] = T.Cast(ts, valid * weight * (prob - onehot))

    return ce_dlogits
