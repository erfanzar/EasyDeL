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

"""tile-lang prim_func factories for fused forward-KL.

Two execution flows are exported:

* **Single-shard fused forward** (``make_fused_kl_prim_func``): per-row
  loss + dstudent in one fused pass.
* **Vocab-parallel two-stage flow** for TP:
  ``make_kl_partial_stats_prim_func`` emits per-shard
  ``(local_max_t, local_se_t, local_max_s, local_se_s, local_partial_loss)``;
  the wrapper merges them with ``pmax`` / ``psum`` over the TP mesh axis;
  ``make_kl_dstudent_prim_func`` then writes the local ``dstudent``.

Computes ``KL(softmax(teacher) || softmax(student))`` per row, fused with the
gradient w.r.t. the student logits — useful for distillation.

For each row ``n`` we compute (in registers, never materialising any
``[N, V]`` probability tensor in HBM):

    p_t[v]        = softmax(teacher[n])[v]
    p_s[v]        = softmax(student[n])[v]
    loss[n]       = weight[n] * sum_v p_t[v] * (log p_t[v] - log p_s[v])
    dstudent[n,v] = weight[n] * (p_s[v] - p_t[v])

The kernel runs four sequential sweeps of the vocabulary per row:

    1. teacher max,        2. teacher sum_exp,
    3. student max+sumexp+loss accumulation,
    4. write dstudent.

Rows with zero weight produce zero loss and zero gradient.
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
        raise TypeError(f"Unsupported dtype for tile-lang fused_kl_divergence: {dtype}")
    return mapping[canonical]


def make_fused_kl_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    threads: int = 128,
):
    """Build the fused forward-KL (loss + dstudent) ``@T.prim_func``.

    Grid: one CTA per row of the logits matrices.

    Args:
        num_rows: Number of rows ``N`` (= ``batch * seq_len`` for an LM).
        vocab_size: Vocabulary dimension ``V``.
        block_v: Chunk size along the vocab axis.
        dtype: Logits dtype (float16, bfloat16, float32). Both inputs must
            share dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(Student[N,V], Teacher[N,V], Weights[N], Loss[N], DStudent[N,V])``.
        ``DStudent`` already absorbs the per-row ``Weights`` factor.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    neg_inf = -3.4028234663852886e38

    @T.prim_func
    def fused_kl_fwd(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        Weights: T.Tensor((N,), accum),
        Loss: T.Tensor((N,), accum),
        DStudent: T.Tensor((N, V), ts),
    ):
        with T.Kernel(N, threads=threads) as bx:
            chunk = T.alloc_fragment((BV,), accum)
            chunk_red = T.alloc_fragment((1,), accum)

            max_t = T.alloc_fragment((1,), accum)
            sum_t = T.alloc_fragment((1,), accum)
            max_s = T.alloc_fragment((1,), accum)
            sum_s = T.alloc_fragment((1,), accum)
            loss_buf = T.alloc_fragment((1,), accum)

            weight = Weights[bx]
            valid = T.if_then_else(weight != 0.0, 1.0, 0.0)

            max_t[0] = neg_inf
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, Teacher[bx, v_idx]),
                        neg_inf,
                    )
                T.reduce_max(chunk, chunk_red, dim=0, clear=True)
                max_t[0] = T.max(max_t[0], chunk_red[0])

            sum_t[0] = 0.0
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.exp(T.Cast(accum, Teacher[bx, v_idx]) - max_t[0]),
                        0.0,
                    )
                T.reduce_sum(chunk, chunk_red, dim=0, clear=True)
                sum_t[0] = sum_t[0] + chunk_red[0]

            max_s[0] = neg_inf
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, Student[bx, v_idx]),
                        neg_inf,
                    )
                T.reduce_max(chunk, chunk_red, dim=0, clear=True)
                max_s[0] = T.max(max_s[0], chunk_red[0])

            sum_s[0] = 0.0
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.exp(T.Cast(accum, Student[bx, v_idx]) - max_s[0]),
                        0.0,
                    )
                T.reduce_sum(chunk, chunk_red, dim=0, clear=True)
                sum_s[0] = sum_s[0] + chunk_red[0]

            lse_t = T.log(sum_t[0]) + max_t[0]
            lse_s = T.log(sum_s[0]) + max_s[0]
            inv_sum_t = 1.0 / sum_t[0]

            loss_buf[0] = 0.0
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    t_logit = T.Cast(accum, Teacher[bx, v_idx])
                    s_logit = T.Cast(accum, Student[bx, v_idx])
                    p_t = T.exp(t_logit - max_t[0]) * inv_sum_t
                    log_p_t = t_logit - lse_t
                    log_p_s = s_logit - lse_s
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        p_t * (log_p_t - log_p_s),
                        0.0,
                    )
                T.reduce_sum(chunk, chunk_red, dim=0, clear=True)
                loss_buf[0] = loss_buf[0] + chunk_red[0]
            Loss[bx] = valid * weight * loss_buf[0]

            inv_sum_s = 1.0 / sum_s[0]
            for vi in T.serial(T.ceildiv(V, BV)):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    if v_idx < V:
                        t_logit = T.Cast(accum, Teacher[bx, v_idx])
                        s_logit = T.Cast(accum, Student[bx, v_idx])
                        p_t = T.exp(t_logit - max_t[0]) * inv_sum_t
                        p_s = T.exp(s_logit - max_s[0]) * inv_sum_s
                        DStudent[bx, v_idx] = T.Cast(ts, valid * weight * (p_s - p_t))

    return fused_kl_fwd


def make_kl_fwd_only_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    direction: str = "forward",
    temperature: float = 1.0,
    block_m: int = 1,
    threads: int = 128,
):
    """Build the lean KL forward kernel.

    Computes per-row KL with optional temperature softening and the
    direction selector:

      * ``direction="forward"`` (default) — ``KL(softmax(t/T) ‖ softmax(s/T))``
        ``= Σ p_t · (log p_t - log p_s)``. Bwd uses
        :func:`make_kl_dstudent_only_prim_func`.
      * ``direction="reverse"`` — ``KL(softmax(s/T) ‖ softmax(t/T))``
        ``= Σ p_s · (log p_s - log p_t)``. Bwd uses
        :func:`make_kl_dstudent_reverse_prim_func` and additionally
        needs the ``acc`` residual emitted here.

    The temperature ``T`` is folded into the logit reads as ``logits/T``
    (a single build-time-constant multiply per element). The kernel
    output is the KL value computed in the *scaled* logit space; the
    caller multiplies by ``T²`` if it wants the EasyDeL-style
    ``distillation_loss`` magnitude.

    Output contract: ``(Loss[N], LseT[N], LseS[N], Acc[N])`` all float32.
    ``Acc`` is the chunk-accumulated KL term needed by the reverse bwd
    (and is harmless for the forward bwd, which ignores it).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_scale = 1.0 / scale
    inv_T = 1.0 / float(temperature)
    if direction not in ("forward", "reverse"):
        raise ValueError(f"direction must be 'forward' or 'reverse'; got {direction!r}")
    is_reverse = direction == "reverse"

    @T.prim_func
    def kl_fwd(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        Weights: T.Tensor((N,), accum),
        Loss: T.Tensor((N,), accum),
        LseT: T.Tensor((N,), accum),
        LseS: T.Tensor((N,), accum),
        Acc: T.Tensor((N,), accum),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            t_smem = T.alloc_shared((BM, BV), ts)
            s_smem = T.alloc_shared((BM, BV), ts)
            t_local = T.alloc_fragment((BM, BV), accum)
            s_local = T.alloc_fragment((BM, BV), accum)
            exp_x = T.alloc_fragment((BM, BV), accum)
            kl_term = T.alloc_fragment((BM, BV), accum)
            max_x = T.alloc_fragment((BM,), accum)
            sum_exp = T.alloc_fragment((BM,), accum)
            sum_chunk = T.alloc_fragment((BM,), accum)
            lse_alpha = T.alloc_fragment((BM,), accum)
            lse_beta = T.alloc_fragment((BM,), accum)
            acc = T.alloc_fragment((BM,), accum)
            w_local = T.alloc_fragment((BM,), accum)
            block_active = T.alloc_fragment((1,), accum)
            for i in T.Parallel(BM):
                row = bx * BM + i
                w_local[i] = T.if_then_else(row < N, T.abs(Weights[row]), 0.0)
            T.reduce_sum(w_local, block_active, dim=0, clear=True)

            with T.If(block_active[0] > 0.0):
                with T.Then():
                    T.fill(lse_alpha, -T.infinity(accum))
                    for k in T.serial(T.ceildiv(V, BV)):
                        if is_reverse:
                            T.copy(Student[bx * BM, k * BV], s_smem)
                        else:
                            T.copy(Teacher[bx * BM, k * BV], t_smem)
                        for i, j in T.Parallel(BM, BV):
                            v_idx = k * BV + j
                            raw = T.Cast(accum, s_smem[i, j] if is_reverse else t_smem[i, j])
                            t_local[i, j] = T.if_then_else(v_idx < V, raw * inv_T, -T.infinity(accum))
                        T.reduce_max(t_local, max_x, dim=1, clear=True)
                        for i, j in T.Parallel(BM, BV):
                            exp_x[i, j] = T.exp2(t_local[i, j] * scale - max_x[i] * scale)
                        T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                        for i in T.Parallel(BM):
                            lse_alpha[i] = max_x[i] * scale + T.log2(
                                T.exp2(lse_alpha[i] - max_x[i] * scale) + sum_exp[i]
                            )

                    T.fill(lse_beta, -T.infinity(accum))
                    T.fill(acc, 0.0)
                    for k in T.serial(T.ceildiv(V, BV)):
                        T.copy(Student[bx * BM, k * BV], s_smem)
                        T.copy(Teacher[bx * BM, k * BV], t_smem)
                        for i, j in T.Parallel(BM, BV):
                            v_idx = k * BV + j
                            s_local[i, j] = T.if_then_else(
                                v_idx < V,
                                T.Cast(accum, s_smem[i, j]) * inv_T,
                                -T.infinity(accum) if not is_reverse else 0.0,
                            )
                            t_local[i, j] = T.if_then_else(
                                v_idx < V,
                                T.Cast(accum, t_smem[i, j]) * inv_T,
                                -T.infinity(accum) if is_reverse else 0.0,
                            )
                        if is_reverse:
                            T.reduce_max(t_local, max_x, dim=1, clear=True)
                            for i, j in T.Parallel(BM, BV):
                                exp_x[i, j] = T.exp2(t_local[i, j] * scale - max_x[i] * scale)
                        else:
                            T.reduce_max(s_local, max_x, dim=1, clear=True)
                            for i, j in T.Parallel(BM, BV):
                                exp_x[i, j] = T.exp2(s_local[i, j] * scale - max_x[i] * scale)
                        T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                        for i in T.Parallel(BM):
                            lse_beta[i] = max_x[i] * scale + T.log2(T.exp2(lse_beta[i] - max_x[i] * scale) + sum_exp[i])
                        for i, j in T.Parallel(BM, BV):
                            v_idx = k * BV + j
                            in_v = T.if_then_else(v_idx < V, 1.0, 0.0)
                            if is_reverse:
                                alpha = s_local[i, j]
                                beta_v = T.if_then_else(v_idx < V, t_local[i, j], 0.0)
                            else:
                                alpha = t_local[i, j]
                                beta_v = T.if_then_else(v_idx < V, s_local[i, j], 0.0)
                            alpha_safe = T.if_then_else(v_idx < V, alpha, 0.0)
                            p_alpha = T.exp2((alpha - lse_alpha[i] * inv_scale) * scale)
                            kl_term[i, j] = in_v * p_alpha * (alpha_safe - beta_v)
                        T.reduce_sum(kl_term, sum_chunk, dim=1, clear=True)
                        for i in T.Parallel(BM):
                            acc[i] = acc[i] + sum_chunk[i]

                    for i in T.Parallel(BM):
                        row = bx * BM + i
                        if row < N:
                            weight = Weights[row]
                            valid = T.if_then_else(weight != 0.0, 1.0, 0.0)
                            lse_a_nat = lse_alpha[i] * inv_scale
                            lse_b_nat = lse_beta[i] * inv_scale
                            Loss[row] = valid * weight * (acc[i] + lse_b_nat - lse_a_nat)
                            if is_reverse:
                                LseS[row] = lse_a_nat
                                LseT[row] = lse_b_nat
                            else:
                                LseT[row] = lse_a_nat
                                LseS[row] = lse_b_nat
                            Acc[row] = acc[i]
                with T.Else():
                    for i in T.Parallel(BM):
                        row = bx * BM + i
                        if row < N:
                            Loss[row] = 0.0
                            LseT[row] = 0.0
                            LseS[row] = 0.0
                            Acc[row] = 0.0

    return kl_fwd


def make_kl_single_pass_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    block_m: int = 1,
    threads: int = 128,
):
    """Single-pass KL forward — online softmax for both streams +
    incremental KL accumulator with running-max correction.

    Read teacher + student exactly once each (``2 N·V`` HBM transfers,
    the minimum to materialise both LSEs + the loss). The chunk loop
    body is compact: shared SMEM tiles for both inputs, one promotion
    + max + exp + sum sweep per stream, then a final correction +
    accumulator update on per-row scalars.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_scale = 1.0 / scale

    @T.prim_func
    def kl_single_pass(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        Weights: T.Tensor((N,), accum),
        Loss: T.Tensor((N,), accum),
        LseT: T.Tensor((N,), accum),
        LseS: T.Tensor((N,), accum),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            x_smem = T.alloc_shared((BM, BV), ts)
            x_local = T.alloc_fragment((BM, BV), accum)
            exp_x = T.alloc_fragment((BM, BV), accum)
            tmp = T.alloc_fragment((BM, BV), accum)
            chunk_max = T.alloc_fragment((BM,), accum)
            chunk_sum = T.alloc_fragment((BM,), accum)
            chunk_kl = T.alloc_fragment((BM,), accum)
            t_cache = T.alloc_fragment((BM, BV), accum)
            m_t = T.alloc_fragment((BM,), accum)
            m_s = T.alloc_fragment((BM,), accum)
            l_t = T.alloc_fragment((BM,), accum)
            l_s = T.alloc_fragment((BM,), accum)
            acc = T.alloc_fragment((BM,), accum)

            T.fill(m_t, -T.infinity(accum))
            T.fill(m_s, -T.infinity(accum))
            T.fill(l_t, 0.0)
            T.fill(l_s, 0.0)
            T.fill(acc, 0.0)

            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Teacher[bx * BM, k * BV], x_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    x_local[i, j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, x_smem[i, j]),
                        -T.infinity(accum),
                    )
                    t_cache[i, j] = x_local[i, j]
                T.reduce_max(x_local, chunk_max, dim=1, clear=True)
                for i in T.Parallel(BM):
                    chunk_max[i] = T.max(m_t[i], chunk_max[i])
                for i, j in T.Parallel(BM, BV):
                    exp_x[i, j] = T.exp2((x_local[i, j] - chunk_max[i]) * scale)
                T.reduce_sum(exp_x, chunk_sum, dim=1, clear=True)
                for i in T.Parallel(BM):
                    shift = T.exp2((m_t[i] - chunk_max[i]) * scale)
                    l_t[i] = l_t[i] * shift + chunk_sum[i]
                    acc[i] = acc[i] * shift
                    m_t[i] = chunk_max[i]

                T.copy(Student[bx * BM, k * BV], x_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    x_local[i, j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, x_smem[i, j]),
                        -T.infinity(accum),
                    )
                T.reduce_max(x_local, chunk_max, dim=1, clear=True)
                for i in T.Parallel(BM):
                    chunk_max[i] = T.max(m_s[i], chunk_max[i])
                for i, j in T.Parallel(BM, BV):
                    exp_x[i, j] = T.exp2((x_local[i, j] - chunk_max[i]) * scale)
                T.reduce_sum(exp_x, chunk_sum, dim=1, clear=True)
                for i in T.Parallel(BM):
                    shift = T.exp2((m_s[i] - chunk_max[i]) * scale)
                    l_s[i] = l_s[i] * shift + chunk_sum[i]
                    m_s[i] = chunk_max[i]

                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    in_v = T.if_then_else(v_idx < V, 1.0, 0.0)
                    pt = T.exp2((t_cache[i, j] - m_t[i]) * scale)
                    s_safe = T.if_then_else(v_idx < V, x_local[i, j], 0.0)
                    t_safe = T.if_then_else(v_idx < V, t_cache[i, j], 0.0)
                    tmp[i, j] = in_v * pt * (t_safe - s_safe)
                T.reduce_sum(tmp, chunk_kl, dim=1, clear=True)
                for i in T.Parallel(BM):
                    acc[i] = acc[i] + chunk_kl[i]

            for i in T.Parallel(BM):
                row = bx * BM + i
                if row < N:
                    weight = Weights[row]
                    valid = T.if_then_else(weight != 0.0, 1.0, 0.0)
                    lse_t_nat = m_t[i] + T.log2(l_t[i]) * inv_scale
                    lse_s_nat = m_s[i] + T.log2(l_s[i]) * inv_scale
                    Loss[row] = valid * weight * (acc[i] / l_t[i] + lse_s_nat - lse_t_nat)
                    LseT[row] = lse_t_nat
                    LseS[row] = lse_s_nat

    return kl_single_pass


def make_kl_unified_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    block_m: int = 1,
    threads: int = 128,
):
    """Build a single unified KL fwd+bwd kernel.

    Outputs ``(loss[N], dstudent_unscaled[N, V])`` from a single launch.
    The custom-VJP stores ``dstudent_unscaled`` in the residual and the
    backward becomes a single ``dy[:, None] * dstudent_unscaled``
    broadcast multiply (~one HBM pass over dstudent).

    Inside the CTA:
      * Pass 1: teacher → online-softmax ``lse_t``  (1 teacher read).
      * Pass 2: student → online-softmax ``lse_s``  (1 student read).
      * Pass 3: teacher + student → ``loss`` accumulation **and**
        ``dstudent_unscaled = factor * (exp(s - lse_s) - exp(t - lse_t))``
        (2 reads, 1 write).

    HBM total: 4 reads × N·V + 1 write × N·V = ~5 N·V per-element
    transfers — the minimum for a 2D-output fused KL kernel that needs
    both LSEs before writing the gradient.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_scale = 1.0 / scale

    @T.prim_func
    def kl_unified(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        Weights: T.Tensor((N,), accum),
        Loss: T.Tensor((N,), accum),
        DStudent: T.Tensor((N, V), ts),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            t_smem = T.alloc_shared((BM, BV), ts)
            s_smem = T.alloc_shared((BM, BV), ts)
            y_smem = T.alloc_shared((BM, BV), ts)
            x_local = T.alloc_fragment((BM, BV), accum)
            x2_local = T.alloc_fragment((BM, BV), accum)
            exp_x = T.alloc_fragment((BM, BV), accum)
            kl_term = T.alloc_fragment((BM, BV), accum)
            max_x = T.alloc_fragment((BM,), accum)
            sum_exp = T.alloc_fragment((BM,), accum)
            lse_t_log2 = T.alloc_fragment((BM,), accum)
            lse_s_log2 = T.alloc_fragment((BM,), accum)
            acc = T.alloc_fragment((BM,), accum)
            sum_chunk = T.alloc_fragment((BM,), accum)
            weight = T.alloc_fragment((BM,), accum)

            for i in T.Parallel(BM):
                row = bx * BM + i
                safe_row = T.if_then_else(row < N, row, 0)
                weight[i] = Weights[safe_row]

            T.fill(lse_t_log2, -T.infinity(accum))
            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Teacher[bx * BM, k * BV], t_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    x_local[i, j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, t_smem[i, j]),
                        -T.infinity(accum),
                    )
                T.reduce_max(x_local, max_x, dim=1, clear=True)
                for i, j in T.Parallel(BM, BV):
                    exp_x[i, j] = T.exp2(x_local[i, j] * scale - max_x[i] * scale)
                T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                for i in T.Parallel(BM):
                    lse_t_log2[i] = max_x[i] * scale + T.log2(T.exp2(lse_t_log2[i] - max_x[i] * scale) + sum_exp[i])

            T.fill(lse_s_log2, -T.infinity(accum))
            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Student[bx * BM, k * BV], s_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    x_local[i, j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, s_smem[i, j]),
                        -T.infinity(accum),
                    )
                T.reduce_max(x_local, max_x, dim=1, clear=True)
                for i, j in T.Parallel(BM, BV):
                    exp_x[i, j] = T.exp2(x_local[i, j] * scale - max_x[i] * scale)
                T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                for i in T.Parallel(BM):
                    lse_s_log2[i] = max_x[i] * scale + T.log2(T.exp2(lse_s_log2[i] - max_x[i] * scale) + sum_exp[i])

            T.fill(acc, 0.0)
            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Teacher[bx * BM, k * BV], t_smem)
                T.copy(Student[bx * BM, k * BV], s_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    in_v = T.if_then_else(v_idx < V, 1.0, 0.0)
                    t_v = T.if_then_else(v_idx < V, T.Cast(accum, t_smem[i, j]), 0.0)
                    s_v = T.if_then_else(v_idx < V, T.Cast(accum, s_smem[i, j]), 0.0)
                    x_local[i, j] = t_v
                    x2_local[i, j] = s_v
                    p_t = T.exp2((t_v - lse_t_log2[i] * inv_scale) * scale)
                    p_s = T.exp2((s_v - lse_s_log2[i] * inv_scale) * scale)
                    kl_term[i, j] = in_v * p_t * (t_v - s_v)
                    y_smem[i, j] = T.Cast(ts, weight[i] * in_v * (p_s - p_t))
                T.reduce_sum(kl_term, sum_chunk, dim=1, clear=True)
                for i in T.Parallel(BM):
                    acc[i] = acc[i] + sum_chunk[i]
                T.copy(y_smem, DStudent[bx * BM, k * BV])

            for i in T.Parallel(BM):
                row = bx * BM + i
                if row < N:
                    valid = T.if_then_else(weight[i] != 0.0, 1.0, 0.0)
                    lse_t_nat = lse_t_log2[i] * inv_scale
                    lse_s_nat = lse_s_log2[i] * inv_scale
                    Loss[row] = valid * weight[i] * (acc[i] + lse_s_nat - lse_t_nat)

    return kl_unified


def make_kl_two_lse_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    block_m: int = 1,
    threads: int = 128,
):
    """Build a single-pass two-stream online-softmax kernel.

    Writes ``(lse_t[N], lse_s[N])`` in float32 from one sweep of the
    vocabulary that maintains TWO online-softmax states (one per stream).
    Same compile-time complexity as the CE single-stream fwd_only
    kernel — one ``T.serial`` chunk loop with vectorised ``T.copy``
    inside, mirroring the ``online_softmax`` example pattern.

    Loss and gradient are then computed in JAX from the returned
    ``lse_t``/``lse_s`` (XLA fuses the elementwise ops; bandwidth is
    optimal because we avoid an extra TileLang pass).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_scale = 1.0 / scale

    @T.prim_func
    def kl_two_lse(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        LseT: T.Tensor((N,), accum),
        LseS: T.Tensor((N,), accum),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            x_smem = T.alloc_shared((BM, BV), ts)
            x_local = T.alloc_fragment((BM, BV), accum)
            exp_x = T.alloc_fragment((BM, BV), accum)
            max_x = T.alloc_fragment((BM,), accum)
            sum_exp = T.alloc_fragment((BM,), accum)
            lse_t_log2 = T.alloc_fragment((BM,), accum)
            lse_s_log2 = T.alloc_fragment((BM,), accum)

            T.fill(lse_t_log2, -T.infinity(accum))
            T.fill(lse_s_log2, -T.infinity(accum))

            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Teacher[bx * BM, k * BV], x_smem)
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
                    lse_t_log2[i] = max_x[i] * scale + T.log2(T.exp2(lse_t_log2[i] - max_x[i] * scale) + sum_exp[i])

            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Student[bx * BM, k * BV], x_smem)
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
                    lse_s_log2[i] = max_x[i] * scale + T.log2(T.exp2(lse_s_log2[i] - max_x[i] * scale) + sum_exp[i])

            for i in T.Parallel(BM):
                row = bx * BM + i
                if row < N:
                    LseT[row] = lse_t_log2[i] * inv_scale
                    LseS[row] = lse_s_log2[i] * inv_scale

    return kl_two_lse


def make_kl_dstudent_only_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    temperature: float = 1.0,
    block_m: int = 1,
    threads: int = 128,
):
    """Build the **forward**-KL backward (``dstudent``) ``@T.prim_func``.

    With ``L_user = T² · KL(softmax(t/T) ‖ softmax(s/T))`` and the
    operation wrapper handling the ``T²`` factor, the per-element
    gradient is::

        d L_user / d s_v = T · (softmax(s/T)_v - softmax(t/T)_v)

    The kernel folds the ``1/T`` from the chain rule into the per-row
    ``factor`` (``weight · dy · inv_T``); the wrapper's ``T²`` scaling
    of the loss propagates the remaining ``T`` factor through ``dy``.

    Row-per-CTA design: grid = ``ceildiv(N, BM)`` CTAs, each streaming
    the whole vocab through a ``T.serial`` chunk loop with vectorised
    cp.async loads and stores. (1D grid avoids a mixed-dtype-SMEM
    ThreadSync bug the 2D variant trips on.)
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_T = 1.0 / float(temperature)

    @T.prim_func
    def kl_dstudent_bwd(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        LseT: T.Tensor((N,), accum),
        LseS: T.Tensor((N,), accum),
        Weights: T.Tensor((N,), accum),
        DY: T.Tensor((N,), accum),
        DStudent: T.Tensor((N, V), ts),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            t_smem = T.alloc_shared((BM, BV), ts)
            s_smem = T.alloc_shared((BM, BV), ts)
            y_smem = T.alloc_shared((BM, BV), ts)
            t_local = T.alloc_fragment((BM, BV), accum)
            s_local = T.alloc_fragment((BM, BV), accum)
            factor = T.alloc_fragment((BM,), accum)
            lse_t_local = T.alloc_fragment((BM,), accum)
            lse_s_local = T.alloc_fragment((BM,), accum)
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
                        safe_row = T.if_then_else(row < N, row, 0)
                        weight = Weights[safe_row]
                        dy = DY[safe_row]
                        valid = T.if_then_else(
                            row_ok * T.if_then_else(weight != 0.0, 1.0, 0.0) > 0.5,
                            1.0,
                            0.0,
                        )
                        factor[i] = valid * weight * dy * inv_T
                        lse_t_local[i] = LseT[safe_row]
                        lse_s_local[i] = LseS[safe_row]

                    for k in T.serial(T.ceildiv(V, BV)):
                        T.copy(Teacher[bx * BM, k * BV], t_smem)
                        T.copy(Student[bx * BM, k * BV], s_smem)
                        for i, j in T.Parallel(BM, BV):
                            t_local[i, j] = T.Cast(accum, t_smem[i, j]) * inv_T
                            s_local[i, j] = T.Cast(accum, s_smem[i, j]) * inv_T
                        for i, j in T.Parallel(BM, BV):
                            v_idx = k * BV + j
                            in_v = T.if_then_else(v_idx < V, 1.0, 0.0)
                            p_s = T.exp2((s_local[i, j] - lse_s_local[i]) * scale)
                            p_t = T.exp2((t_local[i, j] - lse_t_local[i]) * scale)
                            y_smem[i, j] = T.Cast(ts, factor[i] * in_v * (p_s - p_t))
                        T.copy(y_smem, DStudent[bx * BM, k * BV])
                with T.Else():
                    for k in T.serial(T.ceildiv(V, BV)):
                        for i, j in T.Parallel(BM, BV):
                            y_smem[i, j] = T.Cast(ts, 0.0)
                        T.copy(y_smem, DStudent[bx * BM, k * BV])

    return kl_dstudent_bwd


def make_kl_dstudent_reverse_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    temperature: float = 1.0,
    block_m: int = 1,
    threads: int = 128,
):
    """Build the **reverse**-KL backward kernel.

    Reverse KL is ``L = Σ p_s · (log p_s - log p_t)`` (with softmax in
    T-scaled space). The gradient w.r.t. ``s_v`` is::

        d L / d s_v = p_s(v) · [(log p_s(v) - log p_t(v)) - L]
                    = p_s(v) · [(s_v - t_v)/T - acc]

    where ``acc = Σ_v' p_s(v') · (s_{v'} - t_{v'})/T`` is the per-row
    accumulator already computed by ``make_kl_fwd_only_prim_func`` in
    ``direction="reverse"`` mode and threaded in via the ``Acc``
    residual. Folding the user-side ``T²`` scaling: the wrapper's
    cotangent absorbs ``T²``, the kernel multiplies by ``inv_T`` to
    account for the chain rule on ``s/T``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_T = 1.0 / float(temperature)

    @T.prim_func
    def kl_dstudent_reverse_bwd(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        LseS: T.Tensor((N,), accum),
        Acc: T.Tensor((N,), accum),
        Weights: T.Tensor((N,), accum),
        DY: T.Tensor((N,), accum),
        DStudent: T.Tensor((N, V), ts),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            t_smem = T.alloc_shared((BM, BV), ts)
            s_smem = T.alloc_shared((BM, BV), ts)
            y_smem = T.alloc_shared((BM, BV), ts)
            t_local = T.alloc_fragment((BM, BV), accum)
            s_local = T.alloc_fragment((BM, BV), accum)
            factor = T.alloc_fragment((BM,), accum)
            lse_s_local = T.alloc_fragment((BM,), accum)
            acc_local = T.alloc_fragment((BM,), accum)
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
                        safe_row = T.if_then_else(row < N, row, 0)
                        weight = Weights[safe_row]
                        dy = DY[safe_row]
                        valid = T.if_then_else(
                            row_ok * T.if_then_else(weight != 0.0, 1.0, 0.0) > 0.5,
                            1.0,
                            0.0,
                        )
                        factor[i] = valid * weight * dy * inv_T
                        lse_s_local[i] = LseS[safe_row]
                        acc_local[i] = Acc[safe_row]

                    for k in T.serial(T.ceildiv(V, BV)):
                        T.copy(Teacher[bx * BM, k * BV], t_smem)
                        T.copy(Student[bx * BM, k * BV], s_smem)
                        for i, j in T.Parallel(BM, BV):
                            t_local[i, j] = T.Cast(accum, t_smem[i, j]) * inv_T
                            s_local[i, j] = T.Cast(accum, s_smem[i, j]) * inv_T
                        for i, j in T.Parallel(BM, BV):
                            v_idx = k * BV + j
                            in_v = T.if_then_else(v_idx < V, 1.0, 0.0)
                            p_s = T.exp2((s_local[i, j] - lse_s_local[i]) * scale)
                            diff = s_local[i, j] - t_local[i, j]
                            y_smem[i, j] = T.Cast(ts, factor[i] * in_v * p_s * (diff - acc_local[i]))
                        T.copy(y_smem, DStudent[bx * BM, k * BV])
                with T.Else():
                    for k in T.serial(T.ceildiv(V, BV)):
                        for i, j in T.Parallel(BM, BV):
                            y_smem[i, j] = T.Cast(ts, 0.0)
                        T.copy(y_smem, DStudent[bx * BM, k * BV])

    return kl_dstudent_reverse_bwd


def make_kl_jsd_fwd_prim_func(
    *,
    num_rows: int,
    vocab_size: int,
    block_v: int,
    dtype,
    beta: float = 0.5,
    temperature: float = 1.0,
    block_m: int = 1,
    threads: int = 128,
):
    """Generalised Jensen-Shannon divergence forward kernel (Agarwal et al.).

    ``L = β · KL(p_t ‖ m) + (1 - β) · KL(p_s ‖ m)`` where
    ``m = β · p_t + (1 - β) · p_s`` and softmaxes are taken in T-scaled
    space.

    The kernel:
      * Pass 1 builds ``lse_t`` (T-scaled).
      * Pass 2 builds ``lse_s`` (T-scaled).
      * Pass 3 streams ``(t, s)`` once more to evaluate ``log m`` per
        element (``log_m = max + log(exp(log p_t + log(β) - max)
        + exp(log p_s + log(1-β) - max))``, i.e. mixture
        ``m = β·p_t + (1-β)·p_s``) and accumulates the two KL
        contributions.

    The kernel writes ``Loss`` (already scaled by ``weight``) plus the
    ``LseT``/``LseS`` residuals so the JAX-side autodiff bwd can
    recompute log-probs and ``log_m`` without redoing the online
    softmax.
    """
    import math as _math

    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_size
    BV = block_v
    BM = block_m
    scale = 1.4426950408889634
    inv_scale = 1.0 / scale
    inv_T = 1.0 / float(temperature)
    beta = float(beta)
    if not 0.0 < beta < 1.0:
        raise ValueError(
            "JSD kernel expects beta in (0, 1); for the limits use "
            "`direction='reverse'` (β→0) or `direction='forward'` (β→1)."
        )
    log_beta = _math.log(beta)
    log_one_minus_beta = _math.log1p(-beta)

    @T.prim_func
    def kl_jsd_fwd(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        Weights: T.Tensor((N,), accum),
        Loss: T.Tensor((N,), accum),
        LseT: T.Tensor((N,), accum),
        LseS: T.Tensor((N,), accum),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=threads) as bx:
            x_smem = T.alloc_shared((BM, BV), ts)
            t_smem = T.alloc_shared((BM, BV), ts)
            s_smem = T.alloc_shared((BM, BV), ts)
            x_local = T.alloc_fragment((BM, BV), accum)
            t_local = T.alloc_fragment((BM, BV), accum)
            s_local = T.alloc_fragment((BM, BV), accum)
            exp_x = T.alloc_fragment((BM, BV), accum)
            kl_term = T.alloc_fragment((BM, BV), accum)
            max_x = T.alloc_fragment((BM,), accum)
            sum_exp = T.alloc_fragment((BM,), accum)
            sum_chunk = T.alloc_fragment((BM,), accum)
            lse_t = T.alloc_fragment((BM,), accum)
            lse_s = T.alloc_fragment((BM,), accum)
            acc = T.alloc_fragment((BM,), accum)

            T.fill(lse_t, -T.infinity(accum))
            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Teacher[bx * BM, k * BV], x_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    x_local[i, j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, x_smem[i, j]) * inv_T,
                        -T.infinity(accum),
                    )
                T.reduce_max(x_local, max_x, dim=1, clear=True)
                for i, j in T.Parallel(BM, BV):
                    exp_x[i, j] = T.exp2(x_local[i, j] * scale - max_x[i] * scale)
                T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                for i in T.Parallel(BM):
                    lse_t[i] = max_x[i] * scale + T.log2(T.exp2(lse_t[i] - max_x[i] * scale) + sum_exp[i])

            T.fill(lse_s, -T.infinity(accum))
            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Student[bx * BM, k * BV], x_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    x_local[i, j] = T.if_then_else(
                        v_idx < V,
                        T.Cast(accum, x_smem[i, j]) * inv_T,
                        -T.infinity(accum),
                    )
                T.reduce_max(x_local, max_x, dim=1, clear=True)
                for i, j in T.Parallel(BM, BV):
                    exp_x[i, j] = T.exp2(x_local[i, j] * scale - max_x[i] * scale)
                T.reduce_sum(exp_x, sum_exp, dim=1, clear=True)
                for i in T.Parallel(BM):
                    lse_s[i] = max_x[i] * scale + T.log2(T.exp2(lse_s[i] - max_x[i] * scale) + sum_exp[i])

            T.fill(acc, 0.0)
            for k in T.serial(T.ceildiv(V, BV)):
                T.copy(Teacher[bx * BM, k * BV], t_smem)
                T.copy(Student[bx * BM, k * BV], s_smem)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    t_local[i, j] = T.if_then_else(v_idx < V, T.Cast(accum, t_smem[i, j]) * inv_T, 0.0)
                    s_local[i, j] = T.if_then_else(v_idx < V, T.Cast(accum, s_smem[i, j]) * inv_T, 0.0)
                for i, j in T.Parallel(BM, BV):
                    v_idx = k * BV + j
                    in_v = T.if_then_else(v_idx < V, 1.0, 0.0)
                    log_pt = t_local[i, j] - lse_t[i] * inv_scale
                    log_ps = s_local[i, j] - lse_s[i] * inv_scale
                    # Mixture m = beta * p_t + (1 - beta) * p_s (matches docstring / GKD): teacher pairs with log(beta).
                    a = log_pt + log_beta
                    b = log_ps + log_one_minus_beta
                    m_ = T.max(a, b)
                    log_m = m_ + T.log2(T.exp2((a - m_) * scale) + T.exp2((b - m_) * scale)) * inv_scale
                    p_t = T.exp2(log_pt * scale)
                    p_s = T.exp2(log_ps * scale)
                    kl_term[i, j] = in_v * (beta * p_t * (log_pt - log_m) + (1.0 - beta) * p_s * (log_ps - log_m))
                T.reduce_sum(kl_term, sum_chunk, dim=1, clear=True)
                for i in T.Parallel(BM):
                    acc[i] = acc[i] + sum_chunk[i]

            for i in T.Parallel(BM):
                row = bx * BM + i
                if row < N:
                    weight = Weights[row]
                    valid = T.if_then_else(weight != 0.0, 1.0, 0.0)
                    Loss[row] = valid * weight * acc[i]
                    LseT[row] = lse_t[i] * inv_scale
                    LseS[row] = lse_s[i] * inv_scale

    return kl_jsd_fwd


def make_kl_partial_stats_prim_func(
    *,
    num_rows: int,
    vocab_local: int,
    block_v: int,
    dtype,
    threads: int = 128,
):
    """Build the per-shard stats kernel for vocab-parallel KL.

    For each row of the local ``(N, V_local)`` shards, emit:
        local_max_t[n]    = max  teacher_local[n, :]
        local_se_t[n]     = sum_v exp(teacher_local[n, v] - local_max_t[n])
        local_max_s[n]    = max  student_local[n, :]
        local_se_s[n]     = sum_v exp(student_local[n, v] - local_max_s[n])

    The wrapper merges these via the online-softmax trick (psum of
    ``local_se * exp(local_max - global_max)``) then computes the per-shard
    KL contribution and psums it across the TP axis.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_local
    BV = block_v
    neg_inf = -3.4028234663852886e38

    @T.prim_func
    def kl_partial_stats(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        LocalMaxT: T.Tensor((N,), accum),
        LocalSumExpT: T.Tensor((N,), accum),
        LocalMaxS: T.Tensor((N,), accum),
        LocalSumExpS: T.Tensor((N,), accum),
    ):
        with T.Kernel(N, threads=threads) as bx:
            chunk = T.alloc_fragment((BV,), accum)
            chunk_red = T.alloc_fragment((1,), accum)
            m_buf = T.alloc_fragment((1,), accum)
            s_buf = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            m_buf[0] = neg_inf
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(v_idx < V, T.Cast(accum, Teacher[bx, v_idx]), neg_inf)
                T.reduce_max(chunk, chunk_red, dim=0, clear=True)
                m_buf[0] = T.max(m_buf[0], chunk_red[0])
            LocalMaxT[bx] = m_buf[0]

            s_buf[0] = 0.0
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.exp(T.Cast(accum, Teacher[bx, v_idx]) - m_buf[0]),
                        0.0,
                    )
                T.reduce_sum(chunk, chunk_red, dim=0, clear=True)
                s_buf[0] = s_buf[0] + chunk_red[0]
            LocalSumExpT[bx] = s_buf[0]

            m_buf[0] = neg_inf
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(v_idx < V, T.Cast(accum, Student[bx, v_idx]), neg_inf)
                T.reduce_max(chunk, chunk_red, dim=0, clear=True)
                m_buf[0] = T.max(m_buf[0], chunk_red[0])
            LocalMaxS[bx] = m_buf[0]

            s_buf[0] = 0.0
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    chunk[j] = T.if_then_else(
                        v_idx < V,
                        T.exp(T.Cast(accum, Student[bx, v_idx]) - m_buf[0]),
                        0.0,
                    )
                T.reduce_sum(chunk, chunk_red, dim=0, clear=True)
                s_buf[0] = s_buf[0] + chunk_red[0]
            LocalSumExpS[bx] = s_buf[0]

    return kl_partial_stats


def make_kl_local_loss_prim_func(
    *,
    num_rows: int,
    vocab_local: int,
    block_v: int,
    dtype,
    threads: int = 128,
):
    """Build the per-shard local-loss kernel used by vocab-parallel KL.

    Given the **global** ``lse_t`` and ``lse_s`` per row, accumulate the
    local contribution to ``sum_v p_t * (log p_t - log p_s)`` summed over
    this shard's vocab slice. The wrapper then ``psum``s the result over
    the TP axis to recover the global per-row KL.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_local
    BV = block_v

    @T.prim_func
    def kl_local_loss(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        LseT: T.Tensor((N,), accum),
        LseS: T.Tensor((N,), accum),
        LocalLoss: T.Tensor((N,), accum),
    ):
        with T.Kernel(N, threads=threads) as bx:
            chunk = T.alloc_fragment((BV,), accum)
            chunk_red = T.alloc_fragment((1,), accum)
            acc = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            lse_t = LseT[bx]
            lse_s = LseS[bx]

            acc[0] = 0.0
            for vi in T.Pipelined(T.ceildiv(V, BV), num_stages=2):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    t_logit = T.Cast(accum, Teacher[bx, v_idx])
                    s_logit = T.Cast(accum, Student[bx, v_idx])
                    p_t = T.exp(t_logit - lse_t)
                    log_p_t = t_logit - lse_t
                    log_p_s = s_logit - lse_s
                    chunk[j] = T.if_then_else(v_idx < V, p_t * (log_p_t - log_p_s), 0.0)
                T.reduce_sum(chunk, chunk_red, dim=0, clear=True)
                acc[0] = acc[0] + chunk_red[0]
            LocalLoss[bx] = acc[0]

    return kl_local_loss


def make_kl_dstudent_prim_func(
    *,
    num_rows: int,
    vocab_local: int,
    block_v: int,
    dtype,
    threads: int = 128,
):
    """Build the per-shard ``dstudent`` writer for vocab-parallel KL.

    Given the **global** ``lse_t`` and ``lse_s`` per row, write the local
    gradient slab::

        dstudent_local[n, v] = weight[n] * (
            exp(student_local[n, v] - lse_s[n]) - exp(teacher_local[n, v] - lse_t[n])
        )
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    N, V = num_rows, vocab_local
    BV = block_v

    @T.prim_func
    def kl_dstudent(
        Student: T.Tensor((N, V), ts),
        Teacher: T.Tensor((N, V), ts),
        LseT: T.Tensor((N,), accum),
        LseS: T.Tensor((N,), accum),
        Weights: T.Tensor((N,), accum),
        DStudent: T.Tensor((N, V), ts),
    ):
        with T.Kernel(N, threads=threads) as bx:
            weight = Weights[bx]
            valid = T.if_then_else(weight != 0.0, 1.0, 0.0)
            lse_t = LseT[bx]
            lse_s = LseS[bx]

            for vi in T.serial(T.ceildiv(V, BV)):
                for j in T.Parallel(BV):
                    v_idx = vi * BV + j
                    if v_idx < V:
                        t_logit = T.Cast(accum, Teacher[bx, v_idx])
                        s_logit = T.Cast(accum, Student[bx, v_idx])
                        p_t = T.exp(t_logit - lse_t)
                        p_s = T.exp(s_logit - lse_s)
                        DStudent[bx, v_idx] = T.Cast(ts, valid * weight * (p_s - p_t))

    return kl_dstudent
