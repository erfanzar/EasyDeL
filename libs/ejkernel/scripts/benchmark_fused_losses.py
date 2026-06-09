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

"""Benchmark ejkernel fused CE / KL against EasyDeL-native baselines.

Two comparisons:

* **CE** — ``ejkernel.fused_cross_entropy`` (XLA path with analytic VJP)
  vs an inlined port of ``easydel.infra.loss_utils.compute_weighted_cross_entropy``
  (the canonical pure-JAX CE used by every EasyDeL trainer).
* **KL** — ``ejkernel.fused_kl_divergence`` (XLA path with analytic VJP)
  vs a naive JAX implementation of ``sum_v p_t * (log p_t - log p_s)``.
  EasyDeL itself does not ship a full-distribution KL — its trainers use
  the per-token-logps Schulman k3 estimator, which can't be benchmarked
  on logits the same way.

Both forward-only and forward+backward are timed. The XLA-vs-XLA
comparison favours allocator/scheduling differences and analytic-VJP
fusion; the bigger win for our kernel shows up on GPU where the
TileLang path avoids the ``[..., V]`` softmax materialisation, but the
analytic-VJP win is independent of platform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# EasyDeL CE port (vendored to avoid Python-version mismatch on this host).
# Source: easydel.infra.loss_utils (Apache 2.0).
# ---------------------------------------------------------------------------


def _easydel_onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = jax.lax.eq(labels[..., None], jnp.arange(num_classes)[(None,) * labels.ndim])
    y = jax.lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return y.astype(jnp.float32)


@jax.custom_vjp
def _easydel_ce_with_logits(logits, targets, z_loss):
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - logits_sum
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    return loss, total_z_loss


def _easydel_ce_fwd(logits, targets, z_loss=0.0):
    # Rank-agnostic rewrite of EasyDeL's FWD: the upstream version uses
    # ``[:, None]`` which only works for 2D logits; we use keepdims throughout
    # so the same math is valid for any leading rank.
    max_logit = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp_kd = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_sum_exp_kd = jnp.log(sum_exp_kd)
    log_softmax = shifted - log_sum_exp_kd
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    log_z = jnp.squeeze(log_sum_exp_kd + max_logit, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss = loss + total_z_loss
    return (loss, total_z_loss), (logits, targets, z_loss, exp_shifted, sum_exp_kd, log_softmax, log_z)


def _easydel_ce_bwd(res, g):
    g = g[0]
    logits, targets, z_loss, exp_shifted, sum_exp_kd, log_softmax, log_z = res
    deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp_kd - targets
    g_logits = jnp.expand_dims(g, axis=-1) * deriv
    g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
    return (
        jnp.asarray(g_logits, logits.dtype),
        jnp.asarray(g_targets, targets.dtype),
        jnp.array(0.0),
    )


_easydel_ce_with_logits.defvjp(_easydel_ce_fwd, _easydel_ce_bwd)


def easydel_cross_entropy(logits, targets, weights=None, *, label_smoothing=0.0, z_loss=0.0):
    """Port of ``easydel.infra.loss_utils.compute_weighted_cross_entropy``.

    Returns a scalar mean loss (matches what ejkernel's
    ``fused_cross_entropy`` with ``reduction='mean'`` returns).
    """
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1) if label_smoothing > 0 else 0.0
    if label_smoothing > 0:
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_targets = _easydel_onehot(targets, vocab_size, on_value=confidence, off_value=low_confidence)
    else:
        normalizing_constant = 0.0
        soft_targets = _easydel_onehot(targets, vocab_size)

    total_loss, _ = _easydel_ce_with_logits(logits, soft_targets, z_loss=z_loss)
    total_loss = total_loss - normalizing_constant

    if weights is not None:
        total_loss = total_loss * weights
        weight_sum = jnp.sum(weights)
    else:
        weight_sum = reduce(mul, targets.shape, 1)

    return jnp.sum(total_loss) / jnp.maximum(weight_sum, 1e-8)


# ---------------------------------------------------------------------------
# Naive JAX KL baseline.
# ---------------------------------------------------------------------------


def naive_kl(student_logits, teacher_logits, weights=None):
    """Reference KL: pure-JAX ``sum_v p_t * (log p_t - log p_s)`` then mean."""
    log_p_t = jax.nn.log_softmax(teacher_logits, axis=-1)
    log_p_s = jax.nn.log_softmax(student_logits, axis=-1)
    p_t = jnp.exp(log_p_t)
    per_row = jnp.sum(p_t * (log_p_t - log_p_s), axis=-1)
    if weights is None:
        return per_row.mean()
    return jnp.sum(per_row * weights) / jnp.maximum(jnp.sum(weights), 1e-8)


# ---------------------------------------------------------------------------
# Timing helper.
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    median_ms: float
    min_ms: float
    n_iters: int

    def __repr__(self) -> str:
        return f"{self.median_ms:7.2f} ms (min {self.min_ms:7.2f}, n={self.n_iters})"


def time_fn(fn, *args, warmup: int = 3, iters: int = 10) -> Timing:
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return Timing(median_ms=samples[len(samples) // 2], min_ms=samples[0], n_iters=iters)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def bench_ce(shape, dtype, *, warmup=3, iters=10, platforms=("xla",)):
    from ejkernel.modules.operations import fused_cross_entropy

    B, S, V = shape
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    logits = jax.random.normal(k1, (B, S, V), dtype=dtype)
    targets = jax.random.randint(k2, (B, S), 0, V).astype(jnp.int32)

    easy_fwd = jax.jit(lambda x, t: easydel_cross_entropy(x, t))
    easy_fb = jax.jit(jax.value_and_grad(lambda x, t: easydel_cross_entropy(x, t)))
    v_e = float(easy_fwd(logits, targets))
    t_e_fwd = time_fn(easy_fwd, logits, targets, warmup=warmup, iters=iters)
    t_e_fb = time_fn(easy_fb, logits, targets, warmup=warmup, iters=iters)
    print(f"    easydel             : fwd={t_e_fwd}  fwd+bwd={t_e_fb}  value={v_e:.6f}")

    for plat in platforms:
        try:
            ours_fwd = jax.jit(lambda x, t, _p=plat: fused_cross_entropy(x, t, platform=_p, reduction="mean"))
            ours_fb = jax.jit(
                jax.value_and_grad(lambda x, t, _p=plat: fused_cross_entropy(x, t, platform=_p, reduction="mean"))
            )
            v_o = float(ours_fwd(logits, targets))
            t_o_fwd = time_fn(ours_fwd, logits, targets, warmup=warmup, iters=iters)
            t_o_fb = time_fn(ours_fb, logits, targets, warmup=warmup, iters=iters)
            print(
                f"    ejkernel/{plat:<8}    : fwd={t_o_fwd}  fwd+bwd={t_o_fb}  "
                f"value={v_o:.6f}  speedup fwd={t_e_fwd.median_ms / t_o_fwd.median_ms:.2f}x  "
                f"fwd+bwd={t_e_fb.median_ms / t_o_fb.median_ms:.2f}x"
            )
        except Exception as exc:
            print(f"    ejkernel/{plat:<8}    : FAILED ({type(exc).__name__}: {str(exc)[:120]})")


def bench_kl(shape, dtype, *, warmup=3, iters=10, platforms=("xla",)):
    from ejkernel.modules.operations import fused_kl_divergence

    B, S, V = shape
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    student = jax.random.normal(k1, (B, S, V), dtype=dtype)
    teacher = jax.random.normal(k2, (B, S, V), dtype=dtype)

    naive_fwd = jax.jit(naive_kl)
    naive_fb = jax.jit(jax.value_and_grad(naive_kl))
    v_n = float(naive_fwd(student, teacher))
    t_n_fwd = time_fn(naive_fwd, student, teacher, warmup=warmup, iters=iters)
    t_n_fb = time_fn(naive_fb, student, teacher, warmup=warmup, iters=iters)
    print(f"    naive-jax           : fwd={t_n_fwd}  fwd+bwd={t_n_fb}  value={v_n:.6f}")

    for plat in platforms:
        try:
            ours_fwd = jax.jit(lambda s, t, _p=plat: fused_kl_divergence(s, t, platform=_p, reduction="mean"))
            ours_fb = jax.jit(
                jax.value_and_grad(lambda s, t, _p=plat: fused_kl_divergence(s, t, platform=_p, reduction="mean"))
            )
            v_o = float(ours_fwd(student, teacher))
            t_o_fwd = time_fn(ours_fwd, student, teacher, warmup=warmup, iters=iters)
            t_o_fb = time_fn(ours_fb, student, teacher, warmup=warmup, iters=iters)
            print(
                f"    ejkernel/{plat:<8}    : fwd={t_o_fwd}  fwd+bwd={t_o_fb}  "
                f"value={v_o:.6f}  speedup fwd={t_n_fwd.median_ms / t_o_fwd.median_ms:.2f}x  "
                f"fwd+bwd={t_n_fb.median_ms / t_o_fb.median_ms:.2f}x"
            )
        except Exception as exc:
            print(f"    ejkernel/{plat:<8}    : FAILED ({type(exc).__name__}: {str(exc)[:120]})")


SHAPES = [
    # (batch, seq_len, vocab) — representative LLM training-step shapes.
    (1, 1024, 32000),  # Llama-7B/13B-class micro-step
    (2, 2048, 32000),  # 2x larger
    (1, 2048, 128256),  # Llama-3 vocab (128K)
    (2, 4096, 128256),  # large LLM training step
    (4, 8192, 128256),  # very-large LLM training step (target)
]


def main():
    dtype = jnp.float32
    devices = jax.devices()
    device_plat = devices[0].platform
    print(f"jax devices: {devices}  (first: {device_plat})\n")

    # On GPU, benchmark both XLA and TileLang paths. On CPU/TPU, XLA only.
    platforms = ("tilelang", "xla") if device_plat == "gpu" else ("xla",)

    for shape in SHAPES:
        b, s, v = shape
        print(f"--- Cross-Entropy  shape=(B={b}, S={s}, V={v}) dtype={dtype.dtype} ---")
        bench_ce(shape, dtype, platforms=platforms)
        print()
    for shape in SHAPES:
        b, s, v = shape
        print(f"--- KL Divergence  shape=(B={b}, S={s}, V={v}) dtype={dtype.dtype} ---")
        bench_kl(shape, dtype, platforms=platforms)
        print()


if __name__ == "__main__":
    main()
