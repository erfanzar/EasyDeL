"""Final benchmark: ejkernel TileLang+XLA vs EasyDeL CE / naive JAX KL on H100."""

import time

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# EasyDeL CE port (vendored; Apache 2.0). Rank-agnostic fix for the FWD VJP.
# ---------------------------------------------------------------------------
@jax.custom_vjp
def _easydel_ce_with_logits(logits, targets, z_loss):
    log_sm = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    loss = -jnp.sum(targets * log_sm, axis=-1)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    return loss, z_loss * jax.lax.square(log_z)


def _ed_fwd(logits, targets, z_loss=0.0):
    max_logit = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp_kd = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_sm = shifted - jnp.log(sum_exp_kd)
    loss = -jnp.sum(targets * log_sm, axis=-1)
    log_z = jnp.squeeze(jnp.log(sum_exp_kd) + max_logit, axis=-1)
    total_z = z_loss * jax.lax.square(log_z)
    return (loss + total_z, total_z), (logits, targets, z_loss, exp_shifted, sum_exp_kd, log_sm, log_z)


def _ed_bwd(res, g):
    g = g[0]
    logits, targets, z_loss, exp_shifted, sum_exp_kd, log_sm, log_z = res
    deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp_kd - targets
    g_logits = jnp.expand_dims(g, -1) * deriv
    g_targets = -jnp.expand_dims(g, -1) * log_sm
    return (g_logits.astype(logits.dtype), g_targets.astype(targets.dtype), jnp.array(0.0))


_easydel_ce_with_logits.defvjp(_ed_fwd, _ed_bwd)


def easydel_ce(logits, targets):
    soft = jax.nn.one_hot(targets, logits.shape[-1], dtype=jnp.float32)
    loss, _ = _easydel_ce_with_logits(logits, soft, 0.0)
    return loss.mean()


# Naive JAX KL (EasyDeL ships no full-distribution KL).
def naive_kl(student_logits, teacher_logits):
    lpt = jax.nn.log_softmax(teacher_logits, axis=-1)
    lps = jax.nn.log_softmax(student_logits, axis=-1)
    p_t = jnp.exp(lpt)
    return jnp.sum(p_t * (lpt - lps), axis=-1).mean()


def bench(f, *args, warmup=3, iters=8):
    for _ in range(warmup):
        jax.block_until_ready(f(*args))
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        out.append((time.perf_counter() - t0) * 1e3)
    out.sort()
    return out[len(out) // 2]


def bench_ce(shape):
    from ejkernel.modules.operations import fused_cross_entropy

    B, S, V = shape
    print(f"\n=== Cross-Entropy  B={B} S={S} V={V} fp32 ===", flush=True)
    logits = jax.random.normal(jax.random.PRNGKey(0), (B, S, V))
    targets = jax.random.randint(jax.random.PRNGKey(1), (B, S), 0, V).astype(jnp.int32)

    print("compile ejkernel TileLang...", flush=True)
    fwd_tl = jax.jit(lambda x: fused_cross_entropy(x, targets, platform="tilelang"))
    fb_tl = jax.jit(jax.value_and_grad(lambda x: fused_cross_entropy(x, targets, platform="tilelang")))
    jax.block_until_ready(fwd_tl(logits))
    jax.block_until_ready(fb_tl(logits))

    print("compile EasyDeL...", flush=True)
    fwd_ed = jax.jit(lambda x: easydel_ce(x, targets))
    fb_ed = jax.jit(jax.value_and_grad(lambda x: easydel_ce(x, targets)))
    jax.block_until_ready(fwd_ed(logits))
    jax.block_until_ready(fb_ed(logits))

    fwd_tl_ms = bench(fwd_tl, logits)
    fwd_ed_ms = bench(fwd_ed, logits)
    fb_tl_ms = bench(fb_tl, logits)
    fb_ed_ms = bench(fb_ed, logits)
    print(
        f"  fwd       TileLang={fwd_tl_ms:7.2f} ms  EasyDeL={fwd_ed_ms:7.2f} ms  speedup={fwd_ed_ms / fwd_tl_ms:.2f}x",
        flush=True,
    )
    print(
        f"  fwd+bwd   TileLang={fb_tl_ms:7.2f} ms  EasyDeL={fb_ed_ms:7.2f} ms  speedup={fb_ed_ms / fb_tl_ms:.2f}x",
        flush=True,
    )


def bench_kl(shape):
    from ejkernel.modules.operations import fused_kl_divergence

    B, S, V = shape
    print(f"\n=== KL Divergence  B={B} S={S} V={V} fp32 ===", flush=True)
    student = jax.random.normal(jax.random.PRNGKey(0), (B, S, V))
    teacher = jax.random.normal(jax.random.PRNGKey(1), (B, S, V))

    print("compile ejkernel TileLang...", flush=True)
    fwd_tl = jax.jit(lambda s, t: fused_kl_divergence(s, t, platform="tilelang"))
    fb_tl = jax.jit(jax.value_and_grad(lambda s, t: fused_kl_divergence(s, t, platform="tilelang")))
    jax.block_until_ready(fwd_tl(student, teacher))
    jax.block_until_ready(fb_tl(student, teacher))

    print("compile naive-JAX...", flush=True)
    fwd_n = jax.jit(naive_kl)
    fb_n = jax.jit(jax.value_and_grad(naive_kl))
    jax.block_until_ready(fwd_n(student, teacher))
    jax.block_until_ready(fb_n(student, teacher))

    fwd_tl_ms = bench(fwd_tl, student, teacher)
    fwd_n_ms = bench(fwd_n, student, teacher)
    fb_tl_ms = bench(fb_tl, student, teacher)
    fb_n_ms = bench(fb_n, student, teacher)
    print(
        f"  fwd       TileLang={fwd_tl_ms:7.2f} ms  naive-JAX={fwd_n_ms:7.2f} ms  speedup={fwd_n_ms / fwd_tl_ms:.2f}x",
        flush=True,
    )
    print(
        f"  fwd+bwd   TileLang={fb_tl_ms:7.2f} ms  naive-JAX={fb_n_ms:7.2f} ms  speedup={fb_n_ms / fb_tl_ms:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    SHAPES = [
        (2, 2048, 32000),
        (2, 4096, 128256),
        (4, 8192, 128256),  # target
    ]
    print(f"devices: {jax.devices()}", flush=True)
    for shape in SHAPES:
        bench_ce(shape)
    for shape in SHAPES:
        bench_kl(shape)
