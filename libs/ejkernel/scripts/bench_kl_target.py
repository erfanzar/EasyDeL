"""Standalone KL benchmark at (B=4, S=8192, V=128K) on H100."""

import time

import jax
import jax.numpy as jnp


def naive_kl(student_logits, teacher_logits):
    log_p_t = jax.nn.log_softmax(teacher_logits, axis=-1)
    log_p_s = jax.nn.log_softmax(student_logits, axis=-1)
    p_t = jnp.exp(log_p_t)
    return jnp.sum(p_t * (log_p_t - log_p_s), axis=-1).mean()


def main():
    from ejkernel.modules.operations import fused_kl_divergence

    B, S, V = 4, 8192, 128256
    print(f"shape: B={B} S={S} V={V}, ~{B * S * V * 4 / 1e9:.1f} GB per tensor", flush=True)
    s = jax.random.normal(jax.random.PRNGKey(0), (B, S, V))
    t = jax.random.normal(jax.random.PRNGKey(1), (B, S, V))

    # TileLang
    print("compile TileLang fwd...", flush=True)
    fwd_tl = jax.jit(lambda x: fused_kl_divergence(x, t, platform="tilelang"))
    jax.block_until_ready(fwd_tl(s))
    print("compile TileLang fwd+bwd...", flush=True)
    fb_tl = jax.jit(jax.value_and_grad(lambda x: fused_kl_divergence(x, t, platform="tilelang")))
    jax.block_until_ready(fb_tl(s))

    print("warmup...", flush=True)
    for _ in range(2):
        jax.block_until_ready(fwd_tl(s))
        jax.block_until_ready(fb_tl(s))

    def bench(f):
        out = []
        for _ in range(8):
            t0 = time.perf_counter()
            jax.block_until_ready(f(s))
            out.append((time.perf_counter() - t0) * 1e3)
        out.sort()
        return out[len(out) // 2]

    fwd_ms_tl = bench(fwd_tl)
    fb_ms_tl = bench(fb_tl)
    print(f"TileLang KL fwd:     {fwd_ms_tl:.2f} ms", flush=True)
    print(f"TileLang KL fwd+bwd: {fb_ms_tl:.2f} ms  (bwd ~{fb_ms_tl - fwd_ms_tl:.2f} ms)", flush=True)

    # Naive JAX baseline (EasyDeL has no full-distribution KL)
    print("\ncompile naive-JAX fwd / fwd+bwd...", flush=True)
    fwd_naive = jax.jit(naive_kl)
    fb_naive = jax.jit(jax.value_and_grad(naive_kl))
    jax.block_until_ready(fwd_naive(s, t))
    jax.block_until_ready(fb_naive(s, t))
    for _ in range(2):
        jax.block_until_ready(fwd_naive(s, t))
        jax.block_until_ready(fb_naive(s, t))

    def bench2(f, *args):
        out = []
        for _ in range(8):
            t0 = time.perf_counter()
            jax.block_until_ready(f(*args))
            out.append((time.perf_counter() - t0) * 1e3)
        out.sort()
        return out[len(out) // 2]

    fwd_ms_n = bench2(fwd_naive, s, t)
    fb_ms_n = bench2(fb_naive, s, t)
    print(f"naive-JAX KL fwd:     {fwd_ms_n:.2f} ms", flush=True)
    print(f"naive-JAX KL fwd+bwd: {fb_ms_n:.2f} ms", flush=True)

    print(f"\nspeedup fwd:     {fwd_ms_n / fwd_ms_tl:.2f}x", flush=True)
    print(f"speedup fwd+bwd: {fb_ms_n / fb_ms_tl:.2f}x", flush=True)


if __name__ == "__main__":
    main()
