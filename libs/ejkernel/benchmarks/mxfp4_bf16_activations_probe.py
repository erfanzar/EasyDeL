"""FP4 packed weights with BF16 versus approximate INT8 activations.
Storage is unchanged; weight decoding is exact on the tested bounded scales.
Activation requantization is explicitly an approximation, not MXFP4 parity.
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from ejkernel.kernels._xla.quantized_matmul._integer_quantization import quantize_rows

LUT = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def pack(c):
    words = c.astype(np.uint32).reshape(c.shape[0], -1, 8)
    return np.bitwise_or.reduce(words << (4 * np.arange(8, dtype=np.uint32)), axis=-1)


def decode(p):
    c = ((p[..., None] >> (4 * jnp.arange(8, dtype=jnp.uint32))) & 15).reshape(p.shape[0], -1)
    return jnp.asarray(LUT * 2, jnp.int8)[c]


def run(a, p, e, approximate):
    z = decode(p)
    if not approximate:
        w = jnp.ldexp(z.astype(jnp.float32), jnp.repeat(e.astype(jnp.int32) - 1, 32, axis=1)).astype(jnp.bfloat16)
        return jax.lax.dot_general(a, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    m, k = a.shape
    n = z.shape[0]

    def body(g, out):
        ab = jax.lax.dynamic_slice(a, (0, g * 32), (m, 32))
        wb = jax.lax.dynamic_slice(z, (0, g * 32), (n, 32))
        q, s = quantize_rows(ab, 8)
        acc = jax.lax.dot_general(q, wb, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32)
        return out + jnp.ldexp(acc.astype(jnp.float32) * s, e[:, g].astype(jnp.int32)[None, :] - 1)

    return jax.lax.fori_loop(0, k // 32, body, jnp.zeros((m, n), jnp.float32))


rng = np.random.default_rng(81)
print("BACKEND", jax.default_backend(), jax.__version__, flush=True)
for m, k, n in [(8, 256, 128), (128, 1024, 512)]:
    a = jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16)
    c = rng.integers(0, 16, (n, k), dtype=np.uint8)
    e = rng.integers(-3, 4, (n, k // 32), dtype=np.int8)
    w = LUT[c] * np.exp2(np.repeat(e, 32, axis=1).astype(np.float64))
    ax = np.asarray(a).astype(np.float64)
    reference = ax @ w.T
    args = (a, jnp.asarray(pack(c)), jnp.asarray(e))
    for approx in (False, True):
        exe = jax.jit(lambda a, p, e, approx=approx: run(a, p, e, approx)).lower(*args).compile()
        out = np.asarray(exe(*args).block_until_ready())
        expected = reference
        if approx:
            blocks = ax.reshape(m, k // 32, 32)
            mx = np.max(np.abs(blocks), axis=-1, keepdims=True)
            sc = mx / 127
            q = np.round(blocks * 127 / np.where(mx == 0, 1, mx))
            expected = (q * sc).reshape(m, k) @ w.T
        np.testing.assert_allclose(out, expected, rtol=2e-4, atol=0.002)
        for _ in range(4):
            exe(*args).block_until_ready()
        ts = []
        for _ in range(25):
            t = time.perf_counter()
            exe(*args).block_until_ready()
            ts.append((time.perf_counter() - t) * 1000)
        print(
            json.dumps(
                dict(
                    shape=[m, k, n],
                    approximate_activations=approx,
                    median_ms=float(np.median(ts)),
                    relative_l2_vs_fp4_a16=float(np.linalg.norm(out - reference) / np.linalg.norm(reference)),
                )
            ),
            flush=True,
        )
