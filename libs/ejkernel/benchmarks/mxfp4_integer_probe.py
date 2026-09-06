"""Throwaway exact MXFP4-operand integer arithmetic probe on TPU.

Packed nibble storage is unchanged. Both operands here are already FP4;
arbitrary BF16 activations are NOT requantized or claimed equivalent.
Scales use ejkernel's signed-exponent-byte convention, grouped along K.
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

LUT = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def pack(c):
    c = c.astype(np.uint32).reshape(*c.shape[:-1], -1, 8)
    return np.bitwise_or.reduce(c << (4 * np.arange(8, dtype=np.uint32)), axis=-1)


def unpack(p):
    return ((p[..., None] >> (4 * jnp.arange(8, dtype=jnp.uint32))) & 15).reshape(*p.shape[:-1], -1)


def integer(c):
    e = (c >> 1) & 3
    mag = jnp.where(e == 0, c & 1, (2 + (c & 1)) << jnp.maximum(e.astype(jnp.int32) - 1, 0))
    return jnp.where(c & 8, -mag.astype(jnp.int32), mag.astype(jnp.int32)).astype(jnp.int8)


def planes(z):
    # z=lo+2*hi, all values exactly representable in signed INT4.
    hi = jnp.where(z < 0, -((-z) // 2), z // 2)
    lo = z - 2 * hi
    return lo.astype(jnp.int4), hi.astype(jnp.int4)


def kernel(ap, bp, ae, be, method):
    za = integer(unpack(ap))
    zb = integer(unpack(bp))
    m, k = za.shape
    n = zb.shape[0]
    if method == "bf16_full_decode":
        a = jnp.ldexp(za.astype(jnp.float32), jnp.repeat(ae.astype(jnp.int32) - 1, 32, axis=1)).astype(jnp.bfloat16)
        b = jnp.ldexp(zb.astype(jnp.float32), jnp.repeat(be.astype(jnp.int32) - 1, 32, axis=1)).astype(jnp.bfloat16)
        return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    out = jnp.zeros((m, n), jnp.float32)

    def body(g, out):
        a = jax.lax.dynamic_slice(za, (0, g * 32), (m, 32))
        b = jax.lax.dynamic_slice(zb, (0, g * 32), (n, 32))

        def dot(x, y):
            return jax.lax.dot_general(x, y, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32)

        if method == "int8":
            acc = dot(a, b).astype(jnp.float32)
        elif method == "int4_planes":
            al, ah = planes(a)
            bl, bh = planes(b)
            acc = (dot(al, bl) + 2 * dot(al, bh) + 2 * dot(ah, bl) + 4 * dot(ah, bh)).astype(jnp.float32)
        else:
            acc = jax.lax.dot_general(
                a.astype(jnp.bfloat16),
                b.astype(jnp.bfloat16),
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        # Combine exponents before scaling, not two separately rounded scales.
        exponent = ae[:, g].astype(jnp.int32)[:, None] + be[:, g].astype(jnp.int32)[None, :] - 2
        return out + jnp.ldexp(acc, exponent)

    return jax.lax.fori_loop(0, k // 32, body, out)


print("ENV", jax.__version__, pltpu.get_tpu_info(), flush=True)
rng = np.random.default_rng(27)
for m, k, n in [(8, 256, 128), (128, 1024, 512)]:
    ac = rng.integers(0, 16, (m, k), dtype=np.uint8)
    bc = rng.integers(0, 16, (n, k), dtype=np.uint8)
    ae = rng.integers(-3, 4, (m, k // 32), dtype=np.int8)
    be = rng.integers(-3, 4, (n, k // 32), dtype=np.int8)
    # Independent floating decode/reference uses the stored represented values.
    av = LUT[ac] * np.exp2(np.repeat(ae, 32, axis=1).astype(np.float32))
    bv = LUT[bc] * np.exp2(np.repeat(be, 32, axis=1).astype(np.float32))
    want = av.astype(np.float64) @ bv.astype(np.float64).T
    args = tuple(jnp.asarray(x) for x in (pack(ac), pack(bc), ae, be))
    for mode in ("bf16_full_decode", "bf16", "int8", "int4_planes"):
        f = jax.jit(lambda *x, mode=mode: kernel(*x, mode))
        try:
            t = time.perf_counter()
            exe = f.lower(*args).compile()
            compile_s = time.perf_counter() - t
            out = exe(*args).block_until_ready()
            np.testing.assert_allclose(np.asarray(out), want, rtol=1e-5, atol=0.002)
            for _ in range(3):
                exe(*args).block_until_ready()
            times = []
            for _ in range(15):
                t = time.perf_counter()
                exe(*args).block_until_ready()
                times.append((time.perf_counter() - t) * 1000)
            print(
                json.dumps(
                    dict(
                        shape=[m, k, n], mode=mode, correct=True, compile_s=compile_s, median_ms=float(np.median(times))
                    )
                ),
                flush=True,
            )
        except Exception as e:
            print(json.dumps(dict(shape=[m, k, n], mode=mode, error=repr(e))), flush=True)
