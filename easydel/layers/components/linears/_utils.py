import jax
from ejkernel.callib import cdiv, ejit
from jax import numpy as jnp


def nf4xf32_to_f32(x):
    x = x.astype(jnp.float32)
    return (
        x
        * (
            x * (x * (x * (1.82943132356953e-5 * x - 0.00068587779130373) + 0.0100420261313669) - 0.0722703570217226)
            + 0.346075459755188
        )
        - 0.994166218659335
    )


def i4tou4(x):
    """Convert int4 to uint4 (unsigned)."""
    return jnp.where(x < 0, 16 + x, x)


@ejit(static_argnames=["BK", "BM", "BNQL", "compute_dtype"])
def nf4_qmm_jax(
    x: jax.Array,
    wq: jax.Array,
    wscale: jax.Array,
    BK: int = 2048,
    BM: int = 2048,
    BNQL: int = 2048,
    compute_dtype: jnp.dtype = jnp.bfloat16,
):
    qfeatures = wq.shape[-1]
    qnumblocks = wq.shape[-2]
    qblocksize = qfeatures * 2
    UN = qblocksize * qnumblocks
    M = x.shape[0]
    K = x.shape[1]
    BNQL = min(BNQL, qnumblocks)
    BN = min(cdiv(BNQL, qnumblocks) * qnumblocks, UN)
    BM, BK = min(BM, M), min(BK, K)
    num_mblocks = cdiv(M, BM)
    num_kblocks = cdiv(K, BK)
    num_nblocks = cdiv(UN, BN)
    BQ = cdiv(num_nblocks, qnumblocks)
    output = jnp.zeros([M, UN])
    for midx in range(num_mblocks):
        for nidx in range(num_nblocks):
            acc = jnp.zeros([BM, BN], compute_dtype)
            for kidx in range(num_kblocks):
                a = jax.lax.dynamic_slice(x, (midx * BM, kidx * BK), (BM, BK))
                b = jax.lax.dynamic_slice(wq, (kidx * BK, nidx * BQ, 0), (BK, BQ, qfeatures))
                bs = jax.lax.dynamic_slice(wscale, (kidx * BK, nidx * BQ), (BK, BQ))

                b = jnp.stack([(b >> 4) & 0xF, b & 0xF], axis=-1)
                *batch_dims, num_blocks, _ = b.shape
                b = b.reshape(*batch_dims, num_blocks, -1)
                b = i4tou4(b)
                b = nf4xf32_to_f32(b).reshape(*b.shape[:-2], -1)
                w = (b * jnp.expand_dims(bs, -1)).reshape(BK, -1).astype(compute_dtype)
                acc += jnp.dot(a, w)
            output = output.at[
                midx * BM : (midx * BM) + BM,
                nidx * BN : (nidx * BN) + BN,
            ].set(acc)
    return output
