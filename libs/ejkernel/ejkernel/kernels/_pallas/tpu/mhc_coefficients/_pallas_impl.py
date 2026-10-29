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

"""Lane-packed four-stream coefficients with a recomputing TPU reverse pass.

Only the small projection logits and parameters are saved by the VJP. The
normalization tape is recomputed inside each 128-token Pallas program, not
written as a sequence of HBM tensors. First-order reverse AD only; use the XLA
operation for JVPs and higher derivatives. No mesh policy is introduced here.
"""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_BLOCK = 128


def _out_shape(shape, *inputs):
    """Declare outputs varying over exactly the manual axes of their inputs.

    Coefficients are token-local, so each output varies across ``shard_map``
    axes iff an input does. Declaring this lets ``check_vma=True`` callers
    avoid a conservative reverse-mode all-reduce over replicated axes.
    """
    varying = frozenset().union(*(jax.typeof(x).manual_axis_type.varying for x in inputs))
    return jax.ShapeDtypeStruct(shape, jnp.float32, manual_axis_type=jax.sharding.ManualAxisType(varying=varying))


def _gates(z, n_iters, eps):
    """Pure fp32 packed arithmetic used to derive the local transpose."""
    pre = jax.nn.sigmoid(z[:4, :]) + eps
    post = 2 * jax.nn.sigmoid(z[4:8, :])
    matrix = jax.nn.softmax(z[8:, :].reshape(4, 4, _BLOCK), axis=1) + eps
    matrix = matrix / (jnp.sum(matrix, axis=0, keepdims=True) + eps)
    for _ in range(n_iters - 1):
        matrix = matrix / (jnp.sum(matrix, axis=1, keepdims=True) + eps)
        matrix = matrix / (jnp.sum(matrix, axis=0, keepdims=True) + eps)
    return pre, post, matrix


def _forward(logits, base, scale, n_iters, eps):
    """Pack tokens into lanes, compute all coefficients, and remove tail lanes."""
    tokens = logits.shape[0]
    padded = ((tokens + _BLOCK - 1) // _BLOCK) * _BLOCK
    packed = jnp.pad(logits.T, ((0, 0), (0, padded - tokens)))

    def kernel(lr, br, sr, pr, qr, cr):
        l, b, s = lr[...], br[...], sr[...]
        pre_logits = l[:4, :] * s[0, 0] + b[:4, :]
        post_logits = l[4:8, :] * s[1, 0] + b[4:8, :]
        mix_logits = (l[8:, :] * s[2, 0] + b[8:, :]).reshape(4, 4, _BLOCK)
        pr[...] = jax.nn.sigmoid(pre_logits) + eps
        qr[...] = 2 * jax.nn.sigmoid(post_logits)
        matrix = jnp.exp(mix_logits - jnp.max(mix_logits, axis=1, keepdims=True))
        matrix = matrix / jnp.sum(matrix, axis=1, keepdims=True) + eps
        matrix = matrix / (jnp.sum(matrix, axis=0, keepdims=True) + eps)

        def body(_, value):
            value = value / (jnp.sum(value, axis=1, keepdims=True) + eps)
            return value / (jnp.sum(value, axis=0, keepdims=True) + eps)

        cr[...] = jax.lax.fori_loop(0, n_iters - 1, body, matrix)

    pre, post, comb = pl.pallas_call(
        kernel,
        out_shape=(
            _out_shape((4, padded), packed, base, scale),
            _out_shape((4, padded), packed, base, scale),
            _out_shape((4, 4, padded), packed, base, scale),
        ),
        grid=(padded // _BLOCK,),
        in_specs=(
            pl.BlockSpec((24, _BLOCK), lambda t: (0, t)),
            pl.BlockSpec((24, 1), lambda t: (0, 0)),
            pl.BlockSpec((3, 1), lambda t: (0, 0)),
        ),
        out_specs=(
            pl.BlockSpec((4, _BLOCK), lambda t: (0, t)),
            pl.BlockSpec((4, _BLOCK), lambda t: (0, t)),
            pl.BlockSpec((4, 4, _BLOCK), lambda t: (0, 0, t)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="mhc_coefficients_packed",
    )(packed, base[:, None], scale[:, None])
    return pre.T[:tokens], post.T[:tokens], comb.transpose(2, 0, 1)[:tokens]


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def coefficients(logits, base, scale, n_iters, eps):
    """Four-stream local primitive with first-order reverse-mode differentiation."""
    return _forward(logits, base, scale, n_iters, eps)


def _fwd(logits, base, scale, n_iters, eps):
    return _forward(logits, base, scale, n_iters, eps), (logits, base, scale)


def _bwd(n_iters, eps, residual, cotangents):
    logits, base, scale = residual
    indices = jnp.asarray([0] * 4 + [1] * 4 + [2] * 16)
    z = logits * scale[indices] + base
    tokens = z.shape[0]
    pad = (-tokens) % _BLOCK
    packed = jnp.pad(z.T, ((0, 0), (0, pad)))
    dp, dq, dc = cotangents
    dp = jnp.pad(dp.T, ((0, 0), (0, pad)))
    dq = jnp.pad(dq.T, ((0, 0), (0, pad)))
    dc = jnp.pad(dc.transpose(1, 2, 0), ((0, 0), (0, 0), (0, pad)))

    def kernel(zr, pr, qr, cr, out):
        _, backward = jax.vjp(lambda value: _gates(value, n_iters, eps), zr[...])
        out[...] = backward((pr[...], qr[...], cr[...]))[0]

    dz = pl.pallas_call(
        kernel,
        grid=((tokens + pad) // _BLOCK,),
        in_specs=(
            pl.BlockSpec((24, _BLOCK), lambda t: (0, t)),
            pl.BlockSpec((4, _BLOCK), lambda t: (0, t)),
            pl.BlockSpec((4, _BLOCK), lambda t: (0, t)),
            pl.BlockSpec((4, 4, _BLOCK), lambda t: (0, 0, t)),
        ),
        out_specs=pl.BlockSpec((24, _BLOCK), lambda t: (0, t)),
        out_shape=_out_shape(packed.shape, packed, dp, dq, dc),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="mhc_coefficients_vjp_packed",
    )(packed, dp, dq, dc).T[:tokens]
    # Keep the original broadcast/scatter transpose and its fp32 accumulation.
    _, backward = jax.vjp(lambda l, b, s: l * s[indices] + b, logits, base, scale)
    return backward(dz)


coefficients.defvjp(_fwd, _bwd)
