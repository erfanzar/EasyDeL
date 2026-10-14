# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Exact 64-bit unsigned integer arithmetic on ``uint32`` limb pairs.

JAX runs with ``jax_enable_x64`` disabled, so ``jnp.uint64`` silently degrades
to 32 bits. Hashed n-gram embeddings (Qwen4) need genuine 64-bit width: token
ids are multiplied by odd multipliers close to ``2**63 / vocab_size`` and the
products land within a fraction of a percent of the ``int64`` ceiling before
being XOR-folded and reduced modulo a prime. Doing that in 32 bits does not
raise -- it wraps, and every n-gram hashes to the wrong row.

Enabling x64 globally would fix it, but that flag changes default dtypes for
every model, trainer and kernel in the repo, and 64-bit integer ops are
emulated (slowly) on TPU regardless. So the width is emulated locally instead:
a ``u64`` is a ``(hi, lo)`` pair of ``uint32`` arrays, and the three operations
the hash needs are implemented over that representation.

All functions are elementwise and broadcast like ordinary arrays.
"""

from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array

U64 = tuple[Array, Array]  # (hi, lo), both uint32

_U32 = jnp.uint32
# Keep module constants host-native: constructing a JAX array during import can
# initialize/compile a TPU backend before the caller has established its mesh.
_MASK16 = 0xFFFF

__all__ = ("U64", "from_uint32", "mul_by_scalar", "to_python", "u64_mod_small", "u64_xor")


def from_uint32(value: Array) -> U64:
    """Widen a ``uint32`` array to a ``(hi, lo)`` pair."""
    return jnp.zeros_like(value, dtype=_U32), value.astype(_U32)


def _mul32(a: Array, b: Array) -> U64:
    """Exact ``uint32 x uint32 -> uint64`` product as ``(hi, lo)``.

    Schoolbook over 16-bit halves. The cross term ``a0*b1 + a1*b0`` can exceed
    ``2**32``; its carry is recovered by comparing the wrapped sum against one
    addend, which is valid precisely because ``uint32`` addition wraps.
    """
    a0, a1 = a & _MASK16, a >> _U32(16)
    b0, b1 = b & _MASK16, b >> _U32(16)

    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1

    mid = p01 + p10
    mid_carry = (mid < p01).astype(_U32)  # wrapped -> one carry out of bit 32

    lo = p00 + (mid << _U32(16))
    lo_carry = (lo < p00).astype(_U32)

    hi = p11 + (mid >> _U32(16)) + (mid_carry << _U32(16)) + lo_carry
    return hi, lo


def mul_by_scalar(value: Array, scalar: int) -> U64:
    """Multiply a ``uint32`` array by a Python ``int`` up to 64 bits wide.

    Args:
        value: ``uint32`` array (token ids).
        scalar: Non-negative Python integer below ``2**64``.

    Returns:
        The exact 64-bit product as ``(hi, lo)``; bits above 64 are discarded,
        matching two's-complement ``int64`` wrap-around.
    """
    if scalar < 0 or scalar >= 1 << 64:
        raise ValueError(f"scalar must fit in 64 unsigned bits, got {scalar}")
    s_lo = _U32(scalar & 0xFFFFFFFF)
    s_hi = _U32((scalar >> 32) & 0xFFFFFFFF)

    hi, lo = _mul32(value, jnp.full_like(value, s_lo, dtype=_U32))
    # value * s_hi contributes only to the high word (it is already shifted 32).
    hi = hi + value.astype(_U32) * jnp.full_like(value, s_hi, dtype=_U32)
    return hi, lo


def u64_xor(a: U64, b: U64) -> U64:
    """Bitwise XOR of two 64-bit values."""
    return a[0] ^ b[0], a[1] ^ b[1]


def u64_mod_small(value: U64, modulus: Array) -> Array:
    """Reduce a 64-bit value modulo a small ``uint32`` modulus.

    Horner over 8-bit digits, most significant first. The running remainder is
    below ``modulus``; the moduli here are primes just above 20,000,000
    (``< 2**25``), so ``remainder * 256 + digit`` stays below ``2**33`` -- which
    would overflow ``uint32``. The step is therefore split: shift by 4 bits
    twice, reducing in between, keeping every intermediate under ``2**29``.

    Args:
        value: ``(hi, lo)`` pair to reduce.
        modulus: ``uint32`` modulus, broadcastable against ``value``.

    Returns:
        ``value % modulus`` as ``uint32``.
    """
    hi, lo = value
    m = modulus.astype(_U32)
    rem = jnp.zeros_like(m, dtype=_U32)

    for word in (hi, lo):
        for shift in range(28, -1, -4):
            digit = (word >> _U32(shift)) & _U32(0xF)
            rem = ((rem << _U32(4)) + digit) % m
    return rem


def to_python(value: U64) -> Array:
    """Reassemble ``(hi, lo)`` into a Python-int array (host-side, for tests)."""
    import numpy as np

    hi = np.asarray(value[0], dtype=np.uint64)
    lo = np.asarray(value[1], dtype=np.uint64)
    return (hi << np.uint64(32)) | lo
