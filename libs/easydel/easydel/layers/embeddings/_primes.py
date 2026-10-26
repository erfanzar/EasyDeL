# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Pure-Python primality helpers shared by the hashed-table layers.

Both the hashed n-gram embedding (:mod:`easydel.layers.embeddings._ngram`) and the engram prime-bucket layout
(:mod:`easydel.layers.engram._hashing`) size their tables with "the next prime above a base". They share one
primality test and one forward search here so the two layouts can never disagree on what counts as prime.
"""

from __future__ import annotations

import typing as tp

__all__ = ("is_prime", "next_prime_after")

_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test (exact for ``n < 3.3e24``).

    Args:
        n: Integer to test.

    Returns:
        ``True`` when ``n`` is prime.
    """
    if n < 2:
        return False
    for p in _WITNESSES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _WITNESSES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime_after(value: int, exclude: tp.Container[int] = ()) -> int:
    """Smallest prime strictly greater than ``value`` that is not in ``exclude``.

    Args:
        value: Exclusive lower bound of the search.
        exclude: Primes that must be skipped (for example ones already handed out to another bucket).

    Returns:
        The next admissible prime.
    """
    candidate = value + 1
    while not is_prime(candidate) or candidate in exclude:
        candidate += 1
    return candidate
