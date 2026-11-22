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

"""The hashed n-gram embedding and the engram layout share one primality implementation."""

import pytest
from easydel.layers.embeddings._ngram import find_nth_prime_after
from easydel.layers.embeddings._primes import is_prime, next_prime_after
from easydel.layers.engram import engram_bucket_primes

pytestmark = pytest.mark.cpu_ok


def _trial_division(n: int) -> bool:
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n**0.5) + 1))


def test_is_prime_matches_trial_division_on_small_range():
    assert [n for n in range(2000) if is_prime(n)] == [n for n in range(2000) if _trial_division(n)]


def test_is_prime_handles_large_values():
    assert is_prime(2_147_483_647)  # Mersenne prime 2**31 - 1
    assert not is_prime(2_147_483_647 * 3)
    assert is_prime(1_000_003)
    assert not is_prime(1_000_001)  # 101 * 9901
    # A strong pseudoprime for the first twelve prime witnesses, still inside the documented exact range.
    assert not is_prime(399_165_290_221 * 798_330_580_441)


def test_next_prime_after_is_strictly_greater_and_honours_exclude():
    assert next_prime_after(7) == 11
    assert next_prime_after(8) == 11
    assert next_prime_after(7, exclude={11}) == 13


def test_find_nth_prime_after_counts_primes():
    assert find_nth_prime_after(1, 1) == 2
    assert find_nth_prime_after(1, 5) == 11
    assert find_nth_prime_after(10, 3) == 17


def test_engram_bucket_primes_are_distinct_primes_above_the_base():
    primes = engram_bucket_primes(2, 3, 4, 1000)
    flat = [p for layer in primes for per_ngram in layer for p in per_ngram]
    assert len(flat) == len(set(flat)) == 2 * 2 * 4
    assert all(p >= 1000 and _trial_division(p) for p in flat)
