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

"""Autotune cache keys must distinguish dtype *classes* passed as arguments.

``jnp.float32`` / ``jnp.bfloat16`` are callable scalar-type objects; ``stable_json`` used to hit
its generic ``callable`` branch first and serialize every one of them as the same metaclass name,
so e.g. ``preferred_element_type=jnp.float32`` and ``=jnp.bfloat16`` shared one tuned config.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from ejkernel.ops.utils.fingerprint import short_hash, stable_json


def test_dtype_classes_hash_differently():
    assert short_hash({"dtype": jnp.float32}) != short_hash({"dtype": jnp.bfloat16})
    assert short_hash({"dtype": np.float32}) != short_hash({"dtype": np.float16})
    assert short_hash((jnp.int8,)) != short_hash((jnp.int32,))


def test_dtype_spellings_share_one_key():
    """Equivalent dtype spellings serialize identically (``jnp`` class, ``np`` class, ``np.dtype``)."""
    assert stable_json(jnp.float32) == stable_json(np.float32) == stable_json(np.dtype("float32")) == '"float32"'
    assert stable_json(jnp.bfloat16) == stable_json(jnp.dtype(jnp.bfloat16)) == '"bfloat16"'


def test_other_classes_do_not_collide():
    class A:
        pass

    class B:
        pass

    assert stable_json(A) != stable_json(B)
