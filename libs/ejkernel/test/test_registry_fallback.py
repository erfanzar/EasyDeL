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

"""Tests for kernel_registry priority/platform/backend resolution and fallback termination.

These tests only register dummy callables and exercise lookup logic; they do no
array computation, so they are marked ``cpu_ok``.
"""

import jax
import pytest
from ejkernel.kernels._registry import Backend, KernelRegistry, Platform

pytestmark = pytest.mark.cpu_ok


def _dummy(*args, **kwargs):
    return None


def _non_default_backends() -> tuple[Backend, Backend]:
    """Two distinct concrete backends, neither of which is the running JAX default backend."""
    default = jax.default_backend()
    others = [b for b in (Backend.TPU, Backend.GPU, Backend.CPU) if b.value != default]
    return others[0], others[1]


def test_xla_lookup_miss_raises_value_error_not_recursion():
    """A lookup miss for platform=XLA must terminate with ValueError.

    Regression test: the XLA -> Backend.ANY fallback used to be a recursive
    ``self.get`` call with identical arguments, so any algorithm whose XLA
    implementations were registered for specific backends (or missing) raised
    RecursionError instead of the documented ValueError.
    """
    registry = KernelRegistry()
    registered, queried = _non_default_backends()

    @registry.register("test_only_one_backend_xla", Platform.XLA, registered)
    def _impl(q):  # pragma: no cover - trivial
        return q

    with pytest.raises(ValueError, match="No implementation found"):
        registry.get("test_only_one_backend_xla", platform=Platform.XLA, backend=queried)

    # Backend.ANY falls back to jax.default_backend(), which is neither registered backend.
    with pytest.raises(ValueError, match="No implementation found"):
        registry.get("test_only_one_backend_xla", platform=Platform.XLA, backend=Backend.ANY)

    # An algorithm with no XLA implementation at all must also terminate.
    registry.register("test_pallas_only", Platform.PALLAS, registered)(_dummy)
    with pytest.raises(ValueError, match="No implementation found"):
        registry.get("test_pallas_only", platform=Platform.XLA)


def test_unknown_algorithm_raises_value_error():
    registry = KernelRegistry()
    with pytest.raises(ValueError, match="No implementation found"):
        registry.get("definitely_not_registered", platform=Platform.XLA, backend=Backend.ANY)


def test_priority_order_and_any_backend_wildcard():
    """Highest priority wins; Backend.ANY matches any backend query."""
    registry = KernelRegistry()

    def _xla_any(*args, **kwargs):
        return "xla_any"

    def _pallas_tpu(*args, **kwargs):
        return "pallas_tpu"

    def _pallas_tpu_high(*args, **kwargs):
        return "pallas_tpu_high"

    # ``register`` returns the validation wrapper that the registry stores, so compare against it.
    xla_any = registry.register("prio_algo", Platform.XLA, Backend.ANY, priority=0)(_xla_any)
    registry.register("prio_algo", Platform.PALLAS, Backend.TPU, priority=10)(_pallas_tpu)
    pallas_high = registry.register("prio_algo", Platform.PALLAS, Backend.TPU, priority=20)(_pallas_tpu_high)

    assert registry.get("prio_algo", platform=Platform.PALLAS, backend=Backend.TPU) is pallas_high
    # Backend.ANY registrations are wildcards: an XLA query for a concrete backend resolves to them.
    assert registry.get("prio_algo", platform=Platform.XLA, backend=Backend.TPU) is xla_any
    assert registry.get("prio_algo", platform=Platform.XLA, backend=Backend.CPU) is xla_any
