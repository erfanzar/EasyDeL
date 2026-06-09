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


"""High-performance JAX operation framework with automatic configuration management.

ejkernel.ops provides a sophisticated framework for implementing JAX operations with
automatic performance tuning, configuration caching, and flexible execution strategies.

Key Features:
- Automatic configuration selection and caching
- Performance autotuning with persistent storage
- Custom gradient support via VJP
- Batch processing with shared configuration
- Device-aware operation management
- Comprehensive profiling and debugging tools

Main Components:
- Kernel: Base class for implementing operations
- Executor: Main execution engine with config selection
- ConfigSelectorChain: Multi-tier configuration selection
- ConfigCache/PersistentCache: Configuration storage
- Autotuning tools and batch processing utilities

Example Usage:
    ```python
    from ejkernel.ops import Kernel, Executor, ConfigSelectorChain, ConfigCache

    @dataclass
    class MyConfig:
        algorithm: str = "default"

    class MyKernel(Kernel[MyConfig, jax.Array]):
        def run(self, a, b, *, cfg: MyConfig, **kwargs):
            return jax.numpy.dot(a, b)

        def heuristic_cfg(self, inv):
            return MyConfig()


    cache = ConfigCache()
    selector = ConfigSelectorChain(cache)
    executor = Executor(selector)


    kernel = MyKernel("my_op")
    result = executor(kernel, a, b)
    ```
"""

from .config import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    PersistentCache,
    Tuner,
    forward_autotune_only,
    overlay_cache,
    policy_override,
)
from .core import Cfg, Invocation, Kernel, Out
from .execution import (
    AutotuneData,
    Autotuner,
    AutotuningResult,
    ConfigChooser,
    Entry,
    Executor,
    FNAutotuner,
    Measurement,
    autotune,
    autotune_recorded,
    benchmark,
    pmap_with_config,
    vmap_with_config,
)
from .registry import get_invocations, record_invocation
from .utils import (
    BwdParams,
    FwdParams,
    abstractify,
    default_key_builder_with_sharding,
    device_fingerprint,
    device_kind,
    extract_labels_from_hlo_text,
    find_labels_in_lowered,
    from_json,
    get_device_platform,
    label,
    labels_to_configs,
    sharding_fingerprint,
    short_hash,
    stable_json,
    to_json,
)

__all__ = (
    "AutotuneData",
    "AutotunePolicy",
    "Autotuner",
    "AutotuningResult",
    "BwdParams",
    "Cfg",
    "ConfigCache",
    "ConfigChooser",
    "ConfigSelectorChain",
    "Entry",
    "Executor",
    "FNAutotuner",
    "FwdParams",
    "Invocation",
    "Kernel",
    "Measurement",
    "Out",
    "PersistentCache",
    "Tuner",
    "abstractify",
    "autotune",
    "autotune_recorded",
    "benchmark",
    "default_key_builder_with_sharding",
    "device_fingerprint",
    "device_kind",
    "extract_labels_from_hlo_text",
    "find_labels_in_lowered",
    "forward_autotune_only",
    "from_json",
    "get_device_platform",
    "get_invocations",
    "label",
    "labels_to_configs",
    "overlay_cache",
    "pmap_with_config",
    "policy_override",
    "record_invocation",
    "sharding_fingerprint",
    "short_hash",
    "stable_json",
    "to_json",
    "vmap_with_config",
)
