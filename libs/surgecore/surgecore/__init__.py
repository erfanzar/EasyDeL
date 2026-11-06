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

"""surgecore: EasyDeL's native host library for the eSurge inference engine.

The compiled extension module ``surgecore._native`` is built from this package's Rust crate with maturin. It holds
the per-step host computation of eSurge's runner: :class:`HostPayloadBuilder` fills a batch preparer's preallocated
numpy staging arrays with one forward window's step inputs, with the GIL released. :class:`KvCore` is the KV core of
one page pool (LRU free list, prefix map, cache-group rules, transactional allocation) and :class:`SnapshotIndex`
the recurrent-snapshot slot index. :class:`EngineCore` is eSurge's host state machine: the request table, every
data-parallel rank's queues, budgets, KV pages, prefix cache and snapshots, and the scheduler over them, with
speculative drafts; ``schedule()`` returns a :class:`Plan` and ``commit()`` turns the runner output into
:class:`Events`.
"""

try:
    from ._native import (
        PLAN_ASYNC_SAFE,
        PLAN_HAS_PREFILL,
        PLAN_HAS_VERIFY,
        PLAN_PAGES_CHANGED,
        PLAN_STRUCTURED_PENDING,
        CoreClosed,
        EngineCore,
        Events,
        HostPayloadBuilder,
        InvariantError,
        KvCore,
        Plan,
        SnapshotIndex,
        __version__,
        request_distribution_prefix,
    )
except ImportError as exc:  # pragma: no cover - depends on the build
    raise ImportError(
        "surgecore._native (EasyDeL's compiled host library) is not built. Build the wheel with "
        "`maturin build --release` in libs/surgecore and install it, or extract its "
        "surgecore/_native.abi3.so into libs/surgecore/surgecore/ (see libs/surgecore/README.md)."
    ) from exc

__all__ = (
    "PLAN_ASYNC_SAFE",
    "PLAN_HAS_PREFILL",
    "PLAN_HAS_VERIFY",
    "PLAN_PAGES_CHANGED",
    "PLAN_STRUCTURED_PENDING",
    "CoreClosed",
    "EngineCore",
    "Events",
    "HostPayloadBuilder",
    "InvariantError",
    "KvCore",
    "Plan",
    "SnapshotIndex",
    "__version__",
    "request_distribution_prefix",
)
