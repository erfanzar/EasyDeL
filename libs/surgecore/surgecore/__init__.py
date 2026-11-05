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
numpy staging arrays with one forward window's step inputs, with the GIL released.
"""

try:
    from ._native import HostPayloadBuilder, __version__, request_distribution_prefix
except ImportError as exc:  # pragma: no cover - depends on the build
    raise ImportError(
        "surgecore._native (EasyDeL's compiled host library) is not built. Build the wheel with "
        "`maturin build --release` in libs/surgecore and install it, or extract its "
        "surgecore/_native.abi3.so into libs/surgecore/surgecore/ (see libs/surgecore/README.md)."
    ) from exc

__all__ = ("HostPayloadBuilder", "__version__", "request_distribution_prefix")
