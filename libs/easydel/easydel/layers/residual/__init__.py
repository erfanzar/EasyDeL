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

"""Multi-stream residual connections (hyper-connections).

Standard transformers carry a single residual stream; hyper-connection
architectures carry ``hc_count`` parallel streams and learn how sub-layers
read from and write to them. This package holds the stream plumbing shared by
those architectures:

- :func:`expand_streams` widens a single hidden sequence into ``hc_count``
  concatenated streams (the model-entry side).
- :class:`GatedResidual` is the Qwen4-style read/write gate: a low-rank
  element-wise read mixer plus a per-branch scalar write gate.
- :func:`inject_streams` applies the write side back onto the streams.

DeepSeek-V4's ``DeepseekV4HyperConnection`` is deliberately NOT folded into
this package: it shares the N-stream plumbing concept but its mixing law is a
Sinkhorn-projected doubly-stochastic matrix rather than low-rank gates, and
forcing one law onto both families would distort each. See
``.claude/projects/qwen4-port.md`` (Tier 2b) for the rationale.
"""

from ._gated import GatedResidual, expand_streams, inject_streams

__all__ = ("GatedResidual", "expand_streams", "inject_streams")
