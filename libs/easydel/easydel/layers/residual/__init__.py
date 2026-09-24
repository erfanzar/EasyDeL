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

- :class:`ManifoldHyperConnection` is the DeepSeek/GLM-style scalar read/write
  gate and Sinkhorn-projected doubly-stochastic stream mixer, configured by
  :class:`ManifoldHyperConnectionConfig` without model dependencies.
- :func:`manifold_residual_write` applies its transposed mixer and write gates
  to explicit ``[batch, seq, hc, hidden]`` streams.
- :class:`ManifoldHyperHead` learns the final collapse; :class:`MeanHyperHead`
  and :func:`mean_hyper_head` provide a parameter-free mean alternative.

The gated and manifold variants belong to a common residual family, but their
mixing laws and stream layouts remain distinct rather than forced into one
parameterization.
"""

from ._gated import GatedResidual, expand_streams, inject_streams
from ._manifold import (
    HyperStreamSharding,
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    ManifoldHyperHead,
    MeanHyperHead,
    manifold_residual_write,
    mean_hyper_head,
)

__all__ = (
    "GatedResidual",
    "HyperStreamSharding",
    "ManifoldHyperConnection",
    "ManifoldHyperConnectionConfig",
    "ManifoldHyperHead",
    "MeanHyperHead",
    "expand_streams",
    "inject_streams",
    "manifold_residual_write",
    "mean_hyper_head",
)
