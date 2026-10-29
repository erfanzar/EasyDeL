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

"""Registered mHC gate/softmax/Sinkhorn operation."""

from jax import Array

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel

from ..base import detect_platform
from .configs import MHCCoefficientsConfig


class MHCCoefficients(Kernel[MHCCoefficientsConfig, tuple[Array, Array, Array]]):
    """Manifold read/write gates and residual mixer with fixed TPU tiling."""

    def __init__(self):
        super().__init__(op_id="mhc_coefficients")

    def get_impl(self, cfg: MHCCoefficientsConfig):
        """Resolve the registered backend using the usual platform selector."""
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(self, logits, base, scale, hc_mult=4, n_iters=20, eps=1e-6, platform=None, *, cfg):
        """Dispatch fp32 projection logits and parameters to the selected backend."""
        if platform is not None:
            cfg = MHCCoefficientsConfig(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)
        return self.get_impl(cfg)(logits, base, scale, hc_mult=hc_mult, n_iters=n_iters, eps=eps)

    def heuristic_cfg(self, inv: Invocation):
        """Fixed packed tiles: no tuning or persistent benchmark cache required."""
        return MHCCoefficientsConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation):
        """Return the single supported heuristic configuration."""
        return [self.heuristic_cfg(inv)]


_executor = Executor(ConfigSelectorChain(cache=ConfigCache(), policy=AutotunePolicy(allow_autotune=False)))


def mhc_coefficients(logits, base, scale, /, hc_mult=4, n_iters=20, eps=1e-6, *, platform=None, cfg=None):
    """Compute fp32 read gates, write gates and a Sinkhorn residual mixer.

    Args:
        logits: Float32 array ``[..., hc_mult * (hc_mult + 2)]``.
        base: Float32 offset vector matching the last logits dimension.
        scale: Float32 vector with three gate/mixer multipliers.
        hc_mult: Positive static residual-stream count.
        n_iters: Positive static Sinkhorn iteration count.
        eps: Positive finite offset applied at every normalization denominator.
        platform: Implementation override. Use ``'xla'`` for JVPs/higher AD;
            the optimized four-stream TPU path supports first-order reverse AD.
        cfg: Optional operation dispatch configuration.

    Returns:
        ``(pre, post, mixer)`` with shapes ``[..., hc_mult]``,
        ``[..., hc_mult]`` and ``[..., hc_mult, hc_mult]``.

    Raises:
        ValueError: Incompatible shapes or invalid static scalar settings.
        TypeError: Inputs are not float32.
    """
    return _executor(
        MHCCoefficients(), logits, base, scale, hc_mult=hc_mult, n_iters=n_iters, eps=eps, platform=platform, _cfg=cfg
    )
