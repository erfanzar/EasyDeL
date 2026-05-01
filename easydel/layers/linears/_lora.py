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

"""EasyDeL-specific LoRA wrappers.

EasyDeL frequently wraps ``ParallelLinear`` modules with LoRA and then routes
those wrapped modules through utilities that expect the original linear-layer
calling contract. In particular:

- the wrapped module may be called with extra positional or keyword arguments
  (for example ``w=...`` for tied LM-head projection), and
- chunked scoring / generation paths may call ``native_forward`` directly to
  avoid trace-context issues caused by rematerialization wrappers.

Spectrax's stock ``nn.LoRA`` implementation is perfectly fine as a generic LoRA
layer, but it does not preserve those EasyDeL-specific conventions. ``eLoRA``
is therefore a thin compatibility wrapper rather than a different LoRA
algorithm.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from spectrax import nn


class eLoRA(nn.LoRA):
    """LoRA wrapper that behaves like the wrapped EasyDeL linear module.

    The underlying LoRA math is unchanged: the wrapper computes the low-rank
    update ``x @ A @ B`` and, when a base module is present, adds the base
    module output. The value this subclass adds is interface compatibility:

    - ``forward`` forwards ``*args`` / ``**kwargs`` to the wrapped module.
    - ``native_forward`` is exposed for trace-safe LM-head projection paths.

    This lets a LoRA-wrapped ``ParallelLinear`` keep working in places that
    assume they are still talking to an EasyDeL linear layer.
    """

    dtype: jnp.dtype | None = None

    def __init__(
        self,
        d_in: int,
        rank: int,
        d_out: int,
        *,
        base_module: tp.Callable[..., tp.Any] | None = None,
        alpha: float | None = None,
        rngs: spx.Rngs | int | None = None,
        a_init: tp.Callable | None = None,
        b_init: tp.Callable | None = None,
        dtype: jnp.dtype | None = None,
    ) -> None:
        """Initialize the LoRA wrapper around an EasyDeL linear module.

        Args:
            d_in: Input feature dimension of the wrapped module.
            rank: Low-rank LoRA bottleneck size. Smaller values trade
                expressivity for memory and compute.
            d_out: Output feature dimension of the wrapped module.
            base_module: Optional underlying module (typically a
                ``ParallelLinear``) whose output is added to the LoRA update.
                When ``None`` only the LoRA delta is returned.
            alpha: Optional LoRA scaling coefficient. When ``None`` the
                Spectrax default (no scaling) is used.
            rngs: PRNG state used to initialize the LoRA factors.
            a_init: Initializer for the down-projection ``A`` factor of shape
                ``(d_in, rank)``. Defaults to Spectrax's built-in initializer.
            b_init: Initializer for the up-projection ``B`` factor of shape
                ``(rank, d_out)``. Defaults to zeros so the initial residual
                is exactly the base-module output.
            dtype: Computation dtype. When ``None`` no casting is performed.

        Returns:
            None.
        """
        super().__init__(
            d_in=d_in,
            rank=rank,
            d_out=d_out,
            base_module=base_module,
            alpha=alpha,
            rngs=rngs,
            a_init=a_init,
            b_init=b_init,
            dtype=dtype,
        )
        self.dtype = dtype

    @staticmethod
    def _maybe_cast(x: jax.Array, dtype: jnp.dtype | None) -> jax.Array:
        """Cast ``x`` to ``dtype`` if it is provided.

        Args:
            x: Input array.
            dtype: Target dtype, or ``None`` to skip casting.

        Returns:
            ``x`` cast to ``dtype`` if ``dtype is not None``, otherwise ``x``
            itself unchanged.
        """
        return x.astype(dtype) if dtype is not None else x

    def forward(self, x: jax.Array, *args, **kwargs):
        """Apply the LoRA update and delegate extra call arguments to the base module.

        Spectrax's stock ``nn.LoRA.forward`` only accepts the input array. EasyDeL
        sometimes passes additional arguments through linear layers, most notably
        ``w=...`` when reusing embedding weights for tied LM-head projection.
        Forwarding the extra arguments here keeps the wrapped base module's API
        intact while still adding the LoRA residual.
        """
        x = self._maybe_cast(x, self.dtype)
        lora_a = self._maybe_cast(self.lora_a[...], self.dtype)
        lora_b = self._maybe_cast(self.lora_b[...], self.dtype)
        out = x @ lora_a @ lora_b
        if self.base_module is not None:
            if not callable(self.base_module):
                raise ValueError("`self.base_module` must be callable.")
            out += self.base_module(x, *args, **kwargs)
        return out

    def native_forward(
        self,
        inputs: jax.Array,
        *,
        w: jax.Array | None = None,
    ) -> jax.Array:
        """Project through LoRA using EasyDeL's trace-safe linear-layer contract.

        EasyDeL's chunked LM-head utilities call ``native_forward`` directly
        because that path avoids the module-call machinery used by
        rematerialized heads inside nested JAX traces. A LoRA wrapper around the
        LM head therefore also needs to provide ``native_forward`` so those
        utilities can continue to bypass the regular module-call path.

        Args:
            inputs: Hidden states or activations to project.
            w: Optional external weight matrix supplied by tied-embedding paths.
                When present, it is forwarded to the wrapped base module's
                ``native_forward`` implementation if supported.

        Returns:
            The sum of the LoRA residual projection and the wrapped base-module
            projection.
        """
        inputs = self._maybe_cast(inputs, self.dtype)
        lora_a = self._maybe_cast(self.lora_a[...], self.dtype)
        lora_b = self._maybe_cast(self.lora_b[...], self.dtype)
        out = inputs @ lora_a @ lora_b
        if self.base_module is not None:
            if hasattr(self.base_module, "native_forward"):
                out += self.base_module.native_forward(inputs=inputs, w=w)
            elif w is None:
                out += self.base_module(inputs)
            else:
                out += self.base_module(inputs, w=w)
        return out
