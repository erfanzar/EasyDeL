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

"""Mode-bound :class:`OperationExecutor` wrapper for prefill/decode dispatch.

The executor bundles up to three :class:`BaseOperation` instances — a
prefill implementation, a decode implementation, and a shared "mixin"
implementation — and resolves which one services a given
:class:`ExecutionMode` request. It centralises three concerns that
otherwise pollute every caller of the attention layer:

* **Mode dispatch.** :meth:`OperationExecutor.get_operation` returns the
  right operation for ``PREFILL``, ``DECODE``, or ``MIXED`` with the
  documented prefill→mixin and decode→prefill→mixin fallbacks.
* **Requirement aggregation.** :meth:`OperationExecutor.get_requirements`
  and :meth:`OperationExecutor.get_combined_requirements` apply
  instance-level overrides (via
  :meth:`OperationImpl.get_instance_requirements`) and combine prefill /
  decode requirements (intersection of supported caches, union of
  metadata fields).
* **Discovery.** Constructed via either
  :meth:`OperationExecutor.from_flexible_attention` or
  :meth:`OperationExecutor.from_operations`, allowing
  ``iter_module_search``-style traversals to find executors without
  knowing the concrete attention module type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easydel.operations.requirements import ExecutionMode, OperationRequirements

if TYPE_CHECKING:
    from easydel.operations._base_operation import BaseOperation

__all__ = ["OperationExecutor"]


@dataclass
class OperationExecutor:
    """Mode-bound bundle of attention operations with fallback resolution.

    Wraps up to three :class:`BaseOperation` instances and exposes a
    uniform interface for selecting the right one for a given execution
    mode and for combining their requirements.

    Resolution rules:

    * **Prefill.** :attr:`prefill_operation` returns ``prefill_impl`` if
      set, otherwise ``mixin_impl``.
    * **Decode.** :attr:`decode_operation` returns ``decode_impl`` if set;
      otherwise ``prefill_impl``; otherwise ``mixin_impl``.
    * **Mixed.** :meth:`get_operation` for ``ExecutionMode.MIXED``
      delegates to :attr:`prefill_operation`.

    Hence: explicit per-mode impls always take precedence over
    ``mixin_impl``, and decode silently borrows the prefill operation when
    no decode-specific one is registered.

    Example:
        >>> executor = OperationExecutor.from_flexible_attention(flex_attn)
        >>> prefill_op = executor.get_operation(ExecutionMode.PREFILL)
        >>> decode_op = executor.get_operation(ExecutionMode.DECODE)
        >>> reqs = executor.get_combined_requirements()

    Attributes:
        prefill_impl (BaseOperation | None): Operation dedicated to prefill.
        decode_impl (BaseOperation | None): Operation dedicated to decode;
            falls back to ``prefill_impl`` when absent.
        mixin_impl (BaseOperation | None): Shared operation used when no
            per-mode impl is set.
    """

    prefill_impl: BaseOperation | None = None
    decode_impl: BaseOperation | None = None
    mixin_impl: BaseOperation | None = None

    @property
    def prefill_operation(self) -> BaseOperation | None:
        """Resolve the operation that should service prefill requests.

        Returns:
            BaseOperation | None: ``prefill_impl`` if set, otherwise
            ``mixin_impl``. ``None`` when neither is provided.
        """
        if self.prefill_impl is not None:
            return self.prefill_impl
        return self.mixin_impl

    @property
    def decode_operation(self) -> BaseOperation | None:
        """Resolve the operation that should service decode requests.

        Returns:
            BaseOperation | None: ``decode_impl`` if set; otherwise falls back
            to ``prefill_impl`` and finally ``mixin_impl``. ``None`` when no
            operation is available.
        """
        if self.decode_impl is not None:
            return self.decode_impl
        if self.prefill_impl is not None:
            return self.prefill_impl  # Fallback to prefill if decode not set
        return self.mixin_impl

    def get_operation(self, mode: ExecutionMode) -> BaseOperation | None:
        """Return the operation that should service requests in ``mode``.

        ``MIXED`` is treated as a request for the prefill operation, since
        a prefill-capable operator can always also do a single decode step.

        Args:
            mode: Execution mode to resolve. One of ``ExecutionMode.PREFILL``,
                ``ExecutionMode.DECODE``, or ``ExecutionMode.MIXED``.

        Returns:
            The resolved :class:`BaseOperation`, or ``None`` if no
            operation has been registered for ``mode`` (after applying
            the fallback rules described on the class).
        """
        if mode == ExecutionMode.PREFILL:
            return self.prefill_operation
        elif mode == ExecutionMode.DECODE:
            return self.decode_operation
        else:  # MIXED
            return self.prefill_operation  # Default to prefill for mixed

    def get_requirements(self, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Resolve the requirements declared by the operation for ``mode``.

        Prefers :meth:`OperationImpl.get_instance_requirements` so that
        instance-level overrides (such as ``requires_cache=False`` on a
        vision encoder) are honoured; falls back to the class-level
        ``get_requirements`` when the operation does not implement the
        instance variant.

        Args:
            mode: Execution mode whose operation should be queried.

        Returns:
            OperationRequirements describing what the resolved operation
            needs. When no operation is registered for ``mode``,
            :meth:`OperationRequirements.default` is returned.
        """
        op = self.get_operation(mode)
        if op is not None:
            # Use instance requirements to respect metadata overrides
            if hasattr(op, "get_instance_requirements"):
                return op.get_instance_requirements(mode)
            return op.get_requirements(mode)
        return OperationRequirements.default()

    def get_combined_requirements(self) -> OperationRequirements:
        """Merge prefill and decode requirements into a single declaration.

        Computes the intersection of supported cache types (the engine can
        only pick a cache that both prefill and decode support) and the
        union of required metadata fields (both stages must have what
        they need). Instance-level overrides are honoured via
        :meth:`OperationImpl.get_instance_requirements` when available.

        Returns:
            OperationRequirements representing the merged needs.

        Raises:
            RuntimeError: Internal invariant violation when both per-stage
                requirements end up ``None`` after the union step. This
                indicates a logic error rather than user input.
        """
        prefill_reqs = None
        decode_reqs = None

        prefill_op = self.prefill_operation
        decode_op = self.decode_operation

        if prefill_op is not None:
            # Use instance requirements to respect metadata overrides
            if hasattr(prefill_op, "get_instance_requirements"):
                prefill_reqs = prefill_op.get_instance_requirements(ExecutionMode.PREFILL)
            else:
                prefill_reqs = prefill_op.get_requirements(ExecutionMode.PREFILL)
        if decode_op is not None:
            # Use instance requirements to respect metadata overrides
            if hasattr(decode_op, "get_instance_requirements"):
                decode_reqs = decode_op.get_instance_requirements(ExecutionMode.DECODE)
            else:
                decode_reqs = decode_op.get_requirements(ExecutionMode.DECODE)

        if prefill_reqs is None and decode_reqs is None:
            return OperationRequirements.default()
        if prefill_reqs is None:
            if decode_reqs is None:
                raise RuntimeError("decode_reqs must not be None when prefill_reqs is None")
            return decode_reqs
        if decode_reqs is None:
            return prefill_reqs

        # Combine requirements (intersection of cache, union of metadata)
        return prefill_reqs | decode_reqs

    @property
    def requires_cache(self) -> bool:
        """Whether the underlying operations need a KV cache.

        Returns:
            bool: ``True`` when the combined prefill/decode requirements ask
            for a cache (typical for autoregressive decoding); ``False`` for
            cacheless operators such as some vision encoders.
        """
        reqs = self.get_combined_requirements()
        return reqs.cache.requires_cache

    @property
    def has_separate_decode(self) -> bool:
        """Whether decode is dispatched to a distinct operation from prefill.

        Returns:
            bool: ``True`` only when both ``prefill_impl`` and ``decode_impl``
            are set and refer to different objects (e.g. flash for prefill,
            ragged-page for decode).
        """
        return (
            self.decode_impl is not None and self.prefill_impl is not None and self.decode_impl is not self.prefill_impl
        )

    @property
    def is_valid(self) -> bool:
        """Whether at least one backing operation is configured.

        Returns:
            bool: ``True`` if any of ``prefill_impl``, ``decode_impl``, or
            ``mixin_impl`` is non-``None``.
        """
        return self.prefill_impl is not None or self.decode_impl is not None or self.mixin_impl is not None

    def get_operation_name(self, mode: ExecutionMode = ExecutionMode.MIXED) -> str | None:
        """Return the registered name of the operation servicing ``mode``.

        When an operation declares multiple aliases via
        :meth:`BaseOperation.get_impl_name`, the first alias is returned.

        Args:
            mode: Execution mode whose operation should be named.

        Returns:
            The implementation name, or ``None`` if no operation is
            registered for ``mode``.
        """
        op = self.get_operation(mode)
        if op is not None:
            name = op.get_impl_name()
            if isinstance(name, tuple):
                return name[0]
            return name
        return None

    @classmethod
    def from_flexible_attention(cls, flex_attn) -> OperationExecutor:
        """Construct an executor from a ``FlexibleAttentionModule`` instance.

        Reads ``flex_attn.impl`` as the prefill operation and
        ``flex_attn.impl_decode`` as the decode operation; both default to
        ``None`` so an executor can still be built from a partially
        populated module.

        Args:
            flex_attn: A ``FlexibleAttentionModule`` instance carrying
                attention operation implementations on its ``impl`` /
                ``impl_decode`` attributes.

        Returns:
            OperationExecutor wrapping the module's operations.
        """
        return cls(
            prefill_impl=getattr(flex_attn, "impl", None),
            decode_impl=getattr(flex_attn, "impl_decode", None),
            mixin_impl=None,
        )

    @classmethod
    def from_operations(
        cls,
        prefill: BaseOperation | None = None,
        decode: BaseOperation | None = None,
        mixin: BaseOperation | None = None,
    ) -> OperationExecutor:
        """Construct an executor directly from operation instances.

        Args:
            prefill: Operation to use for prefill mode.
            decode: Operation to use for decode mode (falls back to
                ``prefill`` then ``mixin`` per the class-level resolution
                rules).
            mixin: Shared operation used for both modes when no per-mode
                operation is supplied.

        Returns:
            OperationExecutor wrapping the supplied operations.
        """
        return cls(
            prefill_impl=prefill,
            decode_impl=decode,
            mixin_impl=mixin,
        )
