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

"""Extensible speculative-decoding drafter registry and base class.

This module is the single extension point for speculative-decoding drafters
in EasyDeL. It is intentionally light-weight: it only depends on ``re``, JAX,
and the stdlib, so it can be imported eagerly by :mod:`easydel.infra` without
pulling in the (heavy) :mod:`easydel.inference` stack. The one bridge into
inference — resolving the built-in drafter builders — is performed lazily.

Two surfaces are provided:

* :class:`DrafterRegistry` (+ :func:`register_drafter`) — a real registry
  category peer to the operation / trainer registries. Drafter families are
  registered under a canonical slug plus aliases; :meth:`DrafterRegistry.resolve`
  maps a user-supplied ``method`` string to its builder. Built-in builders are
  registered by importing :mod:`easydel.inference.speculative` on first use.
* :class:`SpecDecodeBase` — the mix-in a model-native drafter inherits so that
  ``class Foo(EasyDeLBaseModule, SpecDecodeBase)`` composes. It carries the
  capability flags the eSurge runner probes and provides the shared
  argmax/sample ``draft(...)`` tail; a concrete drafter only implements
  :meth:`SpecDecodeBase._draft_logits`.
"""

from __future__ import annotations

import re
import typing as tp

import jax
import jax.numpy as jnp

if tp.TYPE_CHECKING:
    from easydel.inference.speculative import DraftStep

Builder = tp.Callable[..., tp.Any]


def _normalize_drafter_name(name: tp.Any) -> str:
    """Canonicalize a drafter name to a slug.

    Uses the exact normalization ``EasyDeLBaseModule.drafter`` and the
    ``eSurgeDrafterConfig`` validator use: lowercase, non-alphanumeric runs
    collapsed to ``_``, leading/trailing ``_`` stripped.

    Args:
        name: Raw drafter name / method string (or ``None``).

    Returns:
        The normalized slug (possibly empty).
    """
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")


class DrafterRegistry:
    """Registry mapping drafter method names/aliases to builder callables.

    A builder is any callable ``builder(target_model, *, num_draft_tokens, ...)``
    that returns a drafter implementing
    :class:`~easydel.inference.speculative.DrafterProtocol`. Registration is
    performed with the :meth:`register` decorator (or the module-level
    :func:`register_drafter`). The built-in families are registered lazily the
    first time the registry is queried, by importing
    :mod:`easydel.inference.speculative`.
    """

    _builders: tp.ClassVar[dict[str, Builder]] = {}
    _aliases: tp.ClassVar[dict[str, str]] = {}
    _builtins_loaded: tp.ClassVar[bool] = False

    @classmethod
    def register(cls, name: str, *aliases: str) -> tp.Callable[[Builder], Builder]:
        """Return a decorator registering a builder under ``name`` + ``aliases``.

        Args:
            name: Canonical drafter method name (normalized to a slug).
            *aliases: Additional accepted spellings, each normalized to a slug
                and pointed at the canonical name.

        Returns:
            A decorator that records the builder and returns it unchanged.
        """
        canonical = _normalize_drafter_name(name)
        if not canonical:
            raise ValueError(f"Drafter name {name!r} normalizes to an empty slug.")

        def _decorator(builder: Builder) -> Builder:
            cls._builders[canonical] = builder
            cls._aliases[canonical] = canonical
            for alias in aliases:
                slug = _normalize_drafter_name(alias)
                if slug:
                    cls._aliases[slug] = canonical
            return builder

        return _decorator

    @classmethod
    def _ensure_builtin_drafters(cls) -> None:
        """Import the module that registers the built-in drafter builders once.

        The flag is set before the import so a re-entrant :meth:`resolve` during
        that import does not recurse.
        """
        if cls._builtins_loaded:
            return
        cls._builtins_loaded = True
        import easydel.inference.speculative  # noqa: F401  (registers built-in drafters as a side effect)

    @classmethod
    def resolve(cls, name: str) -> Builder:
        """Resolve a drafter method name/alias to its builder callable.

        Args:
            name: Drafter method name or alias.

        Returns:
            The registered builder callable.

        Raises:
            ValueError: If ``name`` matches no registered drafter.
        """
        cls._ensure_builtin_drafters()
        slug = _normalize_drafter_name(name)
        canonical = cls._aliases.get(slug, slug)
        builder = cls._builders.get(canonical)
        if builder is None:
            raise ValueError(f"Unknown drafter method {name!r}. Registered drafters: {cls.names()}.")
        return builder

    @classmethod
    def has(cls, name: str) -> bool:
        """Return whether ``name`` (or an alias of it) is registered.

        Args:
            name: Drafter method name or alias.

        Returns:
            ``True`` if a builder is registered for the resolved name.
        """
        cls._ensure_builtin_drafters()
        slug = _normalize_drafter_name(name)
        return cls._aliases.get(slug, slug) in cls._builders

    @classmethod
    def names(cls) -> list[str]:
        """Return the sorted list of canonical registered drafter names."""
        cls._ensure_builtin_drafters()
        return sorted(cls._builders.keys())


_DRAFTER_REGISTRY = DrafterRegistry


def register_drafter(name: str, *aliases: str) -> tp.Callable[[Builder], Builder]:
    """Register a drafter builder under ``name`` and optional ``aliases``.

    Module-level convenience wrapping :meth:`DrafterRegistry.register`.

    Args:
        name: Canonical drafter method name.
        *aliases: Additional accepted names.

    Returns:
        A decorator recording the builder.
    """
    return DrafterRegistry.register(name, *aliases)


class SpecDecodeBase:
    """Mix-in base for speculative-decoding drafters.

    A model-native drafter is declared as
    ``class Foo(EasyDeLBaseModule, SpecDecodeBase)``. ``SpecDecodeBase`` is a
    plain ``object`` subclass with no metaclass so it composes with
    ``spx.Module``'s metaclass. Subclasses override :meth:`_draft_logits`
    (required) and may override the capability flags and optional hooks.

    The eSurge runner probes these class attributes (via ``getattr``) to drive
    draft/verify; they are the single documented surface a new drafter must
    reason about:

    Attributes:
        supports_return_full_log_probs: The drafter can materialize a dense
            ``(batch, vocab)`` distribution for distribution-correct sampled
            speculative decoding.
        supports_prefix_draft: The drafter accepts a multi-token prefix (rather
            than a single seed token) to stabilize the first draft step.
        requires_target_kv_cache: The drafter cross-attends to the target's
            per-layer K/V (e.g. the Gemma4 assistant) and must be fed a target
            K/V payload.
        num_draft_tokens: Speculative tokens proposed per verify window. A plain
            ``int`` — never a device array (mutable arrays on an ``spx.Module``
            instance are silently dropped under JIT).
    """

    supports_return_full_log_probs: bool = True
    supports_prefix_draft: bool = False
    requires_target_kv_cache: bool = False
    num_draft_tokens: int = 1

    def _draft_logits(
        self,
        input_ids: jax.Array,
        *,
        target_hidden_states: jax.Array | None,
        position_ids: jax.Array | None = None,
        target_kv_cache: tp.Any | None = None,
        attention_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Return drafter logits and (optionally) drafter hidden rows.

        This is the single required override for a model-native drafter.

        Args:
            input_ids: ``(batch, seq)`` verified/seed tokens.
            target_hidden_states: ``(batch, seq, hidden)`` target hidden state(s)
                the drafter conditions on, or ``None`` for drafters that do not.
            position_ids: Absolute positions for the input rows, or ``None``.
            target_kv_cache: Target K/V payload for cross-attending drafters, or
                ``None``.
            attention_mask: Optional attention mask.

        Returns:
            A tuple ``(logits, hidden)`` where ``logits`` is ``(batch, seq,
            vocab)`` and ``hidden`` is ``(batch, seq, hidden)`` or ``None``.

        Raises:
            NotImplementedError: If the subclass does not implement it.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement `_draft_logits`.")

    def draft(
        self,
        input_ids: jax.Array,
        target_hidden_states: jax.Array | None = None,
        target_kv_cache: tp.Any | None = None,
        position_ids: jax.Array | None = None,
        return_full_log_probs: bool = False,
        sample: bool = False,
        rng_key: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
    ) -> DraftStep:
        """Propose one next-token draft per batch element.

        Shared implementation used by every model-native drafter: it calls
        :meth:`_draft_logits`, takes the last position, then argmaxes (greedy)
        or samples (``sample=True``, requires ``rng_key``) and materializes the
        log-prob tail only when ``sample or return_full_log_probs``.

        Args:
            input_ids: ``(batch, seq)`` verified/seed tokens.
            target_hidden_states: Target hidden state(s) the drafter conditions
                on, or ``None``.
            target_kv_cache: Target K/V payload for cross-attending drafters.
            position_ids: Absolute positions for the input rows.
            return_full_log_probs: Materialize the dense drafter distribution.
            sample: Sample from the drafter distribution instead of argmax.
            rng_key: PRNG key required when ``sample=True``.
            attention_mask: Optional attention mask.

        Returns:
            A :class:`~easydel.inference.speculative.DraftStep`.

        Raises:
            ValueError: If ``sample=True`` but ``rng_key`` is ``None``.
        """
        from easydel.inference.speculative import DraftStep

        logits, hidden = self._draft_logits(
            input_ids,
            target_hidden_states=target_hidden_states,
            position_ids=position_ids,
            target_kv_cache=target_kv_cache,
            attention_mask=attention_mask,
        )
        last = logits[:, -1, :].astype(jnp.float32)
        if sample:
            if rng_key is None:
                raise ValueError("rng_key required when sample=True")
            token_ids = jax.random.categorical(rng_key, last)
        else:
            token_ids = jnp.argmax(last, axis=-1)
        log_probs = None
        token_log_probs = None
        if sample or return_full_log_probs:
            log_probs = jax.nn.log_softmax(last, axis=-1)
            token_log_probs = jnp.take_along_axis(log_probs, token_ids[:, None], axis=-1).squeeze(-1)
        return DraftStep(
            token_ids=token_ids.astype(jnp.int32),
            log_probs=token_log_probs,
            full_log_probs=log_probs,
            hidden_states=hidden,
        )

    def reset(self, batch_size: int) -> None:
        """Reset per-generation drafter state.

        Default no-op: model-native drafters are stateless within their
        JAX-functional forward (any KV cache is owned by the caller / runtime).

        Args:
            batch_size: Number of sequences in the upcoming generation.
        """
        del batch_size

    def reset_request(self, request_id: str | None = None) -> None:
        """Reset per-request / per-verify-window drafter state.

        Default no-op. Drafters that carry an internal KV cache across
        ``draft`` calls (e.g. the inline MTP drafter) override this to clear
        that cache at a request/window boundary so state never leaks across
        requests. The eSurge runner calls it at the start of each verify
        window.

        Args:
            request_id: Identifier of the request being drafted for, or ``None``.
        """
        del request_id

    def set_num_draft_tokens(self, num_draft_tokens: int) -> None:
        """Set the number of speculative tokens proposed per verify window.

        Args:
            num_draft_tokens: Draft tokens per window (clamped up to one).
        """
        self.num_draft_tokens = max(1, int(num_draft_tokens))

    def set_max_length(self, max_length: int) -> None:
        """Configure the runner's maximum sequence length on the drafter.

        The eSurge runner calls this once at construction (guarded by
        ``hasattr``). Model-native drafters own no runner-sized cache, so the
        default is a no-op; drafters with an internal KV cache (inline MTP)
        override it to size that cache.

        Args:
            max_length: Maximum number of tokens the runner may decode.
        """
        del max_length


__all__ = (
    "DrafterRegistry",
    "SpecDecodeBase",
    "register_drafter",
)
