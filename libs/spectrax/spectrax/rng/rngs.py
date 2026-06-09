# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`Rngs` — explicit, named-stream RNG whose state lives in :class:`~spectrax.State`.

:class:`Rngs` is a :class:`~spectrax.Module` that owns a bag of
:class:`RngStream` variables of kind ``'rng'``. Each stream packs a
PRNG key together with a 64-bit counter (stored as two ``uint32``
words: ``[*key_words, counter_hi, counter_lo]``) into a single JAX
array leaf, so the entire stream — including its counter — round-trips
cleanly through ``jit`` / ``grad`` / ``vmap`` / ``scan`` / ``remat``.

Every attribute access on a stream (``rngs.parameters``,
``rngs.dropout``, …) derives a fresh PRNG key from the stream's
``(key, counter)`` pair via two nested :func:`jax.random.fold_in` calls
(once for the high word, once for the low word) and advances the
counter by one with explicit carry. The equivalent method form is
:meth:`Rngs.key`.

A stream that has not been pre-declared at construction is created
lazily by folding the default stream's key with a deterministic hash of
the stream name (:func:`_str_hash`) — this is what lets layers ask for
``rngs.dropout`` even if only ``Rngs(0)`` was constructed. Inside a JAX
transform the lazy creation cannot mutate the module graph, so
:meth:`Rngs.key` instead computes the same fold-in derivation on the
fly without recording a new stream.
"""

from __future__ import annotations

from typing import ClassVar, cast

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, PRNGKey
from ..core.module import Module, _bump_graph_epoch, _inside_transform
from ..core.variable import Variable

__all__ = ["RngStream", "Rngs", "resolve_rngs"]


class RngStream(Variable):
    """A single named RNG stream stored as a packed ``uint32`` JAX array.

    The leaf value is a 1-D ``uint32`` array laid out as
    ``[*key_words, counter_hi, counter_lo]``: the leading ``key_size``
    elements hold the raw PRNG key data (extracted via
    :func:`jax.random.key_data`) and the trailing two elements hold the
    high and low halves of a 64-bit counter. ``key_size`` is recorded in
    ``self.metadata["key_size"]`` so the same packing/unpacking logic
    works for any of JAX's PRNG implementations.

    Counter advancement (:meth:`next_key`) increments the low word and
    propagates carry to the high word; the resulting counter is folded
    twice into the typed key (high then low) to obtain the per-step PRNG
    key. This avoids the ``jax.random.split`` allocation cost while still
    producing keys that are statistically independent across counter
    values.

    Class attributes:
        default_kind: ``"rng"`` — collected separately from
            ``"param"`` / ``"state"`` by :class:`~spectrax.Module` graph
            walks.
        inherit_stage_assignment: ``False`` — RNG streams are kept off
            the MPMD pipeline-stage tagging system; the rank that runs a
            transform owns the stream value directly.
    """

    default_kind: ClassVar[str] = "rng"
    inherit_stage_assignment: ClassVar[bool] = False

    def __init__(self, key: ArrayLike, *, counter: int = 0, ref_id: int | None = None) -> None:
        """Construct a stream from a PRNG key and a starting counter.

        Args:
            key: Either a typed PRNG key (as returned by
                :func:`jax.random.PRNGKey` / :func:`jax.random.key`) or
                raw ``uint32`` key data of shape ``(key_size,)``.
            counter: Initial value of the 64-bit step counter. Defaults
                to ``0``.
            ref_id: Optional shared-variable identifier forwarded to
                :class:`~spectrax.Variable`; leave as ``None`` unless
                explicitly aliasing another variable.

        Raises:
            ValueError: If the raw key data is not 1-D ``uint32``.
        """
        raw = _to_raw_key(key)
        if raw.ndim != 1:
            raise ValueError(f"RngStream key must be 1-D uint32, got shape {raw.shape}")
        hi = jnp.uint32((int(counter) >> 32) & 0xFFFFFFFF)
        lo = jnp.uint32(int(counter) & 0xFFFFFFFF)
        packed = jnp.concatenate([raw, jnp.array([hi, lo], dtype=jnp.uint32)])
        super().__init__(
            packed,
            kind="rng",
            ref_id=ref_id,
            metadata={"key_size": int(raw.shape[0])},
        )

    def _unpack(self) -> tuple[Array, Array, Array]:
        """Return ``(raw_key_u32, counter_hi, counter_lo)`` from the packed leaf.

        Reads ``self.metadata["key_size"]`` (recorded in
        :meth:`__init__`) to know how many leading words make up the
        key; the next two words are the counter halves.

        Returns:
            Return ``(raw_key_u32, counter_hi, counter_lo)`` from the packed leaf.
        """
        n = int(self.metadata["key_size"])
        raw = self._raw_get()
        return raw[:n], raw[n], raw[n + 1]

    def _repack(self, key: Array, hi: Array, lo: Array) -> Array:
        """Pack ``(key, counter_hi, counter_lo)`` back into a single ``uint32`` leaf.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            hi: Hi value consumed by this operation.
            lo: Lo value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return jnp.concatenate([jnp.asarray(key, dtype=jnp.uint32), jnp.asarray([hi, lo], dtype=jnp.uint32)])

    def next_key(self) -> PRNGKey:
        """Return a fresh typed PRNG key and advance the counter by one.

        Algorithm:

        1. Unpack the stored ``(raw_key, hi, lo)``.
        2. Wrap ``raw_key`` as a typed PRNG key.
        3. Compute the output as
           ``fold_in(fold_in(typed, hi), lo)`` so both halves of the
           counter influence the result.
        4. Increment ``lo`` (with explicit overflow detection via
           ``new_lo == 0`` since ``uint32`` wraps silently) and add the
           carry to ``hi``.
        5. Repack and assign back to ``self.value`` so the new state
           lives on the variable.

        Returns:
            A typed :class:`~spectrax.PRNGKey` that callers can hand
            directly to ``jax.random.*`` functions.
        """
        raw, hi, lo = self._unpack()
        typed = jax.random.wrap_key_data(raw)
        out_key = jax.random.fold_in(jax.random.fold_in(typed, hi.astype(jnp.int32)), lo.astype(jnp.int32))
        new_lo = lo + jnp.uint32(1)
        carry = jnp.where(new_lo == jnp.uint32(0), jnp.uint32(1), jnp.uint32(0))
        new_hi = hi + carry
        self.value = self._repack(raw, new_hi, new_lo)
        return out_key

    def fold_in(self, tag: int | str) -> RngStream:
        """Return a derived stream whose key is ``fold_in(self.key, tag)``.

        The new stream's counter is reset to ``0`` — the derivation is
        purely on the key half, so the derived stream is statistically
        independent of the parent and of every sibling derived from a
        different ``tag``.

        Args:
            tag: Either an integer (used directly) or a string (hashed
                deterministically with :func:`_str_hash`). Strings let
                callers tag derivations with semantic names like
                ``"dropout"`` or ``"layer_3"`` without managing a global
                int registry.

        Returns:
            A fresh :class:`RngStream` derived from ``self``.
        """
        raw, *_ = self._unpack()
        h = tag if isinstance(tag, int) else _str_hash(tag)
        typed = jax.random.wrap_key_data(raw)
        return RngStream(jax.random.fold_in(typed, jnp.int32(h)))

    def __call__(self) -> PRNGKey:
        """Alias for :meth:`next_key` — convenient when treating the stream as a callable.

        Returns:
            Result of invoking the wrapped callable or module.
        """
        return self.next_key()


class Rngs(Module):
    """Named collection of :class:`RngStream` variables.

    A :class:`Rngs` is the user-facing entry point for randomness in
    SpectraX. It is itself a :class:`~spectrax.Module`, so the streams
    travel with the model's :class:`~spectrax.State` through every JAX
    transform without manual plumbing.

    Typical usage::

        rngs = Rngs(0)                       # root seed 0; only 'default' stream
        rngs = Rngs(0, dropout=1)            # explicit per-stream seed
        key = rngs.parameters                # fresh key from 'parameters' stream
        key = rngs.dropout                   # fresh key from 'dropout' stream
        key = rngs.key("custom")             # equivalent method form
        stream = rngs.stream("parameters")   # underlying :class:`RngStream`
        rngs.fold_in("layer_3")              # derive a new Rngs branched off this one
        rngs.fork(B)                         # B independent Rngs (for vmap)

    Accessing an undeclared stream by attribute (or via :meth:`stream`)
    derives it lazily by :meth:`RngStream.fold_in`-ing the ``"default"``
    stream with a hash of the name; the new stream is then cached so
    repeated accesses advance a single counter rather than spawning
    independent streams.

    Inside a JAX transform the lazy creation cannot mutate the module
    graph; :meth:`key` instead computes the same fold-in on the fly so
    the call still produces the right key without altering the
    captured graph.

    Class attributes:
        _spx_container_kind: ``"dict"`` — tells the SpectraX module
            graph that the children live in a name-keyed dict.
    """

    _spx_container_kind: ClassVar[str] = "dict"
    _spx_items: dict[str, RngStream]

    def __init__(self, default: int | ArrayLike = 0, **streams: int | ArrayLike) -> None:
        """Construct an :class:`Rngs` with a default stream and optional named streams.

        Args:
            default: Seed for the always-present ``"default"`` stream.
                May be an ``int`` (wrapped via :func:`jax.random.PRNGKey`)
                or an existing PRNG key / raw ``uint32`` key array
                (wrapped by :func:`_coerce_seed`). Defaults to ``0``.
            **streams: Optional additional named streams, each with its
                own seed. Any keyword named ``"default"`` is silently
                ignored — the default stream is always set from the
                ``default`` argument.
        """
        super().__init__()
        object.__setattr__(self, "_spx_items", {"default": RngStream(_coerce_seed(default))})
        for name, seed in streams.items():
            if name == "default":
                continue
            self._spx_items[name] = RngStream(_coerce_seed(seed))

    def _spx_graph_children(self):
        """Yield ``(name, stream)`` for every declared and derived stream.

        Used by the SpectraX module graph walker to discover the
        :class:`RngStream` children — every entry in ``_spx_items``
        becomes a child variable visible to ``state`` extraction,
        sharding, etc.
        """
        yield from self._spx_items.items()

    def stream(self, name: str) -> RngStream:
        """Return (creating if needed) the underlying :class:`RngStream` for ``name``.

        The ``"default"`` stream is always present and was set in
        :meth:`__init__`. Any other name is derived lazily from
        ``"default"`` via :meth:`RngStream.fold_in` (which deterministically
        hashes the string name) and then cached. The cache means a given
        :class:`Rngs` instance will return the *same* stream object on
        subsequent calls with the same ``name``, so calls to
        :meth:`RngStream.next_key` advance a single counter rather than
        spawning independent streams.

        Args:
            name: Stream name. Created on demand if not declared.

        Returns:
            The :class:`RngStream` for ``name``.
        """
        items = self._spx_items
        if name in items:
            return items[name]
        items[name] = items["default"].fold_in(name)
        if not _inside_transform():
            _bump_graph_epoch()
        return items[name]

    def key(self, name: str = "default") -> PRNGKey:
        """Return the next PRNG key from the named stream (method form).

        Equivalent to attribute access ``getattr(rngs, name)`` — both
        routes call into this method.

        In-transform fallback: if ``name`` is not yet a declared stream
        and we are currently tracing a JAX transform (so mutating
        ``_spx_items`` would be unsound), a key is computed on the fly
        as ``fold_in(default.next_key(), hash(name))`` rather than
        creating and caching a new stream. Outside transforms the
        regular :meth:`stream` path is used and the new stream is
        cached.

        Args:
            name: Stream name. Defaults to ``"default"``.

        Returns:
            A typed PRNG key.
        """
        if name != "default" and name not in self._spx_items and _inside_transform():
            key = self._spx_items["default"].next_key()
            tag = _str_hash(name)
            return jax.random.fold_in(key, jnp.int32(tag))
        return self.stream(name).next_key()

    def __getattr__(self, name: str) -> PRNGKey:
        """Attribute access returns a fresh PRNG key from the named stream.

        ``rngs.dropout`` is shorthand for ``rngs.key("dropout")``.
        Names starting with ``_`` and any name that resolves through the
        normal class lookup (declared methods, slots, etc.) are not
        routed here because Python only consults ``__getattr__`` when
        the normal lookup fails.

        Args:
            name: Stream name; must be a non-private string.

        Returns:
            A fresh :class:`PRNGKey` from the named stream.

        Raises:
            AttributeError: If ``name`` starts with ``_`` or the
                instance has not finished initializing.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        items = self.__dict__.get("_spx_items")
        if items is None:
            raise AttributeError(name)
        return self.key(name)

    def fold_in(self, tag: int | str) -> Rngs:
        """Return a new :class:`Rngs` with every stream folded by ``tag``.

        Each stream in ``self._spx_items`` is replaced by
        :meth:`RngStream.fold_in` of the same ``tag``, producing a
        derived :class:`Rngs` whose every stream is statistically
        independent of the parent and of any sibling derivation. The
        resulting :class:`Rngs` is a fresh object — neither the parent
        nor the new instance share counter state.

        Args:
            tag: ``int`` (used as-is) or ``str`` (deterministically
                hashed). See :meth:`RngStream.fold_in`.

        Returns:
            A new :class:`Rngs`.
        """
        new = Rngs.__new__(Rngs)
        Module.__init__(new)
        object.__setattr__(new, "_spx_items", {})
        for name, stream in self._spx_items.items():
            new._spx_items[name] = stream.fold_in(tag)
        return new

    def fork(self, n: int) -> _ForkedRngs:
        """Produce ``n`` independent :class:`Rngs` along a leading axis.

        Splits the next key from the ``"default"`` stream into ``n``
        sub-keys (advancing the ``"default"`` stream counter once) and
        returns them wrapped in a :class:`_ForkedRngs`. Indexing the
        result yields a fresh :class:`Rngs` instance per slot — useful
        as the input to ``vmap`` / ``pmap`` / ``shard_map`` when each
        replica needs its own RNG state.

        Args:
            n: Number of independent RNG streams to produce. Must be
                positive.

        Returns:
            A :class:`_ForkedRngs` of length ``n``.

        Raises:
            ValueError: If ``n <= 0``.
        """
        if n <= 0:
            raise ValueError(f"fork count must be > 0, got {n}.")
        typed = self._spx_items["default"].next_key()
        keys = jax.random.split(typed, n)
        return _ForkedRngs(keys)


class _ForkedRngs:
    """A stack of :class:`Rngs` instances sharing a single keys array.

    Returned by :meth:`Rngs.fork`. The backing ``_keys`` array has
    shape ``(n, key_size)`` (or ``(n,)`` of a typed PRNG-key dtype,
    depending on the JAX PRNG implementation in use). Indexing produces
    fresh :class:`Rngs` instances on demand; calling :meth:`as_stack`
    returns the raw stacked-keys array for use with ``vmap`` /
    ``shard_map``.
    """

    __slots__ = ("_keys",)

    _keys: Array

    def __init__(self, keys: Array) -> None:
        """Wrap a stacked-keys array.

        Args:
            keys: Stacked PRNG keys with leading axis length ``n``,
                produced by :func:`jax.random.split`.
        """
        self._keys = keys

    def __len__(self) -> int:
        """Return the leading axis length (the number of forked Rngs).

        Returns:
            Integer length for the container.
        """
        return int(self._keys.shape[0])

    def __getitem__(self, i: int) -> Rngs:
        """Return a fresh :class:`Rngs` seeded with the ``i``-th key slice.

        Args:
            i: I value consumed by this operation.

        Returns:
            Selected item from the container.
        """
        return Rngs(self._keys[i])

    def as_stack(self) -> Array:
        """Return the backing stacked-keys array unchanged.

        Useful as the ``in_axes``-mapped argument to a vmapped
        constructor or training step.

        Returns:
            Return the backing stacked-keys array unchanged.
        """
        return self._keys


def resolve_rngs(rngs: Rngs | int | None = None) -> Rngs:
    """Coerce an ``rngs`` argument into a concrete :class:`Rngs`.

    Used by every layer constructor that needs PRNG keys at
    initialization time, so that callers can pass a ready-made
    :class:`Rngs`, a bare ``int`` seed, or rely on the
    :func:`spectrax.seed` context manager.

    Args:
        rngs: One of:

            * :class:`Rngs` — returned as-is.
            * ``int`` — wrapped as ``Rngs(seed)``.
            * ``None`` — fall back to the thread-local
              :func:`spectrax.seed` context if any.

    Returns:
        A concrete :class:`Rngs` instance.

    Raises:
        RuntimeError: If ``rngs`` is ``None`` and no
            :func:`spectrax.seed` context is currently active.
    """
    from .seed import default_rngs, has_default_rngs

    if rngs is not None:
        return rngs if isinstance(rngs, Rngs) else Rngs(rngs)
    if has_default_rngs():
        return default_rngs()
    raise RuntimeError("Layer construction requires rngs. Pass rngs=... or wrap with `spectrax.seed(n)`.")


def _coerce_seed(s: int | ArrayLike) -> Array:
    """Coerce an int seed or an existing key into a typed PRNG key.

    ``int`` inputs are wrapped via :func:`jax.random.PRNGKey`; everything
    else is forwarded to :func:`_to_typed_key` (which wraps raw
    ``uint32`` data when necessary).

    Args:
        s: S value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if isinstance(s, int):
        return jax.random.PRNGKey(s)
    return _to_typed_key(s)


def _to_typed_key(x: ArrayLike) -> Array:
    """Return a typed PRNG key, wrapping raw ``uint32`` data when necessary.

    Recognises arrays whose dtype is already in the
    :mod:`jax.dtypes` PRNG-key family and returns them unchanged;
    otherwise calls :func:`jax.random.wrap_key_data` on a ``uint32``
    view of the input.

    Args:
        x: Input value consumed by the operation.

    Returns:
        Return a typed PRNG key, wrapping raw ``uint32`` data when necessary.
    """
    if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
        return cast(Array, x)
    return jax.random.wrap_key_data(jnp.asarray(x, dtype=jnp.uint32))


def _to_raw_key(x: ArrayLike) -> Array:
    """Return raw ``uint32`` key data, unwrapping a typed PRNG key when necessary.

    The inverse of :func:`_to_typed_key`. Used by :class:`RngStream` so
    the packed-leaf representation can store key data alongside the
    counter words inside a single ``uint32`` array.

    Args:
        x: Input value consumed by the operation.

    Returns:
        Return raw ``uint32`` key data, unwrapping a typed PRNG key when necessary.
    """
    if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(x)
    return jnp.asarray(x, dtype=jnp.uint32)


def _str_hash(s: str) -> int:
    """Deterministic FNV-1a 32-bit hash of a UTF-8 string, mapped to ``int32``.

    Used to derive integer tags from semantic stream names (``"dropout"``,
    ``"layer_3"``, …) for :func:`jax.random.fold_in`. The result is
    re-mapped to the signed ``int32`` range expected by ``fold_in`` —
    values >= 2**31 wrap around to the negative half.

    Args:
        s: S value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    h = 0x811C9DC5
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 0x01000193) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return h
