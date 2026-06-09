# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`Selector`: the cross-cutting filter DSL.

A :class:`Selector` is a composable predicate over a module graph. Every
public API in spectrax that takes a "subset of the model" accepts a
:class:`Selector` or one of its sugar forms:

* a string or tuple of strings — shorthand for a collection-name filter,
  so ``"parameters"`` is equivalent to ``select().variables("parameters")``;
* a callable ``(Variable, path) -> bool`` — applied per variable;
* a :class:`Selector` instance;
* ``None`` — a "match nothing" selector.

Selectors do not hold references to modules; they are pure
specifications that match pairs when applied via
:meth:`Selector.apply`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TypeAlias

from ._typing import Path
from .errors import SelectorError
from .module import Module
from .paths import path_to_str, str_to_path
from .state import State, _nested_set
from .variable import Variable

__all__ = [
    "Everything",
    "Nothing",
    "Selector",
    "SelectorSugar",
    "all_of",
    "any_of",
    "as_selector",
    "not_",
    "of_type",
    "path_contains",
    "path_endswith",
    "path_startswith",
    "select",
]


ModulePred: TypeAlias = Callable[[Module, str], bool]
"""``(module, path) -> bool`` predicate type applied per module."""

VariablePred: TypeAlias = Callable[[Variable, str], bool]
"""``(variable, path) -> bool`` predicate type applied per variable."""

AnyPred: TypeAlias = Callable[[object, str], bool]
"""Predicate tolerant of both modules and variables as the first argument."""

SelectorSugar: TypeAlias = "Selector | str | type | Iterable[str | type] | VariablePred | None"
"""Values accepted by :func:`as_selector`.

Accepts:

* a :class:`Selector` (returned unchanged),
* a collection name string (``"parameters"``, ``"adapter"``, …),
* a :class:`~spectrax.Variable` subclass (instance-of filter, like
  ``spx.nn.LoraParameter``),
* an iterable mixing strings and Variable subclasses,
* a callable ``(variable, path) -> bool`` predicate,
* ``None`` — "match nothing".
"""


@dataclass(frozen=True)
class Selector:
    """Composable filter over a module graph.

    Fields are combined with AND semantics. Use ``|`` to construct a
    union and ``~`` to invert the *variable* predicate (module
    predicates are left intact). Instances are frozen dataclasses; all
    builder methods return fresh selectors.

    Attributes:
        module_types: Module must be an instance of at least one of
            these types (when non-empty).
        exclude_module_types: Module must *not* be an instance of any of
            these types.
        variable_kinds: Variable :attr:`~spectrax.Variable.kind` must
            match one of these (when non-empty).
        exclude_kinds: Variable kind must *not* match any of these.
        variable_types: Variable must be an instance of at least one of
            these :class:`~spectrax.Variable` subclasses (when non-empty).
            Mirror of :attr:`module_types` for variables.
        exclude_variable_types: Variable must *not* be an instance of any
            of these :class:`~spectrax.Variable` subclasses.
        path_globs: Path must match at least one of these globs. ``*``
            matches one segment, ``**`` matches any number of segments.
        module_where: Additional module predicates.
        variable_where: Additional variable predicates.
        invert: If ``True``, negate the variable match decision.
        subselectors: Union components; a non-empty tuple makes the
            selector an OR over these.
    """

    module_types: tuple[type, ...] = ()
    exclude_module_types: tuple[type, ...] = ()
    variable_kinds: tuple[str, ...] = ()
    exclude_kinds: tuple[str, ...] = ()
    variable_types: tuple[type, ...] = ()
    exclude_variable_types: tuple[type, ...] = ()
    path_globs: tuple[str, ...] = ()
    module_where: tuple[ModulePred, ...] = ()
    variable_where: tuple[VariablePred, ...] = ()
    invert: bool = False
    subselectors: tuple[Selector, ...] = field(default_factory=tuple)
    combinator: str = "or"

    def at_instances_of(self, *types: type) -> Selector:
        """Return a new selector that also requires module instance-of ``types``.

        Args:
            *types: Module subclasses to match.

        Returns:
            A fresh :class:`Selector` with ``types`` added to
            :attr:`module_types`.
        """
        return replace(self, module_types=self.module_types + tuple(types))

    def not_instances_of(self, *types: type) -> Selector:
        """Return a new selector that also excludes module instance-of ``types``.

        Args:
            *types: Module subclasses whose instances should be rejected.

        Returns:
            A fresh :class:`Selector` with ``types`` added to
            :attr:`exclude_module_types`.
        """
        return replace(self, exclude_module_types=self.exclude_module_types + tuple(types))

    def variables(self, *kinds: str) -> Selector:
        """Return a new selector that also requires variable kind in ``kinds``.

        Args:
            *kinds: Collection-name strings (e.g. ``"parameters"``,
                ``"batch_stats"``) to accept.

        Returns:
            A fresh :class:`Selector` with ``kinds`` added to
            :attr:`variable_kinds`.
        """
        return replace(self, variable_kinds=self.variable_kinds + tuple(kinds))

    def exclude_variables(self, *kinds: str) -> Selector:
        """Return a new selector that also excludes variable kinds in ``kinds``.

        Args:
            *kinds: Collection-name strings to reject.

        Returns:
            A fresh :class:`Selector` with ``kinds`` added to
            :attr:`exclude_kinds`.
        """
        return replace(self, exclude_kinds=self.exclude_kinds + tuple(kinds))

    def of_type(self, *types: type) -> Selector:
        """Return a new selector that also restricts to Variable subclasses.

        Every ``type`` must be a subclass of :class:`~spectrax.Variable`.
        Multiple types union: a variable matches if it is an instance of
        any listed type. Composes with the other Selector filters under
        AND semantics, so combining with :meth:`variables` intersects
        (e.g. ``of_type(LoraParameter).variables("lora")``).

        Args:
            *types: Variable subclasses to accept.

        Returns:
            A fresh :class:`Selector` with ``types`` added to
            :attr:`variable_types`.

        Raises:
            SelectorError: If any argument is not a Variable subclass.

        Example::

            spx.select().of_type(spx.nn.LoraParameter, MyAdapterParam)
        """
        for t in types:
            if not (isinstance(t, type) and issubclass(t, Variable)):
                raise SelectorError(f"of_type expects Variable subclasses, got {t!r}")
        return replace(self, variable_types=self.variable_types + tuple(types))

    def not_of_type(self, *types: type) -> Selector:
        """Return a new selector that additionally excludes Variable subclasses.

        Negation counterpart of :meth:`of_type`. A variable that is an
        instance of any listed type is rejected, even if other selector
        fields would otherwise accept it.

        Args:
            *types: Variable subclasses whose instances should be rejected.

        Returns:
            A fresh :class:`Selector` with ``types`` added to
            :attr:`exclude_variable_types`.

        Raises:
            SelectorError: If any argument is not a Variable subclass.
        """
        for t in types:
            if not (isinstance(t, type) and issubclass(t, Variable)):
                raise SelectorError(f"not_of_type expects Variable subclasses, got {t!r}")
        return replace(self, exclude_variable_types=self.exclude_variable_types + tuple(types))

    def at_path(self, *globs: str) -> Selector:
        """Return a new selector that also requires the path to match ``globs``.

        Args:
            *globs: Dotted path globs. ``*`` matches one segment;
                ``**`` matches any number of segments.

        Returns:
            A fresh :class:`Selector` with ``globs`` added to
            :attr:`path_globs`.
        """
        return replace(self, path_globs=self.path_globs + tuple(globs))

    def where(self, pred: AnyPred) -> Selector:
        """Attach ``pred`` to both module and variable predicate lists.

        ``pred`` is invoked as ``pred(target, path)`` with ``target``
        either a :class:`Module` or a :class:`Variable`; it must be
        tolerant of both. For strictly typed predicates prefer
        :meth:`where_module` / :meth:`where_variable`.

        Args:
            pred: A callable ``(target, path) -> bool``.

        Returns:
            A fresh :class:`Selector` with ``pred`` appended to both
            predicate lists.
        """
        return replace(
            self,
            module_where=(*self.module_where, pred),
            variable_where=(*self.variable_where, pred),
        )

    def where_module(self, pred: ModulePred) -> Selector:
        """Attach a module-only predicate.

        Args:
            pred: A callable ``(module, path) -> bool``.

        Returns:
            A fresh :class:`Selector` with ``pred`` appended to
            :attr:`module_where`.
        """
        return replace(self, module_where=(*self.module_where, pred))

    def where_variable(self, pred: VariablePred) -> Selector:
        """Attach a variable-only predicate.

        Args:
            pred: A callable ``(variable, path) -> bool``.

        Returns:
            A fresh :class:`Selector` with ``pred`` appended to
            :attr:`variable_where`.
        """
        return replace(self, variable_where=(*self.variable_where, pred))

    def __or__(self, other: Selector) -> Selector:
        """Build a union selector matching either operand.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return Selector(subselectors=(self, other), combinator="or")

    def __and__(self, other: Selector) -> Selector:
        """Build an intersection selector matching both operands.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return Selector(subselectors=(self, other), combinator="and")

    def __sub__(self, other: Selector) -> Selector:
        """Set-difference selector: ``a - b`` == ``a & ~b``.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self & ~other

    def __invert__(self) -> Selector:
        """Build a selector whose variable match is inverted.

        Returns:
            Result described by this helper.
        """
        return replace(self, invert=not self.invert)

    def _matches_module(self, m: Module, path: str) -> bool:
        """Return ``True`` iff ``m`` matches this selector's module filter.

        Args:
            m: M value consumed by this operation.
            path: Logical or filesystem path used by the operation.

        Returns:
            Return ``True`` iff ``m`` matches this selector's module filter.
        """
        if self.subselectors:
            if self.combinator == "and":
                return all(s._matches_module(m, path) for s in self.subselectors)
            return any(s._matches_module(m, path) for s in self.subselectors)
        if self.module_types and not isinstance(m, self.module_types):
            return False
        if self.exclude_module_types and isinstance(m, self.exclude_module_types):
            return False
        if self.path_globs and not any(_glob_match(g, path) for g in self.path_globs):
            return False
        for pred in self.module_where:
            try:
                if not pred(m, path):
                    return False
            except Exception as e:
                raise SelectorError(f"Selector predicate raised: {e}") from e
        return True

    def _matches_variable(self, v: Variable, path: str) -> bool:
        """Return ``True`` iff ``v`` matches this selector's variable filter.

        Args:
            v: V value consumed by this operation.
            path: Logical or filesystem path used by the operation.

        Returns:
            Return ``True`` iff ``v`` matches this selector's variable filter.
        """
        if self.subselectors:
            if self.combinator == "and":
                base = all(s._matches_variable(v, path) for s in self.subselectors)
            else:
                base = any(s._matches_variable(v, path) for s in self.subselectors)
            return base ^ self.invert
        matched = True
        if self.variable_kinds and v.kind not in self.variable_kinds:
            matched = False
        if matched and self.exclude_kinds and v.kind in self.exclude_kinds:
            matched = False
        if matched and self.variable_types and not isinstance(v, self.variable_types):
            matched = False
        if matched and self.exclude_variable_types and isinstance(v, self.exclude_variable_types):
            matched = False
        if matched and self.path_globs and not any(_glob_match(g, path) for g in self.path_globs):
            matched = False
        for pred in self.variable_where:
            try:
                if not pred(v, path):
                    matched = False
                    break
            except Exception as e:
                raise SelectorError(f"Selector predicate raised: {e}") from e
        return matched ^ self.invert

    def apply(self, module: Module) -> list[tuple[str, Variable]]:
        """Walk ``module`` and return every ``(path, variable)`` this selector matches.

        Module-level filters (``module_types``, ``exclude_module_types``,
        ``module_where``) prune the traversal: a variable is admitted
        only when some ancestor module satisfies the filter. Variables
        reached by multiple paths (tied / shared weights) are reported
        exactly once, under their first-seen canonical path. Compound
        selectors (``a | b``, ``a & b``) recurse into their
        sub-selectors and combine results respecting :attr:`invert`.

        When no module-level filter is active, the implementation
        consults the :func:`spectrax.export` cache (variable list)
        instead of re-walking, which keeps repeated selector
        application cheap on the dispatch hot path.

        Args:
            module: Live module to scan.

        Returns:
            A list of ``(canonical_path, variable)`` pairs.

        Raises:
            TypeError: If ``module`` is not a :class:`Module`.
            SelectorError: If a user-supplied predicate raises.
        """
        if not isinstance(module, Module):
            raise TypeError(f"apply() expects a Module, got {type(module).__name__}")
        if self.subselectors:
            results = [s.apply(module) for s in self.subselectors]
            seen: dict[int, tuple[str, Variable]] = {}
            if self.combinator == "and":
                ref_sets = [{v.ref_id for _, v in r} for r in results]
                common = set.intersection(*ref_sets) if ref_sets else set()
                for p, v in results[0]:
                    if v.ref_id in common and v.ref_id not in seen:
                        seen[v.ref_id] = (p, v)
            else:
                for r in results:
                    for p, v in r:
                        if v.ref_id not in seen:
                            seen[v.ref_id] = (p, v)
            out_list = list(seen.values())
            if self.invert:
                all_pairs = select().apply(module)
                matched_ids = {v.ref_id for _, v in out_list}
                out_list = [(p, v) for p, v in all_pairs if v.ref_id not in matched_ids]
            return out_list

        has_module_filter = bool(self.module_types) or bool(self.exclude_module_types) or bool(self.module_where)

        cache = module._spx_export_cache
        if not has_module_filter and cache is not None:
            out: list[tuple[str, Variable]] = []
            seen_refs: set[int] = set()
            for _kind, path_str, var in cache[2]:
                rid = var.ref_id
                if rid in seen_refs:
                    continue
                seen_refs.add(rid)
                if self._matches_variable(var, path_str):
                    out.append((path_str, var))
            return out

        out = []
        seen_var_refs: set[int] = set()

        def walk(obj: Module | Variable, path: Path, ancestor_matches: bool) -> None:
            """Recursive traversal helper carrying the "ancestor matched" flag.

            Args:
                obj: Object inspected or transformed by the helper.
                path: Logical or filesystem path used by the operation.
                ancestor_matches: Ancestor matches value consumed by this operation.
            """
            if isinstance(obj, Module):
                path_str = path_to_str(path)
                m_match = ancestor_matches or self._matches_module(obj, path_str)
                for key, child in obj._spx_graph_children():
                    walk(child, (*path, key), m_match)
            elif isinstance(obj, Variable):
                if obj.ref_id in seen_var_refs:
                    return
                seen_var_refs.add(obj.ref_id)
                path_str = path_to_str(path)
                if has_module_filter and not ancestor_matches:
                    return
                if self._matches_variable(obj, path_str):
                    out.append((path_str, obj))

        walk(module, (), False)
        return out

    def partition_state(self, module: Module, state: State) -> tuple[State, State]:
        """Split ``state`` into ``(matched, rest)`` using the live module's paths as reference.

        For every variable picked by :meth:`apply` on ``module`` the
        ``(collection, path)`` pair is collected into a match-set;
        leaves of ``state`` are then routed into ``matched`` if their
        key is in that set and into ``rest`` otherwise. Live writer
        callbacks attached to ``state`` (typically by
        :func:`spectrax.export`) are partitioned alongside the leaves
        so write-through still works on either half.

        Args:
            module: Live module supplying the path ground truth.
            state: :class:`State` to split.

        Returns:
            A pair ``(matched, rest)`` of fresh :class:`State` instances.
        """
        matched_paths: set[tuple[str, str]] = set()
        for p, v in self.apply(module):
            matched_paths.add((v.kind, p))
        matched_nested: dict[str, dict[str, object]] = {}
        rest_nested: dict[str, dict[str, object]] = {}
        matched_writers: dict[tuple[str, str], object] | None = {} if state._writers is not None else None
        rest_writers: dict[tuple[str, str], object] | None = {} if state._writers is not None else None
        for c, path, val in state.items():
            is_match = (c, path) in matched_paths
            tgt = matched_nested if is_match else rest_nested
            _nested_set(tgt.setdefault(c, {}), str_to_path(path), val)
            if state._writers is not None:
                writer = state._writers.get((c, path))
                if writer is not None:
                    if is_match:
                        assert matched_writers is not None
                        matched_writers[(c, path)] = writer
                    else:
                        assert rest_writers is not None
                        rest_writers[(c, path)] = writer
        return State._from_raw(matched_nested, writers=matched_writers), State._from_raw(
            rest_nested, writers=rest_writers
        )

    def set(self, module: Module, fn: Callable[[Variable]]) -> None:
        """In-place update: write ``fn(var)`` to ``var.value`` for every match.

        Walks ``module`` via :meth:`apply` and assigns ``var.value =
        fn(var)`` to each picked variable. Writes go through
        :attr:`Variable.value`, so any installed write hook (from a
        spectrax transform) fires.

        Args:
            module: Live module to mutate.
            fn: Callback receiving the matched :class:`Variable` and
                returning its new value.
        """
        for _, var in self.apply(module):
            var.value = fn(var)


def select() -> Selector:
    """Return an empty :class:`Selector` for chaining.

    Typical usage::

        spx.select().at_instances_of(nn.Linear).variables("parameters")

    Returns:
        A fresh empty :class:`Selector`.
    """
    return Selector()


def as_selector(x: SelectorSugar) -> Selector:
    """Coerce a value into a :class:`Selector`.

    Accepted inputs:

    * ``Selector`` — returned unchanged.
    * ``str`` — treated as a collection name (e.g. ``"parameters"``).
    * :class:`~spectrax.Variable` subclass — instance-of filter
      (e.g. ``spx.nn.LoraParameter``). A variable-subclass filter.
    * ``Iterable`` of strings and/or Variable subclasses — union filter.
    * Callable ``(v, path) -> bool`` — treated as a variable predicate.
    * ``None`` — returns a selector that matches nothing.

    Args:
        x: object :data:`SelectorSugar` value.

    Returns:
        A :class:`Selector` representing the input.

    Raises:
        SelectorError: On an input that cannot be coerced.
    """
    if x is None:
        return Selector(variable_where=(_never,))
    if isinstance(x, Selector):
        return x
    if isinstance(x, str):
        return select().variables(x)
    if isinstance(x, type) and issubclass(x, Variable):
        return select().of_type(x)
    if callable(x):
        return select().where_variable(x)
    if isinstance(x, Iterable):
        lst = list(x)
        if not lst:
            raise SelectorError("Empty iterable cannot be coerced to Selector")
        kinds = tuple(e for e in lst if isinstance(e, str))
        types = tuple(e for e in lst if isinstance(e, type) and issubclass(e, Variable))
        if len(kinds) + len(types) != len(lst):
            raise SelectorError(
                f"Iterable must contain only collection-name strings and/or Variable subclasses, got {lst!r}"
            )
        if kinds and not types:
            return select().variables(*kinds)
        if types and not kinds:
            return select().of_type(*types)
        return select().variables(*kinds) | select().of_type(*types)
    raise SelectorError(f"Cannot coerce to Selector: {x!r}")


def _never(_v: Variable, _p: str) -> bool:
    """Predicate that always returns ``False`` — used by ``as_selector(None)``.

    Args:
        _v:  v value consumed by this operation.
        _p:  p value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    return False


def all_of(*selectors: Selector) -> Selector:
    """Intersection of many selectors. ``all_of()`` returns :data:`Everything`.

    Args:
        *selectors: Selectors to intersect.

    Returns:
        A :class:`Selector` matching variables that match *all* of the
        input selectors.
    """
    if not selectors:
        return Everything
    if len(selectors) == 1:
        return selectors[0]
    return Selector(subselectors=tuple(selectors), combinator="and")


def any_of(*selectors: Selector) -> Selector:
    """Union of many selectors. ``any_of()`` returns :data:`Nothing`.

    Args:
        *selectors: Selectors to union.

    Returns:
        A :class:`Selector` matching variables that match *any* of the
        input selectors.
    """
    if not selectors:
        return Nothing
    if len(selectors) == 1:
        return selectors[0]
    return Selector(subselectors=tuple(selectors), combinator="or")


def not_(selector: Selector) -> Selector:
    """Equivalent to ``~selector``.

    Args:
        selector: The selector to negate.

    Returns:
        A :class:`Selector` whose variable match is inverted.
    """
    return ~selector


Everything: Selector = Selector()
"""A selector matching every variable (an empty filter)."""


Nothing: Selector = Selector(variable_where=(_never,))
"""A selector matching no variable."""


def of_type(*types: type) -> Selector:
    """Build a selector that matches variables by Python class.

    Module-level convenience equivalent to ``spx.select().of_type(*types)``.
    Pick
    parameters by :class:`~spectrax.Variable` subclass rather than by
    the ``kind`` collection-name string. Useful when several Variable
    subclasses share a collection (or vice versa).

    Args:
        *types: One or more :class:`~spectrax.Variable` subclasses.
            Variables that are an instance of any listed class match.

    Returns:
        A :class:`Selector` matching any variable whose runtime type is
        (a subclass of) one of ``types``.

    Raises:
        SelectorError: If any argument is not a Variable subclass.

    Example::

        trainable = spx.of_type(spx.nn.LoraParameter).partition_state(m, state)[0]
        grads = spx.grad(loss, wrt=spx.of_type(spx.nn.LoraParameter))(m, x)
    """
    for t in types:
        if not (isinstance(t, type) and issubclass(t, Variable)):
            raise SelectorError(f"of_type expects Variable subclasses, got {t!r}")
    return select().of_type(*types)


def path_contains(substring: str) -> Selector:
    """Return a selector matching variables whose canonical path contains ``substring``.

    Args:
        substring: The substring to search for in each variable path.

    Returns:
        A :class:`Selector` that matches when ``substring`` is found.
    """

    def pred(_v: Variable, path: str) -> bool:
        """Match when ``substring`` appears inside ``path``.

        Args:
            _v:  v value consumed by this operation.
            path: Logical or filesystem path used by the operation.

        Returns:
            Result described by this helper.
        """
        return substring in path

    return select().where_variable(pred)


def path_endswith(suffix: str) -> Selector:
    """Return a selector matching variables whose canonical path ends with ``suffix``.

    Args:
        suffix: The suffix to test against each variable path.

    Returns:
        A :class:`Selector` that matches when the path ends with
        ``suffix``.
    """

    def pred(_v: Variable, path: str) -> bool:
        """Match when ``path`` ends with ``suffix``.

        Args:
            _v:  v value consumed by this operation.
            path: Logical or filesystem path used by the operation.

        Returns:
            Result described by this helper.
        """
        return path.endswith(suffix)

    return select().where_variable(pred)


def path_startswith(prefix: str) -> Selector:
    """Return a selector matching variables whose canonical path starts with ``prefix``.

    Args:
        prefix: The prefix to test against each variable path.

    Returns:
        A :class:`Selector` that matches when the path starts with
        ``prefix``.
    """

    def pred(_v: Variable, path: str) -> bool:
        """Match when ``path`` starts with ``prefix``.

        Args:
            _v:  v value consumed by this operation.
            path: Logical or filesystem path used by the operation.

        Returns:
            Result described by this helper.
        """
        return path.startswith(prefix)

    return select().where_variable(pred)


def _glob_match(pattern: str, path: str) -> bool:
    """Return ``True`` iff ``path`` matches the dotted glob ``pattern``.

    Segments are separated by ``'.'``. ``*`` matches exactly one segment;
    ``**`` matches any number of segments (including zero).

    Args:
        pattern: Pattern value consumed by this operation.
        path: Logical or filesystem path used by the operation.

    Returns:
        Return ``True`` iff ``path`` matches the dotted glob ``pattern``.
    """
    if pattern == path:
        return True
    pat_parts = pattern.split(".")
    path_parts = path.split(".") if path else []
    return _glob_rec(pat_parts, path_parts)


def _glob_rec(pat: list[str], path: list[str]) -> bool:
    """Recursive glob matcher used by :func:`_glob_match`.

    Args:
        pat: Pat value consumed by this operation.
        path: Logical or filesystem path used by the operation.

    Returns:
        Result described by this helper.
    """
    if not pat:
        return not path
    head = pat[0]
    if head == "**":
        if len(pat) == 1:
            return True
        return any(_glob_rec(pat[1:], path[i:]) for i in range(len(path) + 1))
    if not path:
        return False
    if head == "*" or head == path[0]:
        return _glob_rec(pat[1:], path[1:])
    return False
