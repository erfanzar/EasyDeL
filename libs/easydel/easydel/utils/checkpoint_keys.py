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

"""Declarative checkpoint-key rules: map a released checkpoint's tensor names onto EasyDeL parameter names.

Families that publish weights in a vendor layout (DeepSeek-V4 / V4.1 native naming, GLM-5-Next's composite VLM
checkpoint, Kimi-Linear's ``block_sparse_moe``) declare a :class:`KeyRuleSet` as the class attribute
``checkpoint_key_rules``. The loader (:meth:`EasyDeLBaseModule._get_checkpoint_key_normalizer`, eager and streaming
bridge paths) applies it to every raw key before ``reform_param`` fusion; a ``None`` result marks a tensor the model
does not own (it is skipped). Tensor layout changes (expert stacking, gate/up fusion) stay in ``reform_param``.

A rule set is an ordered list, evaluated top to bottom on one key:

- :class:`Map` -- if ``pattern`` full-matches the current key, the result is ``target`` filled from the match
  (``None`` = unowned) and evaluation stops.
- :class:`Rewrite` -- substitute ``pattern`` (``re.sub`` search semantics) and continue with the rewritten key.
- :class:`Nested` -- if ``pattern`` full-matches, map a derived key through another rule set and wrap the result
  (DSpark stages reuse the backbone's block rules); evaluation stops.

If no terminal rule matches, ``otherwise`` decides: ``None`` (unowned, the default), :data:`KEEP`, or another
:class:`KeyRuleSet` to fall back to.

Templates use :meth:`str.format`: ``{0}`` is the whole matched key and ``{name}`` a named group (unmatched groups
are ``""``). ``rename`` maps a group name to a lookup table applied to that group's value before filling.
"""

from __future__ import annotations

import re
import typing as tp
from dataclasses import dataclass, field

__all__ = ("KEEP", "KeyRuleSet", "Map", "Nested", "Rewrite")

#: Target template returning the key unchanged.
KEEP: tp.Final[str] = "{0}"

_Rename = tp.Mapping[str, tp.Mapping[str, str]]


def _groups(match: re.Match, rename: _Rename) -> dict[str, str]:
    groups = match.groupdict(default="")
    for name, table in rename.items():
        groups[name] = table.get(groups[name], groups[name])
    return groups


@dataclass(frozen=True)
class Map:
    """Terminal rule: on a full match, return ``target`` filled from the match (``None`` = unowned).

    Attributes:
        pattern: Regular expression, full-matched against the current key.
        target: Output template, or ``None`` when the model does not own the tensor.
        rename: ``{group_name: {value: replacement}}`` applied before filling the template.
    """

    pattern: str
    target: str | None
    rename: _Rename = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def apply(self, key: str) -> tuple[bool, str | None]:
        """Return ``(matched, result)``."""
        match = self._compiled.fullmatch(key)
        if match is None:
            return False, None
        if self.target is None:
            return True, None
        return True, self.target.format(match.group(0), **_groups(match, self.rename))


@dataclass(frozen=True)
class Rewrite:
    """Non-terminal rule: substitute ``pattern`` in the key (search semantics) and continue.

    Attributes:
        pattern: Regular expression searched in the current key.
        replacement: Template filled from each match (``{0}`` = the matched text, ``{name}`` = named groups).
        count: Maximum substitutions (``0`` = all, like :meth:`str.replace`).
    """

    pattern: str
    replacement: str
    count: int = 0

    def __post_init__(self):
        object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def apply(self, key: str) -> str:
        """Return the rewritten key."""
        return self._compiled.sub(
            lambda m: self.replacement.format(m.group(0), **m.groupdict(default="")), key, count=self.count
        )


@dataclass(frozen=True)
class Nested:
    """Terminal rule delegating to another rule set.

    On a full match of ``pattern``, ``inner_key`` (filled from the match) is mapped through ``rules``; the inner
    result must full-match ``inner_result`` (filled from the outer match, values regex-escaped) and ``target`` is
    filled from both matches' named groups. Any miss is unowned.

    Attributes:
        pattern: Outer regular expression.
        rules: Rule set the derived key is mapped through.
        inner_key: Template of the derived key.
        inner_result: Template of the regex the inner result must full-match.
        target: Output template (outer and inner named groups; inner names win).
    """

    pattern: str
    rules: KeyRuleSet
    inner_key: str
    inner_result: str
    target: str

    def __post_init__(self):
        object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def apply(self, key: str) -> tuple[bool, str | None]:
        """Return ``(matched, result)``."""
        match = self._compiled.fullmatch(key)
        if match is None:
            return False, None
        outer = match.groupdict(default="")
        inner = self.rules(self.inner_key.format(match.group(0), **outer))
        if inner is None:
            return True, None
        escaped = {name: re.escape(value) for name, value in outer.items()}
        inner_match = re.fullmatch(self.inner_result.format(re.escape(match.group(0)), **escaped), inner)
        if inner_match is None:
            return True, None
        return True, self.target.format(match.group(0), **{**outer, **inner_match.groupdict(default="")})


Rule = Map | Rewrite | Nested


class KeyRuleSet:
    """Ordered checkpoint-key rules; calling it maps one raw key to an EasyDeL name or ``None`` (unowned).

    Args:
        rules: Rules evaluated in order (see the module docstring).
        otherwise: Result when no terminal rule matches: ``None`` (unowned), a template such as :data:`KEEP`, or a
            :class:`KeyRuleSet` applied to the (possibly rewritten) key.
        name: Label used in ``repr``.
    """

    def __init__(
        self,
        rules: tp.Sequence[Rule],
        *,
        otherwise: str | KeyRuleSet | None = None,
        name: str = "",
    ):
        for rule in rules:
            if not isinstance(rule, Map | Rewrite | Nested):
                raise TypeError(f"KeyRuleSet rules must be Map / Rewrite / Nested, got {type(rule).__name__}.")
        self.rules: tuple[Rule, ...] = tuple(rules)
        self.otherwise = otherwise
        self.name = name

    def normalize(self, key: str) -> str | None:
        """Map ``key`` to its EasyDeL parameter name, or ``None`` when the model does not own it."""
        for rule in self.rules:
            if isinstance(rule, Rewrite):
                key = rule.apply(key)
                continue
            matched, result = rule.apply(key)
            if matched:
                return result
        if self.otherwise is None:
            return None
        if isinstance(self.otherwise, KeyRuleSet):
            return self.otherwise(key)
        return self.otherwise.format(key)

    __call__ = normalize

    def __repr__(self) -> str:
        return f"KeyRuleSet({self.name or '<anonymous>'}, {len(self.rules)} rules)"
