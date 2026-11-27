#!/usr/bin/env python3
# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate the EasyDeL trainer-argument catalog the fleet assistant reads.

The assistant must know every argument of every EasyDeL trainer (names, types,
defaults, help text and the inherited ``TrainingArguments`` base) without the
control plane ever importing EasyDeL or JAX. This script therefore reads the
EasyDeL trainer configuration sources *statically* with :mod:`ast` and writes
``eyvan/assistant_knowledge/easydel_trainers.json``:

    uv run python scripts/generate_easydel_catalog.py            # regenerate
    uv run python scripts/generate_easydel_catalog.py --check    # CI / staleness gate

``--check`` exits 1 when the committed catalog differs from a fresh generation
and exits 0 with a skip message when the EasyDeL source tree is not present
(hosted CI has no EasyDeL checkout). Every run also validates the hand-curated
``easydel_essentials.json`` against the catalog, so a field renamed upstream
cannot silently leave the assistant asking about an argument that no longer
exists.

Catalog layout (deterministic, sorted keys):

* ``classes``: every config dataclass that descends from ``TrainingArguments``,
  holding only the fields it declares itself, plus ``__post_init__`` hints.
* ``trainers``: one entry per registered trainer id with its class and the
  resolved inheritance chain (``mro``). Consumers resolve the full field set by
  walking ``mro`` from the base, so base fields are stored once, not 45 times.
* ``elarge``: the eLarge YAML runner's trainer-type aliases and the defaults it
  applies on top of the dataclass defaults.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tokenize
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = REPO_ROOT / "eyvan" / "assistant_knowledge"
CATALOG_PATH = KNOWLEDGE_DIR / "easydel_trainers.json"
ESSENTIALS_PATH = KNOWLEDGE_DIR / "easydel_essentials.json"


#: Locate the sibling EasyDeL trainer sources: monorepo layout (``libs/easydel``)
#: first, then a standalone sibling checkout next to this repository.
def _default_source() -> Path:
    monorepo = REPO_ROOT.parent / "easydel" / "easydel" / "trainers"
    if monorepo.is_dir():
        return monorepo
    return REPO_ROOT.parent / "EasyDeL" / "libs" / "easydel" / "easydel" / "trainers"


DEFAULT_SOURCE = _default_source()
GENERATED_BY = "scripts/generate_easydel_catalog.py"
BASE_CLASS = "TrainingArguments"
#: Auxiliary model blocks the eLarge YAML runner understands (``<role>_model``).
LAUNCH_INPUTS = frozenset({"reference_model", "reward_model", "teacher_model", "target_model"})
#: Launch-level choices every draft needs that are not trainer arguments.
COMMON_LAUNCH_INPUTS = frozenset({"model", "dataset", "sharding"})

_ARG_SECTIONS = frozenset({"Args", "Arguments", "Attributes", "Parameters", "Fields", "Keyword Args"})
_SECTION_HEADER = re.compile(r"^[A-Z][A-Za-z ]*:$")
_DOC_ENTRY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*:\s*(.*)$")
_TOOL_COMMENT = re.compile(r"^(noqa|type:|pyright:|pragma|fmt:)")
_TYPING_PREFIX = re.compile(r"\b(?:tp|typing)\.")
_MAX_CHECK_TEXT = 200
_MAX_MESSAGE = 240

_SEQUENCE_NAMES = frozenset(
    {"list", "tuple", "set", "frozenset", "Sequence", "MutableSequence", "Iterable", "Collection", "List", "Tuple"}
)
_MAPPING_NAMES = frozenset({"dict", "Mapping", "MutableMapping", "Dict"})
_WRAPPER_NAMES = frozenset({"Annotated", "InitVar", "Required", "NotRequired", "Final"})


class CatalogError(RuntimeError):
    """The EasyDeL sources or the curated essentials cannot produce a valid catalog."""


# --------------------------------------------------------------------------- helpers


def _last_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _head_name(annotation: ast.expr) -> str | None:
    if isinstance(annotation, ast.Subscript):
        return _last_name(annotation.value)
    return _last_name(annotation)


def _jsonable(value: Any) -> Any:
    """Return ``value`` as plain JSON data or raise ``ValueError``."""

    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ValueError("non-finite float")
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _jsonable(item) for key, item in value.items()}
    raise ValueError(f"not JSON data: {type(value).__name__}")


def _literal(node: ast.expr) -> Any:
    """Evaluate ``node`` as a JSON-able literal or raise ``ValueError``."""

    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError("not a literal") from exc
    return _jsonable(value)


def _string_value(node: ast.expr | None) -> str | None:
    """Best-effort text of a string expression (implicit concat, ``+`` or f-string)."""

    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append("{" + ast.unparse(value.value) + "}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string_value(node.left), _string_value(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    collapsed = " ".join(text.split())
    return collapsed or None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _type_text(annotation: ast.expr) -> str:
    return _TYPING_PREFIX.sub("", ast.unparse(annotation))


def _summary(doc: str | None) -> str | None:
    if not doc:
        return None
    paragraph = doc.strip().split("\n\n", 1)[0]
    text = _clean(paragraph) or ""
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    sentence = match.group(1) if match else text
    return _clip(sentence, 200) or None


def _docstring_entries(doc: str | None) -> dict[str, str]:
    """Map ``name -> text`` from Google-style ``Args:``/``Attributes:`` sections."""

    entries: dict[str, list[str]] = {}
    if not doc:
        return {}
    in_section = False
    header_indent = 0
    entry_indent: int | None = None
    current: str | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not stripped:
            continue
        if _SECTION_HEADER.match(stripped) and (not in_section or indent <= header_indent):
            in_section = stripped[:-1] in _ARG_SECTIONS
            header_indent = indent
            entry_indent = None
            current = None
            continue
        if not in_section:
            continue
        if indent <= header_indent:
            in_section = False
            current = None
            continue
        match = _DOC_ENTRY.match(stripped)
        if match and (entry_indent is None or indent == entry_indent):
            entry_indent = indent
            current = match.group(1)
            entries[current] = [match.group(2)]
        elif current is not None and entry_indent is not None and indent > entry_indent:
            entries[current].append(stripped)
    return {name: text for name, parts in entries.items() if (text := _clean(" ".join(parts)))}


def _comments(source: str) -> dict[int, str]:
    found: dict[int, str] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                text = token.string.lstrip("#").strip()
                if text and not _TOOL_COMMENT.match(text):
                    found[token.start[0]] = text
    except (tokenize.TokenError, SyntaxError):
        return found
    return found


# --------------------------------------------------------------------------- modules


class _Module:
    """One parsed EasyDeL source file."""

    def __init__(self, path: Path, module: str) -> None:
        self.path = path
        self.module = module
        self.source = path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source, filename=str(path))
        self.lines = self.source.splitlines()
        self.comments = _comments(self.source)
        self.aliases: dict[str, ast.expr] = {}
        for stmt in self.tree.body:
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target, value = stmt.target, stmt.value
            if isinstance(target, ast.Name) and value is not None and _contains_literal(value):
                self.aliases[target.id] = value

    def field_comment(self, stmt: ast.stmt) -> str | None:
        trailing = self.comments.get(stmt.lineno)
        if trailing is not None:
            code = self.lines[stmt.lineno - 1].split("#", 1)[0].strip()
            if code:
                return trailing
        collected: list[str] = []
        line_no = stmt.lineno - 1
        while line_no >= 1 and self.lines[line_no - 1].strip().startswith("#"):
            text = self.comments.get(line_no)
            if text:
                collected.append(text)
            line_no -= 1
        return _clean(" ".join(reversed(collected)))


def _contains_literal(node: ast.expr) -> bool:
    return any(isinstance(sub, ast.Subscript) and _last_name(sub.value) == "Literal" for sub in ast.walk(node))


class _Aliases:
    """Resolve ``Literal`` type aliases: same module first, then a globally unique name."""

    def __init__(self, modules: Iterable[_Module]) -> None:
        self._by_module: dict[str, dict[str, ast.expr]] = {}
        seen: dict[str, list[ast.expr]] = {}
        for module in modules:
            self._by_module[module.module] = module.aliases
            for name, value in module.aliases.items():
                seen.setdefault(name, []).append(value)
        self._unique = {name: values[0] for name, values in seen.items() if len(values) == 1}

    def lookup(self, module: str, name: str) -> ast.expr | None:
        local = self._by_module.get(module, {})
        if name in local:
            return local[name]
        return self._unique.get(name)


class _Ann:
    """What an annotation accepts: value kinds and (optionally) closed choices."""

    def __init__(
        self,
        kinds: frozenset[str] | None,
        choices: tuple[Any, ...] = (),
        open_choices: bool = False,
        sequence: bool = False,
    ) -> None:
        self.kinds = kinds
        self.choices = choices
        self.open_choices = open_choices
        self.sequence = sequence


_ANY = _Ann(None, open_choices=True)


def _union(parts: list[_Ann]) -> _Ann:
    kinds: set[str] | None = set()
    choices: list[Any] = []
    open_choices = False
    sequence = False
    for part in parts:
        if part.kinds is None or kinds is None:
            kinds = None
        else:
            kinds |= part.kinds
        for choice in part.choices:
            if choice not in choices:
                choices.append(choice)
        open_choices = open_choices or part.open_choices
        sequence = sequence or part.sequence
    return _Ann(None if kinds is None else frozenset(kinds), tuple(choices), open_choices, sequence)


def _literal_kind(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _analyze(node: ast.expr, module: str, aliases: _Aliases, depth: int = 0) -> _Ann:
    if depth > 8:
        return _ANY
    if isinstance(node, ast.Constant):
        if node.value is None:
            return _Ann(frozenset({"none"}))
        if isinstance(node.value, str):
            try:
                parsed = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return _ANY
            return _analyze(parsed, module, aliases, depth + 1)
        return _ANY
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union(
            [_analyze(node.left, module, aliases, depth + 1), _analyze(node.right, module, aliases, depth + 1)]
        )
    if isinstance(node, ast.Name | ast.Attribute):
        name = _last_name(node) or ""
        if name == "int":
            return _Ann(frozenset({"int"}))
        if name == "float":
            return _Ann(frozenset({"float", "int"}))
        if name == "bool":
            return _Ann(frozenset({"bool"}))
        if name == "str":
            return _Ann(frozenset({"str"}), open_choices=True)
        if name in _SEQUENCE_NAMES:
            return _Ann(frozenset({"list"}))
        if name in _MAPPING_NAMES:
            return _Ann(frozenset({"dict"}))
        alias = aliases.lookup(module, name) if isinstance(node, ast.Name) else None
        if alias is not None:
            return _analyze(alias, module, aliases, depth + 1)
        return _ANY
    if isinstance(node, ast.Subscript):
        head = _last_name(node.value) or ""
        args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if head == "Literal":
            values: list[Any] = []
            for arg in args:
                try:
                    values.append(_literal(arg))
                except ValueError:
                    return _ANY
            return _Ann(frozenset(_literal_kind(value) for value in values), tuple(values))
        if head == "Optional":
            return _union([_analyze(args[0], module, aliases, depth + 1), _Ann(frozenset({"none"}))])
        if head == "Union":
            return _union([_analyze(arg, module, aliases, depth + 1) for arg in args])
        if head in _WRAPPER_NAMES:
            return _analyze(args[0], module, aliases, depth + 1)
        if head in _SEQUENCE_NAMES:
            items = [arg for arg in args if not (isinstance(arg, ast.Constant) and arg.value is Ellipsis)]
            inner = _union([_analyze(arg, module, aliases, depth + 1) for arg in items]) if items else _ANY
            if inner.choices and not inner.open_choices:
                return _Ann(frozenset({"list"}), inner.choices, False, True)
            return _Ann(frozenset({"list"}))
        if head in _MAPPING_NAMES:
            return _Ann(frozenset({"dict"}))
        return _ANY
    return _ANY


# --------------------------------------------------------------------------- classes


def _registered_ids(cls: ast.ClassDef) -> list[str]:
    for decorator in cls.decorator_list:
        if not (isinstance(decorator, ast.Call) and _last_name(decorator.func) == "register"):
            continue
        if len(decorator.args) < 2:
            continue
        try:
            kind = _literal(decorator.args[0])
            ids = _literal(decorator.args[1])
        except ValueError:
            continue
        if kind != "trainer-arguments":
            continue
        if isinstance(ids, str):
            return [ids]
        if isinstance(ids, list) and all(isinstance(item, str) for item in ids):
            return list(ids)
    return []


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for decorator in cls.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _last_name(target) == "dataclass":
            return True
    return False


def _is_field_call(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and _last_name(node.func) == "field"


def _metadata_help(metadata: ast.expr | None) -> str | None:
    if not isinstance(metadata, ast.Dict):
        return None
    for key, value in zip(metadata.keys, metadata.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == "help":
            return _clean(_string_value(value))
    return None


def _metadata_choices(metadata: ast.expr | None) -> list[Any] | None:
    if not isinstance(metadata, ast.Dict):
        return None
    for key, value in zip(metadata.keys, metadata.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == "choices":
            try:
                choices = _literal(value)
            except ValueError:
                return None
            return choices if isinstance(choices, list) and choices else None
    return None


def _set_default(entry: dict[str, Any], node: ast.expr) -> None:
    try:
        entry["default"] = _literal(node)
    except ValueError:
        entry["default_expr"] = ast.unparse(node)


def _extract_fields(cls: ast.ClassDef, module: _Module, aliases: _Aliases) -> dict[str, dict[str, Any]]:
    doc_entries = _docstring_entries(ast.get_docstring(cls))
    fields: dict[str, dict[str, Any]] = {}
    for stmt in cls.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        name = stmt.target.id
        head = _head_name(stmt.annotation)
        if name.startswith("_") or head == "ClassVar":
            continue
        entry: dict[str, Any] = {"type": _type_text(stmt.annotation)}
        if head == "InitVar":
            entry["init_only"] = True
        help_text: str | None = None
        metadata_choices: list[Any] | None = None
        value = stmt.value
        if value is None:
            entry["required"] = True
        elif _is_field_call(value):
            assert isinstance(value, ast.Call)
            keywords = {keyword.arg: keyword.value for keyword in value.keywords if keyword.arg}
            init = keywords.get("init")
            if isinstance(init, ast.Constant) and init.value is False:
                continue
            if "default" in keywords:
                _set_default(entry, keywords["default"])
            elif "default_factory" in keywords:
                entry["default_factory"] = ast.unparse(keywords["default_factory"])
            else:
                entry["required"] = True
            help_text = _metadata_help(keywords.get("metadata"))
            metadata_choices = _metadata_choices(keywords.get("metadata"))
        else:
            _set_default(entry, value)
        entry.setdefault("required", False)
        if help_text:
            entry["help"] = help_text
            entry["help_source"] = "metadata"
        elif name in doc_entries:
            entry["help"] = doc_entries[name]
            entry["help_source"] = "docstring"
        elif comment := module.field_comment(stmt):
            entry["help"] = comment
            entry["help_source"] = "comment"
        if name in doc_entries and entry.get("help_source") == "metadata" and doc_entries[name] != help_text:
            entry["doc"] = doc_entries[name]
        analysis = _analyze(stmt.annotation, module.module, aliases)
        if analysis.kinds is not None:
            entry["accepts"] = sorted(analysis.kinds)
        if analysis.choices and not analysis.open_choices:
            entry["choices"] = list(analysis.choices)
            entry["choices_source"] = "annotation"
            if analysis.sequence:
                entry["choices_allow_list"] = True
        elif metadata_choices:
            entry["choices"] = metadata_choices
            entry["choices_source"] = "metadata"
        fields[name] = entry
    return fields


def _self_attributes(node: ast.AST) -> set[str]:
    return {
        sub.attr
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self"
    }


def _raise_message(body: list[ast.stmt]) -> str | None:
    for stmt in body:
        if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call) and stmt.exc.args:
            text = _clean(_string_value(stmt.exc.args[0]))
            if text:
                return _clip(text, _MAX_MESSAGE)
    return None


def _post_init_hints(cls: ast.ClassDef) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    """Raise-guarded ``if`` tests and ``self.x not in (...)`` choices from ``__post_init__``."""

    checks: list[dict[str, Any]] = []
    choices: dict[str, list[Any]] = {}
    for stmt in cls.body:
        if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "__post_init__"):
            continue
        for node in ast.walk(stmt):
            if not isinstance(node, ast.If):
                continue
            if not any(isinstance(child, ast.Raise) for child in node.body):
                continue
            checks.append(
                {
                    "if": _clip(_TYPING_PREFIX.sub("", ast.unparse(node.test)), _MAX_CHECK_TEXT),
                    "message": _raise_message(node.body),
                    "fields": sorted(_self_attributes(node.test)),
                }
            )
            for compare in ast.walk(node.test):
                if not (
                    isinstance(compare, ast.Compare)
                    and len(compare.ops) == 1
                    and isinstance(compare.ops[0], ast.NotIn)
                    and isinstance(compare.left, ast.Attribute)
                    and isinstance(compare.left.value, ast.Name)
                    and compare.left.value.id == "self"
                    and isinstance(compare.comparators[0], ast.Tuple | ast.List | ast.Set)
                ):
                    continue
                container = compare.comparators[0]
                if not isinstance(container, ast.Tuple | ast.List | ast.Set):
                    continue
                try:
                    values = [_literal(element) for element in container.elts]
                except ValueError:
                    continue
                choices.setdefault(compare.left.attr, values)
    return checks, choices


# --------------------------------------------------------------------------- eLarge


def _eval_defaults(node: ast.expr, env: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a defaults dict literal, following ``**TABLE["key"]`` spreads."""

    if not isinstance(node, ast.Dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id in env:
                try:
                    ref = _literal(value.slice)
                except ValueError:
                    continue
                spread = env[value.value.id].get(ref)
                if isinstance(spread, dict):
                    out.update(spread)
            continue
        try:
            name = _literal(key)
            out[name] = _literal(value)
        except ValueError:
            continue
    return out


def _elarge_tables(path: Path) -> dict[str, Any]:
    """Aliases and defaults the eLarge runner applies (``infra/elarge/types/training.py``)."""

    if not path.is_file():
        return {"base_defaults": {}, "trainer_defaults": {}, "trainer_type_aliases": {}}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: dict[str, str] = {}
    base: dict[str, Any] = {}
    specific: dict[str, dict[str, Any]] = {}
    for stmt in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target, value = stmt.target, stmt.value
        if isinstance(target, ast.Name) and value is not None:
            if target.id == "_TRAINER_TYPE_ALIASES":
                literal = _literal(value)
                aliases = {str(key): str(val) for key, val in literal.items()}
            elif target.id == "BASE_TRAINER_DEFAULTS":
                base = _eval_defaults(value, {})
            elif target.id == "TRAINER_SPECIFIC_DEFAULTS" and isinstance(value, ast.Dict):
                specific = _eval_trainer_table(value, specific)
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "update"
            and _last_name(stmt.value.func.value) == "TRAINER_SPECIFIC_DEFAULTS"
            and stmt.value.args
            and isinstance(stmt.value.args[0], ast.Dict)
        ):
            specific = _eval_trainer_table(stmt.value.args[0], specific)
    return {"base_defaults": base, "trainer_defaults": specific, "trainer_type_aliases": aliases}


def _eval_trainer_table(node: ast.Dict, current: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    snapshot = {"TRAINER_SPECIFIC_DEFAULTS": dict(current)}
    updated = dict(current)
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            continue
        try:
            name = _literal(key)
        except ValueError:
            continue
        if isinstance(name, str):
            updated[name] = _eval_defaults(value, snapshot)
    return updated


# --------------------------------------------------------------------------- build


def _revision(source: Path) -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    resolved = source.resolve()
    package_root = resolved.parent
    tracked = [
        str(path)
        for path in (
            resolved,
            package_root / "infra" / "elarge" / "types" / "training.py",
            package_root / "infra" / "etils.py",
        )
        if path.exists()
    ]
    if tracked:
        try:
            scoped = subprocess.run(
                [git, "-C", str(resolved), "log", "-n", "1", "--abbrev=9", "--format=%h", "--", *tracked],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if scoped.returncode == 0 and scoped.stdout.strip():
                return scoped.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        result = subprocess.run(
            [git, "-C", str(resolved), "rev-parse", "--short=9", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"


def _config_files(source: Path) -> list[Path]:
    files = [
        path
        for path in source.rglob("*.py")
        if "__pycache__" not in path.parts
        and (path.name.endswith("_config.py") or path.name in {"config.py", "training_configurations.py"})
    ]
    return sorted(files)


def _mro(name: str, bases: dict[str, list[str]], trail: tuple[str, ...] = ()) -> list[str]:
    if name in trail:
        raise CatalogError(f"inheritance cycle through {name}")
    order = [name]
    for base in bases.get(name, []):
        for ancestor in _mro(base, bases, (*trail, name)):
            if ancestor not in order:
                order.append(ancestor)
    return order


def build_catalog(source: Path) -> dict[str, Any]:
    """Statically parse the EasyDeL trainer configs under ``source`` into catalog data."""

    source = source.resolve()
    package_root = source.parent  # .../easydel
    modules = [
        _Module(path, "easydel." + ".".join(path.relative_to(package_root).with_suffix("").parts))
        for path in _config_files(source)
    ]
    if not modules:
        raise CatalogError(f"no trainer config modules under {source}")
    alias_modules = list(modules)
    etils = package_root / "infra" / "etils.py"
    if etils.is_file():
        alias_modules.append(_Module(etils, "easydel.infra.etils"))
    aliases = _Aliases(alias_modules)

    nodes: dict[str, tuple[ast.ClassDef, _Module]] = {}
    for module in modules:
        for stmt in module.tree.body:
            if isinstance(stmt, ast.ClassDef) and _is_dataclass(stmt):
                if stmt.name in nodes:
                    raise CatalogError(f"duplicate config class name {stmt.name} in {module.module}")
                nodes[stmt.name] = (stmt, module)
    bases = {name: [b for base in node.bases if (b := _last_name(base))] for name, (node, _) in nodes.items()}
    if BASE_CLASS not in nodes:
        raise CatalogError(f"{BASE_CLASS} not found under {source}")

    lineages = {name: _mro(name, bases) for name in nodes}
    config_classes = sorted(name for name, lineage in lineages.items() if BASE_CLASS in lineage)

    classes: dict[str, dict[str, Any]] = {}
    for name in config_classes:
        node, module = nodes[name]
        checks, choices = _post_init_hints(node)
        classes[name] = {
            "module": module.module,
            "bases": bases[name],
            "summary": _summary(ast.get_docstring(node)),
            "fields": _extract_fields(node, module, aliases),
            "registered_ids": _registered_ids(node),
            "_checks": checks,
            "_choices": choices,
        }

    # Fill help for redeclared fields that carry none (e.g. ``beta: float = field(default=0.05)``).
    for name in config_classes:
        lineage = [cls for cls in lineages[name] if cls in classes]
        for field_name, entry in classes[name]["fields"].items():
            if "help" in entry:
                continue
            for ancestor in lineage[1:]:
                parent = classes[ancestor]["fields"].get(field_name)
                if parent and parent.get("help"):
                    entry["help"] = parent["help"]
                    entry["help_source"] = f"inherited:{ancestor}"
                    break

    # Keep only checks and choices that reference real constructor fields of the class.
    for name in config_classes:
        lineage = [cls for cls in lineages[name] if cls in classes]
        known = {field_name for cls in lineage for field_name in classes[cls]["fields"]}
        info = classes[name]
        checks = []
        for check in info.pop("_checks"):
            referenced = [field_name for field_name in check["fields"] if field_name in known]
            if referenced:
                checks.append({**check, "fields": referenced})
        if checks:
            info["checks"] = checks
        choices = {key: value for key, value in info.pop("_choices").items() if key in known}
        if choices:
            info["validated_choices"] = choices

    elarge = _elarge_tables(package_root / "infra" / "elarge" / "types" / "training.py")
    type_aliases: dict[str, str] = elarge["trainer_type_aliases"]

    trainers: dict[str, dict[str, Any]] = {}
    for name in config_classes:
        ids = classes[name]["registered_ids"]
        if not ids:
            continue
        primary = ids[0]
        if primary in trainers:
            raise CatalogError(f"trainer id {primary!r} registered by {trainers[primary]['class']} and {name}")
        normalized = primary.lower()
        trainers[primary] = {
            "class": name,
            "module": classes[name]["module"],
            "aliases": ids[1:],
            "elarge_trainer_type": type_aliases.get(normalized, normalized),
            "summary": classes[name]["summary"],
            "mro": [cls for cls in lineages[name] if cls in classes],
        }

    return {
        "easydel_revision": _revision(source),
        "generated_by": GENERATED_BY,
        "source": "/".join(source.relative_to(package_root.parent).parts),
        "elarge": elarge,
        "classes": classes,
        "trainers": trainers,
    }


def render(catalog: dict[str, Any]) -> str:
    return json.dumps(catalog, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def resolved_field_names(catalog: dict[str, Any], trainer_id: str) -> set[str]:
    mro = catalog["trainers"][trainer_id]["mro"]
    return {field_name for cls in mro for field_name in catalog["classes"][cls]["fields"]}


def _check_field_specs(where: str, names: set[str], spec: dict[str, Any], problems: list[str]) -> None:
    for field_name, why in spec.get("fields", {}).items():
        if field_name not in names:
            problems.append(f"essentials: {where}.{field_name} is not a field of the catalog")
        if not isinstance(why, str) or not why.strip():
            problems.append(f"essentials: {where}.{field_name} needs a one-line explanation")
    for group in spec.get("one_of", []):
        group_fields = group.get("fields") if isinstance(group, dict) else None
        if not isinstance(group_fields, list) or len(group_fields) < 2:
            problems.append(f"essentials: {where}.one_of entries need at least two fields")
            continue
        for field_name in group_fields:
            if field_name not in names:
                problems.append(f"essentials: {where}.one_of {field_name} is not a field of the catalog")
        if not isinstance(group.get("why"), str) or not group["why"].strip():
            problems.append(f"essentials: {where}.one_of {group_fields} needs a one-line explanation")


def validate_essentials(catalog: dict[str, Any], essentials: dict[str, Any]) -> list[str]:
    """Return problems with the curated essentials (unknown trainers, fields or launch inputs)."""

    problems: list[str] = []
    common = essentials.get("common", {})
    _check_field_specs("common", set(catalog["classes"][BASE_CLASS]["fields"]), common, problems)
    for key, why in common.get("launch", {}).items():
        if key not in COMMON_LAUNCH_INPUTS:
            problems.append(f"essentials: common launch input {key!r} is not one of {sorted(COMMON_LAUNCH_INPUTS)}")
        if not isinstance(why, str) or not why.strip():
            problems.append(f"essentials: common launch input {key!r} needs a one-line explanation")
    for trainer_id, spec in sorted(essentials.get("trainers", {}).items()):
        if trainer_id not in catalog["trainers"]:
            problems.append(f"essentials: unknown trainer {trainer_id!r}")
            continue
        _check_field_specs(trainer_id, resolved_field_names(catalog, trainer_id), spec, problems)
        for launch_input, detail in spec.get("launch_inputs", {}).items():
            if launch_input not in LAUNCH_INPUTS:
                problems.append(f"essentials: {trainer_id} launch input {launch_input!r} is not an auxiliary model")
            if not isinstance(detail, dict) or not isinstance(detail.get("why"), str):
                problems.append(f"essentials: {trainer_id}.{launch_input} needs {{'required': bool, 'why': str}}")
            elif not isinstance(detail.get("required"), bool):
                problems.append(f"essentials: {trainer_id}.{launch_input}.required must be a boolean")
    return problems


def _load_essentials(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CatalogError(f"{path} must hold a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="EasyDeL easydel/trainers directory")
    parser.add_argument("--output", type=Path, default=CATALOG_PATH, help="catalog JSON path")
    parser.add_argument("--essentials", type=Path, default=ESSENTIALS_PATH, help="curated essentials JSON path")
    parser.add_argument("--check", action="store_true", help="fail when the committed catalog is stale")
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        if args.check:
            print(f"skip: EasyDeL trainer sources not found at {args.source}; catalog not checked")
            return 0
        print(f"error: EasyDeL trainer sources not found at {args.source}", file=sys.stderr)
        return 2

    try:
        catalog = build_catalog(args.source)
        essentials = _load_essentials(args.essentials)
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if essentials is None:
        if args.check:
            print(f"error: {args.essentials} is missing", file=sys.stderr)
            return 1
        print(f"warning: {args.essentials} is missing; essentials not validated", file=sys.stderr)
    else:
        problems = validate_essentials(catalog, essentials)
        if problems:
            for problem in problems:
                print(f"error: {problem}", file=sys.stderr)
            return 1

    text = render(catalog)
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if current != text:
            print(
                f"{args.output} is stale for EasyDeL {catalog['easydel_revision']}; run: uv run python {GENERATED_BY}",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {len(catalog['trainers'])} trainers match EasyDeL {catalog['easydel_revision']}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.output} ({len(text.encode('utf-8'))} bytes): {len(catalog['trainers'])} trainers, "
        f"{len(catalog['classes'])} config classes, EasyDeL {catalog['easydel_revision']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
