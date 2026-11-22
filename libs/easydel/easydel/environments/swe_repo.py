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

"""Multi-turn software-engineering (SWE) repository RL environment."""

from __future__ import annotations

import collections.abc
import io
import json
import os
import posixpath
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import typing as tp

from easydel.trainers._shared.environment import ResetResult, StepResult, create_tool_call_parser

from ._base import SandboxEnvironment, register_environment
from .runtime import (
    ContainerConfig,
    ExecResult,
    KiriRuntime,
    LocalSandboxSession,
    SandboxInfraError,
    SandboxPool,
    SandboxSession,
)

# Files that configure or hook test collection/interpreter start-up. Before grading, every such file is reset to
# its base-revision content (or deleted when the agent added it) so a policy cannot rig the verification tests.
_TEST_INFRA_BASENAMES = frozenset(
    {
        "conftest.py",
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "tox.ini",
        "setup.cfg",
        "pyproject.toml",
        "noxfile.py",
        ".coveragerc",
        "sitecustomize.py",
        "usercustomize.py",
        ".gitattributes",
    }
)
_TEST_INFRA_SUFFIXES = (".pth",)

# Top-level module/package names that would shadow Python test runners or core stdlib modules if newly planted
# by the policy inside the repository working directory.
_RUNNER_SHADOW_NAMES = frozenset(
    {
        "pytest",
        "_pytest",
        "pluggy",
        "iniconfig",
        "py",
        "unittest",
        "nose",
        "nose2",
        "site",
        "sitecustomize",
        "usercustomize",
        "builtins",
        "os",
        "sys",
        "subprocess",
        "runpy",
        "importlib",
        "json",
        "re",
        "io",
        "shutil",
        "pathlib",
        "inspect",
        "traceback",
        "doctest",
        "coverage",
    }
)

_TEST_DIR_NAMES = frozenset({"test", "tests", "__tests__"})
_PYTHON_BIN_RE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|pypy3?(?:\.\d+)*)$")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_C_SIMPLE_ESCAPES: dict[str, int] = {
    "a": 0x07,
    "b": 0x08,
    "f": 0x0C,
    "n": 0x0A,
    "r": 0x0D,
    "t": 0x09,
    "v": 0x0B,
    "\\": 0x5C,
    '"': 0x22,
}


def kill_agent_processes(session: SandboxSession) -> str | None:
    """Kill every process the agent left running in ``session`` before grading starts.

    Background loops started by the policy (``setsid``, ``nohup``, ``&``) could otherwise re-plant
    ``conftest.py`` or write the reward file while verification runs. The kill is mandatory: a session
    without ``kill_sandbox_processes`` (or one whose kill fails) cannot be graded safely, so the caller
    must treat the episode as unscorable.

    Args:
        session: Episode sandbox session about to be graded.

    Returns:
        An error message when the agent's processes could not be killed (unscorable), else ``None``.
    """
    killer = getattr(session, "kill_sandbox_processes", None)
    if not callable(killer):
        return (
            f"{type(session).__name__} cannot kill agent background processes (no kill_sandbox_processes); "
            "refusing to grade an episode whose processes may still be running."
        )
    try:
        killer()
    except Exception as exc:
        return f"failed to kill agent background processes before grading: {exc}"
    return None


def _unquote_git_path(raw: str) -> str:
    """Decode a git C-quoted path (e.g. ``"b/my test.py"`` or ``"b/caf\\303\\251.py"``)."""
    text = raw.strip()
    if len(text) < 2 or not (text.startswith('"') and text.endswith('"')):
        return text
    inner = text[1:-1]
    buf = bytearray()
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if ch != "\\":
            buf.extend(ch.encode("utf-8"))
            i += 1
            continue
        i += 1
        if i >= n:
            buf.append(0x5C)
            break
        esc = inner[i]
        if esc in _C_SIMPLE_ESCAPES:
            buf.append(_C_SIMPLE_ESCAPES[esc])
            i += 1
        elif "0" <= esc <= "7":
            oct_digits = esc
            i += 1
            for _ in range(2):
                if i < n and "0" <= inner[i] <= "7":
                    oct_digits += inner[i]
                    i += 1
                else:
                    break
            buf.append(int(oct_digits, 8) & 0xFF)
        else:
            buf.extend(esc.encode("utf-8"))
            i += 1
    return buf.decode("utf-8", errors="replace")


def _consume_git_token(text: str, pos: int) -> tuple[str, int]:
    """Read one possibly C-quoted path token starting at ``pos``."""
    n = len(text)
    while pos < n and text[pos] == " ":
        pos += 1
    if pos >= n:
        return "", pos
    if text[pos] == '"':
        end = pos + 1
        while end < n:
            if text[end] == "\\" and end + 1 < n:
                end += 2
                continue
            if text[end] == '"':
                end += 1
                break
            end += 1
        return _unquote_git_path(text[pos:end]), end
    end = pos
    while end < n and text[end] != " ":
        end += 1
    return text[pos:end], end


def _split_diff_git_paths(rest: str) -> tuple[str, str] | None:
    """Split the two paths after ``diff --git `` (supporting C-quoted and space-containing paths)."""
    stripped = rest.strip()
    if not stripped:
        return None
    if stripped.startswith('"'):
        first, next_pos = _consume_git_token(stripped, 0)
        second_part = stripped[next_pos:].lstrip()
        if not second_part:
            return None
        second = _unquote_git_path(second_part) if second_part.startswith('"') else second_part
        return first, second
    # Unquoted first path; second path may be quoted (`a/old "b/new name"`) or unquoted (`a/x b/x`).
    if ' "b/' in stripped:
        left, right = stripped.split(' "b/', 1)
        return left, _unquote_git_path('"b/' + right)
    parts = stripped.split(" b/", 1)
    if len(parts) == 2:
        return parts[0], "b/" + parts[1]
    return None


def _normalize_repo_relpath(path: str) -> str | None:
    """Normalize a repository-relative path or return ``None`` if unsafe/empty."""
    if not path or "\x00" in path:
        return None
    cleaned = path.replace("\\", "/")
    if cleaned.startswith("/"):
        return None
    norm = posixpath.normpath(cleaned)
    if not norm or norm == "." or norm.startswith("/") or ".." in norm.split("/"):
        return None
    return norm


def _strip_diff_prefix(path: str) -> str:
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _is_test_infra_path(path: str) -> bool:
    base = posixpath.basename(path)
    return base in _TEST_INFRA_BASENAMES or base.endswith(_TEST_INFRA_SUFFIXES)


def _is_test_path(path: str) -> bool:
    """Return True when ``path`` looks like a repository test file or test directory."""
    norm = _normalize_repo_relpath(path)
    if norm is None:
        return False
    parts = norm.split("/")
    if any(part in _TEST_DIR_NAMES for part in parts):
        return True
    base = parts[-1]
    if base in {"test.py", "tests.py"}:
        return True
    if base.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py")):
        return True
    if ".test." in base or ".spec." in base:
        return True
    return False


def _is_runner_shadow_path(path: str, base_tracked_paths: collections.abc.Collection[str] = ()) -> bool:
    """Return True if ``path`` newly shadows a Python test runner or core stdlib module."""
    norm = _normalize_repo_relpath(path)
    if norm is None:
        return False
    parts = norm.split("/")
    top = parts[0]
    # Strip all extensions (`.py`, `.pyc`, `.cpython-313-x86_64-linux-gnu.so`, `.abi3.so`, etc.).
    stem = top.split(".", 1)[0] if len(parts) == 1 else top
    if stem not in _RUNNER_SHADOW_NAMES:
        return False
    if norm in base_tracked_paths:
        return False
    if len(parts) > 1:
        # Allow adding files inside a package directory that already existed at base (e.g. pytest repo itself).
        base_top_dirs = {p.split("/", 1)[0] for p in base_tracked_paths if "/" in p}
        if top in base_top_dirs:
            return False
    return True


def _patch_touched_paths(patch: str) -> set[str]:
    """Return every repository path a unified diff reads or writes (``/dev/null`` excluded)."""
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            pair = _split_diff_git_paths(line[len("diff --git ") :])
            if pair is not None:
                for raw in pair:
                    norm = _normalize_repo_relpath(_strip_diff_prefix(raw))
                    if norm is not None:
                        paths.add(norm)
        elif line.startswith(("--- ", "+++ ")):
            raw_target = line[4:].split("\t", 1)[0].strip()
            target = _unquote_git_path(raw_target)
            if target == "/dev/null":
                continue
            norm = _normalize_repo_relpath(_strip_diff_prefix(target))
            if norm is not None:
                paths.add(norm)
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            target = _unquote_git_path(line.split(" ", 2)[2].strip())
            norm = _normalize_repo_relpath(target)
            if norm is not None:
                paths.add(norm)
    return paths


def _extract_command_targets(
    commands: collections.abc.Sequence[str],
    base_tracked_paths: collections.abc.Collection[str] = (),
) -> tuple[set[str], set[str]]:
    """Extract ``(file_paths, dir_paths)`` referenced in ``fail_to_pass`` / ``pass_to_pass`` commands.

    Handles direct file paths (``check_p2p.py``), pytest node IDs (``tests/test_a.py::test_x``),
    dotted unittest module targets (``check_p2p``, ``pkg.check_p2p.TestCls``), and non-``tests/``
    directory arguments (``pytest checks``).
    """
    tracked_set = set(base_tracked_paths)
    tracked_dirs: set[str] = set()
    for p in tracked_set:
        parts = p.split("/")
        for k in range(1, len(parts)):
            tracked_dirs.add("/".join(parts[:k]))

    found_files: set[str] = set()
    found_dirs: set[str] = set()
    for cmd in commands:
        if not cmd:
            continue
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        skip_next = False
        for tok in tokens:
            if skip_next:
                skip_next = False
                continue
            if not tok or _ENV_ASSIGN_RE.match(tok):
                continue
            if tok in {";", "&&", "||", "|", "&"}:
                continue
            if tok == "-m":
                # Skip the runner module name itself (e.g. `pytest`, `unittest`).
                skip_next = True
                continue
            if tok.startswith("-"):
                continue
            raw_tok = tok.rstrip(";").split("::", 1)[0]
            if raw_tok in {".", "./"}:
                continue
            if raw_tok.startswith("./"):
                raw_tok = raw_tok[2:]
            norm = _normalize_repo_relpath(raw_tok)
            if norm is None:
                continue
            base = posixpath.basename(norm)
            if norm in tracked_set:
                found_files.add(norm)
                continue
            if norm in tracked_dirs:
                found_dirs.add(norm)
                for p in tracked_set:
                    if p.startswith(norm + "/"):
                        found_files.add(p)
                continue
            # Try dotted Python module target (e.g. `check_p2p`, `pkg.sub.TestCls.test_foo`).
            if "/" not in norm and all(seg.isidentifier() for seg in norm.split(".")):
                mod_parts = norm.split(".")
                matched_mod = False
                for k in range(len(mod_parts), 0, -1):
                    cand_py = "/".join(mod_parts[:k]) + ".py"
                    if cand_py in tracked_set:
                        found_files.add(cand_py)
                        matched_mod = True
                        break
                    cand_dir = "/".join(mod_parts[:k])
                    if cand_dir in tracked_dirs:
                        found_dirs.add(cand_dir)
                        for p in tracked_set:
                            if p.startswith(cand_dir + "/"):
                                found_files.add(p)
                        matched_mod = True
                        break
                if matched_mod:
                    continue
            if "." in base and not base.startswith("."):
                found_files.add(norm)
    return found_files, found_dirs


def _extract_command_file_paths(
    commands: collections.abc.Sequence[str],
    base_tracked_paths: collections.abc.Collection[str] = (),
) -> set[str]:
    """Extract repository file paths explicitly referenced in ``fail_to_pass`` / ``pass_to_pass`` commands."""
    files, _ = _extract_command_targets(commands, base_tracked_paths)
    return files


def _build_protected_predicate(
    test_patch: str | None,
    test_commands: collections.abc.Sequence[str],
    base_tracked_paths: collections.abc.Collection[str],
) -> tp.Callable[[str], bool]:
    """Return a predicate identifying every path an agent diff/untracked file must not alter."""
    patch_paths = _patch_touched_paths(test_patch or "")
    cmd_files, cmd_dirs = _extract_command_targets(test_commands, base_tracked_paths)
    tracked_set = frozenset(base_tracked_paths)

    # Exact protected files (whose path cannot also become a directory containing agent files) and
    # parent directories of protected files (which cannot be replaced by an agent regular file).
    protected_files: set[str] = set(patch_paths) | set(cmd_files)
    for p in tracked_set:
        if _is_test_infra_path(p) or _is_test_path(p):
            protected_files.add(p)

    protected_parent_dirs: set[str] = set(cmd_dirs)
    for p in protected_files:
        parts = p.split("/")
        for k in range(1, len(parts)):
            protected_parent_dirs.add("/".join(parts[:k]))

    def _is_protected(path: str) -> bool:
        norm = _normalize_repo_relpath(path)
        if norm is None:
            return True
        if norm == ".git" or norm.startswith(".git/"):
            return True
        if posixpath.basename(norm).startswith(".ed_"):
            return True
        if _is_test_infra_path(norm):
            return True
        if norm in protected_files or norm in protected_parent_dirs:
            return True
        if _is_test_path(norm):
            return True
        if _is_runner_shadow_path(norm, tracked_set):
            return True
        # Reject any path nested under a protected file (e.g. `test_calc.py/evil.py`) or a protected test dir
        # (e.g. `checks/evil.py`).
        parts = norm.split("/")
        for k in range(1, len(parts)):
            ancestor = "/".join(parts[:k])
            if ancestor in protected_files or ancestor in cmd_dirs:
                return True
        return False

    return _is_protected


def _section_touches_symlink(section_lines: list[str]) -> bool:
    for line in section_lines[1:8]:
        if line.startswith(("new file mode 120000", "old mode 120000", "new mode 120000", "deleted file mode 120000")):
            return True
        if line.startswith("index ") and line.rstrip().endswith(" 120000"):
            return True
    return False


def _filter_agent_patch(patch: str, is_protected: tp.Callable[[str], bool]) -> str:
    """Keep only non-protected, non-symlink file sections from a git unified diff."""
    if not patch:
        return ""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                sections.append(current)
            current = [line]
        else:
            if current:
                current.append(line)
    if current:
        sections.append(current)

    kept: list[str] = []
    for sec in sections:
        if _section_touches_symlink(sec):
            continue
        sec_text = "".join(sec)
        touched = _patch_touched_paths(sec_text)
        if not touched:
            continue
        if any(is_protected(p) for p in touched):
            continue
        kept.append(sec_text)
    return "".join(kept)


def capture_base_revision(session: SandboxSession) -> tuple[str | None, str | None]:
    """Return ``(base_sha, infra_error)`` for the session's current ``HEAD``."""
    res = session.exec(["git", "rev-parse", "HEAD"], env=_CLEAN_GIT_ENV, timeout=10.0)
    if res.infra_error:
        return None, res.infra_error
    sha = res.stdout.strip()
    return (sha if res.ok and sha else None), None


# Grading never lets the agent-writable repository configuration run anything: no system/global config, no
# replace refs, and every grading git command runs against a throw-away git dir (empty config, no hooks, no
# attributes, no filters, no fsmonitor) that only borrows the workspace's object store and index.
_CLEAN_GIT_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_TERMINAL_PROMPT": "0",
}
_CLEAN_GIT_PRELUDE = (
    "e=$(mktemp -d /tmp/.ed_git.XXXXXXXX) || exit 125; trap 'rm -rf \"$e\"' EXIT; "
    'export HOME="$e" XDG_CONFIG_HOME="$e"; '
    'git init -q "$e/w" >/dev/null 2>&1 || exit 125; w=$(pwd -P); '
    'export GIT_DIR="$e/w/.git" GIT_WORK_TREE="$w" GIT_OBJECT_DIRECTORY="$w/.git/objects" '
    'GIT_INDEX_FILE="$w/.git/index"; '
)
_CLEAN_GIT_SCRIPT = _CLEAN_GIT_PRELUDE + 'git -c core.fsmonitor= -c core.hooksPath=/dev/null "$@"'

# Arguments: delete paths, "--", then (mode, blob id, path) triples. Every parent directory that is a symlink is
# removed first (so writes cannot escape through it), each blob is hash-checked against the id recorded at reset
# (so a tampered object store cannot substitute content) and written as plain bytes; nothing runs a filter.
_RESTORE_SCRIPT = (
    _CLEAN_GIT_PRELUDE
    + r"""set -f
unlink_dirs() {
  acc=.; old_ifs=$IFS; IFS=/
  for c in $(dirname -- "$1"); do
    [ "$c" = . ] && continue
    acc="$acc/$c"
    if [ -L "$acc" ]; then rm -f -- "$acc" || { IFS=$old_ifs; return 1; }; fi
  done
  IFS=$old_ifs
}
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  unlink_dirs "$1" || exit 1; rm -rf -- "$1" || exit 1; shift
done
[ $# -gt 0 ] && shift
while [ $# -ge 3 ]; do
  mode=$1; blob=$2; p=$3; shift 3
  unlink_dirs "$p" || exit 1
  rm -rf -- "$p" || exit 1
  mkdir -p -- "$(dirname -- "$p")" || exit 1
  git cat-file blob "$blob" > "$e/blob" || { echo "base blob of $p is missing" >&2; exit 1; }
  got=$(git hash-object --no-filters -t blob "$e/blob") || exit 1
  [ "$got" = "$blob" ] || { echo "base blob of $p failed its hash check" >&2; exit 1; }
  if [ "$mode" = 120000 ]; then
    ln -s -- "$(cat "$e/blob")" "$p" || exit 1
  else
    cp -- "$e/blob" "$p" || exit 1
    if [ "$mode" = 100755 ]; then chmod 755 "$p"; else chmod 644 "$p"; fi
  fi
done
"""
)

GradingSnapshot = dict[str, tuple[str, str]]
"""``path -> (git mode, blob id)`` of every protected path that exists at the base revision."""


def run_clean_git(
    session: SandboxSession,
    args: list[str],
    *,
    stdin: str | None = None,
    timeout: float = 30.0,
) -> ExecResult:
    """Run ``git <args>`` in ``session`` without reading or executing anything the repository configures.

    The command runs against a fresh, empty git dir that only borrows the workspace's object store and index,
    with system/global config, replace refs, hooks and fsmonitor disabled, so agent-written ``.git/config``,
    ``.git/info/attributes`` filters or hooks never run.
    """
    return session.exec(
        ["/bin/sh", "-c", _CLEAN_GIT_SCRIPT, "sh", *args],
        env=_CLEAN_GIT_ENV,
        stdin=stdin,
        timeout=timeout,
    )


def _list_base_tree(session: SandboxSession, base_sha: str) -> tuple[list[tuple[str, str, str]] | None, str | None]:
    """Return ``([(mode, blob_sha, path), ...], infra_error)`` for all blobs at ``base_sha``."""
    tree_file = f".git/.ed_base_tree_{secrets.token_hex(8)}"
    list_script = _CLEAN_GIT_PRELUDE + 'git ls-tree -r -z --full-tree "$1" > "$w/$2"'
    res = session.exec(
        ["/bin/sh", "-c", list_script, "sh", base_sha, tree_file],
        env=_CLEAN_GIT_ENV,
        timeout=60.0,
    )
    if res.infra_error or not res.ok:
        session.exec(["rm", "-f", "--", tree_file], timeout=10.0)
        return None, res.infra_error or f"git ls-tree failed: {res.combined_output(1024)}"
    try:
        listing = session.read_file(tree_file)
    except Exception as exc:
        return None, f"failed to read base tree listing: {exc}"
    finally:
        session.exec(["rm", "-f", "--", tree_file], timeout=10.0)
    entries: list[tuple[str, str, str]] = []
    for entry in listing.split("\0"):
        meta, sep, path = entry.partition("\t")
        fields = meta.split()
        if not sep or len(fields) != 3 or fields[1] != "blob":
            continue
        entries.append((fields[0], fields[2], path))
    return entries, None


def capture_grading_snapshot(
    session: SandboxSession,
    base_sha: str,
    test_patch: str | None,
    *,
    test_commands: collections.abc.Sequence[str] = (),
) -> tuple[GradingSnapshot | None, str | None]:
    """Record the base-revision blob of every path grading protects. Call it at reset, before the agent acts.

    Args:
        session: Episode sandbox session whose working directory is a git checkout.
        base_sha: Commit recorded at reset.
        test_patch: Evaluation patch whose touched paths are protected too.
        test_commands: Evaluation test commands (``fail_to_pass`` / ``pass_to_pass``) whose referenced files
            are also protected.

    Returns:
        ``(snapshot, infra_error)``; ``snapshot`` is ``None`` when the listing failed.
    """
    entries, err = _list_base_tree(session, base_sha)
    if entries is None:
        return None, err
    all_paths = [path for _, _, path in entries]
    is_protected = _build_protected_predicate(test_patch, test_commands, all_paths)
    snapshot: GradingSnapshot = {}
    for mode, blob_id, path in entries:
        if is_protected(path):
            snapshot[path] = (mode, blob_id)
    return snapshot, None


def restore_grading_files(
    session: SandboxSession,
    base_sha: str,
    test_patch: str | None,
    snapshot: GradingSnapshot | None = None,
    *,
    test_commands: collections.abc.Sequence[str] = (),
) -> str | None:
    """Reset test-patch paths, test files, and test-infrastructure files to their base content before grading.

    Every live test-infrastructure file, runner-shadowing file/package, and protected path is deleted, then
    each one that exists at the base revision is rewritten from its recorded blob (undoing edits, deletions
    and symlink swaps). No git command here reads the repository's own config, attributes or hooks.

    Args:
        session: Episode sandbox session whose working directory is a git checkout.
        base_sha: Commit recorded at reset, before the agent acted.
        test_patch: Evaluation patch that will be applied afterwards (its paths are reset too).
        snapshot: Result of :func:`capture_grading_snapshot` taken at reset. When omitted the base tree is
            listed now, which trusts an object store the agent could have written to.
        test_commands: Optional test commands whose referenced files should also be restored.

    Returns:
        An error message when the restore could not run, else ``None``.
    """
    if snapshot is None:
        snapshot, err = capture_grading_snapshot(session, base_sha, test_patch, test_commands=test_commands)
        if snapshot is None:
            return err or "could not list the base revision"
    # Restore user permissions on non-symlink entries first so `chmod 000` by the agent cannot break cleanup.
    chmod_res = session.exec(
        ["find", ".", "-path", "./.git", "-prune", "-o", "!", "-type", "l", "-exec", "chmod", "u+rwx", "--", "{}", "+"],
        timeout=30.0,
    )
    if chmod_res.infra_error:
        return chmod_res.infra_error

    name_tests: list[str] = []
    for name in sorted(_TEST_INFRA_BASENAMES) + [f"*{s}" for s in _TEST_INFRA_SUFFIXES]:
        name_tests += [*(["-o"] if name_tests else []), "-name", name]
    find_argv = ["find", ".", "-path", "./.git", "-prune", "-o", "(", *name_tests, ")", "-prune"]
    find_res = session.exec([*find_argv, "-exec", "rm", "-rf", "--", "{}", "+"], timeout=60.0)
    if find_res.infra_error or not find_res.ok:
        return find_res.infra_error or f"failed to remove test-infrastructure files: {find_res.combined_output(1024)}"

    to_delete_set = {p for p in _patch_touched_paths(test_patch or "") if p not in snapshot}
    to_delete_set.update(p for p in _extract_command_file_paths(test_commands) if p not in snapshot)
    # Also remove top-level runner-shadowing modules/packages that did not exist at base.
    snapshot_top_dirs = {p.split("/", 1)[0] for p in snapshot if "/" in p}
    for name in _RUNNER_SHADOW_NAMES:
        if f"{name}.py" not in snapshot:
            to_delete_set.add(f"{name}.py")
        if name not in snapshot_top_dirs:
            to_delete_set.add(name)

    to_delete = sorted(to_delete_set)
    triples = [item for path in sorted(snapshot) for item in (*snapshot[path], path)]
    res = session.exec(
        ["/bin/sh", "-c", _RESTORE_SCRIPT, "sh", *to_delete, "--", *triples], env=_CLEAN_GIT_ENV, timeout=120.0
    )
    if res.infra_error or not res.ok:
        return res.infra_error or f"failed to restore test files: {res.combined_output(1024)}"
    return None


def _seed_repo_dir_files(repo_dir: str, base_commit: str | None) -> tuple[dict[str, bytes] | None, str | None]:
    """Read a clean, symlink-free, history-free file mapping from host ``repo_dir``.

    When ``base_commit`` is set, extracts the exact tree at ``base_commit`` via ``git archive`` so no
    descendant fix commits, branches, tags, reflogs, or host ``.git/config`` enter the sandbox. When
    ``base_commit`` is omitted, walks ``repo_dir`` skipping ``.git`` and all host symlinks.
    """
    root = os.path.abspath(repo_dir)
    if not os.path.isdir(root):
        return None, f"repo_dir does not exist or is not a directory: {repo_dir!r}"

    if base_commit is not None:
        env = {**os.environ, **_CLEAN_GIT_ENV}
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    root,
                    "-c",
                    "core.fsmonitor=",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "archive",
                    "--format=tar",
                    str(base_commit),
                ],
                capture_output=True,
                env=env,
                timeout=60.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"Failed to archive base_commit {base_commit!r} from repo_dir: {exc}"
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            return None, f"Failed to checkout base_commit {base_commit!r}: {err}"
        copied: dict[str, bytes] = {}
        try:
            with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    rel = _normalize_repo_relpath(member.name)
                    if rel is None or rel == ".git" or rel.startswith(".git/"):
                        continue
                    extracted = tf.extractfile(member)
                    if extracted is not None:
                        copied[rel] = extracted.read()
        except tarfile.TarError as exc:
            return None, f"Failed to unpack git archive for base_commit {base_commit!r}: {exc}"
        return copied, None

    copied = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", ".pytest_cache", "__pycache__"} and not os.path.islink(os.path.join(dirpath, d))
        ]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            if os.path.islink(full) or not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            norm = _normalize_repo_relpath(rel)
            if norm is None or norm == ".git" or norm.startswith(".git/"):
                continue
            with open(full, "rb") as f:
                copied[norm] = f.read()
    return copied, None


def init_clean_git_repo(session: SandboxSession, *, email: str = "swe@easydel.local", name: str = "EasyDeL SWE") -> str | None:
    """Initialize a fresh single-commit git repository in ``session`` without reading host git config."""
    script = (
        "git init -q && "
        f"git config user.email {shlex.quote(email)} && "
        f"git config user.name {shlex.quote(name)} && "
        "git add -A && "
        "git -c core.hooksPath=/dev/null commit -q --allow-empty -m 'base'"
    )
    res = session.exec(["/bin/sh", "-c", script], env=_CLEAN_GIT_ENV, timeout=15.0)
    if res.infra_error:
        return res.infra_error
    if not res.ok:
        return f"Failed to initialize clean git repository: {res.combined_output(1024)}"
    return None


def _split_shell_chain(cmd: str) -> tuple[list[tuple[str, str]], str] | None:
    """Split top-level ``;`` and ``&&`` segments outside quotes, returning ``(prefix_segments, last_segment)``.

    Returns ``None`` if the final segment contains redirections, pipes, subshells, or backgrounding.
    """
    segments: list[tuple[str, str]] = []  # (segment_text, trailing_op)
    buf: list[str] = []
    in_single = False
    in_double = False
    escape = False
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if escape:
            buf.append(ch)
            escape = False
            i += 1
            continue
        if in_single:
            buf.append(ch)
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if ch == "\\":
                buf.append(ch)
                escape = True
                i += 1
                continue
            buf.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            i += 1
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            buf.append(ch)
            i += 1
            continue
        if ch == ";":
            segments.append(("".join(buf).strip(), ";"))
            buf = []
            i += 1
            continue
        if ch == "&" and i + 1 < n and cmd[i + 1] == "&":
            segments.append(("".join(buf).strip(), "&&"))
            buf = []
            i += 2
            continue
        buf.append(ch)
        i += 1

    if in_single or in_double:
        return None
    last = "".join(buf).strip()
    if not last:
        return None

    # Ensure the last segment does not contain shell pipes, redirections, `||`, `&`, or command substitution.
    in_single = False
    in_double = False
    escape = False
    for idx, ch in enumerate(last):
        if escape:
            escape = False
            continue
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_double = False
            elif ch == "`" or (ch == "$" and idx + 1 < len(last) and last[idx + 1] == "("):
                return None
            continue
        if ch == "\\":
            escape = True
        elif ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch in {"|", ">", "<", "&", "`"} or (ch == "$" and idx + 1 < len(last) and last[idx + 1] == "("):
            return None

    prefix = [(seg, op) for seg, op in segments if seg]
    return prefix, last


def _parse_python_test_command(cmd: str) -> dict[str, tp.Any] | None:
    """Recognize pytest, unittest, and Python script/``-c`` test commands for tamper-resistant execution."""
    split = _split_shell_chain(cmd)
    if split is None:
        return None
    prefix_segments, last = split
    try:
        tokens = shlex.split(last)
    except ValueError:
        return None
    if not tokens:
        return None

    extra_env: dict[str, str] = {}
    idx = 0
    while idx < len(tokens) and _ENV_ASSIGN_RE.match(tokens[idx]):
        k, v = tokens[idx].split("=", 1)
        extra_env[k] = v
        idx += 1
    argv = tokens[idx:]
    if not argv:
        return None

    exe_base = posixpath.basename(argv[0])
    if exe_base in {"pytest", "py.test"} or exe_base.startswith("pytest-"):
        return {
            "kind": "pytest",
            "prefix": prefix_segments,
            "extra_env": extra_env,
            "py_bin": None,
            "pytest_bin": argv[0],
            "args": argv[1:],
        }

    if not _PYTHON_BIN_RE.match(exe_base):
        return None

    py_bin = argv[0]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-m" and i + 1 < len(argv):
            mod = argv[i + 1]
            if mod == "pytest":
                return {
                    "kind": "pytest",
                    "prefix": prefix_segments,
                    "extra_env": extra_env,
                    "py_bin": py_bin,
                    "pytest_bin": None,
                    "args": argv[i + 2 :],
                }
            if mod == "unittest":
                return {
                    "kind": "unittest",
                    "prefix": prefix_segments,
                    "extra_env": extra_env,
                    "py_bin": py_bin,
                    "pytest_bin": None,
                    "args": argv[i + 2 :],
                }
            return None
        if arg == "-c" and i + 1 < len(argv):
            return {
                "kind": "py_code",
                "prefix": prefix_segments,
                "extra_env": extra_env,
                "py_bin": py_bin,
                "code": argv[i + 1],
                "args": argv[i + 2 :],
            }
        if arg in {"-W", "-X"} and i + 1 < len(argv):
            i += 2
            continue
        if arg.startswith("-"):
            if arg == "-":
                return None
            i += 1
            continue
        return {
            "kind": "py_script",
            "prefix": prefix_segments,
            "extra_env": extra_env,
            "py_bin": py_bin,
            "script": arg,
            "args": argv[i + 1 :],
        }
    return None


_CHILD_SYSPATH_HELPER = r"""
    def _insert_after_stdlib(paths_to_add):
        insert_at = 0
        for idx, entry in enumerate(sys.path):
            low = entry.replace("\\", "/").lower()
            if "site-packages" in low or "dist-packages" in low:
                break
            insert_at = idx + 1
        for p in reversed(paths_to_add):
            if p and p not in sys.path:
                sys.path.insert(insert_at, p)

    def _env_pythonpaths(cwd):
        out = []
        for raw in os.environ.get("PYTHONPATH", "").split(os.pathsep):
            if not raw:
                continue
            out.append(os.path.realpath(raw if os.path.isabs(raw) else os.path.join(cwd, raw)))
        return out
"""

_PYTEST_CHILD_SRC = (
    r"""
def _run():
    import json, os, sys
"""
    + _CHILD_SYSPATH_HELPER
    + r"""
    tok = __CHILD_TOK__
    args = __ARGS__
    write_fd = os.write
    close_fd = os.close
    hard_exit = os._exit
    dumps = json.dumps

    try:
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.close(null_fd)
    except OSError:
        pass

    rep_fd = os.dup(1)
    os.dup2(2, 1)

    cwd = os.path.realpath(os.getcwd())
    sys.path[:] = [p for p in sys.path if p not in ("", ".") and os.path.realpath(p) != cwd]
    import pytest as _pytest

    _insert_after_stdlib([*_env_pythonpaths(cwd), cwd])

    class _Plugin:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.errors = 0
            self.skipped = 0
            self.session_finished = False

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                if report.passed:
                    self.passed += 1
                elif report.failed:
                    self.failed += 1
                elif report.skipped:
                    self.skipped += 1
            elif report.failed:
                self.errors += 1

        def pytest_sessionfinish(self, session, exitstatus):
            self.session_finished = True

    plugin = _Plugin()
    try:
        rc = int(_pytest.main(args, plugins=[plugin]))
    except BaseException:
        rc = 1

    ok = rc == 0 and plugin.session_finished and plugin.passed > 0 and plugin.failed == 0 and plugin.errors == 0
    line = tok + " " + dumps(
        {
            "ok": ok,
            "rc": rc,
            "passed": plugin.passed,
            "failed": plugin.failed,
            "errors": plugin.errors,
            "session_finished": plugin.session_finished,
        }
    ) + "\n"
    write_fd(rep_fd, line.encode("utf-8"))
    close_fd(rep_fd)
    hard_exit(0 if ok else 1)


_run()
"""
)

_UNITTEST_CHILD_SRC = (
    r"""
def _run():
    import json, os, sys
"""
    + _CHILD_SYSPATH_HELPER
    + r"""
    tok = __CHILD_TOK__
    args = __ARGS__
    write_fd = os.write
    close_fd = os.close
    hard_exit = os._exit
    dumps = json.dumps

    try:
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.close(null_fd)
    except OSError:
        pass

    rep_fd = os.dup(1)
    os.dup2(2, 1)

    cwd = os.path.realpath(os.getcwd())
    sys.path[:] = [p for p in sys.path if p not in ("", ".") and os.path.realpath(p) != cwd]
    import unittest as _unittest

    _insert_after_stdlib([*_env_pythonpaths(cwd), cwd])

    passed = 0
    failed = 0
    errors = 0
    finished = False
    ok = False
    try:
        prog = _unittest.main(module=None, argv=["unittest", *args], exit=False)
        res = getattr(prog, "result", None)
        if res is not None:
            finished = True
            failed = len(getattr(res, "failures", ()))
            errors = len(getattr(res, "errors", ()))
            ran = int(getattr(res, "testsRun", 0))
            passed = max(0, ran - failed - errors)
            ok = bool(ran > 0 and res.wasSuccessful())
    except BaseException:
        ok = False

    line = tok + " " + dumps(
        {
            "ok": ok,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "session_finished": finished,
        }
    ) + "\n"
    write_fd(rep_fd, line.encode("utf-8"))
    close_fd(rep_fd)
    hard_exit(0 if ok else 1)


_run()
"""
)

_PYTHON_SCRIPT_CHILD_SRC = (
    r"""
def _run():
    import json, os, runpy, sys
"""
    + _CHILD_SYSPATH_HELPER
    + r"""
    tok = __CHILD_TOK__
    mode = __MODE__
    target = __TARGET__
    args = __ARGS__
    write_fd = os.write
    close_fd = os.close
    hard_exit = os._exit
    dumps = json.dumps
    realpath = os.path.realpath

    try:
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.close(null_fd)
    except OSError:
        pass

    rep_fd = os.dup(1)
    os.dup2(2, 1)

    cwd = realpath(os.getcwd())
    sys.path[:] = [p for p in sys.path if p not in ("", ".") and realpath(p) != cwd]
    extra_paths = _env_pythonpaths(cwd)
    if mode == "py_script":
        expected_file = realpath(target)
        script_dir = os.path.dirname(expected_file)
        _insert_after_stdlib([*extra_paths, script_dir, cwd])
        sys.argv = [target, *args]
    else:
        expected_file = "<ed_test_code>"
        _insert_after_stdlib([*extra_paths, cwd])
        sys.argv = ["-c", *args]

    def _in_workspace(path_str):
        if not path_str or path_str.startswith("<"):
            return False
        rp = realpath(path_str)
        return rp == cwd or rp.startswith(cwd + os.sep)

    ok = False
    try:
        if mode == "py_script":
            runpy.run_path(target, run_name="__main__")
        else:
            exec(compile(target, "<ed_test_code>", "exec"), {"__name__": "__main__"})
        ok = True
    except SystemExit as se:
        if se.code in (None, 0):
            frames = []
            tb = se.__traceback__
            while tb is not None:
                frames.append(tb.tb_frame)
                tb = tb.tb_next
            # Locate the start of the test script inside the traceback so wrapper frames are excluded.
            start_idx = -1
            for idx, fr in enumerate(frames):
                fn = fr.f_code.co_filename
                if (mode == "py_script" and fn and not fn.startswith("<") and realpath(fn) == expected_file) or (
                    mode != "py_script" and fn == expected_file
                ):
                    start_idx = idx
                    break
            if start_idx >= 0:
                sub = frames[start_idx:]
                deepest_ws = None
                for fr in sub:
                    fn = fr.f_code.co_filename
                    if (mode == "py_script" and _in_workspace(fn)) or (
                        mode != "py_script" and (fn == expected_file or _in_workspace(fn))
                    ):
                        deepest_ws = fn
                is_expected = (
                    deepest_ws is not None
                    and (
                        (mode == "py_script" and realpath(deepest_ws) == expected_file)
                        or (mode != "py_script" and deepest_ws == expected_file)
                    )
                )
                if is_expected:
                    origin_frame = sub[-1]
                    origin_fn = origin_frame.f_code.co_filename
                    if (mode == "py_script" and origin_fn and not origin_fn.startswith("<") and realpath(origin_fn) == expected_file) or (
                        mode != "py_script" and origin_fn == expected_file
                    ):
                        ok = True
                    elif not _in_workspace(origin_fn):
                        # Exit originated in a trusted stdlib/runner module (e.g. unittest.main()) invoked
                        # directly by the trusted test script. If it is unittest.TestProgram, require >= 1 pass.
                        prog_self = origin_frame.f_locals.get("self")
                        prog_res = getattr(prog_self, "result", None)
                        if prog_res is not None and hasattr(prog_res, "testsRun") and hasattr(prog_res, "wasSuccessful"):
                            ok = bool(prog_res.testsRun > 0 and prog_res.wasSuccessful())
                        else:
                            ok = True
    except BaseException:
        ok = False

    line = tok + " " + dumps({"ok": ok, "session_finished": True}) + "\n"
    write_fd(rep_fd, line.encode("utf-8"))
    close_fd(rep_fd)
    hard_exit(0 if ok else 1)


_run()
"""
)

# Trusted parent verdict gate: runs with PR_SET_DUMPABLE=0, closes inherited extra FDs, and keeps the outer
# verdict nonce and real stdout FD strictly inside the parent process (never visible to the child running
# untrusted repository code).
_VERDICT_PARENT_WRAPPER = r"""
def _ed_parent_gate():
    import json, os, secrets, subprocess, sys

    nonce = __ED_NONCE__
    kind = __ED_KIND__
    child_template = __ED_CHILD_SRC__

    try:
        import ctypes

        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE=0: block child ptrace & /proc/<ppid>/fd reads.
    except Exception:
        pass

    try:
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.close(null_fd)
    except OSError:
        pass

    os.closerange(3, 4096)
    out_fd = os.dup(1)
    try:
        os.set_inheritable(out_fd, False)
    except OSError:
        pass
    os.dup2(2, 1)

    child_tok = secrets.token_hex(24)
    child_src = child_template.replace("__CHILD_TOK__", repr(child_tok))

    ok = False
    proc = subprocess.Popen(
        [sys.executable, "-I", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    try:
        out_bytes, err_bytes = proc.communicate(child_src.encode("utf-8"))
    except BaseException:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
        out_bytes, err_bytes = b"", b""

    if err_bytes:
        try:
            os.write(2, err_bytes[-8192:])
        except OSError:
            pass

    if proc.returncode == 0:
        prefix = child_tok + " "
        text = out_bytes.decode("utf-8", "replace")
        carrying = [ln for ln in text.splitlines() if child_tok in ln]
        if len(carrying) == 1 and carrying[0].startswith(prefix):
            try:
                rec = json.loads(carrying[0][len(prefix) :])
            except ValueError:
                rec = None
            if isinstance(rec, dict) and rec.get("ok") is True and rec.get("session_finished") is True:
                if kind in ("pytest", "unittest"):
                    ok = (
                        int(rec.get("passed", 0)) > 0
                        and int(rec.get("failed", 0)) == 0
                        and int(rec.get("errors", 0)) == 0
                    )
                else:
                    ok = True

    os.write(out_fd, (nonce + " " + json.dumps({"ok": bool(ok)}) + "\n").encode("utf-8"))
    os.close(out_fd)
    os._exit(0 if ok else 1)


_ed_parent_gate()
"""


def _parse_test_verdict(stdout: str, nonce: str) -> bool:
    """Return True iff ``stdout`` contains exactly one ``<nonce> <json>`` verdict with ``ok is True``."""
    prefix = nonce + " "
    carrying = [line for line in stdout.splitlines() if nonce in line]
    if len(carrying) != 1 or not carrying[0].startswith(prefix):
        return False
    try:
        payload = json.loads(carrying[0][len(prefix) :])
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("ok") is True


def _resolve_pytest_shell_snippet(pytest_bin: str, fallback_py: str) -> str:
    quoted_pt = shlex.quote(pytest_bin)
    quoted_fb = shlex.quote(fallback_py)
    return (
        f'_ed_py=""; _ed_pt=$(command -v {quoted_pt} 2>/dev/null || true); '
        'if [ -n "$_ed_pt" ] && [ -r "$_ed_pt" ]; then '
        '  _ed_line=$(head -n 1 "$_ed_pt" 2>/dev/null || true); '
        '  case "$_ed_line" in '
        "    '#!'*) _ed_c1=$(printf '%s' \"${_ed_line#\\#!}\" | awk '{print $1}'); "
        '      case "$_ed_c1" in '
        "        */env) _ed_c2=$(printf '%s' \"${_ed_line#\\#!}\" | awk '{print $2}'); "
        '          [ -n "$_ed_c2" ] && _ed_py="$_ed_c2" ;; '
        '        *) [ -x "$_ed_c1" ] && _ed_py="$_ed_c1" ;; '
        "      esac ;; "
        "  esac; "
        "fi; "
        f'[ -z "$_ed_py" ] && _ed_py={quoted_fb}; '
    )


def run_verified_test_command(session: SandboxSession, cmd: str, *, timeout: float) -> tuple[bool, str | None]:
    """Run one verification command and return ``(passed, infra_error)``.

    For Python/pytest/unittest commands, executes through an isolated nonce-verified runner so
    ``os._exit(0)``, imported ``sys.exit(0)``, zero-test collection, or workspace ``pytest/`` shadowing
    cannot forge a passing result.
    """
    parsed = _parse_python_test_command(cmd)
    if parsed is None:
        res = session.run_bash(cmd, timeout=timeout)
        if res.infra_error:
            return False, res.infra_error
        return res.ok, None

    nonce = f"__ED_TEST_VERDICT_{secrets.token_hex(16)}__"
    kind = parsed["kind"]
    if kind == "pytest":
        child_src = _PYTEST_CHILD_SRC.replace("__ARGS__", repr(list(parsed["args"])))
    elif kind == "unittest":
        child_src = _UNITTEST_CHILD_SRC.replace("__ARGS__", repr(list(parsed["args"])))
    elif kind == "py_script":
        child_src = (
            _PYTHON_SCRIPT_CHILD_SRC.replace("__MODE__", repr("py_script"))
            .replace("__TARGET__", repr(parsed["script"]))
            .replace("__ARGS__", repr(list(parsed["args"])))
        )
    else:
        child_src = (
            _PYTHON_SCRIPT_CHILD_SRC.replace("__MODE__", repr("py_code"))
            .replace("__TARGET__", repr(parsed["code"]))
            .replace("__ARGS__", repr(list(parsed["args"])))
        )
    wrapper = (
        _VERDICT_PARENT_WRAPPER.replace("__ED_NONCE__", repr(nonce))
        .replace("__ED_KIND__", repr(kind))
        .replace("__ED_CHILD_SRC__", repr(child_src))
    )

    extra_env = dict(parsed["extra_env"])
    extra_env.setdefault("PYTHONNOUSERSITE", "1")
    extra_env.setdefault("PYTHONSTARTUP", "")
    # Prevent BLAS/OpenMP libraries loaded by pytest plugins (e.g. numpy/jaxtyping) from spawning dozens of
    # threads that exhaust the sandbox's RLIMIT_AS / pids_limit on high-core hosts.
    for _thr_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        extra_env.setdefault(_thr_var, "1")
    prefix_segments: list[tuple[str, str]] = parsed["prefix"]
    py_bin: str | None = parsed["py_bin"]

    if not prefix_segments and py_bin is not None:
        res = session.exec([py_bin, "-I", "-"], env=extra_env, stdin=wrapper, timeout=timeout)
    else:
        parts: list[str] = []
        for seg, op in prefix_segments:
            # Redirect prefix stdin from /dev/null so prefix commands never consume the wrapper script.
            parts.append(f"{{ {seg}; }} </dev/null {op} ")
        if py_bin is not None:
            launch = f"exec {shlex.quote(py_bin)} -I -"
        else:
            pytest_bin = parsed["pytest_bin"] or "pytest"
            fallback_py = sys.executable if isinstance(session, LocalSandboxSession) else "python3"
            if isinstance(session, LocalSandboxSession) and shutil.which(pytest_bin) is None:
                launch = f"exec {shlex.quote(sys.executable)} -I -"
            else:
                launch = _resolve_pytest_shell_snippet(pytest_bin, fallback_py) + 'exec "$_ed_py" -I -'
        shell_cmd = "".join(parts) + launch
        res = session.exec(["/bin/sh", "-c", shell_cmd], env=extra_env, stdin=wrapper, timeout=timeout)

    if res.infra_error:
        return False, res.infra_error
    if not res.ok:
        return False, None
    return _parse_test_verdict(res.stdout, nonce), None


def extract_agent_workspace_changes(
    session: SandboxSession,
    base_sha: str,
    *,
    test_patch: str | None,
    test_commands: collections.abc.Sequence[str],
    base_tracked_paths: collections.abc.Collection[str],
    max_patch_chars: int = 4 * 1024 * 1024,
    max_untracked_file_chars: int = 1024 * 1024,
    max_untracked_files: int = 64,
) -> tuple[str, dict[str, str], str | None, str | None]:
    """Extract only allowed non-test edits and untracked files from the agent's session.

    Returns:
        ``(safe_patch, untracked_files, tamper_reason, infra_error)``:
        - ``infra_error`` is set only for genuine sandbox/daemon errors (unscorable / ``discard=True``).
        - ``tamper_reason`` is set when the agent broke repository metadata (e.g. deleted ``.git/objects``
          or ``chmod 000``), which must score ``0.0`` rather than ``discard=True``.
    """
    is_protected = _build_protected_predicate(test_patch, test_commands, base_tracked_paths)
    nonce = secrets.token_hex(12)
    diff_file = f".ed_diff_{nonce}"
    untracked_file = f".ed_untracked_{nonce}"

    diff_script = _CLEAN_GIT_PRELUDE + 'git -c core.fsmonitor= -c core.hooksPath=/dev/null diff --binary "$1" > "$w/$2"'
    diff_res = session.exec(["/bin/sh", "-c", diff_script, "sh", base_sha, diff_file], env=_CLEAN_GIT_ENV, timeout=30.0)
    if diff_res.infra_error:
        session.exec(["rm", "-f", "--", diff_file], timeout=5.0)
        return "", {}, None, diff_res.infra_error
    if not diff_res.ok:
        session.exec(["rm", "-f", "--", diff_file], timeout=5.0)
        return "", {}, f"repository metadata tampered: {diff_res.combined_output(512)}", None
    try:
        raw_patch = session.read_file(diff_file, max_chars=max_patch_chars)
    except SandboxInfraError as exc:
        return "", {}, None, str(exc)
    except Exception as exc:
        return "", {}, f"could not read agent diff: {exc}", None
    finally:
        session.exec(["rm", "-f", "--", diff_file], timeout=5.0)

    if len(raw_patch) > max_patch_chars:
        return "", {}, "agent diff exceeded maximum grading size", None
    safe_patch = _filter_agent_patch(raw_patch, is_protected)

    ls_script = (
        _CLEAN_GIT_PRELUDE
        + 'git -c core.fsmonitor= -c core.hooksPath=/dev/null ls-files --others --exclude-standard -z > "$w/$1"'
    )
    ls_res = session.exec(["/bin/sh", "-c", ls_script, "sh", untracked_file], env=_CLEAN_GIT_ENV, timeout=15.0)
    if ls_res.infra_error:
        session.exec(["rm", "-f", "--", untracked_file], timeout=5.0)
        return "", {}, None, ls_res.infra_error
    if not ls_res.ok:
        session.exec(["rm", "-f", "--", untracked_file], timeout=5.0)
        return "", {}, f"repository untracked listing tampered: {ls_res.combined_output(512)}", None
    try:
        raw_untracked = session.read_file(untracked_file, max_chars=max_patch_chars)
    except SandboxInfraError as exc:
        return "", {}, None, str(exc)
    except Exception as exc:
        return "", {}, f"could not read untracked file list: {exc}", None
    finally:
        session.exec(["rm", "-f", "--", untracked_file], timeout=5.0)

    untracked_files: dict[str, str] = {}
    for rel in raw_untracked.split("\0"):
        norm = _normalize_repo_relpath(rel)
        if norm is None or is_protected(norm):
            continue
        if len(untracked_files) >= max_untracked_files:
            break
        # Guard against symlinks/FIFOs/directories; skip non-regular or oversized files.
        try:
            content = session.read_file(norm, max_chars=max_untracked_file_chars)
        except SandboxInfraError as exc:
            return "", {}, None, str(exc)
        except Exception:
            continue
        if len(content) <= max_untracked_file_chars:
            untracked_files[norm] = content

    return safe_patch, untracked_files, None, None


def apply_patch_via_stdin(session: SandboxSession, patch_text: str, *, timeout: float = 15.0) -> ExecResult:
    """Apply a unified diff via stdin without creating a predictable patch file in the workspace."""
    return run_clean_git(session, ["apply", "-"], stdin=patch_text, timeout=timeout)


def unscorable_step(error: Exception | str) -> StepResult:
    """Terminal infra-failure result flagged unscorable.

    ``info["discard"] = True`` means the grader could not produce a score (sandbox/infra error, judge failure).
    Its ``reward`` (0.0) is a placeholder and must not be trained on; reward adapters report ``None`` instead.
    """
    result = SandboxEnvironment._infra_failure_step(error)
    result.info["discard"] = True
    return result


def verification_resolved(f2p_results: list[bool], p2p_results: list[bool], *, has_f2p: bool, has_p2p: bool) -> bool:
    """Resolved iff at least one test ran, every fail-to-pass test passed and no pass-to-pass test regressed."""
    if not (has_f2p or has_p2p):
        return False
    f2p_ok = bool(f2p_results) and all(f2p_results) if has_f2p else True
    p2p_ok = bool(p2p_results) and all(p2p_results) if has_p2p else True
    return f2p_ok and p2p_ok


@register_environment(["swe_repo", "swe"])
class SWERepoEnvironment(SandboxEnvironment):
    """Repository issue-resolution environment with F2P / P2F test verification.

    Initializes a git working tree (from an in-memory ``files`` mapping, a host
    ``repo_dir`` snapshot, or an existing repository inside a Kiri/Docker image),
    executes multi-turn ``bash`` / ``read_file`` / ``write_file`` / ``edit_file``
    tool calls, and grades final submissions in a fresh, untouched checkout by
    applying only the agent's non-test diff plus ``test_patch`` and verifying
    all ``fail_to_pass`` and ``pass_to_pass`` commands.
    """

    def __init__(
        self,
        problem_statement: str = "",
        *,
        files: collections.abc.Mapping[str, bytes | str] | None = None,
        repo_dir: str | None = None,
        base_commit: str | None = None,
        test_patch: str | None = None,
        fail_to_pass: collections.abc.Sequence[str] = (),
        pass_to_pass: collections.abc.Sequence[str] = (),
        tool_caller: str | None = None,
        tokenizer: tp.Any | None = None,
        max_tool_calls_per_step: int = 5,
        test_timeout: float = 60.0,
        max_steps: int = 12,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str = "kiri",
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        affinity_key: str | int | None = None,
    ):
        cfg = container_config if container_config is not None else ContainerConfig(use_jail="required")
        super().__init__(
            runtime=runtime,
            backend=backend,
            endpoint=endpoint,
            container_config=cfg,
            max_steps=max_steps,
            affinity_key=affinity_key,
        )
        self._problem_statement = problem_statement
        self._initial_files = dict(files) if files is not None else None
        self._repo_dir = repo_dir
        self._base_commit = base_commit
        self._test_patch = test_patch
        self._fail_to_pass = list(fail_to_pass)
        self._pass_to_pass = list(pass_to_pass)
        self._parser = create_tool_call_parser(tool_caller, tokenizer=tokenizer)
        self._max_tool_calls_per_step = int(max_tool_calls_per_step)
        self._test_timeout = float(test_timeout)
        self._tools_map = {t.name: t for t in self.build_workspace_tools(enable_bash=True, enable_files=True)}
        self._reset_seed: int | None = None
        self._base_sha: str | None = None
        self._grading_snapshot: GradingSnapshot | None = None
        self._base_tracked_paths: set[str] = set()
        self._cached_repo_files: dict[str, bytes] | None = None

    _infra_failure_step = staticmethod(unscorable_step)

    def _all_test_commands(self) -> list[str]:
        return [*self._fail_to_pass, *self._pass_to_pass]

    def _seed_session_files(self, session: SandboxSession, *, capture_metadata: bool) -> str | None:
        if self._repo_dir is not None:
            if self._cached_repo_files is None:
                copied, seed_err = _seed_repo_dir_files(self._repo_dir, self._base_commit)
                if seed_err is not None or copied is None:
                    return seed_err or "Failed to seed repository from repo_dir."
                self._cached_repo_files = copied
            if self._cached_repo_files:
                session.write_files(self._cached_repo_files)

        git_check = session.exec(
            ["/bin/sh", "-c", 'test "$(git rev-parse --show-toplevel 2>/dev/null)" = "$(pwd -P)"'],
            env=_CLEAN_GIT_ENV,
            timeout=5.0,
        )
        if git_check.infra_error:
            return git_check.infra_error

        if git_check.ok and self._base_commit and self._repo_dir is None:
            # Container image already carries a git checkout: check out base_commit and strip all git history so
            # future fix commits, branches, tags, and reflogs cannot leak to the agent.
            co_res = session.exec(
                ["git", "checkout", "-f", self._base_commit],
                env=_CLEAN_GIT_ENV,
                timeout=15.0,
            )
            if co_res.infra_error:
                return co_res.infra_error
            if not co_res.ok:
                return f"Failed to checkout base_commit {self._base_commit!r}: {co_res.combined_output(1024)}"
            strip_res = session.exec(
                ["/bin/sh", "-c", "git clean -fdx >/dev/null 2>&1; rm -rf .git"],
                env=_CLEAN_GIT_ENV,
                timeout=15.0,
            )
            if strip_res.infra_error:
                return strip_res.infra_error
            if not strip_res.ok:
                return f"Failed to strip git history after checkout: {strip_res.combined_output(1024)}"
            git_check = ExecResult(ok=False, stdout="", stderr="", returncode=1)

        if self._initial_files:
            session.write_files(self._initial_files)

        if not git_check.ok:
            init_err = init_clean_git_repo(session, email="swe@easydel.local", name="EasyDeL SWE")
            if init_err is not None:
                return init_err

        if not capture_metadata:
            return None

        base_sha, infra_err = capture_base_revision(session)
        if infra_err:
            return infra_err
        if base_sha is None:
            return "Could not resolve the base revision of the seeded repository."
        entries, list_err = _list_base_tree(session, base_sha)
        if entries is None:
            return list_err or "Could not record the protected files of the base revision."
        all_paths = {path for _, _, path in entries}
        is_protected = _build_protected_predicate(self._test_patch, self._all_test_commands(), all_paths)
        snapshot: GradingSnapshot = {path: (mode, blob_id) for mode, blob_id, path in entries if is_protected(path)}
        self._base_sha = base_sha
        self._base_tracked_paths = all_paths
        self._grading_snapshot = snapshot
        return None

    def _seed_repository(self) -> str | None:
        return self._seed_session_files(self.session, capture_metadata=True)

    def reset(self, seed: int | None = None) -> ResetResult:
        self._reset_seed = seed
        self._base_sha = None
        self._grading_snapshot = None
        self._base_tracked_paths = set()
        self._cached_repo_files = None
        try:
            self._open_session(seed=seed)
            infra_err = self._seed_repository()
        except SandboxInfraError as exc:
            self.close()
            return ResetResult(
                observation=self._problem_statement,
                info={"env_error": True, "env_error_message": str(exc)},
            )
        if infra_err:
            self.close()
            return ResetResult(
                observation=self._problem_statement,
                info={"env_error": True, "env_error_message": infra_err},
            )
        return ResetResult(
            observation=self._problem_statement,
            info={"tools": list(self._tools_map.keys()), "env_error": False},
        )

    def _grade_repository(self) -> tuple[float, dict[str, tp.Any], str | None]:
        agent_session = self.session
        kill_err = kill_agent_processes(agent_session)
        if kill_err is not None:
            return 0.0, {}, kill_err

        if self._base_sha is None or self._grading_snapshot is None:
            return 0.0, {}, "Base revision was not recorded at reset; cannot grade safely."

        safe_patch, untracked_files, tamper_reason, extract_infra = extract_agent_workspace_changes(
            agent_session,
            self._base_sha,
            test_patch=self._test_patch,
            test_commands=self._all_test_commands(),
            base_tracked_paths=self._base_tracked_paths,
        )
        if extract_infra is not None:
            return 0.0, {}, extract_infra
        if tamper_reason is not None:
            return (
                0.0,
                {
                    "resolved": False,
                    "workspace_tampered": True,
                    "error": tamper_reason,
                    "patch": "",
                    "env_error": False,
                },
                None,
            )

        # Grade in a fresh, untouched session/container so container-level mutations, background processes,
        # workspace FIFOs/symlinks, runner shadowing, or test tampering can never influence verification.
        step_count = self._step_count
        grade_session = self._open_session(seed=self._reset_seed)
        self._step_count = step_count
        seed_err = self._seed_session_files(grade_session, capture_metadata=False)
        if seed_err is not None:
            return 0.0, {}, seed_err

        if safe_patch:
            apply_agent = apply_patch_via_stdin(grade_session, safe_patch, timeout=15.0)
            if apply_agent.infra_error:
                return 0.0, {}, apply_agent.infra_error
            if not apply_agent.ok:
                return (
                    0.0,
                    {
                        "resolved": False,
                        "patch": safe_patch,
                        "agent_patch_applied": False,
                        "error": apply_agent.stderr.strip(),
                        "env_error": False,
                    },
                    None,
                )

        if untracked_files:
            try:
                grade_session.write_files(untracked_files)
            except (OSError, ValueError) as exc:
                return (
                    0.0,
                    {
                        "resolved": False,
                        "workspace_tampered": True,
                        "patch": safe_patch,
                        "error": f"could not apply untracked agent files: {exc}",
                        "env_error": False,
                    },
                    None,
                )

        if self._test_patch:
            apply_res = apply_patch_via_stdin(grade_session, self._test_patch, timeout=15.0)
            if apply_res.infra_error:
                return 0.0, {}, apply_res.infra_error
            if not apply_res.ok:
                return (
                    0.0,
                    {
                        "resolved": False,
                        "patch": safe_patch,
                        "test_patch_applied": False,
                        "error": apply_res.stderr.strip(),
                        "env_error": False,
                    },
                    None,
                )

        f2p_results: list[bool] = []
        for cmd in self._fail_to_pass:
            ok, infra_err = run_verified_test_command(grade_session, cmd, timeout=self._test_timeout)
            if infra_err is not None:
                return 0.0, {}, infra_err
            f2p_results.append(ok)

        p2p_results: list[bool] = []
        for cmd in self._pass_to_pass:
            ok, infra_err = run_verified_test_command(grade_session, cmd, timeout=self._test_timeout)
            if infra_err is not None:
                return 0.0, {}, infra_err
            p2p_results.append(ok)

        resolved = verification_resolved(
            f2p_results,
            p2p_results,
            has_f2p=bool(self._fail_to_pass),
            has_p2p=bool(self._pass_to_pass),
        )
        info = {
            "resolved": resolved,
            "f2p_passed": sum(f2p_results),
            "f2p_total": len(f2p_results),
            "p2p_passed": sum(p2p_results),
            "p2p_total": len(p2p_results),
            "patch": safe_patch,
            "env_error": False,
        }
        return (1.0 if resolved else 0.0), info, None

    def step(self, action: str) -> StepResult:
        self._step_count += 1
        calls = self._parser(action)
        if calls and self._step_count < self._max_steps:
            outputs: list[str] = []
            for tool_name, raw_args in calls[: self._max_tool_calls_per_step]:
                tool = self._tools_map.get(tool_name)
                if tool is None:
                    outputs.append(f"[{tool_name}]: Error: unknown tool '{tool_name}'.")
                    continue
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError as exc:
                    outputs.append(f"[{tool_name}]: Error parsing arguments: {exc}")
                    continue
                try:
                    out = tool.call(**parsed_args) if isinstance(parsed_args, dict) else tool.call(parsed_args)
                except SandboxInfraError as exc:
                    return self._infra_failure_step(exc)
                except Exception as exc:
                    outputs.append(f"[{tool_name}]: Error executing tool: {exc}")
                    continue
                if out.startswith("Sandbox infrastructure error:"):
                    return self._infra_failure_step(out)
                outputs.append(f"[{tool_name}]: {out}")
            return StepResult(
                observation="\n\n".join(outputs),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"step": self._step_count, "tool_calls": len(calls), "env_error": False},
            )

        try:
            reward, info, infra_err = self._grade_repository()
        except SandboxInfraError as exc:
            return self._infra_failure_step(exc)

        if infra_err is not None:
            return self._infra_failure_step(infra_err)

        at_limit = self._step_count >= self._max_steps and bool(calls)
        obs = "Issue resolved; all verification tests passed." if reward == 1.0 else "Verification tests failed."
        return StepResult(
            observation=obs,
            reward=reward,
            terminated=not at_limit,
            truncated=at_limit,
            info=info,
        )


__all__ = ("SWERepoEnvironment",)
