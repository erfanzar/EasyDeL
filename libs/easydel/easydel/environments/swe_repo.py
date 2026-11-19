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
import json
import os
import posixpath
import typing as tp

from easydel.trainers._shared.environment import ResetResult, StepResult, create_tool_call_parser

from ._base import SandboxEnvironment, register_environment
from .runtime import ContainerConfig, ExecResult, KiriRuntime, SandboxInfraError, SandboxPool, SandboxSession

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


def _is_test_infra_path(path: str) -> bool:
    base = posixpath.basename(path)
    return base in _TEST_INFRA_BASENAMES or base.endswith(_TEST_INFRA_SUFFIXES)


def _patch_touched_paths(patch: str) -> set[str]:
    """Return every repository path a unified diff reads or writes (``/dev/null`` excluded)."""
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line[len("diff --git ") :].split(" b/", 1)
            if len(parts) == 2:
                paths.add(parts[0][2:] if parts[0].startswith("a/") else parts[0])
                paths.add(parts[1])
        elif line.startswith(("--- ", "+++ ")):
            target = line[4:].split("\t", 1)[0].strip()
            if target == "/dev/null":
                continue
            if target.startswith(("a/", "b/")):
                target = target[2:]
            paths.add(target)
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            paths.add(line.split(" ", 2)[2].strip())
    return {p for p in paths if p and not p.startswith("/") and ".." not in p.split("/")}


def capture_base_revision(session: SandboxSession) -> tuple[str | None, str | None]:
    """Return ``(base_sha, infra_error)`` for the session's current ``HEAD``."""
    res = session.exec(["git", "rev-parse", "HEAD"], timeout=10.0)
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
_BASE_TREE_FILE = ".git/.ed_base_tree"

GradingSnapshot = dict[str, tuple[str, str]]
"""``path -> (git mode, blob id)`` of every protected path that exists at the base revision."""


def run_clean_git(session: SandboxSession, args: list[str], *, timeout: float = 30.0) -> ExecResult:
    """Run ``git <args>`` in ``session`` without reading or executing anything the repository configures.

    The command runs against a fresh, empty git dir that only borrows the workspace's object store and index,
    with system/global config, replace refs, hooks and fsmonitor disabled, so agent-written ``.git/config``,
    ``.git/info/attributes`` filters or hooks never run.
    """
    return session.exec(["/bin/sh", "-c", _CLEAN_GIT_SCRIPT, "sh", *args], env=_CLEAN_GIT_ENV, timeout=timeout)


def capture_grading_snapshot(
    session: SandboxSession, base_sha: str, test_patch: str | None
) -> tuple[GradingSnapshot | None, str | None]:
    """Record the base-revision blob of every path grading protects. Call it at reset, before the agent acts.

    Args:
        session: Episode sandbox session whose working directory is a git checkout.
        base_sha: Commit recorded at reset.
        test_patch: Evaluation patch whose touched paths are protected too.

    Returns:
        ``(snapshot, infra_error)``; ``snapshot`` is ``None`` when the listing failed.
    """
    list_script = _CLEAN_GIT_PRELUDE + 'git ls-tree -r -z --full-tree "$1" > "$w/$2"'
    res = session.exec(
        ["/bin/sh", "-c", list_script, "sh", base_sha, _BASE_TREE_FILE],
        env=_CLEAN_GIT_ENV,
        timeout=60.0,
    )
    if res.infra_error or not res.ok:
        return None, res.infra_error or f"git ls-tree failed: {res.combined_output(1024)}"
    try:
        listing = session.read_file(_BASE_TREE_FILE)
    finally:
        session.exec(["rm", "-f", "--", _BASE_TREE_FILE], timeout=10.0)
    patch_paths = _patch_touched_paths(test_patch or "")
    snapshot: GradingSnapshot = {}
    for entry in listing.split("\0"):
        meta, sep, path = entry.partition("\t")
        fields = meta.split()
        if not sep or len(fields) != 3 or fields[1] != "blob":
            continue
        if _is_test_infra_path(path) or path in patch_paths:
            snapshot[path] = (fields[0], fields[2])
    return snapshot, None


def restore_grading_files(
    session: SandboxSession,
    base_sha: str,
    test_patch: str | None,
    snapshot: GradingSnapshot | None = None,
) -> str | None:
    """Reset test-patch paths and test-infrastructure files to their base content before grading.

    Every live test-infrastructure file and every test-patch path is deleted, then each one that exists at the
    base revision is rewritten from its recorded blob (undoing edits, deletions and symlink swaps). Agent-planted
    ``conftest.py`` / ``sitecustomize.py`` / ``*.pth`` files or pre-created copies of test-patch files therefore
    cannot influence verification. No git command here reads the repository's own config, attributes or hooks.

    Args:
        session: Episode sandbox session whose working directory is a git checkout.
        base_sha: Commit recorded at reset, before the agent acted.
        test_patch: Evaluation patch that will be applied afterwards (its paths are reset too).
        snapshot: Result of :func:`capture_grading_snapshot` taken at reset. When omitted the base tree is
            listed now, which trusts an object store the agent could have written to.

    Returns:
        An error message when the restore could not run (treated as unscorable), else ``None``.
    """
    if snapshot is None:
        snapshot, err = capture_grading_snapshot(session, base_sha, test_patch)
        if snapshot is None:
            return err or "could not list the base revision"
    name_tests: list[str] = []
    for name in sorted(_TEST_INFRA_BASENAMES) + [f"*{s}" for s in _TEST_INFRA_SUFFIXES]:
        name_tests += [*(["-o"] if name_tests else []), "-name", name]
    find_argv = ["find", ".", "-path", "./.git", "-prune", "-o", "(", *name_tests, ")", "-prune"]
    find_res = session.exec([*find_argv, "-exec", "rm", "-rf", "--", "{}", "+"], timeout=60.0)
    if find_res.infra_error or not find_res.ok:
        return find_res.infra_error or f"failed to remove test-infrastructure files: {find_res.combined_output(1024)}"
    to_delete = sorted(p for p in _patch_touched_paths(test_patch or "") if p not in snapshot)
    triples = [item for path in sorted(snapshot) for item in (*snapshot[path], path)]
    res = session.exec(
        ["/bin/sh", "-c", _RESTORE_SCRIPT, "sh", *to_delete, "--", *triples], env=_CLEAN_GIT_ENV, timeout=120.0
    )
    if res.infra_error or not res.ok:
        return res.infra_error or f"failed to restore test files: {res.combined_output(1024)}"
    return None


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
    tool calls, and grades final submissions by applying ``test_patch`` and
    checking that all ``fail_to_pass`` commands pass with zero ``pass_to_pass``
    regressions.
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
        self._base_sha: str | None = None
        self._grading_snapshot: GradingSnapshot | None = None

    _infra_failure_step = staticmethod(unscorable_step)

    def _seed_repository(self) -> str | None:
        session = self.session
        if self._repo_dir is not None:
            copied: dict[str, bytes] = {}
            root = os.path.abspath(self._repo_dir)
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in {".pytest_cache", "__pycache__"}]
                for fname in filenames:
                    full = os.path.join(dirpath, fname)
                    if os.path.isfile(full):
                        rel = os.path.relpath(full, root)
                        with open(full, "rb") as f:
                            copied[rel] = f.read()
            if copied:
                session.write_files(copied)

        if self._initial_files:
            session.write_files(self._initial_files)

        git_check = session.run_bash(
            'test "$(git rev-parse --show-toplevel 2>/dev/null)" = "$(pwd -P)"',
            timeout=5.0,
        )
        if git_check.infra_error:
            return git_check.infra_error
        if not git_check.ok:
            init_res = session.run_bash(
                "git init -q && git config user.email 'swe@easydel.local' && "
                "git config user.name 'EasyDeL SWE' && git add -A && "
                "git commit -q --allow-empty -m 'base'",
                timeout=10.0,
            )
            if init_res.infra_error:
                return init_res.infra_error
        elif self._base_commit:
            co_res = session.exec(["git", "checkout", "-f", self._base_commit], timeout=15.0)
            if co_res.infra_error:
                return co_res.infra_error
            if not co_res.ok:
                return f"Failed to checkout base_commit {self._base_commit!r}: {co_res.combined_output(1024)}"
        base_sha, infra_err = capture_base_revision(session)
        if infra_err:
            return infra_err
        if base_sha is None:
            return "Could not resolve the base revision of the seeded repository."
        snapshot, snap_err = capture_grading_snapshot(session, base_sha, self._test_patch)
        if snapshot is None:
            return snap_err or "Could not record the protected files of the base revision."
        self._base_sha = base_sha
        self._grading_snapshot = snapshot
        return None

    def reset(self, seed: int | None = None) -> ResetResult:
        self._base_sha = None
        self._grading_snapshot = None
        try:
            self._open_session(seed=seed)
            infra_err = self._seed_repository()
        except SandboxInfraError as exc:
            return ResetResult(
                observation=self._problem_statement,
                info={"env_error": True, "env_error_message": str(exc)},
            )
        if infra_err:
            return ResetResult(
                observation=self._problem_statement,
                info={"env_error": True, "env_error_message": infra_err},
            )
        return ResetResult(
            observation=self._problem_statement,
            info={"tools": list(self._tools_map.keys()), "env_error": False},
        )

    def _grade_repository(self) -> tuple[float, dict[str, tp.Any], str | None]:
        session = self.session
        kill_err = kill_agent_processes(session)
        if kill_err is not None:
            return 0.0, {}, kill_err
        diff_res = run_clean_git(session, ["diff"], timeout=10.0)
        if diff_res.infra_error:
            return 0.0, {}, diff_res.infra_error
        model_patch = diff_res.stdout

        if self._base_sha is None or self._grading_snapshot is None:
            return 0.0, {}, "Base revision was not recorded at reset; cannot grade safely."
        restore_err = restore_grading_files(session, self._base_sha, self._test_patch, self._grading_snapshot)
        if restore_err is not None:
            return 0.0, {}, restore_err

        if self._test_patch:
            session.write_files({".ed_eval_tests.patch": self._test_patch})
            apply_res = run_clean_git(session, ["apply", ".ed_eval_tests.patch"], timeout=10.0)
            if apply_res.infra_error:
                return 0.0, {}, apply_res.infra_error
            if not apply_res.ok:
                return (
                    0.0,
                    {"patch": model_patch, "test_patch_applied": False, "error": apply_res.stderr.strip()},
                    None,
                )

        # Second kill: anything that slipped in between the first kill and now must not race the test runs.
        kill_err = kill_agent_processes(session)
        if kill_err is not None:
            return 0.0, {}, kill_err

        f2p_results: list[bool] = []
        for cmd in self._fail_to_pass:
            res = session.run_bash(cmd, timeout=self._test_timeout)
            if res.infra_error:
                return 0.0, {}, res.infra_error
            f2p_results.append(res.ok)

        p2p_results: list[bool] = []
        for cmd in self._pass_to_pass:
            res = session.run_bash(cmd, timeout=self._test_timeout)
            if res.infra_error:
                return 0.0, {}, res.infra_error
            p2p_results.append(res.ok)

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
            "patch": model_patch,
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
