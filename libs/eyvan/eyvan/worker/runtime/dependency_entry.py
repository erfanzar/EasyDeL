# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Install a job's declared packages, then exec the job.

The half of a per-job runtime environment that Eyvan did not have. A workload
could already ship its working tree and its environment variables; its
*dependencies* could only be whatever the host image happened to carry, so
adding one package meant rebuilding an image and rolling a fleet -- which is how
"quick experiment" turns into an afternoon. Ray calls this ``runtime_env``, and
it is the reason people reach for Ray on a cluster they otherwise do not need.

Three decisions are load-bearing:

**The environment sees the host's site-packages.** A clean virtualenv on a TPU
host has no ``jax`` and no ``libtpu``, so every accelerator job would install a
second, CPU-only JAX over the top of a working one and fail on the first device
call. The job's packages are layered *over* what the host provides, never
instead of it.

**One environment per requirement set and interpreter, per host.** Eight ranks on
one host declare the same packages; installing eight times is eight times the work
and a race over the same files. The set is content-addressed, and the marker
file is written only after a successful install, so a crashed attempt leaves a
directory that is rebuilt rather than a half-populated one that is trusted.

**A failed install fails the job.** Running the command anyway produces an
``ImportError`` several minutes later, on an accelerator, attributed to the
user's code rather than to the install that did not happen.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import site
import subprocess
import sys
import typing as tp
from pathlib import Path

#: Where environments are cached when the runtime does not say. A host path, so
#: a second attempt or a second rank on the same machine reuses the first one's
#: work; the per-attempt working directory would defeat exactly that.
DEFAULT_CACHE_ROOT = Path("/var/lib/eyvan/runtime-envs")

#: Host-wide shared cache root for cross-environment assets (uv wheel cache,
#: bare git mirrors, layered base virtualenvs, pre-built Rust wheels).
DEFAULT_SHARED_CACHE_ROOT = DEFAULT_CACHE_ROOT.parent

#: Where environments go when the host path is not writable -- a laptop, or any
#: cell whose agent does not run as root. Named rather than inlined because it
#: appears in a working tree somebody is version-controlling, and the .gitignore
#: that keeps it out of their commits has to spell the same thing.
FALLBACK_CACHE_DIR = ".eyvan-runtime-envs"

#: Written only after an install succeeds. Its absence is what makes a partial
#: environment rebuild instead of being trusted.
COMPLETE_MARKER = ".eyvan-complete"


def environment_key(requirements: tp.Sequence[str], setup_identity: str | None = None) -> str:
    """Return a versioned identity for requirements and their parent interpreter.

    Sorted and de-duplicated first, because the order somebody typed them in is
    not part of what they asked for -- and two identical requests that hash
    differently mean two environments and two installs. A setup step joins the
    identity only when present, so every environment built without one keeps
    the key it always had.
    """

    identity: dict[str, tp.Any] = {
        "format": 3,
        "interpreter": os.path.realpath(sys.executable),
        "version": list(sys.version_info),
        "parent_sites": _parent_site_packages(),
        "requirements": sorted({item.strip() for item in requirements if item.strip()}),
    }
    if setup_identity is not None:
        identity["setup"] = setup_identity
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]


#: Largest setup script fetched from a URL. A setup script installs things; one
#: larger than this is almost certainly not the file somebody meant to point at.
MAX_FETCHED_SETUP_BYTES = 1024 * 1024
_SETUP_FETCH_TIMEOUT_S = 60.0
#: Where the job interpreter is named for the setup script.
SETUP_PYTHON_ENV = "EYVAN_PYTHON"


class SetupStep(tp.NamedTuple):
    """A resolved setup step: the exact script bytes and the arguments it runs with."""

    script: str
    args: tuple[str, ...]

    @property
    def identity(self) -> str:
        """Content address of the step: what decides whether a cached run is reusable."""

        body = json.dumps({"script": self.script, "args": list(self.args)}, sort_keys=True)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _fetch_setup_script(url: str) -> str:
    """Download a setup script over https, refusing a redirect that leaves https."""

    import urllib.request

    if not url.startswith("https://"):
        raise RuntimeError(f"setup URL {url!r} must be https://")
    try:
        with urllib.request.urlopen(url, timeout=_SETUP_FETCH_TIMEOUT_S) as response:
            if not str(response.geturl()).startswith("https://"):
                raise RuntimeError(f"setup URL {url!r} redirected away from https")
            body = response.read(MAX_FETCHED_SETUP_BYTES + 1)
    except OSError as error:
        raise RuntimeError(f"could not fetch setup script {url}: {error}") from error
    if len(body) > MAX_FETCHED_SETUP_BYTES:
        raise RuntimeError(f"setup script {url} is larger than {MAX_FETCHED_SETUP_BYTES} bytes")
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"setup script {url} is not UTF-8 text") from error


def resolve_setup(document: tp.Mapping[str, tp.Any]) -> SetupStep:
    """Turn the wire form of a setup step into the exact script that will run.

    A fetched script is pinned by ``sha256`` when one was given; without one,
    the bytes fetched now are still what the cache is keyed by, so a script
    that changed behind its URL runs again rather than being trusted stale.
    """

    args = tuple(str(item) for item in document.get("args") or ())
    script = document.get("script")
    url = document.get("url")
    if (script is None) == (url is None):
        raise RuntimeError("a setup step needs exactly one of an inline script or a URL")
    if url is not None:
        script = _fetch_setup_script(str(url))
        pinned = document.get("sha256")
        if pinned is not None and hashlib.sha256(script.encode("utf-8")).hexdigest() != pinned:
            raise RuntimeError(f"setup script {url} does not match its pinned sha256")
    return SetupStep(script=str(script), args=args)


def _run_setup(environment: Path, setup: SetupStep, *, env_root: Path | None = None) -> None:
    """Run the setup step inside ``environment``, streaming its output into the job's log."""

    script_path = environment / ".eyvan-setup.sh"
    script_path.write_text(setup.script, encoding="utf-8")
    shell = shutil.which("bash") or "/bin/sh"
    child_environment = activate(environment)
    child_environment[SETUP_PYTHON_ENV] = str(environment / "bin" / "python")
    resolved_root = env_root if env_root is not None else environment.parent
    child_environment.setdefault("EYVAN_RUNTIME_ENV_DIR", str(resolved_root))
    uv_cache = uv_cache_dir(resolved_root)
    if uv_cache is not None:
        child_environment.setdefault("UV_CACHE_DIR", str(uv_cache))
    # The multi-host join belongs to the job. With the marker present, any
    # Python the step starts (``$EYVAN_PYTHON -m pip`` is enough) would load the
    # multi-host ``sitecustomize`` and try to join the gang from here -- the
    # same reason the wrapper itself runs with ``-I``. The job still gets it.
    child_environment.pop("EYVAN_JAX_MULTIHOST", None)
    print(f"eyvan: running setup step ({setup.identity[:12]})", file=sys.stderr, flush=True)
    # Not captured: the step's output is the job's output, so a failing install
    # is readable in `eyvan workload logs` instead of being summarized away.
    completed = subprocess.run(
        [shell, str(script_path), *setup.args], env=child_environment, cwd=str(environment), check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"the setup step exited with status {completed.returncode}")


def _user_cache_root() -> Path | None:
    """Return the user-scoped runtime-env cache when ``HOME`` is explicitly set."""

    home = os.environ.get("HOME")
    if not home:
        return None
    return Path(home).expanduser() / ".eyvan" / "runtime-envs"


def _probe_writable_directory(path: Path) -> None:
    """Ensure ``path`` exists and is writable by the current process."""

    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".writable"
    probe.touch()
    probe.unlink()


def shared_cache_root(env_root: Path | None = None) -> Path | None:
    """Return a writable persistent directory shared across per-job environments.

    When ``env_root`` is named ``runtime-envs`` (e.g. ``/var/lib/eyvan/runtime-envs``
    or ``~/.eyvan/runtime-envs``), its parent hosts sibling shared caches
    (``uv-cache``, ``git-mirrors``, ``base-envs``, ``wheel-cache``). Otherwise a
    writable ``.shared-cache`` subdirectory under ``env_root`` is used before
    falling back to ``/var/lib/eyvan`` and ``~/.eyvan``.
    """

    candidates: list[Path] = []
    if env_root is not None:
        if env_root.name == "runtime-envs":
            candidates.append(env_root.parent)
        candidates.append(env_root / ".shared-cache")
    candidates.append(DEFAULT_SHARED_CACHE_ROOT)
    user_root = _user_cache_root()
    if user_root is not None:
        candidates.append(user_root.parent)
    for candidate in candidates:
        try:
            _probe_writable_directory(candidate)
            return candidate
        except OSError:
            continue
    return None


def uv_cache_dir(env_root: Path | None = None) -> Path | None:
    """Return a writable shared ``UV_CACHE_DIR`` so wheels download once per host."""

    configured = os.environ.get("UV_CACHE_DIR")
    if configured:
        return Path(configured)
    shared = shared_cache_root(env_root)
    if shared is None:
        return None
    target = shared / "uv-cache"
    try:
        _probe_writable_directory(target)
    except OSError:
        return None
    return target


def _installer_environment(env_root: Path) -> dict[str, str]:
    """Return process environment for package installation with shared ``UV_CACHE_DIR``."""

    env = dict(os.environ)
    uv_cache = uv_cache_dir(env_root)
    if uv_cache is not None:
        env.setdefault("UV_CACHE_DIR", str(uv_cache))
    return env


def cache_root() -> Path:
    """Return a writable directory for cached environments.

    Prefers ``EYVAN_RUNTIME_ENV_DIR``, then the host-wide ``DEFAULT_CACHE_ROOT``,
    then ``~/.eyvan/runtime-envs`` when ``HOME`` is set (so local runs reuse
    cached environments without polluting a git working tree), and falls back to
    the working directory only when none of those locations is writable.
    """

    configured = os.environ.get("EYVAN_RUNTIME_ENV_DIR")
    if configured:
        return Path(configured)
    try:
        _probe_writable_directory(DEFAULT_CACHE_ROOT)
        return DEFAULT_CACHE_ROOT
    except OSError as error:
        user_root = _user_cache_root()
        if user_root is not None:
            try:
                _probe_writable_directory(user_root)
                print(
                    f"eyvan: {DEFAULT_CACHE_ROOT} is not writable ({error}); caching this job's packages "
                    f"in {user_root} instead",
                    file=sys.stderr,
                )
                return user_root
            except OSError:
                pass
        print(
            f"eyvan: {DEFAULT_CACHE_ROOT} is not writable ({error}); caching this job's packages "
            f"in ./{FALLBACK_CACHE_DIR} instead, so only a later run from this same "
            "directory reuses them",
            file=sys.stderr,
        )
        return Path.cwd() / FALLBACK_CACHE_DIR


#: How long a found ``uv`` gets to prove it runs before the installer falls back.
_UV_PROBE_TIMEOUT_S = 15.0


def _working_uv() -> str | None:
    """Return a ``uv`` on PATH that actually runs here, or ``None``.

    Found is not the same as runnable. A host whose only ``uv`` is a snap
    (``/snap/bin/uv``) resolves it on PATH, but snap-confine refuses to start
    inside the workload's sandboxed systemd unit ("snap-confine is packaged
    without necessary permissions"). One ``--version`` call decides whether it
    is usable; any failure falls back to
    the interpreter's own ``venv`` and ``pip``, which need nothing from PATH.
    """

    uv = shutil.which("uv")
    if not uv:
        return None
    try:
        probe = subprocess.run(
            [uv, "--version"], capture_output=True, text=True, check=False, timeout=_UV_PROBE_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return uv if probe.returncode == 0 else None


def _install_commands(environment: Path, requirements: tp.Sequence[str]) -> list[list[str]]:
    """Return the commands that build ``environment``, preferring a uv that runs."""

    uv = _working_uv()
    if uv:
        return [
            # `--system-site-packages` is not an optimization. Without it the
            # host's accelerator stack -- jax, libtpu, the vendor's drivers --
            # is invisible, and the job installs a second CPU-only copy over a
            # working one. `--python sys.executable` pins the interpreter that
            # will unpickle the callable; left to uv, it may choose another.
            [uv, "venv", "--python", sys.executable, "--system-site-packages", str(environment)],
            # A setup-only job declares no packages, and an empty install is an error.
            *(
                [[uv, "pip", "install", "--python", str(environment / "bin" / "python"), *requirements]]
                if requirements
                else []
            ),
        ]
    return [
        [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
        *([[str(environment / "bin" / "python"), "-m", "pip", "install", *requirements]] if requirements else []),
    ]


#: Written into the job environment's site-packages. Named so an operator
#: reading the directory can tell where the extra ``sys.path`` entries came from.
PARENT_PATH_FILE = "_eyvan_parent_environment.pth"


def _parent_site_packages() -> list[str]:
    """Return the ordered host directories inherited by a job environment."""

    user_site = site.getusersitepackages() if site.ENABLE_USER_SITE and site.getusersitepackages() in sys.path else ""
    return [path for path in dict.fromkeys((*site.getsitepackages(), user_site)) if path and Path(path).is_dir()]


def _inherit_running_environment(environment: Path) -> None:
    """Make the running interpreter's packages visible inside ``environment``.

    ``--system-site-packages`` exposes the *base* interpreter's site-packages.
    On a native cell the worker runs from ``/opt/eyvan-native/venv`` -- itself a
    virtual environment -- so the base paths alone do not expose the worker's
    Eyvan, JAX, and libtpu packages to jobs declaring dependencies.

    A ``.pth`` file appends the running environment's site directories after
    the job environment's own, so a package the job pins still wins and
    everything else resolves to what the host already provides.
    """

    # Only what the running interpreter itself imports from: a virtualenv
    # disables the user site, and adding it here would hand the job packages the
    # worker never saw.
    parents = [
        path for path in _parent_site_packages() if not Path(path).resolve().is_relative_to(environment.resolve())
    ]
    if not parents:
        return
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    purelib = environment / "lib" / version / "site-packages"
    purelib.mkdir(parents=True, exist_ok=True)
    # Use site.addsitedir() rather than bare path lines: CPython's
    # site.addpackage() appends bare paths to sys.path without processing .pth
    # files inside them, which breaks PEP 660 editable installs (e.g. uv sync).
    # Because the job venv's own site-packages is already on sys.path before
    # this .pth runs, packages pinned by the job still take precedence.
    (purelib / PARENT_PATH_FILE).write_text(
        "".join(f"import site; site.addsitedir({path!r})\n" for path in parents),
        encoding="utf-8",
    )


def prepare(requirements: tp.Sequence[str], *, root: Path | None = None, setup: SetupStep | None = None) -> Path:
    """Return an environment holding ``requirements``, building it if needed.

    Raises:
        RuntimeError: If the environment could not be built. The caller must
            fail the job rather than run it: a command that starts without the
            packages it declared fails minutes later, on an accelerator, and
            blames the user's code.
    """

    base = root if root is not None else cache_root()
    base.mkdir(parents=True, exist_ok=True)
    environment = base / environment_key(requirements, setup.identity if setup is not None else None)
    marker = environment / COMPLETE_MARKER
    if marker.exists():
        return environment

    # Held across the whole build so ranks sharing a host queue rather than
    # write over each other. The marker is re-checked after the lock is taken:
    # by then the rank that was building may have finished.
    lock_path = base / f"{environment.name}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return environment
        if environment.exists():
            # An earlier attempt died mid-install. Its directory is unusable and
            # indistinguishable from a good one except by the missing marker.
            shutil.rmtree(environment, ignore_errors=True)
        install_env = _installer_environment(base)
        for index, command in enumerate(_install_commands(environment, requirements)):
            completed = subprocess.run(command, env=install_env, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                shutil.rmtree(environment, ignore_errors=True)
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"installing {', '.join(requirements)} failed:\n{detail}")
            if index == 0:
                # Right after the environment exists and before anything is
                # installed into it, so the job's own pins are resolved against
                # -- and layered over -- what the running interpreter provides.
                _inherit_running_environment(environment)
        if setup is not None:
            # After the packages, so the step can build on them; before the
            # marker, so a step that failed or was killed is rerun, not trusted.
            try:
                _run_setup(environment, setup, env_root=base)
            except BaseException:
                shutil.rmtree(environment, ignore_errors=True)
                raise
        marker.write_text("\n".join(requirements), encoding="utf-8")
    return environment


def activate(environment: Path, base: tp.Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment variables that put ``environment`` first.

    Prepended rather than replacing ``PATH``: the job still needs the host's
    tooling, and an accelerator host keeps vendor binaries outside any virtual
    environment.
    """

    source = dict(os.environ if base is None else base)
    binaries = str(environment / "bin")
    source["VIRTUAL_ENV"] = str(environment)
    source["PATH"] = f"{binaries}{os.pathsep}{source.get('PATH', '')}".rstrip(os.pathsep)
    return source


def main(argv: tp.Sequence[str] | None = None) -> int:
    """Install the declared packages, then exec the command that follows."""

    parser = argparse.ArgumentParser(prog="eyvan-dependency-entry")
    parser.add_argument("--require", action="append", default=[], metavar="REQUIREMENT")
    parser.add_argument("--setup", default=None, metavar="JSON")
    parser.add_argument("--setup-file", default=None, metavar="PATH")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    command = list(arguments.command)
    # argparse hands the wrapper its own end-of-options marker back inside the
    # remainder; consume exactly that one. Any later `--` belongs to the
    # workload's command line -- `python tool.py -- --fast` means something --
    # and must be exec'd verbatim.
    if command and command[0] == "--":
        del command[0]
    if not command:
        parser.error("a command to execute is required")
    if arguments.setup is not None and arguments.setup_file is not None:
        parser.error("pass at most one of --setup and --setup-file")
    if not arguments.require and arguments.setup is None and arguments.setup_file is None:
        parser.error("at least one --require or a --setup is needed; without one this wrapper does nothing")

    try:
        setup = None
        raw_setup: str | None = None
        setup_source = "--setup"
        if arguments.setup is not None:
            raw_setup = arguments.setup
        elif arguments.setup_file is not None:
            setup_source = "--setup-file"
            try:
                raw_setup = Path(arguments.setup_file).read_text(encoding="utf-8")
            except OSError as error:
                raise RuntimeError(f"could not read --setup-file {arguments.setup_file!r}: {error}") from error
        if raw_setup is not None:
            try:
                document = json.loads(raw_setup)
            except ValueError as error:
                raise RuntimeError(f"{setup_source} is not valid JSON: {error}") from error
            if not isinstance(document, dict):
                raise RuntimeError(f"{setup_source} must be a JSON object")
            setup = resolve_setup(document)
        environment = prepare(arguments.require, setup=setup)
    except RuntimeError as error:
        # Non-zero before the job starts. The alternative is an ImportError
        # minutes later on an accelerator, blamed on the user's code.
        print(f"eyvan: {error}", file=sys.stderr)
        return 1

    os.environ.update(activate(environment))
    # `exec` rather than spawn, for the reason the workspace entry does it: the
    # workload must keep the PID its supervisor started, or signals, exit status
    # and process introspection describe a wrapper instead of the job.
    os.execvp(command[0], command)
    raise AssertionError("execvp returned")  # pragma: no cover - unreachable


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())
