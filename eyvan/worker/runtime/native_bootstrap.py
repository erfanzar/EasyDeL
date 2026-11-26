# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bootstrap rendering for a native Eyvan worker on a Compute Engine VM."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import shlex
import typing as tp
from pathlib import PurePosixPath

from eyvan._internal.json_types import JsonObject, model_json
from eyvan.records import NativeContainerConfig, NativeRuntimeConfig
from eyvan.worker.runtime.dependency_entry import DEFAULT_CACHE_ROOT as DEPENDENCY_CACHE_ROOT

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORKLOAD_RUNTIME_ENVIRONMENT = "/etc/eyvan-native/workload-runtime.json"
_WORKLOAD_CONTAINER_RUNTIME = "/etc/eyvan-native/container-runtime.json"
#: The major.minor head of a declared interpreter, which uv may spell as a full
#: version ("3.11.9"). The venv guard asks the running interpreter for its
#: major.minor, so the declaration is reduced to the same two components
#: before the two are compared.
_DECLARED_PYTHON_MAJOR_MINOR = re.compile(r"\d+\.\d+")

#: Revision of the rendered startup script's *behavior*. The driver's bootstrap
#: identity pins the byte sources it stages (wheel, wheelhouse); the script
#: itself was not pinned, so a script fix with identical artifacts never
#: reached existing cells -- the same staleness the key exists to prevent, one
#: level up. Bump on any bootstrap behavior change existing cells must receive.
#:
#: Known limit, stated rather than implied: a queued resource renders its
#: startup script once and never re-reads it. A revision bump re-renders the
#: *record* and is stamped into every cell created afterwards, but a VM that
#: already exists keeps the script it booted with -- a bootstrap fix reaches
#: running cells only when they are recreated. This is deliberate: pushing a
#: new script under a running cell would re-run provisioning steps with no
#: fence, and the driver has no way to know what the old script already did.
NATIVE_WORKER_BOOTSTRAP_REVISION = "2026-09-26.1"


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"native worker {name} must be a non-empty string")
    if "\x00" in normalized:
        raise ValueError(f"native worker {name} must not contain NUL bytes")
    return normalized


def _runtime_environment(value: tp.Mapping[str, str] | None) -> dict[str, str]:
    """Validate the persistent workload environment passed to the worker."""

    if value is None:
        return {}
    if not isinstance(value, tp.Mapping):
        raise ValueError("native worker runtime environment must be a mapping")
    environment: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError("native worker runtime environment names must be valid shell identifiers")
        if not isinstance(item, str):
            raise ValueError(f"native worker runtime environment variable {name!r} must have a string value")
        if "\x00" in item:
            raise ValueError(f"native worker runtime environment variable {name!r} must not contain NUL bytes")
        environment[name] = item
    return environment


def _container_artifact_mounts(value: tp.Sequence[str] | None) -> tuple[str, ...]:
    """Validate staged host paths automatically exposed to an OCI workload."""

    if value is None:
        return ()
    mounts: list[str] = []
    for item in value:
        if not isinstance(item, str) or item != item.strip() or "\x00" in item:
            raise ValueError("native worker container artifact mounts must be trimmed NUL-free paths")
        path = PurePosixPath(item)
        if not path.is_absolute() or ".." in path.parts or str(path) == "/" or str(path).startswith("//"):
            raise ValueError("native worker container artifact mounts must be absolute non-root paths without '..'")
        mounts.append(str(path))
    if len(mounts) != len(set(mounts)):
        raise ValueError("native worker container artifact mounts must be unique")
    return tuple(mounts)


def _container_runtime_payload(
    container: NativeContainerConfig | None,
    artifact_mounts: tp.Sequence[str] | None,
) -> JsonObject | None:
    """Return the JSON handoff consumed by the native worker's OCI executor."""

    mounts = _container_artifact_mounts(artifact_mounts)
    if container is None:
        if mounts:
            raise ValueError("native worker container artifact mounts require a container runtime")
        return None
    if not isinstance(container, NativeContainerConfig):
        raise ValueError("native worker container runtime must be a NativeContainerConfig")
    return {"artifact_mounts": list(mounts), "container": model_json(container)}


def render_native_runtime_setup(runtime: NativeRuntimeConfig) -> str | None:
    """Render trusted fleet setup after the Eyvan worker venv is ready.

    Artifacts are staged before commands so a command can install a wheel or
    invoke a repository script. This is capacity bootstrap configuration, not
    workload input, and therefore runs on every host in a native TPU slice.
    """

    if not isinstance(runtime, NativeRuntimeConfig):
        raise ValueError("native runtime setup requires a NativeRuntimeConfig")
    if not runtime.artifacts and not runtime.init_commands:
        return None
    lines = ["# Eyvan declarative native runtime setup."]
    for name, value in sorted(runtime.env.items()):
        lines.append(f"export {name}={shlex.quote(value)}")
    for artifact in runtime.artifacts:
        source = shlex.quote(artifact.source)
        destination = shlex.quote(artifact.destination)
        if artifact.recursive:
            lines.extend(
                (
                    f"install -d -m 0755 {destination}",
                    f"gcloud storage cp --recursive {source} {destination}",
                )
            )
        else:
            parent = shlex.quote(str(PurePosixPath(artifact.destination).parent))
            lines.extend(
                (
                    f"install -d -m 0755 {parent}",
                    f"gcloud storage cp {source} {destination}",
                )
            )
    lines.extend(command.rstrip() for command in runtime.init_commands)
    return "\n".join(lines)


def _sha256(value: str | None, name: str, *, required: bool) -> str | None:
    """Validate a lowercase SHA-256 used by a bootstrap integrity check."""

    if value is None:
        if required:
            raise ValueError(f"native worker {name} SHA-256 is required")
        return None
    normalized = _required(value, f"{name} SHA-256")
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"native worker {name} SHA-256 must be 64 lowercase hexadecimal characters")
    return normalized


def _content_address_from_uri(uri: str) -> str | None:
    """Infer the digest from ``.../<sha256>/<filename>`` staging layouts."""

    segments = uri.split("#", 1)[0].rstrip("/").split("/")
    if len(segments) < 2:
        return None
    candidate = segments[-2]
    return candidate if re.fullmatch(r"[0-9a-f]{64}", candidate) else None


def _gcs_source(uri: str, generation: str | None, name: str) -> str:
    """Return a generation-qualified GCS URI when the operator supplied one."""

    if generation is None:
        return uri
    normalized = _required(generation, f"{name} GCS generation")
    if not uri.startswith("gs://"):
        raise ValueError(f"native worker {name} generation is supported only for gs:// artifacts")
    if "#" in uri:
        raise ValueError(f"native worker {name} URI must not include a generation fragment twice")
    if re.fullmatch(r"[1-9][0-9]*", normalized) is None:
        raise ValueError(f"native worker {name} GCS generation must be a positive decimal integer")
    return f"{uri}#{normalized}"


def _verify_sha256(path: str, digest: str, name: str) -> tuple[str, ...]:
    """Render a fail-closed SHA-256 check before an artifact is consumed."""

    record = shlex.quote(f"{digest}  {path}")
    return (
        "command -v sha256sum >/dev/null 2>&1 || "
        f"{{ echo 'Eyvan {name} verification requires sha256sum' >&2; exit 1; }}",
        f"printf '%s\n' {record} | sha256sum --check --strict -",
    )


def _install_command(
    package: str,
    *,
    python: str,
    package_sha256: str | None = None,
    package_generation: str | None = None,
    wheelhouse: str | None = None,
    wheelhouse_sha256: str | None = None,
    wheelhouse_generation: str | None = None,
) -> str:
    """Return an install command with verified wheel and wheelhouse content."""

    quoted = shlex.quote(package)
    wheelhouse_lines: tuple[str, ...] = ()
    pip_flags = "--no-cache-dir --upgrade"
    if wheelhouse is not None:
        expected_wheelhouse = _sha256(wheelhouse_sha256, "wheelhouse", required=True)
        assert expected_wheelhouse is not None
        wheelhouse_source = _gcs_source(wheelhouse, wheelhouse_generation, "wheelhouse")
        wheelhouse_path = "/var/lib/eyvan/bootstrap/wheelhouse.zip"
        wheelhouse_lines = (
            f"gcloud storage cp {shlex.quote(wheelhouse_source)} {wheelhouse_path}",
            *_verify_sha256(wheelhouse_path, expected_wheelhouse, "wheelhouse"),
            "rm -rf /var/lib/eyvan/bootstrap/wheels",
            "mkdir -p /var/lib/eyvan/bootstrap/wheels",
            f"{python} -m zipfile -e {wheelhouse_path} /var/lib/eyvan/bootstrap/wheels",
        )
        pip_flags = "--no-cache-dir --no-index --find-links /var/lib/eyvan/bootstrap/wheels --upgrade"
    elif wheelhouse_sha256 is not None or wheelhouse_generation is not None:
        raise ValueError("native worker wheelhouse digest/generation requires a wheelhouse URI")
    if package.startswith("gs://"):
        wheel_name = package.split("#", 1)[0].rsplit("/", 1)[-1]
        if not wheel_name.endswith(".whl") or wheel_name == ".whl":
            raise ValueError("native worker GCS package must name a .whl file")
        expected_package = _sha256(
            package_sha256 or _content_address_from_uri(package),
            "package",
            required=True,
        )
        assert expected_package is not None
        package_source = _gcs_source(package, package_generation, "package")
        wheel_path = f"/var/lib/eyvan/bootstrap/{wheel_name}"
        quoted_wheel_path = shlex.quote(wheel_path)
        # `--force-reinstall`, because a verified/content-addressed object may
        # retain the same wheel filename and version. Ordinary --upgrade would
        # silently keep an earlier build already installed on the VM.
        return "\n".join(
            (
                "mkdir -p /var/lib/eyvan/bootstrap",
                f"gcloud storage cp {shlex.quote(package_source)} {quoted_wheel_path}",
                *_verify_sha256(wheel_path, expected_package, "package"),
                *wheelhouse_lines,
                f"{python} -m pip install {pip_flags} --force-reinstall --no-deps {quoted_wheel_path}",
                # Dependencies are resolved separately so --force-reinstall does
                # not rebuild the entire tree on every bootstrap.
                f"{python} -m pip install {pip_flags} {quoted_wheel_path}",
            )
        )
    if package_sha256 is not None or package_generation is not None:
        raise ValueError("native worker package digest/generation requires a gs:// wheel URI")
    if package.endswith(".whl"):
        # A wheel file copied to the host (``eyvan gcp connect``) has the same
        # problem as the gs:// one above: every rebuild keeps its filename and
        # version, so ``--upgrade`` alone leaves the earlier build installed and
        # a reconnect would silently leave stale code installed.
        return "\n".join(
            (
                *wheelhouse_lines,
                f"{python} -m pip install {pip_flags} --force-reinstall --no-deps {quoted}",
                f"{python} -m pip install {pip_flags} {quoted}",
            )
        )
    return "\n".join((*wheelhouse_lines, f"{python} -m pip install {pip_flags} {quoted}"))


def derive_native_worker_token(
    shared_secret: str | None,
    management_uid: str,
    *,
    provider_generation: int = 1,
) -> str | None:
    """Derive a VM-scoped worker credential without exposing the agent secret."""

    if shared_secret is None:
        return None
    shared_secret = _required(shared_secret, "shared secret")
    management_uid = _required(management_uid, "management UID")
    if isinstance(provider_generation, bool) or not isinstance(provider_generation, int) or provider_generation < 1:
        raise ValueError("native worker provider generation must be a positive integer")
    return hmac.new(
        shared_secret.encode(),
        f"eyvan-native-worker:{management_uid}:r{provider_generation}".encode(),
        hashlib.sha256,
    ).hexdigest()


def build_native_worker_startup_script(
    *,
    package: str,
    port: int,
    state_dir: str,
    worker_secret_resource: str | None = None,
    worker_secret_file: str | None = None,
    worker_tls_secret_resource: str | None = None,
    package_sha256: str | None = None,
    package_generation: str | None = None,
    wheelhouse: str | None = None,
    wheelhouse_sha256: str | None = None,
    wheelhouse_generation: str | None = None,
    before_start: str | None = None,
    runtime_setup: str | None = None,
    runtime_env: tp.Mapping[str, str] | None = None,
    runtime_container: NativeContainerConfig | None = None,
    container_artifact_mounts: tp.Sequence[str] | None = None,
    runtime_python: str | None = None,
) -> str:
    """Render an idempotent systemd bootstrap for one native worker.

    ``before_start`` preserves the legacy provider setup hook and runs before
    Eyvan is installed. ``runtime_setup`` is generated from the typed fleet
    runtime and runs after the worker virtual environment exists, allowing an
    artifact to be installed without placing bootstrap details in job
    submission. ``runtime_container`` is a typed OCI handoff; its Docker
    engine preflight intentionally runs after setup so fleet init commands can
    install or configure the engine.

    ``worker_secret_file`` opts into a pre-delivered persistent root-only
    credential file instead of Secret Manager. Each systemd start copies it
    into the private RuntimeDirectory. The caller owns secure delivery; the
    default Secret Manager rendering is unchanged.

    ``runtime_python`` builds the worker virtual environment on a named
    interpreter instead of the image's ``python3``. A serialized callable is
    pinned to the interpreter that pickled it, and a TPU image ships whatever
    Python it ships -- 3.11 on the images in use, against a 3.13 laptop, which
    is a failure discovered only once a slice is ACTIVE and the workload
    dispatched.

    **Declaring it gives up the image's site packages.** The default venv is
    built ``--system-site-packages`` so the worker inherits the preinstalled
    JAX and libtpu, which exist for the image's own Python and no other. A venv
    on a different interpreter cannot use them, so a fleet that declares this
    must install its own accelerator stack in ``runtime.init_commands`` -- which
    run after the venv exists and before the worker unit starts, exactly so they
    can. Left unset, nothing about the bootstrap changes.
    """

    package = _required(package, "package")
    state_dir = _required(state_dir, "state directory")
    if (worker_secret_resource is None) == (worker_secret_file is None):
        raise ValueError("native worker requires exactly one secret resource or secret file")
    if worker_secret_resource is not None:
        worker_secret_resource = _required(worker_secret_resource, "secret resource")
    if worker_secret_file is not None:
        worker_secret_file = _required(worker_secret_file, "secret file")
        secret_source = PurePosixPath(worker_secret_file)
        if not secret_source.is_absolute() or ".." in secret_source.parts or str(secret_source) == "/":
            raise ValueError("native worker secret file must be an absolute non-root path without '..'")
    if runtime_python is not None:
        match = _DECLARED_PYTHON_MAJOR_MINOR.match(runtime_python)
        if match is not None and not (3, 11) <= tuple(map(int, match.group(0).split("."))) < (3, 14):
            raise ValueError("Eyvan workers require Python 3.11 through 3.13; choose a supported runtime.python")
    supported_python = shlex.quote("import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))")
    if wheelhouse is not None:
        wheelhouse = _required(wheelhouse, "wheelhouse")
    if port <= 0 or port > 65535:
        raise ValueError("native worker port must be between 1 and 65535")
    if before_start is not None and not before_start.strip():
        raise ValueError("native worker setup script must be non-empty when set")
    if runtime_setup is not None and not runtime_setup.strip():
        raise ValueError("native worker runtime setup must be non-empty when set")
    runtime_env = _runtime_environment(runtime_env)
    container_payload = _container_runtime_payload(runtime_container, container_artifact_mounts)
    runtime_environment_lines: tuple[str, ...] = ()
    if runtime_env:
        encoded_environment = base64.b64encode(
            json.dumps(runtime_env, sort_keys=True, separators=(",", ":")).encode()
        ).decode()
        quoted_environment_path = shlex.quote(_WORKLOAD_RUNTIME_ENVIRONMENT)
        runtime_environment_lines = (
            # Create the file private: there is a window between the shell
            # redirect and the chmod below in which the file would otherwise be
            # world-readable, so the redirect itself runs under a private umask.
            f"(umask 077; printf '%s' {shlex.quote(encoded_environment)} | base64 --decode > {quoted_environment_path})",
            f"chmod 0600 {quoted_environment_path}",
        )
    runtime_container_lines: tuple[str, ...] = ()
    if container_payload is not None:
        encoded_container = base64.b64encode(
            json.dumps(container_payload, sort_keys=True, separators=(",", ":")).encode()
        ).decode()
        quoted_container_path = shlex.quote(_WORKLOAD_CONTAINER_RUNTIME)
        runtime_container_lines = (
            # Same private-at-creation window as the environment file above.
            f"(umask 077; printf '%s' {shlex.quote(encoded_container)} | base64 --decode > {quoted_container_path})",
            f"chmod 0600 {quoted_container_path}",
        )
    runtime_setup_lines: tuple[str, ...] = ()
    if runtime_setup is not None:
        runtime_setup_lines = ("", "# Declarative fleet runtime setup.", runtime_setup.rstrip())
    container_preflight_lines: tuple[str, ...] = ()
    if container_payload is not None:
        container_preflight_lines = (
            "if ! command -v docker >/dev/null 2>&1 || ! docker version >/dev/null 2>&1; then",
            "  echo 'Eyvan OCI runtime requires a running Docker-compatible engine; "
            "install and start it in runtime.init_commands' >&2",
            "  exit 1",
            "fi",
        )

    venv_dir = "/opt/eyvan-native/venv"
    venv_python = f"{venv_dir}/bin/python"
    # A declared interpreter is built with uv, which fetches the interpreter
    # itself rather than asking the image to already have it -- the whole point
    # is that the image's Python is the wrong one. Deliberately *not*
    # `--system-site-packages`: those packages are built for the image's Python
    # and importing them under another interpreter is the failure this is meant
    # to avoid, so the fleet installs its own stack in `runtime.init_commands`.
    #
    # The interpreter's home must be shared, not uv's default: uv installs
    # managed interpreters under the invoking user's $HOME, which on a VM
    # startup script is /root -- mode 0700 -- so the unprivileged workload user
    # would be locked out of its venv. Bootstrap verification would fail and
    # replacements would inherit the same inaccessible interpreter.
    python_home = "/opt/eyvan-native/python"
    declared_python_lines: tuple[str, ...] = ()
    if runtime_python is not None:
        quoted_python = shlex.quote(runtime_python)
        # The guard below compares the running interpreter's major.minor, so a
        # declaration that pins a full version ("3.11.9") is reduced to the
        # same two components: comparing the declared *text* against that join
        # never matched, and the venv was torn down and rebuilt on every boot.
        pinned = _DECLARED_PYTHON_MAJOR_MINOR.match(runtime_python)
        declared_version = pinned.group(0) if pinned else runtime_python
        declared_python_lines = (
            f"# Eyvan fleet declares Python {runtime_python} for its workers.",
            # Rebuild not only on a version mismatch but also when the existing
            # interpreter resolves into a private home directory: that venv
            # predates the shared interpreter home and can never pass the
            # workload-user executability check this script ends with.
            # The declared version is compared as a Python *string*: passed
            # through shlex.quote alone it rendered as a float literal
            # (`== 3.13`), which never equals the joined version string and so
            # rebuilt the venv on every boot.
            f'if ! {venv_python} -c "import sys; raise SystemExit(0 if '
            f"'.'.join(map(str, sys.version_info[:2])) == '{declared_version}' else 1)\" >/dev/null 2>&1 || "
            f'readlink -f {venv_python} | grep -qE "^/(root|home)/"; then',
            "  if ! command -v uv >/dev/null 2>&1; then",
            "    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh",
            "  fi",
            "  if ! command -v uv >/dev/null 2>&1; then",
            f"    echo 'Eyvan needs uv to build a Python {runtime_python} worker environment' >&2",
            "    exit 1",
            "  fi",
            f"  install -d -m 0755 {python_home}",
            f"  rm -rf {venv_dir}",
            # --seed, because the very next block recreates any environment
            # whose `python -m pip` does not answer -- and a uv venv has no pip
            # unless asked. Without it this builds 3.13 and is immediately
            # replaced by one on the image's own interpreter, which is the
            # failure it exists to prevent.
            f"  env UV_PYTHON_INSTALL_DIR={python_home} uv venv --seed --python {quoted_python} {venv_dir}",
            "fi",
        )
    runtime_dir = "/run/eyvan-native"
    secret_path = f"{runtime_dir}/worker-secret"
    workload_user = "eyvan-workload"
    workload_group = "eyvan-workload"
    workload_directory = "/var/lib/eyvan/workload"
    multihost_site_directory = "/opt/eyvan-native/multihost"
    systemd_runtime_environment = "/etc/eyvan-native/systemd-runtime.env"
    environment = [
        "Environment=PYTHONUNBUFFERED=1",
        f"Environment=PATH={venv_dir}/bin:/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"Environment=VIRTUAL_ENV={venv_dir}",
        f"Environment=EYVAN_NATIVE_WORKLOAD_USER={workload_user}",
        f"Environment=EYVAN_NATIVE_WORKLOAD_GROUP={workload_group}",
        f"Environment=EYVAN_NATIVE_WORKLOAD_DIRECTORY={workload_directory}",
        f"Environment=XDG_CACHE_HOME={workload_directory}/.cache",
        f"EnvironmentFile={systemd_runtime_environment}",
    ]
    command_parts = [
        f"{venv_python} -m eyvan.worker.runtime.native_worker",
        "--host 0.0.0.0",
        f"--port {port}",
        f"--state-dir {shlex.quote(state_dir)}",
        f"--secret-file {shlex.quote(secret_path)}",
    ]
    tls_pre_start: tuple[str, ...] = ()
    if worker_tls_secret_resource is not None:
        from eyvan.worker.runtime.native_tls_bootstrap import validate_tls_secret_resource

        validate_tls_secret_resource(worker_tls_secret_resource)
        tls_directory = f"{runtime_dir}/tls"
        tls_pre_start = (
            f"ExecStartPre={venv_python} -m eyvan.worker.runtime.native_tls_bootstrap "
            f"--resource {worker_tls_secret_resource} --directory {tls_directory}",
        )
        command_parts.extend(
            (
                "--mtls-mode required",
                f"--mtls-trust-bundle {tls_directory}/trust_bundle.pem",
                f"--mtls-certificate {tls_directory}/certificate.pem",
                f"--mtls-private-key {tls_directory}/private_key.pem",
            )
        )
    if runtime_env:
        command_parts.extend(("--runtime-env-file", shlex.quote(_WORKLOAD_RUNTIME_ENVIRONMENT)))
    if container_payload is not None:
        command_parts.extend(("--container-runtime-file", shlex.quote(_WORKLOAD_CONTAINER_RUNTIME)))
    command_parts.append("--systemd-transient")
    command = " ".join(command_parts)
    if worker_secret_file is not None:
        # Keep a root-only persistent source across RuntimeDirectory cleanup/reboots.
        # The caller writes it over SSH stdin; neither the unit nor argv holds a token.
        fetch_secret = f"install -o root -g root -m 0600 {shlex.quote(worker_secret_file)} {shlex.quote(secret_path)}"
    else:
        assert worker_secret_resource is not None
        fetch_secret = " ".join(
            (
                "umask 077;",
                "gcloud secrets versions access",
                shlex.quote(worker_secret_resource),
                ">",
                shlex.quote(secret_path),
            )
        )
    unit = "\n".join(
        (
            "[Unit]",
            "Description=Eyvan native worker",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            "RuntimeDirectory=eyvan-native",
            "RuntimeDirectoryMode=0700",
            *environment,
            *tls_pre_start,
            f"ExecStartPre=/bin/sh -ec {shlex.quote(fetch_secret)}",
            f"ExecStart={command}",
            "KillMode=control-group",
            "Restart=always",
            "RestartSec=2",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
        )
    )
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "eyvan_apt_get() {",
        "  if ! command -v apt-get >/dev/null 2>&1; then",
        "    echo 'Eyvan bootstrap requires apt-get for this operation' >&2",
        "    return 127",
        "  fi",
        '  command apt-get -o DPkg::Lock::Timeout=300 "$@"',
        "}",
    ]
    if before_start:
        lines.extend(("", "# Fleet-provided setup runs before the Eyvan worker service.", before_start.rstrip()))
    lines.extend(
        (
            "",
            "export DEBIAN_FRONTEND=noninteractive",
            "mkdir -p /var/lib/eyvan/bootstrap",
            *(
                (
                    "if ! command -v gcloud >/dev/null 2>&1; then",
                    "  echo 'Eyvan native worker requires gcloud in the VM image' >&2",
                    "  exit 1",
                    "fi",
                )
                if worker_secret_file is None
                or package.startswith("gs://")
                or wheelhouse is not None
                or worker_tls_secret_resource is not None
                else ()
            ),
            "if ! command -v systemd-run >/dev/null 2>&1 || ! command -v systemctl >/dev/null 2>&1; then",
            "  echo 'Eyvan native worker requires systemd-run and systemctl in the VM image' >&2",
            "  exit 1",
            "fi",
            "if ! command -v runuser >/dev/null 2>&1; then",
            "  echo 'Eyvan native worker requires runuser in the VM image' >&2",
            "  exit 1",
            "fi",
            "systemd_version=$(systemctl --version | awk 'NR == 1 { print $2 }')",
            "systemd_exit_type_cgroup=0",
            'if [ "${systemd_version}" -ge 250 ] 2>/dev/null; then',
            "  systemd_exit_type_cgroup=1",
            "fi",
            "install -d -m 0755 /etc/eyvan-native",
            "printf 'EYVAN_SYSTEMD_EXIT_TYPE_CGROUP=%s\\n' \"${systemd_exit_type_cgroup}\" "
            "> /etc/eyvan-native/systemd-runtime.env",
            "chmod 0644 /etc/eyvan-native/systemd-runtime.env",
            *runtime_environment_lines,
            *runtime_container_lines,
            *declared_python_lines,
            f"if ! {venv_python} -c {supported_python} >/dev/null 2>&1; then",
            f"  rm -rf {venv_dir}",
            "fi",
            f"if ! {venv_python} -m pip --version >/dev/null 2>&1; then",
            f"  rm -rf {venv_dir}",
            # Ask every interpreter on the host before asking apt for one. The
            # image that motivated this ships `python3` as 3.10 without
            # `ensurepip` *and* a 3.11 that has it, so the bootstrap reached for
            # apt while a usable interpreter sat beside it -- and apt then lost
            # the dpkg lock to a wedged `unattended-upgrade`, which cost a
            # v5p-16 that had taken six attempts to obtain. Nothing here needs
            # the network or a package manager.
            "  eyvan_python=python3",
            f"  if ! python3 -c {supported_python} >/dev/null 2>&1 || "
            "! python3 -m ensurepip --version >/dev/null 2>&1; then",
            "    for eyvan_dir in $(printf '%s' \"${PATH}\" | tr ':' ' ') /usr/local/bin /usr/bin; do",
            '     for candidate in "${eyvan_dir}"/python3.*; do',
            '      [ -x "${candidate}" ] || continue',
            f'      if "${{candidate}}" -c {supported_python} >/dev/null 2>&1 && '
            '"${candidate}" -m ensurepip --version >/dev/null 2>&1; then',
            '        eyvan_python="${candidate}"',
            '        echo "Eyvan native worker is using ${candidate} for its virtual environment; "\\',
            '             "python3 has no ensurepip or is outside supported Python 3.11 through 3.13" >&2',
            "        break",
            "      fi",
            "     done",
            '     if [ "${eyvan_python}" != python3 ]; then break; fi',
            "    done",
            "  fi",
            f'  if ! "${{eyvan_python}}" -c {supported_python} >/dev/null 2>&1; then',
            "    echo 'Eyvan requires Python 3.11 through 3.13; set runtime.python to a supported version' >&2",
            "    exit 1",
            "  fi",
            '  if ! "${eyvan_python}" -m ensurepip --version >/dev/null 2>&1; then',
            "    if ! command -v apt-get >/dev/null 2>&1; then",
            "      echo 'Eyvan native worker requires Python venv support and apt-get is unavailable' >&2",
            "      exit 1",
            "    fi",
            # Keep both commands inside the condition so `set -e` cannot abort
            # before the actionable Eyvan diagnostic below. apt can exit 0 from
            # `update` when individual index fetches fail, so the install may be
            # the first command to report the failure.
            "    if ! eyvan_apt_get update || ! eyvan_apt_get install -y --no-install-recommends python3-venv; then",
            "      echo 'Eyvan native worker could not install python3-venv from apt' >&2",
            "    fi",
            "  fi",
            '  if ! "${eyvan_python}" -m ensurepip --version >/dev/null 2>&1; then',
            "    echo 'Eyvan native worker could not install Python venv support' >&2",
            "    exit 1",
            "  fi",
            f'  "${{eyvan_python}}" -m venv --system-site-packages {venv_dir}',
            "fi",
            _install_command(
                package,
                python=venv_python,
                package_sha256=package_sha256,
                package_generation=package_generation,
                wheelhouse=wheelhouse,
                wheelhouse_sha256=wheelhouse_sha256,
                wheelhouse_generation=wheelhouse_generation,
            ),
            f"chmod -R a+rX {venv_dir}",
            f"install -d -m 0755 {multihost_site_directory}",
            f'multihost_module=$({venv_python} -c "import eyvan.worker.runtime._multihost_site as module; '
            'print(module.__file__)")',
            f'install -m 0644 "${{multihost_module}}" {multihost_site_directory}/eyvan_multihost.py',
            f'distributed_environment_module=$({venv_python} -c "import eyvan.worker.distributed_environment as module; '
            'print(module.__file__)")',
            f'install -m 0644 "${{distributed_environment_module}}" '
            f"{multihost_site_directory}/eyvan_distributed_environment.py",
            f"cat >{multihost_site_directory}/sitecustomize.py <<'EYVAN_MULTIHOST_SITECUSTOMIZE'",
            # Scrub-then-initialize lives in eyvan_multihost.initialize_worker_runtime,
            # not here: ordering expressed in a heredoc can only be asserted as a
            # string, and the order is the whole correctness property.
            "from eyvan_multihost import initialize_worker_runtime",
            "",
            "initialize_worker_runtime()",
            "EYVAN_MULTIHOST_SITECUSTOMIZE",
            f"chmod 0644 {multihost_site_directory}/sitecustomize.py",
            f"if ! getent group {workload_group} >/dev/null 2>&1; then",
            f"  groupadd --system {workload_group}",
            "fi",
            f"if ! id -u {workload_user} >/dev/null 2>&1; then",
            f"  useradd --system --gid {workload_group} --home-dir {workload_directory} --create-home "
            f"--shell /usr/sbin/nologin {workload_user}",
            "fi",
            f"if ! runuser -u {workload_user} -- {venv_python} -c 'import sys' >/dev/null 2>&1; then",
            "  echo 'Eyvan native worker virtual environment is not executable by the workload user; "
            "install its interpreter under a shared path such as /opt/eyvan-native/python' >&2",
            "  exit 1",
            "fi",
            # The per-host cache for per-job ``dependencies``. Owned by the
            # workload user because that is who installs into it: left to root,
            # jobs would fall back to their working directories and reinstall
            # packages instead of sharing the per-host cache.
            f"install -d -o {workload_user} -g {workload_group} -m 0750 {workload_directory} "
            f"{workload_directory}/.cache {DEPENDENCY_CACHE_ROOT}",
            "for device in /dev/vfio/* /dev/vfio/devices/* /dev/accel* /dev/nvidiactl /dev/nvidia-uvm "
            "/dev/nvidia[0-9]*; do",
            '  [ -c "${device}" ] || continue',
            "  device_group=$(stat -c '%G' \"${device}\")",
            '  if [ "${device_group}" != root ] && getent group "${device_group}" >/dev/null 2>&1; then',
            f'    usermod -a -G "${{device_group}}" {workload_user}',
            "  fi",
            "done",
            "needs_device_acl=0",
            "for device in /dev/vfio/* /dev/vfio/devices/* /dev/accel* /dev/nvidiactl /dev/nvidia-uvm "
            "/dev/nvidia[0-9]*; do",
            '  [ -c "${device}" ] || continue',
            f'  if ! runuser -u {workload_user} -- test -r "${{device}}" -a -w "${{device}}"; then',
            "    needs_device_acl=1",
            "    break",
            "  fi",
            "done",
            'if [ "${needs_device_acl}" -eq 1 ]; then',
            "  if ! command -v setfacl >/dev/null 2>&1; then",
            "    if ! command -v apt-get >/dev/null 2>&1; then",
            "      echo 'Eyvan native worker requires setfacl to grant workload access to accelerator devices' >&2",
            "      exit 1",
            "    fi",
            "    eyvan_apt_get update",
            "    eyvan_apt_get install -y --no-install-recommends acl",
            "  fi",
            "  for device in /dev/vfio/* /dev/vfio/devices/* /dev/accel* /dev/nvidiactl /dev/nvidia-uvm "
            "/dev/nvidia[0-9]*; do",
            '    [ -c "${device}" ] || continue',
            f'    setfacl -m u:{workload_user}:rw "${{device}}"',
            "  done",
            "fi",
            "for device in /dev/vfio/* /dev/vfio/devices/* /dev/accel* /dev/nvidiactl /dev/nvidia-uvm "
            "/dev/nvidia[0-9]*; do",
            '  [ -c "${device}" ] || continue',
            f'  if ! runuser -u {workload_user} -- test -r "${{device}}" -a -w "${{device}}"; then',
            '    echo "Eyvan workload user cannot access accelerator device ${device}" >&2',
            "    exit 1",
            "  fi",
            "done",
            f"install -d -m 0700 {shlex.quote(state_dir)}",
            *runtime_setup_lines,
            *container_preflight_lines,
            "cat >/etc/systemd/system/eyvan-native-worker.service <<'EYVAN_NATIVE_WORKER_UNIT'",
            unit,
            "EYVAN_NATIVE_WORKER_UNIT",
            "systemctl daemon-reload",
            "systemctl enable eyvan-native-worker.service",
            "systemctl restart eyvan-native-worker.service",
            "",
        )
    )
    return "\n".join(lines)


__all__ = ("build_native_worker_startup_script", "derive_native_worker_token", "render_native_runtime_setup")
