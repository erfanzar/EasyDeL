# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``sudo eyvan team up``: turn the machine it runs on into a team-mode Eyvan plane.

It runs *on* the target host (a plain Debian or Ubuntu VM, for example a GCE
instance with an external address) as root. No gcloud, no ssh: everything
happens locally through one injected :class:`TeamUpRunner`, so the sequencing,
refusals and secret handling are testable without a machine to break.

What it leaves behind:

* ``/etc/eyvan/plane.env`` (root, 0600): infrastructure settings only -- the
  database URL, repository backend, artifact signing key, metrics token, TLS
  paths and public URL. No mode setting: team mode comes from
* ``/etc/eyvan/platform.yaml`` with ``team: true`` (an existing ``agents``
  section is kept);
* PostgreSQL on the host, a ``eyvan`` system user, ``/opt/eyvan/venv`` with this
  Eyvan version (or ``--wheel``), a CA kept across re-runs and a server
  certificate naming the address (and ``--dns-name``);
* ``eyvan-plane.service`` running ``eyvan-server --host 0.0.0.0 --port N``.

Then it migrates the schema, starts the unit, waits for ``/healthz`` over TLS
verified against the generated CA, and asks ``eyvan-server bootstrap-admin`` for
the first admin's one-time invite. Generated secrets travel only through files
and child-process environments, never through an argv or the printed output.
"""

from __future__ import annotations

import importlib.metadata
import ipaddress
import json
import os
import re
import secrets
import shlex
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import typing as tp
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import click
import yaml

from eyvan._internal.compat import StrEnum
from eyvan._internal.plane_tls import HEAD_CONTROL_PLANE_CA_PATH
from eyvan._internal.platform import EyvanPlatformConfig
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.sdk.commands.head import (
    HEAD_ENVIRONMENT_PATH,
    HEAD_TLS_CERT_PATH,
    HEAD_VENV_PATH,
    HOST_PLANE_SECRET_KEYS,
    PLATFORM_CONFIG_PATH,
    HostInstallSpec,
    host_plane_environment,
    host_postgres_lines,
    host_prelude_lines,
    host_server_unit,
    host_tls_lines,
    host_venv_lines,
)
from eyvan.sdk.team_credentials import certificate_fingerprint, display_fingerprint

#: HTTPS on an unprivileged port. 443 would need the unit to hold
#: CAP_NET_BIND_SERVICE; 8080 is where Eyvan serves *plaintext* (development
#: planes and ``eyvan head up``), and reusing it for TLS invites ``http://``
#: guesses that fail confusingly.
TEAM_PLANE_PORT = 8443
TEAM_ENVIRONMENT_PATH = "/etc/eyvan/plane.env"
TEAM_SERVICE_USER = "eyvan"
TEAM_SERVICE_NAME = "eyvan-plane.service"
TEAM_BOOTSTRAP_PATH = "/opt/eyvan/team-bootstrap.sh"
TEAM_ARTIFACT_DIR = "/var/lib/eyvan/managed-artifacts"
TEAM_FIREWALL_TAG = "eyvan-team-plane"
GCE_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/"
HEALTH_TIMEOUT_S = 120.0
HEALTH_INTERVAL_S = 2.0

#: Minimal, explicit environment for every child process. The operator's own
#: environment is deliberately not inherited: a stray development setting in a
#: sudo session must not decide how the plane is migrated or bootstrapped.
_BASE_ENVIRONMENT = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "HOME": "/root",
}
_SUBJECT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_DNS_PATTERN = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,61}[a-z0-9]$")
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_PATH_PATTERN = re.compile(r"^/[A-Za-z0-9._+/-]+\.whl$")
_ADMIN_EXISTS_MARKER = "already has team accounts"
_DATABASE_URL_KEY = next(key for key in HOST_PLANE_SECRET_KEYS if key.endswith("_DATABASE_URL"))
_METRICS_TOKEN_KEY = next(key for key in HOST_PLANE_SECRET_KEYS if key.endswith("_METRICS_TOKEN"))
_ARTIFACT_KEY_KEY = next(key for key in HOST_PLANE_SECRET_KEYS if key.endswith("_ARTIFACT_TOKEN_KEY"))


class TeamUpStep(StrEnum):
    """One named boundary of ``eyvan team up``."""

    PREFLIGHT = "preflight"
    ENVIRONMENT = "environment"
    PLATFORM = "platform"
    INSTALL = "install"
    MIGRATE = "migrate"
    SERVICE = "service"
    HEALTH = "health"
    CERTIFICATE = "certificate"
    BOOTSTRAP_ADMIN = "bootstrap_admin"


class TeamUpError(RuntimeError):
    """One step did not complete; the message says which and why, without secrets."""

    def __init__(self, step: TeamUpStep, detail: str):
        self.step = step
        self.detail = detail
        super().__init__(f"eyvan team up {step.value.replace('_', ' ')} failed: {detail}")


class TeamUpRunner(tp.Protocol):
    """Typed process boundary for every command ``eyvan team up`` runs on the host."""

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        env: tp.Mapping[str, str],
        timeout: float,
        stream: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run one bounded command; ``stream`` sends its output to the terminal's stderr."""

        ...


def run_host_command(
    args: tp.Sequence[str],
    *,
    env: tp.Mapping[str, str],
    timeout: float,
    stream: bool,
) -> subprocess.CompletedProcess[str]:
    """Run one command on this host with exactly ``env`` and no terminal stdin."""

    if stream:
        try:
            target: int | None = sys.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            target = None
        if target is not None:
            return subprocess.run(
                list(args),
                env=provider_cli_env(env),
                stdin=subprocess.DEVNULL,
                stdout=target,
                stderr=target,
                text=True,
                timeout=timeout,
                check=False,
            )
    return subprocess.run(
        list(args),
        env=provider_cli_env(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def read_gce_metadata(path: str, *, timeout: float = 2.0) -> str | None:
    """Return one GCE metadata value, or ``None`` off GCE or on any failure.

    Proxies are bypassed: the metadata server is link-local, and a proxy that
    answered for it would be answering with someone else's facts.
    """

    request = urllib.request.Request(GCE_METADATA_URL + path, headers={"Metadata-Flavor": "Google"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.headers.get("Metadata-Flavor") != "Google":
                return None
            value = response.read(4096).decode("utf-8").strip()
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return value or None


def probe_health(address: str, port: int, ca_path: Path, timeout: float) -> bool:
    """Return whether ``GET /healthz`` answers 200 over TLS verified for ``address``.

    Connects to loopback -- a GCE external address is NAT, not an interface,
    and the firewall may not admit this host to itself -- but verifies the
    certificate for the advertised address against the plane's own CA, so a
    pass proves the exact certificate clients will be shown.
    """

    try:
        context = ssl.create_default_context(cafile=str(ca_path))
        # Chain and name are still verified. Python 3.13 adds X.509 strict
        # profile checks (key-identifier extensions) that the openssl versions
        # on current Debian/Ubuntu do not always emit; refusing over those
        # would report a healthy plane as down.
        context.verify_flags &= ~getattr(ssl, "VERIFY_X509_STRICT", 0)
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as raw:
            with context.wrap_socket(raw, server_hostname=address) as channel:
                channel.sendall(
                    f"GET /healthz HTTP/1.1\r\nHost: {address}:{port}\r\nConnection: close\r\n\r\n".encode("ascii")
                )
                data = b""
                while len(data) < 65536:
                    chunk = channel.recv(4096)
                    if not chunk:
                        break
                    data += chunk
    except (OSError, ssl.SSLError, ValueError):
        return False
    status = data.split(b"\r\n", 1)[0].split()
    return len(status) >= 2 and status[1] == b"200"


def effective_uid() -> int:
    """Return this process's effective uid (a seam for tests)."""

    return os.geteuid()


@dataclass(frozen=True, slots=True)
class GceFacts:
    """What the GCE metadata server says about this VM; every field may be unknown."""

    external_ip: str | None = None
    instance: str | None = None
    zone: str | None = None
    network: str | None = None


def discover_gce(fetch: tp.Callable[[str], str | None]) -> GceFacts:
    """Read the external address, name, zone and network of this VM, best effort."""

    def last_segment(value: str | None) -> str | None:
        return value.rsplit("/", 1)[-1] or None if value else None

    instance = fetch("instance/name")
    if instance is None:
        # Not GCE (or no metadata server): one bounded miss, not four.
        return GceFacts()
    return GceFacts(
        external_ip=fetch("instance/network-interfaces/0/access-configs/0/external-ip"),
        instance=instance,
        zone=last_segment(fetch("instance/zone")),
        network=last_segment(fetch("instance/network-interfaces/0/network")),
    )


def resolve_address(explicit: str | None, facts: GceFacts) -> str:
    """Return ``--address`` when given, else this VM's external IP, else refuse."""

    if explicit:
        return explicit
    if facts.external_ip:
        return facts.external_ip
    raise TeamUpError(
        TeamUpStep.PREFLIGHT,
        "no --address was given and this machine reported no external IP (not a GCE VM, or one without an "
        "external address); pass --address with the IP people will connect to",
    )


def default_package() -> str:
    """Return the pip requirement for the Eyvan version running this command."""

    try:
        version = importlib.metadata.version("eyvan")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TeamUpError(TeamUpStep.PREFLIGHT, "cannot tell which Eyvan version is running; pass --wheel PATH") from exc
    if "+" in version:
        raise TeamUpError(
            TeamUpStep.PREFLIGHT,
            f"this Eyvan is a local build ({version}) that no package index has; pass --wheel PATH to install it",
        )
    return f"eyvan=={version}"


def default_admin(environ: tp.Mapping[str, str]) -> str | None:
    """Return the account that ran ``sudo``, which is the natural first admin."""

    user = (environ.get("SUDO_USER") or "").strip().lower()
    return user if user and user != "root" else None


@dataclass(frozen=True, slots=True)
class TeamUpSpec:
    """Every value one team plane is made of, validated before anything runs."""

    address: str
    admin: str
    package: str | None = None
    wheel: str | None = None
    port: int = TEAM_PLANE_PORT
    dns_name: str | None = None
    source_ranges: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a value the rendered script or URL could not carry safely."""

        try:
            parsed = ipaddress.ip_address(self.address)
        except ValueError as exc:
            raise ValueError(f"--address {self.address!r} must be an IP address; put a host name in --dns-name") from exc
        if not isinstance(parsed, ipaddress.IPv4Address):
            raise ValueError("--address must be IPv4; use --dns-name for an IPv6-reachable name")
        if parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
            raise ValueError(f"--address {self.address} is not an address other machines can reach")
        if not _SUBJECT_PATTERN.fullmatch(self.admin):
            raise ValueError(f"admin user name {self.admin!r} must match [a-z0-9][a-z0-9._-]{{0,63}}")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        if self.dns_name is not None and not _DNS_PATTERN.fullmatch(self.dns_name):
            raise ValueError(f"--dns-name {self.dns_name!r} is not a lower-case DNS name")
        if (self.package is None) == (self.wheel is None):
            raise ValueError("exactly one of a package requirement or a wheel path is required")
        if self.wheel is not None and not _SAFE_PATH_PATTERN.fullmatch(self.wheel):
            raise ValueError(f"--wheel {self.wheel!r} must be an absolute path to a .whl without spaces or quotes")
        if self.package is not None and (not self.package.strip() or self.package != self.package.strip()):
            raise ValueError("package requirement must be a non-empty trimmed string")
        for value in self.source_ranges:
            try:
                ipaddress.ip_network(value, strict=False)
            except ValueError as exc:
                raise ValueError(f"--source-range {value!r} is not a CIDR range") from exc

    @property
    def host(self) -> str:
        """Return the name clients use: the DNS name when there is one."""

        return self.dns_name or self.address

    @property
    def public_url(self) -> str:
        """Return the URL invites, cookies and ``eyvan login`` use."""

        return f"https://{self.host}:{self.port}"

    def host_install(self) -> HostInstallSpec:
        """Return the host-local installer settings shared with ``eyvan head up``."""

        target = self.wheel or self.package or ""
        return HostInstallSpec(
            port=self.port,
            install_target=shlex.quote(target),
            # A rebuilt wheel usually keeps its version; only an index
            # requirement is real resolution work that --upgrade handles alone.
            force_reinstall=self.wheel is not None,
            local_database=True,
            service_user=TEAM_SERVICE_USER,
            environment_path=TEAM_ENVIRONMENT_PATH,
            unit_description="Eyvan team control plane",
            label="team plane",
        )

    def as_document(self) -> dict[str, tp.Any]:
        """Return the non-secret spec for dry-run and JSON output."""

        return {
            "address": self.address,
            "admin": self.admin,
            "dns_name": self.dns_name,
            "install": self.wheel or self.package,
            "port": self.port,
            "public_url": self.public_url,
            "service": TEAM_SERVICE_NAME,
            "service_user": TEAM_SERVICE_USER,
            "source_ranges": list(self.source_ranges),
        }


def _kept(value: str | None, minimum: int) -> str | None:
    """Return an existing secret worth keeping, or ``None`` to generate one."""

    if value and len(value) >= minimum and _TOKEN_PATTERN.fullmatch(value):
        return value
    return None


@dataclass(frozen=True, slots=True)
class TeamPlaneSecrets:
    """The three credentials a team plane needs, generated once and kept on re-runs."""

    metrics_token: str
    artifact_token_key: str
    database_password: str

    def __post_init__(self) -> None:
        """Refuse material too short, or with characters a shell or SQL literal would interpret."""

        for name, minimum in (("metrics_token", 32), ("artifact_token_key", 32), ("database_password", 24)):
            if _kept(getattr(self, name), minimum) is None:
                raise ValueError(f"{name} must be at least {minimum} URL-safe base64 characters")

    @classmethod
    def from_environment(cls, existing: tp.Mapping[str, str]) -> TeamPlaneSecrets:
        """Keep every usable secret already in the plane's environment file; generate the rest."""

        password = None
        url = existing.get(_DATABASE_URL_KEY)
        if url:
            try:
                password = _kept(_database_password(url), 24)
            except ValueError:
                password = None
        return cls(
            metrics_token=_kept(existing.get(_METRICS_TOKEN_KEY), 32) or secrets.token_urlsafe(32),
            artifact_token_key=_kept(existing.get(_ARTIFACT_KEY_KEY), 32) or secrets.token_urlsafe(32),
            database_password=password or secrets.token_urlsafe(24),
        )

    def __repr__(self) -> str:
        """Keep generated material out of tracebacks."""

        return "TeamPlaneSecrets(metrics_token=***, artifact_token_key=***, database_password=***)"

    def values(self) -> tuple[str, ...]:
        """Return every secret value, for redaction."""

        return (self.metrics_token, self.artifact_token_key, self.database_password)

    def database_url(self) -> str:
        """Return the URL of the plane-local PostgreSQL."""

        return f"postgresql+asyncpg://eyvan:{self.database_password}@127.0.0.1:5432/eyvan"


def _database_password(url: str) -> str | None:
    """Return the decoded password of a database URL, if it has one."""

    password = urllib.parse.urlsplit(url).password
    return urllib.parse.unquote(password) if password else None


def plane_environment(spec: TeamUpSpec, plane_secrets: TeamPlaneSecrets) -> dict[str, str]:
    """Return the plane's environment: infrastructure settings only, never a mode."""

    return host_plane_environment(
        public_url=spec.public_url,
        database_url=plane_secrets.database_url(),
        metrics_token=plane_secrets.metrics_token,
        artifact_token_key=plane_secrets.artifact_token_key,
        artifact_store_uri=TEAM_ARTIFACT_DIR,
    )


def render_environment_file(environment: tp.Mapping[str, str]) -> str:
    """Render a systemd ``EnvironmentFile``."""

    return "".join(f"{key}={value}\n" for key, value in sorted(environment.items()))


def redacted_environment(environment: tp.Mapping[str, str]) -> dict[str, str]:
    """Return the environment with every secret value replaced by ``***``."""

    return {key: "***" if key in HOST_PLANE_SECRET_KEYS else value for key, value in sorted(environment.items())}


def parse_environment_file(text: str) -> dict[str, str]:
    """Parse ``NAME=value`` lines without sourcing them."""

    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() and not key.lstrip().startswith("#"):
            values[key.strip()] = value.strip()
    return values


def merged_platform_yaml(existing: str | None) -> str:
    """Return the platform file with ``team: true``, keeping everything else (notably ``agents``)."""

    document: dict[str, tp.Any] = {}
    if existing is not None and existing.strip():
        try:
            loaded = yaml.safe_load(existing)
        except yaml.YAMLError as exc:
            raise TeamUpError(TeamUpStep.PLATFORM, f"{PLATFORM_CONFIG_PATH} is not valid YAML: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise TeamUpError(TeamUpStep.PLATFORM, f"{PLATFORM_CONFIG_PATH} must be a mapping")
        document = dict(loaded or {})
    merged: dict[str, tp.Any] = {"version": document.pop("version", 1), "team": True}
    document.pop("team", None)
    merged.update(document)
    rendered = yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)
    # Validated with the server's own parser: a file the plane would refuse at
    # startup is refused here, before anything on the host has changed.
    try:
        EyvanPlatformConfig.from_document(yaml.safe_load(rendered))
    except ValueError as exc:
        raise TeamUpError(TeamUpStep.PLATFORM, f"{PLATFORM_CONFIG_PATH}: {exc}") from exc
    return rendered


def render_team_bootstrap_script(spec: TeamUpSpec) -> str:
    """Render the idempotent root installer. It carries no credentials.

    Secrets are read from ``TEAM_ENVIRONMENT_PATH``, which is written before the
    script runs, so the script can be printed in a dry run as-is.
    """

    host = spec.host_install()
    user = TEAM_SERVICE_USER
    address_lines: tuple[str, ...] = (
        "# TLS for the plane listener, under a CA this host owns and keeps across re-runs.",
        f"head_address={shlex.quote(spec.address)}",
    )
    if spec.dns_name is not None:
        address_lines += (f"head_dns_name={shlex.quote(spec.dns_name)}",)
    lines = [
        *host_prelude_lines(host),
        # systemd reads EnvironmentFile as root before dropping to the service
        # user, so nobody else needs to read the plane's credentials.
        f"chown root:root {TEAM_ENVIRONMENT_PATH}",
        f"chmod 0600 {TEAM_ENVIRONMENT_PATH}",
        "",
        "# Eyvan needs Python 3.11 or newer (Debian 12, Ubuntu 24.04 and later ship one).",
        "if ! command -v python3 >/dev/null 2>&1; then",
        "  eyvan_apt_get update",
        "  eyvan_apt_get install -y --no-install-recommends python3",
        "fi",
        "if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then",
        "  echo 'eyvan team up needs python3 >= 3.11 on this host (Debian 12+ or Ubuntu 24.04+)' >&2",
        "  exit 1",
        "fi",
        "if ! command -v openssl >/dev/null 2>&1; then",
        "  eyvan_apt_get update",
        "  eyvan_apt_get install -y --no-install-recommends openssl",
        "fi",
        *host_tls_lines(address_lines=address_lines, group=user, dns_name=spec.dns_name is not None),
        *host_venv_lines(host),
        *host_postgres_lines(host),
        f"install -d -o {user} -g {user} -m 0750 {TEAM_ARTIFACT_DIR}",
        f"cat >/etc/systemd/system/{TEAM_SERVICE_NAME} <<'TEAM_PLANE_UNIT_EOF'",
        host_server_unit(host),
        "TEAM_PLANE_UNIT_EOF",
        f"chmod 0644 /etc/systemd/system/{TEAM_SERVICE_NAME}",
        "systemctl daemon-reload",
        "",
    ]
    return "\n".join(lines)


def firewall_commands(spec: TeamUpSpec, facts: GceFacts) -> tuple[str, ...]:
    """Return the gcloud commands that would open the port; printed, never run."""

    ranges = ",".join(spec.source_ranges) or "YOUR_CIDR"
    return (
        shlex.join(
            (
                "gcloud",
                "compute",
                "firewall-rules",
                "create",
                f"eyvan-team-plane-{spec.port}",
                "--network",
                facts.network or "default",
                "--direction",
                "INGRESS",
                "--allow",
                f"tcp:{spec.port}",
                "--source-ranges",
                ranges,
                "--target-tags",
                TEAM_FIREWALL_TAG,
            )
        ),
        shlex.join(
            (
                "gcloud",
                "compute",
                "instances",
                "add-tags",
                facts.instance or "INSTANCE",
                "--zone",
                facts.zone or "ZONE",
                "--tags",
                TEAM_FIREWALL_TAG,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class TeamUpResult:
    """What one ``eyvan team up`` produced. ``invite_url`` is the one secret it prints."""

    spec: TeamUpSpec
    fingerprint: str
    invite_url: str | None
    invite_expires_at: str | None
    firewall: tuple[str, ...]

    @property
    def admin_existed(self) -> bool:
        """Whether the plane already had accounts, so no invite was created."""

        return self.invite_url is None

    @property
    def login_command(self) -> str:
        """Return the exact command the admin runs on their own machine."""

        return shlex.join(
            ("eyvan", "login", self.spec.public_url, "--username", self.spec.admin, "--fingerprint", self.fingerprint)
        )

    def as_document(self) -> dict[str, tp.Any]:
        """Return the result for ``--json``; includes the one-time invite link."""

        return {
            "admin": self.spec.admin,
            "admin_existed": self.admin_existed,
            "fingerprint": self.fingerprint,
            "firewall_commands": list(self.firewall),
            "invite_expires_at": self.invite_expires_at,
            "invite_url": self.invite_url,
            "login_command": self.login_command,
            "service": TEAM_SERVICE_NAME,
            "url": self.spec.public_url,
        }

    def render_lines(self) -> tuple[str, ...]:
        """Return the human-readable summary."""

        lines = [f"Eyvan team plane is up at {self.spec.public_url}", f"Certificate fingerprint: {self.fingerprint}", ""]
        if self.invite_url is None:
            lines.append(
                "This plane already has team accounts, so no invite was created. A fleet admin invites people "
                "from Settings -> Users or with `eyvan users invite`."
            )
        else:
            lines.extend(
                (
                    f"First admin invite for {self.spec.admin} (expires {self.invite_expires_at}).",
                    "Open it once, privately, and choose a password -- it is shown only here:",
                    f"  {self.invite_url}",
                )
            )
        lines.extend(
            (
                "",
                "Then, on your own machine:",
                f"  {self.login_command}",
                "",
                f"Open TCP {self.spec.port} to the people who use this plane (nothing was changed for you):",
                *(f"  {command}" for command in self.firewall),
            )
        )
        return tuple(lines)


#: Directories the ``eyvan`` service user must traverse (``platform.yaml``, the
#: CA, ``tls/``, the venv). Created and re-moded explicitly so the result never
#: depends on the umask of whoever ran ``sudo eyvan team up``; the secrets inside
#: keep their own 0600/0640 modes.
_SHARED_DIRECTORIES: tuple[str, ...] = ("/etc/eyvan", "/opt/eyvan")


def _ensure_shared_directory(target: Path) -> None:
    """Create ``target`` (and parents) and force it to 0755, repairing a narrower mode left by an earlier run."""

    target.mkdir(parents=True, exist_ok=True)
    os.chmod(target, 0o755)


def _write_atomic(target: Path, text: str, mode: int) -> None:
    """Replace ``target`` with ``text`` so it is never seen partially written or with a wider mode."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        Path(temporary).unlink(missing_ok=True)


class TeamUpInstaller:
    """Run the whole installation on this host through one injected runner."""

    def __init__(
        self,
        spec: TeamUpSpec,
        *,
        runner: TeamUpRunner | None = None,
        facts: GceFacts | None = None,
        root: Path = Path("/"),
        health: tp.Callable[[str, int, Path, float], bool] | None = None,
        sleep: tp.Callable[[float], None] = time.sleep,
        clock: tp.Callable[[], float] = time.monotonic,
        health_timeout_s: float = HEALTH_TIMEOUT_S,
    ):
        self.spec = spec
        self._runner = runner
        self._facts = facts or GceFacts()
        self._root = root
        self._health = health
        self._sleep = sleep
        self._clock = clock
        self._health_timeout_s = health_timeout_s
        self._redact: tuple[str, ...] = ()

    def path(self, absolute: str | Path) -> Path:
        """Return where an absolute host path lives under this installer's filesystem root."""

        return self._root / str(absolute).lstrip("/")

    def _read(self, absolute: str) -> str | None:
        target = self.path(absolute)
        try:
            return target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise TeamUpError(TeamUpStep.PREFLIGHT, f"cannot read {absolute}: {exc.strerror or exc}") from exc

    def _refuse_head(self) -> None:
        if self.path(HEAD_ENVIRONMENT_PATH).exists():
            raise TeamUpError(
                TeamUpStep.PREFLIGHT,
                f"{HEAD_ENVIRONMENT_PATH} exists: this host runs a plane installed by `eyvan head up`, which shares "
                "the venv, TLS and database; manage it with `eyvan head up --team` instead",
            )

    def plan(self) -> dict[str, tp.Any]:
        """Return everything ``up`` would do, with secrets redacted. Changes nothing."""

        self._refuse_head()
        placeholder = TeamPlaneSecrets(metrics_token="x" * 32, artifact_token_key="x" * 32, database_password="x" * 24)
        try:
            platform_existing = self._read(PLATFORM_CONFIG_PATH)
        except TeamUpError:
            platform_existing = None
        venv = f"{HEAD_VENV_PATH}/bin"
        return {
            "dry_run": True,
            "spec": self.spec.as_document(),
            "steps": [
                f"write {TEAM_ENVIRONMENT_PATH} (root, 0600); existing secrets are kept",
                f"set `team: true` in {PLATFORM_CONFIG_PATH}, keeping its other sections",
                f"write {TEAM_BOOTSTRAP_PATH} and run it: packages, user {TEAM_SERVICE_USER}, TLS, venv, "
                f"PostgreSQL, {TEAM_SERVICE_NAME}",
                f"{venv}/eyvan-migrate with the plane's environment",
                f"systemctl enable {TEAM_SERVICE_NAME} && systemctl restart {TEAM_SERVICE_NAME}",
                f"wait for https://{self.spec.address}:{self.spec.port}/healthz (via 127.0.0.1, verified "
                f"against {HEAD_CONTROL_PLANE_CA_PATH})",
                f"{venv}/eyvan-server bootstrap-admin --username {self.spec.admin} --team --json",
            ],
            "environment": redacted_environment(plane_environment(self.spec, placeholder)),
            "platform_yaml": merged_platform_yaml(platform_existing),
            "script": render_team_bootstrap_script(self.spec),
            "firewall_commands": list(firewall_commands(self.spec, self._facts)),
        }

    def up(self) -> TeamUpResult:
        """Install, start and bootstrap the plane; safe to run again."""

        self._refuse_head()
        # Merged before anything is written: an unreadable platform file stops
        # the run with nothing changed.
        platform = merged_platform_yaml(self._read(PLATFORM_CONFIG_PATH))
        existing = self._read(TEAM_ENVIRONMENT_PATH)
        plane_secrets = TeamPlaneSecrets.from_environment(parse_environment_file(existing or ""))
        self._redact = plane_secrets.values()
        environment = plane_environment(self.spec, plane_secrets)
        try:
            for directory in _SHARED_DIRECTORIES:
                _ensure_shared_directory(self.path(directory))
            _write_atomic(self.path(TEAM_ENVIRONMENT_PATH), render_environment_file(environment), 0o600)
        except OSError as exc:
            raise TeamUpError(TeamUpStep.ENVIRONMENT, f"cannot write {TEAM_ENVIRONMENT_PATH}: {exc}") from exc
        try:
            _write_atomic(self.path(PLATFORM_CONFIG_PATH), platform, 0o644)
            _write_atomic(self.path(TEAM_BOOTSTRAP_PATH), render_team_bootstrap_script(self.spec), 0o700)
        except OSError as exc:
            raise TeamUpError(TeamUpStep.PLATFORM, f"cannot write the platform file or installer: {exc}") from exc
        child = {**_BASE_ENVIRONMENT, **environment}
        self._execute(TeamUpStep.INSTALL, ("/bin/bash", TEAM_BOOTSTRAP_PATH), _BASE_ENVIRONMENT, 3600.0, stream=True)
        self._execute(TeamUpStep.MIGRATE, (f"{HEAD_VENV_PATH}/bin/eyvan-migrate",), child, 600.0)
        self._execute(TeamUpStep.SERVICE, ("systemctl", "enable", TEAM_SERVICE_NAME), _BASE_ENVIRONMENT, 120.0)
        self._execute(TeamUpStep.SERVICE, ("systemctl", "restart", TEAM_SERVICE_NAME), _BASE_ENVIRONMENT, 120.0)
        self._wait_healthy()
        fingerprint = self._fingerprint()
        invite_url, expires_at = self._bootstrap_admin(child)
        return TeamUpResult(
            spec=self.spec,
            fingerprint=fingerprint,
            invite_url=invite_url,
            invite_expires_at=expires_at,
            firewall=firewall_commands(self.spec, self._facts),
        )

    def _run(
        self, step: TeamUpStep, args: tp.Sequence[str], env: tp.Mapping[str, str], timeout: float, *, stream: bool
    ) -> subprocess.CompletedProcess[str]:
        # Resolved at call time, never bound as a default argument: a test that
        # forgets to inject a runner must hit the module's guarded boundary.
        runner = self._runner if self._runner is not None else run_host_command
        try:
            return runner(args, env=env, timeout=timeout, stream=stream)
        except FileNotFoundError as exc:
            raise TeamUpError(step, f"{args[0]} is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise TeamUpError(step, f"{shlex.join(args)} exceeded {timeout:g}s") from exc
        except OSError as exc:
            raise TeamUpError(step, f"could not start {args[0]}: {exc}") from exc

    def _execute(
        self,
        step: TeamUpStep,
        args: tp.Sequence[str],
        env: tp.Mapping[str, str],
        timeout: float,
        *,
        stream: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        result = self._run(step, args, env, timeout, stream=stream)
        if result.returncode != 0:
            raise TeamUpError(step, f"{shlex.join(args)} exited {result.returncode}{self._diagnostic(result)}")
        return result

    def _diagnostic(self, result: subprocess.CompletedProcess[str]) -> str:
        text = "\n".join(part for part in (result.stderr or "", result.stdout or "") if part).strip()
        for value in self._redact:
            text = text.replace(value, "***")
        return f": {text[-1500:]}" if text else ""

    def _wait_healthy(self) -> None:
        deadline = self._clock() + self._health_timeout_s
        ca_path = self.path(HEAD_CONTROL_PLANE_CA_PATH)
        health = self._health if self._health is not None else probe_health
        while True:
            if health(self.spec.address, self.spec.port, ca_path, 5.0):
                return
            if self._clock() >= deadline:
                raise TeamUpError(
                    TeamUpStep.HEALTH,
                    f"{TEAM_SERVICE_NAME} did not answer /healthz over TLS within {self._health_timeout_s:g}s; "
                    f"read `journalctl -u {TEAM_SERVICE_NAME} -n 100`",
                )
            self._sleep(HEALTH_INTERVAL_S)

    def _fingerprint(self) -> str:
        try:
            pem = self.path(HEAD_TLS_CERT_PATH).read_text(encoding="utf-8")
            return display_fingerprint(certificate_fingerprint(pem))
        except (OSError, ValueError) as exc:
            raise TeamUpError(TeamUpStep.CERTIFICATE, f"cannot read {HEAD_TLS_CERT_PATH}: {exc}") from exc

    def _bootstrap_admin(self, env: tp.Mapping[str, str]) -> tuple[str | None, str | None]:
        args = (
            f"{HEAD_VENV_PATH}/bin/eyvan-server",
            "bootstrap-admin",
            "--username",
            self.spec.admin,
            "--team",
            "--json",
        )
        result = self._run(TeamUpStep.BOOTSTRAP_ADMIN, args, env, 120.0, stream=False)
        if result.returncode != 0:
            if _ADMIN_EXISTS_MARKER in (result.stderr or ""):
                return None, None
            # bootstrap-admin prints its link only on success, so its failure
            # output carries no invite.
            raise TeamUpError(
                TeamUpStep.BOOTSTRAP_ADMIN, f"bootstrap-admin exited {result.returncode}{self._diagnostic(result)}"
            )
        try:
            document = json.loads((result.stdout or "").strip().splitlines()[-1])
            url = document["url"]
            expires_at = document.get("expires_at")
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            # Never echo the output here: on success it is the invite itself.
            raise TeamUpError(TeamUpStep.BOOTSTRAP_ADMIN, "bootstrap-admin printed no JSON result") from exc
        if not isinstance(url, str) or not url.startswith(f"{self.spec.public_url}/"):
            raise TeamUpError(
                TeamUpStep.BOOTSTRAP_ADMIN, f"bootstrap-admin returned a link that is not under {self.spec.public_url}"
            )
        return url, expires_at if isinstance(expires_at, str) else None


def render_plan_lines(document: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
    """Return the dry-run plan as text."""

    spec = document["spec"]
    lines = [f"eyvan team up would install a team plane at {spec['public_url']} and changes nothing now.", ""]
    lines.append("Steps:")
    lines.extend(f"  {index:2}. {step}" for index, step in enumerate(document["steps"], start=1))
    lines.extend(("", f"{TEAM_ENVIRONMENT_PATH} (secrets redacted):"))
    lines.extend(f"  {key}={value}" for key, value in document["environment"].items())
    lines.extend(("", f"{PLATFORM_CONFIG_PATH}:"))
    lines.extend(f"  {line}" for line in document["platform_yaml"].splitlines())
    lines.extend(("", "Firewall (printed after install, never run):"))
    lines.extend(f"  {command}" for command in document["firewall_commands"])
    lines.extend(("", f"{TEAM_BOOTSTRAP_PATH}:", document["script"]))
    return tuple(lines)


#: Seams the CLI resolves at call time, so tests can replace every host effect.
filesystem_root = Path("/")


@click.group("team")
def team_group() -> None:
    """Run a team-mode Eyvan plane on this machine."""


@team_group.command("up")
@click.option("--address", help="IPv4 address people connect to. [default: this GCE VM's external IP]")
@click.option("--dns-name", help="DNS name that resolves to the address; also put in the certificate.")
@click.option("--port", type=click.IntRange(1, 65535), default=TEAM_PLANE_PORT, show_default=True, help="HTTPS port.")
@click.option("--admin", help="User name of the first admin. [default: the account that ran sudo]")
@click.option(
    "--wheel",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Install this Eyvan wheel instead of this Eyvan version from the package index.",
)
@click.option(
    "--source-range",
    "source_ranges",
    multiple=True,
    help="CIDR allowed to reach the port, used in the printed firewall command; repeatable.",
)
@click.option("--dry-run", is_flag=True, help="Print the plan and installer with secrets redacted; change nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the plan or the result as JSON.")
def team_up(
    address: str | None,
    dns_name: str | None,
    port: int,
    admin: str | None,
    wheel: str | None,
    source_ranges: tuple[str, ...],
    dry_run: bool,
    as_json: bool,
) -> None:
    """Install a team-mode Eyvan plane on this Debian/Ubuntu host (run with sudo).

    Installs PostgreSQL, a venv with Eyvan, a private CA and server certificate,
    and the eyvan-plane systemd unit; then prints the certificate fingerprint,
    the first admin's one-time invite link and the `eyvan login` command.
    Running it again keeps the secrets, the CA and existing accounts.
    """

    if not dry_run and effective_uid() != 0:
        raise click.ClickException("eyvan team up changes this machine and must run as root: sudo eyvan team up")
    resolved_admin = admin.strip().lower() if admin else default_admin(os.environ)
    if resolved_admin is None:
        raise click.ClickException("pass --admin USERNAME (no sudo user to default to)")
    try:
        facts = discover_gce(read_gce_metadata)
        spec = TeamUpSpec(
            address=resolve_address(address, facts),
            admin=resolved_admin,
            package=None if wheel is not None else default_package(),
            wheel=wheel,
            port=port,
            dns_name=dns_name.strip().lower() if dns_name else None,
            source_ranges=tuple(source_ranges),
        )
        installer = TeamUpInstaller(spec, facts=facts, root=filesystem_root)
        if dry_run:
            document = installer.plan()
            if as_json:
                click.echo(json.dumps(document, indent=2, sort_keys=True))
            else:
                for line in render_plan_lines(document):
                    click.echo(line)
            return
        result = installer.up()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except TeamUpError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(result.as_document(), indent=2, sort_keys=True))
        return
    for line in result.render_lines():
        click.echo(line)


def register_team_up_commands(group: click.Group) -> None:
    """Attach ``eyvan team`` to the CLI root."""

    group.add_command(team_group)


__all__ = (
    "TEAM_ENVIRONMENT_PATH",
    "TEAM_PLANE_PORT",
    "TEAM_SERVICE_NAME",
    "GceFacts",
    "TeamPlaneSecrets",
    "TeamUpError",
    "TeamUpInstaller",
    "TeamUpResult",
    "TeamUpSpec",
    "TeamUpStep",
    "discover_gce",
    "merged_platform_yaml",
    "plane_environment",
    "probe_health",
    "read_gce_metadata",
    "register_team_up_commands",
    "render_team_bootstrap_script",
    "resolve_address",
    "run_host_command",
)
