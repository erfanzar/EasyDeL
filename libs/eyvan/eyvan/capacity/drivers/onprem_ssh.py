# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""On-premises capacity: one cell is a fenced claim on operator-owned hosts.

This is the third real capacity backend and the first that needs no cloud
account at all -- an air-gapped rack, a lab of workstations, a reserved row in a
colo. It exists as much to *prove the provider ABI* as to serve those fleets:
every property ``capacity/conformance.py`` demands of an out-of-tree provider is
implemented here against a substrate that is nothing like GCP or Kubernetes.

**How it differs from ``manual.py``.** The manual driver adopts hosts and owns
nothing on them: whatever answers on the declared endpoint is trusted. That is
honest for a fixed lab, and it means a host that was reused by another cell, or
a worker left over from a previous incarnation, is indistinguishable from the
one Eyvan meant to talk to. This driver claims each host by writing a fenced
*incarnation marker* over SSH, so:

* the claim is the leak detector -- a marker naming another cell, an older
  incarnation, or another fencing token is a **leak**, and the cell is reported
  ``NEEDS_RECREATE`` rather than adopted (the role ``management_uid`` plays on
  GCP and the fence-fingerprint annotation plays on Kubernetes);
* release removes the marker, so an abandoned host stops looking claimed;
* a host that cannot be reached is ``UNOBSERVED``, never "gone". SSH failing is
  the ordinary state of a rack during maintenance, and inferring absence from it
  would delete a cell whose workload is still running.

**What Eyvan owns and what the operator owns.** Eyvan owns the claim, the fence,
the marker, and the endpoint list. The operator owns how the native worker
process is started: either it is already running under systemd (the common
on-prem case, and the default here), or a ``start_command``/``stop_command``
argv is declared per pool and this driver runs it after claiming and before
releasing. Eyvan deliberately does not install software on a host it does not
own. Claim publication and lifecycle commands share a host-local marker lock;
stop/removal requires the exact cell, management UID, provider generation and
fence. A foreign, unreadable or missing marker never authorizes a stop.

**Replacement and crash recovery.** Live host inventories and ``state_dir``
cannot be replaced: explicitly release the cell before applying a different
inventory. Interrupted incarnation changes retain exact predecessor cleanup
identities until every host has the successor marker. Marker locks are never
stolen or expired, because a disconnected SSH command or its lifecycle child
may still be running. A host crash or killed shell may leave
``eyvan-cell.marker.lock`` behind; mutations then fail closed with a retryable
error. The operator must quiesce all marker commands and lifecycle children on
that host before removing an abandoned lock. Lifecycle commands must complete
synchronously; backgrounding destructive work escapes this arbitration.

**What is and is not covered by host tests.** The driver logic, the fence
comparisons, and the failure classification are exercised by the conformance
suite against an in-memory substrate. The POSIX marker script is exercised
directly through ``/bin/sh`` in ``tests/test_provider_abi_onprem.py``. What no
host test can cover is a real ``ssh`` binary against a real remote host --
``OpenSshCommandRunner`` is the seam where that begins, and it is deliberately
thin for exactly that reason.
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import typing as tp
from dataclasses import dataclass
from datetime import datetime
from threading import RLock

from eyvan._internal.compat import UTC
from eyvan._internal.identifiers import stable_id
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import (
    CapacityProviderError,
    CapacityProviderFailureCode,
    CellCapabilities,
    CellObservation,
    MutationContext,
    StaleMutationError,
)
from eyvan.capacity.fleet_state import (
    FleetCellNotRegisteredError,
    FleetCellRecord,
    FleetStateStore,
    StaleFenceError,
)

#: Where the release identity lives inside the cell tombstone. A tombstone field
#: rather than a new durable table: the record that survives a release already
#: exists, and this is exactly the information a replay of that release needs.
_CLEANUP_KEY = "onprem_ssh_cleanup"

#: Most hosts one on-premises cell may hold. The same reasoning that bounds a
#: Kubernetes Indexed Job bounds this: past it one failure unit is too large to
#: reason about, and a mistyped inventory should be refused rather than claimed.
MAX_HOSTS = 512

#: Marker fields, one ``key=value`` per line. Chosen over JSON because the
#: remote side is POSIX ``sh``: a format a shell can parse with ``sed`` is a
#: format whose parser can be read in full, and a malformed marker is refused
#: rather than half-understood.
_MARKER_FIELDS = ("cell_id", "management_uid", "provider_generation", "fence")

#: Values interpolated into the marker are restricted to this alphabet before
#: they are sent, so nothing that reaches a remote shell can carry a quote, a
#: newline, or a substitution.
_SAFE_VALUE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-:@/+")

#: Exit codes the marker script uses. ``sh`` reserves 126/127 and signals map
#: above 128, so these stay low and explicit.
_EXIT_OK = 0
_EXIT_CLAIMED_ELSEWHERE = 3
_EXIT_NO_MARKER = 4
_EXIT_HOST_IO = 5
_EXIT_USAGE = 6
_EXIT_BUSY = 7
_EXIT_LIFECYCLE = 8

#: The whole remote protocol. Sent on stdin to ``sh -s`` so no value is ever
#: interpolated into a command string, and shipped as one constant so the
#: semantics an operator audits are the semantics that run.
MARKER_SCRIPT = r"""
set -u
action=$1
marker=$2
cell=$3
uid=$4
generation=$5
fence=$6
shift 6
directory=$(dirname "$marker")

# mkdir is the host-local arbitration primitive, independent of fleet stores.
# Never expire/steal a lock: an old SSH command or its child may still be alive.
# SIGKILL/host failure leaves the lock fail-closed. An operator may remove it
# only after quiescing all marker commands and lifecycle children on this host.
case "$action" in
  claim|release)
    mkdir -p "$directory" || exit 5
    lock="$marker.lock"
    mkdir "$lock" 2>/dev/null || exit 7
    trap 'rmdir "$lock" 2>/dev/null' 0
    # Do not unlock while a lifecycle child is running on transport disconnect.
    trap '' HUP INT TERM
    ;;
  observe) ;;
  *) exit 6 ;;
esac

field() {
  sed -n "s/^$1=//p" "$2" 2>/dev/null | head -n 1
}

emit() {
  printf 'cell_id=%s\nmanagement_uid=%s\nprovider_generation=%s\nfence=%s\n' "$1" "$2" "$3" "$4"
}

case "$action" in
  observe)
    [ -f "$marker" ] || exit 4
    observed_cell=$(field cell_id "$marker")
    observed_uid=$(field management_uid "$marker")
    observed_generation=$(field provider_generation "$marker")
    observed_fence=$(field fence "$marker")
    emit "$observed_cell" "$observed_uid" "$observed_generation" "$observed_fence"
    ;;
  claim)
    if [ -f "$marker" ]; then
      current_cell=$(field cell_id "$marker")
      current_generation=$(field provider_generation "$marker")
      current_uid=$(field management_uid "$marker")
      current_fence=$(field fence "$marker")
      [ "$current_cell" = "$cell" ] && [ "$current_uid" = "$uid" ] || exit 3
      case "$current_generation" in
        ''|*[!0-9]*) exit 3 ;;
      esac
      [ "$current_generation" -le "$generation" ] || exit 3
      if [ "$current_generation" -eq "$generation" ]; then
        [ "$current_fence" = "$fence" ] || exit 3
      fi
    fi
    temporary="$marker.claim.$$"
    emit "$cell" "$uid" "$generation" "$fence" > "$temporary" || exit 5
    mv "$temporary" "$marker" || exit 5
    if [ "$#" -gt 0 ]; then
      "$@" >&2 || { status=$?; printf 'lifecycle command exited %s\n' "$status" >&2; exit 8; }
    fi
    emit "$cell" "$uid" "$generation" "$fence"
    ;;
  release)
    if [ -f "$marker" ]; then
      current_cell=$(field cell_id "$marker")
      current_uid=$(field management_uid "$marker")
      current_generation=$(field provider_generation "$marker")
      current_fence=$(field fence "$marker")
      # Release is destructive: only the exact observed incarnation may stop.
      [ "$current_cell" = "$cell" ] && [ "$current_uid" = "$uid" ] &&
        [ "$current_generation" = "$generation" ] && [ "$current_fence" = "$fence" ] || exit 3
      if [ "$#" -gt 0 ]; then
        "$@" >&2 || { status=$?; printf 'lifecycle command exited %s\n' "$status" >&2; exit 8; }
      fi
      rm -f "$marker" || exit 5
    fi
    ;;
  *)
    exit 6
    ;;
esac
exit 0
"""


class RemoteTransportError(RuntimeError):
    """The command could not be established as having run.

    Deliberately distinct from a non-zero exit code. A refused connection, a
    timeout, or an ``ssh`` transport failure says nothing about whether the
    remote command ran -- it may have claimed the host and lost the answer --
    so it can only ever produce an inconclusive, retryable result.
    """


@dataclass(frozen=True, slots=True)
class RemoteResult:
    """The conclusive outcome of one remote command."""

    exit_code: int
    stdout: str
    stderr: str


class RemoteCommandRunner(tp.Protocol):
    """Run one argv on one host, or say that it could not be established."""

    def run(
        self,
        host: str,
        argv: tp.Sequence[str],
        *,
        stdin: str = "",
        timeout_s: float,
    ) -> RemoteResult:
        """Execute ``argv`` on ``host``.

        Raises:
            RemoteTransportError: If the command's outcome is unknown.
        """
        ...


class OpenSshCommandRunner:
    """Run commands through the system ``ssh`` binary.

    ``BatchMode=yes`` is not a preference: an ``ssh`` that can prompt will block
    a reconcile loop forever on a host whose key changed, and the control plane
    would report the cell as still provisioning while nothing was happening.
    Exit code 255 and every timeout are transport failures, which is what makes
    them inconclusive rather than a claim that the host refused.
    """

    def __init__(self, *, connect_timeout_s: float = 10.0, options: tp.Sequence[str] = ()):
        self.connect_timeout_s = connect_timeout_s
        self.options = tuple(options)

    def run(
        self,
        host: str,
        argv: tp.Sequence[str],
        *,
        stdin: str = "",
        timeout_s: float,
    ) -> RemoteResult:
        """Execute ``argv`` on ``host`` over SSH."""

        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(max(self.connect_timeout_s, 1))}",
            *self.options,
            host,
            "--",
            # SSH joins remote words into a shell command, even though the
            # local subprocess uses argv. Quote once at that actual boundary.
            shlex.join(argv),
        ]
        try:
            completed = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteTransportError(f"ssh to {host} timed out after {timeout_s:.0f}s") from exc
        except OSError as exc:
            raise RemoteTransportError(f"ssh to {host} could not start: {exc}") from exc
        if completed.returncode == 255:
            detail = completed.stderr.strip().splitlines()[-1:] or ["no diagnostic"]
            raise RemoteTransportError(f"ssh transport to {host} failed: {detail[0]}")
        return RemoteResult(exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


@dataclass(frozen=True, slots=True)
class HostMarker:
    """One host's observed incarnation marker."""

    cell_id: str
    management_uid: str
    provider_generation: int
    fence: str


def marker_fingerprint(fencing_token: str) -> str:
    """Return a non-reversible fingerprint of a fencing token.

    The marker is world-readable on a host the operator's users can log into,
    so the token itself cannot go there. A fingerprint answers the only question
    asked of it: does this claim belong to the incarnation the record describes?
    """

    return hashlib.sha256(fencing_token.encode()).hexdigest()[:32]


def _safe_value(name: str, value: str) -> str:
    """Return ``value`` if it is safe to send to a remote shell."""

    if not value or any(character not in _SAFE_VALUE for character in value):
        raise ValueError(f"on-premises capacity {name} {value!r} contains characters that cannot cross a shell")
    return value


def _parse_marker(payload: str) -> HostMarker | None:
    """Parse a marker document, or ``None`` when it is malformed.

    A marker that cannot be parsed is never adopted. It is the same class of
    evidence as a marker naming another cell: something wrote it, and it was not
    this incarnation.
    """

    fields: dict[str, str] = {}
    for line in payload.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in _MARKER_FIELDS:
            fields.setdefault(key, value.strip())
    if set(fields) != set(_MARKER_FIELDS):
        return None
    generation = fields["provider_generation"]
    if not generation.isdigit():
        return None
    return HostMarker(
        cell_id=fields["cell_id"],
        management_uid=fields["management_uid"],
        provider_generation=int(generation),
        fence=fields["fence"],
    )


class OnPremSshCapacityDriver:
    """Claim, fence, observe, and release operator-owned hosts over SSH."""

    provider_name = "onprem-ssh"
    execution_backend = "eyvan-native"
    accelerator_types = ("cpu", "gpu", "tpu")
    capacity_classes = ("on-demand", "reserved")

    def __init__(
        self,
        state: FleetStateStore | None = None,
        *,
        runner: RemoteCommandRunner | None = None,
        native_worker_port: int = 8100,
        native_worker_state_dir: str = "/var/lib/eyvan/native-worker",
        command_timeout_s: float = 30.0,
    ):
        self.state = state or FleetStateStore.from_uri(None)
        self.runner = runner or OpenSshCommandRunner()
        self.native_worker_port = native_worker_port
        self.native_worker_state_dir = native_worker_state_dir.rstrip("/") or "/var/lib/eyvan/native-worker"
        self.command_timeout_s = command_timeout_s
        self._mutation_lock = RLock()

    def capabilities(self) -> CellCapabilities:
        """Describe on-premises capacity.

        Spot is absent for the same reason as in ``manual.py``: a machine the
        operator owns is never preemptible by a provider, and advertising it
        would let the scheduler make an interruption promise with no meaning.

        ``resource_enforcement`` is ``"backend"`` because the native worker
        fences each workload into its own cgroup on these hosts.
        """

        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=self.accelerator_types,
            capacity_classes=self.capacity_classes,
            capacity_backends=(self.provider_name,),
            supports_callables=True,
            supports_checkpoint=True,
            # The operator can already ssh to these machines; routing a
            # diagnostic through Eyvan is what makes it fenced and audited.
            supports_exec=True,
            resource_enforcement="backend",
        )

    # ---- desired-state validation ------------------------------------------

    @staticmethod
    def _nested_options(desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        """Return the provider-specific options block.

        The control plane nests everything provider-specific under
        ``provider_options``, so a driver reading its own fields only from the
        top level is unreachable through the controller however well it works in
        isolation. Both placements are accepted, as in ``manual.py``.
        """

        for key in ("provider_config", "provider_options"):
            value = desired.get(key)
            if is_json_object(value):
                return value
        return {}

    @staticmethod
    def _required_string(desired: tp.Mapping[str, JsonValue], name: str) -> str:
        value = desired.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"on-premises capacity field {name!r} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _optional_string(desired: tp.Mapping[str, JsonValue], name: str) -> str | None:
        value = desired.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"on-premises capacity field {name!r} must be a non-empty string when set")
        return value.strip()

    def _hosts(self, desired: tp.Mapping[str, JsonValue]) -> tuple[str, ...]:
        raw = desired.get("hosts")
        if raw is None:
            raw = self._nested_options(desired).get("hosts")
        if not isinstance(raw, list) or not raw:
            raise ValueError("on-premises capacity requires a non-empty 'hosts' list of ssh destinations")
        if len(raw) > MAX_HOSTS:
            raise ValueError(f"on-premises capacity supports at most {MAX_HOSTS} hosts per cell")
        hosts: list[str] = []
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("every on-premises capacity host must be a non-empty string")
            host = entry.strip()
            if host.startswith("ssh://"):
                host = host[len("ssh://") :]
            if "/" in host or " " in host:
                raise ValueError(f"on-premises capacity host {entry!r} must be an ssh destination, not a URL path")
            _safe_value("host", host)
            hosts.append(host)
        if len(set(hosts)) != len(hosts):
            raise ValueError("on-premises capacity hosts must not contain duplicates")
        return tuple(hosts)

    def _command(self, options: tp.Mapping[str, JsonValue], name: str) -> tuple[str, ...]:
        """Validate an operator-declared lifecycle command.

        An argv list rather than a shell string: the driver never builds a
        command line by concatenation, so there is no place for an inventory
        value to become a shell word.
        """

        value = options.get(name)
        if value is None:
            return ()
        if not isinstance(value, list) or not value:
            raise ValueError(f"on-premises capacity {name!r} must be a non-empty argv list when set")
        argv: list[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"every {name!r} argument must be a non-empty string")
            argv.append(entry)
        return tuple(argv)

    def _worker_port(self, options: tp.Mapping[str, JsonValue]) -> int:
        value = options.get("worker_port", self.native_worker_port)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            raise ValueError("on-premises capacity worker_port must be a TCP port number")
        return value

    def _state_dir(self, options: tp.Mapping[str, JsonValue]) -> str:
        value = options.get("state_dir", self.native_worker_state_dir)
        if not isinstance(value, str) or not value.startswith("/"):
            raise ValueError("on-premises capacity state_dir must be an absolute path")
        return _safe_value("state_dir", value.rstrip("/") or "/")

    def _validate_desired(self, desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        """Validate one apply document into the fields persisted on the record."""

        provider = self._required_string(desired, "provider")
        if provider != self.provider_name:
            raise ValueError(f"on-premises capacity requires provider {self.provider_name!r}, got {provider!r}")
        backend = self._required_string(desired, "capacity_backend")
        if backend != self.provider_name:
            raise ValueError(f"on-premises capacity requires capacity_backend {self.provider_name!r}, got {backend!r}")
        executor = self._required_string(desired, "executor")
        if executor != self.execution_backend:
            raise ValueError(f"on-premises capacity requires executor {self.execution_backend!r}, got {executor!r}")
        capacity_class = self._required_string(desired, "capacity_class").replace("_", "-")
        if capacity_class not in self.capacity_classes:
            choices = ", ".join(self.capacity_classes)
            raise ValueError(
                f"unsupported on-premises capacity class {capacity_class!r}; choose one of {choices}. "
                "Operator-owned hosts are never provider-preemptible, so spot is not offered."
            )
        options = self._nested_options(desired)
        return {
            "accelerator_type": self._required_string(desired, "accelerator_type"),
            "capacity_class": capacity_class,
            "region": self._required_string(desired, "region"),
            "zone": self._optional_string(desired, "zone"),
            "hosts": list(self._hosts(desired)),
            "worker_port": self._worker_port(options),
            "state_dir": self._state_dir(options),
            "start_command": list(self._command(options, "start_command")),
            "stop_command": list(self._command(options, "stop_command")),
        }

    # ---- durable record helpers, shared in shape with manual.py and k8s -----

    @staticmethod
    def _accept_mutation(record: FleetCellRecord, context: MutationContext) -> bool:
        if context.generation < record.generation:
            raise StaleMutationError(
                f"generation {context.generation} is older than cell {record.name!r} generation {record.generation}"
            )
        if (
            context.generation == record.generation
            and record.fencing_token is not None
            and record.fencing_token != context.fencing_token
        ):
            raise StaleMutationError(f"fencing token does not own cell {record.name!r} generation {record.generation}")
        return context.idempotency_key not in record.idempotency_keys

    @staticmethod
    def _stamp(record: FleetCellRecord, context: MutationContext) -> None:
        record.generation = context.generation
        record.fencing_token = context.fencing_token
        record.idempotency_keys = [*record.idempotency_keys, context.idempotency_key][-64:]

    def _revalidate_record_owner(
        self,
        cell_id: str,
        *,
        generation: int,
        fencing_token: str,
        provider_generation: int,
        phase: str,
    ) -> None:
        """Require the live fleet lease and the exact durable incarnation.

        Read the record *between* two lease reads, as the Kubernetes driver
        does, so an ownership change cannot be mistaken for a successful
        provider reconciliation.
        """

        try:
            self.state.require_lease()
            record = self.state.get(cell_id)
            self.state.require_lease()
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {cell_id!r} lost its fleet mutation lease {phase}") from exc
        owned = (
            record is not None
            and record.generation == generation
            and record.fencing_token == fencing_token
            and record.provider_generation == provider_generation
        )
        if not owned:
            raise StaleMutationError(f"cell {cell_id!r} changed durable ownership {phase}")

    # ---- provider failure classification ------------------------------------

    @staticmethod
    def _unavailable(detail: str) -> CapacityProviderError:
        """Report an outcome that could not be established.

        ``retryable`` and ``PROVIDER_UNAVAILABLE`` together are what keep the
        controller from charging this to a stockout belief or discarding the
        command: nothing was learned about the host, so nothing about the pool
        may be concluded.
        """

        return CapacityProviderError(
            detail,
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
            retryable=True,
        )

    @staticmethod
    def _configuration(detail: str) -> CapacityProviderError:
        """Report a conclusive refusal that retrying cannot fix."""

        return CapacityProviderError(
            detail,
            code=CapacityProviderFailureCode.CONFIGURATION,
            retryable=False,
        )

    # ---- remote protocol ----------------------------------------------------

    def _marker_path(self, state_dir: str) -> str:
        return f"{state_dir}/eyvan-cell.marker"

    def _run_marker(
        self,
        host: str,
        action: str,
        *,
        state_dir: str,
        cell_id: str,
        management_uid: str,
        provider_generation: int,
        fence: str,
        command: tp.Sequence[str] = (),
    ) -> RemoteResult:
        """Run ownership arbitration and any lifecycle action in one host lock."""

        argv = (
            "sh",
            "-s",
            "--",
            action,
            self._marker_path(state_dir),
            _safe_value("cell_id", cell_id),
            _safe_value("management_uid", management_uid),
            str(provider_generation),
            _safe_value("fence", fence),
            *command,
        )
        try:
            result = self.runner.run(host, argv, stdin=MARKER_SCRIPT, timeout_s=self.command_timeout_s)
            if result.exit_code == _EXIT_BUSY:
                raise self._unavailable(
                    f"on-premises host {host} has an active or abandoned marker lock; "
                    "retry, or quiesce remote marker/lifecycle processes before operator lock recovery"
                )
            if result.exit_code == _EXIT_LIFECYCLE:
                raise CapacityProviderError(
                    f"on-premises {action} lifecycle command failed on {host}: "
                    f"{result.stderr.strip() or 'no diagnostic'}",
                    code=CapacityProviderFailureCode.PROVIDER_ERROR,
                    retryable=False,
                )
            return result
        except RemoteTransportError as exc:
            raise self._unavailable(f"on-premises host {host} could not be reached: {exc}") from exc

    def _claim(
        self,
        host: str,
        *,
        state_dir: str,
        cell_id: str,
        management_uid: str,
        provider_generation: int,
        fence: str,
        command: tp.Sequence[str] = (),
    ) -> HostMarker:
        """Claim one host and optionally start its worker under the same lock."""

        result = self._run_marker(
            host,
            "claim",
            state_dir=state_dir,
            cell_id=cell_id,
            management_uid=management_uid,
            provider_generation=provider_generation,
            fence=fence,
            command=command,
        )
        if result.exit_code == _EXIT_CLAIMED_ELSEWHERE:
            raise self._configuration(
                f"on-premises host {host} is claimed by another Eyvan cell or a newer incarnation; "
                "release it there or remove it from this pool"
            )
        if result.exit_code == _EXIT_HOST_IO:
            raise CapacityProviderError(
                f"on-premises host {host} could not write its Eyvan claim under {state_dir}: "
                f"{result.stderr.strip() or 'no diagnostic'}",
                code=CapacityProviderFailureCode.PROVIDER_ERROR,
                retryable=False,
            )
        if result.exit_code != _EXIT_OK:
            raise self._configuration(
                f"on-premises host {host} refused the Eyvan claim (exit {result.exit_code}): "
                f"{result.stderr.strip() or 'no diagnostic'}"
            )
        marker = _parse_marker(result.stdout)
        if marker is None:
            raise CapacityProviderError(
                f"on-premises host {host} returned an unreadable claim marker",
                code=CapacityProviderFailureCode.PROVIDER_ERROR,
                retryable=False,
            )
        return marker

    def _observe_host(
        self,
        host: str,
        *,
        state_dir: str,
        cell_id: str,
        management_uid: str,
        provider_generation: int,
        fence: str,
    ) -> HostMarker | None:
        """Return the host's marker, or ``None`` when it holds none."""

        result = self._run_marker(
            host,
            "observe",
            state_dir=state_dir,
            cell_id=cell_id,
            management_uid=management_uid,
            provider_generation=provider_generation,
            fence=fence,
        )
        if result.exit_code == _EXIT_NO_MARKER:
            return None
        if result.exit_code != _EXIT_OK:
            raise CapacityProviderError(
                f"on-premises host {host} could not report its Eyvan claim (exit {result.exit_code}): "
                f"{result.stderr.strip() or 'no diagnostic'}",
                code=CapacityProviderFailureCode.PROVIDER_ERROR,
                retryable=False,
            )
        return _parse_marker(result.stdout)

    def _release_host(
        self,
        host: str,
        *,
        state_dir: str,
        cell_id: str,
        management_uid: str,
        provider_generation: int,
        fence: str,
        command: tp.Sequence[str] = (),
    ) -> None:
        """Stop and release only this exact host incarnation, under one lock."""

        result = self._run_marker(
            host,
            "release",
            state_dir=state_dir,
            cell_id=cell_id,
            management_uid=management_uid,
            provider_generation=provider_generation,
            fence=fence,
            command=command,
        )
        if result.exit_code == _EXIT_CLAIMED_ELSEWHERE:
            # Someone else owns the host now. Deleting their marker would be the
            # one destructive mistake this driver can make, so it does not.
            return
        if result.exit_code != _EXIT_OK:
            raise CapacityProviderError(
                f"on-premises host {host} could not release its Eyvan claim (exit {result.exit_code}): "
                f"{result.stderr.strip() or 'no diagnostic'}",
                code=CapacityProviderFailureCode.PROVIDER_ERROR,
                retryable=False,
            )

    # ---- observation --------------------------------------------------------

    @staticmethod
    def _record_hosts(record: FleetCellRecord) -> tuple[str, ...]:
        value = record.metadata.get("hosts")
        if not isinstance(value, list):
            return ()
        return tuple(entry for entry in value if isinstance(entry, str) and entry)

    @staticmethod
    def _record_string(record: FleetCellRecord, key: str, fallback: str) -> str:
        value = record.metadata.get(key)
        return value if isinstance(value, str) and value else fallback

    def _endpoints(self, record: FleetCellRecord) -> tuple[str, ...]:
        port = record.metadata.get("worker_port")
        resolved = port if isinstance(port, int) and not isinstance(port, bool) else self.native_worker_port
        return tuple(f"http://{host.split('@')[-1]}:{resolved}" for host in self._record_hosts(record))

    def _absent(self, cell_id: str) -> CellObservation:
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.ABSENT,
            generation=0,
            observed_at=datetime.now(UTC),
        )

    def _observation(
        self,
        record: FleetCellRecord,
        state: CapacityStatus,
        detail: str = "",
        *,
        publish_endpoints: bool,
        failure_code: CapacityProviderFailureCode | None = None,
    ) -> CellObservation:
        """Build an observation, publishing endpoints only for a fenced, ready cell.

        Endpoints are the part of an observation that has consequences: the
        agent's readiness probe and every execution resolution follow them. A
        cell whose claim could not be verified publishes none, so nothing can
        connect to a host this incarnation does not own.
        """

        endpoints = self._endpoints(record) if publish_endpoints else ()
        metadata: JsonObject = {
            "provider": self.provider_name,
            "capacity_backend": self.provider_name,
            "resource_id": f"onprem-ssh:{record.name}",
            "desired_state": record.desired_state,
            "provider_generation": record.provider_generation,
            **record.metadata,
            "native_worker_endpoints": list(endpoints),
        }
        return CellObservation(
            cell_id=record.name,
            state=state,
            generation=record.generation,
            execution_endpoint=endpoints[0] if len(endpoints) == 1 else None,
            detail=detail,
            observed_at=datetime.now(UTC),
            metadata=metadata,
            runtime_process_count=len(endpoints) if endpoints else None,
            failure_code=failure_code,
        )

    def observe(self, cell_id: str) -> CellObservation:
        """Report the cell's claim across its hosts.

        Three outcomes, and the difference between them is the whole point of
        this driver:

        * every host holds *this* incarnation's marker -> ``HEALTHY`` with
          endpoints published;
        * a host holds a marker from another cell, another fencing token, an
          older incarnation, or no marker at all -> ``NEEDS_RECREATE`` with no
          endpoints. That is a leak, and adopting it would publish an endpoint
          for a worker this incarnation does not own;
        * a host could not be reached -> ``UNOBSERVED`` with
          ``PROVIDER_UNAVAILABLE``. Not absent, not down, not healthy: unknown.
        """

        record = self.state.get(cell_id)
        if record is None:
            return self._absent(cell_id)
        if record.desired_state != "up":
            return self._observation(record, record.state, "cell is draining", publish_endpoints=False)
        hosts = self._record_hosts(record)
        if not hosts or not isinstance(record.fencing_token, str) or not record.fencing_token:
            return self._observation(
                record,
                CapacityStatus.NEEDS_RECREATE,
                "cell has no durable on-premises claim to verify",
                publish_endpoints=False,
            )
        state_dir = self._record_string(record, "state_dir", self.native_worker_state_dir)
        management_uid = self._record_string(record, "management_uid", "")
        fence = marker_fingerprint(record.fencing_token)
        leaked: list[str] = []
        for host in hosts:
            try:
                marker = self._observe_host(
                    host,
                    state_dir=state_dir,
                    cell_id=record.name,
                    management_uid=management_uid or "unknown",
                    provider_generation=record.provider_generation,
                    fence=fence,
                )
            except CapacityProviderError as exc:
                if exc.code is not CapacityProviderFailureCode.PROVIDER_UNAVAILABLE:
                    raise
                return self._observation(
                    record,
                    CapacityStatus.UNOBSERVED,
                    f"on-premises claim could not be observed: {exc.diagnostic}",
                    publish_endpoints=False,
                    failure_code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
                )
            if marker is None:
                leaked.append(f"{host} holds no readable Eyvan claim")
            elif marker.cell_id != record.name:
                leaked.append(f"{host} is claimed by cell {marker.cell_id!r}")
            elif marker.fence != fence or marker.provider_generation != record.provider_generation:
                leaked.append(f"{host} holds a previous incarnation of this cell")
            elif management_uid and marker.management_uid != management_uid:
                leaked.append(f"{host} holds a claim with a foreign management uid")
        if leaked:
            return self._observation(
                record,
                CapacityStatus.NEEDS_RECREATE,
                "; ".join(leaked[:8]),
                publish_endpoints=False,
            )
        return self._observation(record, CapacityStatus.HEALTHY, "", publish_endpoints=True)

    # ---- lifecycle ----------------------------------------------------------

    def _claim_cleanup(self, record: FleetCellRecord) -> JsonObject:
        """Capture the exact remote claim, independently of later durable fences."""

        return {
            "hosts": list(self._record_hosts(record)),
            "state_dir": self._record_string(record, "state_dir", self.native_worker_state_dir),
            "management_uid": self._record_string(record, "management_uid", stable_id("onprem-ssh", record.name)),
            "provider_generation": record.provider_generation,
            "fence": self._record_string(record, "claim_fence", marker_fingerprint(record.fencing_token or "unknown")),
            "stop_command": record.metadata.get("stop_command", []),
        }

    def _retain_predecessor_claim(self, record: FleetCellRecord) -> None:
        """Keep interrupted replacements releasable without loosening remote fencing."""

        if not self._record_hosts(record):
            return
        previous = record.metadata.get("predecessor_claims", [])
        if not isinstance(previous, list) or not all(is_json_object(item) for item in previous):
            raise self._configuration(f"cell {record.name!r} has malformed predecessor claims")
        if len(previous) >= 64:
            raise self._configuration("too many unfinished on-premises replacements; release this cell before retrying")
        record.metadata["predecessor_claims"] = [*previous, self._claim_cleanup(record)]

    def _settle_claims(self, cell_id: str, record: FleetCellRecord) -> None:
        """Drop predecessor identities only after every host has the current claim."""

        def settle(current: FleetCellRecord) -> None:
            if (
                current.generation != record.generation
                or current.fencing_token != record.fencing_token
                or current.provider_generation != record.provider_generation
            ):
                raise StaleMutationError(f"cell {cell_id!r} changed ownership before settling claims")
            current.metadata.pop("predecessor_claims", None)

        self.state.mutate_record(cell_id, settle, generation=record.generation)

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation:
        """Claim every declared host for this incarnation.

        The durable intent is written first, then the hosts are claimed, then
        the record is settled -- and the record's ownership is revalidated
        around every remote call, so a claim can never be attributed to an
        incarnation that stopped owning the cell while it was in flight. A
        replay of the same mutation does not bump ``provider_generation``,
        which is what makes the claim idempotent: the marker written is
        byte-identical, so the second claim changes nothing.
        """

        validated = self._validate_desired(desired)
        hosts = tuple(str(host) for host in tp.cast(list, validated["hosts"]))
        state_dir = str(validated["state_dir"])
        start_command = tuple(str(part) for part in tp.cast(list, validated["start_command"]))
        try:
            with self._mutation_lock, self.state.exclusive():
                captured: dict[str, int] = {}

                def mutate(record: FleetCellRecord) -> None:
                    accepted = self._accept_mutation(record, context)
                    # The persisted inventory is also the only cleanup identity.
                    # Refuse replacement (including changed replay payloads)
                    # before losing any previously claimed host or marker path.
                    if self._record_hosts(record) and (
                        set(self._record_hosts(record)) != set(hosts)
                        or self._record_string(record, "state_dir", self.native_worker_state_dir) != state_dir
                    ):
                        raise self._configuration(
                            "on-premises hosts and state_dir cannot change on a live cell; release it before replacement"
                        )
                    if not accepted:
                        captured["provider_generation"] = record.provider_generation
                        return
                    self._retain_predecessor_claim(record)
                    record.zone = tp.cast(str | None, validated["zone"])
                    record.accelerator_type = str(validated["accelerator_type"])
                    record.capacity_class = str(validated["capacity_class"])
                    record.desired_state = "up"
                    # Control-plane owned: Eyvan is about to issue a provider
                    # request, and has not yet observed a claim.
                    record.state = CapacityStatus.REQUESTING
                    record.provider_generation = record.provider_generation + 1
                    record.metadata["capacity_backend"] = self.provider_name
                    record.metadata["region"] = validated["region"]
                    record.metadata["hosts"] = list(hosts)
                    record.metadata["worker_port"] = validated["worker_port"]
                    record.metadata["state_dir"] = state_dir
                    record.metadata["start_command"] = list(start_command)
                    record.metadata["stop_command"] = validated["stop_command"]
                    # Host identity survives a drain changing the durable fence.
                    record.metadata["claim_fence"] = marker_fingerprint(context.fencing_token)
                    # Derived, not random: re-applying the same cell reproduces
                    # the same UID so the per-cell worker token stays stable and
                    # already-running workers keep authenticating.
                    record.metadata["management_uid"] = stable_id("onprem-ssh", cell_id)
                    self._stamp(record, context)
                    captured["provider_generation"] = record.provider_generation

                try:
                    record = self.state.mutate_or_create_record(cell_id, mutate, generation=context.generation)
                except StaleFenceError as exc:
                    raise StaleMutationError(str(exc)) from exc
                provider_generation = captured.get("provider_generation", record.provider_generation)
                management_uid = self._record_string(record, "management_uid", stable_id("onprem-ssh", cell_id))
                fence = marker_fingerprint(context.fencing_token)

                def revalidate(phase: str) -> None:
                    self._revalidate_record_owner(
                        cell_id,
                        generation=context.generation,
                        fencing_token=context.fencing_token,
                        provider_generation=provider_generation,
                        phase=phase,
                    )

                for host in self._record_hosts(record):
                    revalidate(f"before claiming {host}")
                    self._claim(
                        host,
                        state_dir=state_dir,
                        cell_id=cell_id,
                        management_uid=management_uid,
                        provider_generation=provider_generation,
                        fence=fence,
                    )
                    revalidate(f"after claiming {host}")
                for host in self._record_hosts(record):
                    revalidate(f"before starting the worker on {host}")
                    # Recheck the marker and start within the same remote lock:
                    # durable revalidation alone cannot protect a reused host.
                    if start_command:
                        self._claim(
                            host,
                            state_dir=state_dir,
                            cell_id=cell_id,
                            management_uid=management_uid,
                            provider_generation=provider_generation,
                            fence=fence,
                            command=start_command,
                        )
                    revalidate(f"after starting the worker on {host}")
                revalidate("before the final on-premises observation")
                observation = self.observe(cell_id)
                revalidate("after the final on-premises observation")
                if observation.state is CapacityStatus.HEALTHY:
                    self._settle_claims(cell_id, record)
                return observation
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {cell_id!r} lost its fleet mutation lease during apply") from exc

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Re-assert this incarnation's claim on every host it already holds.

        Re-claiming rather than merely observing is what recovers a host that
        was rebuilt, or whose marker was removed by a well-meaning cleanup: the
        cell converges back to HEALTHY instead of sitting in NEEDS_RECREATE with
        working hardware underneath it.
        """

        try:
            with self._mutation_lock, self.state.exclusive():
                record = self.state.get(cell_id)
                if record is None:
                    return self._absent(cell_id)

                def stamp(current: FleetCellRecord) -> None:
                    if not self._accept_mutation(current, context):
                        return
                    if current.desired_state == "up":
                        previous_fence = self._record_string(
                            current, "claim_fence", marker_fingerprint(current.fencing_token or "unknown")
                        )
                        next_fence = marker_fingerprint(context.fencing_token)
                        if previous_fence != next_fence:
                            # Equal remote generations cannot change identity.
                            # Persist the new intent before superseding markers.
                            self._retain_predecessor_claim(current)
                            current.provider_generation += 1
                        current.metadata["claim_fence"] = next_fence
                    self._stamp(current, context)

                try:
                    record = self.state.mutate_record(cell_id, stamp, generation=context.generation)
                except FleetCellNotRegisteredError:
                    return self._absent(cell_id)
                except StaleFenceError as exc:
                    raise StaleMutationError(str(exc)) from exc
                if record.desired_state != "up":
                    return self._observation(record, record.state, "cell is draining", publish_endpoints=False)
                fencing_token = record.fencing_token
                if not isinstance(fencing_token, str) or not fencing_token:
                    return self._observation(
                        record,
                        CapacityStatus.NEEDS_RECREATE,
                        "cell has no durable on-premises claim to re-assert",
                        publish_endpoints=False,
                    )
                state_dir = self._record_string(record, "state_dir", self.native_worker_state_dir)
                management_uid = self._record_string(record, "management_uid", stable_id("onprem-ssh", cell_id))
                provider_generation = record.provider_generation
                fence = marker_fingerprint(fencing_token)

                def revalidate(phase: str) -> None:
                    self._revalidate_record_owner(
                        cell_id,
                        generation=record.generation,
                        fencing_token=fencing_token,
                        provider_generation=provider_generation,
                        phase=phase,
                    )

                for host in self._record_hosts(record):
                    revalidate(f"before re-claiming {host}")
                    self._claim(
                        host,
                        state_dir=state_dir,
                        cell_id=cell_id,
                        management_uid=management_uid,
                        provider_generation=provider_generation,
                        fence=fence,
                    )
                    revalidate(f"after re-claiming {host}")
                revalidate("before the on-premises reconcile observation")
                observation = self.observe(cell_id)
                revalidate("after the on-premises reconcile observation")
                if observation.state is CapacityStatus.HEALTHY:
                    self._settle_claims(cell_id, record)
                return observation
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {cell_id!r} lost its fleet mutation lease during reconcile") from exc

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Stop scheduling onto these hosts while keeping the claim for release.

        The claim is deliberately *not* removed here. A draining cell can still
        be running the workload that is checkpointing, and a host whose marker
        disappeared would be claimable by another cell while that work runs.

        Draining is a mutation with consequences -- it takes capacity out of
        service and starts a workload teardown -- so a superseded controller
        must not be able to do it; every driver's drain checks this fence.
        """

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                self._accept_mutation(record, context)  # raises when this fence no longer owns the cell
                if record.fencing_token:
                    record.metadata.setdefault("claim_fence", marker_fingerprint(record.fencing_token))
                record.desired_state = "down"
                record.state = CapacityStatus.DRAINING
                self._stamp(record, context)

            try:
                record = self.state.drain_record(cell_id, context.generation, context.fencing_token, mutate)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.DRAINING,
                generation=context.generation,
                observed_at=datetime.now(UTC),
            )
        return self._observation(record, CapacityStatus.DRAINING, "draining", publish_endpoints=False)

    @staticmethod
    def _released(cell_id: str, generation: int) -> CellObservation:
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.RELEASED,
            generation=generation,
            observed_at=datetime.now(UTC),
        )

    def _tombstone_cleanup(
        self,
        cell_id: str,
        context: MutationContext,
        tombstone: JsonObject | None,
    ) -> JsonObject | None:
        """Validate a release replay and return the predecessor's host identity.

        Without this, a release whose durable record is already gone would have
        nothing to remove and would report success while every host still held
        a marker -- the exact leak this driver exists to detect.
        """

        if tombstone is None:
            return None
        generation = tombstone.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise self._configuration(f"cell {cell_id!r} has a malformed release tombstone generation")
        if context.generation != generation:
            relation = "older than" if context.generation < generation else "newer than"
            raise StaleMutationError(
                f"generation {context.generation} is {relation} released cell {cell_id!r} generation {generation}; "
                "an exact release replay is required without a live record"
            )
        if tombstone.get("fencing_token") != context.fencing_token:
            raise StaleMutationError(f"fencing token does not own released cell {cell_id!r} generation {generation}")
        cleanup = tombstone.get(_CLEANUP_KEY)
        if cleanup is None:
            return None
        if not is_json_object(cleanup):
            raise self._configuration(f"cell {cell_id!r} has a malformed on-premises release identity")
        return cleanup

    def _release_hosts(
        self,
        cell_id: str,
        cleanup: tp.Mapping[str, JsonValue],
        *,
        revalidate: tp.Callable[[str], None],
    ) -> None:
        """Release current and interrupted predecessor claims with exact identities."""

        predecessors = cleanup.get("predecessor_claims", [])
        if not isinstance(predecessors, list) or not all(is_json_object(item) for item in predecessors):
            raise self._configuration(f"cell {cell_id!r} has malformed predecessor release identities")
        self._release_identity_hosts(cell_id, cleanup, revalidate=revalidate)
        for predecessor in predecessors:
            self._release_identity_hosts(cell_id, tp.cast(JsonObject, predecessor), revalidate=revalidate)

    def _release_identity_hosts(
        self,
        cell_id: str,
        cleanup: tp.Mapping[str, JsonValue],
        *,
        revalidate: tp.Callable[[str], None],
    ) -> None:
        """Run fenced stop/removal for one durably captured remote incarnation."""

        hosts = [host for host in tp.cast(list, cleanup.get("hosts", [])) if isinstance(host, str)]
        state_dir = cleanup.get("state_dir")
        management_uid = cleanup.get("management_uid")
        fence = cleanup.get("fence")
        provider_generation = cleanup.get("provider_generation")
        if (
            not isinstance(state_dir, str)
            or not isinstance(management_uid, str)
            or not isinstance(fence, str)
            or isinstance(provider_generation, bool)
            or not isinstance(provider_generation, int)
        ):
            raise self._configuration(f"cell {cell_id!r} has an incomplete on-premises release identity")
        stop_command = tuple(part for part in tp.cast(list, cleanup.get("stop_command", [])) if isinstance(part, str))
        for host in hosts:
            revalidate(f"before stopping and releasing {host}")
            self._release_host(
                host,
                state_dir=state_dir,
                cell_id=cell_id,
                management_uid=management_uid,
                provider_generation=provider_generation,
                fence=fence,
                command=stop_command,
            )
            revalidate(f"after releasing {host}")

    def release(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Stop the workers, drop this cell's claim, and tombstone the record.

        The tombstone carries the host identity, so a replay whose record is
        already gone removes the same markers instead of trusting that a
        previous attempt finished. A release that cannot reach a host raises a
        retryable provider error rather than tombstoning a cell whose hosts are
        still claimed.
        """

        try:
            with self._mutation_lock, self.state.exclusive():
                record = self.state.get(cell_id)
                if record is None:
                    cleanup = self._tombstone_cleanup(cell_id, context, self.state.cell_tombstone(cell_id))
                    if cleanup is not None:

                        def revalidate_replay(phase: str) -> None:
                            self.state.require_lease()
                            if self.state.get(cell_id) is not None:
                                raise StaleMutationError(f"cell {cell_id!r} was recreated {phase}")

                        self._release_hosts(cell_id, cleanup, revalidate=revalidate_replay)
                    self.state.require_lease()
                    return self._released(cell_id, context.generation)

                captured: dict[str, JsonValue] = {}

                def validate_owner(current: FleetCellRecord) -> None:
                    self._accept_mutation(current, context)  # raises before any remote call when stale
                    if not isinstance(current.fencing_token, str) or not current.fencing_token:
                        raise StaleMutationError(f"cell {cell_id!r} has no durable on-premises ownership fence")
                    captured["generation"] = current.generation
                    captured["fencing_token"] = current.fencing_token
                    captured["claim_fence"] = self._record_string(
                        current, "claim_fence", marker_fingerprint(current.fencing_token)
                    )
                    captured["provider_generation"] = current.provider_generation
                    captured["predecessor_claims"] = current.metadata.get("predecessor_claims", [])
                    captured["hosts"] = list(self._record_hosts(current))
                    captured["state_dir"] = self._record_string(current, "state_dir", self.native_worker_state_dir)
                    captured["management_uid"] = self._record_string(
                        current, "management_uid", stable_id("onprem-ssh", cell_id)
                    )
                    captured["stop_command"] = [
                        part for part in tp.cast(list, current.metadata.get("stop_command", [])) if isinstance(part, str)
                    ]

                try:
                    self.state.mutate_record(cell_id, validate_owner, generation=context.generation)
                except FleetCellNotRegisteredError:
                    raise StaleMutationError(
                        f"cell {cell_id!r} disappeared before the on-premises release preflight"
                    ) from None
                except StaleFenceError as exc:
                    raise StaleMutationError(str(exc)) from exc

                durable_generation = int(tp.cast(int, captured["generation"]))
                fencing_token = str(captured["fencing_token"])
                provider_generation = int(tp.cast(int, captured["provider_generation"]))
                cleanup: JsonObject = {
                    "hosts": captured["hosts"],
                    "state_dir": captured["state_dir"],
                    "management_uid": captured["management_uid"],
                    "stop_command": captured["stop_command"],
                    "provider_generation": provider_generation,
                    "fence": captured["claim_fence"],
                    "predecessor_claims": captured["predecessor_claims"],
                }

                def revalidate(phase: str) -> None:
                    self._revalidate_record_owner(
                        cell_id,
                        generation=durable_generation,
                        fencing_token=fencing_token,
                        provider_generation=provider_generation,
                        phase=phase,
                    )

                self._release_hosts(cell_id, cleanup, revalidate=revalidate)
                revalidate("before the durable on-premises tombstone")

                def owned(current: FleetCellRecord) -> bool:
                    self._accept_mutation(current, context)
                    self._stamp(current, context)
                    if current.generation != context.generation or current.fencing_token != context.fencing_token:
                        raise StaleMutationError(f"cell {cell_id!r} changed ownership before on-premises release")
                    return True

                try:
                    self.state.tombstone_record(
                        cell_id,
                        context.generation,
                        context.fencing_token,
                        owned,
                        metadata={_CLEANUP_KEY: cleanup},
                    )
                except StaleFenceError as exc:
                    raise StaleMutationError(str(exc)) from exc
                return self._released(cell_id, context.generation)
        except StaleFenceError as exc:
            raise StaleMutationError(f"cell {cell_id!r} lost its fleet mutation lease during release") from exc


__all__ = (
    "MARKER_SCRIPT",
    "MAX_HOSTS",
    "HostMarker",
    "OnPremSshCapacityDriver",
    "OpenSshCommandRunner",
    "RemoteCommandRunner",
    "RemoteResult",
    "RemoteTransportError",
    "marker_fingerprint",
)
