# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Race one Cloud TPU shape across zones: request it everywhere at once, see who grants it.

Quota is not queue access. A project can hold a Spot quota of a thousand chips
in a zone and still be refused there (code 7, "no permission to submit requests
into this queue"), accepted but left waiting for hours, or granted only to be
suspended minutes later. The only way to learn which zones will actually hand
over a shape is to ask all of them. That is what this module does.

Every request it creates carries the race's id in its name and a label, and is
deleted again unless the caller asked to keep the first grant. Cleanup waits a
request out of ``PROVISIONING`` (where the API refuses deletes) instead of giving
up, and a final sweep lists each raced zone to prove nothing was left behind.
A zone that cannot be listed is reported as unreadable, never as clean.
"""

from __future__ import annotations

import json
import re
import secrets
import subprocess
import threading
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.providers.gcp.queued_resources import default_runtime_version

API = "https://tpu.googleapis.com/v2alpha1"
RACE_LABEL = "eyvan-race"
NAME_PREFIX = "eyvan-race-"
#: Where receipts land when the caller names no path; ``eyvan race sweep`` reads them.
RECEIPT_DIRECTORY = Path("~/.eyvan/races")

_ZONE = re.compile(r"^[a-z]+-[a-z]+[0-9]+-[a-z][a-z0-9]*$")
_ACCELERATOR = re.compile(r"^[a-z0-9]+-[0-9]+$")
_RUN_ID = re.compile(r"^[0-9a-f]{8}$")
_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_PAGE_SIZE = 500

#: Substrings identifying transient rate-limit 429s that should still be retried.
_TRANSIENT_RATE_LIMIT_PHRASES = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "queries per",
    "requests per",
    "per minute",
    "per second",
)

#: Substrings identifying permanent quota exhaustion on HTTP 429 (e.g. Limit: 0).
_PERMANENT_QUOTA_429_PHRASES = (
    "limit: 0",
    "quota limit",
    "quota exceeded",
    "exceeded quota",
    "insufficient quota",
)


def _is_permanent_quota_429(document: JsonObject) -> bool:
    """Return whether an HTTP 429 response represents permanent quota exhaustion rather than a transient rate limit."""

    text = error_text(document).lower()
    if any(phrase in text for phrase in _TRANSIENT_RATE_LIMIT_PHRASES):
        return False
    error = document.get("error")
    if is_json_object(error):
        details = error.get("details")
        if isinstance(details, list):
            for entry in details:
                if not is_json_object(entry):
                    continue
                reason = entry.get("reason")
                if isinstance(reason, str):
                    upper = reason.upper()
                    if "RATE_LIMIT" in upper:
                        return False
                    if upper == "QUOTA_EXCEEDED":
                        return True
    if any(phrase in text for phrase in _PERMANENT_QUOTA_429_PHRASES):
        return True
    if is_json_object(error) and str(error.get("status", "")).upper() == "RESOURCE_EXHAUSTED" and "quota" in text:
        return True
    return False


#: Queued-resource states after which the request will never become ACTIVE.
_ENDED_STATES = frozenset({"FAILED", "SUSPENDING", "SUSPENDED", "DELETING"})


class RaceOutcome(StrEnum):
    """What one zone answered."""

    GRANTED = "granted"
    REFUSED = "refused"  # the create operation failed: no queue access, unsupported, ...
    REJECTED = "rejected"  # the create call itself was refused
    FAILED = "failed"
    SUSPENDED = "suspended"
    WAITING = "waiting"  # still queued when the race ended
    CANCELLED = "cancelled"  # stopped early: another zone won, or the operator interrupted
    ERROR = "error"
    #: Not settled yet. Only an interim receipt (or one left by a crashed race) shows it.
    RUNNING = "running"


class TpuQueueApi(tp.Protocol):
    """The Cloud TPU calls a race needs. ``status`` is the HTTP status; 599 means no answer."""

    def list_zones(self, project: str) -> list[str]: ...

    def offers(self, project: str, zone: str, accelerator_type: str) -> bool | None: ...

    def create(self, project: str, zone: str, name: str, body: JsonObject) -> tuple[int, JsonObject]: ...

    def operation(self, name: str) -> tuple[int, JsonObject]: ...

    def get(self, project: str, zone: str, name: str) -> tuple[int, JsonObject]: ...

    def delete(self, project: str, zone: str, name: str) -> tuple[int, JsonObject]: ...

    def list_requests(self, project: str, zone: str) -> tuple[int, list[str]]: ...

    def list_nodes(self, project: str, zone: str) -> tuple[int, list[str]]: ...

    def delete_node(self, project: str, zone: str, name: str) -> tuple[int, JsonObject]: ...


def gcloud_token(
    account: str | None = None,
    *,
    runner: tp.Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_s: float = 60.0,
) -> str:
    """Mint a bearer token as ``account`` (or gcloud's active account) without touching gcloud config.

    The account is chosen for this one process through ``CLOUDSDK_CORE_ACCOUNT``,
    so a race as one account never changes what every other gcloud call uses.
    """

    environment = provider_cli_env()
    if account:
        environment["CLOUDSDK_CORE_ACCOUNT"] = account
    try:
        result = runner(
            ["gcloud", "auth", "print-access-token"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not run gcloud to mint a token: {exc}") from exc
    token = (result.stdout or "").strip()
    if result.returncode != 0 or not token:
        detail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"gcloud could not mint a token{f' for {account}' if account else ''}: {detail[-1:] or result.returncode}"
        )
    return token


@dataclass
class TpuQueueRest:
    """``TpuQueueApi`` over the Cloud TPU REST API with a gcloud-minted bearer token."""

    token_source: tp.Callable[[], str]
    transport: tp.Any = None
    timeout_s: float = 30.0
    sleep: tp.Callable[[float], None] = time.sleep
    _token: str | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _bearer(self, stale: str | None = None) -> str:
        with self._lock:
            if self._token is None or self._token == stale:
                self._token = self.token_source()
            return self._token

    def _call(
        self, method: str, url: str, *, body: JsonObject | None = None, params: dict[str, tp.Any] | None = None
    ) -> tuple[int, JsonObject]:
        import httpx

        kwargs: dict[str, tp.Any] = {"timeout": self.timeout_s, "trust_env": False}
        if self.transport is not None:
            kwargs["transport"] = self.transport
        token = self._bearer()
        reminted = False
        failure = "no answer"
        for attempt in range(4):
            try:
                with httpx.Client(**kwargs) as client:
                    response = client.request(
                        method, url, json=body, params=params, headers={"Authorization": f"Bearer {token}"}
                    )
            except httpx.HTTPError as exc:
                failure = type(exc).__name__
                self.sleep(2.0 * (attempt + 1))
                continue
            if response.status_code == 401 and not reminted:
                token, reminted = self._bearer(token), True
                continue
            try:
                raw_document = response.json() if response.content else {}
            except ValueError:
                raw_document = {"error": {"message": response.text[:300]}}
            document: JsonObject = raw_document if is_json_object(raw_document) else {}
            if response.status_code in _RETRYABLE and attempt < 3:
                if response.status_code == 429 and _is_permanent_quota_429(document):
                    return response.status_code, document
                self.sleep(5.0 * (attempt + 1))
                continue
            return response.status_code, document
        return 599, {"error": {"message": f"network: {failure}"}}

    @staticmethod
    def _parent(project: str, zone: str) -> str:
        return f"{API}/projects/{project}/locations/{zone}"

    def list_zones(self, project: str) -> list[str]:
        zones: list[str] = []
        page: str | None = None
        while True:
            params: dict[str, tp.Any] = {"pageSize": _PAGE_SIZE, **({"pageToken": page} if page else {})}
            status, document = self._call("GET", f"{API}/projects/{project}/locations", params=params)
            if status != 200:
                raise RuntimeError(f"could not list TPU locations of {project}: {error_text(document) or status}")
            for location in document.get("locations") or []:
                zone = location.get("locationId") if is_json_object(location) else None
                if isinstance(zone, str) and _ZONE.match(zone):
                    zones.append(zone)
            next_page = document.get("nextPageToken")
            if not isinstance(next_page, str) or not next_page:
                return sorted(set(zones))
            page = next_page

    def offers(self, project: str, zone: str, accelerator_type: str) -> bool | None:
        status, _ = self._call("GET", f"{self._parent(project, zone)}/acceleratorTypes/{accelerator_type}")
        return True if status == 200 else False if status == 404 else None

    def create(self, project: str, zone: str, name: str, body: JsonObject) -> tuple[int, JsonObject]:
        return self._call(
            "POST", f"{self._parent(project, zone)}/queuedResources", body=body, params={"queuedResourceId": name}
        )

    def operation(self, name: str) -> tuple[int, JsonObject]:
        return self._call("GET", f"{API}/{name}")

    def get(self, project: str, zone: str, name: str) -> tuple[int, JsonObject]:
        return self._call("GET", f"{self._parent(project, zone)}/queuedResources/{name}")

    def delete(self, project: str, zone: str, name: str) -> tuple[int, JsonObject]:
        return self._call("DELETE", f"{self._parent(project, zone)}/queuedResources/{name}", params={"force": "true"})

    def _names(self, project: str, zone: str, kind: str) -> tuple[int, list[str]]:
        names: list[str] = []
        page: str | None = None
        while True:
            params: dict[str, tp.Any] = {"pageSize": _PAGE_SIZE, **({"pageToken": page} if page else {})}
            status, document = self._call("GET", f"{self._parent(project, zone)}/{kind}", params=params)
            if status != 200:
                return status, []
            for item in document.get(kind) or []:
                name = item.get("name") if is_json_object(item) else None
                if isinstance(name, str):
                    names.append(name.rsplit("/", 1)[-1])
            next_page = document.get("nextPageToken")
            if not isinstance(next_page, str) or not next_page:
                return 200, names
            page = next_page

    def list_requests(self, project: str, zone: str) -> tuple[int, list[str]]:
        return self._names(project, zone, "queuedResources")

    def list_nodes(self, project: str, zone: str) -> tuple[int, list[str]]:
        return self._names(project, zone, "nodes")

    def delete_node(self, project: str, zone: str, name: str) -> tuple[int, JsonObject]:
        return self._call("DELETE", f"{self._parent(project, zone)}/nodes/{name}")

    def get_node(self, project: str, zone: str, name: str) -> tuple[int, JsonObject]:
        return self._call("GET", f"{self._parent(project, zone)}/nodes/{name}")


def error_text(document: JsonObject) -> str:
    """The provider's own words for a failure, as ``"<code> <message>"``."""

    error = document.get("error")
    if not is_json_object(error):
        return ""
    return f"{error.get('code', '')} {error.get('message', '')}".strip()


def error_code(document: JsonObject) -> int | None:
    error = document.get("error")
    code = error.get("code") if is_json_object(error) else None
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def request_name(run_id: str, zone: str) -> str:
    """The queued-resource (and node) id one race uses in one zone."""

    return f"{NAME_PREFIX}{run_id}-{zone}"


def request_body(
    *,
    project: str,
    zone: str,
    name: str,
    accelerator_type: str,
    runtime_version: str,
    spot: bool,
    run_id: str,
    enable_external_ips: bool = True,
) -> JsonObject:
    node: JsonObject = {
        "acceleratorType": accelerator_type,
        "runtimeVersion": runtime_version,
        "labels": {RACE_LABEL: run_id},
    }
    if enable_external_ips:
        # Cloud TPU v2alpha1 defaults external IPs to false; match gcloud's default
        # so --keep-first grants are reachable and healthy for `eyvan gcp connect`.
        node["networkConfig"] = {"enableExternalIps": True}
    body: JsonObject = {
        "tpu": {
            "nodeSpec": [
                {
                    "parent": f"projects/{project}/locations/{zone}",
                    "nodeId": name,
                    "node": node,
                }
            ]
        }
    }
    if spot:
        body["spot"] = {}
    return body


@dataclass
class ZoneResult:
    """One zone's answer and what became of the request made there."""

    zone: str
    name: str
    outcome: RaceOutcome = RaceOutcome.RUNNING
    error: str | None = None
    error_code: int | None = None
    accepted_after_s: float | None = None
    granted_after_s: float | None = None
    states: list[tuple[float, str]] = field(default_factory=list)
    kept: bool = False
    #: ``None`` when nothing was created; ``"deleted"``, ``"kept"`` or a failure otherwise.
    cleanup: str | None = None

    def document(self) -> JsonObject:
        return {
            "zone": self.zone,
            "name": self.name,
            "outcome": self.outcome.value,
            "error": self.error,
            "error_code": self.error_code,
            "accepted_after_s": self.accepted_after_s,
            "granted_after_s": self.granted_after_s,
            "states": [[seconds, state] for seconds, state in self.states],
            "kept": self.kept,
            "cleanup": self.cleanup,
        }


@dataclass
class SweepResult:
    """What a sweep found still present, per zone, after deleting what it could."""

    leftovers: dict[str, list[str]] = field(default_factory=dict)
    #: Zones that could not be listed. Unknown, so never reported as clean.
    unreadable: dict[str, str] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not any(self.leftovers.values()) and not self.unreadable

    def document(self) -> JsonObject:
        return {
            "leftovers": {zone: names for zone, names in self.leftovers.items() if names},
            "unreadable": dict(self.unreadable),
            "clean": self.clean,
        }


@dataclass(frozen=True)
class RaceTiming:
    """Poll intervals and deadlines, in seconds. Tests shrink them."""

    timeout_s: float = 1800.0
    cleanup_timeout_s: float = 3600.0
    poll_s: float = 10.0
    operation_poll_s: float = 2.0
    stagger_s: float = 0.3


class _Cleaner:
    """Deletes requests and sweeps zones, waiting out states the API will not delete from."""

    def __init__(
        self,
        api: TpuQueueApi,
        project: str,
        timing: RaceTiming,
        clock: tp.Callable[[], float],
        on_event: tp.Callable[[str, str], None] = lambda zone, message: None,
    ):
        self.api, self.project, self.timing, self.clock, self.on_event = api, project, timing, clock, on_event

    def delete(self, zone: str, name: str) -> str:
        """Delete one queued resource (force: its node goes with it) and wait until it is gone."""

        deadline = self.clock() + self.timing.cleanup_timeout_s
        last = ""
        last_state: str | None = None
        first_attempt = True
        while True:
            if not first_attempt and self.clock() >= deadline:
                return f"delete_timeout ({last or 'still present'})"
            first_attempt = False
            status, document = self.api.delete(self.project, zone, name)
            if status == 404:
                return "deleted"
            rejected = status >= 300
            if rejected:
                # Typically FAILED_PRECONDITION while PROVISIONING: wait for it
                # to leave PROVISIONING, then re-issue DELETE promptly.
                last = f"http {status}: {error_text(document)}"
            for _ in range(6):
                if rejected:
                    if self.clock() >= deadline:
                        return f"delete_timeout ({last or 'still present'})"
                    time.sleep(self.timing.poll_s)
                gone, qr_doc = self.api.get(self.project, zone, name)
                if gone == 404:
                    return "deleted"
                redelete_after_poll = False
                if gone == 200:
                    state_doc = qr_doc.get("state")
                    qr_state = state_doc.get("state") if is_json_object(state_doc) else None
                    if isinstance(qr_state, str):
                        prev_state = last_state
                        if qr_state != last_state:
                            last_state = qr_state
                            self.on_event(zone, f"deleting: {qr_state.lower()}")
                        if rejected and qr_state != "PROVISIONING":
                            break
                        if not rejected and qr_state == "SUSPENDED":
                            if prev_state != "SUSPENDED":
                                # Fresh transition into SUSPENDED from ACTIVE/SUSPENDING:
                                # re-issue DELETE promptly without waiting 60s.
                                break
                            # Already SUSPENDED (e.g. async DELETE in flight): sleep poll_s
                            # and enforce cleanup_timeout_s before re-issuing DELETE.
                            redelete_after_poll = True
                if not rejected:
                    if self.clock() >= deadline:
                        return f"delete_timeout ({last or 'still present'})"
                    time.sleep(self.timing.poll_s)
                    if redelete_after_poll:
                        break

    def _delete_node(self, zone: str, name: str) -> None:
        """Delete one stray TPU node and wait until the API confirms it is gone."""

        deadline = self.clock() + self.timing.cleanup_timeout_s
        status, _ = self.api.delete_node(self.project, zone, name)
        if status == 404:
            return
        get_node = getattr(self.api, "get_node", None)
        while True:
            if callable(get_node):
                gone, _ = get_node(self.project, zone, name)
                if gone == 404:
                    return
            else:
                list_status, current_nodes = self.api.list_nodes(self.project, zone)
                if list_status == 200 and name not in current_nodes:
                    return
            if self.clock() >= deadline:
                return
            time.sleep(self.timing.poll_s)

    def sweep(self, zones: tp.Iterable[str], run_id: str, *, keep: tp.Collection[str] = ()) -> SweepResult:
        """Delete everything this race left in ``zones`` (except ``keep``), then list again."""

        prefix = f"{NAME_PREFIX}{run_id}-"
        result = SweepResult()
        for zone in sorted(set(zones)):
            status, requests = self.api.list_requests(self.project, zone)
            if status != 200:
                result.unreadable[zone] = f"queued resources: http {status}"
                continue
            for name in requests:
                if name.startswith(prefix) and name not in keep:
                    self.delete(zone, name)
            status, nodes = self.api.list_nodes(self.project, zone)
            if status != 200:
                result.unreadable[zone] = f"nodes: http {status}"
                continue
            for name in nodes:
                if name.startswith(prefix) and name not in keep:
                    self._delete_node(zone, name)
            # Listed again, not assumed: a delete that was accepted is not a delete that happened.
            left: list[str] = []
            for kind, lister in (("queuedResource", self.api.list_requests), ("node", self.api.list_nodes)):
                status, names = lister(self.project, zone)
                if status != 200:
                    result.unreadable[zone] = f"{kind}s: http {status}"
                    break
                left.extend(f"{kind}/{name}" for name in names if name.startswith(prefix) and name not in keep)
            result.leftovers[zone] = left
        return result


@dataclass
class RaceReport:
    run_id: str
    project: str
    accelerator_type: str
    spot: bool
    keep_first: bool
    account: str | None
    started_at: float
    zones: list[ZoneResult] = field(default_factory=list)
    not_offered: list[str] = field(default_factory=list)
    unknown_offer: list[str] = field(default_factory=list)
    interrupted: bool = False
    finished_at: float | None = None
    sweep: SweepResult | None = None

    @property
    def kept(self) -> ZoneResult | None:
        return next((zone for zone in self.zones if zone.kept), None)

    @property
    def clean(self) -> bool:
        """Whether everything this race created is gone, except a deliberately kept grant."""

        failed_cleanup = any(z.cleanup not in (None, "deleted", "kept") for z in self.zones)
        return not failed_cleanup and (self.sweep is not None and self.sweep.clean)

    def document(self) -> JsonObject:
        kept = self.kept
        return {
            "run_id": self.run_id,
            "project": self.project,
            "accelerator_type": self.accelerator_type,
            "capacity_class": "spot" if self.spot else "on-demand",
            "keep_first": self.keep_first,
            "account": self.account,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "interrupted": self.interrupted,
            "raced_zones": [zone.zone for zone in self.zones],
            "not_offered": list(self.not_offered),
            "unknown_offer": list(self.unknown_offer),
            "zones": [zone.document() for zone in self.zones],
            "kept": None if kept is None else {"zone": kept.zone, "name": kept.name},
            "sweep": None if self.sweep is None else self.sweep.document(),
            "clean": self.clean,
        }


class TpuRace:
    """Request one shape in many zones concurrently; delete or keep what is granted."""

    def __init__(
        self,
        api: TpuQueueApi,
        *,
        project: str,
        accelerator_type: str,
        zones: tp.Sequence[str] = (),
        spot: bool = True,
        keep_first: bool = False,
        enable_external_ips: bool = True,
        runtime_version: str | None = None,
        account: str | None = None,
        timing: RaceTiming = RaceTiming(),
        run_id: str | None = None,
        on_event: tp.Callable[[str, str], None] = lambda zone, message: None,
        receipt: Path | None = None,
        clock: tp.Callable[[], float] = time.monotonic,
    ) -> None:
        if not project or not re.fullmatch(r"[a-z][-a-z0-9]{4,28}[a-z0-9]", project):
            raise ValueError(f"{project!r} is not a GCP project id")
        if not _ACCELERATOR.match(accelerator_type):
            raise ValueError(f"{accelerator_type!r} is not a TPU accelerator type such as v6e-8")
        for zone in zones:
            if not _ZONE.match(zone):
                raise ValueError(f"{zone!r} is not a zone such as us-east5-b")
        if run_id is not None and not _RUN_ID.match(run_id):
            raise ValueError("run_id must be 8 lowercase hex characters")
        self.api = api
        self.project = project
        self.accelerator_type = accelerator_type
        self.requested_zones = tuple(dict.fromkeys(zones))
        self.spot = spot
        self.keep_first = keep_first
        self.enable_external_ips = enable_external_ips
        self.runtime_version = runtime_version or default_runtime_version(accelerator_type)
        self.timing = timing
        self.on_event = on_event
        self.receipt = receipt
        self.clock = clock
        self.run_id = run_id or secrets.token_hex(4)
        self.report = RaceReport(
            run_id=self.run_id,
            project=project,
            accelerator_type=accelerator_type,
            spot=spot,
            keep_first=keep_first,
            account=account,
            started_at=time.time(),
        )
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._receipt_lock = threading.Lock()
        self._cleaner = _Cleaner(api, project, timing, clock, on_event=on_event)

    # ------------------------------------------------------------------ planning

    def candidate_zones(self) -> list[str]:
        """Zones to race: the ones named, or every zone of the project that offers the shape."""

        if self.requested_zones:
            return list(self.requested_zones)
        zones = self.api.list_zones(self.project)
        with ThreadPoolExecutor(max_workers=min(32, max(1, len(zones)))) as pool:
            offered = list(pool.map(lambda zone: self.api.offers(self.project, zone, self.accelerator_type), zones))
        self.report.not_offered = [zone for zone, answer in zip(zones, offered, strict=True) if answer is False]
        self.report.unknown_offer = [zone for zone, answer in zip(zones, offered, strict=True) if answer is None]
        return [zone for zone, answer in zip(zones, offered, strict=True) if answer]

    # ------------------------------------------------------------------ the race

    def write_receipt(self) -> None:
        """Atomically rewrite the receipt. Zone threads call this concurrently, so it is serialized."""

        if self.receipt is None:
            return
        with self._receipt_lock:
            self.receipt.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.report.document(), indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.receipt)

    def _claim(self, result: ZoneResult) -> bool:
        with self._lock:
            if not self.keep_first or self.report.interrupted or self.report.kept is not None:
                return False
            result.kept = True
            return True

    def _note_state(self, result: ZoneResult, started: float, state: str) -> None:
        if not result.states or result.states[-1][1] != state:
            result.states.append((round(self.clock() - started, 1), state))
            self.on_event(result.zone, state.lower())

    def _race_zone(self, index: int, result: ZoneResult) -> None:
        created = False
        started = self.clock()
        try:
            if self.stop.wait(index * self.timing.stagger_s):
                result.outcome = RaceOutcome.CANCELLED
                return
            body = request_body(
                project=self.project,
                zone=result.zone,
                name=result.name,
                accelerator_type=self.accelerator_type,
                runtime_version=self.runtime_version,
                spot=self.spot,
                run_id=self.run_id,
                enable_external_ips=self.enable_external_ips,
            )
            started = self.clock()
            status, document = self.api.create(self.project, result.zone, result.name, body)
            if status == 409:
                created = True  # a retried create whose first attempt landed
                document = {"done": True}
            elif status >= 300:
                result.outcome = RaceOutcome.REJECTED
                result.error, result.error_code = error_text(document) or f"http {status}", error_code(document)
                self.on_event(result.zone, f"rejected: {result.error}")
                return
            created = True
            operation = document.get("name")
            # An accepted create is not an accepted request: queue refusals arrive
            # on the operation, after the create call already returned success.
            while isinstance(operation, str) and not document.get("done"):
                if self.clock() - started >= self.timing.timeout_s or self.stop.wait(self.timing.operation_poll_s):
                    result.outcome = RaceOutcome.CANCELLED if self.stop.is_set() else RaceOutcome.WAITING
                    return
                op_status, document = self.api.operation(operation)
                if op_status >= 300:
                    if op_status in _RETRYABLE or op_status == 599:
                        document = {}
                        continue
                    result.outcome = RaceOutcome.ERROR
                    result.error = error_text(document) or f"operation poll http {op_status}"
                    result.error_code = error_code(document)
                    self.on_event(result.zone, f"operation error: {result.error}")
                    return
            if document.get("error"):
                result.outcome = RaceOutcome.REFUSED
                result.error, result.error_code = error_text(document), error_code(document)
                self.on_event(result.zone, f"refused: {result.error}")
                return
            result.accepted_after_s = round(self.clock() - started, 1)
            while True:
                status, document = self.api.get(self.project, result.zone, result.name)
                if status == 404:
                    state = "NOT_FOUND"
                elif status >= 300:
                    state = f"HTTP_{status}"
                else:
                    state_document = document.get("state")
                    value = state_document.get("state") if is_json_object(state_document) else None
                    state = value if isinstance(value, str) else "UNKNOWN"
                    if state in _ENDED_STATES and is_json_object(state_document):
                        detail = {key: item for key, item in state_document.items() if key != "state"}
                        result.error = json.dumps(detail, sort_keys=True)[:500] if detail else None
                self._note_state(result, started, state)
                if state == "ACTIVE":
                    result.outcome = RaceOutcome.GRANTED
                    result.granted_after_s = round(self.clock() - started, 1)
                    if self._claim(result):
                        self.on_event(result.zone, f"granted after {result.granted_after_s}s; kept")
                        self.stop.set()  # the race is won: the others stand down
                    else:
                        self.on_event(result.zone, f"granted after {result.granted_after_s}s")
                    return
                if state in {"FAILED", "NOT_FOUND"}:
                    result.outcome = RaceOutcome.FAILED
                    return
                if state in _ENDED_STATES:
                    result.outcome = RaceOutcome.SUSPENDED
                    return
                if self.clock() - started >= self.timing.timeout_s:
                    result.outcome = RaceOutcome.WAITING
                    return
                if self.stop.wait(self.timing.poll_s):
                    result.outcome = RaceOutcome.CANCELLED
                    return
        except Exception as exc:  # whatever happened, the request below is still cleaned up
            result.outcome = RaceOutcome.ERROR
            result.error = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            if result.kept:
                result.cleanup = "kept"
            elif created:
                result.cleanup = self._cleaner.delete(result.zone, result.name)
                self.on_event(result.zone, f"cleanup: {result.cleanup}")
            self.write_receipt()

    def run(self) -> RaceReport:
        """Race every candidate zone, then sweep. Ctrl-C stops the race, never the cleanup."""

        zones = self.candidate_zones()
        self.report.zones = [ZoneResult(zone=zone, name=request_name(self.run_id, zone)) for zone in zones]
        self.write_receipt()
        threads = [
            threading.Thread(target=self._race_zone, args=(index, result), name=f"race-{result.zone}")
            for index, result in enumerate(self.report.zones)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            while thread.is_alive():
                try:
                    thread.join(0.2)
                except KeyboardInterrupt:
                    with self._lock:
                        self.report.interrupted = True
                    self.stop.set()
                    self.on_event(
                        "*",
                        f"interrupted; deleting every request first (or later: eyvan race sweep {self.run_id})",
                    )
        kept = self.report.kept
        self.report.sweep = self._cleaner.sweep(zones, self.run_id, keep=() if kept is None else (kept.name,))
        self.report.finished_at = time.time()
        self.write_receipt()
        return self.report


def sweep_race(
    api: TpuQueueApi,
    *,
    project: str,
    zones: tp.Iterable[str],
    run_id: str,
    keep: tp.Collection[str] = (),
    timing: RaceTiming = RaceTiming(),
    on_event: tp.Callable[[str, str], None] = lambda zone, message: None,
) -> SweepResult:
    """Delete whatever race ``run_id`` left in ``zones``; the standalone ``eyvan race sweep``."""

    if not _RUN_ID.match(run_id):
        raise ValueError("a race id is 8 lowercase hex characters")
    return _Cleaner(api, project, timing, time.monotonic, on_event=on_event).sweep(zones, run_id, keep=keep)


def default_receipt(run_id: str) -> Path:
    return RECEIPT_DIRECTORY.expanduser() / f"{run_id}.json"


__all__ = (
    "API",
    "NAME_PREFIX",
    "RACE_LABEL",
    "RaceOutcome",
    "RaceReport",
    "RaceTiming",
    "SweepResult",
    "TpuQueueApi",
    "TpuQueueRest",
    "TpuRace",
    "ZoneResult",
    "default_receipt",
    "gcloud_token",
    "request_body",
    "request_name",
    "sweep_race",
)
