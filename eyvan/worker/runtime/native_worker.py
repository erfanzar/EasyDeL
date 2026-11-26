# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Authenticated HTTP transport for one Eyvan native execution worker.

**Deployment order: native workers first, then agents, then the controller.**
This serves one wire method that did not exist before --
``POST /api/v1/cells/{cell_id}/stage-actions``, the second hop of managed
staging. A worker that predates it answers HTTP 404, which
:class:`~eyvan.worker.runtime.native_remote.NativeRemoteExecutor` reports to its
agent as *this worker has no staging executor*, and which the agent reports on
to the control plane the same way: absence of a capability, never a failed
action. So an agent upgraded ahead of its workers stages nothing and records
nothing, exactly as it does today, and the feature appears once the workers
underneath it serve the route.

The executor behind that route is
:class:`~eyvan.worker.runtime.stage_executor.LocalAssetStageExecutor`, built
from this host's own environment and not a second implementation of it. Every
guarantee it makes therefore survives the hop unchanged: the bounded fetch, the
resume point fsynced before a failure propagates, the size and digest read back
off this disk rather than declared, quarantine on a digest mismatch, and the
source allowlist -- which matters most of all here, because a named asset is
user input and an executor that fetched whatever it was handed would be an
arbitrary-read primitive on a machine holding a fleet's accelerators.

``EYVAN_CELL_STAGE_SOURCES`` is that allowlist and is empty by default, so a
worker that ships with this route still stages nothing until an operator
declares what may be fetched *on the worker host*. On a Compute fleet the
worker unit reads ``/etc/eyvan-native/systemd-runtime.env``, and the fleet's
declarative runtime setup runs after that file is written and before the unit
starts, so a line appended there is how the variable reaches this process.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import typing as tp
from pathlib import Path, PurePosixPath

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from eyvan._internal.build_identity import build_identity
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan._internal.observability import trace_span
from eyvan._internal.trace_context import remote_trace_parent
from eyvan.capacity.drivers.base import ExecutionDriver, ExecutionWorkloadReport, MutationContext, StaleMutationError
from eyvan.records import NativeContainerConfig, NativeRuntimeConfig
from eyvan.worker.agent.models import (
    AgentBuild,
    CellStageAction,
    CellStageActionResult,
    CellStageVerdict,
    CheckpointRequest,
    ExecRequest,
    MutationConflictCode,
    NativeStageActionRequest,
    NativeWorkerHealth,
    WorkloadMutationRequest,
    WorkloadReference,
    WorkloadRequest,
)
from eyvan.worker.runtime.local_process import LocalProcessExecutor
from eyvan.worker.runtime.registry import ExecCapableExecutionDriver, IntrospectableExecutionDriver
from eyvan.worker.runtime.stage_executor import AssetStageDriver, AssetStageVerdict, LocalAssetStageExecutor
from eyvan.worker.runtime.systemd_transient import SystemdLogsUnavailableError, SystemdTransientServiceExecutor

logger = logging.getLogger(__name__)

#: Where this worker keeps staged bytes when the operator names no directory.
#:
#: Under the workload's own directory rather than under the worker's state
#: directory, because the point of staging here is that the *workload* can read
#: the bytes. ``EYVAN_NATIVE_WORKLOAD_DIRECTORY`` is the tree the workload user
#: owns and runs in; the state directory is installed ``0700`` for root and a
#: dataset written there would be invisible to the job that asked for it.
_DEFAULT_WORKLOAD_DIRECTORY = "/var/lib/eyvan/workload"

#: What one staging verdict is called on the wire out of this worker.
_STAGE_VERDICTS: dict[AssetStageVerdict, CellStageVerdict] = {
    AssetStageVerdict.OK: CellStageVerdict.OK,
    AssetStageVerdict.REFUSED: CellStageVerdict.REFUSED,
    AssetStageVerdict.INCONCLUSIVE: CellStageVerdict.INCONCLUSIVE,
}


def _context(request: WorkloadRequest | WorkloadMutationRequest | CheckpointRequest) -> MutationContext:
    return MutationContext(**request.mutation.model_dump())


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _runtime_environment(value: tp.Mapping[str, str] | None) -> dict[str, str]:
    """Validate an operator-owned default environment for native workloads."""

    if value is None:
        return {}
    if not isinstance(value, tp.Mapping):
        raise ValueError("native worker runtime environment must be a mapping")
    try:
        return NativeRuntimeConfig(env=dict(value)).env
    except Exception as exc:
        raise ValueError(f"native worker runtime environment is invalid: {exc}") from exc


def _load_runtime_environment(path: Path) -> dict[str, str]:
    """Read the JSON environment emitted by the native bootstrap script."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read native worker runtime environment file: {exc}") from exc
    return _runtime_environment(value)


def _container_artifact_mounts(value: JsonValue) -> tuple[str, ...]:
    """Validate the staged host paths handed to the container executor."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("native worker container artifact mounts must be a list")
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


def _load_container_runtime(path: Path) -> tuple[NativeContainerConfig, tuple[str, ...]]:
    """Load the typed OCI configuration emitted by native capacity bootstrap."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read native worker container runtime file: {exc}") from exc
    if not is_json_object(value) or set(value) != {"artifact_mounts", "container"}:
        raise ValueError("native worker container runtime file must contain container and artifact_mounts")
    try:
        container = NativeContainerConfig.model_validate(value["container"])
    except Exception as exc:
        raise ValueError(f"native worker container runtime is invalid: {exc}") from exc
    return container, _container_artifact_mounts(value["artifact_mounts"])


def _base_environment(runtime_env: tp.Mapping[str, str]) -> dict[str, str]:
    """Return workload defaults injected below job-specific environment values."""

    workload_directory = os.getenv("EYVAN_NATIVE_WORKLOAD_DIRECTORY", _DEFAULT_WORKLOAD_DIRECTORY)
    workload_user = os.getenv("EYVAN_NATIVE_WORKLOAD_USER", "eyvan-workload")
    return {
        # libtpu logs to /tmp/tpu_logs unless told otherwise. On a shared or
        # attached TPU VM another account usually created that directory first
        # (mode 0700), and the workload user then gets two "Could not open the
        # log file ... Permission denied" lines on every libtpu log call -- tens
        # of thousands per job. The workload directory is the tree that user
        # owns. Lowest precedence: the runtime env, then the job, override it.
        "TPU_LOG_DIR": os.getenv("TPU_LOG_DIR", f"{workload_directory}/tpu_logs"),
        **runtime_env,
        "HOME": workload_directory,
        "LOGNAME": workload_user,
        "PATH": os.getenv(
            "PATH",
            "/opt/eyvan-native/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        ),
        "USER": workload_user,
        "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV", "/opt/eyvan-native/venv"),
        "XDG_CACHE_HOME": os.getenv("XDG_CACHE_HOME", f"{workload_directory}/.cache"),
    }


def _default_stager() -> AssetStageDriver | None:
    """Build this host's staging executor, or ``None`` when it declares no sources.

    An empty allowlist is the off switch, and it has to read as *no executor*
    rather than as a refusal of every asset. A refusal is terminal -- it blocks
    the stage permanently, and nothing unblocks it short of replacing the cell --
    so an operator who sized a cache before declaring what may be fetched would
    poison the ledger for every asset in the fleet and have no way back.
    """

    workload_directory = os.getenv("EYVAN_NATIVE_WORKLOAD_DIRECTORY", _DEFAULT_WORKLOAD_DIRECTORY)
    candidate = LocalAssetStageExecutor.from_environment(Path(workload_directory) / "stages")
    return candidate if candidate.allowed_prefixes else None


def create_native_worker_app(
    executor: ExecutionDriver | None = None,
    *,
    state_dir: str | os.PathLike[str] | None = None,
    secret: str | None = None,
    correlation_enabled: bool = False,
    systemd_transient: bool = False,
    systemd_exit_type_cgroup: bool = False,
    runtime_env: tp.Mapping[str, str] | None = None,
    runtime_container: NativeContainerConfig | None = None,
    container_artifact_mounts: tp.Sequence[str] = (),
    stager: AssetStageDriver | None = None,
) -> FastAPI:
    """Create one native execution-worker application.

    Args:
        executor: Worker-side execution driver. When omitted, a durable local
            process executor is constructed from ``state_dir`` unless
            ``systemd_transient`` selects the production systemd executor.
        state_dir: Durable execution state directory used by the default
            executor. It cannot be combined with an explicit ``executor``.
        secret: Bearer credential required by every execution API. The
            liveness endpoint remains unauthenticated.
        correlation_enabled: Join bounded trace parents after authentication
            and emit request spans when an OpenTelemetry SDK is configured.
            Disabled by default; never changes authorization or fencing.
        systemd_transient: Run workloads as cgroup-contained transient systemd
            services. Compute VM bootstrap always enables this mode; the local
            executor remains available for development and in-process tests.
        systemd_exit_type_cgroup: Use systemd 250's ``ExitType=cgroup``
            behavior. The v5p Ubuntu 22.04 image uses cgroup observation
            instead, so it does not need this newer systemd feature.
        runtime_env: Fleet-level workload environment merged below each job's
            explicit environment. It is supported by the production systemd
            executor only.
        runtime_container: Structured OCI runtime settings selected from the
            fleet YAML. It requires the production systemd worker path.
        container_artifact_mounts: Staged host artifact paths automatically
            made read-only at the same paths inside the container.
        stager: Managed-staging executor for the cells whose disk this worker
            owns. Built from this host's environment when omitted, and left
            unset when that environment declares no source prefixes -- which is
            the default, and which the staging route reports as *no executor*.
            Independent of ``executor``: staging writes bytes for a workload
            rather than running one, so a deployment may select one without
            selecting the other.

    Returns:
        The configured FastAPI application.

    Raises:
        ValueError: If explicit executor selection conflicts with default
            executor options.
    """

    if executor is not None and (
        state_dir is not None
        or systemd_transient
        or runtime_env is not None
        or runtime_container is not None
        or container_artifact_mounts
    ):
        raise ValueError("default native worker options cannot be combined with an explicit executor")
    if not isinstance(secret, str) or not secret.strip():
        raise ValueError("native worker requires a non-empty bearer secret")
    if not isinstance(correlation_enabled, bool):
        raise ValueError("native worker correlation setting must be a boolean")
    if not isinstance(systemd_exit_type_cgroup, bool):
        raise ValueError("systemd ExitType cgroup setting must be a boolean")
    runtime_env = _runtime_environment(runtime_env)
    container_artifact_mounts = _container_artifact_mounts(list(container_artifact_mounts))
    if runtime_container is not None and not isinstance(runtime_container, NativeContainerConfig):
        raise ValueError("native worker container runtime must be a NativeContainerConfig")
    if runtime_env and not systemd_transient:
        raise ValueError("native worker runtime environment requires systemd_transient")
    if runtime_container is not None and not systemd_transient:
        raise ValueError("native worker container runtime requires systemd_transient")
    if runtime_container is None and container_artifact_mounts:
        raise ValueError("native worker container artifact mounts require a container runtime")
    if executor is None:
        base_environment = _base_environment(runtime_env)
        if runtime_container is not None:
            from eyvan.worker.runtime.container import DockerContainerExecutor

            executor = DockerContainerExecutor(
                state_dir=state_dir,
                container=runtime_container,
                artifact_mounts=container_artifact_mounts,
                use_exit_type_cgroup=systemd_exit_type_cgroup,
                workload_directory=os.getenv("EYVAN_NATIVE_WORKLOAD_DIRECTORY", "/var/lib/eyvan/workload"),
                base_environment=base_environment,
            )
        elif systemd_transient:
            executor = SystemdTransientServiceExecutor(
                state_dir=state_dir,
                use_exit_type_cgroup=systemd_exit_type_cgroup,
                workload_user=os.getenv("EYVAN_NATIVE_WORKLOAD_USER", "eyvan-workload"),
                workload_group=os.getenv("EYVAN_NATIVE_WORKLOAD_GROUP", "eyvan-workload"),
                workload_directory=os.getenv("EYVAN_NATIVE_WORKLOAD_DIRECTORY", "/var/lib/eyvan/workload"),
                base_environment=base_environment,
            )
        else:
            executor = LocalProcessExecutor(state_dir=state_dir)
    stager = _default_stager() if stager is None else stager
    app = FastAPI(title="Eyvan Native Worker", version="1.0.0", docs_url=None, redoc_url=None)
    app.state.executor = executor
    app.state.stager = stager

    def authenticate_credentials(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {secret}"
        # compare_digest raises TypeError for non-ASCII str, and the header is
        # caller-controlled: unauthenticated garbage must answer 401, never a
        # 500.
        try:
            recognized = authorization is not None and hmac.compare_digest(authorization, expected)
        except TypeError:
            recognized = False
        if not recognized:
            raise HTTPException(status_code=401, detail="invalid native worker credential")

    async def authenticate(
        request: Request, _authenticated: None = Depends(authenticate_credentials)
    ) -> tp.AsyncIterator[None]:
        if not correlation_enabled:
            yield
            return
        with (
            remote_trace_parent(request.headers.get("traceparent"), enabled=True),
            trace_span(
                "eyvan.native_worker_request",
                **{"http.request.method": request.method, "http.route": request.scope["route"].path},
            ),
        ):
            yield

    @app.exception_handler(StaleMutationError)
    async def stale_mutation_handler(_request: Request, exc: StaleMutationError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "code": MutationConflictCode.STALE_MUTATION.value},
        )

    @app.exception_handler(SystemdLogsUnavailableError)
    async def logs_unavailable_handler(_request: Request, exc: SystemdLogsUnavailableError) -> JSONResponse:
        """Report a journal transport failure as a typed 503, not a 500.

        A log read that cannot reach journalctl is a transient dependency
        failure, and it must stay distinguishable from a genuinely silent
        workload -- returning an empty body would look like success.
        """

        return JSONResponse(status_code=503, content={"detail": str(exc), "code": "logs_unavailable"})

    @app.get("/healthz")
    def health() -> NativeWorkerHealth:
        return NativeWorkerHealth(status="ok", runtime="native-worker")

    @app.get("/api/v1/health", dependencies=[Depends(authenticate)])
    def authenticated_health() -> NativeWorkerHealth:
        return NativeWorkerHealth(status="ok", runtime="native-worker")

    @app.get("/api/v1/build", dependencies=[Depends(authenticate)])
    def build() -> AgentBuild:
        """Report the code this worker is running.

        The last component of this system that could not say. It is the process
        that actually executes the workload, and it is installed from whatever
        wheel its cell was provisioned with -- which is not necessarily the one
        the head stages today, because a cell outlives a re-staged head. That
        happened while this was being written: the head's wheel was replaced
        twice while running TPU workers kept their original one.
        """

        identity = build_identity()
        return AgentBuild(
            version=identity.version,
            origin=identity.origin,
            origin_detail=identity.origin_detail,
            source_digest=identity.source_digest,
            digest_detail=identity.digest_detail,
            file_count=identity.file_count,
            python=identity.python,
            platform=identity.platform,
        )

    @app.post("/api/v1/cells/{cell_id}/stage-actions", dependencies=[Depends(authenticate)])
    def stage_asset(cell_id: str, body: NativeStageActionRequest) -> CellStageActionResult:
        """Measure, materialize or remove one named asset on this worker's disk.

        The second hop of managed staging, and the reason a real fleet can stage
        at all: on ``eyvan-native`` the workload runs here rather than beside the
        agent, so this is the only process whose disk a stage row can honestly
        describe. The agent has already fenced the request against its own
        durable record of the cell before it arrives.

        A 501 when this host declares no source prefixes, never a refusal. The
        agent reports it as *no executor*, so nothing is staged, recorded or
        charged -- which is also the answer a worker that predates the route
        gives, by answering 404. A refusal would instead block every asset in
        the fleet permanently over one unset environment variable.
        """

        if stager is None:
            raise HTTPException(
                status_code=501,
                detail=(
                    "this native worker declares no managed staging sources; "
                    "set EYVAN_CELL_STAGE_SOURCES on the worker host to the prefixes this fleet may fetch"
                ),
            )
        if body.action is CellStageAction.MEASURE:
            result = stager.measure(cell_id, body.asset, max_bytes=body.max_bytes)
        elif body.action is CellStageAction.STAGE:
            result = stager.stage(cell_id, body.asset, max_bytes=body.max_bytes)
        else:
            result = stager.evict(cell_id, body.asset)
        return CellStageActionResult(
            asset=body.asset,
            action=body.action,
            verdict=_STAGE_VERDICTS[result.verdict],
            # An eviction frees bytes rather than placing them, and reporting
            # what it removed as bytes now present would charge the budget for a
            # disk that was just emptied.
            size_bytes=0 if body.action is CellStageAction.EVICT else result.size_bytes,
            digest=None if body.action is CellStageAction.EVICT else result.digest,
            detail=result.detail[:2048],
            path=result.path,
        )

    @app.exception_handler(ValueError)
    async def invalid_request_handler(_request: Request, exc: ValueError) -> JSONResponse:
        """Refuse a request the executor rejected as invalid, rather than crash on it.

        The executor validates a launch by raising ``ValueError`` ("... must not
        contain a newline"). An unhandled 500 would be read as a transient fault,
        retrying a dispatch the worker can never accept. Use the same status and
        error code as the cell agent for the same case.
        """

        return JSONResponse(status_code=400, content={"detail": str(exc), "code": "invalid_configuration"})

    @app.post("/api/v1/workloads", dependencies=[Depends(authenticate)])
    def submit(body: WorkloadRequest) -> WorkloadReference:
        metadata = body.spec.get("metadata")
        workload_id = metadata.get("eyvan_workload_id") if isinstance(metadata, dict) else None
        if not isinstance(workload_id, str) or not workload_id.strip():
            raise HTTPException(status_code=422, detail="native worker workload metadata requires eyvan_workload_id")
        remote_id = executor.submit(body.spec, _context(body))
        return WorkloadReference(remote_id=remote_id)

    @app.get("/api/v1/workloads/{remote_id}", dependencies=[Depends(authenticate)])
    def status(remote_id: str) -> ExecutionWorkloadReport:
        return executor.status(remote_id)

    @app.post("/api/v1/workloads/{remote_id}/stop", dependencies=[Depends(authenticate)])
    def stop(remote_id: str, body: WorkloadMutationRequest) -> ExecutionWorkloadReport:
        executor.stop(remote_id, _context(body))
        return executor.status(remote_id)

    @app.post("/api/v1/workloads/{remote_id}/retire", dependencies=[Depends(authenticate)])
    def retire(remote_id: str, body: WorkloadMutationRequest) -> ExecutionWorkloadReport:
        executor.retire(remote_id, _context(body))
        return executor.status(remote_id)

    @app.post("/api/v1/workloads/{remote_id}/checkpoint", dependencies=[Depends(authenticate)])
    def checkpoint(remote_id: str, body: CheckpointRequest) -> WorkloadReference:
        checkpoint_id = executor.checkpoint(
            remote_id,
            hook=body.hook,
            output_uri=body.output_uri,
            context=_context(body),
        )
        return WorkloadReference(remote_id=checkpoint_id)

    @app.get(
        "/api/v1/workloads/{remote_id}/logs",
        response_class=PlainTextResponse,
        dependencies=[Depends(authenticate)],
    )
    def logs(remote_id: str) -> str:
        return executor.logs(remote_id)

    @app.get(
        "/api/v1/workloads/{remote_id}/log-page",
        dependencies=[Depends(authenticate)],
        response_model=None,
    )
    def log_page(remote_id: str, after: str = "", limit: int = 200) -> JsonObject:
        """Return one resumable page of structured records.

        The executor on this host has produced these since Phase 4 -- the
        systemd runtime parses journald's own JSON, with priority, timestamp,
        PID and an opaque cursor. Nothing exposed them over the wire, so the
        control plane could only fetch `logs()`, which is those records with the
        fields thrown away, and `eyvan workload log-page` answered
        `unsupported_capability` for every fleet that runs on a remote worker.

        A 501 when the executor genuinely cannot produce pages, never an empty
        one: "this backend has no structured logs" and "this workload produced
        no output" are different answers, and a follower that cannot tell them
        apart polls forever against a backend that will never answer.
        """

        page = getattr(executor, "log_page", None)
        if page is None:
            raise HTTPException(
                status_code=501,
                detail=f"execution backend {type(executor).__name__} does not expose structured logs",
            )
        result = page(remote_id, after=after, limit=limit)
        return {
            "records": [dict(record.as_json()) for record in result.records],
            "next_cursor": result.next_cursor,
            "unavailable": result.unavailable,
            "detail": result.detail,
            "truncated_ranks": list(result.truncated_ranks),
        }

    @app.get(
        "/api/v1/workloads/{remote_id}/process-status",
        dependencies=[Depends(authenticate)],
        response_model=None,
    )
    def process_status(remote_id: str) -> JsonObject:
        """Return live process detail for debugging a stalled workload."""

        # ``isinstance`` against the declared protocol, not a getattr probe:
        # ``runtime/registry`` already establishes that seam for close and health,
        # and a third ad-hoc probe would make the pattern look optional.
        if not isinstance(executor, IntrospectableExecutionDriver):
            raise HTTPException(
                status_code=501,
                detail="this native execution backend does not expose process introspection",
            )
        return executor.process_status(remote_id)

    @app.post("/api/v1/workloads/{remote_id}/exec", dependencies=[Depends(authenticate)], response_model=None)
    def exec_in_workload(remote_id: str, body: ExecRequest) -> JsonObject:
        """Run one diagnostic command beside a workload on this host.

        The route a multi-host slice had no way to call. ``exec_in_workload``
        was implemented only on the local systemd driver, so a single-host cell
        could exec and a TPU slice -- reached over this API -- could not, which
        is the half of P3 that stayed unqualified.

        Same protocol check and same 501 as process introspection beside it: a
        backend that cannot do this says so, rather than failing somewhere that
        mentions none of it.
        """

        if not isinstance(executor, ExecCapableExecutionDriver):
            raise HTTPException(
                status_code=501,
                detail="this native execution backend cannot exec beside a workload",
            )
        return executor.exec_in_workload(remote_id, list(body.command), timeout_s=body.timeout_s)

    return app


class NativeWorkerArguments(argparse.Namespace):
    """Typed command-line arguments accepted by the native worker process."""

    host: str
    port: int
    state_dir: str | None
    secret: str | None
    secret_file: Path | None
    runtime_env_file: Path | None
    container_runtime_file: Path | None
    systemd_transient: bool
    mtls_mode: str
    mtls_trust_bundle: str
    mtls_certificate: str
    mtls_private_key: str


def main(argv: tp.Sequence[str] | None = None) -> int:
    """Run a native worker HTTP server."""

    parser = argparse.ArgumentParser(description="Run an Eyvan native execution worker")
    parser.add_argument("--host", default=os.getenv("EYVAN_NATIVE_WORKER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("EYVAN_NATIVE_WORKER_PORT", "8100")))
    parser.add_argument("--state-dir", default=os.getenv("EYVAN_NATIVE_WORKER_STATE_DIR"))
    parser.add_argument("--secret", default=os.getenv("EYVAN_NATIVE_WORKER_SECRET"))
    parser.add_argument("--secret-file", type=Path)
    parser.add_argument("--runtime-env-file", type=Path)
    parser.add_argument("--container-runtime-file", type=Path)
    parser.add_argument("--mtls-mode", default=os.getenv("EYVAN_MUTUAL_TLS_MODE", "off"))
    parser.add_argument("--mtls-trust-bundle", default=os.getenv("EYVAN_MUTUAL_TLS_TRUST_BUNDLE", ""))
    parser.add_argument("--mtls-certificate", default=os.getenv("EYVAN_MUTUAL_TLS_CERTIFICATE", ""))
    parser.add_argument("--mtls-private-key", default=os.getenv("EYVAN_MUTUAL_TLS_PRIVATE_KEY", ""))
    parser.add_argument(
        "--systemd-transient",
        action="store_true",
        help="execute workloads as cgroup-contained transient systemd services",
    )
    args = parser.parse_args(list(argv) if argv is not None else None, namespace=NativeWorkerArguments())
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.secret is not None and args.secret_file is not None:
        parser.error("set only one of --secret or --secret-file")
    from eyvan.identity.mtls import MutualTlsFiles, parse_mutual_tls_mode, server_ssl_context, server_ssl_options

    try:
        mode = parse_mutual_tls_mode(args.mtls_mode)
        files = MutualTlsFiles(args.mtls_trust_bundle, args.mtls_certificate, args.mtls_private_key)
        # Validate before reading/deleting a one-use bearer-secret file.
        server_ssl_context(mode, files)
        ssl_options = server_ssl_options(mode, files)
        systemd_exit_type_cgroup = _environment_bool("EYVAN_SYSTEMD_EXIT_TYPE_CGROUP", default=False)
        correlation_enabled = _environment_bool("EYVAN_CORRELATION_ENABLED", default=False)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    secret = args.secret
    # The secret must not be inherited by workload child processes.
    os.environ.pop("EYVAN_NATIVE_WORKER_SECRET", None)
    runtime_env: dict[str, str] = {}
    if args.runtime_env_file is not None:
        try:
            runtime_env = _load_runtime_environment(args.runtime_env_file)
        except ValueError as exc:
            parser.error(str(exc))
    runtime_container: NativeContainerConfig | None = None
    container_artifact_mounts: tuple[str, ...] = ()
    if args.container_runtime_file is not None:
        try:
            runtime_container, container_artifact_mounts = _load_container_runtime(args.container_runtime_file)
        except ValueError as exc:
            parser.error(str(exc))
    if args.secret_file is not None:
        try:
            secret = args.secret_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"could not read --secret-file: {exc}")
        try:
            args.secret_file.unlink()
        except OSError as exc:
            logger.warning("could not remove native worker secret file %s: %s", args.secret_file, exc)
    if not isinstance(secret, str) or not secret.strip():
        # Named here rather than left to the application factory, which raises a
        # ValueError and reaches an operator as a traceback out of a systemd unit
        # that failed to start. Every other missing input on this path is a
        # parser.error naming the flag that supplies it; the credential is the
        # one an operator is most likely to omit, so it says the same thing.
        parser.error(
            "a native worker requires a bearer credential: pass --secret, --secret-file, "
            "or set EYVAN_NATIVE_WORKER_SECRET"
        )

    import uvicorn

    # Imported here, beside uvicorn itself, so importing this module stays free
    # of the server dependency it only needs once it is actually serving.
    from eyvan._internal.access_logging import uvicorn_log_config

    uvicorn.run(
        create_native_worker_app(
            state_dir=args.state_dir,
            secret=secret,
            systemd_transient=args.systemd_transient,
            systemd_exit_type_cgroup=systemd_exit_type_cgroup,
            runtime_env=runtime_env if args.runtime_env_file is not None else None,
            runtime_container=runtime_container,
            container_artifact_mounts=container_artifact_mounts,
            **({"correlation_enabled": True} if correlation_enabled else {}),
        ),
        host=args.host,
        port=args.port,
        # Uvicorn's own graph, with the access formatter that keeps a log-page
        # request's ``?after=`` cursor out of this host's access log.
        log_config=uvicorn_log_config(),
        **ssl_options,
    )
    return 0


__all__ = ("create_native_worker_app", "main")


if __name__ == "__main__":
    raise SystemExit(main())
