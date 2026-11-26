# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ray-like callable submission primitives backed only by native Eyvan APIs."""

from __future__ import annotations

import functools
import hashlib
import hmac
import inspect
import os
import threading
import typing as tp
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from eyvan._internal.json_types import model_json
from eyvan._internal.workspace import WorkspaceError, build_workspace
from eyvan.artifacts import (
    ArtifactStore,
    EyvanManagedArtifactStore,
    GCSArtifactStore,
    artifact_store_from_uri,
)
from eyvan.capacity.providers.gcp.runtime.profile import (
    GcpNativeRuntimeProfile,
    GcpNativeRuntimeProfileError,
    GcpNativeRuntimeProfileStore,
)
from eyvan.capacity.providers.gcp.runtime.supervisor import ensure_gcp_native_runtime, runtime_profile_path_for
from eyvan.records import CapacityClass
from eyvan.sdk.client.callables import EyvanCallableSubmitter
from eyvan.sdk.client.easydel_setup import EasyDeLSetup
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.identity import resolve_control_plane_identity
from eyvan.sdk.client.jobs import (
    AcceleratorConfig,
    CallableSubmission,
    CallableSubmitter,
    CpuConfig,
    EyvanJob,
    EyvanJobError,
    EyvanPlacementError,
    GpuConfig,
    JobBackend,
    JobCompletion,
    SetupScript,
    TpuConfig,
    reject_remote_callable_signing_key,
)
from eyvan.sdk.client.transport import EyvanHTTPError, EyvanUnavailableError
from eyvan.sdk.contexts import (
    DEFAULT_BASE_URL,
    DEFAULT_ORG_ID,
    DEFAULT_PROJECT_ID,
    context_config_path,
    get_active_context,
)
from eyvan.sdk.node import discover_node_context, discover_node_token_or_warn
from eyvan.sdk.team_credentials import team_token_or_warn
from eyvan.server.local.daemon import ensure_loopback_control_plane
from eyvan.worker.runtime.callables import dump_callable
from eyvan.worker.runtime.in_job import in_job_context

P = tp.ParamSpec("P")
R = tp.TypeVar("R")
R_co = tp.TypeVar("R_co", covariant=True)


@dataclass(frozen=True, slots=True)
class ExecutionContext(tp.Generic[R_co]):
    """Process-wide defaults used by decorated Eyvan callables.

    Attributes:
        base_url: The control plane this process resolved. Carried because
            ``init`` picks it from six places -- argument, ``EYVAN_URL``, the
            saved context, this node's own plane, then the loopback default --
            and the returned context was previously unable to say which one won.
            Somebody with a laptop plane and a head has two, and "which am I
            submitting to" is the first question they ask.
    """

    fleet: str | None
    org_id: str
    project_id: str
    artifact_store: ArtifactStore
    submitter: CallableSubmitter[R_co] | None = None
    signing_key: bytes | None = field(default=None, repr=False)
    base_url: str | None = None


_context: ExecutionContext[tp.NoReturn] | None = None
_context_lock = threading.RLock()


def _required(value: str | None, *, argument: str, env_name: str) -> str:
    normalized = value.strip() if value else ""
    if not normalized:
        raise ValueError(f"Eyvan init requires {argument} or {env_name}")
    return normalized


def _optional(value: str | None) -> str | None:
    """Return a trimmed string or None when empty."""

    normalized = value.strip() if value else ""
    return normalized if normalized else None


def _runtime_profile_store(endpoint: ControlPlaneEndpoint) -> GcpNativeRuntimeProfileStore:
    """Return the profile the loopback plane at ``endpoint`` serves, else the default one.

    A plane started by ``eyvan runtime deploy --profile PATH`` records that
    path; following the record is what lets ``init()`` use that plane instead
    of refusing it for not serving the default profile.
    """

    return GcpNativeRuntimeProfileStore(runtime_profile_path_for(endpoint))


def _configured_gcp_runtime_profile(endpoint: ControlPlaneEndpoint) -> GcpNativeRuntimeProfile | None:
    """Load the private GCP profile selected by a loopback execution context, if any."""

    if not endpoint.is_loopback:
        return None
    try:
        return _runtime_profile_store(endpoint).load()
    except FileNotFoundError:
        return None
    except GcpNativeRuntimeProfileError as exc:
        raise RuntimeError(f"Eyvan native GCP runtime profile is invalid: {exc}") from exc


def _prepared_gcp_runtime_profile(
    endpoint: ControlPlaneEndpoint,
    profile: GcpNativeRuntimeProfile | None,
) -> GcpNativeRuntimeProfile | None:
    """Start the private GCP runtime only after its owner has completed one-time setup."""

    if profile is None:
        return None
    return ensure_gcp_native_runtime(endpoint, profile_path=_runtime_profile_store(endpoint).path).profile


def _ensure_fleet_applied(client: tp.Any, fleet: str | None, cache_dir: Path | None) -> None:
    """Bring a named fleet up if it is not already accepting work.

    Submitting is meant to be the whole workflow. Without this the operator runs
    ``eyvan autoscale generate``, then ``eyvan up --select``, then edits their
    script, and any one of those forgotten produces a failure naming none of
    them -- a placement refusal, or a callable rejected for an interpreter the
    stale config still declares. Ray is ``ray job submit``; a second command
    here is a defect, not a prerequisite.

    Only ever *adds* capacity a generated config already describes. A fleet
    already serving is left exactly as it is, and one with no cached config is
    left to the placement error, which names what does exist.

    Failures are narrow on purpose. An earlier version caught every exception
    around the read, which silently swallowed an ``AttributeError`` from calling
    a method the client does not have -- so this did nothing at all and said so
    nowhere.
    """

    if not fleet:
        return
    transport = getattr(client, "_transport", None)
    if transport is None:
        return
    try:
        response = transport.request_checked("GET", f"/api/v1/fleets/{fleet}")
        applied = response.json()
    except EyvanHTTPError:
        # No such fleet, or this caller may not read it. Submission reports both
        # far better than a setup helper can.
        return
    pools = applied.get("pools") if isinstance(applied, dict) else None
    if not isinstance(pools, list):
        return
    if any(isinstance(pool, dict) and pool.get("enabled") for pool in pools):
        return

    from eyvan.capacity.autoscale.cache import GeneratedAutoscaleCache
    from eyvan.sdk.commands.fleets import FleetCommandService

    root = cache_dir if cache_dir is not None else Path("~/.eyvan/autoscale").expanduser()
    cached = GeneratedAutoscaleCache(root).get(fleet)
    if cached is None:
        return
    request = FleetCommandService.apply_request(cached.config)
    transport.mutate(
        "PUT",
        f"/api/v1/fleets/{fleet}",
        json=model_json(request),
        idempotency_key=f"init:{fleet}:{uuid.uuid4().hex}",
    )


def _stderr_note(message: str) -> None:
    import sys

    print(f"Eyvan: {message}", file=sys.stderr, flush=True)


def _fleet_exists(transport: tp.Any, fleet: str) -> bool | None:
    """Return whether the plane has fleet ``fleet``; ``None`` when it could not say."""

    try:
        transport.request_checked("GET", f"/api/v1/fleets/{fleet}")
    except EyvanHTTPError as error:
        return False if error.status_code == 404 else None
    except EyvanUnavailableError:
        return None
    return True


def _run_connect(base_url: str, project: str, zone: str, name: str) -> int:
    """Run ``eyvan gcp connect`` for one discovered TPU, its progress shown on stderr."""

    import subprocess
    import sys

    from eyvan._internal.subprocess_env import provider_cli_env

    argv = [
        sys.executable,
        "-m",
        "eyvan.sdk.cli",
        "--url",
        base_url,
        "gcp",
        "connect",
        "--tpu",
        name,
        "--project",
        project,
        "--zone",
        zone,
    ]
    try:
        stdout: tp.Any = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        stdout = None
    return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stdout, env=provider_cli_env(), check=False).returncode


def _connect_discovered_tpu(
    client: tp.Any,
    fleet: str | None,
    base_url: str,
    *,
    index: tp.Any = None,
    connect: tp.Callable[[str, str, str, str], int] = _run_connect,
    note: tp.Callable[[str], None] | None = None,
) -> str | None:
    """Return the fleet to submit to, connecting a discovered TPU on its first use.

    ``fleet`` may be a discovered TPU's fleet name or its plain node name. Only
    a name the plane does not know is looked up, so a fleet that already exists
    is never re-examined and a typo costs one read. A TPU that was discovered
    (read-only, ``eyvan gcp sync``) but never connected is connected here with
    the same command a person would run, so its failures name the same steps.
    """

    if not fleet:
        return fleet
    transport = getattr(client, "_transport", None)
    if transport is None or _fleet_exists(transport, fleet) is not False:
        return fleet
    from eyvan.capacity.providers.gcp.tpu_discovery import DiscoveryIndex

    say = note if note is not None else _stderr_note
    index = index if index is not None else DiscoveryIndex()
    snapshot = index.load()
    tpu = snapshot.resolve(fleet) if snapshot is not None else None
    if tpu is None and not (snapshot is not None and snapshot.ambiguous(fleet)):
        try:
            snapshot = index.refresh()
        except Exception as exc:  # discovery is best effort: the placement error still names what exists
            say(f"could not list your GCP TPUs to look up {fleet!r}: {exc}")
            return fleet
        tpu = snapshot.resolve(fleet)
    if tpu is None:
        clashes = snapshot.ambiguous(fleet) if snapshot is not None else ()
        if clashes:
            names = ", ".join(f"{t.fleet} ({t.project}/{t.zone})" for t in clashes)
            raise ValueError(f"TPU name {fleet!r} exists in more than one place; use its fleet name: {names}")
        return fleet
    if tpu.fleet != fleet and _fleet_exists(transport, tpu.fleet):
        say(f"{fleet} is TPU {tpu.project}/{tpu.zone}/{tpu.name}; using its fleet {tpu.fleet}")
        return tpu.fleet
    say(
        f"{fleet} is TPU {tpu.project}/{tpu.zone}/{tpu.name} ({tpu.accelerator_type}), not connected yet; "
        "connecting it now (one time)"
    )
    code = connect(base_url, tpu.project, tpu.zone, tpu.name)
    if code != 0:
        raise RuntimeError(
            f"could not connect TPU {tpu.project}/{tpu.zone}/{tpu.name} (see the failed step above); "
            f"retry with: eyvan gcp connect --tpu {tpu.name} --project {tpu.project} --zone {tpu.zone}"
        )
    return tpu.fleet


def _validate_native_gcp_artifact_override(
    artifact_uri: str | None,
    artifact_store: ArtifactStore | None,
) -> None:
    """Require callable inputs and results to remain reachable from native GCP workers."""

    if artifact_uri is not None and not artifact_uri.startswith("gs://"):
        raise ValueError(
            "the local native GCP runtime requires GCS-backed callable artifacts; omit artifact_uri to use its "
            "configured GCS root, or pass a gs:// URI"
        )
    if artifact_store is not None and not isinstance(artifact_store, GCSArtifactStore):
        raise ValueError(
            "the local native GCP runtime requires a GCSArtifactStore; omit artifact_store to use its configured "
            "GCS root, or pass a GCSArtifactStore"
        )


def init(
    address: str | None = None,
    /,
    *,
    fleet: str | None = None,
    org_id: str | None = None,
    project_id: str | None = None,
    artifact_uri: str | None = None,
    artifact_store: ArtifactStore | None = None,
    submitter: CallableSubmitter[tp.NoReturn] | None = None,
    signing_key: bytes | str | None = None,
    base_url: str | None = None,
    token: str | None = None,
    dev_user: str | None = None,
    environ: tp.Mapping[str, str] | None = None,
) -> ExecutionContext[tp.NoReturn]:
    """Connect this process to a control plane, and return what it connected to.

    Called for you the first time a decorated callable is submitted, so a script
    that is happy with the defaults never has to name it. Call it explicitly to
    choose a plane, to pin a fleet, or simply to find out which plane a bare
    ``@execute`` would have used -- the returned context carries the resolved
    ``base_url``.

    Args:
        address: The control plane to use, positionally, so the common case
            reads as one call: ``eyvan.init("http://10.0.0.5:8080")``. Same
            meaning as ``base_url``; passing both is an error rather than a
            silent winner. Omit it and the resolution order is ``EYVAN_URL``,
            the saved CLI context, this node's own plane if it serves one, and
            finally the loopback default -- which is why a laptop needs no
            argument at all.
        fleet: Fleet to submit to, when the saved context does not already name
            one.
        org_id: Owning organization; defaults to the saved context, then Eyvan's
            default org.
        project_id: Owning project, resolved the same way.
        artifact_uri: Where callables and workspaces are staged. Mutually
            exclusive with ``artifact_store``.
        artifact_store: A constructed store, for callers that already have one.
        submitter: Injected submission boundary. Supplying one keeps ``init``
            from building a client, which is what makes it testable offline.
        signing_key: Key for signed callables; remote planes refuse it.
        base_url: Keyword form of ``address``.
        token: Bearer credential, when this plane is not one this node owns.
        dev_user: Development identity for a loopback plane.
        environ: Environment to read, so a caller can resolve against something
            other than the process environment.

    Returns:
        The process-wide context, including the ``base_url`` that won.

    Raises:
        ValueError: If ``address`` and ``base_url`` disagree, or if both
            ``artifact_uri`` and ``artifact_store`` are given.
    """

    if address is not None and base_url is not None and address != base_url:
        raise ValueError("Eyvan init accepts address or base_url, not both")
    base_url = base_url if base_url is not None else address
    environment = os.environ if environ is None else environ
    active_context = get_active_context(context_config_path(environment))
    # What control plane is this *machine* part of. Consulted after the caller's
    # own context and never before it: discovery fills a vacuum, it does not
    # overrule a choice someone made on this machine.
    #
    # Skipped inside a job, where the dispatching plane has already issued this
    # process an endpoint. A workload that happened to land on the head would
    # otherwise prefer the node's own answer over the one it was given.
    node_context = discover_node_context(environment) if in_job_context(environment) is None else None
    configured_fleet = _optional(
        fleet
        or environment.get("EYVAN_FLEET")
        or (active_context.fleet if active_context else None)
        or (node_context.fleet if node_context else None)
    )
    configured_org = (
        _optional(
            org_id
            or environment.get("EYVAN_ORG")
            or (active_context.org_id if active_context else None)
            or (node_context.org_id if node_context else None)
        )
        or DEFAULT_ORG_ID
    )
    configured_project = (
        _optional(
            project_id
            or environment.get("EYVAN_PROJECT")
            or (active_context.project_id if active_context else None)
            or (node_context.project_id if node_context else None)
        )
        or DEFAULT_PROJECT_ID
    )
    if artifact_uri is not None and artifact_store is not None:
        raise ValueError("Eyvan init accepts artifact_uri or artifact_store, not both")
    configured_key = signing_key if signing_key is not None else environment.get("EYVAN_CALLABLE_SIGNING_KEY")
    if isinstance(configured_key, str):
        configured_key = configured_key.encode()
    if submitter is None:
        reject_remote_callable_signing_key(configured_key)
    configured_base_url = (
        base_url
        or environment.get("EYVAN_URL")
        or (active_context.base_url if active_context else None)
        # A head's own address, and the reason a process on a head needs no
        # environment at all. It is also what keeps the endpoint below
        # *non-loopback*: loopback is the signal that starts a local development
        # plane, and on a head that would mean starting a second control plane
        # on the machine already running one.
        or (node_context.base_url if node_context else None)
        or DEFAULT_BASE_URL
    )
    endpoint = ControlPlaneEndpoint.parse(configured_base_url)
    configured_artifact_uri = artifact_uri or environment.get("EYVAN_ARTIFACT_URI")
    requires_default_client = submitter is None or (artifact_store is None and configured_artifact_uri is None)
    runtime_profile = _configured_gcp_runtime_profile(endpoint)
    if runtime_profile is not None:
        _validate_native_gcp_artifact_override(configured_artifact_uri, artifact_store)
    # Not inside a job. ``init()`` running on a TPU worker would otherwise try
    # to *start a control plane there*: the default base URL is loopback, and
    # loopback plus "no client configured" is the local-development signal. A
    # job already has a control plane -- the one that dispatched it -- and
    # starting a second one on a worker would fight for the state directory and
    # answer with an empty fleet.
    if requires_default_client and endpoint.is_loopback and in_job_context(environment) is None:
        runtime_profile = _prepared_gcp_runtime_profile(endpoint, runtime_profile)
        if runtime_profile is None:
            ensure_loopback_control_plane(endpoint)
    client = None
    if requires_default_client:
        from eyvan.sdk.client.sync import EyvanClient

        # The same rule `eyvan` itself applies, and for the same reason: on a
        # loopback development plane with no token and no explicit user, the
        # person submitting owns the plane. Running as `operator` there means
        # init cannot read its own fleet -- 403 lacks manage_fleet -- so it
        # could not bring one up, and the operator was sent to `eyvan up`
        # instead. Anything that names a token, a user, or a role keeps exactly
        # the identity it named, and a remote plane is untouched.
        default_role = (
            "fleet_admin"
            if (
                token is None
                and dev_user is None
                and environment.get("EYVAN_TOKEN") is None
                and environment.get("EYVAN_DEV_USER") is None
                and environment.get("EYVAN_ROLE") is None
                and endpoint.is_loopback
            )
            else "operator"
        )
        configured_role = environment.get("EYVAN_ROLE", default_role)
        identity = resolve_control_plane_identity(
            endpoint,
            # Read from this node's own installer-written credential when the
            # endpoint is exactly the plane this node serves. Never copied
            # anywhere; see ``discover_node_token``.
            token=token
            or environment.get("EYVAN_TOKEN")
            # The personal token `eyvan login` stored for exactly this plane.
            or team_token_or_warn(
                endpoint.url, warn=lambda message: warnings.warn(message, RuntimeWarning, stacklevel=2)
            )
            or discover_node_token_or_warn(
                endpoint,
                environment,
                warn=lambda message: warnings.warn(message, RuntimeWarning, stacklevel=2),
            ),
            dev_user=dev_user or environment.get("EYVAN_DEV_USER"),
            dev_org=configured_org,
            dev_project=configured_project,
            dev_roles=configured_role,
        )
        client = EyvanClient(
            base_url=endpoint.url,
            identity=identity,
        )
    if artifact_store is None:
        if configured_artifact_uri is not None:
            artifact_store = artifact_store_from_uri(configured_artifact_uri)
        elif runtime_profile is not None:
            artifact_store = artifact_store_from_uri(runtime_profile.artifact_root_uri)
        else:
            if client is None:
                raise RuntimeError("Eyvan managed artifacts require the default SDK client")
            artifact_store = EyvanManagedArtifactStore(
                client,
                org_id=configured_org,
                project_id=configured_project,
            )
    if submitter is None:
        from eyvan.capacity.autoscale import AutoscaleConfigCache
        from eyvan.sdk.client.callables import EyvanCallableSubmitter

        if client is None:
            raise RuntimeError("Eyvan callable submission requires the default SDK client")
        submitter = EyvanCallableSubmitter[tp.NoReturn](
            client,
            artifact_store,
            queue_admission_cache=AutoscaleConfigCache(
                active_context.autoscale_cache_dir if active_context is not None else None
            ),
        )
    # Bring the named fleet up before anything is submitted to it. Doing it here
    # rather than leaving it to the operator is the difference between one
    # command and four, and a fleet that is already serving is untouched.
    if client is not None and endpoint.is_loopback and in_job_context(environment) is None:
        # A TPU in the operator's GCP account is usable by name before anyone
        # connected it. Only with a local GCP runtime: that is what owns the
        # credential the connect needs, and without it nothing is looked up.
        if runtime_profile is not None:
            from eyvan.capacity.providers.gcp.tpu_discovery import refresh_in_background

            configured_fleet = _connect_discovered_tpu(client, configured_fleet, endpoint.url)
            refresh_in_background()
        _ensure_fleet_applied(
            client,
            configured_fleet,
            active_context.autoscale_cache_dir if active_context is not None else None,
        )
    context = ExecutionContext(
        fleet=configured_fleet,
        org_id=configured_org,
        project_id=configured_project,
        artifact_store=artifact_store,
        submitter=submitter,
        signing_key=configured_key or None,
        base_url=endpoint.url,
    )
    global _context
    with _context_lock:
        _context = context
    return context


def get_context() -> ExecutionContext[tp.NoReturn]:
    """Return the configured context, lazily loading the active CLI context."""

    with _context_lock:
        context = _context
        if context is None:
            context = init()
        return context


def _zero_arg_invocation(fn: tp.Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> tp.Callable[[], R]:
    """Capture one fully bound invocation without retaining its wrapper context."""

    def invoke() -> R:
        return fn(*args, **kwargs)

    return invoke


def _callable_workspace_root(fn: tp.Callable[..., tp.Any]) -> Path | None:
    """Return the directory the callable's own file lives in, when there is one.

    This is the default a submitter means by "my code": the tree their script
    sits in, which is exactly what relative ``sys.path`` entries inside the
    callable resolve against once the bundle is extracted to the worker's cwd.
    A callable defined interactively -- a REPL, a notebook cell -- has no such
    file, and inventing a root for it would ship a directory the caller never
    pointed at, so it ships nothing until a ``workspace`` names one.
    """

    code = getattr(fn, "__code__", None)
    filename = getattr(code, "co_filename", "")
    if not isinstance(filename, str) or not filename.endswith(".py"):
        return None
    source = Path(filename).resolve()
    return source.parent if source.is_file() else None


def _resolve_fleet_from_tpu(context: ExecutionContext[tp.Any], tpu: AcceleratorConfig) -> str:
    """Ask the control plane which fleet has an enabled pool matching the TpuConfig shape."""

    if not isinstance(context.submitter, EyvanCallableSubmitter):
        raise RuntimeError(
            "Auto-resolving a fleet from a TpuConfig requires the default Eyvan callable submitter; "
            "pass fleet= or configure a submitter that supports capacity lookup"
        )
    explicit = getattr(tpu, "capacity_class_explicit", True)
    pools = context.submitter.client.list_capacity(
        org_id=context.org_id,
        project_id=context.project_id,
        accelerator_type=tpu.type,
        capacity_class=tpu.capacity_class if explicit else None,
    )
    if not explicit:
        # An unnamed class is a preference: the default class first, then
        # reserved capacity the operator already holds, exactly as the
        # submitter's fallback does. One listing answers both. Never on-demand,
        # which would start billing for something nobody asked for.
        pools = [pool for pool in pools if pool.capacity_class is tpu.capacity_class] or [
            pool for pool in pools if pool.capacity_class is CapacityClass.RESERVED
        ]
    if not pools:
        raise EyvanPlacementError(
            requested_accelerator_type=tpu.type,
            available_accelerator_types=(),
            scope=f"{context.org_id}/{context.project_id} with capacity class {tpu.capacity_class.value!r}",
        )
    for pool in pools:
        if pool.max_cells >= tpu.pods:
            return pool.fleet
    return pools[0].fleet


def _refuse_unplaceable_config(config: AcceleratorConfig) -> None:
    """Refuse a config whose GPU the job would not actually be able to use.

    Placement supports GPU callables, but ``_DEFAULT_GPU_IMAGE`` pins
    ``...with-nvidia-580``, and that driver branch dropped Pascal and Volta.
    A P100 or V100 cell can therefore boot, register, and report job success
    even though no process can open its GPU.

    Refuse rather than allow silent CPU fallback or a failure far from the
    cause. Lifting this needs either a driver the accelerator supports --
    ``vm_image`` is settable per pool -- or a cell-side check that refuses to
    register a GPU cell whose driver cannot see its GPU.
    """

    if isinstance(config, GpuConfig):
        raise NotImplementedError(
            "Eyvan places a callable on a GPU pool, but the default GPU image installs NVIDIA driver 580, which "
            "does not support Pascal or Volta -- a P100 or V100 cell runs the job with an unusable GPU and still "
            "reports success. Set a pool `vm_image` whose driver supports the accelerator, or use TpuConfig or "
            "CpuConfig, or submit a command workload with `eyvan workload submit`."
        )


class Executable(tp.Generic[P, R]):
    """Side-effect-free callable wrapper produced by ``@execute``."""

    def __init__(
        self,
        fn: tp.Callable[P, R],
        config: AcceleratorConfig,
        context: ExecutionContext[R] | None,
        workspace: str | os.PathLike[str] | bool | None = None,
    ):
        self._fn = fn
        self._config = config
        self._context = context
        self._workspace = workspace
        functools.update_wrapper(self, fn)

    @property
    def config(self) -> AcceleratorConfig:
        """Return this callable's immutable TPU request."""

        return self._config

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> EyvanJob[R]:
        """Serialize and submit one invocation, returning its durable handle."""

        context = self._context or get_context()
        if context.submitter is None:
            raise RuntimeError("Eyvan callable execution requires a configured CallableSubmitter")
        _refuse_unplaceable_config(self._config)
        if os.environ.get("PYTHONPATH"):
            # A submitter who steered their own interpreter expects it to steer
            # the worker's too. It cannot: only the callable travels, and every
            # import inside it resolves on the worker, where ``PYTHONPATH`` was
            # never set. The job then provisions real accelerators to die on
            # ``ModuleNotFoundError``. The mechanism that does travel is the
            # workspace bundle -- it is extracted to the worker's cwd and
            # ``sys.path[0]``, so relative layout survives the jump -- and
            # naming it here is the difference between a one-line fix and a
            # blind re-submission.
            warnings.warn(
                "PYTHONPATH is set in this process and does not travel to the workers "
                "running the callable; ship the directories it names with "
                "@execute(..., workspace=...) instead",
                stacklevel=2,
            )
        inspect.signature(self._fn).bind(*args, **kwargs)
        bound = _zero_arg_invocation(self._fn, *args, **kwargs)
        artifact = context.artifact_store.put(dump_callable(bound, signing_key=context.signing_key))
        invocation_id = f"call-{uuid.uuid4().hex}"
        fleet = context.fleet
        if fleet is None:
            fleet = _resolve_fleet_from_tpu(context, self._config)
        workspace_uri: str | None = None
        workspace_sha256: str | None = None
        workspace_signature: str | None = None
        # Three contracts share this block, and each is someone's explicit
        # decision. An explicit path is fail-loud: the caller asked for these
        # files by name, so degrading to shipping none would hide the loss.
        # ``False`` is the explicit refusal. The default derives the root from
        # the callable's own file -- the directory a submitter means when they
        # say "my code" -- and *degrades* rather than fails: a callable that
        # needs nothing local must not die at submission because its directory
        # happens to be unpackageable, but the loss is named, never silent.
        if self._workspace is False:
            explicit_root: Path | None = None
        else:
            explicit_root = None if self._workspace is None else Path(self._workspace)
        auto_root = _callable_workspace_root(self._fn) if self._workspace is None else None
        for root, fail_loud in ((explicit_root, True), (auto_root, False)):
            if root is None:
                continue
            try:
                bundle = build_workspace(root)
            except WorkspaceError as error:
                if fail_loud:
                    raise
                warnings.warn(
                    f"the callable's directory {root} could not be shipped as a workspace "
                    f"({error}); submitting without it",
                    stacklevel=2,
                )
                continue
            workspace_uri = context.artifact_store.put_workspace(bundle.payload, bundle.sha256)
            workspace_sha256 = bundle.sha256
            # Signed under the same key as the callable. The bundle is entered
            # and put on sys.path before the pickle is loaded, so an unsigned
            # workspace can shadow any module the signed callable resolves by
            # reference -- and the callable's signature would prove nothing.
            if context.signing_key is not None:
                workspace_signature = hmac.new(context.signing_key, bundle.payload, hashlib.sha256).hexdigest()
            break
        request = CallableSubmission(
            invocation_id=invocation_id,
            fleet=fleet,
            org_id=context.org_id,
            project_id=context.project_id,
            callable_name=f"{self._fn.__module__}.{self._fn.__qualname__}",
            tpu=self._config,
            artifact=artifact,
            result_uri=context.artifact_store.allocate_result_uri(invocation_id),
            workspace_uri=workspace_uri,
            workspace_sha256=workspace_sha256,
            workspace_signature=workspace_signature,
        )
        return context.submitter.submit_callable(request)

    def remote(self, *args: P.args, **kwargs: P.kwargs) -> EyvanJob[R]:
        """Alias for ``submit``, matching familiar remote-execution APIs."""

        return self.submit(*args, **kwargs)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Submit one invocation and block for its durable result."""

        return self.submit(*args, **kwargs).result()


def execute(
    config: AcceleratorConfig,
    *,
    context: ExecutionContext[R] | None = None,
    workspace: str | os.PathLike[str] | bool | None = None,
) -> tp.Callable[[tp.Callable[P, R]], Executable[P, R]]:
    """Decorate a function for durable, scale-from-zero Eyvan execution.

    Takes any ``AcceleratorConfig`` -- ``TpuConfig``, ``GpuConfig`` or
    ``CpuConfig``. The GPU and CPU classes were complete, validated and exported
    while this one entry point refused them, so the only hardware the decorator
    could reach was TPU; everything below it has always been shape-agnostic.

    ``workspace`` ships a working directory alongside the callable, so
    application modules travel with the job instead of having to be baked
    into the image or staged as fleet artifacts. The default is the directory
    the decorated function's own file lives in: that is what a submitter
    means by "my code", and it is why relative ``sys.path`` entries inside the
    callable keep resolving once the bundle is extracted to the worker's cwd
    and ``sys.path[0]``. An explicit path names a different root and fails
    loudly when it cannot be packaged; the derived default instead degrades
    to submitting without a workspace, with a warning naming what was lost,
    because a callable that needs nothing local must not die at submission.
    ``workspace=False`` is the explicit refusal of all of the above.
    """

    # A class object satisfies the protocol's attribute check as readily as an
    # instance does, so ``@execute(TpuConfig)`` -- the missing parentheses --
    # would pass a bare protocol test and fail much later with an error naming
    # none of this. Rejecting the type here is what keeps that a one-line fix.
    if isinstance(config, type) or not isinstance(config, AcceleratorConfig):
        raise TypeError("execute requires a TpuConfig, GpuConfig, or CpuConfig instance")

    def decorate(fn: tp.Callable[P, R]) -> Executable[P, R]:
        if not callable(fn):
            raise TypeError("execute can only decorate callable objects")
        return Executable(fn, config, context, workspace)

    return decorate


__all__ = (
    "AcceleratorConfig",
    "CallableSubmission",
    "CallableSubmitter",
    "CpuConfig",
    "EasyDeLSetup",
    "Executable",
    "ExecutionContext",
    "EyvanJob",
    "EyvanJobError",
    "EyvanPlacementError",
    "EyvanUnavailableError",
    "GpuConfig",
    "JobBackend",
    "JobCompletion",
    "SetupScript",
    "TpuConfig",
    "execute",
    "get_context",
    "init",
)
