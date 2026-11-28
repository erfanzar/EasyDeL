# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A disposable loopback demo using real Eyvan APIs and a local worker."""

from __future__ import annotations

import io
import json
import os
import secrets
import socket
import threading
import time
import typing as tp
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import click


def _sample_callable() -> dict[str, int]:
    """Executed by the worker, not by the CLI that submits it."""

    import os

    return {"value": sum(range(10)), "worker_pid": os.getpid()}


@contextmanager
def _serve_loopback(factory: tp.Callable[[str], tp.Any]) -> tp.Iterator[str]:
    """Reserve a real loopback socket before starting the server; always close it."""

    import uvicorn

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        endpoint = f"http://127.0.0.1:{listener.getsockname()[1]}"
        application = factory(endpoint)
        server = uvicorn.Server(uvicorn.Config(application, log_level="error", access_log=False))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError("the demo's loopback server did not start")
                time.sleep(0.02)
            yield endpoint
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            if thread.is_alive():
                raise RuntimeError("the demo's loopback server did not stop cleanly")


def run_demo(*, timeout: float = 45) -> dict[str, tp.Any]:
    """Run one callable on a disposable local fleet, without saved-context discovery."""

    from eyvan._internal.platform import EyvanPlatformConfig
    from eyvan.artifacts import EyvanManagedArtifactStore
    from eyvan.capacity.fleet_config import EyvanFleetConfig
    from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
    from eyvan.config import EyvanSettings
    from eyvan.records import CapacityClass
    from eyvan.sdk.client.callables import EyvanCallableSubmitter
    from eyvan.sdk.client.identity import ClientIdentity
    from eyvan.sdk.client.jobs import CpuConfig
    from eyvan.sdk.client.sync import EyvanClient
    from eyvan.sdk.client.transport import SyncApiTransport
    from eyvan.sdk.execution import ExecutionContext, execute
    from eyvan.sdk.job_receipt import JobTerminalOutput
    from eyvan.server.api.app import create_app
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.worker.agent.app import create_agent_app
    from eyvan.worker.agent.service import CellAgentService

    with TemporaryDirectory(prefix="eyvan-demo-") as directory, ExitStack() as stack:
        root = Path(directory)
        secret = secrets.token_urlsafe(32)
        settings = EyvanSettings(
            capacity_backend="local",
            execution_backend="native-local",
            agent_shared_secret=secret,
            agent_state_dir=str(root / "agent"),
            artifact_store_uri=str(root / "artifacts"),
            artifact_token_key=secrets.token_urlsafe(32),
            bootstrap_token=secret,
            trusted_callable_projects=("default/default",),
            scheduler_interval_s=0.1,
            cell_reconcile_interval_s=0.1,
            # Both sockets are pre-bound to literal loopback addresses. This
            # opt-in belongs only to the disposable local demo, never a cloud bind.
            allow_insecure_artifact_http=True,
        )
        agent = CellAgentService(settings=settings, state=FleetStateStore(LocalBackend(root / "fleet.json")))
        agent_url = stack.enter_context(_serve_loopback(lambda _: create_agent_app(agent, settings)))

        def plane(endpoint: str):
            return create_app(
                settings=replace(settings, public_url=endpoint),
                repository=InMemoryRepository(),
                platform_config=EyvanPlatformConfig(default_agent_endpoint=agent_url),
            )

        api_url = stack.enter_context(_serve_loopback(plane))
        identity = ClientIdentity(token=secret)
        client = stack.enter_context(EyvanClient(base_url=api_url, identity=identity))
        transport = SyncApiTransport(api_url, identity=identity)
        stack.callback(transport.client.close)
        config = EyvanFleetConfig.model_validate(
            {
                "version": 1,
                "name": "demo",
                "provider": {"type": "local", "region": "local"},
                "executor": "native-local",
                "capacity_class": "spot",
                "pools": {
                    "cpu": {
                        "kind": "cpu",
                        "capacity_backend": "local",
                        "machine_type": "cpu",
                        "resources": {"cpu": 1, "memory_gib": 1},
                        "min_workers": 1,
                        "max_workers": 1,
                        "agent_endpoint": agent_url,
                    }
                },
            }
        )
        transport.mutate(
            "PUT",
            "/api/v1/fleets/demo",
            json={"pools": config.pool_payloads()},
            idempotency_key="demo-fleet",
        )
        store = EyvanManagedArtifactStore(client, org_id="default", project_id="default")
        # Generic job narration promises a persistent plane; this disposable
        # demo has none. Retain its diagnostics for failures, not misleading
        # follow-up commands pointing at a listener we are about to close.
        transcript = io.StringIO()
        submitter = EyvanCallableSubmitter(client, store, terminal_output=JobTerminalOutput(stream=transcript))
        context = ExecutionContext(
            fleet="demo",
            org_id="default",
            project_id="default",
            artifact_store=store,
            submitter=submitter,
            base_url=api_url,
        )
        job = execute(CpuConfig(capacity_class=CapacityClass.SPOT), context=context, workspace=False)(
            _sample_callable
        ).submit()
        try:
            result = job.result(timeout=timeout)
        except BaseException:
            # Cancellation happens while both services are still running.
            # A timeout or interrupt must not leave a worker holding a lease.
            from eyvan.sdk.lifecycle import TERMINAL_STATES

            if job.status() not in TERMINAL_STATES:
                job.cancel()
                job.wait(timeout=10)
            raise
        if result["value"] != 45 or result["worker_pid"] == os.getpid():
            raise RuntimeError("demo did not return the expected result from a separate worker")
        report = {
            "mode": "local-loopback",
            "state": "succeeded",
            "workload_id": job.id,
            "result": result,
            "runner_pid": os.getpid(),
            "control_plane_url": api_url,
            "agent_url": agent_url,
            "cloud_resources_created": 0,
        }
    # Reached only after worker completion, both listener shutdowns and cleanup.
    return report


@click.command("demo")
@click.option("--json", "as_json", is_flag=True, help="Emit the result and cleanup receipt as JSON.")
@click.option("--timeout", type=click.FloatRange(min=0.1, max=600), default=45, show_default=True)
def demo(as_json: bool, timeout: float) -> None:
    """Run a disposable CPU callable through a private loopback fleet; no cloud access."""

    try:
        report = run_demo(timeout=timeout)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    if as_json:
        click.echo(json.dumps(report))
    else:
        click.echo(
            f"Eyvan demo succeeded: result={report['result']['value']} from worker PID {report['result']['worker_pid']}"
        )
        click.echo(
            "The private loopback API, agent and temporary state have been cleaned up. No cloud resources created."
        )


def register_demo_commands(group: click.Group) -> None:
    group.add_command(demo)
