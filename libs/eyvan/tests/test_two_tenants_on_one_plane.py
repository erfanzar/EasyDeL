# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Two organizations on one control plane, and the wall between them.

Eyvan enforces per-org and per-project scope at roughly seventy-five call sites,
and every one of them was individually plausible. What nothing established is the
property those call sites are supposed to add up to: that a principal scoped to
one tenant cannot reach another tenant's workloads, capacity, governance or
audit trail through *any* route.

This asserts that over the real HTTP surface, and it is written so a route added
tomorrow cannot quietly avoid the question. ``ROUTE_CLASSIFICATION`` names every
route in the application; a route missing from it fails
``test_every_route_is_classified`` rather than silently going untested. That is
the difference between a suite that covers the boundary today and one that keeps
covering it.

Classification is the honest part. A route that cannot leak across tenants --
because it takes no tenant scope, or is authenticated by a capability token
rather than a principal -- is exempt *with a stated reason*, not omitted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app

# --------------------------------------------------------------------------- classification

#: Routes exercised below for cross-tenant denial, mapped to the concrete
#: attempt. Every entry is "org B's operator, aimed at org A's resource".
CROSS_TENANT_ROUTES: dict[str, str] = {
    "GET /api/v1/workloads": "list another org's workloads",
    "GET /api/v1/workloads/{workload_id}": "read another org's workload",
    "GET /api/v1/workloads/{workload_id}/explain": "read another org's scheduling verdict",
    "GET /api/v1/workloads/{workload_id}/cost": "read what another org's run cost them",
    "GET /api/v1/usage": "read who in another org spent what, and on which accelerators",
    "GET /api/v1/webhooks": "read where another org sends its events",
    "POST /api/v1/webhooks": "send another org's audit trail somewhere of your choosing",
    "DELETE /api/v1/webhooks/{webhook_id}": "silence another org's notifications",
    "GET /api/v1/workloads/{workload_id}/logs": "read another org's logs",
    "GET /api/v1/workloads/{workload_id}/log-page": "read another org's structured logs",
    "GET /api/v1/workloads/{workload_id}/logs/stream": "follow another org's logs",
    "GET /api/v1/workloads/{workload_id}/progress": "read another org's progress",
    "GET /api/v1/workloads/{workload_id}/process-status": "read another org's process state",
    "GET /api/v1/workloads/{workload_id}/children": "enumerate another org's job tree",
    "GET /api/v1/workloads/{workload_id}/endpoints": "read another org's service endpoints",
    "GET /api/v1/workloads/{workload_id}/service-health": "read another org's replica health and addresses",
    "GET /api/v1/workloads/{workload_id}/service-decisions": "read another org's scaling history and capacity plans",
    "POST /api/v1/workloads": "submit into another org",
    "POST /api/v1/workloads/{workload_id}/cancel": "cancel another org's workload",
    "POST /api/v1/workloads/{workload_id}/retry": "retry another org's workload",
    "POST /api/v1/workloads/{workload_id}/checkpoint": "checkpoint another org's workload",
    "POST /api/v1/workloads/{workload_id}/exec": "run a command beside another org's workload",
    "POST /api/v1/workloads/estimate": "price work in another org",
    "POST /api/v1/batches": "fan out into another org",
    "POST /api/v1/dags": "release a graph into another org",
    "POST /api/v1/groups": "place a heterogeneous group into another org",
    "GET /api/v1/quotas": "read another org's quotas",
    "GET /api/v1/federation/budget-grants": "read another org's federated budget slices",
    "GET /api/v1/federation/quota-grants": "read another org's federated quota slices",
    "GET /api/v1/quotas/current": "read another org's effective quota",
    "PUT /api/v1/quotas": "write another org's quota",
    "DELETE /api/v1/quotas": "delete another org's quota",
    "GET /api/v1/budgets": "read another org's budgets",
    "GET /api/v1/budgets/{org_id}/{project_id}": "read another org's budget",
    "GET /api/v1/budgets/{org_id}/{project_id}/status": "read another org's spend",
    "PUT /api/v1/budgets/{org_id}/{project_id}": "write another org's budget",
    "GET /api/v1/policies": "read another org's policies",
    "GET /api/v1/policies/{org_id}/{project_id}": "read another org's policy",
    "PUT /api/v1/policies/{org_id}/{project_id}": "write another org's policy",
    "GET /api/v1/events": "read another org's audit trail",
    "GET /api/v1/events/stream": "follow another org's audit trail",
    "GET /api/v1/artifacts": "list another org's artifacts",
    "GET /api/v1/artifacts/usage": "read another org's storage usage",
    "GET /api/v1/artifacts/inventory": "inventory another org's physical namespace",
    "POST /api/v1/artifacts/inventory/reindex": "reindex metadata into another org",
    "GET /api/v1/artifacts/collectable": "read another org's collection candidates",
    "POST /api/v1/artifacts/collect": "collect (and with apply, delete) another org's artifacts",
    "GET /api/v1/artifacts/references": "read another org's artifact references",
    "POST /api/v1/artifacts/holds": "hold another org's artifact",
    "DELETE /api/v1/artifacts/holds": "release another org's artifact hold",
    "GET /api/v1/fleets": "list fleets runnable in another org's scope",
    "GET /api/v1/capacity/queue-trend": "read another project's submitted and admitted counts",
    "GET /api/v1/organizations/{org_id}": "read another tenant's registration",
    "GET /api/v1/organizations/{org_id}/projects": "enumerate another tenant's projects",
    "PUT /api/v1/organizations/{org_id}/projects/{project_id}": "create a project in another tenant",
    "POST /api/v1/grants": "grant access inside another tenant",
    "DELETE /api/v1/grants": "strip access inside another tenant",
    "GET /api/v1/tracking/runs": "list another org's tracked training runs",
    "GET /api/v1/tracking/runs/{run_id}": "read another org's run config, summary and attempts",
    "GET /api/v1/tracking/runs/{run_id}/series": "read another org's metric history",
    "GET /api/v1/tracking/runs/{run_id}/histograms": "read another org's logged distributions",
    "GET /api/v1/tracking/runs/{run_id}/episodes": "list another org's RL episodes",
    "GET /api/v1/tracking/runs/{run_id}/episodes/curve": "read another org's episode return curves",
    "GET /api/v1/tracking/runs/{run_id}/episodes/breakdown": "read another org's per-actor and eval results",
    "GET /api/v1/tracking/runs/{run_id}/rollouts": "list another org's sampled trajectories",
    "GET /api/v1/tracking/runs/{run_id}/rollouts/{rollout_id}": "read another org's trajectory and its video",
    "GET /api/v1/tracking/runs/{run_id}/completions": "list another org's prompts and model responses",
    "GET /api/v1/tracking/runs/{run_id}/completions/stats": "read another org's per-step reward signal",
    "GET /api/v1/tracking/runs/{run_id}/completions/{completion_id}": "read another org's full model responses",
    "GET /api/v1/tracking/runs/{run_id}/tables": "read another org's logged tables",
    "GET /api/v1/tracking/runs/{run_id}/media": "list another org's logged media",
    "GET /api/v1/tracking/runs/{run_id}/blobs/{sha256}": "download another org's media bytes",
}

#: Routes that require fleet administration outright. A tenant principal is
#: refused regardless of which tenant's resource it names, so the cross-tenant
#: question does not arise -- but they are listed rather than omitted, because
#: "needs fleet admin" is a claim worth failing on if it ever stops being true.
FLEET_ONLY_ROUTES: dict[str, str] = {
    # Reads whatever the schema holds, across every tenant, so no filter applied
    # to the rows afterwards could make it project-scoped. Fleet administration
    # is the only role whose scope is also the whole database.
    "POST /api/v1/query": "ad-hoc read-only SQL over the whole control plane",
    # Replaces the credential every tenant's capability tokens are signed with;
    # a rotation touches every tenant at once, so it is fleet administration.
    "POST /api/v1/secrets/rotate": "rotate the shared capability-signing credential",
    # Stacks and allocations of the process serving every tenant; there is no
    # per-tenant view of a shared interpreter to scope them to.
    "GET /api/v1/process/stacks": "thread stacks of the shared control-plane process",
    "GET /api/v1/process/allocations": "allocation snapshot of the shared control-plane process",
    # Named for a workload but settled through the preemption path, which is
    # fleet-owned: it moves capacity between tenants' jobs. A tenant kicking its
    # own attempt would be reasonable and is not what this route is.
    "POST /api/v1/workloads/{workload_id}/kick": "interrupt an attempt via fleet preemption settlement",
    "POST /api/v1/attachments/plan": "local CLI attachment preview uses the machine owner's credentials",
    "POST /api/v1/attachments": "local CLI attachment execution uses the machine owner's credentials",
    "GET /api/v1/attachments": "local attachment jobs are fleet-wide operational state",
    "GET /api/v1/attachments/discovered": "the machine owner's GCP TPU inventory is fleet-wide operational state",
    "GET /api/v1/attachments/{attachment_id}": "local attachment progress and results are fleet-wide operational state",
    "GET /api/v1/cells": "fleet capacity inventory",
    "GET /api/v1/cells/{cell_id}": "one cell's provider state",
    "POST /api/v1/cells": "register capacity",
    "DELETE /api/v1/cells/{cell_id}": "destroy capacity",
    "POST /api/v1/cells/{cell_id}/drain": "drain capacity",
    "GET /api/v1/capacity-pools": "fleet pool inventory",
    "GET /api/v1/capacity-pools/{pool_id}": "one pool",
    "POST /api/v1/capacity-pools": "create a pool",
    "DELETE /api/v1/capacity-pools/{pool_id}": "disable a pool",
    "GET /api/v1/capacity/orphans": "capacity no record claims",
    "GET /api/v1/capacity/builds": "which build each cell agent runs",
    "GET /api/v1/capacity/billing": "fleet-wide cost",
    "GET /api/v1/capacity/carbon": "fleet-wide grid intensity",
    "GET /api/v1/capacity/warmth": "fleet-wide warm state",
    "GET /api/v1/capacity/launch-latency": "fleet-wide provisioning latency",
    "GET /api/v1/capacity/quota": "provider quota for the cloud project the fleet's pools declare",
    "GET /api/v1/fleets/{fleet_name}": "one fleet",
    "PUT /api/v1/fleets/{fleet_name}": "declarative fleet apply",
    "DELETE /api/v1/fleets/{fleet_name}": "delete a fleet",
    "GET /api/v1/fleets/{fleet_name}/init-script": "a fleet's host setup script",
    "PUT /api/v1/fleets/{fleet_name}/init-script": "change what every new host of a fleet runs",
    "POST /api/v1/fleets/{fleet_name}/callables": "place a callable on fleet capacity",
    "POST /api/v1/fleets/{fleet_name}/node-types/{node_type}/workloads": "place work on fleet capacity",
    "GET /api/v1/leases": "fleet lease inventory",
    "GET /api/v1/leases/{lease_id}": "one lease",
    "GET /api/v1/fleet/events": "unscoped audit trail",
    "POST /api/v1/preemptions/{command_id}/force-finalize": "break-glass preemption finalize",
    "POST /api/v1/workloads/{workload_id}/force-finalize": "break-glass; it gives up a cell, not just a workload",
    "GET /api/v1/federation/peers": "which control planes this one federates with",
    "PUT /api/v1/federation/peers/{plane_id}": "register a peer plane",
    "POST /api/v1/federation/budget-grants": "cut a peer a slice of a tenant's budget",
    "DELETE /api/v1/federation/budget-grants/{grant_id}": "withdraw a peer's authority to spend",
    "POST /api/v1/federation/quota-grants": "cut a peer a slice of a tenant's quota",
    "DELETE /api/v1/federation/quota-grants/{grant_id}": "withdraw a peer's authority to place work",
    "GET /api/v1/system/invariants": "fleet-wide durability check",
    "POST /api/v1/system/invariants/soak": "fleet-wide proof-scoped durability check",
    "POST /api/v1/system/soak-evidence": "external P8 measurement of the whole plane, not of any tenant",
    "POST /api/v1/system/reconcile": "force a scheduling pass",
    "GET /api/v1/organizations": "tenant list, narrowed to the caller's own grants",
    "PUT /api/v1/organizations/{org_id}": "register a tenant",
    "GET /api/v1/grants": "grant list, narrowed to the caller's own scopes",
    "POST /api/v1/budgets/{org_id}/{project_id}/roll": "close a budget period",
    "GET /api/v1/capacity": "pool availability for placement, org-scoped by query",
}

#: Routes that carry no principal-scoped tenancy at all, each with the reason.
EXEMPT_ROUTES: dict[str, str] = {
    "GET /api/v1/dashboard/config": "public OIDC discovery hints; no tenant data and no secret",
    "GET /api/v1/dashboard/auth/login": "starts the OIDC redirect; names no tenant; sets the transaction cookie",
    "GET /api/v1/dashboard/auth/callback": "completes sign-in; the code exchange is tenant-agnostic",
    "POST /api/v1/dashboard/auth/logout": "drops the caller's own session cookie; reads nothing tenant-scoped",
    "GET /api/v1/session": "authenticated self-introspection; only the caller's resolved grants and scopes",
    "GET /api/v1/health": "liveness; no tenant data",
    "GET /api/v1/tracking/health": "authenticated tracker readiness; store root and segment count, no run data",
    "GET /api/v1/build": "what code this plane runs; authenticated because it fingerprints, but tenant-independent",
    "GET /healthz": "liveness; no tenant data",
    "GET /readyz": "readiness; reports only whether this replica can reach its store",
    "GET /metrics": "separately authenticated by a scrape token; see test_metrics_endpoint_is_not_public",
    "GET /docs": "API documentation",
    "GET /docs/oauth2-redirect": "API documentation",
    "GET /redoc": "API documentation",
    "GET /openapi.json": "API schema",
    "PUT /api/v1/artifacts/callables/{digest}": "content-addressed; authorized by project trust and digest",
    "GET /api/v1/artifacts/callables/{digest}/content": "capability token, not a principal",
    "PUT /api/v1/artifacts/workspaces/{digest}": "content-addressed; authorized by project trust and digest",
    "GET /api/v1/artifacts/workspaces/{digest}/content": "capability token, not a principal",
    "GET /api/v1/artifacts/results/{result_id}": "capability token, not a principal",
    "PUT /api/v1/artifacts/results/{result_id}/content": "capability token, not a principal",
    "POST /api/v1/workloads/{workload_id}/progress-reports": "capability token, not a principal",
    "POST /api/v1/workloads/{workload_id}/children": "workload capability token, not a principal",
    "PUT /api/v1/workloads/{workload_id}/endpoint": "workload capability token, not a principal",
    "GET /api/v1/secrets/content": "secret capability token, not a principal",
    # Team accounts (404 unless team mode is on). Accounts are plane-wide, not
    # per-tenant; authority still comes from the role grants an invite carries.
    "GET /api/v1/team/status": "unauthenticated; reports only whether team mode is on",
    "POST /api/v1/team/login": "password sign-in; the caller's own credentials, no tenant data",
    "POST /api/v1/team/logout": "ends the caller's own session cookie",
    "GET /api/v1/team/me": "authenticated self-introspection; only the caller's account and grants",
    "POST /api/v1/team/tokens": "mints a token for the caller's own account only",
    "GET /api/v1/team/tokens": "lists the caller's own tokens; another subject's needs fleet admin",
    "DELETE /api/v1/team/tokens/{token_id}": "own token, or any token for a fleet admin (test_team_accounts)",
    "POST /api/v1/team/invites/preview": "the invite token is the authority; returns only that invite's email",
    "POST /api/v1/team/invites/accept": "the invite token is the authority; grants are the inviter's, checked at issue",
    "POST /api/v1/team/invites": (
        "each carried grant is checked with the tenancy grant-administration rule, so a project admin "
        "invites only into their own project (test_team_accounts)"
    ),
    # Assistant: every tool call runs as the caller (test_assistant proves a non-admin's
    # fleet read is refused through it); sizing and status read no tenant data.
    "GET /api/v1/assistant/status": "reports only whether the assistant is configured and its model name",
    "GET /api/v1/team/reclaim/plan": (
        "the caller's own verdict and counts; victim ids and per-user borrowing only for a fleet admin (test_governance)"
    ),
    "POST /api/v1/team/reclaim": (
        "the deterministic borrow rule picks victims for the caller's own guarantee, never the caller; "
        "victim ids only for a fleet admin (test_governance)"
    ),
    "POST /api/v1/assistant/chat": "tools call the service with the caller's own principal (test_assistant)",
    "POST /api/v1/assistant/chat/stream": (
        "tools run as the caller; continues only the caller's own conversation, else 404 (test_assistant_stream)"
    ),
    "GET /api/v1/assistant/conversations": "the caller's own saved chats only (test_assistant_stream)",
    "GET /api/v1/assistant/conversations/{conversation_id}": (
        "owner-only; anyone else, a fleet admin included, gets 404 (test_assistant_stream)"
    ),
    "PUT /api/v1/assistant/conversations/{conversation_id}": "rename the caller's own chat only (test_assistant_stream)",
    "DELETE /api/v1/assistant/conversations/{conversation_id}": (
        "delete the caller's own chat only (test_assistant_stream)"
    ),
    "GET /api/v1/assistant/presets": "a fixed built-in list of provider endpoints; no stored data",
    "GET /api/v1/assistant/oauth/openrouter/callback/{state}": (
        "authorized only by a single-use, expiring state minted for one fleet admin (test_assistant_presets)"
    ),
    "GET /api/v1/assistant/providers": (
        "plane-wide provider names and models, never a key; URLs and state only for a fleet admin "
        "(test_assistant_providers)"
    ),
    "POST /api/v1/assistant/size": "pure memory arithmetic over the request body; no stored data",
    "POST /api/v1/team/requests": "files a request as the caller only",
    "GET /api/v1/team/requests": "the caller's own requests; everyone's only for a fleet admin (test_assistant)",
    "POST /api/v1/team/requests/{request_id}/resolve": (
        "approve/decline needs fleet admin, withdraw only the filer's own request (test_assistant)"
    ),
    "POST /api/v1/research/plans": (
        "proposes a plan owned by the caller; needs SUBMIT in that project (test_research_plans)"
    ),
    "GET /api/v1/research/plans": "the caller's own plans; everyone's only for a fleet admin (test_research_plans)",
    "GET /api/v1/research/plans/{plan_id}": ("owner or fleet admin; anyone else gets 404 (test_research_plans)"),
    "POST /api/v1/research/plans/{plan_id}/approve": "owner only; others get 403/404 (test_research_plans)",
    "POST /api/v1/research/plans/{plan_id}/stop": "owner or fleet admin; anyone else gets 404 (test_research_plans)",
    "POST /api/v1/mcp": (
        "every tool runs as the authenticated caller through the service, which refuses other tenants (test_mcp_server)"
    ),
    "GET /api/v1/mcp": "static 405; no server-initiated stream and no data",
    "DELETE /api/v1/mcp": "static 405; the endpoint is sessionless and holds no data",
}

FLEET_ONLY_ROUTES.update(
    {
        "GET /api/v1/team/invites": "every pending invite on the plane",
        "DELETE /api/v1/team/invites/{invite_id}": "revoke any pending invite on the plane",
        "GET /api/v1/team/users": "the plane-wide account directory",
        "POST /api/v1/team/users/{subject}/disable": "disable an account and revoke its tokens",
        "POST /api/v1/team/users/{subject}/enable": "re-enable an account",
        "PUT /api/v1/team/users/{subject}/rules": "set any account's fleet rules, approval threshold and guarantee",
        "PUT /api/v1/team/users/{subject}/tier": "make any account admin, trusted or user (grants + trusted flag)",
        "POST /api/v1/team/workloads/{workload_id}/reclaim": "checkpoint, stop and requeue any tenant's workload",
        "POST /api/v1/assistant/providers": "add a plane-wide assistant provider",
        "POST /api/v1/assistant/models/discover": "read a provider's model list with a typed or stored key",
        "POST /api/v1/assistant/oauth/openrouter/start": "begin connecting a plane-wide OpenRouter provider",
        "PUT /api/v1/assistant/providers/{provider_id}": "change a plane-wide assistant provider or its key",
        "DELETE /api/v1/assistant/providers/{provider_id}": "remove a plane-wide assistant provider",
        "GET /api/v1/assistant/claude-code/status": "the plane host's Claude Code sign-in state (no email/org)",
        "POST /api/v1/assistant/claude-code/login": "start a Claude Code sign-in on the plane host",
    }
)

ROUTE_CLASSIFICATION = {**CROSS_TENANT_ROUTES, **FLEET_ONLY_ROUTES, **EXEMPT_ROUTES}


# --------------------------------------------------------------------------- harness


def _app():
    return create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32))


def _headers(user: str, roles: str, org: str | None = None, project: str | None = None) -> dict[str, str]:
    headers = {"X-Eyvan-User": user, "X-Eyvan-Roles": roles}
    if org is not None:
        headers["X-Eyvan-Org"] = org
    if project is not None:
        headers["X-Eyvan-Project"] = project
    return headers


ADMIN = _headers("root", "fleet_admin")
ACME = _headers("alice", "operator", "acme", "research")
RIVAL = _headers("mallory", "operator", "rival", "research")


@pytest.fixture
def plane() -> TestClient:
    """One control plane serving two registered, unrelated tenants."""

    client = TestClient(_app())
    for org in ("acme", "rival"):
        client.put(
            f"/api/v1/organizations/{org}",
            json={"owner": f"owner@{org}"},
            headers={**ADMIN, "Idempotency-Key": f"o-{org}"},
        )
        client.put(
            f"/api/v1/organizations/{org}/projects/research",
            json={"owner": f"owner@{org}"},
            headers={**ADMIN, "Idempotency-Key": f"p-{org}"},
        )
    return client


def _submit(client: TestClient, headers: dict[str, str], org: str, key: str) -> str:
    response = client.post(
        "/api/v1/workloads",
        json={
            "org_id": org,
            "project_id": "research",
            "spec": {"name": "train", "execution": {"kind": "command", "command": "python train.py"}},
        },
        headers={**headers, "Idempotency-Key": key},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


# --------------------------------------------------------------------------- the guard


def test_every_route_is_classified() -> None:
    """A route added tomorrow must be classified before this suite goes green.

    Without this, the cross-tenant coverage below is a snapshot: correct on the
    day it was written and silently incomplete from the next route onward. The
    failure message names the route so classifying it is a one-line decision,
    not an investigation.
    """

    live = {
        f"{method} {route.path}"
        for route in _app().routes
        if getattr(route, "methods", None)
        for method in sorted(route.methods - {"HEAD", "OPTIONS"})
    }

    unclassified = sorted(live - set(ROUTE_CLASSIFICATION))
    assert not unclassified, (
        "these routes are not classified for tenant isolation; add each to "
        "CROSS_TENANT_ROUTES, FLEET_ONLY_ROUTES, or EXEMPT_ROUTES with a reason: " + ", ".join(unclassified)
    )

    stale = sorted(set(ROUTE_CLASSIFICATION) - live)
    assert not stale, f"these classified routes no longer exist: {', '.join(stale)}"


# --------------------------------------------------------------------------- workloads


def test_a_tenant_cannot_see_or_touch_another_tenants_workload(plane: TestClient) -> None:
    """Read, cancel, retry, checkpoint, and the diagnostic views around them.

    Cancel and retry matter most: they are unauthenticated denial of service
    against a competitor's training run if the scope check is missing on even
    one of them.
    """

    workload = _submit(plane, ACME, "acme", "w-acme")

    reads = {
        "get": plane.get(f"/api/v1/workloads/{workload}", headers=RIVAL),
        "explain": plane.get(f"/api/v1/workloads/{workload}/explain", headers=RIVAL),
        "logs": plane.get(f"/api/v1/workloads/{workload}/logs", headers=RIVAL),
        "log-page": plane.get(f"/api/v1/workloads/{workload}/log-page", headers=RIVAL),
        "log-stream": plane.get(f"/api/v1/workloads/{workload}/logs/stream", params={"follow": "false"}, headers=RIVAL),
        "progress": plane.get(f"/api/v1/workloads/{workload}/progress", headers=RIVAL),
        "process-status": plane.get(f"/api/v1/workloads/{workload}/process-status", headers=RIVAL),
        "children": plane.get(f"/api/v1/workloads/{workload}/children", headers=RIVAL),
        "endpoints": plane.get(f"/api/v1/workloads/{workload}/endpoints", headers=RIVAL),
        # Refused on the owning project before the record's shape is disclosed,
        # so a rival cannot use the managed-service reads as an oracle for which
        # of another tenant's workloads are services.
        "service-health": plane.get(f"/api/v1/workloads/{workload}/service-health", headers=RIVAL),
        "service-decisions": plane.get(f"/api/v1/workloads/{workload}/service-decisions", headers=RIVAL),
    }
    writes = {
        "cancel": plane.post(f"/api/v1/workloads/{workload}/cancel", headers={**RIVAL, "Idempotency-Key": "x1"}),
        "retry": plane.post(f"/api/v1/workloads/{workload}/retry", headers={**RIVAL, "Idempotency-Key": "x2"}),
        "checkpoint": plane.post(f"/api/v1/workloads/{workload}/checkpoint", headers={**RIVAL, "Idempotency-Key": "x3"}),
    }

    for name, response in {**reads, **writes}.items():
        assert response.status_code in {403, 404}, f"{name} leaked across tenants: {response.status_code}"

    # And the owner is unaffected -- a wall that also blocks the tenant is a
    # broken plane, not a secure one.
    assert plane.get(f"/api/v1/workloads/{workload}", headers=ACME).status_code == 200


def test_a_tenant_cannot_read_another_tenants_tracked_runs(plane: TestClient) -> None:
    """Metric history is training data: the scope check must precede any tracker lookup.

    Authorization is decided before the proxy resolves a tracker, so a refused
    request is 403/404 whether or not a tracker is configured -- a 503 here would
    mean the scope was never checked.
    """

    scope = {"org_id": "acme", "project_id": "research"}
    attempts = {
        "runs": plane.get("/api/v1/tracking/runs", params=scope, headers=RIVAL),
        "run": plane.get("/api/v1/tracking/runs/wl_acme", params=scope, headers=RIVAL),
        "series": plane.get(
            "/api/v1/tracking/runs/wl_acme/series", params={**scope, "key": "loss", "x": "step"}, headers=RIVAL
        ),
        "histograms": plane.get(
            "/api/v1/tracking/runs/wl_acme/histograms", params={**scope, "key": "adv"}, headers=RIVAL
        ),
        "episodes": plane.get("/api/v1/tracking/runs/wl_acme/episodes", params=scope, headers=RIVAL),
        "curve": plane.get("/api/v1/tracking/runs/wl_acme/episodes/curve", params=scope, headers=RIVAL),
        "breakdown": plane.get("/api/v1/tracking/runs/wl_acme/episodes/breakdown", params=scope, headers=RIVAL),
        "rollouts": plane.get("/api/v1/tracking/runs/wl_acme/rollouts", params=scope, headers=RIVAL),
        "rollout": plane.get("/api/v1/tracking/runs/wl_acme/rollouts/a1-r0-x-0-0", params=scope, headers=RIVAL),
        "completions": plane.get("/api/v1/tracking/runs/wl_acme/completions", params=scope, headers=RIVAL),
        "completion-stats": plane.get("/api/v1/tracking/runs/wl_acme/completions/stats", params=scope, headers=RIVAL),
        "completion": plane.get("/api/v1/tracking/runs/wl_acme/completions/a1-r0-x-0-0", params=scope, headers=RIVAL),
        "tables": plane.get("/api/v1/tracking/runs/wl_acme/tables", params=scope, headers=RIVAL),
        "media": plane.get("/api/v1/tracking/runs/wl_acme/media", params=scope, headers=RIVAL),
        "blob": plane.get(f"/api/v1/tracking/runs/wl_acme/blobs/{'a' * 64}", params=scope, headers=RIVAL),
    }
    for name, response in attempts.items():
        assert response.status_code in {403, 404}, f"tracking {name} leaked across tenants: {response.status_code}"

    # The owner passes authorization; without a tracker the honest answer is 503, never 403.
    own = plane.get("/api/v1/tracking/runs", params=scope, headers=ACME)
    assert own.status_code not in {401, 403, 404}, own.text


def test_listing_workloads_never_returns_another_tenants(plane: TestClient) -> None:
    """The quietest leak: a list route that ignores the scope it was given."""

    _submit(plane, ACME, "acme", "w-acme")
    _submit(plane, RIVAL, "rival", "w-rival")

    listed = plane.get("/api/v1/workloads", params={"org_id": "acme", "project_id": "research"}, headers=RIVAL)
    own = plane.get("/api/v1/workloads", params={"org_id": "rival", "project_id": "research"}, headers=RIVAL)

    assert listed.status_code == 403
    assert [item["org_id"] for item in own.json()] == ["rival"]


def test_a_tenant_cannot_submit_into_another_tenants_scope(plane: TestClient) -> None:
    """Placing work into another tenant spends their quota and their budget."""

    submissions = {
        "workload": plane.post(
            "/api/v1/workloads",
            json={
                "org_id": "acme",
                "project_id": "research",
                "spec": {"name": "t", "execution": {"kind": "command", "command": "echo hi"}},
            },
            headers={**RIVAL, "Idempotency-Key": "s1"},
        ),
        "estimate": plane.post(
            "/api/v1/workloads/estimate",
            json={
                "org_id": "acme",
                "project_id": "research",
                "spec": {"name": "t", "execution": {"kind": "command", "command": "echo hi"}},
            },
            headers=RIVAL,
        ),
        "batch": plane.post(
            "/api/v1/batches",
            json={
                "org_id": "acme",
                "project_id": "research",
                "count": 2,
                "spec": {"name": "t", "execution": {"kind": "command", "command": "echo hi"}},
            },
            headers={**RIVAL, "Idempotency-Key": "s2"},
        ),
        "dag": plane.post(
            "/api/v1/dags",
            json={
                "org_id": "acme",
                "project_id": "research",
                "nodes": [
                    {
                        "name": "a",
                        "spec": {"name": "a", "execution": {"kind": "command", "command": "echo hi"}},
                    }
                ],
            },
            headers={**RIVAL, "Idempotency-Key": "s3"},
        ),
        "group": plane.post(
            "/api/v1/groups",
            json={
                "org_id": "acme",
                "project_id": "research",
                "members": [
                    {
                        "name": "a",
                        "spec": {"name": "a", "execution": {"kind": "command", "command": "echo hi"}},
                    }
                ],
            },
            headers={**RIVAL, "Idempotency-Key": "s4"},
        ),
    }

    for name, response in submissions.items():
        assert response.status_code in {403, 422}, f"{name} placed work into another tenant: {response.status_code}"


# --------------------------------------------------------------------------- governance


def test_governance_reads_and_writes_stay_inside_a_tenant(plane: TestClient) -> None:
    """Quotas, budgets and policies decide what a tenant may spend.

    Reading them is competitive intelligence; writing them is either a denial of
    service or a way to lift your own ceiling using someone else's admin path.
    """

    attempts = {
        "quota-list": plane.get("/api/v1/quotas", params={"org_id": "acme"}, headers=RIVAL),
        "quota-current": plane.get(
            "/api/v1/quotas/current", params={"org_id": "acme", "project_id": "research"}, headers=RIVAL
        ),
        "quota-write": plane.put(
            "/api/v1/quotas",
            json={"org_id": "acme", "project_id": "research", "limits": {"cpu": 999}},
            headers={**RIVAL, "Idempotency-Key": "q1"},
        ),
        "quota-delete": plane.request(
            "DELETE", "/api/v1/quotas", params={"org_id": "acme", "project_id": "research"}, headers=RIVAL
        ),
        "budget-list": plane.get("/api/v1/budgets", params={"org_id": "acme"}, headers=RIVAL),
        "budget-get": plane.get("/api/v1/budgets/acme/research", headers=RIVAL),
        "budget-status": plane.get("/api/v1/budgets/acme/research/status", headers=RIVAL),
        "budget-write": plane.put(
            "/api/v1/budgets/acme/research",
            json={"limit": 1_000_000},
            headers={**RIVAL, "Idempotency-Key": "b1"},
        ),
        "policy-list": plane.get("/api/v1/policies", params={"org_id": "acme"}, headers=RIVAL),
        "policy-get": plane.get("/api/v1/policies/acme/research", headers=RIVAL),
        "policy-write": plane.put(
            "/api/v1/policies/acme/research",
            json={"max_replicas": 1},
            headers={**RIVAL, "Idempotency-Key": "p1"},
        ),
    }

    for name, response in attempts.items():
        assert response.status_code in {403, 404, 422}, f"{name} crossed tenants: {response.status_code}"


def test_a_tenant_cannot_ask_which_fleets_another_tenant_may_run_on(plane: TestClient) -> None:
    """Runnable-fleet names describe what capacity a competitor has been given."""

    theirs = plane.get("/api/v1/fleets", params={"org_id": "acme", "project_id": "research"}, headers=RIVAL)
    ours = plane.get("/api/v1/fleets", params={"org_id": "rival", "project_id": "research"}, headers=RIVAL)

    assert theirs.status_code == 403
    assert ours.status_code == 200


def test_governance_resolves_per_tenant_rather_than_globally(plane: TestClient) -> None:
    """A limit stored under the right key but resolved globally binds the wrong tenant.

    That failure passes every isolation assertion above -- nothing there submits
    enough to reach a limit -- so it needs its own check. This asserts the
    resolution, which is where the hierarchy is chosen; the scheduler's *use* of
    the resolved limit is covered by ``test_scheduler``, because quotas and
    budgets are scheduling constraints here rather than admission gates: a
    workload over its limit is accepted, queued, and explained.
    """

    plane.put(
        "/api/v1/quotas",
        json={"org_id": "acme", "project_id": "research", "limits": {"cpu": 4}},
        headers={**ADMIN, "Idempotency-Key": "q-acme"},
    )
    plane.put(
        "/api/v1/budgets/acme/research",
        json={"limit": 100.0, "hard_limit": True},
        headers={**ADMIN, "Idempotency-Key": "b-acme"},
    )

    acme_quota = plane.get("/api/v1/quotas/current", params={"org_id": "acme", "project_id": "research"}, headers=ACME)
    rival_quota = plane.get(
        "/api/v1/quotas/current", params={"org_id": "rival", "project_id": "research"}, headers=RIVAL
    )
    rival_budget = plane.get("/api/v1/budgets/rival/research", headers=RIVAL)

    assert acme_quota.status_code == 200
    acme_scopes = {entry["scope"]: entry["quota"] for entry in acme_quota.json()["scopes"]}
    assert acme_scopes["project"]["limits"]["cpu"] == 4.0
    # The neighbouring tenant inherits neither. A global resolution would hand
    # rival acme's ceiling, which reads as a working limit and is the wrong one.
    #
    # Asserted as "declares nothing" rather than as a 404: the read now answers
    # an unconfigured scope instead of refusing it, so the isolation property
    # has to be stated directly. It is also the stronger assertion -- a 404
    # would still pass if the route leaked another tenant's limit and then
    # failed for an unrelated reason.
    assert rival_quota.status_code == 200
    assert all(entry["quota"] is None for entry in rival_quota.json()["scopes"])
    assert rival_budget.status_code == 404


# --------------------------------------------------------------------------- audit and artifacts


def test_the_audit_trail_is_not_readable_across_tenants(plane: TestClient) -> None:
    """An audit stream is a record of who did what, and it is the last thing to leak.

    ``/events/stream`` is checked because a route that authorizes *inside* the
    response generator has already committed a 200 by the time it refuses.
    """

    _submit(plane, ACME, "acme", "w-acme")

    events = plane.get("/api/v1/events", params={"org_id": "acme", "project_id": "research"}, headers=RIVAL)
    stream = plane.get(
        "/api/v1/events/stream",
        params={"org_id": "acme", "project_id": "research", "follow": "false"},
        headers=RIVAL,
    )

    assert events.status_code == 403
    assert stream.status_code == 403, "the stream authorized after committing a response"


def test_artifact_listings_stay_inside_a_tenant(plane: TestClient) -> None:
    """Object names carry model names, dataset names, and experiment structure."""

    scope = {"org_id": "acme", "project_id": "research"}
    uri = "eyvan://results/private-result"
    attempts = {
        "listing": plane.get("/api/v1/artifacts", params=scope, headers=RIVAL),
        "usage": plane.get("/api/v1/artifacts/usage", params=scope, headers=RIVAL),
        "inventory": plane.get("/api/v1/artifacts/inventory", params=scope, headers=RIVAL),
        "reindex": plane.post(
            "/api/v1/artifacts/inventory/reindex",
            json={**scope, "acknowledge_scope": "acme/research"},
            headers=RIVAL,
        ),
        "collectable": plane.get("/api/v1/artifacts/collectable", params=scope, headers=RIVAL),
        "references": plane.get("/api/v1/artifacts/references", params={**scope, "uri": uri}, headers=RIVAL),
        "hold": plane.post(
            "/api/v1/artifacts/holds",
            json={**scope, "uri": uri, "reason": "probe"},
            headers=RIVAL,
        ),
        "release": plane.delete("/api/v1/artifacts/holds", params={**scope, "uri": uri}, headers=RIVAL),
    }

    assert {name: response.status_code for name, response in attempts.items()} == {name: 403 for name in attempts}


# --------------------------------------------------------------------------- tenancy itself


def test_the_tenancy_surface_does_not_leak_between_tenants(plane: TestClient) -> None:
    """Registration and grants are how a tenant would acquire the rest."""

    attempts = {
        "read-org": plane.get("/api/v1/organizations/acme", headers=RIVAL),
        "list-projects": plane.get("/api/v1/organizations/acme/projects", headers=RIVAL),
        "create-project": plane.put(
            "/api/v1/organizations/acme/projects/sneaky",
            json={"owner": "mallory"},
            headers={**RIVAL, "Idempotency-Key": "t1"},
        ),
        "grant": plane.post(
            "/api/v1/grants",
            json={
                "subject": "mallory",
                "grant": {"role": "project_operator", "org_id": "acme", "project_id": "research"},
            },
            headers={**RIVAL, "Idempotency-Key": "t2"},
        ),
        "revoke": plane.request(
            "DELETE",
            "/api/v1/grants",
            params={"subject": "alice", "role": "project_operator", "org_id": "acme", "project_id": "research"},
            headers=RIVAL,
        ),
    }

    for name, response in attempts.items():
        assert response.status_code in {403, 404}, f"{name} crossed tenants: {response.status_code}"


# --------------------------------------------------------------------------- capacity


def test_capacity_administration_is_refused_to_a_tenant_operator(plane: TestClient) -> None:
    """Every fleet-only route, refused to a tenant principal.

    Listed rather than assumed: "this needs fleet administration" is a claim,
    and a claim worth failing on the day it stops being true.
    """

    attempts = {
        "cells": plane.get("/api/v1/cells", headers=RIVAL),
        "pools": plane.get("/api/v1/capacity-pools", headers=RIVAL),
        "orphans": plane.get("/api/v1/capacity/orphans", headers=RIVAL),
        "billing": plane.get("/api/v1/capacity/billing", headers=RIVAL),
        "warmth": plane.get("/api/v1/capacity/warmth", headers=RIVAL),
        # Provider quota describes the cloud project the fleet's pools declare,
        # which is fleet property rather than any tenant's.
        "quota": plane.get("/api/v1/capacity/quota", headers=RIVAL),
        "leases": plane.get("/api/v1/leases", headers=RIVAL),
        "fleet-events": plane.get("/api/v1/fleet/events", headers=RIVAL),
        "reconcile": plane.post("/api/v1/system/reconcile", headers=RIVAL),
        "create-pool": plane.post("/api/v1/capacity-pools", json={}, headers={**RIVAL, "Idempotency-Key": "c1"}),
        "drain-cell": plane.post("/api/v1/cells/any/drain", headers={**RIVAL, "Idempotency-Key": "c2"}),
        "delete-cell": plane.request("DELETE", "/api/v1/cells/any", headers=RIVAL),
    }

    for name, response in attempts.items():
        assert response.status_code in {403, 404, 422}, f"{name} reached fleet capacity: {response.status_code}"


def test_a_tenant_sees_only_its_own_registration_in_a_listing(plane: TestClient) -> None:
    """Narrowed listings are the other half: a 200 that omits, rather than a 403."""

    organizations = plane.get("/api/v1/organizations", headers=RIVAL)

    assert organizations.status_code == 200
    assert [item["id"] for item in organizations.json()] == ["rival"]


# --------------------------------------------------------------------------- federation


@pytest.fixture
def home_plane() -> TestClient:
    """A plane with an identity, serving the same two tenants.

    Federation needs ``plane_id``: a plane with no identity cannot be a home
    plane and refuses to issue. Kept separate from ``plane`` so the isolation
    tests above keep asserting against the default deployment, which has no
    identity and no federation.
    """

    client = TestClient(
        create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32, plane_id="us-plane"))
    )
    for org in ("acme", "rival"):
        client.put(
            f"/api/v1/organizations/{org}",
            json={"owner": f"owner@{org}"},
            headers={**ADMIN, "Idempotency-Key": f"o-{org}"},
        )
        client.put(
            f"/api/v1/organizations/{org}/projects/research",
            json={"owner": f"owner@{org}"},
            headers={**ADMIN, "Idempotency-Key": f"p-{org}"},
        )
    return client


def _grant_acme_a_budget_and_a_peer(plane: TestClient) -> str:
    """Give acme a budget and cut a peer a slice of it. Returns the grant id."""

    plane.put(
        "/api/v1/budgets/acme/research",
        json={"limit": 1000.0},
        headers={**ADMIN, "Idempotency-Key": "b-acme"},
    )
    plane.put(
        "/api/v1/federation/peers/eu-plane",
        json={"endpoint": "https://eu.example"},
        headers=ADMIN,
    )
    issued = plane.post(
        "/api/v1/federation/budget-grants",
        json={
            "org_id": "acme",
            "project_id": "research",
            "grantee_plane_id": "eu-plane",
            "amount": 250.0,
            "expires_at": "2099-01-01T00:00:00Z",
        },
        headers={**ADMIN, "Idempotency-Key": "g-acme"},
    )
    assert issued.status_code == 201, issued.text
    return issued.json()["id"]


def test_a_tenant_cannot_read_another_tenants_federated_budget(home_plane: TestClient) -> None:
    """A grant is a tenant's name attached to a sum of money.

    An unnarrowed listing would make every tenant's budget posture -- how much
    headroom they have and which planes hold it -- readable by anyone with a
    token, which is the same disclosure the organization listing already
    refuses.
    """

    _grant_acme_a_budget_and_a_peer(home_plane)

    rival = home_plane.get("/api/v1/federation/budget-grants", headers=RIVAL)

    assert rival.status_code == 200
    assert rival.json() == []


def test_a_fleet_admin_sees_the_grant_the_tenant_cannot(home_plane: TestClient) -> None:
    """The narrowing must not be indistinguishable from the feature being broken."""

    grant_id = _grant_acme_a_budget_and_a_peer(home_plane)

    listed = home_plane.get("/api/v1/federation/budget-grants", headers=ADMIN)

    assert [item["id"] for item in listed.json()] == [grant_id]


def test_an_operator_cannot_issue_revoke_or_register_a_peer(home_plane: TestClient) -> None:
    """Issuing a grant is spending authority, not reading it.

    A tenant that could cut its own slices could hand a peer plane its entire
    limit, which is exactly the bound federation exists to hold.
    """

    grant_id = _grant_acme_a_budget_and_a_peer(home_plane)

    attempts = {
        "register-peer": home_plane.put(
            "/api/v1/federation/peers/hostile", json={"endpoint": "https://hostile.example"}, headers=RIVAL
        ),
        "list-peers": home_plane.get("/api/v1/federation/peers", headers=RIVAL),
        "issue": home_plane.post(
            "/api/v1/federation/budget-grants",
            json={
                "org_id": "acme",
                "project_id": "research",
                "grantee_plane_id": "eu-plane",
                "amount": 1.0,
                "expires_at": "2099-01-01T00:00:00Z",
            },
            headers={**RIVAL, "Idempotency-Key": "hostile"},
        ),
        "revoke": home_plane.request("DELETE", f"/api/v1/federation/budget-grants/{grant_id}", headers=RIVAL),
    }

    for name, response in attempts.items():
        assert response.status_code in {403, 404, 422}, f"{name} reached federation: {response.status_code}"


def test_a_revoked_grant_leaves_the_listing_but_not_the_default_one(home_plane: TestClient) -> None:
    """ "This peer used to hold headroom and stopped" is the first question asked.

    Deleting the record would answer it with silence, and leaving revoked slices
    in the default listing would make an operator add up authority that no longer
    exists.
    """

    grant_id = _grant_acme_a_budget_and_a_peer(home_plane)
    revoked = home_plane.request("DELETE", f"/api/v1/federation/budget-grants/{grant_id}", headers=ADMIN)

    assert revoked.status_code == 200
    assert revoked.json()["state"] == "revoked"
    assert home_plane.get("/api/v1/federation/budget-grants", headers=ADMIN).json() == []
    everything = home_plane.get("/api/v1/federation/budget-grants?include_inactive=true", headers=ADMIN)
    assert [item["id"] for item in everything.json()] == [grant_id]
