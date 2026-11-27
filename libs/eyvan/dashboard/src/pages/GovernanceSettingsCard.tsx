// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { FolderPlus, KeyRound, Plus, RefreshCw, Trash2, Webhook } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import type { AuthScope } from "../auth/oidc";
import { ErrorCard } from "../components/ErrorCard";
import { StatusBadge } from "../components/StatusBadge";
import type {
  DurableRoleGrant,
  ProjectRecord,
  ProjectRegisterInput,
  RoleGrantIssueInput,
  RoleGrantRevokeInput,
  WebhookCreateInput,
  WebhookSubscription,
} from "../types";

export interface GovernanceSettingsClient {
  listWebhooks?: (scope?: AuthScope, signal?: AbortSignal) => Promise<WebhookSubscription[]>;
  createWebhook?: (input: WebhookCreateInput, scope?: AuthScope, signal?: AbortSignal) => Promise<WebhookSubscription>;
  deleteWebhook?: (webhookId: string, signal?: AbortSignal) => Promise<WebhookSubscription>;
  listProjects?: (orgId: string, signal?: AbortSignal) => Promise<ProjectRecord[]>;
  registerProject?: (input: ProjectRegisterInput, signal?: AbortSignal) => Promise<ProjectRecord>;
  listRoleGrants?: (
    options?: { subject?: string; orgId?: string },
    signal?: AbortSignal,
  ) => Promise<DurableRoleGrant[]>;
  issueRoleGrant?: (input: RoleGrantIssueInput, signal?: AbortSignal) => Promise<DurableRoleGrant>;
  revokeRoleGrant?: (input: RoleGrantRevokeInput, signal?: AbortSignal) => Promise<void>;
}

interface GovernanceSettingsCardProps {
  client: GovernanceSettingsClient;
  defaultOrg: string;
  defaultProject: string;
  onNotify?: (message: string) => void;
}

const ROLE_CHOICES = [
  "project_viewer",
  "project_submitter",
  "project_operator",
  "project_admin",
  "org_admin",
  "fleet_admin",
] as const;

function splitCsv(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function GovernanceSettingsCard({ client, defaultOrg, defaultProject, onNotify }: GovernanceSettingsCardProps) {
  const [revision, setRevision] = useState(0);

  // Webhooks state
  const [webhooks, setWebhooks] = useState<WebhookSubscription[]>([]);
  const [webhooksLoading, setWebhooksLoading] = useState(Boolean(client.listWebhooks));
  const [webhooksError, setWebhooksError] = useState<string | null>(null);
  const [hookUrl, setHookUrl] = useState("");
  const [hookSecret, setHookSecret] = useState("");
  const [hookEntities, setHookEntities] = useState("");
  const [hookActions, setHookActions] = useState("");
  const [hookBusy, setHookBusy] = useState(false);
  const [hookFormError, setHookFormError] = useState<string | null>(null);
  const [deletingHookId, setDeletingHookId] = useState<string | null>(null);

  // Projects state
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [projId, setProjId] = useState("");
  const [projDisplay, setProjDisplay] = useState("");
  const [projOwner, setProjOwner] = useState("");
  const [projBusy, setProjBusy] = useState(false);
  const [projFormError, setProjFormError] = useState<string | null>(null);

  // Role grants state
  const [grants, setGrants] = useState<DurableRoleGrant[]>([]);
  const [grantsError, setGrantsError] = useState<string | null>(null);
  const [grantSubject, setGrantSubject] = useState("");
  const [grantRole, setGrantRole] = useState<string>("project_operator");
  const [grantProject, setGrantProject] = useState(defaultProject);
  const [grantReason, setGrantReason] = useState("");
  const [grantBusy, setGrantBusy] = useState(false);
  const [grantFormError, setGrantFormError] = useState<string | null>(null);
  const [revokingKey, setRevokingKey] = useState<string | null>(null);

  useEffect(() => {
    if (!client.listWebhooks) return;
    const controller = new AbortController();
    setWebhooksLoading(true);
    setWebhooksError(null);
    client
      .listWebhooks({ orgId: defaultOrg, projectId: defaultProject }, controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) setWebhooks(items);
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setWebhooksError(err instanceof Error ? err.message : "Unable to load webhooks");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setWebhooksLoading(false);
      });
    return () => controller.abort();
  }, [client, defaultOrg, defaultProject, revision]);

  useEffect(() => {
    if (!client.listProjects) return;
    const controller = new AbortController();
    setProjectsError(null);
    client
      .listProjects(defaultOrg, controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) setProjects(items);
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setProjectsError(err instanceof Error ? err.message : "Unable to load projects");
        }
      });
    return () => controller.abort();
  }, [client, defaultOrg, revision]);

  useEffect(() => {
    if (!client.listRoleGrants) return;
    const controller = new AbortController();
    setGrantsError(null);
    client
      .listRoleGrants({ orgId: defaultOrg }, controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) setGrants(items);
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setGrantsError(err instanceof Error ? err.message : "Unable to load role grants");
        }
      });
    return () => controller.abort();
  }, [client, defaultOrg, revision]);

  const handleCreateWebhook = async (event: FormEvent) => {
    event.preventDefault();
    if (!client.createWebhook) return;
    const url = hookUrl.trim();
    const secret = hookSecret.trim();
    if (!url) {
      setHookFormError("Webhook URL is required.");
      return;
    }
    if (secret.length < 16) {
      setHookFormError("Signing secret must be at least 16 characters.");
      return;
    }
    setHookBusy(true);
    setHookFormError(null);
    try {
      const created = await client.createWebhook(
        {
          url,
          secret,
          entity_types: splitCsv(hookEntities),
          actions: splitCsv(hookActions),
        },
        { orgId: defaultOrg, projectId: defaultProject },
      );
      setWebhooks((prev) => [created, ...prev]);
      setHookUrl("");
      setHookSecret("");
      setHookEntities("");
      setHookActions("");
      onNotify?.(`Webhook registered for ${defaultOrg}/${defaultProject}`);
    } catch (err) {
      setHookFormError(err instanceof Error ? err.message : "Unable to register webhook");
    } finally {
      setHookBusy(false);
    }
  };

  const handleDeleteWebhook = async (webhookId: string) => {
    if (!client.deleteWebhook) return;
    setDeletingHookId(webhookId);
    setHookFormError(null);
    try {
      await client.deleteWebhook(webhookId);
      setWebhooks((prev) => prev.filter((item) => item.id !== webhookId));
      onNotify?.("Webhook removed");
    } catch (err) {
      setHookFormError(err instanceof Error ? err.message : "Unable to delete webhook");
    } finally {
      setDeletingHookId(null);
    }
  };

  const handleRegisterProject = async (event: FormEvent) => {
    event.preventDefault();
    if (!client.registerProject) return;
    const id = projId.trim();
    const owner = projOwner.trim();
    if (!id || !owner) {
      setProjFormError("Project ID and owner are required.");
      return;
    }
    setProjBusy(true);
    setProjFormError(null);
    try {
      const created = await client.registerProject({
        orgId: defaultOrg,
        projectId: id,
        displayName: projDisplay.trim() || id,
        owner,
        enabled: true,
      });
      setProjects((prev) => [created, ...prev.filter((p) => p.id !== created.id)]);
      setProjId("");
      setProjDisplay("");
      setProjOwner("");
      onNotify?.(`Project ${defaultOrg}/${created.id} registered`);
    } catch (err) {
      setProjFormError(err instanceof Error ? err.message : "Unable to register project");
    } finally {
      setProjBusy(false);
    }
  };

  const handleIssueGrant = async (event: FormEvent) => {
    event.preventDefault();
    if (!client.issueRoleGrant) return;
    const subject = grantSubject.trim();
    if (!subject) {
      setGrantFormError("Subject is required.");
      return;
    }
    const orgScope = grantRole === "fleet_admin" ? null : defaultOrg;
    const projScope =
      grantRole === "fleet_admin" || grantRole === "org_admin" ? null : grantProject.trim() || defaultProject;
    setGrantBusy(true);
    setGrantFormError(null);
    try {
      const issued = await client.issueRoleGrant({
        subject,
        role: grantRole,
        orgId: orgScope,
        projectId: projScope,
        reason: grantReason.trim(),
      });
      setGrants((prev) => [
        issued,
        ...prev.filter(
          (g) =>
            !(
              g.subject === issued.subject &&
              g.grant.role === issued.grant.role &&
              g.grant.org_id === issued.grant.org_id &&
              g.grant.project_id === issued.grant.project_id
            ),
        ),
      ]);
      setGrantSubject("");
      setGrantReason("");
      onNotify?.(`Granted ${issued.grant.role} to ${issued.subject}`);
    } catch (err) {
      setGrantFormError(err instanceof Error ? err.message : "Unable to issue role grant");
    } finally {
      setGrantBusy(false);
    }
  };

  const handleRevokeGrant = async (grant: DurableRoleGrant) => {
    if (!client.revokeRoleGrant) return;
    const key = `${grant.subject}:${grant.grant.role}:${grant.grant.org_id ?? ""}:${grant.grant.project_id ?? ""}`;
    setRevokingKey(key);
    setGrantFormError(null);
    try {
      await client.revokeRoleGrant({
        subject: grant.subject,
        role: grant.grant.role,
        orgId: grant.grant.org_id ?? "",
        projectId: grant.grant.project_id ?? "",
      });
      setGrants((prev) =>
        prev.filter(
          (item) =>
            !(
              item.subject === grant.subject &&
              item.grant.role === grant.grant.role &&
              (item.grant.org_id ?? "") === (grant.grant.org_id ?? "") &&
              (item.grant.project_id ?? "") === (grant.grant.project_id ?? "")
            ),
        ),
      );
      onNotify?.(`Revoked ${grant.grant.role} from ${grant.subject}`);
    } catch (err) {
      setGrantFormError(err instanceof Error ? err.message : "Unable to revoke role grant");
    } finally {
      setRevokingKey(null);
    }
  };

  const hasWebhooks = Boolean(client.listWebhooks);
  const hasTenancy = Boolean(client.listProjects || client.listRoleGrants);
  if (!hasWebhooks && !hasTenancy) return null;

  return (
    <div className="page-stack">
      {hasWebhooks && (
        <section className="panel data-panel" aria-label="Webhook subscriptions">
          <div className="panel-header">
            <Webhook size={16} aria-hidden="true" />
            <div>
              <h2>Webhooks</h2>
              <p>
                Signed lifecycle event deliveries for{" "}
                <span className="monospace">
                  {defaultOrg}/{defaultProject}
                </span>
              </p>
            </div>
            <button
              className="button button-ghost"
              type="button"
              disabled={webhooksLoading}
              onClick={() => setRevision((v) => v + 1)}
            >
              <RefreshCw className={webhooksLoading ? "spin" : undefined} size={13} aria-hidden="true" />
              Refresh
            </button>
          </div>

          {webhooksError && (
            <div className="panel-alert-wrap">
              <ErrorCard compact error={webhooksError} context={{ fallbackTitle: "Unable to load webhooks" }} />
            </div>
          )}

          {webhooks.length > 0 ? (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>URL</th>
                    <th>Entity types</th>
                    <th>Actions</th>
                    <th aria-label="Delete" />
                  </tr>
                </thead>
                <tbody>
                  {webhooks.map((hook) => (
                    <tr key={hook.id}>
                      <td className="monospace">{hook.url}</td>
                      <td>
                        {hook.entity_types.length ? (
                          hook.entity_types.join(", ")
                        ) : (
                          <span className="muted">All entities</span>
                        )}
                      </td>
                      <td>
                        {hook.actions.length ? hook.actions.join(", ") : <span className="muted">All actions</span>}
                      </td>
                      <td className="numeric">
                        {client.deleteWebhook && (
                          <button
                            type="button"
                            className="button button-ghost"
                            aria-label={`Delete webhook ${hook.url}`}
                            disabled={deletingHookId === hook.id}
                            onClick={() => void handleDeleteWebhook(hook.id)}
                          >
                            <Trash2 size={13} aria-hidden="true" />
                            Delete
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            !webhooksLoading &&
            !webhooksError && <p className="panel-note">No webhook subscriptions configured for this project.</p>
          )}

          {client.createWebhook && (
            <form className="governance-inline-form" onSubmit={(event) => void handleCreateWebhook(event)}>
              <div className="storage-subhead">
                <Plus size={12} aria-hidden="true" />
                <span>Add webhook subscription</span>
              </div>
              {hookFormError && (
                <div className="storage-panel-pad">
                  <div className="form-error" role="alert">
                    {hookFormError}
                  </div>
                </div>
              )}
              <div className="governance-form-grid">
                <label className="field">
                  <span>Endpoint URL</span>
                  <input
                    type="url"
                    aria-label="Webhook URL"
                    placeholder="https://hooks.example.org/eyvan"
                    value={hookUrl}
                    disabled={hookBusy}
                    onChange={(event) => setHookUrl(event.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Signing secret (≥ 16 chars)</span>
                  <input
                    type="password"
                    aria-label="Webhook signing secret"
                    placeholder="min 16 characters"
                    value={hookSecret}
                    disabled={hookBusy}
                    onChange={(event) => setHookSecret(event.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Entity types (optional CSV)</span>
                  <input
                    type="text"
                    aria-label="Webhook entity types"
                    placeholder="workload, budget"
                    value={hookEntities}
                    disabled={hookBusy}
                    onChange={(event) => setHookEntities(event.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Actions (optional CSV)</span>
                  <input
                    type="text"
                    aria-label="Webhook actions"
                    placeholder="workload.failed"
                    value={hookActions}
                    disabled={hookBusy}
                    onChange={(event) => setHookActions(event.target.value)}
                  />
                </label>
              </div>
              <div className="governance-form-actions">
                <button className="button button-primary" type="submit" disabled={hookBusy}>
                  {hookBusy ? "Adding…" : "Add webhook"}
                </button>
              </div>
            </form>
          )}
        </section>
      )}

      {hasTenancy && (
        <section className="panel data-panel" aria-label="Projects and role grants">
          <div className="panel-header">
            <KeyRound size={16} aria-hidden="true" />
            <div>
              <h2>Projects &amp; role grants</h2>
              <p>
                Tenancy projects and durable RBAC grants in organization <span className="monospace">{defaultOrg}</span>
              </p>
            </div>
          </div>

          {client.listProjects && (
            <div className="governance-subblock">
              <div className="storage-subhead">
                <FolderPlus size={12} aria-hidden="true" />
                <span>Projects ({projects.length})</span>
              </div>
              {projectsError && (
                <div className="storage-panel-pad">
                  <div className="form-error" role="alert">
                    {projectsError}
                  </div>
                </div>
              )}
              {projects.length > 0 && (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Project ID</th>
                        <th>Display name</th>
                        <th>Owner</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {projects.map((proj) => (
                        <tr key={`${proj.org_id}/${proj.id}`}>
                          <td className="monospace">{proj.id}</td>
                          <td>{proj.display_name || proj.id}</td>
                          <td>{proj.owner}</td>
                          <td>
                            <StatusBadge
                              value={proj.enabled ? "enabled" : "disabled"}
                              tone={proj.enabled ? "healthy" : "expired"}
                              dot={false}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {client.registerProject && (
                <form className="governance-compact-row" onSubmit={(event) => void handleRegisterProject(event)}>
                  {projFormError && (
                    <div className="form-error governance-row-error" role="alert">
                      {projFormError}
                    </div>
                  )}
                  <label className="field">
                    <span>New project ID</span>
                    <input
                      type="text"
                      aria-label="New project ID"
                      placeholder="e.g. evals"
                      value={projId}
                      disabled={projBusy}
                      onChange={(event) => setProjId(event.target.value)}
                    />
                  </label>
                  <label className="field">
                    <span>Display name</span>
                    <input
                      type="text"
                      aria-label="Project display name"
                      placeholder="Evaluations"
                      value={projDisplay}
                      disabled={projBusy}
                      onChange={(event) => setProjDisplay(event.target.value)}
                    />
                  </label>
                  <label className="field">
                    <span>Owner</span>
                    <input
                      type="text"
                      aria-label="Project owner"
                      placeholder="alice"
                      value={projOwner}
                      disabled={projBusy}
                      onChange={(event) => setProjOwner(event.target.value)}
                    />
                  </label>
                  <button className="button button-secondary" type="submit" disabled={projBusy}>
                    {projBusy ? "Saving…" : "Register project"}
                  </button>
                </form>
              )}
            </div>
          )}

          {client.listRoleGrants && (
            <div className="governance-subblock">
              <div className="storage-subhead">
                <KeyRound size={12} aria-hidden="true" />
                <span>Durable role grants ({grants.length})</span>
              </div>
              {grantsError && (
                <div className="storage-panel-pad">
                  <div className="form-error" role="alert">
                    {grantsError}
                  </div>
                </div>
              )}
              {grants.length > 0 && (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Subject</th>
                        <th>Role</th>
                        <th>Scope</th>
                        <th>Reason</th>
                        <th aria-label="Revoke" />
                      </tr>
                    </thead>
                    <tbody>
                      {grants.map((item) => {
                        const rowKey = `${item.subject}:${item.grant.role}:${item.grant.org_id ?? ""}:${item.grant.project_id ?? ""}`;
                        const scopeLabel = item.grant.org_id
                          ? item.grant.project_id
                            ? `${item.grant.org_id}/${item.grant.project_id}`
                            : `${item.grant.org_id}/*`
                          : "global";
                        return (
                          <tr key={rowKey}>
                            <td className="monospace">{item.subject}</td>
                            <td>
                              <StatusBadge value={item.grant.role} dot={false} />
                            </td>
                            <td className="monospace">{scopeLabel}</td>
                            <td>{item.reason || <span className="muted">—</span>}</td>
                            <td className="numeric">
                              {client.revokeRoleGrant && (
                                <button
                                  type="button"
                                  className="button button-ghost"
                                  aria-label={`Revoke ${item.grant.role} from ${item.subject}`}
                                  disabled={revokingKey === rowKey}
                                  onClick={() => void handleRevokeGrant(item)}
                                >
                                  Revoke
                                </button>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
              {client.issueRoleGrant && (
                <form className="governance-compact-row" onSubmit={(event) => void handleIssueGrant(event)}>
                  {grantFormError && (
                    <div className="form-error governance-row-error" role="alert">
                      {grantFormError}
                    </div>
                  )}
                  <label className="field">
                    <span>Subject</span>
                    <input
                      type="text"
                      aria-label="Grant subject"
                      placeholder="bob"
                      value={grantSubject}
                      disabled={grantBusy}
                      onChange={(event) => setGrantSubject(event.target.value)}
                    />
                  </label>
                  <label className="field">
                    <span>Role</span>
                    <select
                      aria-label="Grant role"
                      value={grantRole}
                      disabled={grantBusy}
                      onChange={(event) => setGrantRole(event.target.value)}
                    >
                      {ROLE_CHOICES.map((role) => (
                        <option key={role} value={role}>
                          {role}
                        </option>
                      ))}
                    </select>
                  </label>
                  {grantRole !== "fleet_admin" && grantRole !== "org_admin" && (
                    <label className="field">
                      <span>Project</span>
                      <input
                        type="text"
                        aria-label="Grant project"
                        value={grantProject}
                        disabled={grantBusy}
                        onChange={(event) => setGrantProject(event.target.value)}
                      />
                    </label>
                  )}
                  <label className="field">
                    <span>Reason</span>
                    <input
                      type="text"
                      aria-label="Grant reason"
                      placeholder="optional"
                      value={grantReason}
                      disabled={grantBusy}
                      onChange={(event) => setGrantReason(event.target.value)}
                    />
                  </label>
                  <button className="button button-secondary" type="submit" disabled={grantBusy}>
                    {grantBusy ? "Granting…" : "Grant role"}
                  </button>
                </form>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
