// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Check, Copy, KeyRound, Terminal, UserPlus } from "lucide-react";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import { CopyOnce } from "../components/CopyOnce";
import { ErrorCard } from "../components/ErrorCard";
import { MetricCard } from "../components/MetricCard";
import { ReclaimBadge } from "../components/ReclaimBanner";
import { percent } from "../lib/format";
import type {
  GovernanceSummary,
  GovernanceUserSummary,
  TeamInvite,
  TeamInviteCreated,
  TeamInviteInput,
  TeamTier,
  TeamToken,
  TeamTokenCreated,
  TeamUser,
  TeamUserRules,
} from "../types";
import { ConnectAgentPanel } from "./ConnectAgentPanel";

const TIERS: { id: TeamTier; label: string }[] = [
  { id: "admin", label: "Admin" },
  { id: "trusted", label: "Trusted" },
  { id: "user", label: "User" },
];

type BorrowSetting = "" | "allow" | "deny";

export interface TeamClient {
  listTeamUsers: (signal?: AbortSignal) => Promise<TeamUser[]>;
  setTeamUserDisabled: (subject: string, disabled: boolean) => Promise<TeamUser>;
  /** Optional so older callers and tests that never edit rules still type-check. */
  setTeamUserRules?: (subject: string, rules: Partial<TeamUserRules>) => Promise<TeamUser>;
  setTeamUserTier?: (
    subject: string,
    tier: TeamTier,
    orgId: string | null,
    projectId: string | null,
  ) => Promise<TeamUser>;
  /** Optional 404-safe governance summary; older planes and minimal test doubles omit it. */
  getGovernanceSummary?: (signal?: AbortSignal) => Promise<GovernanceSummary | null>;
  listTeamInvites: (signal?: AbortSignal) => Promise<TeamInvite[]>;
  createTeamInvite: (input: TeamInviteInput) => Promise<TeamInviteCreated>;
  revokeTeamInvite: (inviteId: string) => Promise<TeamInvite>;
  listTeamTokens: (signal?: AbortSignal) => Promise<TeamToken[]>;
  createTeamToken: (name: string, expiresInDays: number | null) => Promise<TeamTokenCreated>;
  revokeTeamToken: (tokenId: string) => Promise<TeamToken>;
}

interface UsersPageProps {
  client: TeamClient;
  /** Fleet administrators manage accounts and invites; everyone manages their own tokens. */
  canAdminister: boolean;
  defaultOrg: string;
  defaultProject: string;
  onNotify?: (message: string) => void;
}

const ROLES = [
  "project_submitter",
  "project_operator",
  "project_viewer",
  "project_admin",
  "org_admin",
  "fleet_admin",
] as const;

const message = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);

/** Build a single pasteable CLI onboarding command from an invite response. */
export function inviteLoginCommand(inviteUrl: string, token: string): string {
  let planeUrl = inviteUrl;
  try {
    const parsed = new URL(inviteUrl);
    const basePath = parsed.pathname.replace(/\/+$/, "");
    planeUrl = `${parsed.origin}${basePath}`;
  } catch {
    planeUrl = inviteUrl.split("#", 1)[0].replace(/\/+$/, "");
  }
  return `eyvan login ${planeUrl} --invite ${token}`;
}

async function copyText(value: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Fall back when Clipboard API is restricted.
  }
  try {
    const area = document.createElement("textarea");
    area.value = value;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(area);
    return ok;
  } catch {
    return false;
  }
}

/** One line a person can read: what this account may do without asking. */
export function rulesSummary(rules: TeamUserRules | undefined): string {
  if (!rules) return "no rules";
  const parts = [
    rules.max_tpu_units != null ? `max ${rules.max_tpu_units} TPU` : null,
    rules.max_cpu_units != null ? `max ${rules.max_cpu_units} CPU` : null,
    rules.max_gpu_units != null ? `max ${rules.max_gpu_units} GPU` : null,
    rules.allowed_fleets ? `fleets: ${rules.allowed_fleets.join(", ") || "none"}` : null,
    rules.approval_tpu_threshold !== null ? `approval above ${rules.approval_tpu_threshold} TPU` : null,
    rules.guaranteed_tpu_units !== null ? `guaranteed ${rules.guaranteed_tpu_units} TPU` : null,
    rules.can_borrow === true ? "can borrow" : rules.can_borrow === false ? "no borrowing" : null,
  ].filter(Boolean);
  return parts.join("; ") || "no rules";
}

function tierBorrowBadge(
  tier: TeamTier | string | undefined,
  explicitBorrow: boolean | null | undefined,
): { label: string; tooltip: string; variant: "admin" | "borrow" | "own" } | null {
  if (!tier && explicitBorrow == null) return null;
  if (tier === "admin") {
    if (explicitBorrow === false) {
      return {
        label: "Admin · borrow denied",
        tooltip: "Fleet admin. Idle capacity borrowing is explicitly denied by rule.",
        variant: "own",
      };
    }
    return {
      label: "Admin",
      tooltip:
        "Fleet admin — manages accounts, quotas, and fleets; may borrow idle TPU capacity above guaranteed units.",
      variant: "admin",
    };
  }
  if (tier === "trusted") {
    if (explicitBorrow === false) {
      return {
        label: "Trusted · own quota",
        tooltip:
          "Trusted tier, but idle capacity borrowing is explicitly denied by rule; limited to own guaranteed quota.",
        variant: "own",
      };
    }
    return {
      label: "Trusted · can borrow",
      tooltip:
        "Trusted tier — may borrow idle TPU capacity above guaranteed units; borrowed jobs get a graceful checkpoint window when reclaimed.",
      variant: "borrow",
    };
  }
  if (explicitBorrow === true) {
    return {
      label: "User · can borrow",
      tooltip:
        "User tier with an explicit rule allowing idle TPU capacity borrowing above guaranteed units (reclaimable).",
      variant: "borrow",
    };
  }
  return {
    label: "User · own quota",
    tooltip: "User tier — limited to own guaranteed TPU quota; cannot borrow idle capacity unless allowed by rule.",
    variant: "own",
  };
}

function borrowToSelect(value: boolean | null | undefined): BorrowSetting {
  if (value === true) return "allow";
  if (value === false) return "deny";
  return "";
}

function selectToBorrow(value: BorrowSetting): boolean | null {
  if (value === "allow") return true;
  if (value === "deny") return false;
  return null;
}

const optionalCount = (value: string, fractional = false): number | null => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || (!fractional && !Number.isInteger(parsed)) || parsed < 0 || parsed > 1_000_000)
    throw new Error(`"${value}" must be a ${fractional ? "number" : "whole number"} of units between 0 and 1,000,000`);
  return parsed;
};

function UsageGuaranteeCell({ user, summary }: { user: TeamUser; summary?: GovernanceUserSummary }) {
  const guaranteed = summary?.guaranteed_tpu_units ?? user.rules?.guaranteed_tpu_units ?? null;
  const maxCap = summary?.max_tpu_units ?? user.rules?.max_tpu_units ?? null;

  if (!summary) {
    return (
      <span className="usage-cell-empty muted" data-testid={`usage-${user.subject}`}>
        {guaranteed !== null ? `${guaranteed} TPU guaranteed` : "—"}
      </span>
    );
  }

  const inUse = Math.max(0, summary.in_use_tpu_units);
  const borrowed = Math.max(0, Math.min(summary.borrowed_tpu_units, inUse));
  const own = Math.max(0, inUse - borrowed);
  const scale = Math.max(inUse, guaranteed ?? 0, maxCap ?? 0, 1);
  const ownPct = (own / scale) * 100;
  const borrowedPct = (borrowed / scale) * 100;
  const guaranteeMarkPct =
    guaranteed !== null && guaranteed > 0 && guaranteed < scale ? (guaranteed / scale) * 100 : null;
  const ariaParts = [
    `${user.subject} TPU usage: ${inUse} in use`,
    guaranteed !== null ? `${guaranteed} guaranteed` : "no guarantee",
    borrowed > 0 ? `${borrowed} borrowed` : "0 borrowed",
    maxCap !== null ? `max ${maxCap}` : null,
  ].filter(Boolean);

  return (
    <div className="usage-bar-cell" data-testid={`usage-${user.subject}`}>
      <div className="usage-bar-numbers">
        <span>
          <strong className="monospace">{inUse}</strong>
          {guaranteed !== null ? (
            <span className="muted"> / {guaranteed} TPU</span>
          ) : (
            <span className="muted"> TPU</span>
          )}
        </span>
        {borrowed > 0 && <span className="usage-borrowed-tag">+{borrowed} borrowed</span>}
        {maxCap !== null && <span className="usage-cap-hint">cap {maxCap}</span>}
      </div>
      <div className="usage-bar-track" role="img" aria-label={ariaParts.join(", ")}>
        {own > 0 && <span className="usage-bar-own" style={{ width: `${ownPct}%` }} />}
        {borrowed > 0 && (
          <span className="usage-bar-borrowed" style={{ left: `${ownPct}%`, width: `${borrowedPct}%` }} />
        )}
        {guaranteeMarkPct !== null && (
          <span
            className="usage-bar-guarantee-mark"
            style={{ left: `${guaranteeMarkPct}%` }}
            title={`Guaranteed quota: ${guaranteed} TPU`}
          />
        )}
      </div>
    </div>
  );
}

function RulesEditor({
  user,
  onSave,
  onCancel,
}: {
  user: TeamUser;
  onSave: (rules: Partial<TeamUserRules>) => Promise<void>;
  onCancel: () => void;
}) {
  // Compare against the rules the admin opened, not a later refresh of the row.
  const [current] = useState(user.rules);
  const [initialBorrow] = useState<boolean | null>(() => user.rules?.can_borrow ?? user.can_borrow ?? null);
  const [denyAll, setDenyAll] = useState(current?.allowed_fleets?.length === 0);
  const [fleets, setFleets] = useState(current?.allowed_fleets?.join(", ") ?? "");
  const [threshold, setThreshold] = useState(current?.approval_tpu_threshold?.toString() ?? "");
  const [guarantee, setGuarantee] = useState(current?.guaranteed_tpu_units?.toString() ?? "");
  const [canBorrow, setCanBorrow] = useState<BorrowSetting>(() => borrowToSelect(initialBorrow));
  const [maxTpu, setMaxTpu] = useState(current?.max_tpu_units?.toString() ?? "");
  const [maxCpu, setMaxCpu] = useState(current?.max_cpu_units?.toString() ?? "");
  const [maxGpu, setMaxGpu] = useState(current?.max_gpu_units?.toString() ?? "");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setProblem(null);
    const rules: Partial<TeamUserRules> = {};
    try {
      const previous = current?.allowed_fleets ?? null;
      if (denyAll !== (previous?.length === 0) || fleets !== (previous?.join(", ") ?? "")) {
        const names = [
          ...new Set(
            fleets
              .split(",")
              .map((name) => name.trim())
              .filter(Boolean),
          ),
        ];
        const next = denyAll ? [] : names.length ? names : null;
        const sameFleets =
          next === previous ||
          (next !== null &&
            previous !== null &&
            next.length === previous.length &&
            next.every((name) => previous.includes(name)));
        if (!sameFleets) rules.allowed_fleets = next;
      }
      for (const [key, input] of [
        ["approval_tpu_threshold", threshold],
        ["guaranteed_tpu_units", guarantee],
        ["max_tpu_units", maxTpu],
        ["max_cpu_units", maxCpu],
        ["max_gpu_units", maxGpu],
      ] as const) {
        const value = optionalCount(input, key === "max_cpu_units");
        if (value !== (current?.[key] ?? null)) rules[key] = value;
      }
      const nextBorrow = selectToBorrow(canBorrow);
      if (nextBorrow !== initialBorrow) {
        rules.can_borrow = nextBorrow;
      }
    } catch (failure) {
      setProblem(message(failure, "Check the numbers"));
      return;
    }
    if (!Object.keys(rules).length) {
      onCancel();
      return;
    }
    setBusy(true);
    try {
      await onSave(rules);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      className="form-grid form-grid-two"
      aria-label={`Rules for ${user.subject}`}
      onSubmit={(event) => void save(event)}
    >
      <label className="field">
        <span>Allowed fleets (comma separated; empty = any)</span>
        <input disabled={denyAll} value={fleets} onChange={(event) => setFleets(event.target.value)} />
      </label>
      <label className="field">
        <span>Deny all fleets</span>
        <input type="checkbox" checked={denyAll} onChange={(event) => setDenyAll(event.target.checked)} />
      </label>
      <label className="field">
        <span>Needs approval above (TPU units; empty = never)</span>
        <input inputMode="numeric" value={threshold} onChange={(event) => setThreshold(event.target.value)} />
      </label>
      <label className="field">
        <span>Guaranteed TPU units (above it is borrowed; empty = none)</span>
        <input inputMode="numeric" value={guarantee} onChange={(event) => setGuarantee(event.target.value)} />
      </label>
      <label className="field">
        <span>Can borrow idle capacity</span>
        <select value={canBorrow} onChange={(event) => setCanBorrow(event.target.value as BorrowSetting)}>
          <option value="">Default by tier</option>
          <option value="allow">Allow</option>
          <option value="deny">Deny</option>
        </select>
        <small className="field-hint">Defaults to Allow for Admin and Trusted, and own quota only for User.</small>
      </label>
      <label className="field">
        <span>Max TPU units (empty = no limit)</span>
        <input inputMode="numeric" value={maxTpu} onChange={(event) => setMaxTpu(event.target.value)} />
        <small className="field-hint">Caps count units in use or queued across all projects.</small>
      </label>
      <label className="field">
        <span>Max CPU units (empty = no limit)</span>
        <input inputMode="decimal" value={maxCpu} onChange={(event) => setMaxCpu(event.target.value)} />
      </label>
      <label className="field">
        <span>Max GPU units (empty = no limit)</span>
        <input inputMode="numeric" value={maxGpu} onChange={(event) => setMaxGpu(event.target.value)} />
      </label>
      <div className="field">
        <span>&nbsp;</span>
        <div>
          <button className="button button-primary" type="submit" disabled={busy}>
            Save rules
          </button>{" "}
          <button className="button button-secondary" type="button" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
      {problem && (
        <p className="panel-note" role="alert">
          {problem}
        </p>
      )}
    </form>
  );
}
const when = (value: string | null) => (value ? new Date(value).toLocaleString() : "—");

export function UsersPage({ client, canAdminister, defaultOrg, defaultProject, onNotify }: UsersPageProps) {
  const [users, setUsers] = useState<TeamUser[]>([]);
  const [invites, setInvites] = useState<TeamInvite[]>([]);
  const [tokens, setTokens] = useState<TeamToken[]>([]);
  const [governance, setGovernance] = useState<GovernanceSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);

  const [email, setEmail] = useState("");
  const [subject, setSubject] = useState("");
  const [tier, setTier] = useState<TeamTier>("user");
  // Empty = use the tier; a role chosen under "Advanced" replaces it (never both).
  const [role, setRole] = useState<string>("");
  const [inviteBusy, setInviteBusy] = useState(false);
  const [createdInvite, setCreatedInvite] = useState<{ url: string; token: string; subject: string } | null>(null);
  const [copiedLoginCmd, setCopiedLoginCmd] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof globalThis.setTimeout> | null>(null);

  const [editingRules, setEditingRules] = useState<string | null>(null);
  const [confirmAdmin, setConfirmAdmin] = useState<string | null>(null);

  const [tokenName, setTokenName] = useState("");
  const [tokenBusy, setTokenBusy] = useState(false);
  const [newToken, setNewToken] = useState<string | null>(null);

  const reload = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(
    () => () => {
      if (copyTimerRef.current !== null) globalThis.clearTimeout(copyTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    const reads: Promise<unknown>[] = [
      client.listTeamTokens(controller.signal).then((value) => {
        if (!controller.signal.aborted) setTokens(value);
      }),
    ];
    if (canAdminister) {
      reads.push(
        client.listTeamUsers(controller.signal).then((value) => {
          if (!controller.signal.aborted) setUsers(value);
        }),
        client.listTeamInvites(controller.signal).then((value) => {
          if (!controller.signal.aborted) setInvites(value);
        }),
      );
      if (typeof client.getGovernanceSummary === "function") {
        // Optional governance telemetry: degrade quietly when absent on older planes.
        void client
          .getGovernanceSummary(controller.signal)
          .then((summary) => {
            if (!controller.signal.aborted) setGovernance(summary);
          })
          .catch(() => {
            if (!controller.signal.aborted) setGovernance(null);
          });
      }
    }
    Promise.all(reads).catch((failure: unknown) => {
      if (!controller.signal.aborted) setError(message(failure, "Could not read team accounts"));
    });
    return () => controller.abort();
  }, [canAdminister, client, revision]);

  const governanceBySubject = useMemo(() => {
    const map = new Map<string, GovernanceUserSummary>();
    for (const item of governance?.users ?? []) {
      map.set(item.subject, item);
    }
    return map;
  }, [governance]);

  const act = async (action: () => Promise<unknown>) => {
    setError(null);
    try {
      await action();
      reload();
    } catch (failure) {
      setError(message(failure, "The change was not applied"));
    }
  };

  const invite = async (event: FormEvent) => {
    event.preventDefault();
    setInviteBusy(true);
    setError(null);
    setCreatedInvite(null);
    setCopiedLoginCmd(false);
    const base = { email: email.trim(), subject: (subject.trim() || email.split("@", 1)[0]).toLowerCase() };
    try {
      const created = await client.createTeamInvite(
        role
          ? {
              ...base,
              role,
              org_id: role !== "fleet_admin" ? defaultOrg : null,
              project_id: role.startsWith("project_") ? defaultProject : null,
            }
          : {
              ...base,
              tier,
              org_id: tier === "admin" ? null : defaultOrg,
              project_id: tier === "admin" ? null : defaultProject,
            },
      );
      setCreatedInvite({
        url: created.url,
        token: created.token,
        subject: created.invite.subject || base.subject,
      });
      setEmail("");
      setSubject("");
      reload();
    } catch (failure) {
      setError(message(failure, "Could not create the invite"));
    } finally {
      setInviteBusy(false);
    }
  };

  const copyInviteCli = async (command: string) => {
    const ok = await copyText(command);
    if (!ok) return;
    setCopiedLoginCmd(true);
    onNotify?.("Copied CLI login command");
    if (copyTimerRef.current !== null) globalThis.clearTimeout(copyTimerRef.current);
    copyTimerRef.current = globalThis.setTimeout(() => setCopiedLoginCmd(false), 1800);
  };

  const createToken = async (event: FormEvent) => {
    event.preventDefault();
    setTokenBusy(true);
    setError(null);
    setNewToken(null);
    try {
      const created = await client.createTeamToken(tokenName.trim(), null);
      setNewToken(created.token);
      setTokenName("");
      reload();
    } catch (failure) {
      setError(message(failure, "Could not create the token"));
    } finally {
      setTokenBusy(false);
    }
  };

  const guaranteedMembers = (governance?.users ?? []).filter(
    (item) => item.guaranteed_tpu_units !== null && item.guaranteed_tpu_units > 0,
  ).length;
  const loginCommand = createdInvite ? inviteLoginCommand(createdInvite.url, createdInvite.token) : null;

  return (
    <div className="page-stack">
      {error && (
        <ErrorCard
          compact
          error={error}
          context={{ fallbackTitle: "Team action failed" }}
          onDismiss={() => setError(null)}
        />
      )}

      {canAdminister && governance && (
        <section className="metric-grid metric-grid-four" aria-label="Team quota and borrowing summary">
          <MetricCard
            label="Guaranteed TPUs"
            value={governance.total_guaranteed_tpu_units.toLocaleString()}
            detail={
              governance.enabled
                ? `reserved across ${guaranteedMembers} member${guaranteedMembers === 1 ? "" : "s"}`
                : "governance enforcement disabled"
            }
            tone="accent"
          />
          <MetricCard
            label="In use"
            value={governance.total_in_use_tpu_units.toLocaleString()}
            detail={
              governance.total_guaranteed_tpu_units > 0
                ? `${percent(governance.total_in_use_tpu_units, governance.total_guaranteed_tpu_units)}% of guaranteed pool`
                : "active TPU units across team"
            }
            tone="positive"
          />
          <MetricCard
            label="Currently borrowed"
            value={governance.total_borrowed_tpu_units.toLocaleString()}
            delta={governance.total_borrowed_tpu_units > 0 ? "reclaimable" : undefined}
            detail={
              governance.total_borrowed_tpu_units > 0
                ? "idle capacity used above member guarantees"
                : "all active usage within member guarantees"
            }
            tone={governance.total_borrowed_tpu_units > 0 ? "warning" : "neutral"}
          />
          <MetricCard
            label="Active reclaims"
            value={governance.active_reclaims.length.toLocaleString()}
            detail={
              governance.active_reclaims.length > 0 ? "graceful checkpoint countdowns running" : "no active reclaims"
            }
            tone={governance.active_reclaims.length > 0 ? "warning" : "neutral"}
          />
        </section>
      )}

      {canAdminister && governance && governance.active_reclaims.length > 0 && (
        <section className="panel data-panel" aria-label="Active team reclaims">
          <div className="panel-header">
            <div>
              <h2>Active reclaims</h2>
              <p>Borrowed workloads currently finishing checkpoints before returning guaranteed capacity</p>
            </div>
          </div>
          <div className="reclaim-strip-list">
            {governance.active_reclaims.map((item) => (
              <div className="reclaim-strip-row" key={item.workload_id}>
                <div className="reclaim-strip-identity">
                  <strong className="monospace">{item.workload_id}</strong>
                  <small>
                    owner <span className="monospace">{item.owner}</span>
                    {item.requested_by ? (
                      <>
                        {" "}
                        · requested by <span className="monospace">{item.requested_by}</span>
                      </>
                    ) : null}
                  </small>
                </div>
                {item.deadline_at ? (
                  <ReclaimBadge
                    deadlineAt={item.deadline_at}
                    requestedBy={item.requested_by}
                    policy={item.policy}
                    showPolicy
                  />
                ) : (
                  <span className="muted monospace">{item.state}</span>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {canAdminister && (
        <section className="panel data-panel" aria-label="Invite a user">
          <div className="panel-header">
            <div>
              <h2>Invite a user</h2>
              <p>
                They open the link, choose a password and are signed in. Project roles apply to {defaultOrg}/
                {defaultProject}.
              </p>
            </div>
          </div>
          <form className="form-grid form-grid-two" onSubmit={(event) => void invite(event)}>
            <label className="field">
              <span>Email</span>
              <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required />
            </label>
            <label className="field">
              <span>User name (optional)</span>
              <input
                value={subject}
                placeholder="From the email when empty"
                onChange={(event) => setSubject(event.target.value)}
              />
            </label>
            <label className="field">
              <span>Tier</span>
              <select
                aria-label="Tier"
                value={tier}
                onChange={(event) => setTier(event.target.value as TeamTier)}
                disabled={!!role}
              >
                {TIERS.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
              <small className="field-hint">
                Admin manages the plane; Trusted runs jobs freely and can borrow idle TPUs; User stays within own quota
                and runs @eyvan.execute once an admin sets a TPU max.
              </small>
            </label>
            <details className="field">
              <summary>Advanced: custom role</summary>
              <label className="field">
                <span>Role</span>
                <select aria-label="Role" value={role} onChange={(event) => setRole(event.target.value)}>
                  <option value="">Use the tier</option>
                  {ROLES.map((name) => (
                    <option key={name} value={name}>
                      {name.replace("_", " ")}
                    </option>
                  ))}
                </select>
              </label>
            </details>
            <div className="field">
              <span>&nbsp;</span>
              <button className="button button-primary" type="submit" disabled={inviteBusy || !email.trim()}>
                <UserPlus size={16} aria-hidden="true" /> Create invite
              </button>
            </div>
          </form>
          {createdInvite && (
            <div className="invite-handoff-stack">
              <CopyOnce label="Invite link — send it privately" value={createdInvite.url} />
              {loginCommand && (
                <div className="invite-cli-handoff" role="status">
                  <div className="invite-cli-meta">
                    <Terminal size={13} aria-hidden="true" />
                    <span>
                      CLI one-liner for <strong className="monospace">{createdInvite.subject}</strong> — redeems invite
                      &amp; signs in
                    </span>
                  </div>
                  <div className="storage-command-code">
                    <code className="monospace" data-testid="invite-login-command">
                      {loginCommand}
                    </code>
                    <button
                      className="button button-secondary"
                      type="button"
                      onClick={() => void copyInviteCli(loginCommand)}
                    >
                      {copiedLoginCmd ? <Check size={13} aria-hidden="true" /> : <Copy size={13} aria-hidden="true" />}
                      {copiedLoginCmd ? "Copied" : "Copy CLI command"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {canAdminister && (
        <section className="panel data-panel" aria-label="Users">
          <div className="panel-header">
            <div>
              <h2>Users</h2>
              <p>
                Disabling an account signs it out everywhere and revokes its tokens. Rules limit fleets, hold big jobs
                for your approval, and set a guarantee; usage above it is borrowed and can be reclaimed.
              </p>
            </div>
          </div>
          {users.length === 0 ? (
            <p className="panel-note">No accounts yet. Invite someone above.</p>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Email</th>
                    <th>Status</th>
                    <th>Tier &amp; borrowing</th>
                    <th>TPU usage</th>
                    <th>Rules</th>
                    <th>Created</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => {
                    const govUser = governanceBySubject.get(user.subject);
                    const explicitBorrow = user.rules?.can_borrow ?? user.can_borrow ?? govUser?.can_borrow ?? null;
                    const effectiveTier = user.tier ?? govUser?.tier;
                    const badge = tierBorrowBadge(effectiveTier, explicitBorrow);
                    return (
                      <Fragment key={user.subject}>
                        <tr>
                          <td className="monospace">{user.subject}</td>
                          <td>{user.email}</td>
                          <td>{user.disabled ? "disabled" : "active"}</td>
                          <td>
                            <div className="tier-cell">
                              {client.setTeamUserTier && user.tier ? (
                                <select
                                  aria-label={`Tier for ${user.subject}`}
                                  value={user.tier}
                                  onChange={(event) => {
                                    const next = event.target.value as TeamTier;
                                    if (next === "admin") {
                                      // Controlled select: it stays on the old tier until confirmed.
                                      setConfirmAdmin(user.subject);
                                      return;
                                    }
                                    setConfirmAdmin(null);
                                    void act(async () => {
                                      // Called as a method: the real client class needs its `this`.
                                      await client.setTeamUserTier?.(user.subject, next, defaultOrg, defaultProject);
                                    });
                                  }}
                                >
                                  {TIERS.map((item) => (
                                    <option key={item.id} value={item.id}>
                                      {item.label}
                                    </option>
                                  ))}
                                </select>
                              ) : !badge ? (
                                "—"
                              ) : null}
                              {badge && (
                                <span
                                  className={`tier-borrow-badge tier-borrow-${badge.variant}`}
                                  title={badge.tooltip}
                                  data-testid={`tier-badge-${user.subject}`}
                                >
                                  {badge.label}
                                </span>
                              )}
                            </div>
                          </td>
                          <td>
                            <UsageGuaranteeCell user={user} summary={govUser} />
                          </td>
                          <td data-testid={`rules-${user.subject}`}>{rulesSummary(user.rules)}</td>
                          <td>{when(user.created_at)}</td>
                          <td>
                            {client.setTeamUserRules && (
                              <>
                                <button
                                  className="button button-secondary"
                                  type="button"
                                  onClick={() => setEditingRules(editingRules === user.subject ? null : user.subject)}
                                >
                                  Rules
                                </button>{" "}
                              </>
                            )}
                            <button
                              className="button button-secondary"
                              type="button"
                              onClick={() => void act(() => client.setTeamUserDisabled(user.subject, !user.disabled))}
                            >
                              {user.disabled ? "Enable" : "Disable"}
                            </button>
                          </td>
                        </tr>
                        {confirmAdmin === user.subject && (
                          <tr>
                            <td colSpan={8}>
                              <div
                                className="panel-note"
                                role="alertdialog"
                                aria-label={`Confirm admin for ${user.subject}`}
                              >
                                Make {user.subject} a fleet admin? They will be able to manage everyone and every fleet.{" "}
                                <button
                                  className="button button-primary"
                                  type="button"
                                  onClick={() => {
                                    setConfirmAdmin(null);
                                    void act(async () => {
                                      await client.setTeamUserTier?.(user.subject, "admin", defaultOrg, defaultProject);
                                    });
                                  }}
                                >
                                  Confirm
                                </button>{" "}
                                <button
                                  className="button button-secondary"
                                  type="button"
                                  onClick={() => setConfirmAdmin(null)}
                                >
                                  Cancel
                                </button>
                              </div>
                            </td>
                          </tr>
                        )}
                        {editingRules === user.subject && client.setTeamUserRules && (
                          <tr>
                            <td colSpan={8}>
                              <RulesEditor
                                user={user}
                                onCancel={() => setEditingRules(null)}
                                onSave={async (rules) => {
                                  if (!client.setTeamUserRules) return;
                                  await act(async () => {
                                    // Called as a method: the real client class needs its `this`.
                                    await client.setTeamUserRules?.(user.subject, rules);
                                    setEditingRules(null);
                                  });
                                }}
                              />
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {canAdminister && (
        <section className="panel data-panel" aria-label="Invites">
          <div className="panel-header">
            <div>
              <h2>Invites</h2>
            </div>
          </div>
          {invites.length === 0 ? (
            <p className="panel-note">No invites.</p>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Email</th>
                    <th>User</th>
                    <th>Role</th>
                    <th>State</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {invites.map((item) => {
                    const open = !item.revoked && !item.accepted_at && new Date(item.expires_at) > new Date();
                    const state = item.accepted_at
                      ? "accepted"
                      : item.revoked
                        ? "revoked"
                        : open
                          ? `open until ${when(item.expires_at)}`
                          : "expired";
                    return (
                      <tr key={item.id}>
                        <td>{item.email}</td>
                        <td className="monospace">{item.subject}</td>
                        <td>{item.roles.map((grant) => grant.role).join(", ") || "—"}</td>
                        <td>{state}</td>
                        <td>
                          {open && (
                            <button
                              className="button button-secondary"
                              type="button"
                              onClick={() => void act(() => client.revokeTeamInvite(item.id))}
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
        </section>
      )}

      <section className="panel data-panel" aria-label="My API tokens">
        <div className="panel-header">
          <div>
            <h2>My API tokens</h2>
            <p>
              For scripts and CI: set <span className="monospace">EYVAN_TOKEN</span>. On your own machine,{" "}
              <span className="monospace">eyvan login</span> creates one for you.
            </p>
          </div>
        </div>
        <form className="form-grid form-grid-two" onSubmit={(event) => void createToken(event)}>
          <label className="field">
            <span>Token name</span>
            <input value={tokenName} onChange={(event) => setTokenName(event.target.value)} maxLength={64} />
          </label>
          <div className="field">
            <span>&nbsp;</span>
            <button className="button button-primary" type="submit" disabled={tokenBusy || !tokenName.trim()}>
              <KeyRound size={16} aria-hidden="true" /> Create token
            </button>
          </div>
        </form>
        {newToken && <CopyOnce label="New token" value={newToken} />}
        {tokens.length === 0 ? (
          <p className="panel-note">No personal tokens.</p>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Starts with</th>
                  <th>Last used</th>
                  <th>State</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {tokens.map((token) => (
                  <tr key={token.id}>
                    <td>{token.name}</td>
                    <td className="monospace">{token.prefix}…</td>
                    <td>{when(token.last_used_at)}</td>
                    <td>{token.revoked_at ? "revoked" : "active"}</td>
                    <td>
                      {!token.revoked_at && (
                        <button
                          className="button button-secondary"
                          type="button"
                          onClick={() => void act(() => client.revokeTeamToken(token.id))}
                        >
                          Revoke
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <ConnectAgentPanel client={client} onMinted={reload} />
    </div>
  );
}
