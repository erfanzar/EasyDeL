// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import {
  Activity,
  ChartNoAxesCombined,
  Boxes,
  ChevronRight,
  CircleDollarSign,
  ClipboardList,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  LogOut,
  Menu,
  Moon,
  Sun,
  PanelLeft,
  Plus,
  Receipt,
  RefreshCw,
  ServerCog,
  Settings,
  Sparkles,
  Trophy,
  X,
} from "lucide-react";

import type { AuthIdentity } from "../auth/oidc";
import type { Route, Section, SectionAvailability } from "../routing";
import type { ControlPlaneHealth, EffectiveProjectScope } from "../types";

/** Where the rail's collapsed state is remembered between sessions. */
const RAIL_STORAGE_KEY = "eyvan.rail.collapsed";

interface ShellProps {
  route: Route;
  mobileNavigationOpen: boolean;
  onMobileNavigationChange: (open: boolean) => void;
  onNavigate: (section: Section) => void;
  onSubmit: () => void;
  /** Re-read the whole snapshot now, rather than waiting for the next sweep. */
  onRefresh?: () => void;
  /** How many records each section holds, when the snapshot can say. */
  counts?: Partial<Record<Section, number>>;
  sectionAvailability: SectionAvailability;
  health?: ControlPlaneHealth;
  /** Whether the change stream is attached; undefined in demo mode. */
  live?: boolean;
  identity?: AuthIdentity;
  scopes: EffectiveProjectScope[];
  selectedScope?: EffectiveProjectScope;
  onScopeChange: (scope: EffectiveProjectScope) => void;
  onLogout?: () => void;
  /** The assistant dock, docked right of the main column on every page. */
  assistantDock?: ReactNode;
  assistantOpen?: boolean;
  assistantWidth?: number;
  /** Present when the dock exists; renders the topbar toggle. */
  onToggleAssistant?: () => void;
  children: ReactNode;
}

const navigation: Array<{
  section: Section;
  label: string;
  icon: typeof LayoutDashboard;
}> = [
  { section: "overview", label: "Overview", icon: LayoutDashboard },
  { section: "workloads", label: "Workloads", icon: ClipboardList },
  { section: "runs", label: "Runs", icon: ChartNoAxesCombined },
  { section: "experiments", label: "Experiments", icon: Trophy },
  { section: "research", label: "Research", icon: FlaskConical },
  { section: "fleet", label: "Fleet", icon: ServerCog },
  { section: "capacity", label: "Quota & capacity", icon: Gauge },
  { section: "billing", label: "Billing", icon: Receipt },
  { section: "budgets", label: "Budgets & policies", icon: CircleDollarSign },
  { section: "audit", label: "Audit events", icon: Activity },
  { section: "settings", label: "Settings", icon: Settings },
];

const pageMeta: Record<Section, { title: string; description: string }> = {
  overview: { title: "Operations overview", description: "Global scheduling and capacity state" },
  workloads: { title: "Workloads", description: "Queue, placement, and execution lifecycle" },
  runs: { title: "Runs", description: "Training metrics, evaluation and attempt history" },
  experiments: {
    title: "Experiments",
    description: "Collaborative experiments, speedrun leaderboards, and team notes",
  },
  research: {
    title: "Research",
    description: "Budgeted research plans the assistant proposed; nothing runs until you approve one",
  },
  fleet: { title: "Fleet", description: "Cells, topology, resources, and capacity leases" },
  capacity: {
    title: "Quota & capacity",
    description: "Where provider TPU quota is free right now, and where nobody could tell",
  },
  billing: { title: "Billing", description: "What the fleet has run, and what anybody can say it cost" },
  budgets: { title: "Budgets & policies", description: "Spend controls and scheduling governance" },
  audit: { title: "Audit events", description: "Durable control-plane decision history" },
  assistant: {
    title: "Assistant",
    description: "Sizing, fleets and quota answered from this plane's data; requests to the fleet admins",
  },
  settings: {
    title: "Settings",
    description: "Storage, experiment tracking, assistant providers, team accounts, invites and personal API tokens",
  },
};

/** The rail's own memory. A rail that forgets is a rail somebody re-collapses every morning. */
function initialCollapsed(): boolean {
  try {
    return globalThis.localStorage?.getItem(RAIL_STORAGE_KEY) === "1";
  } catch {
    // Private windows and blocked site data throw here rather than returning
    // null, and an expanded rail is the right answer when nobody can say.
    return false;
  }
}

export function Shell({
  route,
  mobileNavigationOpen,
  onMobileNavigationChange,
  onNavigate,
  onSubmit,
  onRefresh,
  counts,
  sectionAvailability,
  health,
  live,
  identity,
  scopes,
  selectedScope,
  onScopeChange,
  onLogout,
  assistantDock,
  assistantOpen = false,
  assistantWidth,
  onToggleAssistant,
  children,
}: ShellProps) {
  const drawerRef = useRef<HTMLElement>(null);
  const [collapsed, setCollapsed] = useState(initialCollapsed);
  const [theme, setTheme] = useState(() =>
    typeof document !== "undefined" && document.documentElement.dataset.theme === "light" ? "light" : "dark",
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  const meta = pageMeta[route.section];
  const detailId =
    "workloadId" in route
      ? route.workloadId
      : "compareIds" in route
        ? `${route.compareIds.length} runs`
        : "runId" in route
          ? route.runId
          : "experimentId" in route
            ? route.experimentId
            : "planId" in route
              ? route.planId
              : undefined;
  const connected = health?.status === "ok";
  const initials =
    identity?.displayName
      .split(/\s+/)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase())
      .join("") || "ME";

  useEffect(() => {
    try {
      globalThis.localStorage?.setItem(RAIL_STORAGE_KEY, collapsed ? "1" : "0");
    } catch {
      // Remembering the rail is a convenience, never a requirement.
    }
  }, [collapsed]);

  useEffect(() => {
    if (!mobileNavigationOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onMobileNavigationChange(false);
    };
    globalThis.addEventListener("keydown", onKeyDown);
    drawerRef.current?.querySelector<HTMLButtonElement>("button.sidebar-close")?.focus();
    return () => globalThis.removeEventListener("keydown", onKeyDown);
  }, [mobileNavigationOpen, onMobileNavigationChange]);

  const chooseSection = (section: Section) => {
    onNavigate(section);
    onMobileNavigationChange(false);
  };

  const shellStyle = assistantWidth ? ({ "--dock-width": `${assistantWidth}px` } as CSSProperties) : undefined;

  return (
    <div className={`app-shell ${assistantOpen && assistantDock ? "app-shell-docked" : ""}`} style={shellStyle}>
      {mobileNavigationOpen && (
        <button
          className="drawer-scrim"
          type="button"
          aria-label="Close navigation"
          onClick={() => onMobileNavigationChange(false)}
        />
      )}
      <aside
        className={`sidebar ${mobileNavigationOpen ? "sidebar-open" : ""} ${collapsed ? "sidebar-collapsed" : ""}`}
        ref={drawerRef}
        aria-label="Primary navigation"
      >
        <div className="brand-row">
          <span className="brand-mark" aria-hidden="true">
            <Boxes size={13} strokeWidth={2} />
          </span>
          <span className="brand-name rail-label">Eyvan</span>
          <button
            className="icon-button rail-toggle"
            type="button"
            aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
            aria-pressed={collapsed}
            title={collapsed ? "Expand navigation" : "Collapse navigation"}
            onClick={() => setCollapsed((current) => !current)}
          >
            <PanelLeft size={15} strokeWidth={1.6} />
          </button>
          <button
            className="icon-button sidebar-close"
            type="button"
            aria-label="Close navigation"
            title="Close navigation"
            onClick={() => onMobileNavigationChange(false)}
          >
            <X size={16} />
          </button>
        </div>

        <div className="environment-switcher">
          <span
            className={`environment-dot ${identity?.mode === "demo" ? "environment-dot-demo" : ""}`}
            aria-hidden="true"
          />
          <label>
            <small>Project scope</small>
            <select
              aria-label="Project scope"
              value={selectedScope ? JSON.stringify([selectedScope.orgId, selectedScope.projectId]) : ""}
              disabled={scopes.length < 2}
              onChange={(event) => {
                const scope = scopes.find(
                  (candidate) => JSON.stringify([candidate.orgId, candidate.projectId]) === event.target.value,
                );
                if (scope) onScopeChange(scope);
              }}
            >
              {!selectedScope && <option value="">Authority pending</option>}
              {scopes.map((scope) => (
                <option
                  key={JSON.stringify([scope.orgId, scope.projectId])}
                  value={JSON.stringify([scope.orgId, scope.projectId])}
                >
                  {scope.displayName || `${scope.orgId}/${scope.projectId}`}
                  {!scope.enabled ? " (disabled - history only)" : ""}
                </option>
              ))}
            </select>
          </label>
        </div>

        <nav className="sidebar-nav">
          <span className="nav-label">Control plane</span>
          {navigation
            .filter(({ section }) => sectionAvailability[section] !== false)
            .map(({ section, label, icon: Icon }) => {
              const count = counts?.[section];
              return (
                <button
                  key={section}
                  className={`nav-item ${route.section === section ? "nav-item-active" : ""}`}
                  type="button"
                  title={label}
                  aria-current={route.section === section ? "page" : undefined}
                  onClick={() => chooseSection(section)}
                >
                  <Icon size={15} strokeWidth={1.7} aria-hidden="true" />
                  <span className="rail-label">{label}</span>
                  {/* The count is the reason to look at the row before pressing
                    it. Absent rather than zero when the snapshot has not said:
                    an empty section and an unread one are the same number and
                    opposite facts. */}
                  {count !== undefined && (
                    <span className={`nav-count rail-label${count === 0 ? " nav-count-zero" : ""}`} aria-hidden="true">
                      {count}
                    </span>
                  )}
                </button>
              );
            })}
        </nav>

        <div className="sidebar-footer">
          <div
            className="system-health"
            title={
              live
                ? "Live -- updating as changes commit"
                : health
                  ? `${health.service} reported ${health.status}`
                  : "Awaiting health response"
            }
          >
            <span className={`health-indicator ${connected ? "" : "health-indicator-pending"}`} aria-hidden="true" />
            <span className="nav-connection" aria-hidden="true">
              {live ? "Live" : health?.status === "demo" ? "Demo" : connected ? "Connected" : "Unknown"}
            </span>
            <span className="rail-label">
              <strong>
                {connected
                  ? "Control plane connected"
                  : health?.status === "demo"
                    ? "Demo mode"
                    : "Checking control plane"}
              </strong>
              <small>
                {/* A stream that stopped and a fleet where nothing is happening look
                    identical from the page. The first is a bug; the second is a Tuesday. */}
                {live
                  ? "Live -- updating as changes commit"
                  : health
                    ? `${health.service} reported ${health.status}`
                    : "Awaiting health response"}
              </small>
            </span>
          </div>
          {onLogout ? (
            <button
              className="profile-button"
              type="button"
              aria-label={`Sign out ${identity?.displayName ?? "current user"}`}
              title="Sign out"
              onClick={onLogout}
            >
              <span className="avatar" aria-hidden="true">
                {initials}
              </span>
              <span className="rail-label">
                <strong>{identity?.displayName ?? "Authenticated user"}</strong>
                <small>{identity?.detail ?? "OIDC session"}</small>
              </span>
              <LogOut size={14} aria-hidden="true" />
            </button>
          ) : (
            <div className="profile-button profile-static">
              <span className="avatar" aria-hidden="true">
                {initials}
              </span>
              <span className="rail-label">
                <strong>{identity?.displayName ?? "Unauthenticated"}</strong>
                <small>{identity?.detail ?? "No identity"}</small>
              </span>
            </div>
          )}
        </div>
      </aside>

      <div className={`main-column ${collapsed ? "main-collapsed" : ""}`}>
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            type="button"
            aria-label="Open navigation"
            title="Open navigation"
            onClick={() => onMobileNavigationChange(true)}
          >
            <Menu size={18} />
          </button>
          {/* A breadcrumb, not a title block. The section name is context the
              reader already has from the rail; the record ID is the one thing
              the rail cannot say, and the thing they copy into a terminal. */}
          <nav className="page-title" aria-label="Breadcrumb">
            <h1>
              {detailId
                ? "compareIds" in route
                  ? "Compare runs"
                  : route.section === "runs"
                    ? "Run detail"
                    : route.section === "experiments"
                      ? "Experiment"
                      : route.section === "research"
                        ? "Research plan"
                        : "Workload detail"
                : meta.title}
            </h1>
            {detailId && (
              <>
                <ChevronRight size={13} strokeWidth={1.7} aria-hidden="true" />
                <span className="crumb-target" title={detailId}>
                  {detailId}
                </span>
              </>
            )}
          </nav>
          <div className="topbar-actions">
            <button
              className="icon-button"
              type="button"
              aria-label="Toggle theme"
              title="Toggle theme"
              onClick={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
            >
              {theme === "dark" ? <Sun size={14} aria-hidden="true" /> : <Moon size={14} aria-hidden="true" />}
            </button>
            {onToggleAssistant && (
              <button
                id="assistant-dock-toggle"
                className={`button button-secondary dock-toggle ${assistantOpen ? "dock-toggle-active" : ""}`}
                type="button"
                aria-label="Assistant"
                aria-pressed={assistantOpen}
                aria-controls="assistant-dock"
                aria-keyshortcuts="Control+J Meta+J"
                title="Assistant (Ctrl/⌘+J)"
                onClick={onToggleAssistant}
              >
                <Sparkles size={14} aria-hidden="true" />
                <span className="button-label">Assistant</span>
              </button>
            )}
            {onRefresh && (
              <button className="button button-secondary" type="button" aria-label="Refresh" onClick={onRefresh}>
                <RefreshCw size={14} aria-hidden="true" />
                <span className="button-label">Refresh</span>
              </button>
            )}
            {identity?.capabilities.submitWorkloads && (
              <button className="button button-primary" type="button" aria-label="Submit workload" onClick={onSubmit}>
                <Plus size={14} strokeWidth={2} aria-hidden="true" />
                <span className="button-label">Submit workload</span>
              </button>
            )}
          </div>
        </header>
        <main className="page-content">{children}</main>
      </div>
      {assistantDock}
    </div>
  );
}
