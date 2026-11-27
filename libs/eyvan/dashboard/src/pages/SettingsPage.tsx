// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { CheckCircle2, X } from "lucide-react";
import { useEffect, useState } from "react";

import { ErrorCard } from "../components/ErrorCard";
import { AssistantProvidersSettings, type AssistantSettingsClient } from "./AssistantProvidersSettings";
import { GovernanceSettingsCard, type GovernanceSettingsClient } from "./GovernanceSettingsCard";
import { StorageTrackingSettings, type StorageSettingsClient } from "./StorageTrackingSettings";
import { UsersPage, type TeamClient } from "./UsersPage";

interface SettingsPageProps {
  client: TeamClient & AssistantSettingsClient & StorageSettingsClient & GovernanceSettingsClient;
  /** Accounts, invites and tokens exist only on a team-mode plane. */
  teamMode: boolean;
  /** Fleet admins configure the assistant's providers. */
  canManageFleet: boolean;
  defaultOrg: string;
  defaultProject: string;
  onProvidersChanged?: () => void;
  onNotify?: (message: string) => void;
  assignLocation?: (url: string) => void;
}

export interface SettingsNotice {
  tone: "ok" | "error";
  text: string;
}

/**
 * The plane finishes OpenRouter's sign-in and lands the browser on
 * `#/settings?assistant=openrouter-connected` or `?assistant_error=<code>`.
 */
export function assistantHashNotice(hash: string): SettingsNotice | null {
  const query = hash.split("?")[1];
  if (!query) return null;
  const params = new URLSearchParams(query);
  if (params.get("assistant") === "openrouter-connected") {
    return { tone: "ok", text: "OpenRouter is connected. Its models are now available in the assistant." };
  }
  const code = params.get("assistant_error");
  if (code !== null) {
    const short = code.replace(/[^\w.-]/g, "").slice(0, 64) || "unknown";
    return { tone: "error", text: `OpenRouter sign-in did not finish (${short}). Try again, or paste an API key.` };
  }
  return null;
}

export function SettingsPage({
  client,
  teamMode,
  canManageFleet,
  defaultOrg,
  defaultProject,
  onProvidersChanged,
  onNotify,
  assignLocation,
}: SettingsPageProps) {
  const [notice, setNotice] = useState<SettingsNotice | null>(null);

  useEffect(() => {
    const location = globalThis.location;
    const found = assistantHashNotice(location?.hash ?? "");
    if (!found) return;
    setNotice(found);
    // A reload must not announce the same sign-in again.
    globalThis.history?.replaceState(globalThis.history.state, "", `${location.pathname}${location.search}#/settings`);
    if (found.tone === "ok") onProvidersChanged?.();
    // Read once, on arrival: the hash is cleaned above, so a rerun finds nothing.
  }, [onProvidersChanged]);

  return (
    <div className="page-stack">
      {notice &&
        (notice.tone === "ok" ? (
          <div className="inline-notice" role="status">
            <CheckCircle2 size={15} aria-hidden="true" />
            <span>{notice.text}</span>
            <button
              className="icon-button icon-button-small"
              type="button"
              aria-label="Dismiss notice"
              onClick={() => setNotice(null)}
            >
              <X size={13} aria-hidden="true" />
            </button>
          </div>
        ) : (
          <ErrorCard
            compact
            title="Provider sign-in failed"
            error={notice.text}
            summary={notice.text}
            onDismiss={() => setNotice(null)}
            onCopied={onNotify}
          />
        ))}
      {canManageFleet && client.getStorageStatus && (
        <StorageTrackingSettings
          client={client}
          defaultOrg={defaultOrg}
          defaultProject={defaultProject}
          onNotify={onNotify}
        />
      )}
      {canManageFleet && (
        <AssistantProvidersSettings client={client} onChanged={onProvidersChanged} assignLocation={assignLocation} />
      )}
      {canManageFleet && (client.listWebhooks || client.listProjects || client.listRoleGrants) && (
        <GovernanceSettingsCard
          client={client}
          defaultOrg={defaultOrg}
          defaultProject={defaultProject}
          onNotify={onNotify}
        />
      )}
      {teamMode && (
        <UsersPage
          client={client}
          canAdminister={canManageFleet}
          defaultOrg={defaultOrg}
          defaultProject={defaultProject}
          onNotify={onNotify}
        />
      )}
      {!canManageFleet && !teamMode && <p className="panel-note">Nothing here is configurable with your role.</p>}
    </div>
  );
}
