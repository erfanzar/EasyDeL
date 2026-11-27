// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Plug } from "lucide-react";
import { useState, type FormEvent } from "react";

import { CopyOnce } from "../components/CopyOnce";
import { ErrorCard } from "../components/ErrorCard";
import type { TeamTokenCreated } from "../types";
import { AGENT_CLIENTS, EXPIRY_CHOICES, agentConfig, agentTokenName, type AgentClient } from "./agentConfigs";

export interface ConnectAgentClient {
  createTeamToken?: (name: string, expiresInDays: number | null) => Promise<TeamTokenCreated>;
}

interface Props {
  client: ConnectAgentClient;
  /** Called after a token is minted so the token list can refresh. */
  onMinted?: () => void;
  origin?: string;
}

export function ConnectAgentPanel({ client, onMinted, origin }: Props) {
  const [choice, setChoice] = useState<AgentClient>("xerxes");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState<number | null>(90);
  const [shown, setShown] = useState<{ client: AgentClient; token: string; days: number | null } | null>(null);

  const connect = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setShown(null);
    try {
      const created = await client.createTeamToken?.(agentTokenName(choice), days);
      if (!created) throw new Error("This plane cannot mint tokens");
      setShown({ client: choice, token: created.token, days });
      onMinted?.();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not create the token");
    } finally {
      setBusy(false);
    }
  };

  const config = shown ? agentConfig(shown.client, origin ?? window.location.origin, shown.token) : null;

  return (
    <section className="panel data-panel" aria-label="Connect an agent">
      <div className="panel-header">
        <div>
          <h2>Connect an agent</h2>
          <p>
            Let Xerxes, Claude Code, Codex or Cursor use this plane as you, with your permissions. This creates a
            personal token named <span className="monospace">agent-&lt;client&gt;-&lt;date-time&gt;</span>; revoke it
            above at any time. <span className="monospace">eyvan mcp install</span> mints its own separate token, so you
            do not need both.
          </p>
        </div>
      </div>
      {error && (
        <div className="panel-alert-wrap">
          <ErrorCard
            compact
            error={error}
            context={{ fallbackTitle: "Could not create token" }}
            onDismiss={() => setError(null)}
          />
        </div>
      )}
      <form className="form-grid form-grid-two" onSubmit={(event) => void connect(event)}>
        <label className="field">
          <span>Agent</span>
          <select value={choice} onChange={(event) => setChoice(event.target.value as AgentClient)}>
            {AGENT_CLIENTS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Expires</span>
          <select
            value={days === null ? "never" : String(days)}
            onChange={(event) => setDays(event.target.value === "never" ? null : Number(event.target.value))}
          >
            {EXPIRY_CHOICES.map((item) => (
              <option key={item.label} value={item.days === null ? "never" : String(item.days)}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <div className="field">
          <span>&nbsp;</span>
          <button className="button button-primary" type="submit" disabled={busy}>
            <Plug size={16} aria-hidden="true" /> Create token and show config
          </button>
        </div>
      </form>
      {config && (
        <>
          <p className="panel-note">
            {config.target}. Token expires: {shown?.days == null ? "never" : `in ${shown.days} days`}.
          </p>
          <CopyOnce label="Config" value={config.text} block />
          <p className="panel-note">
            Or on a machine where you are signed in: <code className="monospace">{config.cli}</code>
          </p>
        </>
      )}
    </section>
  );
}
