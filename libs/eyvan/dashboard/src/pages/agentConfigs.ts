// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

export type AgentClient = "xerxes" | "claude-code" | "codex" | "cursor";

export const AGENT_CLIENTS: { id: AgentClient; label: string }[] = [
  { id: "xerxes", label: "Xerxes" },
  { id: "claude-code", label: "Claude Code" },
  { id: "codex", label: "Codex" },
  { id: "cursor", label: "Cursor" },
];

export interface AgentConfig {
  /** Where the text goes. */
  target: string;
  /** Ready-to-paste text; embeds the token and nothing else secret. */
  text: string;
  /** The equivalent one-command install. */
  cli: string;
}

export const EXPIRY_CHOICES: { days: number | null; label: string }[] = [
  { days: null, label: "Never" },
  { days: 30, label: "30 days" },
  { days: 90, label: "90 days" },
  { days: 365, label: "365 days" },
];

const two = (n: number) => String(n).padStart(2, "0");

/** Unique, recognisable token name: agent-<client>-<YYYYMMDD-HHMM> in local time (<= 64 chars). */
export function agentTokenName(client: AgentClient, now: Date = new Date()): string {
  const stamp = `${now.getFullYear()}${two(now.getMonth() + 1)}${two(now.getDate())}-${two(now.getHours())}${two(now.getMinutes())}`;
  return `agent-${client}-${stamp}`.slice(0, 64);
}

const json = (value: unknown) => JSON.stringify(value, null, 2);

/** The MCP endpoint of the plane the dashboard is served from. */
export const mcpUrl = (origin: string) => `${origin.replace(/\/+$/, "")}/api/v1/mcp`;

export function agentConfig(client: AgentClient, origin: string, token: string): AgentConfig {
  const url = mcpUrl(origin);
  const headers = { Authorization: `Bearer ${token}` };
  const cli = `eyvan mcp install --client ${client}`;
  switch (client) {
    case "xerxes":
      return {
        target: "Xerxes mcp.json",
        text: json({ eyvan: { transport: "streamable_http", url, headers } }),
        cli,
      };
    case "claude-code":
      return {
        target: "Merge into ~/.claude.json",
        text: json({ mcpServers: { eyvan: { type: "http", url, headers } } }),
        cli,
      };
    case "codex":
      return {
        target: "~/.codex/config.toml",
        text: [
          "[mcp_servers.eyvan]",
          `url = ${JSON.stringify(url)}`,
          `http_headers = { Authorization = ${JSON.stringify(`Bearer ${token}`)} }`,
        ].join("\n"),
        cli,
      };
    case "cursor":
      return {
        target: "~/.cursor/mcp.json",
        text: json({ mcpServers: { eyvan: { url, headers } } }),
        cli,
      };
  }
}
