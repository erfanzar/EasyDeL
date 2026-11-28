// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import { AGENT_CLIENTS, agentConfig, agentTokenName, mcpUrl } from "./agentConfigs";

const TOKEN = "eyv_secrettoken";
const ORIGIN = "https://plane.example.com/";
const URL = "https://plane.example.com/api/v1/mcp";

describe("agent config templates", () => {
  it("builds the MCP url from the origin", () => {
    expect(mcpUrl(ORIGIN)).toBe(URL);
  });

  it("embeds the token and url in every client's config, plus the CLI line", () => {
    for (const { id } of AGENT_CLIENTS) {
      const config = agentConfig(id, ORIGIN, TOKEN);
      expect(config.text).toContain(TOKEN);
      expect(config.text).toContain(URL);
      expect(config.cli).toBe(`eyvan mcp install --client ${id}`);
      expect(config.cli).not.toContain(TOKEN);
    }
  });

  it("emits valid JSON for Xerxes and Cursor with the expected shapes", () => {
    const xerxes = JSON.parse(agentConfig("xerxes", ORIGIN, TOKEN).text);
    expect(xerxes.eyvan).toEqual({
      transport: "streamable_http",
      url: URL,
      headers: { Authorization: `Bearer ${TOKEN}` },
    });
    const cursor = JSON.parse(agentConfig("cursor", ORIGIN, TOKEN).text);
    expect(cursor.mcpServers.eyvan).toEqual({ url: URL, headers: { Authorization: `Bearer ${TOKEN}` } });
  });

  it("gives Claude Code only the JSON entry, never a command line with the token", () => {
    const { text } = agentConfig("claude-code", ORIGIN, TOKEN);
    expect(text).not.toContain("claude mcp add");
    expect(JSON.parse(text).mcpServers.eyvan).toEqual({
      type: "http",
      url: URL,
      headers: { Authorization: `Bearer ${TOKEN}` },
    });
  });

  it("keeps the token off every command line", () => {
    for (const { id } of AGENT_CLIENTS) {
      const config = agentConfig(id, ORIGIN, TOKEN);
      expect(config.cli).not.toContain(TOKEN);
      expect(config.text).toContain(TOKEN);
      expect(config.text.split("\n").filter((line) => /^\s*(claude|codex|eyvan|curl)\s/.test(line))).toEqual([]);
    }
  });

  it("names tokens agent-<client>-<YYYYMMDD-HHMM> in local time", () => {
    expect(agentTokenName("claude-code", new Date(2026, 10, 17, 9, 5))).toBe("agent-claude-code-20261117-0905");
  });

  it("gives Codex a TOML table", () => {
    const lines = agentConfig("codex", ORIGIN, TOKEN).text.split("\n");
    expect(lines[0]).toBe("[mcp_servers.eyvan]");
    expect(lines[1]).toBe(`url = "${URL}"`);
    expect(lines[2]).toBe(`http_headers = { Authorization = "Bearer ${TOKEN}" }`);
  });
});
