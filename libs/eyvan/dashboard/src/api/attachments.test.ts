// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { afterEach, expect, it, vi } from "vitest";

import { EyvanApiClient } from "./client";
import { mockAttachmentJobs, mockAttachmentPlan } from "./mockData";
import type { AttachmentInput } from "../types";

const input: AttachmentInput = { target: "user@host", kind: "ssh", fleet: "research", project: null, zone: null };
afterEach(() => {
  vi.unstubAllGlobals();
  mockAttachmentJobs.splice(0);
});

it("uses the configured plane base for all four attachment routes without double-prefixing", async () => {
  const plan = mockAttachmentPlan(input);
  const job = { id: "a".repeat(32), status: "running" };
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(new Response(JSON.stringify(plan)))
    .mockResolvedValueOnce(new Response(JSON.stringify(job)))
    .mockResolvedValueOnce(new Response(JSON.stringify(job)))
    .mockResolvedValueOnce(new Response(JSON.stringify({ jobs: [job] })));
  vi.stubGlobal("fetch", fetch);
  const client = new EyvanApiClient("http://plane.test/api/v1", false);
  expect(await client.planAttachment(input)).toEqual(plan);
  expect(await client.startAttachment(input)).toEqual(job);
  expect(await client.getAttachment(job.id)).toEqual(job);
  expect(await client.listAttachments()).toEqual([job]);
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    "http://plane.test/api/v1/attachments/plan",
    "http://plane.test/api/v1/attachments",
    `http://plane.test/api/v1/attachments/${job.id}`,
    "http://plane.test/api/v1/attachments",
  ]);
  for (const [, init] of fetch.mock.calls.slice(0, 2)) {
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(input);
    expect(init.signal).toBeInstanceOf(AbortSignal);
  }
});

it("simulates attachments in demo mode without issuing network requests", async () => {
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  const client = new EyvanApiClient("/api/v1", true);
  expect((await client.planAttachment(input)).fleet).toBe("research");
  const job = await client.startAttachment(input);
  expect(job.status).toBe("succeeded");
  expect((await client.getAttachment(job.id)).result?.fleet).toBe("research");
  expect(await client.listAttachments()).toEqual([job]);
  expect(fetch).not.toHaveBeenCalled();
});
