// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";

import { diagnoseError, groupQuotaFailures, stripRawNoise } from "./errorDiagnostics";

const RAW_GCLOUD_IAM_WALL = [
  "gcloud beta quotas info list failed: ERROR: (gcloud.beta.quotas.info.list) PERMISSION_DENIED:",
  "Permission 'cloudquotas.quotas.get' denied on resource",
  "'//cloudquotas.googleapis.com/projects/example-project-a/locations/global/services/tpu.googleapis.com'",
  "(or it may not exist). This command is authenticated as alice@example.com which is the active account",
  "specified by the [core/account] property.",
  "- '@type': type.googleapis.com/google.rpc.ErrorInfo",
  "  domain: cloudquotas.googleapis.com",
  "  metadata:",
  "    permission: cloudquotas.quotas.get",
  "    resource: //cloudquotas.googleapis.com/projects/example-project-a/locations/global/services/tpu.googleapis.com",
  "  reason: IAM_PERMISSION_DENIED",
  "- '@type': type.googleapis.com/google.rpc.DebugInfo",
  "  detail: Cg1JQU1fUEVSTUlTU0lPTg==",
  "  error_info_id: Cg1JQU1fUEVSTUlTU0lPTg==",
  "  troubleshooter_url: https://console.cloud.google.com/iam-admin/troubleshooter",
].join("\n");

describe("errorDiagnostics", () => {
  it("strips protobuf @type, base64 error_info_id, and troubleshooter_url noise", () => {
    const cleaned = stripRawNoise(RAW_GCLOUD_IAM_WALL);
    expect(cleaned).not.toContain("@type");
    expect(cleaned).not.toContain("type.googleapis.com");
    expect(cleaned).not.toContain("error_info_id");
    expect(cleaned).not.toContain("troubleshooter_url");
    expect(cleaned).toContain("PERMISSION_DENIED");
  });

  it("extracts project, account, permission, and IAM remediation command from raw gcloud wall", () => {
    const parsed = diagnoseError(RAW_GCLOUD_IAM_WALL);
    expect(parsed.code).toBe("permission_denied");
    expect(parsed.tone).toBe("warning");
    expect(parsed.title).toBe("GCP IAM permission denied");
    expect(parsed.project).toBe("example-project-a");
    expect(parsed.account).toBe("alice@example.com");
    expect(parsed.permission).toBe("cloudquotas.quotas.get");
    expect(parsed.summary).toContain("alice@example.com");
    expect(parsed.summary).toContain("cloudquotas.quotas.get");
    expect(parsed.summary).toContain("example-project-a");
    expect(parsed.summary).not.toContain("@type");
    expect(parsed.summary).not.toContain("error_info_id");
    expect(parsed.command).toContain("gcloud projects add-iam-policy-binding example-project-a");
    expect(parsed.command).toContain("roles/cloudquotas.viewer");
    expect(parsed.hasTechnicalDetails).toBe(true);
    expect(parsed.raw).toBe(RAW_GCLOUD_IAM_WALL);
    expect(parsed.badges).toEqual(
      expect.arrayContaining([
        { label: "Project", value: "example-project-a" },
        { label: "Account", value: "alice@example.com" },
        { label: "Permission", value: "cloudquotas.quotas.get" },
      ]),
    );
  });

  it("deduplicates identical Spot and On-demand failures into one grouped card", () => {
    const grouped = groupQuotaFailures(
      [
        { capacity_class: "spot", reason: RAW_GCLOUD_IAM_WALL },
        { capacity_class: "on_demand", reason: RAW_GCLOUD_IAM_WALL },
      ],
      { project: "example-project-a" },
    );

    expect(grouped).toHaveLength(1);
    expect(grouped[0].scopeLabel).toBe("Spot & On-demand");
    expect(grouped[0].classes).toEqual(["spot", "on_demand"]);
    expect(grouped[0].diagnostic.summary).toContain("Spot & On-demand quota could not be read");
    expect(grouped[0].diagnostic.badges).toEqual(
      expect.arrayContaining([
        { label: "Scope", value: "Spot & On-demand" },
        { label: "Project", value: "example-project-a" },
        { label: "Account", value: "alice@example.com" },
        { label: "Permission", value: "cloudquotas.quotas.get" },
      ]),
    );
  });

  it("keeps distinct Spot and On-demand root causes as separate cards", () => {
    const grouped = groupQuotaFailures(
      [
        { capacity_class: "spot", reason: RAW_GCLOUD_IAM_WALL },
        {
          capacity_class: "on_demand",
          reason: "provider quota discovery did not answer within 25s (DEADLINE_EXCEEDED)",
        },
      ],
      { project: "example-project-a" },
    );

    expect(grouped).toHaveLength(2);
    expect(grouped[0].scopeLabel).toBe("Spot");
    expect(grouped[0].diagnostic.code).toBe("permission_denied");
    expect(grouped[1].scopeLabel).toBe("On-demand");
    expect(grouped[1].diagnostic.code).toBe("timeout");
  });

  it("prefers structured backend TpuQuotaFailure fields when present", () => {
    const grouped = groupQuotaFailures(
      [
        {
          capacity_class: "spot",
          reason: RAW_GCLOUD_IAM_WALL,
          code: "permission_denied",
          summary: "Account alice@example.com cannot read Cloud Quotas on example-project-a.",
          permission: "cloudquotas.quotas.get",
          account: "alice@example.com",
          remediation: "Switch to example-project-b or run `gcloud auth login`.",
        },
        {
          capacity_class: "on_demand",
          reason: RAW_GCLOUD_IAM_WALL,
          code: "permission_denied",
          summary: "Account alice@example.com cannot read Cloud Quotas on example-project-a.",
          permission: "cloudquotas.quotas.get",
          account: "alice@example.com",
          remediation: "Switch to example-project-b or run `gcloud auth login`.",
        },
      ],
      { project: "example-project-a" },
    );

    expect(grouped).toHaveLength(1);
    expect(grouped[0].diagnostic.summary).toContain(
      "Account alice@example.com cannot read Cloud Quotas on example-project-a.",
    );
    expect(grouped[0].diagnostic.command).toBe("gcloud auth login");
    expect(grouped[0].diagnostic.remediation).toBe("Switch to example-project-b or run `gcloud auth login`.");
  });

  it("classifies disabled GCP APIs, auth expiry, timeouts, connection errors, and HTTP statuses", () => {
    const disabled = diagnoseError(
      "SERVICE_DISABLED: API [cloudquotas.googleapis.com] has not been used in project 'demo-proj'",
    );
    expect(disabled.code).toBe("api_disabled");
    expect(disabled.command).toBe("gcloud services enable cloudquotas.googleapis.com --project=demo-proj");

    const auth = diagnoseError("gcloud auth token lookup failed: Reauthentication failed");
    expect(auth.code).toBe("auth_required");
    expect(auth.command).toBe("gcloud auth login");

    const unreachable = diagnoseError("Unable to reach Eyvan API: Failed to fetch");
    expect(unreachable.code).toBe("connection_refused");
    expect(unreachable.command).toBe("eyvan dashboard");

    const tracker = diagnoseError(Object.assign(new Error("No tracker configured"), { status: 503 }));
    expect(tracker.code).toBe("tracker_unconfigured");
    expect(tracker.command).toBe("eyvan track up --local");

    const serverErr = diagnoseError(
      Object.assign(new Error("Eyvan API request failed with 502: bad gateway"), { status: 502 }),
    );
    expect(serverErr.code).toBe("server_error");
    expect(serverErr.httpStatus).toBe(502);

    const demo = diagnoseError("Provider quota is unavailable in demo mode; no live readings are requested.");
    expect(demo.code).toBe("demo_unavailable");
    expect(demo.tone).toBe("info");
    expect(demo.hasTechnicalDetails).toBe(false);
  });
});
