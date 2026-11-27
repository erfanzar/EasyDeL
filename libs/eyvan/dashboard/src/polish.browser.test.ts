// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { chromium } from "playwright";
import { build, transformWithEsbuild } from "vite";
import { expect, it } from "vitest";

const processModule = "node:process";
const { platform }: { platform: string } = await import(/* @vite-ignore */ processModule);

it("renders polished CapacityPage IAM diagnostics, project switching, and Storage panel without overflow across desktop/mobile and dark/light themes", async (context) => {
  let browser;
  try {
    browser = await chromium.launch({ headless: true, ...(platform === "darwin" ? { channel: "chrome" } : {}) });
  } catch (error) {
    context.skip(`Real browser unavailable: ${String(error)}`);
    return;
  }

  try {
    const bundle = await build({
      configFile: false,
      envFile: false,
      logLevel: "silent",
      root: new URL("..", import.meta.url).pathname,
      resolve: {
        alias: {
          react: new URL("../node_modules/react", import.meta.url).pathname,
          "react-dom": new URL("../node_modules/react-dom", import.meta.url).pathname,
          "polish:shell": new URL("./components/Shell.tsx", import.meta.url).pathname,
          "polish:capacity": new URL("./pages/CapacityPage.tsx", import.meta.url).pathname,
          "polish:storage": new URL("./pages/StorageTrackingSettings.tsx", import.meta.url).pathname,
          "polish:styles": new URL("./styles.css", import.meta.url).pathname,
        },
      },
      plugins: [
        {
          name: "polish-fixture",
          resolveId: (id) => (id === "polish:entry" ? "\0polish-entry" : undefined),
          load(id) {
            if (id === "\0polish-entry")
              return transformWithEsbuild(
                `
                import { createRoot } from "react-dom/client";
                import { useState } from "react";
                import { Shell } from "polish:shell";
                import { CapacityPage } from "polish:capacity";
                import { StorageTrackingSettings } from "polish:storage";
                import "polish:styles";

                const RAW_IAM = [
                  "gcloud beta quotas info list failed: ERROR: (gcloud.beta.quotas.info.list) PERMISSION_DENIED:",
                  "Permission 'cloudquotas.quotas.get' denied on resource '//cloudquotas.googleapis.com/projects/example-project-a/locations/global/services/tpu.googleapis.com'.",
                  "This command is authenticated as alice@example.com.",
                  "- '@type': type.googleapis.com/google.rpc.ErrorInfo",
                  "  error_info_id: Cg1JQU1fREVOSUVE",
                ].join("\\n");

                const scopes = [{ orgId: "default", projectId: "default", displayName: "default", registered: true, enabled: true, acceptsNewWork: true, roles: ["fleet_admin"], permissions: ["read", "submit", "manage_fleet"] }];

                async function loadQuota(_signal, options) {
                  if (options && options.project === "example-project-b") {
                    return {
                      observed_at: "2026-11-27T10:00:00Z",
                      provider: "gcp",
                      project: "example-project-b",
                      available_projects: ["example-project-a", "example-project-b", "acme-tpus"],
                      status: "ok",
                      refreshed_at: "2026-11-27T09:59:30Z",
                      age_s: 30,
                      cached: false,
                      ttl_s: 120,
                      entries: [{
                        zone: "us-central2-b",
                        region: "us-central2",
                        generation: "v6e",
                        accelerator_type: "v6e-8",
                        capacity_class: "spot",
                        metric: "tpu.googleapis.com/tpu-v6e-preemptible",
                        chips_per_slice: 8,
                        limit_chips: 64,
                        in_use_chips: 16,
                        free_chips: 48,
                        free_slices: 6,
                        state: "known",
                        usage_observed_at: "2026-11-27T09:55:00Z",
                        usage_age_s: 300,
                      }],
                      failures: [],
                    };
                  }
                  return {
                    observed_at: "2026-11-27T10:00:00Z",
                    provider: "gcp",
                    project: "example-project-a",
                    available_projects: ["example-project-a", "example-project-b", "acme-tpus"],
                    status: "unavailable",
                    refreshed_at: "2026-11-27T09:59:30Z",
                    age_s: 30,
                    cached: true,
                    ttl_s: 120,
                    entries: [],
                    failures: [
                      { capacity_class: "spot", reason: RAW_IAM },
                      { capacity_class: "on_demand", reason: RAW_IAM },
                    ],
                  };
                }

                const storageClient = {
                  getStorageStatus: async () => ({
                    configured: true,
                    auto_create: false,
                    config_path: "~/.eyvan/storage.json",
                    project: "example-project-b",
                    monitoring: { bucket: "eyvan-monitoring", uri: "gs://eyvan-monitoring/tracking", location: "us-central1", prefix: "tracking" },
                    regional: { "us-central2": { bucket: "eyvan_us-central2", uri: "gs://eyvan_us-central2", location: "us-central2" } },
                    context: { on_gcp: false, project: "example-project-b", zone: "us-central2-b", region: "us-central2", accelerator_type: null, source: "storage-config" },
                  }),
                  listRuns: async () => [],
                };

                function Fixture() {
                  const [section, setSection] = useState("capacity");
                  return (
                    <Shell
                      route={{ section }}
                      mobileNavigationOpen={false}
                      onMobileNavigationChange={() => {}}
                      onNavigate={setSection}
                      onSubmit={() => {}}
                      counts={{ workloads: 183, fleet: 8, budgets: 0, audit: 1000 }}
                      sectionAvailability={{ capacity: true, settings: true, workloads: true, fleet: true, budgets: true, audit: true }}
                      scopes={scopes}
                      selectedScope={scopes[0]}
                      onScopeChange={() => {}}
                    >
                      {section === "capacity" ? (
                        <CapacityPage loadQuota={loadQuota} />
                      ) : (
                        <StorageTrackingSettings client={storageClient} defaultOrg="default" defaultProject="default" />
                      )}
                    </Shell>
                  );
                }

                createRoot(document.getElementById("root")).render(<Fixture />);
                `,
                "polish-fixture.tsx",
                { loader: "tsx", jsx: "automatic" },
              );
          },
        },
      ],
      build: {
        write: false,
        minify: false,
        rollupOptions: { input: "polish:entry", output: { inlineDynamicImports: true } },
      },
    });

    if (Array.isArray(bundle) || !("output" in bundle)) throw new Error("Missing bundle");
    const js = bundle.output.find((item) => item.type === "chunk");
    const css = bundle.output.find((item) => item.type === "asset" && item.fileName.endsWith(".css"));
    if (!js || js.type !== "chunk" || !css || css.type !== "asset") throw new Error("Missing assets");

    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    page.setDefaultTimeout(5000);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/*", (route) =>
      route.request().isNavigationRequest()
        ? route.fulfill({
            contentType: "text/html",
            body: `<!doctype html><style>${String(css.source)}</style><div id="root"></div><script type="module">${js.code}</script>`,
          })
        : route.abort(),
    );

    await page.goto("http://polish.test/");
    await page.getByText("GCP IAM permission denied").waitFor({ state: "visible" });

    // Single deduplicated error card with collapsed Technical details.
    expect(await page.locator(".error-card").count()).toBe(1);
    expect(await page.locator("details.error-card-details").evaluate((el) => (el as HTMLDetailsElement).open)).toBe(
      false,
    );

    // Zero horizontal overflow on desktop dark & light themes.
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.getByRole("button", { name: "Toggle theme" }).click();
    expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe("light");
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

    // Expanding Technical details reveals raw diagnostic output cleanly.
    await page.getByText("Technical details").click();
    expect(await page.locator("details.error-card-details").evaluate((el) => (el as HTMLDetailsElement).open)).toBe(
      true,
    );
    await page.getByText("Cg1JQU1fREVOSUVE").waitFor({ state: "visible" });

    // Quick-switching project from inside the IAM error card loads healthy quota rows.
    await page.getByRole("button", { name: "Switch to example-project-b" }).click();
    await page.getByText("us-central2-b").waitFor({ state: "visible" });
    expect(await page.locator(".error-card").count()).toBe(0);

    // Settings storage panel renders cleanly without overflow on desktop and mobile.
    await page.getByRole("button", { name: "Settings" }).click();
    await page.getByText("eyvan_us-central2", { exact: true }).waitFor({ state: "visible" });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

    expect(errors).toEqual([]);
  } finally {
    await browser.close();
  }
}, 30000);
