// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// Browser layout is deliberately not tested with jsdom. No server, API or
// provider is contacted: Playwright fulfills the one fixture document locally.
import { chromium, type Browser, type Page } from "playwright";
import { build, transformWithEsbuild } from "vite";
import { afterAll, beforeAll, describe, expect, it, type TestContext } from "vitest";

// Keep Node-only fixture helpers local without adding Node typings to the
// dashboard's browser-only type configuration or changing its dependencies.
const fsModule = "node:fs";
const processModule = "node:process";
const { readFileSync }: { readFileSync: (path: URL, encoding: "utf8") => string } = await import(
  /* @vite-ignore */ fsModule
);
const { platform }: { platform: string } = await import(/* @vite-ignore */ processModule);
const stylesheet = readFileSync(new URL("./styles.css", import.meta.url), "utf8");
const fixture = `
import { useState } from "react";
import { createRoot } from "react-dom/client";
import { Shell } from "writer013:shell";
import { ConfirmDialog } from "writer013:confirm";
import { StagePipeline } from "writer013:pipeline";
import { SubmitWorkloadModal } from "writer013:submit";
import "writer013:styles";
const scopes = [
  { orgId: "acme", projectId: "research", displayName: "Research", registered: true, enabled: true, acceptsNewWork: true, roles: ["project_viewer"], permissions: ["read"] },
  { orgId: "acme", projectId: "platform", displayName: "Platform", registered: true, enabled: true, acceptsNewWork: true, roles: ["project_viewer"], permissions: ["read"] },
];
function Harness() {
  const [open, setOpen] = useState(false);
  const [scope, setScope] = useState(scopes[0]);
  const [error, setError] = useState(null);
  if (location.pathname === "/confirm") return <>
    <button hidden data-testid="inject-error" onClick={() => setError("Agent refused to drain this cell. " + "The active lease must finish before retrying. ".repeat(20))}>Inject error</button>
    <ConfirmDialog open title="Drain cell" confirmLabel="Drain cell" confirmPhrase="research-cell-123" error={error} onConfirm={() => {}} onCancel={() => {}} consequence={<><p>Running work is allowed to finish. The scheduler will not place new work on this cell.</p><p>The cell stays registered and its capacity is not released. Resume the cell to accept work again.</p></>} />
  </>;
  if (location.pathname === "/form") return <SubmitWorkloadModal open project="research" shapes={[]} onClose={() => {}} onSubmit={async () => { throw new Error("Not submitted by layout test"); }} />;
  if (location.pathname === "/pipeline") return <StagePipeline status="running" timeline={[]} variant="rail" />;
  return <Shell route={{section:"overview"}} mobileNavigationOpen={open} onMobileNavigationChange={setOpen} onNavigate={() => {}} onSubmit={() => {}} sectionAvailability={{fleet:true,audit:true}} scopes={scopes} selectedScope={scope} onScopeChange={setScope}><output data-testid="selected-project">{scope.projectId}</output></Shell>;
}
createRoot(document.getElementById("root")).render(<Harness />);
`;

let browser: Browser | undefined;
let browserUnavailable = "";
let html = "";

beforeAll(async () => {
  try {
    browser = await chromium.launch({
      headless: true,
      // The local host has system Chrome, not a downloaded Playwright browser.
      ...(platform === "darwin" ? { channel: "chrome" } : {}),
      args: ["--disable-background-networking", "--disable-component-update", "--no-default-browser-check"],
    });
  } catch (error) {
    // Source contracts still run where no browser has been installed. A skipped
    // layout test is not browser evidence; qualification must install one.
    browserUnavailable = error instanceof Error ? error.message : String(error);
    return;
  }
  const bundle = await build({
    configFile: false,
    envFile: false,
    root: new URL("..", import.meta.url).pathname,
    logLevel: "silent",
    resolve: {
      alias: {
        react: new URL("../node_modules/react", import.meta.url).pathname,
        "react-dom": new URL("../node_modules/react-dom", import.meta.url).pathname,
        "writer013:shell": new URL("./components/Shell.tsx", import.meta.url).pathname,
        "writer013:confirm": new URL("./components/ConfirmDialog.tsx", import.meta.url).pathname,
        "writer013:pipeline": new URL("./components/StagePipeline.tsx", import.meta.url).pathname,
        "writer013:submit": new URL("./components/SubmitWorkloadModal.tsx", import.meta.url).pathname,
        "writer013:styles": new URL("./styles.css", import.meta.url).pathname,
      },
    },
    plugins: [
      {
        name: "writer013-fixture",
        resolveId(id) {
          if (id === "writer013:entry") return "\0writer013-entry";
        },
        async load(id) {
          if (id === "\0writer013-entry") {
            return transformWithEsbuild(fixture, "writer013-fixture.tsx", { loader: "tsx", jsx: "automatic" });
          }
        },
      },
    ],
    build: {
      write: false,
      minify: false,
      rollupOptions: { input: "writer013:entry", output: { inlineDynamicImports: true } },
    },
  });
  if (Array.isArray(bundle) || !("output" in bundle)) throw new Error("Expected an in-memory fixture bundle");
  const script = bundle.output.find((file) => file.type === "chunk");
  const css = bundle.output.find((file) => file.type === "asset" && file.fileName.endsWith(".css"));
  if (!script || script.type !== "chunk" || !css || css.type !== "asset") {
    throw new Error("Fixture must include real components and the full stylesheet");
  }
  html = `<!doctype html><html><head><style>${String(css.source)}</style></head><body><div id="root"></div><script type="module">${script.code}</script></body></html>`;
}, 30000);

afterAll(async () => {
  await browser?.close();
});

async function withPage(
  context: TestContext,
  path: string,
  viewport: { width: number; height: number },
  collapsed: boolean,
  check: (page: Page) => Promise<void>,
) {
  if (!browser) {
    context.skip(`Real-browser qualification unavailable: ${browserUnavailable}`);
    return;
  }
  const session = await browser.newContext({ viewport });
  const page = await session.newPage();
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/*", (route) =>
    route.request().isNavigationRequest() && route.request().url().startsWith("http://writer-013.test/")
      ? route.fulfill({ contentType: "text/html", body: html })
      : route.abort(),
  );
  if (collapsed) {
    await page.addInitScript(() => globalThis.localStorage.setItem("eyvan.rail.collapsed", "1"));
  }
  try {
    await page.goto(`http://writer-013.test/${path}`);
    await check(page);
    expect(errors).toEqual([]);
  } finally {
    await session.close();
  }
}

describe("writer-013 responsive layout contracts", () => {
  it("does not hide the desktop expansion affordance", () => {
    expect(/\.sidebar-collapsed \.rail-toggle\s*\{\s*display:\s*none;/.test(stylesheet)).toBe(false);
  });

  it("can collapse, expand and recover a persisted desktop collapse", async (context) => {
    await withPage(context, "shell", { width: 1400, height: 800 }, false, async (page) => {
      await page.getByRole("button", { name: "Collapse navigation", exact: true }).click();
      expect(await page.evaluate(() => localStorage.getItem("eyvan.rail.collapsed"))).toBe("1");
      const expand = page.getByRole("button", { name: "Expand navigation", exact: true });
      expect(await expand.isVisible()).toBe(true);
      await page.reload();
      await expand.waitFor({ state: "visible" });
      const bounds = await expand.boundingBox();
      const rail = await page.getByRole("complementary", { name: "Primary navigation" }).boundingBox();
      expect(bounds).not.toBeNull();
      expect(rail).not.toBeNull();
      // Compact navigation is now a full-width, 48px top bar, not a 60px rail.
      expect(rail!.width).toBe(1400);
      expect(rail!.height).toBe(48);
      expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(rail!.x + rail!.width);
      await expand.click();
      expect(await page.getByRole("combobox", { name: "Project scope" }).isVisible()).toBe(true);
      expect(await page.evaluate(() => localStorage.getItem("eyvan.rail.collapsed"))).toBe("0");
    });
  });

  it.for(["persisted", "resized"])(
    "keeps scope switching and labels visible in a %s mobile drawer",
    async (mode, context) => {
      await withPage(
        context,
        "shell",
        { width: mode === "resized" ? 1400 : 800, height: 800 },
        mode === "persisted",
        async (page) => {
          if (mode === "resized") {
            await page.getByRole("button", { name: "Collapse navigation", exact: true }).click();
            await page.setViewportSize({ width: 800, height: 800 });
          }
          await page.getByRole("button", { name: "Open navigation", exact: true }).click();
          const selector = page.getByRole("combobox", { name: "Project scope" });
          expect(await selector.isVisible()).toBe(true);
          const labels = page.locator(".sidebar-nav .rail-label");
          expect(await labels.count()).toBeGreaterThan(0);
          for (const label of await labels.all()) expect(await label.isVisible()).toBe(true);
          expect(await page.getByRole("button", { name: "Expand navigation", exact: true }).isVisible()).toBe(false);
          await selector.selectOption(JSON.stringify(["acme", "platform"]));
          expect(await page.getByTestId("selected-project").textContent()).toBe("platform");
          expect(await page.evaluate(() => localStorage.getItem("eyvan.rail.collapsed"))).toBe("1");
          await page.getByRole("complementary").getByRole("button", { name: "Close navigation", exact: true }).click();
          await page.setViewportSize({ width: 1400, height: 800 });
          await page.getByRole("button", { name: "Expand navigation", exact: true }).click();
          expect(await selector.isVisible()).toBe(true);
        },
      );
    },
  );

  it.for([900, 400])("keeps confirmation actions reachable at %ipx width and 300px height", async (width, context) => {
    await withPage(context, "confirm", { width, height: 300 }, false, async (page) => {
      await page.getByRole("dialog", { name: "Drain cell" }).waitFor();
      const body = page.locator(".modal-body");
      const checkLayout = async () => {
        const sizes = await body.evaluate((element) => ({
          client: element.clientHeight,
          scroll: element.scrollHeight,
        }));
        expect(sizes.client).toBeGreaterThan(0);
        expect(sizes.scroll).toBeGreaterThan(sizes.client);
        const dialog = await page.getByRole("dialog").boundingBox();
        const footer = await page.locator(".modal-footer").boundingBox();
        expect(dialog).not.toBeNull();
        expect(footer).not.toBeNull();
        expect(footer!.y).toBeGreaterThanOrEqual(dialog!.y);
        expect(footer!.y + footer!.height).toBeLessThanOrEqual(dialog!.y + dialog!.height);
        expect(footer!.y + footer!.height).toBeLessThanOrEqual(300);
        await body.evaluate((element) => {
          element.scrollTop = 0;
        });
        await body.hover();
        await page.mouse.wheel(0, 600);
        await expect.poll(() => body.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
      };
      await checkLayout();
      await page.getByRole("textbox").fill("research-cell-123");
      expect(await page.getByRole("button", { name: "Drain cell", exact: true }).isEnabled()).toBe(true);
      await checkLayout();
      await page.getByTestId("inject-error").dispatchEvent("click");
      await page.getByRole("alert").waitFor();
      await checkLayout();
      await page.getByRole("button", { name: "Drain cell", exact: true }).click();
      await page.getByRole("button", { name: "Cancel", exact: true }).click();
    });
  });

  it.for([900, 400])("preserves the submission form scroll region at %ipx width", async (width, context) => {
    await withPage(context, "form", { width, height: 300 }, false, async (page) => {
      await page.getByRole("dialog", { name: "Run a job" }).waitFor();
      const sizes = await page
        .locator(".modal-body")
        .evaluate((element) => ({ client: element.clientHeight, scroll: element.scrollHeight }));
      expect(sizes.client).toBeGreaterThan(0);
      expect(sizes.scroll).toBeGreaterThan(sizes.client);
      const dialog = await page.getByRole("dialog").boundingBox();
      const footer = await page.locator(".modal-footer").boundingBox();
      expect(footer!.y + footer!.height).toBeLessThanOrEqual(dialog!.y + dialog!.height);
      expect(footer!.y + footer!.height).toBeLessThanOrEqual(300);
      await page.getByRole("button", { name: "Cancel", exact: true }).click();
    });
  });

  it("restores collapsed mobile descendants with matching specificity", () => {
    const mobile = stylesheet.slice(
      stylesheet.indexOf("@media (max-width: 900px)"),
      stylesheet.indexOf("@media (max-width: 700px)"),
    );
    expect(mobile.includes(".sidebar.sidebar-collapsed .rail-label")).toBe(true);
    expect(/\.sidebar\.sidebar-collapsed \.environment-switcher,[\s\S]*?display:\s*block;/.test(mobile)).toBe(true);
  });

  it("bounds non-form and form modal bodies using flex shrink allocation", () => {
    const modal = stylesheet.match(/^\.modal \{([^}]*)\}/m)?.[1] ?? "";
    const body = stylesheet.match(/^\.modal-body \{([^}]*)\}/m)?.[1] ?? "";
    const form = stylesheet.match(/^\.modal form \{([^}]*)\}/m)?.[1] ?? "";
    expect(/display:\s*flex;/.test(modal)).toBe(true);
    expect(/flex-direction:\s*column;/.test(modal)).toBe(true);
    expect(/min-height:\s*0;/.test(body)).toBe(true);
    expect(/overflow-y:\s*auto;/.test(body)).toBe(true);
    expect(/min-height:\s*0;/.test(form)).toBe(true);
  });

  it("places base pipeline tracks before their responsive overrides", () => {
    const base = stylesheet.indexOf("grid-template-columns: repeat(8, minmax(0, 1fr));");
    expect(base).toBeGreaterThanOrEqual(0);
    expect(base).toBeLessThan(stylesheet.indexOf("@media (max-width: 1280px)"));
  });

  it.for([
    [1400, 8],
    [1000, 4],
    [600, 2],
  ])("renders %ipx pipeline width with %i grid tracks", async ([width, tracks], context) => {
    await withPage(context, "pipeline", { width, height: 800 }, false, async (page) => {
      const pipeline = page.getByRole("list", { name: "Dispatch pipeline" });
      await pipeline.waitFor();
      expect(await pipeline.locator("li").count()).toBe(8);
      const columns = await pipeline.evaluate(
        (element) => getComputedStyle(element).gridTemplateColumns.split(" ").length,
      );
      expect(columns).toBe(tracks);
    });
  });
});
