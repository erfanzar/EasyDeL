// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { chromium } from "playwright";
import { build, transformWithEsbuild } from "vite";
import { expect, it } from "vitest";

const processModule = "node:process";
const { platform }: { platform: string } = await import(/* @vite-ignore */ processModule);

it("38/44: mobile drawer focuses visible close, tabs into scope, and scope has a visible focus outline", async (context) => {
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
          "verify:shell": new URL("./components/Shell.tsx", import.meta.url).pathname,
          "verify:styles": new URL("./styles.css", import.meta.url).pathname,
        },
      },
      plugins: [
        {
          name: "verify-shell",
          resolveId: (id) => (id === "verify:entry" ? "\0verify-entry" : undefined),
          load(id) {
            if (id === "\0verify-entry")
              return transformWithEsbuild(
                `
          import {createRoot} from "react-dom/client";
          import {useState} from "react";
          import {Shell} from "verify:shell";
          import "verify:styles";
          function Fixture() {
            const [open, setOpen] = useState(false);
            const scopes = ["a", "b"].map(projectId => ({orgId:"org",projectId,enabled:true}));
            return <Shell route={{section:"runs"}} mobileNavigationOpen={open} onMobileNavigationChange={setOpen}
              onNavigate={()=>{}} onSubmit={()=>{}} sectionAvailability={{}} scopes={scopes} selectedScope={scopes[0]}
              onScopeChange={()=>{}}><button>Background content</button></Shell>;
          }
          createRoot(document.getElementById("root")).render(<Fixture/>);
        `,
                "verify-shell.tsx",
                { loader: "tsx", jsx: "automatic" },
              );
          },
        },
      ],
      build: {
        write: false,
        minify: false,
        rollupOptions: { input: "verify:entry", output: { inlineDynamicImports: true } },
      },
    });
    if (Array.isArray(bundle) || !("output" in bundle)) throw new Error("Missing bundle");
    const js = bundle.output.find((item) => item.type === "chunk");
    const css = bundle.output.find((item) => item.type === "asset" && item.fileName.endsWith(".css"));
    if (!js || js.type !== "chunk" || !css || css.type !== "asset") throw new Error("Missing assets");
    const page = await browser.newPage({ viewport: { width: 800, height: 1000 } });
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
    await page.goto("http://verify.test/");
    const opener = page.getByRole("button", { name: "Open navigation", exact: true });
    await opener.waitFor({ state: "visible" }).catch((error) => {
      throw new Error(`${String(error)}; browser errors: ${errors.join("; ")}`);
    });
    await opener.focus();
    await page.keyboard.press("Enter");
    const close = page.locator(".sidebar-close");
    await close.waitFor({ state: "visible" });
    expect.soft(await close.evaluate((element) => document.activeElement === element)).toBe(true);
    // Explicit focus also lets the indicator assertion reproduce independently of finding 38.
    await close.focus();
    await page.keyboard.press("Tab");
    const scope = page.getByRole("combobox", { name: "Project scope" });
    expect(await scope.evaluate((element) => document.activeElement === element)).toBe(true);
    const outline = await scope.evaluate((element) => ({
      width: getComputedStyle(element).outlineWidth,
      style: getComputedStyle(element).outlineStyle,
    }));
    expect.soft(outline.width).not.toBe("0px");
    expect.soft(outline.style).not.toBe("none");
    await page.keyboard.press("Escape");
  } finally {
    await browser.close();
  }
}, 30000);
