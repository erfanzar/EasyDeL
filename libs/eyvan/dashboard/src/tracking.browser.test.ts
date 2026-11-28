// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// Canvas qualification uses the installed Chrome, never jsdom's drawing stubs.
import { chromium } from "playwright";
import { build, transformWithEsbuild } from "vite";
import { expect, it } from "vitest";

const processModule = "node:process";
const { platform }: { platform: string } = await import(/* @vite-ignore */ processModule);

it("renders real tracking canvases, switches axes/themes, and retains top navigation", async (context) => {
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
          "tracking:app": new URL("./App.tsx", import.meta.url).pathname,
          "tracking:client": new URL("./api/client.ts", import.meta.url).pathname,
          "tracking:styles": new URL("./styles.css", import.meta.url).pathname,
        },
      },
      plugins: [
        {
          name: "tracking-fixture",
          resolveId(id) {
            if (id === "tracking:entry") return "\0tracking-entry";
          },
          load(id) {
            if (id === "\0tracking-entry")
              return transformWithEsbuild(
                `
          import { createRoot } from "react-dom/client";
          import App from "tracking:app";
          import { EyvanApiClient } from "tracking:client";
          import "tracking:styles";
          createRoot(document.getElementById("root")).render(<App client={new EyvanApiClient("/api/v1", true)} />);
        `,
                "tracking-fixture.tsx",
                { loader: "tsx", jsx: "automatic" },
              );
          },
        },
      ],
      build: {
        write: false,
        minify: false,
        rollupOptions: { input: "tracking:entry", output: { inlineDynamicImports: true } },
      },
    });
    if (Array.isArray(bundle) || !("output" in bundle)) throw new Error("Expected inline tracking fixture");
    const script = bundle.output.find((file) => file.type === "chunk");
    const css = bundle.output.find((file) => file.type === "asset" && file.fileName.endsWith(".css"));
    if (!script || script.type !== "chunk" || !css || css.type !== "asset") throw new Error("Missing fixture assets");
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/*", (route) =>
      route.request().isNavigationRequest()
        ? route.fulfill({
            contentType: "text/html",
            body: `<!doctype html><html><head><style>${String(css.source)}</style></head><body><div id="root"></div><script type="module">${script.code}</script></body></html>`,
          })
        : route.abort(),
    );
    await page.goto("http://tracking.test/#/runs/pong-seed-3");
    await page.getByRole("img", { name: "loss/policy chart" }).waitFor();
    expect(await page.locator(".uplot canvas").count()).toBe(4);
    const nav = await page.getByRole("complementary", { name: "Primary navigation" }).boundingBox();
    expect(nav?.height).toBe(48);
    expect(nav?.width).toBe(1440);
    const topbar = await page.locator("header.topbar").boundingBox();
    const meta = await page.locator(".tracking-meta").boundingBox();
    const heading = await page.locator(".run-heading").boundingBox();
    expect(topbar?.y).toBe(48);
    expect(topbar?.height).toBe(48);
    expect(meta && topbar ? meta.y >= topbar.y + topbar.height : false).toBe(true);
    expect(heading && meta ? heading.y >= meta.y + meta.height : false).toBe(true);
    const overflowX = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflowX).toBeLessThanOrEqual(0);
    const style = await page.locator("body").evaluate((element) => ({
      bg: getComputedStyle(element).backgroundColor,
      grid: getComputedStyle(element).backgroundSize,
      font: getComputedStyle(element).fontFamily,
    }));
    expect(style.bg).toBe("rgb(6, 7, 11)");
    expect(style.grid).toBe("48px 48px, 48px 48px");
    expect(style.font).toContain("JetBrains Mono");
    await page.getByRole("combobox", { name: "X axis" }).selectOption("env_steps");
    await page.getByRole("img", { name: "loss/policy chart" }).waitFor();
    await page.getByRole("slider", { name: "EMA smoothing" }).fill("0.7");
    await page.getByRole("button", { name: "Toggle theme" }).click();
    await expect.poll(() => page.locator("html").getAttribute("data-theme")).toBe("light");
    await page.getByRole("tab", { name: "Rollouts", exact: true }).click();
    const tracks = page.getByRole("img", { name: /Per-step tracks/ });
    await tracks.waitFor();
    expect(
      await tracks.evaluate((element) => {
        const canvas = element as HTMLCanvasElement;
        return (
          canvas.width > 200 &&
          canvas
            .getContext("2d")!
            .getImageData(0, 0, canvas.width, canvas.height)
            .data.some((value) => value > 0)
        );
      }),
    ).toBe(true);
    await page.getByRole("slider", { name: "Rollout step" }).fill("30");
    await tracks.focus();
    await page.keyboard.press("ArrowRight");
    expect(await page.getByTestId("rollout-position").textContent()).toContain("Step 31");
    await page.evaluate(() => {
      location.hash = "#/runs/grpo-math/completions";
    });
    await page.getByText(/Finite rewards have near-zero variance/).waitFor();
    await expect.poll(() => page.locator(".uplot canvas").count()).toBe(3);
    await page.getByRole("button", { name: "Expand response 1", exact: true }).click();
    expect(await page.locator(".completion-response pre").first().textContent()).toContain("<b>x</b>");
    expect(await page.locator(".completion-response b").count()).toBe(0);
    await page.getByRole("tab", { name: "Media", exact: true }).click();
    await page.getByRole("button", { name: /Open media Demo frame/ }).click();
    const image = page.getByRole("img", { name: /Demo frame/ });
    await image.waitFor();
    await image.evaluate((element) => (element as HTMLImageElement).decode());
    expect(await image.getAttribute("src")).toMatch(/^blob:/);
    await page.getByRole("button", { name: "Close viewer" }).click();
    await page.getByRole("button", { name: /Open blob/ }).click();
    await page.getByRole("dialog", { name: "Media viewer" }).waitFor();
    await page.getByRole("button", { name: "Close viewer" }).click();
    await page.getByRole("button", { name: "Runs", exact: true }).click();
    await page.getByRole("button", { name: "offline-eval", exact: true }).waitFor();
    expect(await page.locator(".uplot").count()).toBe(0);
    expect(errors).toEqual([]);
  } finally {
    await browser.close();
  }
}, 30000);
