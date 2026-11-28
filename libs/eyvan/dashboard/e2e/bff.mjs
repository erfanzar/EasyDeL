// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// A real browser, HTTPS and a disposable fake IdP. Never uses saved accounts.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import console from "node:console";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { createInterface } from "node:readline";
import { clearTimeout, setTimeout } from "node:timers";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "playwright";

const { values } = parseArgs({
  options: {
    python: { type: "string" },
    "browser-executable": { type: "string" },
    output: { type: "string" },
  },
});
// Must match SESSION_COOKIE in eyvan/server/api/dashboard_sessions.py. The
// __Host- prefix makes the browser refuse it unless it is Secure, path=/ and
// host-only, so looking for the bare name finds nothing.
const SESSION_COOKIE = "__Host-eyvan_session";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const python = values.python ?? path.join(root, ".venv/bin/python");
const receiptPath = path.resolve(values.output ?? path.join(root, "browser-results/bff.json"));
const receipt = {
  result: "failed",
  checks: [],
  scope: "disposable local HTTPS plane and fake IdP; not deployment qualification",
  limitations: [
    "Browser accepts the fixture's self-signed TLS certificate",
    "Injected verifier uses the fake IdP's local signing key, not production JWKS",
    "No real accounts, ingress, shared-session storage or cloud resources",
  ],
};
await mkdir(path.join(root, ".pytest_cache"), { recursive: true });
const state = await mkdtemp(path.join(root, ".pytest_cache/browser-bff-"));
const env = Object.fromEntries(
  Object.entries(process.env).filter(
    ([key]) => !key.startsWith("EYVAN_") && !key.startsWith("OTEL_") && key !== "PYTHONPATH",
  ),
);
const server = spawn(python, [path.join(root, "tests/browser_bff_server.py"), "--state-dir", state], {
  cwd: root,
  env,
  stdio: ["ignore", "pipe", "pipe"],
});
let diagnostics = "";
server.stderr.on("data", (data) => {
  diagnostics = (diagnostics + data).slice(-12000);
});
let closed = false;
const exited = new Promise((resolve) =>
  server.once("close", () => {
    closed = true;
    resolve();
  }),
);
let browser;
let page;
const stop = () => {
  process.exitCode = 130;
  server.kill("SIGTERM");
  if (browser) void browser.close().catch(() => {});
};
process.once("SIGINT", stop);
process.once("SIGTERM", stop);

try {
  const urls = await new Promise((resolve, reject) => {
    const lines = createInterface({ input: server.stdout });
    const timer = setTimeout(() => reject(new Error("HTTPS fixture readiness timed out")), 30000);
    server.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    server.once("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`HTTPS fixture exited ${code} before readiness`));
    });
    lines.on("line", (line) => {
      let value;
      try {
        value = JSON.parse(line);
      } catch {
        return;
      }
      if (value.plane && value.issuer) {
        clearTimeout(timer);
        lines.close();
        resolve(value);
      }
    });
  });
  browser = await chromium.launch({ headless: true, executablePath: values["browser-executable"] });
  receipt.browser = browser.version();
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  context.setDefaultTimeout(15000);
  await context.route("**/*", (route) => {
    const url = route.request().url();
    return url.startsWith(urls.plane + "/") || url.startsWith(urls.issuer + "/") ? route.continue() : route.abort();
  });
  page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const initial = await page.goto(urls.plane);
  const csp = initial.headers()["content-security-policy"];
  const scriptPolicy = csp?.split(";").find((directive) => directive.trim().startsWith("script-src "));
  assert.equal(scriptPolicy?.trim(), "script-src 'self'");
  await page.getByRole("button", { name: /Sign in with OIDC/i }).click();
  await page.waitForURL(urls.plane + "/");
  await page.getByRole("button", { name: /Sign out/i }).waitFor();
  receipt.checks.push("HTTPS authorization-code/PKCE sign-in under CSP");
  const cookie = (await context.cookies(urls.plane)).find((value) => value.name === SESSION_COOKIE);
  assert(cookie?.httpOnly && cookie?.secure && cookie?.sameSite === "Lax");
  assert(!(await page.evaluate(() => globalThis.document.cookie)).includes("eyvan_session"));
  assert.equal(await page.evaluate(() => globalThis.sessionStorage.length), 0);
  // Only layout preferences may be in web storage, each with the exact value shape
  // it is written in, so a credential stored under any key still fails this check.
  const unexpectedStorage = await page.evaluate(() => {
    const allowed = {
      "eyvan.rail.collapsed": /^[01]$/,
      "eyvan.assistant.open": /^[01]$/,
      "eyvan.assistant.width": /^[0-9]{3,4}$/,
    };
    return Object.entries(globalThis.localStorage).filter(([key, value]) => !allowed[key]?.test(value));
  });
  assert.deepEqual(unexpectedStorage, []);
  assert.equal(await page.evaluate(async () => (await globalThis.fetch("/api/v1/session")).status), 200);
  receipt.checks.push("HttpOnly Secure Lax cookie, no web-storage credentials, authenticated fetch");

  const attack = await context.newPage();
  await attack.goto(urls.issuer + "/attack");
  const rejected = attack.waitForResponse((response) => response.url().endsWith("/api/v1/dashboard/auth/logout"));
  await attack.getByRole("button", { name: "Cross-site logout" }).click();
  assert.equal((await rejected).status(), 403);
  assert((await context.cookies(urls.plane)).some((value) => value.name === SESSION_COOKIE));
  assert.equal(await page.evaluate(async () => (await globalThis.fetch("/api/v1/session")).status), 200);
  const observations = await page.evaluate(async () => (await globalThis.fetch("/qualification")).json());
  assert(observations.logout_attempts.some((value) => !value.has_cookie && value.origin === urls.issuer));
  receipt.checks.push("Actual cross-site cookieless logout POST rejected; session preserved");
  await page.getByRole("button", { name: /Sign out/i }).click();
  await page.getByRole("button", { name: /Sign in with OIDC/i }).waitFor();
  assert(!(await context.cookies(urls.plane)).some((value) => value.name === SESSION_COOKIE));
  assert.equal(await page.evaluate(async () => (await globalThis.fetch("/api/v1/session")).status), 401);
  receipt.checks.push("Same-origin sign-out revokes session and deletes cookie");

  // APIRequestContext shares this browser's cookies. Mint a real transaction,
  // then select the fake IdP's denial path before following its redirect.
  const deniedLogin = await context.request.get(urls.plane + "/api/v1/dashboard/auth/login", { maxRedirects: 0 });
  assert.equal(deniedLogin.status(), 307);
  const authorization = deniedLogin.headers().location;
  assert(authorization.startsWith(urls.issuer + "/authorize?"));
  await page.goto(authorization + "&deny=1");
  await page.getByText("sign-in was cancelled or denied by the identity provider", { exact: true }).waitFor();
  assert(!page.url().includes("auth_error"));
  assert(!(await context.cookies(urls.plane)).some((value) => value.name === SESSION_COOKIE));
  assert.equal(await page.evaluate(async () => (await globalThis.fetch("/api/v1/session")).status), 401);
  receipt.checks.push("Provider denial returns a generic error and does not admit a session");
  assert.deepEqual(errors, []);
  receipt.checks.push("No uncaught dashboard JavaScript errors");
  receipt.result = "passed";
} catch (error) {
  receipt.error = error.message;
  if (page && !page.isClosed()) {
    receipt.pageUrl = page.url();
    receipt.pageText = await page
      .locator("body")
      .innerText({ timeout: 1000 })
      .catch(() => "unavailable");
  }
  process.exitCode = 1;
} finally {
  try {
    if (browser) await browser.close();
  } catch (error) {
    receipt.result = "failed";
    receipt.cleanupError = error.message;
    process.exitCode = 1;
  }
  if (!closed) {
    server.kill("SIGTERM");
    const timer = setTimeout(() => server.kill("SIGKILL"), 10000);
    await exited;
    clearTimeout(timer);
  }
  if (server.exitCode !== 0) {
    receipt.result = "failed";
    receipt.cleanupError = `Fixture exit ${server.exitCode}, signal ${server.signalCode}`;
    process.exitCode = 1;
  }
  await rm(state, { recursive: true, force: true });
  process.removeListener("SIGINT", stop);
  process.removeListener("SIGTERM", stop);
  await mkdir(path.dirname(receiptPath), { recursive: true });
  await writeFile(receiptPath, JSON.stringify(receipt, null, 2) + "\n");
  console.log(JSON.stringify(receipt));
  if (receipt.result !== "passed" && diagnostics.trim()) console.error(diagnostics);
}
