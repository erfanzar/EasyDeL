// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// Real Chromium against a disposable team-mode plane (started by
// tests/test_team_dashboard_browser.py): accept-invite -> Settings -> invite and
// token -> sign out / wrong password / sign in -> a second non-admin user ->
// disabling that user signs them out. Exit 77 when no browser can launch.
import { readFile } from "node:fs/promises";
import { chromium } from "../../dashboard/node_modules/playwright/index.mjs";

const [base, inviteFile] = process.argv.slice(2);
const problems = [];
const ok = (message) => console.log("PASS", message);
const fail = (message) => {
  problems.push(message);
  console.log("FAIL", message);
};
const ADMIN_PW = "admin password 123";
const BOB_PW = "bob password 4567";

function watch(page, tag) {
  page.on("pageerror", (error) => fail(`${tag} page error: ${error.message}`));
}

let browser;
try {
  browser = await chromium.launch({ headless: true });
} catch {
  try {
    browser = await chromium.launch({ headless: true, channel: "chrome" });
  } catch {
    process.exit(77);
  }
}

try {
  // The first admin is invited out-of-band, the way an operator would bootstrap.
  const adminInvite = await readFile(inviteFile, "utf8");

  const contextA = await browser.newContext({ viewport: { width: 1500, height: 1000 }, serviceWorkers: "block" });
  const a = await contextA.newPage();
  watch(a, "A");
  await a.goto(`${base}/`);
  await a.getByRole("form", { name: "Sign in" }).waitFor({ timeout: 15000 });
  ok("a team plane without a session shows the password sign-in form");

  // An invite link opened in the already-open tab changes only the fragment.
  await a.goto(adminInvite);
  await a.getByRole("heading", { name: "Join Eyvan" }).waitFor({ timeout: 15000 });
  if (a.url().includes("accept-invite")) fail(`invite token still in the address bar: ${a.url()}`);
  else ok("the invite token is stripped from the address bar");
  await a.getByLabel("Password", { exact: true }).fill(ADMIN_PW);
  await a.getByLabel("Confirm password").fill(ADMIN_PW);
  await a.getByRole("button", { name: "Create account" }).click();

  const settings = a.getByRole("button", { name: /^Settings/ });
  await settings.waitFor({ timeout: 15000 });
  ok("accepting the invite signs in and the rail offers Settings");

  let attachmentRequests = 0;
  a.on("request", (request) => {
    if (new URL(request.url()).pathname.startsWith("/api/v1/attachments")) attachmentRequests++;
  });
  await a.getByRole("button", { name: /^Fleet\b/ }).click();
  await a.getByRole("heading", { name: "Applied fleets" }).waitFor();
  await a.getByRole("button", { name: /refresh.*fleet|refresh.*inventory/i }).click();
  await a.getByText(/reloaded from Eyvan/i).waitFor();
  if (attachmentRequests || (await a.getByRole("button", { name: "Add TPU" }).count()))
    fail(`team Fleet offered local attachments or made discovery calls (requests: ${attachmentRequests})`);
  else ok("team admin can view and refresh fleet data without invoking local TPU attachments");

  await settings.click();
  await a.getByRole("heading", { name: "Settings" }).waitFor({ timeout: 10000 });
  if (!a.url().endsWith("#/settings")) fail(`Settings navigated to ${a.url()}`);
  else ok("Settings opens #/settings");

  const invitePanel = a.getByRole("region", { name: "Invite a user" });
  await invitePanel.waitFor({ timeout: 15000 });
  await a.getByRole("region", { name: "Users" }).getByText("admin@example.com").waitFor({ timeout: 10000 });
  ok("Settings lists the admin account");
  await invitePanel.getByLabel("Email").fill("bob@example.com");
  await invitePanel.getByRole("button", { name: "Create invite" }).click();
  const link = (await invitePanel.getByTestId("secret-once").innerText({ timeout: 10000 })).trim();
  if (!link.startsWith(`${base}/#/accept-invite/`)) fail(`invite link has an unexpected shape: ${link.slice(0, 60)}`);
  else ok("Create invite shows a one-time link");
  await a.getByRole("region", { name: "Invites" }).getByText("bob@example.com").waitFor({ timeout: 10000 });
  ok("the Invites table lists bob");

  const tokens = a.getByRole("region", { name: "My API tokens" });
  await tokens.getByLabel("Token name").fill("ci");
  await tokens.getByRole("button", { name: "Create token" }).click();
  const raw = (await tokens.getByTestId("secret-once").innerText({ timeout: 10000 })).trim();
  const me = await fetch(`${base}/api/v1/team/me`, { headers: { Authorization: `Bearer ${raw}` } });
  if (!raw.startsWith("eyv_") || me.status !== 200 || (await me.json()).subject !== "admin")
    fail("the dashboard-minted token does not authenticate as admin");
  else ok("a dashboard-minted token authenticates over Bearer");

  await a.evaluate(() => (window.location.hash = "#/users"));
  await a.waitForFunction(() => window.location.hash === "#/users");
  await a.getByRole("region", { name: "My API tokens" }).waitFor({ timeout: 10000 });
  ok("an old #/users link still shows the accounts page");

  await a.getByRole("button", { name: /^Sign out/ }).click();
  await a.getByRole("form", { name: "Sign in" }).waitFor({ timeout: 15000 });
  await a.reload();
  await a.getByRole("form", { name: "Sign in" }).waitFor({ timeout: 15000 });
  ok("sign-out ends the session, also after a reload");
  // Sign in with the user name, not the email: nobody is ever emailed.
  await a.getByLabel("User name or email").fill("admin");
  await a.getByLabel("Password").fill("not the password");
  await a.getByRole("button", { name: "Sign in", exact: true }).click();
  await a.getByRole("alert").getByText("wrong user name or password").waitFor({ timeout: 10000 });
  ok("a wrong password shows the generic refusal");
  await a.getByLabel("Password").fill(ADMIN_PW);
  await a.getByRole("button", { name: "Sign in", exact: true }).click();
  await a.getByRole("button", { name: /^Settings/ }).waitFor({ timeout: 15000 });
  ok("signing in with the user name opens the dashboard");

  const contextB = await browser.newContext({ viewport: { width: 1500, height: 1000 }, serviceWorkers: "block" });
  const b = await contextB.newPage();
  watch(b, "B");
  await b.goto(link);
  await b.getByRole("heading", { name: "Join Eyvan" }).waitFor({ timeout: 15000 });
  await b.getByLabel("Password", { exact: true }).fill(BOB_PW);
  await b.getByLabel("Confirm password").fill(BOB_PW);
  await b.getByRole("button", { name: "Create account" }).click();
  await b.getByRole("button", { name: /^Settings/ }).click({ timeout: 15000 });
  await b.getByRole("region", { name: "My API tokens" }).waitFor({ timeout: 15000 });
  if (await b.getByRole("region", { name: "Invite a user" }).count()) fail("a non-admin sees the invite form");
  else ok("a non-admin sees only their own tokens");

  await a.getByRole("button", { name: /^Settings/ }).click();
  const bobRow = a.getByRole("region", { name: "Users" }).locator("tr", { hasText: "bob@example.com" });
  await bobRow.getByRole("button", { name: "Disable" }).click({ timeout: 15000 });
  await bobRow.getByText("disabled").waitFor({ timeout: 10000 });
  await b.reload();
  await b.getByRole("form", { name: "Sign in" }).waitFor({ timeout: 15000 });
  ok("disabling a user signs their browser out");
} catch (error) {
  fail(`flow stopped: ${String(error?.message ?? error).split("\n")[0]}`);
} finally {
  await browser.close();
}
console.log(problems.length ? `RESULT failed (${problems.length})` : "RESULT passed");
process.exit(problems.length ? 1 : 0);
