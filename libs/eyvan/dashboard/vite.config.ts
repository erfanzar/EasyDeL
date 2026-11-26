// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The control plane is not always on 8080: `eyvan runtime` picks a per-profile
// port, and a developer commonly has more than one running. Hardcoding the
// target sent every request to whichever server happened to own 8080, which
// renders as an empty dashboard rather than an error -- the worst failure mode,
// because it looks like the fleet is idle.
//
// `loadEnv` with an empty prefix reads the real process environment as well as
// .env files, so this works both as `EYVAN_DASHBOARD_API=... npm run dev` and as
// a checked-in .env.local. The cwd is passed as "." rather than `process.cwd()`
// so the config needs no Node type definitions.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const controlPlane = env.EYVAN_DASHBOARD_API || "http://127.0.0.1:8080";
  return {
    plugins: [react()],
    server: {
      port: 4173,
      proxy: {
        "/api": controlPlane,
      },
    },
    preview: {
      port: 4174,
    },
    // The default stays "node". Component suites opt in with a
    // `@vitest-environment jsdom` docblock, per file rather than globally: a DOM
    // for every file is slower and, worse, lets a test that quietly depends on
    // one look like a pure test of the data layer.
    test: {
      environment: "node",
      coverage: {
        provider: "v8",
        // Text only: the default reporter set writes an HTML tree into the
        // working directory, and the repo's route-coverage test scans every
        // *.ts* file it finds there.
        reporter: ["text"],
        include: ["src/**/*.{ts,tsx}"],
        exclude: ["src/**/*.test.*", "src/**/mockData.ts"],
        // Ratchet, not target: measured 2026-09-07 at 82.8% lines / 79.6%
        // statements / 79.4% functions / 65.4% branches. Raise these when
        // coverage improves; never lower them without a reviewed exception.
        thresholds: {
          lines: 80,
          statements: 78,
          functions: 78,
          branches: 60,
        },
      },
    },
  };
});
