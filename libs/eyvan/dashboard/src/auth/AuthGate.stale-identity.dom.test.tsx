// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AuthGate } from "./AuthGate";
import { __resetRuntimeOidcConfigForTests, rejectSession } from "./oidc";

describe("a session probe that resolves after the session was rejected", () => {
  afterEach(() => {
    cleanup();
    __resetRuntimeOidcConfigForTests();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    sessionStorage.removeItem("eyvan.local.credential");
  });

  it("does not re-open the gate with the revoked identity", async () => {
    // The gate asks the authority endpoint whether a session exists. While
    // that probe is in flight, any other call can be refused with a 401 and
    // the API client will reject the session -- clearing identity and firing
    // the auth-changed event. A probe that resolves afterwards must not
    // re-install the identity it captured under the now-revoked cookie.
    vi.stubEnv("VITE_EYVAN_OIDC_ISSUER", "");
    vi.stubEnv("VITE_EYVAN_OIDC_CLIENT_ID", "");

    let answerSession: (body: Record<string, unknown>) => void = () => {};
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: unknown) => {
        const url = String(input);
        if (url.endsWith("/api/v1/session")) {
          return new Promise<Response>((resolve) => {
            answerSession = (body) => resolve(new Response(JSON.stringify(body), { status: 200 }));
          });
        }
        return new Response(
          JSON.stringify({ issuer: "https://idp.example.com", client_id: "eyvan", development_auth: "disabled" }),
          { status: 200 },
        );
      }),
    );

    render(
      <AuthGate>
        <p>fleet</p>
      </AuthGate>,
    );

    // Let the mount effect run to the point where the session probe is
    // actually in flight before the session is rejected underneath it.
    await act(async () => {});

    // The probe is still in flight when the plane refuses some other call.
    rejectSession();

    // ...and only then does the stale probe resolve with an identity.
    answerSession({ subject: "alice", display_name: "Alice Example" });

    await waitFor(() => expect(screen.getByText(/control plane rejected your session/)).toBeTruthy());
    expect(screen.queryByText("fleet")).toBeNull();
  });
});
