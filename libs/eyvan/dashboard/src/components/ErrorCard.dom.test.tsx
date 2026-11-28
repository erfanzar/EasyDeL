// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ErrorBoundary } from "./ErrorBoundary";
import { ErrorCard } from "./ErrorCard";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const RAW_IAM =
  "gcloud beta quotas info list failed: PERMISSION_DENIED: Permission 'cloudquotas.quotas.get' denied on resource '//cloudquotas.googleapis.com/projects/example-project-a/locations/global/services/tpu.googleapis.com'. This command is authenticated as alice@example.com. - '@type': type.googleapis.com/google.rpc.ErrorInfo error_info_id: Cg1JQU0=";

describe("ErrorCard", () => {
  it("renders badges, copyable command with Copied feedback, quick actions, and collapsed technical details", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const onCopied = vi.fn();
    const onSwitch = vi.fn();

    const { container } = render(
      <ErrorCard
        error={RAW_IAM}
        actions={[{ label: "Switch to example-project-b", onClick: onSwitch, variant: "primary" }]}
        onCopied={onCopied}
      />,
    );

    expect(screen.getByText("GCP IAM permission denied")).toBeDefined();
    expect(screen.getByText("example-project-a")).toBeDefined();
    expect(screen.getByText("alice@example.com")).toBeDefined();
    expect(screen.getByText("cloudquotas.quotas.get")).toBeDefined();

    // Technical details exist in DOM but start collapsed.
    const details = container.querySelector("details.error-card-details") as HTMLDetailsElement | null;
    expect(details).not.toBeNull();
    expect(details!.open).toBe(false);
    expect(details!.textContent).toContain("error_info_id: Cg1JQU0=");

    // Primary summary never leaks protobuf noise.
    const summary = container.querySelector(".error-card-summary");
    expect(summary?.textContent).not.toContain("@type");
    expect(summary?.textContent).not.toContain("error_info_id");

    // Copy command copies the gcloud IAM command and shows Copied feedback.
    const copyBtn = screen.getByRole("button", { name: /Copy command/ });
    fireEvent.click(copyBtn);
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(expect.stringContaining("roles/cloudquotas.viewer")));
    expect(await screen.findByRole("button", { name: /Copied/ })).toBeDefined();
    expect(onCopied).toHaveBeenCalledWith("Command copied to clipboard");

    // Quick action triggers cleanly.
    fireEvent.click(screen.getByRole("button", { name: "Switch to example-project-b" }));
    expect(onSwitch).toHaveBeenCalledOnce();
  });

  it("omits Technical details when a short error needs no raw expansion", () => {
    const { container } = render(<ErrorCard error="Something went wrong" />);
    expect(screen.getByText("Something went wrong")).toBeDefined();
    expect(container.querySelector("details")).toBeNull();
  });
});

describe("ErrorBoundary", () => {
  it("catches render crashes, shows recovery controls, and recovers on reload or resetKey change", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    let shouldThrow = true;
    function CrashingView() {
      if (shouldThrow) throw new Error("Simulated chart renderer crash");
      return <div>Recovered dashboard content</div>;
    }

    const onGoHome = vi.fn();
    const view = render(
      <ErrorBoundary resetKey="route-a" onGoHome={onGoHome}>
        <CrashingView />
      </ErrorBoundary>,
    );

    expect(screen.getByText("Unexpected view error")).toBeDefined();
    expect(screen.getByText("Simulated chart renderer crash")).toBeDefined();
    expect(screen.getByText("Technical details")).toBeDefined();

    // Go to Overview callback fires and clears boundary.
    shouldThrow = false;
    fireEvent.click(screen.getByRole("button", { name: "Go to Overview" }));
    expect(onGoHome).toHaveBeenCalledOnce();
    expect(screen.getByText("Recovered dashboard content")).toBeDefined();

    // Changing resetKey also recovers cleanly after another crash.
    shouldThrow = true;
    view.rerender(
      <ErrorBoundary resetKey="route-b" onGoHome={onGoHome}>
        <CrashingView />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Unexpected view error")).toBeDefined();

    shouldThrow = false;
    view.rerender(
      <ErrorBoundary resetKey="route-c" onGoHome={onGoHome}>
        <CrashingView />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Recovered dashboard content")).toBeDefined();
    expect(consoleError).toHaveBeenCalled();
  });
});
