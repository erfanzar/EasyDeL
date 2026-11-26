// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AddTpuModal, type AttachmentClient } from "./components/AddTpuModal";
import { EyvanApiClient, formatErrorDetail } from "./api/client";
import { mockAttachmentPlan, mockDiscoveredTpus } from "./api/mockData";
import { FleetPage } from "./pages/FleetPage";
import type { AttachmentInput, AttachmentJob } from "./types";

const input: AttachmentInput = { target: "alice@203.0.113.20", kind: "ssh", fleet: null, project: null, zone: null };
const running: AttachmentJob = {
  id: "a".repeat(32),
  target: input.target,
  status: "running",
  progress: ["Installing worker"],
  result: null,
  error: null,
  started_at: "2026-10-25T10:00:00Z",
  finished_at: null,
};
const succeeded: AttachmentJob = {
  ...running,
  status: "succeeded",
  progress: ["Installing worker", "Opening tunnel"],
  result: { fleet: "tpu-demo", ready: true, cells: [], submit: 'init(fleet="tpu-demo")', limitations: [] },
  finished_at: "2026-10-25T10:01:00Z",
};

function client(): AttachmentClient {
  return {
    planAttachment: vi.fn(async (request) => mockAttachmentPlan(request)),
    startAttachment: vi.fn(async () => running),
    getAttachment: vi.fn(async () => succeeded),
    listAttachments: vi.fn(async () => []),
  };
}

const close = vi.fn();
const connected = vi.fn(async () => {});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

async function preview(api: AttachmentClient) {
  render(<AddTpuModal open client={api} onClose={close} onConnected={connected} />);
  fireEvent.change(screen.getByLabelText("SSH host or IP"), { target: { value: input.target } });
  fireEvent.click(screen.getByRole("button", { name: "Preview" }));
  await screen.findByRole("region", { name: "Attachment preview" });
}

describe("Add TPU", () => {
  it("shows the plan facts and drops a stale preview on edits without locking Connect", async () => {
    const api = client();
    await preview(api);
    expect(api.planAttachment).toHaveBeenCalledWith(input);
    expect(screen.getByText("v5p-8")).toBeTruthy();
    expect(screen.getByText("spot")).toBeTruthy();
    expect(screen.getByText("tunnel")).toBeTruthy();
    expect(screen.getByText("Install the Eyvan worker on 2 hosts")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Connect" }) as HTMLButtonElement).disabled).toBe(false);
    fireEvent.change(screen.getByLabelText("Fleet name (optional)"), { target: { value: "other" } });
    expect(screen.queryByRole("region", { name: "Attachment preview" })).toBeNull();
    expect((screen.getByRole("button", { name: "Connect" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("connects in one click without a preview", async () => {
    const api = client();
    render(<AddTpuModal open client={api} onClose={close} onConnected={connected} />);
    const connectButton = screen.getByRole("button", { name: "Connect" }) as HTMLButtonElement;
    expect(connectButton.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("SSH host or IP"), { target: { value: input.target } });
    expect(connectButton.disabled).toBe(false);
    fireEvent.click(connectButton);
    await screen.findByText(/Connecting ·/);
    expect(api.startAttachment).toHaveBeenCalledWith(input);
    expect(api.planAttachment).not.toHaveBeenCalled();
  });

  it("accepts a TPU name with project and zone in advanced mode", async () => {
    const api = client();
    render(<AddTpuModal open client={api} onClose={close} onConnected={connected} />);
    fireEvent.click(screen.getByLabelText("Advanced: TPU name + project + zone"));
    fireEvent.change(screen.getByLabelText("TPU name"), { target: { value: "existing-tpu" } });
    fireEvent.change(screen.getByLabelText("GCP project"), { target: { value: "my-project" } });
    fireEvent.change(screen.getByLabelText("Zone"), { target: { value: "us-east5-a" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview" }));
    await waitFor(() =>
      expect(api.planAttachment).toHaveBeenCalledWith({
        ...input,
        target: "existing-tpu",
        kind: "tpu",
        project: "my-project",
        zone: "us-east5-a",
      }),
    );
  });

  it("survives transient polling failures, then shows and copies the successful fleet", async () => {
    const api = client();
    vi.mocked(api.getAttachment).mockRejectedValueOnce(new Error("plane restarting"));
    const copy = vi.fn(async () => {});
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: copy } });
    await preview(api);
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    await act(async () => {});
    expect(screen.getByText(/Connecting ·/)).toBeTruthy();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1500);
    });
    expect(screen.getByText(/Waiting for the local plane to reconnect/)).toBeTruthy();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1500);
    });
    expect(screen.getByText('init(fleet="tpu-demo")')).toBeTruthy();
    expect(screen.getByRole("log").textContent).toContain("Opening tunnel");
    fireEvent.click(screen.getByRole("button", { name: "Copy init snippet" }));
    await act(async () => {});
    expect(copy).toHaveBeenCalledWith('init(fleet="tpu-demo")');
    expect(connected).toHaveBeenCalled();
    expect(api.startAttachment).toHaveBeenCalledTimes(1);
  });

  it("recovers a running job after a fresh modal and reports its final error", async () => {
    const api = client();
    vi.mocked(api.listAttachments).mockResolvedValue([running]);
    vi.mocked(api.getAttachment).mockResolvedValue({ ...running, status: "failed", error: "SSH tunnel refused" });
    vi.useFakeTimers();
    render(<AddTpuModal open client={api} onClose={close} onConnected={connected} />);
    await act(async () => {});
    expect(screen.getByText(/Connecting ·/)).toBeTruthy();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1500);
    });
    expect(screen.getByRole("alert").textContent).toContain("SSH tunnel refused");
    expect(api.startAttachment).not.toHaveBeenCalled();
  });

  it("shows preview failures and still lets Connect report its own result", async () => {
    const api = client();
    vi.mocked(api.planAttachment).mockRejectedValue(new Error("Use a local development plane"));
    render(<AddTpuModal open client={api} onClose={close} onConnected={connected} />);
    fireEvent.change(screen.getByLabelText("SSH host or IP"), { target: { value: input.target } });
    fireEvent.click(screen.getByRole("button", { name: "Preview" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Use a local development plane");
    expect((screen.getByRole("button", { name: "Connect" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("only exposes Add TPU when FleetPage receives fleet-admin attachment authority", async () => {
    const props = {
      cells: [],
      leases: [],
      onRefresh: async () => {},
      loadWarmth: async () => ({ observed_at: "", pools: [] }),
      loadBilling: async () => ({
        observed_at: "",
        accrued_cost: 0,
        idle_cost: 0,
        cells: 0,
        billed_seconds: 0,
        idle_seconds: 0,
        unpriced_cells: 0,
      }),
      loadLaunchLatency: async () => ({ observed_at: "", pools: [] }),
    };
    const view = render(<FleetPage {...props} />);
    expect(screen.queryByRole("button", { name: "Add TPU" })).toBeNull();
    view.rerender(<FleetPage {...props} attachmentClient={client()} />);
    fireEvent.click(screen.getByRole("button", { name: "Add TPU" }));
    expect(screen.getByRole("dialog", { name: "Add TPU" })).toBeTruthy();
    await act(async () => {});
  });

  it("lists every discovered TPU and pre-fills Add TPU from an unconnected one", async () => {
    const api: AttachmentClient = {
      ...client(),
      listDiscoveredTpus: vi.fn(async () => ({
        ...mockDiscoveredTpus,
        skipped: { "locked-project": "no permission to list TPU nodes" },
      })),
    };
    render(
      <FleetPage
        cells={[]}
        leases={[]}
        onRefresh={async () => {}}
        loadWarmth={async () => ({ observed_at: "", pools: [] })}
        loadBilling={async () => ({
          observed_at: "",
          accrued_cost: 0,
          idle_cost: 0,
          cells: 0,
          billed_seconds: 0,
          idle_seconds: 0,
          unpriced_cells: 0,
        })}
        loadLaunchLatency={async () => ({ observed_at: "", pools: [] })}
        attachmentClient={api}
      />,
    );
    const panel = await screen.findByRole("region", { name: "TPUs in your Google Cloud" });
    await waitFor(() => expect(panel.textContent).toContain("tpu-spot-b-5e6f7a8b"));
    expect(panel.textContent).toContain("tpu-train-a-1a2b3c4d");
    expect(panel.textContent).toContain("locked-project: skipped (no permission to list TPU nodes)");
    // Only the unconnected TPU offers Connect.
    const connects = Array.from(panel.querySelectorAll("button")).filter((b) => b.textContent === "Connect");
    expect(connects).toHaveLength(1);
    fireEvent.click(connects[0]);
    expect(screen.getByRole("dialog", { name: "Add TPU" })).toBeTruthy();
    expect((screen.getByLabelText("TPU name") as HTMLInputElement).value).toBe("spot-b");
    expect((screen.getByLabelText("GCP project") as HTMLInputElement).value).toBe("demo-lab");
    expect((screen.getByLabelText("Zone") as HTMLInputElement).value).toBe("us-east5-a");
    expect((screen.getByLabelText("Fleet name (optional)") as HTMLInputElement).value).toBe("tpu-spot-b-5e6f7a8b");
    // The preview loads by itself: no Preview click, and Connect unlocks.
    await screen.findByRole("region", { name: "Attachment preview" });
    expect(vi.mocked(api.planAttachment).mock.calls[0][0]).toEqual({
      target: "spot-b",
      kind: "tpu",
      fleet: "tpu-spot-b-5e6f7a8b",
      project: "demo-lab",
      zone: "us-east5-a",
    });
    const dialog = screen.getByRole("dialog", { name: "Add TPU" });
    const connect = Array.from(dialog.querySelectorAll("button")).find((b) => b.textContent === "Connect");
    expect((connect as HTMLButtonElement).disabled).toBe(false);
  });

  it("reopens a plain Add TPU blank after a discovered TPU was used", async () => {
    const api = client();
    const initial: AttachmentInput = {
      target: "spot-b",
      kind: "tpu",
      fleet: "tpu-x",
      project: "demo-lab",
      zone: "us-east5-a",
    };
    const view = render(<AddTpuModal open client={api} initial={initial} onClose={close} onConnected={connected} />);
    await screen.findByRole("region", { name: "Attachment preview" });
    view.rerender(<AddTpuModal open={false} client={api} initial={null} onClose={close} onConnected={connected} />);
    view.rerender(<AddTpuModal open client={api} initial={null} onClose={close} onConnected={connected} />);
    // The very first render after reopening must already be the blank SSH form.
    expect(screen.queryByLabelText("TPU name")).toBeNull();
    expect((screen.getByLabelText("SSH host or IP") as HTMLInputElement).value).toBe("");
    await act(async () => {});
  });

  it("says Preview is optional", async () => {
    render(<AddTpuModal open client={client()} onClose={close} onConnected={connected} />);
    expect(screen.getByRole("note").textContent).toContain("Preview is optional");
    await act(async () => {});
  });

  it("renders a FastAPI validation error as text, not [object Object]", () => {
    const detail = [{ loc: ["body", "target"], msg: "Value error, invalid TPU name", type: "value_error" }];
    expect(formatErrorDetail(detail)).toBe(": target: invalid TPU name");
    expect(formatErrorDetail("plain")).toBe(": plain");
    expect(formatErrorDetail(undefined)).toBe("");
  });

  it("reads discovered TPUs through the real client class without losing its receiver", async () => {
    // A plain object of vi.fn()s never touches `this`, so it hid a detached
    // method call that crashed the real client with "this.forceDemo".
    const real = new EyvanApiClient("/api/v1", true);
    render(
      <FleetPage
        cells={[]}
        leases={[]}
        onRefresh={async () => {}}
        loadWarmth={async () => ({ observed_at: "", pools: [] })}
        loadBilling={async () => ({
          observed_at: "",
          accrued_cost: 0,
          idle_cost: 0,
          cells: 0,
          billed_seconds: 0,
          idle_seconds: 0,
          unpriced_cells: 0,
        })}
        loadLaunchLatency={async () => ({ observed_at: "", pools: [] })}
        attachmentClient={real}
      />,
    );
    const panel = await screen.findByRole("region", { name: "TPUs in your Google Cloud" });
    await waitFor(() => expect(panel.textContent).toContain(mockDiscoveredTpus.tpus[0].fleet));
    expect(panel.textContent).not.toContain("forceDemo");
    expect(panel.querySelector('[role="alert"]')).toBeNull();
  });

  it("shows a connected TPU's real cell state, with the error and automatic retry when it is failing", async () => {
    const [ready] = mockDiscoveredTpus.tpus;
    const api: AttachmentClient = {
      ...client(),
      listDiscoveredTpus: vi.fn(async () => ({
        ...mockDiscoveredTpus,
        tpus: [
          ready,
          {
            ...ready,
            fleet: "tpu-broken-9f9f9f9f",
            name: "broken",
            cell_state: "failing" as const,
            last_error: "could not read attached TPU node (HTTP 403)",
            next_attempt_at: "2026-11-01T12:10:00Z",
          },
        ],
      })),
    };
    render(
      <FleetPage
        cells={[]}
        leases={[]}
        onRefresh={async () => {}}
        loadWarmth={async () => ({ observed_at: "", pools: [] })}
        loadBilling={async () => ({
          observed_at: "",
          accrued_cost: 0,
          idle_cost: 0,
          cells: 0,
          billed_seconds: 0,
          idle_seconds: 0,
          unpriced_cells: 0,
        })}
        loadLaunchLatency={async () => ({ observed_at: "", pools: [] })}
        attachmentClient={api}
      />,
    );
    const panel = await screen.findByRole("region", { name: "TPUs in your Google Cloud" });
    await waitFor(() => expect(panel.textContent).toContain("tpu-broken-9f9f9f9f"));
    expect(panel.textContent).toContain("Ready");
    expect(panel.textContent).toContain("Failing");
    expect(panel.textContent).toContain("could not read attached TPU node (HTTP 403)");
    expect(panel.textContent).toContain("Retrying automatically after");
    expect(panel.textContent).not.toContain("Connected");
  });

  it("names the listing account and every project with its TPU count, and clips long fleet names", async () => {
    const [first, second] = mockDiscoveredTpus.tpus;
    const longFleet = "tpu-ray-sc-foundation-us-east1-d-worker-90884e43-tpu-bb413fbf2";
    const api: AttachmentClient = {
      ...client(),
      listDiscoveredTpus: vi.fn(async () => ({
        ...mockDiscoveredTpus,
        account: "someone@example.com",
        projects: ["demo-empty", "demo-lab", "demo-research"],
        tpus: [first, { ...second, fleet: longFleet }],
      })),
    };
    render(
      <FleetPage
        cells={[]}
        leases={[]}
        onRefresh={async () => {}}
        loadWarmth={async () => ({ observed_at: "", pools: [] })}
        loadBilling={async () => ({
          observed_at: "",
          accrued_cost: 0,
          idle_cost: 0,
          cells: 0,
          billed_seconds: 0,
          idle_seconds: 0,
          unpriced_cells: 0,
        })}
        loadLaunchLatency={async () => ({ observed_at: "", pools: [] })}
        attachmentClient={api}
      />,
    );
    const panel = await screen.findByRole("region", { name: "TPUs in your Google Cloud" });
    await waitFor(() => expect(panel.textContent).toContain("someone@example.com"));
    expect(panel.textContent).toContain("3 projects: demo-empty (0), demo-lab (1), demo-research (1)");
    const fleetCell = Array.from(panel.querySelectorAll("td")).find((td) => td.textContent?.trim() === longFleet);
    expect(fleetCell?.classList.contains("discovered-fleet")).toBe(true);
    expect(fleetCell?.getAttribute("title")).toBe(longFleet);
  });

  it("keeps the last discovered list and names the error when a refresh fails", async () => {
    const api: AttachmentClient = {
      ...client(),
      listDiscoveredTpus: vi.fn(async () => {
        throw new Error("plane unreachable");
      }),
    };
    render(
      <FleetPage
        cells={[]}
        leases={[]}
        onRefresh={async () => {}}
        loadWarmth={async () => ({ observed_at: "", pools: [] })}
        loadBilling={async () => ({
          observed_at: "",
          accrued_cost: 0,
          idle_cost: 0,
          cells: 0,
          billed_seconds: 0,
          idle_seconds: 0,
          unpriced_cells: 0,
        })}
        loadLaunchLatency={async () => ({ observed_at: "", pools: [] })}
        attachmentClient={api}
      />,
    );
    const panel = await screen.findByRole("region", { name: "TPUs in your Google Cloud" });
    await waitFor(() => expect(panel.textContent).toContain("plane unreachable"));
    expect(panel.textContent).not.toContain("No TPU VMs observed");
  });
});
