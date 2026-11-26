// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EyvanApiClient } from "../api/client";
import { FleetPage } from "../pages/FleetPage";
import type { CapacityPoolSummary, FleetInitScriptDocument } from "../types";
import { FleetInitScript, type FleetInitScriptClient } from "./FleetInitScript";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function doc(overrides: Partial<FleetInitScriptDocument> = {}): FleetInitScriptDocument {
  return {
    fleet: "research",
    script: "pip install torch",
    consistent: true,
    supported: true,
    unsupported_reasons: [],
    other_init_commands: ["echo from-yaml"],
    pools: ["research/cpu"],
    max_bytes: 32768,
    ...overrides,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((ok, fail) => {
    resolve = ok;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function editor(): HTMLTextAreaElement {
  return screen.getByRole("textbox", { name: /Script \(bash\)/ }) as HTMLTextAreaElement;
}

function button(name: string): HTMLButtonElement {
  return screen.getByRole("button", { name }) as HTMLButtonElement;
}

/** The real client against a stubbed fetch, so the request itself is what is asserted. */
function realClientWith(responses: Array<() => Response>) {
  const fetch = vi.fn(async () => {
    const next = responses.shift();
    if (!next) throw new Error("unexpected request");
    return next();
  });
  vi.stubGlobal("fetch", fetch);
  return { client: new EyvanApiClient("/api/v1", false), fetch };
}

const json =
  (body: unknown, status = 200) =>
  () =>
    new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("fleet init script editor", () => {
  it("loads and shows the script, the YAML commands that also run, and the byte budget", async () => {
    const client: FleetInitScriptClient = {
      getFleetInitScript: vi.fn(async () => doc()),
      setFleetInitScript: vi.fn(),
    };
    render(<FleetInitScript fleet="research" client={client} />);

    await waitFor(() => expect(editor().value).toBe("pip install torch"));
    expect(client.getFleetInitScript).toHaveBeenCalledWith("research", expect.any(AbortSignal));
    expect(editor().getAttribute("spellcheck")).toBe("false");
    const yaml = screen.getByRole("region", { name: "Also runs (from fleet YAML)" });
    expect(within(yaml).getByText("echo from-yaml")).toBeTruthy();
    expect(screen.getByText("17 / 32768 bytes")).toBeTruthy();
    // Unchanged: nothing to save.
    expect(button("Save script").disabled).toBe(true);
  });

  it("saves with a PUT carrying an Idempotency-Key and the script, only after the confirmation", async () => {
    const saved = doc({ script: "echo new" });
    const { client, fetch } = realClientWith([json(doc()), json(saved)]);
    render(<FleetInitScript fleet="research" client={client} />);
    await waitFor(() => expect(editor().value).toBe("pip install torch"));

    fireEvent.change(editor(), { target: { value: "echo new" } });
    fireEvent.click(button("Save script"));
    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toMatch(/Existing idle cells are\s+replaced/);
    expect(fetch).toHaveBeenCalledTimes(1);

    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    const [url, init] = fetch.mock.calls[1] as unknown as [string, RequestInit];
    expect(url).toBe("/api/v1/fleets/research/init-script");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(String(init.body))).toEqual({ script: "echo new" });
    const headers = init.headers as Record<string, string>;
    expect(headers["Idempotency-Key"]).toBeTruthy();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByRole("status").textContent).toContain("Saved");
    expect(button("Save script").disabled).toBe(true);
  });

  it("uses a fresh Idempotency-Key for each save", async () => {
    const { client, fetch } = realClientWith([json(doc()), json(doc({ script: "a" })), json(doc({ script: "b" }))]);
    render(<FleetInitScript fleet="research" client={client} />);
    await waitFor(() => expect(editor().value).toBe("pip install torch"));
    for (const script of ["a", "b"]) {
      fireEvent.change(editor(), { target: { value: script } });
      fireEvent.click(button("Save script"));
      fireEvent.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Save" }));
      await waitFor(() => expect(editor().value).toBe(script));
    }
    const keys = fetch.mock.calls
      .slice(1)
      .map(
        (call) => ((call as unknown as [string, RequestInit])[1].headers as Record<string, string>)["Idempotency-Key"],
      );
    expect(keys).toHaveLength(2);
    expect(keys[0]).not.toBe(keys[1]);
  });

  it("removes by sending an empty script after a confirmation", async () => {
    const client: FleetInitScriptClient = {
      getFleetInitScript: vi.fn(async () => doc()),
      setFleetInitScript: vi.fn(async () => doc({ script: "" })),
    };
    render(<FleetInitScript fleet="research" client={client} />);
    await waitFor(() => expect(editor().value).toBe("pip install torch"));

    fireEvent.click(button("Remove script"));
    const dialog = await screen.findByRole("dialog");
    expect(client.setFleetInitScript).not.toHaveBeenCalled();
    fireEvent.click(within(dialog).getByRole("button", { name: "Remove" }));

    await waitFor(() => expect(client.setFleetInitScript).toHaveBeenCalledWith("research", ""));
    await waitFor(() => expect(editor().value).toBe(""));
    expect(button("Remove script").disabled).toBe(true);
  });

  it("shows why an attached TPU fleet cannot run a script and disables editing", async () => {
    const reason =
      "pool research/tpu is an attached TPU: Eyvan never creates its hosts, so a script here would never run; " +
      "pass setup to `eyvan gcp connect --init-command` instead";
    const client: FleetInitScriptClient = {
      getFleetInitScript: vi.fn(async () =>
        doc({ script: "", supported: false, unsupported_reasons: [reason], other_init_commands: [] }),
      ),
      setFleetInitScript: vi.fn(),
    };
    render(<FleetInitScript fleet="research" client={client} />);

    expect((await screen.findByRole("alert")).textContent).toContain(reason);
    expect(editor().disabled).toBe(true);
    expect(button("Save script").disabled).toBe(true);
    expect(button("Remove script").disabled).toBe(true);
  });

  it("warns when the fleet's pools disagree", async () => {
    const client: FleetInitScriptClient = {
      getFleetInitScript: vi.fn(async () => doc({ consistent: false })),
      setFleetInitScript: vi.fn(),
    };
    render(<FleetInitScript fleet="research" client={client} />);
    expect((await screen.findByRole("status")).textContent).toContain("different init scripts");
  });

  it("shows the server's refusal verbatim and keeps the dialog open", async () => {
    const detail = "the script may not contain a line reading END_OF_EYVAN_FLEET_INIT_SCRIPT";
    const { client } = realClientWith([json(doc()), json({ detail }, 422)]);
    render(<FleetInitScript fleet="research" client={client} />);
    await waitFor(() => expect(editor().value).toBe("pip install torch"));

    fireEvent.change(editor(), { target: { value: "a\nEND_OF_EYVAN_FLEET_INIT_SCRIPT\nb" } });
    fireEvent.click(button("Save script"));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));

    expect((await within(dialog).findByRole("alert")).textContent).toContain(detail);
    expect(screen.getByRole("dialog")).toBeTruthy();
  });

  it("disables Save when the script is over the byte limit (counted in UTF-8 bytes)", async () => {
    const client: FleetInitScriptClient = {
      getFleetInitScript: vi.fn(async () => doc({ script: "", max_bytes: 4 })),
      setFleetInitScript: vi.fn(),
    };
    render(<FleetInitScript fleet="research" client={client} />);
    await waitFor(() => expect(editor().disabled).toBe(false));

    fireEvent.change(editor(), { target: { value: "ééé" } }); // 3 characters, 6 bytes
    expect(screen.getByText(/6 \/ 4 bytes/).textContent).toContain("over the limit");
    expect(button("Save script").disabled).toBe(true);
    fireEvent.change(editor(), { target: { value: "éé" } });
    expect(button("Save script").disabled).toBe(false);
  });

  it("never lets an older load overwrite a newer save", async () => {
    const first = deferred<FleetInitScriptDocument>();
    const reload = deferred<FleetInitScriptDocument>();
    const reads = [first, reload];
    const client: FleetInitScriptClient = {
      getFleetInitScript: vi.fn(() => reads.shift()!.promise),
      setFleetInitScript: vi.fn(async () => doc({ script: "echo saved" })),
    };
    render(<FleetInitScript fleet="research" client={client} />);
    await act(async () => first.resolve(doc()));
    await waitFor(() => expect(editor().value).toBe("pip install torch"));

    // A reload starts, then a save finishes before it answers.
    fireEvent.click(button("Reload"));
    fireEvent.change(editor(), { target: { value: "echo saved" } });
    fireEvent.click(button("Save script"));
    fireEvent.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(editor().value).toBe("echo saved");

    await act(async () => reload.resolve(doc({ script: "stale" })));
    expect(editor().value).toBe("echo saved");
  });

  it("ignores a read for a fleet the operator has already left", async () => {
    const research = deferred<FleetInitScriptDocument>();
    const client: FleetInitScriptClient = {
      getFleetInitScript: vi.fn((fleet: string) =>
        fleet === "research" ? research.promise : Promise.resolve(doc({ fleet: "other", script: "echo other" })),
      ),
      setFleetInitScript: vi.fn(),
    };
    const view = render(<FleetInitScript fleet="research" client={client} />);
    view.rerender(<FleetInitScript fleet="other" client={client} />);
    await waitFor(() => expect(editor().value).toBe("echo other"));
    await act(async () => research.resolve(doc({ script: "echo research" })));
    expect(editor().value).toBe("echo other");
  });

  it("refuses in demo mode instead of pretending to save", async () => {
    const client = new EyvanApiClient("/api/v1", true);
    await expect(client.getFleetInitScript("research")).rejects.toEqual(
      expect.objectContaining({ name: "ApiError", status: 400 }),
    );
    await expect(client.setFleetInitScript("research", "x")).rejects.toEqual(
      expect.objectContaining({ name: "ApiError", status: 400 }),
    );
  });
});

describe("init script on the fleet page", () => {
  const pools: CapacityPoolSummary[] = [
    {
      id: "research/cpu",
      fleetName: "research",
      acceleratorType: null,
      capacityClass: "on_demand",
      zone: "us-central1-a",
      region: "us-central1",
      maxCells: 2,
      minCells: 0,
      enabled: true,
    },
  ];
  const base = {
    cells: [],
    leases: [],
    onRefresh: async () => {},
    loadWarmth: async () => ({ observed_at: "", pools: [] }),
    loadBilling: async () =>
      ({
        observed_at: "",
        accrued_cost: 0,
        idle_cost: 0,
        cells: 0,
        billed_seconds: 0,
        idle_seconds: 0,
        unpriced_cells: 0,
      }) as never,
    loadLaunchLatency: async () => ({ observed_at: "", pools: [] }),
    loadPools: async () => pools,
  };

  it("is never offered to a session that cannot manage the fleet", async () => {
    render(<FleetPage {...base} />);
    const region = await screen.findByRole("region", { name: "Applied fleets" });
    await waitFor(() => expect(within(region).getByText("research")).toBeTruthy());
    expect(screen.queryByRole("button", { name: /Init script/ })).toBeNull();
  });

  it("opens the editor for a fleet when a fleet admin asks", async () => {
    const client: FleetInitScriptClient = {
      getFleetInitScript: vi.fn(async () => doc()),
      setFleetInitScript: vi.fn(),
    };
    render(<FleetPage {...base} initScriptClient={client} />);
    fireEvent.click(await screen.findByRole("button", { name: "Init script for research" }));

    const panel = await screen.findByRole("region", { name: "Init script for research" });
    await waitFor(() =>
      expect((within(panel).getByRole("textbox") as HTMLTextAreaElement).value).toBe("pip install torch"),
    );
    fireEvent.click(within(panel).getByRole("button", { name: "Close init script" }));
    expect(screen.queryByRole("region", { name: "Init script for research" })).toBeNull();
  });
});
