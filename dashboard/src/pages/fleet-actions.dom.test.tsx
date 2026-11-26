// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Cell } from "../types";
import { FleetPage } from "./FleetPage";

afterEach(cleanup);

const never = () => new Promise<never>(() => {});
const loaders = { loadWarmth: never, loadBilling: never, loadLaunchLatency: never };

function cell(overrides: Partial<Cell> = {}): Cell {
  return {
    id: "cell-a",
    name: "us-east5-b-v6e-01",
    backend: "gcp-tpu",
    region: "us-east5",
    zone: "us-east5-b",
    status: "healthy",
    state: "ready",
    utilization: 62,
    activeWorkloads: 2,
    topology: "v6e-16",
    resources: [{ name: "TPU", used: 8, total: 16 }],
    lastHeartbeat: "2026-08-16T09:00:00Z",
    ...overrides,
  };
}

function renderFleet(options: { cells?: Cell[]; onDrainCell?: (id: string) => Promise<void> } = {}) {
  return render(
    <FleetPage
      cells={options.cells ?? [cell()]}
      leases={[]}
      onRefresh={() => Promise.resolve()}
      {...loaders}
      onDrainCell={options.onDrainCell}
    />,
  );
}

describe("draining a cell", () => {
  it("is not offered to a session that cannot manage the fleet", () => {
    renderFleet();

    expect(screen.queryByRole("button", { name: "Drain" })).toBeNull();
  });

  it("reports selected-project work without claiming to know whole-fleet occupancy", async () => {
    renderFleet({ onDrainCell: () => Promise.resolve() });

    fireEvent.click(screen.getByRole("button", { name: "Drain" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/2 selected-project workloads currently hold capacity here/)).toBeTruthy();
    expect(within(dialog).getByText(/Other-project occupancy is unknown/)).toBeTruthy();
    expect(screen.getByText("Selected-project workloads")).toBeTruthy();
    expect(within(dialog).getAllByText(/us-east5-b-v6e-01/).length).toBeGreaterThan(0);
  });

  it("reports zero selected-project workloads without claiming the cell is empty", async () => {
    renderFleet({ cells: [cell({ activeWorkloads: 0 })], onDrainCell: () => Promise.resolve() });

    fireEvent.click(screen.getByRole("button", { name: "Drain" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/No selected-project workloads currently hold capacity here/)).toBeTruthy();
    expect(within(dialog).getByText(/Other-project occupancy is unknown/)).toBeTruthy();
    expect(within(dialog).queryByText(/No workload currently holds capacity here/)).toBeNull();
  });

  it("cannot be reached by a mis-click, because a drain cannot be undone", async () => {
    // The reason this action stayed off the dashboard. There is no route back
    // to READY, and on spot the slice it gives up may not be grantable again
    // for hours -- so the gate is not a louder warning but a gesture a reflex
    // cannot perform.
    renderFleet({ onDrainCell: () => Promise.resolve() });

    fireEvent.click(screen.getByRole("button", { name: "Drain" }));
    const dialog = await screen.findByRole("dialog");

    expect(within(dialog).getByRole("button", { name: "Drain cell" }).hasAttribute("disabled")).toBe(true);
    // Focus lands on the field, never on the destructive button: a dialog that
    // opens with "drain" under the return key turns a keystroke meant for the
    // page behind it into a fleet mutation.
    await waitFor(() => expect(document.activeElement).toBe(within(dialog).getByRole("textbox")));
  });

  it("refuses a phrase that names a different cell", async () => {
    const drained: string[] = [];
    renderFleet({
      cells: [cell(), cell({ id: "cell-b", name: "us-east5-b-v6e-02" })],
      onDrainCell: (id) => {
        drained.push(id);
        return Promise.resolve();
      },
    });

    fireEvent.click(screen.getAllByRole("button", { name: "Drain" })[1]);
    fireEvent.change(await screen.findByRole("textbox"), { target: { value: "us-east5-b-v6e-01" } });

    expect(screen.getByRole("button", { name: "Drain cell" }).hasAttribute("disabled")).toBe(true);
    expect(drained).toEqual([]);
  });

  it("drains only after the confirmation, and only the cell that was named", async () => {
    const drained: string[] = [];
    renderFleet({
      cells: [cell(), cell({ id: "cell-b", name: "us-east5-b-v6e-02" })],
      onDrainCell: (id) => {
        drained.push(id);
        return Promise.resolve();
      },
    });

    fireEvent.click(screen.getAllByRole("button", { name: "Drain" })[1]);
    expect(drained).toEqual([]);
    fireEvent.change(await screen.findByRole("textbox"), { target: { value: "us-east5-b-v6e-02" } });
    fireEvent.click(screen.getByRole("button", { name: "Drain cell" }));

    await waitFor(() => expect(drained).toEqual(["cell-b"]));
    expect(await screen.findByText(/us-east5-b-v6e-02 is draining/)).toBeTruthy();
  });

  it("keeps a refusal on screen instead of closing over an unchanged fleet", async () => {
    // Closing and dropping a toast puts the reason two seconds away from a
    // fleet that did not change, which reads as a fleet that did.
    renderFleet({ onDrainCell: () => Promise.reject(new Error("'alice' lacks manage_fleet permission")) });

    fireEvent.click(screen.getByRole("button", { name: "Drain" }));
    fireEvent.change(await screen.findByRole("textbox"), { target: { value: "us-east5-b-v6e-01" } });
    fireEvent.click(screen.getByRole("button", { name: "Drain cell" }));

    expect((await screen.findByRole("alert")).textContent).toContain("'alice' lacks manage_fleet permission");
    expect(screen.getByRole("dialog")).toBeTruthy();
  });

  it("holds the dialog open while the request is in flight", async () => {
    // The window between the click and the answer is where a second click
    // lands. It is also where an optimistic close would report a success the
    // plane has not confirmed.
    let settle = () => {};
    renderFleet({ onDrainCell: () => new Promise<void>((resolve) => (settle = resolve)) });

    fireEvent.click(screen.getByRole("button", { name: "Drain" }));
    fireEvent.change(await screen.findByRole("textbox"), { target: { value: "us-east5-b-v6e-01" } });
    fireEvent.click(screen.getByRole("button", { name: "Drain cell" }));

    const busy = await screen.findByRole("button", { name: "Working" });
    expect(busy.hasAttribute("disabled")).toBe(true);
    expect(screen.getByRole("button", { name: "Cancel" }).hasAttribute("disabled")).toBe(true);

    settle();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("distinguishes a cell being drained from one being released", () => {
    // Both render as `maintenance`, and an operator deciding what to do about a
    // machine needs to know which of the two it is already doing.
    renderFleet({
      cells: [
        cell({ status: "maintenance", state: "draining" }),
        cell({ id: "cell-b", name: "us-east5-b-v6e-02", status: "maintenance", state: "deleting" }),
      ],
      onDrainCell: () => Promise.resolve(),
    });

    expect(screen.getByRole("button", { name: "Draining" }).hasAttribute("disabled")).toBe(true);
    expect(screen.getByRole("button", { name: "Releasing" }).hasAttribute("disabled")).toBe(true);
  });

  it("offers to drain an unhealthy cell, which is when it is most needed", () => {
    renderFleet({ cells: [cell({ status: "degraded", state: "unreachable" })], onDrainCell: () => Promise.resolve() });

    expect(screen.getByRole("button", { name: "Drain" }).hasAttribute("disabled")).toBe(false);
  });
});
