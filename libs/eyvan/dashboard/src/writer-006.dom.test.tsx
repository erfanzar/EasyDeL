// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ConfirmDialog, type ConfirmDialogProps } from "./components/ConfirmDialog";
import { FleetPage } from "./pages/FleetPage";
import type { Cell } from "./types";

afterEach(cleanup);

const cell: Cell = {
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
};

const never = () => new Promise<never>(() => {});
const loaders = { loadWarmth: never, loadBilling: never, loadLaunchLatency: never };
const dialogProps: ConfirmDialogProps = {
  open: true,
  title: "Drain cell-a",
  consequence: "Release this cell",
  confirmLabel: "Drain cell",
  confirmPhrase: "cell-a",
  onConfirm: () => {},
  onCancel: () => {},
};

describe("confirmation lifecycle", () => {
  it("preserves a typed drain confirmation across an actual fleet refresh", async () => {
    const drained: string[] = [];
    const onDrainCell = async (id: string) => {
      drained.push(id);
    };
    const props = { cells: [cell], leases: [], onRefresh: async () => {}, ...loaders, onDrainCell };
    const view = render(<FleetPage {...props} refreshRevision={0} />);
    fireEvent.click(screen.getByRole("button", { name: "Drain" }));
    const dialog = await screen.findByRole("dialog");
    const field = within(dialog).getByRole("textbox") as HTMLInputElement;
    fireEvent.change(field, { target: { value: cell.name } });
    expect(within(dialog).getByRole("button", { name: "Drain cell" }).hasAttribute("disabled")).toBe(false);

    view.rerender(<FleetPage {...props} refreshRevision={1} />);

    expect(field.value).toBe(cell.name);
    expect(within(dialog).getByRole("button", { name: "Drain cell" }).hasAttribute("disabled")).toBe(false);
    expect(drained).toEqual([]);
    fireEvent.click(within(dialog).getByRole("button", { name: "Drain cell" }));
    expect(await screen.findByText(/us-east5-b-v6e-01 is draining/)).toBeTruthy();
    expect(drained).toEqual([cell.id]);
  });

  it("resets the phrase on a genuine close and reopen", () => {
    const view = render(<ConfirmDialog {...dialogProps} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "cell-a" } });
    view.rerender(<ConfirmDialog {...dialogProps} open={false} />);
    view.rerender(<ConfirmDialog {...dialogProps} />);
    expect((screen.getByRole("textbox") as HTMLInputElement).value).toBe("");
    expect(screen.getByRole("button", { name: "Drain cell" }).hasAttribute("disabled")).toBe(true);
  });

  it("resets the phrase when the confirmation target changes while open", () => {
    const view = render(<ConfirmDialog {...dialogProps} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "cell-a" } });
    view.rerender(<ConfirmDialog {...dialogProps} title="Drain cell-b" confirmPhrase="cell-b" />);
    expect((screen.getByRole("textbox") as HTMLInputElement).value).toBe("");
    expect(screen.getByRole("button", { name: "Drain cell" }).hasAttribute("disabled")).toBe(true);
  });

  it("uses the latest cancellation callback without resetting the phrase", () => {
    const cancellations: string[] = [];
    const view = render(<ConfirmDialog {...dialogProps} onCancel={() => cancellations.push("old")} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "cell-a" } });
    view.rerender(<ConfirmDialog {...dialogProps} onCancel={() => cancellations.push("current")} />);
    expect((screen.getByRole("textbox") as HTMLInputElement).value).toBe("cell-a");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(cancellations).toEqual(["current"]);
  });

  it.each(["Escape", "backdrop"])("keeps a pending fleet drain open after %s", async (route) => {
    let settle = () => {};
    const pending = new Promise<void>((resolve) => {
      settle = resolve;
    });
    const drained: string[] = [];
    render(
      <FleetPage
        cells={[cell, { ...cell, id: "cell-b", name: "us-east5-b-v6e-02" }]}
        leases={[]}
        onRefresh={async () => {}}
        {...loaders}
        onDrainCell={(id) => {
          drained.push(id);
          return pending;
        }}
      />,
    );
    fireEvent.click(screen.getAllByRole("button", { name: "Drain" })[0]);
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: cell.name } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Drain cell" }));
    expect(within(dialog).getByRole("button", { name: "Working" }).hasAttribute("disabled")).toBe(true);
    expect(within(dialog).getByRole("button", { name: "Cancel" }).hasAttribute("disabled")).toBe(true);

    if (route === "Escape") fireEvent.keyDown(window, { key: "Escape" });
    else fireEvent.click(screen.getByRole("button", { name: `Close Drain ${cell.name} dialog` }));

    expect(screen.queryByRole("dialog")).toBe(dialog);
    expect(within(dialog).getByRole("button", { name: "Working" }).hasAttribute("disabled")).toBe(true);
    expect(drained).toEqual([cell.id]);
    settle();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByText(/us-east5-b-v6e-01 is draining/)).toBeTruthy();
  });

  it.each(["Escape", "backdrop"])("allows idle fleet confirmation dismissal with %s", async (route) => {
    const drained: string[] = [];
    render(
      <FleetPage
        cells={[cell]}
        leases={[]}
        onRefresh={async () => {}}
        {...loaders}
        onDrainCell={async (id) => {
          drained.push(id);
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Drain" }));
    await screen.findByRole("dialog");
    if (route === "Escape") fireEvent.keyDown(window, { key: "Escape" });
    else fireEvent.click(screen.getByRole("button", { name: `Close Drain ${cell.name} dialog` }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(drained).toEqual([]);
  });
});
