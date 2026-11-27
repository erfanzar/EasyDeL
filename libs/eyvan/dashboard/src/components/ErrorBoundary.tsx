// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Component, type ErrorInfo, type ReactNode } from "react";

import { ErrorCard, type ErrorCardAction } from "./ErrorCard";

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Changing this key clears any captured crash so navigating away recovers automatically. */
  resetKey?: string;
  onGoHome?: () => void;
  onReload?: () => void;
}

interface ErrorBoundaryState {
  error: Error | null;
  componentStack: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null, componentStack: "" };

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ error, componentStack: info.componentStack ?? "" });
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    if (this.state.error && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ error: null, componentStack: "" });
    }
  }

  private reset = () => {
    this.setState({ error: null, componentStack: "" });
    this.props.onReload?.();
  };

  private goHome = () => {
    this.setState({ error: null, componentStack: "" });
    if (this.props.onGoHome) {
      this.props.onGoHome();
    } else if (typeof globalThis.location !== "undefined") {
      globalThis.location.hash = "#/overview";
    }
  };

  render(): ReactNode {
    const { error, componentStack } = this.state;
    if (!error) return this.props.children;

    const rawDetails = [
      `${error.name}: ${error.message}`,
      error.stack ? `\n${error.stack}` : "",
      componentStack ? `\nComponent stack:${componentStack}` : "",
    ]
      .filter(Boolean)
      .join("\n");

    const actions: ErrorCardAction[] = [
      { label: "Reload view", onClick: this.reset, variant: "primary" },
      { label: "Go to Overview", onClick: this.goHome, variant: "secondary" },
    ];

    return (
      <div className="error-boundary-shell">
        <ErrorCard
          error={{
            code: "unknown",
            tone: "danger",
            title: "Unexpected view error",
            summary: error.message || "A rendering error interrupted this view.",
            raw: rawDetails,
            hasTechnicalDetails: true,
            remediation: "Reload this view or return to Overview. Copy the technical details below if reporting a bug.",
            badges: [{ label: "Type", value: error.name || "Error" }],
          }}
          actions={actions}
        />
      </div>
    );
  }
}
