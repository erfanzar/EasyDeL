// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { Copy } from "lucide-react";
import { useState } from "react";

/** A secret or ready-to-paste block the plane shows exactly once, with a Copy button. */
export function CopyOnce({ label, value, block = false }: { label: string; value: string; block?: boolean }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="panel-note" role="status">
      <strong>{label}</strong> (shown once):{" "}
      {block ? (
        <pre className="monospace" data-testid="secret-once">
          {value}
        </pre>
      ) : (
        <code className="monospace" data-testid="secret-once">
          {value}
        </code>
      )}{" "}
      <button
        className="button button-secondary"
        type="button"
        onClick={() => {
          void navigator.clipboard?.writeText(value).then(() => setCopied(true));
        }}
      >
        <Copy size={14} aria-hidden="true" /> {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}
