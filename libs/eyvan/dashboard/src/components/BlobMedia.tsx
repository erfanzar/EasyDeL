// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useState } from "react";
import type { BlobRef } from "../types";
import { TrackingState } from "./TrackingState";

export type BlobLoader = (runId: string, digest: string, signal: AbortSignal) => Promise<Blob>;

/** Never put bearer-protected API paths in media src attributes. Own and revoke each URL. */
export function useRunBlob(runId: string, blob: BlobRef | null, loadBlob: BlobLoader, retry = 0) {
  const digest = blob?.sha256;
  const available = blob?.available !== false;
  // Blobs are immutable: polling must not revoke a playing video's source.
  const key = JSON.stringify([runId, digest, available, retry]);
  const [result, setResult] = useState<{ key: string; loader: BlobLoader; url?: string; error?: Error } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let url: string | undefined;
    if (digest && available) {
      loadBlob(runId, digest, controller.signal)
        .then((data) => {
          if (controller.signal.aborted) return;
          url = URL.createObjectURL(data);
          setResult({ key, loader: loadBlob, url });
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted)
            setResult({
              key,
              loader: loadBlob,
              error: reason instanceof Error ? reason : new Error("Unable to load media"),
            });
        });
    }
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [runId, digest, available, key, loadBlob]);
  return result?.key === key && result.loader === loadBlob ? result : null;
}

export function BlobMedia({
  runId,
  blob,
  caption,
  loadBlob,
  onRetry,
}: {
  runId: string;
  blob: BlobRef;
  caption: string;
  loadBlob: BlobLoader;
  revision: number;
  onRetry: () => void;
}) {
  const [retry, setRetry] = useState(0);
  const result = useRunBlob(runId, blob, loadBlob, retry);
  if (blob.available === false) return <p className="run-panel-empty">Media not uploaded · unavailable</p>;
  if (result?.error)
    return (
      <TrackingState
        error={result.error}
        onRetry={() => {
          setRetry((value) => value + 1);
          onRetry();
        }}
      />
    );
  if (!result?.url) return <p className="run-panel-empty">Loading media…</p>;
  if (blob.content_type.startsWith("image/")) return <img className="tracking-media" src={result.url} alt={caption} />;
  if (blob.content_type.startsWith("video/"))
    return <video className="tracking-media" src={result.url} controls aria-label={caption} />;
  if (blob.content_type.startsWith("audio/"))
    return <audio className="tracking-media" src={result.url} controls aria-label={caption} />;
  return <p className="run-panel-empty">Unsupported media type</p>;
}
