// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertTriangle, LoaderCircle, LogIn, UserPlus } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";

import { acceptTeamInvite, previewTeamInvite, teamLogin } from "./oidc";

/** The hash an invite link opens: `#/accept-invite/<token>`. */
export const ACCEPT_INVITE_PREFIX = "#/accept-invite/";

/**
 * Read an invite token out of the address bar and strip it, so it does not
 * linger in history, a copied URL or a screenshot. Returns undefined when the
 * page was not opened from an invite link.
 */
export function takeInviteTokenFromLocation(): string | undefined {
  const hash = globalThis.location?.hash ?? "";
  if (!hash.startsWith(ACCEPT_INVITE_PREFIX)) return undefined;
  const token = decodeURIComponent(hash.slice(ACCEPT_INVITE_PREFIX.length)).trim();
  globalThis.history?.replaceState(null, "", `${globalThis.location.pathname}${globalThis.location.search}`);
  return token || undefined;
}

const message = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);

export function TeamLoginScreen({
  notice,
  onOidcSignIn,
}: {
  notice?: string | null;
  /** Set when the plane also names an identity provider: offer it beside the password form. */
  onOidcSignIn?: () => void;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(notice ?? null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await teamLogin(email.trim(), password);
    } catch (failure) {
      setError(message(failure, "Sign-in failed"));
      setBusy(false);
    }
  };

  return (
    <main className="auth-screen">
      <form className="auth-panel team-auth-form" onSubmit={(event) => void submit(event)} aria-label="Sign in">
        <span className="brand-mark">
          <LogIn size={22} />
        </span>
        <h1>Sign in to Eyvan</h1>
        <p>Use your user name (or the email on your invite) and your password.</p>
        {error && (
          <div className="form-error" role="alert">
            {error}
          </div>
        )}
        <label className="field">
          <span>User name or email</span>
          <input
            type="text"
            autoCapitalize="none"
            spellCheck={false}
            autoComplete="username"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
            disabled={busy}
          />
        </label>
        <label className="field">
          <span>Password</span>
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
            disabled={busy}
          />
        </label>
        <button className="button button-primary" type="submit" disabled={busy || !email.trim() || !password}>
          {busy && <LoaderCircle className="spin" size={14} />}Sign in
        </button>
        {onOidcSignIn && (
          <>
            <p className="muted">or</p>
            <button className="button button-secondary" type="button" onClick={onOidcSignIn} disabled={busy}>
              <LogIn size={17} /> Sign in with OIDC
            </button>
          </>
        )}
      </form>
    </main>
  );
}

export function AcceptInviteScreen({ token, onCancel }: { token: string; onCancel: () => void }) {
  const [invite, setInvite] = useState<{ email: string; subject: string; expires_at: string } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    previewTeamInvite(token)
      .then((value) => {
        if (!cancelled) setInvite(value);
      })
      .catch((failure: unknown) => {
        if (!cancelled) setLoadError(message(failure, "This invite could not be read"));
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  const mismatch = confirm.length > 0 && confirm !== password;
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (password !== confirm) return;
    setBusy(true);
    setError(null);
    try {
      await acceptTeamInvite(token, password, displayName.trim());
    } catch (failure) {
      setError(message(failure, "Could not accept the invite"));
      setBusy(false);
    }
  };

  if (loadError) {
    return (
      <main className="auth-screen">
        <section className="auth-panel" role="alert">
          <span className="brand-mark">
            <AlertTriangle size={22} />
          </span>
          <h1>This invite cannot be used</h1>
          <p>{loadError}. Ask your admin for a new invite link.</p>
          <button className="button button-secondary" type="button" onClick={onCancel}>
            Go to sign-in
          </button>
        </section>
      </main>
    );
  }
  if (!invite) {
    return (
      <main className="auth-screen">
        <section className="auth-panel" role="status">
          <span className="brand-mark">
            <LoaderCircle className="spin" size={22} />
          </span>
          <h1>Reading your invite</h1>
        </section>
      </main>
    );
  }
  return (
    <main className="auth-screen">
      <form className="auth-panel team-auth-form" onSubmit={(event) => void submit(event)} aria-label="Accept invite">
        <span className="brand-mark">
          <UserPlus size={22} />
        </span>
        <h1>Join Eyvan</h1>
        <p>
          You were invited as <strong>{invite.subject}</strong> ({invite.email}). Choose a password to finish.
        </p>
        {error && (
          <div className="form-error" role="alert">
            {error}
          </div>
        )}
        <label className="field">
          <span>Display name (optional)</span>
          <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} disabled={busy} />
        </label>
        <label className="field">
          <span>Password</span>
          <input
            type="password"
            autoComplete="new-password"
            minLength={10}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
            disabled={busy}
          />
        </label>
        <label className="field">
          <span>Confirm password</span>
          <input
            type="password"
            autoComplete="new-password"
            value={confirm}
            onChange={(event) => setConfirm(event.target.value)}
            required
            disabled={busy}
          />
        </label>
        {mismatch && <p className="muted">The passwords do not match.</p>}
        <p className="muted">At least 10 characters.</p>
        <button
          className="button button-primary"
          type="submit"
          disabled={busy || password.length < 10 || password !== confirm}
        >
          {busy && <LoaderCircle className="spin" size={14} />}Create account
        </button>
      </form>
    </main>
  );
}
