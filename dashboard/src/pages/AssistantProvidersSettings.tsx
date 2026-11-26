// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertTriangle, ExternalLink, LoaderCircle, LogIn, Plus, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useRef, useState, type FormEvent } from "react";

import type {
  AssistantModelDiscovery,
  AssistantModelDiscoveryInput,
  AssistantPreset,
  AssistantProvider,
  AssistantProviderInput,
  AssistantProviderList,
  AssistantProviderUpdate,
  ClaudeCodeLoginStart,
  ClaudeCodeStatus,
  OpenRouterSignInInput,
  OpenRouterSignInStart,
} from "../types";

/** Methods the settings call -- always as `client.method(...)`: the real class needs its `this`. */
export interface AssistantSettingsClient {
  listAssistantProviders(signal?: AbortSignal): Promise<AssistantProviderList>;
  listAssistantPresets(signal?: AbortSignal): Promise<AssistantPreset[]>;
  discoverAssistantModels(body: AssistantModelDiscoveryInput, signal?: AbortSignal): Promise<AssistantModelDiscovery>;
  createAssistantProvider(input: AssistantProviderInput): Promise<AssistantProvider>;
  updateAssistantProvider(id: string, changes: AssistantProviderUpdate): Promise<AssistantProvider>;
  deleteAssistantProvider(id: string): Promise<{ deleted: string }>;
  startOpenRouterSignIn(body: OpenRouterSignInInput): Promise<OpenRouterSignInStart>;
  getClaudeCodeStatus(signal?: AbortSignal): Promise<ClaudeCodeStatus>;
  startClaudeCodeLogin(): Promise<ClaudeCodeLoginStart>;
}

interface AssistantProvidersSettingsProps {
  client: AssistantSettingsClient;
  /** Called after any change, so the assistant dock re-reads its picker. */
  onChanged?: () => void;
  /** Sends the browser to the provider's sign-in page; replaceable in tests. */
  assignLocation?: (url: string) => void;
  /** How often a running Claude Code sign-in is re-checked; shortened in tests. */
  claudeCodePollMs?: number;
}

/** Typing a key re-asks the provider for models once the typing pauses. */
export const DISCOVERY_DEBOUNCE_MS = 500;
/** A Claude Code sign-in is re-checked this often, for at most CLAUDE_CODE_POLL_LIMIT_MS. */
export const CLAUDE_CODE_POLL_MS = 3_000;
export const CLAUDE_CODE_POLL_LIMIT_MS = 180_000;

const statusOf = (error: unknown): unknown =>
  error && typeof error === "object" && "status" in error ? (error as { status?: unknown }).status : undefined;

/** "Signed in · Claude Max via claude.ai", "Not signed in", or why neither can be said. */
export function claudeCodeStatusText(status: ClaudeCodeStatus): string {
  if (!status.installed) return "Claude Code isn't installed on the plane host";
  if (!status.logged_in) return "Not signed in";
  const kind = status.subscription_type?.trim();
  const plan = kind ? (/^claude\b/i.test(kind) ? kind : `Claude ${kind[0].toUpperCase()}${kind.slice(1)}`) : "";
  const via = status.auth_method?.trim() ? `via ${status.auth_method.trim()}` : "";
  const tail = [plan, via].filter(Boolean).join(" ");
  return tail ? `Signed in · ${tail}` : "Signed in";
}

function ClaudeCodeCard({
  client,
  pollMs,
  onStatus,
}: {
  client: AssistantSettingsClient;
  pollMs: number;
  onStatus: (status: ClaudeCodeStatus) => void;
}) {
  const [status, setStatus] = useState<ClaudeCodeStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [starting, setStarting] = useState(false);
  const [login, setLogin] = useState<{ url: string | null; alreadyRunning: boolean } | null>(null);
  const [pollUntil, setPollUntil] = useState<number | null>(null);
  const [timedOut, setTimedOut] = useState(false);
  const alive = useRef(true);
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const check = useCallback(async () => {
    setChecking(true);
    try {
      const value = await client.getClaudeCodeStatus();
      if (!alive.current) return null;
      setStatus(value);
      setError(null);
      onStatusRef.current(value);
      return value;
    } catch (failure) {
      if (alive.current) setError(message(failure, "Could not ask the plane host about Claude Code"));
      return null;
    } finally {
      if (alive.current) setChecking(false);
    }
  }, [client]);

  useEffect(() => {
    void check();
  }, [check]);

  // While a sign-in runs on the plane host, re-check on a fixed cadence until
  // it lands or the three-minute window closes; "Check again" stays available
  // after that. An interval rather than a timer re-armed per answer: a refused
  // read, or an answer identical to the last, must not end the wait.
  const inFlight = useRef(false);
  useEffect(() => {
    if (pollUntil === null) return;
    const timer = globalThis.setInterval(() => {
      if (Date.now() >= pollUntil) {
        setPollUntil(null);
        setTimedOut(true);
        return;
      }
      if (inFlight.current) return;
      inFlight.current = true;
      void check().finally(() => {
        inFlight.current = false;
      });
    }, pollMs);
    return () => globalThis.clearInterval(timer);
  }, [check, pollMs, pollUntil]);

  useEffect(() => {
    if (status?.logged_in) setPollUntil(null);
  }, [status]);

  const signIn = async () => {
    setStarting(true);
    setError(null);
    setTimedOut(false);
    try {
      const started = await client.startClaudeCodeLogin();
      if (!alive.current) return;
      setLogin({ url: started.url ?? null, alreadyRunning: false });
      setPollUntil(Date.now() + CLAUDE_CODE_POLL_LIMIT_MS);
    } catch (failure) {
      if (!alive.current) return;
      if (statusOf(failure) === 409) {
        setLogin((current) => ({ url: current?.url ?? null, alreadyRunning: true }));
        setPollUntil(Date.now() + CLAUDE_CODE_POLL_LIMIT_MS);
      } else {
        setError(message(failure, "Could not start the Claude Code sign-in"));
      }
    } finally {
      if (alive.current) setStarting(false);
    }
  };

  const signedIn = Boolean(status?.logged_in);
  return (
    <div className="claude-code-card" role="group" aria-label="Claude Code">
      <div className="claude-code-status">
        {status === null ? (
          <span className="field-hint">
            {error ? "Claude Code status unknown" : "Checking Claude Code on the plane host…"}
          </span>
        ) : (
          <strong className={signedIn ? "claude-code-ok" : status.installed ? "" : "claude-code-missing"} role="status">
            {claudeCodeStatusText(status)}
          </strong>
        )}
        {status?.version && <span className="badge-tag">v{status.version}</span>}
      </div>
      {status?.detail && <span className="field-hint">{status.detail}</span>}
      {status && !signedIn && (
        <div className="provider-signin">
          {status.installed && (
            <button className="button button-primary" type="button" disabled={starting} onClick={() => void signIn()}>
              <LogIn size={14} aria-hidden="true" /> Sign in with Claude
            </button>
          )}
          <button className="button button-secondary" type="button" disabled={checking} onClick={() => void check()}>
            {checking ? (
              <LoaderCircle className="spin" size={13} aria-hidden="true" />
            ) : (
              <RefreshCw size={13} aria-hidden="true" />
            )}
            Check again
          </button>
        </div>
      )}
      {login && !signedIn && (
        <div className="claude-code-login">
          {login.url && (
            <a className="button button-secondary" href={login.url} target="_blank" rel="noopener noreferrer">
              Open sign-in page <ExternalLink size={12} aria-hidden="true" />
            </a>
          )}
          <span className="field-hint">
            {login.alreadyRunning ? "A sign-in is already running on the plane host. " : ""}
            The sign-in completes on the plane host.{" "}
            {pollUntil !== null
              ? "Waiting for it to finish…"
              : timedOut
                ? "Still not signed in. Finish the sign-in, then press Check again."
                : ""}
          </span>
        </div>
      )}
      {error && (
        <div className="inline-error" role="alert">
          <AlertTriangle size={14} aria-hidden="true" />
          {error}
        </div>
      )}
    </div>
  );
}

const LOCAL_PRESETS = new Set(["ollama", "lmstudio"]);
const message = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);
const unique = (values: string[]) => [...new Set(values.map((value) => value.trim()).filter(Boolean))];

export function presetBadge(preset: AssistantPreset): { text: string; tone: "accent" | "ok" | "plain" } {
  if (preset.sign_in) return { text: "Sign in", tone: "accent" };
  if (LOCAL_PRESETS.has(preset.id)) return { text: "Local", tone: "ok" };
  return { text: "Key", tone: "plain" };
}

/** A provider configured before presets existed is edited as a custom endpoint. */
function presetFor(provider: AssistantProvider, presets: AssistantPreset[] | null): AssistantPreset {
  const known = presets?.find((preset) => preset.id === provider.preset);
  if (known) return known;
  return {
    id: provider.preset ?? "custom",
    label: provider.preset ?? "Custom",
    base_url: provider.base_url ?? "",
    editable_url: !provider.preset,
    key_required: false,
    key_url: null,
    sign_in: null,
    suggested_models: [],
  };
}

const defaultAssignLocation = (url: string) => globalThis.location.assign(url);

type DiscoveryState =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "done"; source: AssistantModelDiscovery["source"]; detail: string | null; count: number }
  | { state: "failed" };

function ProviderForm({
  client,
  preset,
  editing,
  onSaved,
  onCancel,
  assignLocation,
  claudeCodePollMs,
}: {
  client: AssistantSettingsClient;
  preset: AssistantPreset;
  editing?: AssistantProvider;
  onSaved: (text: string) => void;
  onCancel: () => void;
  assignLocation: (url: string) => void;
  claudeCodePollMs: number;
}) {
  const ids = useId();
  const signIn = !editing && preset.sign_in === "openrouter";
  // Claude Code holds its own sign-in on the plane host: no key, no URL.
  const claudeCode = preset.sign_in === "claude-code" || preset.id === "claude-code";
  const [claudeStatus, setClaudeStatus] = useState<ClaudeCodeStatus | null>(null);
  const claudeReady = !claudeCode || Boolean(editing) || Boolean(claudeStatus?.logged_in);
  const keyNeeded = !editing && preset.key_required && !claudeCode;
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [baseUrl, setBaseUrl] = useState(editing?.base_url ?? preset.base_url);
  const [models, setModels] = useState<string[]>(editing?.models ?? []);
  const [checked, setChecked] = useState<string[]>(editing?.models ?? []);
  const [defaultModel, setDefaultModel] = useState(editing?.default_model ?? "");
  const [filter, setFilter] = useState("");
  const [extra, setExtra] = useState("");
  const [name, setName] = useState(editing?.name ?? "");
  const [maxSteps, setMaxSteps] = useState(editing?.max_steps ? String(editing.max_steps) : "");
  const [showKey, setShowKey] = useState(!signIn && !claudeCode);
  const [discovery, setDiscovery] = useState<DiscoveryState>({ state: "idle" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const discoverAbort = useRef<AbortController | null>(null);
  // Once the admin ticks or adds a model, a later listing never overrides it.
  const choicesTouched = useRef(Boolean(editing));

  const url = baseUrl.trim();
  const key = clearKey ? "" : apiKey.trim();
  const canDiscover = claudeReady && (!preset.editable_url || Boolean(url)) && (!keyNeeded || Boolean(key) || signIn);

  const discover = useCallback(
    async (currentKey: string, currentUrl: string) => {
      discoverAbort.current?.abort();
      const controller = new AbortController();
      discoverAbort.current = controller;
      const body: AssistantModelDiscoveryInput = editing ? { provider_id: editing.id } : { preset: preset.id };
      if (currentKey) body.api_key = currentKey;
      if (preset.editable_url && currentUrl && (!editing || currentUrl !== (editing.base_url ?? ""))) {
        body.base_url = currentUrl;
      }
      setDiscovery({ state: "loading" });
      try {
        const result = await client.discoverAssistantModels(body, controller.signal);
        if (controller.signal.aborted) return;
        setDiscovery({ state: "done", source: result.source, detail: result.detail, count: result.models.length });
        setModels((current) => unique([...result.models, ...current]));
        if (!choicesTouched.current) {
          const suggested = preset.suggested_models.filter((model) => result.models.includes(model));
          const initial = suggested.length ? suggested : result.models.slice(0, 10);
          setChecked(initial);
          setDefaultModel((current) => (initial.includes(current) ? current : (initial[0] ?? "")));
        }
      } catch (failure) {
        if (controller.signal.aborted) return;
        setDiscovery({ state: "failed" });
        setError(message(failure, "Could not load the provider's models"));
      }
    },
    [client, editing, preset],
  );

  useEffect(() => () => discoverAbort.current?.abort(), []);

  // New providers list their models without a button press: at once when no
  // key is needed, otherwise once the pasted key (or typed URL) settles.
  useEffect(() => {
    if (editing || !canDiscover) return;
    const typed = Boolean(key) || (preset.editable_url && Boolean(url));
    const timer = globalThis.setTimeout(() => void discover(key, url), typed ? DISCOVERY_DEBOUNCE_MS : 0);
    return () => globalThis.clearTimeout(timer);
  }, [canDiscover, discover, editing, key, preset.editable_url, url]);

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return needle ? models.filter((model) => model.toLowerCase().includes(needle)) : models;
  }, [filter, models]);
  const chosenDefault = checked.includes(defaultModel) ? defaultModel : (checked[0] ?? "");

  const toggleModel = (model: string, on: boolean) => {
    choicesTouched.current = true;
    setChecked((current) => (on ? unique([...current, model]) : current.filter((item) => item !== model)));
  };

  const addModel = () => {
    const model = extra.trim();
    if (!model) return;
    choicesTouched.current = true;
    setModels((current) => unique([...current, model]));
    setChecked((current) => unique([...current, model]));
    setExtra("");
  };

  const parsedSteps = (): number | undefined | null => {
    if (!maxSteps.trim()) return undefined;
    const steps = Number(maxSteps);
    return Number.isInteger(steps) && steps >= 1 && steps <= 20 ? steps : null;
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    if (preset.editable_url && !url) return setError("Enter the provider's base URL.");
    if (keyNeeded && !key) return setError(`Paste an API key for ${preset.label}.`);
    if (!checked.length) return setError("Choose at least one model.");
    const steps = parsedSteps();
    if (steps === null) return setError("Max steps must be a whole number from 1 to 20.");
    setBusy(true);
    try {
      if (editing) {
        const changes: AssistantProviderUpdate = { models: checked, default_model: chosenDefault };
        if (name.trim() && name.trim() !== editing.name) changes.name = name.trim();
        if (preset.editable_url && url !== (editing.base_url ?? "")) changes.base_url = url;
        if (steps !== undefined && steps !== editing.max_steps) changes.max_steps = steps;
        if (clearKey) changes.clear_api_key = true;
        else if (key) changes.api_key = key;
        await client.updateAssistantProvider(editing.id, changes);
      } else {
        const input: AssistantProviderInput = { preset: preset.id, models: checked, default_model: chosenDefault };
        if (key) input.api_key = key;
        if (preset.editable_url) input.base_url = url;
        if (name.trim()) input.name = name.trim();
        if (steps !== undefined) input.max_steps = steps;
        await client.createAssistantProvider(input);
      }
      setApiKey("");
      onSaved(editing ? `${name.trim() || editing.name} saved.` : `${name.trim() || preset.label} added.`);
    } catch (failure) {
      setError(message(failure, "The assistant provider was not saved"));
    } finally {
      setBusy(false);
    }
  };

  const signInWithOpenRouter = async () => {
    setError(null);
    setBusy(true);
    try {
      const body: OpenRouterSignInInput = {};
      if (checked.length) {
        body.models = checked;
        body.default_model = chosenDefault;
      }
      if (name.trim()) body.name = name.trim();
      const started = await client.startOpenRouterSignIn(body);
      assignLocation(started.authorize_url);
    } catch (failure) {
      setError(message(failure, "Could not start the OpenRouter sign-in"));
      setBusy(false);
    }
  };

  const badge = presetBadge(preset);
  const keyHint = editing
    ? "Leave empty to keep the stored key."
    : preset.key_required
      ? "Stored on the plane and never shown again."
      : "Optional for this provider. Stored on the plane and never shown again.";

  return (
    <form
      className="provider-form"
      noValidate
      aria-label={editing ? `Edit ${editing.name}` : `Add ${preset.label}`}
      onSubmit={(event) => void save(event)}
    >
      <div className="provider-form-head">
        <strong>{editing ? `Edit ${editing.name}` : preset.label}</strong>
        <span className={`badge-tag badge-tag-${badge.tone}`}>{badge.text}</span>
      </div>

      {signIn && (
        <div className="provider-signin">
          <button
            className="button button-primary"
            type="button"
            disabled={busy}
            onClick={() => void signInWithOpenRouter()}
          >
            <LogIn size={14} aria-hidden="true" /> Sign in with OpenRouter
          </button>
          <span className="field-hint">OpenRouter asks you to approve, then sends you back here.</span>
          {!showKey && (
            <button className="text-button" type="button" onClick={() => setShowKey(true)}>
              or paste an API key
            </button>
          )}
        </div>
      )}

      {claudeCode && <ClaudeCodeCard client={client} pollMs={claudeCodePollMs} onStatus={setClaudeStatus} />}

      {!claudeCode && (
        <div className="form-grid form-grid-two">
          {showKey && (
            <div className="field">
              <label htmlFor={`${ids}-key`}>API key</label>
              <input
                id={`${ids}-key`}
                type="password"
                autoComplete="off"
                spellCheck={false}
                placeholder={
                  editing ? "Stored key is kept when left empty" : preset.key_required ? "Paste your key" : ""
                }
                value={apiKey}
                disabled={busy || clearKey}
                onChange={(event) => setApiKey(event.target.value)}
              />
              <span className="field-hint">
                {keyHint}{" "}
                {preset.key_url && (
                  <a href={preset.key_url} target="_blank" rel="noopener noreferrer">
                    Get a key <ExternalLink size={11} aria-hidden="true" />
                  </a>
                )}
              </span>
            </div>
          )}
          <div className="field">
            <label htmlFor={`${ids}-url`}>Base URL</label>
            <input
              id={`${ids}-url`}
              type="url"
              required={preset.editable_url}
              readOnly={!preset.editable_url}
              placeholder={preset.editable_url ? "https://api.example.com/v1" : ""}
              value={baseUrl}
              disabled={busy}
              onChange={(event) => setBaseUrl(event.target.value)}
            />
            {!preset.editable_url && <span className="field-hint">Set by the {preset.label} preset.</span>}
          </div>
          {editing?.has_api_key && (
            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={clearKey}
                disabled={busy}
                onChange={(event) => {
                  setClearKey(event.target.checked);
                  if (event.target.checked) setApiKey("");
                }}
              />
              <span>Remove stored key</span>
            </label>
          )}
        </div>
      )}

      {claudeReady && (
        <>
          <div className="provider-models">
            <div className="model-toolbar">
              <strong>Models</strong>
              <button
                className="button button-secondary"
                type="button"
                disabled={busy || !canDiscover || discovery.state === "loading"}
                onClick={() => void discover(key, url)}
              >
                {discovery.state === "loading" ? (
                  <LoaderCircle className="spin" size={13} aria-hidden="true" />
                ) : (
                  <RefreshCw size={13} aria-hidden="true" />
                )}
                Load models
              </button>
              <span className="field-hint" role="status">
                {discovery.state === "loading"
                  ? "Asking the provider…"
                  : discovery.state === "done"
                    ? discovery.source === "live"
                      ? `${discovery.count} models from the provider`
                      : `Showing suggested models${discovery.detail ? ` — ${discovery.detail}` : ""}`
                    : !canDiscover
                      ? preset.editable_url && !url
                        ? "Enter the base URL to list models."
                        : "Paste a key to list models."
                      : ""}
              </span>
              {models.length > 0 && <span className="field-hint">{checked.length} selected</span>}
            </div>
            {models.length > 8 && (
              <input
                className="model-filter"
                aria-label="Filter models"
                placeholder="Filter models"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              />
            )}
            <div className="model-checklist" role="group" aria-label="Models">
              {visible.map((model) => (
                <label key={model} className="model-check" title={model}>
                  <input
                    type="checkbox"
                    checked={checked.includes(model)}
                    disabled={busy}
                    onChange={(event) => toggleModel(model, event.target.checked)}
                  />
                  <span>{model}</span>
                </label>
              ))}
              {models.length === 0 && <span className="field-hint">No models listed yet.</span>}
              {models.length > 0 && visible.length === 0 && <span className="field-hint">No models match.</span>}
            </div>
            <div className="form-grid form-grid-two">
              <div className="field">
                <label htmlFor={`${ids}-default`}>Default model</label>
                <select
                  id={`${ids}-default`}
                  value={chosenDefault}
                  disabled={busy || checked.length === 0}
                  onChange={(event) => setDefaultModel(event.target.value)}
                >
                  {checked.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor={`${ids}-extra`}>Add model id</label>
                <span className="field-inline">
                  <input
                    id={`${ids}-extra`}
                    placeholder="A model the provider did not list"
                    value={extra}
                    disabled={busy}
                    onChange={(event) => setExtra(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        addModel();
                      }
                    }}
                  />
                  <button className="button button-secondary" type="button" disabled={!extra.trim()} onClick={addModel}>
                    <Plus size={13} aria-hidden="true" /> Add model
                  </button>
                </span>
              </div>
            </div>
          </div>

          <details className="provider-advanced">
            <summary>Advanced</summary>
            <div className="form-grid form-grid-two">
              <div className="field">
                <label htmlFor={`${ids}-name`}>Name</label>
                <input
                  id={`${ids}-name`}
                  placeholder={preset.label}
                  value={name}
                  disabled={busy}
                  onChange={(event) => setName(event.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor={`${ids}-steps`}>Max steps</label>
                <input
                  id={`${ids}-steps`}
                  type="number"
                  min={1}
                  max={20}
                  step={1}
                  placeholder="8"
                  value={maxSteps}
                  disabled={busy}
                  onChange={(event) => setMaxSteps(event.target.value)}
                />
              </div>
            </div>
          </details>
        </>
      )}

      {error && (
        <div className="inline-error" role="alert">
          <AlertTriangle size={14} aria-hidden="true" />
          {error}
        </div>
      )}

      <div className="provider-form-actions">
        {(showKey || (claudeCode && claudeReady)) && (
          <button className="button button-primary" type="submit" disabled={busy}>
            {busy ? "Saving…" : "Save provider"}
          </button>
        )}
        <button className="button button-secondary" type="button" disabled={busy} onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

type Mode =
  | { kind: "list" }
  | { kind: "choose" }
  | { kind: "add"; preset: AssistantPreset }
  | { kind: "edit"; provider: AssistantProvider };

export function AssistantProvidersSettings({
  client,
  onChanged,
  assignLocation = defaultAssignLocation,
  claudeCodePollMs = CLAUDE_CODE_POLL_MS,
}: AssistantProvidersSettingsProps) {
  const [list, setList] = useState<AssistantProviderList | null>(null);
  const [presets, setPresets] = useState<AssistantPreset[] | null>(null);
  const [revision, setRevision] = useState(0);
  const [mode, setMode] = useState<Mode>({ kind: "list" });
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    client
      .listAssistantProviders(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setList(value);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) setError(message(failure, "Could not read assistant providers"));
      });
    return () => controller.abort();
  }, [client, revision]);

  useEffect(() => {
    const controller = new AbortController();
    client
      .listAssistantPresets(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setPresets(value);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) {
          setPresets([]);
          setError(message(failure, "Could not read the provider presets"));
        }
      });
    return () => controller.abort();
  }, [client]);

  const changed = (text: string | null) => {
    setRevision((value) => value + 1);
    setNotice(text);
    onChanged?.();
  };

  const act = async (id: string, operation: () => Promise<unknown>, done: string) => {
    setBusyId(id);
    setError(null);
    try {
      await operation();
      changed(done);
    } catch (failure) {
      setError(message(failure, "The assistant provider was not updated"));
    } finally {
      setBusyId(null);
    }
  };

  const remove = (provider: AssistantProvider) => {
    if (!globalThis.confirm(`Delete assistant provider “${provider.name}”? Its stored key is deleted too.`)) return;
    void act(provider.id, () => client.deleteAssistantProvider(provider.id), `${provider.name} deleted.`);
  };

  const providers = list?.providers ?? [];
  const canManage = list?.can_manage !== false;
  const presetLabel = (provider: AssistantProvider) =>
    presets?.find((preset) => preset.id === provider.preset)?.label ?? provider.preset ?? "Custom";

  return (
    <section className="panel data-panel" aria-label="Assistant providers">
      <div className="panel-header">
        <div>
          <h2>Assistant providers</h2>
          <p>The models the fleet assistant can use. Keys stay on the plane and are never shown again.</p>
        </div>
        {mode.kind === "list" && (
          <button
            className="button button-primary"
            type="button"
            disabled={!canManage || list === null}
            onClick={() => {
              setNotice(null);
              setMode({ kind: "choose" });
            }}
          >
            <Plus size={14} aria-hidden="true" /> Add provider
          </button>
        )}
      </div>

      {error && (
        <div className="inline-error provider-alert" role="alert">
          <AlertTriangle size={14} aria-hidden="true" />
          {error}
        </div>
      )}
      {notice && (
        <p className="panel-note" role="status">
          {notice}
        </p>
      )}
      {list !== null && !canManage && (
        <p className="panel-note">Only fleet admins can change the assistant providers.</p>
      )}

      {mode.kind === "list" &&
        (list === null ? (
          <p className="panel-note">Reading providers…</p>
        ) : providers.length === 0 ? (
          <p className="panel-note">No assistant provider yet. Add one: pick a provider and paste its API key.</p>
        ) : (
          <ul className="provider-list" aria-label="Configured providers">
            {providers.map((provider) => {
              const enabled = provider.enabled !== false;
              const busy = busyId === provider.id;
              return (
                <li key={provider.id} className="provider-row" aria-label={provider.name}>
                  <div className="provider-row-main">
                    <span className="provider-row-title">
                      <strong>{provider.name}</strong>
                      <span className="badge-tag">{presetLabel(provider)}</span>
                      <span className={`badge-tag ${provider.has_api_key ? "badge-tag-ok" : ""}`}>
                        {provider.has_api_key ? "key set" : "no key"}
                      </span>
                    </span>
                    <span className="provider-row-meta">
                      {provider.models.length} {provider.models.length === 1 ? "model" : "models"} · default{" "}
                      {provider.default_model || "—"}
                    </span>
                  </div>
                  <div className="provider-row-actions">
                    <label className="toggle-field">
                      <input
                        type="checkbox"
                        aria-label={`Enable ${provider.name}`}
                        checked={enabled}
                        disabled={!canManage || busy}
                        onChange={() =>
                          void act(
                            provider.id,
                            () => client.updateAssistantProvider(provider.id, { enabled: !enabled }),
                            `${provider.name} ${enabled ? "disabled" : "enabled"}.`,
                          )
                        }
                      />
                      <span>{enabled ? "Enabled" : "Disabled"}</span>
                    </label>
                    <button
                      className="button button-secondary"
                      type="button"
                      disabled={!canManage || busy}
                      onClick={() => {
                        setNotice(null);
                        setMode({ kind: "edit", provider });
                      }}
                    >
                      Edit
                    </button>
                    <button
                      className="button button-secondary"
                      type="button"
                      disabled={!canManage || busy}
                      onClick={() => remove(provider)}
                    >
                      Delete
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        ))}

      {mode.kind === "choose" && (
        <div className="provider-choose">
          <p className="panel-note">Choose a provider. You only need its API key; the rest is filled in.</p>
          {presets === null ? (
            <p className="panel-note">Reading presets…</p>
          ) : (
            <div className="preset-grid" role="group" aria-label="Provider presets">
              {presets.map((preset) => {
                const badge = presetBadge(preset);
                return (
                  <button
                    key={preset.id}
                    className="preset-card"
                    type="button"
                    aria-label={preset.label}
                    title={preset.base_url || preset.label}
                    onClick={() => setMode({ kind: "add", preset })}
                  >
                    <span>{preset.label}</span>
                    <span className={`badge-tag badge-tag-${badge.tone}`} aria-hidden="true">
                      {badge.text}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
          <div className="provider-form-actions provider-choose-actions">
            <button className="button button-secondary" type="button" onClick={() => setMode({ kind: "list" })}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {mode.kind === "add" && (
        <ProviderForm
          key={`add-${mode.preset.id}`}
          client={client}
          preset={mode.preset}
          assignLocation={assignLocation}
          claudeCodePollMs={claudeCodePollMs}
          onCancel={() => setMode({ kind: "choose" })}
          onSaved={(text) => {
            setMode({ kind: "list" });
            changed(text);
          }}
        />
      )}
      {mode.kind === "edit" && (
        <ProviderForm
          key={`edit-${mode.provider.id}`}
          client={client}
          preset={presetFor(mode.provider, presets)}
          editing={mode.provider}
          assignLocation={assignLocation}
          claudeCodePollMs={claudeCodePollMs}
          onCancel={() => setMode({ kind: "list" })}
          onSaved={(text) => {
            setMode({ kind: "list" });
            changed(text);
          }}
        />
      )}
    </section>
  );
}
